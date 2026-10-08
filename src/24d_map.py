"""Шаг 24d. Статическая карта типов МО (фигура 2) с врезками Москвы и Санкт-Петербурга; проверки и компоновка (24d-2).

Классы МО (каждый действующий полигон ровно в одном): 1) тип из рабочей выборки (kmeans_labels_final.parquet),
2) тип присвоен вне выборки (types_outside_sample.parquet, status = assigned) — цвет типа со штриховкой,
3) тип не присвоен — светлая штриховка. Границы МО обводятся тонкой линией цвета заливки. На карте технические
имена типов (full_type_name из src/23_type_portraits.py), интерпретационные имена не используются; площади и
население не печатаются. Источник границ в подписи читается из data/geo/ATTRIBUTION.md.
Проверки до записи: данные (сдвиг долгот, МО 409, регионы), компоновка по границам элементов и по массиву PNG,
независимая проверка рисунка по пикселям (цвет в центре вписанной окружности каждого МО). Параметры и ожидания —
config.yaml, группа step24d_map. При нарушении любого контроля код выхода 1 и ничего не записывается.
Вход:  data/geo/mo_national.geojson, mo_cities.geojson, geometry_report.json, ATTRIBUTION.md (шаг 24б),
       data/processed/kmeans_labels_final.parquet, data/processed/types_outside_sample.parquet (шаг 24c)
Выход: notebooks/figures/24_fig2_map.png и .svg, data/processed/figure_data_24d.parquet, notebooks/24d_map.md
Запуск из корня проекта:  .venv/bin/python src/24d_map.py          (--layout-only: только компоновка, без записи)
"""
import argparse
import importlib
import io
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402

import config as C  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
P = lambda rel: PROJECT_DIR / rel  # noqa: E731
LABELS_PATH = PROJECT_DIR / "data" / "processed" / "kmeans_labels_final.parquet"
LABELS_REL, OUTSIDE_REL = "kmeans_labels_final.parquet", "types_outside_sample.parquet"
NB23_PATH = PROJECT_DIR / "notebooks" / "23_type_portraits.md"
STATUSES = ["no_data_in_dataset", "no_dec2024", "too_far"]
FIG = "фигура 2"
LEG_ASSIGNED = "штриховка: тип присвоен вне выборки (цвет — присвоенный тип; МО не входили в обучение)"
LEG_NO_TYPE = "светлая штриховка: тип не присвоен (нет данных за декабрь 2024, МО нет в наборе данных или МО слишком далеко от центров типов)"
C_DEBUG = False
CHANGED = " (порог изменён решением автора 04.10.2026)"


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def hex_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


# ------------------------------------------------------------------ данные
def read_geo(path: Path) -> list[dict]:
    if not path.exists():
        stop(f"нет файла {path}")
    return json.loads(path.read_text(encoding="utf-8"))["features"]


def type_names() -> list[str]:
    """Технические имена типов: full_type_name из src/23_type_portraits.py (вызовы повторяют начало main шага 23)."""
    s23 = importlib.import_module("23_type_portraits")
    d = s23.step18.load()
    s23.step18.clean_rosstat(d, [])
    sup, _ = s23.step21.supply_values(d["ids"])
    D = s23.build_frame(d, sup)
    z = s23.z_profile(d, D)
    za = s23.alr_profile(d, D)
    return [s23.full_type_name(D, z, za, t) for t in range(C.FINAL_K)]


def nb23_names() -> list[str]:
    out = []
    for line in NB23_PATH.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"- Имя типа: (.*) \(расчёт шага 23\)\.$", line)
        if m:
            out.append(m.group(1))
    return out


def feature_parts(f: dict, shift: bool):
    """Части МО как списки колец; долготы меньше 0 сдвигаются на +360. Возвращает (части, колец с разрезом)."""
    parts, cut = [], 0
    for rings in f["geometry"]["coordinates"]:
        rr = []
        for ring in rings:
            a = np.array(ring, float)
            if shift:
                a[a[:, 0] < 0, 0] += 360.0
            if np.abs(np.diff(a[:, 0])).max() > 180:   # контур, пересекающий меридиан 180 скачком, считался бы разрезанным
                cut += 1
            rr.append(a)
        parts.append(rr)
    return parts, cut


def mo_geom(f: dict, shift: bool) -> MultiPolygon:
    parts, _ = feature_parts(f, shift)
    return MultiPolygon([Polygon(rr[0], rr[1:]) for rr in parts])


def to_path(parts: list) -> MPath:
    """Составной контур МО: оболочки против часовой стрелки, отверстия по часовой (orient), заливка по правилу nonzero."""
    verts, codes = [], []
    for rings in parts:
        poly = orient(Polygon(rings[0], rings[1:]), 1.0)
        for r in [poly.exterior, *poly.interiors]:
            c = np.asarray(r.coords)
            verts.append(c)
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
    return MPath(np.vstack(verts), codes)


def style(cls: int, typ) -> dict:
    lw = C.MAP_BORDER_LINEWIDTH_PT
    if cls == 1:
        col = C.MAP_TYPE_COLORS[typ]
        return dict(facecolor=col, edgecolor=col, linewidth=lw)
    if cls == 2:
        col = C.MAP_TYPE_COLORS[typ]
        return dict(facecolor=col, edgecolor=col, linewidth=lw, hatch=C.MAP_ASSIGNED_HATCH, hatchcolor=C.MAP_ASSIGNED_HATCH_COLOR)
    return dict(facecolor=C.MAP_NO_TYPE_FACE, edgecolor=C.MAP_NO_TYPE_FACE, linewidth=lw, hatch=C.MAP_NO_TYPE_HATCH,
                hatchcolor=C.MAP_NO_TYPE_HATCH_COLOR)


def draw_features(ax, feats: list[dict], klass: dict, ktype: dict, shift: bool) -> None:
    order = {1: 0, 3: 1, 2: 2}   # от больших классов к малым
    items = sorted(feats, key=lambda f: (order[klass[int(f["properties"]["territory_id"])]], int(f["properties"]["territory_id"])))
    for f in items:
        i = int(f["properties"]["territory_id"])
        parts, _ = feature_parts(f, shift)
        ax.add_patch(PathPatch(to_path(parts), **style(klass[i], ktype.get(i))))


def extent(feats: list[dict], shift: bool):
    xs, ys = [], []
    for f in feats:
        parts, _ = feature_parts(f, shift)
        for rings in parts:
            xs.append(rings[0][:, 0]); ys.append(rings[0][:, 1])
    x, y = np.concatenate(xs), np.concatenate(ys)
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def read_caption() -> str:
    text = P(C.MAP_ATTRIBUTION_PATH).read_text(encoding="utf-8").split("\n")
    if C.MAP_ATTRIBUTION_CAPTION_HEADING not in text:
        stop(f"в ATTRIBUTION.md нет заголовка {C.MAP_ATTRIBUTION_CAPTION_HEADING}")
    k = text.index(C.MAP_ATTRIBUTION_CAPTION_HEADING) + 1
    para = []
    while k < len(text) and not text[k].strip():
        k += 1
    while k < len(text) and text[k].strip():
        para.append(text[k].strip()); k += 1
    if not para:
        stop("под заголовком подписи в ATTRIBUTION.md пусто")
    return " ".join(para)


# ------------------------------------------------------------------ рисунок
def build_figure(ctx: dict) -> dict:
    plt.rcParams.update({"font.family": C.MAP_FONT, "svg.fonttype": "path", "svg.hashsalt": C.MAP_SVG_HASHSALT,
                         "hatch.linewidth": C.MAP_HATCH_LINEWIDTH_PT, "axes.linewidth": C.MAP_INSET_FRAME_LW})
    W, H = C.MAP_FIG_SIZE_IN
    fig = plt.figure(figsize=(W, H), dpi=C.MAP_DPI, facecolor="white")
    fx = lambda inch: inch / W      # noqa: E731
    fy = lambda inch: inch / H      # noqa: E731
    aspect = 1.0 / np.cos(np.radians(C.MAP_PROJECTION_LAT_DEG))
    cx0, cx1, cy0, cy1 = ctx["extent_nat"]
    pad = 0.5
    x0, x1, y0, y1 = cx0 - pad, cx1 + pad, cy0 - pad, cy1 + pad
    mr = C.MAP_MAP_RECT_IN
    map_h = mr["width"] * (y1 - y0) * aspect / (x1 - x0)
    ax = fig.add_axes([fx(mr["left"]), fy(mr["top"] - map_h), fx(mr["width"]), fy(map_h)])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect(aspect, adjustable="box"); ax.axis("off")
    draw_features(ax, ctx["nat"], ctx["klass"], ctx["ktype"], True)
    title = fig.text(0.5, fy(C.MAP_TITLE_TOP_IN), C.MAP_TITLE, ha="center", va="top", fontsize=C.MAP_TITLE_FONTSIZE)
    fig.canvas.draw()
    insets, leaders = [], []
    for key, cfgd, label in (("moscow", C.MAP_MOSCOW_INSET_IN, "Москва"), ("spb", C.MAP_SPB_INSET_IN, "Санкт-Петербург")):
        feats = [f for f in ctx["cit"] if f["properties"]["city"] == key]
        ex0, ex1, ey0, ey1 = extent(feats, False)
        px_, py_ = (ex1 - ex0) * C.MAP_INSET_PAD_FRACTION, (ey1 - ey0) * C.MAP_INSET_PAD_FRACTION
        ex0, ex1, ey0, ey1 = ex0 - px_, ex1 + px_, ey0 - py_, ey1 + py_
        phi = np.radians((ey0 + ey1) / 2)
        hh = cfgd["width"] * (ey1 - ey0) / ((ex1 - ex0) * np.cos(phi))
        iax = fig.add_axes([fx(cfgd["left"]), fy(cfgd["bottom"]), fx(cfgd["width"]), fy(hh)])
        iax.set_xlim(ex0, ex1); iax.set_ylim(ey0, ey1); iax.set_aspect(1.0 / np.cos(phi), adjustable="box")
        iax.set_xticks([]); iax.set_yticks([])
        for sp in iax.spines.values():
            sp.set_edgecolor(C.MAP_INSET_FRAME_COLOR)
        draw_features(iax, feats, ctx["klass"], ctx["ktype"], False)
        ititle = iax.set_title(f"{label}: {len(feats)} МО", loc="left", fontsize=C.MAP_INSET_FONTSIZE, pad=2)
        rect = Rectangle((ex0, ey0), ex1 - ex0, ey1 - ey0, fill=False, edgecolor=C.MAP_INSET_FRAME_COLOR, linewidth=C.MAP_INSET_FRAME_LW)
        ax.add_patch(rect)
        p1 = fig.transFigure.transform((fx(cfgd["left"] + cfgd["width"]), fy(cfgd["bottom"] + hh / 2)))   # правая сторона врезки
        corners = {"bottom-left": ey0, "top-left": ey1, "left-middle": (ey0 + ey1) / 2}                    # левая сторона рамки на карте
        cands = {k: np.array(ax.transData.transform((ex0, v))) for k, v in corners.items()}
        mode = C.MAP_LEADER_RECT_CORNER
        p0 = min((cands["bottom-left"], cands["top-left"]), key=lambda c: float(np.hypot(*(c - p1)))) if mode == "nearest" else cands[mode]
        n = W * C.MAP_DPI, H * C.MAP_DPI
        line = Line2D([p0[0] / n[0], p1[0] / n[0]], [p0[1] / n[1], p1[1] / n[1]], transform=fig.transFigure,
                      color="black", linewidth=C.MAP_LEADER_LW, path_effects=[pe.withStroke(linewidth=C.MAP_LEADER_HALO_LW, foreground="white")])
        fig.add_artist(line)
        insets.append({"key": key, "ax": iax, "title": ititle, "bounds": (ex0, ex1, ey0, ey1), "height_in": hh, "feats": feats, "label": label,
                       "rect_end_px": p0})
        leaders.append(line)
    handles = [Patch(facecolor=C.MAP_TYPE_COLORS[t], edgecolor=C.MAP_TYPE_COLORS[t], linewidth=C.MAP_BORDER_LINEWIDTH_PT,
                     label=f"тип {t}: {ctx['names'][t]}") for t in range(C.FINAL_K)]
    handles.append(Patch(label=LEG_ASSIGNED, **style(2, 0)))
    handles.append(Patch(label=LEG_NO_TYPE, **style(3, None)))
    ax_bottom_in = mr["top"] - map_h
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, fy(ax_bottom_in - C.MAP_LEGEND_GAP_IN)),
                        ncol=C.MAP_LEGEND_NCOL, fontsize=C.MAP_LEGEND_FONTSIZE, frameon=False, handlelength=2.2, handleheight=1.0,
                        labelspacing=0.45, columnspacing=1.6)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    leg_bb = legend.get_window_extent(rend)
    cap_top_in = leg_bb.y0 / C.MAP_DPI - C.MAP_CAPTION_GAP_IN
    cap_text = "\n".join(textwrap.wrap(ctx["caption"], C.MAP_CAPTION_WRAP_CHARS) + textwrap.wrap(ctx["extra"], C.MAP_CAPTION_WRAP_CHARS))
    caption = fig.text(fx(C.MAP_CAPTION_LEFT_IN), fy(cap_top_in), cap_text, ha="left", va="top", fontsize=C.MAP_CAPTION_FONTSIZE, linespacing=1.25)
    fig.canvas.draw()
    return {"fig": fig, "ax": ax, "title": title, "insets": insets, "leaders": leaders, "legend": legend, "caption": caption, "handles": handles,
            "map_h": map_h, "contour": (cx0, cx1, cy0, cy1), "rend": rend}


def bb_in(bb) -> tuple:
    return tuple(v / C.MAP_DPI for v in (bb.x0, bb.y0, bb.x1, bb.y1))


def overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def layout_checks(F: dict, png: np.ndarray | None) -> list[dict]:
    """Измеримые критерии компоновки по get_window_extent и массиву PNG (дюймы от нижнего левого угла холста)."""
    W, H = C.MAP_FIG_SIZE_IN
    rend = F["rend"]
    m = C.MAP_MIN_MARGIN_IN
    el = {"заголовок": bb_in(F["title"].get_window_extent(rend)), "ось карты": bb_in(F["ax"].get_window_extent(rend)),
          "легенда": bb_in(F["legend"].get_window_extent(rend)), "подпись": bb_in(F["caption"].get_window_extent(rend))}
    for s in F["insets"]:
        el[f"врезка {s['label']} с заголовком"] = bb_in(s["ax"].get_tightbbox(rend))
    out = []

    def add(name, value, ok):
        out.append({"name": name, "value": value, "ok": bool(ok)})
    # (а) отступ от края холста
    for k, b in el.items():
        mg = min(b[0], b[1], W - b[2], H - b[3])
        add(f"(а) отступ от края холста: {k}, дюймов (не менее {m})", round(mg, 3), mg >= m)
    # (б) пересечения
    if C_DEBUG:
        for k, b in el.items():
            print("  bbox", k, tuple(round(v, 3) for v in b))
    names = list(el)
    bad = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if overlap(el[a], el[b])]
    add("(б) пересечений между элементами (заголовок, ось карты, врезки, легенда, подпись)", len(bad) if not bad else f"{len(bad)}: " + "; ".join(f"{a_} x {b_}" for a_, b_ in bad), not bad)
    # (в) зазоры
    leg, cap, axb = el["легенда"], el["подпись"], el["ось карты"]
    g1, g2 = axb[1] - leg[3], leg[1] - cap[3]
    add(f"(в) зазор ось карты - легенда, дюймов (0 - {C.MAP_MAX_GAP_IN})", round(g1, 3), 0 <= g1 <= C.MAP_MAX_GAP_IN)
    add(f"(в) зазор легенда - подпись, дюймов (0 - {C.MAP_MAX_GAP_IN})", round(g2, 3), 0 <= g2 <= C.MAP_MAX_GAP_IN)
    # прочее по тексту компоновки
    t = el["заголовок"]
    add(f"заголовок по центру холста: смещение центра, дюймов (не более {C.MAP_TITLE_CENTER_TOL_IN})", round(abs((t[0] + t[2]) / 2 - W / 2), 4),
        abs((t[0] + t[2]) / 2 - W / 2) <= C.MAP_TITLE_CENTER_TOL_IN)
    add("размер заголовка больше заголовков врезок", f"{C.MAP_TITLE_FONTSIZE} > {C.MAP_INSET_FONTSIZE}", C.MAP_TITLE_FONTSIZE > C.MAP_INSET_FONTSIZE)
    add(f"легенда по центру холста: смещение центра, дюймов (не более {C.MAP_TITLE_CENTER_TOL_IN})", round(abs((leg[0] + leg[2]) / 2 - W / 2), 4),
        abs((leg[0] + leg[2]) / 2 - W / 2) <= C.MAP_TITLE_CENTER_TOL_IN)
    add(f"ширина легенды не больше ширины холста минус поля, дюймов (не более {round(W - 2 * m, 2)})", round(leg[2] - leg[0], 3), leg[2] - leg[0] <= W - 2 * m)
    add(f"подпись в пределах полей: правый край, дюймов (не более {round(W - m, 2)})", round(cap[2], 3), cap[2] <= W - m and cap[0] >= m)
    cx0, cx1, cy0, cy1 = F["contour"]
    right = F["ax"].transData.transform((cx1, cy0))[0] / C.MAP_DPI
    top = F["ax"].transData.transform((cx0, cy1))[1] / C.MAP_DPI
    add(f"самая правая точка контуров не ближе {m} дюйма к правому краю: отступ, дюймов", round(W - right, 3), W - right >= m)
    add(f"верхняя точка контуров не ближе {m} дюйма к верхнему краю: отступ, дюймов", round(H - top, 3), H - top >= m)
    bottoms = [s["ax"].get_window_extent(rend).y0 / C.MAP_DPI for s in F["insets"]]
    add("нижний край нижней врезки не ниже нижнего края оси карты: разность, дюймов", round(min(bottoms) - axb[1], 3), min(bottoms) >= axb[1] - 1e-9)
    # конец линии-выноски на рамке врезки (в пределах 1 пикселя)
    for s, ln in zip(F["insets"], F["leaders"]):
        e = ln.get_transform().transform(np.array([ln.get_xdata()[0], ln.get_ydata()[0]]))
        d = float(np.hypot(*(e - s["rect_end_px"])))
        add(f"линия-выноска {s['label']} заканчивается на рамке врезки на карте: расстояние, пикселей (не более 1)", round(d, 3), d <= 1.0)
    # (г) поля по массиву PNG
    if png is not None:
        nonwhite = (np.round(png[..., :3] * 255) < 255).any(axis=2)
        rows, cols = np.where(nonwhite.any(axis=1))[0], np.where(nonwhite.any(axis=0))[0]
        hpx, wpx = nonwhite.shape
        marg = {"слева": cols.min() / C.MAP_DPI, "справа": (wpx - 1 - cols.max()) / C.MAP_DPI, "сверху": rows.min() / C.MAP_DPI,
                "снизу": (hpx - 1 - rows.max()) / C.MAP_DPI}
        for k, v in marg.items():
            add(f"(г) поле холста {k} по массиву PNG, дюймов ({m} - {C.MAP_PNG_MARGIN_MAX_IN})", round(v, 3), m <= v <= C.MAP_PNG_MARGIN_MAX_IN)
    return out


# ------------------------------------------------------------------ проверка по пикселям
def seg_dist(px: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((px - a) @ ab) / float(ab @ ab), 0, 1)
    return np.linalg.norm(px - (a + t[..., None] * ab), axis=-1)


def pixel_check(F: dict, ctx: dict, png: np.ndarray) -> dict:
    """Цвет пикселя в центре максимальной вписанной окружности каждого МО в каждой оси, где оно нарисовано."""
    arr = np.round(png[..., :3] * 255).astype(float)
    hpx, wpx = arr.shape[:2]
    thr, tol1, tols = C.MAP_PIXEL_CHECK_MIN_RADIUS_PX, C.MAP_PIXEL_TOL_CLASS1, C.MAP_PIXEL_TOL_SEGMENT
    tc = [hex_rgb(h) for h in C.MAP_TYPE_COLORS]
    h2, f3, h3 = hex_rgb(C.MAP_ASSIGNED_HATCH_COLOR), hex_rgb(C.MAP_NO_TYPE_FACE), hex_rgb(C.MAP_NO_TYPE_HATCH_COLOR)
    rec, mism = [], []
    axes = [("карта страны", F["ax"], ctx["nat"], True)] + [(f"врезка {s['label']}", s["ax"], s["feats"], False) for s in F["insets"]]
    for axname, ax, feats, shift in axes:
        T = ax.transData
        for f in feats:
            i = int(f["properties"]["territory_id"])
            g = mo_geom(f, shift)
            line = shapely.maximum_inscribed_circle(g, C.MAP_PIXEL_MIC_TOLERANCE_DEG)
            (cx, cy) = line.coords[0][:2]
            r = line.length
            d0 = np.array(T.transform((cx, cy)))
            dx = np.array(T.transform((cx + 1, cy))) - d0
            dy = np.array(T.transform((cx, cy + 1))) - d0
            r_px = r * min(np.hypot(*dx), np.hypot(*dy))
            cls, typ = ctx["klass"][i], ctx["ktype"].get(i)
            row = {"territory_id": i, "class": cls, "type": typ, "axis": axname, "radius_px": float(r_px), "checked": False, "ok": None,
                   "area": float(g.area)}
            if r_px >= thr:
                col, rw = int(d0[0]), int(hpx - d0[1])
                got = arr[rw, col]
                if cls == 1:
                    ok = bool(np.all(np.abs(got - tc[typ]) <= tol1)); exp = C.MAP_TYPE_COLORS[typ]
                elif cls == 2:
                    ok = bool(seg_dist(got, tc[typ], h2) <= tols); exp = f"{C.MAP_TYPE_COLORS[typ]}..{C.MAP_ASSIGNED_HATCH_COLOR}"
                else:
                    ok = bool(seg_dist(got, f3, h3) <= tols); exp = f"{C.MAP_NO_TYPE_FACE}..{C.MAP_NO_TYPE_HATCH_COLOR}"
                row.update(checked=True, ok=ok)
                if not ok:
                    mism.append({"territory_id": i, "class": cls, "expected": exp, "got": "#%02X%02X%02X" % tuple(int(v) for v in got), "axis": axname})
            rec.append(row)
    df = pd.DataFrame(rec)
    # сведение: доля пикселей оси карты, не близких ни к одному допустимому цвету
    b = F["ax"].get_window_extent(F["rend"])
    sub = arr[int(hpx - b.y1):int(hpx - b.y0), int(b.x0):int(b.x1)].reshape(-1, 3)
    dmin = np.full(len(sub), np.inf)
    for c in tc + [f3, h2, h3, np.array([255.0, 255.0, 255.0])]:
        dmin = np.minimum(dmin, np.abs(sub - c).max(axis=1))
    near = dmin <= tol1
    for k in range(C.FINAL_K):
        near |= seg_dist(sub, tc[k], h2) <= tols
    near |= seg_dist(sub, f3, h3) <= tols
    return {"df": df, "mismatch": mism, "boundary_share": float((~near).mean()), "axis_pixels": int(len(sub))}


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-bbox", action="store_true", help="печатать границы элементов компоновки")
    ap.add_argument("--layout-only", action="store_true", help="только компоновка (без проверок по пикселям и без записи)")
    args = ap.parse_args()
    global C_DEBUG
    C_DEBUG = args.debug_bbox
    for rel in (C.MAP_GEO_NATIONAL_PATH, C.MAP_GEO_CITIES_PATH, C.MAP_TYPES_OUTSIDE_PATH, C.MAP_ATTRIBUTION_PATH, C.MAP_GEO_REPORT_PATH):
        if not P(rel).exists():
            stop(f"нет файла {rel} (шаги 24б и 24c)")
    nat = read_geo(P(C.MAP_GEO_NATIONAL_PATH))
    cit = read_geo(P(C.MAP_GEO_CITIES_PATH))
    labels = pd.read_parquet(LABELS_PATH)
    out = pd.read_parquet(P(C.MAP_TYPES_OUTSIDE_PATH))
    controls = []

    def add(name, expect, got, ok, note=""):
        controls.append({"name": name + note, "expect": expect, "got": got, "ok": bool(ok)})

    # --- классы
    nat_ids = [int(f["properties"]["territory_id"]) for f in nat]
    props = {int(f["properties"]["territory_id"]): f["properties"] for f in nat}
    sample_ids = set(labels["territory_id"].astype(int))
    ktype_sample = dict(zip(labels["territory_id"].astype(int), labels["cluster"].astype(int)))
    asg = out[out["status"] == "assigned"]
    ktype_asg = dict(zip(asg["territory_id"].astype(int), asg["type"].astype(int)))
    notype = out[out["status"] != "assigned"]
    cls1, cls2, cls3 = sample_ids, set(ktype_asg), set(notype["territory_id"].astype(int))
    klass = {**{i: 1 for i in cls1}, **{i: 2 for i in cls2}, **{i: 3 for i in cls3}}
    ktype = {**ktype_sample, **ktype_asg}
    sample_by_type = np.bincount(labels["cluster"].astype(int), minlength=C.FINAL_K).tolist()
    asg_by_type = np.bincount(asg["type"].astype(int), minlength=C.FINAL_K).tolist()
    nt_counts = {s: int((notype["status"] == s).sum()) for s in STATUSES}
    p95 = C.ASSIGN_SAMPLE_DIST["p95"]
    beyond = int((asg["dist_nearest"] > p95).sum())
    in_sample_prop = {i: bool(p["in_sample"]) for i, p in props.items()}
    add("полигонов в mo_national (уникальных territory_id)", C.MAP_EXPECT_ACTIVE, len(set(nat_ids)), len(nat_ids) == len(set(nat_ids)) == C.MAP_EXPECT_ACTIVE)
    add("сумма классов 1 + 2 + 3", C.MAP_EXPECT_ACTIVE, len(cls1) + len(cls2) + len(cls3), len(cls1) + len(cls2) + len(cls3) == C.MAP_EXPECT_ACTIVE)
    add("пересечения классов (пар)", 0, len(cls1 & cls2) + len(cls1 & cls3) + len(cls2 & cls3), not (cls1 & cls2 or cls1 & cls3 or cls2 & cls3))
    add("territory_id в mo_national = выборка + types_outside_sample", True, set(nat_ids) == cls1 | set(out["territory_id"].astype(int)),
        set(nat_ids) == cls1 | set(out["territory_id"].astype(int)) and len(out) == C.MAP_EXPECT_ACTIVE - len(cls1))
    add("in_sample в mo_national согласован с выборкой", True, all(in_sample_prop[i] == (i in sample_ids) for i in nat_ids),
        all(in_sample_prop[i] == (i in sample_ids) for i in nat_ids))
    add("класс 1 по типам 0–5", C.MAP_EXPECT_SAMPLE_BY_TYPE, sample_by_type, sample_by_type == C.MAP_EXPECT_SAMPLE_BY_TYPE)
    add("класс 2 по типам 0–5", C.MAP_EXPECT_ASSIGNED_BY_TYPE, asg_by_type, asg_by_type == C.MAP_EXPECT_ASSIGNED_BY_TYPE)
    add("класс 3 по статусам", C.MAP_EXPECT_NO_TYPE, nt_counts, nt_counts == C.MAP_EXPECT_NO_TYPE)
    add(f"из присвоенных МО с dist_nearest больше p95 выборки ({p95})", C.MAP_EXPECT_BEYOND_P95, beyond, beyond == C.MAP_EXPECT_BEYOND_P95)
    names = type_names()
    nb = nb23_names()
    add("технические имена типов (full_type_name) совпадают с notebooks/23_type_portraits.md", True, names == nb, names == nb and len(names) == C.FINAL_K)

    # --- 1a. сдвиг долгот
    shift_rows, cut_after = [], 0
    for f in nat:
        raw = np.concatenate([np.array(r)[:, 0] for rings in f["geometry"]["coordinates"] for r in rings])
        _, cut1 = feature_parts(f, True)
        cut_after += cut1
        if (raw < 0).any():
            sh = np.where(raw < 0, raw + 360.0, raw)
            p = f["properties"]
            shift_rows.append({"territory_id": int(p["territory_id"]), "name": p["name"], "region_name": p["region_name"], "lon_min_before": float(raw.min()),
                               "lon_max_before": float(raw.max()), "lon_min_after": float(sh.min()), "lon_max_after": float(sh.max())})
    shifted_ids = [r["territory_id"] for r in shift_rows]
    shift_regions = sorted({r["region_name"] for r in shift_rows})
    lo_b, hi_b = C.MAP_LON_BEFORE_RANGE
    max_after = max(r["lon_max_after"] for r in shift_rows)
    min_after = min(r["lon_min_after"] for r in shift_rows)
    add(f"сдвиг долгот: region_name у всех сдвинутых МО содержит «{C.MAP_SHIFT_REGION_SUBSTRING}»", True, shift_regions,
        all(C.MAP_SHIFT_REGION_SUBSTRING in r["region_name"] for r in shift_rows) and len(shift_rows) > 0)
    add("сдвиг долгот: максимум после сдвига не больше", C.MAP_LON_AFTER_MAX, round(max_after, 4), max_after <= C.MAP_LON_AFTER_MAX, CHANGED)
    add("сдвиг долгот: минимум после сдвига не меньше", C.MAP_LON_AFTER_MIN, round(min_after, 4), min_after >= C.MAP_LON_AFTER_MIN, CHANGED)
    add("сдвиг долгот: до сдвига все долготы в пределах", f"{lo_b}, {hi_b}", f"{min(r['lon_min_before'] for r in shift_rows)}, {max(r['lon_max_before'] for r in shift_rows)}",
        all(lo_b <= r["lon_min_before"] and r["lon_max_before"] <= hi_b for r in shift_rows), CHANGED)
    shifted_in_sample = [i for i in shifted_ids if i in sample_ids]
    cities_neg = sum(1 for f in cit if any((np.array(r)[:, 0] < 0).any() for rings in f["geometry"]["coordinates"] for r in rings))
    add("полигонов выборки с долготой меньше 0", 0, len(shifted_in_sample), len(shifted_in_sample) == 0)
    add("контуров с разрезом после сдвига (скачок долготы больше 180)", 0, cut_after, cut_after == 0)
    # --- 1b. МО 409
    c409 = C.MAP_CHECK_409
    i409 = c409["territory_id"]
    r409 = out[out["territory_id"] == i409]
    ok409 = (c409["name_substring"] in props[i409]["name"] and klass.get(i409) == 3 and len(r409) == 1 and r409["status"].iloc[0] == c409["status"]
             and pd.isna(r409["type"].iloc[0]) and i409 not in ktype_asg and i409 not in sample_ids)
    add(f"МО {i409}: название содержит «{c409['name_substring']}», класс 3, status {c409['status']}, type пусто, не в присвоенных и не в выборке", True, props[i409]["name"], ok409)
    # --- 1c. Чукотский АО по классам (для сведения)
    chuk = {k: sum(1 for i, p in props.items() if C.MAP_SHIFT_REGION_SUBSTRING in p["region_name"] and klass[i] == k) for k in (1, 2, 3)}
    # --- 1d. регионы
    n_reg_all = len({p["region_name"] for p in props.values()})
    n_reg_smp = len({p["region_name"] for i, p in props.items() if i in sample_ids})
    add("регионов в mo_national (различных region_name)", C.MAP_EXPECT_REGIONS_ALL, n_reg_all, n_reg_all == C.MAP_EXPECT_REGIONS_ALL)
    add("регионов среди МО выборки", C.MAP_EXPECT_REGIONS_SAMPLE, n_reg_smp, n_reg_smp == C.MAP_EXPECT_REGIONS_SAMPLE)
    add("врезка Москвы, полигонов", C.MAP_EXPECT_INSET_POLYGONS["moscow"], sum(1 for f in cit if f["properties"]["city"] == "moscow"),
        sum(1 for f in cit if f["properties"]["city"] == "moscow") == C.MAP_EXPECT_INSET_POLYGONS["moscow"])
    add("врезка Санкт-Петербурга, полигонов", C.MAP_EXPECT_INSET_POLYGONS["spb"], sum(1 for f in cit if f["properties"]["city"] == "spb"),
        sum(1 for f in cit if f["properties"]["city"] == "spb") == C.MAP_EXPECT_INSET_POLYGONS["spb"])
    add("полигонов врезок с долготой меньше 0", 0, cities_neg, cities_neg == 0)

    # --- тексты
    region_text = ", ".join(shift_regions)
    caption = read_caption()
    extra = (f"Проекция: равнопромежуточная с поправкой aspect = 1/cos {C.MAP_PROJECTION_LAT_DEG:.0f}°; проекция искажает формы и площади на севере. "
             f"Долготы меньше 0 сдвинуты на +360 у {len(shifted_ids)} МО. Классы МО из {C.MAP_EXPECT_ACTIVE}: тип из рабочей выборки {len(cls1)}, "
             f"тип присвоен вне выборки {len(cls2)} (дальше 95-го процентиля расстояний выборки {beyond} из {len(cls2)}), тип не присвоен {len(cls3)} "
             f"(нет в данных расходов {nt_counts['no_data_in_dataset']}, нет данных за декабрь 2024 {nt_counts['no_dec2024']}, "
             f"расстояние до центроида больше максимума по выборке {nt_counts['too_far']}). "
             "Присвоенный МО, граничащий с МО выборки того же типа, отличается от соседей только штриховкой. "
             "Крупные малонаселённые МО Севера и Дальнего Востока занимают на карте много места: площадь не равна населению. "
             f"Справочник границ содержит {n_reg_all} регионов, рабочая выборка — {n_reg_smp}.")
    ctx = {"nat": nat, "cit": cit, "klass": klass, "ktype": ktype, "names": names, "caption": caption, "extra": extra,
           "extent_nat": extent(nat, True)}

    # --- рисунок и компоновка
    F = build_figure(ctx)
    fig = F["fig"]
    if args.layout_only:
        for c in layout_checks(F, None):
            print(f"| {c['name']} | {c['value']} | {'пройден' if c['ok'] else 'НЕ ПРОЙДЕН'} |")
        return
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=C.MAP_DPI, facecolor="white", metadata={"Software": None})
    png_bytes = png.getvalue()
    arr = mpimg.imread(io.BytesIO(png_bytes), format="png")
    lay = layout_checks(F, arr)
    for c in lay:
        add("компоновка: " + c["name"], "выполнено", c["value"], c["ok"])
    add("элементов легенды", C.MAP_EXPECT_LEGEND_ITEMS, len(F["handles"]), len(F["handles"]) == C.MAP_EXPECT_LEGEND_ITEMS)
    texts = [t.get_text() for t in fig.findobj(Text) if t.get_text().strip()]
    bad_words = [t for t in texts if any(w in t.lower() for w in C.MAP_FORBIDDEN_WORDS)]
    bad_br = [t for t in texts if "[" in t or "]" in t]
    add("запрещённых слов в текстах рисунка", 0, len(bad_words), not bad_words)
    add("квадратных скобок в текстах рисунка", 0, len(bad_br), not bad_br)

    # --- проверка по пикселям
    pc = pixel_check(F, ctx, arr)
    df = pc["df"]
    allowed = set(json.loads(P(C.MAP_GEO_REPORT_PATH).read_text(encoding="utf-8"))["original"]["coverage_invalid_edge_ids"])
    for pair in json.loads(P(C.MAP_GEO_REPORT_PATH).read_text(encoding="utf-8"))["original"]["overlap_pair_ids"]:
        allowed |= set(pair)
    chk_any = df[df["checked"]].groupby("territory_id").size().index
    checked_ids = set(int(i) for i in chk_any)
    nat_df = df[df["axis"] == "карта страны"]
    area_tot = nat_df["area"].sum()
    area_by_cls = {k: float(nat_df[(nat_df["class"] == k) & nat_df["territory_id"].isin(checked_ids)]["area"].sum() / nat_df[nat_df["class"] == k]["area"].sum()) for k in (1, 2, 3)}
    area_share = float(nat_df[nat_df["territory_id"].isin(checked_ids)]["area"].sum() / area_tot)
    typ_checked = [len(checked_ids & {i for i in cls1 if ktype[i] == t}) for t in range(C.FINAL_K)]
    cls_checked = {k: len(checked_ids & s) for k, s in ((1, cls1), (2, cls2), (3, cls3))}
    mism = pc["mismatch"]
    mism_not_allowed = [m for m in mism if m["territory_id"] not in allowed]
    add("проверка по пикселям: МО проверено хотя бы в одной оси", f"не менее {C.MAP_EXPECT_PIXEL_CHECKED_MIN}", len(checked_ids), len(checked_ids) >= C.MAP_EXPECT_PIXEL_CHECKED_MIN)
    add("проверка по пикселям: класс 1, проверено по каждому типу 0–5", f"не менее {C.MAP_EXPECT_PIXEL_TYPE_MIN}", typ_checked, min(typ_checked) >= C.MAP_EXPECT_PIXEL_TYPE_MIN)
    add("проверка по пикселям: класс 2, проверено МО", f"не менее {C.MAP_EXPECT_PIXEL_CLASS_MIN}", cls_checked[2], cls_checked[2] >= C.MAP_EXPECT_PIXEL_CLASS_MIN)
    add("проверка по пикселям: класс 3, проверено МО", f"не менее {C.MAP_EXPECT_PIXEL_CLASS_MIN}", cls_checked[3], cls_checked[3] >= C.MAP_EXPECT_PIXEL_CLASS_MIN)
    add("проверка по пикселям: доля площади карты страны на проверенных МО", f"не менее {C.MAP_EXPECT_PIXEL_AREA_SHARE_MIN}", round(area_share, 4), area_share >= C.MAP_EXPECT_PIXEL_AREA_SHARE_MIN)
    add("проверка по пикселям: несовпадений", f"не более {C.MAP_MAX_PIXEL_MISMATCH}", len(mism), len(mism) <= C.MAP_MAX_PIXEL_MISMATCH)
    add("проверка по пикселям: каждое несовпадение входит в список неверных рёбер или перекрытий (geometry_report.json)", 0, len(mism_not_allowed), not mism_not_allowed)

    # --- figure data
    rows = []

    def row(panel, key, typ, value, n, src, col):
        rows.append({"фигура": FIG, "панель": panel, "ключ": key, "тип": typ, "значение": float(value), "n": int(n), "источник_файл": src,
                     "источник_блок_или_колонка": col})
    row("классы", "класс 1: тип из рабочей выборки", None, len(cls1), len(cls1), LABELS_REL, "territory_id")
    row("классы", "класс 2: тип присвоен вне выборки", None, len(cls2), len(cls2), OUTSIDE_REL, "status = assigned")
    row("классы", "класс 3: тип не присвоен", None, len(cls3), len(cls3), OUTSIDE_REL, "status != assigned")
    row("классы", "всего МО", None, len(nat_ids), len(nat_ids), Path(C.MAP_GEO_NATIONAL_PATH).name, "territory_id")
    for t in range(C.FINAL_K):
        row("классы по типам", "класс 1: тип из рабочей выборки", t, sample_by_type[t], len(cls1), LABELS_REL, "cluster")
    for t in range(C.FINAL_K):
        row("классы по типам", "класс 2: тип присвоен вне выборки", t, asg_by_type[t], len(cls2), OUTSIDE_REL, "type (status = assigned)")
    for s in STATUSES:
        row("класс 3 по статусам", s, None, nt_counts[s], len(cls3), OUTSIDE_REL, "status")
    row("проверки", f"полигонов со сдвигом долготы на +360 ({region_text})", None, len(shifted_ids), len(nat_ids), Path(C.MAP_GEO_NATIONAL_PATH).name, "geometry; region_name")
    row("проверки", "контуров с разрезом после сдвига", None, cut_after, len(nat_ids), Path(C.MAP_GEO_NATIONAL_PATH).name, "geometry")
    row("проверки", f"присвоенных МО дальше p95 расстояний выборки ({p95})", None, beyond, len(cls2), OUTSIDE_REL, "dist_nearest")
    row("проверки", f"МО {i409}: условия п. 1b выполнены (1 = да)", None, int(ok409), 1, Path(C.MAP_GEO_NATIONAL_PATH).name + "; " + OUTSIDE_REL, "name; status; type")
    row("проверки", "регионов в справочнике границ (различных region_name)", None, n_reg_all, len(nat_ids), Path(C.MAP_GEO_NATIONAL_PATH).name, "region_name")
    row("проверки", "регионов в рабочей выборке", None, n_reg_smp, len(cls1), Path(C.MAP_GEO_NATIONAL_PATH).name, "region_name; in_sample")
    for axname in ["карта страны"] + [f"врезка {s['label']}" for s in F["insets"]]:
        d = df[df["axis"] == axname]
        row("проверка по пикселям", f"проверено МО: {axname}", None, int(d["checked"].sum()), len(d), "src/24d_map.py", "maximum_inscribed_circle, цвет пикселя")
        row("проверка по пикселям", f"пропущено (радиус меньше порога): {axname}", None, int((~d["checked"]).sum()), len(d), "src/24d_map.py", "радиус меньше порога")
        row("проверка по пикселям", f"несовпадений: {axname}", None, int((d["checked"] & (d["ok"] == False)).sum()), int(d["checked"].sum()), "src/24d_map.py", "цвет вне допуска")  # noqa: E712
    row("проверка по пикселям", "проверено МО хотя бы в одной оси", None, len(checked_ids), len(nat_ids), "src/24d_map.py", "объединение осей")
    row("проверка по пикселям", "несовпадений всего", None, len(mism), len(checked_ids), "src/24d_map.py", "объединение осей")
    row("врезки", "полигонов во врезке: moscow", None, sum(1 for f in cit if f["properties"]["city"] == "moscow"), sum(1 for f in cit if f["properties"]["city"] == "moscow"), Path(C.MAP_GEO_CITIES_PATH).name, "city")
    row("врезки", "полигонов во врезке: spb", None, sum(1 for f in cit if f["properties"]["city"] == "spb"), sum(1 for f in cit if f["properties"]["city"] == "spb"), Path(C.MAP_GEO_CITIES_PATH).name, "city")
    row("легенда", "элементов легенды", None, len(F["handles"]), len(F["handles"]), "src/24d_map.py", "легенда")
    fd = pd.DataFrame(rows).astype({"фигура": "string", "панель": "string", "ключ": "string", "тип": "Int8", "значение": "float64", "n": "int32",
                                    "источник_файл": "string", "источник_блок_или_колонка": "string"})
    # второй способ подсчёта по исходным таблицам и geojson
    reg2_all = len(pd.Series([p["region_name"] for p in props.values()]).unique())
    reg2_smp = pd.DataFrame(list(props.values()))[lambda d_: d_["in_sample"]]["region_name"].nunique()
    sh2 = sum(1 for f in nat if min(min(np.array(r)[:, 0]) for rings in f["geometry"]["coordinates"] for r in rings) < 0)
    d2 = df.groupby("axis").agg(n=("territory_id", "size"), c=("checked", "sum"))
    chk = [("класс 1", len(cls1), int(labels["territory_id"].nunique())), ("класс 2", len(cls2), int((out["status"] == "assigned").sum())),
           ("класс 3", len(cls3), int((out["status"] != "assigned").sum())), ("всего", len(nat_ids), sum(1 for _ in nat)),
           ("класс 1 по типам", sample_by_type, labels.groupby("cluster").size().reindex(range(C.FINAL_K)).tolist()),
           ("класс 2 по типам", asg_by_type, asg.groupby("type").size().reindex(range(C.FINAL_K)).fillna(0).astype(int).tolist()),
           ("геометрии с in_sample", int(sum(in_sample_prop.values())), len(cls1)), ("регионов всего", n_reg_all, reg2_all),
           ("регионов в выборке", n_reg_smp, reg2_smp), ("МО со сдвигом", len(shifted_ids), sh2),
           ("МО в осях (карта + врезки)", int(d2["n"].sum()), len(nat_ids) + len(cit)), ("несовпадений = списку", len(mism), int(((df["ok"] == False)).sum()))]  # noqa: E712
    bad_chk = [c for c in chk if c[1] != c[2]]
    add("значения figure_data сверены со вторым способом подсчёта", 0, len(bad_chk), not bad_chk)

    # --- таблица контролей
    fmt = lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else (str(v).replace("[", "(").replace("]", ")") if isinstance(v, dict) else str(v))  # noqa: E731
    print("МО со сдвигом долготы: territory_id | название | регион | долготы до (мин / макс) | после (мин / макс)")
    for r in shift_rows:
        print(f"{r['territory_id']} | {r['name']} | {r['region_name']} | {r['lon_min_before']:.4f} / {r['lon_max_before']:.4f} | {r['lon_min_after']:.4f} / {r['lon_max_after']:.4f}")
    print(f"МО Чукотского АО по классам 1, 2, 3: {chuk[1]}, {chuk[2]}, {chuk[3]}; МО {i409}: {props[i409]['name']}; регионов: {n_reg_all} и {n_reg_smp}")
    print("| контроль | ожидание | получено | результат |")
    for c in controls:
        print(f"| {c['name']} | {fmt(c['expect'])} | {fmt(c['got'])} | {'пройден' if c['ok'] else 'НЕ ПРОЙДЕН'} |")
    rad = df["radius_px"].to_numpy()
    print("радиус вписанной окружности, пикселей (все МО во всех осях): " + ", ".join(f"p{q} {np.percentile(rad, q):.2f}" for q in (5, 10, 25, 50, 75, 90)))
    unchecked = df[~df["checked"]].drop_duplicates("territory_id")
    unchecked = unchecked[~unchecked["territory_id"].isin(checked_ids)]
    print("не проверено по классам:", {k: int((unchecked["class"] == k).sum()) for k in (1, 2, 3)}, "| по типам (классы 1 и 2):",
          {t: int((unchecked["type"] == t).sum()) for t in range(C.FINAL_K)})
    print(f"доля площади на проверенных МО: всего {area_share:.4f}, по классам {({k: round(v, 4) for k, v in area_by_cls.items()})}")
    print(f"граничные пиксели оси карты (не близки ни к одному допустимому цвету): {pc['boundary_share'] * 100:.2f}% из {pc['axis_pixels']}")
    for k, d in df.groupby("axis"):
        print(f"ось {k}: нарисовано МО {len(d)}, проверено {int(d['checked'].sum())}, пропущено {int((~d['checked']).sum())}, несовпадений {int((d['checked'] & (d['ok'] == False)).sum())}")  # noqa: E712
    for m_ in mism:
        print("несовпадение:", m_, "| входит в список геометрии:", m_["territory_id"] in allowed)
    print(f"источник списка: {C.MAP_GEO_REPORT_PATH}, original.coverage_invalid_edge_ids ({len(json.loads(P(C.MAP_GEO_REPORT_PATH).read_text(encoding='utf-8'))['original']['coverage_invalid_edge_ids'])} МО) и original.overlap_pair_ids")
    if not all(c["ok"] for c in controls):
        cand = Path("/tmp/step24d2_candidate")
        cand.mkdir(parents=True, exist_ok=True)
        (cand / "24_fig2_map.png").write_bytes(png_bytes)     # кандидат для осмотра, только во временной папке
        print(f"STOP: контроли не пройдены; файлы репозитория не записаны, кандидат PNG: {cand}")
        raise SystemExit(1)

    # --- запись
    svg = io.BytesIO()
    fig.savefig(svg, format="svg", facecolor="white", metadata={"Date": None, "Creator": None})
    svg_bytes = svg.getvalue()
    svg_ok = len(svg_bytes) <= C.MAP_MAX_SVG_BYTES
    P(C.MAP_PNG_PATH).parent.mkdir(parents=True, exist_ok=True)
    P(C.MAP_PNG_PATH).write_bytes(png_bytes)
    if svg_ok:
        P(C.MAP_SVG_PATH).write_bytes(svg_bytes)
    print(f"PNG: {len(png_bytes)} байт; SVG: {len(svg_bytes)} байт, предел {C.MAP_MAX_SVG_BYTES}: {'записан' if svg_ok else 'НЕ ЗАПИСАН (больше предела)'}")
    fd.to_parquet(P(C.MAP_FIGURE_DATA_PATH), engine="pyarrow", index=False)
    extra_md = {"region_text": region_text, "n_reg_all": n_reg_all, "n_reg_smp": n_reg_smp, "lay": lay, "df": df, "mism": mism, "area_share": area_share,
                "area_by_cls": area_by_cls, "boundary": pc["boundary_share"], "checked": len(checked_ids), "typ_checked": typ_checked, "cls_checked": cls_checked}
    P(C.MAP_REPORT_PATH).write_text(report(fd, caption, extra, names, len(shifted_ids), beyond, len(cls2), svg_ok, len(svg_bytes), controls, nt_counts, extra_md), encoding="utf-8")
    plt.close(fig)
    print("Готово.")


def report(fd, caption, extra, names, n_shift, beyond, n_asg, svg_ok, svg_len, controls, nt, em) -> str:
    fmt = lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else (str(v).replace("[", "(").replace("]", ")") if isinstance(v, dict) else str(v))  # noqa: E731
    L = ["# 24d. Статическая карта типов МО (фигура 2)", "",
         "Сгенерировано `src/24d_map.py`. Файлы: `notebooks/figures/24_fig2_map.png`" + (", `notebooks/figures/24_fig2_map.svg`" if svg_ok else "")
         + ", данные фигуры `data/processed/figure_data_24d.parquet`. Типы подписаны техническими именами (`full_type_name`, шаг 23); "
         "интерпретационные имена на карте не используются.", "",
         f"Заголовок фигуры: {C.MAP_TITLE}.", "", "## Подпись", "", caption, "", extra, "",
         "## Классы МО", "", "| класс | число МО | тип | источник (файл, колонка) |", "|---|---|---|---|"]
    cl = fd[fd["панель"] == "классы"]
    bt = fd[fd["панель"] == "классы по типам"]
    st = fd[fd["панель"] == "класс 3 по статусам"]
    for _, r in cl.iterrows():
        L.append(f"| {r['ключ']} | {int(r['значение'])} | все | {r['источник_файл']}, {r['источник_блок_или_колонка']} |")
    for _, r in bt.iterrows():
        L.append(f"| {r['ключ']} | {int(r['значение'])} | {int(r['тип'])} ({names[int(r['тип'])]}) | {r['источник_файл']}, {r['источник_блок_или_колонка']} |")
    for _, r in st.iterrows():
        L.append(f"| класс 3: тип не присвоен | {int(r['значение'])} | статус {r['ключ']} | {r['источник_файл']}, {r['источник_блок_или_колонка']} |")
    L += ["", "## Что на карте не показано", "",
          "- площади МО и население: на карте не печатаются;",
          "- МО за пределами врезок: внутригородские территории Москвы и Санкт-Петербурга различимы только на врезках, на карте страны их контуры меньше пикселя;",
          f"- МО без типа показаны светлой штриховкой: нет данных за декабрь 2024 или в наборе данных ({nt['no_dec2024'] + nt['no_data_in_dataset']} МО) "
          f"либо МО слишком далеко от центров типов ({nt['too_far']} МО); различия по статусам лежат в types_outside_sample.parquet;",
          f"- SVG: {'записан' if svg_ok else 'не записан'} (размер {svg_len} байт, предел {C.MAP_MAX_SVG_BYTES} байт).", "",
          "## Ограничения", "",
          "- Карта показывает территорию, а не население: крупные малонаселённые МО занимают много места, а плотно населённые МО малы.",
          "- Присвоенный МО, граничащий с МО выборки того же типа, отличается от соседей только штриховкой.",
          "- Крупные малонаселённые МО Севера и Дальнего Востока занимают на карте много места: площадь не равна населению.",
          f"- Справочник границ содержит {em['n_reg_all']} регионов, рабочая выборка — {em['n_reg_smp']}.",
          "- Типы вне выборки присвоены по ближайшему центроиду канонического разбиения; эти МО не входили в обучение и в статистики не входят.",
          f"- {beyond} из {n_asg} присвоенных МО лежат дальше 95-го процентиля расстояний выборки до центроида.",
          "- Проекция равнопромежуточная с поправкой aspect = 1/cos 60°: форма и площадь МО на севере искажены.",
          f"- Долготы меньше 0 сдвинуты на +360 у {n_shift} МО ({em['region_text']}), контуры при этом не разрезаются.", "",
          "## Проверка рисунка по пикселям", "",
          f"Для каждого МО в каждой оси, где оно нарисовано, берётся центр максимальной вписанной окружности (допуск {C.MAP_PIXEL_MIC_TOLERANCE_DEG} градуса); "
          f"проверка выполняется при радиусе не меньше {C.MAP_PIXEL_CHECK_MIN_RADIUS_PX} пикселя. Класс 1: цвет типа (допуск {C.MAP_PIXEL_TOL_CLASS1} из 255 по каналу); "
          f"классы 2 и 3: любой цвет на отрезке между цветом заливки и цветом штриховки (расстояние до отрезка не более {C.MAP_PIXEL_TOL_SEGMENT} из 255).", "",
          "| ось | нарисовано МО | проверено | пропущено | несовпадений |", "|---|---|---|---|---|"]
    for k, d in em["df"].groupby("axis"):
        L.append(f"| {k} | {len(d)} | {int(d['checked'].sum())} | {int((~d['checked']).sum())} | {int((d['checked'] & (d['ok'] == False)).sum())} |")  # noqa: E712
    L += ["", f"Проверено МО хотя бы в одной оси: {em['checked']}; по типам 0–5 в классе 1: {', '.join(map(str, em['typ_checked']))}; "
          f"в классе 2: {em['cls_checked'][2]}; в классе 3: {em['cls_checked'][3]}. Доля площади карты страны на проверенных МО: {em['area_share']:.3f} "
          f"(по классам: {', '.join(f'{k}: {v:.3f}' for k, v in em['area_by_cls'].items())}). "
          f"Доля пикселей оси карты, не близких ни к одному допустимому цвету (граничные пиксели): {em['boundary'] * 100:.2f}%.", ""]
    if em["mism"]:
        L += ["Несовпадения: " + "; ".join(f"{m['territory_id']} (класс {m['class']}, ось {m['axis']}, ожидалось {m['expected']}, получено {m['got']})" for m in em["mism"]) + ".", ""]
    L += ["## Контроли", "", "| контроль | ожидание | получено | результат |", "|---|---|---|---|"]
    for c in controls:
        L.append(f"| {c['name']} | {fmt(c['expect'])} | {fmt(c['got'])} | {'пройден' if c['ok'] else 'НЕ ПРОЙДЕН'} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
