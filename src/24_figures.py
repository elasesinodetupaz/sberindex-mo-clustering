"""Шаг 24. Фигуры 1 (профили типов), 3 (маркетплейсы между уровнями), 4 (размер МО), 5 (надёжность типов).

Описательный шаг: данные не пересчитываются. Каждое число на рисунке берётся из data/processed/figure_data_24.parquet,
который строится из parquet шагов 19–28 (type_portraits, composition_robustness, hypothesis_verdicts,
hypothesis_results_19b) и перед рисованием сверяется с заново прочитанными исходными файлами (допуск config SOURCE_TOL, n точно).
Имена и пометки типов — функции src/23_type_portraits.py (вызовы повторяют начало main шага 23); технические имена
сверяются с notebooks/23_type_portraits.md. Интерпретационные имена (docs/type_names.md) в рисунки не входят.
Контроли выполняются до записи файлов; при провале код выхода 1 и ничего не записывается. Параметры и ожидания —
config.yaml, группа step24_figures.
Вход:  data/processed/{type_portraits, composition_robustness, hypothesis_verdicts, hypothesis_results_19b}.parquet,
       prediction_strength_clusters_2024_12.parquet, kmeans_k6_matching.parquet (независимая сверка), notebooks/23_type_portraits.md
Выход: notebooks/figures/24_fig{1_profiles,3_marketplaces,4_size,5_reliability}.{png,svg},
       data/processed/figure_data_24.parquet, notebooks/24_figures.md
Запуск из корня проекта:  .venv/bin/python src/24_figures.py
"""
import hashlib
import importlib
import io
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

import config as C  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
F = C.FIGS
K = C.FINAL_K
CATS = ["Продовольствие", "Здоровье", "Общепит", "Транспорт", "Маркетплейсы"]
GROUPS = F["GROUPS"]
TYPES = [str(t) for t in range(K)]
FD_COLS = ["фигура", "панель", "ключ", "тип", "значение", "n", "источник_файл", "источник_блок_или_колонка"]
SRC_VCOL = {"tp": ("value", "n"), "comp": ("value", "n"), "verd": ("obs", None), "r19b": ("obs", "n")}
SRC_KEY = {"tp": "TYPE_PORTRAITS_PATH", "comp": "COMP_PATH", "verd": "VERDICTS_PATH", "r19b": "R19B_PATH"}
BRACKET_OK = re.compile(r"\[[+−±-]?\d+\.\d+; [+−±-]?\d+\.\d+\]")
FD: list = []
SPECS: list = []
CK: list = []
LOG: list = []


def P(rel: str) -> Path:
    return PROJECT_DIR / rel


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def ck(name: str, got, exp, tol=0.0) -> bool:
    ok = (got == exp) if isinstance(exp, (str, bool)) or tol is None else abs(float(got) - float(exp)) <= tol + 1e-12
    CK.append({"контроль": name, "получено": got if isinstance(got, str) else (f"{got:g}" if isinstance(got, (int, float, np.integer, np.floating)) else str(got)),
               "ожидается": exp if isinstance(exp, str) else f"{exp:g}", "допуск": "точно" if (isinstance(exp, (str, bool)) or tol == 0) else f"{tol:g}",
               "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})
    return ok


def sgn(v: float, nd: int = 2) -> str:
    return f"{v:+.{nd}f}".replace("-", "−")


def fmt_val(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".").replace("-", "−")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen(when: str) -> None:
    for name, exp in F["FROZEN_SHA256"].items():
        path = next((P(F[k]) for k in ("TYPE_PORTRAITS_PATH", "COMP_PATH", "VERDICTS_PATH", "R19B_PATH") if P(F[k]).name == name), None)
        got = sha(path)
        print(f"{when}: {got}  {path.relative_to(PROJECT_DIR)} {'OK' if got == exp else 'MISMATCH'}")
        LOG.append(f"- {when}: `{path.relative_to(PROJECT_DIR)}` — {'совпадает' if got == exp else 'НЕ СОВПАДАЕТ'}")
        if got != exp:
            stop(f"sha256 {name} = {got}, ожидается {exp}")


# ------------------------------------------------------------------ источники и figure_data
def read_src() -> dict:
    return {k: pd.read_parquet(P(F[v])) for k, v in SRC_KEY.items()}


def pick(df: pd.DataFrame, spec: dict) -> pd.Series:
    q = df
    for col, v in spec.items():
        q = q[q[col] == v]
    if len(q) != 1:
        stop(f"ожидалась ровно одна строка {spec}, найдено {len(q)}")
    return q.iloc[0]


def reg(fig, panel, key, typ, src, spec, vcol=None, ncol="default", block=None) -> None:
    """Строка figure_data из исходного parquet; спецификация запоминается для сверки с заново прочитанным файлом."""
    row = pick(SRC_A[src], spec)
    vcol = vcol or SRC_VCOL[src][0]
    nc = SRC_VCOL[src][1] if ncol == "default" else (None if ncol == "-" else ncol)
    n = None if nc is None or pd.isna(row[nc]) else int(row[nc])
    FD.append({"фигура": fig, "панель": panel, "ключ": key, "тип": typ, "значение": float(row[vcol]), "n": n,
               "источник_файл": Path(F[SRC_KEY[src]]).name, "источник_блок_или_колонка": block or spec.get("block", vcol)})
    SPECS.append((src, spec, vcol, nc))


def reg_cfg(fig, panel, key, value, cfg_path: tuple, idx=None, file="config.yaml") -> None:
    FD.append({"фигура": fig, "панель": panel, "ключ": key, "тип": "—", "значение": float(value), "n": None,
               "источник_файл": file, "источник_блок_или_колонка": ".".join(cfg_path) + ("" if idx is None else f", элемент {idx}")})
    SPECS.append(("cfg", {"path": cfg_path, "i": idx}, None, None))


def verify_sources() -> int:
    """Каждая строка figure_data против заново прочитанного исходного файла (допуск SOURCE_TOL, n точно)."""
    B = read_src()
    cfg = yaml.safe_load(P("config.yaml").read_text(encoding="utf-8"))
    bad = []
    for row, (src, spec, vcol, nc) in zip(FD, SPECS):
        if src == "cfg":
            v = cfg
            for part in spec["path"]:
                v = v[part]
            v = v if spec["i"] is None else v[spec["i"]]
            n = None
        elif src == "mark":
            continue
        else:
            r = pick(B[src], spec)
            v, n = float(r[vcol]), (None if nc is None or pd.isna(r[nc]) else int(r[nc]))
        if abs(float(v) - row["значение"]) > F["SOURCE_TOL"] or n != row["n"]:
            bad.append((row["фигура"], row["ключ"], row["тип"], row["значение"], v, row["n"], n))
    if bad:
        stop(f"значения figure_data не совпали с исходными файлами (фигура, ключ, тип, рисунок, исходник, n, n исходника): {bad[:10]}")
    return sum(1 for s in SPECS if s[0] != "mark")


# ------------------------------------------------------------------ имена и пометки типов (шаг 23)
def s23_objects():
    s23 = importlib.import_module("23_type_portraits")
    d = s23.step18.load()
    s23.step18.clean_rosstat(d, [])
    sup, _ = s23.step21.supply_values(d["ids"])
    D = s23.build_frame(d, sup)
    z = s23.z_profile(d, D)
    za = s23.alr_profile(d, D)
    return s23, D, z, za


def nb23_names() -> list:
    out = []
    for line in P(F["NB23_PATH"]).read_text(encoding="utf-8").split("\n"):
        m = re.match(r"- Имя типа: (.*) \(расчёт шага 23\)\.$", line)
        if m:
            out.append(m.group(1))
    return out


# ------------------------------------------------------------------ данные фигур
def build_data(s23, z, za) -> None:
    for t in range(K):
        for c in CATS:
            reg("фигура 1", "a", f"медиана {c}, %", str(t), "tp", {"type": str(t), "block": "расходы", "metric": f"медиана {c}, %"})
            reg("фигура 1", "b", f"{c}: σ по долям", str(t), "tp", {"type": str(t), "block": "признаки профиля", "metric": f"{c}: σ по долям"})
            reg("фигура 1", "c", f"{c}: σ по лог-отношению", str(t), "tp", {"type": str(t), "block": "признаки профиля", "metric": f"{c}: σ по лог-отношению"})
            mark = s23.feature_mark(z, za, t, c)
            FD.append({"фигура": "фигура 1", "панель": "b, c", "ключ": f"пометка «зависит от записи»: {c}", "тип": str(t),
                       "значение": 1.0 if mark == "зависит от записи" else 0.0, "n": None, "источник_файл": "src/23_type_portraits.py",
                       "источник_блок_или_колонка": "функция feature_mark"})
            SPECS.append(("mark", {"t": t, "c": c}, None, None))
        for c, m in s23.border_features(z, za, t):
            FD.append({"фигура": "фигура 1", "панель": "b, c", "ключ": f"граница порога, меньшее |σ|: {c}", "тип": str(t), "значение": float(m),
                       "n": None, "источник_файл": "src/23_type_portraits.py", "источник_блок_или_колонка": "функция border_features"})
            SPECS.append(("mark", {"t": t, "c": c}, None, None))
    # фигура 3
    base = {"block": "5. связи", "metric": "Г2: ρ(Маркетплейсы, ln market_access)"}
    for lev, lab in [("между регионами", "между регионами"), ("внутри регионов", "внутри регионов")]:
        for var, vl in [("доли", "по долям"), ("лог-отношение", "по log(доля / «прочее»)")]:
            reg("фигура 3", "столбики", f"ρ {vl}, {lab}", "—", "comp", {**base, "variant": var, "level": lev}, block="5. связи")
        reg("фигура 3", "подпись под столбиками", f"регионов, {lab}", "—", "comp", {**base, "variant": "доли", "level": lev}, vcol="n_regions", ncol="-", block="5. связи")
    reg_cfg("фигура 3", "подпись под столбиками", "порог числа МО в регионе", C.COMP_MIN_MO_REGION, ("step25_composition", "MIN_MO_REGION"))
    labs = C.SUMMARY_RHO_LABELS
    for i, b in enumerate(C.SUMMARY_RHO_BINS):
        reg_cfg("фигура 3", "линии порогов", f"порог |ρ| между «{labs[i]}» и «{labs[i + 1]}»", b, ("step26_summary", "RHO_SIZE_BINS"), i)
    reg("фигура 3", "подпись", "ρ все МО по долям (вердикты шага 19c)", "—", "verd", {"row_type": "level", "hypothesis": "Г2", "level": "общий"}, block="obs; Г2, общий")
    reg("фигура 3", "подпись", "ρ все МО по лог-отношению (шаг 25)", "—", "comp", {**base, "variant": "лог-отношение", "level": "все МО"}, block="5. связи")
    reg("фигура 3", "подпись", "ρ внутри регионов, шаг 19b (вердикты шага 19c)", "—", "verd", {"row_type": "level", "hypothesis": "Г2", "level": "внутри регионов"}, block="obs; Г2, внутри регионов")
    reg("фигура 3", "подпись", "МО в выборке шага 19b, внутри регионов", "—", "r19b", {"row_type": "corr", "variable": "share_Маркетплейсы~log_market_access",
        "variant": "основной", "level": "внутри регионов"}, vcol="n", ncol="-", block="n; Г2, основной, внутри регионов")
    # фигура 4
    for g in GROUPS:
        for m in (f"{g}: n МО", f"{g}: медиана Транспорт", f"{g}: медиана Общепит"):
            reg("фигура 4", "a", m, "все", "tp", {"type": "все", "block": "1.1 группы размера", "metric": m})
        for m in (f"{g}: транспорт / «прочее» (медиана)", f"{g}: общепит / «прочее» (медиана)"):
            reg("фигура 4", "b", m, "все", "tp", {"type": "все", "block": "1.2 относительно «прочего»", "metric": m})
    reg("фигура 4", "под осью", "МО без населения 2023", "все", "tp", {"type": "все", "block": "1.1 группы размера", "metric": "МО без населения 2023"})
    for a, b in zip(GROUPS, GROUPS[1:]):
        m = f"изменение медианы {a} → {b}"
        blk = "тип 1: транспорт по размеру"
        reg("фигура 4", "a, ▲", f"{m}, п.п.", "все", "tp", {"type": "1", "block": blk, "metric": f"{m}, п.п."}, ncol="-")
        for side, w in (("нижняя", "lo"), ("верхняя", "hi")):
            reg("фигура 4", "a, ▲", f"{m}: {side} граница 95% интервала", "все", "tp", {"type": "1", "block": blk, "metric": f"{m}: {side} граница 95% интервала"})
    # фигура 5
    for t in range(K):
        for m in ("PS типа (среднее доля пар)", "уверенных сопоставлений из 23", "месяцев без якоря", "Г6: S по типу", "Г6: базовый уровень"):
            reg("фигура 5", {"PS типа (среднее доля пар)": "a", "уверенных сопоставлений из 23": "b", "месяцев без якоря": "b"}.get(m, "c"), m, str(t),
                "tp", {"type": str(t), "block": "надёжность", "metric": m})


def make_V() -> dict:
    return {(r["фигура"], r["панель"], r["ключ"], r["тип"]): (r["значение"], r["n"]) for r in FD}


def v(V, fig, panel, key, typ="—"):
    return V[(f"фигура {fig}", panel, key, typ)][0]


def vn(V, fig, panel, key, typ="—"):
    return V[(f"фигура {fig}", panel, key, typ)][1]


# ------------------------------------------------------------------ рисование: общее
class Canvas:
    def __init__(self, size):
        self.W, self.H = size
        self.fig = plt.figure(figsize=size, dpi=F["DPI"], facecolor="white")

    def axes(self, r: dict, **kw):
        return self.fig.add_axes([r["left"] / self.W, r["bottom"] / self.H, r["width"] / self.W, r["height"] / self.H], **kw)

    def text(self, x, y, s, **kw):
        return self.fig.text(x / self.W, y / self.H, s, **kw)

    def rect(self, x, y, w, h, **kw):
        r = Rectangle((x / self.W, y / self.H), w / self.W, h / self.H, transform=self.fig.transFigure, **kw)
        self.fig.add_artist(r)
        return r


def lum(rgba) -> float:
    return 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]


def wrap(s: str) -> str:
    return "\n".join(textwrap.fill(p, F["CAPTION_WRAP_CHARS"]) for p in s.split("\n"))


def style_axes(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=F["TICK_FS"])


# ------------------------------------------------------------------ фигура 1
def draw_fig1(V, names, z, za, s23, sizes) -> Canvas:
    cv = Canvas(F["F1_SIZE_IN"])
    W, H = cv.W, cv.H
    ch, left = F["F1_CELL_H_IN"], F["F1_LEFT_IN"]
    cw = (W - left - F["F1_RIGHT_IN"]) / K
    cv.text(W / 2, H - 0.42, "Профили типов МО, декабрь 2024", ha="center", va="center", fontsize=F["TITLE_FS"], fontweight="bold")
    panels = [("a", "(а) медиана доли расходов в «Все категории», %; цвет: внутри строки от меньшей медианы к большей", "медиана {c}, %"),
              ("b", "(б) профиль σ по долям: средний z-score типа (z-score внутри месяца)", "{c}: σ по долям"),
              ("c", "(в) профиль σ по log(доля / «прочее»): средний z-score типа", "{c}: σ по лог-отношению")]
    pitch = F["F1_AXES_TOP_OFFSET_IN"] + 5 * ch + F["F1_PANEL_GAP_IN"]
    bottom_last = None
    for pi, (pn, sub, keyf) in enumerate(panels):
        y_sub = H - F["F1_TOP_IN"] - pi * pitch
        top = y_sub - F["F1_AXES_TOP_OFFSET_IN"]
        cv.text(0.30, y_sub, sub, ha="left", va="center", fontsize=F["PANEL_FS"], fontweight="bold")
        ax = cv.axes({"left": left, "bottom": top - 5 * ch, "width": K * cw, "height": 5 * ch})
        ax.set_xlim(0, K)
        ax.set_ylim(5, 0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.xaxis.tick_top()
        ax.set_xticks(np.arange(K) + 0.5)
        ax.set_xticklabels([f"Тип {t}, n = {vn(V, 1, 'a', 'медиана Продовольствие, %', str(t))}" for t in range(K)], fontsize=F["LABEL_FS"])
        ax.set_yticks(np.arange(5) + 0.5)
        ax.set_yticklabels(CATS, fontsize=F["LABEL_FS"])
        ax.tick_params(length=0, pad=6)
        vals = np.array([[v(V, 1, pn, keyf.format(c=c), str(t)) for t in range(K)] for c in CATS])   # строки — категории
        if pn == "a":
            cmap = plt.get_cmap(F["F1_SHARE_CMAP"])
            lo_r, hi_r = F["F1_SHARE_CMAP_RANGE"]
            colors = np.empty(vals.shape, dtype=object)
            for r_ in range(5):
                lo, hi = vals[r_].min(), vals[r_].max()
                for c_ in range(K):
                    colors[r_, c_] = cmap(lo_r + (hi_r - lo_r) * (vals[r_, c_] - lo) / (hi - lo))
        else:
            vmax = F["F1_SIGMA_VMAX"]
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            cmap = plt.get_cmap(F["F1_SIGMA_CMAP"])
            colors = np.empty(vals.shape, dtype=object)
            for r_ in range(5):
                for c_ in range(K):
                    colors[r_, c_] = cmap(norm(vals[r_, c_]))
        for r_, c in enumerate(CATS):
            for c_ in range(K):
                ax.add_patch(Rectangle((c_, r_), 1, 1, facecolor=colors[r_, c_], edgecolor="white", linewidth=1.2))
                txt = f"{vals[r_, c_]:.1f}" if pn == "a" else sgn(vals[r_, c_])
                kw = {}
                if pn != "a":
                    dep = v(V, 1, "b, c", f"пометка «зависит от записи»: {c}", str(c_)) == 1.0
                    bord = ("фигура 1", "b, c", f"граница порога, меньшее |σ|: {c}", str(c_)) in V
                    if bord:
                        txt += "*"
                    if dep:
                        ax.add_patch(Rectangle((c_, r_), 1, 1, fill=False, hatch=F["F1_HATCH"], edgecolor=F["F1_HATCH_COLOR"], linewidth=0))
                        kw["bbox"] = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)
                col = "black" if (lum(colors[r_, c_]) > 0.5 or kw) else "white"
                ax.text(c_ + 0.5, r_ + 0.5, txt, ha="center", va="center", fontsize=F["CELL_FS"], color=col, zorder=5, **kw)
        for c_ in range(K):                                                       # цветная полоса типа над колонкой (цвета карты 24d)
            ax.add_patch(Rectangle((c_ + 0.03, -0.13), 0.94, 0.10, facecolor=F["TYPE_COLORS"][c_], edgecolor="none", clip_on=False))
        if pn != "a":
            cax = cv.axes({"left": W - F["F1_RIGHT_IN"] + 0.25, "bottom": top - 5 * ch, "width": F["F1_CBAR_W_IN"], "height": 5 * ch})
            cb = cv.fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cb.ax.tick_params(labelsize=F["NOTE_FS"])
            cb.set_label("σ", fontsize=F["NOTE_FS"])
        bottom_last = top - 5 * ch
    y = bottom_last - 0.45
    cv.text(0.30, y, "Технические имена типов (расчёт шага 23, функция full_type_name):", ha="left", va="center", fontsize=F["LABEL_FS"], fontweight="bold")
    for t in range(K):
        y -= F["F1_LEGEND_LINE_IN"]
        cv.rect(0.30, y - 0.055, 0.13, 0.11, facecolor=F["TYPE_COLORS"][t], edgecolor="none")
        cv.text(0.52, y, f"Тип {t} (n = {sizes[t]}): {names[t]}", ha="left", va="center", fontsize=F["NOTE_FS"])
    y -= F["F1_LEGEND_LINE_IN"] + 0.10
    cv.rect(0.30, y - 0.055, 0.13, 0.11, fill=False, hatch=F["F1_HATCH"], edgecolor=F["F1_HATCH_COLOR"], linewidth=0)
    cv.text(0.52, y, "штриховка (б, в): признак с пометкой «зависит от записи» (знак или величина σ по log(доля / «прочее») не повторяют σ по долям)",
            ha="left", va="center", fontsize=F["NOTE_FS"])
    y -= F["F1_LEGEND_LINE_IN"]
    cv.text(0.30, y, "*", ha="left", va="center", fontsize=F["LABEL_FS"], fontweight="bold")
    cv.text(0.52, y, "признак из пометки «на границе порога» в имени типа: меньшее из двух |σ| лишь немного выше порога подписи (запас задан после просмотра профилей)",
            ha="left", va="center", fontsize=F["NOTE_FS"])
    y -= F["F1_LEGEND_LINE_IN"]
    cv.text(0.30, y, "Источник: data/processed/type_portraits.parquet, блоки «расходы» и «признаки профиля» (шаг 23); пометки — feature_mark и border_features шага 23.",
            ha="left", va="center", fontsize=F["NOTE_FS"])
    return cv


# ------------------------------------------------------------------ фигура 3
def draw_fig3(V) -> tuple:
    cv = Canvas(F["F3_SIZE_IN"])
    W, H = cv.W, cv.H
    ax = cv.axes(F["F3_AXES_IN"])
    style_axes(ax)
    ylo, yhi = F["F3_YLIM"]
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-0.6, 1.6)
    cv.text(W / 2, H - 0.42, "Доля маркетплейсов и ln market_access: ρ Спирмена на двух уровнях", ha="center", va="center", fontsize=F["TITLE_FS"], fontweight="bold")
    levels = ["между регионами", "внутри регионов"]
    variants = ["по долям", "по log(доля / «прочее»)"]
    for gi, lev in enumerate(levels):
        for vi, var in enumerate(variants):
            val = v(V, 3, "столбики", f"ρ {var}, {lev}")
            x = gi + (vi - 0.5) * 2 * F["F3_BAR_OFFSET"]
            ax.bar(x, val, width=F["F3_BAR_WIDTH"], color=F["F3_BAR_COLORS"][vi], zorder=3, label=var if gi == 0 else None)
            ax.text(x, val + (0.02 if val >= 0 else -0.02), sgn(val, 3), ha="center", va="bottom" if val >= 0 else "top", fontsize=F["CELL_FS"], zorder=4)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    bins = FD_RHO_BINS
    for b in bins:
        for s_ in (1, -1):
            ax.axhline(s_ * b, color=F["F3_LINE_COLOR"], linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    edges = [0.0] + bins
    for s_ in (1, -1):
        for i, lab in enumerate(C.SUMMARY_RHO_LABELS):
            mid = (edges[i] + edges[i + 1]) / 2 if i < len(bins) else bins[-1] + 0.06
            ax.text(1.012, s_ * mid, lab, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=F["NOTE_FS"], color="#444444")
    ax.set_yticks([-b for b in reversed(bins)] + [0.0] + bins)
    ax.set_yticklabels([sgn(t, 1) if t else "0" for t in [-b for b in reversed(bins)] + [0.0] + bins])
    ax.set_ylabel("ρ Спирмена: доля маркетплейсов ~ ln market_access", fontsize=F["LABEL_FS"])
    ax.set_xticks([0, 1])
    mm = C.COMP_MIN_MO_REGION
    ax.set_xticklabels([
        f"между регионами\nмедианы {int(v(V, 3, 'подпись под столбиками', 'регионов, между регионами'))} регионов с не менее {int(v(V, 3, 'подпись под столбиками', 'порог числа МО в регионе'))} МО\n"
        f"n = {vn(V, 3, 'столбики', 'ρ по долям, между регионами')} МО",
        f"внутри регионов\nранги внутри региона, {int(v(V, 3, 'подпись под столбиками', 'регионов, внутри регионов'))} регионов с не менее {int(v(V, 3, 'подпись под столбиками', 'порог числа МО в регионе'))} МО\n"
        f"n = {vn(V, 3, 'столбики', 'ρ по долям, внутри регионов')} МО"], fontsize=F["LABEL_FS"])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.legend(loc="upper right", fontsize=F["NOTE_FS"], frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, title="кодирование доли", title_fontsize=F["NOTE_FS"])
    cap = (f"Уровень «все МО» (ρ {sgn(v(V, 3, 'подпись', 'ρ все МО по долям (вердикты шага 19c)'), 3)} по долям, "
           f"{sgn(v(V, 3, 'подпись', 'ρ все МО по лог-отношению (шаг 25)'), 3)} по лог-отношению) приведён в таблице шага 26; значение "
           f"{sgn(v(V, 3, 'подпись', 'ρ внутри регионов, шаг 19b (вердикты шага 19c)'), 3)} в шаге 19b посчитано по другой выборке "
           f"({int(v(V, 3, 'подпись', 'МО в выборке шага 19b, внутри регионов'))} МО); знак зависит от уровня.\n"
           "Источник: data/processed/composition_robustness.parquet, блок «5. связи» (шаг 25); подпись: hypothesis_verdicts.parquet и hypothesis_results_19b.parquet. "
           "Горизонтальные линии — границы подписей размера |ρ| (config, step26_summary). Диаграмма рассеяния не показана.")
    cv.text(0.30, F["F3_CAPTION_TOP_IN"], wrap(cap), ha="left", va="top", fontsize=F["NOTE_FS"])
    return cv, cap


# ------------------------------------------------------------------ фигура 4
def draw_fig4(V, tri) -> tuple:
    cv = Canvas(F["F4_SIZE_IN"])
    W, H = cv.W, cv.H
    cv.text(W / 2, H - 0.42, "Структура расходов и размер МО (группы по населению 2023, тыс. жителей), все МО, декабрь 2024", ha="center", va="center",
            fontsize=F["TITLE_FS"], fontweight="bold")
    xl = [f"{g}\nn = {vn(V, 4, 'a', f'{g}: n МО', 'все')}" for g in GROUPS]
    spec = [("a", F["F4_AXES_A_IN"], F["F4_YLIM_A"], "(а) медиана доли расходов, %", "медиана доли в «Все категории», %",
             (lambda g: f"{g}: медиана Транспорт", lambda g: f"{g}: медиана Общепит"), ".1f"),
            ("b", F["F4_AXES_B_IN"], F["F4_YLIM_B"], "(б) отношение к «прочему»", "медиана отношения доли к доле «прочего»",
             (lambda g: f"{g}: транспорт / «прочее» (медиана)", lambda g: f"{g}: общепит / «прочее» (медиана)"), ".3f")]
    for pn, rect, ylim, title, ylab, (kt, kc), nf in spec:
        ax = cv.axes(rect)
        style_axes(ax)
        ax.set_xlim(-0.4, len(GROUPS) - 0.6)
        ax.set_ylim(*ylim)
        xs = np.arange(len(GROUPS))
        yt = np.array([v(V, 4, pn, kt(g), "все") for g in GROUPS])
        yc = np.array([v(V, 4, pn, kc(g), "все") for g in GROUPS])
        ax.plot(xs, yt, marker="o", color=F["F4_COLORS"]["transport"], linewidth=1.6, label="транспорт", zorder=3)
        ax.plot(xs, yc, marker="s", color=F["F4_COLORS"]["catering"], linewidth=1.6, linestyle="--", label="общепит", zorder=3)
        off = (ylim[1] - ylim[0]) * 0.035
        for x, a_, b_ in zip(xs, yt, yc):
            ax.text(x, a_ + off, format(a_, nf), ha="center", va="bottom", fontsize=F["CELL_FS"], color=F["F4_COLORS"]["transport"])
            ax.text(x, b_ - off, format(b_, nf), ha="center", va="top", fontsize=F["CELL_FS"], color=F["F4_COLORS"]["catering"])
        if pn == "a":
            for (a, b, d, lo, hi, sig) in tri:
                if sig:
                    i = GROUPS.index(a)
                    ytop = max(yt[i], yt[i + 1]) + (ylim[1] - ylim[0]) * 0.13
                    ax.text(i + 0.5, ytop, "▲", ha="center", va="center", fontsize=F["PANEL_FS"] + 1, color="#B00020", zorder=5)
                    ax.text(i + 0.5, ytop + (ylim[1] - ylim[0]) * 0.055, f"{sgn(d)} [{sgn(lo)}; {sgn(hi)}]", ha="center", va="bottom", fontsize=F["NOTE_FS"], color="#B00020")
        ax.set_xticks(xs)
        ax.set_xticklabels(xl, fontsize=F["LABEL_FS"])
        ax.set_xlabel("группа размера МО, тыс. жителей", fontsize=F["LABEL_FS"])
        ax.set_ylabel(ylab, fontsize=F["LABEL_FS"])
        ax.set_title(title, fontsize=F["PANEL_FS"], fontweight="bold", loc="left")
        ax.legend(loc="upper left", fontsize=F["NOTE_FS"], frameon=False)
    n_tot = sum(vn(V, 4, "a", f"{g}: n МО", "все") for g in GROUPS)
    cap = (f"▲ — интервал бутстрепа (95%) разности медиан доли транспорта соседних групп не включает 0; разность в п.п. и интервал указаны у отметки. "
           f"Показаны медианы без усов. МО без населения 2023 ({int(v(V, 4, 'под осью', 'МО без населения 2023', 'все'))}) в группы не входят: n групп {n_tot}.\n"
           "Источник: data/processed/type_portraits.parquet, блоки «1.1 группы размера» (а), «1.2 относительно «прочего»» (б) и «тип 1: транспорт по размеру» "
           "(интервалы разностей по всем МО двух соседних групп; шаг 23).")
    cv.text(0.30, F["F4_CAPTION_TOP_IN"], wrap(cap), ha="left", va="top", fontsize=F["NOTE_FS"])
    return cv, cap


# ------------------------------------------------------------------ фигура 5
def draw_fig5(V) -> tuple:
    cv = Canvas(F["F5_SIZE_IN"])
    W, H = cv.W, cv.H
    cv.text(W / 2, H - 0.42, "Надёжность типов МО (k = 6)", ha="center", va="center", fontsize=F["TITLE_FS"], fontweight="bold")
    xs = np.arange(K)
    xl = [f"Тип {t}" for t in range(K)]
    cols = F["TYPE_COLORS"]
    ps = [v(V, 5, "a", "PS типа (среднее доля пар)", str(t)) for t in range(K)]
    mt = [v(V, 5, "b", "уверенных сопоставлений из 23", str(t)) for t in range(K)]
    nm = [v(V, 5, "b", "месяцев без якоря", str(t)) for t in range(K)]
    ax = cv.axes(F["F5_AXES_A_IN"]); style_axes(ax)
    ax.bar(xs, ps, color=cols, width=0.62, zorder=3)
    ax.set_ylim(0, 1.08)
    for x, y in zip(xs, ps):
        ax.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=F["CELL_FS"])
    ax.set_xticks(xs); ax.set_xticklabels(xl, fontsize=F["LABEL_FS"])
    ax.set_ylabel("PS (среднее «доля пар»)", fontsize=F["LABEL_FS"])
    ax.set_title("(а) prediction strength по типам", fontsize=F["PANEL_FS"], fontweight="bold", loc="left")
    ax = cv.axes(F["F5_AXES_B_IN"]); style_axes(ax)
    ax.bar(xs, mt, color=cols, width=0.62, zorder=3)
    ax.set_ylim(0, max(nm) * 1.15)
    for x, y, n_ in zip(xs, mt, nm):
        ax.text(x, y + 0.4, f"{int(y)} из {int(n_)}", ha="center", va="bottom", fontsize=F["CELL_FS"])
    ax.set_xticks(xs); ax.set_xticklabels(xl, fontsize=F["LABEL_FS"])
    ax.set_ylabel("сопоставлений с якорем из 23", fontsize=F["LABEL_FS"])
    ax.set_title("(б) уверенные сопоставления типа по месяцам", fontsize=F["PANEL_FS"], fontweight="bold", loc="left")
    ax = cv.axes(F["F5_AXES_C_IN"]); style_axes(ax)
    s6 = [v(V, 5, "c", "Г6: S по типу", str(t)) for t in range(K)]
    b6 = [v(V, 5, "c", "Г6: базовый уровень", str(t)) for t in range(K)]
    w = 0.34
    for x in xs:
        ax.bar(x - w / 2 - 0.01, s6[x], width=w, color=cols[x], zorder=3)
        ax.bar(x + w / 2 + 0.01, b6[x], width=w, color=F["F5_BASE_COLOR"], hatch=F["F5_BASE_HATCH"], edgecolor="#666666", linewidth=0.4, zorder=3)
        ax.text(x - w / 2 - 0.01, s6[x] + 0.02, f"{s6[x]:.2f}", ha="center", va="bottom", fontsize=F["CELL_FS"])
        ax.text(x + w / 2 + 0.01, b6[x] + 0.02, f"{b6[x]:.2f}", ha="center", va="bottom", fontsize=F["CELL_FS"])
    ax.set_ylim(0, 1.12)
    ax.set_xticks(xs); ax.set_xticklabels(xl, fontsize=F["LABEL_FS"])
    ax.set_ylabel("доля переходов в ближайшие типы", fontsize=F["LABEL_FS"])
    ax.set_title("(в) Г6: доля переходов в ближайшие типы (цвет типа) и базовый уровень нуля (штриховка)", fontsize=F["PANEL_FS"], fontweight="bold", loc="left")
    cap = ("Базовый уровень нуля нереалистичен; это описательное свойство, не проверка.\n"
           "Источник: data/processed/type_portraits.parquet, блок «надёжность» (шаг 23): (а) среднее «доля пар» по строкам k = 6 с данным доминирующим типом "
           "(prediction_strength_clusters_2024_12.parquet), (б) сопоставления с якорем по 23 месяцам без якоря (kmeans_k6_matching.parquet), "
           "(в) Г6 из hypothesis_results_19b.parquet (основной вариант, два ближайших типа).")
    cv.text(0.30, F["F5_CAPTION_TOP_IN"], wrap(cap), ha="left", va="top", fontsize=F["NOTE_FS"])
    return cv, cap


# ------------------------------------------------------------------ проверки рисунка
def drawn_texts(fig) -> list:
    out = list(fig.texts)
    for ax in fig.axes:
        out += list(ax.texts) + [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label]
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            out += [tk.label1 for tk in axis.get_major_ticks() if lo - 1e-9 <= tk.get_loc() <= hi + 1e-9]   # метки вне осей не рисуются
        lg = ax.get_legend()
        if lg is not None:
            out += list(lg.get_texts()) + [lg.get_title()]
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def layout_report(cv: Canvas) -> dict:
    fig = cv.fig
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(r)
    m = F["MIN_MARGIN_IN"]
    margins = [tb.x0, tb.y0, cv.W - tb.x1, cv.H - tb.y1]
    texts = drawn_texts(fig)
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    tol = F["TEXT_OVERLAP_TOL_PX"]
    pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.x0 + tol < b.x1 and b.x0 + tol < a.x1 and a.y0 + tol < b.y1 and b.y0 + tol < a.y1:
                pairs.append((boxes[i][0].get_text()[:40], boxes[j][0].get_text()[:40]))
    allt = [t.get_text() for t in fig.findobj(Text) if t.get_text().strip()]
    bad_w = [t for t in allt if any(w in t.lower() for w in F["FORBIDDEN_WORDS"])]
    bad_br = [t for t in allt if ("[" in t or "]" in t) and BRACKET_OK.sub("", t).count("[") + BRACKET_OK.sub("", t).count("]") > 0]
    return {"margin_min": min(margins), "margin_ok": min(margins) >= m - 1e-9, "overlaps": pairs, "n_texts": len(allt), "bad_words": bad_w, "bad_brackets": bad_br}


def pngs_svgs(cv: Canvas) -> tuple:
    png = io.BytesIO()
    cv.fig.savefig(png, format="png", dpi=F["DPI"], facecolor="white", metadata={"Software": None})
    svg = io.BytesIO()
    cv.fig.savefig(svg, format="svg", facecolor="white", metadata={"Date": None, "Creator": None})
    return png.getvalue(), svg.getvalue()


# ------------------------------------------------------------------ отчёт
def md(df: pd.DataFrame) -> str:
    return df.astype(str).apply(lambda col: col.str.replace("|", "\\|", regex=False)).to_markdown(index=False, disable_numparse=True)


def fig_table(fig: str) -> str:
    d = pd.DataFrame(FD)
    d = d[d["фигура"] == fig]
    out = pd.DataFrame({"панель": d["панель"], "показатель": [k + (f", тип {t}" if t.isdigit() else "") for k, t in zip(d["ключ"], d["тип"])],
                        "значение": [fmt_val(x) for x in d["значение"]], "n": ["—" if pd.isna(n) else str(int(n)) for n in d["n"]],
                        "источник (файл, блок)": [f"{f_}, {b}" for f_, b in zip(d["источник_файл"], d["источник_блок_или_колонка"])]})
    return md(out)


def main() -> None:
    global SRC_A, FD_RHO_BINS
    plt.rcParams.update({"font.family": F["FONT"], "svg.fonttype": "path", "svg.hashsalt": F["SVG_HASHSALT"], "axes.unicode_minus": True,
                         "hatch.linewidth": F["F1_HATCH_LW"]})
    check_frozen("до", )
    ck("цвета типов совпадают с картой 24d", ",".join(F["TYPE_COLORS"]), ",".join(C.MAP_TYPE_COLORS))
    s23, D, z, za = s23_objects()
    names = [s23.full_type_name(D, z, za, t) for t in range(K)]
    nb = nb23_names()
    ck("технические имена типов = notebooks/23_type_portraits.md (число имён)", len(nb), K)
    for t in range(min(K, len(nb))):
        ck(f"имя типа {t} дословно как в md шага 23", names[t], nb[t])
    ck("группы размера = size_labels шага 23", "|".join(GROUPS), "|".join(s23.size_labels(C.SIZE_BINS_TH)))
    SRC_A = read_src()
    build_data(s23, z, za)
    FD_RHO_BINS = [r["значение"] for r in FD if r["ключ"].startswith("порог |ρ| между")]
    ck("число строк figure_data без дубликатов ключа", len(FD) - len({(r["фигура"], r["панель"], r["ключ"], r["тип"]) for r in FD}), 0)
    n_ver = verify_sources()
    ck("строк figure_data сверено с исходными файлами (расхождений)", 0, 0)
    CK[-1]["получено"] = f"0 из {n_ver}"
    V = make_V()
    tp = SRC_A["tp"]
    # --- независимая сверка пометок и границы порога с parquet шага 23
    th = importlib.import_module("10b_kmeans_cluster_profiles").PROFILE_Z_THRESHOLD
    marks_bad, border_got = [], []
    for t in range(K):
        for c in CATS:
            zs = pick(tp, {"type": str(t), "block": "признаки профиля", "metric": f"{c}: σ по долям"})["value"]
            za_ = pick(tp, {"type": str(t), "block": "признаки профиля", "metric": f"{c}: σ по лог-отношению"})["value"]
            exp_mark = 0.0 if abs(zs) < th else (0.0 if (np.sign(za_) == np.sign(zs) and abs(za_) >= th) else 1.0)
            if exp_mark != V[("фигура 1", "b, c", f"пометка «зависит от записи»: {c}", str(t))][0]:
                marks_bad.append((t, c))
            kb = ("фигура 1", "b, c", f"граница порога, меньшее |σ|: {c}", str(t))
            if kb in V:
                border_got.append((t, c))
                pq = tp[(tp["type"] == str(t)) & (tp["block"] == "признаки профиля") & (tp["metric"] == f"{c}: на границе порога, меньшее |σ|")]
                if len(pq) != 1 or abs(pq["value"].iloc[0] - V[kb][0]) > F["SOURCE_TOL"]:
                    marks_bad.append((t, c, "граница"))
    ck("пометки «зависит от записи» и «граница порога» = пересчёт по σ из parquet (расхождений)", len(marks_bad), 0)
    ck("признаки на границе порога (шаг 23)", ";".join(f"{a} {b}" for a, b in sorted(border_got)), ";".join(f"{a} {b}" for a, b in sorted(map(tuple, F["EXPECT_F1_BORDER"]))))
    smax = max(abs(r["значение"]) for r in FD if r["фигура"] == "фигура 1" and r["панель"] in ("b", "c") and not r["ключ"].startswith(("пометка", "граница")))
    ck("шкала σ охватывает все значения (max |σ| ≤ F1_SIGMA_VMAX)", int(smax <= F["F1_SIGMA_VMAX"]), 1)
    # --- контроли фигуры 1
    for t, c, x in F["EXPECT_F1_MEDIAN"]:
        ck(f"Ф1: медиана тип {t} {c.lower()}", v(V, 1, "a", f"медиана {c}, %", str(t)), x, F["TOL_F1_MEDIAN"])
    for t, c, xs_, xa in F["EXPECT_F1_SIGMA"]:
        ck(f"Ф1: σ по долям тип {t} {c.lower()}", v(V, 1, "b", f"{c}: σ по долям", str(t)), xs_, F["TOL_F1_SIGMA"])
        ck(f"Ф1: σ по лог-отношению тип {t} {c.lower()}", v(V, 1, "c", f"{c}: σ по лог-отношению", str(t)), xa, F["TOL_F1_SIGMA"])
    # --- контроли фигуры 3
    e3 = F["EXPECT_F3"]
    for lev, ex in (("между регионами", e3["between"]), ("внутри регионов", e3["within"])):
        ck(f"Ф3: ρ по долям, {lev}", v(V, 3, "столбики", f"ρ по долям, {lev}"), ex[0], F["TOL_F3"])
        ck(f"Ф3: ρ по лог-отношению, {lev}", v(V, 3, "столбики", f"ρ по log(доля / «прочее»), {lev}"), ex[1], F["TOL_F3"])
    ck("Ф3: регионов, между регионами", v(V, 3, "подпись под столбиками", "регионов, между регионами"), e3["between_n_regions"])
    ck("Ф3: МО, между регионами", vn(V, 3, "столбики", "ρ по долям, между регионами"), e3["between_n"])
    print(f"Ф3 (из источника): внутри регионов {int(v(V, 3, 'подпись под столбиками', 'регионов, внутри регионов'))} регионов, "
          f"{vn(V, 3, 'столбики', 'ρ по долям, внутри регионов')} МО")
    ck("Ф3: ρ все МО по долям (подпись)", v(V, 3, "подпись", "ρ все МО по долям (вердикты шага 19c)"), F["EXPECT_F3_VERDICT_ALL"], F["TOL_F3"])
    ck("Ф3: ρ все МО по лог-отношению (подпись)", v(V, 3, "подпись", "ρ все МО по лог-отношению (шаг 25)"), F["EXPECT_F3_ALR_ALL"], F["TOL_F3"])
    ck("Ф3: ρ шага 19b внутри регионов (подпись)", v(V, 3, "подпись", "ρ внутри регионов, шаг 19b (вердикты шага 19c)"), F["EXPECT_F3_19B_WITHIN"][0], F["TOL_F3"])
    ck("Ф3: МО шага 19b внутри регионов (подпись)", v(V, 3, "подпись", "МО в выборке шага 19b, внутри регионов"), F["EXPECT_F3_19B_WITHIN"][1])
    ck("Ф3: линии порогов = config step26_summary", ",".join(f"{x:g}" for x in FD_RHO_BINS), ",".join(f"{x:g}" for x in C.SUMMARY_RHO_BINS))
    # --- контроли фигуры 4
    for g, ex in zip(GROUPS, F["EXPECT_F4_N"]):
        ck(f"Ф4: n группы {g}", v(V, 4, "a", f"{g}: n МО", "все"), ex)
        ck(f"Ф4: n группы {g} (колонка n)", vn(V, 4, "a", f"{g}: n МО", "все"), ex)
    for g, ex in zip(GROUPS, F["EXPECT_F4_TRANSPORT"]):
        ck(f"Ф4: медиана транспорта {g}", v(V, 4, "a", f"{g}: медиана Транспорт", "все"), ex, F["TOL_F4_MEDIAN"])
    tri = []
    for a, b in zip(GROUPS, GROUPS[1:]):
        m = f"изменение медианы {a} → {b}"
        d_, lo, hi = (v(V, 4, "a, ▲", f"{m}, п.п.", "все"), v(V, 4, "a, ▲", f"{m}: нижняя граница 95% интервала", "все"), v(V, 4, "a, ▲", f"{m}: верхняя граница 95% интервала", "все"))
        tri.append((a, b, d_, lo, hi, bool(lo > 0 or hi < 0)))
    for a, b, ex_d, ex_lo, ex_hi in F["EXPECT_F4_DIFF"]:
        row = next(r for r in tri if r[0] == a and r[1] == b)
        ck(f"Ф4: разность {a} → {b}", row[2], ex_d, F["TOL_F4_DIFF"])
        ck(f"Ф4: нижняя граница {a} → {b}", row[3], ex_lo, F["TOL_F4_DIFF"])
        ck(f"Ф4: верхняя граница {a} → {b}", row[4], ex_hi, F["TOL_F4_DIFF"])
    ck("Ф4: ▲ над переходами", ";".join(f"{a}→{b}" for a, b, *_, s in tri if s), ";".join(f"{a}→{b}" for a, b, *_ in F["EXPECT_F4_DIFF"]))
    n_sum = sum(vn(V, 4, "a", f"{g}: n МО", "все") for g in GROUPS)
    ck("Ф4: n групп в сумме = 2004 − МО без населения", n_sum, 2004 - int(v(V, 4, "под осью", "МО без населения 2023", "все")))
    # --- контроли фигуры 5
    for t in range(K):
        ck(f"Ф5: PS тип {t}", v(V, 5, "a", "PS типа (среднее доля пар)", str(t)), F["EXPECT_F5_PS"][t], F["TOL_F5_SHARE"])
        ck(f"Ф5: сопоставлений тип {t}", v(V, 5, "b", "уверенных сопоставлений из 23", str(t)), F["EXPECT_F5_MATCH"][t])
        ck(f"Ф5: Г6 S тип {t}", v(V, 5, "c", "Г6: S по типу", str(t)), F["EXPECT_F5_G6"][t][0], F["TOL_F5_SHARE"])
        ck(f"Ф5: Г6 базовый уровень тип {t}", v(V, 5, "c", "Г6: базовый уровень", str(t)), F["EXPECT_F5_G6"][t][1], F["TOL_F5_SHARE"])
    ps_up = pd.read_parquet(P(F["PS_PATH"]))
    mt_up = pd.read_parquet(P(F["MATCHING_PATH"]))
    up_bad = 0
    for t in range(K):
        p_ = ps_up[(ps_up["k"] == K) & (ps_up["доминирующий тип k=6"] == t)]
        m_ = mt_up[(mt_up["cluster"] == t) & (mt_up["месяц"] != C.MONTH)]
        up_bad += abs(float(p_["доля пар"].mean()) - v(V, 5, "a", "PS типа (среднее доля пар)", str(t))) > F["SOURCE_TOL"]
        up_bad += int((m_["уверенность"] == "уверенно").sum()) != int(v(V, 5, "b", "уверенных сопоставлений из 23", str(t)))
        up_bad += len(m_) != int(v(V, 5, "b", "месяцев без якоря", str(t)))
    ck("Ф5: PS и сопоставления против исходных файлов шагов 10d и 12 (расхождений)", up_bad, 0)
    bad = [r for r in CK if r["статус"] != "совпало"]
    for r in CK:
        print(f"{r['статус']:12s} {r['контроль']}: {r['получено']} (ожидается {r['ожидается']}, допуск {r['допуск']})")
    if bad:
        stop("расхождения с контрольными значениями:\n" + pd.DataFrame(bad).to_string(index=False))
    sizes = [int(vn(V, 1, "a", "медиана Продовольствие, %", str(t))) for t in range(K)]
    # --- рисование
    figs = {}
    cv1 = draw_fig1(V, names, z, za, s23, sizes)
    cv3, cap3 = draw_fig3(V)
    cv4, cap4 = draw_fig4(V, tri)
    cv5, cap5 = draw_fig5(V)
    capt = {"fig1": "", "fig3": cap3, "fig4": cap4, "fig5": cap5}
    cvs = {"fig1": cv1, "fig3": cv3, "fig4": cv4, "fig5": cv5}
    lay_fail = False
    layout = {}
    for k, cv in cvs.items():
        rp = layout_report(cv)
        layout[k] = rp
        print(f"{k}: поля min {rp['margin_min']:.3f} дюйма; перекрытий текстов {len(rp['overlaps'])}; текстов {rp['n_texts']}; "
              f"запрещённых слов {len(rp['bad_words'])}; скобок вне интервала {len(rp['bad_brackets'])}")
        ck(f"{k}: поля холста не менее {F['MIN_MARGIN_IN']} дюйма", int(rp["margin_ok"]), 1)
        ck(f"{k}: перекрытий рамок текстов", len(rp["overlaps"]), 0)
        ck(f"{k}: запрещённых слов в текстах рисунка", len(rp["bad_words"]), 0)
        ck(f"{k}: скобок вне формата интервала", len(rp["bad_brackets"]), 0)
        if rp["overlaps"]:
            print("  перекрытия:", rp["overlaps"][:10])
        lay_fail |= (not rp["margin_ok"]) or bool(rp["overlaps"]) or bool(rp["bad_words"]) or bool(rp["bad_brackets"])
    ck("шрифт рисунков", plt.rcParams["font.family"][0], F["FONT"])
    if lay_fail:
        stop("компоновка или тексты рисунков не прошли проверки (см. вывод выше)")
    out = {}
    for k, cv in cvs.items():
        out[k] = pngs_svgs(cv)
    check_frozen("после")
    fd = pd.DataFrame(FD)[FD_COLS]
    fd["n"] = fd["n"].astype("Int64")
    fd = fd.sort_values(["фигура", "панель", "ключ", "тип"], kind="stable").reset_index(drop=True)
    if fd.duplicated(["фигура", "панель", "ключ", "тип"]).any():
        stop("дубликаты ключа figure_data")
    report = build_report(fd, capt, layout, out)
    low = report.lower()
    hits = {w: len(re.findall(w, low)) for w in F["FORBIDDEN_WORDS"]}
    nobr = BRACKET_OK.sub("", report)
    if any(hits.values()) or "[" in nobr or "]" in nobr:
        stop(f"в md запрещённые слова {hits} или скобки вне формата интервала: "
             f"{[ln[:120] for ln in nobr.splitlines() if '[' in ln or ']' in ln][:5]}")
    for k, name in F["NAMES"].items():
        png, svg = out[k]
        P(f"{F['OUT_DIR']}/{name}.png").write_bytes(png)
        P(f"{F['OUT_DIR']}/{name}.svg").write_bytes(svg)
    fd.to_parquet(P(F["FIGURE_DATA_PATH"]), engine="pyarrow", index=False)
    P(F["REPORT_PATH"]).write_text(report, encoding="utf-8")
    print(f"записано: 4 рисунка (png, svg), {Path(F['FIGURE_DATA_PATH']).name} ({len(fd)} строк), {Path(F['REPORT_PATH']).name}")


def build_report(fd: pd.DataFrame, capt: dict, layout: dict, out: dict) -> str:
    V = make_V()
    n_tot = sum(vn(V, 4, "a", f"{g}: n МО", "все") for g in GROUPS)
    L = ["# 24. Фигуры 1, 3, 4, 5", "",
         "Сгенерировано `src/24_figures.py`. Описательный шаг: данные не пересчитываются, на рисунки попадают только числа из parquet шагов 19–28 "
         "(`data/processed/figure_data_24.parquet`). Технические имена типов — функция `full_type_name` шага 23 (сверены с `notebooks/23_type_portraits.md`); "
         "интерпретационные имена на рисунки не входят. Фигура 2 (карта) — шаг 24d.", "",
         "## Заморозка", "", *LOG, "",
         "## Файлы", ""]
    for k, name in F["NAMES"].items():
        L.append(f"- `{F['OUT_DIR']}/{name}.png` ({len(out[k][0])} байт), `{F['OUT_DIR']}/{name}.svg` ({len(out[k][1])} байт)")
    L += ["", "## Фигура 1. Профили типов", "",
          "Подпись. Три панели для типов 0–5 и пяти категорий: (а) медиана доли расходов в «Все категории», %, декабрь 2024; (б) профиль σ по долям; "
          "(в) профиль σ по log(доля / «прочее»). Штриховка в (б) и (в) — пометка «зависит от записи» (feature_mark шага 23); «*» — признак из пометки "
          "«на границе порога» (border_features шага 23). Под рисунком технические имена типов. Источник: `type_portraits.parquet`, блоки «расходы» и «признаки профиля»; "
          f"n по типам: {', '.join(str(vn(V, 1, 'a', 'медиана Продовольствие, %', str(t))) for t in range(K))}. Ограничение: шкала цвета в (а) относительная внутри строки, "
          "сравнивать нужно числа; σ в (б) и (в) — средние z-score типа, не проверки.", "",
          fig_table("фигура 1"), "",
          "Что на рисунке не показано:", "",
          "- разброс внутри типов (квартили, интервалы): только медианы и средние σ;",
          "- декабрь 2024 — единственный месяц; остальные 23 месяца и переходы между типами не показаны (фигура 5, часть в);",
          "- МО вне выборки (шаги 24c, 24g) в профили не входят.", "",
          "## Фигура 3. Маркетплейсы и market_access на двух уровнях", "",
          "Подпись. " + capt["fig3"].replace("\n", " "), "",
          "Ограничения: ρ — по МО выборки шага 25 (регионы с не менее чем указанным числом МО); «между регионами» — ρ медиан регионов; "
          "значения шага 25 и шага 19b посчитаны по разным выборкам, поэтому внутри регионов приведено значение шага 25.", "",
          fig_table("фигура 3"), "",
          "Что на рисунке не показано:", "",
          "- уровень «все МО» (в таблице шага 26);",
          "- значения по месяцам и интервалы ρ (в шаге 25 для Г2 их нет);",
          "- диаграмма рассеяния.", "",
          "## Фигура 4. Размер МО", "",
          "Подпись. " + capt["fig4"].replace("\n", " "), "",
          "Ограничения: группы размера условные (границы — config шага 23); интервалы разностей получены бутстрепом шага 23 по МО двух соседних групп, "
          "а не по типам; значения отношения к «прочему» в (б) — медианы отношения.", "",
          fig_table("фигура 4"), "",
          "Что на рисунке не показано:", "",
          f"- {int(v(V, 4, 'под осью', 'МО без населения 2023', 'все'))} МО без населения 2023 (в группы не входят; n групп {n_tot});",
          "- малые группы (более 300 тыс. жителей: n = " + str(vn(V, 4, "a", "более 300: n МО", "все")) + ") показаны без интервалов;",
          "- усы у медиан не рисуются; интервалы только у разностей соседних групп (отметка ▲);",
          "- остальные категории расходов и деление по типам.", "",
          "## Фигура 5. Надёжность типов", "",
          "Подпись. " + capt["fig5"].replace("\n", " "), "",
          "Ограничения: PS — среднее по строкам prediction strength с данным доминирующим типом при k = 6 (n — число строк); "
          "число сопоставлений — из 23 месяцев без якоря; Г6 — описательное свойство.", "",
          fig_table("фигура 5"), "",
          "Что на рисунке не показано:", "",
          "- интервалы для PS и S; поправки и вердикты;",
          "- Г6 при числе ближайших типов, отличном от основного варианта.", "",
          "## Контроли", "", md(pd.DataFrame(CK)), "",
          "## Осмотр компоновки (программный)", ""]
    for k, rp in layout.items():
        L.append(f"- {k}: поля холста не менее {rp['margin_min']:.2f} дюйма, перекрытий рамок текстов {len(rp['overlaps'])}, "
                 f"запрещённых слов {len(rp['bad_words'])}, скобок вне интервала {len(rp['bad_brackets'])}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
