"""Шаг 24б. Геометрия МО для карты: справочник границ, упрощение покрытия, отчёт и подпись источника.

Что делает:
  1. находит gpkg справочника МО СберИндекса в MUNICIPAL_DICT_DIR; если файла нет, скачивает архив
     по https (сертификаты Минцифры certs/ca_bundle.pem подставляются только в запрос, системное
     хранилище не меняется), сверяет sha256 архива и gpkg с config.yaml и распаковывает bsdtar;
  2. выбирает полигоны, действующие в декабре 2024 (правило year_to == 9999; список территорий
     сверяется с xlsx справочника), помечает МО рабочей выборки (kmeans_labels_final.parquet);
  3. упрощает всё покрытие одним вызовом shapely.coverage_simplify: карта страны (допуск
     NATIONAL_TOLERANCE_DEG) и отдельно Москва и Санкт-Петербург для врезок (CITIES_TOLERANCE_DEG);
  4. пишет data/geo/mo_national.geojson, mo_cities.geojson, geometry_report.md/.json, ATTRIBUTION.md.
Все метрики считаются по итоговым файлам (после округления координат); все контроли выполняются ДО записи в data/geo/:
если хотя бы один не пройден, скрипт печатает таблицу контролей, пишет кандидаты в каталог временных файлов и
завершается с кодом 1, не изменяя data/geo/. Параметры — в config.yaml,
группа step24b_geometry. Площади в км² в отчёты не печатаются: |ΔS| считается как относительное
изменение площади (равновеликая проекция, формула шага 0б).
Вход:  справочник МО (gpkg и xlsx, вне проекта), data/processed/kmeans_labels_final.parquet,
       data/raw/territories.parquet
Выход: data/geo/*  (при --out-dir другая папка)
Запуск из корня проекта:  .venv/bin/python src/24b_geometry.py
Нужен shapely==2.1.2 (coverage_simplify: GEOS 3.12 и новее) и bsdtar (только при скачивании).
"""
import argparse
import collections
import gzip
import hashlib
import json
import sqlite3
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from shapely.validation import explain_validity

import config as C

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
LABELS_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
TERRITORIES_PATH = PROJECT_DIR / "data" / "raw" / "territories.parquet"
GPKG_NAME = "t_dict_municipal_districts_poly.gpkg"
XLSX_NAME = "t_dict_municipal_districts.xlsx"
GPKG_TABLE = "t_dict_municipal_districts_poly"
ARCHIVE_NAME = "t_dict_municipal.rar"
SOURCE_URL = "https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/legalcode"

warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- справочник
def ensure_dictionary(dict_dir: Path) -> tuple[Path, Path, bool]:
    gpkg, xlsx = dict_dir / GPKG_NAME, dict_dir / XLSX_NAME
    downloaded = False
    if not gpkg.exists():
        dict_dir.mkdir(parents=True, exist_ok=True)
        cafile = PROJECT_DIR / C.GEO_CA_BUNDLE
        if not cafile.exists():
            stop(f"нет файла сертификатов {cafile}")
        ctx = ssl.create_default_context(cafile=str(cafile))   # только этот бандл, системное хранилище не читается
        req = urllib.request.Request(C.GEO_ARCHIVE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=600) as r:
            data = r.read()
        got = hashlib.sha256(data).hexdigest()
        if got != C.GEO_ARCHIVE_SHA256:
            stop(f"sha256 архива {got} не совпадает с ожидаемым {C.GEO_ARCHIVE_SHA256}")
        arc = dict_dir / ARCHIVE_NAME
        arc.write_bytes(data)
        res = subprocess.run(["bsdtar", "-xf", str(arc), "-C", str(dict_dir)], capture_output=True, text=True)
        if res.returncode != 0:
            stop(f"bsdtar не распаковал архив: {res.stderr.strip()}")
        downloaded = True
    if not gpkg.exists() or not xlsx.exists():
        stop(f"в {dict_dir} нет {GPKG_NAME} или {XLSX_NAME}")
    got = sha256_file(gpkg)
    if got != C.GEO_GPKG_SHA256:
        stop(f"sha256 gpkg {got} не совпадает с ожидаемым {C.GEO_GPKG_SHA256}")
    return gpkg, xlsx, downloaded


def parse_gpkg_blob(blob: bytes) -> shapely.Geometry:
    if blob[:2] != b"GP":
        stop("заголовок GPKG не найден")
    flags = blob[3]
    if flags & 0b100000 or (flags >> 4) & 1:
        stop("неожиданный тип геометрии GPKG (ExtendedGeoPackageBinary или пустая)")
    env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    return shapely.from_wkb(bytes(blob[8 + env:]))


def load_dictionary(gpkg: Path, xlsx: Path):
    con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    rows = con.execute(f'select territory_id, year_from, year_to, geom from "{GPKG_TABLE}"').fetchall()
    con.close()
    tab = pd.DataFrame({"territory_id": [int(r[0]) for r in rows], "year_from": [r[1] for r in rows],
                        "year_to": [r[2] for r in rows]})
    if tab.territory_id.duplicated().any():
        stop("в gpkg повторяются territory_id")
    geoms = {int(r[0]): parse_gpkg_blob(r[3]) for r in rows}
    x = pd.read_excel(xlsx)
    return tab, geoms, x


# ---------------------------------------------------------------- площадь, перекрытия, соседи
_A, _F = 6378137.0, 1 / 298.257223563
_E2 = _F * (2 - _F)
_E = np.sqrt(_E2)


def _q(s):
    return (1 - _E2) * (s / (1 - _E2 * s ** 2) - (1 / (2 * _E)) * np.log((1 - _E * s) / (1 + _E * s)))


_QP = float(_q(np.array(1.0)))
_RA = _A * np.sqrt(_QP / 2)


def ring_area(ring: np.ndarray) -> float:
    """Равновеликая цилиндрическая проекция на авталической сфере (как geom_area в шаге 0б), м²."""
    lon, phi = np.radians(ring[:, 0]), np.radians(ring[:, 1])
    s = _q(np.sin(phi)) / _QP
    return abs(0.5 * _RA ** 2 * np.sum(np.diff(lon) * (s[:-1] + s[1:])))


def polygons_of(g):
    if g is None or g.is_empty:
        return []
    return list(g.geoms) if g.geom_type in ("MultiPolygon", "GeometryCollection") else [g]


def geom_area(g) -> float:
    tot = 0.0
    for p in polygons_of(g):
        tot += ring_area(np.asarray(p.exterior.coords)) - sum(ring_area(np.asarray(h.coords)) for h in p.interiors)
    return tot


def counts(g) -> tuple[int, int]:
    ps = polygons_of(g)
    return len(ps), sum(len(p.interiors) for p in ps)


def unmatched_parts(p0: list, p1: list) -> list[int]:
    """Части исходного МО без пары в итоговом: жадное сопоставление по расстоянию между центроидами частей."""
    if len(p1) >= len(p0):
        return []
    c0 = np.array([[q.centroid.x, q.centroid.y] for q in p0])
    c1 = np.array([[q.centroid.x, q.centroid.y] for q in p1]) if p1 else np.zeros((0, 2))
    D = np.sqrt(((c0[:, None, :] - c1[None, :, :]) ** 2).sum(2)) if len(p1) else np.zeros((len(p0), 0))
    free = set(range(len(p0)))
    for _ in range(len(p1)):
        r, c = np.unravel_index(np.argmin(D), D.shape)
        free.discard(int(r))
        D[r, :] = np.inf
        D[:, c] = np.inf
    return sorted(free)


def lost_parts(ids: list[int], names: list[str], orig, pre, final) -> list[dict]:
    """МО, у которых после упрощения и округления пропали части: число частей на каждом этапе и площади потерянных."""
    rows = []
    for k, i in enumerate(ids):
        p0, pp, pf = polygons_of(orig[k]), polygons_of(pre[k]), polygons_of(final[k])
        lost = unmatched_parts(p0, pf)
        if not lost:
            continue
        rows.append({"territory_id": int(i), "name": names[k], "parts_before": len(p0), "parts_after_simplify": len(pp),
                     "parts_after_rounding": len(pf),
                     "lost": [{"area_deg2": float(p0[j].area), "area_m2": float(geom_area(p0[j])),
                               "large": bool(p0[j].area >= C.GEO_MAX_LOST_PART_AREA)} for j in lost]})
    return rows


def overlap_pairs(ids: list[int], geoms: np.ndarray, min_area: float):
    """Пары с площадью пересечения выше min_area. Для недопустимых полигонов (упрощённые файлы) пересечение
    считается через make_valid только внутри измерения (в файлы make_valid не попадает); число таких пар возвращается."""
    tree = shapely.STRtree(geoms)
    out, fallback = [], 0
    for k, g in enumerate(geoms):
        if g.is_empty:
            continue
        for j in tree.query(g, predicate="intersects"):
            if j <= k:
                continue
            try:
                a = shapely.intersection(g, geoms[j]).area
            except shapely.errors.GEOSException:
                fallback += 1
                a = shapely.intersection(shapely.make_valid(g), shapely.make_valid(geoms[j])).area
            if a > min_area:
                out.append((ids[k], ids[j]))
    return out, fallback


def build_pairs(ids: list[int], geoms: np.ndarray):
    """Пары МО с >= 2 точно совпадающими вершинами исходных полигонов; на пару до GAP_SAMPLE вершин."""
    xs, own = [], []
    for k, g in enumerate(geoms):
        for p in polygons_of(g):
            for r in [p.exterior, *p.interiors]:
                c = np.asarray(r.coords)[:-1]
                xs.append(c)
                own.append(np.full(len(c), k, dtype=np.int64))
    X, O = np.concatenate(xs), np.concatenate(own)
    order = np.lexsort((X[:, 1], X[:, 0]))
    Xs, Os = X[order], O[order]
    newg = np.r_[True, (Xs[1:] != Xs[:-1]).any(axis=1)]
    starts = np.nonzero(newg)[0]
    ends = np.r_[starts[1:], len(Xs)]
    pa, pb, px = [], [], []
    for s, e in zip(starts[ends - starts >= 2], ends[ends - starts >= 2]):
        u = np.unique(Os[s:e])
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                pa.append(u[i]); pb.append(u[j]); px.append(Xs[s])
    pa, pb, px = np.array(pa), np.array(pb), np.array(px)
    n = len(ids)
    key = pa * n + pb
    o2 = np.lexsort((px[:, 1], px[:, 0], key))
    key, px = key[o2], px[o2]
    ukeys, first, cnt = np.unique(key, return_index=True, return_counts=True)
    rng = np.random.default_rng(C.GEO_GAP_SEED)
    pairs = {}
    for kk, f, c in zip(ukeys, first, cnt):
        if c < 2:
            continue
        V = px[f:f + c]
        if c > C.GEO_GAP_SAMPLE:
            V = V[np.sort(rng.choice(c, C.GEO_GAP_SAMPLE, replace=False))]
        pairs[(int(kk // n), int(kk % n))] = (int(c), V)
    return pairs


def segments(g) -> np.ndarray:
    parts = []
    for p in polygons_of(g):
        for r in [p.exterior, *p.interiors]:
            c = np.asarray(r.coords)
            parts.append(np.hstack([c[:-1], c[1:]]))
    return np.vstack(parts) if parts else np.zeros((0, 4))


def dist_inside(P: np.ndarray, Sg: np.ndarray):
    a, b = Sg[:, :2], Sg[:, 2:]
    ab = b - a
    L2 = (ab ** 2).sum(1)
    L2[L2 == 0] = 1e-30
    D = np.empty(len(P))
    IN = np.zeros(len(P), bool)
    for c in range(0, len(P), 200):
        p = P[c:c + 200]
        t = np.clip(((p[:, None, 0] - a[None, :, 0]) * ab[None, :, 0] + (p[:, None, 1] - a[None, :, 1]) * ab[None, :, 1]) / L2[None, :], 0, 1)
        qx, qy = a[None, :, 0] + t * ab[None, :, 0], a[None, :, 1] + t * ab[None, :, 1]
        D[c:c + 200] = np.sqrt(((p[:, None, 0] - qx) ** 2 + (p[:, None, 1] - qy) ** 2).min(1))
        x, y = p[:, None, 0], p[:, None, 1]
        x1, y1, x2, y2 = a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1]
        cr = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / np.where(y2 == y1, 1, y2 - y1) + x1)
        IN[c:c + 200] = (cr.sum(1) % 2) == 1
    return D, IN


def gap_widths(pairs, geoms: np.ndarray, keep: set | None = None):
    """Максимум по общим вершинам |p_A - p_B|: расхождение двух упрощённых границ вдоль исходной общей границы."""
    segs, res = {}, {}

    def sg(k):
        if k not in segs:
            segs[k] = segments(geoms[k])
        return segs[k]
    for (a, b), (n, V) in pairs.items():
        if keep is not None and (a not in keep or b not in keep):
            continue
        Sa, Sb = sg(a), sg(b)
        if len(Sa) == 0 or len(Sb) == 0:
            res[(a, b)] = np.nan
            continue
        dA, inA = dist_inside(V, Sa)
        dB, inB = dist_inside(V, Sb)
        res[(a, b)] = float(np.abs(np.where(inA, dA, -dA) - np.where(inB, -dB, dB)).max())
    return res


# ---------------------------------------------------------------- GeoJSON
def ring_coords(ring, decimals: int):
    q = np.round(np.asarray(ring.coords), decimals)
    keep = np.ones(len(q), bool)
    keep[1:] = np.any(q[1:] != q[:-1], axis=1)
    q = q[keep]
    if len(q) < 4 or not np.array_equal(q[0], q[-1]):
        return None
    return q.tolist()


def feature_json(props: dict, g, decimals: int) -> str:
    parts = []
    for p in polygons_of(g):
        shell = ring_coords(p.exterior, decimals)
        if shell is None:
            continue
        rings = [shell] + [h for h in (ring_coords(r, decimals) for r in p.interiors) if h is not None]
        parts.append(rings)
    feat = {"type": "Feature", "properties": props, "geometry": {"type": "MultiPolygon", "coordinates": parts}}
    return json.dumps(feat, separators=(",", ":"), ensure_ascii=False)


def make_geojson(feats: list[str]) -> bytes:
    return ('{"type":"FeatureCollection","features":[\n' + ",\n".join(feats) + "\n]}\n").encode("utf-8")


def read_geojson_geoms(data: bytes) -> dict[int, shapely.Geometry]:
    out = {}
    for f in json.loads(data.decode("utf-8"))["features"]:
        parts = [shapely.Polygon(r[0], r[1:]) for r in f["geometry"]["coordinates"]]
        out[int(f["properties"]["territory_id"])] = shapely.MultiPolygon(parts)
    return out


# ---------------------------------------------------------------- метрики файла
def file_metrics(name: str, data: bytes, ids: list[int], names: list[str], orig: np.ndarray, pre: np.ndarray,
                 in_sample: np.ndarray, pairs_all):
    back = read_geojson_geoms(data)
    simp = np.array([back[i] for i in ids], dtype=object)
    n = len(ids)
    a0 = np.array([geom_area(g) for g in orig])
    a1 = np.array([geom_area(g) for g in simp])
    rel = (a1 - a0) / a0
    big = np.abs(rel) > C.GEO_AREA_CHANGE_THRESHOLD
    cb = np.array([counts(g) for g in orig])
    ca = np.array([counts(g) for g in simp])
    gw = gap_widths(pairs_all, simp, keep=set(range(n)))
    w = np.array(list(gw.values()))
    over = [k for k, v in gw.items() if v > C.GEO_GAP_THRESHOLD]
    ov, ov_fallback = overlap_pairs(ids, simp, C.GEO_OVERLAP_MIN_AREA)
    inv_edges = shapely.coverage_invalid_edges(simp)
    with np.errstate(all="ignore"):
        cov_ok = bool(shapely.coverage_is_valid(simp))
    lost_rows = lost_parts(ids, names, orig, pre, simp)
    inv_rows = [{"territory_id": int(ids[k]), "name": names[k], "explain_validity": explain_validity(simp[k]).replace("[", "(").replace("]", ")")}
                for k in range(n) if not shapely.is_valid(simp[k])]
    m = {
        "lost_parts": lost_rows,
        "lost_parts_large": int(sum(1 for r in lost_rows for q in r["lost"] if q["large"])),
        "lost_parts_small": int(sum(1 for r in lost_rows for q in r["lost"] if not q["large"])),
        "invalid_after": inv_rows,
        "file": name, "bytes": len(data), "bytes_gzip9": len(gzip.compress(data, 9, mtime=0)),
        "polygons": n, "vertices": int(sum(shapely.get_num_coordinates(g) for g in simp)),
        "vertices_original": int(sum(shapely.get_num_coordinates(g) for g in orig)),
        "sample_polygons": int(in_sample.sum()),
        "area_change_gt_threshold_sample": int((big & in_sample).sum()),
        "area_change_gt_threshold_all": int(big.sum()),
        "area_change_ids_sample": [ids[k] for k in np.nonzero(big & in_sample)[0]],
        "area_change_ids_not_sample": [ids[k] for k in np.nonzero(big & ~in_sample)[0]],
        "max_abs_area_change_sample": float(np.abs(rel[in_sample]).max()),
        "parts_before": int(cb[:, 0].sum()), "parts_after": int(ca[:, 0].sum()),
        "polygons_with_fewer_parts": int((ca[:, 0] < cb[:, 0]).sum()),
        "holes_before": int(cb[:, 1].sum()), "holes_after": int(ca[:, 1].sum()),
        "polygons_with_fewer_holes": int((ca[:, 1] < cb[:, 1]).sum()),
        "empty_polygons": int(sum(1 for g in simp if g.is_empty)),
        "is_valid_original_invalid": int((~shapely.is_valid(orig)).sum()),
        "is_valid_simplified_invalid": int((~shapely.is_valid(simp)).sum()),
        "neighbour_pairs": len(gw), "pairs_gap_gt_threshold": len(over),
        "pairs_with_empty_geometry": int(np.isnan(w).sum()) if len(w) else 0,
        "gap_max_deg": float(np.nanmax(w)) if len(w) else None,
        "gap_p95_deg": float(np.nanpercentile(w, 95)) if len(w) else None,
        "overlap_pairs_simplified": len(ov), "overlap_pairs_measured_via_make_valid": ov_fallback,
        "coverage_is_valid_simplified": cov_ok,
        "coverage_invalid_edge_polygons_simplified": int(sum(1 for e in inv_edges if e is not None and not e.is_empty)),
    }
    keep_s = set(np.nonzero(in_sample)[0].tolist())
    gw_s = {k: v for k, v in gw.items() if k[0] in keep_s and k[1] in keep_s}
    m["neighbour_pairs_both_in_sample"] = len(gw_s)
    m["pairs_gap_gt_threshold_both_in_sample"] = int(sum(1 for v in gw_s.values() if v > C.GEO_GAP_THRESHOLD))
    return m


def md_report(R: dict) -> str:
    o, nat, cit = R["original"], R["national"], R["cities"]
    L = [
        "# Геометрия МО для карты (шаг 24б)", "",
        "Сгенерировано `src/24b_geometry.py`. Все числа взяты из расчёта скрипта; параметры — `config.yaml`, "
        "группа `step24b_geometry`. Метрики файлов считаны по записанным GeoJSON (после округления координат). "
        "Площади в км² не печатаются: изменение площади |ΔS| считается как относительное.", "",
        "## Источник и входы", "",
        f"- справочник: `{R['source_url']}`; срез: полигоны, действующие в {R['snapshot_month']} (правило `year_to = {R['active_year_to']}`); "
        f"список территорий сверен с xlsx справочника (совпал)",
        f"- sha256 gpkg: `{R['inputs']['gpkg_sha256']}`",
        f"- sha256 xlsx: `{R['inputs']['xlsx_sha256']}`",
        f"- sha256 архива (ожидаемый, config; проверяется при скачивании): `{R['inputs']['archive_sha256_expected']}`",
        f"- sha256 `kmeans_labels_final.parquet`: `{R['inputs']['labels_sha256']}`",
        f"- sha256 `territories.parquet`: `{R['inputs']['territories_sha256']}`",
        f"- shapely {R['versions']['shapely']}, GEOS {R['versions']['geos']}, numpy {R['versions']['numpy']}", "",
        "## Исходное покрытие (по полному набору действующих полигонов)", "",
        "| показатель | значение |", "|---|---|",
        f"| действующих полигонов | {o['active_polygons']} |",
        f"| из них в рабочей выборке | {o['sample_polygons']} |",
        f"| вне выборки | {o['not_sample_polygons']} |",
        f"| вершин | {o['vertices']} |",
        f"| shapely.is_valid: недопустимых полигонов | {o['is_valid_invalid']} |",
        f"| shapely.coverage_is_valid | {o['coverage_is_valid']} |",
        f"| полигонов с неверными рёбрами (coverage_invalid_edges) | {o['coverage_invalid_edge_polygons']} |",
        f"| пар с перекрытием (площадь пересечения больше {R['overlap_min_area_deg2']} град²) | {o['overlap_pairs']} |",
        f"| из них обе стороны в выборке | {o['overlap_pairs_both_in_sample']} |",
        f"| из них с участием МО вне выборки | {o['overlap_pairs_with_not_sample']} |", "",
        "Территориальные id с неверными рёбрами: " + ", ".join(map(str, o["coverage_invalid_edge_ids"])) + ".", "",
        "Пары с перекрытием (territory_id): " + ", ".join(f"{a}–{b}" for a, b in o["overlap_pair_ids"]) + ".", "",
        "## Упрощение (shapely.coverage_simplify)", "",
        "| показатель | mo_national.geojson | mo_cities.geojson |", "|---|---|---|",
        f"| допуск, градусы | {R['national_tolerance_deg']} | {R['cities_tolerance_deg']} |",
        f"| знаков координат | {R['national_decimals']} | {R['cities_decimals']} |",
        f"| полигонов | {nat['polygons']} | {cit['polygons']} |",
        f"| из них в выборке | {nat['sample_polygons']} | {cit['sample_polygons']} |",
        f"| вершин (исходных вершин) | {nat['vertices']} ({nat['vertices_original']}) | {cit['vertices']} ({cit['vertices_original']}) |",
        f"| размер файла, байт | {nat['bytes']} | {cit['bytes']} |",
        f"| размер, gzip -9, байт | {nat['bytes_gzip9']} | {cit['bytes_gzip9']} |",
        f"| МО выборки с \\|ΔS\\| > {R['area_change_threshold']} | {nat['area_change_gt_threshold_sample']} | {cit['area_change_gt_threshold_sample']} |",
        f"| МО вне выборки с \\|ΔS\\| > {R['area_change_threshold']} | {len(nat['area_change_ids_not_sample'])} | {len(cit['area_change_ids_not_sample'])} |",
        f"| максимум \\|ΔS\\| среди МО выборки | {nat['max_abs_area_change_sample']:.4f} | {cit['max_abs_area_change_sample']:.4f} |",
        f"| частей до / после | {nat['parts_before']} / {nat['parts_after']} | {cit['parts_before']} / {cit['parts_after']} |",
        f"| МО с потерей частей | {nat['polygons_with_fewer_parts']} | {cit['polygons_with_fewer_parts']} |",
        f"| отверстий до / после | {nat['holes_before']} / {nat['holes_after']} | {cit['holes_before']} / {cit['holes_after']} |",
        f"| МО с потерей отверстий | {nat['polygons_with_fewer_holes']} | {cit['polygons_with_fewer_holes']} |",
        f"| пустых полигонов | {nat['empty_polygons']} | {cit['empty_polygons']} |",
        f"| shapely.is_valid: недопустимых (исходные / упрощённые) | {nat['is_valid_original_invalid']} / {nat['is_valid_simplified_invalid']} | {cit['is_valid_original_invalid']} / {cit['is_valid_simplified_invalid']} |",
        f"| shapely.coverage_is_valid (упрощённые) | {nat['coverage_is_valid_simplified']} | {cit['coverage_is_valid_simplified']} |",
        f"| полигонов с неверными рёбрами (упрощённые) | {nat['coverage_invalid_edge_polygons_simplified']} | {cit['coverage_invalid_edge_polygons_simplified']} |",
        f"| пар с перекрытием (упрощённые) | {nat['overlap_pairs_simplified']} | {cit['overlap_pairs_simplified']} |",
        f"| из них измерено через make_valid (недопустимые полигоны) | {nat['overlap_pairs_measured_via_make_valid']} | {cit['overlap_pairs_measured_via_make_valid']} |",
        f"| пар соседей (общих вершин не менее 2) | {nat['neighbour_pairs']} | {cit['neighbour_pairs']} |",
        f"| пар с расхождением общей границы более {R['gap_threshold_deg']}° | {nat['pairs_gap_gt_threshold']} | {cit['pairs_gap_gt_threshold']} |",
        f"| из них пары, где обе стороны в выборке: пар / с расхождением | {nat['neighbour_pairs_both_in_sample']} / {nat['pairs_gap_gt_threshold_both_in_sample']} | {cit['neighbour_pairs_both_in_sample']} / {cit['pairs_gap_gt_threshold_both_in_sample']} |",
        f"| максимум расхождения, градусы | {nat['gap_max_deg']:.5f} | {cit['gap_max_deg']:.5f} |",
        f"| 95-й процентиль расхождения, градусы | {nat['gap_p95_deg']:.5f} | {cit['gap_p95_deg']:.5f} |",
        f"| sha256 файла | `{nat['sha256']}` | `{cit['sha256']}` |", "",
        "МО выборки с \\|ΔS\\| выше порога в mo_national.geojson (territory_id): " +
        (", ".join(map(str, nat["area_change_ids_sample"])) or "нет") + ".", "",
        "Расхождение общей границы считается по точно совпадающим вершинам исходных полигонов (не более "
        f"{R['gap_sample_per_pair']} вершин на пару, seed {R['gap_seed']}): для каждой вершины сравниваются положения двух "
        "упрощённых границ относительно исходной, пара получает максимум. Метод тот же, что в шаге 0в.", "",
    ]
    thr = R["max_lost_part_area_deg2"]
    L += [f"## Исчезнувшие части (крупной считается часть площадью не менее {thr} град²)", "",
          "Часть сопоставляется с итоговой геометрией по расстоянию между центроидами; площадь в м² считается способом шага 0б "
          "(равновеликая проекция). Мелкие части, исчезнувшие при упрощении или округлении, нарушением не считаются.", ""]
    for key, label in (("national", "mo_national.geojson"), ("cities", "mo_cities.geojson")):
        f = R[key]
        L.append(f"{label}: крупных потерянных частей {f['lost_parts_large']}, мелких {f['lost_parts_small']}.")
        L.append("")
        if f["lost_parts"]:
            L += ["| territory_id | название | частей до упрощения | после упрощения | после округления | потерянные части: площадь, град² | площадь, м² |",
                  "|---|---|---|---|---|---|---|"]
            for r in f["lost_parts"]:
                L.append(f"| {r['territory_id']} | {r['name']} | {r['parts_before']} | {r['parts_after_simplify']} | {r['parts_after_rounding']} | "
                         + "; ".join(f"{q['area_deg2']:.3e}" for q in r["lost"]) + " | " + "; ".join(f"{q['area_m2']:.1f}" for q in r["lost"]) + " |")
            L.append("")
    L += ["## МО с недопустимой геометрией после упрощения и округления (shapely.is_valid, без исправлений)", ""]
    for key, label in (("national", "mo_national.geojson"), ("cities", "mo_cities.geojson")):
        f = R[key]
        L.append(f"{label}: недопустимых {len(f['invalid_after'])}.")
        L.append("")
        if f["invalid_after"]:
            L += ["| territory_id | название | shapely.explain_validity |", "|---|---|---|"]
            for r in f["invalid_after"]:
                L.append(f"| {r['territory_id']} | {r['name']} | {r['explain_validity']} |")
            L.append("")
    L += ["## Контроли", "", "| контроль | ожидание | получено | результат |", "|---|---|---|---|"]
    for c in R["controls"]:
        L.append(f"| {c['name']} | {c['expect']} | {c['got']} | {'пройден' if c['ok'] else 'НЕ ПРОЙДЕН'} |")
    return "\n".join(L) + "\n"


def md_attribution(R: dict) -> str:
    o = R["original"]
    d = R["data_download_date"]
    ru = (f"Данные о границах и преобразованиях муниципальных образований. СберИндекс. Данные доступны по адресу "
          f"{SOURCE_URL} (данные скачаны {d}).")
    en = (f"Data on the boundaries and transformations of municipalities. Sberindex. Available at "
          f"{SOURCE_URL} (data downloaded on {d}).")
    L = [
        "# Источник и лицензия геометрии МО (data/geo/)", "",
        "## Источник", "", ru, "", en, "",
        "Набор построен СберИндексом по данным Росстата и OpenStreetMap (по странице набора). "
        "Условия лицензии OpenStreetMap (ODbL) в проекте не проверялись. Атрибуция: Contains data © OpenStreetMap contributors.", "",
        "## Лицензия", "",
        f"Данные набора доступны по лицензии CC BY-SA 4.0 (по официальной странице набора): {LICENSE_URL}", "",
        "## Что изменено в файлах этой папки", "",
        "Файлы `mo_national.geojson` и `mo_cities.geojson` являются адаптацией исходных полигонов и распространяются на тех же "
        "условиях (CC BY-SA 4.0). Изменения:", "",
        f"- выбраны полигоны, действующие в {R['snapshot_month']} (`year_to = {R['active_year_to']}`), всего {o['active_polygons']};",
        f"- геометрия упрощена одним вызовом `shapely.coverage_simplify` ко всему покрытию: допуск {R['national_tolerance_deg']}° для "
        f"`mo_national.geojson` и {R['cities_tolerance_deg']}° для `mo_cities.geojson` (все действующие полигоны Москвы и Санкт-Петербурга, "
        "включая посёлки вне выборки);",
        f"- координаты округлены до {R['national_decimals']} знаков (`mo_national.geojson`) и {R['cities_decimals']} знаков (`mo_cities.geojson`);",
        "- атрибуты заменены: `territory_id`, `name`, `region_name` (из справочника), `in_sample` (входит ли МО в рабочую выборку проекта), "
        "в `mo_cities.geojson` ещё `city` (moscow или spb). Цвет и тип МО в файлы не записываются.", "",
        "Параметры упрощения и метрики качества: `data/geo/geometry_report.md`. Скрипт: `src/24b_geometry.py`.", "",
        "## Известные ошибки исходного справочника (не исправлены)", "",
        "Числа по полному покрытию действующих полигонов:", "",
        f"- пар МО с перекрытием: {o['overlap_pairs']} (в {o['overlap_pairs_both_in_sample']} обе стороны в рабочей выборке, "
        f"в {o['overlap_pairs_with_not_sample']} участвует МО вне выборки);",
        f"- полигонов с неверными рёбрами по `shapely.coverage_is_valid` (`coverage_invalid_edges`): {o['coverage_invalid_edge_polygons']} "
        f"из {o['active_polygons']}; shapely.coverage_is_valid для всего покрытия: {o['coverage_is_valid']};",
        f"- полигонов, недопустимых по `shapely.is_valid`: {o['is_valid_invalid']}.", "",
        "Перечень территориальных id: `data/geo/geometry_report.md`. Исходные ошибки не исправлялись.", "",
        "## Что показали проверки упрощённых файлов (подробности в `data/geo/geometry_report.md`)", "",
        f"- `mo_national.geojson`: МО, недопустимых по `shapely.is_valid` после упрощения и округления: {len(R['national']['invalid_after'])} "
        f"(перечень и shapely.explain_validity в отчёте, не исправлялись); пар с перекрытием: {R['national']['overlap_pairs_simplified']};",
        f"- `mo_cities.geojson`: недопустимых МО {len(R['cities']['invalid_after'])}; пар с перекрытием: {R['cities']['overlap_pairs_simplified']};",
        "- мелкие части МО, исчезнувшие при упрощении или округлении, перечислены в отчёте.", "",
        "## Подпись для карты", "",
        f"Границы МО: СберИндекс, «Данные о границах и преобразованиях муниципальных образований» (скачано {d}), CC BY-SA 4.0; "
        f"геометрия упрощена (shapely.coverage_simplify, допуск {R['national_tolerance_deg']}°, врезки Москвы и Санкт-Петербурга "
        f"{R['cities_tolerance_deg']}°). Набор построен по данным Росстата и OpenStreetMap: Contains data © OpenStreetMap contributors.", "",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--municipal-dict-dir", default=C.GEO_DICT_DIR, help="каталог справочника (по умолчанию из config.yaml)")
    ap.add_argument("--out-dir", default=str(PROJECT_DIR / "data" / "geo"))
    args = ap.parse_args()
    dict_dir = Path(args.municipal_dict_dir).expanduser()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpkg, xlsx, downloaded = ensure_dictionary(dict_dir)
    print(f"справочник: {dict_dir} (скачан в этом запуске: {downloaded}); sha256 gpkg совпал с config")
    tab, gdict, x = load_dictionary(gpkg, xlsx)

    # --- срез «действует в декабре 2024»: year_to == 9999, сверка с xlsx
    active = sorted(tab[tab.year_to == C.GEO_ACTIVE_YEAR_TO].territory_id)
    xl_active = set(x[x.year_to == C.GEO_ACTIVE_YEAR_TO].territory_id.astype(int))
    if set(active) != xl_active:
        stop(f"действующие территории gpkg ({len(active)}) и xlsx ({len(xl_active)}) не совпадают")
    if len(active) != C.GEO_EXPECT_ACTIVE:
        stop(f"действующих полигонов {len(active)}, ожидалось {C.GEO_EXPECT_ACTIVE}")
    labels = pd.read_parquet(LABELS_PATH)
    sample = set(labels.territory_id.astype(int))
    if not sample <= set(active) or len(sample) != C.GEO_EXPECT_SAMPLE:
        stop(f"выборка: {len(sample)} МО, из них действующих {len(sample & set(active))}, ожидалось {C.GEO_EXPECT_SAMPLE}")
    terr = pd.read_parquet(TERRITORIES_PATH).set_index("territory_id")
    if not set(active) <= set(terr.index):
        stop("часть действующих территорий отсутствует в territories.parquet")

    ids = active
    orig = np.array([gdict[i] for i in ids], dtype=object)
    in_sample = np.array([i in sample for i in ids])
    name = [str(terr.at[i, "name"]) for i in ids]
    region = [str(terr.at[i, "region_name"]) for i in ids]
    rcode = [int(terr.at[i, "region_code"]) for i in ids]
    print(f"действующих полигонов {len(ids)}, в выборке {int(in_sample.sum())}, вне выборки {int((~in_sample).sum())}")

    # --- исходное покрытие
    ov_orig, fb_orig = overlap_pairs(ids, orig, C.GEO_OVERLAP_MIN_AREA)
    if fb_orig:
        stop(f"в исходном покрытии {fb_orig} пар с ошибкой пересечения (недопустимая геометрия)")
    inv_orig = shapely.coverage_invalid_edges(orig)
    inv_ids = [ids[k] for k, e in enumerate(inv_orig) if e is not None and not e.is_empty]
    with np.errstate(all="ignore"):
        cov_orig = bool(shapely.coverage_is_valid(orig))
    smp = set(sample)
    both = sum(1 for a, b in ov_orig if a in smp and b in smp)
    original = {
        "active_polygons": len(ids), "sample_polygons": int(in_sample.sum()), "not_sample_polygons": int((~in_sample).sum()),
        "vertices": int(sum(shapely.get_num_coordinates(g) for g in orig)),
        "is_valid_invalid": int((~shapely.is_valid(orig)).sum()), "coverage_is_valid": cov_orig,
        "coverage_invalid_edge_polygons": len(inv_ids), "coverage_invalid_edge_ids": inv_ids,
        "overlap_pairs": len(ov_orig), "overlap_pair_ids": [list(p) for p in ov_orig],
        "overlap_pairs_both_in_sample": both, "overlap_pairs_with_not_sample": len(ov_orig) - both,
    }
    print("исходное покрытие:", {k: v for k, v in original.items() if not isinstance(v, list)})

    pairs_all = build_pairs(ids, orig)
    print(f"пар соседей (общих вершин не менее 2): {len(pairs_all)}")

    # --- карта страны: всё покрытие одним вызовом
    nat_geoms = shapely.coverage_simplify(orig, C.GEO_NATIONAL_TOL)
    feats = [feature_json({"territory_id": i, "name": name[k], "region_name": region[k], "in_sample": bool(in_sample[k])},
                          nat_geoms[k], C.GEO_NATIONAL_DECIMALS) for k, i in enumerate(ids)]
    nat_bytes = make_geojson(feats)
    nat = file_metrics("mo_national.geojson", nat_bytes, ids, name, orig, nat_geoms, in_sample, pairs_all)
    nat["sha256"] = hashlib.sha256(nat_bytes).hexdigest()

    # --- врезки: все действующие полигоны Москвы и Санкт-Петербурга, отдельное покрытие
    city_of = {v: k for k, v in C.GEO_CITY_REGION_CODES.items()}
    cidx = [k for k in range(len(ids)) if rcode[k] in city_of]
    cids = [ids[k] for k in cidx]
    corig = orig[cidx]
    cin = in_sample[cidx]
    cit_geoms = shapely.coverage_simplify(corig, C.GEO_CITIES_TOL)
    cfeats = [feature_json({"territory_id": ids[k], "name": name[k], "region_name": region[k], "in_sample": bool(in_sample[k]),
                            "city": city_of[rcode[k]]}, cit_geoms[j], C.GEO_CITIES_DECIMALS) for j, k in enumerate(cidx)]
    cit_bytes = make_geojson(cfeats)
    cpos = {k: j for j, k in enumerate(cidx)}
    cpairs = {(cpos[a], cpos[b]): v for (a, b), v in pairs_all.items() if a in cpos and b in cpos}
    cname = [name[k] for k in cidx]
    cit = file_metrics("mo_cities.geojson", cit_bytes, cids, cname, corig, cit_geoms, cin, cpairs)
    cit["sha256"] = hashlib.sha256(cit_bytes).hexdigest()
    cit["moscow_polygons"] = int(sum(1 for k in cidx if rcode[k] == C.GEO_CITY_REGION_CODES["moscow"]))
    cit["spb_polygons"] = int(sum(1 for k in cidx if rcode[k] == C.GEO_CITY_REGION_CODES["spb"]))

    controls = [
        {"name": "действующих полигонов", "expect": C.GEO_EXPECT_ACTIVE, "got": len(ids), "ok": len(ids) == C.GEO_EXPECT_ACTIVE},
        {"name": "МО выборки среди действующих", "expect": C.GEO_EXPECT_SAMPLE, "got": int(in_sample.sum()), "ok": int(in_sample.sum()) == C.GEO_EXPECT_SAMPLE},
        {"name": "пар с перекрытием в исходном покрытии", "expect": C.GEO_EXPECT_OVERLAP_PAIRS, "got": len(ov_orig), "ok": len(ov_orig) == C.GEO_EXPECT_OVERLAP_PAIRS},
        {"name": "mo_national: МО выборки с |ΔS| выше порога", "expect": f"не более {C.GEO_MAX_NATIONAL_AREA_CHANGE_MO}",
         "got": nat["area_change_gt_threshold_sample"], "ok": nat["area_change_gt_threshold_sample"] <= C.GEO_MAX_NATIONAL_AREA_CHANGE_MO},
        {"name": "mo_cities: МО с |ΔS| выше порога", "expect": 0, "got": cit["area_change_gt_threshold_all"], "ok": cit["area_change_gt_threshold_all"] == 0},
        {"name": "mo_cities: пар с перекрытием после упрощения", "expect": 0, "got": cit["overlap_pairs_simplified"], "ok": cit["overlap_pairs_simplified"] == 0},
        {"name": f"mo_national: потеряно частей площадью не менее {C.GEO_MAX_LOST_PART_AREA} град²", "expect": 0, "got": nat["lost_parts_large"], "ok": nat["lost_parts_large"] == 0},
        {"name": f"mo_cities: потеряно частей площадью не менее {C.GEO_MAX_LOST_PART_AREA} град²", "expect": 0, "got": cit["lost_parts_large"], "ok": cit["lost_parts_large"] == 0},
        {"name": "размер mo_national.geojson, байт", "expect": f"не более {C.GEO_MAX_NATIONAL_BYTES}", "got": nat["bytes"], "ok": nat["bytes"] <= C.GEO_MAX_NATIONAL_BYTES},
    ]
    R = {
        "source_url": SOURCE_URL, "snapshot_month": C.GEO_SNAPSHOT_MONTH, "active_year_to": C.GEO_ACTIVE_YEAR_TO,
        "data_download_date": C.GEO_DOWNLOAD_DATE, "overlap_min_area_deg2": C.GEO_OVERLAP_MIN_AREA,
        "national_tolerance_deg": C.GEO_NATIONAL_TOL, "national_decimals": C.GEO_NATIONAL_DECIMALS,
        "cities_tolerance_deg": C.GEO_CITIES_TOL, "cities_decimals": C.GEO_CITIES_DECIMALS,
        "area_change_threshold": C.GEO_AREA_CHANGE_THRESHOLD, "max_lost_part_area_deg2": C.GEO_MAX_LOST_PART_AREA, "gap_threshold_deg": C.GEO_GAP_THRESHOLD,
        "gap_sample_per_pair": C.GEO_GAP_SAMPLE, "gap_seed": C.GEO_GAP_SEED,
        "inputs": {"gpkg_sha256": sha256_file(gpkg), "xlsx_sha256": sha256_file(xlsx), "archive_sha256_expected": C.GEO_ARCHIVE_SHA256,
                   "labels_sha256": sha256_file(LABELS_PATH), "territories_sha256": sha256_file(TERRITORIES_PATH)},
        "versions": {"shapely": shapely.__version__, "geos": ".".join(map(str, shapely.geos_version)), "numpy": np.__version__},
        "original": original, "national": nat, "cities": cit, "controls": controls,
    }
    files = {"mo_national.geojson": nat_bytes, "mo_cities.geojson": cit_bytes,
             "geometry_report.json": (json.dumps(R, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8"),
             "geometry_report.md": md_report(R).encode("utf-8"), "ATTRIBUTION.md": md_attribution(R).encode("utf-8")}
    print("| контроль | ожидание | получено | результат |")
    for c in controls:
        print(f"| {c['name']} | {c['expect']} | {c['got']} | {'пройден' if c['ok'] else 'НЕ ПРОЙДЕН'} |")
    bad = [c for c in controls if not c["ok"]]
    if bad:
        cand = (Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())) / "step24b_candidate"
        cand.mkdir(parents=True, exist_ok=True)
        for fn, data in files.items():
            (cand / fn).write_bytes(data)
        print(f"STOP: контроли не пройдены; data/geo/ не изменён, кандидаты записаны в {cand}")
        raise SystemExit(1)
    for fn, data in files.items():       # запись только после прохождения всех контролей
        (out_dir / fn).write_bytes(data)
    print("Готово.")


if __name__ == "__main__":
    main()
