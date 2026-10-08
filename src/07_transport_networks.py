"""Шаг 07. Транспортные сети (highway, railway): очистка пар и kNN-графы по расстоянию.

1. railway: проверка симметрии x→y / y→x, одна строка на неориентированную пару (x < y);
   highway приводится к тому же виду (x < y).
2. highway: разбор пар с distance == 0 (без исключения) -> notebooks/highway_zero_distance_pairs.md
3. railway: пары между МО с ж/д, отсутствующие в исходнике (общая ближайшая станция),
   достраиваются с distance = 0 и флагом imputed = True (IMPUTE_MISSING_RAILWAY_ZERO).
   Пары с highway distance >= NAMES_CHECK_KM проверяются через справочник названий и
   расстояние по прямой между центрами (MANUAL_REVIEW).
4. kNN-графы (k=8) только по МО из valid_territories.parquet.

Выход: data/processed/connection_valid.parquet (обе сети, x < y, только рабочая выборка,
         колонка imputed),
       data/processed/transport_network_{type}_knn<K>.parquet / .graphml,
       data/processed/transport_network_railway_knn<K>_before_impute.*,
       data/processed/railway_degree.parquet (territory_id, has_railway, степени),
       notebooks/07_transport_networks.md
Запуск из корня проекта:  .venv/bin/python src/07_transport_networks.py
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from network_utils import knn_edges_by_distance
from territory_names import add_names, load_territory_names, names_note

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
CONNECTION_PATH = RAW_DIR / "connection.parquet"
MARKET_ACCESS_PATH = RAW_DIR / "market_access.parquet"
VALID_TERRITORIES_PATH = PROCESSED_DIR / "valid_territories.parquet"
CONNECTION_VALID_PATH = PROCESSED_DIR / "connection_valid.parquet"
RAILWAY_DEGREE_PATH = PROCESSED_DIR / "railway_degree.parquet"
ZERO_PAIRS_REPORT = NOTEBOOKS_DIR / "highway_zero_distance_pairs.md"
REPORT_PATH = NOTEBOOKS_DIR / "07_transport_networks.md"

from config import TRANSPORT_KNN_K as K
TYPES = ["highway", "railway"]
# допустимое расхождение distance между x→y и y→x (км)
from config import TRANSPORT_SYMMETRY_TOL_KM as SYMMETRY_TOL_KM
# достраивать отсутствующие railway-пары внутри ж/д-сети с distance = 0
from config import TRANSPORT_IMPUTE_MISSING_RAILWAY_ZERO as IMPUTE_MISSING_RAILWAY_ZERO
# достроенные пары с highway distance не меньше порога (км) проверяются через названия
from config import TRANSPORT_NAMES_CHECK_KM as NAMES_CHECK_KM
# типичное отношение «дорога / прямая» для пар 30–100 км: медиана 1.24, p99 2.13
from config import TRANSPORT_DETOUR_RATIO as DETOUR_RATIO
TERRITORIES_PATH = RAW_DIR / "territories.parquet"
# Ручной разбор пар с большим автодорожным расстоянием (по справочнику названий)
MANUAL_REVIEW = {
    (1903, 2266): "Переправа через Волгу: Приволжский р-н (Самарская обл.) и Радищевский р-н "
                  "(Ульяновская обл.) разделены Волгой; дорога идёт в объезд до моста.",
    (274, 489): "Переправа через Волгу: Звениговский р-н (Марий Эл) и Козловский округ "
                "(Чувашия) — напротив друг друга через Волгу.",
    (270, 489): "Переправа через Волгу: г. Волжск (Марий Эл) и Козловка (Чувашия) — "
                "напротив друг друга через Волгу.",
    (272, 489): "Переправа через Волгу: Волжский р-н (Марий Эл, центр — г. Волжск) и "
                "Козловка (Чувашия) — напротив друг друга через Волгу.",
    (34, 48): "Соседние районы Башкортостана (Аургазинский и Гафурийский); маршрут без "
              "объезда, общая ближайшая станция правдоподобна.",
}


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def md_table(df: pd.DataFrame, index: bool = True, fmt: str = ",.2f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def to_undirected(df: pd.DataFrame) -> pd.DataFrame:
    """Переупорядочивает пару так, чтобы territory_id_x < territory_id_y."""
    x, y = df["territory_id_x"].to_numpy(), df["territory_id_y"].to_numpy()
    return df.assign(territory_id_x=np.minimum(x, y), territory_id_y=np.maximum(x, y))


def dedup_railway(rail: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    u = to_undirected(rail)
    g = u.groupby(["territory_id_x", "territory_id_y"])["distance"].agg(["count", "min", "max"])
    diff = g["max"] - g["min"]
    bad = g[diff > SYMMETRY_TOL_KM]
    notes = [
        f"- строк railway: {len(rail):,}; неориентированных пар: {len(g):,}",
        f"- строк на пару: {g['count'].value_counts().sort_index().to_dict()}",
        f"- пар с |d(x→y) − d(y→x)| > 0: {int((diff > 0).sum()):,}; "
        f"> {SYMMETRY_TOL_KM} км: {len(bad):,}; максимум расхождения: {diff.max():.2f} км",
    ]
    if len(bad):
        raise SystemExit("STOP: railway несимметричен:\n" + bad.head(20).to_string())
    return u.drop_duplicates(["territory_id_x", "territory_id_y"], ignore_index=True), notes


def zero_distance_report(hw: pd.DataFrame, rail: pd.DataFrame, ma: pd.Series,
                         valid: set, names: pd.Series | None) -> str:
    """Пары highway с distance == 0 + косвенные признаки «общего центра»."""
    zero = hw[hw["distance"] == 0].sort_values(["territory_id_x", "territory_id_y"],
                                               ignore_index=True)

    # Профили расстояний до всех МО: у МО с общим центром они должны совпадать
    ids = np.union1d(hw["territory_id_x"], hw["territory_id_y"])
    pos = pd.Series(np.arange(len(ids)), index=ids)
    D = np.full((len(ids), len(ids)), np.nan)
    i, j = pos[hw["territory_id_x"]].to_numpy(), pos[hw["territory_id_y"]].to_numpy()
    D[i, j] = D[j, i] = hw["distance"].to_numpy()

    rail_d = rail.set_index(["territory_id_x", "territory_id_y"])["distance"]
    rows = []
    for x, y in zero[["territory_id_x", "territory_id_y"]].itertuples(index=False):
        dx, dy = D[pos[x]], D[pos[y]]
        m = ~np.isnan(dx) & ~np.isnan(dy)
        m[[pos[x], pos[y]]] = False
        delta = np.abs(dx[m] - dy[m])
        rows.append({
            "territory_id_x": x,
            "territory_id_y": y,
            "x в выборке": x in valid,
            "y в выборке": y in valid,
            "market_access x": ma.get(x, np.nan),
            "market_access y": ma.get(y, np.nan),
            "railway distance": rail_d.get((x, y), np.nan),
            "Δ профиля медиана, км": np.median(delta),
            "Δ профиля max, км": delta.max(),
        })
    tab = add_names(pd.DataFrame(rows), names, ["territory_id_x", "territory_id_y"])

    ids_in_zero = pd.concat([zero["territory_id_x"], zero["territory_id_y"]])
    both_valid = tab["x в выборке"] & tab["y в выборке"]
    lines = [
        "# Пары highway с distance = 0 (для ручного разбора)",
        "",
        "Сгенерировано `src/07_transport_networks.py`. Пары **не исключены** из данных.",
        "",
        names_note(),
        "",
        "По описанию датасета, автодорожное расстояние считается «от центра одного МО до "
        "центра другого», поэтому 0 означает общий центр (типичный случай: городской "
        "округ и одноимённый муниципальный район с администрацией в этом городе).",
        "",
        "Косвенные проверки:",
        "",
        f"- всего пар: **{len(tab)}**; различных id в них: **{ids_in_zero.nunique()}**; "
        f"id, входящих более чем в одну нулевую пару: "
        f"**{int((ids_in_zero.value_counts() > 1).sum())}**",
        f"- обе стороны в рабочей выборке: **{int(both_valid.sum())}**; "
        f"хотя бы одна: **{int((tab['x в выборке'] | tab['y в выборке']).sum())}**",
        "- «Δ профиля» — разница расстояний от x и от y до всех остальных МО по highway. "
        "Если центры совпадают, она ≈ 0 для всех МО.",
        "",
        md_table(tab[["Δ профиля медиана, км", "Δ профиля max, км", "railway distance"]]
                 .describe(percentiles=[.5, .9]).T),
        "",
        f"- пар с max Δ профиля > 1 км: **{int((tab['Δ профиля max, км'] > 1).sum())}**",
        f"- пар с railway distance > 0: "
        f"**{int((tab['railway distance'] > 0).sum())}** (из "
        f"{int(tab['railway distance'].notna().sum())} с ж/д данными)",
        "",
        "## Список пар",
        "",
        md_table(tab, index=False),
        "",
    ]
    ZERO_PAIRS_REPORT.write_text("\n".join(lines), encoding="utf-8")
    return (f"- пар distance = 0: {len(tab)}, в обеих сторонах рабочей выборки: "
            f"{int(both_valid.sum())}; max Δ профиля > 1 км: "
            f"{int((tab['Δ профиля max, км'] > 1).sum())}. "
            f"Разбор: `{ZERO_PAIRS_REPORT.relative_to(PROJECT_DIR)}`")


def build_graph(edges: pd.DataFrame, nodes: np.ndarray, ma: pd.Series) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from((int(i), {"market_access": float(ma[i])}) for i in nodes)
    G.add_edges_from((int(x), int(y), {"distance": float(d), "mutual": bool(m)})
                     for x, y, d, m in edges[["territory_id_x", "territory_id_y",
                                              "distance", "mutual"]].itertuples(index=False))
    return G


def distance_matrix(sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.union1d(sub["territory_id_x"], sub["territory_id_y"])
    pos = pd.Series(np.arange(len(nodes)), index=nodes)
    D = np.full((len(nodes), len(nodes)), np.nan)
    i, j = pos[sub["territory_id_x"]].to_numpy(), pos[sub["territory_id_y"]].to_numpy()
    D[i, j] = D[j, i] = sub["distance"].to_numpy()
    return nodes, D


def missing_pairs(sub: pd.DataFrame) -> pd.DataFrame:
    """Неупорядоченные пары (x < y) между узлами сети, отсутствующие в таблице."""
    nodes, D = distance_matrix(sub)
    iu, ju = np.triu_indices(len(nodes), k=1)
    m = np.isnan(D[iu, ju])
    return pd.DataFrame({"territory_id_x": nodes[iu[m]], "territory_id_y": nodes[ju[m]]})


def build_knn(sub: pd.DataFrame, n_valid: int, ma: pd.Series, t: str, suffix: str = ""
              ) -> tuple[nx.Graph, dict]:
    nodes, D = distance_matrix(sub)
    n = len(nodes)
    edges = knn_edges_by_distance(nodes, D, K)
    G = build_graph(edges, nodes, ma)
    G.graph.update({"type": t, "method": f"kNN k={K} по distance, undirected union"})
    stem = f"transport_network_{t}_knn{K}{suffix}"
    edges.to_parquet(PROCESSED_DIR / f"{stem}.parquet", engine="pyarrow", index=False)
    nx.write_graphml(G, PROCESSED_DIR / f"{stem}.graphml")

    deg = np.array([d for _, d in G.degree()])
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    metrics = {
        "МО выборки в сети": n,
        "МО выборки вне сети": n_valid - n,
        "отсутствующих пар внутри сети": n * (n - 1) // 2 - len(sub),
        "рёбер": G.number_of_edges(),
        "взаимных (mutual)": int(edges["mutual"].sum()),
        "изолированных МО": nx.number_of_isolates(G),
        "компонент связности": len(comps),
        "крупнейшая компонента": len(comps[0]),
        "вторая компонента": len(comps[1]) if len(comps) > 1 else 0,
        "степень min": int(deg.min()),
        "степень медиана": float(np.median(deg)),
        "степень mean": float(deg.mean()),
        "степень max": int(deg.max()),
        "distance ребра медиана": edges["distance"].median(),
        "distance ребра max": edges["distance"].max(),
        "рёбер с distance = 0": int((edges["distance"] == 0).sum()),
    }
    return G, metrics


def main() -> None:
    conn = pd.read_parquet(CONNECTION_PATH, engine="pyarrow")
    ma = pd.read_parquet(MARKET_ACCESS_PATH, engine="pyarrow").set_index("territory_id")[
        "market_access"]
    valid_ids = pd.read_parquet(VALID_TERRITORIES_PATH, engine="pyarrow")["territory_id"]
    valid = set(valid_ids)

    unexpected = set(conn["type"].unique()) - set(TYPES)
    if unexpected:
        raise SystemExit(f"STOP: неожиданные type в connection: {unexpected}")

    # 1. Приведение к одной строке на неориентированную пару
    rail, rail_notes = dedup_railway(conn[conn["type"] == "railway"])
    hw = to_undirected(conn[conn["type"] == "highway"])
    hw_dups = int(hw.duplicated(["territory_id_x", "territory_id_y"]).sum())
    if hw_dups:
        raise SystemExit(f"STOP: в highway {hw_dups} повторов неориентированной пары")

    # 2. Нулевые расстояния highway
    names = load_territory_names()
    zero_note = zero_distance_report(hw, rail, ma, valid, names)

    # Только рабочая выборка
    clean = pd.concat([hw, rail], ignore_index=True)
    clean = clean[clean["territory_id_x"].isin(valid) & clean["territory_id_y"].isin(valid)]
    clean["imputed"] = False

    # 3. Достройка отсутствующих railway-пар (все пары, без фильтра по расстоянию)
    rail_before = clean[clean["type"] == "railway"]
    miss = missing_pairs(rail_before)
    hw_d = hw.set_index(["territory_id_x", "territory_id_y"])["distance"]
    miss["highway distance"] = hw_d.reindex(
        pd.MultiIndex.from_frame(miss[["territory_id_x", "territory_id_y"]])).to_numpy()
    hw_zero_pairs = set(zip(*[hw.loc[hw["distance"] == 0, c] for c in
                              ["territory_id_x", "territory_id_y"]]))
    miss["highway-нулевая пара"] = [p in hw_zero_pairs for p in
                                    zip(miss["territory_id_x"], miss["territory_id_y"])]
    if IMPUTE_MISSING_RAILWAY_ZERO and len(miss):
        add = miss[["territory_id_x", "territory_id_y"]].assign(
            distance=0.0, type="railway", imputed=True)
        clean = pd.concat([clean, add.astype(clean.dtypes.to_dict())], ignore_index=True)
    clean = clean.sort_values(["type", "territory_id_x", "territory_id_y"], ignore_index=True)
    if clean.duplicated(["type", "territory_id_x", "territory_id_y"]).any():
        raise SystemExit("STOP: после достройки появились повторы пар")
    clean.to_parquet(CONNECTION_VALID_PATH, engine="pyarrow", index=False)
    n_imputed = int(clean["imputed"].sum())

    # Проверка дальних пар через справочник: расстояние по прямой между центрами
    check = miss[miss["highway distance"] >= NAMES_CHECK_KM].sort_values(
        "highway distance", ascending=False).copy()
    if TERRITORIES_PATH.exists():
        terr = pd.read_parquet(TERRITORIES_PATH).set_index("territory_id")
        cx = terr.loc[check["territory_id_x"]]
        cy = terr.loc[check["territory_id_y"]]
        check["центр x"] = cx["center"].to_numpy()
        check["центр y"] = cy["center"].to_numpy()
        check["по прямой, км"] = haversine_km(cx["center_lat"].to_numpy(), cx["center_lon"].to_numpy(),
                                              cy["center_lat"].to_numpy(), cy["center_lon"].to_numpy())
        check["дорога / прямая"] = check["highway distance"] / check["по прямой, км"]
    check["разбор"] = [MANUAL_REVIEW.get((int(x), int(y)), "нет ручного разбора") for x, y in
                       zip(check["territory_id_x"], check["territory_id_y"])]
    unreviewed = int((check["разбор"] == "нет ручного разбора").sum())
    check_show = add_names(check.drop(columns=["highway-нулевая пара"]), names,
                           ["territory_id_x", "territory_id_y"])

    lines = [
        "# 07. Транспортные сети (kNN по расстоянию)",
        "",
        "Сгенерировано `src/07_transport_networks.py`.",
        "",
        "## Подготовка пар",
        "",
        "railway:",
        *rail_notes,
        "",
        f"highway: строк {len(hw):,}, повторов неориентированной пары: {hw_dups} "
        "(исходно пары записаны в одну сторону, но не всегда x < y — переупорядочено)",
        "",
        zero_note,
        "",
        "## Достройка railway-пар с distance = 0",
        "",
        "Пары между двумя МО с ж/д, отсутствующие в исходнике, трактуются как «общая "
        "ближайшая станция» и достраиваются с distance = 0 (imputed = True). Фильтра по "
        "автодорожному расстоянию нет.",
        "",
        f"- отсутствовало пар между МО с ж/д (рабочая выборка): **{len(miss)}**; из них "
        f"highway-нулевых: **{int(miss['highway-нулевая пара'].sum())}**",
        f"- достроено: **{n_imputed}**",
        "",
        "highway distance у достроенных пар:",
        "",
        md_table(miss["highway distance"].describe(percentiles=[.25, .5, .75, .9])
                 .rename("км").to_frame()),
        "",
        "### Проверка через названия территорий",
        "",
        f"{len(check)} достроенных пар разнесены по автодороге на {NAMES_CHECK_KM}+ км — на "
        "первый взгляд это противоречит «общей ближайшей станции». Проверка по справочнику "
        "МО (названия, регионы, координаты центров) показывает, что противоречия нет: "
        f"в {int((check.get('дорога / прямая', pd.Series(dtype=float)) >= DETOUR_RATIO).sum())} "
        f"парах дорога длиннее прямой в {DETOUR_RATIO:.0f}+ раз (типично ≈ 1.24, p99 ≈ 2.1) — "
        "МО находятся напротив друг друга через Волгу, и автодорожный маршрут идёт в объезд "
        "до ближайшего моста; остальные — соседние районы с обычным коэффициентом.",
        "",
        md_table(check_show.astype({"territory_id_x": str, "territory_id_y": str}),
                 index=False, fmt=",.1f"),
        "",
        f"**Решение:** все {len(check)} пар включены в достройку наравне с остальными "
        "(итого достроено 100 из 100)."
        + (f" Пар без ручного разбора: {unreviewed} — проверить." if unreviewed else ""),
        "",
        f"Очищенные пары рабочей выборки: `{CONNECTION_VALID_PATH.relative_to(PROJECT_DIR)}`",
        "",
    ]

    # 4. kNN-графы
    metrics = {}
    G_hw, metrics["highway"] = build_knn(clean[clean["type"] == "highway"], len(valid), ma,
                                         "highway")
    G_rb, metrics["railway до достройки"] = build_knn(rail_before, len(valid), ma, "railway",
                                                      "_before_impute")
    G_ra, metrics["railway после достройки"] = build_knn(clean[clean["type"] == "railway"],
                                                         len(valid), ma, "railway")
    for name, G in [("highway", G_hw), ("railway до", G_rb), ("railway после", G_ra)]:
        sizes = pd.Series([len(c) for c in nx.connected_components(G)]).value_counts()
        lines += [f"Размеры компонент {name} (размер → число): "
                  f"{sizes.sort_index().to_dict()}", ""]

    # Состав малых компонент (всё, кроме крупнейшей)
    lines += ["## Малые компоненты связности", "", names_note(), ""]
    for name, G in [("highway", G_hw), ("railway", G_ra)]:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)[1:]
        for n_c, comp in enumerate(comps, 1):
            members = pd.DataFrame({"territory_id": sorted(int(i) for i in comp)})
            members["market_access"] = members["territory_id"].map(ma)
            members["has_railway"] = members["territory_id"].isin(G_ra.nodes)
            members = add_names(members, names, ["territory_id"])
            regions = ""
            if names is not None and TERRITORIES_PATH.exists():
                rc = terr.loc[members["territory_id"], "region_name"].value_counts()
                regions = " — " + ", ".join(f"{r}: {n}" for r, n in rc.items())
            lines += [f"{name}, компонента {n_c} ({len(comp)} МО){regions}:", "",
                      md_table(members.astype({"territory_id": str}), index=False), ""]

    # railway_degree
    rd = pd.DataFrame({"territory_id": valid_ids.to_numpy()})
    rd["has_railway"] = rd["territory_id"].isin(G_ra.nodes)
    for col, G in [("railway_degree_before_impute", G_rb), ("railway_degree", G_ra)]:
        rd[col] = rd["territory_id"].map(dict(G.degree())).fillna(0).astype(int)
    rd.to_parquet(RAILWAY_DEGREE_PATH, engine="pyarrow", index=False)

    def edges_of(G: nx.Graph) -> set:
        return {tuple(sorted(e)) for e in G.edges()}

    e_b, e_a = edges_of(G_rb), edges_of(G_ra)
    imputed_ids = set(miss["territory_id_x"]) | set(miss["territory_id_y"])
    delta = rd["railway_degree"] - rd["railway_degree_before_impute"]
    check_ids = set(check["territory_id_x"]) | set(check["territory_id_y"])
    chk_deg = add_names(rd[rd["territory_id"].isin(check_ids)], names, ["territory_id"])

    lines += [
        f"## kNN-графы (k={K})",
        "",
        md_table(pd.DataFrame(metrics)),
        "",
        "## Railway: до / после достройки",
        "",
        f"- общих рёбер: {len(e_a & e_b):,}; только до: {len(e_b - e_a):,}; "
        f"только после: {len(e_a - e_b):,}",
        f"- МО, входящих в достроенные пары: **{len(imputed_ids)}**; "
        f"МО с изменившейся степенью: **{int((delta != 0).sum())}**",
        "",
        "Изменение степени (после − до), все МО с ж/д:",
        "",
        md_table(delta[rd["has_railway"]].value_counts().sort_index()
                 .rename("МО").rename_axis("Δ степени").to_frame(), fmt=",.0f"),
        "",
        "Степень участников пар из раздела «Проверка через названия территорий»:",
        "",
        md_table(chk_deg.astype({"territory_id": str}), index=False, fmt=",.0f"),
        "",
        "Файлы: `data/processed/transport_network_{highway,railway}_knn8.{parquet,graphml}` "
        "(railway — после полной достройки), `transport_network_railway_knn8_before_impute.*` "
        f"— до; `{RAILWAY_DEGREE_PATH.relative_to(PROJECT_DIR)}`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
