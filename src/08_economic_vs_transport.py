"""Шаг 08. Экономическая близость vs транспортная близость.

1. Корреляция economic similarity ↔ highway distance (на общих kNN-рёбрах + справочно по всем парам).
2. Экономически похожие пары (топ-10% рёбер экономической сети), не являющиеся
   транспортными соседями ни по highway, ни по railway.
3. Economic centrality (средняя similarity к соседям) vs market_access и railway_degree.
4. Пункты 1 и 3 для двух месяцев (2024-12 и 2023-01).

Вход:  data/processed/economic_networks/{YYYY-MM}_std_knn8.parquet,
       data/processed/transport_network_{highway,railway}_knn8.parquet,
       data/processed/connection_valid.parquet, category_shares.parquet, railway_degree.parquet,
       data/raw/market_access.parquet
Выход: notebooks/08_economic_vs_transport.md,
       data/processed/econ_similar_not_transport_neighbors_{YYYY-MM}.parquet
Запуск из корня проекта:  .venv/bin/python src/08_economic_vs_transport.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from network_utils import SHARE_COLUMNS, std_cosine_similarity
from territory_names import add_names, load_territory_names, names_note

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
ECON_DIR = PROCESSED_DIR / "economic_networks"
HIGHWAY_KNN_PATH = PROCESSED_DIR / "transport_network_highway_knn8.parquet"
RAILWAY_KNN_PATH = PROCESSED_DIR / "transport_network_railway_knn8.parquet"
CONNECTION_VALID_PATH = PROCESSED_DIR / "connection_valid.parquet"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
RAILWAY_DEGREE_PATH = PROCESSED_DIR / "railway_degree.parquet"
MARKET_ACCESS_PATH = RAW_DIR / "market_access.parquet"
TERRITORIES_PATH = RAW_DIR / "territories.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "08_economic_vs_transport.md"

from config import ECON_TRANSPORT_MONTHS as MONTHS
from config import ECON_KNN_K as K
from config import ECON_TRANSPORT_TOP_SIMILARITY_SHARE as TOP_SIMILARITY_SHARE  # топ-10% рёбер экономической сети по similarity
N_PAIRS_IN_REPORT = 60        # сколько пар из п.2 показать в отчёте (полный список — в parquet)
# п.2: «похожие, но далеко» — дальше этого порога сходство перестаёт зависеть от расстояния
from config import ECON_TRANSPORT_FAR_MIN_KM as FAR_MIN_KM
# кластер пар на ~700 км (гипотеза: две агломерации); границы для выделения в отчёте
from config import ECON_TRANSPORT_CLUSTER_KM as CLUSTER_KM
from config import ECON_TRANSPORT_CLUSTER_MA_LOW as CLUSTER_MA_LOW
from config import ECON_TRANSPORT_CLUSTER_MA_HIGH as CLUSTER_MA_HIGH
KEY = ["territory_id_x", "territory_id_y"]


def md_table(df: pd.DataFrame, index: bool = True, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def corr_row(a: pd.Series, b: pd.Series) -> dict:
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    return {"n": int(m.sum()), "Pearson r": pr, "p (Pearson)": pp,
            "Spearman ρ": sr, "p (Spearman)": sp}


def pair_set(df: pd.DataFrame) -> set:
    return set(zip(df["territory_id_x"].astype(int), df["territory_id_y"].astype(int)))


def dense(ids: np.ndarray, df: pd.DataFrame, col: str) -> np.ndarray:
    pos = pd.Series(np.arange(len(ids)), index=ids)
    M = np.full((len(ids), len(ids)), np.nan)
    i, j = pos[df["territory_id_x"]].to_numpy(), pos[df["territory_id_y"]].to_numpy()
    M[i, j] = M[j, i] = df[col].to_numpy()
    return M


def analyse_month(month: str, hw_knn: pd.DataFrame, rw_knn: pd.DataFrame,
                  hw_dist: pd.Series, D_hw: np.ndarray, ids: np.ndarray,
                  shares: pd.DataFrame, nodes: pd.DataFrame, names: pd.Series | None
                  ) -> tuple[list[str], dict, dict]:
    econ = pd.read_parquet(ECON_DIR / f"{month}_std_knn{K}.parquet", engine="pyarrow")
    econ["highway distance"] = hw_dist.reindex(pd.MultiIndex.from_frame(econ[KEY])).to_numpy()

    # --- 1. similarity ↔ highway distance ---
    both = econ.merge(hw_knn[KEY + ["distance"]], on=KEY, how="inner")
    c_both = corr_row(both["similarity"], both["distance"])

    c_econ = corr_row(econ["similarity"], econ["highway distance"])

    # по всем парам: полная матрица similarity за месяц
    m = shares[shares["date"] == pd.Timestamp(month)].set_index("territory_id").loc[ids]
    S, _ = std_cosine_similarity(m[SHARE_COLUMNS].to_numpy())
    iu, ju = np.triu_indices(len(ids), k=1)
    s_all, d_all = pd.Series(S[iu, ju]), pd.Series(D_hw[iu, ju])
    c_all = corr_row(s_all, d_all)
    c_all_log = corr_row(s_all, np.log1p(d_all))

    # сверка: similarity в edge list = пересчитанной
    pos = pd.Series(np.arange(len(ids)), index=ids)
    chk = S[pos[econ["territory_id_x"]].to_numpy(), pos[econ["territory_id_y"]].to_numpy()]
    if not np.allclose(chk, econ["similarity"].to_numpy()):
        raise SystemExit(f"STOP: {month}: similarity в edge list не совпадает с пересчётом")

    # similarity по дистанционным поясам (все пары)
    bands = pd.cut(d_all, [-0.1, 50, 100, 200, 500, 1000, 2000, np.inf])
    by_band = s_all.groupby(bands, observed=True).agg(["count", "mean", "median"])
    by_band.index = by_band.index.astype(str)
    by_band.index.name = "highway distance, км"

    e_econ, e_hw, e_rw = pair_set(econ), pair_set(hw_knn), pair_set(rw_knn)
    n_pairs = len(ids) * (len(ids) - 1) // 2
    overlap = pd.DataFrame([
        ("рёбер экономической сети", len(e_econ)),
        ("из них — рёбра highway kNN", len(e_econ & e_hw)),
        ("из них — рёбра railway kNN", len(e_econ & e_rw)),
        ("из них — рёбра highway или railway kNN", len(e_econ & (e_hw | e_rw))),
        ("ожидалось бы при случайном совпадении (highway)",
         round(len(e_econ) * len(e_hw) / n_pairs, 1)),
    ], columns=["показатель", "значение"])

    corr = pd.DataFrame({
        "общие kNN-рёбра econ ∩ highway (similarity ↔ distance)": c_both,
        "все рёбра econ (similarity ↔ highway distance)": c_econ,
        "все пары МО (similarity ↔ highway distance)": c_all,
        "все пары МО (similarity ↔ log(1+distance))": c_all_log,
    }).T

    # --- 2. похожие, но не соседи ---
    thr = econ["similarity"].quantile(1 - TOP_SIMILARITY_SHARE)
    top = econ[econ["similarity"] >= thr]
    far = top[[p not in e_hw and p not in e_rw for p in pair_set_iter(top)]]
    far = far.merge(nodes.rename(columns=lambda c: f"{c}_x" if c != "territory_id" else c)
                    .rename(columns={"territory_id": "territory_id_x"}), on="territory_id_x")
    far = far.merge(nodes.rename(columns=lambda c: f"{c}_y" if c != "territory_id" else c)
                    .rename(columns={"territory_id": "territory_id_y"}), on="territory_id_y")
    far = far.sort_values(["similarity", "highway distance"], ascending=[False, False],
                          ignore_index=True)
    far = add_names(far, names, KEY)
    far["дальше порога"] = far["highway distance"] >= FAR_MIN_KM
    far_path = PROCESSED_DIR / f"econ_similar_not_transport_neighbors_{month.replace('-', '_')}.parquet"
    far.to_parquet(far_path, engine="pyarrow", index=False)
    far200 = far[far["дальше порога"]]

    def in_rng(v: pd.Series, r: tuple) -> pd.Series:
        return v.between(*r)

    ma_lo = far200[["market_access_x", "market_access_y"]].min(axis=1)
    ma_hi = far200[["market_access_x", "market_access_y"]].max(axis=1)
    cluster = far200[in_rng(far200["highway distance"], CLUSTER_KM)
                     & in_rng(ma_lo, CLUSTER_MA_LOW) & in_rng(ma_hi, CLUSTER_MA_HIGH)]
    # сторона кластера с низким / высоким market_access
    lo_side = np.where(cluster["market_access_x"] <= cluster["market_access_y"],
                       cluster["territory_id_x"], cluster["territory_id_y"])
    hi_side = np.where(cluster["market_access_x"] <= cluster["market_access_y"],
                       cluster["territory_id_y"], cluster["territory_id_x"])

    def region_line(lo: np.ndarray, hi: np.ndarray) -> str:
        if not TERRITORIES_PATH.exists():
            return ("- Гипотеза (без справочника не проверена): внутригородские МО "
                    "Санкт-Петербурга и Москвы (≈ 700 км по трассе).")
        reg = pd.read_parquet(TERRITORIES_PATH).set_index("territory_id")["region_name"]

        def comp(ids_: np.ndarray) -> str:
            vc = reg.loc[sorted(set(ids_))].value_counts()
            return ", ".join(f"{r} — {n}" for r, n in vc.head(4).items()) + (
                f", прочие — {vc.iloc[4:].sum()}" if len(vc) > 4 else "")
        return ("- Состав по регионам (справочник МО): сторона низкого market_access: "
                f"{comp(lo)}; сторона высокого: {comp(hi)}. Ядро кластера — внутригородские "
                "территории Санкт-Петербурга и Москвы (≈ 700 км по трассе): структура "
                "расходов двух столиц почти идентична, при том что транспортными соседями "
                "они не являются.")

    def side_table(side_ids: np.ndarray) -> pd.DataFrame:
        t = (pd.Series(side_ids, name="territory_id").value_counts()
             .rename("пар в кластере").rename_axis("territory_id").reset_index())
        t = t.merge(nodes[["territory_id", "market_access", "has_railway"]], on="territory_id")
        return add_names(t.sort_values(["пар в кластере", "territory_id"],
                                       ascending=[False, True]), names, ["territory_id"])

    # --- 3. economic centrality ---
    long = pd.concat([econ[["territory_id_x", "similarity"]].rename(
        columns={"territory_id_x": "territory_id"}),
        econ[["territory_id_y", "similarity"]].rename(columns={"territory_id_y": "territory_id"})])
    cent = long.groupby("territory_id")["similarity"].mean().rename("econ_centrality")
    nd = nodes.set_index("territory_id").join(cent)
    rail = nd[nd["has_railway"]]
    cent_corr = pd.DataFrame({
        "centrality ↔ market_access (все МО)": corr_row(nd["econ_centrality"], nd["market_access"]),
        "centrality ↔ railway_degree (все МО, 0 = нет ж/д)":
            corr_row(nd["econ_centrality"], nd["railway_degree"].astype(float)),
        "centrality ↔ railway_degree (только МО с ж/д)":
            corr_row(rail["econ_centrality"], rail["railway_degree"].astype(float)),
        "centrality ↔ has_railway (точечно-бисериальная)":
            corr_row(nd["econ_centrality"], nd["has_railway"].astype(float)),
        "market_access ↔ railway_degree (контроль)":
            corr_row(nd["market_access"], nd["railway_degree"].astype(float)),
    }).T
    by_rail = nd.groupby("has_railway")["econ_centrality"].agg(["count", "mean", "median", "std"])
    mw = stats.mannwhitneyu(nd.loc[nd["has_railway"], "econ_centrality"],
                            nd.loc[~nd["has_railway"], "econ_centrality"])
    ma_q = nd.groupby(pd.qcut(nd["market_access"], 5), observed=True)["econ_centrality"].agg(
        ["count", "mean", "median"])
    ma_q.index = ma_q.index.astype(str)
    ma_q.index.name = "квинтиль market_access"

    lines = [
        f"## {month}",
        "",
        "### 1. Economic similarity ↔ highway distance",
        "",
        md_table(overlap, index=False, fmt=",.1f"),
        "",
        md_table(corr, fmt=",.3g"),
        "",
        "Средняя similarity по поясам автодорожного расстояния (все пары МО):",
        "",
        md_table(by_band),
        "",
        f"### 2. Экономически похожие, но не транспортные соседи",
        "",
        names_note(),
        "",
        f"- порог топ-{TOP_SIMILARITY_SHARE:.0%} similarity среди рёбер экономической сети: "
        f"**{thr:.4f}** → {len(top):,} пар",
        f"- из них не являются рёбрами ни highway kNN, ни railway kNN: **{len(far):,}** "
        f"({len(far) / len(top):.1%}); медиана highway distance "
        f"{far['highway distance'].median():,.0f} км — многие из них близко, просто не в топ-8",
        f"- **из них дальше {FAR_MIN_KM} км по автодороге: {len(far200):,}** — это и есть "
        "«экономически похожие, но территориально не соседствующие» пары; "
        f"медиана {far200['highway distance'].median():,.0f} км, "
        f"> 1000 км: {(far200['highway distance'] > 1000).mean():.1%}",
        f"- полный список (все {len(far):,}, флаг `дальше порога`): "
        f"`{far_path.relative_to(PROJECT_DIR)}`",
        "",
        f"#### Кластер пар на {CLUSTER_KM[0]}–{CLUSTER_KM[1]} км "
        f"(market_access {CLUSTER_MA_LOW[0]}–{CLUSTER_MA_LOW[1]} ↔ "
        f"{CLUSTER_MA_HIGH[0]}–{CLUSTER_MA_HIGH[1]})",
        "",
        f"- пар: **{len(cluster)}** из {len(far200)} дальних "
        f"({len(cluster) / max(len(far200), 1):.0%}); МО на стороне низкого market_access: "
        f"{len(set(lo_side))}, высокого: {len(set(hi_side))}",
        region_line(lo_side, hi_side),
        f"- has_railway на стороне низкого / высокого market_access: "
        f"{nodes.set_index('territory_id').loc[sorted(set(lo_side)), 'has_railway'].mean():.0%} / "
        f"{nodes.set_index('territory_id').loc[sorted(set(hi_side)), 'has_railway'].mean():.0%}",
        "",
    ]
    if month == MONTHS[0]:
        cols = KEY + (["name_x", "name_y"] if names is not None else []) + [
            "similarity", "highway distance", "market_access_x", "market_access_y"]
        lines += [
            "Сторона низкого market_access:",
            "",
            md_table(side_table(lo_side).astype({"territory_id": str}), index=False, fmt=",.1f"),
            "",
            "Сторона высокого market_access:",
            "",
            md_table(side_table(hi_side).astype({"territory_id": str}), index=False, fmt=",.1f"),
            "",
            f"#### Дальние пары (> {FAR_MIN_KM} км), первые {N_PAIRS_IN_REPORT} по similarity",
            "",
            md_table(far200.head(N_PAIRS_IN_REPORT)
                     .astype({"territory_id_x": str, "territory_id_y": str})[cols],
                     index=False, fmt=",.4f"),
            "",
            "Самые удалённые (топ-20 по highway distance):",
            "",
            md_table(far200.sort_values("highway distance", ascending=False).head(20)
                     .astype({"territory_id_x": str, "territory_id_y": str})[cols],
                     index=False, fmt=",.4f"),
            "",
        ]
    lines += [
        "### 3. Economic centrality (средняя similarity к соседям в экономической сети)",
        "",
        md_table(nd["econ_centrality"].describe().rename("econ_centrality").to_frame()),
        "",
        md_table(cent_corr, fmt=",.3g"),
        "",
        "Centrality по наличию ж/д:",
        "",
        md_table(by_rail),
        "",
        f"Mann–Whitney (с ж/д vs без): U = {mw.statistic:,.0f}, p = {mw.pvalue:.3g}",
        "",
        "Centrality по квинтилям market_access:",
        "",
        md_table(ma_q),
        "",
    ]
    summary = {
        "econ ∩ highway: Spearman": c_both["Spearman ρ"],
        "все пары: Spearman": c_all["Spearman ρ"],
        "все пары: Pearson": c_all["Pearson r"],
        "все рёбра econ: Spearman": c_econ["Spearman ρ"],
        "рёбер econ ∩ highway": len(e_econ & e_hw),
        "похожих не-соседей (п.2)": len(far),
        f"из них > {FAR_MIN_KM} км": len(far200),
        "из них в кластере ~700 км": len(cluster),
    }
    cent_summary = {
        "centrality ↔ market_access, Spearman": cent_corr.iloc[0]["Spearman ρ"],
        "centrality ↔ railway_degree (все), Spearman": cent_corr.iloc[1]["Spearman ρ"],
        "centrality ↔ railway_degree (с ж/д), Spearman": cent_corr.iloc[2]["Spearman ρ"],
        "centrality ↔ has_railway, r": cent_corr.iloc[3]["Pearson r"],
    }
    return lines, summary, {**cent_summary, "_cent": cent}


def pair_set_iter(df: pd.DataFrame):
    return zip(df["territory_id_x"].astype(int), df["territory_id_y"].astype(int))


def main() -> None:
    hw_knn = pd.read_parquet(HIGHWAY_KNN_PATH, engine="pyarrow")
    rw_knn = pd.read_parquet(RAILWAY_KNN_PATH, engine="pyarrow")
    conn = pd.read_parquet(CONNECTION_VALID_PATH, engine="pyarrow")
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    rd = pd.read_parquet(RAILWAY_DEGREE_PATH, engine="pyarrow")
    ma = pd.read_parquet(MARKET_ACCESS_PATH, engine="pyarrow")

    nodes = rd[["territory_id", "has_railway", "railway_degree"]].merge(
        ma, on="territory_id", how="left")
    if nodes["market_access"].isna().any():
        raise SystemExit("STOP: у части МО рабочей выборки нет market_access")
    ids = np.sort(nodes["territory_id"].to_numpy())
    names = load_territory_names()

    hw = conn[conn["type"] == "highway"]
    hw_dist = hw.set_index(KEY)["distance"]
    D_hw = dense(ids, hw, "distance")

    lines = [
        "# 08. Экономическая близость vs транспортная близость",
        "",
        "Сгенерировано `src/08_economic_vs_transport.py`.",
        "",
        "- Экономическая сеть: z-score долей внутри месяца + cosine + kNN (k=8).",
        "- Транспорт: kNN (k=8) по highway / railway (railway — с достройкой 95 пар).",
        "- market_access читается из `data/raw/market_access.parquet` (в processed его нет).",
        "- Корреляции на пересечении kNN-рёбер сжаты по диапазону (обе величины уже отобраны "
        "как «ближайшие»), поэтому рядом приведены корреляции по всем 2 007 006 парам МО.",
        "- Отрицательная корреляция similarity ↔ distance = «чем ближе, тем похожее».",
        "",
    ]
    summaries, cent_summaries, cents = {}, {}, {}
    for month in MONTHS:
        part, summ, cs = analyse_month(month, hw_knn, rw_knn, hw_dist, D_hw, ids, shares,
                                         nodes, names)
        cents[month] = cs.pop("_cent")
        lines += part
        summaries[month], cent_summaries[month] = summ, cs

    # --- 4. Устойчивость во времени ---
    c1, c2 = cents[MONTHS[0]], cents[MONTHS[1]]
    cc = corr_row(c1, c2.reindex(c1.index))
    lines += [
        f"## 4. Устойчивость: {MONTHS[0]} vs {MONTHS[1]}",
        "",
        md_table(pd.DataFrame(summaries), fmt=",.3f"),
        "",
        md_table(pd.DataFrame(cent_summaries), fmt=",.3f"),
        "",
        f"Корреляция economic centrality между месяцами: Pearson {cc['Pearson r']:.3f}, "
        f"Spearman {cc['Spearman ρ']:.3f}",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
