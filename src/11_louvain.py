"""Шаг 11. Louvain-разбиение экономической сети и сетевые метрики качества (MQ, AVI, AVU).

Граф: экономическая сеть месяца (z-score + cosine + kNN k=8), вес ребра — similarity.
Для сравнения те же метрики считаются для канонического KMeans (kmeans_labels_final) на
том же графе. Формулы метрик — src/partition_metrics.py (Shalileh, Antonov, Tsyplakova, 2025).

Вход:  data/processed/economic_networks/<YYYY-MM>_std_knn8.parquet,
       data/processed/kmeans_labels_final.parquet, data/raw/territories.parquet,
       data/raw/market_access.parquet
Выход: data/processed/louvain_labels_<YYYY_MM>_seed<SEED>.parquet (предварительное разбиение;
         итоговое, лучшее по MQ из 100 запусков, — шаг 11b),
       notebooks/11_louvain_metrics.md
Запуск из корня проекта:  .venv/bin/python src/11_louvain.py
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from config import FINAL_K
from partition_metrics import partition_metrics, self_check

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TERRITORIES_PATH = RAW_DIR / "territories.parquet"
MARKET_ACCESS_PATH = RAW_DIR / "market_access.parquet"
KMEANS_FINAL_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "11_louvain_metrics.md"

from config import MONTH
from config import ECON_KNN_K as K_NN
from config import LOUVAIN_WEIGHT as WEIGHT
from config import LOUVAIN_RESOLUTION as RESOLUTION
from config import LOUVAIN_SEED as SEED
from config import LOUVAIN_STABILITY_SEEDS as STABILITY_SEEDS  # проверка чувствительности Louvain к seed
TOP_N_REGIONS = 3

NETWORK_PATH = PROCESSED_DIR / "economic_networks" / f"{MONTH}_std_knn{K_NN}.parquet"
# предварительный запуск (один seed); итоговое разбиение (лучшее по MQ из 100 запусков)
# пишет шаг 11b в louvain_labels_<YYYY_MM>.parquet
LOUVAIN_LABELS_PATH = PROCESSED_DIR / f"louvain_labels_{MONTH.replace('-', '_')}_seed{SEED}.parquet"


def md_table(df: pd.DataFrame, index: bool = False, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def louvain_labels(G: nx.Graph, seed: int) -> dict:
    comms = nx.community.louvain_communities(G, weight=WEIGHT, resolution=RESOLUTION, seed=seed)
    comms = sorted(comms, key=lambda c: (-len(c), min(c)))   # 0 = крупнейшее сообщество
    return {n: i for i, c in enumerate(comms) for n in c}


def main() -> None:
    edges = pd.read_parquet(NETWORK_PATH, engine="pyarrow")
    G = nx.Graph()
    G.add_weighted_edges_from(
        ((int(x), int(y), float(w)) for x, y, w in
         edges[["territory_id_x", "territory_id_y", WEIGHT]].itertuples(index=False)),
        weight=WEIGHT)
    if (edges[WEIGHT] <= 0).any():
        raise SystemExit("STOP: в сети есть рёбра с неположительным весом")

    km = pd.read_parquet(KMEANS_FINAL_PATH).set_index("territory_id")["cluster"]
    if set(km.index) != set(G.nodes):
        raise SystemExit("STOP: множества МО в сети и в kmeans_labels_final не совпадают")

    # Louvain
    lab = louvain_labels(G, SEED)
    lv = pd.Series(lab, name="cluster").rename_axis("territory_id").sort_index()
    lv.reset_index().to_parquet(LOUVAIN_LABELS_PATH, engine="pyarrow", index=False)

    # Чувствительность к seed
    stab = []
    for s in STABILITY_SEEDS:
        l_s = louvain_labels(G, s)
        m_s, _ = partition_metrics(G, l_s, weight=WEIGHT)
        stab.append({"seed": s, "K": m_s["K"], "MQ": m_s["MQ"],
                     f"ARI с seed={SEED}": adjusted_rand_score(
                         lv.to_numpy(), pd.Series(l_s).reindex(lv.index).to_numpy())})
    stab = pd.DataFrame(stab)

    # Метрики: Louvain и KMeans на одном графе
    m_lv, pc_lv = partition_metrics(G, lab, weight=WEIGHT)
    m_km, pc_km = partition_metrics(G, km.to_dict(), weight=WEIGHT)
    # проверка весов: те же разбиения на невзвешенном графе (источник рассматривает невзвешенные графы)
    m_lv_u, _ = partition_metrics(G, lab, weight=None)
    m_km_u, _ = partition_metrics(G, km.to_dict(), weight=None)
    weight_delta = max(abs(w[k] - u[k]) for w, u in [(m_lv, m_lv_u), (m_km, m_km_u)]
                       for k in ("MQ", "AVI", "AVU"))
    avu_equal = (m_lv["K"] - 1) / (2 * m_lv["K"] - 3)   # AVU для равных кластеров и равномерных связей
    comp = pd.DataFrame([{"разбиение": f"Louvain (resolution={RESOLUTION}, seed={SEED})", **m_lv},
                         {"разбиение": f"KMeans k={FINAL_K} (kmeans_labels_final)", **m_km}])

    # Состав сообществ Louvain
    terr = pd.read_parquet(TERRITORIES_PATH).set_index("territory_id")
    ma = pd.read_parquet(MARKET_ACCESS_PATH).set_index("territory_id")["market_access"]
    comp_rows = []
    for c, ids in lv.groupby(lv).groups.items():
        regions = terr.loc[ids, "region_name"].value_counts()
        comp_rows.append({
            "cluster": c,
            "МО": len(ids),
            "market_access медиана": ma.loc[ids].median(),
            f"топ-{TOP_N_REGIONS} регионов": "; ".join(
                f"{r} {n}" for r, n in regions.head(TOP_N_REGIONS).items()),
            "регионов": len(regions),
        })
    lv_comp = pd.DataFrame(comp_rows).merge(
        pc_lv.drop(columns="узлов"), on="cluster")

    check_tab, check_notes = self_check()

    lines = [
        f"# 11. Louvain-разбиение экономической сети ({MONTH}) и метрики MQ / AVI / AVU",
        "",
        "Сгенерировано `src/11_louvain.py`.",
        "",
        f"- Граф: `{NETWORK_PATH.relative_to(PROJECT_DIR)}` — {G.number_of_nodes()} МО, "
        f"{G.number_of_edges():,} рёбер, вес = {WEIGHT}.",
        f"- Louvain: `networkx.community.louvain_communities`, resolution = {RESOLUTION}, "
        f"seed = {SEED}. Метки: `{LOUVAIN_LABELS_PATH.relative_to(PROJECT_DIR)}` "
        "(сообщество 0 — крупнейшее). **Предварительный запуск** — итоговое разбиение "
        "(лучшее по MQ из 100 запусков) и анализ устойчивости: `11b_louvain_stability.md`.",
        "",
        "## Формулы (Shalileh, Antonov, Tsyplakova, 2025, Doklady Mathematics 112(3), (18)–(21))",
        "",
        "- E_int(k) — сумма весов рёбер внутри кластера k; B_k — сумма весов рёбер с ровно "
        "одним концом в k; E(k, l) — сумма весов рёбер между k и l.",
        "- Isolability(k) = 2·E_int(k) / (2·E_int(k) + B_k); **AVI** = (1/K)·Σ_k Isolability(k).",
        "- Unifiability(k, l) = E(k, l) / (B_k + B_l − E(k, l)); **AVU** = (1/K)·Σ_{k≠l} "
        "Unifiability(k, l) по всем K(K−1) упорядоченным парам, нормировка 1/K — как в (21). "
        "Она же гарантирует AVU ∈ [0, 1]: U(k, l) ≤ E(k, l)/B_k и Σ_l E(k, l) = B_k.",
        "- **MQ** — модулярность Ньюмана (`networkx.community.modularity`, weight = similarity).",
        "- Хорошее разбиение: MQ ↑, AVI ↑, AVU ↓ (как в источнике: «maximize AVI and ANUI, "
        "while minimizing AVU»); интерпретацию AVU см. в проверке ниже.",
        "- В источнике при N ≈ 2000 плато AVU — 0.67 / 0.56 / 0.55 / 0.53 / 0.52 для "
        "K = 3 / 6 / 9 / 12 / 15 (табл. 3). Это совпадает в пределах точности чтения графиков с "
        "(K−1)/(2K−3) для равных кластеров и равномерных межкластерных рёбер; слабая зависимость "
        "AVU от p — следствие формулы, а не эмпирическая закономерность. AVU предварительного "
        f"Louvain (seed {SEED}, K = {m_lv['K']}) — {m_lv['AVU']:.3f}, близко к значению "
        f"(K−1)/(2K−3) = {avu_equal:.3f} для этого K, поэтому разделённость читаем по AVI и MQ, "
        "а не по AVU.",
        "",
        "## Итоговые метрики",
        "",
        md_table(comp),
        "",
        f"Проверка весов: макс. |Δ| между взвешенным ({WEIGHT}) и невзвешенным вариантом = "
        f"{weight_delta:.4f}; источник (Shalileh et al., 2025) рассматривает невзвешенные графы, на "
        "наших данных различие несущественно.",
        "",
        "## Сообщества Louvain",
        "",
        md_table(lv_comp, fmt=",.3f"),
        "",
        f"## Чувствительность Louvain к seed ({len(stab)} запусков)",
        "",
        md_table(stab),
        "",
        f"K: {stab['K'].min()}–{stab['K'].max()}; MQ: {stab['MQ'].min():.4f}–"
        f"{stab['MQ'].max():.4f}; ARI с основным запуском: медиана "
        f"{stab.iloc[:, 3].median():.3f}, минимум {stab.iloc[:, 3].min():.3f}.",
        "",
        "## Проверка метрик на синтетических графах",
        "",
        "4 клики по 10 узлов (вес внутренних рёбер 1) + 6 межкластерных рёбер веса w — "
        "равномерно по парам кластеров или сосредоточенных в двух парах; для сравнения — "
        "случайный граф (n = 200, p = 0.05) со случайной разметкой на 4 кластера.",
        "",
        md_table(check_tab),
        "",
        *check_notes,
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(comp.to_string(index=False))
    print(stab.to_string(index=False))
    print(lv_comp.to_string(index=False))


if __name__ == "__main__":
    main()
