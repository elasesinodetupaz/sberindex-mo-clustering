"""Шаг 15c. Честное сравнение с Louvain (проверка 3 батча 2), экономическая сеть декабря 2024.

1. resolution из config.ROBUST_LOUVAIN_RESOLUTIONS, по ROBUST_LOUVAIN_SEEDS запусков: лучший по MQ;
   K, MQ, AVI, AVU (partition_metrics), ARI и взвешенная чистота относительно канона k = FINAL_K
   (чистота в обе стороны). Выделяется resolution с K ближе всего к ROBUST2_TARGET_K.
2. kNN из config.ROBUST_KNN_VARIANTS при resolution = 1: K, MQ, ARI между вариантами и с каноном.

Выход: data/processed/robust2_louvain_resolution.parquet, data/processed/robust2_louvain_knn.parquet,
       data/processed/robust2_louvain_knn_ari.parquet
Запуск из корня проекта:  .venv/bin/python src/15c_louvain_resolution.py
"""
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from config import (FINAL_K, MONTH, ROBUST2_EXTRA_RESOLUTIONS, ROBUST2_TARGET_K, ROBUST_KNN_VARIANTS, ROBUST_LOUVAIN_RESOLUTIONS,
                    ROBUST_LOUVAIN_SEEDS)
from network_utils import SHARE_COLUMNS, knn_edges, month_shares, std_cosine_similarity
from partition_metrics import partition_metrics

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
CANON_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
BASE_NETWORK = PROCESSED_DIR / "economic_networks" / f"{MONTH}_std_knn8.parquet"
RES_PATH = PROCESSED_DIR / "robust2_louvain_resolution.parquet"
KNN_PATH = PROCESSED_DIR / "robust2_louvain_knn.parquet"
KNN_ARI_PATH = PROCESSED_DIR / "robust2_louvain_knn_ari.parquet"
from config import LOUVAIN_WEIGHT as WEIGHT


def graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    G.add_weighted_edges_from(((int(x), int(y), float(w)) for x, y, w in
                               edges[["territory_id_x", "territory_id_y", WEIGHT]].itertuples(index=False)),
                              weight=WEIGHT)
    return G


def best_louvain(G: nx.Graph, resolution: float) -> tuple[dict, float, list[int]]:
    best, ks = None, []
    for s in range(ROBUST_LOUVAIN_SEEDS):
        comms = nx.community.louvain_communities(G, weight=WEIGHT, resolution=resolution, seed=s)
        mq = nx.community.modularity(G, comms, weight=WEIGHT)
        ks.append(len(comms))
        if best is None or round(mq, 12) > round(best[1], 12):
            best = (comms, mq)
    comms = sorted(best[0], key=lambda c: (-len(c), min(c)))
    return {n: i for i, c in enumerate(comms) for n in c}, best[1], ks


def purity(a: np.ndarray, b: np.ndarray) -> float:
    """Доля объектов в «основном» кластере b своего кластера a."""
    ct = pd.crosstab(a, b)
    return float(ct.max(axis=1).sum() / ct.to_numpy().sum())


def main() -> None:
    canon = pd.read_parquet(CANON_PATH).set_index("territory_id")["cluster"]
    G = graph(pd.read_parquet(BASE_NETWORK))
    nodes = np.array(sorted(G.nodes))
    km = canon.loc[nodes].to_numpy()

    rows = []
    for r in sorted(set(ROBUST_LOUVAIN_RESOLUTIONS) | set(ROBUST2_EXTRA_RESOLUTIONS)):
        lab, mq, ks = best_louvain(G, r)
        m, _ = partition_metrics(G, lab, weight=WEIGHT)
        lv = np.array([lab[n] for n in nodes])
        rows.append({"resolution": r, "подобрано под K": r in ROBUST2_EXTRA_RESOLUTIONS, "K": m["K"], "K по 20 запускам: min": min(ks),
                     "K по 20 запускам: max": max(ks), "MQ": m["MQ"], "AVI": m["AVI"], "AVU": m["AVU"],
                     "ARI с каноном": adjusted_rand_score(km, lv),
                     "чистота Louvain → KMeans": purity(lv, km), "чистота KMeans → Louvain": purity(km, lv)})
        print(rows[-1])
    m_km, _ = partition_metrics(G, dict(zip(nodes.tolist(), km.tolist())), weight=WEIGHT)
    rows.append({"resolution": np.nan, "K": FINAL_K, "MQ": m_km["MQ"], "AVI": m_km["AVI"], "AVU": m_km["AVU"],
                 "ARI с каноном": 1.0, "чистота Louvain → KMeans": 1.0, "чистота KMeans → Louvain": 1.0,
                 "метод": f"KMeans k={FINAL_K} (канон) на том же графе"})
    res = pd.DataFrame(rows)
    res["метод"] = res["метод"].fillna("Louvain")
    lv_rows = res[res["метод"] == "Louvain"]
    res["K ближе всего к целевому"] = False
    res.loc[(lv_rows["K"] - ROBUST2_TARGET_K).abs().idxmin(), "K ближе всего к целевому"] = True
    res.to_parquet(RES_PATH, engine="pyarrow", index=False)

    # kNN
    sh = month_shares(pd.read_parquet(SHARES_PATH), MONTH)
    ids = sh["territory_id"].to_numpy()
    sim, _ = std_cosine_similarity(sh[SHARE_COLUMNS].to_numpy())
    parts, krows = {}, []
    for kk in ROBUST_KNN_VARIANTS:
        e = knn_edges(ids, sim, kk)
        if kk == 8:
            base = pd.read_parquet(BASE_NETWORK)
            if not e[["territory_id_x", "territory_id_y"]].equals(base[["territory_id_x", "territory_id_y"]]):
                raise SystemExit("STOP: kNN = 8 не совпал с сетью шага 06")
        Gk = graph(e)
        lab, mq, ks = best_louvain(Gk, 1.0)
        lv = np.array([lab[n] for n in ids])
        parts[kk] = lv
        krows.append({"kNN": kk, "рёбер": Gk.number_of_edges(), "K": len(set(lv)),
                      "K по 20 запускам: min": min(ks), "K по 20 запускам: max": max(ks), "MQ": mq,
                      "ARI с каноном": adjusted_rand_score(canon.loc[ids], lv)})
    pd.DataFrame(krows).to_parquet(KNN_PATH, engine="pyarrow", index=False)
    pd.DataFrame([{"kNN a": a, "kNN b": b, "ARI": adjusted_rand_score(parts[a], parts[b])}
                  for a, b in combinations(ROBUST_KNN_VARIANTS, 2)]).to_parquet(KNN_ARI_PATH, index=False)
    print(pd.DataFrame(krows).to_string(index=False))


if __name__ == "__main__":
    main()
