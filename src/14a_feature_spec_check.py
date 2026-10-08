"""Шаг 14a. Чувствительность канона к спецификации признаков (проверка 1 шага 14), декабрь 2024.

Варианты (z-score внутри месяца во всех):
  A — 5 долей от «Все категории» (текущий);
  B — 5 долей, перенормированных к сумме пяти;
  C — 6 признаков: 5 долей + share_Прочее (= 1 − сумма пяти).
Для каждого: KMeans k = FINAL_K, лучший по inertia из ROBUST_N_RUNS запусков; ARI с каноном
(kmeans_labels_final) и доля МО в другом кластере после сопоставления. Для B и C (и A — для
сравнения) — prediction strength для ROBUST_K_COMPARE на разбиениях шага 10d. Экономическая сеть
(z-score + cosine + kNN) — Жаккар рёбер с текущей сетью декабря.

Выход: data/processed/robust_feature_spec.parquet, data/processed/robust_feature_spec_ps.parquet
Запуск из корня проекта:  .venv/bin/python src/14a_feature_spec_check.py
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from config import FINAL_K, MONTH, ROBUST_K_COMPARE, ROBUST_KNN_K, ROBUST_N_RUNS, ROBUST_PS_SPLITS
from network_utils import SHARE_COLUMNS, knn_edges, month_shares, standardize_shares, std_cosine_similarity

step10 = importlib.import_module("10_kmeans_k_selection")
step10d = importlib.import_module("10d_prediction_strength")
step12a = importlib.import_module("12a_kmeans_runs")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
CANON_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
NETWORK_PATH = PROCESSED_DIR / "economic_networks" / f"{MONTH}_std_knn{ROBUST_KNN_K}.parquet"
OUT_PATH = PROCESSED_DIR / "robust_feature_spec.parquet"
OUT_PS_PATH = PROCESSED_DIR / "robust_feature_spec_ps.parquet"


def variants(sh: pd.DataFrame) -> dict[str, np.ndarray]:
    S = sh[SHARE_COLUMNS].to_numpy()
    other = 1 - S.sum(axis=1)
    if (other < 0).any():
        raise SystemExit("STOP: сумма пяти долей > 1 у части МО — «прочее» отрицательно")
    return {"A: 5 долей от «Все категории»": S,
            "B: 5 долей, нормированы к сумме пяти": S / S.sum(axis=1, keepdims=True),
            "C: 5 долей + share_Прочее": np.column_stack([S, other])}


def best_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    runs = [KMeans(n_clusters=k, random_state=s, n_init=step10.N_INIT).fit(X) for s in range(ROBUST_N_RUNS)]
    best = min(range(len(runs)), key=lambda i: (round(runs[i].inertia_, 6), i))
    return runs[best].labels_


def prediction_strength(X: np.ndarray, k: int) -> tuple[float, float]:
    rng = np.random.default_rng(step10d.SPLIT_SEED)
    vals = []
    for _ in range(ROBUST_PS_SPLITS):
        perm = rng.permutation(len(X))
        ia, ib = perm[: len(X) // 2], perm[len(X) // 2: 2 * (len(X) // 2)]
        ka = KMeans(n_clusters=k, random_state=step10.RANDOM_STATE, n_init=step10.N_INIT).fit(X[ia])
        kb = KMeans(n_clusters=k, random_state=step10.RANDOM_STATE, n_init=step10.N_INIT).fit(X[ib])
        for train, test, idx in [(ka, kb, ib), (kb, ka, ia)]:
            vals.append(min(step10d.cluster_ratios(test.labels_, train.predict(X[idx]), k).values()))
    return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def edge_set(e: pd.DataFrame) -> set:
    return set(zip(e["territory_id_x"].astype(int), e["territory_id_y"].astype(int)))


def main() -> None:
    sh = month_shares(pd.read_parquet(SHARES_PATH), MONTH)
    ids = sh["territory_id"].to_numpy()
    canon = pd.read_parquet(CANON_PATH).set_index("territory_id")["cluster"].loc[ids].to_numpy()
    net_now = edge_set(pd.read_parquet(NETWORK_PATH))

    rows, ps_rows = [], []
    for name, raw in variants(sh).items():
        X, _ = standardize_shares(raw)
        lab = best_kmeans(X, FINAL_K)
        rel = step12a.overlap_relabel(canon, lab, FINAL_K)
        sim, _ = std_cosine_similarity(raw)
        net = edge_set(knn_edges(ids, sim, ROBUST_KNN_K))
        rows.append({"вариант": name, "признаков": raw.shape[1],
                     "ARI с каноном": adjusted_rand_score(canon, lab),
                     "доля МО в другом кластере": float((rel != canon).mean()),
                     "Жаккар рёбер сети с текущей": len(net & net_now) / len(net | net_now)})
        for k in ROBUST_K_COMPARE:
            m, se = prediction_strength(X, k)
            ps_rows.append({"вариант": name, "k": k, "PS": m, "PS se": se})
        print(rows[-1], [r for r in ps_rows if r["вариант"] == name])
    pd.DataFrame(rows).to_parquet(OUT_PATH, engine="pyarrow", index=False)
    pd.DataFrame(ps_rows).to_parquet(OUT_PS_PATH, engine="pyarrow", index=False)


if __name__ == "__main__":
    main()
