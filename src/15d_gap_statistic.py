"""Шаг 15d. Gap statistic (Tibshirani, Walther, Hastie, 2001) для k из config.ROBUST_GAP_K_RANGE, декабрь 2024.

Признаки канона (z-score 5 долей). W_k — inertia KMeans (параметры шага 10). Эталон —
ROBUST_GAP_REFS выборок, равномерных в ограничивающем прямоугольнике данных в осях главных компонент
(вариант статьи с поворотом по SVD). Gap(k) = E*[log W_k] − log W_k; s_k = sd·√(1 + 1/B).
Правило статьи: наименьшее k с Gap(k) ≥ Gap(k+1) − s_{k+1}.

Выход: data/processed/robust2_gap.parquet
Запуск из корня проекта:  .venv/bin/python src/15d_gap_statistic.py
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import MONTH, ROBUST_GAP_K_RANGE, ROBUST_GAP_REFS
from network_utils import SHARE_COLUMNS, month_shares, standardize_shares

step10 = importlib.import_module("10_kmeans_k_selection")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
OUT_PATH = PROCESSED_DIR / "robust2_gap.parquet"
from config import GAP_SEED


def log_w(X: np.ndarray, k: int) -> float:
    return float(np.log(KMeans(n_clusters=k, random_state=step10.RANDOM_STATE, n_init=step10.N_INIT).fit(X).inertia_))


def main() -> None:
    X, _ = standardize_shares(month_shares(pd.read_parquet(SHARES_PATH), MONTH)[SHARE_COLUMNS].to_numpy())
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xp = Xc @ Vt.T
    lo, hi = Xp.min(axis=0), Xp.max(axis=0)
    rng = np.random.default_rng(GAP_SEED)
    refs = [rng.uniform(lo, hi, size=Xp.shape) @ Vt for _ in range(ROBUST_GAP_REFS)]
    rows = []
    for k in ROBUST_GAP_K_RANGE:
        lw = log_w(X, k)
        lw_ref = np.array([log_w(R, k) for R in refs])
        rows.append({"k": k, "log W": lw, "E*[log W]": lw_ref.mean(), "Gap": lw_ref.mean() - lw,
                     "s_k": lw_ref.std(ddof=0) * np.sqrt(1 + 1 / ROBUST_GAP_REFS)})
        print(rows[-1])
    res = pd.DataFrame(rows)
    ks = res["k"].tolist()
    res["Gap(k) ≥ Gap(k+1) − s(k+1)"] = [
        bool(res.at[i, "Gap"] >= res.at[i + 1, "Gap"] - res.at[i + 1, "s_k"]) if i + 1 < len(ks) else np.nan
        for i in range(len(ks))]
    res.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    sel = res[res["Gap(k) ≥ Gap(k+1) − s(k+1)"] == True]  # noqa: E712
    print("k по правилу статьи:", int(sel["k"].iloc[0]) if len(sel) else "не найдено в диапазоне",
          "; argmax Gap:", int(res.loc[res["Gap"].idxmax(), "k"]))


if __name__ == "__main__":
    main()
