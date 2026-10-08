"""Шаг 14b. Устойчивость выбора k во времени (проверки 2–3 шага 14).

Для каждого из 24 месяцев и k из config.ROBUST_K_RANGE (z-score 5 долей внутри месяца):
  - prediction strength (алгоритм и разбиения шага 10d, M = ROBUST_PS_SPLITS): стандартный
    (минимум по кластерам) и средний по кластерам, плюс доля пар по каждому кластеру;
  - SW и S_Dbw с baseline-нормализацией (функции и seed шага 10) для KMeans на всём месяце.
Декабрь 2024 обязан совпасть с шагами 10 и 10d (встроенная проверка).

Выход: data/processed/robust_k_by_month.parquet, data/processed/robust_ps_clusters_by_month.parquet
Запуск из корня проекта:  .venv/bin/python src/14b_k_over_time.py
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from s_dbw import S_Dbw
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from config import MONTH, ROBUST_K_RANGE, ROBUST_N_JOBS, ROBUST_N_RANDOM, ROBUST_PS_SPLITS
from network_utils import SHARE_COLUMNS, standardize_shares

step10 = importlib.import_module("10_kmeans_k_selection")
step10d = importlib.import_module("10d_prediction_strength")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
SUMMARY_PATH = PROCESSED_DIR / "robust_k_by_month.parquet"
CLUSTERS_PATH = PROCESSED_DIR / "robust_ps_clusters_by_month.parquet"


def make_splits(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Те же разбиения, что в шаге 10d."""
    rng = np.random.default_rng(step10d.SPLIT_SEED)
    out = []
    for _ in range(ROBUST_PS_SPLITS):
        perm = rng.permutation(n)
        out.append((perm[: n // 2], perm[n // 2: 2 * (n // 2)]))
    return out


def month_job(m: str, X: np.ndarray) -> tuple[list[dict], list[dict]]:
    with threadpool_limits(1):
        splits = make_splits(len(X))
        rows, crow = [], []
        for k in ROBUST_K_RANGE:
            km = KMeans(n_clusters=k, random_state=step10.RANDOM_STATE, n_init=step10.N_INIT).fit(X)
            s_dbw = S_Dbw(X, km.labels_, **step10.S_DBW_KW)
            base = step10.s_dbw_baseline(X, km.labels_, k)
            ps_min, ps_mean = [], []
            for s, (ia, ib) in enumerate(splits):
                ka = KMeans(n_clusters=k, random_state=step10.RANDOM_STATE, n_init=step10.N_INIT).fit(X[ia])
                kb = KMeans(n_clusters=k, random_state=step10.RANDOM_STATE, n_init=step10.N_INIT).fit(X[ib])
                for d, train, test, idx in [("A→B", ka, kb, ib), ("B→A", kb, ka, ia)]:
                    r = step10d.cluster_ratios(test.labels_, train.predict(X[idx]), k)
                    mn = min(r.values())
                    ps_min.append(mn)
                    ps_mean.append(float(np.mean(list(r.values()))))
                    for c, v in r.items():
                        crow.append({"месяц": m, "k": k, "split": s, "направление": d, "кластер": c,
                                     "МО": int((test.labels_ == c).sum()), "доля пар": v,
                                     "минимум": v == mn})
            rows.append({"месяц": m, "k": k,
                         "PS": np.mean(ps_min), "PS se": np.std(ps_min, ddof=1) / np.sqrt(len(ps_min)),
                         "PS (среднее по кластерам)": np.mean(ps_mean),
                         "PS (среднее по кластерам) se": np.std(ps_mean, ddof=1) / np.sqrt(len(ps_mean)),
                         "SW": silhouette_score(X, km.labels_), "S_Dbw": s_dbw,
                         "S_Dbw random mean": base.mean(), "S_Dbw random std": base.std(ddof=1),
                         "S_Dbw norm": (s_dbw - base.mean()) / base.std(ddof=1)})
    return rows, crow


def main() -> None:
    if ROBUST_N_RANDOM != step10.N_RANDOM:
        raise SystemExit("STOP: ROBUST_N_RANDOM должен совпадать с N_RANDOM шага 10")
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    months = [f"{d:%Y-%m}" for d in sorted(shares["date"].unique())]
    ids = np.sort(shares["territory_id"].unique())
    data = {m: standardize_shares(shares[shares["date"] == pd.Timestamp(m)].set_index("territory_id")
                                  .loc[ids, SHARE_COLUMNS].to_numpy())[0] for m in months}
    out = Parallel(n_jobs=ROBUST_N_JOBS, verbose=5)(delayed(month_job)(m, data[m]) for m in months)
    summary = pd.DataFrame([r for rows, _ in out for r in rows])
    clusters = pd.DataFrame([r for _, crow in out for r in crow])

    # встроенная проверка: декабрь = шаги 10 и 10d
    dec = summary[summary["месяц"] == MONTH].set_index("k")
    ps10d = pd.read_parquet(PROCESSED_DIR / f"prediction_strength_{MONTH.replace('-', '_')}.parquet").set_index("k")
    m10 = pd.read_parquet(PROCESSED_DIR / f"kmeans_k_selection_{MONTH.replace('-', '_')}.parquet").set_index("k")
    common = [k for k in dec.index if k in ps10d.index]
    if not (np.allclose(dec.loc[common, "PS"], ps10d.loc[common, "PS mean"])
            and np.allclose(dec.loc[common, "S_Dbw norm"], m10.loc[common, "S_Dbw norm"])):
        raise SystemExit("STOP: декабрь 2024 не совпал с шагами 10/10d")
    summary.to_parquet(SUMMARY_PATH, engine="pyarrow", index=False)
    clusters.to_parquet(CLUSTERS_PATH, engine="pyarrow", index=False)
    print("OK: декабрь совпал с шагами 10/10d;", len(summary), "строк")


if __name__ == "__main__":
    main()
