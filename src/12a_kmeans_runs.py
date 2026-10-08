"""Шаг 12a. KMeans (k из --k, по умолчанию config.FINAL_K): 100 запусков на месяц, официальное разбиение, уверенность принадлежности МО.

Для каждого месяца: z-score 5 долей внутри месяца; KMeans(k) с random_state = 0..99
(каждый запуск — как в основном пайплайне: n_init из шага 10). Официальное разбиение —
запуск с минимальной inertia. Каждый запуск сопоставляется с официальным через
linear_sum_assignment по пересечению состава; уверенность принадлежности МО = доля запусков,
в которых МО попало в тот же (сопоставленный) кластер, что и в официальном разбиении.

Выход: data/processed/kmeans_labels_k{K}/{YYYY-MM}.parquet (territory_id, cluster — официальное
         разбиение месяца, исходные номера KMeans),
       data/processed/kmeans_k{K}_runs_summary.parquet (month, seed, inertia, ARI с официальным),
       data/processed/kmeans_k{K}_runs_labels.parquet (month, seed, territory_id, cluster — метки
         всех запусков, исходная нумерация каждого запуска),
       data/processed/kmeans_k{K}_official_confidence_raw.parquet (territory_id, month,
         cluster_raw, confidence) — нумерация официального разбиения; согласованную с якорем
         нумерацию добавляет шаг 12
Запуск из корня проекта:  .venv/bin/python src/12a_kmeans_runs.py [--k 6]
"""
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from config import FINAL_K
from network_utils import SHARE_COLUMNS, standardize_shares

step10 = importlib.import_module("10_kmeans_k_selection")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"


def paths(k: int) -> dict:
    return {"labels_dir": PROCESSED_DIR / f"kmeans_labels_k{k}",
            "runs_summary": PROCESSED_DIR / f"kmeans_k{k}_runs_summary.parquet",
            "conf_raw": PROCESSED_DIR / f"kmeans_k{k}_official_confidence_raw.parquet",
            "runs_labels": PROCESSED_DIR / f"kmeans_k{k}_runs_labels.parquet"}


from config import KMEANS_RUN_SEEDS as SEEDS
N_INIT = step10.N_INIT


def overlap_relabel(official: np.ndarray, other: np.ndarray, k: int) -> np.ndarray:
    """Перенумеровывает other в номера official (макс. суммарное пересечение состава)."""
    inter = np.zeros((k, k))
    np.add.at(inter, (official, other), 1)
    r, c = linear_sum_assignment(-inter)
    mapping = np.empty(k, dtype=int)
    mapping[c] = r
    return mapping[other]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=FINAL_K)
    K = ap.parse_args().k
    P = paths(K)
    LABELS_DIR, RUNS_SUMMARY_PATH = P["labels_dir"], P["runs_summary"]
    CONF_RAW_PATH, RUNS_LABELS_PATH = P["conf_raw"], P["runs_labels"]
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    months = [f"{d:%Y-%m}" for d in sorted(shares["date"].unique())]
    ids = np.sort(shares["territory_id"].unique())
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    summary, conf_rows, run_labels = [], [], []
    for m in months:
        d = shares[shares["date"] == pd.Timestamp(m)].set_index("territory_id").loc[ids]
        X, _ = standardize_shares(d[SHARE_COLUMNS].to_numpy())
        runs = []
        for s in SEEDS:
            km = KMeans(n_clusters=K, random_state=s, n_init=N_INIT).fit(X)
            runs.append((s, km.inertia_, km.labels_))
        # при равной (до 1e-6) inertia — наименьший seed: многопоточное суммирование во float
        # иначе делает выбор среди равных решений невоспроизводимым
        best_seed, best_inertia, official = min(runs, key=lambda r: (round(r[1], 6), r[0]))
        same = np.zeros(len(ids))
        for s, inertia, lab in runs:
            run_labels.append(pd.DataFrame({"month": m, "seed": np.int16(s),
                                            "territory_id": ids, "cluster": lab.astype(np.int8)}))
            relab = overlap_relabel(official, lab, K)
            same += relab == official
            summary.append({"month": m, "seed": s, "inertia": inertia,
                            "ARI с официальным": adjusted_rand_score(official, lab),
                            "официальный": s == best_seed})
        conf = same / len(runs)
        pd.DataFrame({"territory_id": ids, "cluster": official}).to_parquet(
            LABELS_DIR / f"{m}.parquet", engine="pyarrow", index=False)
        conf_rows.append(pd.DataFrame({"territory_id": ids, "month": m, "cluster_raw": official,
                                       "confidence": conf}))
        print(f"{m}: official seed {best_seed}, inertia {best_inertia:.2f}, "
              f"conf median {np.median(conf):.2f}, share<0.5 {(conf < .5).mean():.1%}")

    pd.DataFrame(summary).to_parquet(RUNS_SUMMARY_PATH, engine="pyarrow", index=False)
    pd.concat(run_labels, ignore_index=True).to_parquet(RUNS_LABELS_PATH, engine="pyarrow", index=False)
    pd.concat(conf_rows, ignore_index=True).to_parquet(CONF_RAW_PATH, engine="pyarrow", index=False)


if __name__ == "__main__":
    main()
