"""Шаг 15a. Выбор спецификации признаков A/B/C по всем 24 месяцам (проверка 1 батча 2).

A — 5 долей от «Все категории» (канон); B — 5 долей, нормированных к сумме пяти;
C — 5 долей + share_Прочее. Во всех вариантах z-score внутри месяца.
Для каждого варианта и месяца: KMeans k = ROBUST2_K, лучший по inertia из ROBUST2_N_RUNS
запусков (правило выбора — как в шаге 12a); для k из ROBUST2_K_RANGE — SW, CH/N (KMeans шага 10)
и prediction strength (разбиения и алгоритм шага 10d). Сопоставление месяцев с декабрём и
критерий проблемных месяцев — функции шага 12 (z-score). Для варианта A результат обязан
совпасть с шагами 12a/12 (встроенная проверка).

Выход: data/processed/robust2_spec_labels.parquet (вариант, месяц, territory_id, cluster),
       data/processed/robust2_spec_k.parquet (вариант, месяц, k, SW, CH/N, PS),
       data/processed/robust2_spec_months.parquet (вариант, месяц: проблемный, ARI с пред. мес.)
Запуск из корня проекта:  .venv/bin/python src/15a_feature_spec_full.py
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score
from threadpoolctl import threadpool_limits

from config import (FINAL_K, MONTH, ROBUST2_K, ROBUST2_K_RANGE, ROBUST2_N_RUNS, ROBUST2_VARIANTS,
                    ROBUST_N_JOBS, ROBUST_PS_SPLITS)
from network_utils import SHARE_COLUMNS, standardize_shares

step10 = importlib.import_module("10_kmeans_k_selection")
step10d = importlib.import_module("10d_prediction_strength")
step12 = importlib.import_module("12_kmeans_temporal_tracking")
step12a = importlib.import_module("12a_kmeans_runs")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
LABELS_PATH = PROCESSED_DIR / "robust2_spec_labels.parquet"
K_PATH = PROCESSED_DIR / "robust2_spec_k.parquet"
MONTHS_PATH = PROCESSED_DIR / "robust2_spec_months.parquet"


def features(S: np.ndarray, variant: str) -> np.ndarray:
    """Сырые признаки варианта из 5 долей от «Все категории» (S)."""
    if variant == "A":
        return S
    if variant == "B":
        return S / S.sum(axis=1, keepdims=True)
    if variant == "C":
        other = 1 - S.sum(axis=1)
        if (other < 0).any():
            raise SystemExit("STOP: сумма пяти долей > 1 — «прочее» отрицательно")
        return np.column_stack([S, other])
    raise SystemExit(f"STOP: неизвестный вариант {variant}")


def best_of_runs(X: np.ndarray, k: int) -> np.ndarray:
    runs = [(round(KMeans(n_clusters=k, random_state=s, n_init=step10.N_INIT).fit(X).inertia_, 6), s)
            for s in range(ROBUST2_N_RUNS)]
    best_seed = min(runs)[1]
    return KMeans(n_clusters=k, random_state=best_seed, n_init=step10.N_INIT).fit(X).labels_


def ps(X: np.ndarray, k: int) -> float:
    rng = np.random.default_rng(step10d.SPLIT_SEED)
    vals = []
    for _ in range(ROBUST_PS_SPLITS):
        perm = rng.permutation(len(X))
        ia, ib = perm[: len(X) // 2], perm[len(X) // 2: 2 * (len(X) // 2)]
        ka = KMeans(n_clusters=k, random_state=step10.RANDOM_STATE, n_init=step10.N_INIT).fit(X[ia])
        kb = KMeans(n_clusters=k, random_state=step10.RANDOM_STATE, n_init=step10.N_INIT).fit(X[ib])
        for train, test, idx in [(ka, kb, ib), (kb, ka, ia)]:
            vals.append(min(step10d.cluster_ratios(test.labels_, train.predict(X[idx]), k).values()))
    return float(np.mean(vals))


def job(variant: str, month: str, S: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    with threadpool_limits(1):
        X, _ = standardize_shares(features(S, variant))
        lab = best_of_runs(X, ROBUST2_K)
        rows = []
        for k in ROBUST2_K_RANGE:
            km = KMeans(n_clusters=k, random_state=step10.RANDOM_STATE, n_init=step10.N_INIT).fit(X)
            rows.append({"вариант": variant, "месяц": month, "k": k,
                         "SW": silhouette_score(X, km.labels_),
                         "CH/N": calinski_harabasz_score(X, km.labels_) / len(X), "PS": ps(X, k)})
        return lab, rows


def temporal(variant: str, months: list, S: dict, L: dict, canon_dec: np.ndarray) -> pd.DataFrame:
    """Сопоставление с декабрём (z-score) и проблемные месяцы — функции и критерии шага 12."""
    k = ROBUST2_K
    Z = {m: standardize_shares(features(S[m], variant))[0] for m in months}
    anchor = step12a.overlap_relabel(canon_dec, L[MONTH], k)
    anchor_c = step12.raw_centroids(Z[MONTH], anchor, k)
    aligned, amb = {}, {}
    for m in months:
        mapping, q = step12.match_to_anchor(anchor_c, step12.raw_centroids(Z[m], L[m], k))
        aligned[m] = np.vectorize(mapping.get)(L[m])
        amb[m] = int((q["уверенность"] == "неоднозначно").sum())
    rows = []
    for i, m in enumerate(months):
        rows.append({"вариант": variant, "месяц": m, "неоднозначных": amb[m],
                     "доля МО сменивших кластер": float((aligned[m] != aligned[months[i - 1]]).mean()) if i else np.nan,
                     "ARI с пред. мес.": adjusted_rand_score(L[months[i - 1]], L[m]) if i else np.nan})
    mt = pd.DataFrame(rows)
    sc = mt["доля МО сменивших кластер"].dropna()
    q1, q3 = sc.quantile([.25, .75])
    mt["проблемный"] = (mt["неоднозначных"] >= step12.PROBLEM_MONTH_MIN_AMBIGUOUS) | (
        mt["доля МО сменивших кластер"] > q3 + step12.PROBLEM_SWITCH_IQR * (q3 - q1))
    return mt


def main() -> None:
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    months = [f"{d:%Y-%m}" for d in sorted(shares["date"].unique())]
    ids = np.sort(shares["territory_id"].unique())
    S = {m: shares[shares["date"] == pd.Timestamp(m)].set_index("territory_id").loc[ids, SHARE_COLUMNS]
         .to_numpy() for m in months}
    tasks = [(v, m) for v in ROBUST2_VARIANTS for m in months]
    out = Parallel(n_jobs=ROBUST_N_JOBS, verbose=2)(delayed(job)(v, m, S[m]) for v, m in tasks)
    labels = {(v, m): lab for (v, m), (lab, _) in zip(tasks, out)}
    ktab = pd.DataFrame([r for _, rows in out for r in rows])

    # встроенная проверка: вариант A = официальные разбиения шага 12a (k = FINAL_K)
    if ROBUST2_K == FINAL_K:
        for m in months:
            off = pd.read_parquet(PROCESSED_DIR / f"kmeans_labels_k{FINAL_K}" / f"{m}.parquet"
                                  ).set_index("territory_id")["cluster"].loc[ids].to_numpy()
            if adjusted_rand_score(off, labels[("A", m)]) < 1 - 1e-9:
                raise SystemExit(f"STOP: вариант A, {m}: не совпал с официальным разбиением шага 12a")

    canon = pd.read_parquet(PROCESSED_DIR / "kmeans_labels_final.parquet").set_index("territory_id")[
        "cluster"].loc[ids].to_numpy()
    mts = pd.concat([temporal(v, months, S, {m: labels[(v, m)] for m in months}, canon)
                     for v in ROBUST2_VARIANTS], ignore_index=True)
    ref = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{FINAL_K}_months.parquet")
    a = mts[mts["вариант"] == "A"].reset_index(drop=True)
    if not (a["проблемный"].to_numpy() == ref["проблемный"].to_numpy()).all():
        raise SystemExit("STOP: вариант A: проблемные месяцы не совпали с шагом 12")

    pd.DataFrame([{"вариант": v, "месяц": m, "territory_id": t, "cluster": int(c)}
                  for (v, m), lab in labels.items() for t, c in zip(ids, lab)]).to_parquet(
        LABELS_PATH, engine="pyarrow", index=False)
    ktab.to_parquet(K_PATH, engine="pyarrow", index=False)
    mts.to_parquet(MONTHS_PATH, engine="pyarrow", index=False)
    print("OK: вариант A совпал с шагами 12a/12;",
          mts.groupby("вариант")["проблемный"].sum().to_dict(),
          mts.groupby("вариант")["ARI с пред. мес."].median().round(3).to_dict())


if __name__ == "__main__":
    main()
