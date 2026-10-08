"""Шаг 10d. Prediction strength (Tibshirani & Walther, 2005) для KMeans, k = 2..15, декабрь 2024.

Для каждого из M случайных разбиений МО на равные половины A и B и каждого k:
  - KMeans (параметры шага 10) обучается отдельно на A и на B;
  - «истинные» кластеры тестовой половины — по её собственной модели; точки тестовой половины
    классифицируются по ближайшему центроиду модели обучающей половины;
  - для каждого тестового кластера — доля пар его точек, которые модель обучающей половины
    тоже относит в один кластер; PS разбиения = МИНИМУМ этой доли по тестовым кластерам.
Оба направления (A→B и B→A) дают 2·M значений; итог для k — их среднее.
Тестовые кластеры из одной точки (пар нет) в минимуме не участвуют и учитываются отдельно.

Диагностика: каждому тестовому кластеру приписывается доминирующий тип по меткам шага 10b для
k = config.FINAL_K и config.SUPERSEDED_K (по большинству его МО) — видно, какие типы
воспроизводятся хуже.

Выход: data/processed/prediction_strength_<YYYY_MM>.parquet (по k),
       data/processed/prediction_strength_clusters_<YYYY_MM>.parquet (по тестовым кластерам),
       data/processed/prediction_strength_runs_<YYYY_MM>.parquet (по разбиениям),
       notebooks/figures/prediction_strength_by_k.png
Запуск из корня проекта:  .venv/bin/python src/10d_prediction_strength.py
"""
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import FINAL_K, SUPERSEDED_K
from network_utils import SHARE_COLUMNS, month_shares, standardize_shares

step10 = importlib.import_module("10_kmeans_k_selection")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
FIGURES_DIR = PROJECT_DIR / "notebooks" / "figures"

MONTH = step10.MONTH
K_RANGE = step10.K_RANGE
RANDOM_STATE = step10.RANDOM_STATE
N_INIT = step10.N_INIT
from config import PS_M_SPLITS as M_SPLITS  # как параметр M в fpc::prediction.strength
from config import PS_SPLIT_SEED as SPLIT_SEED
from config import PS_THRESHOLDS as THRESHOLDS  # пороги из Tibshirani & Walther (2005)

TAG = MONTH.replace("-", "_")
PS_PATH = PROCESSED_DIR / f"prediction_strength_{TAG}.parquet"
PS_RUNS_PATH = PROCESSED_DIR / f"prediction_strength_runs_{TAG}.parquet"
PS_CLUSTERS_PATH = PROCESSED_DIR / f"prediction_strength_clusters_{TAG}.parquet"
# типы для диагностики — метки шага 10b (kmeans_labels_final — копия для FINAL_K, пишется позже, в 10c)
DIAG_KS = [FINAL_K, SUPERSEDED_K]
FIG_PATH = FIGURES_DIR / "prediction_strength_by_k.png"


def cluster_ratios(test_true: np.ndarray, test_pred: np.ndarray, k: int) -> dict:
    """Для каждого тестового кластера (≥ 2 точек) — доля его пар, совпадающих по предсказанию."""
    out = {}
    for c in range(k):
        pred = test_pred[test_true == c]
        n = len(pred)
        if n >= 2:
            m = np.bincount(pred, minlength=k)
            out[c] = (m * (m - 1)).sum() / (n * (n - 1))
    return out


def main() -> None:
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    month = month_shares(shares, MONTH)
    X, _ = standardize_shares(month[SHARE_COLUMNS].to_numpy())
    canon = {kc: pd.read_parquet(PROCESSED_DIR / f"kmeans_labels_k{kc}_{TAG}.parquet")
             .set_index("territory_id")["cluster"].loc[month["territory_id"].to_numpy()].to_numpy()
             for kc in DIAG_KS}
    n = len(X)
    rng = np.random.default_rng(SPLIT_SEED)
    splits = []
    for _ in range(M_SPLITS):
        perm = rng.permutation(n)
        splits.append((perm[: n // 2], perm[n // 2: 2 * (n // 2)]))   # равные половины

    rows, crow = [], []
    for k in K_RANGE:
        for s, (ia, ib) in enumerate(splits):
            km_a = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X[ia])
            km_b = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X[ib])
            for direction, train, test_km, test_idx in [("A→B", km_a, km_b, ib), ("B→A", km_b, km_a, ia)]:
                ratios = cluster_ratios(test_km.labels_, train.predict(X[test_idx]), k)
                ps = min(ratios.values()) if ratios else np.nan
                rows.append({"k": k, "split": s, "направление": direction, "PS": ps,
                             "кластеров-синглтонов": k - len(ratios)})
                for c, r in ratios.items():
                    mem = test_km.labels_ == c
                    crow.append({"k": k, "split": s, "направление": direction, "кластер": c,
                                 "МО": int(mem.sum()), "доля пар": r, "минимум": r == ps,
                                 **{f"доминирующий тип k={kc}": int(np.bincount(
                                     canon[kc][test_idx][mem], minlength=kc).argmax())
                                    for kc in DIAG_KS}})
        print(f"k={k}: PS mean {np.nanmean([r['PS'] for r in rows if r['k'] == k]):.3f}")
    runs = pd.DataFrame(rows)
    runs.to_parquet(PS_RUNS_PATH, engine="pyarrow", index=False)
    pd.DataFrame(crow).to_parquet(PS_CLUSTERS_PATH, engine="pyarrow", index=False)

    g = runs.groupby("k")["PS"]
    res = pd.DataFrame({
        "PS mean": g.mean(), "PS sd": g.std(), "PS se": g.std() / np.sqrt(g.count()),
        "PS медиана": g.median(), "PS 10%": g.quantile(.1),
        **{f"доля разбиений с PS ≥ {t}": g.apply(lambda x, t=t: (x >= t).mean()) for t in THRESHOLDS},
        "значений": g.count(),
        "кластеров-синглтонов (всего)": runs.groupby("k")["кластеров-синглтонов"].sum(),
    }).reset_index()
    res.to_parquet(PS_PATH, engine="pyarrow", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(res["k"], res["PS mean"], yerr=1.96 * res["PS se"], marker="o", color="#4C72B0",
                capsize=3, label=f"PS, среднее по {2 * M_SPLITS} (±95% ДИ среднего)")
    ax.fill_between(res["k"], res["PS 10%"], res["PS mean"], color="#4C72B0", alpha=.12,
                    label="10-й перцентиль … среднее")
    for t, c in zip(THRESHOLDS, ["#DD8452", "#C44E52"]):
        ax.axhline(t, color=c, ls="--", lw=1, label=f"порог {t}")
    ax.set(xlabel="k (число кластеров)", ylabel="prediction strength", xticks=list(K_RANGE),
           ylim=(0, 1.02), title=f"Prediction strength (Tibshirani & Walther, 2005), {MONTH}")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=120)
    plt.close(fig)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
