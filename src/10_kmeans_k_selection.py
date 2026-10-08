"""Шаг 10. Выбор числа кластеров KMeans: метрики качества для k = K_MIN..K_MAX.

Признаки: те же 5 долей категорий, стандартизированные (z-score) внутри месяца,
что и для экономической сети (network_utils.standardize_shares).
Итоговое k здесь НЕ выбирается — только метрики и графики для ручного решения.

Вход:  data/processed/category_shares.parquet
Выход: notebooks/10_kmeans_k_selection.md, notebooks/figures/*.png,
       data/processed/kmeans_k_selection_<YYYY_MM>.parquet (таблица метрик)
Запуск из корня проекта:  .venv/bin/python src/10_kmeans_k_selection.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from s_dbw import S_Dbw
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from network_utils import SHARE_COLUMNS, month_shares, standardize_shares

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
REPORT_PATH = NOTEBOOKS_DIR / "10_kmeans_k_selection.md"

from config import MONTH
from config import KMEANS_K_RANGE as K_RANGE
from config import KMEANS_RANDOM_STATE as RANDOM_STATE
from config import KMEANS_N_INIT as N_INIT
# S_Dbw: метод из оригинальной статьи Halkidi & Vazirgiannis (2001).
# Центр кластера — ближайшая к среднему точка (nearest_centr=True, по умолчанию в s-dbw):
# при центре-среднем плотность в окрестности центра в 5-D бывает нулевой, и для k >= 9
# индекс не определён (ValueError). Вариант с центром-средним — справочная колонка.
from config import SDBW_KW as S_DBW_KW
S_DBW_MEAN_CENTR_KW = {**S_DBW_KW, "nearest_centr": False}

# Baseline S_Dbw: N_RANDOM случайных разбиений на k групп — перестановка меток реального разбиения
# этого k (размеры кластеров сохраняются точно). Принцип (нормировать индексы, чьи случайные уровни
# зависят от K, на среднее и дисперсию случайного разбиения) рекомендован в Shalileh et al. (2025);
# сама процедура в статье не описана, авторы откладывают её. Перестановка меток с сохранением
# размеров кластеров и бутстреп-ДИ — наши.
from config import SDBW_N_RANDOM as N_RANDOM
from config import SDBW_BASELINE_SEED as BASELINE_SEED  # rng seed = BASELINE_SEED + k
from config import SDBW_N_BOOT as N_BOOT  # бутстреп по случайным разбиениям — 95% ДИ для S_Dbw norm

METRICS_PATH = PROCESSED_DIR / f"kmeans_k_selection_{MONTH.replace('-', '_')}.parquet"


def s_dbw_or_nan(X: np.ndarray, labels: np.ndarray) -> float:
    try:
        return S_Dbw(X, labels, **S_DBW_MEAN_CENTR_KW)
    except ValueError:  # нулевая плотность у центров двух и более кластеров
        return np.nan


def s_dbw_baseline(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """S_Dbw для N_RANDOM перестановок меток реального разбиения."""
    rng = np.random.default_rng(BASELINE_SEED + k)
    vals, failed = [], 0
    for _ in range(N_RANDOM):
        try:
            vals.append(S_Dbw(X, rng.permutation(labels), **S_DBW_KW))
        except ValueError:
            failed += 1
    if failed:
        raise SystemExit(f"STOP: k={k}: S_Dbw не определён для {failed} случайных разбиений")
    return np.array(vals)


def norm_ci(real: float, vals: np.ndarray, k: int) -> tuple[float, float]:
    """95% бутстреп-интервал для (real − mean) / std по выборке случайных значений."""
    rng = np.random.default_rng(BASELINE_SEED + 1000 + k)
    b = rng.choice(vals, size=(N_BOOT, len(vals)), replace=True)
    z = (real - b.mean(axis=1)) / b.std(axis=1, ddof=1)
    return float(np.quantile(z, .025)), float(np.quantile(z, .975))


def plot_metric(ks, values, ylabel: str, title: str, path: Path, note: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, values, marker="o", color="#4C72B0")
    ax.set(xlabel="k (число кластеров)", ylabel=ylabel, title=title, xticks=list(ks))
    ax.grid(alpha=.3)
    ax.text(0.99, 0.97, note, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#999999"))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    month = month_shares(shares, MONTH)
    X, scaler = standardize_shares(month[SHARE_COLUMNS].to_numpy())

    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X)
        labels = km.labels_
        sizes = np.bincount(labels)
        s_dbw = S_Dbw(X, labels, **S_DBW_KW)
        vals = s_dbw_baseline(X, labels, k)
        base_mean, base_std = float(vals.mean()), float(vals.std(ddof=1))
        lo, hi = norm_ci(s_dbw, vals, k)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "SW": silhouette_score(X, labels),
            "CH": calinski_harabasz_score(X, labels),
            "S_Dbw": s_dbw,
            "S_Dbw random mean": base_mean,
            "S_Dbw random std": base_std,
            "S_Dbw norm": (s_dbw - base_mean) / base_std,
            "S_Dbw norm 95% ДИ: низ": lo,
            "S_Dbw norm 95% ДИ: верх": hi,
            "S_Dbw (центр = среднее)": s_dbw_or_nan(X, labels),
            "мин. размер кластера": int(sizes.min()),
            "макс. размер кластера": int(sizes.max()),
        })
        print(f"k={k} done")
    res = pd.DataFrame(rows)
    res.to_parquet(METRICS_PATH, engine="pyarrow", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ks = res["k"]
    figs = [
        ("elbow_inertia.png", "inertia", "inertia (сумма квадратов до центров)",
         "Elbow: inertia KMeans", "меньше = плотнее; ищем «локоть»"),
        ("silhouette_by_k.png", "SW", "Silhouette Width",
         "Silhouette Width (SW)", "больше = лучше"),
        ("calinski_harabasz_by_k.png", "CH", "Calinski–Harabasz",
         "Calinski–Harabasz (CH)", "больше = лучше"),
        ("s_dbw_by_k.png", "S_Dbw", "S_Dbw (Halkidi)",
         "S_Dbw (Halkidi)", "МЕНЬШЕ = лучше"),
        ("s_dbw_normalized_by_k.png", "S_Dbw norm", "(S_Dbw − mean_random) / std_random",
         "S_Dbw относительно случайного baseline", "МЕНЬШЕ = лучше"),
    ]
    for fname, col, ylabel, title, note in figs:
        plot_metric(ks, res[col], ylabel, f"{title}, {MONTH}", FIGURES_DIR / fname, note)

    lines = [
        f"# 10. Выбор k для KMeans ({MONTH})",
        "",
        "Сгенерировано `src/10_kmeans_k_selection.py`. **Итоговое k не выбирается** — "
        "таблица и графики для ручного решения.",
        "",
        f"- Признаки: {len(SHARE_COLUMNS)} долей ({', '.join(SHARE_COLUMNS)}), z-score внутри "
        f"месяца — та же стандартизация, что для экономической сети; МО: **{len(X)}**",
        f"- KMeans: random_state={RANDOM_STATE}, n_init={N_INIT}; k = {K_RANGE.start}..{K_RANGE.stop - 1}",
        f"- S_Dbw: пакет `s-dbw`, {S_DBW_KW}. Центр кластера — ближайшая к среднему точка: "
        "при центре-среднем (как буквально в статье) плотность у центров в 5-мерном "
        "пространстве бывает нулевой, и для k ≥ 9 индекс не определён. Этот вариант "
        "приведён справочно в колонке «S_Dbw (центр = среднее)» (NaN = не определён).",
        f"- **Baseline S_Dbw**: для каждого k — {N_RANDOM} случайных "
        "разбиений тех же МО на k групп (перестановка меток реального разбиения, размеры "
        "кластеров сохраняются), S_Dbw тем же способом; `S_Dbw norm` = (S_Dbw − mean_random) / "
        "std_random — чем отрицательнее, тем сильнее реальное разбиение лучше случайного. "
        f"rng seed = {BASELINE_SEED} + k. 95% ДИ для S_Dbw norm — бутстреп ({N_BOOT} повторов) по "
        f"{N_RANDOM} случайным значениям (неопределённость оценки mean/std baseline). "
        "Принцип (нормировать индексы, чьи случайные уровни зависят от K, на среднее и дисперсию случайного разбиения) рекомендован в Shalileh et al. (2025); сама процедура в статье не описана, авторы откладывают её. Перестановка меток с сохранением размеров кластеров и бутстреп-ДИ — наши.",
        "- KMeans и все метрики — в евклидовой геометрии z-score признаков "
        "(экономическая сеть строилась по косинусу тех же признаков).",
        "",
        "Направление метрик: inertia — ищем «локоть»; **SW ↑, CH ↑ — больше лучше; "
        "S_Dbw ↓ — меньше лучше.**",
        "",
        res.to_markdown(index=False, floatfmt=",.4f"),
        "",
        f"Таблица также сохранена: `{METRICS_PATH.relative_to(PROJECT_DIR)}`",
        "",
        "## Графики",
        "",
    ]
    for fname, _, _, title, note in figs:
        lines += [f"### {title} ({note})", "", f"![{title}](figures/{fname})", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
