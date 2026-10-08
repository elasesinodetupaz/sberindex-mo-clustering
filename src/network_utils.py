"""Общие функции построения экономической сети (используются шагами 05, 06)."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

SHARE_COLUMNS = ["share_Продовольствие", "share_Здоровье", "share_Общепит",
                 "share_Транспорт", "share_Маркетплейсы"]


def knn_pairs(score: np.ndarray, k: int) -> pd.DataFrame:
    """Топ-k по score (больше = ближе) для каждого узла; пары с score = -inf не берутся.

    Возвращает неориентированные пары индексов a < b и флаг mutual
    (True, если каждый из пары входит в топ-k другого).
    """
    s = score.copy()
    np.fill_diagonal(s, -np.inf)
    nbr = np.argsort(-s, axis=1, kind="stable")[:, :k]
    src = np.repeat(np.arange(len(s)), nbr.shape[1])
    dst = nbr.ravel()
    finite = np.isfinite(s[src, dst])
    src, dst = src[finite], dst[finite]
    a, b = np.minimum(src, dst), np.maximum(src, dst)
    counts = pd.DataFrame({"a": a, "b": b}).value_counts().rename("n_dir").reset_index()
    counts["mutual"] = counts.pop("n_dir").to_numpy() == 2
    return counts


def knn_edges(ids: np.ndarray, sim: np.ndarray, k: int) -> pd.DataFrame:
    """Топ-k соседей по сходству, объединение x→y и y→x в неориентированные рёбра."""
    p = knn_pairs(sim, k)
    return pd.DataFrame({
        "territory_id_x": ids[p["a"]],
        "territory_id_y": ids[p["b"]],
        "similarity": sim[p["a"], p["b"]],
        "mutual": p["mutual"].to_numpy(),
    }).sort_values(["territory_id_x", "territory_id_y"], ignore_index=True)


def knn_edges_by_distance(ids: np.ndarray, dist: np.ndarray, k: int) -> pd.DataFrame:
    """Топ-k ближайших по расстоянию (NaN = нет связи), неориентированные рёбра."""
    score = np.where(np.isnan(dist), -np.inf, -dist)
    p = knn_pairs(score, k)
    return pd.DataFrame({
        "territory_id_x": ids[p["a"]],
        "territory_id_y": ids[p["b"]],
        "distance": dist[p["a"], p["b"]],
        "mutual": p["mutual"].to_numpy(),
    }).sort_values(["territory_id_x", "territory_id_y"], ignore_index=True)


def standardize_shares(X_raw: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """z-score каждой доли по всем МО переданного среза (обычно — одного месяца)."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_raw), scaler


def month_shares(shares: pd.DataFrame, month: str) -> pd.DataFrame:
    """Строки category_shares за месяц 'YYYY-MM', отсортированные по territory_id."""
    m = shares[shares["date"] == pd.Timestamp(month)].sort_values("territory_id")
    if m.empty or m["territory_id"].duplicated().any():
        raise SystemExit(f"STOP: за {month} нет данных или дубликаты territory_id")
    return m


def std_cosine_similarity(X_raw: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """z-score по каждому признаку, затем попарный косинус. STOP при нулевом z-векторе."""
    X, scaler = standardize_shares(X_raw)
    zero_norm = int((np.linalg.norm(X, axis=1) < 1e-12).sum())
    if zero_norm:
        raise SystemExit(f"STOP: {zero_norm} МО с нулевым z-вектором — косинус не определён")
    return cosine_similarity(X), scaler
