"""Общие функции проверки гипотез о типах МО (шаги 18a и 19).

Краскел–Уоллис с поправкой на связки и мера эффекта η²_H = max(0, (H − k_eff + 1) / (n − k_eff));
Спирмен ρ = Пирсон на рангах. Нулевые распределения — перестановки меток типов (или одной из
переменных); ранги считаются один раз, суммы рангов по группам — через bincount, перестановки — батчами.
Уровни: «общий» и «внутри регионов» (значения заменяются нормированным рангом внутри региона,
перестановки — только внутри регионов, регионы с малым числом МО с данными не участвуют).
"""
import numpy as np
from scipy.stats import rankdata

LEVEL_ALL = "общий"
LEVEL_REGION = "внутри регионов"
LEVELS = [LEVEL_ALL, LEVEL_REGION]


def tie_factor(values: np.ndarray) -> float:
    """1 − Σ(t³ − t) / (n³ − n) по группам совпадающих значений."""
    n = len(values)
    _, t = np.unique(values, return_counts=True)
    return 1.0 - float(((t.astype(float) ** 3 - t).sum()) / (n ** 3 - n))


def within_region_rank(y: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Нормированный ранг внутри региона: (ранг − 0.5) / n_региона, ранги средние при связках."""
    out = np.empty(len(y), dtype=float)
    for r in np.unique(region):
        m = region == r
        out[m] = (rankdata(y[m]) - 0.5) / m.sum()
    return out


def prepare(y, labels, region, level: str, min_group: int, min_types: int, min_region_n: int):
    """Отбор МО для одного теста. y — np.ndarray (NaN = нет данных), labels — метки типов или None
    (для корреляций y — матрица n×2). Возвращает dict с отобранными значениями и сведениями для status."""
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y).any(axis=1) if y.ndim == 2 else ~np.isnan(y)
    excluded_types, excluded_regions = [], []
    while True:
        changed = False
        if level == LEVEL_REGION:
            reg, cnt = np.unique(region[ok], return_counts=True)
            small = reg[cnt < min_region_n]
            if len(small):
                excluded_regions += small.tolist()
                ok &= ~np.isin(region, small)
                changed = True
        if labels is not None:
            typ, cnt = np.unique(labels[ok], return_counts=True)
            small = typ[cnt < min_group]
            if len(small):
                excluded_types += small.tolist()
                ok &= ~np.isin(labels, small)
                changed = True
        if not changed:
            break
    res = {"mask": ok, "n": int(ok.sum()), "excluded_types": sorted(set(excluded_types)),
           "excluded_regions": sorted(set(excluded_regions))}
    yy = y[ok]
    rr = region[ok] if region is not None else None
    if level == LEVEL_REGION:
        yy = (np.column_stack([within_region_rank(yy[:, j], rr) for j in range(yy.shape[1])])
              if yy.ndim == 2 else within_region_rank(yy, rr))
    res["y"], res["region"] = yy, rr
    if labels is not None:
        lab = labels[ok]
        res["labels"] = lab
        res["k_eff"] = int(len(np.unique(lab)))
        res["min_group"] = int(np.unique(lab, return_counts=True)[1].min()) if len(lab) else 0
        res["status"] = ("не оценивается" if res["k_eff"] < min_types
                         else f"исключены типы {res['excluded_types']}" if res["excluded_types"] else "ok")
    return res


def _dense(codes: np.ndarray) -> np.ndarray:
    return np.unique(codes, return_inverse=True)[1]


def perm_index(rng: np.random.Generator, n_rows: int, n: int, region_dense=None) -> np.ndarray:
    """n_rows перестановок индексов 0..n−1: полностью случайных или только внутри регионов."""
    if region_dense is None:
        return rng.permuted(np.tile(np.arange(n), (n_rows, 1)), axis=1)
    base = np.argsort(region_dense, kind="stable")
    keys = rng.random((n_rows, n)) + 2.0 * region_dense[base][None, :]
    order = np.argsort(keys, axis=1)
    idx = np.empty((n_rows, n), dtype=np.int64)
    idx[:, base] = base[order]
    return idx


class KW:
    """Краскел–Уоллис для фиксированных значений: ранги и поправка на связки считаются один раз."""

    def __init__(self, y: np.ndarray, labels: np.ndarray):
        self.n = len(y)
        self.r = rankdata(y)
        self.codes = _dense(labels)
        self.k = int(self.codes.max()) + 1
        self.counts = np.bincount(self.codes, minlength=self.k).astype(float)
        self.c = tie_factor(y)

    def h_batch(self, codes_batch: np.ndarray) -> np.ndarray:
        b = codes_batch.shape[0]
        idx = codes_batch + self.k * np.arange(b)[:, None]
        s = np.bincount(idx.ravel(), weights=np.broadcast_to(self.r, codes_batch.shape).ravel(),
                        minlength=b * self.k).reshape(b, self.k)
        h = 12.0 / (self.n * (self.n + 1)) * (s ** 2 / self.counts).sum(axis=1) - 3.0 * (self.n + 1)
        return h / self.c

    def eta2(self, h: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, (h - self.k + 1) / (self.n - self.k))

    def observed(self) -> float:
        return float(self.eta2(self.h_batch(self.codes[None, :]))[0])

    def null(self, n_perm: int, seed: int, batch: int, region=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        reg = _dense(region) if region is not None else None
        out = []
        for start in range(0, n_perm, batch):
            idx = perm_index(rng, min(batch, n_perm - start), self.n, reg)
            out.append(self.eta2(self.h_batch(self.codes[idx])))
        return np.concatenate(out)


def kw_eta2(y: np.ndarray, labels: np.ndarray) -> tuple:
    """(η²_H, H, k_eff) для одной переменной."""
    kw = KW(np.asarray(y, float), np.asarray(labels))
    h = float(kw.h_batch(kw.codes[None, :])[0])
    return float(kw.eta2(np.array([h]))[0]), h, kw.k


class Spearman:
    """ρ = Пирсон на рангах (на уровне «внутри регионов» — на нормированных рангах внутри региона)."""

    def __init__(self, x: np.ndarray, y: np.ndarray, ranked: bool = False):
        rx = x if ranked else rankdata(x)
        ry = y if ranked else rankdata(y)
        self.xc = rx - rx.mean()
        self.yc = ry - ry.mean()
        self.den = np.sqrt((self.xc ** 2).sum() * (self.yc ** 2).sum())
        self.n = len(x)

    def observed(self) -> float:
        return float(self.xc @ self.yc / self.den)

    def null(self, n_perm: int, seed: int, batch: int, region=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        reg = _dense(region) if region is not None else None
        out = []
        for start in range(0, n_perm, batch):
            idx = perm_index(rng, min(batch, n_perm - start), self.n, reg)
            out.append(self.xc[idx] @ self.yc / self.den)
        return np.concatenate(out)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    return Spearman(np.asarray(x, float), np.asarray(y, float)).observed()


def p_upper(obs: float, null: np.ndarray) -> float:
    """p = (1 + #{нуль ≥ obs}) / (1 + N_PERM)."""
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


def verdict_kw(obs_dec: float, null_dec: np.ndarray, obs_months, p95_months, alpha: float, months_rule: int) -> dict:
    """«подтверждено», если p_dec < alpha и obs_m > p95_m не менее чем в months_rule месяцах."""
    p_dec = p_upper(obs_dec, null_dec)
    months = int(np.sum(np.asarray(obs_months) > np.asarray(p95_months)))
    ok = p_dec < alpha and months >= months_rule
    return {"verdict": "подтверждено" if ok else "не подтверждено", "p_dec": p_dec, "months_above_p95": months}


# --- Шаг 18b: Г6 (переходы в ближайшие типы) и Г9 (переходы и общие сдвиги долей) ---

def nearest_types(dist: np.ndarray, n_nearest: int) -> dict:
    """Для каждого типа i — n_nearest других типов с наименьшим расстоянием между центроидами."""
    out = {}
    for i in range(dist.shape[0]):
        order = [j for j in np.argsort(dist[i], kind="stable") if j != i]
        out[i] = set(int(j) for j in order[:n_nearest])
    return out


def share_to_nearest(from_types: np.ndarray, to_types: np.ndarray, nearest: dict) -> float:
    """S = доля переходов, у которых конечный тип входит в ближайшие к исходному."""
    hits = np.array([t in nearest[f] for f, t in zip(from_types, to_types)])
    return float(hits.mean())


def g6_target_probs(to_types: np.ndarray, n_types: int) -> np.ndarray:
    """p[i, j] — вероятность конечного типа j для исходного i: частота j среди конечных типов всех
    переходов, без j = i, с нормировкой по строке."""
    freq = np.bincount(to_types, minlength=n_types).astype(float)
    p = np.tile(freq, (n_types, 1))
    np.fill_diagonal(p, 0.0)
    return p / p.sum(axis=1, keepdims=True)


def g6_null(from_types: np.ndarray, probs: np.ndarray, nearest: dict, n_perm: int, seed: int) -> tuple:
    """Нуль S: каждому переходу (исходный тип сохраняется) конечный тип назначается случайно с
    вероятностями probs[i]. Попадание в ближайшие для перехода из i — событие с вероятностью
    q_i = Σ_{j ∈ nearest(i)} probs[i, j], поэтому число попаданий среди n_i переходов из i имеет
    распределение Binomial(n_i, q_i) — это точная запись того же нуля.
    Возвращает (вектор S длины n_perm, q по исходным типам, n по исходным типам)."""
    rng = np.random.default_rng(seed)
    k = probs.shape[0]
    n_from = np.bincount(from_types, minlength=k)
    q = np.array([probs[i, sorted(nearest[i])].sum() for i in range(k)])
    hits = np.zeros(n_perm)
    for i in range(k):
        if n_from[i]:
            hits += rng.binomial(n_from[i], q[i], size=n_perm)
    return hits / n_from.sum(), q, n_from


def g6_simulate_targets(from_types: np.ndarray, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Одна симуляция из нуля Г6 напрямую: конечный тип каждого перехода выбирается по probs[i]."""
    to = np.empty(len(from_types), dtype=int)
    for i in np.unique(from_types):
        m = from_types == i
        to[m] = rng.choice(probs.shape[0], size=int(m.sum()), p=probs[i])
    return to


def spearman_cyclic_null(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ρ Спирмена для всех ненулевых циклических сдвигов y относительно x (n − 1 значений)."""
    return np.array([spearman_rho(x, np.roll(y, s)) for s in range(1, len(y))])


# --- Шаг 19b: точный p для Г6 ---

def binomial_sum_sf(n: np.ndarray, q: np.ndarray, x_obs: int) -> float:
    """P(X ≥ x_obs), X = сумма независимых Binomial(n_i, q_i) (свёртка распределений)."""
    from scipy.stats import binom
    pmf = np.array([1.0])
    for ni, qi in zip(n, q):
        pmf = np.convolve(pmf, binom.pmf(np.arange(int(ni) + 1), int(ni), qi))
    return float(min(1.0, pmf[int(x_obs):].sum()))


# --- Шаг 21: остаток на рангах (как для Г7 в шаге 18) ---

def rank_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Ранги y минус линейная регрессия (МНК) на ранги x, по МО, где есть обе величины; иначе NaN."""
    m = ~np.isnan(y) & ~np.isnan(x)
    ry, rx = rankdata(y[m]), rankdata(x[m])
    slope, intercept = np.polyfit(rx, ry, 1)
    out = np.full(len(y), np.nan)
    out[m] = ry - (intercept + slope * rx)
    return out
