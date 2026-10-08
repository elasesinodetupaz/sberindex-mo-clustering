"""Шаг 32. Правила построения рёбер сети: структура сети и результаты кластеризации (декабрь 2024, описательно).

Правила (все на 2004 МО выборки, кроме R6): R1 канон (kNN8 по косинусу z-оценок декабря), R2 порог по сырому косинусу
(в репозитории нет функции — пропущено), R3 kNN8 по корреляции рядов (120 значений), R4 kNN8 по максимальной лаговой
корреляции (лаги -2..+2), R5 kNN8 по расстоянию highway, R6 kNN8 по расстоянию railway с достройкой нулей (шаг 07; сеть
построена на 1260 МО с железной дорогой). Новых методов кластеризации нет: Louvain с параметрами канона (resolution, seed
из config) и готовые индексы (SW, CH, S_Dbw — sklearn и пакет s-dbw с config.SDBW_KW; MQ, AVI, AVU — src/partition_metrics.py).
Все контроли выполняются до записи; при нарушении код выхода 1 и ничего не записывается.

Вход:  data/processed/{category_shares, kmeans_labels_final, louvain_labels_2024_12_seed42, louvain_labels_2024_12,
       transport_network_{highway,railway}_knn8, connection_valid}, economic_networks/2024-12_std_knn8.parquet,
       notebooks/{05, 07, 08, 10, 11, 11b, 13}*.md (напечатанные значения для контролей)
Выход: data/processed/edge_rules_32.parquet, edge_rules_partitions_32.parquet, notebooks/32_edge_rules.md,
       notebooks/figures/32_edge_rules.{png,svg}
Запуск из корня проекта:  .venv/bin/python src/32_edge_rules.py
"""
import hashlib
import importlib
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from s_dbw import S_Dbw
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, normalized_mutual_info_score,
                             silhouette_score)

import config as C
from network_utils import SHARE_COLUMNS, knn_edges, knn_edges_by_distance, month_shares, standardize_shares, std_cosine_similarity
from partition_metrics import partition_metrics

PROJECT_DIR = Path(__file__).resolve().parents[1]
S = C.STEP32
P = lambda rel: PROJECT_DIR / rel  # noqa: E731
CONTROLS: list[tuple[str, str, str, bool]] = []
SRC = "расчёт шага 32"


def stop(msg: str) -> None:
    print(f"STOP: {msg}")
    sys.exit(1)


def ctl(name: str, got, expected, ok: bool | None = None) -> None:
    """Запись контроля; при расхождении STOP (до записи любых файлов)."""
    ok = (got == expected) if ok is None else ok
    sh = lambda v: str(v) if len(str(v)) <= 150 else str(v)[:150] + "…"  # noqa: E731
    CONTROLS.append((name, sh(got), sh(expected), bool(ok)))
    print(f"[{'ok' if ok else 'НАРУШЕН'}] {name}: {sh(got)} (ожидалось {sh(expected)})")
    if not ok:
        stop(f"{name}: получено {sh(got)}, ожидалось {sh(expected)}")


def need(rel: str) -> Path:
    if not P(rel).exists():
        stop(f"нет файла {rel}: предыдущий шаг не выполнен")
    return P(rel)


# ============================================================================ напечатанные значения notebooks
def num_tol(s: str) -> float:
    """Допуск: половина последней напечатанной цифры."""
    s = s.replace(",", "")
    m = re.search(r"\.(\d+)$", s)
    return 0.5 * 10 ** -(len(m.group(1)) if m else 0)


def fnum(s: str) -> float:
    return float(s.replace(",", ""))


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def find_line(rel: str, pattern: str, start: int = 0) -> tuple[int, str]:
    lines = need(rel).read_text(encoding="utf-8").split("\n")
    for i in range(start, len(lines)):
        if re.search(pattern, lines[i]):
            return i + 1, lines[i]
    stop(f"в {rel} нет строки по шаблону {pattern}")


CMP_ROWS: list[list[str]] = []   # файл и строка / значение в notebooks / значение в новом расчёте / совпало


def compare(rel: str, ln: int, label: str, printed_s: str, got: float, tol: float | None = None) -> None:
    """Сверка с напечатанным: допуск — половина последней напечатанной цифры."""
    v = fnum(printed_s)
    t = num_tol(printed_s) if tol is None else tol
    ok = abs(got - v) <= t + 1e-12
    CMP_ROWS.append([f"{rel}, строка {ln}", label, printed_s, f"{got:.6g}", "да" if ok else "НЕТ"])
    ctl(f"{label}: {rel} строка {ln} (допуск {t:g})", round(float(got), 6), v, ok)


# ============================================================================ данные и правила
def load_series() -> dict:
    shares = pd.read_parquet(need(S["SHARES_PATH"]))
    dates = sorted(shares["date"].unique())
    ids = None
    Zc = None
    for t, d in enumerate(dates):
        m = month_shares(shares, pd.Timestamp(d).strftime("%Y-%m"))
        if ids is None:
            ids = m["territory_id"].to_numpy()
            Zc = np.zeros((len(SHARE_COLUMNS), len(ids), len(dates)))
        elif not np.array_equal(ids, m["territory_id"].to_numpy()):
            stop(f"набор МО в месяце {d} отличается от первого месяца")
        Z, _ = standardize_shares(m[SHARE_COLUMNS].to_numpy())
        Zc[:, :, t] = Z.T
    raw_dec = month_shares(shares, C.MONTH)[SHARE_COLUMNS].to_numpy()
    return {"ids": ids, "Zc": Zc, "dates": dates, "raw_dec": raw_dec, "n_rows": len(shares)}


def row_normalize(M: np.ndarray) -> np.ndarray:
    M = M - M.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    if (nrm < 1e-12).any():
        stop("ряд с нулевой дисперсией: корреляция не определена")
    return M / nrm


def clip_corr(W: np.ndarray, name: str) -> np.ndarray:
    tol = S["WEIGHT_TOL"]
    lo, hi = float(np.nanmin(W)), float(np.nanmax(W))
    if lo < -1 - tol or hi > 1 + tol or np.isnan(W).any():
        stop(f"{name}: коэффициенты корреляции вне [-1, 1] или NaN (min {lo}, max {hi})")
    return np.clip(W, -1.0, 1.0)


def corr_matrix_r3(Zc: np.ndarray) -> np.ndarray:
    X = np.concatenate([Zc[c] for c in range(Zc.shape[0])], axis=1)   # N × (5·24)
    Xn = row_normalize(X)
    R = Xn @ Xn.T
    return clip_corr((R + R.T) / 2, "R3")


def corr_matrix_r4(Zc: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """w_ij = max по l от -2 до +2 среднего по категориям rho_ij(l); rho_ij(l) = corr(x_i(t), x_j(t + l)) на пересечении месяцев."""
    T = Zc.shape[2]
    lags = S["LAGS"]
    if sorted(lags) != [-l for l in sorted(lags, reverse=True)] or 0 not in lags:
        stop("LAGS должны быть симметричны и содержать 0")
    pos_lags = [l for l in lags if l > 0]
    months = {l: T - abs(l) for l in lags}
    R: dict[int, np.ndarray] = {}
    R0 = np.mean([row_normalize(Zc[c]) @ row_normalize(Zc[c]).T for c in range(Zc.shape[0])], axis=0)
    R[0] = (R0 + R0.T) / 2
    for l in pos_lags:
        acc = np.zeros_like(R0)
        for c in range(Zc.shape[0]):
            A = row_normalize(Zc[c][:, :T - l])    # x_i(t)
            B = row_normalize(Zc[c][:, l:])        # x_j(t + l)
            acc += A @ B.T
        R[l] = acc / Zc.shape[0]
    stack = np.stack([R[l] if l >= 0 else R[-l].T for l in lags])    # rho_ij(l) для l = lags
    W = stack.max(axis=0)
    best = np.asarray(lags)[stack.argmax(axis=0)]
    return clip_corr(W, "R4"), best, months


def edges_std(df: pd.DataFrame, wcol: str | None) -> pd.DataFrame:
    out = pd.DataFrame({"x": df["territory_id_x"].astype(int), "y": df["territory_id_y"].astype(int),
                        "w": df[wcol].astype(float) if wcol else 1.0, "mutual": df["mutual"].astype(bool)})
    return out.sort_values(["x", "y"], ignore_index=True)


def adjacency_sha(ids: np.ndarray, e: pd.DataFrame) -> str:
    pos = pd.Series(np.arange(len(ids)), index=ids)
    A = np.zeros((len(ids), len(ids)))
    i, j = pos[e["x"]].to_numpy(), pos[e["y"]].to_numpy()
    A[i, j] = A[j, i] = e["w"].to_numpy()
    return hashlib.sha256(A.tobytes()).hexdigest()


def build_all(D: dict) -> dict:
    """Рёбра R1, R3, R4 (по ids выборки) и вспомогательные матрицы; вызывается дважды для проверки повторяемости."""
    ids, Zc, K = D["ids"], D["Zc"], S["K"]
    sim, _ = std_cosine_similarity(D["raw_dec"])
    e1 = edges_std(knn_edges(ids, sim, K), "similarity")
    R3 = corr_matrix_r3(Zc)
    e3 = edges_std(knn_edges(ids, R3, K), "similarity")
    W4, best, months = corr_matrix_r4(Zc)
    e4 = edges_std(knn_edges(ids, W4, K), "similarity")
    return {"R1": e1, "R3": e3, "R4": e4, "best": best, "W4": W4, "months": months}


def graph_from(e: pd.DataFrame, unit: bool = False) -> nx.Graph:
    """Граф строится из рёбер, отсортированных по (x, y), как в шаге 11 (порядок узлов влияет на Louvain)."""
    G = nx.Graph()
    G.add_weighted_edges_from(((int(x), int(y), 1.0 if unit else float(w)) for x, y, w in e[["x", "y", "w"]].itertuples(index=False)),
                              weight="weight")
    return G


def louvain_labels(G: nx.Graph, seed: int) -> dict:
    """Те же параметры и порядок нумерации, что в шаге 11 (src/11_louvain.py)."""
    comms = nx.community.louvain_communities(G, weight="weight", resolution=C.LOUVAIN_RESOLUTION, seed=seed)
    comms = sorted(comms, key=lambda c: (-len(c), min(c)))
    return {n: i for i, c in enumerate(comms) for n in c}


# ============================================================================ показатели
def lab_arr(lab: dict, ids) -> np.ndarray:
    return np.array([lab[int(i)] for i in ids])


def sizes_str(labels) -> str:
    return ", ".join(str(int(x)) for x in sorted(np.bincount(pd.factorize(pd.Series(labels))[0]).tolist(), reverse=True))


def structure(G: nx.Graph, e: pd.DataFrame, types: dict, km_metrics: dict) -> dict:
    N, M = G.number_of_nodes(), G.number_of_edges()
    deg = np.array([d for _, d in G.degree()], float)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    nx.set_node_attributes(G, {n: types[n] for n in G}, "type")
    inside = float(np.mean([types[u] == types[v] for u, v in G.edges()]))
    return {"узлов": N, "рёбер": M, "плотность": 2 * M / (N * (N - 1)), "степень, среднее": float(deg.mean()),
            "степень, ст. откл.": float(deg.std()), "компонент связности": len(comps),
            "доля узлов в наибольшей компоненте": len(comps[0]) / N,
            "доля взаимных рёбер": float(e["mutual"].mean()),
            "средний коэффициент кластеризации": float(nx.average_clustering(G)),
            "ассортативность по типам": float(nx.attribute_assortativity_coefficient(G, "type")),
            "доля рёбер внутри типа": inside, "модулярность типов": float(km_metrics["MQ"])}


def icvi(X: np.ndarray, G: nx.Graph, ids, labels, notes: list, tag: str) -> dict:
    """SW, CH, S_Dbw в атрибутивном пространстве A (декабрь 2024) и MQ, AVI, AVU на графе G; недоопределённые — NaN и примечание."""
    vals = {}
    for name, fn in (("SW", lambda: silhouette_score(X, labels)), ("CH", lambda: calinski_harabasz_score(X, labels)),
                     ("S_Dbw", lambda: S_Dbw(X, labels, **C.SDBW_KW))):
        try:
            vals[name] = float(fn())
        except Exception as e:  # noqa: BLE001
            vals[name] = np.nan
            notes.append((tag, name, f"не определён: {type(e).__name__}: {e}"))
    vals.update(net_indices(G, ids, labels, notes, tag))
    return vals


def net_indices(G, ids, labels, notes, tag) -> dict:
    try:
        m, _ = partition_metrics(G, {int(i): int(l) for i, l in zip(ids, labels)}, weight="weight")
        return {k: float(m[k]) for k in ("MQ", "AVI", "AVU")}
    except Exception as e:  # noqa: BLE001
        for k in ("MQ", "AVI", "AVU"):
            notes.append((tag, k, f"не определён: {type(e).__name__}: {e}"))
        return {k: np.nan for k in ("MQ", "AVI", "AVU")}


def evaluate(code: str, G: nx.Graph, e: pd.DataFrame, ids, X, lv_lab: np.ndarray, km: np.ndarray, ref_lv: np.ndarray,
             types: dict, notes: list, tag: str, with_structure: bool = True) -> dict:
    """Все показатели одного правила на одном наборе МО (ids, X, метки согласованы по порядку)."""
    km_m = net_indices(G, ids, km, notes, tag + " KMeans")
    lv_rel = pd.factorize(pd.Series(lv_lab))[0]
    out = {"code": code, "K": int(len(set(lv_lab.tolist()))), "sizes": sizes_str(lv_lab),
           "ARI с KMeans": float(adjusted_rand_score(km, lv_lab)), "NMI с KMeans": float(normalized_mutual_info_score(km, lv_lab)),
           "ARI с Louvain R1": float(adjusted_rand_score(ref_lv, lv_lab)),
           "L": icvi(X, G, ids, lv_rel, notes, tag + " Louvain"), "KM": km_m}
    if with_structure:
        out["struct"] = structure(G, e, types, km_m)
    return out


def rank_codes(vals: dict, codes: list, metric: str) -> dict:
    d = S["RANK_DIRECTIONS"][metric]
    v = np.array([vals[c][metric] for c in codes], float)
    if np.isnan(v).any():
        return {c: np.nan for c in codes}
    sign = 1.0 if d == "меньше лучше" else -1.0
    r = rankdata(np.round(sign * v, 12), method="average")
    return {c: float(x) for c, x in zip(codes, r)}


def metric_value(res: dict, metric: str) -> float:
    return res["L"][metric] if metric in ("SW", "CH", "S_Dbw") else res[metric]


# ============================================================================ основной расчёт
def main() -> None:
    ctl("k во всех kNN: STEP32.K = ECON_KNN_K = TRANSPORT_KNN_K", (S["K"], C.ECON_KNN_K, C.TRANSPORT_KNN_K), (8, 8, 8))
    K = S["K"]
    D = load_series()
    ids, Zc = D["ids"], D["Zc"]
    N = len(ids)
    ctl("category_shares: 2004 МО × 24 месяца = 48096 строк, нет NaN", (N, len(D["dates"]), D["n_rows"], bool(np.isfinite(Zc).all())),
        (S["EXPECT_N_SAMPLE"], 24, 48096, True))
    km_s = pd.read_parquet(need(S["KMEANS_LABELS_PATH"])).set_index("territory_id")["cluster"]
    ctl("выборка: множества МО category_shares и kmeans_labels_final совпадают", sorted(ids.tolist()) == sorted(km_s.index.tolist()), True)
    km = km_s.loc[ids].to_numpy().astype(int)
    types = {int(i): int(t) for i, t in zip(ids, km)}
    X_A, _ = standardize_shares(D["raw_dec"])

    # ---- построение R1, R3, R4 (дважды: повторяемость графа)
    B1 = build_all(D)
    B2 = build_all(D)
    E = {"R1": B1["R1"], "R3": B1["R3"], "R4": B1["R4"]}
    hashes = {}
    for c in ("R1", "R3", "R4"):
        h1, h2 = adjacency_sha(ids, B1[c]), adjacency_sha(ids, B2[c])
        hashes[c] = h1
        ctl(f"{c}: повторное построение даёт идентичный граф (sha256 матрицы смежности)", h1 == h2, True)
    ctl("R4: число доступных месяцев по лагам l = -2..+2", {l: m for l, m in sorted(B1["months"].items())}, {-2: 22, -1: 23, 0: 24, 1: 23, 2: 22})
    ctl("R4: матрица w_ij симметрична (макс. |w − wᵀ|)", float(np.abs(B1["W4"] - B1["W4"].T).max()), 0.0)

    # ---- R1: сверка с сохранённой сетью шага 05
    net = pd.read_parquet(need(S["ECON_NETWORK_PATH"]))
    sv = edges_std(net, "similarity")
    mg = E["R1"].merge(sv, on=["x", "y"], how="outer", suffixes=("", "_s"), indicator=True)
    ctl("R1: рёбра нового расчёта = рёбрам economic_networks/2024-12_std_knn8.parquet (расхождений по составу)", int((mg["_merge"] != "both").sum()), 0)
    ctl("R1: макс. |Δ similarity| с сохранённой сетью", float(np.abs(mg["w"] - mg["w_s"]).max()), 0.0, float(np.abs(mg["w"] - mg["w_s"]).max()) < 1e-12)

    # ---- R5, R6: готовые рёбра шага 07
    hw = pd.read_parquet(need(S["HIGHWAY_PATH"]))
    rw = pd.read_parquet(need(S["RAILWAY_PATH"]))
    E["R5"], E["R6"] = edges_std(hw, None), edges_std(rw, None)
    cv = pd.read_parquet(need(S["CONNECTION_VALID_PATH"]))
    n_imputed = int(cv["imputed"].sum())
    step07 = importlib.import_module("07_transport_networks")
    for code, t, saved in (("R5", "highway", hw), ("R6", "railway", rw)):
        sub = cv[cv["type"] == t]
        nodes, Dm = step07.distance_matrix(sub)
        re_ = knn_edges_by_distance(nodes, Dm, K)
        m_ = re_.merge(saved, on=["territory_id_x", "territory_id_y"], how="outer", indicator=True)
        ctl(f"{code}: рёбра, пересчитанные функцией шага 07 из connection_valid = сохранённым (расхождений по составу)", int((m_["_merge"] != "both").sum()), 0)

    # ---- множества узлов
    nodes_R = {c: np.array(sorted(set(E[c]["x"]) | set(E[c]["y"]))) for c in ("R1", "R3", "R4", "R5", "R6")}
    ctl("R1, R3, R4, R5: узлов 2004 (все МО выборки)", [len(nodes_R[c]) for c in ("R1", "R3", "R4", "R5")], [2004] * 4)
    ctl("R6: узлов в железнодорожной сети шага 07 и они входят в выборку", (len(nodes_R["R6"]), bool(set(nodes_R["R6"]) <= set(ids.tolist()))),
        (S["EXPECT_N_RAILWAY_NODES"], True))
    for c in ("R1", "R3", "R4", "R5", "R6"):
        directed = len(E[c]) + int(E[c]["mutual"].sum())
        ctl(f"{c}: число направленных пар до симметризации = {K} × узлов", directed, K * len(nodes_R[c]))
    for c in ("R3", "R4"):
        w = E[c]["w"]
        ctl(f"{c}: веса рёбер от −1 до 1; все веса положительны (требование Louvain)", (float(w.min()) >= -1.0 and float(w.max()) <= 1.0, float(w.min()) > 0), (True, True))
    ids_set = {c: set(nodes_R[c].tolist()) for c in nodes_R}
    in_rail = np.array([int(i) in ids_set["R6"] for i in ids])
    ids_x = ids[in_rail]

    # ---- сверка R1 с notebooks
    rel05, rel07, rel08, rel10, rel11, rel11b = (S["NB05_PATH"], S["NB07_PATH"], S["NB08_PATH"], S["NB10_PATH"], S["NB11_PATH"], S["NB11B_PATH"])
    G1 = graph_from(E["R1"])
    comps1 = sorted(nx.connected_components(G1), key=len, reverse=True)
    deg1 = np.array([d for _, d in G1.degree()])
    for lab_, pat, val in (("R1: рёбер", r"^\| рёбер\s+\|", len(E["R1"])), ("R1: взаимных рёбер", r"^\| из них взаимных", int(E["R1"]["mutual"].sum())),
                           ("R1: плотность", r"^\| плотность\s+\|", 2 * len(E["R1"]) / (N * (N - 1))),
                           ("R1: компонент связности", r"^\| компонент связности", len(comps1)),
                           ("R1: степень min", r"^\| степень min", int(deg1.min())), ("R1: степень медиана", r"^\| степень медиана", float(np.median(deg1))),
                           ("R1: степень max", r"^\| степень max", int(deg1.max()))):
        ln, t = find_line(rel05, pat)
        compare(rel05, ln, lab_, cells(t)[1], float(val))
    ln, t = find_line(rel11, r"2004 МО, [\d,]+ рёбер")
    compare(rel11, ln, "R1: рёбер (шаг 11)", re.search(r"2004 МО, ([\d,]+) рёбер", t).group(1), float(len(E["R1"])))
    ln, t = find_line(rel11b, r"2004 МО, [\d,]+ рёбер")
    compare(rel11b, ln, "R1: рёбер (шаг 11b)", re.search(r"2004 МО, ([\d,]+) рёбер", t).group(1), float(len(E["R1"])))

    # ---- Louvain на R1 (seed 42) и сверка с шагом 11
    lv = {}
    lv["R1"] = louvain_labels(G1, C.LOUVAIN_SEED)
    saved = pd.read_parquet(need(S["LOUVAIN_STEP11_LABELS_PATH"])).set_index("territory_id")["cluster"].loc[ids].to_numpy()
    ctl("R1: метки Louvain (seed 42) = louvain_labels_2024_12_seed42.parquet, расхождений", int((lab_arr(lv["R1"], ids) != saved).sum()), 0)
    ln, t = find_line(rel11, r"\| Louvain \(resolution=1\.0, seed=42\)")
    c_ = cells(t)
    compare(rel11, ln, "R1: K Louvain (seed 42)", c_[1], float(len(set(lv["R1"].values()))))
    m_lv, _ = partition_metrics(G1, lv["R1"], weight="weight")
    for name, i in (("MQ", 2), ("AVI", 3), ("AVU", 4)):
        compare(rel11, ln, f"R1: {name} Louvain (seed 42)", c_[i], float(m_lv[name]))
    ln, t = find_line(rel11, r"\| KMeans k=6 \(kmeans_labels_final\)")
    c_ = cells(t)
    m_km, _ = partition_metrics(G1, {int(i): int(l) for i, l in zip(ids, km)}, weight="weight")
    for name, i in (("MQ", 2), ("AVI", 3), ("AVU", 4)):
        compare(rel11, ln, f"R1: {name} KMeans k=6", c_[i], float(m_km[name]))
    ln, t = find_line(rel11b, r"\| KMeans k=6 \(kmeans_labels_final\)")
    c_ = cells(t)
    for name, i in (("MQ", 2), ("AVI", 3), ("AVU", 4)):
        compare(rel11b, ln, f"R1: {name} KMeans k=6 (4 знака)", c_[i], float(m_km[name]))
    ln, t = find_line(rel10, r"^\|\s+6\.0000 \|")
    c_ = cells(t)
    kmv = icvi(X_A, G1, ids, km, [], "контроль")
    for name, i in (("SW", 2), ("CH", 3), ("S_Dbw", 4)):
        compare(rel10, ln, f"R1: {name} KMeans k=6 (шаг 10)", c_[i], kmv[name])
    # лучший по MQ из 20 seed (шаг 13)
    best = None
    for sd in range(C.LOUVAIN_MONTHLY_N_SEEDS):
        lb = louvain_labels(G1, sd)
        mq = partition_metrics(G1, lb, weight="weight")[0]
        if best is None or mq["MQ"] > best[1]["MQ"]:
            best = (sd, mq, len(set(lb.values())))
    nb13 = C.STEP31["NB13_PATH"]
    ln, t = find_line(nb13, r"^\| 2024-12 \|\s+\d+ \|\s+\d+ \|\s+[\d.]+ \|")
    c_ = cells(t)
    compare(nb13, ln, f"R1: seed лучшего по MQ из {C.LOUVAIN_MONTHLY_N_SEEDS}", c_[1], float(best[0]), 0.0)
    compare(nb13, ln, "R1: K лучшего запуска", c_[2], float(best[2]), 0.0)
    for name, i in (("MQ", 3), ("AVI", 4), ("AVU", 5)):
        compare(nb13, ln, f"R1: {name} лучшего запуска", c_[i], float(best[1][name]))
    # ARI итогового Louvain 11b с KMeans (в notebooks напечатан для итогового запуска, не для seed 42)
    fin = pd.read_parquet(need(S["LOUVAIN_FINAL_LABELS_PATH"])).set_index("territory_id")["cluster"].loc[ids].to_numpy()
    ln, t = find_line(rel11b, r"ARI\(итоговый Louvain, KMeans k=6\) = \*\*[\d.]+\*\*")
    compare(rel11b, ln, "R1: ARI итогового Louvain 11b с KMeans (справочно; для seed 42 в notebooks не напечатан)",
            re.search(r"\*\*([\d.]+)\*\*", t).group(1), float(adjusted_rand_score(km, fin)))

    # ---- сверка R5, R6 с notebooks/07
    ln0, _ = find_line(rel07, r"^## kNN-графы")

    def nb07_row(label: str) -> tuple[int, list[str]]:
        ln, t = find_line(rel07, r"^\| " + re.escape(label) + r"\s+\|", ln0)
        return ln, cells(t)

    G5, G6 = graph_from(E["R5"]), graph_from(E["R6"])
    for code, G, col in (("R5", G5, 1), ("R6", G6, 3)):
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        deg = np.array([d for _, d in G.degree()], float)
        for lab_, row, val in (("МО выборки в сети", "МО выборки в сети", G.number_of_nodes()), ("рёбер", "рёбер", G.number_of_edges()),
                               ("взаимных рёбер", "взаимных (mutual)", int(E[code]["mutual"].sum())),
                               ("изолированных МО", "изолированных МО", nx.number_of_isolates(G)), ("компонент связности", "компонент связности", len(comps)),
                               ("крупнейшая компонента", "крупнейшая компонента", len(comps[0])), ("вторая компонента", "вторая компонента", len(comps[1])),
                               ("степень min", "степень min", int(deg.min())), ("степень медиана", "степень медиана", float(np.median(deg))),
                               ("степень mean", "степень mean", float(deg.mean())), ("степень max", "степень max", int(deg.max())),
                               ("рёбер с distance = 0", "рёбер с distance = 0", int(((hw if code == "R5" else rw)["distance"] == 0).sum()))):
            ln, cs = nb07_row(row)
            compare(rel07, ln, f"{code}: {lab_}", cs[col], float(val))
    ln, cs = nb07_row("рёбер")
    rb = pd.read_parquet(need(S["RAILWAY_BEFORE_PATH"]))
    compare(rel07, ln, "R6: рёбер до достройки (справочно)", cs[2], float(len(rb)))
    ln, t = find_line(rel07, r"^- достроено: \*\*\d+\*\*")
    compare(rel07, ln, "R6: достроено пар (notebooks/07)", re.search(r"\*\*(\d+)\*\*", t).group(1), float(n_imputed), 0.0)
    ctl("R6: достроено пар в connection_valid.parquet (imputed) = ожидание config", n_imputed, S["EXPECT_IMPUTED_RAILWAY_PAIRS"])
    ctl("R6: достройка включена в config (TRANSPORT_IMPUTE_MISSING_RAILWAY_ZERO)", C.TRANSPORT_IMPUTE_MISSING_RAILWAY_ZERO, True)
    ln8, t8 = find_line(rel08, r"с достройкой \d+ пар")
    n08 = int(re.search(r"с достройкой (\d+) пар", t8).group(1))
    ctl(f"notebooks/08 строка {ln8}: напечатано пар достройки (расхождение с notebooks/07 фиксируется, файлы не меняются)", n08, S["EXPECT_NB08_IMPUTED_PRINTED"])
    ctl("расхождение достройки: notebooks/07 и данные 100, notebooks/08 95", (n_imputed, n08), (100, 95))

    # ---- Louvain и разбиения по остальным правилам; взвешенные (основные) и невзвешенные (чувствительность)
    graphs = {"R1": G1, "R3": graph_from(E["R3"]), "R4": graph_from(E["R4"]), "R5": G5, "R6": G6}
    graphs_u = {c: graph_from(E[c], unit=True) for c in graphs}
    lv_u = {}
    for c in graphs:
        if c != "R1":
            lv[c] = louvain_labels(graphs[c], C.LOUVAIN_SEED)
        lv_u[c] = louvain_labels(graphs_u[c], C.LOUVAIN_SEED)
    codes = ["R1", "R3", "R4", "R5", "R6"]
    for c in codes:
        ctl(f"{c}: метки Louvain (взвешенные и вес 1) целые, по одной на узел", (len(lv[c]), len(lv_u[c]), len(graphs[c])), (len(nodes_R[c]),) * 3)
    ctl("R5 и R6: основной вариант (вес 1) совпадает с вариантом «вес 1» (расхождений меток)",
        [int(sum(lv[c][n] != lv_u[c][n] for n in lv[c])) for c in ("R5", "R6")], [0, 0])

    # ---- показатели: основная таблица (R1, R3, R4, R5 на 2004 МО, R6 на 1260 МО)
    notes: list = []
    ref_full = lab_arr(lv["R1"], ids)
    ref_x = lab_arr(lv["R1"], ids_x)
    km_x = km[in_rail]
    types_x = {int(i): int(t) for i, t in zip(ids_x, km_x)}
    X_x = X_A[in_rail]
    main_res, res1260, res_u, ari_wu = {}, {}, {}, {}
    for c in codes:
        if c != "R6":
            main_res[c] = evaluate(c, graphs[c], E[c], ids, X_A, lab_arr(lv[c], ids), km, ref_full, types, notes, f"{c} основная")
        else:
            main_res[c] = evaluate(c, graphs[c], E[c], ids_x, X_x, lab_arr(lv[c], ids_x), km_x, ref_x, types_x, notes, "R6 основная")
    undefined_main = list(notes)
    ctl("неопределённых индексов в основной таблице", len(undefined_main), 0)
    notes_extra: list = []
    for c in codes:
        if c == "R6":
            res1260[c] = main_res[c]
            continue
        sub_nodes = [int(i) for i in ids_x]
        Gs = graphs[c].subgraph(sub_nodes).copy()
        es = E[c][E[c]["x"].isin(ids_set["R6"]) & E[c]["y"].isin(ids_set["R6"])]
        res1260[c] = evaluate(c, Gs, es, ids_x, X_x, lab_arr(lv[c], ids_x), km_x, ref_x, types_x, notes_extra, f"{c} пересечение 1260")
    for c in codes:
        if c != "R6":
            res_u[c] = evaluate(c, graphs_u[c], E[c], ids, X_A, lab_arr(lv_u[c], ids), km, lab_arr(lv_u["R1"], ids), types, notes_extra, f"{c} вес 1",
                                with_structure=False)
            ari_wu[c] = float(adjusted_rand_score(lab_arr(lv[c], ids), lab_arr(lv_u[c], ids)))
        else:
            res_u[c] = evaluate(c, graphs_u[c], E[c], ids_x, X_x, lab_arr(lv_u[c], ids_x), km_x, lab_arr(lv_u["R1"], ids_x), types_x, notes_extra, "R6 вес 1",
                                with_structure=False)
            ari_wu[c] = float(adjusted_rand_score(lab_arr(lv[c], ids_x), lab_arr(lv_u[c], ids_x)))
    ctl("ARI с KMeans от −1 до 1, NMI от 0 до 1 и K ≥ 2 для всех правил", all(-1 <= r["ARI с KMeans"] <= 1 and 0 <= r["NMI с KMeans"] <= 1 and r["K"] >= 2
                                                                           for t in (main_res, res1260, res_u) for r in t.values()), True)
    ctl("R1: ARI Louvain R1 с самим собой = 1", main_res["R1"]["ARI с Louvain R1"], 1.0)
    # распределение 744 МО вне R6 по типам
    out_types = pd.Series(km[~in_rail]).value_counts().sort_index()
    in_types = pd.Series(km[in_rail]).value_counts().sort_index()
    tt = pd.DataFrame({"в R6": in_types, "вне R6": out_types}).fillna(0).astype(int)
    ctl("МО вне R6: 744, по типам сумма равна 744; в R6 + вне R6 = 2004", (int(tt["вне R6"].sum()), int(tt.sum().sum())), (N - S["EXPECT_N_RAILWAY_NODES"], N))
    # лаги R4: доля рёбер с лучшим лагом ≠ 0
    bl = B1["best"]
    pos = pd.Series(np.arange(N), index=ids)
    e4 = E["R4"]
    lag_edge = bl[pos[e4["x"]].to_numpy(), pos[e4["y"]].to_numpy()]
    lag_share = {int(l): float(np.mean(lag_edge == l)) for l in S["LAGS"]}
    lag_all = {int(l): float(np.mean(bl[np.triu_indices(N, 1)] == l)) for l in S["LAGS"]}
    # интервалы весов рёбер
    wr = {c: (float(E[c]["w"].min()), float(E[c]["w"].max())) for c in ("R1", "R3", "R4")}

    # ---- ранги
    rank_all, rank_1260 = {}, {}
    for met in S["RANK_METRICS"]:
        tab_main = {c: {met: metric_value(main_res[c], met)} for c in codes}
        tab_1260 = {c: {met: metric_value(res1260[c], met)} for c in codes}
        rank_all[met] = rank_codes(tab_main, ["R1", "R3", "R4", "R5"], met)
        rank_1260[met] = rank_codes(tab_1260, codes, met)

    # ---- выход: таблицы длинного формата
    rows = []

    def add(table, code, metric, value, direction, source, nodes, weight, rank=np.nan, note=""):
        rows.append({"таблица": table, "правило": code, "показатель": metric, "значение": float(value) if value == value else np.nan,
                     "направление": direction, "источник": source, "узлов": int(nodes), "вес": weight, "ранг": rank, "примечание": note})

    wname = {"R1": "similarity", "R3": "ρ", "R4": "max ρ по лагам", "R5": "1", "R6": "1"}
    for tab_name, res, rk, nn in (("основная", main_res, rank_all, None), ("пересечение 1260 МО", res1260, rank_1260, int(len(ids_x)))):
        for c in codes:
            r = res[c]
            nodes_n = (len(nodes_R[c]) if nn is None else nn)
            for k_, v_ in r["struct"].items():
                add(tab_name, c, k_, v_, "описательно", SRC, nodes_n, wname[c])
            add(tab_name, c, "K Louvain", r["K"], "описательно", SRC, nodes_n, wname[c])
            for k_ in ("ARI с KMeans", "NMI с KMeans", "ARI с Louvain R1"):
                add(tab_name, c, k_, r[k_], "больше = ближе к сравниваемому разбиению", SRC, nodes_n, wname[c], rk.get(k_, {}).get(c, np.nan))
            for k_ in S["INDICES"]:
                add(tab_name, c, f"{k_} (Louvain)", r["L"][k_], S["DIRECTIONS"][k_], SRC, nodes_n, wname[c], rk.get(k_, {}).get(c, np.nan))
            for k_ in ("MQ", "AVI", "AVU"):
                add(tab_name, c, f"{k_} (KMeans, канон)", r["KM"][k_], S["DIRECTIONS"][k_], SRC, nodes_n, wname[c])
    for c in codes:
        r = res_u[c]
        nodes_n = len(nodes_R[c])
        add("вес 1", c, "K Louvain", r["K"], "описательно", SRC, nodes_n, "1")
        for k_ in ("ARI с KMeans", "NMI с KMeans"):
            add("вес 1", c, k_, r[k_], "больше = ближе к сравниваемому разбиению", SRC, nodes_n, "1")
        add("вес 1", c, "ARI взвешенного и невзвешенного Louvain", ari_wu[c], "больше = ближе", SRC, nodes_n, "1")
        for k_ in S["INDICES"]:
            add("вес 1", c, f"{k_} (Louvain)", r["L"][k_], S["DIRECTIONS"][k_], SRC, nodes_n, "1")
        for k_ in ("MQ", "AVI", "AVU"):
            add("вес 1", c, f"{k_} (KMeans, канон)", r["KM"][k_], S["DIRECTIONS"][k_], SRC, nodes_n, "1")
    add("основная", "R2", "статус", np.nan, "", SRC, 0, "", note=S["R2_STATUS"])
    for l_, v_ in lag_share.items():
        add("основная", "R4", f"доля рёбер R4 с лучшим лагом {l_:+d}", v_, "описательно", SRC, N, wname["R4"])
    for r_ in CMP_ROWS:
        f_, ln_ = r_[0].split(", строка ")
        add("сверка", r_[1], "значение в notebooks", fnum(r_[2]), "", f"{f_}, строка {ln_}", 0, "", note=f"расчёт {r_[3]}, совпало: {r_[4]}")
    tab = pd.DataFrame(rows)
    part = pd.DataFrame({"territory_id": ids.astype(int), "тип KMeans": km})
    for c in codes:
        for suffix, d_ in ((" Louvain", lv[c]), (" Louvain, вес 1", lv_u[c])):
            part[c + suffix] = pd.Series([d_.get(int(i), pd.NA) for i in ids], dtype="Int64")
    ctl("edge_rules_partitions: 2004 строки, territory_id уникален, R6 покрывает 1260", (len(part), bool(part["territory_id"].is_unique), int(part["R6 Louvain"].notna().sum())),
        (2004, True, 1260))

    fig = draw_figure(codes, main_res, res1260)
    report = build_report(codes, E, main_res, res1260, res_u, ari_wu, rank_all, rank_1260, tt, lag_share, lag_all, wr, hashes, B1["months"], undefined_main,
                          notes_extra, fig["margins"], len(ids_x), n_imputed, n08)
    fig_text = " ".join(t.get_text() for t in fig["fig"].findobj(matplotlib.text.Text))
    ctl("запрещённые слова на рисунке", sum(len(re.findall(re.escape(w), fig_text, flags=re.I)) for w in S["FORBIDDEN_STEMS"]), 0)
    cmp_tab = md_table(["файл и строка", "показатель", "значение в notebooks", "значение в новом расчёте", "совпало"], CMP_ROWS)
    draft = report.replace("@@CMP@@", cmp_tab).replace("@@NCTL@@", "0").replace("@@CONTROLS@@", controls_table())
    stems = {w: len(re.findall(re.escape(w), draft, flags=re.I)) for w in S["FORBIDDEN_STEMS"]}
    ctl("запрещённые слова в md (основы)", {k: v for k, v in stems.items() if v}, {})
    nobr = re.sub(r"\[[+-]?\d+\.\d+; [+-]?\d+\.\d+\]", "", draft)
    ctl("md: квадратные скобки только в формате интервала", ("[" in nobr or "]" in nobr), False)
    report = report.replace("@@CMP@@", cmp_tab).replace("@@NCTL@@", str(len(CONTROLS))).replace("@@CONTROLS@@", controls_table())
    if not all(c[3] for c in CONTROLS):
        stop("контроли")
    for rel in (S["OUT_TABLE_PATH"], S["OUT_PARTITIONS_PATH"], S["REPORT_PATH"], S["FIG_PNG_PATH"]):
        P(rel).parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(P(S["OUT_TABLE_PATH"]), engine="pyarrow", index=False)
    part.to_parquet(P(S["OUT_PARTITIONS_PATH"]), engine="pyarrow", index=False)
    fig["fig"].savefig(P(S["FIG_PNG_PATH"]), dpi=S["FIG_DPI"], metadata={"Software": None})
    fig["fig"].savefig(P(S["FIG_SVG_PATH"]), format="svg", metadata={"Date": None})
    P(S["REPORT_PATH"]).write_text(report, encoding="utf-8")
    print("записано:", S["OUT_TABLE_PATH"], S["OUT_PARTITIONS_PATH"], S["REPORT_PATH"], S["FIG_PNG_PATH"], S["FIG_SVG_PATH"])


def controls_table() -> str:
    head = "| контроль | получено | ожидалось | статус |\n|---|---|---|---|"
    cl = lambda x: x.replace("|", "/").replace("[", "(").replace("]", ")")  # noqa: E731
    return head + "\n" + "\n".join(f"| {cl(a)} | {cl(b)} | {cl(c)} | {'пройден' if d else 'НАРУШЕН'} |" for a, b, c, d in CONTROLS)


# ============================================================================ рисунок
LABEL = {"R1": "R1\nканон", "R3": "R3\nкорр.\nрядов", "R4": "R4\nлаговая\nкорр.", "R5": "R5\nhighway", "R6": "R6\nrailway\n(1260 МО)"}
COLORS = {"assort": "#2F6690", "inside": "#E09F3E", "ari": "#2F6690", "K": "#9E2A2B"}


def draw_figure(codes, main_res, res1260) -> dict:
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "step32", "svg.fonttype": "path"})
    w, h = S["FIG_SIZE_IN"]
    fig = plt.figure(figsize=(w, h))
    axa = fig.add_axes([0.07, 0.31, 0.40, 0.49])
    axb = fig.add_axes([0.58, 0.31, 0.32, 0.49])
    x = np.arange(len(codes))
    bw = 0.38
    a_vals = [main_res[c]["struct"]["ассортативность по типам"] for c in codes]
    i_vals = [main_res[c]["struct"]["доля рёбер внутри типа"] for c in codes]
    b1 = axa.bar(x - bw / 2, a_vals, bw, color=COLORS["assort"], label="ассортативность по типам")
    b2 = axa.bar(x + bw / 2, i_vals, bw, color=COLORS["inside"], label="доля рёбер внутри типа")
    for bars in (b1, b2):
        for r in bars:
            axa.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.012, f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    axa.set_xticks(x)
    axa.set_xticklabels([LABEL[c] for c in codes], fontsize=9)
    axa.set_ylim(0, max(max(a_vals), max(i_vals)) * 1.22)
    axa.set_ylabel("значение", fontsize=10)
    axa.legend(fontsize=9, loc="upper right", frameon=False)
    axa.set_title("а) Структура сети и канонические типы", fontsize=11, loc="left", fontweight="bold")
    ari = [main_res[c]["ARI с KMeans"] for c in codes]
    kk = [main_res[c]["K"] for c in codes]
    br = axb.bar(x, ari, 0.55, color=COLORS["ari"], label="ARI с каноническим KMeans (левая шкала)")
    for r in br:
        axb.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008, f"{r.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    axb.set_xticks(x)
    axb.set_xticklabels([f"{LABEL[c]}\nK = {k_}" for c, k_ in zip(codes, kk)], fontsize=9)
    axb.set_ylim(0, max(ari) * 1.3)
    axb.set_ylabel("ARI", fontsize=10)
    ax2 = axb.twinx()
    ax2.plot(x, kk, "D", color=COLORS["K"], markersize=7, label="K Louvain (правая шкала)")
    ax2.set_ylim(0, max(kk) * 1.35)
    ax2.set_ylabel("число кластеров K", fontsize=10)
    axb.set_title("б) Louvain: согласие с KMeans и число кластеров", fontsize=11, loc="left", fontweight="bold")
    h1, l1 = axb.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axb.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right", frameon=False)
    for ax in (axa, axb, ax2):
        ax.spines["top"].set_visible(False)
    fig.suptitle("Правила построения рёбер: структура сети и результаты Louvain (декабрь 2024)", x=0.03, ha="left", y=0.96, fontsize=12.5, fontweight="bold")
    fig.text(0.03, 0.095, "R1, R3, R4, R5 — 2004 МО выборки; R6 — 1260 МО с железнодорожной сетью (шаг 07), показатели считаны на них.\n"
                         "R2 в репозитории не реализован как функция. DTW не реализовано. kNN, k = 8 для всех правил.\n"
                         "Louvain: resolution и seed канона; веса: R1 — сходство, R3 и R4 — корреляция, R5 и R6 — 1.",
             fontsize=9, ha="left", va="center", color="#333333", linespacing=1.5)
    fig.canvas.draw()
    rd = fig.canvas.get_renderer()
    bbs = [t.get_window_extent(rd) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
    bb = matplotlib.transforms.Bbox.union(bbs + [axa.get_window_extent(rd), axb.get_window_extent(rd), ax2.get_window_extent(rd)])
    dpi = fig.dpi
    margins = [float(v) for v in (bb.x0 / dpi, (fig.get_figwidth() * dpi - bb.x1) / dpi, bb.y0 / dpi, (fig.get_figheight() * dpi - bb.y1) / dpi)]
    ctl("рисунок: размер холста 11.7×6.5 дюйма", [round(float(v), 2) for v in fig.get_size_inches()], [11.7, 6.5])
    ctl("рисунок: поля не менее 0.25 дюйма (лево, право, низ, верх)", [round(m, 3) for m in margins], [S["FIG_MARGIN_MIN_IN"]] * 4,
        all(m >= S["FIG_MARGIN_MIN_IN"] for m in margins))
    return {"fig": fig, "margins": margins}


# ============================================================================ отчёт
def md_table(header, rows) -> str:
    cell = lambda x: str(x).replace("|", "/").replace("\n", " ")  # noqa: E731
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"] + ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows])


def order_line(vals: dict, fmt: str = "{:.3f}") -> str:
    items = sorted(vals.items(), key=lambda kv: (-round(kv[1], 9), kv[0]))
    out = fmt.format(items[0][1])
    parts = [f"{items[0][0]} ({out})"]
    for (k0, v0), (k1, v1) in zip(items, items[1:]):
        parts.append(">" if round(v0, 9) > round(v1, 9) else "=")
        parts.append(f"{k1} ({fmt.format(v1)})")
    return " ".join(parts)


def build_report(codes, E, main_res, res1260, res_u, ari_wu, rank_all, rank_1260, tt, lag_share, lag_all, wr, hashes, months, undefined, notes_extra,
                 margins, n1260, n_imputed, n08) -> str:
    f4 = lambda v: "н/д" if v != v else f"{v:.4f}"  # noqa: E731
    f2 = lambda v: "н/д" if v != v else (f"{v:.2f}" if abs(v) < 1000 else f"{v:.1f}")  # noqa: E731
    base = ["R1", "R3", "R4", "R5"]
    names = {r["code"]: r["name"] for r in S["RULES"]}
    arrows = {"больше лучше": "↑", "меньше лучше": "↓"}

    def struct_rows(res, cs):
        keys = list(res[cs[0]]["struct"].keys())
        return [[k] + [(f"{res[c]['struct'][k]:.4f}" if isinstance(res[c]["struct"][k], float) else f"{res[c]['struct'][k]:,}") for c in cs] for k in keys]

    def clus_rows(res, cs):
        return [[c, res[c]["K"], res[c]["sizes"], f4(res[c]["ARI с KMeans"]), f4(res[c]["NMI с KMeans"]), f4(res[c]["ARI с Louvain R1"])] for c in cs]

    def icvi_rows(res, cs, rk=None):
        rows_ = []
        for c in cs:
            L, KM = res[c]["L"], res[c]["KM"]
            cells_ = [c, res[c]["K"]]
            for k in ("SW", "CH", "S_Dbw"):
                r = (rk or {}).get(k, {}).get(c, np.nan)
                cells_.append(f2(L[k]) if k == "CH" else f4(L[k]))
                if rk is not None and r == r:
                    cells_[-1] += f" (ранг {r:g})"
            cells_ += [f4(L["MQ"]), f4(L["AVI"]), f4(L["AVU"]), f4(KM["MQ"]), f4(KM["AVI"]), f4(KM["AVU"])]
            rows_.append(cells_)
        return rows_

    icvi_head = ["правило", "K", f"SW {arrows['больше лучше']}", f"CH {arrows['больше лучше']}", f"S_Dbw {arrows['меньше лучше']}", "MQ ↑ (Louvain)", "AVI ↑ (Louvain)",
                 "AVU ↓ (Louvain)", "MQ ↑ (KMeans)", "AVI ↑ (KMeans)", "AVU ↓ (KMeans)"]
    clus_head = ["правило", "K", "размеры кластеров (по убыванию)", "ARI с KMeans", "NMI с KMeans", "ARI с Louvain R1"]
    rk_rows = lambda rk, cs: [[m] + [f"{rk[m][c]:g}" if rk[m][c] == rk[m][c] else "н/д" for c in cs] for m in S["RANK_METRICS"]]  # noqa: E731

    obs = []
    mets = [("ассортативность по типам", lambda r: r["struct"]["ассортативность по типам"]), ("доля рёбер внутри типа", lambda r: r["struct"]["доля рёбер внутри типа"]),
            ("плотность", lambda r: r["struct"]["плотность"]), ("доля взаимных рёбер", lambda r: r["struct"]["доля взаимных рёбер"]),
            ("средний коэффициент кластеризации", lambda r: r["struct"]["средний коэффициент кластеризации"]),
            ("ARI Louvain с KMeans", lambda r: r["ARI с KMeans"]), ("NMI Louvain с KMeans", lambda r: r["NMI с KMeans"]),
            ("K Louvain", lambda r: float(r["K"])), ("SW разбиения Louvain", lambda r: r["L"]["SW"]), ("CH разбиения Louvain", lambda r: r["L"]["CH"]),
            ("S_Dbw разбиения Louvain (меньше лучше)", lambda r: r["L"]["S_Dbw"]), ("MQ разбиения Louvain на собственном графе", lambda r: r["L"]["MQ"])]
    for nm, fn in mets:
        fm = "{:.0f}" if nm == "K Louvain" else ("{:.1f}" if nm.startswith("CH") else "{:.3f}")
        obs.append(f"- {nm}, правила R1, R3, R4, R5 (2004 МО): " + order_line({c: fn(main_res[c]) for c in base}, fm) + ".")
    obs = ["Знак «>» означает, что значение у стоящего слева правила выше, чем у стоящего справа; «=» — значения равны до 9 знаков.", ""] + obs
    obs += ["", "На пересечении 1260 МО (R1–R5 — индуцированный подграф, R6 — своя сеть):"]
    for nm, fn in mets[:2] + mets[5:8] + mets[8:11]:
        fm = "{:.0f}" if nm == "K Louvain" else ("{:.1f}" if nm.startswith("CH") else "{:.3f}")
        obs.append(f"- {nm}, R1–R6: " + order_line({c: fn(res1260[c]) for c in codes}, fm) + ".")
    obs += ["", "Чувствительность к весам (вес 1 вместо веса правила):"]
    for c in codes:
        w = main_res[c]
        u = res_u[c]
        obs.append(f"- {c}: K {w['K']} при весе правила и {u['K']} при весе 1; ARI с KMeans {w['ARI с KMeans']:.3f} и {u['ARI с KMeans']:.3f}; "
                   f"ARI между двумя Louvain-разбиениями {ari_wu[c]:.3f}.")
    obs += ["", f"R4: доля рёбер по лучшему лагу, l = −2, −1, 0, +1, +2: " + ", ".join(f"{lag_share[l]:.3f}" for l in S["LAGS"]) +
            "; по всем парам МО: " + ", ".join(f"{lag_all[l]:.3f}" for l in S["LAGS"]) + "."]

    w1_rows = []
    for c in codes:
        r, u = main_res[c], res_u[c]
        w1_rows.append([c, "1", u["K"], u["sizes"], f4(u["ARI с KMeans"]), f4(u["NMI с KMeans"]), f4(ari_wu[c]), f4(u["L"]["SW"]), f2(u["L"]["CH"]), f4(u["L"]["S_Dbw"]),
                        f4(u["L"]["MQ"]), f4(u["L"]["AVI"]), f4(u["L"]["AVU"]), f4(u["KM"]["MQ"]), f4(u["KM"]["AVI"]), f4(u["KM"]["AVU"])])
    w1_head = ["правило", "вес", "K", "размеры кластеров", "ARI с KMeans", "NMI с KMeans", "ARI с Louvain основной таблицы", "SW ↑", "CH ↑", "S_Dbw ↓", "MQ ↑ (Louvain)",
               "AVI ↑ (Louvain)", "AVU ↓ (Louvain)", "MQ ↑ (KMeans)", "AVI ↑ (KMeans)", "AVU ↓ (KMeans)"]
    wr_txt = {c: f"[{wr[c][0]:+.2f}; {wr[c][1]:+.2f}]" for c in wr}
    L = [
        "# 32. Правила построения рёбер: структура сети и результаты кластеризации (декабрь 2024)", "",
        "Сгенерировано `src/32_edge_rules.py`. Шаг описательный: новых методов кластеризации нет, Louvain с параметрами канона (resolution "
        f"{C.LOUVAIN_RESOLUTION}, seed {C.LOUVAIN_SEED}) и готовые индексы проекта; предрегистрации и порогов нет. Все контроли выполнены до записи файлов (раздел 10).", "",
        "## 1. Правила построения рёбер", "",
        f"Обозначения: i, j — МО выборки (2004), s_ic(m) — доля категории c в месяце m (пять категорий: {', '.join(c.replace('share_', '') for c in SHARE_COLUMNS)}), "
        "z_ic(m) — её z-оценка внутри месяца m по МО выборки (`network_utils.standardize_shares`), m = 1..24 (2023-01 … 2024-12); декабрь 2024 — m = 24; k = 8 во всех kNN-правилах.", "",
        "- **R1 (канон).** x_i = (z_i1, …, z_i5) за декабрь 2024; w_ij = cos(x_i, x_j) = ⟨x_i, x_j⟩ / (‖x_i‖·‖x_j‖). Ребро i–j, если j входит в 8 наибольших w_i· или i входит в 8 наибольших w_j·. Вес ребра — w_ij (similarity).",
        f"- **R2.** Порог по косинусу сырых долей декабря 2024: ребро, если cos(s_i, s_j) > {C.RAW_NET_EDGE_THRESHOLD} (шаг 04, отклонённый вариант; 99.8% пар). **{S['R2_STATUS']}**: порог применяется внутри `main()` шага 04, отдельной функции нет; вариант не воссоздавался, правило пропущено.",
        "- **R3.** Вектор x_i = (z_ic(m)), c = 1..5, m = 1..24, размерность 120. Вес ρ_ij = коэффициент корреляции Пирсона между x_i и x_j (по 120 значениям). Ребро i–j, если j входит в 8 наибольших ρ_i· или i входит в 8 наибольших ρ_j·. Вес ребра — ρ_ij.",
        "- **R4.** Для лага l ∈ {−2, −1, 0, +1, +2} и категории c: ρ_ij^c(l) = corr(z_ic(m), z_jc(m + l)) по месяцам m, для которых определены оба значения (пересечение месяцев); ρ_ij(l) = (1/5)·Σ_c ρ_ij^c(l); "
        "w_ij = max по l от −2 до +2 из ρ_ij(l). Так как ρ_ji(l) = ρ_ij(−l), матрица w симметрична. Ребро i–j — как в R3 по w; вес ребра — w_ij. "
        "Доступно месяцев: " + ", ".join(f"l = {l}: {months[l]}" for l in sorted(months)) + ".",
        "- **R5.** Расстояние highway из шага 07 (`transport_network_highway_knn8.parquet`, функция `knn_edges_by_distance`): ребро, если j входит в 8 ближайших к i или i входит в 8 ближайших к j. Вес в основных таблицах — 1.",
        f"- **R6.** Расстояние railway из шага 07 с достройкой отсутствующих пар между МО с железной дорогой нулём (достроено {n_imputed} пар); сеть построена на {n1260} МО с железной дорогой. Правило kNN и вес — как в R5.",
        "",
        "Симметризация во всех kNN-правилах одна (`network_utils.knn_pairs`): неориентированное ребро {i, j} существует, если j в топ-8 для i **или** i в топ-8 для j (объединение направленных рёбер); "
        "ребро взаимное, если выполняются оба условия. До симметризации у каждого узла ровно 8 направленных рёбер.", "",
        f"Диапазоны весов рёбер: R1 {wr_txt['R1']}, R3 {wr_txt['R3']}, R4 {wr_txt['R4']}; у R5 и R6 вес 1. Пересчёт с весом 1 — в разделе 6.", "",
        "## 2. Множество МО и железнодорожная сеть", "",
        f"R1, R3, R4, R5 построены на 2004 МО выборки. Железнодорожная сеть шага 07 содержит {n1260} из 2004 МО; {2004 - n1260} МО без железной дороги в R6 не входят, и это не случайное подмножество. "
        "Распределение МО по каноническим типам (число МО):", "",
        md_table(["тип KMeans", "в R6", "вне R6", "всего"], [[int(t), int(r["в R6"]), int(r["вне R6"]), int(r.sum())] for t, r in tt.iterrows()] +
                 [["итого", int(tt["в R6"].sum()), int(tt["вне R6"].sum()), int(tt.values.sum())]]), "",
        f"R6 считается на {n1260} МО и в рейтинги на 2004 МО не входит; ранги R6 даны только в таблице на пересечении 1260 МО (раздел 7). Для R1–R5 на пересечении 1260 МО берутся те же графы, "
        "ограниченные этими МО (индуцированный подграф), и те же метки Louvain, что в основной таблице; ARI и NMI считаются только на этих МО. Признак «взаимное ребро» сохраняется из полного графа правила.", "",
        "## 3. Структура сетей", "",
        "Степень, плотность, компоненты, коэффициент кластеризации и ассортативность — по невзвешенному графу; плотность = 2E / (N(N−1)); ст. откл. степени — по совокупности узлов; "
        "ассортативность — `networkx.attribute_assortativity_coefficient` по каноническим типам (kmeans_labels_final); модулярность типов — `partition_metrics` с весом правила. "
        "Столбец R2 отсутствует: " + S["R2_STATUS"] + ".", "",
        md_table(["показатель"] + [f"{c}" + (" (1260 МО)" if c == "R6" else " (2004 МО)") for c in codes], struct_rows(main_res, codes)), "",
        "## 4. Результаты кластеризации (Louvain на графе правила)", "",
        f"Louvain: `networkx.community.louvain_communities`, resolution {C.LOUVAIN_RESOLUTION}, seed {C.LOUVAIN_SEED}, вес ребра — вес правила (R5 и R6 — 1). Нумерация кластеров по размеру, как в шаге 11. "
        "ARI и NMI — с каноническим KMeans (k = 6, kmeans_labels_final), последний столбец — ARI с Louvain на R1. R6: узлы только из R6, сравнение на тех же МО.", "",
        md_table(clus_head, clus_rows(main_res, codes)), "",
        "## 5. Индексы качества (ICVI)", "",
        "SW, CH, S_Dbw — для разбиения Louvain в атрибутивном пространстве варианта A (z-оценки пяти долей, декабрь 2024); MQ, AVI, AVU — на графе этого правила; "
        "последние три столбца — для канонического KMeans на графе того же правила. Функции: SW и CH — `sklearn.metrics`, S_Dbw — пакет `s-dbw` с `SDBW_KW` из config (как в шаге 10), "
        "MQ, AVI, AVU — `src/partition_metrics.py`. Стрелки — направление (↑ больше лучше, ↓ меньше лучше). Ранги (1 — лучший) даны для SW, CH, S_Dbw среди R1, R3, R4, R5; "
        "значения MQ, AVI, AVU на разных графах напрямую не сопоставляются.", "",
        md_table(icvi_head, icvi_rows(main_res, codes, {k: rank_all[k] for k in ("SW", "CH", "S_Dbw")})), "",
        "Ранги среди R1, R3, R4, R5 (SW, CH — больше лучше, S_Dbw — меньше лучше; для ARI и NMI ранг 1 — наибольшее значение):", "",
        md_table(["показатель"] + base, rk_rows(rank_all, base)), "",
        "## 6. Чувствительность к весам (вес ребра 1 для всех правил)", "",
        "В основных таблицах веса у правил разные по их определению (R1 — сходство, R3 и R4 — корреляция, R5 и R6 — 1), поэтому различие между правилами может отражать вес, а не правило. "
        "Ниже Louvain и индексы MQ, AVI, AVU пересчитаны на тех же рёбрах с весом 1 (для R5 и R6 вес и так 1, строки совпадают с основной таблицей). "
        "Выводы о лучшем правиле осторожны и опираются на обе таблицы.", "",
        md_table(w1_head, w1_rows), "",
        "## 7. Пересечение 1260 МО: те же показатели для R1–R6", "",
        f"Все правила рассматриваются на тех же {n1260} МО. R1–R5 — индуцированные подграфы, метки Louvain из основной таблицы; R6 — своя сеть. Ранги — среди R1–R6.", "",
        "Структура:", "",
        md_table(["показатель"] + codes, struct_rows(res1260, codes)), "",
        "Кластеризация (K — число различных меток среди этих МО):", "",
        md_table(clus_head, clus_rows(res1260, codes)), "",
        "ICVI:", "",
        md_table(icvi_head, icvi_rows(res1260, codes, {k: rank_1260[k] for k in ("SW", "CH", "S_Dbw")})), "",
        "Ранги среди R1–R6:", "",
        md_table(["показатель"] + codes, rk_rows(rank_1260, codes)), "",
        "## 8. Наблюдения (только сопоставление значений)", "",
        *obs, "",
        "## 9. Ограничения", "",
        "- Один месяц (декабрь 2024) для R1, R2, R5, R6; 24 месяца (2023-01 … 2024-12) для R3 и R4. Правила R3 и R4 используют информацию за весь период, R1, R5, R6 — нет.",
        "- Индексы MQ, AVI, AVU на графах разных правил несопоставимы напрямую: у графов разные веса и разная структура. SW, CH, S_Dbw считаются в одном пространстве, но при разном K.",
        "- kNN с k = 8 одинаково для всех правил; число рёбер у kNN-правил почти одинаково по построению, плотность определяется числом узлов.",
        f"- R6 построено на {n1260} МО с железнодорожной сетью; {2004 - n1260} МО без неё не входят в R6, и состав этого множества по типам неслучаен (раздел 2). Сравнение R6 с остальными правилами — на пересечении (раздел 7).",
        "- Louvain — один запуск (seed 42) без выбора лучшего из многих; в шаге 11b показано, что запуски различаются по K и по составу сообществ.",
        "- R2 не вычислялось (нет функции). Лаговые связи (R4) ограничены лагами ±2 месяца; DTW не реализовано.",
        f"- В notebooks/08 напечатано {n08} достроенных железнодорожных пар, в notebooks/07 и в данных — {n_imputed}; файлы не менялись.", "",
        "## 10. Сверка с напечатанными значениями и контроли", "",
        "Сверка (допуск — половина последней напечатанной цифры):", "",
        "@@CMP@@", "",
        f"Хэши матриц смежности (sha256, повторное построение даёт тот же хэш): R1 `{hashes['R1']}`, R3 `{hashes['R3']}`, R4 `{hashes['R4']}`.", "",
        "Контроли (выполнены до записи файлов; всего @@NCTL@@; квадратные скобки в списках заменены круглыми):", "",
        "@@CONTROLS@@", "",
        "## 11. Неопределённые индексы", "",
        ("В основной таблице неопределённых индексов нет." if not undefined else md_table(["набор", "индекс", "примечание"], undefined)), "",
        ("В остальных таблицах (пересечение 1260 МО, вес 1) неопределённых индексов нет." if not notes_extra else md_table(["набор", "индекс", "примечание"], notes_extra)), "",
        "## 12. Рисунок", "",
        f"`{S['FIG_PNG_PATH']}`, `{S['FIG_SVG_PATH']}`: панель (а) — ассортативность по типам и доля рёбер внутри типа по правилам; панель (б) — ARI Louvain с каноническим KMeans и K Louvain. "
        f"Холст {S['FIG_SIZE_IN'][0]}×{S['FIG_SIZE_IN'][1]} дюйма, PNG {S['FIG_DPI']} dpi, шрифт DejaVu Sans. Поля (лево, право, низ, верх), дюймы: " + ", ".join(f"{m:.2f}" for m in margins) + ".", "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    main()
