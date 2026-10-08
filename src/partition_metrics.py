"""Сетевые метрики качества разбиения: MQ (модулярность), AVI (isolability), AVU (unifiability).

Формулы — Shalileh, Antonov, Tsyplakova (2025), Doklady Mathematics 112(3), (18)–(21):

  E_int(k)     — сумма весов рёбер внутри кластера k
  B_k          — сумма весов рёбер, у которых ровно один конец в k (граница k)
  E(k, l)      — сумма весов рёбер между кластерами k и l
  Isolability(k)     = 2·E_int(k) / (2·E_int(k) + B_k)
  Unifiability(k, l) = E(k, l) / (B_k + B_l − E(k, l))
  AVI = (1/K) · Σ_k Isolability(k)
  AVU = (1/K) · Σ_{k≠l, упорядоченные пары} Unifiability(k, l)

Нормировка AVU именно 1/K (не 1/(K(K−1))), как в формуле (21). Она же гарантирует
AVU ∈ [0, 1]: U(k, l) ≤ E(k, l) / B_k (так как B_l ≥ E(k, l)), а Σ_l E(k, l) = B_k,
поэтому Σ_l U(k, l) ≤ 1 для каждого k.
Высокий AVI и низкий AVU = хорошо разделённые кластеры.
"""
import networkx as nx
import numpy as np
import pandas as pd


def cluster_edge_sums(G: nx.Graph, labels: dict, weight: str | None = "weight"
                      ) -> tuple[list, np.ndarray]:
    """Матрица W[k, l]: сумма весов рёбер между кластерами k и l; W[k, k] = E_int(k)."""
    clusters = sorted(set(labels.values()))
    idx = {c: i for i, c in enumerate(clusters)}
    W = np.zeros((len(clusters), len(clusters)))
    for u, v, d in G.edges(data=True):
        if u == v:
            continue
        w = d.get(weight, 1.0) if weight else 1.0
        a, b = idx[labels[u]], idx[labels[v]]
        if a == b:
            W[a, a] += w
        else:
            W[a, b] += w
            W[b, a] += w
    return clusters, W


def isolability(W: np.ndarray) -> np.ndarray:
    e_int = np.diag(W)
    boundary = W.sum(axis=1) - e_int
    denom = 2 * e_int + boundary
    if (denom == 0).any():
        raise ValueError("кластер без рёбер (E_int = 0 и B = 0): isolability не определена")
    return 2 * e_int / denom


def unifiability(W: np.ndarray) -> np.ndarray:
    """Матрица U[k, l] для k ≠ l (диагональ = 0)."""
    boundary = W.sum(axis=1) - np.diag(W)
    E = W.copy()
    np.fill_diagonal(E, 0.0)
    denom = boundary[:, None] + boundary[None, :] - E
    with np.errstate(divide="ignore", invalid="ignore"):
        U = np.where(denom > 0, E / denom, 0.0)
    np.fill_diagonal(U, 0.0)
    return U


def partition_metrics(G: nx.Graph, labels: dict, weight: str | None = "weight"
                      ) -> tuple[dict, pd.DataFrame]:
    """MQ, AVI, AVU для разбиения labels (узел -> кластер) и таблица по кластерам."""
    missing = set(G.nodes) - set(labels)
    if missing:
        raise ValueError(f"{len(missing)} узлов без метки кластера")
    clusters, W = cluster_edge_sums(G, labels, weight)
    K = len(clusters)
    iso = isolability(W)
    U = unifiability(W)
    communities = [{n for n, c in labels.items() if c == cl and n in G} for cl in clusters]
    summary = {
        "K": K,
        "MQ": nx.community.modularity(G, communities, weight=weight),
        "AVI": iso.mean(),
        "AVU": U.sum() / K,
    }
    per_cluster = pd.DataFrame({
        "cluster": clusters,
        "узлов": [len(c) for c in communities],
        "E_int": np.diag(W),
        "B": W.sum(axis=1) - np.diag(W),
        "Isolability": iso,
        "Σ_l Unifiability(k,l)": U.sum(axis=1),
    })
    return summary, per_cluster


def _cliques(n: int, K: int, inter: list[tuple[int, int]], w: float) -> tuple[nx.Graph, dict]:
    G = nx.disjoint_union_all([nx.complete_graph(n)] * K)
    nx.set_edge_attributes(G, 1.0, "weight")
    G.add_edges_from((a, b, {"weight": w}) for a, b in inter)
    return G, {v: v // n for v in G}


def self_check() -> tuple[pd.DataFrame, list[str]]:
    """Проверки свойств метрик на синтетических графах (при нарушении — AssertionError).

    Граф: 4 клики по 10 узлов (вес внутренних рёбер 1) + 6 межкластерных рёбер веса w,
    либо равномерно по всем парам кластеров, либо сосредоточенных в парах (0,1) и (2,3).
    """
    even = [(0, 10), (1, 20), (2, 30), (11, 21), (12, 31), (22, 32)]
    paired = [(0, 10), (1, 11), (2, 12), (20, 30), (21, 31), (22, 32)]
    rows = []
    for layout, inter in [("равномерно по парам", even), ("сосредоточены в 2 парах", paired)]:
        for w in [0.01, 1.0, 10.0]:
            G, lab = _cliques(10, 4, inter, w)
            s, _ = partition_metrics(G, lab)
            rows.append({"межкластерные рёбра": layout, "вес w": w, **s})
    R = nx.erdos_renyi_graph(200, 0.05, seed=1)
    rng = np.random.default_rng(0)
    s, _ = partition_metrics(R, {n: int(rng.integers(0, 4)) for n in R}, weight=None)
    rows.append({"межкластерные рёбра": "случайный граф, случайная разметка", "вес w": np.nan, **s})
    tab = pd.DataFrame(rows)

    bounds_ok = True
    for seed in range(30):
        H = nx.erdos_renyi_graph(60, 0.1, seed=seed)
        H.remove_nodes_from(list(nx.isolates(H)))
        k = int(rng.integers(2, 8))
        m, _ = partition_metrics(H, {n: int(rng.integers(0, k)) for n in H}, weight=None)
        bounds_ok &= 0 <= m["AVU"] <= 1 + 1e-12 and 0 <= m["AVI"] <= 1 + 1e-12

    ev = tab[tab["межкластерные рёбра"] == "равномерно по парам"]
    pr = tab[tab["межкластерные рёбра"] == "сосредоточены в 2 парах"]
    rnd = tab.iloc[-1]
    assert bounds_ok, "AVI/AVU вышли за [0, 1]"
    assert ev["AVI"].is_monotonic_decreasing and ev["MQ"].is_monotonic_decreasing
    assert np.allclose(ev["AVU"], ev["AVU"].iloc[0]) and np.allclose(pr["AVU"], pr["AVU"].iloc[0])
    assert (pr["AVU"].to_numpy() > ev["AVU"].to_numpy()).all()
    assert ev["AVI"].iloc[0] > rnd["AVI"] and ev["MQ"].iloc[0] > rnd["MQ"]
    avu_vs_random = ev["AVU"].iloc[0] - rnd["AVU"]

    notes = [
        "- AVI и AVU лежат в [0, 1] на 30 случайных графах со случайными разметками (K = 2..7) — ✔",
        "- AVI и MQ падают при росте веса межкластерных рёбер (w: 0.01 → 1 → 10) — ✔",
        "- **AVU не зависит от веса (объёма) межкластерных рёбер** — одинаков при w = 0.01, 1, 10. "
        "Это прямое следствие формулы (20): числитель E(k,l) и знаменатель B_k + B_l − E(k,l) "
        "состоят только из межкластерных рёбер, поэтому умножение их всех на константу U не "
        "меняет.",
        "- AVU реагирует на **распределение** границы по кластерам: если граница каждого "
        "кластера уходит в одного соседа, AVU = 1; если размазана по всем — ниже (0.6 при K = 4). "
        "Высокий AVU = есть пары кластеров, «перетекающие» друг в друга (кандидаты на слияние).",
        f"- Следствие: почти идеально разделённые клики (w = 0.01) и случайная разметка "
        f"случайного графа имеют почти одинаковый AVU ({ev['AVU'].iloc[0]:.3f} против "
        f"{rnd['AVU']:.3f}, разница {avu_vs_random:+.3f}), тогда как AVI ({ev['AVI'].iloc[0]:.3f} "
        f"против {rnd['AVI']:.3f}) и MQ их чётко различают. Ожидание «мало межкластерных рёбер "
        "→ AVU низкий» для формулы (20)–(21) в заданном виде **не выполняется**; разделённость "
        "по объёму рёбер измеряют AVI и MQ, AVU — структуру связей между кластерами.",
    ]
    return tab, notes
