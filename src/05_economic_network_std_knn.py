"""Шаг 05. Экономическая сеть за месяц: z-score долей -> косинусное сходство -> kNN-граф.

Вход:  data/processed/category_shares.parquet
Выход: data/processed/economic_network_<YYYY_MM>_std_knn<K>.parquet  (edge list),
       data/processed/economic_network_<YYYY_MM>_std_knn<K>.graphml,
       notebooks/05_economic_network_<YYYY_MM>_std_knn<K>.md, гистограмма .png
Предыдущая версия без стандартизации (шаг 04) не затрагивается.
Запуск из корня проекта:  .venv/bin/python src/05_economic_network_std_knn.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from network_utils import SHARE_COLUMNS, knn_edges, std_cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"

from config import MONTH
from config import ECON_KNN_K as K
QUANTILES = [0, .01, .1, .25, .5, .75, .9, 1]

TAG = f"{MONTH.replace('-', '_')}_std_knn{K}"
EDGES_PATH = PROCESSED_DIR / f"economic_network_{TAG}.parquet"
GRAPHML_PATH = PROCESSED_DIR / f"economic_network_{TAG}.graphml"
REPORT_PATH = NOTEBOOKS_DIR / f"05_economic_network_{TAG}.md"
HIST_PATH = NOTEBOOKS_DIR / f"05_similarity_hist_{TAG}.png"


def md_table(df: pd.DataFrame, index: bool = True, fmt: str = ",.4f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def main() -> None:
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    month = shares[shares["date"] == pd.Timestamp(MONTH)].sort_values("territory_id")
    if month.empty or month["territory_id"].duplicated().any():
        raise SystemExit(f"STOP: за {MONTH} нет данных или дубликаты territory_id")
    ids = month["territory_id"].to_numpy()
    X_raw = month[SHARE_COLUMNS].to_numpy()

    # 1–2. z-score по каждой категории + косинусное сходство
    sim, scaler = std_cosine_similarity(X_raw)
    iu, ju = np.triu_indices(len(ids), k=1)
    pair_std = sim[iu, ju]
    pair_raw = cosine_similarity(X_raw)[iu, ju]

    # 3. Сравнение квантилей
    q = pd.DataFrame({
        "без стандартизации": pd.Series(pair_raw).quantile(QUANTILES),
        "z-score": pd.Series(pair_std).quantile(QUANTILES),
    }).rename_axis("квантиль")

    # 4. kNN-граф
    edges = knn_edges(ids, sim, K)
    G = nx.Graph()
    G.add_nodes_from(int(i) for i in ids)
    G.add_weighted_edges_from(
        ((int(x), int(y), float(w)) for x, y, w in
         edges[["territory_id_x", "territory_id_y", "similarity"]].itertuples(index=False)),
        weight="similarity")
    nx.set_edge_attributes(G, {(int(x), int(y)): bool(m) for x, y, m in
                               edges[["territory_id_x", "territory_id_y", "mutual"]]
                               .itertuples(index=False)}, "mutual")

    # 5. Метрики графа
    deg = pd.Series(dict(G.degree()))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    metrics = pd.DataFrame([
        ("узлов", G.number_of_nodes()),
        ("рёбер", G.number_of_edges()),
        ("из них взаимных (mutual)", int(edges["mutual"].sum())),
        ("плотность", nx.density(G)),
        ("изолированных МО", nx.number_of_isolates(G)),
        ("компонент связности", len(comps)),
        ("размер крупнейшей компоненты", len(comps[0])),
        ("степень min", deg.min()),
        ("степень медиана", deg.median()),
        ("степень max", deg.max()),
        ("similarity ребра min", edges["similarity"].min()),
        ("similarity ребра медиана", edges["similarity"].median()),
    ], columns=["метрика", "значение"])

    # 6. Сохранение
    edges.to_parquet(EDGES_PATH, engine="pyarrow", index=False)
    G.graph.update({"month": MONTH, "method": f"zscore+cosine, kNN k={K}, undirected union"})
    nx.write_graphml(G, GRAPHML_PATH)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pair_raw, bins=200, alpha=.6, label="без стандартизации", color="#8C8C8C")
    ax.hist(pair_std, bins=200, alpha=.7, label="z-score", color="#4C72B0")
    ax.set(title=f"Косинусное сходство пар МО, {MONTH}", xlabel="similarity",
           ylabel="число пар")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HIST_PATH, dpi=120)

    small = [sorted(int(i) for i in c) for c in comps[1:]]
    lines = [
        f"# 05. Экономическая сеть {MONTH}: z-score + cosine + kNN (k={K})",
        "",
        "Сгенерировано `src/05_economic_network_std_knn.py`.",
        "",
        "Средние и std долей, использованные для z-score:",
        "",
        md_table(pd.DataFrame({"mean": scaler.mean_, "std": scaler.scale_},
                              index=SHARE_COLUMNS)),
        "",
        "## Квантили similarity по всем парам",
        "",
        f"![hist]({HIST_PATH.name})",
        "",
        md_table(q),
        "",
        f"## kNN-граф (k={K}, объединение x→y и y→x)",
        "",
        md_table(metrics, index=False),
        "",
    ]
    if small:
        lines += [f"Мелкие компоненты (кроме крупнейшей): {small}", ""]
    lines += [
        f"- edge list: `{EDGES_PATH.relative_to(PROJECT_DIR)}` "
        "(territory_id_x < territory_id_y, similarity, mutual)",
        f"- граф: `{GRAPHML_PATH.relative_to(PROJECT_DIR)}` (атрибуты рёбер similarity, mutual)",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
