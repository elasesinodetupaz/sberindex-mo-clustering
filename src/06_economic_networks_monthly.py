"""Шаг 06. Экономическая сеть по каждому месяцу: z-score внутри месяца -> cosine -> kNN.

Вход:  data/processed/category_shares.parquet
Выход: data/processed/economic_networks/{YYYY-MM}_std_knn<K>.parquet (по месяцу),
       data/processed/economic_networks_summary.md
Сверка: декабрь 2024 должен совпасть с результатом шага 05.
Запуск из корня проекта:  .venv/bin/python src/06_economic_networks_monthly.py
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from network_utils import SHARE_COLUMNS, knn_edges, std_cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
VALID_TERRITORIES_PATH = PROCESSED_DIR / "valid_territories.parquet"
NETWORKS_DIR = PROCESSED_DIR / "economic_networks"
SUMMARY_PATH = PROCESSED_DIR / "economic_networks_summary.md"

from config import ECON_KNN_K as K
from config import MONTH as REFERENCE_MONTH
REFERENCE_PATH = PROCESSED_DIR / f"economic_network_{REFERENCE_MONTH.replace('-', '_')}_std_knn{K}.parquet"


def edge_set(edges: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(edges["territory_id_x"].astype(int), edges["territory_id_y"].astype(int)))


def main() -> None:
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    n_valid = len(pd.read_parquet(VALID_TERRITORIES_PATH, engine="pyarrow"))
    NETWORKS_DIR.mkdir(parents=True, exist_ok=True)

    rows, prev = [], None
    for date, month in shares.groupby("date", sort=True):
        label = f"{date:%Y-%m}"
        month = month.sort_values("territory_id")
        if len(month) != n_valid or month["territory_id"].duplicated().any():
            raise SystemExit(f"STOP: {label}: {len(month)} МО вместо {n_valid} или дубликаты")
        ids = month["territory_id"].to_numpy()

        sim, _ = std_cosine_similarity(month[SHARE_COLUMNS].to_numpy())
        edges = knn_edges(ids, sim, K)
        edges.to_parquet(NETWORKS_DIR / f"{label}_std_knn{K}.parquet", engine="pyarrow",
                         index=False)

        G = nx.Graph()
        G.add_nodes_from(int(i) for i in ids)
        G.add_edges_from(edge_set(edges))
        deg = np.array([d for _, d in G.degree()])
        cur = edge_set(edges)
        rows.append({
            "месяц": label,
            "рёбер": G.number_of_edges(),
            "взаимных": int(edges["mutual"].sum()),
            "степень медиана": float(np.median(deg)),
            "степень max": int(deg.max()),
            "компонент": nx.number_connected_components(G),
            "изолированные МО": "да" if nx.number_of_isolates(G) else "нет",
            "similarity ребра медиана": edges["similarity"].median(),
            # доля рёбер, общих с предыдущим месяцем (Жаккар)
            "Жаккар с пред. мес.": (len(cur & prev) / len(cur | prev)) if prev else np.nan,
        })
        prev = cur

    summary = pd.DataFrame(rows)

    # Сверка с эталоном (шаг 05)
    ref = pd.read_parquet(REFERENCE_PATH, engine="pyarrow")
    new = pd.read_parquet(NETWORKS_DIR / f"{REFERENCE_MONTH}_std_knn{K}.parquet", engine="pyarrow")
    ref_ok = ref.equals(new)
    if not ref_ok:
        raise SystemExit(f"STOP: {REFERENCE_MONTH} не совпал с эталоном {REFERENCE_PATH.name}")

    # Стабильность относительно первого месяца
    first = edge_set(pd.read_parquet(NETWORKS_DIR / f"{summary['месяц'].iloc[0]}_std_knn{K}.parquet"))
    last = edge_set(new)
    jac_first_last = len(first & last) / len(first | last)

    lines = [
        f"# Экономические сети по месяцам (z-score внутри месяца + cosine + kNN, k={K})",
        "",
        "Сгенерировано `src/06_economic_networks_monthly.py`. "
        f"Файлы: `data/processed/economic_networks/{{YYYY-MM}}_std_knn{K}.parquet`.",
        "",
        f"- Сверка {REFERENCE_MONTH} с `{REFERENCE_PATH.name}`: "
        f"**{'совпадает' if ref_ok else 'НЕ совпадает'}**",
        f"- Жаккар рёбер {summary['месяц'].iloc[0]} vs {summary['месяц'].iloc[-1]}: "
        f"**{jac_first_last:.3f}**",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Разброс показателей по 24 месяцам:",
        "",
        summary[["рёбер", "степень медиана", "компонент", "similarity ребра медиана",
                 "Жаккар с пред. мес."]].describe().loc[["min", "mean", "max"]]
        .to_markdown(floatfmt=".3f"),
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
