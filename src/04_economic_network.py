"""Шаг 04. Экономическая сеть за один месяц: косинусное сходство МО по 5 долям категорий.

Вход:  data/processed/category_shares.parquet
Выход: data/processed/economic_network_<YYYY_MM>.parquet  (рёбра с similarity > порога),
       notebooks/04_economic_network_<YYYY_MM>.md, notebooks/04_similarity_hist_<YYYY_MM>.png
Запуск из корня проекта:  .venv/bin/python src/04_economic_network.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"

from config import MONTH
from config import RAW_NET_EDGE_THRESHOLD as EDGE_THRESHOLD
SHARE_COLUMNS = ["share_Продовольствие", "share_Здоровье", "share_Общепит",
                 "share_Транспорт", "share_Маркетплейсы"]
from config import RAW_NET_CANDIDATE_THRESHOLDS as CANDIDATE_THRESHOLDS

TAG = MONTH.replace("-", "_")
NETWORK_PATH = PROCESSED_DIR / f"economic_network_{TAG}.parquet"
REPORT_PATH = NOTEBOOKS_DIR / f"04_economic_network_{TAG}.md"
HIST_PATH = NOTEBOOKS_DIR / f"04_similarity_hist_{TAG}.png"


def md_table(df: pd.DataFrame, index: bool = True, fmt: str = ",.4f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def main() -> None:
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    month = shares[shares["date"] == pd.Timestamp(MONTH)].sort_values("territory_id")
    if month.empty or month["territory_id"].duplicated().any():
        raise SystemExit(f"STOP: за {MONTH} нет данных или дубликаты territory_id")

    ids = month["territory_id"].to_numpy()
    sim = cosine_similarity(month[SHARE_COLUMNS].to_numpy())

    # Все неупорядоченные пары (i < j), без петель
    iu, ju = np.triu_indices(len(ids), k=1)
    pair_sim = sim[iu, ju]

    keep = pair_sim > EDGE_THRESHOLD
    edges = pd.DataFrame({
        "territory_id_x": ids[iu[keep]],
        "territory_id_y": ids[ju[keep]],
        "similarity": pair_sim[keep],
    })
    edges.to_parquet(NETWORK_PATH, engine="pyarrow", index=False)

    # --- Распределение similarity ---
    s = pd.Series(pair_sim)
    quantiles = s.quantile([0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])
    bins = [-1e-9, .5, .8, .9, .95, .97, .98, .99, .995, .999, 1 + 1e-9]
    hist = (pd.cut(s, bins=bins).value_counts().sort_index()
            .rename("пар").rename_axis("интервал similarity").to_frame())
    hist["доля пар"] = hist["пар"] / len(s)

    # Число рёбер и степень узлов при разных порогах
    n = len(ids)
    rows = []
    for t in CANDIDATE_THRESHOLDS:
        adj = sim > t
        np.fill_diagonal(adj, False)
        deg = adj.sum(axis=1)
        rows.append({"порог": t, "рёбер": int((pair_sim > t).sum()),
                     "плотность": (pair_sim > t).mean(),
                     "степень медиана": np.median(deg), "степень мин": deg.min(),
                     "изолированных МО": int((deg == 0).sum())})
    thr = pd.DataFrame(rows).set_index("порог")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(pair_sim, bins=200, color="#4C72B0")
    axes[0].set(title=f"Косинусное сходство, все пары ({MONTH})",
                xlabel="similarity", ylabel="число пар")
    tail = pair_sim[pair_sim > 0.9]
    axes[1].hist(tail, bins=200, color="#4C72B0")
    axes[1].set(title="Фрагмент similarity > 0.9", xlabel="similarity")
    for t in [0.95, 0.99]:
        axes[1].axvline(t, color="#C44E52", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(HIST_PATH, dpi=120)

    lines = [
        f"# 04. Экономическая сеть, {MONTH}",
        "",
        "Сгенерировано `src/04_economic_network.py`. Косинусное сходство по 5 долям "
        f"({', '.join(SHARE_COLUMNS)}).",
        "",
        f"- МО: **{n}**, всех неупорядоченных пар: **{len(pair_sim):,}**",
        f"- рёбер с similarity > {EDGE_THRESHOLD}: **{len(edges):,}** "
        f"({len(edges) / len(pair_sim):.1%} всех пар) → "
        f"`{NETWORK_PATH.relative_to(PROJECT_DIR)}` (каждая пара один раз, x < y)",
        "",
        "## Распределение similarity по всем парам",
        "",
        f"![hist]({HIST_PATH.name})",
        "",
        md_table(quantiles.rename("similarity").rename_axis("квантиль").to_frame()),
        "",
        md_table(hist, fmt=",.4f"),
        "",
        "## Порог → размер сети",
        "",
        md_table(thr, fmt=",.4f"),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Рёбер > {EDGE_THRESHOLD}: {len(edges):,} -> {NETWORK_PATH}")
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
