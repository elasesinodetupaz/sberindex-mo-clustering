"""Шаг 11b. Устойчивость Louvain: 100 запусков, выбор лучшего по MQ, сравнение с каноническим KMeans.

Параметры графа и Louvain (месяц, вес, resolution) — из шага 11.
Итоговое разбиение — реальный запуск с максимальной модулярностью (не консенсус меток).

Вход:  data/processed/economic_networks/<YYYY-MM>_std_knn8.parquet,
       data/processed/kmeans_labels_final.parquet, data/raw/territories.parquet
Выход: data/processed/louvain_runs_<YYYY_MM>.parquet     (seed, K, MQ по 100 запускам),
       data/processed/louvain_runs_labels_<YYYY_MM>.parquet (seed, territory_id, cluster),
       data/processed/louvain_labels_<YYYY_MM>.parquet    (итоговое разбиение: territory_id, cluster),
       data/processed/louvain_final_seed_<YYYY_MM>.txt    (seed итогового разбиения),
       notebooks/11b_louvain_stability.md, notebooks/figures/louvain_*.png
Запуск из корня проекта:  .venv/bin/python src/11b_louvain_stability.py
"""
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from config import FINAL_K
from partition_metrics import partition_metrics

step11 = importlib.import_module("11_louvain")

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
FIGURES_DIR = NOTEBOOKS_DIR / "figures"
TERRITORIES_PATH = RAW_DIR / "territories.parquet"
KMEANS_FINAL_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
REPORT_PATH = NOTEBOOKS_DIR / "11b_louvain_stability.md"

MONTH = step11.MONTH
WEIGHT = step11.WEIGHT
RESOLUTION = step11.RESOLUTION
NETWORK_PATH = step11.NETWORK_PATH
from config import LOUVAIN_SEEDS as SEEDS
from config import LOUVAIN_N_RANDOM_PAIRS as N_RANDOM_PAIRS  # непересекающиеся пары запусков для «типичного» ARI
from config import LOUVAIN_PAIRS_RNG_SEED as PAIRS_RNG_SEED  # воспроизводимый выбор пар
TOP_N_REGIONS = 3

TAG = MONTH.replace("-", "_")
RUNS_PATH = PROCESSED_DIR / f"louvain_runs_{TAG}.parquet"
RUNS_LABELS_PATH = PROCESSED_DIR / f"louvain_runs_labels_{TAG}.parquet"
FINAL_LABELS_PATH = PROCESSED_DIR / f"louvain_labels_{TAG}.parquet"
FINAL_SEED_PATH = PROCESSED_DIR / f"louvain_final_seed_{TAG}.txt"


def md_table(df: pd.DataFrame, index: bool = False, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def describe(s: pd.Series) -> pd.Series:
    return s.describe(percentiles=[.05, .25, .5, .75, .95])


def hist(values, bins, xlabel: str, title: str, path: Path, marks: dict | None = None,
         discrete: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if discrete:
        vc = pd.Series(values).value_counts().sort_index()
        ax.bar(vc.index, vc.values, color="#4C72B0", width=0.8)
        ax.set_xticks(vc.index)
    else:
        ax.hist(values, bins=bins, color="#4C72B0", edgecolor="white")
    for label, (x, color) in (marks or {}).items():
        ax.axvline(x, color=color, ls="--", lw=1.2, label=label)
    if marks:
        ax.legend(fontsize=8)
    ax.set(xlabel=xlabel, ylabel="число запусков / пар", title=title)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    edges = pd.read_parquet(NETWORK_PATH, engine="pyarrow")
    G = nx.Graph()
    G.add_weighted_edges_from(
        ((int(x), int(y), float(w)) for x, y, w in
         edges[["territory_id_x", "territory_id_y", WEIGHT]].itertuples(index=False)),
        weight=WEIGHT)
    nodes = np.array(sorted(G.nodes))

    # 1. 100 запусков
    runs, labels = [], {}
    for s in SEEDS:
        lab = step11.louvain_labels(G, s)
        comms = [{n for n, c in lab.items() if c == k} for k in range(max(lab.values()) + 1)]
        runs.append({"seed": s, "K": len(comms),
                     "MQ": nx.community.modularity(G, comms, weight=WEIGHT)})
        labels[s] = np.array([lab[n] for n in nodes])
    runs = pd.DataFrame(runs)
    runs.to_parquet(RUNS_PATH, engine="pyarrow", index=False)
    pd.DataFrame([(s, int(n), int(c)) for s, arr in labels.items() for n, c in zip(nodes, arr)],
                 columns=["seed", "territory_id", "cluster"]).to_parquet(
        RUNS_LABELS_PATH, engine="pyarrow", index=False)

    # 2. Случайные непересекающиеся пары
    rng = np.random.default_rng(PAIRS_RNG_SEED)
    pick = rng.choice(list(SEEDS), size=2 * N_RANDOM_PAIRS, replace=False)
    pairs = pick.reshape(-1, 2)
    ari_pairs = pd.Series([adjusted_rand_score(labels[a], labels[b]) for a, b in pairs],
                          name="ARI (случайные пары)")

    # 3. Лучший по MQ запуск (при равенстве — наименьший seed)
    best_mq = runs["MQ"].max()
    ties = runs[np.isclose(runs["MQ"], best_mq, rtol=0, atol=1e-12)]
    best_seed = int(ties["seed"].min())
    best = labels[best_seed]
    final = pd.DataFrame({"territory_id": nodes, "cluster": best})
    final.to_parquet(FINAL_LABELS_PATH, engine="pyarrow", index=False)
    FINAL_SEED_PATH.write_text(f"{best_seed}\n", encoding="utf-8")

    # 4. ARI лучшего против остальных 99
    others = runs[runs["seed"] != best_seed].copy()
    others["ARI с лучшим"] = [adjusted_rand_score(best, labels[s]) for s in others["seed"]]
    ari_best = others["ARI с лучшим"]
    # связь «чем ближе MQ к лучшему, тем похожее разбиение?»
    rho_mq_ari = others[["MQ", "ARI с лучшим"]].corr(method="spearman").iloc[0, 1]

    # 5. Метрики итогового разбиения
    lab_best = dict(zip(nodes.tolist(), best.tolist()))
    m_lv, pc_lv = partition_metrics(G, lab_best, weight=WEIGHT)

    # 6. Сравнение с каноническим KMeans
    km = pd.read_parquet(KMEANS_FINAL_PATH).set_index("territory_id")["cluster"].reindex(nodes)
    if km.isna().any():
        raise SystemExit("STOP: не у всех МО сети есть метка KMeans")
    m_km, _ = partition_metrics(G, km.to_dict(), weight=WEIGHT)
    ari_km = adjusted_rand_score(best, km.to_numpy())
    ari_km_all = pd.Series([adjusted_rand_score(labels[s], km.to_numpy()) for s in SEEDS])
    comp = pd.DataFrame([
        {"разбиение": f"Louvain, лучший по MQ (seed={best_seed})", **m_lv},
        {"разбиение": f"KMeans k={FINAL_K} (kmeans_labels_final)", **m_km},
    ])
    ct = pd.crosstab(pd.Series(best, name="Louvain"), pd.Series(km.to_numpy(), name=f"KMeans k={FINAL_K}"))
    purity = ct.max(axis=1).sum() / ct.to_numpy().sum()
    dom = pd.DataFrame({
        "МО": ct.sum(axis=1),
        "основной кластер KMeans": ct.idxmax(axis=1),
        "доля в нём": ct.max(axis=1) / ct.sum(axis=1),
    })

    # Состав сообществ итогового разбиения
    terr = pd.read_parquet(TERRITORIES_PATH).set_index("territory_id")
    comp_rows = []
    for c in sorted(set(best)):
        ids = nodes[best == c]
        regions = terr.loc[ids, "region_name"].value_counts()
        comp_rows.append({"cluster": c, "МО": len(ids), f"топ-{TOP_N_REGIONS} регионов":
                          "; ".join(f"{r} {n}" for r, n in regions.head(TOP_N_REGIONS).items()),
                          "регионов": len(regions)})
    lv_comp = pd.DataFrame(comp_rows).merge(pc_lv.drop(columns="узлов"), on="cluster")

    # Графики
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    hist(runs["K"], None, "K (число сообществ)", f"Louvain, {len(runs)} запусков: K",
         FIGURES_DIR / "louvain_K_hist.png",
         {f"лучший по MQ (K={m_lv['K']})": (m_lv["K"], "#C44E52")}, discrete=True)
    hist(runs["MQ"], 20, "MQ (модулярность)", f"Louvain, {len(runs)} запусков: MQ",
         FIGURES_DIR / "louvain_MQ_hist.png", {f"лучший (seed={best_seed})": (best_mq, "#C44E52")})
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(ari_best.min(), ari_pairs.min()) - .02,
                       max(ari_best.max(), ari_pairs.max()) + .02, 25)
    ax.hist(ari_best, bins=bins, alpha=.75, color="#4C72B0", label="лучший vs остальные 99")
    ax.hist(ari_pairs, bins=bins, alpha=.6, color="#DD8452",
            label=f"{N_RANDOM_PAIRS} случайных непересекающихся пар")
    ax.axvline(ari_km, color="#C44E52", ls="--", lw=1.2, label=f"лучший Louvain vs KMeans k={FINAL_K}")
    ax.set(xlabel="ARI", ylabel="число сравнений", title="Совпадение разбиений Louvain (ARI)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "louvain_ARI_hist.png", dpi=120)
    plt.close(fig)

    K_mode = runs["K"].mode().tolist()
    share_high = (ari_best >= 0.8).mean()
    lines = [
        f"# 11b. Устойчивость Louvain ({MONTH}): {len(runs)} запусков",
        "",
        "Сгенерировано `src/11b_louvain_stability.py`.",
        "",
        f"- Граф: `{NETWORK_PATH.relative_to(PROJECT_DIR)}` ({G.number_of_nodes()} МО, "
        f"{G.number_of_edges():,} рёбер), вес = {WEIGHT}; Louvain `networkx`, "
        f"resolution = {RESOLUTION}, seed = {SEEDS.start}..{SEEDS.stop - 1}.",
        f"- Все запуски: `{RUNS_PATH.relative_to(PROJECT_DIR)}` (seed, K, MQ), метки — "
        f"`{RUNS_LABELS_PATH.relative_to(PROJECT_DIR)}`.",
        "",
        "## 1–2. Распределение K и MQ",
        "",
        "![K](figures/louvain_K_hist.png)",
        "",
        "![MQ](figures/louvain_MQ_hist.png)",
        "",
        md_table(pd.DataFrame({"K": describe(runs["K"]), "MQ": describe(runs["MQ"])}),
                 index=True, fmt=",.4f"),
        "",
        f"- K: min {runs['K'].min()}, max {runs['K'].max()}, медиана {runs['K'].median():.0f}, "
        f"мода {', '.join(map(str, K_mode))}",
        f"- MQ: min {runs['MQ'].min():.4f}, max {runs['MQ'].max():.4f}, медиана "
        f"{runs['MQ'].median():.4f} (размах {runs['MQ'].max() - runs['MQ'].min():.4f})",
        "",
        "Частоты K:",
        "",
        md_table(runs["K"].value_counts().sort_index().rename("запусков").rename_axis("K")
                 .reset_index(), fmt=",.0f"),
        "",
        f"### ARI между независимыми запусками ({N_RANDOM_PAIRS} случайных непересекающихся пар, "
        f"выбор пар — rng seed {PAIRS_RNG_SEED})",
        "",
        md_table(describe(ari_pairs).to_frame(), index=True),
        "",
        "## 3. Итоговое разбиение: лучший запуск по MQ",
        "",
        f"- **seed = {best_seed}** (сохранён в `{FINAL_SEED_PATH.relative_to(PROJECT_DIR)}`); "
        f"запусков с тем же максимальным MQ: {len(ties)}",
        f"- метки: `{FINAL_LABELS_PATH.relative_to(PROJECT_DIR)}` (заменяет прежнее разбиение "
        "seed = 42; оно сохранено как `louvain_labels_2024_12_seed42.parquet`)",
        "",
        "## 5. Метрики итогового разбиения",
        "",
        md_table(pd.DataFrame([{"seed": best_seed, **m_lv}]), fmt=",.4f"),
        "",
        "Формулы — `src/partition_metrics.py` без изменений. AVU нечувствителен к объёму "
        "межкластерных рёбер (см. `11_louvain_metrics.md`, раздел проверки) — интерпретировать "
        "как меру концентрации связей между парами кластеров.",
        "",
        md_table(lv_comp, fmt=",.3f"),
        "",
        "## 4. Устойчивость итогового разбиения: ARI лучшего против остальных 99",
        "",
        "![ARI](figures/louvain_ARI_hist.png)",
        "",
        md_table(describe(ari_best).to_frame(), index=True),
        "",
        f"- Медианный ARI с остальными запусками: **{ari_best.median():.3f}** "
        f"(межквартильный размах {ari_best.quantile(.25):.3f}–{ari_best.quantile(.75):.3f}, "
        f"min {ari_best.min():.3f}, max {ari_best.max():.3f}); запусков с ARI ≥ 0.8: "
        f"{share_high:.0%}.",
        f"- Связь MQ запуска и его ARI с лучшим (Спирмен): {rho_mq_ari:+.3f}.",
        "",
        "**Ограничение метода.** "
        + (
            "Итоговое разбиение не воспроизводится независимыми запусками: типичный запуск "
            f"совпадает с ним лишь на ARI ≈ {ari_best.median():.2f}, а независимые запуски "
            f"между собой — на ARI ≈ {ari_pairs.median():.2f}. При этом MQ почти одинаков у "
            f"всех запусков (размах {runs['MQ'].max() - runs['MQ'].min():.4f}): у графа много "
            "разных разбиений почти равной модулярности (вырожденный ландшафт модулярности). "
            f"Лучший запуск выше медианы по MQ лишь на {best_mq - runs['MQ'].median():.4f} "
            f"({(best_mq - runs['MQ'].mean()) / runs['MQ'].std():.1f} ст. откл.). Частично это "
            f"смягчается тем, что запуски с более высоким MQ ближе к лучшему (Спирмен "
            f"{rho_mq_ari:+.2f}), — т. е. в области высокой модулярности разбиения сходятся, но "
            "даже ближайший запуск совпадает с лучшим не полностью "
            f"(max ARI {ari_best.max():.2f}). "
            "Поэтому границы и число сообществ Louvain (K от "
            f"{runs['K'].min()} до {runs['K'].max()}) нельзя интерпретировать как устойчивую "
            "структуру на уровне отдельных МО; устойчивыми можно считать только крупные "
            "блоки, которые повторяются во всех запусках (проверить отдельно, например по "
            "матрице совместной встречаемости)."
            if ari_best.median() < 0.8 else
            f"Итоговое разбиение хорошо воспроизводится (медианный ARI {ari_best.median():.2f})."
        ),
        "",
        f"## 6. Сравнение с KMeans k = {FINAL_K}",
        "",
        f"- ARI(итоговый Louvain, KMeans k={FINAL_K}) = **{ari_km:.3f}**; по всем 100 запускам Louvain: "
        f"медиана {ari_km_all.median():.3f}, диапазон {ari_km_all.min():.3f}–{ari_km_all.max():.3f}.",
        f"- ARI занижен разницей в числе кластеров ({m_lv['K']} против {m_km['K']}): он штрафует "
        "и вложенные разбиения. Мера вложенности — доля МО, попавших в «основной» кластер "
        f"KMeans своего сообщества Louvain (взвешенная чистота): **{purity:.3f}**. То есть "
        "сообщества Louvain в основном — части кластеров KMeans, а не поперечное разбиение.",
        "",
        md_table(comp, fmt=",.4f"),
        "",
        "MQ, AVI, AVU для KMeans посчитаны на том же графе (как в шаге 11). Louvain "
        "оптимизирует модулярность прямо на графе, KMeans — евклидово расстояние в пространстве "
        "признаков, поэтому преимущество Louvain по MQ/AVI ожидаемо и не означает лучшую "
        "типологию само по себе.",
        "",
        "Какому кластеру KMeans соответствует каждое сообщество Louvain:",
        "",
        md_table(dom.reset_index().rename(columns={"Louvain": "сообщество Louvain"}), fmt=",.2f"),
        "",
        f"Таблица сопряжённости (строки — Louvain, столбцы — KMeans k={FINAL_K}):",
        "",
        md_table(ct, index=True, fmt=",.0f"),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(runs.describe().to_string())
    print(f"best seed {best_seed}, ties {len(ties)}; metrics {m_lv}")
    print("ARI best vs others:", describe(ari_best).round(3).to_dict())
    print("ARI random pairs:", describe(ari_pairs).round(3).to_dict())
    print(f"ARI best vs KMeans: {ari_km:.3f}; all runs median {ari_km_all.median():.3f}")
    print(comp.to_string(index=False))
    print(dom.to_string())


if __name__ == "__main__":
    main()
