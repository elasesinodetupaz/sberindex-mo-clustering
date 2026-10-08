"""Шаг 13. Louvain по всем 24 месяцам (облегчённо) и сравнение с каноническим KMeans того же месяца.

Для каждого месяца: экономическая сеть (z-score + cosine + kNN k=8, шаг 06), Louvain
N_SEEDS раз (seed 0..N_SEEDS-1), выбор разбиения с максимальной модулярностью; MQ/AVI/AVU
(src/partition_metrics.py); ARI и вложенность относительно официального KMeans k=config.FINAL_K месяца
(шаг 12a --k FINAL_K, согласованная нумерация — шаг 12). Полный анализ устойчивости и трекинг сообществ
Louvain во времени сознательно не делаются (см. 11b/11c для декабря 2024).

Выход: data/processed/louvain_monthly_summary.parquet, data/processed/louvain_monthly_labels.parquet
       (month, territory_id, cluster), notebooks/13_louvain_monthly_check.md,
       notebooks/figures/louvain_monthly_K_MQ.png
Запуск из корня проекта:  .venv/bin/python src/13_louvain_monthly_check.py
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
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
NETWORKS_DIR = PROCESSED_DIR / "economic_networks"
TRAJ_PATH = PROCESSED_DIR / f"kmeans_k{FINAL_K}_trajectories.parquet"
DEC_LOUVAIN_100_PATH = PROCESSED_DIR / "louvain_labels_2024_12.parquet"   # шаг 11b (100 seed)
DEC_SEED_PATH = PROCESSED_DIR / "louvain_final_seed_2024_12.txt"
KMEANS_CANONICAL_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
SUMMARY_PATH = PROCESSED_DIR / "louvain_monthly_summary.parquet"
LABELS_PATH = PROCESSED_DIR / "louvain_monthly_labels.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "13_louvain_monthly_check.md"
FIG_PATH = PROJECT_DIR / "notebooks" / "figures" / "louvain_monthly_K_MQ.png"

from config import ECON_KNN_K as K_NN
from config import LOUVAIN_MONTHLY_N_SEEDS as N_SEEDS
WEIGHT = step11.WEIGHT
from config import NEST_PURITY  # сообщество «вложено», если ≥ 80% его МО — в одном кластере KMeans
# «картина декабря типична», если его показатели в пределах TYPICAL_Z ст. откл. от среднего по месяцам
from config import TYPICAL_Z


def md_table(df: pd.DataFrame, index: bool = False, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def nesting(louv: np.ndarray, km: np.ndarray) -> dict:
    ct = pd.crosstab(louv, km)
    purity = ct.max(axis=1) / ct.sum(axis=1)
    top2 = np.sort((ct.T / ct.sum(axis=1)).T.to_numpy(), axis=1)[:, -2:].sum(axis=1)
    mixed = (purity < NEST_PURITY).to_numpy()
    return {"_mixed": int(mixed.sum()), "_mixed_two_types": int((top2[mixed] >= NEST_PURITY).sum()),"ARI с KMeans": adjusted_rand_score(louv, km),
            f"доля сообществ с чистотой ≥ {NEST_PURITY:.0%}": (purity >= NEST_PURITY).mean(),
            f"МО в сообществах с чистотой ≥ {NEST_PURITY:.0%}":
                ct.sum(axis=1)[purity >= NEST_PURITY].sum() / ct.to_numpy().sum(),
            "взвешенная чистота": ct.max(axis=1).sum() / ct.to_numpy().sum(),
            "кластеров KMeans, раздробленных на ≥ 2 сообщества (≥10% кластера)":
                int(((ct / ct.sum(axis=0) >= 0.1).sum(axis=0) >= 2).sum())}


def main() -> None:
    traj = pd.read_parquet(TRAJ_PATH)
    months = sorted(traj["month"].unique())
    rows, all_labels = [], []
    for m in months:
        edges = pd.read_parquet(NETWORKS_DIR / f"{m}_std_knn{K_NN}.parquet")
        G = nx.Graph()
        G.add_weighted_edges_from(
            ((int(x), int(y), float(w)) for x, y, w in
             edges[["territory_id_x", "territory_id_y", WEIGHT]].itertuples(index=False)),
            weight=WEIGHT)
        nodes = np.array(sorted(G.nodes))
        best = None
        mqs = []
        for s in range(N_SEEDS):
            lab = step11.louvain_labels(G, s)
            comms = [{n for n, c in lab.items() if c == k} for k in range(max(lab.values()) + 1)]
            mq = nx.community.modularity(G, comms, weight=WEIGHT)
            mqs.append(mq)
            if best is None or round(mq, 12) > round(best[1], 12):
                best = (s, mq, lab)
        seed, _, lab = best
        metrics, _ = partition_metrics(G, lab, weight=WEIGHT)
        louv = np.array([lab[n] for n in nodes])
        km = traj[traj["month"] == m].set_index("territory_id").loc[nodes, "cluster"].to_numpy()
        rows.append({"месяц": m, "seed": seed, "K": metrics["K"], "MQ": metrics["MQ"],
                     "AVI": metrics["AVI"], "AVU": metrics["AVU"],
                     "MQ по 20 seed: min": min(mqs), "MQ по 20 seed: max": max(mqs),
                     **nesting(louv, km)})
        all_labels.append(pd.DataFrame({"month": m, "territory_id": nodes, "cluster": louv}))
        print(m, rows[-1]["K"], round(rows[-1]["MQ"], 4), round(rows[-1]["ARI с KMeans"], 3),
              round(rows[-1]["взвешенная чистота"], 3))
    res = pd.DataFrame(rows)
    mixed_total, mixed_two = int(res.pop("_mixed").sum()), int(res.pop("_mixed_two_types").sum())
    res.to_parquet(SUMMARY_PATH, engine="pyarrow", index=False)
    pd.concat(all_labels, ignore_index=True).to_parquet(LABELS_PATH, engine="pyarrow", index=False)

    # Справка: декабрь из шага 11b (лучший из 100 seed) против канонического KMeans
    dec_ref = None
    if DEC_LOUVAIN_100_PATH.exists():
        l100 = pd.read_parquet(DEC_LOUVAIN_100_PATH).set_index("territory_id")["cluster"]
        kmc = pd.read_parquet(KMEANS_CANONICAL_PATH).set_index("territory_id")["cluster"].loc[l100.index]
        kmo = traj[traj["month"] == "2024-12"].set_index("territory_id").loc[l100.index, "cluster"]
        dec_ref = pd.DataFrame([
            {"вариант": f"11b: Louvain seed {DEC_SEED_PATH.read_text().strip()} (лучший из 100) vs "
                        "канонический KMeans (seed 42)",
             **{k: v for k, v in nesting(l100.to_numpy(), kmc.to_numpy()).items() if not k.startswith("_")}},
            {"вариант": "11b: Louvain (лучший из 100) vs официальный KMeans (шаг 12a)",
             **{k: v for k, v in nesting(l100.to_numpy(), kmo.to_numpy()).items() if not k.startswith("_")}},
            {"вариант": f"этот шаг: Louvain (лучший из {N_SEEDS}) vs официальный KMeans",
             **{k: v for k, v in res[res["месяц"] == "2024-12"].iloc[0].items() if k in
                nesting(np.zeros(2), np.zeros(2)) and not k.startswith("_")}},
        ])

    # График K и MQ
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(res["месяц"], res["K"], marker="o", color="#4C72B0", label="K")
    ax1.set_ylabel("K (число сообществ)", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(res["месяц"], res["MQ"], marker="s", color="#C44E52", label="MQ")
    ax2.set_ylabel("MQ (модулярность)", color="#C44E52")
    ax1.set_xticks(range(len(months)))
    ax1.set_xticklabels(months, rotation=60, ha="right", fontsize=8)
    ax1.grid(alpha=.3)
    ax1.set_title(f"Louvain по месяцам (лучший по MQ из {N_SEEDS} seed)")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=120)
    plt.close(fig)

    nest_col = f"доля сообществ с чистотой ≥ {NEST_PURITY:.0%}"
    zcols = ["ARI с KMeans", "взвешенная чистота", nest_col, "K"]
    dec_z = {c: (res.loc[res["месяц"] == "2024-12", c].iloc[0] - res[c].mean()) / res[c].std()
             for c in zcols}
    typical = all(abs(v) <= TYPICAL_Z for v in dec_z.values())
    desc = res[["K", "MQ", "AVI", "AVU", "ARI с KMeans", nest_col, "взвешенная чистота"]].describe(
    ).loc[["mean", "std", "min", "50%", "max"]]
    dec = res[res["месяц"] == "2024-12"].iloc[0]
    rank_ari = int((res["ARI с KMeans"] < dec["ARI с KMeans"]).sum()) + 1
    rank_pur = int((res["взвешенная чистота"] > dec["взвешенная чистота"]).sum()) + 1

    lines = [
        f"# 13. Louvain по месяцам: облегчённая проверка и сравнение с KMeans k = {FINAL_K}",
        "",
        "Сгенерировано `src/13_louvain_monthly_check.py`.",
        "",
        f"- Для каждого месяца: экономическая сеть `economic_networks/{{YYYY-MM}}_std_knn{K_NN}.parquet`, "
        f"Louvain {N_SEEDS} раз (seed 0..{N_SEEDS - 1}), resolution = {step11.RESOLUTION}, вес — "
        f"{WEIGHT}; выбрано разбиение с максимальной MQ. Метки: "
        f"`{LABELS_PATH.relative_to(PROJECT_DIR)}`, сводка: `{SUMMARY_PATH.relative_to(PROJECT_DIR)}`.",
        "- KMeans — официальное разбиение месяца (min inertia из 100 запусков, шаг 12a) в "
        "согласованной нумерации (шаг 12).",
        f"- Вложенность: доля сообществ Louvain, у которых ≥ {NEST_PURITY:.0%} МО из одного "
        "кластера KMeans (как задано), и взвешенная чистота — доля МО в «основном» кластере KMeans "
        "своего сообщества (метрика из 11b для декабря).",
        "- Осознанно не делались: анализ устойчивости Louvain на 100 seed, co-occurrence, трекинг "
        "сообществ во времени (выполнены только для декабря 2024 — шаги 11b, 11c).",
        "",
        "## K, MQ, AVI, AVU по месяцам",
        "",
        "![K и MQ](figures/louvain_monthly_K_MQ.png)",
        "",
        md_table(res[["месяц", "seed", "K", "MQ", "AVI", "AVU", "MQ по 20 seed: min",
                      "MQ по 20 seed: max"]], fmt=",.4f"),
        "",
        f"## Сравнение с KMeans k = {FINAL_K} того же месяца",
        "",
        md_table(res[["месяц", "K", "ARI с KMeans", nest_col,
                      f"МО в сообществах с чистотой ≥ {NEST_PURITY:.0%}", "взвешенная чистота",
                      "кластеров KMeans, раздробленных на ≥ 2 сообщества (≥10% кластера)"]],
                 fmt=",.3f"),
        "",
        "Сводно по 24 месяцам:",
        "",
        md_table(desc, index=True, fmt=",.3f"),
        "",
    ]
    if dec_ref is not None:
        lines += [
            "Справка — декабрь 2024 в разных вариантах (разница из-за числа seed Louvain и выбора "
            "разбиения KMeans):",
            "",
            md_table(dec_ref, fmt=",.3f"),
            "",
        ]
    lines += [
        "## Вывод",
        "",
        f"- K: {res['K'].min()}–{res['K'].max()} (медиана {res['K'].median():.0f}); MQ: "
        f"{res['MQ'].min():.3f}–{res['MQ'].max():.3f}. ARI Louvain–KMeans: "
        f"{res['ARI с KMeans'].min():.3f}–{res['ARI с KMeans'].max():.3f} (медиана "
        f"{res['ARI с KMeans'].median():.3f}); взвешенная чистота: "
        f"{res['взвешенная чистота'].min():.3f}–{res['взвешенная чистота'].max():.3f} (медиана "
        f"{res['взвешенная чистота'].median():.3f}); доля сообществ с чистотой ≥ {NEST_PURITY:.0%}: "
        f"{res[nest_col].min():.2f}–{res[nest_col].max():.2f} (медиана {res[nest_col].median():.2f}).",
        f"- Декабрь 2024 среди 24 месяцев: {rank_ari}-й по наименьшему ARI и {rank_pur}-й по "
        "наибольшей взвешенной чистоте.",
        "- Отклонение декабря от среднего по месяцам (в ст. откл.): "
        + ", ".join(f"{c} {v:+.1f}" for c, v in dec_z.items()) + ".",
        f"- Разброс по месяцам мал: ст. откл. ARI {res['ARI с KMeans'].std():.3f}, взвешенной "
        f"чистоты {res['взвешенная чистота'].std():.3f}.",
        "",
        "**" + ("Картина декабря типична для всего периода, а не особенность декабря" if typical else
                "Декабрь выделяется на фоне остальных месяцев") + ".** "
        + (f"Во всех 24 месяцах Louvain даёт более дробное разбиение (K ≈ {res['K'].median():.0f} "
           f"против {FINAL_K}), которое слабо совпадает с KMeans по ARI (≈ {res['ARI с KMeans'].median():.2f}), "
           f"но в основном вложено в его кластеры: ≈ {res['взвешенная чистота'].median():.0%} МО "
           "находятся в «основном» кластере KMeans своего сообщества. Вложенность при этом "
           f"**частичная**: строгому критерию (≥ {NEST_PURITY:.0%} МО из одного кластера) отвечает "
           f"лишь около {res[nest_col].median():.0%} сообществ. Остальные в основном стыкуют два типа "
           f"KMeans: у {mixed_two / max(mixed_total, 1):.0%} из {mixed_total} таких сообществ (все месяцы) "
           f"≥ {NEST_PURITY:.0%} МО приходятся на два кластера KMeans. Итоговая формулировка для всего периода: сообщества "
           "Louvain в основном — части типов KMeans или стыки двух соседних типов, а не "
           "поперечное деление; это свойство сочетания методов, а не особенность декабря."
           if typical else
           "См. показатели декабря выше: декабрьский вывод нельзя переносить на весь период без "
           "оговорок."),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(desc.to_string())
    if dec_ref is not None:
        print(dec_ref.to_string(index=False))
    print("dec z", {k: round(v, 2) for k, v in dec_z.items()}, "typical", typical)


if __name__ == "__main__":
    main()
