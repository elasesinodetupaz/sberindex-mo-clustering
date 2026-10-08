"""Шаг 11c. Устойчивость сообществ итогового Louvain по матрице совместной встречаемости.

co(i, j) = доля из 100 запусков Louvain, в которых МО i и j попали в одно сообщество.
Матрица считается разреженно: C = M·Mᵀ / R, где M — объединённая one-hot матрица
принадлежности (N × Σ_r K_r) по всем R запускам.

Вход:  data/processed/louvain_runs_labels_<YYYY_MM>.parquet, louvain_labels_<YYYY_MM>.parquet,
       louvain_final_seed_<YYYY_MM>.txt, kmeans_labels_final.parquet, category_shares.parquet,
       data/raw/territories.parquet
Выход: data/processed/louvain_cooccurrence_<YYYY_MM>.parquet (territory_id_x < territory_id_y,
         co — только пары, хотя бы раз оказавшиеся вместе),
       data/processed/louvain_node_stability_<YYYY_MM>.parquet (по МО),
       notebooks/11c_louvain_core_stability.md
Запуск из корня проекта:  .venv/bin/python src/11c_louvain_core_stability.py
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

from config import FINAL_K
from network_utils import SHARE_COLUMNS, month_shares, standardize_shares

step10b = importlib.import_module("10b_kmeans_cluster_profiles")

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TERRITORIES_PATH = RAW_DIR / "territories.parquet"
KMEANS_FINAL_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "11c_louvain_core_stability.md"

from config import MONTH
TAG = MONTH.replace("-", "_")
RUNS_LABELS_PATH = PROCESSED_DIR / f"louvain_runs_labels_{TAG}.parquet"
FINAL_LABELS_PATH = PROCESSED_DIR / f"louvain_labels_{TAG}.parquet"
FINAL_SEED_PATH = PROCESSED_DIR / f"louvain_final_seed_{TAG}.txt"
COOC_PATH = PROCESSED_DIR / f"louvain_cooccurrence_{TAG}.parquet"
NODE_STAB_PATH = PROCESSED_DIR / f"louvain_node_stability_{TAG}.parquet"

TOP_N = 5                 # топ устойчивых / нестабильных сообществ
from config import LOUVAIN_CORE_THRESHOLD as CORE_THRESHOLD  # МО — «ядро» сообщества, если его средняя co с сообществом ≥ порога
N_CORE_EXAMPLES = 10      # сколько МО ядра показывать
TOP_N_REGIONS = 5


def md_table(df: pd.DataFrame, index: bool = False, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def main() -> None:
    runs = pd.read_parquet(RUNS_LABELS_PATH)
    final = pd.read_parquet(FINAL_LABELS_PATH).set_index("territory_id")["cluster"]
    best_seed = int(FINAL_SEED_PATH.read_text().strip())
    nodes = np.array(sorted(runs["territory_id"].unique()))
    pos = pd.Series(np.arange(len(nodes)), index=nodes)
    R = runs["seed"].nunique()
    if set(nodes) != set(final.index):
        raise SystemExit("STOP: МО в запусках и в итоговом разбиении не совпадают")
    chk = runs[runs["seed"] == best_seed].set_index("territory_id")["cluster"].reindex(final.index)
    if not (chk == final).all():
        raise SystemExit(f"STOP: итоговое разбиение не совпадает с запуском seed={best_seed}")

    # 1. Разреженная co-occurrence
    col_offset = runs.groupby("seed")["cluster"].max().add(1).cumsum().shift(fill_value=0)
    cols = runs["cluster"].to_numpy() + runs["seed"].map(col_offset).to_numpy()
    M = sparse.csr_matrix((np.ones(len(runs)), (pos[runs["territory_id"]].to_numpy(), cols)),
                          shape=(len(nodes), int(cols.max()) + 1))
    C = (M @ M.T).tocoo()
    up = C.row < C.col
    cooc = pd.DataFrame({"territory_id_x": nodes[C.row[up]], "territory_id_y": nodes[C.col[up]],
                         "co": C.data[up] / R})
    cooc.to_parquet(COOC_PATH, engine="pyarrow", index=False)
    Cd = (M @ M.T).toarray() / R      # 2004×2004 float — для агрегатов по сообществам (≈32 МБ)

    # 2. Устойчивость сообществ итогового разбиения
    lab = final.reindex(nodes).to_numpy()
    km = pd.read_parquet(KMEANS_FINAL_PATH).set_index("territory_id")["cluster"].reindex(nodes)
    terr = pd.read_parquet(TERRITORIES_PATH).set_index("territory_id")

    node_rows, comm_rows = [], []
    for c in sorted(set(lab)):
        idx = np.where(lab == c)[0]
        sub = Cd[np.ix_(idx, idx)]
        n = len(idx)
        iu = np.triu_indices(n, k=1)
        inner = sub[iu]
        # средняя co каждого члена с остальными членами сообщества
        node_mean = (sub.sum(axis=1) - 1.0) / (n - 1)
        out_idx = np.where(lab != c)[0]
        node_out_max = Cd[np.ix_(idx, out_idx)].max(axis=1)
        for i, v, o in zip(idx, node_mean, node_out_max):
            node_rows.append({"territory_id": nodes[i], "louvain": c, "kmeans": int(km.iloc[i]),
                              "co со своим сообществом": v, "max co вне сообщества": o})
        kmc = km.iloc[idx].value_counts()
        comm_rows.append({
            "cluster": c, "МО": n,
            "средняя внутр. co": inner.mean(),
            "медиана внутр. co": np.median(inner),
            "доля пар co ≥ 0.9": (inner >= 0.9).mean(),
            f"МО в ядре (co ≥ {CORE_THRESHOLD})": int((node_mean >= CORE_THRESHOLD).sum()),
            "основной кластер KMeans": int(kmc.index[0]),
            "доля в нём": kmc.iloc[0] / n,
            "регионов": terr.loc[nodes[idx], "region_name"].nunique(),
            f"топ-3 регионов": "; ".join(f"{r} {k}" for r, k in
                                         terr.loc[nodes[idx], "region_name"].value_counts()
                                         .head(3).items()),
        })
    node_stab = pd.DataFrame(node_rows)
    node_stab = node_stab.join(terr[["name", "region_name"]], on="territory_id")
    node_stab.to_parquet(NODE_STAB_PATH, engine="pyarrow", index=False)
    comm = (pd.DataFrame(comm_rows).sort_values("средняя внутр. co", ascending=False)
            .reset_index(drop=True))
    comm.insert(0, "ранг", np.arange(1, len(comm) + 1))

    # Профили канонического KMeans (подписи из шага 10b)
    shares = month_shares(pd.read_parquet(SHARES_PATH), MONTH).set_index("territory_id")
    X, _ = standardize_shares(shares[SHARE_COLUMNS].to_numpy())
    z = pd.DataFrame(X, index=shares.index, columns=SHARE_COLUMNS)
    km_prof = z.join(km.rename("k")).groupby("k")[SHARE_COLUMNS].mean().apply(
        step10b.describe_profile, axis=1)
    km_reg = {k: "; ".join(f"{r} {n}" for r, n in
                          terr.loc[km[km == k].index, "region_name"].value_counts().head(3).items())
              for k in sorted(km.unique())}
    comm["профиль кластера KMeans"] = comm["основной кластер KMeans"].map(km_prof)

    # Baseline: co для случайной пары МО
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(nodes), 20000), rng.integers(0, len(nodes), 20000)
    base = Cd[a[a != b], b[a != b]].mean()
    rho, p = stats.spearmanr(comm["средняя внутр. co"], comm["доля в нём"])
    rho_n, p_n = stats.spearmanr(comm["средняя внутр. co"], comm["МО"])

    def community_block(row: pd.Series) -> list[str]:
        c = row["cluster"]
        members = node_stab[node_stab["louvain"] == c].sort_values(
            "co со своим сообществом", ascending=False)
        regions = members["region_name"].value_counts()
        core = members.head(N_CORE_EXAMPLES)[["territory_id", "name", "region_name",
                                                "co со своим сообществом"]]
        return [
            f"#### Сообщество {c} — {row['МО']} МО, средняя внутр. co = "
            f"{row['средняя внутр. co']:.3f}",
            "",
            f"- KMeans: {row['доля в нём']:.0%} в кластере {row['основной кластер KMeans']} "
            f"({row['профиль кластера KMeans']})",
            f"- МО в ядре (co ≥ {CORE_THRESHOLD}): {row[f'МО в ядре (co ≥ {CORE_THRESHOLD})']} из "
            f"{row['МО']}",
            f"- регионы ({len(regions)}): " + "; ".join(
                f"{r} — {n}" for r, n in regions.head(TOP_N_REGIONS).items()),
            "",
            f"Самые «прочные» члены (топ-{N_CORE_EXAMPLES} по co со своим сообществом):",
            "",
            md_table(core.astype({"territory_id": str})),
            "",
        ]

    show_cols = ["ранг", "cluster", "МО", "средняя внутр. co", "медиана внутр. co",
                 "доля пар co ≥ 0.9", f"МО в ядре (co ≥ {CORE_THRESHOLD})", "регионов",
                 "топ-3 регионов", "основной кластер KMeans", "доля в нём"]
    km_map = (comm.groupby("основной кластер KMeans")
              .apply(lambda g: ", ".join(f"{c} ({v:.2f})" for c, v in
                                         zip(g["cluster"], g["средняя внутр. co"])),
                     include_groups=False))
    km_table = pd.DataFrame({
        "кластер KMeans": sorted(km.unique()),
        "МО": [int((km == k).sum()) for k in sorted(km.unique())],
        "профиль": [km_prof[k] for k in sorted(km.unique())],
        "топ-3 регионов": [km_reg[k] for k in sorted(km.unique())],
        "сообщества Louvain (средняя внутр. co)": [km_map.get(k, "—") for k in sorted(km.unique())],
    })

    lines = [
        f"# 11c. Устойчивость сообществ Louvain ({MONTH}) по совместной встречаемости",
        "",
        "Сгенерировано `src/11c_louvain_core_stability.py`.",
        "",
        f"- co(i, j) — доля из {R} запусков Louvain, где МО i и j в одном сообществе. "
        f"Разреженная матрица: `{COOC_PATH.relative_to(PROJECT_DIR)}` — {len(cooc):,} пар "
        f"с co > 0 из {len(nodes) * (len(nodes) - 1) // 2:,} ({len(cooc) / (len(nodes) * (len(nodes) - 1) / 2):.1%}).",
        f"- Итоговое разбиение: seed = {best_seed}, K = {len(comm)}.",
        "- Устойчивость сообщества — средняя co по всем парам его членов (1 = состав "
        "держится вместе во всех запусках).",
        f"- Для сравнения: средняя co случайной пары МО = {base:.3f}.",
        f"- По МО: `{NODE_STAB_PATH.relative_to(PROJECT_DIR)}` (co со своим сообществом, max co "
        "вне сообщества, кластер KMeans, название, регион).",
        "",
        "## Рейтинг сообществ по устойчивости",
        "",
        md_table(comm[show_cols], fmt=",.3f"),
        "",
        f"## Топ-{TOP_N} самых устойчивых сообществ",
        "",
    ]
    for _, row in comm.head(TOP_N).iterrows():
        lines += community_block(row)
    lines += [f"## Топ-{TOP_N} самых нестабильных сообществ", ""]
    for _, row in comm.tail(TOP_N).iloc[::-1].iterrows():
        lines += community_block(row)
    lines += [
        f"## Сопоставление с KMeans k = {FINAL_K}",
        "",
        "Какие сообщества Louvain лежат в каждом кластере KMeans (по основному кластеру "
        "сообщества; в скобках — средняя внутренняя co):",
        "",
        md_table(km_table),
        "",
        f"- Связь устойчивости сообщества и его «чистоты» относительно KMeans (доля МО в "
        f"основном кластере KMeans), Спирмен по {len(comm)} сообществам: ρ = {rho:+.2f} "
        f"(p = {p:.2g}).",
        f"- Связь устойчивости и размера сообщества (средняя по парам может быть выше у малых "
        f"сообществ): ρ = {rho_n:+.2f} (p = {p_n:.2g}).",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"pairs co>0: {len(cooc):,}; baseline {base:.3f}; rho {rho:+.2f} p {p:.2g}")
    print(comm[show_cols + ["профиль кластера KMeans"]].to_string(index=False))
    print(km_table.to_string(index=False))


if __name__ == "__main__":
    main()
