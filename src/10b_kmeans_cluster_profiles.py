"""Шаг 10b. Финальные KMeans для k-кандидатов: метки, профили долей, региональный состав.

Параметры KMeans (месяц, random_state, n_init) берутся из шага 10, чтобы совпадать с ним.

Вход:  data/processed/category_shares.parquet, data/raw/territories.parquet,
       data/raw/market_access.parquet, data/processed/railway_degree.parquet,
       data/processed/kmeans_k_selection_<YYYY_MM>.parquet (сверка inertia)
Выход: data/processed/kmeans_labels_k{N}_<YYYY_MM>.parquet (territory_id, cluster),
       notebooks/10b_kmeans_cluster_profiles.md
Запуск из корня проекта:  .venv/bin/python src/10b_kmeans_cluster_profiles.py
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from network_utils import SHARE_COLUMNS, month_shares, standardize_shares

step10 = importlib.import_module("10_kmeans_k_selection")

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
TERRITORIES_PATH = RAW_DIR / "territories.parquet"
MARKET_ACCESS_PATH = RAW_DIR / "market_access.parquet"
RAILWAY_DEGREE_PATH = PROCESSED_DIR / "railway_degree.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "10b_kmeans_cluster_profiles.md"

MONTH = step10.MONTH
RANDOM_STATE = step10.RANDOM_STATE
N_INIT = step10.N_INIT
from config import STABILITY_KS as K_CANDIDATES
TOP_N_EXAMPLES = 10
TOP_N_REGIONS = 8
# порог (в стандартных отклонениях) для автоматической подписи профиля кластера
PROFILE_Z_THRESHOLD = 0.5

TAG = MONTH.replace("-", "_")
SHORT = {c: c.replace("share_", "") for c in SHARE_COLUMNS}


def md_table(df: pd.DataFrame, index: bool = True, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def describe_profile(z_mean: pd.Series) -> str:
    """Короткая подпись: категории, заметно отклоняющиеся от среднего по всем МО."""
    parts = [f"{'↑' if v > 0 else '↓'} {SHORT[c]} ({v:+.1f}σ)"
             for c, v in z_mean.reindex(z_mean.abs().sort_values(ascending=False).index).items()
             if abs(v) >= PROFILE_Z_THRESHOLD]
    return ", ".join(parts) if parts else "близко к среднему по всем категориям"


def main() -> None:
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    month = month_shares(shares, MONTH).reset_index(drop=True)
    X, _ = standardize_shares(month[SHARE_COLUMNS].to_numpy())

    terr = pd.read_parquet(TERRITORIES_PATH, engine="pyarrow")
    ma = pd.read_parquet(MARKET_ACCESS_PATH, engine="pyarrow")
    rd = pd.read_parquet(RAILWAY_DEGREE_PATH, engine="pyarrow")[["territory_id", "has_railway"]]
    base = (month[["territory_id"] + SHARE_COLUMNS]
            .merge(terr[["territory_id", "name", "municipal_district_type", "region_name"]],
                   on="territory_id", how="left")
            .merge(ma, on="territory_id", how="left")
            .merge(rd, on="territory_id", how="left"))
    if base[["name", "market_access", "has_railway"]].isna().any().any():
        raise SystemExit("STOP: не у всех МО есть название / market_access / has_railway")
    z = pd.DataFrame(X, columns=SHARE_COLUMNS)

    ref = pd.read_parquet(PROCESSED_DIR / f"kmeans_k_selection_{TAG}.parquet").set_index("k")
    overall = base[SHARE_COLUMNS].mean()

    lines = [
        f"# 10b. Профили кластеров KMeans ({MONTH}), k = {', '.join(map(str, K_CANDIDATES))}",
        "",
        "Сгенерировано `src/10b_kmeans_cluster_profiles.py`. KMeans на 5 z-score долях "
        f"(random_state={RANDOM_STATE}, n_init={N_INIT} — как в шаге 10).",
        "",
        "- Профиль кластера — средние **исходные** доли (от «Все категории»), рядом — "
        "отклонение от среднего по всем МО в стандартных отклонениях (σ, z-score).",
        f"- Подпись профиля: категории с |z| ≥ {PROFILE_Z_THRESHOLD}σ.",
        f"- Примеры МО: топ-{TOP_N_EXAMPLES} по market_access внутри кластера.",
        "",
        "Средние доли по всем МО:",
        "",
        md_table(overall.rename(SHORT).rename("доля").to_frame().T, index=False),
        "",
    ]

    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X)
        if not np.isclose(km.inertia_, ref.loc[k, "inertia"]):
            raise SystemExit(f"STOP: k={k}: inertia {km.inertia_:.4f} не совпадает с шагом 10 "
                             f"({ref.loc[k, 'inertia']:.4f})")
        df = base.assign(cluster=km.labels_)
        df[["territory_id", "cluster"]].to_parquet(
            PROCESSED_DIR / f"kmeans_labels_k{k}_{TAG}.parquet", engine="pyarrow", index=False)

        zc = z.assign(cluster=km.labels_).groupby("cluster")[SHARE_COLUMNS].mean()
        g = df.groupby("cluster")
        summary = pd.DataFrame({
            "МО": g.size(),
            "подпись": zc.apply(describe_profile, axis=1),
            "market_access медиана": g["market_access"].median(),
            "с ж/д": g["has_railway"].mean(),
            "регионов": g["region_name"].nunique(),
        })
        prof = g[SHARE_COLUMNS].mean().rename(columns=SHORT)
        prof["сумма 5 долей"] = prof.sum(axis=1)
        prof_z = zc.rename(columns=lambda c: f"{SHORT[c]}, σ")

        lines += [
            f"## k = {k}",
            "",
            f"Файл меток: `data/processed/kmeans_labels_k{k}_{TAG}.parquet`; "
            f"inertia {km.inertia_:,.1f}, SW {ref.loc[k, 'SW']:.3f}, CH {ref.loc[k, 'CH']:,.1f}, "
            f"S_Dbw {ref.loc[k, 'S_Dbw']:.3f}",
            "",
            "### Сводка",
            "",
            md_table(summary.astype({"с ж/д": float}), fmt=",.2f"),
            "",
            "### Средние доли (исходные)",
            "",
            md_table(prof),
            "",
            "### Отклонение от среднего (σ)",
            "",
            md_table(prof_z, fmt="+.2f"),
            "",
        ]
        for c in range(k):
            sub = df[df["cluster"] == c]
            types = sub["municipal_district_type"].value_counts()
            regions = sub["region_name"].value_counts()
            reg_tab = pd.DataFrame({"МО": regions, "доля кластера": regions / len(sub)}
                                   ).head(TOP_N_REGIONS)
            top = (sub.nlargest(TOP_N_EXAMPLES, "market_access")
                   [["territory_id", "name", "region_name", "market_access"]]
                   .astype({"territory_id": str}))
            lines += [
                f"### k = {k}, кластер {c} — {len(sub)} МО: {summary.loc[c, 'подпись']}",
                "",
                "Типы МО: " + ", ".join(f"{t} — {n}" for t, n in types.items()),
                "",
                f"Регионы (всего {len(regions)}; топ-{TOP_N_REGIONS}):",
                "",
                md_table(reg_tab.rename_axis("регион"), fmt=",.2f"),
                "",
                f"Топ-{TOP_N_EXAMPLES} по market_access:",
                "",
                md_table(top, index=False, fmt=",.1f"),
                "",
            ]
        print(f"k={k}: sizes {summary['МО'].tolist()}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
