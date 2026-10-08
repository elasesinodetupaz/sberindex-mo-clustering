"""Шаг 01. Обзор исходных данных: структура, полнота, пересечение territory_id.

Только чтение data/raw; результат — отчёт notebooks/01_data_overview.md.
Запуск из корня проекта:  .venv/bin/python src/01_data_overview.py
"""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
CONSUMPTION_PATH = RAW_DIR / "consumption.parquet"
MARKET_ACCESS_PATH = RAW_DIR / "market_access.parquet"
CONNECTION_PATH = RAW_DIR / "connection.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "01_data_overview.md"

STATS = ["count", "min", "max", "mean", "median"]


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    return df.to_markdown(index=index, floatfmt=",.2f")


def dtypes_table(df: pd.DataFrame) -> str:
    return md_table(df.dtypes.astype(str).rename("dtype").to_frame())


def basic_block(name: str, path: Path, df: pd.DataFrame) -> list[str]:
    nulls = df.isna().sum()
    return [
        f"## {name}",
        "",
        f"Путь: `{path.relative_to(PROJECT_DIR)}`",
        "",
        f"- shape: **{df.shape[0]:,} × {df.shape[1]}**",
        f"- полных дубликатов строк: **{df.duplicated().sum():,}**",
        f"- пропусков по колонкам: {nulls.to_dict()}",
        "",
        "dtypes:",
        "",
        dtypes_table(df),
        "",
    ]


def consumption_section(df: pd.DataFrame) -> tuple[list[str], set]:
    out = basic_block("consumption.parquet", CONSUMPTION_PATH, df)
    dates = pd.to_datetime(df["date"], format="%Y-%m")
    n_months = dates.nunique()
    out += [
        f"- дата после парсинга: min **{dates.min():%Y-%m}**, max **{dates.max():%Y-%m}**, "
        f"уникальных месяцев: **{n_months}**",
        f"- territory_id.nunique(): **{df['territory_id'].nunique():,}**",
        f"- категории: {list(df['category'].unique())}",
        "",
    ]

    # Паттерн «6 строк на (territory_id, date)»
    n_cat = df["category"].nunique()
    per_pair = df.groupby(["territory_id", "date"]).size()
    bad = per_pair[per_pair != n_cat]
    dup_keys = df.duplicated(["territory_id", "date", "category"]).sum()
    out += [
        f"### Полнота: строк на пару (territory_id, date), ожидается {n_cat}",
        "",
        f"- пар (territory_id, date): **{len(per_pair):,}**",
        f"- пар с числом строк ≠ {n_cat}: **{len(bad):,}**",
        f"- дубликатов ключа (territory_id, date, category): **{dup_keys:,}**",
        "",
        "Распределение числа строк на пару:",
        "",
        md_table(per_pair.value_counts().sort_index().rename_axis("строк на пару")
                 .rename("число пар").to_frame()),
        "",
    ]
    if len(bad):
        missing = (df.assign(one=1)
                   .pivot_table(index=["territory_id", "date"], columns="category",
                                values="one", aggfunc="size", fill_value=0)
                   .loc[bad.index])
        out += [
            "Каких категорий не хватает в неполных парах (число пар без категории):",
            "",
            md_table((missing == 0).sum().rename("пар без категории").to_frame()),
            "",
        ]

    # Полнота по месяцам: у всех ли МО есть все месяцы
    months_per_terr = df.groupby("territory_id")["date"].nunique()
    out += [
        f"Месяцев на территорию (из {n_months} возможных):",
        "",
        md_table(months_per_terr.value_counts().sort_index(ascending=False)
                 .rename_axis("месяцев").rename("территорий").to_frame()),
        "",
    ]

    # Статистика value по категориям
    stats = df.groupby("category")["value"].agg(STATS)
    out += [
        "### value по категориям",
        "",
        md_table(stats),
        "",
        f"- value ≤ 0: **{(df['value'] <= 0).sum():,}** строк "
        f"(из них = 0: {(df['value'] == 0).sum():,}, < 0: {(df['value'] < 0).sum():,})",
        "",
    ]

    # Согласованность «Все категории» с суммой пяти категорий
    wide = df.pivot_table(index=["territory_id", "date"], columns="category",
                          values="value", aggfunc="sum")
    if "Все категории" in wide:
        parts = wide.drop(columns="Все категории").sum(axis=1, min_count=1)
        ratio = (wide["Все категории"] / parts).dropna()
        out += [
            "### «Все категории» / сумма пяти категорий",
            "",
            md_table(ratio.describe(percentiles=[.01, .25, .5, .75, .99])
                     .rename("ratio").to_frame()),
            "",
        ]
    return out, set(df["territory_id"].unique())


def market_access_section(df: pd.DataFrame) -> tuple[list[str], set]:
    out = basic_block("market_access.parquet", MARKET_ACCESS_PATH, df)
    out += [
        f"- territory_id.nunique(): **{df['territory_id'].nunique():,}** "
        f"(строк: {len(df):,})",
        "",
        "Статистика market_access:",
        "",
        md_table(df["market_access"].agg(STATS).rename("market_access").to_frame()),
        "",
    ]
    return out, set(df["territory_id"].unique())


def connection_section(df: pd.DataFrame) -> tuple[list[str], dict[str, set]]:
    out = basic_block("connection.parquet", CONNECTION_PATH, df)
    out += [f"- type.unique(): {list(df['type'].unique())}", ""]

    rows, ids_by_type = [], {}
    for t, g in df.groupby("type"):
        ids = set(g["territory_id_x"]) | set(g["territory_id_y"])
        ids_by_type[t] = ids
        n = len(ids)
        pair_key = pd.DataFrame({"a": g[["territory_id_x", "territory_id_y"]].min(axis=1),
                                 "b": g[["territory_id_x", "territory_id_y"]].max(axis=1)})
        rows.append({
            "type": t,
            "уник. territory_id_x": g["territory_id_x"].nunique(),
            "уник. territory_id_y": g["territory_id_y"].nunique(),
            "уник. id (x ∪ y)": n,
            "пар (строк)": len(g),
            "уник. неупоряд. пар": len(pair_key.drop_duplicates()),
            "пар x==y": int((g["territory_id_x"] == g["territory_id_y"]).sum()),
            "полный граф n(n-1)/2": n * (n - 1) // 2,
            "distance min": g["distance"].min(),
            "distance max": g["distance"].max(),
            "distance mean": g["distance"].mean(),
            "distance median": g["distance"].median(),
            "distance ≤ 0": int((g["distance"] <= 0).sum()),
        })
    out += [
        "### По типам связи",
        "",
        md_table(pd.DataFrame(rows).set_index("type").T),
        "",
        "Если «уник. неупоряд. пар» ≈ «полный граф n(n-1)/2», то таблица — полная матрица "
        "расстояний между всеми достижимыми парами, а не список смежности.",
        "",
    ]
    return out, ids_by_type


def overlap_section(cons: set, ma: set, conn_by_type: dict[str, set]) -> list[str]:
    conn_any = set().union(*conn_by_type.values())
    hw = conn_by_type.get("highway", set())
    rw = conn_by_type.get("railway", set())
    sets = {"consumption": cons, "market_access": ma, "connection (любой type)": conn_any,
            "connection highway": hw, "connection railway": rw}
    sizes = pd.Series({k: len(v) for k, v in sets.items()}, name="уник. id").to_frame()

    rows = {
        "во всех трёх (consumption ∩ market_access ∩ connection)": len(cons & ma & conn_any),
        "consumption ∩ market_access": len(cons & ma),
        "consumption ∩ connection": len(cons & conn_any),
        "только в consumption, нет в market_access": len(cons - ma),
        "только в consumption, нет в connection": len(cons - conn_any),
        "в consumption, нет в connection highway": len(cons - hw),
        "в consumption, нет в connection railway": len(cons - rw),
        "в market_access, нет в consumption": len(ma - cons),
        "в connection, нет в consumption": len(conn_any - cons),
        "в market_access, нет в connection": len(ma - conn_any),
    }
    out = [
        "## Пересечение territory_id между таблицами",
        "",
        md_table(sizes),
        "",
        md_table(pd.Series(rows, name="число id").to_frame()),
        "",
    ]
    for label, s in [("consumption \\ market_access", cons - ma),
                     ("consumption \\ connection", cons - conn_any),
                     ("consumption \\ highway", cons - hw)]:
        if s and len(s) <= 50:
            out += [f"- id {label}: {sorted(int(x) for x in s)}", ""]
    return out


def main() -> None:
    cons = pd.read_parquet(CONSUMPTION_PATH, engine="pyarrow")
    ma = pd.read_parquet(MARKET_ACCESS_PATH, engine="pyarrow")
    conn = pd.read_parquet(CONNECTION_PATH, engine="pyarrow")

    lines = ["# 01. Обзор исходных данных", "",
             "Сгенерировано `src/01_data_overview.py`. Исходники не изменялись.", ""]
    part, cons_ids = consumption_section(cons)
    lines += part
    part, ma_ids = market_access_section(ma)
    lines += part
    part, conn_ids = connection_section(conn)
    lines += part
    lines += overlap_section(cons_ids, ma_ids, conn_ids)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
