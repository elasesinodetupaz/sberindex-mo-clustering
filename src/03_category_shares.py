"""Шаг 03. Структура расходов: доли 5 категорий от «Все категории» по (territory_id, date).

Вход:  data/raw/consumption.parquet, data/processed/valid_territories.parquet
Выход: data/processed/category_shares.parquet, отчёт notebooks/03_category_shares.md
Запуск из корня проекта:  .venv/bin/python src/03_category_shares.py
"""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
CONSUMPTION_PATH = RAW_DIR / "consumption.parquet"
VALID_TERRITORIES_PATH = PROCESSED_DIR / "valid_territories.parquet"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "03_category_shares.md"

TOTAL_CATEGORY = "Все категории"
# исходная категория -> имя колонки в итоговой таблице
SHARE_COLUMNS = {
    "Продовольствие": "share_Продовольствие",
    "Здоровье": "share_Здоровье",
    "Общественное питание": "share_Общепит",
    "Транспорт": "share_Транспорт",
    "Маркетплейсы": "share_Маркетплейсы",
}
N_MONTHS = 24


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    return df.to_markdown(index=index, floatfmt=",.4f")


def main() -> None:
    valid_ids = pd.read_parquet(VALID_TERRITORIES_PATH, engine="pyarrow")["territory_id"]
    cons = pd.read_parquet(CONSUMPTION_PATH, engine="pyarrow")
    cons = cons[cons["territory_id"].isin(valid_ids)].copy()
    cons["date"] = pd.to_datetime(cons["date"], format="%Y-%m")

    unexpected = set(cons["category"].unique()) - set(SHARE_COLUMNS) - {TOTAL_CATEGORY}
    if unexpected:
        raise SystemExit(f"STOP: неожиданные категории в consumption: {unexpected}")

    # 1. Итог по «Все категории»
    df_total = (cons[cons["category"] == TOTAL_CATEGORY]
                .rename(columns={"value": "total_value"})
                [["territory_id", "date", "total_value"]])

    # 2. Доли пяти категорий от total_value
    df_parts = cons[cons["category"] != TOTAL_CATEGORY]
    long = df_parts.merge(df_total, on=["territory_id", "date"], how="left",
                          validate="many_to_one")
    if long["total_value"].isna().any():
        raise SystemExit("STOP: у части (territory_id, date) нет строки «Все категории»")
    long["category_share"] = long["value"] / long["total_value"]

    # 3. Сумма долей по (territory_id, date)
    share_sum = long.groupby(["territory_id", "date"])["category_share"].sum()
    deviation = share_sum - 1

    # 4. Широкая таблица
    wide = (long.pivot(index=["territory_id", "date"], columns="category",
                       values="category_share")
            .rename(columns=SHARE_COLUMNS)[list(SHARE_COLUMNS.values())]
            .reset_index()
            .sort_values(["territory_id", "date"], ignore_index=True))
    wide.columns.name = None

    # 5. Проверка размера
    expected_rows = len(valid_ids) * N_MONTHS
    n_nan = int(wide[list(SHARE_COLUMNS.values())].isna().sum().sum())
    if len(wide) != expected_rows or wide["territory_id"].nunique() != len(valid_ids) or n_nan:
        raise SystemExit(f"STOP: размер/полнота не совпадают: строк {len(wide)} "
                         f"(ожидалось {expected_rows}), МО {wide['territory_id'].nunique()}, "
                         f"NaN {n_nan}")

    # 6. Сохранение
    wide.to_parquet(SHARES_PATH, engine="pyarrow", index=False)

    share_cols = list(SHARE_COLUMNS.values())
    lines = [
        "# 03. Доли категорий расходов",
        "",
        "Сгенерировано `src/03_category_shares.py`. "
        "Доля = value категории / value «Все категории» за тот же (territory_id, date).",
        "",
        f"- МО: **{wide['territory_id'].nunique()}**, месяцев: **{wide['date'].nunique()}** "
        f"({wide['date'].min():%Y-%m} – {wide['date'].max():%Y-%m})",
        f"- строк: **{len(wide):,}** (ожидалось {expected_rows:,}) — совпадает",
        f"- NaN в долях: {n_nan}",
        f"- сохранено: `{SHARES_PATH.relative_to(PROJECT_DIR)}`, колонки: "
        f"{list(wide.columns)}",
        "",
        "## Сумма долей 5 категорий по (territory_id, date)",
        "",
        "Отклонение от 1 систематическое: «Все категории» включает прочие расходы.",
        "",
        md_table(pd.DataFrame({
            "сумма долей": share_sum.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]),
            "отклонение (сумма − 1)": deviation.describe(
                percentiles=[.01, .05, .25, .5, .75, .95, .99]),
        })),
        "",
        "Гистограмма суммы долей:",
        "",
        md_table(pd.cut(share_sum, bins=[0, .4, .5, .6, .7, .8, .9, 1.0])
                 .value_counts().sort_index().rename("пар (МО, месяц)")
                 .rename_axis("интервал").to_frame()),
        "",
        f"- пар с суммой долей > 1: **{int((share_sum > 1).sum())}**",
        "",
        "## Описательная статистика долей",
        "",
        md_table(wide[share_cols].describe().T),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"category_shares: {wide.shape} -> {SHARES_PATH}")
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
