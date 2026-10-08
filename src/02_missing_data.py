"""Шаг 02. Фильтрация рабочей выборки МО и проверка пропусков / неположительных значений.

Фильтры (решение по итогам шага 01):
  1. МО с полными 24 месяцами в consumption;
  2. из них — только присутствующие в market_access.
Результат: data/processed/valid_territories.parquet, отчёт notebooks/02_missing_data.md.
Запуск из корня проекта:  .venv/bin/python src/02_missing_data.py
"""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
CONSUMPTION_PATH = RAW_DIR / "consumption.parquet"
MARKET_ACCESS_PATH = RAW_DIR / "market_access.parquet"
CONNECTION_PATH = RAW_DIR / "connection.parquet"
VALID_TERRITORIES_PATH = PROCESSED_DIR / "valid_territories.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "02_missing_data.md"

N_MONTHS_REQUIRED = 24


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    return df.to_markdown(index=index, floatfmt=",.2f")


def nan_table(df: pd.DataFrame) -> str:
    return md_table(df.isna().sum().rename("NaN").to_frame())


def main() -> None:
    cons = pd.read_parquet(CONSUMPTION_PATH, engine="pyarrow")
    ma = pd.read_parquet(MARKET_ACCESS_PATH, engine="pyarrow")
    conn = pd.read_parquet(CONNECTION_PATH, engine="pyarrow")

    # --- Фильтр 1: полная история ---
    all_ids = set(cons["territory_id"].unique())
    months = cons.groupby("territory_id")["date"].nunique()
    full_ids = set(months[months == N_MONTHS_REQUIRED].index)
    incomplete_ids = all_ids - full_ids

    # --- Фильтр 2: наличие в market_access ---
    ma_ids = set(ma["territory_id"].unique())
    no_ma_all = all_ids - ma_ids
    no_ma_after_f1 = full_ids - ma_ids
    valid_ids = sorted(full_ids & ma_ids)

    valid = pd.DataFrame({"territory_id": pd.Series(valid_ids, dtype=cons["territory_id"].dtype)})
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    valid.to_parquet(VALID_TERRITORIES_PATH, engine="pyarrow", index=False)

    # Покрытие рабочей выборки транспортной сетью (справочно)
    conn_ids = {t: set(g["territory_id_x"]) | set(g["territory_id_y"])
                for t, g in conn.groupby("type")}
    vset = set(valid_ids)

    lines = [
        "# 02. Рабочая выборка и пропуски",
        "",
        "Сгенерировано `src/02_missing_data.py`. Исходники не изменялись.",
        "",
        "## Фильтрация МО",
        "",
        md_table(pd.DataFrame([
            ("МО в consumption", len(all_ids)),
            (f"исключено: неполная история (< {N_MONTHS_REQUIRED} мес.)", len(incomplete_ids)),
            (f"осталось с полными {N_MONTHS_REQUIRED} мес.", len(full_ids)),
            ("исключено: нет в market_access (после фильтра 1)", len(no_ma_after_f1)),
            ("**итоговая рабочая выборка**", len(valid_ids)),
        ], columns=["шаг", "число МО"]), index=False),
        "",
        f"- Всего в consumption без market_access: {len(no_ma_all)} МО; из них "
        f"{len(no_ma_all & incomplete_ids)} уже отсеяны фильтром 1, "
        f"фильтр 2 дополнительно исключил **{len(no_ma_after_f1)}**.",
        f"- id, исключённые фильтром 2: {sorted(int(x) for x in no_ma_after_f1)}",
        f"- id, исключённые фильтром 1 ({len(incomplete_ids)}): "
        f"{sorted(int(x) for x in incomplete_ids)}",
        "",
        f"Список сохранён: `{VALID_TERRITORIES_PATH.relative_to(PROJECT_DIR)}` "
        f"({len(valid)} строк, столбец `territory_id`).",
        "",
        "Справочно — покрытие рабочей выборки транспортной сетью:",
        "",
        md_table(pd.DataFrame([
            (t, len(vset & ids), len(vset - ids)) for t, ids in conn_ids.items()
        ], columns=["type", "МО выборки есть в сети", "МО выборки нет в сети"]), index=False),
        "",
    ]

    # --- Неположительные значения value ---
    nonpos = cons[cons["value"] <= 0]
    by_cat = pd.DataFrame({
        "всего строк": cons.groupby("category").size(),
        "value = 0": cons[cons["value"] == 0].groupby("category").size(),
        "value < 0": cons[cons["value"] < 0].groupby("category").size(),
    }).fillna(0).astype(int)
    cons_valid = cons[cons["territory_id"].isin(vset)]
    lines += [
        "## Неположительные значения value (consumption)",
        "",
        f"- во всём файле: **{len(nonpos):,}** строк с value ≤ 0 "
        f"(= 0: {(cons['value'] == 0).sum():,}, < 0: {(cons['value'] < 0).sum():,})",
        f"- в рабочей выборке: **{(cons_valid['value'] <= 0).sum():,}**",
        "",
        md_table(by_cat),
        "",
        "## NaN",
        "",
        "consumption.parquet:",
        "",
        nan_table(cons),
        "",
        "market_access.parquet:",
        "",
        nan_table(ma),
        "",
        "connection.parquet:",
        "",
        nan_table(conn),
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Рабочая выборка: {len(valid_ids)} МО -> {VALID_TERRITORIES_PATH}")
    print(f"Отчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
