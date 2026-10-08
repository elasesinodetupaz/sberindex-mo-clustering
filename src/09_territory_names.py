"""Шаг 09. Справочник названий МО из версионной таблицы t_dict_municipal_districts.

Для каждого territory_id берётся версия с максимальным year_from.
Полигоны (t_dict_municipal_districts_poly.gpkg) не читаются.

Вход:  MUNICIPAL_DICT_PATH (xlsx, вне проекта), data/processed/valid_territories.parquet
Выход: data/raw/territories.parquet (territory_id, name, municipal_district_type, region_code,
       region_name, center, center_lat, center_lon),
       notebooks/09_territory_names.md
Запуск из корня проекта:  .venv/bin/python src/09_territory_names.py
"""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
MUNICIPAL_DICT_PATH = Path.home() / "Documents" / "municipal_dict" / "t_dict_municipal_districts.xlsx"
VALID_TERRITORIES_PATH = PROJECT_DIR / "data" / "processed" / "valid_territories.parquet"
TERRITORIES_PATH = PROJECT_DIR / "data" / "raw" / "territories.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "09_territory_names.md"

OUTPUT_COLUMNS = {
    "territory_id": "territory_id",
    "municipal_district_name": "name",
    "municipal_district_type": "municipal_district_type",
    "region_code": "region_code",
    "region_name": "region_name",
    "municipal_district_center": "center",
    "municipal_district_center_lat": "center_lat",
    "municipal_district_center_lon": "center_lon",
}


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    return df.to_markdown(index=index)


def main() -> None:
    raw = pd.read_excel(MUNICIPAL_DICT_PATH, sheet_name=0)
    missing = (set(OUTPUT_COLUMNS) | {"year_from"}) - set(raw.columns)
    if missing:
        raise SystemExit(f"STOP: в справочнике нет колонок {missing}")

    versions = raw.groupby("territory_id").size()
    # самая свежая версия: максимальный year_from для каждого territory_id
    max_from = raw.groupby("territory_id")["year_from"].transform("max")
    latest = raw[raw["year_from"] == max_from]
    dups = latest[latest["territory_id"].duplicated(keep=False)].sort_values("territory_id")

    lines = [
        "# 09. Справочник названий МО",
        "",
        "Сгенерировано `src/09_territory_names.py`.",
        f"Источник: `{MUNICIPAL_DICT_PATH}` (лист 1; полигоны не читались).",
        "",
        f"- shape исходной таблицы: **{raw.shape[0]:,} × {raw.shape[1]}**",
        f"- колонки: {list(raw.columns)}",
        f"- уникальных territory_id: **{raw['territory_id'].nunique():,}**",
        f"- версий на territory_id: {versions.value_counts().sort_index().to_dict()}",
        f"- year_from: {sorted(raw['year_from'].unique().tolist())}; "
        f"year_to: {sorted(raw['year_to'].unique().tolist())}",
        "",
        "Первые 10 строк:",
        "",
        md_table(raw.head(10)),
        "",
        f"- дублей territory_id после отбора максимального year_from: **{dups['territory_id'].nunique()}**",
        "",
    ]
    if len(dups):
        lines += ["Дубли для ручного разбора (файл не сохранён):", "", md_table(dups), ""]
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        raise SystemExit("STOP: дубли territory_id в самой свежей версии — нужен ручной разбор")

    out = (latest[list(OUTPUT_COLUMNS)].rename(columns=OUTPUT_COLUMNS)
           .sort_values("territory_id", ignore_index=True))
    out.to_parquet(TERRITORIES_PATH, engine="pyarrow", index=False)

    # Сверка с рабочей выборкой
    valid = set(pd.read_parquet(VALID_TERRITORIES_PATH, engine="pyarrow")["territory_id"])
    dict_ids = set(out["territory_id"])
    not_found = sorted(int(i) for i in valid - dict_ids)
    name_changed = raw[raw["territory_id"].isin(valid)].groupby("territory_id")[
        "municipal_district_name"].nunique()
    lines += [
        f"Сохранено: `{TERRITORIES_PATH.relative_to(PROJECT_DIR)}` ({len(out):,} строк, "
        f"колонки {list(out.columns)})",
        "",
        "## Сверка с рабочей выборкой (valid_territories.parquet)",
        "",
        f"- МО в выборке: **{len(valid)}**; найдено в справочнике: **{len(valid & dict_ids)}**; "
        f"не найдено: **{len(not_found)}**",
        f"- не найденные id: {not_found}" if not_found else "- не найденных id нет",
        f"- МО выборки, у которых название менялось между версиями: "
        f"**{int((name_changed > 1).sum())}**",
        "",
        "Типы МО в рабочей выборке:",
        "",
        md_table(out[out["territory_id"].isin(valid)]["municipal_district_type"]
                 .value_counts().rename_axis("тип").rename("МО").reset_index()),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
