"""Справочник territory_id -> название МО (используется шагами 07, 08).

В исходной поставке справочника нет. Когда он появится, положить файл в data/raw/
под одним из имён из CANDIDATE_FILES (parquet / csv / xlsx) с колонками territory_id и
name (другие имена колонок — указать в ID_COLUMN / NAME_COLUMN; опционально REGION_COLUMN).
"""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
CANDIDATE_FILES = ["territories.parquet", "territories.csv", "territories.xlsx"]
ID_COLUMN = "territory_id"
NAME_COLUMN = "name"
REGION_COLUMN = "region_name"   # если в справочнике есть регион — добавится к названию


def names_path() -> Path | None:
    for f in CANDIDATE_FILES:
        if (RAW_DIR / f).exists():
            return RAW_DIR / f
    return None


def load_territory_names() -> pd.Series | None:
    """Series territory_id -> название (с регионом, если есть), или None, если файла нет."""
    p = names_path()
    if p is None:
        return None
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_excel(p)
    missing = {ID_COLUMN, NAME_COLUMN} - set(df.columns)
    if missing:
        raise SystemExit(f"STOP: в {p.name} нет колонок {missing}; есть: {list(df.columns)}. "
                         "Укажите ID_COLUMN / NAME_COLUMN в src/territory_names.py")
    if df[ID_COLUMN].duplicated().any():
        raise SystemExit(f"STOP: в {p.name} дубликаты {ID_COLUMN}: "
                         f"{df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].head(10).tolist()}")
    name = df[NAME_COLUMN].astype(str)
    if REGION_COLUMN in df.columns:
        name = name + " (" + df[REGION_COLUMN].astype(str) + ")"
    return pd.Series(name.to_numpy(), index=df[ID_COLUMN].astype(int), name="name")


def names_note() -> str:
    p = names_path()
    if p is None:
        return ("**Справочник названий не найден** (ожидается один из "
                f"{', '.join('data/raw/' + f for f in CANDIDATE_FILES)}). Названия появятся "
                "после его добавления и перезапуска скрипта.")
    return f"Названия МО: `{p.relative_to(PROJECT_DIR)}`."


def add_names(df: pd.DataFrame, names: pd.Series | None, id_cols: list[str]) -> pd.DataFrame:
    """Добавляет колонку name_<суффикс> рядом с каждой id-колонкой (если справочник есть)."""
    if names is None:
        return df
    out = df.copy()
    for col in id_cols:
        suffix = col.rsplit("_", 1)[-1] if col != "territory_id" else ""
        new = f"name_{suffix}" if suffix else "name"
        out.insert(out.columns.get_loc(col) + 1, new, out[col].map(names))
    return out
