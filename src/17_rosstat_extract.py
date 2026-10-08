"""Шаг 17. Извлечение показателей Росстата (БД ПМО) для 2004 МО рабочей выборки.

Данные нужны только для проверки гипотез о готовых типах: метки кластеров и отчёты шагов 10–16
этот шаг не читает и не меняет. Источник — выгрузки tochno.st «Муниципальная статистика России
с 2005 года» (CC BY 4.0), см. data/external/README.md.

Архивы не распаковываются: csv читаются потоком из zip по частям, только нужные колонки.
Показатели (МО верхнего уровня):
  Y48112027 — оценка численности населения на 1 января -> population;
  Y48112014 — численность населения по полу и возрасту -> share_younger / share_working_age / share_older;
  Y48423007 — среднемесячная зарплата (без субъектов малого предпринимательства) -> wage;
  Y48423005 — среднесписочная численность работников (то же) -> workers, share_okved_<буква>.
Склейка с МО — по 8 цифрам ОКТМО из справочника МО (все версии кода), колонка matched_by.
Значения не исправляются; подозрительные МО-годы отмечены флагом anomaly.

Поведение: есть оба архива -> файл пересчитывается; архивов нет, но файл есть -> используется
готовый файл (проверяется схема); нет ни того, ни другого -> STOP.

Вход:  data/raw/rosstat/population.zip, data/raw/rosstat/employment_wages.zip,
       MUNICIPAL_DICT_PATH (xlsx, вне проекта), data/processed/valid_territories.parquet
Выход: data/external/rosstat_mo_2023_2025.parquet
Запуск из корня проекта:  .venv/bin/python src/17_rosstat_extract.py
"""
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROSSTAT_ANOMALY_SHARE_TOL as ANOMALY_SHARE_TOL
from config import ROSSTAT_CHUNKSIZE as CHUNKSIZE
from config import ROSSTAT_LABOUR_YEARS, ROSSTAT_POP_YEARS

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROSSTAT_DIR = PROJECT_DIR / "data" / "raw" / "rosstat"
POPULATION_ZIP = ROSSTAT_DIR / "population.zip"
EMPLOYMENT_ZIP = ROSSTAT_DIR / "employment_wages.zip"
# тот же справочник, что в src/09_territory_names.py (MUNICIPAL_DICT_PATH)
MUNICIPAL_DICT_PATH = Path.home() / "Documents" / "municipal_dict" / "t_dict_municipal_districts.xlsx"
VALID_TERRITORIES_PATH = PROJECT_DIR / "data" / "processed" / "valid_territories.parquet"
OUT_PATH = PROJECT_DIR / "data" / "external" / "rosstat_mo_2023_2025.parquet"

UPPER_LEVEL = "Муниципальное образование верхнего уровня"
OKVED_TOTAL = "Всего по обследуемым видам экономической деятельности"
VOZR = {"Всего": "total", "Моложе трудоспособного возраста": "younger",
        "Трудоспособный возраст": "working_age", "Старше трудоспособного возраста": "older"}
BASE_COLS = ["region_id", "mun_level", "municipality", "oktmo", "oktmo_stable", "oktmo_year_from",
             "oktmo_year_to", "year", "indicator_value", "indicator_period"]
# кириллические буквы разделов ОКВЭД-2 в выгрузке -> латинские
CYR2LAT = {"А": "A", "В": "B", "Е": "E", "Н": "H"}
OKVED_LETTERS = list("ABCDEFGHIJKLMNOPQRS")
MATCH_PRIORITY = {"oktmo=последний код": 1, "oktmo=код другой версии": 2, "oktmo_stable": 3}
OUT_COLUMNS = (["territory_id", "year", "population", "share_younger", "share_working_age", "share_older",
                "wage", "wage_rel_region", "workers", "workers_per_pop"]
               + [f"share_okved_{c}" for c in OKVED_LETTERS] + ["matched_by", "anomaly"])


def read_member(zip_path: Path, code: str, extra_cols: list, keep) -> pd.DataFrame:
    """Читает csv показателя из архива потоком по частям и оставляет строки, для которых keep(chunk) истинно."""
    member = f"data_{code}_112_v20250918.csv"
    start, n_rows, parts = time.time(), 0, []
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
        for chunk in pd.read_csv(f, sep=";", usecols=BASE_COLS + extra_cols, dtype=str, chunksize=CHUNKSIZE):
            n_rows += len(chunk)
            parts.append(chunk[(chunk["mun_level"] == UPPER_LEVEL) & keep(chunk)])
    df = pd.concat(parts, ignore_index=True)
    print(f"{zip_path.name}/{member}: прочитано строк {n_rows:,}, отобрано {len(df):,}, время {time.time() - start:.1f} с")
    return df


def load_key(valid: pd.DataFrame):
    """Коды ОКТМО (8 цифр) всех версий МО выборки; последний код — версия с максимальным year_from."""
    d = pd.read_excel(MUNICIPAL_DICT_PATH, sheet_name=0)
    d = d[d["territory_id"].isin(valid["territory_id"])].copy()
    d["ok8"] = d["oktmo"].astype(str).str.replace("-", "", regex=False).str[:8]
    if not d["ok8"].str.fullmatch(r"\d{8}").all():
        raise SystemExit("STOP: в справочнике есть oktmo, из которых не получается 8 цифр")
    latest = d[d["year_from"] == d.groupby("territory_id")["year_from"].transform("max")]
    if latest["territory_id"].duplicated().any() or len(latest) != len(valid):
        raise SystemExit("STOP: последняя версия справочника не даёт ровно одну строку на МО выборки")
    shared = d.groupby("ok8")["territory_id"].nunique()
    if (shared > 1).any():
        raise SystemExit(f"STOP: коды ОКТМО у нескольких МО: {shared[shared > 1].index.tolist()}")
    n_codes = d.groupby("territory_id")["ok8"].nunique()
    print(f"Ключ: МО {len(latest)}, с несколькими кодами по версиям {(n_codes > 1).sum()}, кодов всего {d['ok8'].nunique()}")
    return dict(zip(d["ok8"], d["territory_id"])), set(latest["ok8"])


def match(df: pd.DataFrame, code2tid: dict, latest_codes: set) -> pd.DataFrame:
    by_oktmo = df["oktmo"].map(code2tid)
    by_stable = df["oktmo_stable"].map(code2tid)
    df = df.assign(territory_id=by_oktmo.fillna(by_stable),
                   matched_by=np.where(by_oktmo.notna(),
                                       np.where(df["oktmo"].isin(latest_codes), "oktmo=последний код",
                                                "oktmo=код другой версии"),
                                       np.where(by_stable.notna(), "oktmo_stable", None)))
    df = df[df["territory_id"].notna()].copy()
    df["territory_id"] = df["territory_id"].astype(int)
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["indicator_value"])
    return df


def resolve(df: pd.DataFrame, name: str, keys: list) -> pd.DataFrame:
    """Одна строка на (МО, год, keys): полные дубли удаляются; затем лучший matched_by; затем строка,
    где год входит в [oktmo_year_from, oktmo_year_to]; если и так неоднозначно — значение пустое."""
    k = ["territory_id", "year"] + keys
    n0 = len(df)
    df = df.drop_duplicates()
    counts = df.groupby(k).size()
    print(f"{name}: полных дублей удалено {n0 - len(df)}; строк на (МО, год{', ' + ', '.join(keys) if keys else ''}): "
          f"{counts.value_counts().sort_index().to_dict()}")
    pri = df["matched_by"].map(MATCH_PRIORITY)
    df = df[pri == pri.groupby([df[c] for c in k]).transform("min")]
    year_from = pd.to_numeric(df["oktmo_year_from"].str[:4], errors="coerce")
    year_to = pd.to_numeric(df["oktmo_year_to"].str[-4:], errors="coerce")
    df = df.assign(in_years=(year_from <= df["year"]) & (df["year"] <= year_to))
    size = df.groupby(k)["value"].transform("size")
    single = df[size == 1]
    multi = df[size > 1]
    by_years = multi[multi["in_years"]]
    by_years = by_years[by_years.groupby(k)["value"].transform("size") == 1]
    unresolved = multi.drop_duplicates(k).set_index(k).index.difference(by_years.set_index(k).index)
    print(f"{name}: неоднозначно после matched_by {multi.drop_duplicates(k).shape[0]}, "
          f"разрешено по годам действия кода {len(by_years)}, оставлено пустым {len(unresolved)}")
    if len(unresolved):
        print(f"{name}: пустые из-за неоднозначности (МО, год{', ' + ', '.join(keys) if keys else ''}):",
              unresolved.tolist())
    return pd.concat([single, by_years])


def okved_column(label: str) -> str:
    if label == OKVED_TOTAL:
        return "total"
    letter = label.split()[1]
    return "share_okved_" + CYR2LAT.get(letter, letter)


def check_schema(df: pd.DataFrame, n_mo: int) -> None:
    if list(df.columns) != OUT_COLUMNS:
        raise SystemExit(f"STOP: колонки файла {list(df.columns)} не совпадают с ожидаемыми {OUT_COLUMNS}")
    if len(df) != n_mo * len(ROSSTAT_POP_YEARS) or df.duplicated(["territory_id", "year"]).any():
        raise SystemExit(f"STOP: в файле {len(df)} строк или дубли (МО, год); ожидается {n_mo} × {len(ROSSTAT_POP_YEARS)}")
    if df["anomaly"].dtype != bool:
        raise SystemExit(f"STOP: тип anomaly {df['anomaly'].dtype}, ожидается bool")
    print(f"Схема проверена: {len(df)} строк, {len(df.columns)} колонок")


def build(valid: pd.DataFrame) -> pd.DataFrame:
    code2tid, latest_codes = load_key(valid)
    pop_years = {str(y) for y in ROSSTAT_POP_YEARS}
    lab_years = {str(y) for y in ROSSTAT_LABOUR_YEARS}
    raw = {
        "population": read_member(POPULATION_ZIP, "Y48112027", ["mest"], lambda c: (
            (c["mest"] == "Все население") & (c["indicator_period"] == "На 1 января") & c["year"].isin(pop_years))),
        "age": read_member(POPULATION_ZIP, "Y48112014", ["grup_2", "vozr"], lambda c: (
            (c["grup_2"] == "Всего") & c["vozr"].isin(VOZR) & (c["indicator_period"] == "На 1 января")
            & c["year"].isin(pop_years))),
        "wage": read_member(EMPLOYMENT_ZIP, "Y48423007", ["okved2"], lambda c: (
            (c["okved2"] == OKVED_TOTAL) & (c["indicator_period"] == "Январь-декабрь") & c["year"].isin(lab_years))),
        "workers": read_member(EMPLOYMENT_ZIP, "Y48423005", ["okved2"], lambda c: (
            (c["indicator_period"] == "Январь-декабрь") & c["year"].isin(lab_years))),
    }
    pop = resolve(match(raw["population"], code2tid, latest_codes), "population", [])
    age = match(raw["age"], code2tid, latest_codes).assign(group=lambda d: d["vozr"].map(VOZR))
    age = resolve(age, "age", ["group"])
    wage = resolve(match(raw["wage"], code2tid, latest_codes), "wage", [])
    wk = match(raw["workers"], code2tid, latest_codes)
    wk["col"] = wk["okved2"].map(okved_column)
    unknown = set(wk["col"]) - {"total"} - {f"share_okved_{c}" for c in OKVED_LETTERS}
    if unknown:
        raise SystemExit(f"STOP: разделы ОКВЭД-2 вне A–S у МО выборки: {sorted(unknown)}")
    wk = resolve(wk, "workers", ["col"])

    idx = pd.MultiIndex.from_product([sorted(valid["territory_id"].astype(int)), ROSSTAT_POP_YEARS],
                                     names=["territory_id", "year"])
    out = pd.DataFrame(index=idx)
    out["population"] = pop.set_index(["territory_id", "year"])["value"]
    ag = age.pivot(index=["territory_id", "year"], columns="group", values="value")
    for g in ["younger", "working_age", "older"]:
        out[f"share_{g}"] = ag[g] / ag["total"]
    out["wage"] = wage.set_index(["territory_id", "year"])["value"]
    wkw = wk.pivot(index=["territory_id", "year"], columns="col", values="value")
    out["workers"] = wkw["total"]
    for c in OKVED_LETTERS:
        col = f"share_okved_{c}"
        out[col] = wkw[col] / wkw["total"] if col in wkw else np.nan
    out = out.reset_index()

    regions = pd.concat([d[["territory_id", "region_id"]] for d in (pop, age, wage, wk)]).drop_duplicates()
    if regions["territory_id"].duplicated().any():
        raise SystemExit("STOP: у МО несколько region_id Росстата")
    out["region_id"] = out["territory_id"].map(regions.set_index("territory_id")["region_id"])
    out["wage_rel_region"] = out["wage"] / out.groupby(["region_id", "year"])["wage"].transform("median")
    out["workers_per_pop"] = out["workers"] / out["population"]

    sources = {"pop": pop, "age": age[age["group"] == "total"], "wage": wage, "workers": wk[wk["col"] == "total"]}
    mb = pd.concat([d.set_index(["territory_id", "year"])["matched_by"].rename(n) for n, d in sources.items()], axis=1)
    mb = mb.reindex(pd.MultiIndex.from_frame(out[["territory_id", "year"]]))
    out["matched_by"] = ["; ".join(f"{n}={m}" for n, m in row.items() if isinstance(m, str)) or None
                         for row in mb.to_dict("records")]
    share_sum = out[["share_younger", "share_working_age", "share_older"]].sum(axis=1, min_count=3)
    out["anomaly"] = ((out["population"] == 0) | ((share_sum - 1).abs() > ANOMALY_SHARE_TOL)).astype(bool)

    out["territory_id"] = out["territory_id"].astype(valid["territory_id"].dtype)
    out["year"] = out["year"].astype("int64")
    return out[OUT_COLUMNS].sort_values(["territory_id", "year"], ignore_index=True)


def main() -> None:
    valid = pd.read_parquet(VALID_TERRITORIES_PATH)
    have_zips = POPULATION_ZIP.exists() and EMPLOYMENT_ZIP.exists()
    if not have_zips:
        missing = [p.name for p in (POPULATION_ZIP, EMPLOYMENT_ZIP) if not p.exists()]
        if OUT_PATH.exists():
            print(f"Архивов Росстата нет ({', '.join(missing)}): используется готовый файл {OUT_PATH}")
            check_schema(pd.read_parquet(OUT_PATH), len(valid))
            return
        raise SystemExit(f"STOP: нет архивов Росстата ({', '.join(missing)} в {ROSSTAT_DIR}) и нет готового "
                         f"файла {OUT_PATH}. Положите архивы tochno.st (см. data/external/README.md) и запустите снова.")
    out = build(valid)
    check_schema(out, len(valid))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    print(f"Записано: {OUT_PATH} ({OUT_PATH.stat().st_size:,} байт)")
    print("Непустых значений по годам:")
    print(out.groupby("year")[["population", "share_older", "wage", "workers"]].count().to_string())
    print("anomaly == True по годам:", out[out["anomaly"]].groupby("year").size().to_dict())


if __name__ == "__main__":
    main()
