"""Шаг 20. Извлечение данных о розничной торговле и общепите (БД ПМО Росстата, раздел 2) для 2004 МО.

Только извлечение: связи этих показателей с долями расходов здесь не считаются. Ключ ОКТМО и правила
склейки — те же функции, что в шаге 17 (импорт из src/17_rosstat_extract.py без запуска его расчёта).
Входной parquet читается по частям (iter_batches), только нужные колонки.

Показатели (МО верхнего уровня, период RETAIL_PERIOD, годы RETAIL_YEARS, непустые значения):
  retail_floor_m2    — Y48002002 «Площадь торгового зала объектов розничной торговли», obroz = «Магазины»;
  retail_stores      — Y48002001 «Количество объектов розничной торговли и общественного питания», «Магазины»;
  catering_seats_pub — Y48002004 «Число мест в объектах общественного питания», сумма «Рестораны, кафе, бары»
                       и «Общедоступные столовые, закусочные» (пусто, если нет ни одного из двух);
  catering_seats_all — Y48002004, сумма трёх видов (плюс «Столовые учебных заведений, организаций,
                       промышленных предприятий»; пусто, если нет ни одного);
  cafes_objects      — Y48002001, obroz = «Рестораны, кафе, бары».

Поведение: есть входной parquet -> файл пересчитывается; входа нет, но файл есть -> используется готовый
файл (проверяется схема); нет ни того, ни другого -> STOP.

Вход:  data/raw/rosstat/retail/data_section2_112_v20250918.parquet, MUNICIPAL_DICT_PATH (как в шаге 17),
       data/processed/valid_territories.parquet; для проверок — kmeans_labels_final.parquet и
       data/external/rosstat_mo_2023_2025.parquet (население)
Выход: data/external/rosstat_retail_2023_2024.parquet
Запуск из корня проекта:  .venv/bin/python src/20_rosstat_retail_extract.py
"""
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config import (RETAIL_BATCH_ROWS, RETAIL_CHECK_BAND, RETAIL_CHECK_MAX_FLOOR_PER_1000, RETAIL_PERIOD,
                    RETAIL_YEARS)

step17 = importlib.import_module("17_rosstat_extract")

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_DIR / "data" / "raw" / "rosstat" / "retail" / "data_section2_112_v20250918.parquet"
VALID_TERRITORIES_PATH = PROJECT_DIR / "data" / "processed" / "valid_territories.parquet"
LABELS_FINAL_PATH = PROJECT_DIR / "data" / "processed" / "kmeans_labels_final.parquet"
POPULATION_PATH = PROJECT_DIR / "data" / "external" / "rosstat_mo_2023_2025.parquet"
OUT_PATH = PROJECT_DIR / "data" / "external" / "rosstat_retail_2023_2024.parquet"

READ_COLUMNS = ["indicator_code", "year", "mun_level", "obroz", "indicator_period", "indicator_value", "oktmo",
                "oktmo_stable", "oktmo_history", "oktmo_year_from", "oktmo_year_to"]
UPPER_LEVEL = step17.UPPER_LEVEL
CODE_OBJECTS, CODE_FLOOR, CODE_SEATS = "Y48002001", "Y48002002", "Y48002004"
SHOPS = "Магазины"
CAFES = "Рестораны, кафе, бары"
CANTEENS_PUBLIC = "Общедоступные столовые, закусочные"
CANTEENS_INSTITUTIONAL = "Столовые учебных заведений, организаций, промышленных предприятий"
FLOOR_SUBTYPES = ["Аптеки и аптечные магазины", "Гипермаркеты", "Минимаркеты", "Павильоны", "Прочие магазины",
                  "Специализированные непродовольственные магазины", "Специализированные продовольственные магазины",
                  "Супермаркеты", "Универмаги"]   # проверка в): «Магазины» / сумма этих девяти видов
OUT_COLUMNS = ["territory_id", "year", "retail_floor_m2", "retail_stores", "catering_seats_pub", "catering_seats_all",
               "cafes_objects", "matched_by"]
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def read_input() -> pd.DataFrame:
    start, n_rows, parts = time.time(), 0, []
    pf = pq.ParquetFile(INPUT_PATH)
    for batch in pf.iter_batches(batch_size=RETAIL_BATCH_ROWS, columns=READ_COLUMNS):
        d = batch.to_pandas()
        n_rows += len(d)
        keep = ((d["mun_level"] == UPPER_LEVEL) & d["year"].isin(RETAIL_YEARS) & (d["indicator_period"] == RETAIL_PERIOD)
                & d["indicator_value"].notna() & d["indicator_code"].isin([CODE_OBJECTS, CODE_FLOOR, CODE_SEATS]))
        parts.append(d[keep])
    df = pd.concat(parts, ignore_index=True)
    df["year"] = df["year"].astype(str)          # как в выгрузках шага 17 (match приводит год к int)
    print(f"{INPUT_PATH.name}: прочитано строк {n_rows:,} ({pf.metadata.num_row_groups} групп), отобрано {len(df):,}, "
          f"время {time.time() - start:.1f} с")
    return df


def check_schema(df: pd.DataFrame, n_mo: int) -> None:
    if list(df.columns) != OUT_COLUMNS:
        raise SystemExit(f"STOP: колонки {list(df.columns)} не совпадают с ожидаемыми {OUT_COLUMNS}")
    if len(df) != n_mo * len(RETAIL_YEARS) or df.duplicated(["territory_id", "year"]).any():
        raise SystemExit(f"STOP: {len(df)} строк или дубли (МО, год); ожидается {n_mo} × {len(RETAIL_YEARS)}")
    print(f"а) схема: {len(df)} строк, пара (territory_id, year) уникальна, {len(df.columns)} колонок")


def build(valid: pd.DataFrame, raw: pd.DataFrame) -> tuple:
    code2tid, latest_codes = step17.load_key(valid)
    m = step17.match(raw, code2tid, latest_codes)
    res = step17.resolve(m, "retail", ["indicator_code", "obroz"])
    wide = res.pivot(index=["territory_id", "year"], columns=["indicator_code", "obroz"], values="value")
    idx = pd.MultiIndex.from_product([sorted(valid["territory_id"].astype(int)), RETAIL_YEARS], names=["territory_id", "year"])
    wide = wide.reindex(idx)

    def col(code, obroz):
        return wide[(code, obroz)] if (code, obroz) in wide.columns else pd.Series(np.nan, index=idx)

    out = pd.DataFrame(index=idx)
    out["retail_floor_m2"] = col(CODE_FLOOR, SHOPS)
    out["retail_stores"] = col(CODE_OBJECTS, SHOPS)
    out["catering_seats_pub"] = pd.concat([col(CODE_SEATS, CAFES), col(CODE_SEATS, CANTEENS_PUBLIC)], axis=1).sum(axis=1, min_count=1)
    out["catering_seats_all"] = pd.concat([col(CODE_SEATS, CAFES), col(CODE_SEATS, CANTEENS_PUBLIC),
                                           col(CODE_SEATS, CANTEENS_INSTITUTIONAL)], axis=1).sum(axis=1, min_count=1)
    out["cafes_objects"] = col(CODE_OBJECTS, CAFES)
    used = res[((res["indicator_code"] == CODE_FLOOR) & (res["obroz"] == SHOPS))
               | ((res["indicator_code"] == CODE_OBJECTS) & res["obroz"].isin([SHOPS, CAFES]))
               | ((res["indicator_code"] == CODE_SEATS) & res["obroz"].isin([CAFES, CANTEENS_PUBLIC, CANTEENS_INSTITUTIONAL]))]
    mb = used.groupby(["territory_id", "year"])["matched_by"].agg(lambda s: "; ".join(sorted(set(s))))
    out["matched_by"] = mb.reindex(idx)
    out = out.reset_index()
    out["territory_id"] = out["territory_id"].astype(valid["territory_id"].dtype)
    out["year"] = out["year"].astype("int64")
    return out[OUT_COLUMNS].sort_values(["territory_id", "year"], ignore_index=True), wide


def checks(out: pd.DataFrame, wide: pd.DataFrame) -> None:
    value_cols = OUT_COLUMNS[2:-1]
    cov = out.groupby("year")[value_cols].count()
    print(f"б) МО с непустым значением (из {out['territory_id'].nunique()}):\n{cov.to_string()}")
    lab = pd.read_parquet(LABELS_FINAL_PATH).set_index("territory_id")["cluster"]
    o = out.assign(cluster=out["territory_id"].map(lab))
    by_cl = o.groupby(["cluster", "year"])[["retail_floor_m2", "catering_seats_pub"]].agg(lambda s: f"{s.notna().sum()} из {len(s)}")
    print(f"б) по кластерам k = 6:\n{by_cl.unstack('year').to_string()}")

    sub = pd.concat([wide[(CODE_FLOOR, SHOPS)].rename("shops")]
                    + [wide[(CODE_FLOOR, t)].rename(t) if (CODE_FLOOR, t) in wide.columns else pd.Series(np.nan, index=wide.index, name=t)
                       for t in FLOOR_SUBTYPES], axis=1)
    full = sub.dropna()
    ratio = full["shops"] / full[FLOOR_SUBTYPES].sum(axis=1)
    lo, hi = RETAIL_CHECK_BAND
    rows = []
    for y, r in list(ratio.groupby(level="year")) + [("оба года", ratio)]:
        rows.append({"год": y, "МО-лет со всеми девятью видами": len(r), "медиана отношения": round(float(r.median()), 4),
                     f"доля в [{lo}, {hi}]": round(float(((r >= lo) & (r <= hi)).mean()), 4)})
    print(f"в) справочно, вариант 1 — МО-годы со всеми девятью видами; площадь торгового зала: «{SHOPS}» / сумма "
          f"девяти видов ({', '.join(FLOOR_SUBTYPES)}):\n{pd.DataFrame(rows).to_string(index=False)}")
    shops = sub[sub["shops"].notna()]
    rest = shops[FLOOR_SUBTYPES].fillna(0).sum(axis=1)
    ratio2 = (shops["shops"] / rest)[rest > 0]
    rows = [{"год": y, "МО с «Магазины»": int((shops.index.get_level_values("year") == y).sum()),
             "из них с суммой видов > 0": len(r), "медиана отношения": round(float(r.median()), 4),
             f"доля в [{lo}, {hi}]": round(float(((r >= lo) & (r <= hi)).mean()), 4)}
            for y, r in ratio2.groupby(level="year")]
    print("в) справочно, вариант 2 — все МО с «Магазины», сумма имеющихся из девяти видов (пропуск = 0):\n"
          f"{pd.DataFrame(rows).to_string(index=False)}")

    pop = pd.read_parquet(POPULATION_PATH)
    pop = pop[~pop["anomaly"] & (pop["population"] > 0)].set_index(["territory_id", "year"])["population"]
    o = out.set_index(["territory_id", "year"])
    rows = []
    for c in ["retail_floor_m2", "catering_seats_pub"]:
        for y in RETAIL_YEARS:
            per = (o[c] / pop.reindex(o.index) * 1000).xs(y, level="year").dropna()
            q = per.quantile(QUANTILES)
            rows.append({"показатель на 1000 жителей": c, "год": y, "n": len(per),
                         **{f"{int(p * 100)}%": round(float(q[p]), 2) for p in QUANTILES},
                         "≤ 0": int((per <= 0).sum()),
                         f"> {RETAIL_CHECK_MAX_FLOOR_PER_1000}": int((per > RETAIL_CHECK_MAX_FLOOR_PER_1000).sum())})
    print("г) на 1000 жителей (население шага 17 того же года; МО-годы с anomaly и населением ≤ 0 не участвуют):\n"
          f"{pd.DataFrame(rows).to_string(index=False)}")


def main() -> None:
    t0 = time.time()
    valid = pd.read_parquet(VALID_TERRITORIES_PATH)
    if not INPUT_PATH.exists():
        if OUT_PATH.exists():
            print(f"Входного файла нет ({INPUT_PATH}): используется готовый файл {OUT_PATH}")
            check_schema(pd.read_parquet(OUT_PATH), len(valid))
            return
        raise SystemExit(f"STOP: нет входного файла {INPUT_PATH} и нет готового файла {OUT_PATH}. Положите выгрузку "
                         "раздела 2 БД ПМО (tochno.st, см. data/external/README.md) и запустите снова.")
    out, wide = build(valid, read_input())
    check_schema(out, len(valid))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    print(f"Записано: {OUT_PATH} ({OUT_PATH.stat().st_size:,} байт)")
    checks(out, wide)
    print(f"весь шаг: {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
