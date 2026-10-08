"""Шаг 21. Пороги для дополнительных гипотез Г10–Г13 (предложение розницы и общепита). Предрегистрация.

Гипотезы сформулированы после основных результатов 19a–19c, поэтому это отдельное «дополнительное»
семейство. Пороги фиксируются ДО просмотра результатов: для переменных Г10–Г13 здесь нет наблюдаемых
эффектов — только нулевые распределения, число МО с данными, размеры групп и квантили самих показателей
предложения. Статистики, уровни, отбор МО, MIN_GROUP, N_PERM и разбиения — те же, что в шагах 18/18b
(функции kw_job, corr_job, validate, load импортируются из src/18_hypothesis_thresholds.py без запуска его
расчёта). Замороженные файлы шагов 18–20 сверяются по sha256 до и после работы.

Г10: ρ(share_Маркетплейсы, floor_pc), знак «−»; Г11: ρ(share_Общепит, seats_pc), знак «+» — основной вариант по
24 месяцам, чувствительность — декабрь 2024. Г12, Г13: η²_H по типам для доли и её остатка после МНК на
рангах переменной предложения — 25 разбиений.

Вход:  входы шага 18, data/external/rosstat_retail_2023_2024.parquet, data/external/rosstat_mo_2023_2025.parquet
Выход: data/processed/hypothesis_thresholds_supply.parquet, data/processed/hypothesis_thresholds_supply.sha256,
       notebooks/21_supply_thresholds.md
Запуск из корня проекта:  .venv/bin/python src/21_supply_thresholds.py
"""
import hashlib
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import (ALPHA, HYP_MIN_REGION_N, HYP_MIN_TYPES, HYP_NEG1_N_VARS, HYP_NEG2_N_VARS, HYP_NEG_MAX_SHARE,
                    HYP_PERM_BATCH, MIN_GROUP, MONTH, N_PERM, ROBUST_N_JOBS, SEED_21, SUPPLY_CORR_PAIRS, SUPPLY_KW,
                    SUPPLY_VARIABLES)
from hypothesis_tools import LEVEL_REGION, LEVELS, Spearman, prepare, rank_residual

step18 = importlib.import_module("18_hypothesis_thresholds")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RETAIL_PATH = PROJECT_DIR / "data" / "external" / "rosstat_retail_2023_2024.parquet"
POPULATION_PATH = PROJECT_DIR / "data" / "external" / "rosstat_mo_2023_2025.parquet"
THRESHOLDS_PATH = PROCESSED_DIR / "hypothesis_thresholds_supply.parquet"
SHA_PATH = PROCESSED_DIR / "hypothesis_thresholds_supply.sha256"
REPORT_PATH = PROJECT_DIR / "notebooks" / "21_supply_thresholds.md"

FROZEN = {
    PROCESSED_DIR / "hypothesis_thresholds.parquet": "11081afbfc7e88dbd301004e4c739e3f25e00bf8a47704cf7c74e15e8e6665f7",
    PROCESSED_DIR / "hypothesis_null_dec.parquet": "40735907ad605157e95ab29e05b7288ec0af61ea085360f9838e515a830a1178",
    PROCESSED_DIR / "hypothesis_thresholds_g6g9.parquet": "1926061b7319ba9210d9520e087dbcf7c5107a7bcaea9cb2e7cacdba687022a1",
    PROCESSED_DIR / "hypothesis_series_g9.parquet": "dbf1fce8c74b5ea373cd61f4ec909f41d22fa6c6cfa2be413e16fafe2e8bce6e",
    PROCESSED_DIR / "hypothesis_results_19a.parquet": "58e1516ccbf03b2a1fca82971d5170f581815fe18fd8d22bbf261862fdad9d8e",
    PROCESSED_DIR / "hypothesis_results_19b.parquet": "d66d64d68a82867a5add988d4441fb0df2d35297b8c94c1514bc9e513c615d9d",
    RETAIL_PATH: "58f32111b48c4189811f61c6f462d9b0b8e06d3da431dd99866c8f33e1378a80",
}
DEC = step18.DEC
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
CONTROL_SEED_BASE = SEED_21 * 10          # seeds отрицательных контролей (данные и нули), отдельно от комбинаций


def md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(index=False, **kw)


def check_frozen(when: str, log: list) -> None:
    bad = []
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{when}: {got}  {path.relative_to(PROJECT_DIR)} {'OK' if got == expected else 'MISMATCH'}")
        log.append(f"- {when}: `{path.relative_to(PROJECT_DIR)}` — {'совпадает' if got == expected else 'НЕ СОВПАДАЕТ'}")
        if got != expected:
            bad.append(path.name)
    if bad:
        raise SystemExit(f"STOP: sha256 не совпадает: {bad}")


def supply_values(ids: np.ndarray) -> tuple:
    """Переменные предложения на 1000 жителей; МО-годы с anomaly, населением ≤ 0 или пропуском — NaN."""
    retail = pd.read_parquet(RETAIL_PATH).set_index(["territory_id", "year"])
    pop = pd.read_parquet(POPULATION_PATH)
    pop = pop[~pop["anomaly"] & (pop["population"] > 0)].set_index(["territory_id", "year"])["population"]
    out, excluded = {}, []
    for v in SUPPLY_VARIABLES:
        idx = pd.MultiIndex.from_arrays([ids, np.full(len(ids), v["year"])])
        num = retail[v["column"]].reindex(idx).to_numpy(float)
        den = pop.reindex(idx).to_numpy(float)
        out[v["name"]] = num / den * 1000
        excluded.append({"переменная": v["name"], "показатель": v["column"], "год": v["year"],
                         "МО с показателем": int(np.sum(~np.isnan(num))),
                         "из них без населения (anomaly, ≤ 0, пропуск)": int(np.sum(~np.isnan(num) & np.isnan(den))),
                         "МО с данными": int(np.sum(~np.isnan(out[v['name']])))})
    return out, pd.DataFrame(excluded)


def share(d: dict, column: str, month: str) -> np.ndarray:
    return step18.values({"source": "shares", "column": column, "period": "month"}, month, d)


def build_jobs(d: dict, sup: dict) -> list:
    jobs = []
    parts = [DEC] + d["months"]
    for s in SUPPLY_CORR_PAIRS:
        periods = d["months"] if s["variant"] == "основной" else [MONTH]
        for level in LEVELS:
            for period in periods:
                xy = np.column_stack([share(d, s["share"], period), sup[s["supply"]]])
                meta = {"hypothesis": s["hypothesis"], "variable": f"{s['share']}~{s['supply']}", "variant": s["variant"],
                        "level": level, "partition": period, "expected_sign": s["sign"]}
                jobs.append(("corr", meta, xy, None))
    for s in SUPPLY_KW:
        for kind in ["доля", "остаток"]:
            for level in LEVELS:
                for part in parts:
                    month = MONTH if part == DEC else part
                    y = share(d, s["share"], month)
                    y = np.where(~np.isnan(sup[s["supply"]]), y, np.nan)
                    if kind == "остаток":
                        y = rank_residual(y, sup[s["supply"]])
                    name = s["share"] if kind == "доля" else f"{s['share']}_resid_{s['supply']}"
                    meta = {"hypothesis": s["hypothesis"], "variable": name, "variant": "основной", "level": level,
                            "partition": part}
                    jobs.append(("kw", meta, y, d["parts"][part]))
    for i, j in enumerate(jobs):
        j[1]["seed"] = SEED_21 + i
    return jobs


def run_job(job, region):
    kind, meta, y, labels = job
    row = (step18.kw_job(meta, y, labels, region, False) if kind == "kw" else step18.corr_job(meta, y, region, False))[0]
    return row


def negative_controls(d: dict, log: list) -> None:
    """500 пар N(0,1) и 200 пар «региональный шум» (общая региональная компонента у обеих переменных):
    двусторонний p по перестановкам. У пар без связок и пропусков нуль ρ зависит только от набора рангов
    (и регионов), поэтому он считается один раз на уровень."""
    region = d["region"]
    n = len(d["ids"])
    rng = np.random.default_rng(CONTROL_SEED_BASE)
    reg_codes, reg_inv = np.unique(region, return_inverse=True)
    x1, y1 = rng.standard_normal((n, HYP_NEG1_N_VARS)), rng.standard_normal((n, HYP_NEG1_N_VARS))
    a = rng.standard_normal((len(reg_codes), HYP_NEG2_N_VARS))[reg_inv]
    x2 = a + rng.standard_normal((n, HYP_NEG2_N_VARS))
    y2 = a + rng.standard_normal((n, HYP_NEG2_N_VARS))
    rows = []
    for li, level in enumerate(LEVELS):
        ref = prepare(np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float)[::-1]]), None, region,
                      level, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
        null = Spearman(ref["y"][:, 0], ref["y"][:, 1], ranked=level == LEVEL_REGION).null(
            N_PERM, CONTROL_SEED_BASE + 1 + li, HYP_PERM_BATCH, ref["region"] if level == LEVEL_REGION else None)
        abs_null = np.abs(null)
        for name, xs, ys in [("N(0,1)", x1, y1), ("региональный шум", x2, y2)]:
            p = []
            for j in range(xs.shape[1]):
                q = prepare(np.column_stack([xs[:, j], ys[:, j]]), None, region, level, MIN_GROUP, HYP_MIN_TYPES,
                            HYP_MIN_REGION_N)
                obs = Spearman(q["y"][:, 0], q["y"][:, 1], ranked=level == LEVEL_REGION).observed()
                p.append((1 + np.sum(abs_null >= abs(obs))) / (1 + N_PERM))
            rows.append({"контроль": name, "уровень": level, "пар": xs.shape[1],
                         "доля p < ALPHA": round(float(np.mean(np.array(p) < ALPHA)), 4)})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    log += ["### Отрицательные контроли (ρ Спирмена, двусторонний p)", "",
            "Нуль ρ для пар без связок и пропусков зависит только от набора рангов (и регионов), поэтому он "
            "считается один раз на уровень.", "", md(tab), "",
            f"Требование: доля p < {ALPHA} на уровне «внутри регионов» не больше {HYP_NEG_MAX_SHARE} у обоих контролей; "
            "у регионального шума на общем уровне ожидается высокой.", ""]
    fail = tab[(tab["уровень"] == LEVEL_REGION) & (tab["доля p < ALPHA"] > HYP_NEG_MAX_SHARE)]
    if len(fail):
        raise SystemExit(f"STOP: отрицательный контроль превысил {HYP_NEG_MAX_SHARE}:\n{fail.to_string(index=False)}")


def main() -> None:
    t0 = time.time()
    log, frozen_log = [], []
    check_frozen("до", frozen_log)
    d = step18.load()
    ros_log = []
    step18.clean_rosstat(d, ros_log)
    sup, sup_tab = supply_values(d["ids"])
    print(sup_tab.to_string(index=False))
    val_log = []
    step18.validate(val_log)

    t1 = time.time()
    jobs = build_jobs(d, sup)
    print(f"комбинаций: {len(jobs)} (spearman {sum(j[0] == 'corr' for j in jobs)}, kw {sum(j[0] == 'kw' for j in jobs)})")
    rows = Parallel(n_jobs=ROBUST_N_JOBS)(delayed(run_job)(j, d["region"]) for j in jobs)
    thr = pd.DataFrame(rows).reindex(columns=step18.COLUMNS)
    for c in ["n", "k_eff", "min_group", "regions_excluded", "seed", "n_perm"]:
        thr[c] = thr[c].astype("Int64")
    t_thr = time.time() - t1

    t2 = time.time()
    ctrl_log = []
    negative_controls(d, ctrl_log)
    t_ctrl = time.time() - t2

    thr.to_parquet(THRESHOLDS_PATH, engine="pyarrow", index=False)
    sha = hashlib.sha256(THRESHOLDS_PATH.read_bytes()).hexdigest()
    SHA_PATH.write_text(f"{sha}  {THRESHOLDS_PATH.name}\n", encoding="utf-8")
    check_frozen("после", frozen_log)

    dec = thr[(thr["partition"] == DEC) | ((thr["test"] == "spearman_rho") & (thr["partition"] == MONTH))]
    dec_kw = dec[dec["test"] == "kw_eta2_H"][["hypothesis", "variable", "variant", "level", "n", "k_eff", "min_group",
                                             "regions_excluded", "null_mean", "p95", "p99", "status"]]
    dec_corr = dec[dec["test"] == "spearman_rho"][["hypothesis", "variable", "variant", "level", "partition", "n",
                                                   "regions_excluded", "expected_sign", "null_mean", "p95", "p99",
                                                   "p95_two", "p99_two"]]
    mon = thr[thr["partition"] != DEC]
    summ = mon[mon["variant"] == "основной"].groupby(["hypothesis", "variable", "test", "level"], sort=False)["p95"].agg(
        ["min", "median", "max"]).reset_index()
    lab = d["parts"][DEC]
    by_type = pd.DataFrame({v["name"]: pd.Series(~np.isnan(sup[v["name"]])).groupby(lab).sum() for v in SUPPLY_VARIABLES})
    by_type.index.name = "тип (декабрь 2024)"
    by_type = by_type.reset_index()
    quant = pd.DataFrame([{"переменная": k, "n": int(np.sum(~np.isnan(sup[k]))),
                           **{f"{int(q * 100)}%": round(float(np.nanquantile(sup[k], q)), 2) for q in QUANTILES}}
                          for k in ["floor_pc", "seats_pc"]])

    lines = [
        "# 21. Пороги для дополнительных гипотез Г10–Г13 (предрегистрация)", "",
        "Сгенерировано `src/21_supply_thresholds.py`. Гипотезы сформулированы после основных результатов 19a–19c "
        "и портретов типов — это отдельное «дополнительное» семейство. Пороги зафиксированы до просмотра результатов: "
        "наблюдаемых эффектов для переменных Г10–Г13 здесь нет.", "",
        "## Заморозка", "", f"sha256 `{THRESHOLDS_PATH.name}`: `{sha}` (также в `{SHA_PATH.name}`).", "",
        "Файлы шагов 18–20 до и после работы:", "", *frozen_log, "",
        "## Определения", "",
        "- floor_pc = retail_floor_m2 / население на 1 января × 1000; seats_pc = catering_seats_pub / население × 1000 "
        "(2023 — основной вариант). Чувствительность: floor_pc_2024, seats_pc_2024 (данные и население 2024), stores_pc = "
        "retail_stores / население × 1000 (Г10), seats_all_pc = catering_seats_all / население × 1000 (Г11). Население и "
        "флаг anomaly — `rosstat_mo_2023_2025.parquet`; МО с anomaly, населением ≤ 0 или пропуском исключаются.",
        f"- Г10: ρ(share_Маркетплейсы, floor_pc), ожидаемый знак «−»; Г11: ρ(share_Общепит, seats_pc), знак «+». "
        f"Основной вариант — 24 месяца (доля месяца m, статичная переменная предложения); чувствительность — только {MONTH}. "
        "p95, p99 — односторонние в ожидаемом направлении (для «−» — 5-й и 1-й процентили), p95_two, p99_two — по |ρ|.",
        "- Г12, Г13: η²_H по типам на МО, где есть и доля, и переменная предложения: сырая доля (share_Маркетплейсы, "
        "share_Общепит) и её остаток после МНК на рангах floor_pc / seats_pc (как для Г7: ранги, МНК на глобальных рангах, "
        "на уровне «внутри регионов» затем ранг внутри региона). 25 разбиений (каноническое декабрьское и 24 месячных), "
        "доли — за месяц разбиения.",
        f"- Как в шагах 18/18b: уровни «общий» и «внутри регионов», регионы с < {HYP_MIN_REGION_N} МО с данными не "
        f"участвуют, тип с < {MIN_GROUP} МО исключается, {N_PERM} перестановок; seed комбинации = {SEED_21} + номер строки.", "",
        "## Данные", "", md(sup_tab), "", "Число МО с данными по переменным и типам (канонические типы, декабрь 2024):", "",
        md(by_type), "", "Квантили переменных предложения по всем МО (без разбивки по типам):", "", md(quant), "",
        *ros_log, "",
        f"## Пороги: декабрь ({MONTH}) и каноническое декабрьское разбиение", "", "### Спирмен (Г10, Г11)", "",
        md(dec_corr), "", "### Краскел–Уоллис (Г12, Г13)", "", md(dec_kw), "",
        "## Сводка p95 по 24 месяцам (основной вариант; min / медиана / max)", "", md(summ), "",
        "## Проверки", "", *val_log, *ctrl_log,
        "## Время", "", f"- пороги ({len(thr)} комбинаций): {t_thr:.1f} с; контроли: {t_ctrl:.1f} с; весь шаг: "
        f"{time.time() - t0:.1f} с", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    for name, tab in [("СПИРМЕН, декабрь", dec_corr), ("КРАСКЕЛ–УОЛЛИС, декабрь", dec_kw), ("МО ПО ТИПАМ", by_type),
                      ("КВАНТИЛИ", quant)]:
        print(f"===== {name}\n{tab.to_string(index=False)}")
    print(f"sha256 {sha}\nвесь шаг: {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
