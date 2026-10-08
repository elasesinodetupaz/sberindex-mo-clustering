"""Шаг 18a. Пороги для проверки гипотез Г1–Г4, Г7, Г8 (предрегистрация).

Пороги фиксируются ДО просмотра результатов гипотез: для переменных гипотез считаются только
нулевые распределения (перестановки), число МО с данными и размеры групп. Наблюдаемые эффекты
считаются только для контролей (положительный: share_Общепит, share_Транспорт; отрицательные:
независимый шум и «региональный шум»).

Разбиения: каноническое декабрьское (kmeans_labels_final) и 24 месячных (kmeans_labels_k6/).
Краскел–Уоллис (η²_H) — для переменных по типам; Спирмен — для пар Г2, Г3 (МО-уровень, по месяцам,
от разбиения не зависят). Уровни «общий» и «внутри регионов». Функции — src/hypothesis_tools.py.
Хэш sha256 таблицы порогов записывается в hypothesis_thresholds.sha256 и в отчёт; шаг 19 сверяет его.

Вход:  data/processed/{kmeans_labels_final, category_shares, railway_degree, valid_territories}.parquet,
       data/processed/kmeans_labels_k6/{YYYY-MM}.parquet, data/raw/{market_access, territories}.parquet,
       data/external/rosstat_mo_2023_2025.parquet
Выход: data/processed/hypothesis_thresholds.parquet, data/processed/hypothesis_null_dec.parquet,
       data/processed/hypothesis_thresholds.sha256, notebooks/18_hypothesis_thresholds.md
Запуск из корня проекта:  .venv/bin/python src/18_hypothesis_thresholds.py
"""
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kruskal, rankdata, spearmanr

from config import (ALPHA, HYP_CONTROL_SEED, HYP_CORR_PAIRS, HYP_KW_VARIABLES, HYP_MIN_REGION_N, HYP_MIN_TYPES,
                    HYP_NEG1_N_VARS, HYP_NEG2_N_VARS, HYP_NEG_MAX_SHARE, HYP_PERM_BATCH, HYP_POSITIVE_CONTROLS,
                    HYP_SEED, MIN_GROUP, MONTH, MONTHS_RULE, N_PERM, ROBUST_N_JOBS)
from hypothesis_tools import KW, LEVEL_ALL, LEVEL_REGION, LEVELS, Spearman, kw_eta2, perm_index, prepare, verdict_kw

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
LABELS_FINAL_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
MONTHLY_LABELS_DIR = PROCESSED_DIR / "kmeans_labels_k6"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
RAILWAY_PATH = PROCESSED_DIR / "railway_degree.parquet"
VALID_TERRITORIES_PATH = PROCESSED_DIR / "valid_territories.parquet"
MARKET_ACCESS_PATH = PROJECT_DIR / "data" / "raw" / "market_access.parquet"
TERRITORIES_PATH = PROJECT_DIR / "data" / "raw" / "territories.parquet"
ROSSTAT_PATH = PROJECT_DIR / "data" / "external" / "rosstat_mo_2023_2025.parquet"
THRESHOLDS_PATH = PROCESSED_DIR / "hypothesis_thresholds.parquet"
NULL_DEC_PATH = PROCESSED_DIR / "hypothesis_null_dec.parquet"
SHA_PATH = PROCESSED_DIR / "hypothesis_thresholds.sha256"
REPORT_PATH = PROJECT_DIR / "notebooks" / "18_hypothesis_thresholds.md"

DEC = "dec"                     # каноническое декабрьское разбиение (kmeans_labels_final)
STATIC = "mean2024"             # «партиция» корреляций, где обе переменные статичны
FEDERAL = ["Москва", "Санкт-Петербург", "Севастополь"]
EXPECTED_FEDERAL = {"Москва": 144, "Санкт-Петербург": 98, "Севастополь": 0}
EXPECTED_FEDERAL_CLUSTERS = {1: 13, 3: 229}
OKVED_LETTERS = list("ABCDEFGHIJKLMNOPQRS")
COLUMNS = ["hypothesis", "variable", "variant", "test", "level", "partition", "n", "k_eff", "min_group",
           "regions_excluded", "null_mean", "p95", "p99", "expected_sign", "p95_two", "p99_two", "seed", "n_perm",
           "status"]


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def load() -> dict:
    ids = np.sort(pd.read_parquet(VALID_TERRITORIES_PATH)["territory_id"].to_numpy())
    terr = pd.read_parquet(TERRITORIES_PATH).set_index("territory_id").loc[ids]
    parts = {DEC: pd.read_parquet(LABELS_FINAL_PATH).set_index("territory_id")["cluster"].reindex(ids).to_numpy()}
    months = sorted(p.stem for p in MONTHLY_LABELS_DIR.glob("*.parquet"))
    if len(months) != 24 or months[-1] != MONTH:
        raise SystemExit(f"STOP: ожидается 24 месячных разбиения до {MONTH}, найдено {months}")
    for m in months:
        parts[m] = pd.read_parquet(MONTHLY_LABELS_DIR / f"{m}.parquet").set_index("territory_id")["cluster"].reindex(ids).to_numpy()
    if any(np.isnan(p.astype(float)).any() for p in parts.values()):
        raise SystemExit("STOP: не у всех МО выборки есть метка в одном из разбиений")
    shares = pd.read_parquet(SHARES_PATH)
    shares["month"] = shares["date"].dt.strftime("%Y-%m")
    return {"ids": ids, "terr": terr, "region": terr["region_code"].to_numpy(), "parts": parts, "months": months,
            "shares": shares, "ma": pd.read_parquet(MARKET_ACCESS_PATH).set_index("territory_id")["market_access"],
            "rw": pd.read_parquet(RAILWAY_PATH).set_index("territory_id").reindex(ids),
            "ros": pd.read_parquet(ROSSTAT_PATH)}


def check_federal(d: dict, log: list) -> np.ndarray:
    fed = d["terr"]["region_name"].isin(FEDERAL).to_numpy()
    by_region = {r: int((d["terr"]["region_name"] == r).sum()) for r in FEDERAL}
    by_cluster = pd.Series(d["parts"][DEC][fed]).value_counts().sort_index().to_dict()
    log += [f"- МО городов федерального значения (по region_name): всего **{int(fed.sum())}**; {by_region}",
            f"- по кластерам канонического разбиения: {by_cluster}"]
    if {r: by_region[r] for r in ("Москва", "Санкт-Петербург")} != {r: EXPECTED_FEDERAL[r] for r in ("Москва", "Санкт-Петербург")} \
            or int(fed.sum()) != sum(EXPECTED_FEDERAL.values()) or by_cluster != EXPECTED_FEDERAL_CLUSTERS:
        raise SystemExit(f"STOP: МО городов федерального значения {by_region}, по кластерам {by_cluster}; "
                         f"ожидалось {EXPECTED_FEDERAL}, {EXPECTED_FEDERAL_CLUSTERS}")
    return fed


def check_railway(d: dict, log: list) -> None:
    rw = d["rw"]
    tab = (pd.DataFrame({"has_railway": rw["has_railway"], "before_impute>0": rw["railway_degree_before_impute"] > 0,
                         "railway_degree>0": rw["railway_degree"] > 0})
           .value_counts().rename("МО").reset_index())
    log += ["", "Сопряжённость has_railway × (railway_degree_before_impute > 0) × (railway_degree > 0):", "", md(tab), ""]
    print(tab.to_string(index=False))
    if rw["has_railway"].isna().any() or not (rw["has_railway"] == (rw["railway_degree_before_impute"] > 0)).all():
        raise SystemExit("STOP: has_railway не совпадает с (railway_degree_before_impute > 0)")


def clean_rosstat(d: dict, log: list) -> None:
    ros = d["ros"]
    bad = ros[ros["anomaly"]]
    names = d["terr"]["name"].reindex(bad["territory_id"]).to_numpy()
    regions = d["terr"]["region_name"].reindex(bad["territory_id"]).to_numpy()
    tab = pd.DataFrame({"territory_id": bad["territory_id"].to_numpy(), "year": bad["year"].to_numpy(),
                        "название": names, "регион": regions})
    log += ["МО-годы Росстата с anomaly == True исключены из всех переменных Росстата этого года:", "", md(tab), ""]
    value_cols = [c for c in ros.columns if c not in ("territory_id", "year", "matched_by", "anomaly")]
    ros.loc[ros["anomaly"], value_cols] = np.nan
    d["ros_by_year"] = {y: g.set_index("territory_id").reindex(d["ids"]) for y, g in ros.groupby("year")}


def values(spec: dict, month: str, d: dict) -> np.ndarray:
    """Значения переменной по МО выборки (NaN — нет данных). month — месяц разбиения."""
    src = spec["source"]
    if src == "market_access":
        v = d["ma"].reindex(d["ids"]).to_numpy(float)
    elif src == "railway":
        v = d["rw"][spec["column"]].astype(float).to_numpy()
    elif src == "rosstat":
        v = d["ros_by_year"][spec["year"]][spec["column"]].to_numpy(float)
    elif src == "shares":
        sh = d["shares"]
        if spec["period"] == "month":
            s = sh[sh["month"] == month].set_index("territory_id")[spec["column"]]
        elif spec["period"] == "mean2024":
            s = sh[sh["month"].str.startswith("2024")].groupby("territory_id")[spec["column"]].mean()
        else:
            raise SystemExit(f"STOP: неизвестный period {spec['period']}")
        v = s.reindex(d["ids"]).to_numpy(float)
    elif src == "health_resid":
        h = values({"source": "shares", "column": spec["column"], "period": spec["period"]}, month, d)
        o = d["ros_by_year"][spec["older_year"]]["share_older"].to_numpy(float)
        m = ~np.isnan(h) & ~np.isnan(o)
        rh, ro = rankdata(h[m]), rankdata(o[m])
        slope, intercept = np.polyfit(ro, rh, 1)
        v = np.full(len(h), np.nan)
        v[m] = rh - (intercept + slope * ro)
    else:
        raise SystemExit(f"STOP: неизвестный source {src}")
    if spec.get("transform") == "log":
        if (v[~np.isnan(v)] <= 0).any():
            raise SystemExit(f"STOP: log от неположительных значений в {spec}")
        v = np.log(v)
    return v


def expand_kw_specs() -> list:
    out = []
    for s in HYP_KW_VARIABLES:
        if "*" in s.get("column", ""):
            for c in OKVED_LETTERS:
                out.append({**s, "column": s["column"].replace("*", c), "variable": s["variable"].replace("*", c)})
        else:
            out.append(dict(s))
    return out


def monthly(spec: dict) -> bool:
    return spec.get("period") == "month"


def kw_job(meta: dict, y: np.ndarray, labels: np.ndarray, region: np.ndarray, keep_null: bool):
    p = prepare(y, labels, region, meta["level"], MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
    row = {**meta, "test": "kw_eta2_H", "n": p["n"], "k_eff": p["k_eff"], "min_group": p["min_group"],
           "regions_excluded": len(p["excluded_regions"]), "status": p["status"], "n_perm": N_PERM}
    if p["status"] == "не оценивается":
        return row, None
    kw = KW(p["y"], p["labels"])
    if kw.c == 0:
        return {**row, "status": "не оценивается (все значения равны)"}, None
    null = kw.null(N_PERM, meta["seed"], HYP_PERM_BATCH, p["region"] if meta["level"] == LEVEL_REGION else None)
    row.update(null_mean=float(null.mean()), p95=float(np.quantile(null, 0.95)), p99=float(np.quantile(null, 0.99)))
    return row, (null if keep_null else None)


def corr_job(meta: dict, xy: np.ndarray, region: np.ndarray, keep_null: bool):
    p = prepare(xy, None, region, meta["level"], MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
    sp = Spearman(p["y"][:, 0], p["y"][:, 1], ranked=meta["level"] == LEVEL_REGION)
    null = sp.null(N_PERM, meta["seed"], HYP_PERM_BATCH, p["region"] if meta["level"] == LEVEL_REGION else None)
    lo = meta["expected_sign"] == "-"
    row = {**meta, "test": "spearman_rho", "n": p["n"], "regions_excluded": len(p["excluded_regions"]),
           "status": "ok", "n_perm": N_PERM, "null_mean": float(null.mean()),
           "p95": float(np.quantile(null, 0.05 if lo else 0.95)), "p99": float(np.quantile(null, 0.01 if lo else 0.99)),
           "p95_two": float(np.quantile(np.abs(null), 0.95)), "p99_two": float(np.quantile(np.abs(null), 0.99))}
    return row, (null if keep_null else None)


def build_jobs(d: dict, fed: np.ndarray) -> list:
    parts = [DEC] + d["months"]
    jobs = []
    for s in expand_kw_specs():
        for level in LEVELS:
            for part in parts:
                month = MONTH if part == DEC else part
                meta = {"hypothesis": s["hypothesis"], "variable": s["variable"], "variant": s["variant"],
                        "level": level, "partition": part}
                jobs.append(("kw", meta, values(s, month, d), d["parts"][part], part == DEC))
    for s in HYP_CORR_PAIRS:
        periods = d["months"] if (monthly(s["x"]) or monthly(s["y"])) else [STATIC]
        for level in LEVELS:
            for period in periods:
                month = MONTH if period == STATIC else period
                xy = np.column_stack([values(s["x"], month, d), values(s["y"], month, d)])
                if s["exclude_federal"]:
                    xy[fed] = np.nan
                meta = {"hypothesis": s["hypothesis"], "variable": s["variable"], "variant": s["variant"],
                        "level": level, "partition": period, "expected_sign": s["sign"]}
                jobs.append(("corr", meta, xy, None, period in (MONTH, STATIC)))
    for i, j in enumerate(jobs):
        j[1]["seed"] = HYP_SEED + i
    return jobs


def run_job(job, region):
    kind, meta, y, labels, keep = job
    return kw_job(meta, y, labels, region, keep) if kind == "kw" else corr_job(meta, y, region, keep)


def validate(log: list) -> None:
    rng = np.random.default_rng(HYP_CONTROL_SEED)
    rows = []
    for i in range(5):
        n, k = int(rng.integers(40, 400)), int(rng.integers(3, 8))
        y = rng.integers(0, 15, n).astype(float)          # целые значения — со связками
        lab = rng.integers(0, k, n)
        h_scipy = kruskal(*[y[lab == g] for g in np.unique(lab)]).statistic
        h_ours = kw_eta2(y, lab)[1]
        x = rng.integers(0, 10, n).astype(float)
        rho_scipy = spearmanr(x, y).statistic
        rho_ours = Spearman(x, y).observed()
        rows.append({"набор": i + 1, "n": n, "k": k, "H scipy": h_scipy, "H ours": h_ours, "|ΔH|": abs(h_scipy - h_ours),
                     "ρ scipy": rho_scipy, "ρ ours": rho_ours, "|Δρ|": abs(rho_scipy - rho_ours)})
    tab = pd.DataFrame(rows)
    region = rng.integers(0, 20, 300)
    idx = perm_index(rng, 50, 300, np.unique(region, return_inverse=True)[1])
    same_region = bool((region[idx] == region[None, :]).all())
    log += ["## Проверка реализации", "", md(tab), "",
            f"- max |ΔH| = {tab['|ΔH|'].max():.3g}, max |Δρ| = {tab['|Δρ|'].max():.3g}; "
            f"перестановки «внутри регионов» сохраняют регион каждого МО: {same_region}", ""]
    print(tab.to_string(index=False))
    print(f"max |ΔH| = {tab['|ΔH|'].max():.3g}, max |Δρ| = {tab['|Δρ|'].max():.3g}, внутри регионов: {same_region}")
    if tab["|ΔH|"].max() >= 1e-9 or tab["|Δρ|"].max() >= 1e-9 or not same_region:
        raise SystemExit("STOP: реализация расходится со scipy или перестановки не сохраняют регион")


def positive_controls(d: dict, region: np.ndarray, log: list) -> None:
    parts = [DEC] + d["months"]
    jobs = []
    for col in HYP_POSITIVE_CONTROLS:
        spec = {"source": "shares", "column": col, "period": "month"}
        for level in LEVELS:
            for i, part in enumerate(parts):
                meta = {"variable": col, "level": level, "partition": part,
                        "seed": HYP_CONTROL_SEED * 1000 + len(jobs)}
                jobs.append((meta, values(spec, MONTH if part == DEC else part, d), d["parts"][part]))

    def job(meta, y, labels):
        row, null = kw_job(meta, y, labels, region, True)
        p = prepare(y, labels, region, meta["level"], MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
        return {**row, "obs": KW(p["y"], p["labels"]).observed()}, null

    res = Parallel(n_jobs=ROBUST_N_JOBS)(delayed(job)(*j) for j in jobs)
    rows = []
    for col in HYP_POSITIVE_CONTROLS:
        for level in LEVELS:
            sel = [(r, nl) for r, nl in res if r["variable"] == col and r["level"] == level]
            dec = [(r, nl) for r, nl in sel if r["partition"] == DEC][0]
            mon = [r for r, _ in sel if r["partition"] != DEC]
            v = verdict_kw(dec[0]["obs"], dec[1], [r["obs"] for r in mon], [r["p95"] for r in mon], ALPHA, MONTHS_RULE)
            rows.append({"переменная": col, "уровень": level, "η²_H (дек.)": round(dec[0]["obs"], 4),
                         "p95 нуля (дек.)": round(dec[0]["p95"], 5), "p_dec": v["p_dec"],
                         "месяцев obs > p95 из 24": v["months_above_p95"], "вердикт": v["verdict"]})
    tab = pd.DataFrame(rows)
    log += ["### Положительный контроль (ожидается «подтверждено»)", "", md(tab), ""]
    print(tab.to_string(index=False))


def negative_controls(d: dict, region: np.ndarray, log: list) -> None:
    rng = np.random.default_rng(HYP_CONTROL_SEED)
    n = len(d["ids"])
    noise = rng.standard_normal((n, HYP_NEG1_N_VARS))
    reg_codes, reg_inv = np.unique(region, return_inverse=True)
    reg_noise = rng.standard_normal((len(reg_codes), HYP_NEG2_N_VARS))[reg_inv] + rng.standard_normal((n, HYP_NEG2_N_VARS))
    ys = np.column_stack([noise, reg_noise])
    parts = [DEC] + d["months"]
    ref = np.arange(n, dtype=float)   # у непрерывных переменных без пропусков нуль зависит только от набора рангов
    part_labels = d["parts"]

    def job(part, level, seed):
        labels = part_labels[part]
        p = prepare(ref, labels, region, level, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
        null = KW(p["y"], p["labels"]).null(N_PERM, seed, HYP_PERM_BATCH, p["region"] if level == LEVEL_REGION else None)
        obs = []
        for j in range(ys.shape[1]):
            q = prepare(ys[:, j], labels, region, level, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
            obs.append(KW(q["y"], q["labels"]).observed())
        return part, level, null, np.array(obs)

    tasks = [(part, level, HYP_CONTROL_SEED * 10000 + i) for i, (part, level) in
             enumerate((p, lv) for lv in LEVELS for p in parts)]
    res = {(p, lv): (nl, ob) for p, lv, nl, ob in Parallel(n_jobs=ROBUST_N_JOBS)(delayed(job)(*t) for t in tasks)}
    rows = []
    for level in LEVELS:
        null_dec, obs_dec = res[(DEC, level)]
        p95_m = np.array([np.quantile(res[(m, level)][0], 0.95) for m in d["months"]])
        obs_m = np.column_stack([res[(m, level)][1] for m in d["months"]])
        conf = np.array([verdict_kw(obs_dec[j], null_dec, obs_m[j], p95_m, ALPHA, MONTHS_RULE)["verdict"] == "подтверждено"
                         for j in range(ys.shape[1])])
        pdec = np.array([(1 + np.sum(null_dec >= o)) / (1 + N_PERM) for o in obs_dec])
        for name, sl in [("№1: N(0,1)", slice(0, HYP_NEG1_N_VARS)), ("№2: a_region + e", slice(HYP_NEG1_N_VARS, None))]:
            rows.append({"контроль": name, "уровень": level, "переменных": int(conf[sl].size),
                         "доля p_dec < ALPHA": round(float((pdec[sl] < ALPHA).mean()), 4),
                         "доля «подтверждено»": round(float(conf[sl].mean()), 4)})
    tab = pd.DataFrame(rows)
    log += ["### Отрицательные контроли", "",
            "Нулевое распределение для контролей считается один раз на (разбиение, уровень): у непрерывных "
            "переменных без пропусков оно зависит только от набора рангов (и регионов), а не от самих значений.", "",
            md(tab), "",
            f"Требование: №1 на обоих уровнях и №2 «внутри регионов» — не больше {HYP_NEG_MAX_SHARE}; "
            "№2 «общий» ожидается высоким (типы сосредоточены в регионах).", ""]
    print(tab.to_string(index=False))
    fail = tab[((tab["контроль"].str.startswith("№1")) | (tab["уровень"] == LEVEL_REGION))
               & (tab["доля «подтверждено»"] > HYP_NEG_MAX_SHARE)]
    if len(fail):
        raise SystemExit(f"STOP: отрицательный контроль превысил {HYP_NEG_MAX_SHARE}:\n{fail.to_string(index=False)}")


def main() -> None:
    t0 = time.time()
    log = []
    d = load()
    fed = check_federal(d, log)
    print("\n".join(log))
    check_railway(d, log)
    clean_rosstat(d, log)
    region = d["region"]
    val_log = []
    validate(val_log)

    t1 = time.time()
    jobs = build_jobs(d, fed)
    print(f"комбинаций: {len(jobs)} (kw {sum(j[0] == 'kw' for j in jobs)}, spearman {sum(j[0] == 'corr' for j in jobs)})")
    res = Parallel(n_jobs=ROBUST_N_JOBS, verbose=0)(delayed(run_job)(j, region) for j in jobs)
    thr = pd.DataFrame([r for r, _ in res]).reindex(columns=COLUMNS)
    for c in ["n", "k_eff", "min_group", "regions_excluded", "seed", "n_perm"]:
        thr[c] = thr[c].astype("Int64")
    nulls = pd.concat([pd.DataFrame({"hypothesis": r["hypothesis"], "variable": r["variable"], "level": r["level"],
                                     "partition": r["partition"], "perm": np.arange(len(nl)), "value": nl})
                       for r, nl in res if nl is not None], ignore_index=True)
    t_thr = time.time() - t1
    print(f"пороги: {t_thr:.1f} с")

    t2 = time.time()
    ctrl_log = []
    positive_controls(d, region, ctrl_log)
    negative_controls(d, region, ctrl_log)
    t_ctrl = time.time() - t2
    print(f"контроли: {t_ctrl:.1f} с")

    thr.to_parquet(THRESHOLDS_PATH, engine="pyarrow", index=False)
    nulls.to_parquet(NULL_DEC_PATH, engine="pyarrow", index=False)
    sha = hashlib.sha256(THRESHOLDS_PATH.read_bytes()).hexdigest()
    SHA_PATH.write_text(f"{sha}  {THRESHOLDS_PATH.name}\n", encoding="utf-8")

    dec = thr[thr["partition"] == DEC]
    dec_tab = dec[["hypothesis", "variable", "variant", "level", "n", "k_eff", "min_group", "regions_excluded",
                   "null_mean", "p95", "p99", "status"]]
    corr_dec = thr[(thr["test"] == "spearman_rho") & thr["partition"].isin([MONTH, STATIC])][
        ["hypothesis", "variable", "variant", "level", "partition", "n", "regions_excluded", "expected_sign",
         "null_mean", "p95", "p99", "p95_two", "p99_two"]]
    mon = thr[~thr["partition"].isin([DEC, STATIC])]
    summ = (mon.groupby(["hypothesis", "variable", "test", "level"], sort=False)["p95"]
            .agg(["min", "median", "max"]).reset_index())
    base = prepare(np.zeros(len(d["ids"])), None, region, LEVEL_REGION, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
    small_regions = d["terr"].drop_duplicates("region_code").set_index("region_code").loc[base["excluded_regions"], "region_name"]
    counts = pd.DataFrame([{"переменная": s["variable"], "МО с данными (дек.)": int((~np.isnan(values(s, MONTH, d))).sum())}
                           for s in expand_kw_specs()])

    lines = [
        "# 18a. Пороги для проверки гипотез Г1–Г4, Г7, Г8 (предрегистрация)", "",
        "Сгенерировано `src/18_hypothesis_thresholds.py`. Пороги зафиксированы до просмотра результатов гипотез: "
        "для переменных гипотез здесь нет наблюдаемых эффектов — только нулевые распределения, число МО с "
        "данными и размеры групп. Наблюдаемые эффекты приведены только для контролей.", "",
        "## Заморозка", "",
        f"sha256 `{THRESHOLDS_PATH.name}`: `{sha}` (также в `{SHA_PATH.name}`). Шаг 19 сверяет хэш перед работой.", "",
        "## Формулы", "",
        "- H Краскела–Уоллиса с поправкой на связки: H = [12 / (n(n+1)) · Σ R_g² / n_g − 3(n+1)] / "
        "[1 − Σ(t³ − t) / (n³ − n)]; k_eff — число непустых типов; η²_H = max(0, (H − k_eff + 1) / (n − k_eff)).",
        "- Спирмен: ρ = коэффициент Пирсона на рангах (средние ранги при связках).",
        f"- Нуль: {N_PERM} перестановок меток типов (для ρ — одной из переменных); seed комбинации = "
        f"{HYP_SEED} + номер строки в таблице порогов. Порог p95 / p99 — 95-й / 99-й процентиль нуля; для ρ — "
        "односторонний в ожидаемом направлении (для «−» — 5-й / 1-й процентиль) и двусторонний по |ρ|.",
        f"- Уровень «внутри регионов»: значение заменяется нормированным рангом внутри региона (region_code), "
        f"(ранг − 0,5) / n_региона; перестановки только внутри регионов; регионы с числом МО с данными "
        f"< {HYP_MIN_REGION_N} не участвуют (колонка regions_excluded).",
        f"- Тип с числом МО с данными < {MIN_GROUP} исключается из теста переменной (status); если осталось "
        f"< {HYP_MIN_TYPES} типов — «не оценивается».",
        f"- Вердикт verdict_kw: «подтверждено», если p_dec = (1 + #{{нуль ≥ obs}}) / (1 + {N_PERM}) < {ALPHA} "
        f"и obs_m > p95_m не менее чем в {MONTHS_RULE} из 24 месяцев.",
        "- Месячные проверки: доли category_shares за месяц m сопоставляются с разбиением месяца m; "
        "статичные переменные (market_access, ж/д, Росстат, среднее долей за 2024) — с разбиением каждого месяца. "
        f"Для канонического декабрьского разбиения доли берутся за {MONTH}. Корреляции Г2, Г3 не зависят от "
        f"разбиения: partition — месяц долей; `{STATIC}` — обе переменные статичны.", "",
        "## Данные", "", *log, "",
        "Число МО с данными (декабрь 2024, до отбора по типам и регионам):", "", md(counts), "",
        f"Регионы с числом МО выборки < {HYP_MIN_REGION_N} (не участвуют во «внутри регионов» ни в одной переменной): "
        f"{len(small_regions)} — {', '.join(small_regions.tolist()) or 'нет'}. У отдельных переменных из-за "
        "пропусков исключается больше регионов (колонка regions_excluded).", "",
        *val_log,
        "## Пороги: каноническое декабрьское разбиение (Краскел–Уоллис, η²_H)", "", md(dec_tab), "",
        f"## Пороги корреляций (Спирмен): месяц {MONTH} и статичные варианты", "", md(corr_dec), "",
        "## Сводка p95 по 24 месячным разбиениям (min / медиана / max)", "", md(summ), "",
        "## Контроли", "", *ctrl_log,
        "## Время", "",
        f"- пороги ({len(thr)} комбинаций): {t_thr:.1f} с; контроли: {t_ctrl:.1f} с; весь шаг: {time.time() - t0:.1f} с", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"sha256 {sha}")
    print(f"записано: {THRESHOLDS_PATH.name} ({len(thr)} строк), {NULL_DEC_PATH.name} ({len(nulls)} строк), "
          f"{SHA_PATH.name}, {REPORT_PATH.name}; весь шаг {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
