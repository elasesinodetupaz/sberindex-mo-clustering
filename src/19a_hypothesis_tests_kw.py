"""Шаг 19a. Проверка гипотез Г1, Г4, Г7, Г8: критерий Краскела–Уоллиса (η²_H) по типам.

Пороги заморожены шагом 18a: перед расчётом сверяется sha256 таблицы порогов и нулевых векторов.
Переменные, уровни и разбиения — те же, что в шаге 18 (функции построения переменных импортируются
из src/18_hypothesis_thresholds.py без запуска его расчёта). Никаких новых переменных, порогов, фильтров
и объединений типов. Вердикт по гипотезам здесь не выносится (это шаг 19c): считаются obs_dec, p_dec,
месяцы выше порога, confirmed_level по правилу 18a, описание по типам, подпись размера по Коэну,
эталон — η²_H разбиения на федеральные округа; для Г1 — порядок типов по медиане log_market_access,
для Г7 — Δ = η²_H(share_health) − η²_H(остаток после учёта возраста) с бутстрепом по МО.

Вход:  файлы шага 18a (hypothesis_thresholds.parquet, hypothesis_null_dec.parquet), входы шага 18,
       data/processed/kmeans_k6_trajectories.parquet, data/external/federal_districts.csv
Выход: data/processed/hypothesis_results_19a.parquet, data/processed/hypothesis_results_19a.sha256,
       notebooks/19a_hypothesis_results.md
Запуск из корня проекта:  .venv/bin/python src/19a_hypothesis_tests_kw.py
"""
import hashlib
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import (ALPHA, BOOT_B, FINAL_K, HYP_CONTROL_SEED, HYP_MIN_REGION_N, HYP_MIN_TYPES, HYP_PERM_BATCH,
                    MIN_GROUP, MONTH, MONTHS_RULE, N_PERM, ROBUST_N_JOBS, SEED_G7)
from hypothesis_tools import KW, LEVEL_ALL, LEVEL_REGION, LEVELS, kw_eta2, p_upper, prepare, verdict_kw

step18 = importlib.import_module("18_hypothesis_thresholds")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
THRESHOLDS_PATH = PROCESSED_DIR / "hypothesis_thresholds.parquet"
NULL_DEC_PATH = PROCESSED_DIR / "hypothesis_null_dec.parquet"
TRAJ_PATH = PROCESSED_DIR / f"kmeans_k{FINAL_K}_trajectories.parquet"
FD_PATH = PROJECT_DIR / "data" / "external" / "federal_districts.csv"
RESULTS_PATH = PROCESSED_DIR / "hypothesis_results_19a.parquet"
SHA_PATH = PROCESSED_DIR / "hypothesis_results_19a.sha256"
REPORT_PATH = PROJECT_DIR / "notebooks" / "19a_hypothesis_results.md"

EXPECTED_SHA = {THRESHOLDS_PATH: "11081afbfc7e88dbd301004e4c739e3f25e00bf8a47704cf7c74e15e8e6665f7",
                NULL_DEC_PATH: "40735907ad605157e95ab29e05b7288ec0af61ea085360f9838e515a830a1178"}
DEC = step18.DEC
G1_VARIABLE = "log_market_access"
G1_TOP_TYPE, G1_BOTTOM_TYPE = 3, 4          # Г1: месяц засчитывается, если медиана max у типа 3 и min у типа 4
G7_HEALTH = "share_health"
G7_PAIRS = [("основной", "share_health", "share_health_resid_older2023"),
            ("чувствительность", "share_health", "share_health_resid_older2024")]
G7_NO_RESID = "share_health_mean2024"       # чувствительность без остатка в шаге 18: Δ не определена
G7_NO_RESID_NOTE = ("Δ не определена: в шаге 18 нет остатка для этой переменной; приведён confirmed_level "
                    "самой переменной из основной таблицы")
POSITIVE_CONTROL = "share_Общепит"
COHEN = [(0.01, "ничтожный"), (0.06, "малый"), (0.14, "средний"), (np.inf, "большой")]


def md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(index=False, **kw)


def size_label(eta2: float) -> str:
    return next(lab for bound, lab in COHEN if eta2 < bound)


def check_frozen(log: list) -> None:
    for path, expected in EXPECTED_SHA.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        log.append(f"- sha256 `{path.name}`: `{got}` — {'совпадает' if got == expected else 'НЕ СОВПАДАЕТ'}")
        print(f"sha256 {path.name} {got} {'OK' if got == expected else 'MISMATCH'}")
        if got != expected:
            raise SystemExit(f"STOP: sha256 {path.name} = {got}, ожидается {expected}")


def federal_district_labels(d: dict, log: list) -> np.ndarray:
    fd = pd.read_csv(FD_PATH, dtype=str, keep_default_na=False)
    missing = sorted(set(d["terr"]["region_name"]) - set(fd["region_name"]))
    if missing:
        raise SystemExit(f"STOP: регионов выборки нет в {FD_PATH.name}: {missing}")
    mapping = fd.set_index("region_name")["federal_district"].replace("", np.nan)
    lab = d["terr"]["region_name"].map(mapping)
    no_fd = lab.isna()
    log.append(f"- эталон «федеральные округа»: округов {lab.nunique()}; МО без округа (не участвуют): "
               f"{int(no_fd.sum())} в регионах {sorted(d['terr'].loc[no_fd, 'region_name'].unique())}")
    codes = pd.Series(pd.Categorical(lab).codes, dtype=float).where(~no_fd.to_numpy())
    return codes.to_numpy()


def eta2_on(y: np.ndarray, labels: np.ndarray, region: np.ndarray, level: str) -> tuple:
    p = prepare(y, labels, region, level, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
    return KW(p["y"], p["labels"]).observed(), p


def kw_tests(d: dict, thr: pd.DataFrame, nulls: pd.DataFrame, fd: np.ndarray) -> tuple:
    region = d["region"]
    rows, month_rows, desc_rows = [], [], []
    thr_idx = thr[thr["test"] == "kw_eta2_H"].set_index(["variable", "level", "partition"])
    null_idx = {k: g.sort_values("perm")["value"].to_numpy()
                for k, g in nulls[nulls["partition"] == DEC].groupby(["variable", "level"])}
    for s in step18.expand_kw_specs():
        y_dec = step18.values(s, MONTH, d)
        for level in LEVELS:
            obs_dec, p = eta2_on(y_dec, d["parts"][DEC], region, level)
            t = thr_idx.loc[(s["variable"], level, DEC)]
            got = (p["n"], p["k_eff"], len(p["excluded_regions"]))
            exp = (int(t["n"]), int(t["k_eff"]), int(t["regions_excluded"]))
            if got != exp:
                raise SystemExit(f"STOP: {s['variable']} / {level}: (n, k_eff, regions_excluded) = {got}, в порогах {exp}")
            p_dec = p_upper(obs_dec, null_idx[(s["variable"], level)])
            obs_m, p95_m = [], []
            for m in d["months"]:
                o, pm = eta2_on(step18.values(s, m, d), d["parts"][m], region, level)
                tm = thr_idx.loc[(s["variable"], level, m)]
                if (pm["n"], pm["k_eff"], len(pm["excluded_regions"])) != (int(tm["n"]), int(tm["k_eff"]), int(tm["regions_excluded"])):
                    raise SystemExit(f"STOP: {s['variable']} / {level} / {m}: n, k_eff или regions_excluded не как в порогах")
                obs_m.append(o)
                p95_m.append(float(tm["p95"]))
                month_rows.append({"row_type": "kw_month", "hypothesis": s["hypothesis"], "variable": s["variable"],
                                   "variant": s["variant"], "level": level, "partition": m, "obs": o, "p95": float(tm["p95"])})
            months_ok = int(np.sum(np.array(obs_m) > np.array(p95_m)))
            ok = t["status"] == "ok"
            row = {"row_type": "kw", "hypothesis": s["hypothesis"], "variable": s["variable"], "variant": s["variant"],
                   "level": level, "status": t["status"], "n": p["n"], "k_eff": p["k_eff"],
                   "regions_excluded": len(p["excluded_regions"]), "obs": obs_dec, "p_dec": p_dec,
                   "p95": float(t["p95"]), "months_ok": months_ok,
                   "confirmed_level": (p_dec < ALPHA and months_ok >= MONTHS_RULE) if ok else None,
                   "size_label": size_label(obs_dec)}
            if level == LEVEL_ALL:
                m_fd = ~np.isnan(y_dec) & ~np.isnan(fd)
                e_fd, _, k_fd = kw_eta2(y_dec[m_fd], fd[m_fd].astype(int))
                row.update(eta2_fd=e_fd, n_fd=int(m_fd.sum()), k_fd=int(k_fd))
            rows.append(row)
        ok_mask = ~np.isnan(y_dec)
        lab = d["parts"][DEC]
        for typ in sorted(np.unique(lab)):
            v = y_dec[ok_mask & (lab == typ)]
            desc_rows.append({"row_type": "describe", "hypothesis": s["hypothesis"], "variable": s["variable"],
                              "variant": s["variant"], "level": LEVEL_ALL, "type": int(typ), "n": int(len(v)),
                              "q25": float(np.quantile(v, 0.25)) if len(v) else np.nan,
                              "median": float(np.median(v)) if len(v) else np.nan,
                              "q75": float(np.quantile(v, 0.75)) if len(v) else np.nan})
    return rows, month_rows, desc_rows


def g1_order(d: dict) -> list:
    spec = next(s for s in step18.expand_kw_specs() if s["variable"] == G1_VARIABLE)
    y = pd.Series(step18.values(spec, MONTH, d), index=d["ids"])
    traj = pd.read_parquet(TRAJ_PATH).pivot(index="territory_id", columns="month", values="cluster").loc[d["ids"]]
    rows = []
    for m in d["months"]:
        med = y.groupby(traj[m]).median().sort_values(ascending=False)
        ok = bool(med.index[0] == G1_TOP_TYPE and med.index[-1] == G1_BOTTOM_TYPE)
        rows.append({"row_type": "g1_order_month", "hypothesis": "Г1", "variable": G1_VARIABLE, "partition": m,
                     "order": " > ".join(str(int(i)) for i in med.index), "order_ok": ok})
    rows.append({"row_type": "g1_order", "hypothesis": "Г1", "variable": G1_VARIABLE,
                 "months_ok": int(sum(r["order_ok"] for r in rows)),
                 "order": rows[-1]["order"], "partition": MONTH})
    return rows


def g7_delta(d: dict, confirmed: dict) -> list:
    specs = {s["variable"]: s for s in step18.expand_kw_specs()}
    region, lab = d["region"], d["parts"][DEC]
    rows = []
    for variant, hv, rv in G7_PAIRS:
        h = step18.values(specs[hv], MONTH, d)
        r = step18.values(specs[rv], MONTH, d)
        common = ~np.isnan(h) & ~np.isnan(r)
        h, r = np.where(common, h, np.nan), np.where(common, r, np.nan)
        for li, level in enumerate(LEVELS):
            ph = prepare(h, lab, region, level, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
            pr = prepare(r, lab, region, level, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
            if not np.array_equal(ph["mask"], pr["mask"]):
                raise SystemExit(f"STOP: Г7 {variant}/{level}: разные множества МО у доли и остатка")
            yh, yr, lb = ph["y"], pr["y"], ph["labels"]
            e_h, e_r = KW(yh, lb).observed(), KW(yr, lb).observed()
            rng = np.random.default_rng(SEED_G7 + 10 * G7_PAIRS.index((variant, hv, rv)) + li)
            boot = np.empty(BOOT_B)
            for b in range(BOOT_B):
                idx = rng.integers(0, len(lb), len(lb))
                boot[b] = kw_eta2(yh[idx], lb[idx])[0] - kw_eta2(yr[idx], lb[idx])[0]
            lo, hi = np.quantile(boot, [0.025, 0.975])
            conf_h = confirmed.get((hv, level))
            rows.append({"row_type": "g7_delta", "hypothesis": "Г7", "variable": f"{hv} − {rv}", "variant": variant,
                         "level": level, "n": int(len(lb)), "eta2_health": e_h, "eta2_resid": e_r, "delta": e_h - e_r,
                         "q025": float(lo), "q975": float(hi),
                         "p_boot": float((1 + np.sum(boot <= 0)) / (1 + BOOT_B)),
                         "confirmed_level": (None if conf_h is None else bool(conf_h and lo > 0)), "status": "ok"})
    for level in LEVELS:
        rows.append({"row_type": "g7_delta", "hypothesis": "Г7", "variable": G7_NO_RESID, "variant": "чувствительность",
                     "level": level, "confirmed_level": confirmed.get((G7_NO_RESID, level)), "status": G7_NO_RESID_NOTE})
    return rows


def control_job(values_by_part: dict, parts: dict, region: np.ndarray, level: str, part: str, seed: int):
    obs, p = eta2_on(values_by_part[part], parts[part], region, level)
    null = KW(p["y"], p["labels"]).null(N_PERM, seed, HYP_PERM_BATCH, p["region"] if level == LEVEL_REGION else None)
    return level, part, obs, null


def controls(d: dict) -> list:
    parts_order = [DEC] + d["months"]
    pos = {part: step18.values({"source": "shares", "column": POSITIVE_CONTROL, "period": "month"},
                               MONTH if part == DEC else part, d) for part in parts_order}
    noise = np.random.default_rng(HYP_CONTROL_SEED).standard_normal(len(d["ids"]))
    neg = {part: noise for part in parts_order}
    rows = []
    for name, vals, base in [(POSITIVE_CONTROL, pos, HYP_CONTROL_SEED * 1000),
                             ("N(0,1)", neg, HYP_CONTROL_SEED * 10000)]:
        tasks = [(level, part, base + li * len(parts_order) + pi)
                 for li, level in enumerate(LEVELS) for pi, part in enumerate(parts_order)]
        res = Parallel(n_jobs=ROBUST_N_JOBS)(delayed(control_job)(vals, d["parts"], d["region"], lv, pt, sd)
                                             for lv, pt, sd in tasks)
        res = {(lv, pt): (o, nl) for lv, pt, o, nl in res}
        for level in LEVELS:
            obs_dec, null_dec = res[(level, DEC)]
            v = verdict_kw(obs_dec, null_dec, [res[(level, m)][0] for m in d["months"]],
                           [np.quantile(res[(level, m)][1], 0.95) for m in d["months"]], ALPHA, MONTHS_RULE)
            rows.append({"row_type": "control", "variable": name, "level": level, "obs": obs_dec, "p_dec": v["p_dec"],
                         "months_ok": v["months_above_p95"], "confirmed_level": v["verdict"] == "подтверждено"})
    tab = pd.DataFrame(rows)
    bad = tab[(tab["variable"] == POSITIVE_CONTROL) != tab["confirmed_level"]]
    if len(bad):
        raise SystemExit(f"STOP: контроль не прошёл:\n{tab.to_string(index=False)}")
    return rows


def main() -> None:
    t0 = time.time()
    log = []
    check_frozen(log)
    d = step18.load()
    ros_log = []
    step18.clean_rosstat(d, ros_log)
    fd = federal_district_labels(d, log)
    thr = pd.read_parquet(THRESHOLDS_PATH)
    nulls = pd.read_parquet(NULL_DEC_PATH)

    rows, month_rows, desc_rows = kw_tests(d, thr, nulls, fd)
    confirmed = {(r["variable"], r["level"]): r["confirmed_level"] for r in rows}
    t_kw = time.time() - t0
    g1 = g1_order(d)
    t1 = time.time()
    g7 = g7_delta(d, confirmed)
    t_g7 = time.time() - t1
    t2 = time.time()
    ctrl = controls(d)
    t_ctrl = time.time() - t2

    res = pd.DataFrame(rows + g7 + g1 + ctrl + desc_rows + month_rows)
    cols = ["row_type", "hypothesis", "variable", "variant", "level", "partition", "status", "type", "n", "k_eff",
            "regions_excluded", "obs", "p_dec", "p95", "months_ok", "confirmed_level", "size_label", "eta2_fd", "n_fd",
            "k_fd", "eta2_health", "eta2_resid", "delta", "q025", "q975", "p_boot", "order", "order_ok", "q25",
            "median", "q75"]
    res = res.reindex(columns=cols)
    for c in ["type", "n", "k_eff", "regions_excluded", "months_ok", "n_fd", "k_fd"]:
        res[c] = res[c].astype("Int64")
    for c in ["confirmed_level", "order_ok"]:
        res[c] = res[c].astype("boolean")
    res.to_parquet(RESULTS_PATH, engine="pyarrow", index=False)
    sha = hashlib.sha256(RESULTS_PATH.read_bytes()).hexdigest()
    SHA_PATH.write_text(f"{sha}  {RESULTS_PATH.name}\n", encoding="utf-8")

    kw = res[res["row_type"] == "kw"]
    kw_tab = kw[["hypothesis", "variable", "variant", "level", "status", "n", "obs", "p95", "p_dec", "months_ok",
                 "confirmed_level", "size_label", "eta2_fd", "n_fd"]]
    desc = res[res["row_type"] == "describe"][["hypothesis", "variable", "type", "n", "q25", "median", "q75"]]
    g1_tab = res[res["row_type"].isin(["g1_order_month", "g1_order"])][["row_type", "partition", "order", "order_ok", "months_ok"]]
    g7_tab = res[res["row_type"] == "g7_delta"][["variable", "variant", "level", "n", "eta2_health", "eta2_resid",
                                                  "delta", "q025", "q975", "p_boot", "confirmed_level", "status"]]
    ctrl_tab = res[res["row_type"] == "control"][["variable", "level", "obs", "p_dec", "months_ok", "confirmed_level"]]
    lines = [
        "# 19a. Проверка гипотез Г1, Г4, Г7, Г8 (Краскел–Уоллис по типам)", "",
        "Сгенерировано `src/19a_hypothesis_tests_kw.py`. Пороги — замороженные шага 18a; переменные, уровни и "
        "разбиения — те же. Вердикта по гипотезам здесь нет (шаг 19c); confirmed_level — правило 18a для одной "
        "переменной на одном уровне.", "",
        "## Заморозка и данные", "", *log, *ros_log, "",
        "## Правила", "",
        f"- obs — η²_H на каноническом декабрьском разбиении; p_dec = (1 + #{{нуль ≥ obs}}) / (1 + {N_PERM}) по "
        "векторам `hypothesis_null_dec.parquet`; months_ok — число месяцев из 24 с obs_m > p95_m (p95_m из таблицы "
        f"порогов); confirmed_level = (p_dec < {ALPHA}) и (months_ok ≥ {MONTHS_RULE}); для status не «ok» — пусто.",
        "- size_label — подпись по сетке Коэна для η²_H (< 0.01 ничтожный, 0.01–< 0.06 малый, 0.06–< 0.14 средний, "
        "≥ 0.14 большой); на confirmed_level не влияет.",
        "- eta2_fd — эталон: η²_H разбиения на федеральные округа для той же переменной (декабрь, только общий "
        f"уровень), таблица `{FD_PATH.relative_to(PROJECT_DIR)}`.", "",
        "## Результаты по переменным", "", md(kw_tab), "",
        "## Описание по типам (декабрь, общий уровень)", "", md(desc), "",
        f"## Г1: порядок типов по медиане {G1_VARIABLE}", "",
        f"Типы — колонка cluster `{TRAJ_PATH.name}` в каждом месяце; месяц засчитывается, если наибольшая медиана у "
        f"типа {G1_TOP_TYPE} и наименьшая у типа {G1_BOTTOM_TYPE}.", "", md(g1_tab), "",
        "## Г7: Δ = η²_H(share_health) − η²_H(остаток после учёта возраста)", "",
        f"Множество — МО, у которых есть и доля здравоохранения, и доля старших; бутстреп по МО, {BOOT_B} повторов, "
        f"seed {SEED_G7}; значения (и остатки, и ранги внутри регионов) фиксированы, в бутстрепе пересчитывается только "
        "η²_H. p_boot = (1 + #{Δ_boot ≤ 0}) / (1 + B). confirmed_level(Г7) = confirmed_level(share_health) и "
        "нижняя граница Δ > 0.", "", md(g7_tab), "",
        "## Контроли", "", md(ctrl_tab), "",
        "## Время", "",
        f"- Краскел–Уоллис по всем переменным и месяцам: {t_kw:.1f} с; Г7-бутстреп: {t_g7:.1f} с; контроли: {t_ctrl:.1f} с; "
        f"весь шаг: {time.time() - t0:.1f} с", "",
        f"sha256 `{RESULTS_PATH.name}`: `{sha}`", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    for name, tab in [("РЕЗУЛЬТАТЫ", kw_tab), ("Г1", g1_tab), ("Г7", g7_tab), ("КОНТРОЛИ", ctrl_tab)]:
        print(f"===== {name}\n{tab.to_string(index=False)}")
    print(f"sha256 {sha}\nвесь шаг: {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
