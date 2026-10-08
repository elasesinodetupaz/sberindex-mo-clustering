"""Шаг 22. Проверка дополнительных гипотез Г10–Г13 (предложение розницы и общепита) и вердикты.

Определения и пороги заморожены шагом 21 (sha256 сверяется). Переменные, разбиения, уровни и отбор МО —
функции шагов 18 и 21; ρ и η²_H — функции шагов 19a/19b; поправка Холма и ступени — шаг 19c (импорт без
запуска расчёта). Нулевые векторы декабря шаг 21 в файл не сохранял: они восстанавливаются тем же кодом с
seed каждой комбинации из замороженной таблицы, и их сводки (n, null_mean, p95, p99, p95_two, p99_two)
сверяются с таблицей на точное совпадение; при расхождении — STOP.

Г10, Г11 — ρ Спирмена (основной вариант: декабрь + 24 месяца; чувствительность: только декабрь);
Г12, Г13 — η²_H доли и остатка, Δ с бутстрепом по МО. Холм по {Г10, Г12, Г13} на каждом уровне; Г11 —
контроль вне поправки. Разведочно (exploratory): частная корреляция Г10, Г11 с поправкой на размер МО.
Все Г10–Г13 — дополнительное семейство, сформулированное после основных результатов.

Вход:  замороженные файлы шагов 18–21, входы шагов 18 и 21
Выход: data/processed/hypothesis_results_supply.parquet (+ .sha256), notebooks/22_supply_results.md
Запуск из корня проекта:  .venv/bin/python src/22_supply_tests.py
"""
import hashlib
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import (ALPHA, BOOT_B, HYP_MIN_REGION_N, HYP_MIN_TYPES, HYP_PERM_BATCH, MIN_GROUP, MONTH, MONTHS_RULE,
                    N_PERM, ROBUST_N_JOBS, SEED_22, SUPPLY_CORR_PAIRS, SUPPLY_KW)
from hypothesis_tools import LEVEL_ALL, LEVEL_REGION, LEVELS, kw_eta2, prepare, rank_residual

step18 = importlib.import_module("18_hypothesis_thresholds")
step19a = importlib.import_module("19a_hypothesis_tests_kw")
step19b = importlib.import_module("19b_hypothesis_tests_corr_trans")
step19c = importlib.import_module("19c_hypothesis_verdicts")
step21 = importlib.import_module("21_supply_thresholds")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
THRESHOLDS_PATH = PROCESSED_DIR / "hypothesis_thresholds_supply.parquet"
RESULTS_PATH = PROCESSED_DIR / "hypothesis_results_supply.parquet"
SHA_PATH = PROCESSED_DIR / "hypothesis_results_supply.sha256"
REPORT_PATH = PROJECT_DIR / "notebooks" / "22_supply_results.md"

FROZEN = {**step21.FROZEN, THRESHOLDS_PATH: "c7d0a880d7cbac769ed7c7c8505dffea2d49eb6d3ce525b824238bc5368ddbf6"}
DEC = step18.DEC
CONFIRMED, NOT_CONFIRMED = step19c.CONFIRMED, step19c.NOT_CONFIRMED
FAMILY = ["Г10", "Г12", "Г13"]            # Холм на каждом уровне
CONTROL_HYP = "Г11"                       # контроль вне поправки
SUPPLEMENTARY_NOTE = "дополнительное семейство, сформулировано после основных результатов"
UNRELIABLE_NOTE = "показатель предложения ненадёжен"
SIZE_BINS = [0, 10_000, 30_000, 100_000, 300_000, np.inf]
SIZE_LABELS = ["до 10 тыс.", "10–30 тыс.", "30–100 тыс.", "100–300 тыс.", "более 300 тыс."]
CHECK_COLS = ["n", "k_eff", "min_group", "regions_excluded", "null_mean", "p95", "p99", "p95_two", "p99_two", "seed"]


def md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(index=False, **kw)


def check_frozen(when: str, log: list) -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{when}: {got}  {path.relative_to(PROJECT_DIR)} {'OK' if got == expected else 'MISMATCH'}")
        log.append(f"- {when}: `{path.relative_to(PROJECT_DIR)}` — {'совпадает' if got == expected else 'НЕ СОВПАДАЕТ'}")
        if got != expected:
            raise SystemExit(f"STOP: sha256 {path.name} = {got}, ожидается {expected}")


def same(a, b) -> bool:
    return (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and a == b)


def null_job(job, region):
    kind, meta, y, labels = job
    fn = step18.kw_job if kind == "kw" else step18.corr_job
    return fn(meta, y, labels, region, True) if kind == "kw" else fn(meta, y, region, True)


def restore_nulls(jobs: list, thr: pd.DataFrame, region: np.ndarray, log: list) -> dict:
    """Нулевые векторы декабря: тот же код и seed шага 21; сводки сверяются с замороженной таблицей."""
    keys = ["hypothesis", "variable", "variant", "level", "partition", "seed"]
    job_keys = [tuple(str(j[1][k]) for k in keys) for j in jobs]
    thr_keys = [tuple(str(r[k]) for k in keys) for _, r in thr.iterrows()]
    if job_keys != thr_keys:
        raise SystemExit("STOP: комбинации шага 21 (порядок, seed) не совпадают с замороженной таблицей порогов")
    need = [i for i, j in enumerate(jobs) if j[1]["partition"] == (DEC if j[0] == "kw" else MONTH)]
    res = Parallel(n_jobs=ROBUST_N_JOBS)(delayed(null_job)(jobs[i], region) for i in need)
    nulls, bad = {}, []
    for i, (row, null) in zip(need, res):
        t = thr.iloc[i]
        for c in CHECK_COLS:
            if not same(row.get(c, np.nan), t[c]):
                bad.append((t["variable"], t["level"], t["partition"], c, row.get(c), t[c]))
        nulls[(t["variable"], t["variant"], t["level"])] = null
    log.append(f"- восстановлено нулевых векторов декабря: {len(need)}; расхождений со сводками замороженной таблицы "
               f"({', '.join(CHECK_COLS)}): {len(bad)}")
    print(log[-1])
    if bad:
        raise SystemExit(f"STOP: восстановленный нуль не совпадает с порогами шага 21: {bad[:5]}")
    return nulls


def corr_tests(jobs: list, thr: pd.DataFrame, nulls: dict, region: np.ndarray) -> tuple:
    tidx = thr.set_index(["variable", "variant", "level", "partition"])
    by_key = {(j[1]["variable"], j[1]["variant"], j[1]["level"], j[1]["partition"]): j[2] for j in jobs if j[0] == "corr"}
    rows, month_rows = [], []
    for s in SUPPLY_CORR_PAIRS:
        var = f"{s['share']}~{s['supply']}"
        for level in LEVELS:
            sp, p = step19b.rho_on(by_key[(var, s["variant"], level, MONTH)], region, level)
            t = tidx.loc[(var, s["variant"], level, MONTH)]
            if p["n"] != int(t["n"]):
                raise SystemExit(f"STOP: {var} / {level}: n = {p['n']}, в порогах {int(t['n'])}")
            obs = sp.observed()
            p_dec = step19b.p_one_sided(obs, nulls[(var, s["variant"], level)], s["sign"])
            same_sign = obs > 0 if s["sign"] == "+" else obs < 0
            row = {"row_type": "corr", "hypothesis": s["hypothesis"], "variable": var, "variant": s["variant"],
                   "level": level, "partition": MONTH, "expected_sign": s["sign"], "n": p["n"], "obs": obs,
                   "p95": float(t["p95"]), "p95_two": float(t["p95_two"]), "p_dec": p_dec,
                   "opposite_sign": bool((not same_sign) and obs != 0 and abs(obs) > float(t["p95_two"]))}
            if s["variant"] == "основной":
                ok = 0
                for (v, vr, lv, m), xy in by_key.items():
                    if (v, vr, lv) != (var, s["variant"], level):
                        continue
                    spm, pm = step19b.rho_on(xy, region, level)
                    tm = tidx.loc[(v, vr, lv, m)]
                    if pm["n"] != int(tm["n"]):
                        raise SystemExit(f"STOP: {var} / {level} / {m}: n не как в порогах")
                    o = spm.observed()
                    ok += step19b.beyond(o, float(tm["p95"]), s["sign"])
                    month_rows.append({"row_type": "corr_month", "hypothesis": s["hypothesis"], "variable": var,
                                       "variant": s["variant"], "level": level, "partition": m, "obs": o,
                                       "p95": float(tm["p95"])})
                row.update(months_ok=int(ok), confirmed_level=bool(p_dec < ALPHA and ok >= MONTHS_RULE))
            else:
                row.update(confirmed_level=bool(same_sign and p_dec < ALPHA))
            rows.append(row)
    return rows, month_rows


def kw_tests(jobs: list, thr: pd.DataFrame, nulls: dict, region: np.ndarray) -> tuple:
    tidx = thr.set_index(["variable", "variant", "level", "partition"])
    rows, month_rows = [], []
    kw_jobs = [j for j in jobs if j[0] == "kw"]
    for key in dict.fromkeys((j[1]["hypothesis"], j[1]["variable"], j[1]["variant"], j[1]["level"]) for j in kw_jobs):
        hyp, var, variant, level = key
        obs_m, p95_m, obs_dec, n_dec = [], [], None, None
        for _, meta, y, labels in [j for j in kw_jobs if (j[1]["hypothesis"], j[1]["variable"], j[1]["variant"],
                                                          j[1]["level"]) == key]:
            o, p = step19a.eta2_on(y, labels, region, level)
            t = tidx.loc[(var, variant, level, meta["partition"])]
            if (p["n"], p["k_eff"], len(p["excluded_regions"])) != (int(t["n"]), int(t["k_eff"]), int(t["regions_excluded"])):
                raise SystemExit(f"STOP: {var} / {level} / {meta['partition']}: n, k_eff или regions_excluded не как в порогах")
            if meta["partition"] == DEC:
                obs_dec, n_dec, status = o, p["n"], t["status"]
            else:
                obs_m.append(o)
                p95_m.append(float(t["p95"]))
                month_rows.append({"row_type": "kw_month", "hypothesis": hyp, "variable": var, "variant": variant,
                                   "level": level, "partition": meta["partition"], "obs": o, "p95": float(t["p95"])})
        p_dec = step19b.p_one_sided(obs_dec, nulls[(var, variant, level)], "+")
        months_ok = int(np.sum(np.array(obs_m) > np.array(p95_m)))
        rows.append({"row_type": "kw", "hypothesis": hyp, "variable": var, "variant": variant, "level": level,
                     "partition": DEC, "n": n_dec, "obs": obs_dec, "p95": float(tidx.loc[(var, variant, level, DEC)]["p95"]),
                     "p_dec": p_dec, "months_ok": months_ok, "status": status,
                     "confirmed_level": bool(p_dec < ALPHA and months_ok >= MONTHS_RULE) if status == "ok" else None})
    return rows, month_rows


def delta_tests(jobs: list, d: dict, kw_rows: list) -> list:
    region, rows = d["region"], []
    conf = {(r["variable"], r["level"]): r["confirmed_level"] for r in kw_rows}
    dec = {(j[1]["hypothesis"], j[1]["variable"], j[1]["level"]): j for j in jobs if j[0] == "kw" and j[1]["partition"] == DEC}
    for hi, s in enumerate(SUPPLY_KW):
        share_var, resid_var = s["share"], f"{s['share']}_resid_{s['supply']}"
        for li, level in enumerate(LEVELS):
            js, jr = dec[(s["hypothesis"], share_var, level)], dec[(s["hypothesis"], resid_var, level)]
            ps = prepare(js[2], js[3], region, level, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
            pr = prepare(jr[2], jr[3], region, level, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
            if not np.array_equal(ps["mask"], pr["mask"]):
                raise SystemExit(f"STOP: {s['hypothesis']} {level}: разные множества МО у доли и остатка")
            ys, yr, lb = ps["y"], pr["y"], ps["labels"]
            e_s, e_r = kw_eta2(ys, lb)[0], kw_eta2(yr, lb)[0]
            rng = np.random.default_rng(SEED_22 + 10 * hi + li)
            boot = np.empty(BOOT_B)
            for b in range(BOOT_B):
                idx = rng.integers(0, len(lb), len(lb))
                boot[b] = kw_eta2(ys[idx], lb[idx])[0] - kw_eta2(yr[idx], lb[idx])[0]
            lo, hi_q = np.quantile(boot, [0.025, 0.975])
            c = conf[(share_var, level)]
            rows.append({"row_type": "delta", "hypothesis": s["hypothesis"], "variable": f"{share_var} − {resid_var}",
                         "variant": "основной", "level": level, "n": int(len(lb)), "eta2_share": e_s, "eta2_resid": e_r,
                         "delta": e_s - e_r, "q025": float(lo), "q975": float(hi_q),
                         "p_boot": float((1 + np.sum(boot <= 0)) / (1 + BOOT_B)),
                         "confirmed_level": None if c is None else bool(c and lo > 0)})
    return rows


def verdicts(corr_rows: list, delta_rows: list) -> tuple:
    rows = []
    for r in corr_rows:
        if r["variant"] == "основной":
            rows.append({"hypothesis": r["hypothesis"], "variable": r["variable"], "level": r["level"], "p_source": "p_dec",
                         "p_raw": r["p_dec"], "confirmed_level": r["confirmed_level"], "stat": "ρ", "obs": r["obs"],
                         "opposite_sign": r["opposite_sign"]})
    for r in delta_rows:
        rows.append({"hypothesis": r["hypothesis"], "variable": r["variable"], "level": r["level"], "p_source": "p_boot",
                     "p_raw": r["p_boot"], "confirmed_level": r["confirmed_level"], "stat": "Δ η²_H", "obs": r["delta"]})
    v = pd.DataFrame(rows)
    v["p_holm"] = np.nan
    for level in LEVELS:
        m = v["hypothesis"].isin(FAMILY) & (v["level"] == level)
        v.loc[m, "p_holm"] = step19c.holm(v.loc[m, "p_raw"])
    p_final = v["p_holm"].where(v["hypothesis"].isin(FAMILY), v["p_raw"])
    v["result"] = np.where(v["confirmed_level"].astype(bool) & (p_final < ALPHA), CONFIRMED, NOT_CONFIRMED)
    g11_region = v[(v["hypothesis"] == CONTROL_HYP) & (v["level"] == LEVEL_REGION)]["result"].iloc[0]

    def note(r):
        parts = [SUPPLEMENTARY_NOTE]
        if r["hypothesis"] == CONTROL_HYP:
            parts.append("контроль, вне поправки Холма")
        if r.get("opposite_sign") is True or (not pd.isna(r.get("opposite_sign")) and bool(r.get("opposite_sign"))):
            parts.append(step19c.OPPOSITE_NOTE)
        if r["hypothesis"] in FAMILY and g11_region != CONFIRMED:
            parts.append(UNRELIABLE_NOTE)
        return "; ".join(parts)

    v["note"] = v.apply(note, axis=1)
    stages = []
    for hyp in ["Г10", "Г11", "Г12", "Г13"]:
        s = v[v["hypothesis"] == hyp].set_index("level")
        key = (s.loc[LEVEL_ALL, "result"] == CONFIRMED, s.loc[LEVEL_REGION, "result"] == CONFIRMED)
        stages.append({"hypothesis": hyp, "level": "итог по уровням", "result": step19c.STAGES[key],
                       "note": s.loc[LEVEL_ALL, "note"] if s.loc[LEVEL_ALL, "note"] == s.loc[LEVEL_REGION, "note"]
                       else f"общий: {s.loc[LEVEL_ALL, 'note']} | внутри регионов: {s.loc[LEVEL_REGION, 'note']}"})
    return v, pd.DataFrame(stages)


def partial_corr(d: dict, sup: dict) -> list:
    """Разведочно: ρ Спирмена между остатками рангов доли и переменной предложения после МНК на рангах
    log(население 2023); декабрь 2024, оба уровня; нуль — перестановки одного вектора остатков."""
    log_pop = np.log(d["ros_by_year"][2023]["population"].to_numpy(float))
    rows = []
    for k, s in enumerate([p for p in SUPPLY_CORR_PAIRS if p["variant"] == "основной"]):
        x = step21.share(d, s["share"], MONTH)
        y = sup[s["supply"]]
        m = ~np.isnan(x) & ~np.isnan(y) & np.isfinite(log_pop)
        rx = rank_residual(np.where(m, x, np.nan), np.where(m, log_pop, np.nan))
        ry = rank_residual(np.where(m, y, np.nan), np.where(m, log_pop, np.nan))
        for li, level in enumerate(LEVELS):
            sp, p = step19b.rho_on(np.column_stack([rx, ry]), d["region"], level)
            obs = sp.observed()
            null = sp.null(N_PERM, SEED_22 + 100 + 10 * k + li, HYP_PERM_BATCH, p["region"] if level == LEVEL_REGION else None)
            rows.append({"row_type": "partial", "hypothesis": s["hypothesis"], "variable": f"{s['share']}~{s['supply']} | log(население 2023)",
                         "level": level, "partition": MONTH, "expected_sign": s["sign"], "n": p["n"], "obs": obs,
                         "p_dec": step19b.p_one_sided(obs, null, s["sign"]), "exploratory": True})
    return rows


def describe(d: dict, sup: dict) -> tuple:
    pop = d["ros_by_year"][2023]["population"].to_numpy(float)
    frame = pd.DataFrame({"floor_pc": sup["floor_pc"], "seats_pc": sup["seats_pc"], "тип": d["parts"][DEC],
                          "группа размера": pd.cut(pop, SIZE_BINS, labels=SIZE_LABELS, right=False)})
    out = []
    for by in ["тип", "группа размера"]:
        g = frame.groupby(by, observed=True)
        t = pd.DataFrame({"floor_pc: МО с данными": g["floor_pc"].count(), "floor_pc: медиана": g["floor_pc"].median().round(2),
                          "seats_pc: МО с данными": g["seats_pc"].count(), "seats_pc: медиана": g["seats_pc"].median().round(2)})
        out.append(t.reset_index())
    no_pop = int(np.isnan(pop).sum())
    return out[0], out[1], no_pop


def main() -> None:
    t0 = time.time()
    frozen_log, check_log = [], []
    check_frozen("до", frozen_log)
    d = step18.load()
    step18.clean_rosstat(d, [])
    sup, _ = step21.supply_values(d["ids"])
    thr = pd.read_parquet(THRESHOLDS_PATH)
    jobs = step21.build_jobs(d, sup)
    nulls = restore_nulls(jobs, thr, d["region"], check_log)
    t_null = time.time() - t0

    corr_rows, corr_months = corr_tests(jobs, thr, nulls, d["region"])
    kw_rows, kw_months = kw_tests(jobs, thr, nulls, d["region"])
    t1 = time.time()
    delta_rows = delta_tests(jobs, d, kw_rows)
    t_delta = time.time() - t1
    v, stages = verdicts(corr_rows, delta_rows)
    partial = partial_corr(d, sup)
    by_type, by_size, no_pop = describe(d, sup)
    ctrl = step19b.controls(d)
    check_frozen("после", frozen_log)

    res = pd.concat([pd.DataFrame(corr_rows + kw_rows + delta_rows + partial + ctrl + corr_months + kw_months),
                     v.assign(row_type="verdict"), stages.assign(row_type="stage"),
                     by_type.assign(row_type="describe_type").rename(columns={"тип": "type"}),
                     by_size.assign(row_type="describe_size").rename(columns={"группа размера": "size_group"})
                     .astype({"size_group": "string"})],
                    ignore_index=True)
    for c in ["n", "months_ok", "type"]:
        res[c] = res[c].astype("Int64")
    for c in ["confirmed_level", "opposite_sign", "exploratory"]:
        res[c] = res[c].astype("boolean")
    res = res[["row_type"] + [c for c in res.columns if c != "row_type"]]
    res.to_parquet(RESULTS_PATH, engine="pyarrow", index=False)
    sha = hashlib.sha256(RESULTS_PATH.read_bytes()).hexdigest()
    SHA_PATH.write_text(f"{sha}  {RESULTS_PATH.name}\n", encoding="utf-8")

    corr_tab = pd.DataFrame(corr_rows)[["hypothesis", "variable", "variant", "level", "expected_sign", "n", "obs", "p95",
                                        "p95_two", "p_dec", "months_ok", "opposite_sign", "confirmed_level"]]
    kw_tab = pd.DataFrame(kw_rows)[["hypothesis", "variable", "level", "n", "obs", "p95", "p_dec", "months_ok",
                                    "status", "confirmed_level"]]
    delta_tab = pd.DataFrame(delta_rows)[["hypothesis", "variable", "level", "n", "eta2_share", "eta2_resid", "delta",
                                          "q025", "q975", "p_boot", "confirmed_level"]]
    v_tab = pd.concat([v[["hypothesis", "level", "p_source", "p_raw", "p_holm", "confirmed_level", "result", "note"]],
                       stages[["hypothesis", "level", "result", "note"]]], ignore_index=True)
    v_tab["_o"] = v_tab["hypothesis"].map({"Г10": 0, "Г11": 1, "Г12": 2, "Г13": 3})
    v_tab = v_tab.sort_values(["_o"], kind="stable").drop(columns="_o")
    part_tab = pd.DataFrame(partial)[["hypothesis", "variable", "level", "expected_sign", "n", "obs", "p_dec", "exploratory"]]
    ctrl_tab = pd.DataFrame(ctrl)[["variable", "level", "n", "obs", "p_dec", "confirmed_level"]]
    lines = [
        "# 22. Проверка дополнительных гипотез Г10–Г13 (предложение розницы и общепита)", "",
        "Сгенерировано `src/22_supply_tests.py`. Г10–Г13 — дополнительное семейство, сформулированное после основных "
        "результатов 19a–19c. Пороги — замороженные шага 21.", "",
        "## Заморозка", "", *frozen_log, *check_log,
        "  (шаг 21 не сохранял нулевые векторы; они восстановлены тем же кодом с seed из замороженной таблицы и "
        "совпали с её сводками)", "",
        "## Правила", "",
        f"- Г10, Г11: ρ Спирмена; p_dec — односторонний по ожидаемому знаку; months_ok — месяцы, где ρ за ожидаемым "
        f"знаком дальше p95_m; confirmed_level = p_dec < {ALPHA} и months_ok ≥ {MONTHS_RULE}. Варианты чувствительности — "
        f"только декабрь: тот же знак и p_dec < {ALPHA}. opposite_sign — ρ противоположного знака и |ρ| > p95_two.",
        f"- Г12, Г13: η²_H доли и её остатка (декабрь и 24 месяца); Δ = η²_H(доля) − η²_H(остаток); бутстреп по МО, "
        f"{BOOT_B} повторов, seed {SEED_22}; на уровне подтверждено, если confirmed_level(доля) и нижняя граница Δ > 0.",
        f"- Холм (ALPHA = {ALPHA}) по семейству {{Г10 (p_dec), Г12 (p_boot), Г13 (p_boot)}} на каждом уровне; Г11 — "
        f"контроль вне поправки. Если Г11 не подтверждена внутри регионов, у Г10, Г12, Г13 — «{UNRELIABLE_NOTE}».", "",
        "## Вердикты", "", md(v_tab), "",
        "## Г10, Г11: ρ Спирмена", "", md(corr_tab), "",
        "## Г12, Г13: η²_H доли и остатка", "", md(kw_tab), "", "Δ = η²_H(доля) − η²_H(остаток), бутстреп по МО:", "",
        md(delta_tab), "",
        "## Разведочно: частная корреляция с поправкой на размер МО (exploratory, вне вердиктов и Холма)", "",
        f"Остатки рангов доли и рангов переменной предложения после МНК на рангах log(население 2023); ρ Спирмена между "
        f"остатками, декабрь 2024; нуль — {N_PERM} перестановок одного вектора остатков.", "", md(part_tab), "",
        "## Описание (не для вердикта)", "", "По типам k = 6 (декабрь 2024):", "", md(by_type), "",
        f"По группам размера МО (население на 1 января 2023; МО без населения: {no_pop}):", "", md(by_size), "",
        "## Контроли", "", md(ctrl_tab), "",
        "## Время", "", f"- восстановление нулей: {t_null:.1f} с; бутстреп Δ: {t_delta:.1f} с; весь шаг: {time.time() - t0:.1f} с", "",
        f"sha256 `{RESULTS_PATH.name}`: `{sha}`", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    for name, tab in [("ВЕРДИКТЫ", v_tab), ("Г10, Г11", corr_tab), ("Г12, Г13", kw_tab), ("Δ", delta_tab),
                      ("ЧАСТНАЯ КОРРЕЛЯЦИЯ (exploratory)", part_tab), ("ПО ТИПАМ", by_type), ("ПО РАЗМЕРУ", by_size),
                      ("КОНТРОЛИ", ctrl_tab)]:
        print(f"===== {name}\n{tab.to_string(index=False)}")
    print(f"МО без населения 2023: {no_pop}\nsha256 {sha}\nвесь шаг: {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
