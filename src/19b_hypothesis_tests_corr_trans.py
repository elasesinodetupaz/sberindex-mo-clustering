"""Шаг 19b. Проверка гипотез Г2, Г3 (корреляции Спирмена), Г6 (переходы в ближайшие типы), Г9 (переходы и
общие сдвиги долей). Пороги заморожены шагами 18a и 18b: перед расчётом сверяются sha256.

Определения переменных, переходов, ближайших типов и рядов — те же, что в шагах 18 и 18b (функции
импортируются из их модулей без запуска расчёта). Вердикт по гипотезам здесь не выносится (шаг 19c).

Вход:  файлы шагов 18a и 18b (hypothesis_thresholds, hypothesis_null_dec, hypothesis_thresholds_g6g9,
       hypothesis_series_g9), входы шагов 18 и 18b
Выход: data/processed/hypothesis_results_19b.parquet, data/processed/hypothesis_results_19b.sha256,
       notebooks/19b_hypothesis_results.md
Запуск из корня проекта:  .venv/bin/python src/19b_hypothesis_tests_corr_trans.py
"""
import hashlib
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (ALPHA, BOOT_B, HYP_CONTROL_SEED, HYP_CORR_PAIRS, HYP_MIN_REGION_N, HYP_MIN_TYPES, HYP_PERM_BATCH,
                    MIN_GROUP, MONTH, MONTHS_RULE, N_NEAREST, N_NEAREST_SENS, N_PERM, TEST_SEED_G6, TEST_SEED_G9)
from hypothesis_tools import (LEVEL_REGION, LEVELS, Spearman, binomial_sum_sf, nearest_types, prepare,
                              share_to_nearest, spearman_cyclic_null)

step18 = importlib.import_module("18_hypothesis_thresholds")
step18b = importlib.import_module("18b_hypothesis_thresholds_g6_g9")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
THRESHOLDS_PATH = PROCESSED_DIR / "hypothesis_thresholds.parquet"
NULL_DEC_PATH = PROCESSED_DIR / "hypothesis_null_dec.parquet"
THRESHOLDS_G6G9_PATH = PROCESSED_DIR / "hypothesis_thresholds_g6g9.parquet"
SERIES_G9_PATH = PROCESSED_DIR / "hypothesis_series_g9.parquet"
RESULTS_PATH = PROCESSED_DIR / "hypothesis_results_19b.parquet"
SHA_PATH = PROCESSED_DIR / "hypothesis_results_19b.sha256"
REPORT_PATH = PROJECT_DIR / "notebooks" / "19b_hypothesis_results.md"

EXPECTED_SHA = {THRESHOLDS_PATH: "11081afbfc7e88dbd301004e4c739e3f25e00bf8a47704cf7c74e15e8e6665f7",
                NULL_DEC_PATH: "40735907ad605157e95ab29e05b7288ec0af61ea085360f9838e515a830a1178",
                THRESHOLDS_G6G9_PATH: "1926061b7319ba9210d9520e087dbcf7c5107a7bcaea9cb2e7cacdba687022a1",
                SERIES_G9_PATH: "dbf1fce8c74b5ea373cd61f4ec909f41d22fa6c6cfa2be413e16fafe2e8bce6e"}
STATIC = step18.STATIC
EXPLORATORY = {"Г3"}               # Г3 помечается exploratory=True
G9_GOVERNING_ROW = "управляющий порог (больший p95)"
TOP_N = 3                          # Г9: описательно — месяцы с наибольшими M_t и P_t


def md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(index=False, **kw)


def check_frozen(log: list) -> None:
    for path, expected in EXPECTED_SHA.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        log.append(f"- sha256 `{path.name}`: `{got}` — {'совпадает' if got == expected else 'НЕ СОВПАДАЕТ'}")
        print(f"sha256 {path.name} {got} {'OK' if got == expected else 'MISMATCH'}")
        if got != expected:
            raise SystemExit(f"STOP: sha256 {path.name} = {got}, ожидается {expected}")


def rho_on(xy: np.ndarray, region: np.ndarray, level: str) -> tuple:
    """ρ так же, как в шаге 18 (corr_job): отбор МО, на уровне «внутри регионов» — нормированные ранги."""
    p = prepare(xy, None, region, level, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
    return Spearman(p["y"][:, 0], p["y"][:, 1], ranked=level == LEVEL_REGION), p


def p_one_sided(obs: float, null: np.ndarray, sign: str) -> float:
    hits = np.sum(null >= obs) if sign == "+" else np.sum(null <= obs)
    return float((1 + hits) / (1 + len(null)))


def beyond(obs: float, p95: float, sign: str) -> bool:
    """obs за ожидаемым знаком и дальше порога p95 (для «−» порог отрицательный)."""
    return obs > p95 if sign == "+" else obs < p95


def corr_tests(d: dict, fed: np.ndarray, thr: pd.DataFrame, nulls: pd.DataFrame) -> tuple:
    thr_idx = thr[thr["test"] == "spearman_rho"].set_index(["variable", "level", "partition"])
    null_idx = {k: g.sort_values("perm")["value"].to_numpy()
                for k, g in nulls.groupby(["variable", "level", "partition"])}
    rows, month_rows = [], []
    for s in HYP_CORR_PAIRS:
        is_monthly = step18.monthly(s["x"]) or step18.monthly(s["y"])
        dec_part = MONTH if is_monthly else STATIC
        sign = s["sign"]

        def xy_for(month):
            xy = np.column_stack([step18.values(s["x"], month, d), step18.values(s["y"], month, d)])
            if s["exclude_federal"]:
                xy[fed] = np.nan
            return xy

        for level in LEVELS:
            sp, p = rho_on(xy_for(MONTH), d["region"], level)
            t = thr_idx.loc[(s["variable"], level, dec_part)]
            if (p["n"], len(p["excluded_regions"])) != (int(t["n"]), int(t["regions_excluded"])):
                raise SystemExit(f"STOP: {s['variable']} / {level}: n = {p['n']}, в порогах {int(t['n'])}")
            obs = sp.observed()
            p_dec = p_one_sided(obs, null_idx[(s["variable"], level, dec_part)], sign)
            same_sign = obs > 0 if sign == "+" else obs < 0
            opposite = (not same_sign) and obs != 0 and abs(obs) > float(t["p95_two"])
            row = {"row_type": "corr", "hypothesis": s["hypothesis"], "variable": s["variable"], "variant": s["variant"],
                   "level": level, "partition": dec_part, "expected_sign": sign, "n": p["n"], "obs": obs,
                   "p95": float(t["p95"]), "p95_two": float(t["p95_two"]), "p_dec": p_dec,
                   "opposite_sign": bool(opposite), "exploratory": s["hypothesis"] in EXPLORATORY}
            if is_monthly:
                ok_m = 0
                for m in d["months"]:
                    sp_m, pm = rho_on(xy_for(m), d["region"], level)
                    tm = thr_idx.loc[(s["variable"], level, m)]
                    if pm["n"] != int(tm["n"]):
                        raise SystemExit(f"STOP: {s['variable']} / {level} / {m}: n = {pm['n']}, в порогах {int(tm['n'])}")
                    o = sp_m.observed()
                    ok_m += beyond(o, float(tm["p95"]), sign)
                    month_rows.append({"row_type": "corr_month", "hypothesis": s["hypothesis"], "variable": s["variable"],
                                       "variant": s["variant"], "level": level, "partition": m, "expected_sign": sign,
                                       "obs": o, "p95": float(tm["p95"])})
                row.update(months_ok=int(ok_m), confirmed_level=bool(p_dec < ALPHA and ok_m >= MONTHS_RULE),
                           rule="p_dec < ALPHA и months_ok ≥ MONTHS_RULE")
            else:
                row.update(confirmed_level=bool(same_sign and p_dec < ALPHA), rule="тот же знак и p_dec < ALPHA (без правила месяцев)")
            rows.append(row)
    return rows, month_rows


def g6_tests(check_log: list) -> list:
    log = []
    wide = step18b.load_trajectories(log)
    tr = step18b.transitions(wide)
    final = pd.read_parquet(step18b.LABELS_FINAL_PATH).set_index("territory_id")["cluster"].sort_index()
    dist = step18b.centroid_distances(final)
    k = dist.shape[0]
    thr = pd.read_parquet(THRESHOLDS_G6G9_PATH)
    thr = thr[thr["hypothesis"] == "Г6"]
    frm, to = tr["from_type"].to_numpy(), tr["to_type"].to_numpy()
    mo = tr["territory_id"].to_numpy()
    mo_ids, mo_inv = np.unique(mo, return_inverse=True)
    rng = np.random.default_rng(TEST_SEED_G6)
    weights = rng.multinomial(len(mo_ids), np.full(len(mo_ids), 1 / len(mo_ids)), size=BOOT_B)
    trans_per_mo = np.bincount(mo_inv, minlength=len(mo_ids))
    rows = []
    for n_near in [N_NEAREST] + list(N_NEAREST_SENS):
        near = nearest_types(dist, n_near)
        hit = np.array([t in near[f] for f, t in zip(frm, to)])
        s_obs = share_to_nearest(frm, to, near)
        t_all = thr[(thr["row"] == "итог") & (thr["n_nearest"] == n_near)].iloc[0]
        t_type = thr[(thr["row"] == "по исходному типу") & (thr["n_nearest"] == n_near)].set_index("from_type")
        n_i = np.bincount(frm, minlength=k)
        if len(tr) != int(t_all["n_transitions"]) or tr["territory_id"].nunique() != int(t_all["n_distinct_mo"]) \
                or not all(n_i[i] == int(t_type.loc[i, "n_transitions"]) for i in range(k)):
            raise SystemExit(f"STOP: Г6 N={n_near}: число переходов или МО не совпадает с порогами 18b")
        q = np.array([float(t_type.loc[i, "null_mean"]) for i in range(k)])
        p_exact = binomial_sum_sf(n_i, q, int(hit.sum()))
        hits_per_mo = np.bincount(mo_inv, weights=hit.astype(float), minlength=len(mo_ids))
        boot = (weights @ hits_per_mo) / (weights @ trans_per_mo)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        variant = "основной" if n_near == N_NEAREST else "чувствительность"
        rows.append({"row_type": "g6", "hypothesis": "Г6", "variant": variant, "n_nearest": n_near, "obs": s_obs,
                     "n": len(tr), "n_distinct_mo": int(tr["territory_id"].nunique()),
                     "null_mean": float(t_all["null_mean"]), "p95": float(t_all["p95"]), "p_exact": p_exact,
                     "q025": float(lo), "q975": float(hi),
                     "confirmed_level": bool(s_obs > float(t_all["p95"]) and lo > float(t_all["null_mean"]))})
        for i in range(k):
            m = frm == i
            rows.append({"row_type": "g6_type", "hypothesis": "Г6", "variant": variant, "n_nearest": n_near,
                         "type": i, "n": int(m.sum()), "n_distinct_mo": int(tr.loc[m, "territory_id"].nunique()),
                         "obs": float(hit[m].mean()) if m.any() else np.nan, "null_mean": float(q[i]),
                         "nearest": " ".join(str(j) for j in sorted(near[i]))})
        first = {i: min(near[i], key=lambda j: dist[i, j]) for i in range(k)}
        s_syn = share_to_nearest(frm, np.array([first[f] for f in frm]), near)
        check_log.append(f"- Г6 N = {n_near}: синтетика «все конечные типы — ближайшие» даёт S = {s_syn}")
        if s_syn != 1.0:
            raise SystemExit("STOP: синтетическая проверка Г6 не прошла")
    return rows


def g9_tests() -> list:
    series = pd.read_parquet(SERIES_G9_PATH)
    thr = pd.read_parquet(THRESHOLDS_G6G9_PATH)
    thr = thr[thr["hypothesis"] == "Г9"].set_index("row")
    p_ser, m_ser = series["P"].to_numpy(float), series["M"].to_numpy(float)
    sp = Spearman(p_ser, m_ser)
    obs = sp.observed()
    perm = sp.null(N_PERM, TEST_SEED_G9, HYP_PERM_BATCH)
    cyc = spearman_cyclic_null(p_ser, m_ser)
    governing = float(thr.loc[G9_GOVERNING_ROW, "p95"])
    rows = [{"row_type": "g9", "hypothesis": "Г9", "variant": "основной", "n": len(series), "obs": obs,
             "p_perm": float((1 + np.sum(perm >= obs)) / (1 + N_PERM)),
             "p_cyc": float((1 + np.sum(cyc >= obs)) / (1 + len(cyc))), "p95": governing,
             "confirmed_level": bool(obs > governing)}]
    top_m = series.nlargest(TOP_N, "M")["month"].tolist()
    top_p = series.nlargest(TOP_N, "P")["month"].tolist()
    rows.append({"row_type": "g9_top", "hypothesis": "Г9", "top_M": ", ".join(sorted(top_m)),
                 "top_P": ", ".join(sorted(top_p)), "top_both": ", ".join(sorted(set(top_m) & set(top_p))) or "—"})
    return rows


def controls(d: dict) -> list:
    rows = []
    col = {"source": "shares", "column": "share_Общепит", "period": "month"}
    x = step18.values(col, MONTH, d)
    rng = np.random.default_rng(HYP_CONTROL_SEED)
    noise = rng.standard_normal((len(d["ids"]), 2))
    for name, xy in [("share_Общепит ~ share_Общепит", np.column_stack([x, x])), ("N(0,1) ~ N(0,1)", noise)]:
        for li, level in enumerate(LEVELS):
            sp, p = rho_on(xy, d["region"], level)
            obs = sp.observed()
            null = sp.null(N_PERM, HYP_CONTROL_SEED * 100 + li, HYP_PERM_BATCH, p["region"] if level == LEVEL_REGION else None)
            p_dec = p_one_sided(obs, null, "+")
            rows.append({"row_type": "control", "variable": name, "level": level, "n": p["n"], "obs": obs, "p_dec": p_dec,
                         "confirmed_level": bool(obs > 0 and p_dec < ALPHA)})
    tab = pd.DataFrame(rows)
    pos = tab[tab["variable"].str.startswith("share_")]
    neg = tab[tab["variable"].str.startswith("N(")]
    if not (np.allclose(pos["obs"], 1.0, atol=1e-12) and pos["confirmed_level"].all() and not neg["confirmed_level"].any()):
        raise SystemExit(f"STOP: контроль корреляций не прошёл:\n{tab.to_string(index=False)}")
    return rows


def main() -> None:
    t0 = time.time()
    log, check_log = [], []
    check_frozen(log)
    d = step18.load()
    step18.clean_rosstat(d, [])
    fed = step18.check_federal(d, [])
    thr = pd.read_parquet(THRESHOLDS_PATH)
    nulls = pd.read_parquet(NULL_DEC_PATH)
    corr, corr_months = corr_tests(d, fed, thr, nulls)
    t_corr = time.time() - t0
    t1 = time.time()
    g6 = g6_tests(check_log)
    t_g6 = time.time() - t1
    g9 = g9_tests()
    ctrl = controls(d)

    res = pd.DataFrame(corr + g6 + g9 + ctrl + corr_months)
    cols = ["row_type", "hypothesis", "variable", "variant", "level", "partition", "expected_sign", "n_nearest", "type",
            "nearest", "n", "n_distinct_mo", "obs", "p95", "p95_two", "null_mean", "p_dec", "p_exact", "p_perm", "p_cyc",
            "q025", "q975", "months_ok", "opposite_sign", "confirmed_level", "exploratory", "rule", "top_M", "top_P",
            "top_both"]
    res = res.reindex(columns=cols)
    for c in ["n_nearest", "type", "n", "n_distinct_mo", "months_ok"]:
        res[c] = res[c].astype("Int64")
    for c in ["opposite_sign", "confirmed_level", "exploratory"]:
        res[c] = res[c].astype("boolean")
    res.to_parquet(RESULTS_PATH, engine="pyarrow", index=False)
    sha = hashlib.sha256(RESULTS_PATH.read_bytes()).hexdigest()
    SHA_PATH.write_text(f"{sha}  {RESULTS_PATH.name}\n", encoding="utf-8")

    corr_tab = res[res["row_type"] == "corr"][["hypothesis", "variable", "variant", "level", "partition", "expected_sign",
                                               "n", "obs", "p95", "p95_two", "p_dec", "months_ok", "opposite_sign",
                                               "confirmed_level", "exploratory"]]
    g6_tab = res[res["row_type"] == "g6"][["variant", "n_nearest", "n", "n_distinct_mo", "obs", "null_mean", "p95",
                                           "p_exact", "q025", "q975", "confirmed_level"]]
    g6_type_tab = res[res["row_type"] == "g6_type"][["variant", "n_nearest", "type", "nearest", "n", "n_distinct_mo",
                                                     "obs", "null_mean"]]
    g9_tab = res[res["row_type"] == "g9"][["n", "obs", "p95", "p_perm", "p_cyc", "confirmed_level"]]
    g9_top = res[res["row_type"] == "g9_top"][["top_M", "top_P", "top_both"]]
    ctrl_tab = res[res["row_type"] == "control"][["variable", "level", "n", "obs", "p_dec", "confirmed_level"]]
    lines = [
        "# 19b. Проверка гипотез Г2, Г3 (корреляции), Г6 (переходы), Г9 (переходы и сдвиги долей)", "",
        "Сгенерировано `src/19b_hypothesis_tests_corr_trans.py`. Пороги — замороженные шагов 18a и 18b; определения — "
        "те же. Вердикта по гипотезам здесь нет (шаг 19c).", "",
        "## Заморозка", "", *log, "",
        "## Правила", "",
        f"- Г2, Г3: ρ Спирмена как в шаге 18 (те же переменные, преобразования и множество МО; n сверено с порогами). "
        f"p_dec — односторонний по ожидаемому знаку по вектору нуля `{NULL_DEC_PATH.name}` (декабрь {MONTH} или "
        f"`{STATIC}`). months_ok — месяцы, где obs_m за ожидаемым знаком и дальше p95_m. Для вариантов с месячными "
        f"долями confirmed_level = (p_dec < {ALPHA}) и (months_ok ≥ {MONTHS_RULE}); для `{STATIC}` правила месяцев нет: "
        f"тот же знак и p_dec < {ALPHA}. opposite_sign — ρ противоположного знака и |ρ| > двустороннего p95. Г3 "
        "помечена exploratory.",
        "- Правило месяцев применено ко всем вариантам с месячными долями — основному и Г3 «без городов федерального "
        "значения» (у него пороги тоже есть для всех 24 месяцев).",
        f"- Г6: переходы и ближайшие типы как в 18b. p_exact = P(X ≥ x_obs), X — сумма независимых Binomial(n_i, q_i) "
        f"(q_i — базовый уровень типа из порогов 18b). Бутстреп по МО, у которых есть переходы: {BOOT_B} повторов, "
        f"seed {TEST_SEED_G6}, МО берутся с возвращением вместе со всеми своими переходами. confirmed_level = "
        "S_obs > p95 и нижняя граница (2,5%) > null_mean.",
        f"- Г9: ряды P_t, M_t — замороженные 18b. p_perm — {N_PERM} перестановок месяцев (seed {TEST_SEED_G9}); "
        "p_cyc — по 22 циклическим сдвигам (грубый). confirmed_level = ρ > управляющего порога 18b. p_perm сохранён для "
        "поправки Холма в 19c.", "",
        "## Г2, Г3", "", md(corr_tab), "",
        "## Г6", "", md(g6_tab), "", "По исходным типам (описание):", "", md(g6_type_tab), "",
        "## Г9", "", md(g9_tab), "", f"Месяцы с {TOP_N} наибольшими M_t и P_t и их пересечение:", "", md(g9_top), "",
        "## Проверки и контроли", "", *check_log, "", md(ctrl_tab), "",
        "## Время", "", f"- корреляции (24 месяца × варианты × уровни): {t_corr:.1f} с; Г6: {t_g6:.1f} с; "
        f"весь шаг: {time.time() - t0:.1f} с", "", f"sha256 `{RESULTS_PATH.name}`: `{sha}`", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    for name, tab in [("Г2, Г3", corr_tab), ("Г6", g6_tab), ("Г6 по типам", g6_type_tab), ("Г9", g9_tab),
                      ("Г9 месяцы", g9_top), ("КОНТРОЛИ", ctrl_tab)]:
        print(f"===== {name}\n{tab.to_string(index=False)}")
    print("\n".join(check_log))
    print(f"sha256 {sha}\nвесь шаг: {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
