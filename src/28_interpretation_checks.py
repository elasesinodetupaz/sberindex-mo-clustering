"""Шаг 28. Проверка интерпретаций: описание набора, сверка чисел прежнего отчёта, разведочные проверки.

Разведочно, вне предрегистрации. Новых данных нет; данные не меняются. Считается только названное в задании шага 28:
  A) дословные цитаты описания набора о расходах (из config: цитаты собраны вручную, адрес и дата обращения рядом);
  B) независимый пересчёт восьми чисел прежнего отчёта (чат 03.10.2026) с ожиданиями и допусками из config;
     при расхождении — оба числа и STOP, выходные файлы не пишутся;
  C) разведочные проверки: состав типов по виду МО; ρ Спирмена на уровне МО с бутстрепом по МО;
     медианы доли транспорта по типам в группах размера.
Выборка, доли и показатели Росстата — build_frame шага 23 (как в шаге 27); месячные типы — колонка cluster
траекторий шага 12 (как в шаге 18b); уровень «внутри регионов» — prepare из hypothesis_tools (как в шагах 19 и 22).

Выход: data/processed/interp_checks_28.parquet (long), notebooks/28_interp_checks.md
Запуск из корня проекта:  .venv/bin/python src/28_interpretation_checks.py
"""
import hashlib
import importlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score

from config import (BOOT_B, CHECKS_EXPECT, DATA_ANSWERS, DATA_QUOTES, FAR_EAST_DISTRICT, FINAL_K, HEALTH_TYPES,
                    HYP_MIN_REGION_N, HYP_MIN_TYPES, MIDDLE_TYPE, MIN_GROUP, MIN_TYPE_N_GROUP, MONTH, N_NEAREST,
                    PREV_DEC_MONTH, PREV_SOURCES, REMOTE_TYPE, SEED_28, SIZE_BINS_TH, SMALL_TYPES, STABLE_SHARE,
                    SUMMER_MONTHS, TOL_ARI, TOL_RHO, TOL_SHARE, TOP_TRANSITIONS, YAKUTIA)
from hypothesis_tools import LEVEL_ALL, LEVEL_REGION, LEVELS, Spearman, prepare, rank_residual, spearman_rho

step23 = importlib.import_module("23_type_portraits")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TRAJ_PATH = PROCESSED_DIR / f"kmeans_k{FINAL_K}_trajectories.parquet"
MONTHLY_DIR = PROCESSED_DIR / f"kmeans_labels_k{FINAL_K}"
RESULTS_19B = PROCESSED_DIR / "hypothesis_results_19b.parquet"
README_PATH = PROJECT_DIR / "README.md"
OUT_PATH = PROCESSED_DIR / "interp_checks_28.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "28_interp_checks.md"

FROZEN = {
    PROCESSED_DIR / "hypothesis_thresholds.parquet": "11081afbfc7e88dbd301004e4c739e3f25e00bf8a47704cf7c74e15e8e6665f7",
    PROCESSED_DIR / "hypothesis_results_19a.parquet": "58e1516ccbf03b2a1fca82971d5170f581815fe18fd8d22bbf261862fdad9d8e",
    RESULTS_19B: "d66d64d68a82867a5add988d4441fb0df2d35297b8c94c1514bc9e513c615d9d",
    PROCESSED_DIR / "hypothesis_results_supply.parquet": "fbfd86ec33e27ad61378987ccd53e727678017cdb1f091808db8f7545bed97db",
    PROCESSED_DIR / "hypothesis_verdicts.parquet": "01177c53c539fbabded34f03e85d6a8838726df29f93ecb9d47739f96323659f",
    PROCESSED_DIR / "composition_robustness.parquet": "41bfe1d9ba8a0925e8bf3c8c17d6b8232cf5f33db2f749c796c77a32834e37e1",
    PROCESSED_DIR / "type_portraits.parquet": "cb3ccd6e72d6a703e78a9f4e93c4b13029292fd28131e39f79f68ba76d267f7c",
    PROJECT_DIR / "data" / "raw" / "territories.parquet": "a489f1d8c948d6cba6d4404292a1937d0d1da40a0df958a25aacdb8cbb56d675",
    PROCESSED_DIR / "kmeans_labels_final.parquet": "46cf888d54ec9c6c8ca148f001f867b9408727257d9c98f6240fbe36aa60b22e",
}
CATS = ["Продовольствие", "Здоровье", "Общепит", "Транспорт", "Маркетплейсы"]
EXPL = "разведочно, вне предрегистрации"
FORBIDDEN = ["подтверждено", "доказано", "причина", "вызывает", "объясняется", "из-за", "следует"]

ROWS: list = []


def rec(block, item, metric, value, n=None, type_="", group="", level="", q025=np.nan, q975=np.nan,
        expected="", tol=np.nan, status="", note="") -> float:
    ROWS.append({"block": block, "item": item, "metric": metric, "type": str(type_), "group": str(group), "level": level,
                 "value": float(value), "n": pd.NA if n is None else int(n), "q025": float(q025), "q975": float(q975),
                 "expected": str(expected), "tol": float(tol), "status": status, "note": note})
    return float(value)


def check_frozen(when: str, log: list) -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(PROJECT_DIR)
        print(f"{when}: {got}  {rel} {'OK' if got == expected else 'MISMATCH'}")
        log.append(f"- {when}: `{rel}` — {'совпадает' if got == expected else 'НЕ СОВПАДАЕТ'}")
        if got != expected:
            raise SystemExit(f"STOP: sha256 {rel} = {got}, ожидается {expected}")


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, disable_numparse=True)


def load_wide(ids: np.ndarray) -> pd.DataFrame:
    """МО × 24 месяца → канонический тип (колонка cluster траекторий шага 12, как в шаге 18b)."""
    wide = pd.read_parquet(TRAJ_PATH).pivot(index="territory_id", columns="month", values="cluster")
    if wide.shape != (len(ids), 24) or not np.array_equal(wide.index.to_numpy(), ids) or wide.isna().any().any():
        raise SystemExit(f"STOP: траектории не дают таблицу 2004 МО × 24 месяца: {wide.shape}")
    if wide.columns[-1] != MONTH or PREV_DEC_MONTH not in wide.columns or not set(SUMMER_MONTHS) <= set(wide.columns):
        raise SystemExit("STOP: в траекториях нет нужных месяцев")
    return wide.astype(int)


def raw_month(month: str, ids: np.ndarray) -> np.ndarray:
    """Номера KMeans месяца (без сопоставления) — для независимой сверки ARI."""
    s = pd.read_parquet(MONTHLY_DIR / f"{month}.parquet").set_index("territory_id")["cluster"]
    return s.reindex(ids).to_numpy()


# --- C2: ρ на уровне МО с бутстрепом по МО ---

def _region_ranks(v: np.ndarray, reg: np.ndarray) -> np.ndarray:
    """Нормированный ранг внутри региона, (ранг − 0.5) / n_региона, как within_region_rank (векторно)."""
    s = pd.Series(v)
    g = s.groupby(reg)
    return ((g.rank(method="average") - 0.5) / g.transform("size")).to_numpy()


def _rho_sample(x: np.ndarray, y: np.ndarray, reg: np.ndarray, level: str, ctrl, incl: np.ndarray) -> float:
    """Остатки (для частной ρ) — по всем МО с данными, как в шаге 22; затем отбор МО (incl) и ρ на нужном уровне."""
    if ctrl is not None:
        x, y = rank_residual(x, ctrl), rank_residual(y, ctrl)
    x, y, reg = x[incl], y[incl], reg[incl]
    if level == LEVEL_REGION:
        return Spearman(_region_ranks(x, reg), _region_ranks(y, reg), ranked=True).observed()
    return spearman_rho(x, y)


def rho_boot(x, y, region, level: str, seed: int, ctrl=None) -> dict:
    """ρ как в шагах 19 и 22 (prepare: отбор МО, «внутри регионов» — нормированные ранги внутри региона, регион с числом
    МО с данными меньше HYP_MIN_REGION_N не участвует); частная ρ — остатки рангов после МНК на ранги ctrl (как в шаге 22).
    Бутстреп: BOOT_B выборок с возвращением из всех МО с данными; в каждой остатки (по всем МО выборки), отбор МО
    исключённых регионов и ранги внутри региона пересчитываются так же, как для наблюдаемого ρ."""
    m = ~np.isnan(x) & ~np.isnan(y)
    if ctrl is not None:
        m &= np.isfinite(ctrl)
    xx, yy = np.where(m, x, np.nan), np.where(m, y, np.nan)
    if ctrl is not None:
        xr = rank_residual(xx, np.where(m, ctrl, np.nan))
        yr = rank_residual(yy, np.where(m, ctrl, np.nan))
    else:
        xr, yr = xx, yy
    p = prepare(np.column_stack([xr, yr]), None, region, level, MIN_GROUP, HYP_MIN_TYPES, HYP_MIN_REGION_N)
    obs = Spearman(p["y"][:, 0], p["y"][:, 1], ranked=level == LEVEL_REGION).observed()
    xs, ys, rs = x[m], y[m], region[m]                   # все МО с данными
    cs = ctrl[m] if ctrl is not None else None
    incl = p["mask"][m]                                  # из них участвуют (регион не исключён)
    check = _rho_sample(xs, ys, rs, level, cs, incl)
    if abs(check - obs) > 1e-9:
        raise SystemExit(f"STOP: ρ бутстреп-функции {check} не совпадает с ρ шагов 19/22 {obs}")
    rng = np.random.default_rng(seed)
    n_all = len(xs)
    boot = np.empty(BOOT_B)
    for b in range(BOOT_B):
        idx = rng.integers(0, n_all, n_all)
        boot[b] = _rho_sample(xs[idx], ys[idx], rs[idx], level, None if cs is None else cs[idx], incl[idx])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"rho": obs, "n": p["n"], "q025": float(lo), "q975": float(hi), "n_regions": int(len(np.unique(rs[incl]))),
            "excluded_regions": len(p["excluded_regions"])}


def main() -> None:
    frozen_log: list = []
    check_frozen("до", frozen_log)
    d = step23.step18.load()
    step23.step18.clean_rosstat(d, [])
    sup, _ = step23.step21.supply_values(d["ids"])
    D = step23.build_frame(d, sup)                       # выборка, доли (×100), Росстат 2023 — как в шаге 23
    z = step23.z_profile(d, D)                           # средний z-score пяти долей по типу (декабрь 2024)
    ids = d["ids"]
    region = d["region"]
    if len(D) != 2004:
        raise SystemExit(f"STOP: ожидалось 2004 МО, получено {len(D)}")
    wide = load_wide(ids)
    canon = D["type"].to_numpy()
    E = CHECKS_EXPECT
    ctrl_rows, bad = [], []

    def ctrl(item, name, got, exp, tol, fmt="{:.4f}"):
        ok = abs(got - exp) <= tol + 1e-12
        ctrl_rows.append({"пункт": item, "что": name, "результат": fmt.format(got), "ожидание": fmt.format(exp),
                          "допуск": f"{tol:g}", "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})
        if not ok:
            bad.append(ctrl_rows[-1])
        return ok

    lines_b = []

    # B1. МО, менявшие тип хотя бы раз
    changed = wide.nunique(axis=1) > 1
    n_changed = int(changed.sum())
    W = wide.to_numpy()
    trans = W[:, 1:] != W[:, :-1]
    n_trans = int(trans.sum())
    r19 = pd.read_parquet(RESULTS_19B)
    g6 = r19[(r19["row_type"] == "g6") & (r19["n_nearest"] == N_NEAREST)]
    ctrl("B1", "МО выборки, менявшие тип хотя бы раз", n_changed, E["changed_any"], 0, "{:.0f}")
    ctrl("B1", "сверка: различных МО с переходами в шаге 19b (Г6)", n_changed, int(g6["n_distinct_mo"].iloc[0]), 0, "{:.0f}")
    ctrl("B1", "сверка: переходов всего, шаг 19b (Г6)", n_trans, int(g6["n"].iloc[0]), 0, "{:.0f}")
    rec("B", "B1", "МО, менявшие тип хотя бы раз", n_changed, len(ids), expected=E["changed_any"], tol=0)
    rec("B", "B1", "переходов (МО-месяцев со сменой типа)", n_trans, len(ids))

    # B2. ARI декабрь 2023 — декабрь 2024
    ari = adjusted_rand_score(wide[PREV_DEC_MONTH], wide[MONTH])
    ari_raw = adjusted_rand_score(raw_month(PREV_DEC_MONTH, ids), raw_month(MONTH, ids))
    if abs(ari - ari_raw) > 1e-12:
        raise SystemExit(f"STOP: ARI по траекториям {ari} ≠ ARI по номерам KMeans месяцев {ari_raw}")
    ctrl("B2", f"ARI {PREV_DEC_MONTH} — {MONTH}", ari, E["ari_dec23_dec24"], TOL_ARI)
    rec("B", "B2", f"ARI {PREV_DEC_MONTH} — {MONTH}", ari, len(ids), expected=E["ari_dec23_dec24"], tol=TOL_ARI,
        note="по колонке cluster траекторий и по номерам KMeans месяцев — одно значение")

    # B3. Доля МО, остающихся в декабрьском типе не менее STABLE_SHARE месяцев
    share_in = (W == canon[:, None]).mean(axis=1)
    b3 = []
    for t in range(FINAL_K):
        m = canon == t
        v = float((share_in[m] >= STABLE_SHARE).mean())
        ctrl("B3", f"тип {t}: доля МО с долей месяцев в своём типе ≥ {STABLE_SHARE}", v, E["stable_share_by_type"][t], TOL_SHARE)
        rec("B", "B3", "доля МО, остающихся в типе", v, int(m.sum()), type_=t, expected=E["stable_share_by_type"][t], tol=TOL_SHARE)
        b3.append({"тип": t, "МО типа": int(m.sum()), "остаются в типе": int((share_in[m] >= STABLE_SHARE).sum()),
                   "доля": f"{v:.3f}", "медиана доли месяцев в своём типе": f"{np.median(share_in[m]):.3f}"})

    # B4. Дальневосточные МО в типе 4
    fe = (D["fd"] == FAR_EAST_DISTRICT).to_numpy()
    m4 = canon == REMOTE_TYPE
    ctrl("B4", f"тип {REMOTE_TYPE}, {MONTH} (kmeans_labels_final): МО Дальнего Востока", int((m4 & fe).sum()),
         E["remote_far_east_dec"][0], 0, "{:.0f}")
    ctrl("B4", f"тип {REMOTE_TYPE}, {MONTH}: всего МО", int(m4.sum()), E["remote_far_east_dec"][1], 0, "{:.0f}")
    rec("B", "B4", "доля МО Дальнего Востока в типе", (m4 & fe).sum() / m4.sum(), int(m4.sum()), type_=REMOTE_TYPE,
        group=MONTH, expected="93 из 111", note="kmeans_labels_final")
    b4 = [{"месяц": f"{MONTH} (канон)", "МО типа 4": int(m4.sum()), "из них Дальний Восток": int((m4 & fe).sum()),
           "доля": f"{(m4 & fe).sum() / m4.sum():.3f}"}]
    lo_e, hi_e = E["remote_far_east_summer"]
    for mo in SUMMER_MONTHS:
        mm = W[:, list(wide.columns).index(mo)] == REMOTE_TYPE
        v = float((mm & fe).sum() / mm.sum())
        ok = lo_e - TOL_SHARE - 1e-12 <= v <= hi_e + TOL_SHARE + 1e-12
        ctrl_rows.append({"пункт": "B4", "что": f"тип {REMOTE_TYPE}, {mo} (траектории): доля МО Дальнего Востока",
                          "результат": f"{v:.4f}", "ожидание": f"{lo_e:.2f}–{hi_e:.2f}", "допуск": f"{TOL_SHARE:g}",
                          "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})
        if not ok:
            bad.append(ctrl_rows[-1])
        rec("B", "B4", "доля МО Дальнего Востока в типе", v, int(mm.sum()), type_=REMOTE_TYPE, group=mo,
            expected=f"{lo_e}–{hi_e}", tol=TOL_SHARE, note="траектории, колонка cluster")
        b4.append({"месяц": mo, "МО типа 4": int(mm.sum()), "из них Дальний Восток": int((mm & fe).sum()), "доля": f"{v:.3f}"})

    # B5. Доля дисперсии между регионами
    b5 = []
    for c, exp in E["between_region_var"].items():
        x = D[c].to_numpy(float) / 100
        within = x - pd.Series(x).groupby(D["region"].to_numpy()).transform("mean").to_numpy()
        v = 1 - float((within ** 2).sum() / ((x - x.mean()) ** 2).sum())
        ctrl("B5", f"доля дисперсии между регионами: {c}", v, exp, TOL_SHARE)
        rec("B", "B5", "доля дисперсии между регионами", v, len(x), group=c, expected=exp, tol=TOL_SHARE)
        b5.append({"категория": c, "R": f"{v:.4f}", "МО": len(x), "регионов": int(D["region"].nunique())})

    # B6. Дальний Восток вне типа 4
    sub = D[fe & ~m4]
    v6 = float(sub["Маркетплейсы"].median())
    ctrl("B6", f"медиана доли маркетплейсов, %, МО Дальнего Востока вне типа {REMOTE_TYPE}", v6,
         E["far_east_non_remote_mp"]["median_pct"], TOL_SHARE * 100, "{:.3f}")
    ctrl("B6", f"число МО Дальнего Востока вне типа {REMOTE_TYPE}", len(sub), E["far_east_non_remote_mp"]["n"], 0, "{:.0f}")
    rec("B", "B6", "медиана доли маркетплейсов, %", v6, len(sub), group="Дальний Восток вне типа 4",
        expected=E["far_east_non_remote_mp"]["median_pct"], tol=TOL_SHARE * 100)
    v6b = float(D.loc[fe & m4, "Маркетплейсы"].median())
    rec("B", "B6", "медиана доли маркетплейсов, %", v6b, int((fe & m4).sum()), type_=REMOTE_TYPE, group="Дальний Восток")

    # B7. Якутия: ρ(размер, market_access)
    yk = D[(D["region"] == YAKUTIA) & D["log_pop"].notna() & D["market_access"].notna()]
    v7 = spearman_rho(yk["log_pop"].to_numpy(), yk["market_access"].to_numpy())
    ctrl("B7", "Якутия: ρ Спирмена (население 2023, market_access)", v7, E["yakutia_rho_logpop_ma"]["rho"], TOL_RHO)
    ctrl("B7", "Якутия: n", len(yk), E["yakutia_rho_logpop_ma"]["n"], 0, "{:.0f}")
    rec("B", "B7", "ρ Спирмена (население 2023, market_access)", v7, len(yk), group=YAKUTIA,
        expected=E["yakutia_rho_logpop_ma"]["rho"], tol=TOL_RHO)

    # B8. Тип 1: переходы и ближайшие типы
    frm, to = W[:, :-1][trans], W[:, 1:][trans]
    dest = pd.Series(to[frm == MIDDLE_TYPE]).value_counts().sort_values(ascending=False, kind="stable")
    top = [int(t) for t in dest.index[:TOP_TRANSITIONS]]
    cen = z.loc[range(FINAL_K), CATS].to_numpy()
    dist = np.sqrt(((cen[:, None, :] - cen[None, :, :]) ** 2).sum(axis=2))
    order = [int(j) for j in np.argsort(dist[MIDDLE_TYPE], kind="stable") if j != MIDDLE_TYPE]
    nearest = order[:N_NEAREST]
    g6t = r19[(r19["row_type"] == "g6_type") & (r19["n_nearest"] == N_NEAREST) & (r19["type"] == MIDDLE_TYPE)]
    near19 = sorted(int(s) for s in str(g6t["nearest"].iloc[0]).split())
    for name, got, exp in [("основные направления переходов из типа 1", top, E["middle_top_transitions"]),
                           ("ближайшие к типу 1 по центроиду", nearest, E["middle_nearest"]),
                           ("сверка: ближайшие к типу 1 в шаге 19b (как множество)", sorted(nearest), near19)]:
        ok = got == exp
        ctrl_rows.append({"пункт": "B8", "что": name, "результат": ", ".join(map(str, got)),
                          "ожидание": ", ".join(map(str, exp)), "допуск": "точно", "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})
        if not ok:
            bad.append(ctrl_rows[-1])
    b8 = [{"в тип": int(t), "переходов": int(c), "доля переходов из типа 1": f"{c / dest.sum():.3f}"} for t, c in dest.items()]
    for t, c in dest.items():
        rec("B", "B8", "переходов из типа 1", c, int(dest.sum()), type_=MIDDLE_TYPE, group=f"в тип {int(t)}")
    b8d = [{"тип": j, "расстояние от центроида типа 1": f"{dist[MIDDLE_TYPE, j]:.3f}"} for j in order]
    for j in order:
        rec("B", "B8", "евклидово расстояние между центроидами (z-score)", dist[MIDDLE_TYPE, j], type_=MIDDLE_TYPE, group=f"тип {j}")

    ctrl_tab = pd.DataFrame(ctrl_rows)
    print(ctrl_tab.to_string(index=False))
    if bad:
        raise SystemExit("STOP: расхождения с ожиданиями (оба числа выше):\n" + pd.DataFrame(bad).to_string(index=False))

    # A. Цитаты: строку README сверяем с файлом
    readme = README_PATH.read_text(encoding="utf-8")
    for q in DATA_QUOTES:
        if q["source"].startswith("README.md") and q["quote"] not in readme:
            raise SystemExit(f"STOP: цитата не найдена в README.md: {q['quote']}")

    # C1. Состав типов по виду МО
    ct = pd.crosstab(D["mdt"], D["type"])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False, kind="stable").index]
    c1 = []
    for kind, row in ct.iterrows():
        r = {"вид МО": kind}
        for t in range(FINAL_K):
            share = row[t] / ct[t].sum()
            r[f"тип {t}"] = f"{row[t]} ({share * 100:.1f}%)"
            rec("C1", "C1", "МО вида в типе", row[t], int(ct[t].sum()), type_=t, group=kind, note=EXPL)
            rec("C1", "C1", "доля вида в типе", share, int(ct[t].sum()), type_=t, group=kind, note=EXPL)
        r["всего"] = int(row.sum())
        c1.append(r)

    # C2. ρ на уровне МО
    specs = [
        ("продовольствие ~ занятые в сельском хозяйстве (A)", "Продовольствие", "okved_A", SMALL_TYPES, False),
        ("здоровье ~ занятые в здравоохранении (Q)", "Здоровье", "okved_Q", HEALTH_TYPES, False),
        ("общепит ~ работники на жителя", "Общепит", "wpp", None, False),
        ("общепит ~ работники на жителя, частная ρ при учёте log(население 2023)", "Общепит", "wpp", None, True),
    ]
    c2 = []
    for k, (name, xc, yc, types, partial) in enumerate(specs):
        keep = np.isin(canon, types) if types is not None else np.ones(len(D), bool)
        x = np.where(keep, D[xc].to_numpy(float), np.nan)
        y = np.where(keep, D[yc].to_numpy(float), np.nan)
        cv = D["log_pop"].to_numpy(float) if partial else None
        for li, level in enumerate(LEVELS):
            r = rho_boot(x, y, region, level, SEED_28 + 10 * k + li, cv)
            grp = "все МО" if types is None else "типы " + ", ".join(map(str, types))
            rec("C2", "C2", name, r["rho"], r["n"], group=grp, level=level, q025=r["q025"], q975=r["q975"],
                note=f"{EXPL}; регионов {r['n_regions']}" + ("; частная ρ" if partial else ""))
            c2.append({"пара": name, "МО": grp, "уровень": level, "ρ": f"{r['rho']:+.3f}",
                       "95% интервал бутстрепа": f"[{r['q025']:+.3f}; {r['q975']:+.3f}]", "n": r["n"],
                       "регионов": r["n_regions"], "исключено регионов": r["excluded_regions"]})
            print(name, level, r)

    # C3. Доля транспорта по типам в группах размера
    c3 = []
    for g in step23.size_labels(SIZE_BINS_TH):
        s = D[D["size_group"] == g]
        cnt = s.groupby("type").size()
        med = s.groupby("type")["Транспорт"].median()
        ok_t = [t for t in cnt.index if cnt[t] >= MIN_TYPE_N_GROUP]
        ranked = sorted(ok_t, key=lambda t: (-med[t], t))
        for place, t in enumerate(ranked, 1):
            rec("C3", "C3", "медиана доли транспорта, %", med[t], int(cnt[t]), type_=t, group=g, note=f"{EXPL}; место {place}")
            c3.append({"группа, тыс. жителей": g, "место": place, "тип": int(t), "медиана транспорта, %": f"{med[t]:.2f}",
                       "n": int(cnt[t]), "МО группы": len(s)})

    # Отчёт
    src = PREV_SOURCES
    lines = [f"# 28. Проверка интерпретаций ({EXPL})", "",
             "Сгенерировано `src/28_interpretation_checks.py`. Данные не меняются: пересчёт чисел прежнего отчёта (чат 03.10.2026) "
             "и разведочные расчёты по готовым меткам и готовым показателям. Без тестов, порогов и вердиктов.", "",
             "## Заморозка", "", *frozen_log, "",
             "## A. Описание набора: к чему привязаны расходы и как покупка относится к категории", "",
             "Цитаты дословные; собраны вручную, адрес и дата обращения указаны рядом. Строка README сверяется с файлом при каждом запуске.", ""]
    for qn, title in [("1", "(1) К чему привязаны расходы"), ("2", "(2) Как покупка относится к категории")]:
        lines += [f"### {title}", ""]
        for q in DATA_QUOTES:
            if str(q["question"]) == qn:
                lines += [f"> «{q['quote']}»", "", f"Источник: {q['source']}.", ""]
        lines += [f"Ответ: {DATA_ANSWERS[qn]}", ""]
    lines += ["## B. Пересчёт чисел прежнего отчёта", "",
              f"Где получены числа прежнего отчёта: журнал сессии Claude Code `{src['log_path']}` (вне репозитория; номера строк "
              "журнала JSONL) и файлы проекта:", ""]
    lines += [f"- {it['item']}: {it['where']}" for it in src["items"]]
    lines += ["", "Определения пересчёта:", "",
              f"- Месячный тип МО — колонка `cluster` файла `{TRAJ_PATH.name}` (тип в нумерации якоря {MONTH}, как в шаге 18b); "
              f"тип {MONTH} в B3, B4 (канон), B6 и B8 (центроиды) — `kmeans_labels_final.parquet`. В {MONTH} траектории и канон "
              f"расходятся у {int((W[:, -1] != canon).sum())} МО.",
              "- B1: МО, у которых месячный тип принимает не менее двух разных значений за 24 месяца (то же, что хотя бы одна смена "
              "между соседними месяцами).",
              f"- B2: ARI (sklearn adjusted_rand_score) между разбиениями {PREV_DEC_MONTH} и {MONTH} по всем 2004 МО; ARI не зависит "
              "от нумерации, сверено с номерами KMeans месяцев.",
              f"- B3: для МО канонического типа t — доля месяцев (из 24), в которых месячный тип равен t; МО «остаётся в типе», если "
              f"доля ≥ {STABLE_SHARE}, то есть не менее {int(np.ceil(STABLE_SHARE * 24))} месяцев.",
              f"- B4: федеральный округ — `data/external/federal_districts.csv`, значение «{FAR_EAST_DISTRICT}».",
              "- B5: R = 1 − Σ_i (x_i − x̄_r(i))² / Σ_i (x_i − x̄)², x — доля категории в декабре 2024, x̄_r(i) — среднее по региону МО i, "
              "все 2004 МО и все регионы (включая регионы с одним МО).",
              "- B6: медиана доли (в процентах от «Все категории», декабрь 2024) по МО Дальнего Востока с каноническим типом ≠ 4.",
              "- B7: ρ Спирмена между log(население 2023) и market_access по МО Якутии с обоими значениями (Росстат, МО-годы с anomaly исключены).",
              f"- B8: переходы — пары соседних месяцев, где месячный тип сменился с {MIDDLE_TYPE} на другой; ближайшие — {N_NEAREST} "
              "типа с наименьшим евклидовым расстоянием между центроидами (средний z-score пяти долей МО типа, декабрь 2024, "
              "стандартизация шага 10b), как в шаге 18b.", "",
              "### Таблица ожиданий и результатов (B)", "", md(ctrl_tab), "",
              "### B3. Доля МО, остающихся в декабрьском типе", "", md(pd.DataFrame(b3)), "",
              "### B4. МО Дальнего Востока в типе 4", "", md(pd.DataFrame(b4)), "",
              "### B5. Доля дисперсии между регионами", "", md(pd.DataFrame(b5)), "",
              f"### B6. Медиана доли маркетплейсов: тип 4 на Дальнем Востоке {v6b:.2f}% (n = {int((fe & m4).sum())}), "
              f"МО Дальнего Востока вне типа 4 {v6:.2f}% (n = {len(sub)})", "",
              "### B8. Переходы из типа 1 по конечному типу", "", md(pd.DataFrame(b8)), "",
              "Расстояния между центроидами от типа 1:", "", md(pd.DataFrame(b8d)), "",
              "## C. Разведочные проверки (вне предрегистрации)", "",
              "### C1. Состав типов по виду МО (`municipal_district_type`, territories.parquet); в скобках доля вида в типе", "",
              md(pd.DataFrame(c1)), "",
              "### C2. ρ Спирмена на уровне МО", "",
              f"- Данные: доли декабря 2024 (в процентах от «Все категории»); Росстат 2023 (шаг 17, МО-годы с anomaly исключены); "
              "доли занятых по ОКВЭД — по организациям без малого бизнеса по месту нахождения; работники на жителя пусты у МО без "
              "населения 2023.",
              f"- Правило уровня «внутри регионов» (шаги 19 и 22, `prepare` и `within_region_rank` в `src/hypothesis_tools.py`): "
              f"значения заменяются нормированным рангом внутри региона, (ранг − 0.5) / n_региона, средние ранги при связках; регион с "
              f"числом МО с данными меньше {HYP_MIN_REGION_N} (HYP_MIN_REGION_N) не участвует; ρ — Пирсон на этих рангах. «Общий» — ρ "
              "Спирмена по всем отобранным МО.",
              "- Частная ρ: ранги каждой переменной минус МНК на ранги log(население 2023) (`rank_residual`, как в шаге 22), затем ρ "
              "на нужном уровне.",
              f"- Интервал: {BOOT_B} выборок с возвращением из всех МО с данными, seed {SEED_28} + 10 · номер пары + номер уровня; "
              "в каждой выборке остатки (по всем МО выборки), исключение МО малых регионов и ранги внутри региона пересчитываются "
              "так же, как для наблюдаемого ρ; 2.5% и 97.5% квантили.",
              f"- Типы: продовольствие — {', '.join(map(str, SMALL_TYPES))} (SMALL_TYPES шага 23); здоровье — "
              f"{', '.join(map(str, HEALTH_TYPES))}.", "",
              md(pd.DataFrame(c2)), "",
              f"### C3. Медиана доли транспорта по типам в группах размера (типы с числом МО в группе не менее {MIN_TYPE_N_GROUP}, по убыванию)", "",
              "Население 2023; МО без населения в группы не входят.", "",
              md(pd.DataFrame(c3)), ""]
    text = "\n".join(lines)
    hits = {w: len(re.findall(w, text, flags=re.I)) for w in FORBIDDEN}
    nobr = re.sub(r"\[[+-]?\d+\.\d+; [+-]?\d+\.\d+\]", "", text)
    if any(hits.values()) or "[" in nobr or "]" in nobr:
        raise SystemExit(f"STOP: запрещённые слова {hits} или квадратные скобки вне интервала")
    print("запрещённые слова:", hits, "| квадратные скобки вне интервала: 0")
    check_frozen("после", frozen_log)
    i = lines.index("## Заморозка") + 2 + len(FROZEN)
    lines[i:i] = frozen_log[len(FROZEN):]
    out = pd.DataFrame(ROWS)
    out["n"] = out["n"].astype("Int64")
    if out.duplicated(["block", "item", "metric", "type", "group", "level"]).any():
        raise SystemExit("STOP: дубликаты ключа")
    out.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"записано: {OUT_PATH.name} ({len(out)} строк), {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
