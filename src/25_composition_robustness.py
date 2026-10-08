"""Шаг 25. Устойчивость типологии и ключевых связей к композиционной природе долей расходов.

Всё РАЗВЕДОЧНО, вне предрегистрации: никаких p-значений, порогов и вердиктов. Замороженные файлы шагов 18–22
только проверяются по sha256 (до и после) и не читаются для расчётов.

Данные: доли пяти категорий category_shares.parquet за MONTH (2004 МО), rest = 1 − сумма пяти долей.
1. Воспроизведение канона: KMeans (k = FINAL_K, n_init и random_state шага 10) на z-score пяти долей — ARI = 1 с
   kmeans_labels_final, иначе STOP; ARI канона с запусками на KMEANS_SEEDS_CHECK.
2. Кодировки: (а) clr по шести частям + z-score, (б) clr по шести частям, (в) clr по пяти частям, нормированным на
   сумму 1, + z-score; ARI с каноном и SW в преобразованном пространстве.
3. SVD на z-score пяти долей: доля дисперсии PC1, нагрузки, ρ Спирмена PC1 и rest.
4. rest по типам канона: медиана, квартили, n; η²_H (общий уровень и внутри регионов).
5. Г2 и Г3 по долям и по alr_i = ln(доля_i / rest): ρ Спирмена на трёх уровнях.
6. η²_H по типам для долей и alr пяти категорий, общий уровень и внутри регионов.
Сверка с контрольными значениями пользователя (CONTROLS ниже): при расхождении выводятся оба числа, STOP.

Выход: data/processed/composition_robustness.parquet (long: block, metric, variant, level, value, n, n_regions),
       notebooks/25_composition_robustness.md
Запуск из корня проекта:  .venv/bin/python src/25_composition_robustness.py
"""
import hashlib
import importlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from config import COMP_MIN_MO_REGION, FINAL_K, KMEANS_N_INIT, KMEANS_RANDOM_STATE, KMEANS_SEEDS_CHECK, MONTH
from hypothesis_tools import kw_eta2, spearman_rho, within_region_rank
from network_utils import SHARE_COLUMNS, month_shares, standardize_shares

step18 = importlib.import_module("18_hypothesis_thresholds")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
EXTERNAL_DIR = PROJECT_DIR / "data" / "external"
OUT_PATH = PROCESSED_DIR / "composition_robustness.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "25_composition_robustness.md"

FROZEN = {
    PROCESSED_DIR / "hypothesis_thresholds.parquet": "11081afbfc7e88dbd301004e4c739e3f25e00bf8a47704cf7c74e15e8e6665f7",
    PROCESSED_DIR / "hypothesis_results_19a.parquet": "58e1516ccbf03b2a1fca82971d5170f581815fe18fd8d22bbf261862fdad9d8e",
    PROCESSED_DIR / "hypothesis_results_19b.parquet": "d66d64d68a82867a5add988d4441fb0df2d35297b8c94c1514bc9e513c615d9d",
    EXTERNAL_DIR / "rosstat_retail_2023_2024.parquet": "58f32111b48c4189811f61c6f462d9b0b8e06d3da431dd99866c8f33e1378a80",
    PROCESSED_DIR / "hypothesis_thresholds_supply.parquet": "c7d0a880d7cbac769ed7c7c8505dffea2d49eb6d3ce525b824238bc5368ddbf6",
    PROCESSED_DIR / "hypothesis_results_supply.parquet": "fbfd86ec33e27ad61378987ccd53e727678017cdb1f091808db8f7545bed97db",
    PROCESSED_DIR / "hypothesis_verdicts.parquet": "01177c53c539fbabded34f03e85d6a8838726df29f93ecb9d47739f96323659f",
}
CATS = [c.replace("share_", "") for c in SHARE_COLUMNS]
SHARES = "доли"
ALR = "лог-отношение"
L_ALL, L_IN, L_BETWEEN = "все МО", "внутри регионов", "между регионами"
REF = "(расчёт шага 25, разведочно)"
FORBIDDEN = ["доказано", "причина", "вызывает", "объясняется", "из-за"]

# Контрольные значения пользователя (сообщение с заданием шага 25); допуск ARI 0.03, остальное 0.02
TOL_ARI, TOL = 0.03, 0.02
TOL_REST_MED = 0.05      # медианы rest по типам, п.п.: контрольные значения пользователя округлены до 0.1 п.п.
CONTROLS = [
    ("ARI с каноном: (а) clr6 + z", "ari_a", 0.28, TOL_ARI),
    ("ARI с каноном: (б) clr6", "ari_b", 0.30, TOL_ARI),
    ("ARI с каноном: (в) clr5 + z", "ari_c", 0.28, TOL_ARI),
    ("ARI канона с другими seed: min", "seed_min", 0.89, TOL_ARI),
    ("ARI канона с другими seed: медиана", "seed_med", 0.97, TOL_ARI),
    ("PC1: доля дисперсии", "pc1_var", 0.57, TOL),
    ("ρ(PC1, rest)", "pc1_rest", -0.76, TOL),
    *[(f"медиана rest, тип {t}, %", f"rest_med_{t}", v, TOL_REST_MED) for t, v in enumerate([24.5, 29.2, 23.2, 33.2, 32.4, 24.4])],
    ("Г2, все МО: по долям", "g2_all_s", 0.117, TOL), ("Г2, все МО: по alr", "g2_all_a", 0.015, TOL),
    ("Г2, внутри регионов: по долям", "g2_in_s", -0.298, TOL), ("Г2, внутри регионов: по alr", "g2_in_a", -0.330, TOL),
    ("Г2, между регионами: по долям", "g2_btw_s", 0.49, TOL), ("Г2, между регионами: по alr", "g2_btw_a", 0.50, TOL),
    ("Г2, между регионами: n регионов", "g2_btw_nreg", 69, 0),
    ("Г3, все МО: по долям", "g3_all_s", -0.267, TOL), ("Г3, все МО: по alr", "g3_all_a", -0.278, TOL),
    ("Г3, внутри регионов: по долям", "g3_in_s", -0.462, TOL), ("Г3, внутри регионов: по alr", "g3_in_a", -0.479, TOL),
    ("η²_H Здоровье, общий: по долям", "eta_Здоровье_s", 0.637, TOL), ("η²_H Здоровье, общий: по alr", "eta_Здоровье_a", 0.418, TOL),
    ("η²_H Маркетплейсы, общий: по долям", "eta_Маркетплейсы_s", 0.661, TOL), ("η²_H Маркетплейсы, общий: по alr", "eta_Маркетплейсы_a", 0.658, TOL),
    ("η²_H Общепит, общий: по долям", "eta_Общепит_s", 0.665, TOL), ("η²_H Общепит, общий: по alr", "eta_Общепит_a", 0.617, TOL),
    ("η²_H rest, общий", "eta_rest", 0.596, TOL),
]

ROWS: list = []


def rec(block, metric, variant, level, value, n=None, n_regions=None) -> float:
    ROWS.append({"block": block, "metric": metric, "variant": variant, "level": level, "value": float(value),
                 "n": None if n is None else int(n), "n_regions": None if n_regions is None else int(n_regions)})
    return float(value)


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, disable_numparse=True)


def check_frozen(when: str, log: list) -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(PROJECT_DIR)
        print(f"{when}: {got}  {rel} {'OK' if got == expected else 'MISMATCH'}")
        log.append(f"- {when}: `{rel}` — {'совпадает' if got == expected else 'НЕ СОВПАДАЕТ'}")
        if got != expected:
            raise SystemExit(f"STOP: sha256 {rel} = {got}, ожидается {expected}")


def kmeans(X: np.ndarray, seed: int = KMEANS_RANDOM_STATE) -> np.ndarray:
    return KMeans(n_clusters=FINAL_K, random_state=seed, n_init=KMEANS_N_INIT).fit_predict(X)


def clr(P: np.ndarray) -> np.ndarray:
    L = np.log(P)
    return L - L.mean(axis=1, keepdims=True)


def big_regions(region: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Маска МО из регионов, где МО с данными не меньше COMP_MIN_MO_REGION."""
    cnt = pd.Series(region[ok]).value_counts()
    return ok & np.isin(region, cnt[cnt >= COMP_MIN_MO_REGION].index)


def rho_levels(y: np.ndarray, x: np.ndarray, region: np.ndarray) -> dict:
    ok = ~np.isnan(x) & ~np.isnan(y)
    out = {L_ALL: (spearman_rho(y[ok], x[ok]), int(ok.sum()), None)}
    okk = big_regions(region, ok)
    ry, rx = within_region_rank(y[okk], region[okk]), within_region_rank(x[okk], region[okk])
    n_reg = int(len(np.unique(region[okk])))
    out[L_IN] = (float(np.corrcoef(ry, rx)[0, 1]), int(okk.sum()), n_reg)
    g = pd.DataFrame({"r": region[okk], "x": x[okk], "y": y[okk]}).groupby("r").median()
    out[L_BETWEEN] = (spearman_rho(g["y"].to_numpy(), g["x"].to_numpy()), int(okk.sum()), int(len(g)))
    return out


def eta_levels(y: np.ndarray, labels: np.ndarray, region: np.ndarray) -> dict:
    ok = ~np.isnan(y)
    okk = big_regions(region, ok)
    return {"общий": (kw_eta2(y[ok], labels[ok])[0], int(ok.sum()), None),
            "внутри регионов": (kw_eta2(within_region_rank(y[okk], region[okk]), labels[okk])[0], int(okk.sum()),
                                int(len(np.unique(region[okk]))))}


def f2(v) -> str:
    return f"{v:+.3f}"


def main() -> None:
    frozen_log: list = []
    check_frozen("до", frozen_log)
    d = step18.load()
    step18.clean_rosstat(d, [])                                     # МО-годы с anomaly — пустые во всех показателях Росстата
    ids, region, labels = d["ids"], d["region"], d["parts"][step18.DEC]
    m = month_shares(d["shares"].drop(columns=["month"]), MONTH).reset_index(drop=True)
    if not np.array_equal(m["territory_id"].to_numpy(), ids):
        raise SystemExit("STOP: порядок МО в category_shares не совпадает с выборкой")
    S = m[list(SHARE_COLUMNS)].to_numpy(float)
    rest = 1.0 - S.sum(axis=1)
    if np.isnan(S).any() or (S <= 0).any() or (rest <= 0).any():
        raise SystemExit(f"STOP: пропуски, нулевые доли или rest ≤ 0: NaN {int(np.isnan(S).sum())}, "
                         f"доли ≤ 0 {int((S <= 0).sum())}, rest ≤ 0 {int((rest <= 0).sum())}")
    print(f"МО: {len(S)}; rest: min {rest.min():.4f}, max {rest.max():.4f}")
    rec("данные", "rest: минимум", SHARES, "все МО", rest.min(), len(S))
    rec("данные", "rest: максимум", SHARES, "все МО", rest.max(), len(S))
    my: dict = {}

    # 1. канон и seed
    Z, _ = standardize_shares(S)
    ari0 = adjusted_rand_score(labels, kmeans(Z))
    print(f"ARI воспроизведения канона: {ari0:.6f}")
    if not np.isclose(ari0, 1.0):
        raise SystemExit(f"STOP: KMeans на z-score пяти долей не воспроизводит канон: ARI = {ari0:.6f}")
    rec("1. канон", "ARI воспроизведения канона", "z-score пяти долей", "все МО", ari0, len(S))
    seed_rows = []
    for s in KMEANS_SEEDS_CHECK:
        a = rec("1. канон", f"ARI канона с seed {s}", "z-score пяти долей", "все МО", adjusted_rand_score(labels, kmeans(Z, s)), len(S))
        seed_rows.append({"seed": str(s), "ARI с каноном": f"{a:.3f}"})
    seeds = [r["value"] for r in ROWS if r["metric"].startswith("ARI канона с seed")]
    my["seed_min"] = rec("1. канон", "ARI канона с другими seed: min", "z-score пяти долей", "все МО", min(seeds), len(seeds))
    my["seed_med"] = rec("1. канон", "ARI канона с другими seed: медиана", "z-score пяти долей", "все МО", float(np.median(seeds)), len(seeds))

    # 2. кодировки
    P5 = S / S.sum(axis=1, keepdims=True)
    enc = [("а", "ari_a", "clr по шести частям (пять долей и rest), затем z-score", standardize_shares(clr(np.column_stack([S, rest])))[0]),
           ("б", "ari_b", "clr по шести частям без стандартизации", clr(np.column_stack([S, rest]))),
           ("в", "ari_c", "clr по пяти частям, нормированным на сумму 1, затем z-score", standardize_shares(clr(P5))[0])]
    enc_rows = []
    sw0 = rec("2. кодировки", "SW канона в z-score пяти долей", "канон", "все МО", silhouette_score(Z, labels), len(S))
    enc_rows.append({"кодировка": "канон: z-score пяти долей", "ARI с каноном": f"{ari0:.3f}", "SW": f"{sw0:.3f}"})
    for key, cname, desc, X in enc:
        lab_x = kmeans(X)
        my[cname] = rec("2. кодировки", "ARI с каноном", f"({key}) {desc}", "все МО", adjusted_rand_score(labels, lab_x), len(S))
        sw = rec("2. кодировки", "SW в преобразованном пространстве", f"({key}) {desc}", "все МО", silhouette_score(X, lab_x), len(S))
        enc_rows.append({"кодировка": f"({key}) {desc}", "ARI с каноном": f"{my[cname]:.3f}", "SW": f"{sw:.3f}"})

    # 3. главная ось
    Zc = Z - Z.mean(axis=0)
    _, sv, Vt = np.linalg.svd(Zc, full_matrices=False)
    v1 = Vt[0] * (1 if Vt[0][CATS.index("Продовольствие")] > 0 else -1)   # знак: нагрузка продовольствия > 0
    pc1 = Zc @ v1
    my["pc1_var"] = rec("3. главная ось", "PC1: доля дисперсии", "z-score пяти долей", "все МО", sv[0] ** 2 / (sv ** 2).sum(), len(S))
    load_rows = [{"категория": c, "нагрузка PC1": f"{rec('3. главная ось', f'PC1: нагрузка {c}', 'z-score пяти долей', 'все МО', v, len(S)):+.3f}"}
                 for c, v in zip(CATS, v1)]
    my["pc1_rest"] = rec("3. главная ось", "ρ(PC1, rest)", "z-score пяти долей", "все МО", spearman_rho(pc1, rest), len(S))

    # 4. rest по типам
    rest_rows = []
    for t in range(FINAL_K):
        r = rest[labels == t] * 100
        q1, med, q3 = np.percentile(r, [25, 50, 75])
        my[f"rest_med_{t}"] = rec("4. rest по типам", f"тип {t}: медиана rest, %", SHARES, "все МО", med, len(r))
        rec("4. rest по типам", f"тип {t}: нижний квартиль rest, %", SHARES, "все МО", q1, len(r))
        rec("4. rest по типам", f"тип {t}: верхний квартиль rest, %", SHARES, "все МО", q3, len(r))
        rest_rows.append({"тип": str(t), "медиана, %": f"{med:.1f}", "квартиль 1, %": f"{q1:.1f}", "квартиль 3, %": f"{q3:.1f}", "n": str(len(r))})
    eta_rest = eta_levels(rest, labels, region)
    for lev, (v, n, nr) in eta_rest.items():
        rec("4. rest по типам", "η²_H rest по типам", SHARES, lev, v, n, nr)
    my["eta_rest"] = eta_rest["общий"][0]

    # 5. Г2, Г3: доли и alr
    alr = np.log(S / rest[:, None])
    x_g2 = np.log(d["ma"].reindex(ids).to_numpy(float))
    x_g3 = d["ros_by_year"][2023]["wage_rel_region"].to_numpy(float)
    rel_rows = []
    for h, cat, xname, x in [("Г2", "Маркетплейсы", "ln market_access", x_g2), ("Г3", "Продовольствие", "зарплата относительно региона, 2023", x_g3)]:
        j = CATS.index(cat)
        hk = {"Г2": "g2", "Г3": "g3"}[h]                              # ключи контролей
        res = {SHARES: rho_levels(S[:, j], x, region), ALR: rho_levels(alr[:, j], x, region)}
        for var, lv in res.items():
            for lev, (v, n, nr) in lv.items():
                rec("5. связи", f"{h}: ρ({cat}, {xname})", var, lev, v, n, nr)
        short = {L_ALL: "all", L_IN: "in", L_BETWEEN: "btw"}
        for lev in (L_ALL, L_IN, L_BETWEEN):
            my[f"{hk}_{short[lev]}_s"], my[f"{hk}_{short[lev]}_a"] = res[SHARES][lev][0], res[ALR][lev][0]
            vs, ns, nrs = res[SHARES][lev]
            va, na, nra = res[ALR][lev]
            if (ns, nrs) != (na, nra):
                raise SystemExit(f"STOP: {h}, {lev}: разное число МО или регионов для долей и alr")
            rel_rows.append({"связь": f"{h}: ρ({cat}, {xname})", "уровень": lev, "по долям": f2(vs), "по лог-отношению": f2(va),
                             "n МО": str(ns), "n регионов": "—" if nrs is None else str(nrs)})
        my[f"{hk}_btw_nreg"] = res[SHARES][L_BETWEEN][2]

    # 6. η²_H долей и alr
    eta_rows = []
    for j, cat in enumerate(CATS):
        es, ea = eta_levels(S[:, j], labels, region), eta_levels(alr[:, j], labels, region)
        for lev in ("общий", "внутри регионов"):
            rec("6. η²_H по типам", f"η²_H {cat}", SHARES, lev, *es[lev])
            rec("6. η²_H по типам", f"η²_H {cat}", ALR, lev, *ea[lev])
            eta_rows.append({"категория": cat, "уровень": lev, "по долям": f"{es[lev][0]:.3f}", "по лог-отношению": f"{ea[lev][0]:.3f}",
                             "n МО": str(es[lev][1]), "n регионов": "—" if es[lev][2] is None else str(es[lev][2])})
        my[f"eta_{cat}_s"], my[f"eta_{cat}_a"] = es["общий"][0], ea["общий"][0]

    # сверка
    ctrl_rows, bad = [], []
    for name, key, ctrl, tol in CONTROLS:
        ok = abs(my[key] - ctrl) <= tol + 1e-9
        ctrl_rows.append({"контроль": name, "шаг 25": f"{my[key]:.3f}", "контрольное": f"{ctrl:g}", "допуск": f"{tol:g}",
                          "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})
        if not ok:
            bad.append(ctrl_rows[-1])
    ctrl = pd.DataFrame(ctrl_rows)
    print(ctrl.to_string(index=False))
    if bad:
        raise SystemExit("STOP: расхождения с контрольными значениями (значения не подгонялись):\n" + pd.DataFrame(bad).to_string(index=False))

    # сводная таблица «по долям / по лог-отношению»
    summary = [{"показатель": r["связь"] + f", {r['уровень']}", "по долям": r["по долям"], "по лог-отношению": r["по лог-отношению"]} for r in rel_rows]
    summary += [{"показатель": f"η²_H {r['категория']} по типам, {r['уровень']}", "по долям": r["по долям"], "по лог-отношению": r["по лог-отношению"]}
                for r in eta_rows]

    # текст выводов (шаблоны; числа подставляются)
    enc_ari = [my["ari_a"], my["ari_b"], my["ari_c"]]
    sign_g2 = all(np.sign(my[f"g2_{k}_s"]) == np.sign(my[f"g2_{k}_a"]) for k in ("all", "in", "btw"))
    sign_g3 = all(np.sign(my[f"g3_{k}_s"]) == np.sign(my[f"g3_{k}_a"]) for k in ("all", "in", "btw"))
    eta_drop = sorted(CATS, key=lambda c: my[f"eta_{c}_a"] - my[f"eta_{c}_s"])
    conclusions = [
        f"- Канон воспроизводится (ARI = {ari0:.3f}); при смене seed ARI канона с перезапуском от {my['seed_min']:.3f} до "
        f"{max(seeds):.3f}, медиана {my['seed_med']:.3f} {REF}.",
        f"- При clr-кодировках разбиение на {FINAL_K} типов заметно отличается от канона: ARI от {min(enc_ari):.3f} до "
        f"{max(enc_ari):.3f} {REF}. Типология описывает структуру в z-score пяти долей и в этом смысле зависит от кодировки.",
        f"- Главная ось z-score пяти долей несёт {my['pc1_var'] * 100:.0f}% дисперсии; ρ(PC1, rest) = {my['pc1_rest']:+.2f} {REF}. "
        "Существенная часть различий между МО по пяти долям согласуется с различиями в доле «прочего».",
        f"- Медиана rest по типам — от {min(my[f'rest_med_{t}'] for t in range(FINAL_K)):.1f}% до "
        f"{max(my[f'rest_med_{t}'] for t in range(FINAL_K)):.1f}%; η²_H rest по типам {my['eta_rest']:.3f} (общий уровень), "
        f"{eta_rest['внутри регионов'][0]:.3f} внутри регионов {REF}.",
        f"- Г2: знак ρ по долям и по лог-отношению {'совпадает' if sign_g2 else 'совпадает не на всех уровнях'}; "
        f"на уровне всех МО ρ меняется с {f2(my['g2_all_s'])} до {f2(my['g2_all_a'])}, внутри регионов с {f2(my['g2_in_s'])} "
        f"до {f2(my['g2_in_a'])}, между регионами с {f2(my['g2_btw_s'])} до {f2(my['g2_btw_a'])} {REF}.",
        f"- Г3: знак ρ по долям и по лог-отношению {'совпадает' if sign_g3 else 'совпадает не на всех уровнях'}; "
        f"на уровне всех МО {f2(my['g3_all_s'])} → {f2(my['g3_all_a'])}, внутри регионов {f2(my['g3_in_s'])} → {f2(my['g3_in_a'])} {REF}.",
        f"- η²_H по типам при переходе к лог-отношению сильнее всего снижается для категорий {eta_drop[0]} "
        f"({my[f'eta_{eta_drop[0]}_s']:.3f} → {my[f'eta_{eta_drop[0]}_a']:.3f}) и {eta_drop[1]} "
        f"({my[f'eta_{eta_drop[1]}_s']:.3f} → {my[f'eta_{eta_drop[1]}_a']:.3f}), меньше всего — для {eta_drop[-1]} "
        f"({my[f'eta_{eta_drop[-1]}_s']:.3f} → {my[f'eta_{eta_drop[-1]}_a']:.3f}) {REF}. Различия типов по этим категориям "
        "отчасти совпадают с различиями в доле «прочего»; метки типов при этом взяты из канона, а не перестроены.",
    ]

    lines = [
        "# 25. Устойчивость типологии к композиционной природе долей (k = 6)", "",
        "Сгенерировано `src/25_composition_robustness.py`. Всё **разведочно, вне предрегистрации**: без p-значений, порогов "
        "и вердиктов; замороженные результаты шагов 18–22 не используются и не пересчитываются.", "",
        "## Заморозка", "", *frozen_log, "",
        "## Данные", "",
        f"- Доли пяти категорий за {MONTH}, {len(S)} МО; rest = 1 − сумма пяти долей (от {rest.min() * 100:.1f}% до "
        f"{rest.max() * 100:.1f}%); нулевых долей нет. alr_i = ln(доля_i / rest). Типы — канонические метки "
        "`kmeans_labels_final.parquet`.",
        f"- KMeans: k = {FINAL_K}, n_init = {KMEANS_N_INIT}, random_state = {KMEANS_RANDOM_STATE} (как в шаге 10). "
        f"«Внутри регионов» — нормированный ранг внутри региона, регионы с числом МО с данными < {COMP_MIN_MO_REGION} не участвуют; "
        f"«между регионами» — ρ медиан регионов с не менее {COMP_MIN_MO_REGION} МО. η²_H — `hypothesis_tools.kw_eta2`.",
        "- Г2: доля маркетплейсов ~ ln market_access. Г3: доля продовольствия ~ зарплата относительно региона (Росстат 2023, "
        "МО-годы с anomaly исключены).", "",
        "## 1. Воспроизведение канона и seed", "",
        f"ARI KMeans на z-score пяти долей с каноном: {ari0:.3f}.", "", md(pd.DataFrame(seed_rows)), "",
        "## 2. Альтернативные кодировки", "", "SW — силуэт в пространстве соответствующей кодировки.", "", md(pd.DataFrame(enc_rows)), "",
        "## 3. Главная ось", "",
        f"SVD на z-score пяти долей: PC1 — {my['pc1_var']:.3f} дисперсии; знак выбран так, что нагрузка продовольствия "
        f"положительна. ρ Спирмена(PC1, rest) = {my['pc1_rest']:+.3f}.", "", md(pd.DataFrame(load_rows)), "",
        "## 4. «Прочее» (rest) по типам канона", "", md(pd.DataFrame(rest_rows)), "",
        f"η²_H rest по типам: общий уровень {eta_rest['общий'][0]:.3f} (n = {eta_rest['общий'][1]}), внутри регионов "
        f"{eta_rest['внутри регионов'][0]:.3f} (n = {eta_rest['внутри регионов'][1]}, регионов {eta_rest['внутри регионов'][2]}).", "",
        "## 5. Связи Г2 и Г3: доли и лог-отношение", "", md(pd.DataFrame(rel_rows)), "",
        "## 6. η²_H по типам канона: доли и лог-отношение", "", md(pd.DataFrame(eta_rows)), "",
        "## 7. Сводка: по долям и по лог-отношению", "", md(pd.DataFrame(summary)), "",
        "### Выводы (разведочно)", "", *conclusions, "",
        "## Контрольные сверки", "", md(ctrl), "",
    ]
    text = "\n".join(lines)
    hits = [w for w in FORBIDDEN if re.search(w, text, flags=re.I)]
    if hits:
        raise SystemExit(f"STOP: запрещённые слова в отчёте: {hits}")
    check_frozen("после", frozen_log)
    i = lines.index("## Заморозка") + 2 + len(FROZEN)
    lines[i:i] = frozen_log[len(FROZEN):]
    out = pd.DataFrame(ROWS)
    if out.duplicated(["block", "metric", "variant", "level"]).any():
        raise SystemExit("STOP: дубликаты ключа (block, metric, variant, level)")
    out["n"] = out["n"].astype("Int64")
    out["n_regions"] = out["n_regions"].astype("Int64")
    out.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"записано: {OUT_PATH.name} ({len(out)} строк), {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
