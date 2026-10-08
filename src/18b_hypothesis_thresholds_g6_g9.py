"""Шаг 18b. Пороги для гипотез Г6 и Г9 (предрегистрация).

Пороги фиксируются ДО просмотра результатов: наблюдаемые статистики Г6 и Г9 (доля переходов в
ближайшие типы, значения рядов P_t и M_t, корреляция между ними) здесь не считаются и не печатаются.
Считаются только нулевые распределения, число переходов, число МО, размеры типов и длины рядов.
Ряды Г9 сохраняются в файл (для шага 19), но не выводятся. Замороженные файлы шага 18a не трогаются.

Г6: переход — смена канонического типа МО между соседними месяцами (траектории шага 12 при k = 6).
S = доля переходов, у которых конечный тип входит в N_NEAREST ближайших к исходному (евклидово
расстояние между центроидами z-score-признаков типов канонического разбиения, декабрь 2024).
Нуль: конечный тип каждого перехода случаен среди типов ≠ исходного, с вероятностями по общей частоте
конечных типов.
Г9: P_t — доля МО, сменивших тип между t−1 и t; M_t — максимум по 5 категориям |робастный z| медианного
сдвига сырых долей. Статистика — ρ Спирмена (P, M), ожидаемый знак «+»; нуль — перестановки месяцев и
циклические сдвиги.

Вход:  data/processed/{kmeans_k6_trajectories, kmeans_k6_switches, kmeans_labels_final,
       category_shares}.parquet
Выход: data/processed/hypothesis_thresholds_g6g9.parquet, data/processed/hypothesis_series_g9.parquet,
       data/processed/hypothesis_thresholds_g6g9.sha256, notebooks/18b_hypothesis_thresholds_g6_g9.md
Запуск из корня проекта:  .venv/bin/python src/18b_hypothesis_thresholds_g6_g9.py
"""
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import FINAL_K, HYP_PERM_BATCH, MONTH, N_NEAREST, N_NEAREST_SENS, N_PERM, SEED_G6, SEED_G9
from hypothesis_tools import (Spearman, g6_null, g6_simulate_targets, g6_target_probs, nearest_types,
                              share_to_nearest, spearman_cyclic_null, spearman_rho)
from network_utils import SHARE_COLUMNS, month_shares, standardize_shares

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TRAJ_PATH = PROCESSED_DIR / f"kmeans_k{FINAL_K}_trajectories.parquet"
SWITCHES_PATH = PROCESSED_DIR / f"kmeans_k{FINAL_K}_switches.parquet"
LABELS_FINAL_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
THRESHOLDS_PATH = PROCESSED_DIR / "hypothesis_thresholds_g6g9.parquet"
SERIES_PATH = PROCESSED_DIR / "hypothesis_series_g9.parquet"
SHA_PATH = PROCESSED_DIR / "hypothesis_thresholds_g6g9.sha256"
REPORT_PATH = PROJECT_DIR / "notebooks" / "18b_hypothesis_thresholds_g6_g9.md"

MAX_DEC_MISMATCH = 100          # допустимое число МО, где декабрь траекторий ≠ канонического разбиения
N_SIM_CHECK = 200               # проверка (б) Г6: симуляций из нуля
MAD_SCALE = 1.4826
COLUMNS = ["hypothesis", "variant", "row", "n_nearest", "from_type", "n_transitions", "n_distinct_mo", "n",
           "null_mean", "p95", "p99", "seed", "n_perm"]


def md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(index=False, **kw)


def load_trajectories(log: list) -> pd.DataFrame:
    traj = pd.read_parquet(TRAJ_PATH)
    switches = pd.read_parquet(SWITCHES_PATH)
    for name, df in [(TRAJ_PATH.name, traj), (SWITCHES_PATH.name, switches)]:
        log += [f"`{name}`: {df.shape[0]} строк; колонки {list(df.columns)}", "", md(df.head(5)), ""]
        print(f"===== {name} {df.shape}\n{list(df.columns)}\n{df.head(5).to_string()}")
    final = pd.read_parquet(LABELS_FINAL_PATH).set_index("territory_id")["cluster"]
    wide = traj.pivot(index="territory_id", columns="month", values="cluster")
    if wide.isna().any().any() or wide.shape != (len(final), 24) or set(wide.index) != set(final.index):
        raise SystemExit(f"STOP: траектории не дают полную таблицу МО × 24 месяца: {wide.shape}")
    if not set(np.unique(wide.to_numpy())) <= set(final.unique()):
        raise SystemExit("STOP: в траекториях есть типы, которых нет в kmeans_labels_final")
    mismatch = int((wide[MONTH] != final.reindex(wide.index)).sum())
    log += ["Таблица «МО × месяц → канонический тип» — колонка `cluster` траекторий (в шаге 12: нумерация "
            f"якоря {MONTH}, для k = {FINAL_K} та же, что в kmeans_labels_final; `cluster_raw` — номера KMeans месяца).",
            f"- МО, у которых тип {MONTH} в траекториях ≠ kmeans_labels_final: **{mismatch}** (допустимо ≤ "
            f"{MAX_DEC_MISMATCH}: траектории построены по официальному месячному разбиению шага 12a, "
            "каноническое — по минимальной inertia шага 10c). Для переходов используются траектории во всех месяцах.", ""]
    print(f"МО с различающимся типом {MONTH} (траектории vs kmeans_labels_final): {mismatch}")
    if mismatch > MAX_DEC_MISMATCH:
        raise SystemExit(f"STOP: {mismatch} МО с различающимся типом {MONTH} (> {MAX_DEC_MISMATCH})")
    return wide.astype(int)


def transitions(wide: pd.DataFrame) -> pd.DataFrame:
    months = list(wide.columns)
    rows = []
    for prev, cur in zip(months[:-1], months[1:]):
        ch = wide[cur] != wide[prev]
        rows.append(pd.DataFrame({"territory_id": wide.index[ch], "month": cur,
                                  "from_type": wide.loc[ch, prev].to_numpy(), "to_type": wide.loc[ch, cur].to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def centroid_distances(final: pd.Series) -> np.ndarray:
    sh = month_shares(pd.read_parquet(SHARES_PATH), MONTH).set_index("territory_id").loc[final.index]
    z = standardize_shares(sh[SHARE_COLUMNS].to_numpy())[0]
    types = sorted(final.unique())
    cent = np.array([z[final.to_numpy() == t].mean(axis=0) for t in types])
    return np.sqrt(((cent[:, None, :] - cent[None, :, :]) ** 2).sum(axis=-1))


def g6(wide: pd.DataFrame, log: list, check_log: list) -> list:
    final = pd.read_parquet(LABELS_FINAL_PATH).set_index("territory_id")["cluster"].sort_index()
    k = int(final.max()) + 1
    tr = transitions(wide)
    frm, to = tr["from_type"].to_numpy(), tr["to_type"].to_numpy()
    dist = centroid_distances(final)
    variants = [(N_NEAREST, "основной")] + [(n, "чувствительность") for n in N_NEAREST_SENS]

    sizes = pd.DataFrame({"тип": range(k), f"МО в каноническом разбиении ({MONTH})": np.bincount(final, minlength=k),
                          "переходов из типа": np.bincount(frm, minlength=k)})
    dist_tab = pd.DataFrame(dist, columns=[f"тип {j}" for j in range(k)]).round(4)
    dist_tab.insert(0, "тип", range(k))
    near_tab = pd.DataFrame({"тип": range(k), **{f"N = {n}": [sorted(nearest_types(dist, n)[i]) for i in range(k)]
                                                 for n in sorted({N_NEAREST, *N_NEAREST_SENS})}})
    ties = [i for i in range(k) for n in sorted({N_NEAREST, *N_NEAREST_SENS})
            if np.isclose(np.sort(dist[i][np.arange(k) != i])[n - 1], np.sort(dist[i][np.arange(k) != i])[n])]
    log += ["## Г6. Переходы между типами", "",
            f"- переходов (МО-месяцев со сменой типа, 23 пары месяцев): **{len(tr)}**; различных МО: "
            f"**{tr['territory_id'].nunique()}** из {wide.shape[0]}", "", md(sizes), "",
            f"Евклидовы расстояния между центроидами типов (z-score 5 долей, {MONTH}, kmeans_labels_final):", "",
            md(dist_tab), "", "Ближайшие типы:", "", md(near_tab), "",
            f"- совпадающие расстояния на границе списка ближайших: {sorted(set(ties)) or 'нет'}", ""]
    print(sizes.to_string(index=False)); print(dist_tab.to_string(index=False)); print(near_tab.to_string(index=False))

    probs = g6_target_probs(to, k)
    rows, base_rows = [], []
    for n, variant in variants:
        near = nearest_types(dist, n)
        seed = SEED_G6 + n
        null, q, n_from = g6_null(frm, probs, near, N_PERM, seed)
        rows.append({"hypothesis": "Г6", "variant": variant, "row": "итог", "n_nearest": n,
                     "n_transitions": len(tr), "n_distinct_mo": int(tr["territory_id"].nunique()), "n": len(tr),
                     "null_mean": float(null.mean()), "p95": float(np.quantile(null, 0.95)),
                     "p99": float(np.quantile(null, 0.99)), "seed": seed, "n_perm": N_PERM})
        for i in range(k):
            rows.append({"hypothesis": "Г6", "variant": variant, "row": "по исходному типу", "n_nearest": n,
                         "from_type": i, "n_transitions": int(n_from[i]),
                         "n_distinct_mo": int(tr.loc[tr["from_type"] == i, "territory_id"].nunique()),
                         "n": int(n_from[i]), "null_mean": float(q[i]), "seed": seed, "n_perm": N_PERM})
        # проверки реализации
        first_near = {i: min(near[i], key=lambda j: dist[i, j]) for i in range(k)}
        s_all_near = share_to_nearest(frm, np.array([first_near[f] for f in frm]), near)
        rng = np.random.default_rng(seed + 1000)
        sims = np.array([share_to_nearest(frm, g6_simulate_targets(frm, probs, rng), near) for _ in range(N_SIM_CHECK)])
        se = null.std() / np.sqrt(N_SIM_CHECK)
        ok = s_all_near == 1.0 and abs(sims.mean() - null.mean()) <= 2 * se
        base_rows.append({"N_NEAREST": n, "(а) S при всех конечных = ближайшие": s_all_near,
                          f"(б) среднее S {N_SIM_CHECK} прямых симуляций": round(float(sims.mean()), 6),
                          "null_mean": round(float(null.mean()), 6), "2 SE": round(float(2 * se), 6),
                          "|разность|": round(float(abs(sims.mean() - null.mean())), 6), "проверка": ok})
    check = pd.DataFrame(base_rows)
    check_log += ["### Г6: синтетика", "",
                  f"(а) все конечные типы — ближайшие к исходному, ожидается S = 1; (б) {N_SIM_CHECK} симуляций с прямым "
                  "выбором конечного типа по тем же вероятностям, ожидается среднее S в пределах 2 стандартных "
                  f"ошибок (SE = sd нуля / √{N_SIM_CHECK}) от null_mean биномиальной записи нуля.", "", md(check), ""]
    print(check.to_string(index=False))
    if not check["проверка"].all():
        raise SystemExit("STOP: проверка реализации Г6 не прошла")
    return rows


def g9(wide: pd.DataFrame, log: list, check_log: list) -> list:
    months = list(wide.columns)
    p_t = [(wide[cur] != wide[prev]).mean() for prev, cur in zip(months[:-1], months[1:])]
    sh = pd.read_parquet(SHARES_PATH)
    sh["month"] = sh["date"].dt.strftime("%Y-%m")
    if sorted(sh["month"].unique()) != months:
        raise SystemExit("STOP: месяцы category_shares не совпадают с месяцами траекторий")
    sw = {c: sh.pivot(index="territory_id", columns="month", values=c)[months] for c in SHARE_COLUMNS}
    d = pd.DataFrame({c: [(w[cur] - w[prev]).dropna().median() for prev, cur in zip(months[:-1], months[1:])]
                      for c, w in sw.items()}, index=months[1:])
    n_pairs = {c: int(min((w[cur].notna() & w[prev].notna()).sum() for prev, cur in zip(months[:-1], months[1:])))
               for c, w in sw.items()}
    med = d.median()
    mad = (d - med).abs().median()
    if (mad == 0).any():
        raise SystemExit(f"STOP: MAD = 0 у категорий {mad[mad == 0].index.tolist()}")
    z = (d - med) / (MAD_SCALE * mad)
    m_t = z.abs().max(axis=1)
    series = pd.DataFrame({"month": months[1:], "P": p_t, "M": m_t.to_numpy()})
    series.to_parquet(SERIES_PATH, engine="pyarrow", index=False)

    sp = Spearman(series["P"].to_numpy(float), series["M"].to_numpy(float))
    perm = sp.null(N_PERM, SEED_G9, HYP_PERM_BATCH)
    cyc = spearman_cyclic_null(series["P"].to_numpy(float), series["M"].to_numpy(float))
    rows = [{"hypothesis": "Г9", "variant": "основной", "row": "перестановки месяцев", "n": len(series),
             "null_mean": float(perm.mean()), "p95": float(np.quantile(perm, 0.95)), "p99": float(np.quantile(perm, 0.99)),
             "seed": SEED_G9, "n_perm": N_PERM},
            {"hypothesis": "Г9", "variant": "основной", "row": "циклические сдвиги", "n": len(series),
             "null_mean": float(cyc.mean()), "p95": float(np.quantile(cyc, 0.95)), "p99": float(np.quantile(cyc, 0.99)),
             "n_perm": len(cyc)}]
    rows.append({"hypothesis": "Г9", "variant": "основной", "row": "управляющий порог (больший p95)", "n": len(series),
                 "p95": max(rows[0]["p95"], rows[1]["p95"])})

    rng = np.random.default_rng(SEED_G9 + 1000)
    chk = []
    for i in range(5):
        a, b = rng.integers(0, 8, 23).astype(float), rng.normal(size=23)
        chk.append({"ряд": i + 1, "ρ ours": spearman_rho(a, b), "ρ scipy": spearmanr(a, b).statistic})
    chk = pd.DataFrame(chk)
    chk["|Δ|"] = (chk["ρ ours"] - chk["ρ scipy"]).abs()
    check_log += ["### Г9: ρ против scipy.stats.spearmanr (случайные ряды длины 23, в первом ряду связки)", "",
                  md(chk), "", f"- max |Δ| = {chk['|Δ|'].max():.3g}", ""]
    print(chk.to_string(index=False)); print(f"max |Δ| = {chk['|Δ|'].max():.3g}")
    if chk["|Δ|"].max() >= 1e-9:
        raise SystemExit("STOP: ρ расходится со scipy")
    log += ["## Г9. Переходы и общие сдвиги долей", "",
            f"- длина рядов P_t и M_t: **{len(series)}** месяцев ({series['month'].iloc[0]} … {series['month'].iloc[-1]}); "
            f"пропусков в рядах: P {int(series['P'].isna().sum())}, M {int(series['M'].isna().sum())}",
            f"- МО, по которым считается P_t: {wide.shape[0]} в каждом месяце; минимальное число МО со значением доли "
            f"в обоих месяцах пары: {n_pairs}",
            "- Г9 проверяется через непрерывную долю сменивших тип P_t, а не через бинарный флаг «проблемный "
            "месяц» шага 12.",
            "- D_{c,t} считается по СЫРЫМ долям (без z-score внутри месяца, который убрал бы общий сдвиг).",
            f"- ряды сохранены в `{SERIES_PATH.name}`; их значения в отчёт и вывод не включаются.", ""]
    return rows


def main() -> None:
    t0 = time.time()
    log, check_log = [], []
    wide = load_trajectories(log)
    rows = g6(wide, log, check_log)
    t_g6 = time.time() - t0
    rows += g9(wide, log, check_log)
    thr = pd.DataFrame(rows).reindex(columns=COLUMNS)
    for c in ["n_nearest", "from_type", "n_transitions", "n_distinct_mo", "n", "seed", "n_perm"]:
        thr[c] = thr[c].astype("Int64")
    thr.to_parquet(THRESHOLDS_PATH, engine="pyarrow", index=False)
    shas = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (THRESHOLDS_PATH, SERIES_PATH)}
    SHA_PATH.write_text("".join(f"{h}  {n}\n" for n, h in shas.items()), encoding="utf-8")

    g6_tab = thr[thr["hypothesis"] == "Г6"].drop(columns=["hypothesis", "p95", "p99"]).loc[lambda d: d["row"] != "итог"]
    lines = [
        "# 18b. Пороги для гипотез Г6 и Г9 (предрегистрация)", "",
        "Сгенерировано `src/18b_hypothesis_thresholds_g6_g9.py`. Пороги зафиксированы до просмотра результатов: "
        "наблюдаемые статистики Г6 и Г9 (доля переходов в ближайшие типы, значения рядов P_t и M_t, их корреляция) "
        "здесь не считались и не выводятся.", "",
        "## Заморозка", "", *[f"- sha256 `{n}`: `{h}`" for n, h in shas.items()],
        f"- (оба хэша записаны в `{SHA_PATH.name}`; шаг 19 сверяет их перед работой)", "",
        "## Формулы", "",
        "- Г6: переход — МО, у которого канонический тип в месяце t отличается от типа в t−1 (t = 2023-02 … 2024-12). "
        "S = доля переходов, у которых конечный тип входит в N_NEAREST ближайших к исходному. Ближайшие — по "
        f"евклидову расстоянию между центроидами (средний z-score 5 долей МО типа, {MONTH}, kmeans_labels_final); "
        f"N_NEAREST = {N_NEAREST} — основной вариант, {N_NEAREST_SENS} — чувствительность.",
        f"- Нуль Г6: исходный тип перехода сохраняется, конечный выбирается среди типов ≠ исходного с вероятностями, "
        f"пропорциональными общей частоте конечных типов всех реальных переходов (с нормировкой без исходного). "
        f"{N_PERM} повторов; seed = SEED_G6 + N_NEAREST. Число попаданий в ближайшие среди n_i переходов из типа i в "
        "этом нуле распределено как Binomial(n_i, q_i), q_i — вероятность попасть в ближайшие; так нуль и считается "
        "(совпадение с прямыми симуляциями — в проверке ниже). null_mean по исходному типу = q_i.",
        "- **Оговорка Г6:** переходы одного МО зависимы (одно МО даёт несколько переходов), поэтому порог по "
        "переходам как по независимым наблюдениям оптимистичен; на шаге 19 будет бутстреп по МО.",
        "- Г9: P_t — доля МО, сменивших канонический тип между t−1 и t. D_{c,t} — медиана по МО (со значением в обоих "
        "месяцах) разности сырых долей share_c(t) − share_c(t−1); z_{c,t} = (D_{c,t} − медиана_t) / "
        f"({MAD_SCALE} · MAD_t) по 23 месяцам категории; M_t = max_c |z_{{c,t}}|. Статистика — ρ Спирмена (P, M), "
        "n = 23, ожидаемый знак «+», односторонний порог.",
        f"- Нуль Г9: (а) {N_PERM} перестановок месяцев одного ряда (seed SEED_G9); (б) 22 ненулевых циклических сдвига "
        "одного ряда относительно другого. **У способа (б) всего 22 значения, его p95 и p99 грубые** (интерполяция "
        "между крайними значениями). Управляющий порог — больший из двух p95.", "",
        "## Данные", "", *log,
        "## Пороги", "", "### Г6: итог по вариантам", "",
        md(thr[(thr["hypothesis"] == "Г6") & (thr["row"] == "итог")][
            ["variant", "n_nearest", "n_transitions", "n_distinct_mo", "null_mean", "p95", "p99", "seed", "n_perm"]]), "",
        "### Г6: базовый уровень по исходному типу (null_mean = вероятность попасть в ближайшие)", "",
        md(g6_tab[["variant", "n_nearest", "from_type", "n_transitions", "n_distinct_mo", "null_mean"]]), "",
        "### Г9", "", md(thr[thr["hypothesis"] == "Г9"][["row", "n", "null_mean", "p95", "p99", "seed", "n_perm"]]), "",
        "## Проверка реализации", "", *check_log,
        "## Время", "", f"- Г6: {t_g6:.1f} с; весь шаг: {time.time() - t0:.1f} с", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(thr.to_string(index=False))
    for n, h in shas.items():
        print(f"sha256 {n} {h}")
    print(f"весь шаг: {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
