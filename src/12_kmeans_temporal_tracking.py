"""Шаг 12. KMeans (k из --k, по умолчанию config.FINAL_K) по всем 24 месяцам + согласование кластеров по профилю с декабрём 2024.

1. Для каждого месяца: z-score 5 долей внутри месяца, KMeans k с нуля (центроиды не
   переносятся), параметры KMeans — из шага 10.
2. Согласование (вариант A): центроид кластера = средние ИСХОДНЫЕ доли его МО. 7 центроидов
   месяца сопоставляются НАПРЯМУЮ с 7 каноническими центроидами ANCHOR_MONTH
   (метки k шага 10b, kmeans_labels_k{K}_2024_12) — linear_sum_assignment на евклидовых расстояниях. Без цепочки.
3. Качество каждого сопоставления, траектории МО, метрики стабильности, проблемные месяцы.

Выход: data/processed/kmeans_labels_k{K}/{YYYY-MM}.parquet (territory_id, cluster — исходные
         номера KMeans этого месяца),
       data/processed/kmeans_k{K}_matching.parquet (сопоставление каждого месяца с якорем + качество),
       data/processed/kmeans_k{K}_trajectories.parquet (territory_id, month, cluster_raw, cluster),
       data/processed/kmeans_k{K}_switches.parquet (по МО),
       data/processed/kmeans_k{K}_months.parquet (по месяцам),
       notebooks/12_kmeans_temporal_tracking.md
Запуск из корня проекта:  .venv/bin/python src/12_kmeans_temporal_tracking.py [--k 6]
"""
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from config import FINAL_K
from network_utils import SHARE_COLUMNS, standardize_shares

step10 = importlib.import_module("10_kmeans_k_selection")
step10b = importlib.import_module("10b_kmeans_cluster_profiles")

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
TERRITORIES_PATH = RAW_DIR / "territories.parquet"


def configure(k: int) -> None:
    """Задаёт k и все зависящие от него пути (глобально для модуля)."""
    global K, KMEANS_FINAL_PATH, LABELS_DIR, MATCHING_PATH, TRAJ_PATH, RUNS_SUMMARY_PATH
    global CONF_RAW_PATH, CONFIDENCE_PATH, SWITCHES_PATH, REPORT_PATH, BORDERLINE_PATH, MONTHS_PATH
    K = k
    # якорная нумерация — метки k шага 10b за ANCHOR_MONTH (для FINAL_K — то же, что kmeans_labels_final)
    KMEANS_FINAL_PATH = PROCESSED_DIR / f"kmeans_labels_k{k}_{ANCHOR_MONTH.replace('-', '_')}.parquet"
    LABELS_DIR = PROCESSED_DIR / f"kmeans_labels_k{k}"
    MATCHING_PATH = PROCESSED_DIR / f"kmeans_k{k}_matching.parquet"
    TRAJ_PATH = PROCESSED_DIR / f"kmeans_k{k}_trajectories.parquet"
    RUNS_SUMMARY_PATH = PROCESSED_DIR / f"kmeans_k{k}_runs_summary.parquet"
    CONF_RAW_PATH = PROCESSED_DIR / f"kmeans_k{k}_official_confidence_raw.parquet"
    CONFIDENCE_PATH = PROCESSED_DIR / f"kmeans_k{k}_membership_confidence.parquet"
    SWITCHES_PATH = PROCESSED_DIR / f"kmeans_k{k}_switches.parquet"
    MONTHS_PATH = PROCESSED_DIR / f"kmeans_k{k}_months.parquet"   # по месяцам: проблемные, ARI и т. п.
    BORDERLINE_PATH = PROCESSED_DIR / f"kmeans_k{k}_borderline.parquet"
    REPORT_PATH = PROJECT_DIR / "notebooks" / f"12_kmeans_temporal_tracking_k{k}.md"


RANDOM_STATE = step10.RANDOM_STATE
N_INIT = step10.N_INIT
from config import MONTH as ANCHOR_MONTH
# Уверенность сопоставления: margin = (расстояние до ближайшего конкурента) / (расстояние до
# сопоставленного). Конкурент — ближайший из остальных центроидов и со стороны кластера месяца,
# и со стороны канонического кластера.
from config import CONFIDENT_MARGIN  # уверенно: взаимно ближайшие и margin ≥ 1.5
from config import AMBIGUOUS_MARGIN  # неоднозначно: не взаимно ближайшие или margin < 1.2
from config import CONFIDENT_SHARE_MIN  # кластер «слабо устойчив во времени», если уверенных месяцев < 50%
from config import PROBLEM_MONTH_MIN_AMBIGUOUS  # проблемный месяц: ≥ 2 неоднозначных сопоставлений ...
from config import PROBLEM_SWITCH_IQR  # ... или доля смен выше Q3 + 1.5·IQR
from config import PERIOD_WINDOW
TOP_N_MIGRANTS = 40
# смена кластера «устойчивая», если новый кластер держится ≥ STABLE_MIN_MONTHS месяцев подряд
from config import STABLE_MIN_MONTHS
# уверенность принадлежности МО (шаг 12a): доля из 100 запусков KMeans в том же кластере
from config import CONF_THRESHOLD
from config import CONF_THRESHOLDS
from config import BORDERLINE_CONF  # «пограничное» МО: уверенность < 0.5 ...
from config import BORDERLINE_MIN_SHARE  # ... более чем в половине месяцев
# «Пограничные во времени»: доля месяцев в основном кластере ниже порога
from config import TEMPORAL_THRESHOLDS
from config import TEMPORAL_THRESHOLD
from config import SWING_TOP2_MIN  # «качающиеся»: ≥ 80% месяцев в двух самых частых кластерах
from config import SWING_TOP2_ALT
N_BORDER_SHOW = 50
from config import MIN_CONF_MONTHS_IN_WINDOW  # (в) с фильтром: мода окна — по месяцам с уверенностью ≥ порога,
                               # окно учитывается, если таких месяцев не меньше 3 из 6
step12a = importlib.import_module("12a_kmeans_runs")
# Пространство центроидов для сопоставления:
#   "raw"          — исходные доли (как задано);
#   "raw_centered" — исходные доли минус среднее месяца (убирает общий тренд, масштаб сохраняется);
#   "zscore"       — z-score внутри месяца (пространство, в котором строится KMeans).
from config import CENTROID_SPACE
SPACES = ["raw", "raw_centered", "zscore"]   # для обоснования выбора в отчёте
# |corr(медианное расстояние до якоря, номер месяца)| выше порога → расстояние определяется
# общим трендом долей, а не различием типов; уверенность нельзя трактовать как свойство типов
from config import TREND_CONFOUND_ABS_CORR


def md_table(df: pd.DataFrame, index: bool = False, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def raw_centroids(shares_raw: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    return np.vstack([shares_raw[labels == c].mean(axis=0) for c in range(k)])


def match_to_anchor(anchor_c: np.ndarray, month_c: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Биекция кластер месяца -> канонический номер + качество каждой пары."""
    D = np.sqrt(((anchor_c[:, None, :] - month_c[None, :, :]) ** 2).sum(axis=-1))  # [канон, месяц]
    rows, cols = linear_sum_assignment(D)
    med_all = np.median(D)
    q = []
    for c, j in zip(rows, cols):
        d = D[c, j]
        comp_row = np.delete(D[c], j).min()          # канонический c ↔ другие кластеры месяца
        comp_col = np.delete(D[:, j], c).min()       # кластер месяца j ↔ другие канонические
        competitor = min(comp_row, comp_col)
        mutual = D[c].argmin() == j and D[:, j].argmin() == c
        margin = competitor / d if d > 0 else np.inf
        rank = int((D < d).sum()) + 1                # ранг среди всех 7×7 пар
        level = ("уверенно" if mutual and margin >= CONFIDENT_MARGIN else
                 "неоднозначно" if (not mutual) or margin < AMBIGUOUS_MARGIN else "умеренно")
        q.append({"cluster": int(c), "исходный номер": int(j), "расстояние": d,
                  "расстояние / медиана 7×7": d / med_all, "ранг среди 49 пар": rank,
                  "взаимно ближайшие": bool(mutual), "margin": margin, "уверенность": level})
    return {int(j): int(c) for c, j in zip(rows, cols)}, pd.DataFrame(q)


def to_space(X: np.ndarray, space: str = CENTROID_SPACE) -> np.ndarray:
    if space == "raw":
        return X
    if space == "raw_centered":
        return X - X.mean(axis=0)
    if space == "zscore":
        return standardize_shares(X)[0]
    raise SystemExit(f"STOP: неизвестное пространство центроидов {space}")


def space_comparison(S: dict, labels: dict, final: np.ndarray, months: list) -> pd.DataFrame:
    """Число уверенных сопоставлений (из месяцев без якоря) по кластерам для каждого пространства."""
    out = {}
    for space in SPACES:
        ac = raw_centroids(to_space(S[ANCHOR_MONTH], space), final, K)
        qs = [match_to_anchor(ac, raw_centroids(to_space(S[m], space), labels[m], K))[1]
              for m in months if m != ANCHOR_MONTH]
        q = pd.concat(qs)
        med = [np.median(x["расстояние"]) for x in qs]
        out[space] = (q[q["уверенность"] == "уверенно"].groupby("cluster").size()
                      .reindex(range(K), fill_value=0))
        out[space]["corr(расстояние, месяц)"] = np.corrcoef(np.arange(len(med)), med)[0, 1]
    return pd.DataFrame(out)


configure(FINAL_K)
MONTH_NAMES = {1: "январь", 2: "февраль", 3: "март", 4: "апрель", 5: "май", 6: "июнь", 7: "июль",
               8: "август", 9: "сентябрь", 10: "октябрь", 11: "ноябрь", 12: "декабрь"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=FINAL_K)
    configure(ap.parse_args().k)
    shares = pd.read_parquet(SHARES_PATH, engine="pyarrow")
    terr = pd.read_parquet(TERRITORIES_PATH).set_index("territory_id")
    months = [f"{d:%Y-%m}" for d in sorted(shares["date"].unique())]
    ids = np.sort(shares["territory_id"].unique())

    # 1. Официальные разбиения месяцев (шаг 12a: min inertia из 100 запусков KMeans)
    raw_lab, S, Z = {}, {}, {}
    for m in months:
        d = shares[shares["date"] == pd.Timestamp(m)].set_index("territory_id").loc[ids]
        S[m] = d[SHARE_COLUMNS].to_numpy()
        Z[m], _ = standardize_shares(S[m])
        lab = pd.read_parquet(LABELS_DIR / f"{m}.parquet").set_index("territory_id")["cluster"]
        raw_lab[m] = lab.loc[ids].to_numpy()
    runs = pd.read_parquet(RUNS_SUMMARY_PATH)

    # Якорь: официальное разбиение ANCHOR_MONTH в нумерации меток k шага 10b
    canonical = pd.read_parquet(KMEANS_FINAL_PATH).set_index("territory_id")["cluster"].loc[ids].to_numpy()
    final = step12a.overlap_relabel(canonical, raw_lab[ANCHOR_MONTH], K)
    anchor_vs_canonical = {"ARI": adjusted_rand_score(canonical, final),
                           "МО с другим кластером": int((canonical != final).sum())}
    anchor_c = raw_centroids(to_space(S[ANCHOR_MONTH]), final, K)

    # 2–3. Прямое сопоставление каждого месяца с якорем
    aligned, qual = {}, []
    for m in months:
        mapping, q = match_to_anchor(anchor_c, raw_centroids(to_space(S[m]), raw_lab[m], K))
        aligned[m] = np.vectorize(mapping.get)(raw_lab[m])
        q.insert(0, "месяц", m)
        q["МО"] = [int((aligned[m] == c).sum()) for c in q["cluster"]]
        qual.append(q)
    qual = pd.concat(qual, ignore_index=True).sort_values(["месяц", "cluster"], ignore_index=True)
    qual.to_parquet(MATCHING_PATH, engine="pyarrow", index=False)
    if not (aligned[ANCHOR_MONTH] == final).all():
        raise SystemExit("STOP: сопоставление якоря с самим собой не тождественно")

    # 4. Траектории
    A = np.column_stack([aligned[m] for m in months])
    Rw = np.column_stack([raw_lab[m] for m in months])
    pd.DataFrame({"territory_id": np.repeat(ids, len(months)), "month": np.tile(months, len(ids)),
                  "cluster_raw": Rw.ravel(), "cluster": A.ravel()}).to_parquet(
        TRAJ_PATH, engine="pyarrow", index=False)

    # Уверенность принадлежности в согласованной нумерации
    conf_raw = pd.read_parquet(CONF_RAW_PATH)
    C = np.column_stack([conf_raw[conf_raw["month"] == m].set_index("territory_id")
                         .loc[ids, "confidence"].to_numpy() for m in months])
    chk = np.column_stack([conf_raw[conf_raw["month"] == m].set_index("territory_id")
                           .loc[ids, "cluster_raw"].to_numpy() for m in months])
    if not (chk == Rw).all():
        raise SystemExit("STOP: метки в confidence не совпадают с официальными разбиениями")
    pd.DataFrame({"territory_id": np.repeat(ids, len(months)), "month": np.tile(months, len(ids)),
                  "cluster": A.ravel(), "confidence": C.ravel(), "cluster_raw": Rw.ravel()}
                 ).to_parquet(CONFIDENCE_PATH, engine="pyarrow", index=False)

    # 5. Метрики стабильности
    n = len(ids)
    n_switch = (A[:, 1:] != A[:, :-1]).sum(axis=1)
    flicker = ((A[:, :-2] != A[:, 1:-1]) & (A[:, :-2] == A[:, 2:])).sum(axis=1)
    share_main = np.array([np.bincount(r, minlength=K).max() / len(r) for r in A])
    # Классификация смен: переход в столбец j (месяц j) устойчив, если A[:, j..j+L-1] одинаковы;
    # для последних L-1 месяцев проверить нельзя — «не определено» (цензурировано)
    changed = np.zeros_like(A, dtype=bool)
    changed[:, 1:] = A[:, 1:] != A[:, :-1]
    L = STABLE_MIN_MONTHS
    holds = np.zeros_like(A, dtype=bool)
    for j in range(1, A.shape[1] - L + 1):
        holds[:, j] = (A[:, j:j + L] == A[:, [j]]).all(axis=1)
    censored = np.zeros_like(A, dtype=bool)
    censored[:, A.shape[1] - L + 1:] = changed[:, A.shape[1] - L + 1:]
    stable_tr = changed & holds
    flick_tr = changed & ~holds & ~censored
    n_stable, n_flick, n_cens = stable_tr.sum(1), flick_tr.sum(1), censored.sum(1)
    # Переход в НОВЫЙ кластер — целевой кластер МО не посещало ни в одном из предыдущих месяцев
    new_target = np.zeros_like(A, dtype=bool)
    for i, j in zip(*np.nonzero(changed)):
        new_target[i, j] = A[i, j] not in A[i, :j]
    stable_back = stable_tr & ~new_target        # устойчивая смена в уже посещённый кластер
    # (а) новый кластер без учёта длительности
    n_new_any = new_target.sum(1)
    # (б) содержательный переход: новый кластер И держится ≥ STABLE_MIN_MONTHS мес. подряд
    meaningful = new_target & holds
    n_meaningful = meaningful.sum(1)
    n_new_censored = (new_target & censored).sum(1)   # новый кластер в последнем месяце — не проверить
    early = np.array([np.bincount(r[:PERIOD_WINDOW], minlength=K).argmax() for r in A])
    late = np.array([np.bincount(r[-PERIOD_WINDOW:], minlength=K).argmax() for r in A])
    sw = pd.DataFrame({
        "territory_id": ids, "смен кластера": n_switch,
        "разных кластеров": [len(set(r)) for r in A], "возвратов A→B→A": flicker,
        "содержательных переходов": n_meaningful, "переходов в новый кластер (любой длит.)": n_new_any,
        "устойчивых смен": n_stable,
        "мерцаний": n_flick, "смен не определено": n_cens,
        "доля месяцев в основном кластере": share_main,
        f"мода первых {PERIOD_WINDOW} мес.": early, f"мода последних {PERIOD_WINDOW} мес.": late,
        f"кластер {ANCHOR_MONTH}": A[:, -1], "траектория": ["".join(map(str, r)) for r in A],
    }).join(terr[["name", "region_name"]], on="territory_id")
    sw["чистая миграция"] = early != late
    sw.to_parquet(SWITCHES_PATH, engine="pyarrow", index=False)

    stab = pd.DataFrame([
        ("ни разу не меняли кластер", int((n_switch == 0).sum())),
        ("ровно 1 смена", int((n_switch == 1).sum())),
        ("2–3 смены", int(((n_switch >= 2) & (n_switch <= 3)).sum())),
        ("4–9 смен", int(((n_switch >= 4) & (n_switch <= 9)).sum())),
        ("10 и более смен", int((n_switch >= 10).sum())),
    ], columns=["группа", "МО"])
    stab["доля"] = stab["МО"] / n
    stab_st = pd.DataFrame([
        ("ни одной устойчивой смены", int((n_stable == 0).sum())),
        ("1 устойчивая смена", int((n_stable == 1).sum())),
        ("2–3 устойчивые смены", int(((n_stable >= 2) & (n_stable <= 3)).sum())),
        ("4–6 устойчивых смен", int(((n_stable >= 4) & (n_stable <= 6)).sum())),
        ("7 и более устойчивых смен", int((n_stable >= 7).sum())),
    ], columns=["группа", "МО"])
    stab_st["доля"] = stab_st["МО"] / n
    tr_tot = pd.DataFrame({"тип смены": ["устойчивая (≥ %d мес.)" % L, "мерцание",
                                         "не определено (последний месяц)"],
                           "смен": [int(n_stable.sum()), int(n_flick.sum()), int(n_cens.sum())]})
    tr_tot["доля"] = tr_tot["смен"] / tr_tot["смен"].sum()
    share_back = stable_back.sum() / max(n_stable.sum(), 1)
    only_flicker = int(((n_stable == 0) & (n_flick > 0)).sum())

    # Жаккар одноимённых кластеров соседних месяцев
    jac_rows = []
    for a, b in zip(months[:-1], months[1:]):
        for c in range(K):
            x, y = aligned[a] == c, aligned[b] == c
            union = (x | y).sum()
            jac_rows.append({"переход": f"{a} → {b}", "cluster": c,
                             "jaccard": (x & y).sum() / union if union else np.nan})
    jac = pd.DataFrame(jac_rows)
    jac_pivot = jac.pivot(index="переход", columns="cluster", values="jaccard")
    jac_pivot["среднее"] = jac_pivot.mean(axis=1)

    # 6. Уверенность по кластерам (23 месяца без якоря)
    n_other = len(months) - 1
    q23 = qual[qual["месяц"] != ANCHOR_MONTH]
    conf = (q23.pivot_table(index="cluster", columns="уверенность", values="месяц",
                            aggfunc="count", fill_value=0)
            .reindex(columns=["уверенно", "умеренно", "неоднозначно"], fill_value=0))
    conf["доля уверенных"] = conf["уверенно"] / n_other
    conf["margin медиана"] = q23.groupby("cluster")["margin"].median()
    anchor_sig = pd.Series({c: step10b.describe_profile(
        pd.Series(Z[ANCHOR_MONTH][final == c].mean(axis=0), index=SHARE_COLUMNS)) for c in range(K)})
    conf.insert(0, f"профиль ({ANCHOR_MONTH})", anchor_sig)
    Da = np.sqrt(((anchor_c[:, None] - anchor_c[None]) ** 2).sum(-1))
    np.fill_diagonal(Da, np.inf)
    conf["ближайший канонический сосед"] = [int(Da[c].argmin()) for c in conf.index]
    weak = conf[conf["доля уверенных"] < CONFIDENT_SHARE_MIN]
    conf_month = qual.pivot(index="месяц", columns="cluster", values="уверенность")

    # Проблемные месяцы
    month_rows = []
    for i, m in enumerate(months):
        qm = qual[qual["месяц"] == m]
        month_rows.append({
            "месяц": m,
            "неоднозначных": int((qm["уверенность"] == "неоднозначно").sum()),
            "уверенных": int((qm["уверенность"] == "уверенно").sum()),
            "расстояние медиана": qm["расстояние"].median(),
            "доля МО сменивших кластер vs пред. мес.":
                float((A[:, i] != A[:, i - 1]).mean()) if i else np.nan,
            "ARI с пред. мес. (разбиения)":
                adjusted_rand_score(raw_lab[months[i - 1]], raw_lab[m]) if i else np.nan,
        })
    mt = pd.DataFrame(month_rows)
    sc = mt["доля МО сменивших кластер vs пред. мес."].dropna()
    q1, q3 = sc.quantile([.25, .75])
    sw_thr = q3 + PROBLEM_SWITCH_IQR * (q3 - q1)
    mt["проблемный"] = (mt["неоднозначных"] >= PROBLEM_MONTH_MIN_AMBIGUOUS) | (
        mt["доля МО сменивших кластер vs пред. мес."] > sw_thr)
    problem = mt[mt["проблемный"]]
    mt.to_parquet(MONTHS_PATH, engine="pyarrow", index=False)
    dist_trend = np.corrcoef(np.arange(n_other), mt["расстояние медиана"].iloc[:-1])[0, 1]

    # Сезонность: показатели месяца, не зависящие от якоря, + контроль чувствительности к seed
    ari_prev = mt["ARI с пред. мес. (разбиения)"].to_numpy()
    ari_next = np.append(ari_prev[1:], np.nan)
    seed_ari = runs.groupby("month")["ARI с официальным"].mean().reindex(months).to_numpy()
    mean_shift = [np.nan] + [float(np.abs(S[b].mean(0) - S[a].mean(0)).sum())
                             for a, b in zip(months[:-1], months[1:])]
    seas = pd.DataFrame({
        "месяц": months,
        "год": [m[:4] for m in months],
        "календарный месяц": [int(m[5:]) for m in months],
        "ARI с соседями (среднее)": np.nanmean(np.vstack([ari_prev, ari_next]), axis=0),
        "доля МО сменивших кластер": mt["доля МО сменивших кластер vs пред. мес."].to_numpy(),
        "доля мерцаний среди смен": [flick_tr[:, j].sum() / changed[:, j].sum()
                                     if changed[:, j].sum() else np.nan for j in range(len(months))],
        "неоднозначных сопоставлений": mt["неоднозначных"].to_numpy(),
        f"ARI 100 запусков KMeans с официальным": seed_ari,
        "Σ|Δ средних долей| к пред. мес.": mean_shift,
    })
    PROBLEM_MONTHS = mt.loc[mt["проблемный"], "месяц"].tolist()
    seas["проблемный"] = seas["месяц"].isin(PROBLEM_MONTHS)
    low_thr = seas["ARI с соседями (среднее)"].quantile(1 / 3)
    seas_pairs = []
    for pm in PROBLEM_MONTHS:
        cm = int(pm[5:])
        other = seas[(seas["календарный месяц"] == cm) & (seas["месяц"] != pm)].iloc[0]
        this = seas[seas["месяц"] == pm].iloc[0]
        seas_pairs.append({
            "проблемный месяц": pm, "тот же месяц другого года": other["месяц"],
            "ARI с соседями: проблемный": this["ARI с соседями (среднее)"],
            "ARI с соседями: другой год": other["ARI с соседями (среднее)"],
            "другой год в нижней трети": bool(other["ARI с соседями (среднее)"] <= low_thr),
            "мерцания: проблемный": this["доля мерцаний среди смен"],
            "мерцания: другой год": other["доля мерцаний среди смен"],
            "другой год — якорь": other["месяц"] == ANCHOR_MONTH,
        })
    seas_pairs = pd.DataFrame(seas_pairs)
    yy = seas.pivot(index="календарный месяц", columns="год", values="ARI с соседями (среднее)")
    yy_ok = yy.dropna()
    rho_yy, p_yy = stats.spearmanr(yy_ok.iloc[:, 0], yy_ok.iloc[:, 1])
    yy_shift = seas.pivot(index="календарный месяц", columns="год",
                          values="Σ|Δ средних долей| к пред. мес.")
    rho_shift = yy_shift.corr(method="spearman").iloc[0, 1]
    rho_seed = seas[["ARI с соседями (среднее)", f"ARI 100 запусков KMeans с официальным"]
                    ].corr(method="spearman").iloc[0, 1]
    rho_shift_ari = seas[["ARI с соседями (среднее)", "Σ|Δ средних долей| к пред. мес."]
                         ].corr(method="spearman").iloc[0, 1]
    # проверка по календарным месяцам (август 2023 и 2024 — одна проверка, не две)
    seas_pairs["календарный месяц"] = [int(m[5:]) for m in seas_pairs["проблемный месяц"]]
    cal = (seas_pairs[~seas_pairs["другой год — якорь"]]
           .groupby("календарный месяц")["другой год в нижней трети"].any())
    cal_untestable = sorted(set(seas_pairs.loc[seas_pairs["другой год — якорь"], "календарный месяц"]))
    repeated_months = [m for m, v in cal.items() if v]
    seasonal_confirmed = len(repeated_months) > len(cal) / 2 and rho_yy > 0 and p_yy < 0.05
    seed_prob = seas.loc[seas["проблемный"], f"ARI 100 запусков KMeans с официальным"]
    seed_rest = seas.loc[~seas["проблемный"], f"ARI 100 запусков KMeans с официальным"]
    seed_col = "ARI 100 запусков KMeans с официальным"
    non_periodic = [pm for pm in PROBLEM_MONTHS if int(pm[5:]) not in repeated_months
                    and int(pm[5:]) not in cal_untestable]

    migrants = sw.sort_values(["содержательных переходов", "мерцаний", "устойчивых смен"],
                              ascending=[False, True, False]).head(TOP_N_MIGRANTS)
    # сравнение трёх критериев
    def crit_row(name: str, per_mo: np.ndarray | None, has: np.ndarray, note: str) -> dict:
        return {"критерий": name,
                "событий всего": int(per_mo.sum()) if per_mo is not None else np.nan,
                "на МО, среднее": per_mo.mean() if per_mo is not None else np.nan,
                "МО хотя бы с одним": int(has.sum()), "доля МО": has.mean(), "что отсекает": note}
    crit = pd.DataFrame([
        crit_row("(а) переход в ранее не посещённый кластер, любой длительности", n_new_any,
                 n_new_any > 0, "возвраты в уже посещённые кластеры"),
        crit_row(f"(б) содержательный: новый кластер И держится ≥ {L} мес.", n_meaningful,
                 n_meaningful > 0, "возвраты + одномесячные выбросы в новый кластер"),
        crit_row(f"(в) мода первых {PERIOD_WINDOW} мес. ≠ мода последних {PERIOD_WINDOW} мес.", None,
                 early != late, "всё, что не меняет доминирующий тип окна"),
    ])
    new_one_month = int((new_target & ~holds & ~censored).sum())
    per_month = meaningful[:, PERIOD_WINDOW + 1:].sum(axis=0)   # всплеск — вне стартового окна
    peak_j = PERIOD_WINDOW + 1 + int(per_month.argmax())
    peak_month, peak_n = months[peak_j], int(meaningful[:, peak_j].sum())
    tgt = np.bincount(A[meaningful[:, peak_j], peak_j], minlength=K)
    peak_top, peak_top_n = int(tgt.argmax()), int(tgt.max())
    net = sw[sw["чистая миграция"]]
    flows = (net.groupby([f"мода первых {PERIOD_WINDOW} мес.", f"мода последних {PERIOD_WINDOW} мес."])
             .size().rename("МО").reset_index().sort_values("МО", ascending=False))
    flows.columns = ["из кластера", "в кластер", "МО"]
    main_cl = np.array([np.bincount(r, minlength=K).argmax() for r in A])

    # --- Фильтр по уверенности принадлежности ---
    def window_mode(block: np.ndarray, okb: np.ndarray) -> np.ndarray:
        out = np.full(block.shape[0], -1)
        for i in range(block.shape[0]):
            v = block[i][okb[i]]
            if len(v) >= MIN_CONF_MONTHS_IN_WINDOW:
                out[i] = np.bincount(v, minlength=K).argmax()
        return out

    def filtered_metrics(thr: float | None) -> dict:
        ok = np.ones_like(C, dtype=bool) if thr is None else C >= thr
        pair_ok = np.zeros_like(ok)
        pair_ok[:, 1:] = ok[:, 1:] & ok[:, :-1]
        a_ev = new_target & pair_ok
        b_ev = meaningful & pair_ok
        e = window_mode(A[:, :PERIOD_WINDOW], ok[:, :PERIOD_WINDOW])
        l_ = window_mode(A[:, -PERIOD_WINDOW:], ok[:, -PERIOD_WINDOW:])
        defined = (e >= 0) & (l_ >= 0)
        return {"порог": "без фильтра" if thr is None else f"{thr:.1f}",
                "доля МО-месяцев ≥ порога": ok.mean(),
                "(а) событий": int(a_ev.sum()), "(а) доля МО": (a_ev.sum(1) > 0).mean(),
                "(б) событий": int(b_ev.sum()), "(б) доля МО": (b_ev.sum(1) > 0).mean(),
                "(в) МО": int((defined & (e != l_)).sum()),
                "(в) доля МО": (defined & (e != l_)).mean(),
                "(в) не определено (мало уверенных мес.)": int((~defined).sum()),
                "_b_per_mo": b_ev.sum(1), "_a_per_mo": a_ev.sum(1)}

    sens_raw = [filtered_metrics(None)] + [filtered_metrics(t) for t in CONF_THRESHOLDS]
    sens = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in sens_raw])
    main_f = sens_raw[1 + CONF_THRESHOLDS.index(CONF_THRESHOLD)]
    sw[f"содержательных переходов (conf ≥ {CONF_THRESHOLD})"] = main_f["_b_per_mo"]
    sw["уверенность средняя"] = C.mean(1)
    sw[f"доля месяцев с уверенностью < {BORDERLINE_CONF}"] = (C < BORDERLINE_CONF).mean(1)
    sw.to_parquet(SWITCHES_PATH, engine="pyarrow", index=False)
    migrants_f = sw.sort_values([f"содержательных переходов (conf ≥ {CONF_THRESHOLD})",
                                 "содержательных переходов", "мерцаний"],
                                ascending=[False, False, True]).head(TOP_N_MIGRANTS)

    # Уверенность по месяцам и кластерам
    conf_month_tab = pd.DataFrame({"месяц": months, "медиана": np.median(C, 0), "среднее": C.mean(0),
                                   f"доля < {BORDERLINE_CONF}": (C < BORDERLINE_CONF).mean(0),
                                   f"доля ≥ {CONF_THRESHOLD}": (C >= CONF_THRESHOLD).mean(0)})
    conf_cluster_tab = (pd.DataFrame({"cluster": A.ravel(), "c": C.ravel()}).groupby("cluster")["c"]
                        .agg(среднее="mean", **{f"доля < {BORDERLINE_CONF}":
                                                lambda x: (x < BORDERLINE_CONF).mean()}))
    conf_cluster_tab.insert(0, "профиль", anchor_sig)

    # Пограничные МО
    low_share = (C < BORDERLINE_CONF).mean(1)
    border_mask = low_share > BORDERLINE_MIN_SHARE

    def top_two(row: np.ndarray) -> str:
        bc = np.bincount(row, minlength=K)
        o = np.argsort(-bc)[:2]
        return f"{o[0]} ({bc[o[0]] / len(row):.0%}) / {o[1]} ({bc[o[1]] / len(row):.0%})"
    border = sw[border_mask].copy()
    border["основные кластеры (доля мес.)"] = [top_two(r) for r in A[border_mask]]
    border["пара кластеров"] = ["–".join(map(str, sorted(np.argsort(-np.bincount(r, minlength=K))[:2])))
                                for r in A[border_mask]]
    border = border.sort_values(f"доля месяцев с уверенностью < {BORDERLINE_CONF}", ascending=False)
    near_border = int(((low_share > 0.25) & ~border_mask).sum())
    low_share_bins = pd.cut(pd.Series(low_share), [-0.01, 0, .05, .1, .2, .3, 1],
                            labels=["0", "≤5%", "5–10%", "10–20%", "20–30%", ">30%"]
                            ).value_counts().sort_index()
    temporal_hybrid, temporal_hybrid60 = int((share_main < .5).sum()), int((share_main < .6).sum())

    # Пограничные во времени: подтипы «качающиеся» (2 кластера) и «рассеянные» (3+)
    counts = np.vstack([np.bincount(r, minlength=K) for r in A])
    order = np.argsort(-counts, axis=1)
    top2_share = (np.take_along_axis(counts, order[:, :2], 1).sum(1)) / A.shape[1]
    tb = sw[["territory_id", "name", "region_name", "доля месяцев в основном кластере",
             "смен кластера", "разных кластеров", "уверенность средняя", "траектория"]].copy()
    tb["доля месяцев в топ-2 кластерах"] = top2_share
    tb["основной кластер"] = order[:, 0]
    tb["второй кластер"] = order[:, 1]
    tb["пара кластеров"] = [f"{min(a, b)}–{max(a, b)}" for a, b in order[:, :2]]
    tb["кластеры (доля мес.)"] = [", ".join(f"{c}: {counts[i, c] / A.shape[1]:.0%}"
                                            for c in order[i] if counts[i, c])
                                  for i in range(len(ids))]

    def subtype(thr: float, top2_min: float) -> pd.Series:
        return pd.Series(np.where(share_main >= thr, "",
                                  np.where(top2_share >= top2_min, "качающееся", "рассеянное")),
                         index=tb.index)
    for thr in TEMPORAL_THRESHOLDS:
        tb[f"подтип (порог {thr:.1f})"] = subtype(thr, SWING_TOP2_MIN)
    tb.to_parquet(BORDERLINE_PATH, engine="pyarrow", index=False)
    tsens = pd.DataFrame([{"порог доли основного кластера": f"< {thr:.0%}",
                           "пограничных": int((share_main < thr).sum()),
                           "доля МО": (share_main < thr).mean(),
                           "качающиеся": int((subtype(thr, SWING_TOP2_MIN) == "качающееся").sum()),
                           "рассеянные": int((subtype(thr, SWING_TOP2_MIN) == "рассеянное").sum())}
                          for thr in TEMPORAL_THRESHOLDS])
    tsens_top2 = pd.DataFrame([{"порог топ-2 для «качающихся»": f"≥ {t2:.0%}",
                                "качающиеся": int((subtype(TEMPORAL_THRESHOLD, t2) == "качающееся").sum()),
                                "рассеянные": int((subtype(TEMPORAL_THRESHOLD, t2) == "рассеянное").sum())}
                               for t2 in SWING_TOP2_ALT])
    main_sub = tb[f"подтип (порог {TEMPORAL_THRESHOLD:.1f})"]
    swing = tb[main_sub == "качающееся"].sort_values("доля месяцев в основном кластере")
    scatter = tb[main_sub == "рассеянное"].sort_values("доля месяцев в топ-2 кластерах")
    share_hist = pd.cut(pd.Series(share_main), [0, .4, .5, .6, .7, .8, .9, 1.0],
                        include_lowest=True).value_counts().sort_index()

    trend_confounded = abs(dist_trend) > TREND_CONFOUND_ABS_CORR
    shift = S[months[-1]].mean(axis=0) - S[months[0]].mean(axis=0)
    finding = []
    if trend_confounded:
        finding = [
            "> **ВНИМАНИЕ: уверенность сопоставления в этом варианте определяется общим трендом "
            "долей, а не различием типов — не интерпретировать низкую уверенность как свойство "
            "кластеров.**",
            ">",
            f"> - Медианное расстояние до якоря почти линейно падает к {ANCHOR_MONTH} "
            f"(корреляция с номером месяца {dist_trend:+.2f}).",
            "> - Средние доли за период сдвинулись у всех МО: "
            + ", ".join(f"{c.replace('share_', '')} {v:+.3f}" for c, v in zip(SHARE_COLUMNS, shift))
            + " — сдвиг маркетплейсов больше типичных различий между кластерами, поэтому "
            f"центроиды ранних месяцев в пространстве «{CENTROID_SPACE}» сдвинуты целиком.",
            "> - Пространства без этого эффекта: `CENTROID_SPACE = 'raw_centered'` "
            "(доли минус среднее месяца) или `'zscore'` (пространство KMeans).",
            "",
        ]
    elif len(weak):
        finding = [
            "### Находка: слабая временная устойчивость отдельных типов",
            "",
            f"Кластеры **{', '.join(map(str, weak.index))}** уверенно сопоставляются с профилем "
            f"{ANCHOR_MONTH} менее чем в {CONFIDENT_SHARE_MIN:.0%} месяцев ("
            + ", ".join(f"{c}: {int(r['уверенно'])} из {n_other}" for c, r in weak.iterrows())
            + "). Это свойство самих типов, а не метода сопоставления: их профили близки к "
            "соседним кластерам (см. «ближайший канонический сосед»), поэтому независимый KMeans "
            "в разные месяцы проводит границу между ними по-разному. Как самостоятельные "
            "устойчивые во времени типы их интерпретировать нельзя; кластеры с высокой долей "
            "уверенных месяцев — устойчивые типы.",
            "",
        ]

    comp_spaces = space_comparison(S, raw_lab, final, months)
    choice = [
        "## Выбор пространства для сопоставления",
        "",
        "Сопоставление кластеров во времени сначала было выполнено по центроидам в **исходных "
        "долях** и **отклонено**: в этом пространстве уверенность сопоставления определялась не "
        "различием типов, а общим временным трендом структуры расходов. За период средние доли "
        "сдвинулись у всех МО: "
        + ", ".join(f"{c.replace('share_', '')} {v:+.3f}" for c, v in zip(SHARE_COLUMNS, shift))
        + f". Рост доли маркетплейсов (**{shift[SHARE_COLUMNS.index('share_Маркетплейсы')]:+.3f}**) "
        "превышает типичные различия между кластерами, поэтому все центроиды ранних месяцев "
        "оказываются сдвинуты целиком: расстояние до якоря почти линейно зависит от удалённости "
        f"месяца (r = {comp_spaces.loc['corr(расстояние, месяц)', 'raw']:+.2f}), а уверенно "
        f"хотя бы в половине месяцев сопоставлялось лишь "
        f"{int((comp_spaces.loc[list(range(K)), 'raw'] >= n_other / 2).sum())} из {K} кластеров.",
        "",
        "Выбрано пространство **z-score внутри месяца** — то же, в котором строится сама "
        "кластеризация: оно убирает общий сдвиг и масштаб каждого месяца, и типы сравниваются "
        "по тем же признакам, по которым они выделены. Сравнение (число месяцев с уверенным "
        f"сопоставлением из {n_other}; последняя строка — корреляция медианного расстояния до "
        "якоря с номером месяца):",
        "",
        md_table(comp_spaces.rename(index={c: f"кластер {c}" for c in range(K)})
                 .rename(columns={"raw": "исходные доли (отклонено)",
                                  "raw_centered": "доли минус среднее месяца",
                                  "zscore": "z-score (выбрано)"}), index=True, fmt=",.2f"),
        "",
    ]
    lines = [
        f"# 12. KMeans k = {K} во времени: 24 месяца, сопоставление по профилю с {ANCHOR_MONTH}",
        "",
        "Сгенерировано `src/12_kmeans_temporal_tracking.py`. Заменяет прежний вариант с цепочкой "
        "сопоставлений через соседние месяцы (он давал дрейф нумерации).",
        "",
        f"- Для каждого месяца: z-score 5 долей внутри месяца, KMeans k = {K} с нуля "
        f"(random_state = {RANDOM_STATE}, n_init = {N_INIT}); центроиды между месяцами не "
        f"переносятся. Исходные метки: `{LABELS_DIR.relative_to(PROJECT_DIR)}/{{YYYY-MM}}.parquet`.",
        f"- **Сопоставление (вариант A):** центроид = средние исходные доли МО кластера; 7 "
        f"центроидов каждого месяца сопоставляются **напрямую** с 7 каноническими центроидами "
        f"{ANCHOR_MONTH} (метки k = {K} шага 10b) — `linear_sum_assignment` на евклидовых "
        "расстояниях. Номер кластера = номер ближайшего по профилю канонического типа (10c).",
        f"- Уверенность пары: margin = расстояние до ближайшего конкурента / расстояние до "
        f"сопоставленного центроида. **Уверенно** — взаимно ближайшие и margin ≥ {CONFIDENT_MARGIN}; "
        f"**неоднозначно** — не взаимно ближайшие или margin < {AMBIGUOUS_MARGIN}; иначе умеренно.",
        f"- Пространство центроидов: **{CENTROID_SPACE}** (обоснование — раздел ниже).",
        f"- Корреляция медианного расстояния до якоря с номером месяца: {dist_trend:+.2f} "
        f"(порог признания тренда доминирующим: |r| > {TREND_CONFOUND_ABS_CORR}).",
        f"- Разбиение каждого месяца — «официальное» из шага 12a: минимальная inertia среди 100 "
        "запусков KMeans (seed 0..99). Якорь — официальное разбиение "
        f"{ANCHOR_MONTH} в нумерации меток k = {K} шага 10b (`{KMEANS_FINAL_PATH.name}`, "
        f"seed 42{'; = kmeans_labels_final' if K == FINAL_K else ''}); с ними оно совпадает на ARI "
        f"{anchor_vs_canonical['ARI']:.3f} ({anchor_vs_canonical['МО с другим кластером']} МО в "
        "другом кластере).",
        f"- Файлы: сопоставления `{MATCHING_PATH.relative_to(PROJECT_DIR)}`, траектории "
        f"`{TRAJ_PATH.relative_to(PROJECT_DIR)}`, по МО `{SWITCHES_PATH.relative_to(PROJECT_DIR)}`.",
        "",
        *choice,
        f"## Уверенность сопоставления по каноническим кластерам ({n_other} месяцев без якоря)",
        "",
        md_table(conf.reset_index(), fmt=",.2f"),
        "",
        *finding,
        "Уверенность по месяцам (строка — месяц, столбец — канонический кластер):",
        "",
        md_table(conf_month.reset_index()),
        "",
        "## Проблемные месяцы",
        "",
        f"Критерий: ≥ {PROBLEM_MONTH_MIN_AMBIGUOUS} неоднозначных сопоставлений или доля МО, "
        f"сменивших кластер относительно предыдущего месяца, выше Q3 + {PROBLEM_SWITCH_IQR}·IQR "
        f"({sw_thr:.1%}). Сглаживание не применялось — решение о варианте C (кварталы) для этих "
        "месяцев принимается отдельно.",
        "",
        md_table(problem.drop(columns="проблемный")) if len(problem) else "Проблемных месяцев нет.",
        "",
        "Все месяцы:",
        "",
        md_table(mt, fmt=",.3f"),
        "",
        "## Метрики стабильности",
        "",
        f"- Средний Жаккар одноимённых кластеров соседних месяцев: **{jac['jaccard'].mean():.3f}** "
        f"(медиана {jac['jaccard'].median():.3f}); шаг 06 для рёбер сети — 0.18.",
        f"- ARI разбиений соседних месяцев (не зависит от нумерации): медиана "
        f"{mt['ARI с пред. мес. (разбиения)'].median():.3f} — даже при идеальной нумерации часть "
        "смен вызвана тем, что KMeans разных месяцев режет пространство по-разному.",
        "",
        "Число смен кластера за 23 перехода:",
        "",
        md_table(stab),
        "",
        md_table(pd.Series(n_switch).describe(percentiles=[.25, .5, .75, .9]).rename(
            "смен на МО").to_frame(), index=True, fmt=",.2f"),
        "",
        f"- Возвраты A→B→A: {flicker.sum() / max(n_switch.sum(), 1):.1%} всех смен.",
        f"- Доля месяцев в основном кластере: медиана {np.median(share_main):.2f}; МО ≥ 80% "
        f"времени в одном кластере: {(share_main >= .8).mean():.1%}.",
        f"- Чистая миграция (мода первых {PERIOD_WINDOW} мес. ≠ мода последних {PERIOD_WINDOW}): "
        f"{len(net)} МО ({len(net) / n:.1%}).",
        "",
        f"## Устойчивые переходы и мерцания (порог: новый кластер держится ≥ {L} мес. подряд)",
        "",
        f"Смена кластера в месяце t устойчива, если МО остаётся в новом кластере в месяцах t … "
        f"t+{L - 1}; иначе — мерцание (boundary noise: МО у границы кластеров, которую KMeans "
        f"разных месяцев проводит по-разному). Смены в последнем месяце ({ANCHOR_MONTH}) проверить "
        "нельзя — «не определено».",
        "",
        md_table(tr_tot),
        "",
        "Число устойчивых смен на МО:",
        "",
        md_table(stab_st),
        "",
        f"- **Ограничение порога {L} мес.:** из устойчивых смен {share_back:.0%} возвращают МО в "
        "кластер, где оно уже было раньше, — это многомесячные колебания между соседними "
        "типами, а не переход в новый тип. Смен в ранее не посещённый кластер на МО: медиана "
        f"{np.median(n_meaningful):.0f}, среднее {n_meaningful.mean():.2f} (см. раздел о критериях ниже).",
        f"- МО, у которых все смены — мерцания (ни одной устойчивой): **{only_flicker}** "
        f"({only_flicker / n:.1%}); без смен вовсе — {int((n_switch == 0).sum())}.",
        f"- Устойчивых смен на МО: медиана {np.median(n_stable):.0f}, среднее {n_stable.mean():.2f}, "
        f"max {n_stable.max()}; мерцаний: медиана {np.median(n_flick):.0f}, среднее "
        f"{n_flick.mean():.2f}.",
        "",
        "По основному кластеру МО:",
        "",
        md_table(pd.DataFrame({"основной кластер": main_cl, "устойчивых": n_stable,
                               "мерцаний": n_flick})
                 .groupby("основной кластер").agg(МО=("устойчивых", "size"),
                                                   устойчивых_медиана=("устойчивых", "median"),
                                                   устойчивых_среднее=("устойчивых", "mean"),
                                                   мерцаний_среднее=("мерцаний", "mean"))
                 .reset_index(), fmt=",.2f"),
        "",
        "Смены (все, без фильтра) по основному (модальному за 24 мес.) кластеру МО:",
        "",
        md_table(pd.DataFrame({"основной кластер": main_cl, "смен": n_switch})
                 .groupby("основной кластер")["смен"].agg(["count", "median", "mean"])
                 .reset_index(), fmt=",.2f"),
        "",
        "Жаккар одноимённых кластеров соседних месяцев:",
        "",
        md_table(jac_pivot.reset_index(), fmt=",.2f"),
        "",
        "## Уверенность принадлежности и пограничные МО",
        "",
        "Для каждого месяца KMeans запущен 100 раз (seed 0..99); каждый запуск сопоставлен с "
        "официальным разбиением месяца по пересечению состава (`linear_sum_assignment`). "
        "**Уверенность принадлежности** МО в месяце — доля запусков, где МО попало в тот же "
        f"кластер, что и в официальном разбиении. Таблица: `{CONFIDENCE_PATH.relative_to(PROJECT_DIR)}` "
        "(territory_id, month, cluster — согласованный номер, confidence).",
        "",
        f"- Уверенность по всем МО-месяцам: медиана {np.median(C):.2f}, среднее {C.mean():.3f}; "
        f"< {BORDERLINE_CONF}: {(C < BORDERLINE_CONF).mean():.1%}; ≥ {CONF_THRESHOLD}: "
        f"{(C >= CONF_THRESHOLD).mean():.1%}.",
        "",
        "По месяцам:",
        "",
        md_table(conf_month_tab, fmt=",.3f"),
        "",
        "По согласованным кластерам:",
        "",
        md_table(conf_cluster_tab.reset_index(), fmt=",.3f"),
        "",
        f"### Чувствительность метрик (а)/(б)/(в) к порогу уверенности",
        "",
        "Фильтр: переход из месяца t в t+1 засчитывается, только если уверенность МО и в t, и в "
        "t+1 не ниже порога. Для (в) мода каждого окна считается только по месяцам с "
        f"уверенностью ≥ порога; окно учитывается, если таких месяцев ≥ {MIN_CONF_MONTHS_IN_WINDOW} "
        f"из {PERIOD_WINDOW}. Основной порог — {CONF_THRESHOLD}.",
        "",
        md_table(sens, fmt=",.3f"),
        "",
        f"### Пограничные МО (уверенность < {BORDERLINE_CONF} более чем в "
        f"{BORDERLINE_MIN_SHARE:.0%} месяцев)",
        "",
        (f"**{len(border)} МО** ({len(border) / n:.1%}). Это не «мигранты», а отдельная "
         "содержательная категория: территории с гибридным профилем расходов, который объективно "
         f"не укладывается ни в один из {K} типов — даже в пределах одного месяца разные запуски "
         "KMeans относят их к разным кластерам. Ещё "
         f"{near_border} МО имеют уверенность < {BORDERLINE_CONF} в 25–50% месяцев."
         if len(border) else
         f"**Пограничных МО по этому критерию нет (0).** Максимальная доля месяцев с уверенностью "
         f"< {BORDERLINE_CONF} у одного МО — {low_share.max():.0%}; распределение МО по этой доле: "
         + ", ".join(f"{k}: {v}" for k, v in low_share_bins.items())
         + f". Низкая уверенность сосредоточена в отдельных месяцах (прежде всего "
         f"{conf_month_tab.loc[conf_month_tab[f'доля < {BORDERLINE_CONF}'].idxmax(), 'месяц']}: "
         f"{conf_month_tab[f'доля < {BORDERLINE_CONF}'].max():.0%} МО-месяцев против медианы "
         f"{conf_month_tab[f'доля < {BORDERLINE_CONF}'].median():.1%}), а не у отдельных МО.\n\n"
         "**Вывод:** внутри месяца разбиение KMeans почти детерминировано — уверенность "
         "принадлежности отражает только чувствительность к начальной инициализации, и она мала. "
         "Смены кластера от месяца к месяцу (ARI разбиений соседних месяцев ≈ "
         f"{mt['ARI с пред. мес. (разбиения)'].median():.2f}) поэтому вызваны **изменением данных "
         "между месяцами**, а не случайностью алгоритма; фильтр по уверенности убирает лишь их "
         "небольшую часть (см. таблицу чувствительности). Гибридность профиля проявляется "
         f"**во времени**, а не между запусками: {temporal_hybrid} МО проводят в своём основном "
         f"кластере меньше половины месяцев (< 60% — {temporal_hybrid60}). Это альтернативное "
         "определение «пограничных» МО; список по нему не формировался — определение категории "
         "за автором исследования; см. следующий раздел."),
        "",
        *([
            "Между какими типами они находятся (два самых частых согласованных кластера):",
            "",
            md_table(border["пара кластеров"].value_counts().rename("МО").rename_axis(
                "пара кластеров").reset_index(), fmt=",.0f"),
            "",
            "Регионы:",
            "",
            md_table(border["region_name"].value_counts().head(10).rename("МО")
                     .rename_axis("регион").reset_index(), fmt=",.0f"),
            "",
            md_table(border[["territory_id", "name", "region_name", "уверенность средняя",
                             f"доля месяцев с уверенностью < {BORDERLINE_CONF}",
                             "основные кластеры (доля мес.)", "смен кластера", "траектория"]]
                     .astype({"territory_id": str}), fmt=",.2f"),
            "",
        ] if len(border) else []),
        "### Пограничные во времени: «качающиеся» и «рассеянные» МО",
        "",
        "Определение: МО проводит в своём основном (самом частом) согласованном кластере меньше "
        f"порога месяцев. Подтипы: **качающиеся** — не менее {SWING_TOP2_MIN:.0%} месяцев в двух "
        "самых частых кластерах (колеблются между двумя типами); **рассеянные** — месяцы "
        "распределены по трём и более кластерам без доминирующей пары. Полная таблица по всем МО: "
        f"`{BORDERLINE_PATH.relative_to(PROJECT_DIR)}`.",
        "",
        "Распределение МО по доле месяцев в основном кластере:",
        "",
        md_table(share_hist.rename("МО").rename_axis("доля месяцев").reset_index()
                 .astype({"доля месяцев": str}), fmt=",.0f"),
        "",
        "Чувствительность к порогу доли основного кластера (подтипы — при топ-2 ≥ "
        f"{SWING_TOP2_MIN:.0%}):",
        "",
        md_table(tsens, fmt=",.3f"),
        "",
        f"Чувствительность подтипов к порогу топ-2 (при пороге основного кластера "
        f"< {TEMPORAL_THRESHOLD:.0%}):",
        "",
        md_table(tsens_top2, fmt=",.0f"),
        "",
        f"- Основной порог — **< {TEMPORAL_THRESHOLD:.0%}**. Естественного излома в распределении "
        "нет (см. гистограмму), поэтому порог — соглашение, а таблица показывает его цену: при "
        + ", ".join(f"{r['порог доли основного кластера']} — {r['пограничных']} МО ({r['доля МО']:.0%})"
                    for _, r in tsens.iterrows())
        + f". При < {TEMPORAL_THRESHOLDS[0]:.0%} в категорию попадают только МО без явного "
        f"основного типа, при < {TEMPORAL_THRESHOLDS[-1]:.0%} — уже больше трети всех МО, и "
        "категория теряет специфичность.",
        "- **Соотношение подтипов чувствительно к обоим порогам**: доля качающихся среди "
        "пограничных — "
        + ", ".join(f"{r['качающиеся'] / r['пограничных']:.0%} при {r['порог доли основного кластера']}"
                    for _, r in tsens.iterrows())
        + "; при фиксированном пороге основного кластера — "
        + ", ".join(f"{r['качающиеся'] / (r['качающиеся'] + r['рассеянные']):.0%} при топ-2 "
                    f"{r['порог топ-2 для «качающихся»']}" for _, r in tsens_top2.iterrows())
        + ". Устойчивый вывод — не точные размеры подтипов, а то, что среди МО с наиболее "
        "размытым профилем (строгий порог) преобладают рассеянные, а по мере ослабления порога "
        "добавляются в основном качающиеся между двумя соседними типами.",
        "- Оговорка: часть временной «пограничности» порождена методом — разбиения соседних "
        f"месяцев различаются (ARI ≈ {mt['ARI с пред. мес. (разбиения)'].median():.2f}) даже без "
        "изменения самих МО, поэтому МО у границы двух типов чаще попадают в эту категорию.",
        "",
        f"#### Качающиеся ({len(swing)} МО)",
        "",
        "Между какими типами (пара самых частых кластеров):",
        "",
        md_table(swing["пара кластеров"].value_counts().rename("МО").rename_axis("пара").reset_index()
                 .assign(профили=lambda d: [" | ".join(anchor_sig[int(x)] for x in p_.split("–"))
                                            for p_ in d["пара"]]), fmt=",.0f"),
        "",
        "Регионы (топ-10):",
        "",
        md_table(swing["region_name"].value_counts().head(10).rename("МО").rename_axis("регион")
                 .reset_index(), fmt=",.0f"),
        "",
        f"Список (первые {N_BORDER_SHOW} по наименьшей доле основного кластера; полный — в parquet):",
        "",
        md_table(swing.head(N_BORDER_SHOW)[["territory_id", "name", "region_name",
                                            "доля месяцев в основном кластере",
                                            "доля месяцев в топ-2 кластерах", "кластеры (доля мес.)",
                                            "уверенность средняя", "траектория"]]
                 .astype({"territory_id": str}), fmt=",.2f"),
        "",
        f"#### Рассеянные ({len(scatter)} МО)",
        "",
        "Основной кластер:",
        "",
        md_table(scatter["основной кластер"].value_counts().rename("МО").rename_axis("кластер")
                 .reset_index().assign(профиль=lambda d: d["кластер"].map(anchor_sig)), fmt=",.0f"),
        "",
        "Регионы (топ-10):",
        "",
        md_table(scatter["region_name"].value_counts().head(10).rename("МО").rename_axis("регион")
                 .reset_index(), fmt=",.0f"),
        "",
        f"Список (первые {N_BORDER_SHOW} по наименьшей доле топ-2; полный — в parquet):",
        "",
        md_table(scatter.head(N_BORDER_SHOW)[["territory_id", "name", "region_name",
                                              "доля месяцев в основном кластере",
                                              "доля месяцев в топ-2 кластерах", "разных кластеров",
                                              "кластеры (доля мес.)", "траектория"]]
                 .astype({"territory_id": str}), fmt=",.2f"),
        "",
        "## Критерии «содержательного перехода»: сравнение",
        "",
        "Содержательный переход (критерий б) засчитывается, только если одновременно: (1) целевой "
        "кластер МО раньше не посещало ни в одном из предыдущих месяцев и (2) остаётся в нём "
        f"минимум {L} месяца подряд после перехода. Новый кластер, продержавшийся 1 месяц, — шум.",
        "",
        md_table(crit, fmt=",.3f"),
        "",
        f"До и после фильтра по уверенности (порог {CONF_THRESHOLD}):",
        "",
        md_table(sens[sens["порог"].isin(["без фильтра", f"{CONF_THRESHOLD:.1f}"])], fmt=",.3f"),
        "",
        f"- Переходов в новый кластер всего {int(n_new_any.sum()):,}; из них одномесячных выбросов "
        f"(откат или уход дальше через месяц) — **{new_one_month:,}** "
        f"({new_one_month / max(n_new_any.sum(), 1):.0%}); в последнем месяце, где длительность "
        f"проверить нельзя, — {int(n_new_censored.sum())} (в (б) не засчитаны).",
        f"- Чувствительность интерпретации к критерию: доля МО, «сменивших тип», — "
        f"{crit.loc[0, 'доля МО']:.0%} по (а), {crit.loc[1, 'доля МО']:.0%} по (б), "
        f"{crit.loc[2, 'доля МО']:.0%} по (в). Разница между (а) и (б) — это цена одномесячных "
        "выбросов; (б) и (в) отвечают на разные вопросы: (б) — «появлялся ли у МО устойчиво новый "
        "тип хотя бы раз» (включая временный), (в) — «отличается ли доминирующий тип в конце "
        "периода от начала» (итоговый сдвиг). Headline-метрика — (в).",
        "",
        "Содержательные переходы (б) по месяцу перехода и целевому кластеру:",
        "",
        md_table(pd.DataFrame({"месяц": np.repeat([months], n, axis=0)[meaningful],
                               "в кластер": A[meaningful]})
                 .pivot_table(index="месяц", columns="в кластер", aggfunc="size", fill_value=0)
                 .reindex(months, fill_value=0).assign(всего=lambda d: d.sum(axis=1))
                 .reset_index(), fmt=",.0f"),
        "",
        f"- **Смещение критерия (1) к началу периода:** «не посещённый ранее» зависит от длины "
        f"истории — в первые месяцы почти любая смена «новая». На первые {PERIOD_WINDOW} переходов "
        f"({months[1]} … {months[PERIOD_WINDOW]}) приходится "
        f"{int(meaningful[:, :PERIOD_WINDOW + 1].sum())} из {int(meaningful.sum())} событий (б) "
        f"({meaningful[:, :PERIOD_WINDOW + 1].sum() / max(meaningful.sum(), 1):.0%}). Счётчики (а) и (б) "
        "поэтому нельзя сравнивать между началом и концом периода; (в) этого смещения не имеет.",
        f"- **Всплеск в {peak_month}:** {peak_n} содержательных переходов, из них {peak_top_n} — в "
        f"кластер {peak_top}. Чувствительность KMeans к seed в этом месяце — ARI "
        f"{seas.loc[seas['месяц'] == peak_month, seas.columns[7]].iloc[0]:.3f} (медиана по месяцам "
        f"{seas[seas.columns[7]].median():.3f}): вероятнее сдвиг границы кластера в разбиении "
        "этого месяца, чем одновременная смена типа у всех этих МО.",
        "",
        f"## МО с наибольшим числом содержательных переходов (топ-{TOP_N_MIGRANTS}, фильтр "
        f"уверенности {CONF_THRESHOLD})",
        "",
        f"Сортировка — по содержательным переходам (б) с фильтром уверенности ≥ {CONF_THRESHOLD}, "
        "затем по (б) без фильтра и по меньшему числу мерцаний. Траектория — согласованный номер в "
        "каждом из 24 месяцев (2023-01 … 2024-12).",
        "",
        md_table(migrants_f[["territory_id", "name", "region_name",
                             f"содержательных переходов (conf ≥ {CONF_THRESHOLD})",
                             "уверенность средняя", "содержательных переходов",
                           "переходов в новый кластер (любой длит.)", "устойчивых смен", "мерцаний",
                           "смен кластера", "разных кластеров",
                           "доля месяцев в основном кластере", "траектория"]]
                 .astype({"territory_id": str}), fmt=",.2f"),
        "",
        f"Регионы среди МО с ≥ 2 содержательными переходами:",
        "",
        md_table(sw[sw["содержательных переходов"] >= 2]["region_name"].value_counts().head(10)
                 .rename("МО").rename_axis("регион").reset_index(), fmt=",.0f"),
        "",
        "## Проверка гипотезы сезонности проблемных месяцев",
        "",
        f"Проблемные месяцы этого прогона: {', '.join(PROBLEM_MONTHS) or 'нет'}. Основной показатель — "
        "**ARI разбиения месяца с соседними месяцами** (не зависит от якоря и нумерации); "
        "дополнительно — доля мерцаний среди смен в этот месяц, число неоднозначных сопоставлений "
        f"(зависит от якоря: {ANCHOR_MONTH} сопоставляется сам с собой и тривиально «уверен»), "
        "контроль чувствительности KMeans к seed (среднее ARI 100 запусков с официальным разбиением) и "
        "величина сдвига средних долей к предыдущему месяцу (сезонность в самих данных).",
        "",
        md_table(seas.drop(columns=["год", "календарный месяц"]), fmt=",.3f"),
        "",
        "Проблемный месяц против того же календарного месяца другого года (нижняя треть по ARI "
        f"с соседями: ≤ {low_thr:.3f}):",
        "",
        md_table(seas_pairs, fmt=",.3f"),
        "",
        f"- Проверяемые календарные месяцы (другой год — не якорь): "
        f"{', '.join(MONTH_NAMES.get(m, str(m)) for m in cal.index)}; провал повторяется в "
        f"обоих годах: **{', '.join(MONTH_NAMES.get(m, str(m)) for m in repeated_months) or 'нет'}** "
        f"({len(repeated_months)} из {len(cal)}). Непроверяемы: "
        f"{', '.join(MONTH_NAMES.get(m, str(m)) for m in cal_untestable) or '—'} "
        f"(парный месяц — якорь {ANCHOR_MONTH}).",
        f"- Корреляция «ARI с соседями» 2023 vs 2024 по календарным месяцам: Спирмен "
        f"**{rho_yy:+.2f}** (p = {p_yy:.2f}, n = {len(yy_ok)}); сдвига средних долей 2023 vs 2024: "
        f"{rho_shift:+.2f}.",
        f"- ARI с соседями vs сдвиг средних долей к пред. месяцу (все месяцы): {rho_shift_ari:+.2f}; "
        f"vs чувствительность KMeans к seed: {rho_seed:+.2f}. ARI между seed: проблемные месяцы — "
        f"медиана {seed_prob.median():.3f}, остальные — {seed_rest.median():.3f}.",
        "",
        "**Вывод.** " + (
            "Сезонность подтверждается: провалы устойчивости повторяются в те же календарные "
            "месяцы обоих лет, и профиль по календарным месяцам значимо согласован между годами."
            if seasonal_confirmed else
            "Сезонность как общее объяснение проблемных месяцев **не подтверждается**: из "
            f"{len(cal)} проверяемых календарных месяцев провал повторяется в обоих годах только "
            f"в {len(repeated_months)} "
            f"({', '.join(MONTH_NAMES.get(m, str(m)) for m in repeated_months) or '—'}), а профиль "
            f"устойчивости по календарным месяцам между годами согласован слабо и незначимо "
            f"(ρ = {rho_yy:+.2f}, p = {p_yy:.2f}). "
            + "".join(
                f"**{MONTH_NAMES[m].capitalize()}** повторяется в обоих годах — кандидат на сезонный "
                "эффект; чувствительность KMeans к seed в эти месяцы: "
                + ", ".join(f"{x} — ARI {seas.loc[seas['месяц'] == x, seed_col].iloc[0]:.3f}"
                            for x in seas.loc[seas['календарный месяц'] == m, 'месяц'])
                + " (высокий ARI = разбиение месяца чёткое, но отличается от соседних, т. е. сдвиг "
                "в данных, а не шум алгоритма). На двух наблюдениях эффект не доказан. "
                for m in repeated_months)
            + ("Пара **декабрь 2023 – январь 2024**: доля мерцаний в декабре "
               f"{seas.loc[seas['месяц'] == '2023-12', 'доля мерцаний среди смен'].iloc[0]:.0%} "
               "(смены откатываются в следующем месяце) — картина, согласующаяся с новогодними "
               f"тратами, но проверить повторение нельзя ({ANCHOR_MONTH} — якорь, января 2025 в "
               "данных нет). "
               if {"2023-12", "2024-01"} <= set(PROBLEM_MONTHS) else "")
            + (f"Альтернативные гипотезы для непериодичных провалов ({', '.join(non_periodic)}): "
               "(1) **геометрическая неоднозначность разбиения** — ARI между seed в эти месяцы: "
               + ", ".join(f"{x} — {seas.loc[seas['месяц'] == x, seed_col].iloc[0]:.3f}"
                           for x in non_periodic)
               + f" (медиана по остальным месяцам {seed_rest.median():.3f}); (2) **разовые "
               "обновления данных или методологии СберИндекса** (пересчёт моделей оценки, "
               "изменение охвата транзакций) — проверяется только по документации источника, меток "
               "в датасете нет."
               if non_periodic else "")
        ),
        "",
        f"## Чистая миграция: основные потоки (мода первых → последних {PERIOD_WINDOW} мес.)",
        "",
        "Профили: " + "; ".join(f"**{c}**: {s}" for c, s in anchor_sig.items()),
        "",
        md_table(flows.head(15), fmt=",.0f"),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(conf.drop(columns=f"профиль ({ANCHOR_MONTH})").to_string())
    print(stab.to_string(index=False))
    print(f"jaccard mean {jac['jaccard'].mean():.3f}; dist trend {dist_trend:+.2f}; sw_thr {sw_thr:.3f}")
    print(mt.to_string(index=False))
    print(flows.head(10).to_string(index=False))
    print(tr_tot.to_string(index=False)); print(stab_st.to_string(index=False))
    print(seas.to_string(index=False)); print(seas_pairs.to_string(index=False))
    print(f"rho_yy {rho_yy:+.2f} rho_shift {rho_shift:+.2f} rho_seed {rho_seed:+.2f} "
          f"rho_shift_ari {rho_shift_ari:+.2f} p_yy {p_yy:.2f} repeated {repeated_months}")
    print(crit.to_string(index=False)); print("new_one_month", new_one_month)
    print(anchor_vs_canonical); print(sens.to_string(index=False))
    print("borderline", len(border), "near", near_border)
    print(tsens.to_string(index=False)); print(tsens_top2.to_string(index=False))
    print(swing["пара кластеров"].value_counts().head(8).to_string())
    print(scatter["region_name"].value_counts().head(5).to_string())
    print(swing["region_name"].value_counts().head(5).to_string())
    print(border["пара кластеров"].value_counts().head(8).to_string())
    print(conf_cluster_tab.drop(columns="профиль").to_string())
    print(pd.Series(n_meaningful).value_counts().sort_index().to_string())
    print(migrants[["name", "region_name", "содержательных переходов", "мерцаний", "траектория"]]
          .head(12).to_string(index=False))


if __name__ == "__main__":
    main()
