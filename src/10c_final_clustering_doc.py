"""Шаг 10c. Фиксация канонического KMeans (k = config.FINAL_K, 2024-12) и документирование выбора.

Кластеризация НЕ пересчитывается: используются сохранённые результаты шагов 10, 10b, 10d,
12a/12 (k = FINAL_K и k = SUPERSEDED_K) и раздел профилей из отчёта 10b.

Вход:  data/processed/kmeans_labels_k{4,6,7,8}_<YYYY_MM>.parquet (шаг 10b),
       data/processed/kmeans_k_selection_<YYYY_MM>.parquet (шаг 10),
       data/processed/prediction_strength*_<YYYY_MM>.parquet (шаг 10d),
       data/processed/kmeans_k{FINAL_K}_runs_*.parquet (шаг 12a — плоский минимум),
       data/processed/kmeans_k{SUPERSEDED_K}_matching.parquet,
       kmeans_k{SUPERSEDED_K}_membership_confidence.parquet (шаги 12a/12 для прежнего k),
       notebooks/10b_kmeans_cluster_profiles.md, data/raw/territories.parquet
Выход: data/processed/kmeans_labels_final.parquet (копия меток k = FINAL_K),
       data/processed/kmeans_labels_final_k{SUPERSEDED_K}_superseded.parquet (прежний канон),
       notebooks/10c_final_clustering_k{FINAL_K}.md
Запуск из корня проекта:  .venv/bin/python src/10c_final_clustering_doc.py
"""
import importlib
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score

from config import FINAL_K, MONTH, SUPERSEDED_K
from network_utils import SHARE_COLUMNS, month_shares, standardize_shares

step10b = importlib.import_module("10b_kmeans_cluster_profiles")

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
TERRITORIES_PATH = RAW_DIR / "territories.parquet"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
PROFILES_REPORT = NOTEBOOKS_DIR / "10b_kmeans_cluster_profiles.md"
REPORT_PATH = NOTEBOOKS_DIR / f"10c_final_clustering_k{FINAL_K}.md"
FINAL_LABELS_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
SUPERSEDED_LABELS_PATH = PROCESSED_DIR / f"kmeans_labels_final_k{SUPERSEDED_K}_superseded.parquet"
TAG = MONTH.replace("-", "_")

from config import NEAR_MIN_INERTIA  # «плоский минимум»: запуски с inertia ≤ min + 0.5
from config import SAME_PARTITION_ARI  # запуски с ARI ≥ порога считаются одним и тем же разбиением
from config import CANONICAL_SEED
from config import PS_THRESHOLDS
COMPARE_K = [4, 6, 7, 8]
DECISION_DATE = "2026-09-28"
# кластеры прежнего канона k = 7 (номера из 10b), на которых строилось прежнее обоснование
TRANSPORT_CLUSTER_K7 = 4   # ↑ транспорт, ↓ здоровье: Забайкалье, Иркутская обл., Красноярский край
REMOTE_CLUSTER_K7 = 6      # ↓ маркетплейсы, ↓ здоровье: Якутия, Сахалин, Амурская обл., Хабаровский край
AVERAGE_CLUSTER_K8 = 6     # «близко к среднему по всем категориям»
REMOTE_TOP_REGIONS = 7     # сколько крупнейших регионов удалённой группы перечислять в «Итоге»
NUM_WORDS = {5: "пять", 6: "шесть", 7: "семь", 8: "восемь", 9: "девять", 10: "десять"}
# ориентиры Shalileh et al. (2025, табл. 2; синтетика): SW ≥ 0.5; CH/N ≥ 1 (в идеале ≥ 1.5);
# S_Dbw ≤ 0.6 (в идеале ≤ 0.2); случайный уровень S_Dbw в статье ≈ 1–1.5
ICVI_SW_MIN, ICVI_CHN_MIN, ICVI_SDBW_MAX = 0.5, 1.0, 0.6

SOURCE_ICVI = ("Shalileh S., Antonov E. A., Tsyplakova D. A. Cluster Validity across Attribute and "
               "Network Spaces: Empirical Benchmarks for Attributed Networks Clustering // Doklady "
               "Mathematics. 2025. Vol. 112, No. 3. P. 553–564. DOI 10.1134/S1064562425700589")


def md(df: pd.DataFrame, index: bool = True, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def labels(k: int) -> pd.Series:
    return pd.read_parquet(PROCESSED_DIR / f"kmeans_labels_k{k}_{TAG}.parquet").set_index(
        "territory_id")["cluster"]


def best_match(src: pd.Series, c: int, dst: pd.Series) -> dict:
    """Кластер в dst, лучше всего покрывающий кластер c из src, и Жаккар между ними."""
    a = set(src[src == c].index)
    cb = dst[dst.index.isin(a)].value_counts().index[0]
    b = set(dst[dst == cb].index)
    return {"cluster": int(cb), "size": len(b), "common": len(a & b),
            "jaccard": len(a & b) / len(a | b), "split": dst.loc[sorted(a)].value_counts()}


def section_from_md(path: Path, header: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(header)
    nxt = text.find("\n## ", start + len(header))
    return text[start:nxt if nxt != -1 else len(text)].strip()


def flat_minimum_stats(ids: np.ndarray, canonical: np.ndarray, k: int) -> dict:
    """Числа проверки плоского минимума (для шага 16): запуски у минимума, ARI, «подвижные» МО."""
    relabel = importlib.import_module("12a_kmeans_runs").overlap_relabel
    runs = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{k}_runs_summary.parquet")
    runs = runs[runs["month"] == MONTH].set_index("seed")
    lab_all = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{k}_runs_labels.parquet",
                              filters=[("month", "==", MONTH)])
    L = {int(sd): g.set_index("territory_id")["cluster"].loc[ids].to_numpy()
         for sd, g in lab_all.groupby("seed")}
    near = runs.index[runs["inertia"] <= runs["inertia"].min() + NEAR_MIN_INERTIA].tolist()
    moved = np.zeros(len(ids), dtype=bool)
    for sd in near:
        moved |= relabel(canonical, L[sd], k) != canonical
    aris = [adjusted_rand_score(L[a], L[b]) for a, b in combinations(near, 2)]
    return {"запусков у минимума": len(near), "доля подвижных МО": float(moved.mean()),
            "медианный ARI у минимума": float(np.median(aris)) if aris else 1.0}


def flat_minimum_check(ids: np.ndarray, canonical: np.ndarray, k: int) -> list[str]:
    """Систематическая проверка: одинаковы ли по структуре все запуски около минимума inertia."""
    relabel = importlib.import_module("12a_kmeans_runs").overlap_relabel
    runs = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{k}_runs_summary.parquet")
    runs = runs[runs["month"] == MONTH].set_index("seed")
    lab_all = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{k}_runs_labels.parquet",
                              filters=[("month", "==", MONTH)])
    L = {int(sd): g.set_index("territory_id")["cluster"].loc[ids].to_numpy()
         for sd, g in lab_all.groupby("seed")}
    best = runs["inertia"].min()
    # порядок как в шаге 12a: inertia с округлением до 1e-6, затем seed
    near = (runs[runs["inertia"] <= best + NEAR_MIN_INERTIA].assign(_r=lambda d: d["inertia"].round(6))
            .reset_index().sort_values(["_r", "seed"])["seed"].tolist())
    pairs = pd.DataFrame([{"a": a, "b": b, "ARI": adjusted_rand_score(L[a], L[b]),
                           "Δinertia": abs(runs.at[a, "inertia"] - runs.at[b, "inertia"])}
                          for a, b in combinations(near, 2)])
    all_pairs = np.array([adjusted_rand_score(L[a], L[b]) for a, b in combinations(L, 2)])
    groups, seen = [], set()
    for sd in near:
        if sd in seen:
            continue
        g = [x for x in near if x not in seen and adjusted_rand_score(L[sd], L[x]) >= SAME_PARTITION_ARI]
        seen |= set(g)
        groups.append(g)
    official = int(runs.index[runs["официальный"]][0])   # официальный seed из шага 12a
    parts = pd.DataFrame([{
        "разбиение": i, "запусков": len(g), "seed (пример)": g[0], "inertia": runs.at[g[0], "inertia"],
        "ARI с официальным": adjusted_rand_score(L[official], L[g[0]]),
        f"ARI с каноническим (seed {CANONICAL_SEED})": adjusted_rand_score(canonical, L[g[0]]),
        "МО не как в каноне": int((relabel(canonical, L[g[0]], k) != canonical).sum()),
        f"содержит seed {CANONICAL_SEED}": CANONICAL_SEED in g} for i, g in enumerate(groups)])
    moved = np.zeros(len(ids), dtype=bool)
    for sd in near:
        moved |= relabel(canonical, L[sd], k) != canonical
    core_diff = relabel(canonical, L[official], k) != canonical
    q = pairs["ARI"].quantile([0, .05, .25, .5, .75, 1]) if len(pairs) else pd.Series(
        1.0, index=[0, .05, .25, .5, .75, 1])
    one_family = q[.5] >= 0.9 and q[0] >= 0.8
    lines = [
        f"## Систематическая проверка плоского минимума (k = {k}, {MONTH})",
        "",
        f"Из 100 запусков KMeans (seed 0..99, шаг 12a) **{len(near)}** имеют inertia в пределах "
        f"{NEAR_MIN_INERTIA} от лучшего ({best:.3f}; официальный — seed {official}). Попарный ARI "
        f"посчитан для **всех** {len(pairs)} пар этих запусков; для сравнения — все "
        f"{len(all_pairs)} пар из 100 запусков.",
        "",
        md(pd.DataFrame({f"{len(near)} запусков у минимума": pairs["ARI"].describe(
            percentiles=[.05, .25, .5, .75]) if len(pairs) else pd.Series(dtype=float),
            "все 100 запусков": pd.Series(all_pairs).describe(percentiles=[.05, .25, .5, .75])}),
           fmt=",.4f"),
        "",
        f"- Различимых разбиений среди запусков у минимума (объединение при ARI ≥ "
        f"{SAME_PARTITION_ARI}): **{len(groups)}**.",
        f"- Официальное разбиение (seed {official}) и каноническое (seed {CANONICAL_SEED}, "
        f"`kmeans_labels_final`) различаются у **{int(core_diff.sum())}** МО "
        f"(ARI {adjusted_rand_score(canonical, L[official]):.3f}).",
        f"- МО, которые хотя бы в одном из запусков у минимума попадают не в канонический кластер: "
        f"**{int(moved.sum())}** ({moved.mean():.1%}); остальные {int((~moved).sum())} во всех "
        f"{len(near)} запусках в одном и том же кластере.",
        "",
        md(parts, index=False, fmt=",.4f"),
        "",
        "**Вывод.** " + (
            f"Все запуски у минимума — варианты одной структуры: медианный ARI {q[.5]:.3f}, "
            f"минимальный {q[0]:.3f}. Качественно других типов нет; «подвижны» лишь "
            f"{int(moved.sum())} МО ({moved.mean():.1%}). Минимум "
            + ("пологий, но не хрупкий." if len(near) > 1 else "выражен (один запуск у минимума).")
            if one_family else
            f"Среди запусков у минимума есть качественно разные структуры: медианный ARI "
            f"{q[.5]:.3f}, минимальный {q[0]:.3f} — выбор запуска влияет на типологию."),
        "",
    ]
    return lines


def main() -> None:
    ks = sorted(set(COMPARE_K) | {FINAL_K, SUPERSEDED_K})
    L = {k: labels(k) for k in ks}
    metrics = pd.read_parquet(PROCESSED_DIR / f"kmeans_k_selection_{TAG}.parquet").set_index("k")
    ps = pd.read_parquet(PROCESSED_DIR / f"prediction_strength_{TAG}.parquet").set_index("k")
    psc = pd.read_parquet(PROCESSED_DIR / f"prediction_strength_clusters_{TAG}.parquet")
    terr = pd.read_parquet(TERRITORIES_PATH).set_index("territory_id")
    ids = L[FINAL_K].index.to_numpy()

    # 1. Канонические метки и прежний канон
    final = L[FINAL_K].reset_index()
    final.to_parquet(FINAL_LABELS_PATH, engine="pyarrow", index=False)
    L[SUPERSEDED_K].reset_index().to_parquet(SUPERSEDED_LABELS_PATH, engine="pyarrow", index=False)

    # Профили (подписи 10b) для канонических кластеров
    sh = month_shares(pd.read_parquet(SHARES_PATH), MONTH).set_index("territory_id").loc[ids]
    Z = pd.DataFrame(standardize_shares(sh[SHARE_COLUMNS].to_numpy())[0], index=ids, columns=SHARE_COLUMNS)

    def sig(k: int, c: int) -> str:
        return step10b.describe_profile(Z[L[k] == c].mean())

    # a) сравнение кандидатов
    rows = []
    for k in COMPARE_K:
        sizes = L[k].value_counts().sort_index()
        rows.append({"k": k, "SW": metrics.at[k, "SW"], "CH": metrics.at[k, "CH"],
                     "CH/N": metrics.at[k, "CH"] / len(L[k]), "S_Dbw": metrics.at[k, "S_Dbw"],
                     "S_Dbw norm": metrics.at[k, "S_Dbw norm"],
                     "norm 95% ДИ": f"[{metrics.at[k, 'S_Dbw norm 95% ДИ: низ']:.2f}; "
                                    f"{metrics.at[k, 'S_Dbw norm 95% ДИ: верх']:.2f}]",
                     "PS": ps.at[k, "PS mean"],
                     "PS 95% ДИ": f"[{ps.at[k, 'PS mean'] - 1.96 * ps.at[k, 'PS se']:.2f}; "
                                  f"{ps.at[k, 'PS mean'] + 1.96 * ps.at[k, 'PS se']:.2f}]",
                     "мин. кластер": sizes.min(), "размеры кластеров": ", ".join(map(str, sizes.tolist()))})
    comp = pd.DataFrame(rows)
    sw_best, ch_best = metrics["SW"].idxmax(), metrics["CH"].idxmax()
    sd, sn = metrics["S_Dbw"], metrics["S_Dbw norm"]
    lo, hi = metrics["S_Dbw norm 95% ДИ: низ"], metrics["S_Dbw norm 95% ДИ: верх"]
    base = metrics["S_Dbw random mean"]
    sw_f, chn_f = metrics.at[FINAL_K, "SW"], metrics.at[FINAL_K, "CH"] / len(L[FINAL_K])
    n_icvi_ok = int(sw_f >= ICVI_SW_MIN) + int(chn_f >= ICVI_CHN_MIN) + int(sd[FINAL_K] <= ICVI_SDBW_MAX)
    ci_overlap_678 = max(lo[[6, 7, 8]]) <= min(hi[[6, 7, 8]])
    best_norm_k = int(sn.idxmin())
    small_k_better = hi[[2, 3]].max() < lo[range(6, 12)].min()
    d67 = sd[6] - sd[7]
    base_share_67 = (base[6] - base[7]) / d67 if d67 else np.nan

    # d) prediction strength
    def ps_diff(a: int, b: int) -> tuple[float, float]:
        return (ps.at[a, "PS mean"] - ps.at[b, "PS mean"],
                float(np.hypot(ps.at[a, "PS se"], ps.at[b, "PS se"])))
    ps_ge4 = ps.loc[ps.index >= 4, "PS mean"]
    ps_best_ge4 = int(ps_ge4.idxmax())
    d_fs, se_fs = ps_diff(FINAL_K, SUPERSEDED_K)
    above = {t: [int(k) for k in ps.index if ps.at[k, "PS mean"] >= t] for t in PS_THRESHOLDS}

    # e) транспортная группа k=7: три независимые проверки
    col7 = f"доминирующий тип k={SUPERSEDED_K}"
    t7 = psc[psc["k"] == SUPERSEDED_K].groupby(col7)
    ps_pairs7 = t7["доля пар"].mean()
    ps_min7 = t7["минимум"].sum()
    match7 = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{SUPERSEDED_K}_matching.parquet")
    match7 = match7[match7["месяц"] != MONTH]
    conf_match7 = (match7["уверенность"] == "уверенно").groupby(match7["cluster"]).sum()
    n_months = match7["месяц"].nunique()
    mc7 = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{SUPERSEDED_K}_membership_confidence.parquet")
    mem7 = mc7.groupby("cluster")["confidence"].agg(средняя="mean", **{"доля < 0.5": lambda x: (x < .5).mean()})
    evid = pd.DataFrame({
        "профиль (k = 7)": [sig(SUPERSEDED_K, c) for c in range(SUPERSEDED_K)],
        "PS: доля верно предсказанных пар": ps_pairs7.reindex(range(SUPERSEDED_K)),
        "PS: раз был минимумом (из 100)": ps_min7.reindex(range(SUPERSEDED_K)).fillna(0).astype(int),
        f"время: уверенных сопоставлений (из {n_months})": conf_match7.reindex(range(SUPERSEDED_K)),
        "уверенность принадлежности: средняя": mem7["средняя"],
        "уверенность принадлежности: доля < 0.5": mem7["доля < 0.5"],
    }).rename_axis("тип k = 7")
    tr = TRANSPORT_CLUSTER_K7
    worst = {"PS": int(ps_pairs7.idxmin()), "время": int(conf_match7.idxmin()),
             "уверенность": int(mem7["средняя"].idxmin())}
    n_confirm = sum(v == tr for v in worst.values())

    # f) что происходит при k = FINAL_K
    tr_members = L[SUPERSEDED_K][L[SUPERSEDED_K] == tr].index
    tr_regions = terr.loc[tr_members, "region_name"].value_counts().head(4)
    tr_split = L[FINAL_K].loc[tr_members].value_counts()
    rm = best_match(L[SUPERSEDED_K], REMOTE_CLUSTER_K7, L[FINAL_K])
    rm4 = best_match(L[SUPERSEDED_K], REMOTE_CLUSTER_K7, L[4])
    rm_regions = terr.loc[L[FINAL_K][L[FINAL_K] == rm["cluster"]].index, "region_name"].value_counts().head(4)
    avg8 = int((L[8] == AVERAGE_CLUSTER_K8).sum())
    # тип k = FINAL_K с наихудшей воспроизводимостью по PS
    colF = f"доминирующий тип k={FINAL_K}"
    tF = psc[psc["k"] == FINAL_K].groupby(colF)["доля пар"].mean()

    def regs(vc: pd.Series) -> str:
        return ", ".join(f"{r} — {n}" for r, n in vc.items())

    top2 = tr_split.head(2)
    # удалённая группа k = FINAL_K: устойчивость при k + 1, k + 2, география, воспроизводимость
    rm_c = rm["cluster"]
    rm_members = L[FINAL_K].index[L[FINAL_K] == rm_c]
    rm_next = {k: best_match(L[FINAL_K], rm_c, L[k])["jaccard"] for k in (FINAL_K + 1, FINAL_K + 2)}
    hw = pd.read_parquet(PROCESSED_DIR / "transport_network_highway_knn8.parquet")
    Ghw = nx.Graph()
    Ghw.add_edges_from(zip(hw["territory_id_x"], hw["territory_id_y"]))
    rm_comps = sorted((len(c) for c in nx.connected_components(Ghw.subgraph(rm_members))), reverse=True)
    rm_nreg = terr.loc[rm_members, "region_name"].nunique()
    rm_top = terr.loc[rm_members, "region_name"].value_counts().head(REMOTE_TOP_REGIONS)
    rm4_final = best_match(L[FINAL_K], rm_c, L[4])   # удалённая группа канона k = FINAL_K при k = 4
    matchF = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{FINAL_K}_matching.parquet")
    matchF = matchF[matchF["месяц"] != MONTH]
    rm_conf = int(((matchF["cluster"] == rm_c) & (matchF["уверенность"] == "уверенно")).sum())
    nF_months = matchF["месяц"].nunique()
    rm_is_weakest = int(tF.idxmin()) == rm_c
    # PS удалённой группы против следующего по слабости типа: парно по разбиениям (split × направление)
    rm_next_type = int(tF.drop(rm_c).idxmin())
    ps_by_split = psc[psc["k"] == FINAL_K].groupby(["split", "направление", colF])["доля пар"].mean().unstack()
    ps_pair = ps_by_split[[rm_c, rm_next_type]].dropna()
    rm_ps_p = wilcoxon(ps_pair[rm_c], ps_pair[rm_next_type]).pvalue

    # сведения шагов 14–15 (есть только после их выполнения; run_all.sh перезапускает 10c в конце)
    later = {}
    p14 = PROCESSED_DIR / "robust_k_by_month.parquet"
    if p14.exists():
        r14 = pd.read_parquet(p14).pivot(index="месяц", columns="k", values="PS")
        best_m = r14[[k for k in r14.columns if k >= 4]].idxmax(axis=1)
        later["ps_months"] = (int((best_m == FINAL_K).sum()), len(best_m),
                              int((r14[FINAL_K] > r14[SUPERSEDED_K]).sum()))
        later["ps_best_dist"] = best_m.value_counts().sort_index()
    p15a = PROCESSED_DIR / "robust2_spec_months.parquet"
    p15e = PROCESSED_DIR / "robust2_external_validity.parquet"
    if p15a.exists() and p15e.exists():
        sm = pd.read_parquet(p15a)
        ari = sm.dropna(subset=["ARI с пред. мес."]).pivot(index="месяц", columns="вариант", values="ARI с пред. мес.")
        ev = pd.read_parquet(p15e)
        ev = ev[ev["набор"] == "варианты"]
        ext_cols = [c for c in ev.columns if c not in ("набор", "разбиение", "месяц")]
        ext_p = {}
        for a_, b_ in [("A", "B"), ("A", "C")]:
            xa = ev[ev["разбиение"] == a_].set_index("месяц")
            xb = ev[ev["разбиение"] == b_].set_index("месяц").reindex(xa.index)
            ext_p[(a_, b_)] = {c: (wilcoxon(xa[c], xb[c]).pvalue, float((xa[c] - xb[c]).median()))
                               for c in ext_cols}
        later["variants"] = {
            "ari_median": ari.median(), "p_AB": wilcoxon(ari["A"], ari["B"]).pvalue,
            "p_AC": wilcoxon(ari["A"], ari["C"]).pvalue,
            "ext_mean": ev.groupby("разбиение")[ext_cols].mean(), "ext_p": ext_p}
    JUSTIFICATION = (
        f"k = {FINAL_K} выбран не статистическим критерием. Внутренние показатели (SW, CH/N, S_Dbw "
        "после нормализации, gap statistic, prediction strength, устойчивость по времени) монотонно "
        "ухудшаются или смещены к малым k, внешние (связь с регионами, market_access, рёбрами "
        f"highway) растут с k. k = {FINAL_K} — компромисс между устойчивостью и связью с внешними "
        f"данными; на нём выделяется удалённая группа ({len(rm_members)} МО; "
        f"{NUM_WORDS.get(len(rm_top), str(len(rm_top)))} регионов дают {int(rm_top.sum())} МО: "
        f"{regs(rm_top)}; всего {rm_nreg} регионов); в графе highway kNN8 она не образует единой связной "
        f"области — {len(rm_comps)} компонент, крупнейшая {rm_comps[0]} МО; устойчива при "
        f"k = {FINAL_K + 1}–{FINAL_K + 2} (Жаккар {rm_next[FINAL_K + 1]:.2f} и {rm_next[FINAL_K + 2]:.2f}). "
        f"Транспортный тип k = {SUPERSEDED_K} слабо воспроизводится и собран в основном из двух "
        f"кластеров k = {FINAL_K} ({' + '.join(str(n) for n in top2.values)} из {len(tr_members)} МО). "
        f"Слабое место k = {FINAL_K}: удалённая группа — "
        + ("наименее воспроизводимый тип" if rm_is_weakest else "один из наименее воспроизводимых типов")
        + f" (PS {tF[rm_c]:.2f} "
        + (f"(точечно; разница с типом {rm_next_type} — {tF[rm_next_type]:.2f} — незначима, p = {rm_ps_p:.2f})"
           if rm_ps_p >= 0.05 else
           f"(разница с типом {rm_next_type} — {tF[rm_next_type]:.2f} — значима, p = {rm_ps_p:.2g})")
        + f", во времени {rm_conf} из {nF_months} месяцев).")

    if "variants" in later:
        vv = later["variants"]
        sig_ext = {pair: [c for c, (pv, _) in d.items() if pv < 0.05] for pair, d in vv["ext_p"].items()}
        b_lower = [(c, pv) for c, (pv, dmed) in vv["ext_p"][("A", "B")].items() if pv < 0.05 and dmed > 0]
        n_ext, n_pairs = len(vv["ext_p"][("A", "B")]), len(vv["ext_p"])
        bonf = 0.05 / n_ext   # поправка Бонферрони по числу показателей
        VARIANT_TEXT = (
            "Выбран вариант A — доли пяти категорий от «Все категории». На внешних данных A и C "
            "равноценны (значимых различий нет"
            + (f", кроме {', '.join(sig_ext[('A', 'C')])}" if sig_ext[("A", "C")] else "")
            + "), B хуже (значимо ниже A по "
            + (", ".join(f"{c} (p = {pv:.2g}" + ("; не проходит поправку Бонферрони" if pv >= bonf else "")
                         + ")" for c, pv in b_lower) or "—")
            + "; критерий Уилкоксона по 24 месяцам, p < 0.05, без поправки на множественные сравнения "
            f"({n_ext} показателей × {n_pairs} пары)). "
            f"Медианный ARI соседних месяцев A / B / C = {vv['ari_median']['A']:.3f} / "
            f"{vv['ari_median']['B']:.3f} / {vv['ari_median']['C']:.3f}; по парному критерию Уилкоксона "
            f"различие не значимо ни против B (p = {vv['p_AB']:.2f}), ни против C (p = {vv['p_AC']:.3f}); "
            "число проблемных месяцев между вариантами не сопоставляется: критерий зависит от геометрии "
            "признаков. "
            "«Прочее» (траты вне пяти категорий) входит в A неявно — сумма пяти долей меньше 1, — "
            "а «Все категории» охватывает только безналичные траты.")
    else:
        VARIANT_TEXT = ("Выбран вариант A — доли пяти категорий от «Все категории». Сравнение вариантов "
                        "(шаги 15a, 15e) появится в документе после их выполнения.")

    lines = [
        f"# 10c. Итоговая кластеризация KMeans: k = {FINAL_K} ({MONTH})",
        "",
        "Сгенерировано `src/10c_final_clustering_doc.py` из уже посчитанных результатов "
        "(шаги 10, 10b, 10d, 12a, 12; если выполнены — 14, 15a, 15e); кластеризация не "
        "пересчитывалась.",
        "",
        f"- **Канонические метки:** `{FINAL_LABELS_PATH.relative_to(PROJECT_DIR)}` (копия "
        f"`kmeans_labels_k{FINAL_K}_{TAG}.parquet`; {len(final)} МО). На этот файл ссылаются "
        "дальнейшие шаги (сравнение с Louvain, трекинг во времени).",
        f"- **Прежний канон k = {SUPERSEDED_K}** заменён {DECISION_DATE}; его метки сохранены в "
        f"`{SUPERSEDED_LABELS_PATH.relative_to(PROJECT_DIR)}`, прежний отчёт — "
        f"`10c_final_clustering_k{SUPERSEDED_K}_superseded.md`.",
        "- Признаки: 5 долей категорий расходов, z-score внутри месяца; KMeans random_state = 42, "
        "n_init = 10.",
        "",
        f"## Обоснование выбора k = {FINAL_K}",
        "",
        f"**Итог:** {JUSTIFICATION}",
        "",
        "### a) Сравнение кандидатов",
        "",
        md(comp, index=False, fmt=",.3f"),
        "",
        "SW, CH ↑ — больше лучше; S_Dbw ↓ — меньше лучше (Halkidi, центр — ближайшая к среднему "
        "точка); S_Dbw norm ↓ — относительно случайного baseline; PS ↑ — prediction strength.",
        "",
        f"- **CH/N** — CH, делённый на число объектов (N = {len(final)}), как рекомендовано в "
        f"{SOURCE_ICVI}. При фиксированном N это масштабирование: порядок кандидатов тот же, что по "
        "CH; нормировка нужна для сопоставимости между выборками разного размера.",
        "- **Baseline-нормализация S_Dbw**: для каждого k — 100 случайных "
        "разбиений тех же 2004 МО (перестановка меток реального разбиения, размеры кластеров "
        "сохраняются); S_Dbw norm = (S_Dbw − mean_random) / std_random; 95% ДИ — бутстреп "
        "(шаг 10). Принцип (нормировать индексы, чьи случайные уровни зависят от K, на среднее и дисперсию случайного разбиения) рекомендован в Shalileh et al. (2025); сама процедура в статье не описана, авторы откладывают её. Перестановка меток с сохранением размеров кластеров и бутстреп-ДИ — наши.",
        "- Ориентиры Shalileh et al. (2025, табл. 2; получены на синтетике: гауссовы блобы, "
        f"N = 5000, 10 признаков): SW ≥ {ICVI_SW_MIN}; CH/N ≥ {ICVI_CHN_MIN:.0f} (в идеале ≥ 1.5); "
        f"S_Dbw ≤ {ICVI_SDBW_MAX} (в идеале ≤ 0.2). При k = {FINAL_K}: SW {sw_f:.3f}, CH/N {chn_f:.3f}, "
        f"S_Dbw {sd[FINAL_K]:.3f} — "
        + ("ни один из трёх не достигает ориентира: выраженной кластерной структуры нет, типология — "
           "интерпретируемая сегментация. " if n_icvi_ok == 0 else
           f"ориентира достигают {n_icvi_ok} из 3. ")
        + f"Случайный уровень S_Dbw в статье около 1–1.5, у нашего baseline — {base[FINAL_K]:.3f}, "
        "поэтому для S_Dbw пороги статьи — лишь ориентир, а вывод о слабой структуре опирается на "
        "SW и CH/N.",
        "",
        "### b) SW и CH",
        "",
        f"Оба индекса максимальны при k = 2 (SW = {metrics.at[sw_best, 'SW']:.3f}, CH = "
        f"{metrics.at[ch_best, 'CH']:,.0f}) и почти монотонно убывают с ростом k. Сырые значения SW "
        "и CH нельзя напрямую сравнивать между разными k: в эталонных экспериментах источника даже "
        "для истинного разбиения CH резко падает с ростом K, а SW слабо снижается; CH к тому же "
        "растёт с N. Разбиение на 2 группы содержательно бедно, поэтому SW и CH используются как "
        "ограничение, а не как критерий выбора среди k ≥ 4.",
        "",
        "### c) S_Dbw: сырые значения и baseline-нормализация",
        "",
        f"Сырой S_Dbw: k=5 — {sd[5]:.3f}, k=6 — {sd[6]:.3f}, k=7 — {sd[7]:.3f}, k=8 — {sd[8]:.3f}. "
        f"После нормализации: k=6 — {sn[6]:.2f}, k=7 — {sn[7]:.2f}, k=8 — {sn[8]:.2f}. "
        + (f"95% ДИ для k = 6, 7, 8 перекрываются — эти k по S_Dbw статистически неразличимы. "
           if ci_overlap_678 else "")
        + f"Падение сырого S_Dbw на 6 → 7 на {base_share_67:.0%} повторяется в самом случайном "
        f"baseline (эффект числа и размеров кластеров). Наилучшее нормализованное значение — при "
        f"k = {best_norm_k}"
        + ("; k = 2–3 значимо лучше k = 6–11 — нормализованный S_Dbw, как SW и CH, смещён к малым "
           "k." if small_k_better else "."),
        "",
        "### d) Prediction strength",
        "",
        "Prediction strength — независимая проверка воспроизводимости разбиения на половинах "
        "выборки (детали — раздел ниже). Как и остальные индексы, она смещена к малым k: пороги "
        f"0.8/0.9 проходят только k = {', '.join(map(str, above[0.8])) or '—'}. В декабре 2024 она "
        f"различает соседние кандидаты (k = {FINAL_K} против {SUPERSEDED_K}, 8, 4, 5):",
        "",
        md(pd.DataFrame([{"сравнение": f"k = {a} vs k = {b}", "разница PS": ps_diff(a, b)[0],
                          "95% ДИ разницы": f"± {1.96 * ps_diff(a, b)[1]:.3f}",
                          "значимо": abs(ps_diff(a, b)[0]) > 1.96 * ps_diff(a, b)[1]}
                         for a, b in [(FINAL_K, SUPERSEDED_K), (FINAL_K, 8), (FINAL_K, 4), (FINAL_K, 5)]]),
           index=False, fmt=",.3f"),
        "",
        f"В декабре k = {SUPERSEDED_K} воспроизводится значимо хуже k = {FINAL_K}. "
        + (f"По 24 месяцам (шаг 14) картина другая: среди k ≥ 4 PS наибольшая при k = {FINAL_K} лишь в "
           f"{later['ps_months'][0]} из {later['ps_months'][1]} месяцев (месяцев с наибольшим PS: "
           + ", ".join(f"k = {k} — {n}" for k, n in later["ps_best_dist"].items()) + "), а "
           f"PS({FINAL_K}) > PS({SUPERSEDED_K}) — в {later['ps_months'][2]} из {later['ps_months'][1]}; "
           "декабрь по этому показателю нетипичен, поэтому PS не рассматривается как критерий выбора k."
           if "ps_months" in later else "Сравнение по всем месяцам — шаг 14."),
        "",
        f"### e) Воспроизводимость транспортной группы при k = {SUPERSEDED_K}",
        "",
        f"Основное содержательное отличие k = {SUPERSEDED_K} от k = {FINAL_K} — отдельная "
        f"транспортная группа (тип {tr} при k = {SUPERSEDED_K}, {len(tr_members)} МО: "
        f"{regs(tr_regions)}). Три независимые проверки воспроизводимости типов k = {SUPERSEDED_K}:",
        "",
        md(evid, fmt=",.2f"),
        "",
        f"- **Prediction strength (шаг 10d):** худшая доля воспроизводимых пар — у типа "
        f"{worst['PS']} ({ps_pairs7[worst['PS']]:.2f}); он чаще всех даёт минимум PS "
        f"({int(ps_min7.get(tr, 0))} из 100 для транспортного типа).",
        f"- **Сопоставление во времени (шаг 12, k = {SUPERSEDED_K}):** меньше всего уверенных "
        f"сопоставлений с профилем декабря у типа {worst['время']} "
        f"({int(conf_match7[worst['время']])} из {n_months}).",
        f"- **Уверенность принадлежности внутри месяца (шаг 12a, k = {SUPERSEDED_K}):** самая низкая "
        f"средняя уверенность у типа {worst['уверенность']} ({mem7['средняя'][worst['уверенность']]:.3f}; "
        f"доля МО-месяцев < 0.5 — {mem7['доля < 0.5'][worst['уверенность']]:.1%}).",
        f"- Транспортный тип ({tr}) — худший в **{n_confirm} из 3** проверок. Содержательно "
        "группа привлекательна, но её граница в признаковом пространстве статистически не "
        "держится.",
        f"- Дополнительно, с оговоркой о другом пространстве: в сети (Louvain, шаг 11c при "
        f"прежнем каноне k = {SUPERSEDED_K}) транспортный тип распадался на устойчивое ядро и "
        f"нестабильную периферию (`11c_louvain_core_stability_k{SUPERSEDED_K}_superseded.md`) — "
        "это согласуется с размытой границей, но в число подтверждений не включается.",
        "",
        f"### f) Что происходит при k = {FINAL_K}",
        "",
        f"- **Удалённая группа сохраняется:** кластер {rm['cluster']} ({rm['size']} МО, "
        f"{sig(FINAL_K, rm['cluster'])}; {regs(rm_regions)}); Жаккар с удалённой группой "
        f"k = {SUPERSEDED_K} — {rm['jaccard']:.2f} (общих МО {rm['common']}).",
        f"- **МО транспортного типа k = {SUPERSEDED_K}** распределяются по кластерам k = {FINAL_K}: "
        + "; ".join(f"кластер {c} — {n} МО ({sig(FINAL_K, c)})" for c, n in tr_split.items())
        + f". Основная часть ({int(top2.sum())} из {len(tr_members)}) — в двух соседних кластерах.",
        f"- Воспроизводимость типов k = {FINAL_K} по PS (доля верно предсказанных пар): "
        + ", ".join(f"{int(c)} — {v:.2f}" for c, v in tF.items())
        + f"; наименее воспроизводимый — тип {int(tF.idxmin())} ({sig(FINAL_K, int(tF.idxmin()))}).",
        "",
        "### g) Что дают k = 4, 5 и 8",
        "",
        "PS ниже — за декабрь 2024; по 24 месяцам соотношение другое (п. d).",
        "",
        f"- **k = 4:** PS за декабрь {ps.at[4, 'PS mean']:.2f}; удалённая группа k = {FINAL_K} "
        f"({len(rm_members)} МО) не выделяется — её МО распределены по кластерам k = 4 "
        f"({', '.join(f'кластер {c} — {n} МО' for c, n in rm4_final['split'].items())}; Жаккар с "
        f"ближайшим {rm4_final['jaccard']:.2f}).",
        f"- **k = 5:** PS за декабрь {ps.at[5, 'PS mean']:.2f}.",
        f"- **k = 8:** PS за декабрь {ps.at[8, 'PS mean']:.2f}; появляется кластер {AVERAGE_CLUSTER_K8} "
        f"({avg8} МО) с профилем «{sig(8, AVERAGE_CLUSTER_K8)}».",
        "",
        "### h) Оговорка о принципе",
        "",
        "Единого «правильного» числа типов территорий не существует — в том числе в "
        "специализированной литературе по типологии муниципальных образований. Все индексы "
        "(включая PS) смещены к малым k, внешние показатели растут с k, поэтому выбор среди k ≥ 4 "
        f"— это компромисс между детальностью типологии и воспроизводимостью, а не результат "
        f"применения критерия. Отказ от k = {SUPERSEDED_K} означает, что транспортная специализация "
        "описывается не отдельным "
        "компактным типом: "
        + ("; ".join(f"{n} МО бывшего транспортного типа входят в кластер {c} ({sig(FINAL_K, c)})"
                     for c, n in top2.items()))
        + ". Транспортный акцент, таким образом, частично сохраняется в более широком кластере, "
        "а частично растворяется в соседнем типе.",
        "",
        "## Вариант признаков",
        "",
        VARIANT_TEXT,
        "",
        "## Prediction strength (Tibshirani & Walther, 2005)",
        "",
        "2004 МО 50 раз случайно делятся на две равные половины A и B; для каждого k KMeans (те же "
        "параметры, что в шаге 10) обучается отдельно на A и на B; точки тестовой половины "
        "классифицируются по ближайшему центроиду модели другой половины; для каждого кластера "
        "тестовой половины считается доля пар точек, которые и другая модель относит в один кластер; "
        "PS разбиения — **минимум** этой доли по кластерам. Итог для k — среднее по 100 значениям "
        f"(A→B и B→A). Расчёт — шаг 10d; пороги Tibshirani & Walther (2005) — "
        f"{', '.join(map(str, PS_THRESHOLDS))}.",
        "",
        "![Prediction strength](figures/prediction_strength_by_k.png)",
        "",
        md(ps[["PS mean", "PS se", "PS sd", "PS 10%"] + [f"доля разбиений с PS ≥ {t}" for t in PS_THRESHOLDS]],
           fmt=",.3f"),
        "",
        "Prediction strength проверяет воспроизводимость разбиения на независимых подвыборках, а не "
        "компактность; критерий по минимуму и пороги 0.8–0.9 делают его строгим. Как "
        "отмечено в Akhanli & Hennig (2020, arXiv:2002.01822), высокие значения PS легче "
        "достигаются при меньшем числе кластеров, поэтому Tibshirani & Walther берут наибольшее k "
        "с PS выше 0.8–0.9. Здесь "
        "это проявляется так же, как у остальных индексов: надёжно воспроизводится только деление "
        "на 2–3 группы.",
        "",
        *flat_minimum_check(ids, L[FINAL_K].to_numpy(), FINAL_K),
        f"## Профили кластеров k = {FINAL_K}",
        "",
        f"Раздел перенесён без изменений из `{PROFILES_REPORT.name}`.",
        "",
        section_from_md(PROFILES_REPORT, f"## k = {FINAL_K}").replace(f"## k = {FINAL_K}", "", 1).strip(),
        "",
        f"## История решения: k = {SUPERSEDED_K} → k = {FINAL_K}",
        "",
        f"До {DECISION_DATE} каноническим было k = {SUPERSEDED_K}: выбор внутри статистически "
        "равноценного диапазона k = 6–8 опирался на содержательный аргумент — появление отдельной "
        f"транспортной группы. Prediction strength за декабрь (шаг 10d) показала, что k = {FINAL_K} "
        f"значимо воспроизводимее k = {SUPERSEDED_K}, а транспортная группа — "
        + ("наименее воспроизводимый тип (худший в 3 из 3 проверок: PS, трекинг во времени, "
           "уверенность принадлежности). " if n_confirm == 3 else
           f"слабо воспроизводимый тип (худший в {n_confirm} из 3 проверок). ")
        + "Последующие проверки "
        "(шаги 14–16, 15e) показали, что ни один статистический критерий не выделяет k = "
        f"{FINAL_K} однозначно; итоговое обоснование — компромисс (см. «Итог»). Канон заменён на "
        f"k = {FINAL_K}; все зависимые шаги (11, 11b, 11c, 12, 13) пересчитаны на новом каноне, "
        "прежние отчёты сохранены с суффиксом `_k7_superseded`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Метки: {FINAL_LABELS_PATH} ({len(final)} МО, k = {FINAL_K})")
    print(f"Отчёт: {REPORT_PATH}")
    print(evid.drop(columns=f"профиль (k = {SUPERSEDED_K})").round(3).to_string())
    print("worst:", worst, "; transport split:", tr_split.to_dict(), "; remote:", {k: rm[k] for k in ("cluster", "jaccard", "common")})


if __name__ == "__main__":
    main()
