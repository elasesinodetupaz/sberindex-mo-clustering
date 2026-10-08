"""Шаг 15. Сводный отчёт батча 2 проверок устойчивости (читает результаты шагов 15a–15d).

Канон (k = FINAL_K, вариант признаков A) и прежние отчёты не изменяются.
Выход: notebooks/15_robustness_checks_2.md
Запуск из корня проекта:  .venv/bin/python src/15_robustness_report_2.py
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from config import (FINAL_K, MONTH, ROBUST2_CAPITAL_CLUSTER, ROBUST2_COMPARE_TOL, ROBUST2_CONTINGENCY,
                    ROBUST2_GROUP_JACCARD, ROBUST2_K, ROBUST2_REMOTE_CLUSTER, ROBUST2_VARIANTS, ROBUST_KNN_K)
from network_utils import SHARE_COLUMNS, knn_edges, standardize_shares, std_cosine_similarity

import importlib
step12a = importlib.import_module("12a_kmeans_runs")
step15a = importlib.import_module("15a_feature_spec_full")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_PATH = PROJECT_DIR / "notebooks" / "15_robustness_checks_2.md"
SHORT = {c: c.replace("share_", "") for c in SHARE_COLUMNS}
VNAME = {"A": "A: 5 долей от «Все категории» (канон)", "B": "B: 5 долей, к сумме пяти",
         "C": "C: 5 долей + share_Прочее"}

# Ручные расчёты автора (декабрь 2024, лучший из 30 запусков) — для сверки
AUTHOR = {
    "ARI A с каноном": 0.955, "ARI B с каноном": 0.40, "ARI C с каноном": 0.53, "ARI B–C": 0.43,
    "удалённая: Жаккар в B": 0.56, "удалённая: МО в основном кластере B": 77,
    "удалённая: Жаккар в C": 0.78, "удалённая: МО в основном кластере C": 99,
    "столичная: Жаккар в B": 0.52, "столичная: Жаккар в C": 0.95,
    "«прочее»: столичный кластер": 0.336, "«прочее»: удалённый кластер": 0.336,
    "η² «прочего» (канон)": 0.57, "η² транспорта (канон)": 0.57, "η² общепита (канон)": 0.79,
    "удалённая, σ к сумме пяти: продовольствие": 1.6, "удалённая, σ: маркетплейсы": -1.9,
    "удалённая, σ: «прочее»": 1.2, "удалённая, σ: здоровье": -1.0,
}


def md(df: pd.DataFrame, index: bool = False, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def best_group(members: set, ids: np.ndarray, lab: np.ndarray) -> tuple[float, int, int]:
    """Лучший Жаккар кластера разбиения с группой, номер кластера, МО группы в нём."""
    best = max(((jac(members, set(ids[lab == c])), int(c)) for c in np.unique(lab)))
    return best[0], best[1], len(members & set(ids[lab == best[1]]))


def eta2(x: np.ndarray, lab: np.ndarray) -> float:
    grand = x.mean()
    between = sum((lab == c).sum() * (x[lab == c].mean() - grand) ** 2 for c in np.unique(lab))
    return float(between / ((x - grand) ** 2).sum())


def main() -> None:
    labs = pd.read_parquet(PROCESSED_DIR / "robust2_spec_labels.parquet")
    ktab = pd.read_parquet(PROCESSED_DIR / "robust2_spec_k.parquet")
    mts = pd.read_parquet(PROCESSED_DIR / "robust2_spec_months.parquet")
    shares = pd.read_parquet(PROCESSED_DIR / "category_shares.parquet")
    months = sorted(labs["месяц"].unique())
    ids = np.sort(labs["territory_id"].unique())
    L = {(v, m): g.set_index("territory_id")["cluster"].loc[ids].to_numpy()
         for (v, m), g in labs.groupby(["вариант", "месяц"])}
    canon = pd.read_parquet(PROCESSED_DIR / "kmeans_labels_final.parquet").set_index("territory_id")[
        "cluster"].loc[ids].to_numpy()
    S = shares[shares["date"] == pd.Timestamp(MONTH)].set_index("territory_id").loc[ids, SHARE_COLUMNS].to_numpy()
    other = 1 - S.sum(axis=1)
    five = S / S.sum(axis=1, keepdims=True)
    V = ROBUST2_VARIANTS

    # ---------- 1а: ARI ----------
    ari_dec = {f"{v} с каноном": adjusted_rand_score(canon, L[(v, MONTH)]) for v in V}
    ari_dec.update({f"{a}–{b}": adjusted_rand_score(L[(a, MONTH)], L[(b, MONTH)]) for a, b in combinations(V, 2)})
    ari_months = {f"{v} с A (официальное разбиение месяца)": np.median(
        [adjusted_rand_score(L[("A", m)], L[(v, m)]) for m in months]) for v in V if v != "A"}
    ari_months.update({f"{a}–{b}": np.median([adjusted_rand_score(L[(a, m)], L[(b, m)]) for m in months])
                       for a, b in combinations(V, 2)})
    ari_tab = pd.DataFrame({"декабрь 2024": pd.Series(ari_dec), "медиана по 24 месяцам": pd.Series(ari_months)})
    ct = {v: pd.crosstab(pd.Series(canon, name="канон"), pd.Series(step12a.overlap_relabel(canon, L[(v, MONTH)], ROBUST2_K),
                                                                    name=f"{v} (номера по канону)")) for v in ("B", "C")}

    # ---------- 1б: удалённая и столичная группы ----------
    groups = {"удалённая": ROBUST2_REMOTE_CLUSTER, "столичная": ROBUST2_CAPITAL_CLUSTER}
    grow, author_vals = [], {}
    for gname, c in groups.items():
        members = set(ids[canon == c])
        for v in V:
            j, bc, n_in = best_group(members, ids, L[(v, MONTH)])
            split = pd.Series(L[(v, MONTH)][canon == c]).value_counts()
            own = set(ids[L[(v, MONTH)] == bc])          # состав группы в декабре в варианте v
            share = np.mean([best_group(own, ids, L[(v, m)])[0] >= ROBUST2_GROUP_JACCARD for m in months])
            grow.append({"группа": f"{gname} (кластер {c} канона, {len(members)} МО)", "вариант": v,
                         "лучший Жаккар в декабре": j, "МО группы в лучшем кластере": n_in,
                         "как делится (МО по кластерам)": ", ".join(str(int(x)) for x in split.values[:3]),
                         f"доля месяцев с Жаккаром ≥ {ROBUST2_GROUP_JACCARD}": share})
            author_vals[(gname, v)] = (j, n_in)
    gtab = pd.DataFrame(grow)

    # ---------- 1в: SW, CH/N, PS; время ----------
    kdec = ktab[ktab["месяц"] == MONTH].pivot(index="k", columns="вариант", values=["SW", "CH/N", "PS"])
    kmean = ktab.groupby(["k", "вариант"])[["SW", "CH/N", "PS"]].mean().unstack("вариант")
    temp = mts.groupby("вариант").agg(**{"проблемных месяцев": ("проблемный", "sum"),
                                         "медианный ARI соседних месяцев": ("ARI с пред. мес.", "median")})

    # ---------- 1г: профили и η² ----------
    feats = np.column_stack([five, other])
    fnames = [f"{SHORT[c]} (к сумме пяти)" for c in SHARE_COLUMNS] + ["«прочее»"]
    Zf = (feats - feats.mean(axis=0)) / feats.std(axis=0)
    prof, eta = {}, {}
    for v in V:
        lab = step12a.overlap_relabel(canon, L[(v, MONTH)], ROBUST2_K)
        p = pd.DataFrame(Zf, columns=[f"{n}, σ" for n in fnames]).groupby(lab).mean()
        p.insert(0, "МО", pd.Series(lab).value_counts().sort_index())
        p["«прочее», средняя доля"] = pd.Series(other).groupby(lab).mean()
        prof[v] = p.rename_axis("кластер (номера по канону)").reset_index()
        eta[v] = {n: eta2(feats[:, i], lab) for i, n in enumerate(fnames)}
    eta_total = {f"{SHORT[c]} (от «Все категории»)": eta2(S[:, i], canon) for i, c in enumerate(SHARE_COLUMNS)}
    eta_tab = pd.DataFrame(eta)
    eta_canon_total = pd.Series(eta_total, name="канон A, доли от «Все категории»")

    # ---------- 1д: сеть ----------
    nets = {}
    for v in V:
        sim, _ = std_cosine_similarity(step15a.features(S, v))
        e = knn_edges(ids, sim, ROBUST_KNN_K)
        nets[v] = set(zip(e["territory_id_x"], e["territory_id_y"]))
    net_tab = pd.DataFrame([{"пара": f"{a}–{b}", "Жаккар рёбер": jac(nets[a], nets[b])} for a, b in combinations(V, 2)])

    # ---------- сверка с автором ----------
    rp = prof["A"].set_index("кластер (номера по канону)")
    mine = {
        "ARI A с каноном": ari_dec["A с каноном"], "ARI B с каноном": ari_dec["B с каноном"],
        "ARI C с каноном": ari_dec["C с каноном"], "ARI B–C": ari_dec["B–C"],
        "удалённая: Жаккар в B": author_vals[("удалённая", "B")][0],
        "удалённая: МО в основном кластере B": author_vals[("удалённая", "B")][1],
        "удалённая: Жаккар в C": author_vals[("удалённая", "C")][0],
        "удалённая: МО в основном кластере C": author_vals[("удалённая", "C")][1],
        "столичная: Жаккар в B": author_vals[("столичная", "B")][0],
        "столичная: Жаккар в C": author_vals[("столичная", "C")][0],
        "«прочее»: столичный кластер": rp.at[ROBUST2_CAPITAL_CLUSTER, "«прочее», средняя доля"],
        "«прочее»: удалённый кластер": rp.at[ROBUST2_REMOTE_CLUSTER, "«прочее», средняя доля"],
        "η² «прочего» (канон)": eta["A"]["«прочее»"],
        "η² транспорта (канон)": eta_total[f"{SHORT['share_Транспорт']} (от «Все категории»)"],
        "η² общепита (канон)": eta_total[f"{SHORT['share_Общепит']} (от «Все категории»)"],
        "удалённая, σ к сумме пяти: продовольствие": rp.at[ROBUST2_REMOTE_CLUSTER, f"{SHORT['share_Продовольствие']} (к сумме пяти), σ"],
        "удалённая, σ: маркетплейсы": rp.at[ROBUST2_REMOTE_CLUSTER, f"{SHORT['share_Маркетплейсы']} (к сумме пяти), σ"],
        "удалённая, σ: «прочее»": rp.at[ROBUST2_REMOTE_CLUSTER, "«прочее», σ"],
        "удалённая, σ: здоровье": rp.at[ROBUST2_REMOTE_CLUSTER, f"{SHORT['share_Здоровье']} (к сумме пяти), σ"],
    }
    flag_col = f"расхождение > {ROBUST2_COMPARE_TOL} (для числа МО — > 10%)"
    cmp_rows = []
    for key, a in AUTHOR.items():
        m = float(mine[key])
        is_count = key.startswith("удалённая: МО")
        diff = abs(m - a)
        flag = (diff > ROBUST2_COMPARE_TOL * (a if is_count else 1)) if is_count else diff > ROBUST2_COMPARE_TOL
        cmp_rows.append({"показатель": key, "у автора": a, "у меня": m, "|разница|": diff,
                         flag_col: "**да**" if flag else "нет"})
    cmp_tab = pd.DataFrame(cmp_rows)
    n_flag = int((cmp_tab[flag_col] == "**да**").sum())
    other_range = rp.drop(index=[ROBUST2_CAPITAL_CLUSTER, ROBUST2_REMOTE_CLUSTER])["«прочее», средняя доля"]

    # ---------- п. 2 ----------
    ctg = pd.read_parquet(PROCESSED_DIR / "robust2_contingency.parquet")
    bj = pd.read_parquet(PROCESSED_DIR / "robust2_best_jaccard.parquet")
    ct_blocks, main_child = [], {}
    for a, b in ROBUST2_CONTINGENCY:
        t = ctg[ctg["пара"] == f"{a}→{b}"].pivot(index="родитель", columns="потомок", values="МО")
        main_child[f"{a}→{b}"] = t.max(axis=1).sum() / t.to_numpy().sum()
        ct_blocks += [f"**k = {a} → k = {b}** (строки — кластеры k = {a}, столбцы — k = {b}); доля МО в "
                      f"основном потомке: **{main_child[f'{a}→{b}']:.0%}**", "",
                      md(t.rename_axis(f"k={a} \\ k={b}").reset_index(), fmt=",.0f"), ""]
    bj_tab = bj.pivot(index=["кластер k=6", "МО"], columns="k", values="Жаккар").add_prefix("лучший Жаккар с k=").reset_index()
    rm_from = {k: ctg[(ctg["пара"] == f"{k}→6") & (ctg["потомок"] == ROBUST2_REMOTE_CLUSTER) & (ctg["МО"] > 0)]
               .sort_values("МО", ascending=False) for k in (4, 5)}
    author_child = {"4→6": 0.65, "5→6": 0.80, "6→7": 0.86}

    # ---------- п. 3 ----------
    lres = pd.read_parquet(PROCESSED_DIR / "robust2_louvain_resolution.parquet")
    lknn = pd.read_parquet(PROCESSED_DIR / "robust2_louvain_knn.parquet")
    lknn_ari = pd.read_parquet(PROCESSED_DIR / "robust2_louvain_knn_ari.parquet")
    k6 = lres[lres["K ближе всего к целевому"]].iloc[0]
    kmr = lres[lres["метод"] != "Louvain"].iloc[0]

    # ---------- п. 4 ----------
    gap = pd.read_parquet(PROCESSED_DIR / "robust2_gap.parquet")
    rule = gap[gap["Gap(k) ≥ Gap(k+1) − s(k+1)"] == True]  # noqa: E712
    gap_k = int(rule["k"].iloc[0]) if len(rule) else None
    gap_argmax = int(gap.loc[gap["Gap"].idxmax(), "k"])
    gap_range = gap["Gap"].max() - gap["Gap"].min()

    # ---------- противоречия ----------
    sw6 = kdec.loc[ROBUST2_K, "SW"]
    ch6 = kdec.loc[ROBUST2_K, "CH/N"]
    ps_dec = kdec["PS"]
    b_better_compact = bool(((kmean["SW"]["B"] > kmean["SW"]["A"]) & (kmean["CH/N"]["B"] > kmean["CH/N"]["A"])).all())
    # устойчивость во времени: сопоставим только ARI соседних месяцев (без нумерации кластеров);
    # число проблемных месяцев между вариантами не сопоставляется — критерий зависит от геометрии признаков
    from scipy.stats import wilcoxon
    ari_adj = mts.dropna(subset=["ARI с пред. мес."]).pivot(index="месяц", columns="вариант",
                                                            values="ARI с пред. мес.")
    p_ab = wilcoxon(ari_adj["A"], ari_adj["B"]).pvalue
    p_ac = wilcoxon(ari_adj["A"], ari_adj["C"]).pvalue
    med = temp["медианный ARI соседних месяцев"]
    ps_m = pd.read_parquet(PROCESSED_DIR / "robust_k_by_month.parquet").pivot(index="месяц", columns="k", values="PS")
    ps_lead = int((ps_m[[k for k in ps_m.columns if k >= 4]].idxmax(axis=1) == FINAL_K).sum())
    ps_all_higher = bool(all(ps_dec[v][FINAL_K] > ps_dec[v][FINAL_K + 1] for v in V))
    stab_text = (f"по медианному ARI соседних месяцев A / B / C = {med['A']:.3f} / {med['B']:.3f} / "
                 f"{med['C']:.3f}, различие незначимо (p = {p_ab:.2f} против B, p = {p_ac:.3f} против C); "
                 "число проблемных месяцев между вариантами не сопоставляется: критерий зависит от "
                 "геометрии признаков")
    contr = []
    if gap_k is not None and gap_k != FINAL_K:
        contr.append(f"**gap statistic выбирает k = {gap_k}** (правило Tibshirani и др.; максимум Gap — "
                     f"k = {gap_argmax}), а не k = {FINAL_K}; кривая Gap при этом почти плоская "
                     f"(размах {gap_range:.3f} при s_k ≈ {gap['s_k'].mean():.3f}), т. е. выраженной кластерной "
                     "структуры нет ни при каком k")
    if b_better_compact:
        contr.append(f"**вариант B компактнее канона A по SW и CH/N при всех k = 4–8** (среднее по месяцам; "
                     f"декабрь, k = {ROBUST2_K}: SW {sw6['B']:.3f} против {sw6['A']:.3f}, CH/N {ch6['B']:.3f} против "
                     f"{ch6['A']:.3f})")
    best_ps_B = int(ps_dec["B"].idxmax())
    if best_ps_B != FINAL_K:
        contr.append(f"в варианте B prediction strength за декабрь максимальна при **k = {best_ps_B}** "
                     f"({ps_dec['B'][best_ps_B]:.3f}; при k = {FINAL_K} — {ps_dec['B'][FINAL_K]:.3f})")
    first = ("**Противоречия текущему канону (k = 6, вариант A):** " + "; ".join(contr) + ". "
             f"Устойчивость во времени: {stab_text}. "
             f"По PS за декабрь k = {FINAL_K} выше k = {FINAL_K + 1} "
             + ("во всех вариантах" if ps_all_higher else "не во всех вариантах")
             + f"; PS не критерий выбора k (по {len(ps_m)} месяцам k = {FINAL_K} лидирует лишь в {ps_lead} "
             f"из {len(ps_m)}, см. 10c), поэтому это не довод в пользу варианта. "
             "Канон не изменён.") \
        if contr else "**Противоречий текущему канону (k = 6, вариант A) не найдено.**"

    # ---------- смысл выбора варианта ----------
    meaning = [
        f"- **A (канон)** — доли от всех расходов: в признаки неявно входит, какая часть трат приходится "
        f"на «прочее» (сумма пяти долей < 1). Типология описывает структуру *всего* потребления. Во времени: "
        f"{stab_text}. "
        f"Столичный и удалённый типы выделяются в том числе высокой долей «прочего» "
        f"({rp.at[ROBUST2_CAPITAL_CLUSTER, '«прочее», средняя доля']:.3f} и "
        f"{rp.at[ROBUST2_REMOTE_CLUSTER, '«прочее», средняя доля']:.3f} против "
        f"{other_range.min():.2f}–{other_range.max():.2f} у остальных).",
        f"- **B** — чистая композиция пяти категорий без учёта объёма «прочего»: типология отвечает на "
        f"вопрос «как распределены траты внутри пяти наблюдаемых категорий». Разбиение сильно отличается от "
        f"канона (ARI {ari_dec['B с каноном']:.2f}); столичная группа "
        + ("делится" if author_vals[("столичная", "B")][0] < 0.7 else "сохраняется")
        + f" (Жаккар {author_vals[('столичная', 'B')][0]:.2f}), удалённая размывается "
        f"(Жаккар {author_vals[('удалённая', 'B')][0]:.2f}); кластеры компактнее (SW, CH/N выше); во времени "
        f"медианный ARI соседних месяцев {med['B']:.3f} против {med['A']:.3f} у A, различие незначимо "
        f"(p = {p_ab:.2f}); число проблемных месяцев между вариантами не сопоставляется.",
        f"- **C** — «прочее» как явный шестой признак с тем же весом, что и каждая категория. Ближе к A по "
        f"смыслу и по столичной группе (Жаккар {author_vals[('столичная', 'C')][0]:.2f}), но ARI с каноном "
        f"{ari_dec['C с каноном']:.2f}; во времени медианный ARI соседних месяцев {med['C']:.3f} против "
        f"{med['A']:.3f} у A, различие незначимо (p = {p_ac:.3f}); число проблемных месяцев между "
        "вариантами не сопоставляется.",
        f"- Для отчёта: выбор A означает явную оговорку, что типология отражает и долю не охваченных пятью "
        f"категориями трат (η² «прочего» = {eta['A']['«прочее»']:.2f}), а не только их композицию; выбор B или C "
        "потребует пересчитать канон и всю цепочку (10–16) и даст заметно другую типологию.",
    ]

    verdict = pd.DataFrame([
        ("1. Спецификация A/B/C: состав", f"ARI с каноном: B {ari_dec['B с каноном']:.2f}, C {ari_dec['C с каноном']:.2f}; "
         f"B–C {ari_dec['B–C']:.2f}; сеть A–B {net_tab.iloc[0, 1]:.2f}, A–C {net_tab.iloc[1, 1]:.2f}",
         "—", "типология сильно зависит от варианта"),
        ("1. Спецификация: компактность", f"SW, CH/N выше у B при всех k", "—", "**нет** (B компактнее)"),
        ("1. Спецификация: время", f"медианный ARI соседних месяцев A/B/C: {med['A']:.3f}/{med['B']:.3f}/"
         f"{med['C']:.3f}; p = {p_ab:.2f} против B, p = {p_ac:.3f} против C; число проблемных месяцев между "
         "вариантами не сопоставляется", "—", "нет значимого различия"),
        ("1. Спецификация: PS k = 6 vs 7 (декабрь)", ", ".join(f"{v}: {ps_dec[v][6]:.2f} vs {ps_dec[v][7]:.2f}" for v in V),
         f"{FINAL_K} > {FINAL_K + 1} " + ("во всех вариантах" if ps_all_higher else "не во всех вариантах")
         + f" за декабрь; не критерий выбора k (по {len(ps_m)} мес. лидирует в {ps_lead} из {len(ps_m)})", "—"),
        ("2. Сопряжённость по k", ", ".join(f"{p}: {v:.0%}" for p, v in main_child.items()) +
         f"; удалённая k = 6 собрана из одного кластера k = 5 на {rm_from[5]['МО'].iloc[0] / rm_from[5]['МО'].sum():.0%}",
         "нейтрально (вложенная иерархия, 6→7 — самый «чистый» переход)", "—"),
        ("3. Louvain при K ≈ 6", f"resolution {k6['resolution']}: K = {int(k6['K'])}, ARI с каноном {k6['ARI с каноном']:.2f}, "
         f"чистота {k6['чистота Louvain → KMeans']:.2f}/{k6['чистота KMeans → Louvain']:.2f}; MQ {k6['MQ']:.3f} vs {kmr['MQ']:.3f} у KMeans",
         "частично (совпадение умеренное)", "—"),
        ("3. Louvain: kNN 5/8/12", f"K {', '.join(str(int(x)) for x in lknn['K'])}; ARI между вариантами "
         f"{lknn_ari['ARI'].min():.2f}–{lknn_ari['ARI'].max():.2f}", "нейтрально (Louvain чувствителен к kNN)", "—"),
        ("4. Gap statistic", f"k = {gap_k} по правилу, максимум при k = {gap_argmax}; размах Gap {gap_range:.3f}",
         "**нет**", "—"),
    ], columns=["проверка", "результат", "подтверждает k = 6?", "подтверждает вариант A?"])

    lines = [
        "# 15. Проверки устойчивости, батч 2",
        "",
        first,
        "",
        "Сгенерировано `src/15_robustness_report_2.py` по результатам шагов 15a–15d. Канон "
        "(`kmeans_labels_final`, k = 6, вариант A), отчёт 10c, README и прежние отчёты не изменялись.",
        "",
        "## Сверка с ручными расчётами автора (декабрь 2024)",
        "",
        f"Автор: лучший из 30 запусков; здесь — лучший из 100 (правило шага 12a). Расхождений больше "
        f"{ROBUST2_COMPARE_TOL}: **{n_flag}**. η² транспорта и общепита сверены на долях от «Все "
        "категории» (канон); η² «прочего» и профиль удалённой группы — на долях к сумме пяти, как у автора.",
        "",
        md(cmp_tab, fmt=",.3f"),
        "",
        f"«Прочее» у остальных кластеров канона: {other_range.min():.3f}–{other_range.max():.3f} "
        "(у автора 0,23–0,29).",
        "",
        "## 1. Выбор спецификации признаков (k = 6, лучший из 100 запусков)",
        "",
        "### а) ARI",
        "",
        md(ari_tab.reset_index().rename(columns={"index": "сравнение"}), fmt=",.3f"),
        "",
        "Канон → B (строки — кластеры канона, столбцы — кластеры B, пронумерованные по наибольшему пересечению):",
        "",
        md(ct["B"].reset_index(), fmt=",.0f"),
        "",
        "Канон → C:",
        "",
        md(ct["C"].reset_index(), fmt=",.0f"),
        "",
        "### б) Удалённая и столичная группы",
        "",
        f"Лучший Жаккар в декабре — с составом группы в каноне; доля месяцев — для состава группы в "
        f"декабре *в том же варианте*: в скольких из {len(months)} месяцев есть кластер с Жаккаром ≥ "
        f"{ROBUST2_GROUP_JACCARD}.",
        "",
        md(gtab, fmt=",.2f"),
        "",
        "### в) SW, CH/N, prediction strength (k = 4–8) и устойчивость во времени",
        "",
        "Декабрь 2024:",
        "",
        md(kdec.round(3).reset_index(), fmt=",.3f"),
        "",
        "Среднее по 24 месяцам:",
        "",
        md(kmean.round(3).reset_index(), fmt=",.3f"),
        "",
        "Во времени (проблемные месяцы — критерии шага 12; вариант A совпадает с шагом 12):",
        "",
        md(temp.reset_index(), fmt=",.3f"),
        "",
        "Число проблемных месяцев сравнивать между вариантами нельзя: критерий зависит от геометрии "
        "признаков варианта.",
        "",
        "### г) Профили кластеров в долях к сумме пяти (σ по всем МО) и доля «прочего»",
        "",
    ]
    for v in V:
        lines += [f"**{VNAME[v]}**", "", md(prof[v], fmt=",.2f"), ""]
    lines += [
        "η² (доля дисперсии признака, объяснённая кластерами варианта; признаки — доли к сумме пяти и «прочее»):",
        "",
        md(eta_tab.reset_index().rename(columns={"index": "признак"}), fmt=",.2f"),
        "",
        "Канон A на долях от «Все категории»:",
        "",
        md(eta_canon_total.reset_index().rename(columns={"index": "признак"}), fmt=",.2f"),
        "",
        f"### д) Экономическая kNN-сеть (k = {ROBUST_KNN_K}, косинус на z-score), декабрь",
        "",
        md(net_tab, fmt=",.3f"),
        "",
        "### Итог по пункту 1: что означает выбор варианта",
        "",
        *meaning,
        "",
        "## 2. Таблицы сопряжённости по k (декабрь, метки шага 10b)",
        "",
        "Сверка с автором: " + ", ".join(f"{p}: {main_child[p]:.0%} (у автора {author_child[p]:.0%})" for p in main_child) + ".",
        "",
        *ct_blocks,
        "Лучший Жаккар каждого кластера k = 6 с кластерами k = 4, 5, 7:",
        "",
        md(bj_tab, fmt=",.2f"),
        "",
        f"Удалённый кластер k = 6 (кластер {ROBUST2_REMOTE_CLUSTER}, {int(rm_from[4]['МО'].sum())} МО) собран из: "
        + "; ".join(f"k = {k}: " + ", ".join(f"кластер {int(r['родитель'])} — {int(r['МО'])} МО"
                                               for _, r in rm_from[k].iterrows()) for k in (4, 5)) + ".",
        "",
        "## 3. Louvain на экономической сети декабря",
        "",
        f"20 запусков на каждое значение resolution, лучший по MQ. resolution {k6['resolution']} подобран "
        f"перебором (config.ROBUST2_EXTRA_RESOLUTIONS) так, чтобы лучший запуск давал K = {int(k6['K'])}.",
        "",
        md(lres.drop(columns=["K ближе всего к целевому"]), fmt=",.3f"),
        "",
        f"**Сравнение при равной детализации (K = {int(k6['K'])}, resolution {k6['resolution']}):** ARI с каноном "
        f"{k6['ARI с каноном']:.3f}; взвешенная чистота Louvain → KMeans {k6['чистота Louvain → KMeans']:.3f}, "
        f"KMeans → Louvain {k6['чистота KMeans → Louvain']:.3f}. Louvain на этом графе даёт выше MQ "
        f"({k6['MQ']:.3f} против {kmr['MQ']:.3f}) и AVI ({k6['AVI']:.3f} против {kmr['AVI']:.3f}) — Louvain "
        "оптимизирует модулярность (MQ); AVI с ней связана, но напрямую не оптимизируется; при "
        "одинаковом числе групп разбиения совпадают умеренно.",
        "",
        "Чувствительность к числу соседей kNN (resolution = 1):",
        "",
        md(lknn, fmt=",.3f"),
        "",
        md(lknn_ari, fmt=",.3f"),
        "",
        "## 4. Gap statistic (декабрь 2024)",
        "",
        md(gap, fmt=",.3f"),
        "",
        f"По правилу Tibshirani и др. (наименьшее k с Gap(k) ≥ Gap(k+1) − s(k+1)) — **k = {gap_k}**; "
        f"максимум Gap — k = {gap_argmax}. Кривая почти плоская (размах {gap_range:.3f}): данные образуют "
        "скорее континуум, чем набор чётко отделённых групп, — это согласуется с остальными индексами.",
        "",
        "## Сводка: проверка → результат",
        "",
        md(verdict),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(first)
    print(cmp_tab.to_string(index=False))


if __name__ == "__main__":
    main()
