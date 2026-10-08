"""Шаг 14. Сводный отчёт проверок устойчивости выбора k (читает результаты шагов 14a, 14b).

Проверки 4 (Louvain при разных resolution / kNN) и 5 (gap statistic) в этом прогоне не
выполнялись: по правилу батча выполнение остановлено, так как проверка 2 противоречит части
обоснования k = FINAL_K (см. первый абзац отчёта).

Выход: notebooks/14_robustness_checks.md, notebooks/figures/robust_ps_by_month.png
Запуск из корня проекта:  .venv/bin/python src/14_robustness_report.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FINAL_K, MONTH, ROBUST_DECOMP_K, ROBUST_SE_MULT, SUPERSEDED_K

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_PATH = PROJECT_DIR / "notebooks" / "14_robustness_checks.md"
FIG_PATH = PROJECT_DIR / "notebooks" / "figures" / "robust_ps_by_month.png"


def md(df: pd.DataFrame, index: bool = False, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def main() -> None:
    spec = pd.read_parquet(PROCESSED_DIR / "robust_feature_spec.parquet")
    spec_ps = pd.read_parquet(PROCESSED_DIR / "robust_feature_spec_ps.parquet")
    s = pd.read_parquet(PROCESSED_DIR / "robust_k_by_month.parquet")
    cl = pd.read_parquet(PROCESSED_DIR / "robust_ps_clusters_by_month.parquet")
    F, S = FINAL_K, SUPERSEDED_K

    # --- п. 1 ---
    psw = spec_ps.pivot(index="вариант", columns="k", values="PS")
    pse = spec_ps.pivot(index="вариант", columns="k", values="PS se")
    spec_tab = spec.set_index("вариант").join(pd.DataFrame({
        f"PS k={F}": psw[F], f"PS k={S}": psw[S], "разница": psw[F] - psw[S],
        "разница / SE": (psw[F] - psw[S]) / np.hypot(pse[F], pse[S])}))
    spec_pref = bool((spec_tab["разница / SE"] > ROBUST_SE_MULT).all())
    others = spec_tab.iloc[1:]

    # --- п. 2 ---
    P = s.pivot(index="месяц", columns="k", values="PS")
    SE = s.pivot(index="месяц", columns="k", values="PS se")
    Pm = s.pivot(index="месяц", columns="k", values="PS (среднее по кластерам)")
    SEm = s.pivot(index="месяц", columns="k", values="PS (среднее по кластерам) se")
    SW = s.pivot(index="месяц", columns="k", values="SW")
    SN = s.pivot(index="месяц", columns="k", values="S_Dbw norm")
    ge4 = [k for k in P.columns if k >= 4]
    best_ps = P[ge4].idxmax(axis=1)
    best_sw = SW[ge4].idxmax(axis=1)
    best_sn = SN[ge4].idxmin(axis=1)
    d, dse = P[F] - P[S], np.hypot(SE[F], SE[S])
    dm, dmse = Pm[F] - Pm[S], np.hypot(SEm[F], SEm[S])
    n = len(P)
    n_best_F = int((best_ps == F).sum())
    months_best_F = best_ps.index[best_ps == F].tolist()
    by_month = pd.DataFrame({
        f"PS k={F}": P[F], f"PS k={S}": P[S], "разница / SE": d / dse,
        "k с макс. PS (k ≥ 4)": best_ps, "k с макс. SW (k ≥ 4)": best_sw,
        "k с мин. S_Dbw norm (k ≥ 4)": best_sn}).reset_index()
    count_tab = pd.DataFrame([
        {"агрегация PS": "минимум по кластерам (стандарт)", f"PS({F}) > PS({S})": int((d > 0).sum()),
         f"больше чем на {ROBUST_SE_MULT:.0f} SE": int((d > ROBUST_SE_MULT * dse).sum()),
         f"PS({S}) > PS({F}) больше чем на {ROBUST_SE_MULT:.0f} SE": int((-d > ROBUST_SE_MULT * dse).sum())},
        {"агрегация PS": "среднее по кластерам", f"PS({F}) > PS({S})": int((dm > 0).sum()),
         f"больше чем на {ROBUST_SE_MULT:.0f} SE": int((dm > ROBUST_SE_MULT * dmse).sum()),
         f"PS({S}) > PS({F}) больше чем на {ROBUST_SE_MULT:.0f} SE": int((-dm > ROBUST_SE_MULT * dmse).sum())},
    ])
    mean_k = pd.DataFrame({"PS (минимум), среднее по месяцам": P.mean(),
                           "PS (среднее по кластерам), среднее по месяцам": Pm.mean(),
                           "SW, среднее": SW.mean(), "S_Dbw norm, среднее": SN.mean()}).reset_index()
    dec_rank = {k: float(P.loc[MONTH, k] - P[k].mean()) / float(P[k].std()) for k in ge4}

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, row in P.iterrows():
        ax.plot(row.index, row.values, color="#4C72B0" if m != MONTH else "#C44E52",
                lw=0.6 if m != MONTH else 1.8, alpha=0.35 if m != MONTH else 1,
                label=f"{MONTH}" if m == MONTH else None)
    ax.plot(P.columns, P.mean(), color="black", lw=2.2, marker="o", label="среднее по 24 месяцам")
    for t in (0.8, 0.9):
        ax.axhline(t, color="#999999", ls="--", lw=0.8)
    ax.set(xlabel="k", ylabel="prediction strength (минимум по кластерам)", xticks=list(P.columns),
           ylim=(0, 1.02), title="Prediction strength по месяцам (тонкие линии — месяцы)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=120)
    plt.close(fig)

    # --- п. 3: декабрь, по кластерам ---
    dec = cl[(cl["месяц"] == MONTH) & cl["k"].isin(ROBUST_DECOMP_K)]
    dec_tab = pd.DataFrame({
        "PS (минимум)": P.loc[MONTH, ROBUST_DECOMP_K], "PS (среднее по кластерам)": Pm.loc[MONTH, ROBUST_DECOMP_K],
        "доли пар по кластерам (средние, по возрастанию)": [
            ", ".join(f"{v:.2f}" for v in sorted(dec[dec["k"] == k].groupby("кластер")["доля пар"].mean()))
            for k in ROBUST_DECOMP_K]}).rename_axis("k").reset_index()
    best_mean_dec = int(Pm.loc[MONTH, ROBUST_DECOMP_K].idxmax())

    contradiction = n_best_F < n / 2
    verdict_rows = [
        ("1. Спецификация признаков (A/B/C)",
         f"PS(k={F}) > PS(k={S}) во всех вариантах (разница {', '.join(f'{v:.1f}' for v in spec_tab['разница / SE'])} SE); "
         f"но разбиение k = {F} меняется: ARI с каноном {', '.join(f'{v:.2f}' for v in others['ARI с каноном'])} "
         f"(B, C), в другом кластере {', '.join(f'{v:.0%}' for v in others['доля МО в другом кластере'])} МО",
         "да — выбор k" if spec_pref else "нет",
         "**нет** — состав типов чувствителен к определению долей"),
        ("2. Выбор k во времени (24 мес., PS)",
         f"k = {F} — максимум PS среди k ≥ 4 в {n_best_F} из {n} месяцев ({', '.join(months_best_F) or '—'}); "
         f"в остальных — k = 4 или 5. PS({F}) > PS({S}) в {int((d > 0).sum())} из {n} "
         f"(> {ROBUST_SE_MULT:.0f} SE — в {int((d > ROBUST_SE_MULT * dse).sum())}; обратное значимо — в "
         f"{int((-d > ROBUST_SE_MULT * dse).sum())})",
         f"**нет** — «k = {F} лучше всех k ≥ 4» верно только для {MONTH}",
         f"да, в большинстве месяцев — «k = {F} лучше k = {S}»"),
        ("2. SW и S_Dbw norm во времени",
         f"среди k ≥ 4 SW максимален при k = {best_sw.mode()[0]} во всех {int((best_sw == best_sw.mode()[0]).sum())} "
         f"месяцах; S_Dbw norm минимален при k = {best_sn.mode()[0]} в {int((best_sn == best_sn.mode()[0]).sum())}",
         "нет (смещены к малым k, как и ожидалось)", "—"),
        (f"3. Агрегация PS (минимум vs среднее)",
         f"{MONTH}: по среднему по кластерам максимум среди k = 4–8 — k = {best_mean_dec}; по месяцам "
         f"PS({F}) > PS({S}) по среднему — в {int((dm > 0).sum())} из {n} (значимо — "
         f"{int((dm > ROBUST_SE_MULT * dmse).sum())}, обратное — {int((-dm > ROBUST_SE_MULT * dmse).sum())})",
         f"да для {F} vs {S}; не зависит от агрегации по минимуму в {MONTH}", "—"),
        ("4. Louvain: resolution, kNN", "не выполнялось — остановлено по правилу батча", "—", "—"),
        ("5. Gap statistic", "не выполнялось — остановлено по правилу батча", "—", "—"),
    ]
    verdict = pd.DataFrame(verdict_rows, columns=["проверка", "результат", f"подтверждает k = {F}?",
                                                  "дополнительно"])

    lines = [
        "# 14. Проверки устойчивости выбора k",
        "",
        "Сгенерировано `src/14_robustness_report.py` по результатам `src/14a_feature_spec_check.py` и "
        "`src/14b_k_over_time.py`. Предыдущие отчёты не изменялись; канон не менялся.",
        "",
    ]
    if contradiction:
        lines += [
            f"> **СТОП: проверка 2 противоречит части обоснования k = {F}.** Максимум prediction "
            f"strength среди k ≥ 4 приходится на k = {F} только в **{n_best_F} из {n}** месяцев "
            f"({', '.join(months_best_F) or '—'}); в остальных — k = 4 "
            f"({int((best_ps == 4).sum())} мес.) или k = 5 ({int((best_ps == 5).sum())} мес.). Средний PS "
            f"по месяцам: k = 4 — {P[4].mean():.3f}, k = 5 — {P[5].mean():.3f}, k = {F} — "
            f"{P[F].mean():.3f}, k = {S} — {P[S].mean():.3f}. {MONTH} нетипичен именно для малых k: "
            f"его PS при k = 4 и 5 ниже среднего по месяцам на {abs(dec_rank[4]):.1f} и "
            f"{abs(dec_rank[5]):.1f} ст. откл. Утверждение 10c «k = {F} лучше всех k ≥ 4» верно "
            f"только для {MONTH}. Сравнение k = {F} vs k = {S} в основном держится: PS({F}) > PS({S}) "
            f"в {int((d > 0).sum())} из {n} месяцев (значимо — в {int((d > ROBUST_SE_MULT * dse).sum())}), "
            f"но в {int((-d > ROBUST_SE_MULT * dse).sum())} месяцах значимо лучше k = {S}. "
            "По правилу батча проверки 4 и 5 не выполнялись; канон не изменён — решение за автором.",
            "",
        ]
    lines += [
        "## Сводка",
        "",
        md(verdict),
        "",
        "## 1. Чувствительность к спецификации признаков (декабрь 2024)",
        "",
        "A — 5 долей от «Все категории» (текущий); B — 5 долей, нормированных к сумме пяти; C — 5 долей "
        "+ share_Прочее. Во всех вариантах z-score внутри месяца; KMeans — лучший по inertia из 100 "
        "запусков; ARI и доля МО в другом кластере — относительно `kmeans_labels_final`; сеть — "
        "z-score + cosine + kNN8.",
        "",
        md(spec_tab.reset_index(), fmt=",.3f"),
        "",
        f"- Предпочтение k = {F} над k = {S} по PS сохраняется во всех вариантах, хотя в B разница "
        f"меньше ({spec_tab['разница'].iloc[1]:+.3f}).",
        f"- **Сам состав кластеров k = {F} сильно зависит от определения долей:** при нормировке к сумме "
        f"пяти (B) ARI с каноном {spec_tab['ARI с каноном'].iloc[1]:.2f}, в другом кластере "
        f"{spec_tab['доля МО в другом кластере'].iloc[1]:.0%} МО; экономическая сеть совпадает с текущей "
        f"лишь на {spec_tab['Жаккар рёбер сети с текущей'].iloc[1]:.0%} рёбер. Доля «прочих» расходов "
        "(не входящих в 5 категорий) — самостоятельная ось различий между МО; выбор знаменателя долей "
        "меняет, что считается «похожими» МО.",
        "",
        "## 2. Выбор k во времени (24 месяца, k = 2..10)",
        "",
        "![PS по месяцам](figures/robust_ps_by_month.png)",
        "",
        md(count_tab),
        "",
        "Средние по месяцам:",
        "",
        md(mean_k, fmt=",.3f"),
        "",
        "По месяцам:",
        "",
        md(by_month, fmt=",.3f"),
        "",
        f"## 3. Разложение prediction strength ({MONTH}, k = {ROBUST_DECOMP_K[0]}–{ROBUST_DECOMP_K[-1]})",
        "",
        md(dec_tab, fmt=",.3f"),
        "",
        f"- В {MONTH} предпочтение k = {F} не зависит от агрегации по минимуму: по среднему по "
        f"кластерам k = {F} тоже максимален среди k = 4–8 "
        f"({Pm.loc[MONTH, F]:.3f})." if best_mean_dec == F else
        f"- В {MONTH} по среднему по кластерам максимум среди k = 4–8 — k = {best_mean_dec}.",
        f"- По всем месяцам агрегация влияет слабо: по среднему по кластерам PS({F}) > PS({S}) в "
        f"{int((dm > 0).sum())} из {n} месяцев против {int((d > 0).sum())} по минимуму. По среднему по "
        f"кластерам PS тоже убывает с k (среднее по месяцам: k = 4 — {Pm[4].mean():.3f}, "
        f"k = {F} — {Pm[F].mean():.3f}).",
        "",
        "## 4–5. Не выполнялись",
        "",
        "Louvain при разных resolution и kNN, gap statistic — остановлено по правилу батча после "
        "противоречия в проверке 2.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
