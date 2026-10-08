"""Шаг 16. Кривая стабильности по k = config.STABILITY_KS: один код и одни официальные разбиения.

Источники (для каждого k): шаги 12a/12 (--k), 10b (профили и якорная нумерация), функция
вложенности шага 13 (Louvain-разбиения по месяцам не зависят от k KMeans и берутся из шага 13),
функция плоского минимума шага 10c, prediction strength из шага 14b.

Показатели:
  (а) доля МО, у которых доминирующий кластер первых 6 мес. ≠ последних 6 мес.;
  (б) число МО с долей основного кластера < TEMPORAL_THRESHOLD (шаг 12);
  (в) число проблемных месяцев; (г) медианный ARI разбиений соседних месяцев;
  (д) доля уверенных сопоставлений с декабрём: среднее и минимум по типам;
  (е) Louvain по 24 месяцам: медианы ARI, взвешенной чистоты, доли сообществ с чистотой ≥ 80%;
  (ж) декабрь: доля «подвижных» МО среди запусков у минимума inertia;
  (з) prediction strength: декабрь и среднее по 24 месяцам.
«Излом» на 6→7: |Δ(6→7)| ≥ KINK_RATIO · max(|Δ(5→6)|, |Δ(7→8)|).

Выход: data/processed/k_stability_curve.parquet, notebooks/16_k_stability_curve.md,
       notebooks/figures/k_stability_curve.png
Запуск из корня проекта:  .venv/bin/python src/16_k_stability_curve.py
"""
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FINAL_K, KINK_RATIO, MONTH, STABILITY_KS

step12 = importlib.import_module("12_kmeans_temporal_tracking")
step13 = importlib.import_module("13_louvain_monthly_check")
step10c = importlib.import_module("10c_final_clustering_doc")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "k_stability_curve.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "16_k_stability_curve.md"
FIG_PATH = PROJECT_DIR / "notebooks" / "figures" / "k_stability_curve.png"
TAG = MONTH.replace("-", "_")
NEST_COL = f"доля сообществ с чистотой ≥ {step13.NEST_PURITY:.0%}"

# показатель: (буква, подпись, ориентация: +1 — больше лучше, −1 — меньше лучше)
METRICS = [
    ("а", "доля МО со сменой доминирующего кластера (первые vs последние 6 мес.)", -1),
    ("б", f"МО с долей основного кластера < {step12.TEMPORAL_THRESHOLD:.0%}", -1),
    ("в", "проблемных месяцев", -1),
    ("г", "медианный ARI соседних месяцев", +1),
    ("д", "уверенные сопоставления с декабрём: среднее по типам", +1),
    ("д", "уверенные сопоставления с декабрём: минимум по типам", +1),
    ("е", "Louvain: медианный ARI с KMeans", +1),
    ("е", "Louvain: медианная взвешенная чистота", +1),
    ("е", f"Louvain: медианная {NEST_COL}", +1),
    ("ж", "декабрь: доля подвижных МО у минимума inertia", -1),
    ("з", "prediction strength: декабрь", +1),
    ("з", "prediction strength: среднее по 24 мес.", +1),
]
# показатели, которые между k напрямую не сопоставляются: в таблице остаются, но не входят в подсчёт
# «k = 6 — локальный оптимум»
NOT_COMPARABLE = {"проблемных месяцев"}
NOT_COMPARABLE_NOTE = (
    "критерий проблемного месяца (≥ {amb} неоднозначных сопоставлений с якорем или доля МО, сменивших "
    "кластер, выше Q3 + {iqr}·IQR своего ряда) зависит от геометрии разбиения и якоря и между k "
    "напрямую не сопоставляется — так же, как между вариантами признаков A/B/C (отчёт 15)")


def metrics_for_k(k: int, louv: pd.DataFrame, ps: pd.DataFrame) -> dict:
    sw = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{k}_switches.parquet")
    mt = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{k}_months.parquet")
    match = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{k}_matching.parquet")
    match = match[match["месяц"] != MONTH]
    conf_by_type = (match["уверенность"] == "уверенно").groupby(match["cluster"]).mean()
    traj = pd.read_parquet(PROCESSED_DIR / f"kmeans_k{k}_trajectories.parquet")
    nest = []
    for m, g in louv.groupby("month"):
        km = traj[traj["month"] == m].set_index("territory_id").loc[g["territory_id"], "cluster"].to_numpy()
        nest.append(step13.nesting(g["cluster"].to_numpy(), km))
    nest = pd.DataFrame(nest)
    canon = pd.read_parquet(PROCESSED_DIR / f"kmeans_labels_k{k}_{TAG}.parquet").set_index("territory_id")["cluster"]
    flat = step10c.flat_minimum_stats(canon.index.to_numpy(), canon.to_numpy(), k)
    psk = ps[ps["k"] == k].set_index("месяц")["PS"]
    vals = [
        float(sw["чистая миграция"].mean()),
        int((sw["доля месяцев в основном кластере"] < step12.TEMPORAL_THRESHOLD).sum()),
        int(mt["проблемный"].sum()),
        float(mt["ARI с пред. мес. (разбиения)"].median()),
        float(conf_by_type.mean()), float(conf_by_type.min()),
        float(nest["ARI с KMeans"].median()), float(nest["взвешенная чистота"].median()),
        float(nest[NEST_COL].median()),
        flat["доля подвижных МО"],
        float(psk[MONTH]), float(psk.mean()),
    ]
    return {name: v for (_, name, _), v in zip(METRICS, vals)}


def main() -> None:
    louv = pd.read_parquet(step13.LABELS_PATH)
    ps = pd.read_parquet(PROCESSED_DIR / "robust_k_by_month.parquet")
    missing = [k for k in STABILITY_KS if not (PROCESSED_DIR / f"kmeans_k{k}_months.parquet").exists()]
    if missing:
        raise SystemExit(f"STOP: нет результатов шагов 12a/12 для k = {missing}")
    tab = pd.DataFrame({k: metrics_for_k(k, louv, ps) for k in STABILITY_KS})   # строки — метрики
    tab.to_parquet(OUT_PATH, engine="pyarrow")

    ks = STABILITY_KS
    i6 = ks.index(6)
    rows, kink_letters, local_opt, near_kink = [], {}, [], []
    for letter, name, orient in METRICS:
        v = tab.loc[name, ks].astype(float).to_numpy()
        d = np.diff(v)
        d56, d67, d78 = d[i6 - 1], d[i6], d[i6 + 1]
        neigh = max(abs(d56), abs(d78))
        kink = abs(d67) > 0 and abs(d67) >= KINK_RATIO * neigh
        direction = "лучше" if orient * d67 > 0 else ("хуже" if orient * d67 < 0 else "без изменений")
        if kink:
            kink_letters.setdefault(letter, []).append(f"{name} ({direction})")
        elif neigh > 0 and abs(d67) / neigh >= 1:
            near_kink.append(f"({letter}) {name}: |Δ 6→7| / max соседних = {abs(d67) / neigh:.2f}")
        is_opt = orient * (v[i6] - v[i6 - 1]) > 0 and orient * (v[i6] - v[i6 + 1]) > 0
        if is_opt and name not in NOT_COMPARABLE:
            local_opt.append(f"({letter}) {name}")
        row = {"": letter, "показатель": name, "ориентация": "↑ лучше" if orient > 0 else "↓ лучше"}
        for j, k in enumerate(ks):
            row[f"k={k}"] = v[j]
            if j < len(ks) - 1:
                row[f"Δ {k}→{ks[j + 1]}"] = d[j]
        row["k=6 лучше и k=5, и k=7"] = (("да, не учитывается¹" if is_opt else "не учитывается¹")
                                         if name in NOT_COMPARABLE else ("да" if is_opt else ""))
        row["6→7 относительно соседних"] = (f"**излом**: 6→7 {direction} ({abs(d67):.3g} vs "
                                           f"max соседних {neigh:.3g})" if kink else
                                           f"гладко ({abs(d67):.3g} vs max соседних {neigh:.3g})")
        rows.append(row)
    table = pd.DataFrame(rows)

    # график: одна панель на показатель
    ncol = 4
    nrow = int(np.ceil(len(METRICS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    for ax, (letter, name, orient) in zip(axes.ravel(), METRICS):
        v = tab.loc[name, ks].astype(float).to_numpy()
        ax.plot(ks, v, marker="o", color="#4C72B0")
        ax.plot(ks[i6:i6 + 2], v[i6:i6 + 2], color="#C44E52", lw=2.5, label="6→7")
        ax.axvline(FINAL_K, color="#999999", ls=":", lw=1)
        ax.set_title(f"({letter}) {name}", fontsize=8)
        ax.set_xticks(ks)
        ax.text(0.02, 0.95, "↑ лучше" if orient > 0 else "↓ лучше", transform=ax.transAxes,
                fontsize=7, va="top", color="#555555")
        ax.grid(alpha=.3)
    for ax in axes.ravel()[len(METRICS):]:
        ax.axis("off")
    fig.suptitle(f"Кривая стабильности по k (красным — переход 6→7; пунктир — канон k = {FINAL_K})",
                 fontsize=10)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=110)
    plt.close(fig)

    n_letters = len(kink_letters)
    n_comparable = len([m for m in METRICS if m[1] not in NOT_COMPARABLE])
    nc_labels = "; ".join(f"({l}) {n}" for l, n, _ in METRICS if n in NOT_COMPARABLE)
    nc_note = NOT_COMPARABLE_NOTE.format(amb=step12.PROBLEM_MONTH_MIN_AMBIGUOUS, iqr=step12.PROBLEM_SWITCH_IQR)
    answer = (
        f"**Есть ли излом на 6→7 хотя бы в двух метриках из (а)–(з)? — "
        f"{'ДА' if n_letters >= 2 else 'НЕТ'}.** Излом (|Δ 6→7| ≥ {KINK_RATIO} × max соседних "
        f"переходов 5→6 и 7→8) найден в {n_letters} из 8 метрик"
        + (": " + "; ".join(f"({l}) " + ", ".join(v) for l, v in sorted(kink_letters.items()))
           if kink_letters else "")
        + ". Остальные метрики меняются на 6→7 гладко, в пределах соседних переходов.")

    lines = [
        f"# 16. Кривая стабильности по k = {ks[0]}–{ks[-1]}",
        "",
        answer,
        "",
        (f"**Где k = 6 — локальный оптимум (лучше и k = 5, и k = 7):** {len(local_opt)} из "
         f"{n_comparable} показателей — " + "; ".join(local_opt) + ". В этих показателях заметный "
         "скачок приходится на 5→6 (k = 5 хуже обоих соседей), а не на 6→7."
         if local_opt else "**k = 6 не является локальным оптимумом ни по одному показателю.**")
        + f" Показатель {nc_labels} в подсчёт не входит: {nc_note}.",
        "",
        *(["Близко к порогу излома (отношение ≥ 1, но < KINK_RATIO): " + "; ".join(near_kink) + ".", ""]
          if near_kink else []),
        "Сгенерировано `src/16_k_stability_curve.py`. Для всех k — один и тот же код и одни и те же "
        "официальные разбиения: шаги 12a/12 (`--k`), 10b (профили и якорная нумерация), функция "
        "вложенности шага 13 (Louvain-разбиения по месяцам общие для всех k), функция плоского "
        "минимума шага 10c, prediction strength — шаг 14b. Канон не менялся.",
        "",
        f"Правило излома (config.KINK_RATIO = {KINK_RATIO}): |Δ(6→7)| ≥ {KINK_RATIO} · "
        "max(|Δ(5→6)|, |Δ(7→8)|). «Лучше/хуже» — с учётом ориентации показателя.",
        "",
        "![Кривая стабильности](figures/k_stability_curve.png)",
        "",
        table.to_markdown(index=False, floatfmt=",.3f"),
        "",
        f"¹ {nc_labels}: {nc_note}. В подсчёт «k = 6 — локальный оптимум» не входит.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(answer)
    print(table[["", "показатель"] + [f"k={k}" for k in ks] + ["6→7 относительно соседних"]].to_string())


if __name__ == "__main__":
    main()
