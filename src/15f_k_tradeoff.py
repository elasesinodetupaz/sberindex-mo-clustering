"""Шаг 15f. Иллюстрация компромисса по k: устойчивость во времени против связи с внешними данными.

Вариант A, k = 4..8. Две кривые, каждая приведена к шкале 0–1 (min-max по k):
  - стабильность — медианный ARI разбиений соседних месяцев (шаг 12, kmeans_k{k}_months.parquet);
  - внешняя связь — среднее по 24 месяцам z рёбер highway (панель 1) или AMI с регионами (панель 2),
    шаг 15e (robust2_external_validity.parquet).
Иллюстрация, а не критерий выбора k. Ничего, кроме рисунка и раздела в отчёте 15e (между метками
<!-- 15f:start --> и <!-- 15f:end -->), не изменяется. Запускать после шага 15e.

Выход: notebooks/figures/k_tradeoff.png, раздел в notebooks/15e_external_validity.md
Запуск из корня проекта:  .venv/bin/python src/15f_k_tradeoff.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FINAL_K

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
EXT_PATH = PROCESSED_DIR / "robust2_external_validity.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "15e_external_validity.md"
FIG_PATH = PROJECT_DIR / "notebooks" / "figures" / "k_tradeoff.png"
from config import STABILITY_KS as KS  # вариант A, официальные метки шага 12a
MARK_START, MARK_END = "<!-- 15f:start -->", "<!-- 15f:end -->"
EXTERNAL = [("z рёбер highway", "внешняя связь: z рёбер highway"),
            ("AMI регион", "внешняя связь: AMI с регионами")]
STABILITY_LABEL = "стабильность: медианный ARI соседних месяцев"


def minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())


def main() -> None:
    stab = pd.Series({k: pd.read_parquet(PROCESSED_DIR / f"kmeans_k{k}_months.parquet")[
        "ARI с пред. мес. (разбиения)"].median() for k in KS})
    ext = pd.read_parquet(EXT_PATH)
    ext = ext[ext["набор"] == "k"].assign(k=lambda d: d["разбиение"].str[2:].astype(int))
    ext_mean = ext.groupby("k")[[c for c, _ in EXTERNAL]].mean().loc[KS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
    for ax, (col, label) in zip(axes, EXTERNAL):
        for k in KS:
            ax.axvline(k, color="#DDDDDD", lw=0.8, zorder=0)
        ax.axvspan(FINAL_K - 0.12, FINAL_K + 0.12, color="#F2D7D5", zorder=0, label=f"k = {FINAL_K} (канон)")
        s1, s2 = minmax(stab), minmax(ext_mean[col])
        ax.plot(KS, s1, marker="o", color="#4C72B0", label=STABILITY_LABEL)
        ax.plot(KS, s2, marker="s", color="#DD8452", label=label)
        for k in KS:
            ax.annotate(f"{stab[k]:.3f}", (k, s1[k]), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color="#4C72B0")
            ax.annotate(f"{ext_mean.at[k, col]:.3g}", (k, s2[k]), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7, color="#DD8452")
        ax.set(xlabel="k", xticks=KS, ylim=(-0.12, 1.12), title=label.replace("внешняя связь: ", "Стабильность и "))
        ax.grid(alpha=.25, axis="y")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    axes[0].set_ylabel("шкала 0–1 (min-max по k = 4–8)")
    fig.suptitle("Компромисс по k: иллюстрация, а не критерий выбора (подписи — исходные значения)", fontsize=10)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=120)
    plt.close(fig)

    tab = pd.DataFrame({"k": KS, "медианный ARI соседних месяцев": stab.values,
                        "стабильность (0–1)": minmax(stab).values})
    for col, _ in EXTERNAL:
        tab[f"{col}, среднее"] = ext_mean[col].values
        tab[f"{col} (0–1)"] = minmax(ext_mean[col]).values
    section = [
        MARK_START,
        "## Компромисс по k (иллюстрация)",
        "",
        "Сгенерировано `src/15f_k_tradeoff.py`. Вариант A, k = 4–8. Стабильность — медианный ARI "
        "разбиений соседних месяцев (шаг 12); внешняя связь — среднее по 24 месяцам z рёбер highway или "
        "AMI с регионами (таблицы выше). Каждая кривая приведена к шкале 0–1 min-max по k.",
        "",
        "![Компромисс по k](figures/k_tradeoff.png)",
        "",
        tab.to_markdown(index=False, floatfmt=",.3f"),
        "",
        "**Как читать.** Это иллюстрация компромисса, а не критерий выбора k. С ростом k разбиения "
        "соседних месяцев совпадают хуже, а связь кластеров с внешними данными (регион, транспортная "
        "близость) усиливается — отчасти механически, из-за большего числа групп. Шкалы 0–1 построены "
        "по диапазону k = 4–8 и произвольны: положение точки пересечения кривых и «расстояния» между "
        f"ними зависят от выбранного диапазона и нормировки и сами по себе не указывают на оптимальное "
        f"k. На рисунке видно лишь, что k = {FINAL_K} находится внутри этого компромисса, а не на "
        "одном из его краёв.",
        MARK_END,
    ]
    text = REPORT_PATH.read_text(encoding="utf-8")
    if MARK_START in text and MARK_END in text:
        head, rest = text.split(MARK_START, 1)
        text = head.rstrip() + "\n\n" + "\n".join(section) + rest.split(MARK_END, 1)[1]
    else:
        text = text.rstrip() + "\n\n" + "\n".join(section) + "\n"
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
