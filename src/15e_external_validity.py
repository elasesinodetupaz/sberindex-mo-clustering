"""Шаг 15e. Внешняя валидность вариантов признаков A/B/C (k = 6) и выбора k (вариант A, k = 4..8).

Внешние данные (не входили в признаки): log(market_access), region_code (справочник МО),
граф highway kNN8 (шаг 07). Для каждого разбиения (24 месяца):
  а) R², скорректированный R² и ω² для log(market_access) по кластерам (однофакторный ANOVA);
  б) AMI между кластерами и region_code;
  в) доля рёбер highway kNN8 с концами в одном кластере и z-оценка относительно EXTVAL_PERMUTATIONS
     перестановок меток (число кластеров и их размеры сохраняются).
Сравнения: парный критерий Уилкоксона по 24 месяцам (месяцы не независимы — p-значения ориентировочные).

Выход: data/processed/robust2_external_validity.parquet, notebooks/15e_external_validity.md
Запуск из корня проекта:  .venv/bin/python src/15e_external_validity.py
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_mutual_info_score

from config import (EXTVAL_ALPHA, EXTVAL_KS, EXTVAL_PERMUTATIONS, EXTVAL_SEED, FINAL_K, MONTH,
                    ROBUST2_K, ROBUST2_VARIANTS)

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
SPEC_LABELS_PATH = PROCESSED_DIR / "robust2_spec_labels.parquet"
HIGHWAY_PATH = PROCESSED_DIR / "transport_network_highway_knn8.parquet"
OUT_PATH = PROCESSED_DIR / "robust2_external_validity.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "15e_external_validity.md"

METRICS = [("adj R² log(MA)", "скорректированный R² для log(market_access)"),
           ("ω² log(MA)", "ω² для log(market_access)"),
           ("AMI регион", "AMI кластеров и region_code"),
           ("доля рёбер highway внутри", "доля рёбер highway kNN8 внутри кластеров"),
           ("z рёбер highway", "z-оценка доли рёбер highway внутри кластеров против перестановок")]


def md(df: pd.DataFrame, index: bool = False, fmt: str = ",.3f") -> str:
    return df.to_markdown(index=index, floatfmt=fmt)


def anova(y: np.ndarray, lab: np.ndarray) -> tuple[float, float, float]:
    n, k = len(y), len(np.unique(lab))
    grand = y.mean()
    sst = ((y - grand) ** 2).sum()
    ssb = sum((lab == c).sum() * (y[lab == c].mean() - grand) ** 2 for c in np.unique(lab))
    msw = (sst - ssb) / (n - k)
    r2 = ssb / sst
    return r2, 1 - (1 - r2) * (n - 1) / (n - k), (ssb - (k - 1) * msw) / (sst + msw)


def edge_stats(lab: np.ndarray, ei: np.ndarray, ej: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    obs = float((lab[ei] == lab[ej]).mean())
    perm = np.array([(p[ei] == p[ej]).mean() for p in (rng.permutation(lab) for _ in range(EXTVAL_PERMUTATIONS))])
    return obs, float((obs - perm.mean()) / perm.std(ddof=1))


def main() -> None:
    ma = pd.read_parquet(RAW_DIR / "market_access.parquet").set_index("territory_id")["market_access"]
    reg = pd.read_parquet(RAW_DIR / "territories.parquet").set_index("territory_id")["region_code"]
    spec = pd.read_parquet(SPEC_LABELS_PATH)
    months = sorted(spec["месяц"].unique())
    ids = np.sort(spec["territory_id"].unique())
    if ma.reindex(ids).isna().any() or reg.reindex(ids).isna().any() or (ma.reindex(ids) <= 0).any():
        raise SystemExit("STOP: не у всех МО есть положительный market_access и region_code")
    y = np.log(ma.loc[ids].to_numpy())
    region = reg.loc[ids].to_numpy()
    pos = pd.Series(np.arange(len(ids)), index=ids)
    hw = pd.read_parquet(HIGHWAY_PATH)
    hw = hw[hw["territory_id_x"].isin(pos.index) & hw["territory_id_y"].isin(pos.index)]
    ei, ej = pos[hw["territory_id_x"]].to_numpy(), pos[hw["territory_id_y"]].to_numpy()

    parts = {}   # (набор, метка, месяц) -> метки
    for (v, m), g in spec.groupby(["вариант", "месяц"]):
        parts[("варианты", v, m)] = g.set_index("territory_id")["cluster"].loc[ids].to_numpy()
    for k in EXTVAL_KS:
        for m in months:
            parts[("k", f"k={k}", m)] = pd.read_parquet(PROCESSED_DIR / f"kmeans_labels_k{k}" / f"{m}.parquet"
                                                        ).set_index("territory_id")["cluster"].loc[ids].to_numpy()
    for m in months:   # вариант A при k = 6 — те же официальные разбиения шага 12a
        if ROBUST2_K == FINAL_K and not (pd.Series(parts[("варианты", "A", m)]).groupby(
                parts[("k", f"k={FINAL_K}", m)]).nunique() == 1).all():
            raise SystemExit(f"STOP: {m}: вариант A (15a) не совпал с метками шага 12a")

    rows = []
    for (grp, name, m), lab in parts.items():
        r2, adj, omega = anova(y, lab)
        # одинаковые перестановки для всех разбиений одного месяца (одинаковое разбиение → одинаковый z)
        share, z = edge_stats(lab, ei, ej, np.random.default_rng([EXTVAL_SEED, months.index(m)]))
        rows.append({"набор": grp, "разбиение": name, "месяц": m, "R² log(MA)": r2, "adj R² log(MA)": adj,
                     "ω² log(MA)": omega, "AMI регион": adjusted_mutual_info_score(region, lab),
                     "доля рёбер highway внутри": share, "z рёбер highway": z})
    res = pd.DataFrame(rows)
    res.to_parquet(OUT_PATH, engine="pyarrow", index=False)

    def summary(grp: str) -> pd.DataFrame:
        d = res[res["набор"] == grp]
        out = []
        for name, g in d.groupby("разбиение", sort=False):
            row = {"разбиение": name}
            for col, _ in METRICS:
                row[f"{col}: {MONTH}"] = g.loc[g["месяц"] == MONTH, col].iloc[0]
                row[f"{col}: среднее ± ст. откл."] = f"{g[col].mean():.3f} ± {g[col].std():.3f}"
            out.append(row)
        return pd.DataFrame(out)

    def wilcox(grp: str, pairs: list[tuple[str, str]]) -> pd.DataFrame:
        d = res[res["набор"] == grp]
        out = []
        for a, b in pairs:
            for col, _ in METRICS:
                x = d[d["разбиение"] == a].set_index("месяц")[col]
                w = d[d["разбиение"] == b].set_index("месяц")[col].reindex(x.index)
                diff = x - w
                p = wilcoxon(x, w).pvalue if (diff != 0).any() else 1.0
                out.append({"сравнение": f"{a} vs {b}", "показатель": col, "медиана разницы": diff.median(),
                            f"{a} выше (мес.)": int((diff > 0).sum()), "p (Уилкоксон)": p,
                            f"значимо (p < {EXTVAL_ALPHA})": "да" if p < EXTVAL_ALPHA else "нет",
                            "выше у": (a if diff.median() > 0 else b) if p < EXTVAL_ALPHA else "—"})
        return pd.DataFrame(out)

    var_names = ROBUST2_VARIANTS
    k_names = [f"k={k}" for k in EXTVAL_KS]
    w_var = wilcox("варианты", list(combinations(var_names, 2)))
    w_k = wilcox("k", [(f"k={FINAL_K}", n) for n in k_names if n != f"k={FINAL_K}"])

    def leaders(grp: str) -> pd.DataFrame:
        d = res[res["набор"] == grp].groupby("разбиение")[[c for c, _ in METRICS]].mean()
        return pd.DataFrame({"показатель": [c for c, _ in METRICS],
                             "максимум среднего по 24 мес. у": [d[c].idxmax() for c, _ in METRICS],
                             "значение": [d[c].max() for c, _ in METRICS]})

    note = ("Сравнение между k имеет ограничение: с ростом k R² растёт механически (скорректированный R² "
            "и ω² это частично учитывают), а доля рёбер внутри кластеров падает (z-оценка против "
            "перестановок с теми же размерами кластеров это учитывает); AMI скорректирован на случайность. "
            "Месяцы не независимы, поэтому p-значения критерия Уилкоксона ориентировочные.")
    lines = [
        "# 15e. Внешняя валидность вариантов признаков и выбора k",
        "",
        "Сгенерировано `src/15e_external_validity.py`. Решение не принимается; канон, 10c, README и "
        "прежние отчёты не изменялись.",
        "",
        "Внешние данные, не входившие в признаки: log(market_access); region_code (справочник МО); граф "
        f"highway kNN8 (шаг 07). z-оценка — против {EXTVAL_PERMUTATIONS} перестановок меток (размеры "
        f"кластеров сохраняются). Сравнения — парный критерий Уилкоксона по 24 месяцам, α = {EXTVAL_ALPHA}.",
        "",
        f"> {note}",
        "",
        "Показатели: " + "; ".join(f"**{c}** — {d}" for c, d in METRICS) + ". Для всех показателей больше — "
        "значит сильнее связь кластеров с внешними данными.",
        "",
        f"## Варианты признаков A/B/C при k = {ROBUST2_K}",
        "",
        md(summary("варианты"), fmt=",.3f"),
        "",
        "Лидер по среднему за 24 месяца:",
        "",
        md(leaders("варианты"), fmt=",.3f"),
        "",
        "Парные сравнения:",
        "",
        md(w_var, fmt=",.4g"),
        "",
        f"## Вариант A при k = {EXTVAL_KS[0]}–{EXTVAL_KS[-1]}",
        "",
        md(summary("k"), fmt=",.3f"),
        "",
        "Лидер по среднему за 24 месяца:",
        "",
        md(leaders("k"), fmt=",.3f"),
        "",
        f"Парные сравнения k = {FINAL_K} с остальными:",
        "",
        md(w_k, fmt=",.4g"),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(leaders("варианты").to_string(index=False)); print(leaders("k").to_string(index=False))
    print(w_var[["сравнение", "показатель", "медиана разницы", "p (Уилкоксон)", "выше у"]].to_string(index=False))
    print(w_k[["сравнение", "показатель", "медиана разницы", "p (Уилкоксон)", "выше у"]].to_string(index=False))


if __name__ == "__main__":
    main()
