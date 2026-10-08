"""Шаг 15b. Таблицы сопряжённости по k (проверка 2 батча 2), декабрь 2024, метки шага 10b.

Для пар (k_родитель → k_потомок) из config.ROBUST2_CONTINGENCY: кросс-таблица, доля МО в основном
«потомке» (для каждого кластера-родителя — доля его МО в самом крупном кластере-потомке, взвешенно
по всем МО); Жаккар лучшего соответствия для каждого кластера k = FINAL_K с кластерами других k;
из каких кластеров k = 4 и k = 5 собрана удалённая группа k = FINAL_K.

Выход: data/processed/robust2_contingency.parquet (пара, родитель, потомок, МО),
       data/processed/robust2_best_jaccard.parquet
Запуск из корня проекта:  .venv/bin/python src/15b_k_contingency.py
"""
from pathlib import Path

import pandas as pd

from config import FINAL_K, MONTH, ROBUST2_CONTINGENCY, ROBUST2_REMOTE_CLUSTER

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TAG = MONTH.replace("-", "_")
CT_PATH = PROCESSED_DIR / "robust2_contingency.parquet"
BJ_PATH = PROCESSED_DIR / "robust2_best_jaccard.parquet"


def labels(k: int) -> pd.Series:
    return pd.read_parquet(PROCESSED_DIR / f"kmeans_labels_k{k}_{TAG}.parquet").set_index("territory_id")["cluster"]


def main() -> None:
    ks = sorted({k for pair in ROBUST2_CONTINGENCY for k in pair} | {FINAL_K})
    L = {k: labels(k) for k in ks}
    ct_rows = []
    for a, b in ROBUST2_CONTINGENCY:
        ct = pd.crosstab(L[a], L[b])
        for p, row in ct.iterrows():
            for c, n in row.items():
                ct_rows.append({"пара": f"{a}→{b}", "родитель": int(p), "потомок": int(c), "МО": int(n)})
        print(f"{a}→{b}: доля МО в основном потомке {ct.max(axis=1).sum() / ct.to_numpy().sum():.3f}")
    pd.DataFrame(ct_rows).to_parquet(CT_PATH, engine="pyarrow", index=False)

    bj = []
    for c in sorted(L[FINAL_K].unique()):
        a = set(L[FINAL_K].index[L[FINAL_K] == c])
        for k in ks:
            if k == FINAL_K:
                continue
            best = max(((len(a & set(L[k].index[L[k] == d])) / len(a | set(L[k].index[L[k] == d])), int(d))
                        for d in L[k].unique()))
            bj.append({"кластер k=6": int(c), "МО": len(a), "k": k, "лучший кластер": best[1],
                       "Жаккар": best[0]})
    pd.DataFrame(bj).to_parquet(BJ_PATH, engine="pyarrow", index=False)
    rm = L[FINAL_K].index[L[FINAL_K] == ROBUST2_REMOTE_CLUSTER]
    for k in (4, 5):
        print(f"удалённая группа k={FINAL_K} из кластеров k={k}:", L[k].loc[rm].value_counts().to_dict())


if __name__ == "__main__":
    main()
