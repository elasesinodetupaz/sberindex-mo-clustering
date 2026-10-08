"""Шаг 24g. Тип МО по последнему доступному месяцу (для МО со статусом no_dec2024 в types_outside_sample.parquet).

Отдельный класс «тип по последнему месяцу»: эти МО не входят в статистики и портреты типов, в проверки Г1–Г13 и в
статическую карту 24d; types_outside_sample.parquet шаг не меняет. Что делает:
  1. берёт МО со статусом no_dec2024 и для каждого находит последний месяц в consumption.parquet, где есть все пять
     категорий и «Все категории» (функции complete_ids и shares_for_month импортируются из шага 24c);
  2. обучает KMeans(FINAL_K, random_state, n_init) на МО выборки за декабрь 2024 (как в шаге 24c) и сверяет разбиение
     с kmeans_labels_final.parquet; центроиды декабря 2024 фиксируются;
  3. для каждого из 24 месяцев стандартизирует доли МО выборки параметрами этого месяца, присваивает их ближайшему
     зафиксированному центроиду и считает согласие с колонкой cluster в kmeans_k6_trajectories.parquet (доля совпавших
     МО и ARI), а также медиану, p95 и максимум расстояния до центроида;
  4. кандидату присваивает ближайший центроид по параметрам его последнего месяца; месяц допускается при согласии не меньше
     AGREEMENT_MIN (config.yaml), тип присваивается при расстоянии не больше максимума по выборке в том же месяце.
Все контроли выполняются до записи файлов; при расхождении печатается таблица контролей и код выхода 1.
Вход:  data/processed/types_outside_sample.parquet, data/processed/kmeans_labels_final.parquet,
       data/processed/category_shares.parquet, data/processed/kmeans_k6_trajectories.parquet,
       data/raw/consumption.parquet, data/geo/mo_national.geojson
Выход: data/processed/types_last_month.parquet, data/processed/month_agreement_24g.parquet,
       notebooks/24g_last_month_types.md
Запуск из корня проекта:  .venv/bin/python src/24g_last_month_types.py [--no-write]
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import config as C
from network_utils import standardize_shares

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
GEO_PATH = PROJECT_DIR / C.ASSIGN_GEO_PATH
LABELS_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
OUTSIDE_PATH = PROCESSED_DIR / "types_outside_sample.parquet"
TRAJ_PATH = PROJECT_DIR / C.LASTM_TRAJECTORIES_PATH
CONSUMPTION_PATH = RAW_DIR / "consumption.parquet"
OUT_TYPES = PROCESSED_DIR / "types_last_month.parquet"
OUT_AGREE = PROCESSED_DIR / "month_agreement_24g.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "24g_last_month_types.md"

m24c = importlib.import_module("24c_assign_outside")   # функции и константы шага 24c импортируются, не переписываются
SHARE_NAMES = m24c.SHARE_NAMES
SHARE_COLUMNS = m24c.SHARE_COLUMNS
TOTAL_CATEGORY = m24c.TOTAL_CATEGORY
DEC = C.ASSIGN_DIST_DECIMALS


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def fmt(v):
    return ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)


def main() -> None:
    write = "--no-write" not in sys.argv
    for p in (GEO_PATH, LABELS_PATH, SHARES_PATH, OUTSIDE_PATH, TRAJ_PATH, CONSUMPTION_PATH):
        if not p.exists():
            stop(f"нет файла {p}")
    geo = json.loads(GEO_PATH.read_text(encoding="utf-8"))
    region_of = {int(f["properties"]["territory_id"]): f["properties"]["region_name"] for f in geo["features"]}
    labels = pd.read_parquet(LABELS_PATH)
    sample = sorted(labels["territory_id"].astype(int))
    outside = pd.read_parquet(OUTSIDE_PATH)
    cand_ids = sorted(outside.loc[outside["status"] == "no_dec2024", "territory_id"].astype(int))
    cons = pd.read_parquet(CONSUMPTION_PATH, engine="pyarrow")
    months = sorted(cons["date"].unique())
    if len(months) != C.LASTM_EXPECT_MONTHS:
        stop(f"месяцев в consumption {len(months)}, ожидалось {C.LASTM_EXPECT_MONTHS}")
    by_month = {m: g for m, g in cons.groupby("date")}

    # --- 1. кандидаты: последний полный месяц
    last_month = {}
    for i in cand_ids:
        for m in reversed(months):
            if i in m24c.complete_ids(by_month[m], [i]):
                last_month[i] = m
                break

    # --- 2. центроиды декабря 2024: обучение на выборке и сверка с kmeans_labels_final
    cm = by_month[C.MONTH]
    ws = m24c.shares_for_month(cm, sample)
    if len(ws) != len(sample):
        stop(f"долей за {C.MONTH} для выборки {len(ws)}, ожидалось {len(sample)}")
    cs = pd.read_parquet(SHARES_PATH)
    cs = cs[cs["date"] == pd.Timestamp(C.MONTH)].sort_values("territory_id").reset_index(drop=True)
    shares_match = bool(np.allclose(ws[SHARE_NAMES].to_numpy(), cs[SHARE_NAMES].to_numpy(), rtol=0, atol=1e-12)
                        and (ws["territory_id"].to_numpy() == cs["territory_id"].to_numpy()).all())
    X, scaler = standardize_shares(ws[SHARE_NAMES].to_numpy())
    km = KMeans(n_clusters=C.FINAL_K, random_state=C.KMEANS_RANDOM_STATE, n_init=C.KMEANS_N_INIT).fit(X)
    lab = labels.set_index("territory_id")["cluster"].reindex(ws["territory_id"]).to_numpy()
    mism = int((km.labels_ != lab).sum())
    sizes = np.bincount(lab, minlength=C.FINAL_K).tolist()
    cent = km.cluster_centers_

    def dists(Z):
        D = np.sqrt(((Z[:, None, :] - cent[None, :, :]) ** 2).sum(2))
        srt = np.sort(D, axis=1)
        return D.argmin(1), srt[:, 0], srt[:, 1]

    # --- 3. по месяцам: параметры месяца, фиксированные центроиды, согласие с месячными метками
    traj = pd.read_parquet(TRAJ_PATH)
    traj_schema = [(c, str(t)) for c, t in traj.dtypes.items()]
    rows, month_info = [], {}
    for m in months:
        wm = m24c.shares_for_month(by_month[m], sample)
        wm = wm.dropna(subset=SHARE_NAMES).reset_index(drop=True)
        ids = wm["territory_id"].astype(int).to_numpy()
        Zm, sc_m = standardize_shares(wm[SHARE_NAMES].to_numpy())
        pm, d1m, _ = dists(Zm)
        tm = traj[traj["month"] == m].set_index("territory_id")["cluster"].reindex(ids).to_numpy()
        valid = ~pd.isna(tm)
        tmv = tm[valid].astype(int)
        agree = float((pm[valid] == tmv).mean())
        ari = float(adjusted_rand_score(tmv, pm[valid]))
        month_info[m] = {"scaler": sc_m, "dmax": float(d1m.max()), "agree": agree, "pred": pm, "ids": ids}
        rows.append({"month": m, "n_sample": int(len(ids)), "agreement": agree, "ari": ari,
                     "dist_median": float(np.median(d1m)), "dist_p95": float(np.percentile(d1m, 95)),
                     "dist_max": float(d1m.max()), "eligible": bool(agree >= C.LASTM_AGREEMENT_MIN)})
    # декабрь 2024: присвоение к центроидам против kmeans_labels_final
    dec = month_info[C.MONTH]
    dec_vs_final = float((pd.Series(dec["pred"], index=dec["ids"]).reindex(labels["territory_id"].astype(int))
                          .to_numpy() == labels["cluster"].to_numpy()).mean())

    # --- 4. присвоение кандидатам
    res = []
    for i in cand_ids:
        reg = region_of.get(i)
        if i not in last_month:
            res.append({"territory_id": i, "region_name": reg, "last_month": None, "status": "no_complete_month", "type": None,
                        "dist_nearest": np.nan, "dist_second": np.nan, "ratio": np.nan, "month_agreement": np.nan})
            continue
        m = last_month[i]
        mi = month_info[m]
        w = m24c.shares_for_month(by_month[m], [i])
        t, d1, d2 = dists(mi["scaler"].transform(w[SHARE_NAMES].to_numpy()))
        d1, d2 = float(d1[0]), float(d2[0])
        if mi["agree"] < C.LASTM_AGREEMENT_MIN:
            status, typ = "month_not_eligible", None
        elif d1 > mi["dmax"]:
            status, typ = "too_far", None
        else:
            status, typ = "assigned_last_month", int(t[0])
        res.append({"territory_id": i, "region_name": reg, "last_month": m, "status": status, "type": typ,
                    "dist_nearest": d1, "dist_second": d2, "ratio": d2 / d1, "month_agreement": mi["agree"]})
    out = pd.DataFrame(res)
    for c in ("dist_nearest", "dist_second", "ratio"):
        out[c] = out[c].round(DEC)
    out["ratio"] = (out["dist_second"] / out["dist_nearest"]).round(DEC)
    out = out[["territory_id", "region_name", "last_month", "status", "type", "dist_nearest", "dist_second", "ratio",
               "month_agreement"]].astype({"territory_id": "int32", "region_name": "string", "last_month": "string",
                                           "status": "string", "type": "Int8", "dist_nearest": "float64",
                                           "dist_second": "float64", "ratio": "float64", "month_agreement": "float64"})
    out = out.sort_values("territory_id", ignore_index=True)
    agree_df = pd.DataFrame(rows)
    cnt = out.groupby("last_month").size()
    agree_df["n_candidates_last_month"] = agree_df["month"].map(cnt).fillna(0).astype(int)
    for c in ("agreement", "ari", "dist_median", "dist_p95", "dist_max"):
        agree_df[c] = agree_df[c].round(DEC)
    agree_df = agree_df.astype({"month": "string", "n_sample": "int32", "agreement": "float64", "ari": "float64",
                                "dist_median": "float64", "dist_p95": "float64", "dist_max": "float64",
                                "eligible": "bool", "n_candidates_last_month": "int32"})
    st = out["status"].value_counts()

    # --- контроли
    controls = []

    def add(name, expect, got, ok):
        controls.append({"name": name, "expect": expect, "got": got, "ok": bool(ok)})

    def near(a, b, tol):
        return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= tol))

    reg_counts = out["region_name"].value_counts()
    got_reg = {k: int(reg_counts.get(k, 0)) for k in C.LASTM_EXPECT_REGION_COUNTS}
    other = out[~out["region_name"].isin(C.LASTM_EXPECT_REGION_COUNTS)]
    add("версия scikit-learn", C.ASSIGN_SKLEARN_VERSION, sklearn.__version__, sklearn.__version__ == C.ASSIGN_SKLEARN_VERSION)
    add("доли за декабрь 2024, пересчёт совпадает с category_shares.parquet", True, shares_match, shares_match)
    add("scaler mean_", C.ASSIGN_SCALER_MEAN, np.round(scaler.mean_, 6).tolist(), near(scaler.mean_, C.ASSIGN_SCALER_MEAN, C.ASSIGN_SCALER_TOL))
    add("расхождений с kmeans_labels_final (2004 МО)", 0, mism, mism == 0)
    add("размеры типов 0–5", C.ASSIGN_TYPE_SIZES, sizes, sizes == C.ASSIGN_TYPE_SIZES)
    add("inertia", C.ASSIGN_INERTIA, round(float(km.inertia_), 3), abs(km.inertia_ - C.ASSIGN_INERTIA) <= C.ASSIGN_INERTIA_TOL)
    add("кандидатов (no_dec2024)", C.LASTM_EXPECT_CANDIDATES, len(cand_ids), len(cand_ids) == C.LASTM_EXPECT_CANDIDATES)
    add("согласие декабря 2024: присвоение к центроидам против kmeans_labels_final", C.LASTM_EXPECT_DEC_AGREEMENT, round(dec_vs_final, 10),
        dec_vs_final == C.LASTM_EXPECT_DEC_AGREEMENT)
    for m, n in C.LASTM_EXPECT_LAST_MONTH_COUNTS.items():
        g = int((out["last_month"] == m).sum())
        add(f"кандидатов с последним месяцем {m}", n, g, g == n)
    add("кандидатов по регионам (10 регионов)", C.LASTM_EXPECT_REGION_COUNTS, got_reg, got_reg == C.LASTM_EXPECT_REGION_COUNTS)
    add("кандидатов в остальных регионах", C.LASTM_EXPECT_OTHER_REGIONS_N, len(other), len(other) == C.LASTM_EXPECT_OTHER_REGIONS_N)
    ssum = int(sum(st.get(k, 0) for k in ("assigned_last_month", "month_not_eligible", "too_far", "no_complete_month")))
    add("присвоено + month_not_eligible + too_far + no_complete_month", len(cand_ids), ssum, ssum == len(cand_ids))
    add("согласий в таблице месяцев", C.LASTM_EXPECT_MONTHS, len(agree_df), len(agree_df) == C.LASTM_EXPECT_MONTHS)
    add("согласие каждого месяца от 0 до 1", True, bool(agree_df["agreement"].between(0, 1).all()), bool(agree_df["agreement"].between(0, 1).all()))
    add("строк в таблице результата", len(cand_ids), len(out), len(out) == len(cand_ids) and out["territory_id"].is_unique)

    print(f"sklearn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
    print("Схема kmeans_k6_trajectories.parquet:", fmt([f"{c}: {t}" for c, t in traj_schema]), "; колонка месяца: month, метка: cluster")
    print("| контроль | ожидание | получено | результат |")
    for c in controls:
        print(f"| {c['name']} | {fmt(c['expect'])} | {fmt(c['got'])} | {'пройден' if c['ok'] else 'НЕ ПРОЙДЕН'} |")
    print("Кандидаты в остальных регионах:", other["region_name"].value_counts().to_dict())
    print("Таблица месяцев:")
    print(agree_df.to_string(index=False))
    print("Статусы:", {k: int(v) for k, v in st.items()}, "; присвоено по типам:",
          np.bincount(out.loc[out["status"] == "assigned_last_month", "type"].astype(int), minlength=C.FINAL_K).tolist())
    n_ok = int(st.get("assigned_last_month", 0))
    if n_ok < C.LASTM_NOTICE_MIN_ELIGIBLE:
        print(f"СООБЩЕНИЕ АВТОРУ: присвоено {n_ok} из {len(cand_ids)} (меньше {C.LASTM_NOTICE_MIN_ELIGIBLE}); решение за автором.")
    if not all(c["ok"] for c in controls):
        print("STOP: контроли не пройдены; файлы не записаны")
        raise SystemExit(1)
    if not write:
        print("Режим --no-write: файлы не записаны.")
        return
    md = report(out, agree_df, dec_vs_final, controls, len(sample))
    out.to_parquet(OUT_TYPES, engine="pyarrow", index=False)
    agree_df.to_parquet(OUT_AGREE, engine="pyarrow", index=False)
    REPORT_PATH.write_text(md, encoding="utf-8")
    print("Готово.")


def report(out, agree_df, dec_vs_final, controls, n_sample) -> str:
    st = out["status"].value_counts()
    asg = out[out["status"] == "assigned_last_month"]
    L = ["# 24g. Тип МО по последнему доступному месяцу", "",
         "Сгенерировано `src/24g_last_month_types.py`. Все числа взяты из расчёта скрипта. Класс «тип по последнему месяцу» отдельный: "
         "эти МО не входили в обучение и не входят в статистики и портреты типов, в проверки Г1–Г13 и в статическую карту 24d; "
         "`types_outside_sample.parquet` шаг не меняет.", "",
         "## Метод", "",
         f"1. Кандидаты — МО со статусом no_dec2024 в `types_outside_sample.parquet` ({len(out)} МО). Для каждого берётся последний месяц "
         "в `consumption.parquet`, где есть все пять категорий и «Все категории» (функции шага 24c).",
         f"2. Центроиды: KMeans (k = {C.FINAL_K}, random_state = {C.KMEANS_RANDOM_STATE}, n_init = {C.KMEANS_N_INIT}) на {n_sample} МО выборки за {C.MONTH} "
         "в z-пространстве шага 24c; разбиение совпало с `kmeans_labels_final.parquet` на всех МО выборки. Центроиды зафиксированы.",
         "3. Для каждого из 24 месяцев доли МО выборки стандартизируются параметрами этого месяца (StandardScaler, ddof = 0) и присваиваются "
         "ближайшему зафиксированному центроиду. Согласие месяца — доля МО выборки, у которых это присвоение равно колонке cluster в "
         f"`kmeans_k6_trajectories.parquet` за тот же месяц; рядом указан ARI между этими метками. Для {C.MONTH} присвоение к центроидам "
         f"против `kmeans_labels_final.parquet` совпало на доле {dec_vs_final:.4f}.",
         f"4. Месяц допускается, если согласие не меньше {C.LASTM_AGREEMENT_MIN:.2f} (`AGREEMENT_MIN` в `config.yaml`).",
         "5. Кандидат стандартизируется параметрами своего последнего месяца и получает ближайший центроид. Тип присваивается, если месяц допущен "
         "и расстояние до ближайшего центроида не больше максимума по выборке в том же месяце.", "",
         "## Согласие метода с месячными метками на выборке", "",
         "| месяц | МО выборки | согласие | ARI | расстояние, медиана | расстояние, p95 | расстояние, максимум | допущен | кандидатов с последним месяцем |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in agree_df.itertuples():
        L.append(f"| {r.month} | {r.n_sample} | {r.agreement:.4f} | {r.ari:.4f} | {r.dist_median:.3f} | {r.dist_p95:.3f} | {r.dist_max:.3f} | "
                 f"{'да' if r.eligible else 'нет'} | {r.n_candidates_last_month} |")
    L += ["", "## Кандидаты по последнему месяцу и статусам", "",
          "| последний месяц | всего | assigned_last_month | month_not_eligible | too_far |", "|---|---|---|---|---|"]
    has = out[out["last_month"].notna()]
    for m, g in has.groupby("last_month"):
        s = g["status"].value_counts()
        L.append(f"| {m} | {len(g)} | {int(s.get('assigned_last_month', 0))} | {int(s.get('month_not_eligible', 0))} | {int(s.get('too_far', 0))} |")
    L += ["", "| статус | число МО |", "|---|---|"]
    for k in ("assigned_last_month", "month_not_eligible", "too_far", "no_complete_month"):
        L.append(f"| {k} | {int(st.get(k, 0))} |")
    L.append(f"| всего | {len(out)} |")
    L += ["", "## Присвоенные типы", "", "| тип | МО |", "|---|---|"]
    bt = np.bincount(asg["type"].astype(int), minlength=C.FINAL_K)
    for t in range(C.FINAL_K):
        L.append(f"| {t} | {int(bt[t])} |")
    L += ["", "## По регионам", "", "| регион | кандидатов | присвоено | не присвоено |", "|---|---|---|---|"]
    for reg, g in sorted(out.groupby("region_name"), key=lambda kv: (-len(kv[1]), kv[0])):
        a = int((g["status"] == "assigned_last_month").sum())
        L.append(f"| {reg} | {len(g)} | {a} | {len(g) - a} |")
    L += ["", "## Ограничения", "",
          "- Тип присвоен по данным месяца, отличного от декабря 2024; у разных месяцев сезонность отличается, поэтому доли категорий и расстояния "
          "до центроидов сравнимы между месяцами лишь отчасти.",
          "- Согласие метода с месячными метками на выборке показано в таблице выше; для месяцев с более высоким согласием присвоение ближе к месячным меткам выборки.",
          "- Кандидаты не входили в обучение KMeans, в стандартизацию и в статистики шагов 10–28.",
          "- Месячные метки в `kmeans_k6_trajectories.parquet` сопоставлены с якорем декабря 2024; согласие измерено только на МО выборки.",
          "- Для МО со статусами month_not_eligible, too_far и no_complete_month тип не присваивается.", ""]
    L += ["## Контроли", "", "| контроль | ожидание | получено | результат |", "|---|---|---|---|"]
    for c in controls:
        L.append(f"| {c['name']} | {fmt(c['expect'])} | {fmt(c['got'])} | {'пройден' if c['ok'] else 'НЕ ПРОЙДЕН'} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
