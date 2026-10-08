"""Шаг 24c. Типы МО вне рабочей выборки: присвоение по ближайшему центроиду канонического разбиения.

Присвоенные типы НЕ входят ни в один расчёт шагов 10–28 и не меняют замороженные файлы: этот шаг только
читает входы и пишет свой файл. Что делает:
  1. берёт действующие МО из data/geo/mo_national.geojson (шаг 24б), рабочую выборку из kmeans_labels_final.parquet;
     группа A — нет в consumption.parquet, группа B — есть в consumption, но вне выборки;
  2. строит доли пяти категорий за канонический месяц тем же выражением, что шаг 03 (value / «Все категории»);
  3. стандартизирует доли по МО выборки (StandardScaler, ddof = 0, как network_utils.standardize_shares), обучает
     KMeans(FINAL_K, random_state, n_init из config.yaml) на выборке — центроиды в файлах не хранятся, поэтому
     разбиение обучается заново и сверяется с kmeans_labels_final.parquet (0 расхождений);
  4. МО группы B с пятью долями и «Все категории» за месяц получают ближайший центроид (евклид в z-пространстве),
     но тип присваивается только если расстояние не больше максимума по выборке; остальные МО типа не получают.
Расстояния в файле округлены до DIST_DECIMALS знаков (config.yaml). Все контроли выполняются до записи файлов;
при расхождении печатается таблица контролей и код выхода 1.
Вход:  data/geo/mo_national.geojson, data/processed/kmeans_labels_final.parquet, data/processed/category_shares.parquet,
       data/raw/consumption.parquet, data/raw/market_access.parquet
Выход: data/processed/types_outside_sample.parquet, notebooks/24c_types_outside_sample.md
Запуск из корня проекта:  .venv/bin/python src/24c_assign_outside.py
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans

import config as C
from network_utils import standardize_shares

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
GEO_PATH = PROJECT_DIR / C.ASSIGN_GEO_PATH
LABELS_PATH = PROCESSED_DIR / "kmeans_labels_final.parquet"
SHARES_PATH = PROCESSED_DIR / "category_shares.parquet"
CONSUMPTION_PATH = RAW_DIR / "consumption.parquet"
MARKET_ACCESS_PATH = RAW_DIR / "market_access.parquet"
OUT_PATH = PROCESSED_DIR / "types_outside_sample.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "24c_types_outside_sample.md"

# константы шагов 02 и 03 берутся импортом, не переписываются
step02 = importlib.import_module("02_missing_data")
step03 = importlib.import_module("03_category_shares")
N_MONTHS_REQUIRED = step02.N_MONTHS_REQUIRED
SHARE_COLUMNS = step03.SHARE_COLUMNS
TOTAL_CATEGORY = step03.TOTAL_CATEGORY
SHARE_NAMES = list(SHARE_COLUMNS.values())


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def shares_for_month(cons_month: pd.DataFrame, ids) -> pd.DataFrame:
    """Доли пяти категорий за месяц для МО из ids — те же выражения, что в src/03_category_shares.py:33–47."""
    cons = cons_month[cons_month["territory_id"].isin(ids)].copy()
    cons["date"] = pd.to_datetime(cons["date"], format="%Y-%m")
    df_total = (cons[cons["category"] == TOTAL_CATEGORY]
                .rename(columns={"value": "total_value"})
                [["territory_id", "date", "total_value"]])
    df_parts = cons[cons["category"] != TOTAL_CATEGORY]
    long = df_parts.merge(df_total, on=["territory_id", "date"], how="left", validate="many_to_one")
    long["category_share"] = long["value"] / long["total_value"]
    wide = (long.pivot(index=["territory_id", "date"], columns="category", values="category_share")
            .rename(columns=SHARE_COLUMNS)[SHARE_NAMES]
            .reset_index()
            .sort_values(["territory_id", "date"], ignore_index=True))
    wide.columns.name = None
    return wide


def complete_ids(cons_month: pd.DataFrame, ids) -> set:
    """МО, у которых за месяц есть все пять категорий и «Все категории» (значение итога положительное)."""
    d = cons_month[cons_month["territory_id"].isin(ids)]
    piv = d.pivot_table(index="territory_id", columns="category", values="value", aggfunc="first")
    need = list(SHARE_COLUMNS) + [TOTAL_CATEGORY]
    if any(c not in piv.columns for c in need):
        return set()
    ok = piv[need].dropna()
    return set(ok[ok[TOTAL_CATEGORY] > 0].index.astype(int))


def fmt(v):
    return ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)


def pct(a, q):
    return float(np.percentile(a, q))


def main() -> None:
    if not GEO_PATH.exists():
        stop(f"нет файла {GEO_PATH} (шаг 24б)")
    geo = json.loads(GEO_PATH.read_text(encoding="utf-8"))
    active = sorted({int(f["properties"]["territory_id"]) for f in geo["features"]})
    labels = pd.read_parquet(LABELS_PATH)
    sample = sorted(labels["territory_id"].astype(int))
    if not set(sample) <= set(active):
        stop("часть МО выборки отсутствует среди действующих полигонов")
    outside = sorted(set(active) - set(sample))
    cons = pd.read_parquet(CONSUMPTION_PATH, engine="pyarrow")
    ma_ids = set(pd.read_parquet(MARKET_ACCESS_PATH, engine="pyarrow")["territory_id"].astype(int))
    cons_ids = set(cons["territory_id"].astype(int).unique())
    group_a = [i for i in outside if i not in cons_ids]
    group_b = [i for i in outside if i in cons_ids]
    months = cons.groupby("territory_id")["date"].nunique()

    # --- доли за канонический месяц, стандартизация и обучение на выборке
    cm = cons[cons["date"] == C.MONTH]
    ws = shares_for_month(cm, sample)
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

    _, d_sample, _ = dists(X)
    sd = {"median": float(np.median(d_sample)), "p95": pct(d_sample, 95), "max": float(d_sample.max())}

    # --- группа B: полные доли за месяц
    comp = complete_ids(cm, group_b)
    b_complete = sorted(comp)
    b_incomplete = sorted(set(group_b) - comp)
    wb = shares_for_month(cm, b_complete)
    Zb = scaler.transform(wb[SHARE_NAMES].to_numpy())
    tb, d1, d2 = dists(Zb)
    bd = {"median": float(np.median(d1)), "p95": pct(d1, 95), "max": float(d1.max())}
    beyond = [int(i) for i, d in zip(wb["territory_id"], d1) if d > sd["max"]]
    assigned_mask = d1 <= sd["max"]
    by_type_assigned = np.bincount(tb[assigned_mask], minlength=C.FINAL_K).tolist()

    # --- таблица результата: одна строка на каждое МО вне выборки
    info = pd.DataFrame({"territory_id": wb["territory_id"].astype(int), "type": tb, "dist_nearest": d1, "dist_second": d2,
                         "assigned": assigned_mask}).set_index("territory_id")
    rows = []
    for i in outside:
        grp = "A" if i in group_a else "B"
        if grp == "A":
            status, typ, dn, ds, reason = "no_data_in_dataset", None, np.nan, np.nan, None
        else:
            inc = int(months[i]) < N_MONTHS_REQUIRED
            nom = i not in ma_ids
            reason = "both" if (inc and nom) else "incomplete_history" if inc else "no_market_access" if nom else None
            if reason is None:
                stop(f"МО {i} группы B не имеет кода исключения по шагу 02")
            if i in info.index:
                r = info.loc[i]
                dn, ds = float(r["dist_nearest"]), float(r["dist_second"])
                if bool(r["assigned"]):
                    status, typ = "assigned", int(r["type"])
                else:
                    status, typ = "too_far", None
            else:
                status, typ, dn, ds = "no_dec2024", None, np.nan, np.nan
        rows.append({"territory_id": i, "group": grp, "status": status, "type": typ, "dist_nearest": dn, "dist_second": ds,
                     "months_available": int(months[i]) if i in months.index else 0, "exclusion_reason": reason})
    out = pd.DataFrame(rows)
    out["dist_nearest"] = out["dist_nearest"].round(C.ASSIGN_DIST_DECIMALS)   # округление убирает шум многопоточных вычислений
    out["dist_second"] = out["dist_second"].round(C.ASSIGN_DIST_DECIMALS)
    out["ratio"] = (out["dist_second"] / out["dist_nearest"]).round(C.ASSIGN_DIST_DECIMALS)
    out = out[["territory_id", "group", "status", "type", "dist_nearest", "dist_second", "ratio", "months_available",
               "exclusion_reason"]].astype({"territory_id": "int32", "group": "string", "status": "string", "type": "Int8",
                                             "dist_nearest": "float64", "dist_second": "float64", "ratio": "float64",
                                             "months_available": "int16", "exclusion_reason": "string"})
    out = out.sort_values("territory_id", ignore_index=True)
    if (out.dtypes == object).any():
        stop("в таблице результата остались колонки типа object")
    st = out["status"].value_counts()

    # --- контроли
    controls = []

    def add(name, expect, got, ok):
        controls.append({"name": name, "expect": expect, "got": got, "ok": bool(ok)})

    def near(a, b, tol):
        return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= tol))

    add("версия scikit-learn", C.ASSIGN_SKLEARN_VERSION, sklearn.__version__, sklearn.__version__ == C.ASSIGN_SKLEARN_VERSION)
    add("доли за месяц, пересчёт совпадает с category_shares.parquet", True, shares_match, shares_match)
    add("scaler mean_", C.ASSIGN_SCALER_MEAN, np.round(scaler.mean_, 6).tolist(), near(scaler.mean_, C.ASSIGN_SCALER_MEAN, C.ASSIGN_SCALER_TOL))
    add("scaler scale_", C.ASSIGN_SCALER_SCALE, np.round(scaler.scale_, 6).tolist(), near(scaler.scale_, C.ASSIGN_SCALER_SCALE, C.ASSIGN_SCALER_TOL))
    add("расхождений с kmeans_labels_final (2004 МО)", 0, mism, mism == 0)
    add("размеры типов 0–5", C.ASSIGN_TYPE_SIZES, sizes, sizes == C.ASSIGN_TYPE_SIZES)
    add("inertia", C.ASSIGN_INERTIA, round(float(km.inertia_), 3), abs(km.inertia_ - C.ASSIGN_INERTIA) <= C.ASSIGN_INERTIA_TOL)
    add("действующих МО", C.ASSIGN_EXPECT_ACTIVE, len(active), len(active) == C.ASSIGN_EXPECT_ACTIVE)
    add("МО вне выборки", len(active) - len(sample), len(outside), len(outside) == len(active) - len(sample))
    add("группа A", C.ASSIGN_EXPECT_A, len(group_a), len(group_a) == C.ASSIGN_EXPECT_A)
    add("группа B", C.ASSIGN_EXPECT_B, len(group_b), len(group_b) == C.ASSIGN_EXPECT_B)
    add("B с полными пятью долями за месяц", C.ASSIGN_EXPECT_B_COMPLETE, len(b_complete), len(b_complete) == C.ASSIGN_EXPECT_B_COMPLETE)
    add("B без полных долей за месяц", C.ASSIGN_EXPECT_B_INCOMPLETE, len(b_incomplete), len(b_incomplete) == C.ASSIGN_EXPECT_B_INCOMPLETE)
    for k in ("median", "p95", "max"):
        add(f"расстояние до центроида, выборка, {k}", C.ASSIGN_SAMPLE_DIST[k], round(sd[k], 3), abs(sd[k] - C.ASSIGN_SAMPLE_DIST[k]) <= C.ASSIGN_DIST_TOL)
    for k in ("median", "p95", "max"):
        add(f"расстояние до центроида, все МО B с долями, {k}", C.ASSIGN_B_DIST[k], round(bd[k], 3), abs(bd[k] - C.ASSIGN_B_DIST[k]) <= C.ASSIGN_DIST_TOL)
    add("МО с расстоянием больше максимума по выборке", C.ASSIGN_BEYOND_IDS, beyond, beyond == C.ASSIGN_BEYOND_IDS)
    add("присвоено (status = assigned)", C.ASSIGN_EXPECT_ASSIGNED, int(st.get("assigned", 0)), int(st.get("assigned", 0)) == C.ASSIGN_EXPECT_ASSIGNED)
    add("присвоено по типам 0–5", C.ASSIGN_ASSIGNED_BY_TYPE, by_type_assigned, by_type_assigned == C.ASSIGN_ASSIGNED_BY_TYPE)
    add("строк в таблице результата", len(outside), len(out), len(out) == len(outside) and out["territory_id"].is_unique)
    add("status: no_data_in_dataset / no_dec2024 / too_far", [C.ASSIGN_EXPECT_A, C.ASSIGN_EXPECT_B_INCOMPLETE, len(C.ASSIGN_BEYOND_IDS)],
        [int(st.get(k, 0)) for k in ("no_data_in_dataset", "no_dec2024", "too_far")],
        [int(st.get(k, 0)) for k in ("no_data_in_dataset", "no_dec2024", "too_far")] == [C.ASSIGN_EXPECT_A, C.ASSIGN_EXPECT_B_INCOMPLETE, len(C.ASSIGN_BEYOND_IDS)])

    print(f"sklearn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
    print("| контроль | ожидание | получено | результат |")
    for c in controls:
        print(f"| {c['name']} | {fmt(c['expect'])} | {fmt(c['got'])} | {'пройден' if c['ok'] else 'НЕ ПРОЙДЕН'} |")
    if not all(c["ok"] for c in controls):
        print("STOP: контроли не пройдены; файлы не записаны")
        raise SystemExit(1)

    # --- запись
    md = report(out, sd, bd, d_sample, d1, assigned_mask, by_type_assigned, group_b, months, C.MONTH, controls, len(sample))
    out.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    REPORT_PATH.write_text(md, encoding="utf-8")
    print("Готово.")


def report(out, sd, bd, d_sample, d1, assigned_mask, by_type_assigned, group_b, months, month, controls, n_sample) -> str:
    st = out["status"].value_counts()
    asg = out[out["status"] == "assigned"]
    da = asg["dist_nearest"].to_numpy()
    mb = months.reindex(group_b).astype(int)
    hist = mb.value_counts().sort_index()
    p95s = sd["p95"]
    L = [
        "# 24c. Типы МО вне рабочей выборки", "",
        "Сгенерировано `src/24c_assign_outside.py`. Все числа взяты из расчёта скрипта. Присвоенные типы не входят в расчёты шагов 10–28, "
        "в портреты типов и в проверки гипотез Г1–Г13; замороженные файлы шаг не меняет.", "",
        "## Метод", "",
        f"1. Действующие МО — территории из `data/geo/mo_national.geojson` (шаг 24б). Рабочая выборка — `kmeans_labels_final.parquet` ({n_sample} МО). "
        "МО вне выборки делятся на группу A (нет в `consumption.parquet`) и группу B (есть в `consumption.parquet`).",
        f"2. Для месяца {month} строятся доли пяти категорий тем же выражением, что в шаге 03: значение категории, делённое на «Все категории».",
        f"3. Доли выборки стандартизируются по МО выборки (`StandardScaler`, ddof = 0, как в `network_utils.standardize_shares`). "
        f"KMeans (k = {C.FINAL_K}, random_state = {C.KMEANS_RANDOM_STATE}, n_init = {C.KMEANS_N_INIT}, параметры из `config.yaml`) обучается заново на выборке: "
        "центроиды в файлах не хранятся. Разбиение совпало с `kmeans_labels_final.parquet` на всех МО выборки.",
        "4. МО группы B, у которых за месяц есть все пять категорий и «Все категории», стандартизируются параметрами выборки; "
        "для каждого считаются расстояния (евклид в z-пространстве) до ближайшего и второго центроида.",
        "5. Тип присваивается, если расстояние до ближайшего центроида не больше максимума по выборке. Остальные МО типа не получают.", "",
        "## Статусы", "", "| статус | число МО | смысл |", "|---|---|---|",
        f"| assigned | {int(st.get('assigned', 0))} | тип присвоен |",
        f"| no_data_in_dataset | {int(st.get('no_data_in_dataset', 0))} | группа A: нет в `consumption.parquet` |",
        f"| no_dec2024 | {int(st.get('no_dec2024', 0))} | группа B: за {month} нет пяти категорий и «Все категории» |",
        f"| too_far | {int(st.get('too_far', 0))} | группа B: расстояние до ближайшего центроида больше максимума по выборке |",
        f"| всего вне выборки | {len(out)} | |", "",
        "## Исключение МО группы B из выборки (коды шага 02)", "", "| exclusion_reason | число МО группы B |", "|---|---|"]
    for k, v in out[out["group"] == "B"]["exclusion_reason"].value_counts().items():
        L.append(f"| {k} | {int(v)} |")
    L += ["", "## Присвоенные типы", "", "| тип | МО вне выборки | МО в выборке |", "|---|---|---|"]
    sizes = C.ASSIGN_TYPE_SIZES
    for t in range(C.FINAL_K):
        L.append(f"| {t} | {by_type_assigned[t]} | {sizes[t]} |")
    L += ["", "## Расстояние до ближайшего центроида (z-пространство)", "",
          "| группа МО | n | медиана | p95 | максимум |", "|---|---|---|---|---|",
          f"| выборка | {len(d_sample)} | {sd['median']:.3f} | {sd['p95']:.3f} | {sd['max']:.3f} |",
          f"| все МО группы B с долями за {month} (до применения правила) | {len(d1)} | {bd['median']:.3f} | {bd['p95']:.3f} | {bd['max']:.3f} |",
          f"| присвоенные | {len(da)} | {np.median(da):.3f} | {pct(da, 95):.3f} | {da.max():.3f} |", ""]
    L.append(f"Медиана отношения расстояния до второго центроида к расстоянию до ближайшего у присвоенных МО: {asg['ratio'].median():.2f}.")
    L.append("")
    L += ["## Ограничения", "",
          "- Типы присвоены МО вне обучающей выборки: эти МО не входили ни в обучение KMeans, ни в стандартизацию, ни в статистики и проверки шагов 10–28.",
          f"- У МО группы B неполная история в `consumption.parquet` (число месяцев с данными по группе B: {', '.join(f'{int(k)}: {int(v)}' for k, v in hist.items())}); "
          f"полные {N_MONTHS_REQUIRED} месяца есть только у МО группы B без `market_access` ({int((mb == N_MONTHS_REQUIRED).sum())} МО).",
          f"- Расстояние до центроида у присвоенных МО больше, чем у выборки: медиана {np.median(da):.3f} против {sd['median']:.3f}, p95 {pct(da, 95):.3f} против {p95s:.3f}; "
          f"выше p95 выборки {int((da > p95s).sum())} из {len(da)} присвоенных МО.",
          "- Присвоенные типы не входят в статистики, портреты типов и проверки гипотез Г1–Г13; на карте они помечаются отдельно.",
          "- Для МО без типа (группа A, нет данных за месяц, слишком далёкие от центроидов) тип не присваивается и не оценивается.", ""]
    L += ["## Контроли", "", "| контроль | ожидание | получено | результат |", "|---|---|---|---|"]
    for c in controls:
        L.append(f"| {c['name']} | {fmt(c['expect'])} | {fmt(c['got'])} | {'пройден' if c['ok'] else 'НЕ ПРОЙДЕН'} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
