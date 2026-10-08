"""Шаг 24f. Данные слоёв лендинга: доступность рынков, устойчивость типа, покрытие по регионам, тип по последнему месяцу.

Только чтение готовых файлов, новых моделей нет. Что делает:
  1. собирает таблицу из 2594 МО (границы шага 24б): класс 1 — выборка (kmeans_labels_final), 2 — тип присвоен вне
     выборки (types_outside_sample, шаг 24c), 3 — без типа, 4 — тип по последнему месяцу (types_last_month, шаг 24g);
  2. добавляет market_access (data/raw/market_access.parquet), для класса 1 — число месяцев из 24, в которых метка
     траектории (колонка cluster шага 12) равна канонической, и долю месяцев (функция load_wide и выражение доли месяцев
     взяты из шага 28 — импортом и тем же выражением, не переписаны);
  3. считает покрытие по 85 регионам.
Все контроли выполняются до записи файлов; при расхождении печатается таблица контролей и код выхода 1.
Вход:  data/geo/mo_national.geojson, data/processed/{kmeans_labels_final, types_outside_sample, types_last_month,
       kmeans_k6_trajectories, kmeans_k6_matching, type_portraits}.parquet, data/raw/market_access.parquet
Выход: data/processed/layer_data_24f.parquet, data/processed/region_coverage_24f.parquet, notebooks/24f_layer_data.md
Запуск из корня проекта:  .venv/bin/python src/24f_layer_data.py [--no-write]
"""
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import config as C

PROJECT_DIR = Path(__file__).resolve().parents[1]
L = C.LAYERS
P = {k: PROJECT_DIR / L[k] for k in ("GEO_PATH", "LABELS_PATH", "OUTSIDE_PATH", "LAST_MONTH_PATH", "MARKET_ACCESS_PATH",
                                     "TRAJECTORIES_PATH", "PORTRAITS_PATH", "MATCHING_PATH",
                                     "OUT_LAYER_PATH", "OUT_REGION_PATH", "REPORT_PATH")}
N_MONTHS = L["N_MONTHS"]
TYPES = list(range(C.FINAL_K))
CLASS_NAMES = {1: "выборка", 2: "присвоено вне выборки", 3: "без типа", 4: "тип по последнему месяцу"}

m28 = importlib.import_module("28_interpretation_checks")   # load_wide и выражение доли месяцев — из шага 28
assert Path(m28.TRAJ_PATH).resolve() == P["TRAJECTORIES_PATH"].resolve(), "траектории шага 28 и шага 24f различаются"


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


controls: list = []


def ctrl(name: str, got, exp, tol=0.0, fmt="{}") -> bool:
    """Числовая сверка с допуском (tol = 0 — точно) или сверка списков/множеств/строк."""
    if isinstance(got, (list, tuple, np.ndarray)):
        got = [float(x) for x in got]
        exp = [float(x) for x in exp]
        ok = len(got) == len(exp) and all(abs(a - b) <= tol + 1e-12 for a, b in zip(got, exp))
        g, e = ", ".join(fmt.format(x) for x in got), ", ".join(fmt.format(x) for x in exp)
    elif isinstance(got, (set, frozenset)):
        ok, g, e = got == exp, "; ".join(sorted(got)), "; ".join(sorted(exp))
    else:
        ok = abs(float(got) - float(exp)) <= tol + 1e-12
        g, e = fmt.format(got), fmt.format(exp)
    controls.append({"контроль": name, "результат": g, "ожидание": e, "допуск": f"{tol:g}",
                     "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})
    return ok


def finish_controls() -> pd.DataFrame:
    df = pd.DataFrame(controls)
    print(df.to_markdown(index=False, disable_numparse=True))
    bad = df[df["статус"] != "совпало"]
    if len(bad):
        print(f"\nРасхождений: {len(bad)}")
        sys.exit(1)
    return df


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    write = "--no-write" not in sys.argv
    for k, p in P.items():
        if not k.startswith("OUT_") and k != "REPORT_PATH" and not p.exists():
            stop(f"нет файла {p}")
    for rel, expected in L["FROZEN_SHA256"].items():
        got = sha256(PROJECT_DIR / rel)
        if got != expected:
            stop(f"sha256 {rel} = {got}, ожидается {expected}")

    # --- входы
    geo = json.loads(P["GEO_PATH"].read_text(encoding="utf-8"))
    g = pd.DataFrame([f["properties"] for f in geo["features"]])[["territory_id", "name", "region_name", "in_sample"]]
    if not g["territory_id"].is_unique:
        stop("в границах повторяются territory_id")
    g = g.sort_values("territory_id").reset_index(drop=True)
    labels = pd.read_parquet(P["LABELS_PATH"]).sort_values("territory_id").reset_index(drop=True)
    outside = pd.read_parquet(P["OUTSIDE_PATH"]).sort_values("territory_id").reset_index(drop=True)
    lastm = pd.read_parquet(P["LAST_MONTH_PATH"]).sort_values("territory_id").reset_index(drop=True)
    ma = pd.read_parquet(P["MARKET_ACCESS_PATH"])

    ids_sample = labels["territory_id"].astype(int).to_numpy()
    canon = labels["cluster"].astype(int).to_numpy()
    structural = {
        "в таблице границ территории выборки = kmeans_labels_final": set(g.loc[g["in_sample"], "territory_id"]) == set(ids_sample),
        "выборка и вне выборки не пересекаются, вместе дают все границы": (
            not (set(ids_sample) & set(outside["territory_id"])) and set(ids_sample) | set(outside["territory_id"]) == set(g["territory_id"])),
        "territory_id в types_outside_sample и types_last_month уникальны": outside["territory_id"].is_unique and lastm["territory_id"].is_unique,
        "market_access: territory_id уникальны, без пропусков, все есть в границах": (
            ma["territory_id"].is_unique and not ma["market_access"].isna().any() and set(ma["territory_id"]) <= set(g["territory_id"])),
        "статусы types_outside_sample из известного набора": set(outside["status"]) <= {"assigned", "no_data_in_dataset", "no_dec2024", "too_far"},
        "статусы types_last_month из известного набора": set(lastm["status"]) <= {"assigned_last_month", "month_not_eligible", "too_far", "no_complete_month"},
        "types_last_month только из no_dec2024": set(lastm["territory_id"]) <= set(outside.loc[outside["status"] == "no_dec2024", "territory_id"]),
        "тип задан ровно у assigned и assigned_last_month": (
            bool(outside["type"].notna().eq(outside["status"] == "assigned").all()) and bool(lastm["type"].notna().eq(lastm["status"] == "assigned_last_month").all())),
        "типы в диапазоне 0–5": set(outside["type"].dropna().astype(int)) | set(lastm["type"].dropna().astype(int)) | set(canon) <= set(TYPES),
        "region_name в types_last_month совпадает с границами": bool(
            (lastm.set_index("territory_id")["region_name"] == g.set_index("territory_id")["region_name"].reindex(lastm["territory_id"])).all()),
        "last_month — строка YYYY-MM": bool(lastm["last_month"].str.fullmatch(r"\d{4}-\d{2}").all()),
        "dist_nearest непусто у assigned и assigned_last_month": (
            bool(outside.loc[outside["status"] == "assigned", "dist_nearest"].notna().all())
            and bool(lastm.loc[lastm["status"] == "assigned_last_month", "dist_nearest"].notna().all())),
    }
    for name, ok in structural.items():
        controls.append({"контроль": name, "результат": str(bool(ok)), "ожидание": "True", "допуск": "0",
                         "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})

    # --- устойчивость типа: траектории шага 12 через load_wide шага 28 и то же выражение доли месяцев
    wide = m28.load_wide(ids_sample)
    W = wide.to_numpy()
    if W.shape != (len(ids_sample), N_MONTHS):
        stop(f"траектории: {W.shape}")
    months_same = (W == canon[:, None]).sum(axis=1)
    share_in = (W == canon[:, None]).mean(axis=1)   # то же выражение, что в шаге 28, п. B3
    stable_share = months_same / N_MONTHS
    dec_differs = W[:, -1] != canon
    ari_dec = float(adjusted_rand_score(canon, W[:, -1]))
    controls.append({"контроль": "months_same / 24 совпадает с выражением шага 28 (макс. расхождение)",
                     "результат": f"{np.abs(stable_share - share_in).max():.1e}", "ожидание": "0", "допуск": "1e-12",
                     "статус": "совпало" if np.abs(stable_share - share_in).max() <= 1e-12 else "РАСХОЖДЕНИЕ"})
    if wide.columns[-1] != C.MONTH:
        stop(f"последний месяц траекторий {wide.columns[-1]}, ожидается {C.MONTH}")

    # --- таблица МО
    t = g[["territory_id", "name", "region_name"]].copy()
    t["territory_id"] = t["territory_id"].astype("int32")
    o = outside.set_index("territory_id")
    lm = lastm.set_index("territory_id")
    in_s = t["territory_id"].isin(ids_sample).to_numpy()
    in_a = t["territory_id"].map(o["status"]).eq("assigned").fillna(False).to_numpy(bool)
    in_4 = t["territory_id"].map(lm["status"]).eq("assigned_last_month").fillna(False).to_numpy(bool)
    cls = np.where(in_s, 1, np.where(in_a, 2, np.where(in_4, 4, 3))).astype("int8")
    t["class"] = cls
    typ = pd.Series(pd.array([pd.NA] * len(t), dtype="Int8"))
    typ[cls == 1] = pd.Series(canon, index=ids_sample).reindex(t.loc[cls == 1, "territory_id"]).to_numpy()
    typ[cls == 2] = o["type"].reindex(t.loc[cls == 2, "territory_id"]).to_numpy()
    typ[cls == 4] = lm["type"].reindex(t.loc[cls == 4, "territory_id"]).to_numpy()
    t["type"] = typ.astype("Int8")
    st = pd.Series(pd.array([pd.NA] * len(t), dtype="string"))
    out_status = t["territory_id"].map(o["status"])
    st[cls == 3] = out_status[cls == 3].map({"no_data_in_dataset": "no_data_in_dataset", "too_far": "too_far",
                                              "no_dec2024": "no_dec2024_no_type"}).to_numpy()
    t["status"] = st.astype("string")
    t["market_access"] = t["territory_id"].map(ma.set_index("territory_id")["market_access"]).astype("float64")
    k1 = pd.Series(np.arange(len(ids_sample)), index=ids_sample)
    row1 = k1.reindex(t["territory_id"]).to_numpy()
    c1 = cls == 1
    ms = pd.Series(pd.array([pd.NA] * len(t), dtype="Int8"))
    ms[c1] = months_same[row1[c1].astype(int)]
    t["months_same"] = ms.astype("Int8")
    ss = np.full(len(t), np.nan)
    ss[c1] = stable_share[row1[c1].astype(int)]
    t["stable_share"] = ss
    dd = pd.Series(pd.array([pd.NA] * len(t), dtype="boolean"))
    dd[c1] = dec_differs[row1[c1].astype(int)]
    t["dec_label_differs"] = dd.astype("boolean")
    lmo = pd.Series(pd.array([pd.NA] * len(t), dtype="string"))
    lmo[cls == 4] = lm["last_month"].reindex(t.loc[cls == 4, "territory_id"]).to_numpy()
    t["last_month"] = lmo.astype("string")
    ag = np.full(len(t), np.nan)
    ag[cls == 4] = lm["month_agreement"].reindex(t.loc[cls == 4, "territory_id"]).to_numpy()
    t["month_agreement"] = ag
    dn = np.full(len(t), np.nan)
    dn[cls == 2] = o["dist_nearest"].reindex(t.loc[cls == 2, "territory_id"]).to_numpy()
    dn[cls == 4] = lm["dist_nearest"].reindex(t.loc[cls == 4, "territory_id"]).to_numpy()
    t["dist_nearest"] = dn
    t = t[["territory_id", "name", "region_name", "class", "type", "status", "market_access", "months_same",
           "stable_share", "dec_label_differs", "last_month", "month_agreement", "dist_nearest"]]

    n4 = int(len(lastm.loc[lastm["status"] == "assigned_last_month"]))
    cnt = t["class"].value_counts().to_dict()
    print(f"N4 (status = assigned_last_month в types_last_month.parquet) = {n4}")

    # --- контроли таблицы МО
    ctrl("строк в layer_data", len(t), L["EXPECT_ROWS"])
    ctrl("territory_id отсортированы и уникальны (1 = да)", int(t["territory_id"].is_monotonic_increasing and t["territory_id"].is_unique), 1)
    ctrl("класс 1: выборка", cnt.get(1, 0), L["EXPECT_CLASS_SAMPLE"])
    ctrl("класс 2: присвоено вне выборки", cnt.get(2, 0), L["EXPECT_CLASS_ASSIGNED"])
    ctrl("класс 4: тип по последнему месяцу (N4 из файла)", cnt.get(4, 0), n4)
    ctrl("класс 3: без типа = 492 − N4", cnt.get(3, 0), L["EXPECT_NO_TYPE_TOTAL"] - n4)
    ctrl("no_data_in_dataset", int((t["status"] == "no_data_in_dataset").sum()), L["EXPECT_NO_DATA"])
    ctrl("too_far (класс 3, исходный)", int((t["status"] == "too_far").sum()), L["EXPECT_TOO_FAR"])
    ctrl("no_dec2024_no_type = 84 − N4", int((t["status"] == "no_dec2024_no_type").sum()), 84 - n4)
    ctrl("тип пуст ровно у класса 3 (1 = да)", int(bool(t["type"].isna().eq(t["class"] == 3).all())), 1)
    ctrl("status непусто ровно у класса 3 (1 = да)", int(bool(t["status"].notna().eq(t["class"] == 3).all())), 1)
    for col, cl in (("months_same", 1), ("stable_share", 1), ("dec_label_differs", 1), ("last_month", 4), ("month_agreement", 4)):
        ctrl(f"{col} непусто ровно у класса {cl} (1 = да)", int(bool(t[col].notna().eq(t["class"] == cl).all())), 1)
    ctrl("dist_nearest непусто ровно у классов 2 и 4 (1 = да)", int(bool(t["dist_nearest"].notna().eq(t["class"].isin([2, 4])).all())), 1)
    m = t["market_access"]
    ctrl("market_access непусто, всего", int(m.notna().sum()), L["EXPECT_MA_NONNULL"])
    ctrl("market_access непусто у класса 1", int(m[t["class"] == 1].notna().sum()), L["EXPECT_MA_CLASS1"])
    ctrl("market_access непусто у класса 2", int(m[t["class"] == 2].notna().sum()), L["EXPECT_MA_CLASS2"])
    ctrl("market_access непусто у классов 1+2+3+4 в сумме", int(sum(m[t["class"] == c].notna().sum() for c in (1, 2, 3, 4))), L["EXPECT_MA_NONNULL"])
    ctrl("market_access: минимум и максимум", [m.min(), m.max()], L["EXPECT_MA_RANGE"], 1e-9, "{:.1f}")
    ctrl("market_access: медиана по непустым", float(m.median()), L["EXPECT_MA_MEDIAN"], L["MA_MEDIAN_TOL"], "{:.2f}")
    ctrl("МО без market_access (справочно: 2594 − 2571)", int(m.isna().sum()), L["EXPECT_ROWS"] - L["EXPECT_MA_NONNULL"])

    # --- покрытие по регионам
    gb = t.groupby("region_name", sort=True)
    reg = pd.DataFrame({
        "n_mo": gb.size(),
        "n_sample": gb["class"].apply(lambda s: int((s == 1).sum())),
        "n_assigned": gb["class"].apply(lambda s: int((s == 2).sum())),
        "n_last_month": gb["class"].apply(lambda s: int((s == 4).sum())),
        "n_no_data_in_dataset": gb["status"].apply(lambda s: int((s == "no_data_in_dataset").sum())),
        "n_no_dec2024_no_type": gb["status"].apply(lambda s: int((s == "no_dec2024_no_type").sum())),
        "n_too_far": gb["status"].apply(lambda s: int((s == "too_far").sum())),
    }).reset_index()
    reg["n_no_type"] = reg["n_no_data_in_dataset"] + reg["n_no_dec2024_no_type"] + reg["n_too_far"]
    reg["share_no_type"] = reg["n_no_type"] / reg["n_mo"]
    for c in reg.columns:
        if c.startswith("n_"):
            reg[c] = reg[c].astype("int32")
    reg = reg[["region_name", "n_mo", "n_sample", "n_assigned", "n_last_month", "n_no_data_in_dataset",
               "n_no_dec2024_no_type", "n_too_far", "n_no_type", "share_no_type"]]
    ctrl("строк в region_coverage", len(reg), L["EXPECT_REGIONS"])
    ctrl("сумма n_mo", int(reg["n_mo"].sum()), L["EXPECT_ROWS"])
    sum4 = reg["n_sample"] + reg["n_assigned"] + reg["n_last_month"] + reg["n_no_type"]
    ctrl("регионов, где n_sample + n_assigned + n_last_month + n_no_type ≠ n_mo", int((sum4 != reg["n_mo"]).sum()), 0)
    ctrl("n_no_type = n_no_data + n_no_dec2024_no_type + n_too_far во всех регионах (регионов с расхождением)", int(
        (reg["n_no_type"] != reg["n_no_data_in_dataset"] + reg["n_no_dec2024_no_type"] + reg["n_too_far"]).sum()), 0)
    ctrl("регион МО в layer_data = регион границ (регионов в layer_data)", t["region_name"].nunique(), L["EXPECT_REGIONS"])
    empty = set(reg.loc[reg["n_sample"] == 0, "region_name"])
    ctrl("регионов с n_sample = 0", len(empty), len(L["EXPECT_EMPTY_SAMPLE_REGIONS"]))
    ctrl("множество регионов с n_sample = 0", empty, set(L["EXPECT_EMPTY_SAMPLE_REGIONS"]))
    for rname, n_exp in L["EXPECT_REGION_N"].items():
        r = reg[reg["region_name"] == rname]
        ok = len(r) == 1 and int(r["n_mo"].iloc[0]) == n_exp and int(r["n_no_data_in_dataset"].iloc[0]) == n_exp
        controls.append({"контроль": f"{rname}: n_mo = n_no_data_in_dataset = {n_exp}",
                         "результат": f"{int(r['n_mo'].iloc[0])} / {int(r['n_no_data_in_dataset'].iloc[0])}" if len(r) == 1 else "нет региона",
                         "ожидание": f"{n_exp} / {n_exp}", "допуск": "0", "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})

    # --- устойчивость типа
    s1 = t[t["class"] == 1]
    sh = [float((s1.loc[s1["type"] == k, "stable_share"] >= C.STABLE_SHARE - 1e-12).mean()) for k in TYPES]
    ctrl(f"доля МО класса 1 с stable_share ≥ {C.STABLE_SHARE}, типы 0–5", sh, L["EXPECT_STABLE_BY_TYPE"], L["STABLE_BY_TYPE_TOL"], "{:.3f}")
    q = np.quantile(s1["stable_share"].to_numpy(float), L["STABLE_QUANTILES"])
    ctrl("квантили stable_share 5/25/50/75/95", q, L["EXPECT_STABLE_QUANTILES"], L["STABLE_QUANTILES_TOL"], "{:.4f}")
    full = s1["stable_share"] == 1.0
    ctrl("МО с stable_share = 1.0, всего", int(full.sum()), L["EXPECT_FULL_STABLE"])
    ctrl("МО с stable_share = 1.0, типы 0–5", [int((full & (s1["type"] == k)).sum()) for k in TYPES], L["EXPECT_FULL_STABLE_BY_TYPE"], 0, "{:.0f}")
    ctrl("МО с dec_label_differs", int(s1["dec_label_differs"].sum()), L["EXPECT_DEC_DIFFERS"])
    ctrl("ARI меток декабря 2024 в траекториях и kmeans_labels_final", ari_dec, L["EXPECT_ARI_DEC"], L["ARI_DEC_TOL"], "{:.4f}")
    tp = pd.read_parquet(P["PORTRAITS_PATH"])
    cp = tp[(tp["metric"] == "уверенных сопоставлений из 23") & (tp["block"] == "надёжность")]
    conf = [int(cp.loc[cp["type"].astype(str) == str(k), "value"].iloc[0]) for k in TYPES]
    ctrl("уверенных сопоставлений из 23 по type_portraits, типы 0–5", conf, L["EXPECT_CONFIDENT_BY_TYPE"], 0, "{:.0f}")
    mt = pd.read_parquet(P["MATCHING_PATH"])
    mt = mt[mt["месяц"] != C.MONTH]
    conf_m = [int(((mt["cluster"] == k) & (mt["уверенность"] == "уверенно")).sum()) for k in TYPES]
    ctrl("то же по kmeans_k6_matching (независимая сверка)", conf_m, L["EXPECT_CONFIDENT_BY_TYPE"], 0, "{:.0f}")

    # --- справочно: регионы с большой долей МО без типа до и после шага 24g
    thr = L["COVERAGE_SHARE_MIN"]
    reg["share_before_24g"] = (reg["n_no_type"] + reg["n_last_month"]) / reg["n_mo"]
    before = reg[reg["share_before_24g"] >= thr - 1e-12]
    after = reg[reg["share_no_type"] >= thr - 1e-12]
    left = sorted(set(before["region_name"]) - set(after["region_name"]))
    print(f"\nРегионов с share_no_type ≥ {thr} до шага 24g (класс 4 считается без типа): {len(before)} (ожидалось {L['EXPECT_COVERAGE_BEFORE_24G']})")
    print(f"Регионов с share_no_type ≥ {thr} после шага 24g: {len(after)}")
    print("Список после:", "; ".join(f"{r.region_name} {r.n_no_type}/{r.n_mo}" for r in after.itertuples()))
    print("Вышли из списка:", "; ".join(left) if left else "нет")
    print()

    df_ctrl = finish_controls()

    # --- md
    md = build_md(t, reg, df_ctrl, n4, cnt, before, after, left, ari_dec, mt)
    low = md.lower()
    bad_words = [w for w in L["FORBIDDEN_WORDS"] if w in low]
    if bad_words:
        stop(f"в тексте запрещённые слова: {bad_words}")
    rest = re.sub(r"\[±\d+\.\d\d; ±\d+\.\d\d\]", "", md)   # единственный допустимый формат квадратных скобок
    if "[" in rest or "]" in rest:
        stop("в тексте квадратные скобки вне формата интервала")
    if "км²" in md or "км2" in low:
        stop("в тексте площади")
    if write:
        P["OUT_LAYER_PATH"].parent.mkdir(parents=True, exist_ok=True)
        t.to_parquet(P["OUT_LAYER_PATH"], engine="pyarrow", index=False)
        reg.drop(columns=["share_before_24g"]).to_parquet(P["OUT_REGION_PATH"], engine="pyarrow", index=False)
        with open(P["REPORT_PATH"], "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
        print("Записано:", P["OUT_LAYER_PATH"].name, P["OUT_REGION_PATH"].name, P["REPORT_PATH"].name)


def table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, disable_numparse=True)


def build_md(t, reg, df_ctrl, n4, cnt, before, after, left, ari_dec, mt) -> str:
    out = []
    add = out.append
    s1 = t[t["class"] == 1]
    add("# Шаг 24f. Данные слоёв лендинга")
    add("")
    add("Описательный шаг: только чтение готовых файлов, новых моделей, тестов и порогов нет. "
        "Скрипт `src/24f_layer_data.py`; параметры и ожидаемые значения — группа `step24f_layers` в `config.yaml`.")
    add("")
    add("Входы: `data/geo/mo_national.geojson`, `kmeans_labels_final.parquet`, `types_outside_sample.parquet` (шаг 24c), "
        "`types_last_month.parquet` (шаг 24g), `data/raw/market_access.parquet`, `kmeans_k6_trajectories.parquet`, "
        "`kmeans_k6_matching.parquet`, `type_portraits.parquet`. "
        "Выходы: `layer_data_24f.parquet` (одна строка на МО), `region_coverage_24f.parquet` (одна строка на регион).")
    add("")
    add("## 1. Определения полей `layer_data_24f.parquet`")
    add("")
    fields = [
        ("territory_id", "int32", "идентификатор МО; строки отсортированы по нему"),
        ("name, region_name", "string", "название МО и региона из `mo_national.geojson`"),
        ("class", "int8", "1 — МО выборки; 2 — тип присвоен вне выборки (шаг 24c); 3 — без типа; 4 — тип по последнему месяцу (шаг 24g)"),
        ("type", "Int8", "тип 0–5; пусто у класса 3. Класс 1 — канонический тип декабря 2024, классы 2 и 4 — ближайший центроид"),
        ("status", "string", "только класс 3: `no_data_in_dataset`; `no_dec2024_no_type` (нет данных за декабрь 2024 и тип по последнему месяцу не присвоен: "
                             "`month_not_eligible`, `too_far`, `no_complete_month` шага 24g объединены); `too_far` (исходный статус шага 24c)"),
        ("market_access", "float64", "индекс доступности рынков 2024 из `market_access.parquet`; пусто, если значения нет"),
        ("months_same", "Int8", "только класс 1: число из 24 месяцев, в которых метка траектории (колонка `cluster` шага 12, после сопоставления с якорем) равна канонической"),
        ("stable_share", "float64", "только класс 1: `months_same` / 24"),
        ("dec_label_differs", "boolean", "только класс 1: метка декабря 2024 в траекториях отличается от канонической"),
        ("last_month", "string", "только класс 4: последний месяц с полными пятью категориями, YYYY-MM"),
        ("month_agreement", "float64", "только класс 4: согласие метода с месячными метками на выборке в этом месяце (шаг 24g)"),
        ("dist_nearest", "float64", "классы 2 и 4: расстояние до ближайшего центроида (в пространстве шагов 24c и 24g)"),
    ]
    add(table(pd.DataFrame(fields, columns=["поле", "тип", "определение"])))
    add("")
    add("Пустые значения хранятся как пропуски (`NA`/`NaN`); для `dec_label_differs` используется тип `boolean` с пропусками, чтобы не путать «нет» и «не определено».")
    add("")
    add("Поля `region_coverage_24f.parquet`: `region_name`, `n_mo`, `n_sample` (класс 1), `n_assigned` (класс 2), `n_last_month` (класс 4), "
        "`n_no_data_in_dataset`, `n_no_dec2024_no_type`, `n_too_far`, `n_no_type` (их сумма), `share_no_type` = `n_no_type` / `n_mo`.")
    add("")
    add("## 2. Классы")
    add("")
    st = t["status"].value_counts()
    rows = [(1, CLASS_NAMES[1], cnt[1], ""), (2, CLASS_NAMES[2], cnt[2], ""), (4, CLASS_NAMES[4], cnt[4], f"N4 = {n4} из `types_last_month.parquet`")]
    rows.append((3, CLASS_NAMES[3], cnt[3], "; ".join(f"{k}: {int(st[k])}" for k in ("no_data_in_dataset", "no_dec2024_no_type", "too_far"))))
    add(table(pd.DataFrame(rows, columns=["класс", "смысл", "МО", "примечание"])))
    add("")
    lmc = t.loc[t["class"] == 4, "last_month"].value_counts().sort_index()
    add("Последний месяц класса 4: " + ", ".join(f"{k} — {int(v)}" for k, v in lmc.items()) + ".")
    add("")
    add("## 3. Контроли (выполнены до записи файлов)")
    add("")
    add(table(df_ctrl))
    add("")
    add("## 4. Устойчивость типа МО выборки")
    add("")
    add("Доля месяцев считается по колонке `cluster` траекторий шага 12 относительно канонической метки (функция `load_wide` и выражение доли месяцев "
        "из `src/28_interpretation_checks.py`, п. B3). Порог `STABLE_SHARE` из `config.yaml` (группа `step28_checks`).")
    add("")
    rows = []
    for k in TYPES:
        x = s1[s1["type"] == k]
        rows.append((k, len(x), f"{(x['stable_share'] >= C.STABLE_SHARE - 1e-12).mean():.3f}", int((x["stable_share"] == 1.0).sum()),
                     int(x["dec_label_differs"].sum()), f"{x['stable_share'].median():.4f}"))
    add(table(pd.DataFrame(rows, columns=["тип", "МО", f"доля с stable_share ≥ {C.STABLE_SHARE}", "stable_share = 1.0", "dec_label_differs", "медиана stable_share"])))
    add("")
    q = np.quantile(s1["stable_share"].to_numpy(float), L["STABLE_QUANTILES"])
    add("Квантили `stable_share` по МО выборки 5/25/50/75/95: " + ", ".join(f"{v:.4f}" for v in q) + ".")
    add("")
    add(f"ARI между метками декабря 2024 в траекториях и `kmeans_labels_final`: {ari_dec:.4f}.")
    add("")
    add("## 5. Покрытие по регионам")
    add("")
    add(f"Регионов с `share_no_type` не менее {L['COVERAGE_SHARE_MIN']}: {len(after)}. До присвоения типов по последнему месяцу (класс 4 считался без типа) — {len(before)}.")
    add("")
    cols = ["region_name", "n_mo", "n_sample", "n_assigned", "n_last_month", "n_no_type", "share_no_type"]
    a = after[cols].copy()
    a["share_no_type"] = a["share_no_type"].map(lambda v: f"{v:.3f}")
    add(table(a.sort_values(["share_no_type", "region_name"], ascending=[False, True])))
    add("")
    if left:
        b = before[before["region_name"].isin(left)][["region_name", "n_mo", "n_no_type", "n_last_month", "share_no_type", "share_before_24g"]].copy()
        b["share_no_type"] = b["share_no_type"].map(lambda v: f"{v:.3f}")
        b["share_before_24g"] = b["share_before_24g"].map(lambda v: f"{v:.3f}")
        add("Регионы, вышедшие из списка после присвоения типов классу 4 (в них доля МО без типа опустилась ниже порога, потому что МО класса 4 перестали считаться без типа):")
        add("")
        add(table(b.rename(columns={"share_no_type": "share_no_type после", "share_before_24g": "доля до шага 24g"})))
    else:
        add("Ни один регион не вышел из списка после присвоения типов классу 4.")
    add("")
    e = reg[reg["n_sample"] == 0]
    add(f"Регионов без МО выборки (`n_sample` = 0): {len(e)} — " + "; ".join(e["region_name"]) + ".")
    add("")
    add("## 6. Ограничения")
    add("")
    add(f"1. Доля месяцев считается относительно канонической метки декабря 2024 (seed {C.CANONICAL_SEED}). Месячные метки в траекториях получены из официального "
        "разбиения каждого месяца (минимум inertia из 100 запусков, шаг 12a; для декабря 2024 это seed 66) с последующим сопоставлением с якорем. "
        f"Метка декабря в траекториях отличается от канонической у {int(s1['dec_label_differs'].sum())} МО (ARI {ari_dec:.4f}), "
        "поэтому у этих МО `months_same` не больше 23 по построению сравнения.")
    w = C.LAYERS["WEAK_TYPE"]
    mw = mt[mt["cluster"] == w]
    amb = ", ".join(mw.loc[mw["уверенность"] == "неоднозначно", "месяц"])
    mod = ", ".join(mw.loc[mw["уверенность"] == "умеренно", "месяц"])
    n_conf_w = int((mw["уверенность"] == "уверенно").sum())
    others = [23 - int(((mt["cluster"] == k) & (mt["уверенность"] == "уверенно")).sum()) for k in TYPES if k != w]
    add(f"2. Сопоставление типов между месяцами неуверенное для типа {w}: неоднозначно в месяцах {amb}; умеренно в месяцах {mod} "
        f"(по `kmeans_k6_matching.parquet`; уверенных сопоставлений {n_conf_w} из 23). У остальных типов неуверенных сопоставлений не больше {max(others)} из 23. "
        f"У МО типа {w} `stable_share` отражает в том числе качество сопоставления месяцев, а не только смену типа самим МО.")
    nm = int(t["market_access"].isna().sum())
    noma = "; ".join(f"{k} — {int(v)}" for k, v in t.loc[t["market_access"].isna(), "region_name"].value_counts().sort_index().items())
    add(f"3. `market_access` есть не у всех МО: пусто у {nm} (у класса 1 — {int(t.loc[t['class'] == 1, 'market_access'].isna().sum())}); "
        f"регионы пропусков: {noma}. В слоях такие МО нужно показывать как «нет значения».")
    lo, hi = t.loc[t["class"] == 4, "last_month"].min(), t.loc[t["class"] == 4, "last_month"].max()
    add(f"4. Класс 4 присвоен по месяцу, отличному от декабря 2024 (у {n4} МО последний полный месяц — от {lo} до {hi}); тип класса 4 "
        "получен по параметрам этого месяца и не входит в статистики шагов 10–28 и в карту 24d. Качество присвоения зависит от согласия метода "
        "с месячными метками на выборке (`month_agreement`).")
    add("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
