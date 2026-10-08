"""Шаг 19c. Вердикты по гипотезам: механическое применение правил к результатам шагов 19a и 19b.

Ничего не пересчитывается: берутся p, confirmed_level, метки размера и эталон из hypothesis_results_19a/19b
(их sha256 сверяются с записанными в .sha256). Поправка Холма (ALPHA) — отдельно по семействам и уровням;
«подтверждено» = confirmed_level и скорректированный p < ALPHA. Для Г1, Г2, Г3, Г7, Г8 — четыре ступени по
сочетанию уровней «общий» и «внутри регионов»; Г4 — число и список отраслей; Г6, Г9 — один уровень.

Вход:  data/processed/hypothesis_results_19a.parquet (+ .sha256), data/processed/hypothesis_results_19b.parquet (+ .sha256)
Выход: data/processed/hypothesis_verdicts.parquet, notebooks/19_hypothesis_verdicts.md
Запуск из корня проекта:  .venv/bin/python src/19c_hypothesis_verdicts.py
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import ALPHA, MONTHS_RULE
from hypothesis_tools import LEVEL_ALL, LEVEL_REGION, LEVELS

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_19A = PROCESSED_DIR / "hypothesis_results_19a.parquet"
RESULTS_19B = PROCESSED_DIR / "hypothesis_results_19b.parquet"
VERDICTS_PATH = PROCESSED_DIR / "hypothesis_verdicts.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "19_hypothesis_verdicts.md"

CONFIRMED, NOT_CONFIRMED = "подтверждено", "не подтверждено"
STAGES = {(True, True): "эффект есть и между регионами, и внутри регионов",
          (True, False): "различие есть, но его объясняет состав регионов",
          (False, True): "внутри региона различие есть, на общем уровне оно перекрыто различиями между регионами",
          (False, False): "не подтверждено"}
SMALL_LABELS = {"ничтожный", "малый"}
SMALL_NOTE = "устойчиво, но мало по размеру"
OPPOSITE_NOTE = "знак противоположен ожидаемому (|ρ| больше двустороннего p95)"
EXPLORATORY_NOTE = "разведочная"
LIMITATIONS = [
    "Зарплата и численность работников считаются по месту нахождения организации и только по организациям без "
    "малого предпринимательства (Росстат не формирует эти показатели по полному кругу организаций на уровне МО).",
    "У столичных МО нет возрастных данных Санкт-Петербурга (в БД ПМО нет возрастной структуры для его "
    "внутригородских МО), поэтому Г7 с остатком по возрасту считается без них.",
    "Г9 имеет малую мощность: ряды длиной 23 месяца.",
    "Переходы одного МО зависимы: точный p для Г6 оптимистичен; бутстреп по МО это учитывает только в интервале.",
]


def check_frozen() -> list:
    log = []
    for path in (RESULTS_19A, RESULTS_19B):
        sha_file = path.with_suffix(".sha256")
        if not sha_file.exists():
            raise SystemExit(f"STOP: нет {sha_file.name}")
        expected = sha_file.read_text(encoding="utf-8").split()[0]
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        log.append(f"- sha256 `{path.name}`: `{got}` — {'совпадает' if got == expected else 'НЕ СОВПАДАЕТ'} с `{sha_file.name}`")
        print(f"sha256 {path.name} {got} {'OK' if got == expected else 'MISMATCH'}")
        if got != expected:
            raise SystemExit(f"STOP: sha256 {path.name} = {got}, в {sha_file.name} {expected}")
    return log


def holm(p: pd.Series) -> pd.Series:
    """Поправка Холма (step-down), скорректированные p с монотонностью, не больше 1."""
    order = np.argsort(p.to_numpy(), kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p.iloc[i])
        adj[i] = min(1.0, running)
    return pd.Series(adj, index=p.index)


def one(df: pd.DataFrame, what: str) -> pd.Series:
    if len(df) != 1:
        raise SystemExit(f"STOP: для {what} найдено строк {len(df)}, ожидается 1")
    return df.iloc[0]


def collect(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Строки-кандидаты: гипотеза, уровень, p, confirmed_level и описание размера — без пересчёта."""
    kw = a[a["row_type"] == "kw"]
    corr = b[b["row_type"] == "corr"]
    g7 = a[a["row_type"] == "g7_delta"]
    rows = []

    def kw_row(hyp, var, level, family):
        r = one(kw[(kw["variable"] == var) & (kw["level"] == level)], f"{hyp} {var} {level}")
        rows.append({"hypothesis": hyp, "variable": var, "level": level, "family": family, "p_source": "p_dec",
                     "p_raw": r["p_dec"], "confirmed_level": r["confirmed_level"], "stat": "η²_H", "obs": r["obs"],
                     "size_label": r["size_label"], "eta2_fd": r["eta2_fd"]})

    for level in LEVELS:
        fam = f"основное, {level}"
        kw_row("Г1", "log_market_access", level, fam)
        r = one(corr[(corr["hypothesis"] == "Г2") & (corr["variant"] == "основной") & (corr["level"] == level)], f"Г2 {level}")
        rows.append({"hypothesis": "Г2", "variable": r["variable"], "level": level, "family": fam, "p_source": "p_dec",
                     "p_raw": r["p_dec"], "confirmed_level": r["confirmed_level"], "stat": "ρ", "obs": r["obs"],
                     "opposite_sign": r["opposite_sign"]})
        r = one(g7[(g7["variant"] == "основной") & (g7["level"] == level)], f"Г7 {level}")
        h = one(kw[(kw["variable"] == "share_health") & (kw["level"] == level)], f"Г7 share_health {level}")
        rows.append({"hypothesis": "Г7", "variable": r["variable"], "level": level, "family": fam, "p_source": "p_boot",
                     "p_raw": r["p_boot"], "confirmed_level": r["confirmed_level"], "stat": "η²_H (share_health)",
                     "obs": h["obs"], "size_label": h["size_label"], "eta2_fd": h["eta2_fd"]})
        kw_row("Г8", "log_population_2023", level, fam)
        for c in "ABCDEFGHIJKLMNOPQRS":
            kw_row("Г4", f"share_okved_{c}_2023", level, f"Г4 (19 отраслей), {level}")
        r = one(corr[(corr["hypothesis"] == "Г3") & (corr["variant"] == "основной") & (corr["level"] == level)], f"Г3 {level}")
        rows.append({"hypothesis": "Г3", "variable": r["variable"], "level": level, "family": "вне Холма (разведочная)",
                     "p_source": "p_dec", "p_raw": r["p_dec"], "confirmed_level": r["confirmed_level"], "stat": "ρ",
                     "obs": r["obs"], "opposite_sign": r["opposite_sign"], "exploratory": True})
    fam = f"основное, {LEVEL_ALL}"
    r = one(b[(b["row_type"] == "g6") & (b["variant"] == "основной")], "Г6")
    rows.append({"hypothesis": "Г6", "variable": f"доля переходов в ближайшие (N = {int(r['n_nearest'])})",
                 "level": LEVEL_ALL, "family": fam, "p_source": "p_exact", "p_raw": r["p_exact"],
                 "confirmed_level": r["confirmed_level"], "stat": "S", "obs": r["obs"]})
    r = one(b[b["row_type"] == "g9"], "Г9")
    rows.append({"hypothesis": "Г9", "variable": "ρ(P_t, M_t)", "level": LEVEL_ALL, "family": fam, "p_source": "p_perm",
                 "p_raw": r["p_perm"], "confirmed_level": r["confirmed_level"], "stat": "ρ", "obs": r["obs"]})
    df = pd.DataFrame(rows)
    if df[["p_raw", "confirmed_level"]].isna().any().any():
        raise SystemExit("STOP: не хватает p или confirmed_level:\n" + df[df[["p_raw", "confirmed_level"]].isna().any(axis=1)].to_string())
    return df


def main() -> None:
    log = check_frozen()
    a = pd.read_parquet(RESULTS_19A)
    b = pd.read_parquet(RESULTS_19B)
    df = collect(a, b)
    df["p_holm"] = np.nan
    for fam, idx in df.groupby("family").groups.items():
        if fam.startswith("вне Холма"):
            continue
        df.loc[idx, "p_holm"] = holm(df.loc[idx, "p_raw"])
    p_final = df["p_holm"].where(df["p_holm"].notna(), df["p_raw"])
    df["result"] = np.where(df["confirmed_level"].astype(bool) & (p_final < ALPHA), CONFIRMED, NOT_CONFIRMED)
    df["ratio_fd"] = np.where(df["level"] == LEVEL_ALL, df.get("obs") / df.get("eta2_fd"), np.nan)
    df.loc[df["stat"].isin(["ρ", "S"]), "ratio_fd"] = np.nan

    def flag(v) -> bool:
        return v is not None and not pd.isna(v) and bool(v)

    def note(r):
        parts = []
        if flag(r.get("exploratory")):
            parts.append(EXPLORATORY_NOTE)
        if flag(r.get("opposite_sign")):
            parts.append(OPPOSITE_NOTE)
        if r["result"] == CONFIRMED and r.get("size_label") in SMALL_LABELS:
            parts.append(SMALL_NOTE)
        return "; ".join(parts)

    df["note"] = df.apply(note, axis=1)

    stage_rows = []
    for hyp in ["Г1", "Г2", "Г7", "Г8", "Г3"]:
        s = df[df["hypothesis"] == hyp].set_index("level")
        key = (s.loc[LEVEL_ALL, "result"] == CONFIRMED, s.loc[LEVEL_REGION, "result"] == CONFIRMED)
        notes = [f"{lv}: {s.loc[lv, 'note']}" for lv in LEVELS if s.loc[lv, "note"]]
        stage_rows.append({"hypothesis": hyp, "level": "итог по уровням", "result": STAGES[key], "note": " | ".join(notes)})
    g4_rows = []
    for level in LEVELS:
        s = df[(df["hypothesis"] == "Г4") & (df["level"] == level)]
        conf = s[s["result"] == CONFIRMED]["variable"].str.replace("share_okved_", "").str.replace("_2023", "")
        g4_rows.append({"hypothesis": "Г4", "level": level, "result": f"{len(conf)} из {len(s)} отраслей подтверждено",
                        "note": "подтверждены: " + (", ".join(conf) if len(conf) else "—")})
    order = one(a[a["row_type"] == "g1_order"], "Г1 порядок")
    order_row = {"hypothesis": "Г1 (порядок типов)", "level": "по 24 месяцам",
                 "result": "выполнено" if int(order["months_ok"]) >= MONTHS_RULE else "не выполнено",
                 "note": f"месяцев с максимумом медианы у типа 3 и минимумом у типа 4: {int(order['months_ok'])} из 24; "
                         f"декабрь: {order['order']}"}

    df["row_type"] = "level"
    out = pd.concat([df, pd.DataFrame(stage_rows).assign(row_type="stage"), pd.DataFrame(g4_rows).assign(row_type="g4_summary"),
                     pd.DataFrame([order_row]).assign(row_type="g1_order")], ignore_index=True)
    cols = ["row_type", "hypothesis", "variable", "level", "family", "p_source", "p_raw", "p_holm", "confirmed_level",
            "result", "stat", "obs", "size_label", "eta2_fd", "ratio_fd", "opposite_sign", "exploratory", "note"]
    out = out.reindex(columns=cols)
    for c in ["confirmed_level", "opposite_sign", "exploratory"]:
        out[c] = out[c].astype("boolean")
    out.to_parquet(VERDICTS_PATH, engine="pyarrow", index=False)

    def size_text(r):
        if r["row_type"] != "level" or r["result"] != CONFIRMED:
            return ""
        if r["stat"].startswith("η²_H"):
            t = f"η²_H = {r['obs']:.3f}, {r['size_label']}"
            return t + (f"; к эталону ФО: {r['ratio_fd']:.2f}" if pd.notna(r["ratio_fd"]) else "")
        return f"{r['stat']} = {r['obs']:.3f} (метка Коэна для η²_H не применяется)"

    main_rows = out[out["row_type"] == "level"]
    main_rows = main_rows[main_rows["hypothesis"] != "Г4"]
    order_h = {h: i for i, h in enumerate(["Г1", "Г2", "Г3", "Г6", "Г7", "Г8", "Г9"])}
    tab_rows = []
    for hyp in sorted(order_h, key=order_h.get):
        for _, r in main_rows[main_rows["hypothesis"] == hyp].iterrows():
            p_txt = f"p = {r['p_raw']:.4g}" + (f", p_Холм = {r['p_holm']:.4g}" if pd.notna(r["p_holm"]) else " (без Холма)")
            tab_rows.append({"гипотеза": hyp, "уровень": r["level"], "результат": f"{r['result']} ({p_txt})",
                             "размер": size_text(r), "оговорка": r["note"]})
        st = out[(out["row_type"] == "stage") & (out["hypothesis"] == hyp)]
        if len(st):
            tab_rows.append({"гипотеза": hyp, "уровень": "итог по уровням", "результат": st.iloc[0]["result"],
                             "размер": "", "оговорка": st.iloc[0]["note"]})
        if hyp == "Г1":
            tab_rows.append({"гипотеза": order_row["hypothesis"], "уровень": order_row["level"],
                             "результат": order_row["result"], "размер": "", "оговорка": order_row["note"]})
    for _, r in out[out["row_type"] == "g4_summary"].iterrows():
        tab_rows.append({"гипотеза": "Г4", "уровень": r["level"], "результат": r["result"], "размер": "",
                         "оговорка": r["note"]})
    table = pd.DataFrame(tab_rows)
    g4_detail = out[(out["row_type"] == "level") & (out["hypothesis"] == "Г4")][
        ["level", "variable", "p_raw", "p_holm", "confirmed_level", "result", "size_label", "ratio_fd", "note"]]
    lines = [
        "# 19. Вердикты по гипотезам", "",
        "Сгенерировано `src/19c_hypothesis_verdicts.py`: механическое применение правил к результатам шагов 19a и 19b, "
        "без пересчёта.", "", "## Заморозка", "", *log, "",
        "## Правила", "",
        f"- Поправка Холма (ALPHA = {ALPHA}) отдельно по семействам: «основное» на общем уровне — Г1 (log_market_access, "
        "p_dec), Г2 (основной, p_dec), Г7 (p_boot), Г8 (log_population_2023, p_dec), Г6 (p_exact), Г9 (p_perm); "
        "«основное» внутри регионов — Г1, Г2, Г7, Г8; Г4 — 19 отраслей основного варианта, отдельно на каждом уровне; "
        "Г3 — разведочная, вне Холма (сырые p).",
        f"- «подтверждено» = confirmed_level (шаги 19a, 19b) и скорректированный p < {ALPHA}.",
        "- Ступени для Г1, Г2, Г3, Г7, Г8 — по сочетанию итогов на общем уровне и внутри регионов.",
        "- Размер для подтверждённых результатов: метка Коэна для η²_H и отношение η²_H к эталону «федеральные округа» "
        "(только общий уровень); при метке «ничтожный» или «малый» — «устойчиво, но мало по размеру». Для ρ и S метка "
        "Коэна для η²_H не применяется, приводится значение статистики.", "",
        "## Вердикты", "", table.to_markdown(index=False), "",
        "## Г4 по отраслям", "", g4_detail.to_markdown(index=False), "",
        "## Ограничения", "", *[f"- {x}" for x in LIMITATIONS], ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"sha256 {hashlib.sha256(VERDICTS_PATH.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
