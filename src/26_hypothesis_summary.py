"""Шаг 26. Сводка проверок внешней валидности.

ЧИСТО СВОДНЫЙ шаг: числа только читаются из замороженных результатов шагов 18–22 (hypothesis_results_19a/19b,
hypothesis_verdicts, hypothesis_results_supply), из шага 25 (composition_robustness) и из шага 23 (type_portraits);
ничего не пересчитывается, новых тестов, порогов и вердиктов нет. Слова «подтверждено» / «не подтверждено» из колонки
result входов в отчёт не копируются: текст результата строится по правилу «отличимо / не отличимо от случайного»
с подписью размера (условные границы — config step26_summary). Пометка «держится / зависит от записи» в таблице
профилей выводится из записанных в шаге 23 σ по тому же правилу, что в шаге 23 (порог подписи 10b).

Выход: data/processed/hypothesis_summary.parquet (long), notebooks/26_hypothesis_summary.md
Запуск из корня проекта:  .venv/bin/python src/26_hypothesis_summary.py
"""
import ast
import hashlib
import importlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (COMP_MIN_MO_REGION, HYP_MIN_REGION_N, MONTHS_RULE, PROFILE_BORDER_MARGIN, SUMMARY_ETA2_BINS,
                    SUMMARY_ETA2_LABELS, SUMMARY_NEG_EXPECT, SUMMARY_NEG_SOURCES, SUMMARY_RHO_BINS, SUMMARY_RHO_LABELS)

step10b = importlib.import_module("10b_kmeans_cluster_profiles")
step19a = importlib.import_module("19a_hypothesis_tests_kw")      # только константы правила порядка типов Г1

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
R19A = PROCESSED_DIR / "hypothesis_results_19a.parquet"
R19B = PROCESSED_DIR / "hypothesis_results_19b.parquet"
RSUP = PROCESSED_DIR / "hypothesis_results_supply.parquet"
VERD = PROCESSED_DIR / "hypothesis_verdicts.parquet"
COMP = PROCESSED_DIR / "composition_robustness.parquet"
PORT = PROCESSED_DIR / "type_portraits.parquet"
OUT_PATH = PROCESSED_DIR / "hypothesis_summary.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "26_hypothesis_summary.md"

FROZEN = {
    PROCESSED_DIR / "hypothesis_thresholds.parquet": "11081afbfc7e88dbd301004e4c739e3f25e00bf8a47704cf7c74e15e8e6665f7",
    R19A: "58e1516ccbf03b2a1fca82971d5170f581815fe18fd8d22bbf261862fdad9d8e",
    R19B: "d66d64d68a82867a5add988d4441fb0df2d35297b8c94c1514bc9e513c615d9d",
    RSUP: "fbfd86ec33e27ad61378987ccd53e727678017cdb1f091808db8f7545bed97db",
    VERD: "01177c53c539fbabded34f03e85d6a8838726df29f93ecb9d47739f96323659f",
    COMP: "41bfe1d9ba8a0925e8bf3c8c17d6b8232cf5f33db2f749c796c77a32834e37e1",
    PORT: "cb3ccd6e72d6a703e78a9f4e93c4b13029292fd28131e39f79f68ba76d267f7c",
}
CATS = ["Продовольствие", "Здоровье", "Общепит", "Транспорт", "Маркетплейсы"]
L_ALL, L_IN = "общий", "внутри регионов"
FORBIDDEN = ["подтверждено", "доказано", "причина", "вызывает", "объясняется", "из-за", "следует"]

# Контрольные значения пользователя (задание шага 26), допуск 0.002
TOL = 0.002
CONTROLS = [("Г8 η²_H, общий", "g8_all", 0.411), ("Г8 η²_H, внутри регионов", "g8_in", 0.148),
            ("Г1 η²_H, общий", "g1_all", 0.393), ("Г1 η²_H, внутри регионов", "g1_in", 0.045),
            ("Г1 эталон «федеральные округа» η²_H", "g1_fd", 0.746),
            ("Г2 ρ, общий", "g2_all", 0.117), ("Г2 ρ, внутри регионов", "g2_in", -0.299),
            ("Г3 ρ, общий", "g3_all", -0.267), ("Г3 ρ, внутри регионов", "g3_in", -0.463),
            ("Г7 Δ, общий", "g7_all", -0.0259), ("Г7 Δ, внутри регионов", "g7_in", -0.0105),
            ("Г6 S", "g6_s", 0.693), ("Г9 ρ", "g9", 0.116),
            ("Г10 ρ, внутри регионов", "g10_in", -0.321), ("Г11 ρ, внутри регионов", "g11_in", 0.338),
            ("Г12 Δ, общий", "g12_all", 0.0093), ("Г13 Δ, общий", "g13_all", 0.0343)]
G7_EXPECTED_SIGN = 1          # Г7: ожидалось Δ = η²_H(доля) − η²_H(остаток после учёта возраста) > 0 (учёт возраста уменьшает различие);
                              # в hypothesis_thresholds у Г7 expected_sign пуст, в hypothesis_thresholds_g6g9 Г7 нет — поэтому константа
EXPECT_G2_ALR_ALL, EXPECT_G2_BETWEEN_DIFF = 0.015, 0.006   # ожидания пользователя (правки №1 шага 26), допуск TOL
DELTA_HEAD = "Δ = η²_H до учёта − η²_H после учёта (положительное: различие типов уменьшилось; без подписи размера)"
RATIO_EXPECTED = {("Г12", "общий"): 0.0143, ("Г12", "внутри регионов"): 0.0612, ("Г13", "общий"): 0.0554,
                  ("Г13", "внутри регионов"): 0.0906, ("Г7", "общий"): -0.0433, ("Г7", "внутри регионов"): -0.0517}
EXPECT_G3_ALR_IN = -0.481     # ожидание пользователя (правки №2 шага 26), допуск TOL
BORDER_EXPECTED = {(0, "Общепит"): 0.5585, (2, "Здоровье"): 0.5746, (5, "Общепит"): 0.5116}
NOTE_ALL = "общий уровень: см. раздел 0 про контроль «региональный шум»"   # оговорка правок №3 (пункт C)
NEG_ROW_18 = re.compile(r"^\|\s*№2: a_region \+ e\s*\|\s*(общий|внутри регионов)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*$")
NEG_ROW_21 = re.compile(r"^\|\s*региональный шум\s*\|\s*(общий|внутри регионов)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*$")

ROWS: list = []


def rec(block, item, level, metric, value, n=None, q025=None, q975=None, result="", size="", source="") -> float:
    ROWS.append({"block": block, "item": item, "level": level, "metric": metric, "value": float(value),
                 "n": None if n is None else int(n), "q025": np.nan if q025 is None else float(q025),
                 "q975": np.nan if q975 is None else float(q975), "result": result, "size": size, "source": source})
    return float(value)


def check_frozen(when: str, log: list) -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        rel = path.relative_to(PROJECT_DIR)
        print(f"{when}: {got}  {rel} {'OK' if got == expected else 'MISMATCH'}")
        log.append(f"- {when}: `{rel}` — {'совпадает' if got == expected else 'НЕ СОВПАДАЕТ'}")
        if got != expected:
            raise SystemExit(f"STOP: sha256 {rel} = {got}, ожидается {expected}")


def neg_controls() -> dict:
    """Доли контроля «региональный шум» из md-отчётов шагов 18 и 21 (разбор регулярными выражениями, файлы не меняются);
    сверка с ожиданием из config: STOP при расхождении."""
    got: dict = {}
    for step, rx, cols in [("step18", NEG_ROW_18, ("all_p", "all_pos", "in_p", "in_pos")), ("step21", NEG_ROW_21, ("all_p", "in_p"))]:
        path = PROJECT_DIR / SUMMARY_NEG_SOURCES[step]
        hits = []
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = rx.match(line)
            if m:
                hits.append((no, m.groups()))
        if len(hits) != 2 or {h[1][0] for h in hits} != {L_ALL, L_IN}:
            raise SystemExit(f"STOP: {path.name}: ожидалось по одной строке контроля на уровень, найдено {hits}")
        for no, g_ in hits:
            lv = "all" if g_[0] == L_ALL else "in"
            if int(g_[1]) != SUMMARY_NEG_EXPECT["n_repeats"]:
                raise SystemExit(f"STOP: {path.name}, строка {no}: повторов {g_[1]}, ожидается {SUMMARY_NEG_EXPECT['n_repeats']}")
            vals = [float(x) for x in g_[2:]]
            for col, x in zip([c for c in cols if c.startswith(lv)], vals):
                got[(step, col)] = (x, no)
    bad = [(k, got[k][0], SUMMARY_NEG_EXPECT[k[0]][k[1]]) for k in got if abs(got[k][0] - SUMMARY_NEG_EXPECT[k[0]][k[1]]) > 1e-12]
    for k in sorted(got):
        print(f"контроль «региональный шум» {k[0]} {k[1]}: {got[k][0]:g} (строка {got[k][1]}); ожидание {SUMMARY_NEG_EXPECT[k[0]][k[1]]:g}")
    if bad or len(got) != 6:
        raise SystemExit(f"STOP: контроль «региональный шум» не совпал с config (ключ, файл, ожидание): {bad}")
    return got


def sha_neg_sources() -> dict:
    return {s_: hashlib.sha256((PROJECT_DIR / pth).read_bytes()).hexdigest() for s_, pth in SUMMARY_NEG_SOURCES.items()}


def one(df: pd.DataFrame, what: str) -> pd.Series:
    if len(df) != 1:
        raise SystemExit(f"STOP: {what}: ожидалась ровно одна строка, найдено {len(df)}")
    return df.iloc[0]


def size_eta(v: float) -> str:
    return SUMMARY_ETA2_LABELS[int(np.searchsorted(SUMMARY_ETA2_BINS, v, side="right"))]


def size_rho(v: float) -> str:
    return SUMMARY_RHO_LABELS[int(np.searchsorted(SUMMARY_RHO_BINS, abs(v), side="right"))]


def tiny(label: str) -> bool:
    return label in (SUMMARY_ETA2_LABELS[0], SUMMARY_RHO_LABELS[0])


def verdict_text(distinct: bool, size: str, why: str = "после поправки Холма") -> str:
    """Правило текста: «отличимо от случайного (…)» / «не отличимо от случайного»; затем размер."""
    base = f"отличимо от случайного ({why})" if distinct else "не отличимо от случайного"
    out = f"{base}; размер: {size}" if size else base
    if distinct and size and tiny(size):
        out += "; различие обнаружимо, практически ничтожно"
    return out


RATIOS: dict = {}


def ratio(block, item, level, delta, eta_before, n, source) -> float:
    """Δ / η²_H до учёта (из той же строки входа); запись в parquet отдельной строкой."""
    RATIOS[(item, level)] = rec(block, item, level, "Δ / η²_H до учёта", delta / eta_before, n, source=source)
    return RATIOS[(item, level)]


def f(v, fmt=".3f") -> str:
    return format(v, fmt)


def ci(lo, hi, fmt="+.4f") -> str:
    return f"[{format(lo, fmt)}; {format(hi, fmt)}]"


def md(df: pd.DataFrame) -> str:
    """Таблица markdown; «|» внутри ячеек (|ρ|, |σ|) экранируется, чтобы не ломать столбцы."""
    return df.astype(str).apply(lambda col: col.str.replace("|", "\\|", regex=False)).to_markdown(index=False, disable_numparse=True)


def template_literals() -> dict:
    """Числа в строковых константах шаблонов: список lines и словари строк таблиц (*_rows.append({...})), а также
    строки результата verdict_text; без вложенных вызовов выборки и форматирования (one, cval, kw19a, verd, rec, f, ci)."""
    skip = {"one", "cval", "kw19a", "verd", "rec", "f", "ci", "format", "size_eta", "size_rho", "md"}
    found: dict = {}

    def walk(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in skip:
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for t in re.findall(r"(?<![Г\w.])\d+(?:\.\d+)?", node.value):
                found.setdefault(t, set()).add(node.lineno)
        for ch in ast.iter_child_nodes(node):
            walk(ch)

    for node in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "lines" for t in node.targets):
            walk(node.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name) and node.func.value.id.endswith("_rows")):
            for arg in node.args:
                walk(arg)
        if isinstance(node, ast.FunctionDef) and node.name == "verdict_text":
            walk(node)
    return found


def main() -> None:
    frozen_log: list = []
    check_frozen("до", frozen_log)
    neg_sha_before = sha_neg_sources()
    neg = neg_controls()
    NEG_META = {("step18", "all_p"): ("доля p_dec < ALPHA, общий (шаг 18)", L_ALL), ("step18", "all_pos"): ("доля положительных решений правила шага 18, общий", L_ALL),
                ("step18", "in_p"): ("доля p_dec < ALPHA, внутри регионов (шаг 18)", L_IN), ("step18", "in_pos"): ("доля положительных решений правила шага 18, внутри регионов", L_IN),
                ("step21", "all_p"): ("доля p < ALPHA, общий (шаг 21)", L_ALL), ("step21", "in_p"): ("доля p < ALPHA, внутри регионов (шаг 21)", L_IN)}
    for k_, (x_, no_) in neg.items():
        rec("0. контроль", "региональный шум", NEG_META[k_][1], NEG_META[k_][0], x_, SUMMARY_NEG_EXPECT["n_repeats"],
            source=f"{SUMMARY_NEG_SOURCES[k_[0]]}, строка {no_}")
    a, b = pd.read_parquet(R19A), pd.read_parquet(R19B)
    s, v = pd.read_parquet(RSUP), pd.read_parquet(VERD)
    comp, port = pd.read_parquet(COMP), pd.read_parquet(PORT)
    my: dict = {}
    main_rows, delta_rows, g4_rows = [], [], []

    def kw19a(var, level):
        return one(a[(a.row_type == "kw") & (a.variable == var) & (a.variant == "основной") & (a.level == level)], f"19a kw {var} {level}")

    def verd(hyp, level):
        return one(v[(v.row_type == "level") & (v.hypothesis == hyp) & (v.level == level)], f"verdicts {hyp} {level}")

    # ── Г1 market_access, Г8 размер МО (η²_H, Холм) ──
    for hyp, var, what, key in [("Г1", "log_market_access", "типы различаются по log market_access", "g1"),
                                ("Г8", "log_population_2023", "типы различаются по log населения 2023", "g8")]:
        for level, k in [(L_ALL, "all"), (L_IN, "in")]:
            r, vv = kw19a(var, level), verd(hyp, level)
            if not np.isclose(r["obs"], vv["obs"]):
                raise SystemExit(f"STOP: {hyp} {level}: η²_H в 19a и verdicts различаются")
            sz = size_eta(r["obs"])
            res = verdict_text(bool(vv["confirmed_level"]), sz)
            my[f"{key}_{k}"] = rec("1. проверки", hyp, level, f"η²_H {var}", r["obs"], r["n"], result=res, size=sz, source="19a; verdicts")
            note = ""
            if level == L_ALL:
                my[f"{key}_fd"] = rec("1. проверки", hyp, level, "эталон «федеральные округа» η²_H", r["eta2_fd"], r["n_fd"], source="19a")
                note = f"эталон «федеральные округа»: η²_H {f(r['eta2_fd'])}; {NOTE_ALL}"
            main_rows.append({"проверка": hyp, "что проверяется": what, "уровень": level,
                              "показатель": f"η²_H {f(r['obs'])} (n = {int(r['n'])})", "результат": res, "размер": sz, "оговорка": note})
        if hyp == "Г1":
            o = one(a[a.row_type == "g1_order"], "19a g1_order")
            n_months = int((a.row_type == "g1_order_month").sum())
            rec("1. проверки", "Г1 порядок типов", "по месяцам", "месяцев с выполненным порядком", o["months_ok"], n_months, source="19a")
            ok = int(o["months_ok"]) >= MONTHS_RULE
            main_rows.append({"проверка": "Г1", "что проверяется": f"в каждом из {n_months} месячных разбиений: наибольшая медиана "
                              f"log market_access у типа {step19a.G1_TOP_TYPE}, наименьшая у типа {step19a.G1_BOTTOM_TYPE}",
                              "уровень": "по месяцам",
                              "показатель": f"порядок в декабре (для справки, не часть правила): {o['order']}",
                              "результат": f"выполнено в {int(o['months_ok'])} месяцах из {n_months}, правило ≥ {MONTHS_RULE}: "
                                           f"{'выполнено' if ok else 'не выполнено'}", "размер": "—", "оговорка": ""})
            for level in (L_ALL, L_IN):
                r = kw19a("has_railway", level)
                sz = size_eta(r["obs"])
                res = verdict_text(bool(r["confirmed_level"]), sz, "без поправки Холма: вне семейства шага 19c")
                rec("1. проверки", "Г1 железная дорога", level, "η²_H has_railway", r["obs"], r["n"], result=res, size=sz, source="19a")
                note = "в проверку с поправкой Холма не входила; о том, почему, в файлах проекта сведений нет"
                if level == L_ALL:
                    rec("1. проверки", "Г1 железная дорога", level, "эталон «федеральные округа» η²_H", r["eta2_fd"], r["n_fd"], source="19a")
                    note = f"эталон «федеральные округа»: η²_H {f(r['eta2_fd'], '.4f')}; " + note + f"; {NOTE_ALL}"
                main_rows.append({"проверка": "Г1", "что проверяется": "типы различаются по наличию железной дороги (has_railway)",
                                  "уровень": level, "показатель": f"η²_H {f(r['obs'], '.4f')} (n = {int(r['n'])}); правило ≥ {MONTHS_RULE} "
                                  f"месяцев: {'выполнено' if int(r['months_ok']) >= MONTHS_RULE else 'не выполнено'} ({int(r['months_ok'])})",
                                  "результат": res, "размер": sz, "оговорка": note})

    # ── Г2, Г3 (ρ) ──
    for hyp, var, what, why in [("Г2", "share_Маркетплейсы~log_market_access", "доля маркетплейсов ~ log market_access", "после поправки Холма"),
                                ("Г3", "share_Продовольствие~wage_rel_region_2023", "доля продовольствия ~ зарплата относительно региона (разведочная)",
                                 "вне поправки Холма: разведочная")]:
        for level, k in [(L_ALL, "all"), (L_IN, "in")]:
            r = one(b[(b.row_type == "corr") & (b.variable == var) & (b.variant == "основной") & (b.level == level)], f"19b {hyp} {level}")
            vv = verd(hyp, level)
            sz = size_rho(r["obs"])
            res = verdict_text(bool(vv["confirmed_level"]), sz, why)
            exp_sign = 1 if r["expected_sign"] == "+" else -1
            reversed_sig = (not bool(vv["confirmed_level"])) and bool(r["opposite_sign"]) and np.sign(r["obs"]) == -exp_sign
            if reversed_sig:   # знак обратный ожидаемому и |ρ| больше двустороннего порога шага 19c
                res = ("ожидаемое направление не выполнено (знак обратный); по двустороннему порогу шага 19c отличимо от нуля; "
                       f"размер: {sz}")
            my[f"{hyp.lower().replace('г', 'g')}_{k}"] = rec("1. проверки", hyp, level, f"ρ {var}", r["obs"], r["n"], result=res, size=sz, source="19b; verdicts")
            notes = []
            if hyp == "Г2":
                notes.append("знак зависит от уровня")
                if level == L_ALL:
                    x = one(comp[(comp.block == "5. связи") & comp.metric.str.startswith("Г2") & (comp.variant == "лог-отношение")
                                 & (comp.level == "все МО")], "шаг 25 Г2 все МО лог-отношение")["value"]
                    print(f"Г2 общий, ρ по лог-отношению: {x:+.4f}; ожидание {EXPECT_G2_ALR_ALL:+.3f}")
                    if abs(x - EXPECT_G2_ALR_ALL) > TOL + 1e-12:
                        raise SystemExit(f"STOP: Г2 общий по лог-отношению {x:+.4f}, ожидается {EXPECT_G2_ALR_ALL:+.3f}")
                    rec("1. проверки", "Г2", level, "ρ по log(доля / «прочее») (шаг 25)", x, None, source="composition_robustness")
                    notes.append(f"при записи в лог-отношении ρ {f(x, '+.3f')} ({size_rho(x)}; шаг 25)")
                    notes.append(NOTE_ALL)
                if bool(vv["opposite_sign"]):
                    notes.append(f"знак противоположен ожидаемому ({vv['note'].split('(')[1].rstrip(')')}, шаг 19c)")
                if hyp == "Г2" and level == L_IN:
                    c25 = one(comp[(comp.block == "5. связи") & comp.metric.str.startswith("Г2") & (comp.variant == "доли")
                                   & (comp.level == "внутри регионов")], "шаг 25 Г2 внутри регионов доли")
                    notes.append(f"значения шагов 19b и 25 посчитаны по разным выборкам: шаг 19b — {int(r['n'])} МО, регионы с не менее "
                                 f"{HYP_MIN_REGION_N} МО; шаг 25 — {int(c25['n'])} МО, регионы с не менее {COMP_MIN_MO_REGION} МО")
            else:
                notes.append("зарплата по месту нахождения организации, а не жителей")
            main_rows.append({"проверка": hyp, "что проверяется": what, "уровень": level,
                              "показатель": f"ρ {f(r['obs'], '+.3f')} (n = {int(r['n'])})", "результат": res, "размер": sz, "оговорка": "; ".join(notes)})
        if hyp == "Г2":
            btw = {vc: one(comp[(comp.block == "5. связи") & comp.metric.str.startswith("Г2") & (comp.variant == vc) & (comp.level == "между регионами")],
                           f"шаг 25 Г2 между регионами {vc}") for vc in ("доли", "лог-отношение")}
            d_btw = abs(btw["лог-отношение"]["value"] - btw["доли"]["value"])
            print(f"Г2 между регионами: разность ρ по долям и по лог-отношению {d_btw:.4f}; ожидание {EXPECT_G2_BETWEEN_DIFF:.3f}")
            if abs(d_btw - EXPECT_G2_BETWEEN_DIFF) > TOL + 1e-12:
                raise SystemExit(f"STOP: Г2 между регионами: разность {d_btw:.4f}, ожидается {EXPECT_G2_BETWEEN_DIFF:.3f}")
            rec("1. проверки", "Г2", "между регионами", "разность ρ по долям и по log(доля / «прочее»)", d_btw, None, source="composition_robustness")
            labs = {vc: size_rho(r_["value"]) for vc, r_ in btw.items()}
            cut = [t for t in SUMMARY_RHO_BINS if min(abs(r_["value"]) for r_ in btw.values()) < t <= max(abs(r_["value"]) for r_ in btw.values())]
            btw_note = f"знак зависит от уровня; значения по долям и по лог-отношению различаются на {f(d_btw)}"
            if labs["доли"] != labs["лог-отношение"]:
                if len(cut) != 1:
                    raise SystemExit(f"STOP: подписи размера различаются, но порог между значениями не единственный: {cut}")
                btw_note += f"; подпись размера разная: значения лежат по разные стороны порога {cut[0]}"
            for var_c, lab in [("доли", "ρ по долям"), ("лог-отношение", "ρ по log(доля / «прочее»)")]:
                r = btw[var_c]
                sz = size_rho(r["value"])
                rec("1. проверки", "Г2", "между регионами", lab, r["value"], r["n"], result="не проверялось (шаг 25, разведочно)", size=sz, source="composition_robustness")
                main_rows.append({"проверка": "Г2", "что проверяется": what, "уровень": "между регионами",
                                  "показатель": f"{lab} {f(r['value'], '+.3f')} (медианы {int(r['n_regions'])} регионов с не менее "
                                                f"{COMP_MIN_MO_REGION} МО; МО {int(r['n'])})",
                                  "результат": "не проверялось (шаг 25, разведочно)", "размер": sz, "оговорка": btw_note})

    # ── Г4 ──
    g4v = v[(v.row_type == "level") & (v.hypothesis == "Г4")]
    letters = sorted({re.search(r"okved_(\w)_", x).group(1) for x in g4v.variable})
    for L in letters:
        row = {"отрасль": L}
        for level in (L_ALL, L_IN):
            r = kw19a(f"share_okved_{L}_2023", level)
            vv = one(g4v[(g4v.variable == f"share_okved_{L}_2023") & (g4v.level == level)], f"Г4 {L} {level}")
            sz = size_eta(r["obs"])
            res = verdict_text(bool(vv["confirmed_level"]), sz)
            rec("1. Г4", f"Г4 {L}", level, "η²_H доли занятых", r["obs"], r["n"], result=res, size=sz, source="19a; verdicts")
            row[f"{level}: η²_H (n)"] = f"{f(r['obs'])} ({int(r['n'])})"
            row[f"{level}: размер"] = sz
            row[f"{level}: результат"] = res
        g4_rows.append(row)
    for level in (L_ALL, L_IN):
        sub = [r for r in ROWS if r["block"] == "1. Г4" and r["level"] == level]
        dist = [r for r in sub if r["result"].startswith("отличимо")]
        mid = [r for r in dist if r["size"] in SUMMARY_ETA2_LABELS[2:]]
        rec("1. проверки", "Г4", level, "отраслей отличимо от случайного", len(dist), len(sub), source="verdicts")
        rec("1. проверки", "Г4", level, "из них со средним и большим размером", len(mid), len(dist), source="19a; verdicts")
        main_rows.append({"проверка": "Г4", "что проверяется": "типы различаются по отраслевой структуре занятых", "уровень": level,
                          "показатель": f"η²_H по {len(sub)} отраслям (таблица Г4 ниже)",
                          "результат": f"отличимо от случайного (после поправки Холма) по {len(dist)} отраслям из {len(sub)}",
                          "размер": f"из них средний или большой: {len(mid)}", "оговорка": NOTE_ALL if level == L_ALL else ""})

    # ── Г6 ──
    g6 = one(b[(b.row_type == "g6") & (b.variant == "основной")], "19b Г6")
    my["g6_s"] = rec("1. проверки", "Г6", L_ALL, "S: доля переходов в ближайшие типы", g6["obs"], g6["n"], g6["q025"], g6["q975"],
                     result="описательное свойство, не проверка", source="19b")
    rec("1. проверки", "Г6", L_ALL, "базовый уровень нуля (null_mean)", g6["null_mean"], g6["n"], source="19b")
    main_rows.append({"проверка": "Г6", "что проверяется": "переходы МО между месяцами идут в ближайшие типы", "уровень": L_ALL,
                      "показатель": f"S {f(g6['obs'])} {ci(g6['q025'], g6['q975'], '+.3f')}, null_mean {f(g6['null_mean'])}; переходов {int(g6['n'])}",
                      "результат": "описательное свойство, не проверка", "размер": "—", "оговорка": "нуль нереалистичен"})

    # ── Г7 (Δ) ──
    for level, k in [(L_ALL, "all"), (L_IN, "in")]:
        r = one(a[(a.row_type == "g7_delta") & (a.variant == "основной") & (a.level == level)], f"19a Г7 {level}")
        vv = verd("Г7", level)
        sz = size_eta(r["eta2_health"])
        excl_zero = r["q025"] > 0 or r["q975"] < 0
        if (not bool(vv["confirmed_level"])) and excl_zero and np.sign(r["delta"]) == -G7_EXPECTED_SIGN:
            res7 = ("ожидаемое направление не выполнено (знак обратный); интервал изменения η²_H не включает ноль; учёт возраста "
                    "различие типов по доле здравоохранения не уменьшает")
        else:
            res7 = verdict_text(bool(vv["confirmed_level"]), "") + "; учёт возраста различие типов по доле здравоохранения не уменьшает"
        my[f"g7_{k}"] = rec("1. проверки", "Г7", level, "Δ η²_H (доля здравоохранения − остаток после учёта возраста)", r["delta"], r["n"],
                            r["q025"], r["q975"], result=res7, source="19a; verdicts")
        rec("1. проверки", "Г7", level, "η²_H доли здравоохранения до учёта", r["eta2_health"], r["n"], size=sz, source="19a")
        rt = ratio("1. проверки", "Г7", level, r["delta"], r["eta2_health"], r["n"], "19a; verdicts")
        delta_rows.append({"проверка": "Г7", "уровень": level, "η²_H доли до учёта (n)": f"{f(r['eta2_health'])} ({int(r['n'])})",
                           "размер η²_H доли": sz,
                           DELTA_HEAD: f"{f(r['delta'], '+.4f')} {ci(r['q025'], r['q975'])}",
                           "Δ в долях η²_H до учёта": f"{rt * 100:+.1f}%",
                           "результат": res7,
                           "оговорка": "доля здравоохранения входит в признаки кластеризации"})

    # ── Г9 ──
    g9 = one(b[b.row_type == "g9"], "19b Г9")
    vv = verd("Г9", L_ALL)
    sz = size_rho(g9["obs"])
    my["g9"] = rec("1. проверки", "Г9", L_ALL, "ρ(P_t, M_t)", g9["obs"], g9["n"], result=verdict_text(bool(vv["confirmed_level"]), sz), size=sz, source="19b; verdicts")
    rec("1. проверки", "Г9", L_ALL, "порог p95", g9["p95"], g9["n"], source="19b")
    main_rows.append({"проверка": "Г9", "что проверяется": "доля сменивших тип связана с общим сдвигом долей", "уровень": L_ALL,
                      "показатель": f"ρ {f(g9['obs'], '+.3f')} (n = {int(g9['n'])}), порог p95 {f(g9['p95'])}",
                      "результат": verdict_text(bool(vv["confirmed_level"]), sz), "размер": sz,
                      "оговорка": f"малая мощность, {int(g9['n'])} месяца"})

    # ── Г10–Г13 ──
    sup_rows, sup_delta = [], []
    sens = {}
    for hyp, var, lab in [("Г10", "share_Маркетплейсы~stores_pc", "при замене торговой площади числом магазинов на 1000 жителей"),
                          ("Г11", "share_Общепит~seats_all_pc", "при замене мест общепита всеми местами общепита")]:
        vals = {}
        for level in (L_ALL, L_IN):
            r = one(s[(s.row_type == "corr") & (s.variable == var) & (s.variant == "чувствительность") & (s.level == level)], f"supply {var} {level}")
            exp_sign = 1 if r["expected_sign"] == "+" else -1
            szs = size_rho(r["obs"])
            txt = "знак обратный ожидаемому" if np.sign(r["obs"]) == -exp_sign else "знак совпадает с ожидаемым"
            if tiny(szs):
                txt += ", связь едва различима"
            vals[level] = (r["obs"], txt, szs)
            rec("1. дополнительное семейство", hyp, level, f"ρ {var} (чувствительность)", r["obs"], r["n"], size=szs, source="supply")
        same = vals[L_ALL][1] == vals[L_IN][1]
        tail = (f": {vals[L_ALL][1]}" if same else f"; общий: {vals[L_ALL][1]}; внутри регионов: {vals[L_IN][1]}")
        sens[hyp] = (f"{lab} ρ {f(vals[L_ALL][0], '+.3f')} (общий, {vals[L_ALL][2]}), {f(vals[L_IN][0], '+.3f')} "
                     f"(внутри регионов, {vals[L_IN][2]}){tail}", vals)
    for hyp, what, why in [("Г10", "доля маркетплейсов ~ торговая площадь на 1000 жителей", "после поправки Холма"),
                           ("Г11", "доля общепита ~ места в общепите на 1000 жителей (контроль)", "вне поправки Холма: контроль")]:
        for level, k in [(L_ALL, "all"), (L_IN, "in")]:
            r = one(s[(s.row_type == "corr") & (s.hypothesis == hyp) & (s.variant == "основной") & (s.level == level)], f"supply {hyp} {level}")
            vv = one(s[(s.row_type == "verdict") & (s.hypothesis == hyp) & (s.level == level)], f"supply verdict {hyp} {level}")
            p = one(s[(s.row_type == "partial") & (s.hypothesis == hyp) & (s.level == level)], f"supply partial {hyp} {level}")
            sz, szp = size_rho(r["obs"]), size_rho(p["obs"])
            res = verdict_text(bool(vv["confirmed_level"]), sz, why)
            my[f"{hyp.lower().replace('г', 'g')}_{k}"] = rec("1. дополнительное семейство", hyp, level, f"ρ {r['variable']}", r["obs"], r["n"], result=res, size=sz, source="supply")
            rec("1. дополнительное семейство", hyp, level, "частная ρ с поправкой на log населения 2023 (разведочно)", p["obs"], p["n"], result="разведочно", size=szp, source="supply")
            sup_rows.append({"проверка": hyp, "что проверяется": what, "уровень": level, "показатель": f"ρ {f(r['obs'], '+.3f')} (n = {int(r['n'])})",
                             "результат": res, "размер": sz,
                             "частная ρ с поправкой на размер МО (разведочно)": f"{f(p['obs'], '+.3f')} ({szp})",
                             "оговорка": sens[hyp][0] + (f"; {NOTE_ALL}" if level == L_ALL and hyp == "Г10" else "")})
    for hyp, what in [("Г12", "маркетплейсы: учёт торговой площади"), ("Г13", "общепит: учёт мест в общепите")]:
        for level, k in [(L_ALL, "all"), (L_IN, "in")]:
            r = one(s[(s.row_type == "delta") & (s.hypothesis == hyp) & (s.level == level)], f"supply delta {hyp} {level}")
            vv = one(s[(s.row_type == "verdict") & (s.hypothesis == hyp) & (s.level == level)], f"supply verdict {hyp} {level}")
            sz = size_eta(r["eta2_share"])
            res = verdict_text(bool(vv["confirmed_level"]), "")
            my[f"{hyp.lower().replace('г', 'g')}_{k}"] = rec("1. дополнительное семейство", hyp, level, "Δ η²_H (доля − остаток после учёта)", r["delta"], r["n"],
                                                            r["q025"], r["q975"], result=res, source="supply")
            rec("1. дополнительное семейство", hyp, level, "η²_H доли до учёта", r["eta2_share"], r["n"], size=sz, source="supply")
            rt = ratio("1. дополнительное семейство", hyp, level, r["delta"], r["eta2_share"], r["n"], "supply")
            sup_delta.append({"проверка": hyp, "что проверяется": what, "уровень": level, "η²_H доли до учёта (n)": f"{f(r['eta2_share'])} ({int(r['n'])})",
                              "размер η²_H доли": sz, DELTA_HEAD: f"{f(r['delta'], '+.4f')} {ci(r['q025'], r['q975'])}",
                              "Δ в долях η²_H до учёта": f"{rt * 100:+.1f}%",
                              "результат": res, "оговорка": NOTE_ALL if level == L_ALL else ""})

    g11_in = one(s[(s.row_type == "verdict") & (s.hypothesis == "Г11") & (s.level == L_IN)], "supply verdict Г11 внутри регионов")
    g11_dist = bool(g11_in["confirmed_level"])
    g11_rule = ("Правило шага 22: пометка «показатель предложения ненадёжен» ставится у Г10, Г12, Г13, если Г11 внутри регионов "
                "не отличима от случайного; у Г11 внутри регионов результат "
                + ("«отличимо от случайного», пометка не ставится." if g11_dist else "«не отличимо от случайного», пометка ставится."))

    # ── Δ / η²_H до учёта: сверка с ожиданием ──
    bad_r = [(k_, round(RATIOS[k_], 4), v_) for k_, v_ in RATIO_EXPECTED.items() if abs(RATIOS[k_] - v_) > TOL + 1e-12]
    print("Δ / η²_H до учёта:", {k_: round(v_, 4) for k_, v_ in RATIOS.items()})
    if bad_r or set(RATIOS) != set(RATIO_EXPECTED):
        raise SystemExit(f"STOP: Δ / η²_H до учёта не совпадает с ожиданием (ключ, шаг 26, ожидание): {bad_r}")

    # ── сверка ──
    ctrl_rows, bad = [], []
    for name, key, val in CONTROLS:
        ok = abs(my[key] - val) <= TOL + 1e-12
        ctrl_rows.append({"контроль": name, "шаг 26": f"{my[key]:.4f}", "контрольное": f"{val:g}", "допуск": f"{TOL:g}", "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})
        bad += [] if ok else [ctrl_rows[-1]]
    ctrl = pd.DataFrame(ctrl_rows)
    print(ctrl.to_string(index=False))
    if bad:
        raise SystemExit("STOP: расхождения с контрольными значениями:\n" + pd.DataFrame(bad).to_string(index=False))

    # ── шаг 25 ──
    def cval(block, metric, variant=None, level=None):
        q = comp[(comp.block == block) & (comp.metric == metric)]
        if variant is not None:
            q = q[q.variant == variant]
        if level is not None:
            q = q[q.level == level]
        return one(q, f"шаг 25 {block} {metric} {variant} {level}")
    comp_rows = []
    for metric, lab in [("Г2: ρ(Маркетплейсы, ln market_access)", "Г2: ρ(маркетплейсы, ln market_access)"),
                        ("Г3: ρ(Продовольствие, зарплата относительно региона, 2023)", "Г3: ρ(продовольствие, зарплата относительно региона)")]:
        for level in ("все МО", "внутри регионов", "между регионами"):
            x, y = cval("5. связи", metric, "доли", level), cval("5. связи", metric, "лог-отношение", level)
            comp_rows.append({"показатель": f"{lab}, {level}", "по долям": f(x["value"], "+.3f"), "по лог-отношению": f(y["value"], "+.3f")})
    for c in CATS:
        for level in (L_ALL, L_IN):
            x, y = cval("6. η²_H по типам", f"η²_H {c}", "доли", level), cval("6. η²_H по типам", f"η²_H {c}", "лог-отношение", level)
            comp_rows.append({"показатель": f"η²_H {c} по типам, {level}", "по долям": f(x["value"]), "по лог-отношению": f(y["value"])})
    ari_rows = [{"кодировка": r["variant"], "ARI с каноном": f(r["value"])} for _, r in comp[(comp.block == "2. кодировки") & (comp.metric == "ARI с каноном")].iterrows()]
    pc1 = cval("3. главная ось", "PC1: доля дисперсии")
    pc1_rest = cval("3. главная ось", "ρ(PC1, rest)")
    g3_alr_in = cval("5. связи", "Г3: ρ(Продовольствие, зарплата относительно региона, 2023)", "лог-отношение", "внутри регионов")["value"]
    print(f"Г3 внутри регионов по лог-отношению: {g3_alr_in:+.4f}; ожидание {EXPECT_G3_ALR_IN:+.3f}")
    if abs(g3_alr_in - EXPECT_G3_ALR_IN) > TOL + 1e-12:
        raise SystemExit(f"STOP: Г3 внутри регионов по лог-отношению {g3_alr_in:+.4f}, ожидается {EXPECT_G3_ALR_IN:+.3f}")
    rec("3. что устойчиво", "Г3", "внутри регионов", "ρ по log(доля / «прочее») (шаг 25)", g3_alr_in, None, source="composition_robustness")
    eta_all = {c: (cval("6. η²_H по типам", f"η²_H {c}", "доли", L_ALL)["value"], cval("6. η²_H по типам", f"η²_H {c}", "лог-отношение", L_ALL)["value"])
               for c in CATS}
    rest_rows = [{"тип": str(t), "медиана «прочего», %": f(cval("4. rest по типам", f"тип {t}: медиана rest, %")["value"], ".1f"),
                  "квартили, %": f"{f(cval('4. rest по типам', f'тип {t}: нижний квартиль rest, %')['value'], '.1f')} – "
                                f"{f(cval('4. rest по типам', f'тип {t}: верхний квартиль rest, %')['value'], '.1f')}",
                  "n": str(int(cval("4. rest по типам", f"тип {t}: медиана rest, %")["n"]))} for t in range(6)]
    eta_rest = cval("4. rest по типам", "η²_H rest по типам", "доли", L_ALL)["value"]

    # ── профили типов (шаг 23) ──
    pp = port[port.block == "признаки профиля"]
    border = pp[pp.metric.str.contains(", меньшее |σ|", regex=False)]
    got_b = {(int(r["type"]), r["metric"].split(":")[0]): r["value"] for _, r in border.iterrows()}
    print("строки границы порога:", {k: round(x, 4) for k, x in got_b.items()})
    if set(got_b) != set(BORDER_EXPECTED) or any(abs(got_b[k] - BORDER_EXPECTED[k]) > TOL + 1e-12 for k in BORDER_EXPECTED):
        raise SystemExit(f"STOP: строки «на границе порога» не совпадают с ожиданием: {got_b} против {BORDER_EXPECTED}")
    th = step10b.PROFILE_Z_THRESHOLD
    prof_rows = []
    for t in range(6):
        for c in CATS:
            zs = one(pp[(pp.type == str(t)) & (pp.metric == f"{c}: σ по долям")], f"σ по долям {t} {c}")["value"]
            za = one(pp[(pp.type == str(t)) & (pp.metric == f"{c}: σ по лог-отношению")], f"σ по лог-отношению {t} {c}")["value"]
            mark = "" if abs(zs) < th else ("держится" if np.sign(za) == np.sign(zs) and abs(za) >= th else "зависит от записи")
            prof_rows.append({"тип": str(t), "категория": c, "σ по долям": f(zs, "+.2f"), "σ по log(доля / «прочее»)": f(za, "+.2f"),
                              "пометка": mark or "—", "граница порога": f"меньшее |σ| {f(got_b[(t, c)], '.2f')}" if (t, c) in got_b else "—"})
    cover = one(port[(port.block == "ограничения") & (port.metric == "средняя сумма пяти долей за 24 месяца, %")], "шаг 23: покрытие")["value"]

    # ── текст ──
    g = {r["item"] + "|" + r["level"] + "|" + r["metric"]: r for r in ROWS}
    lines = [
        "# 26. Сводка проверок внешней валидности", "",
        "Сгенерировано `src/26_hypothesis_summary.py`. Сводный шаг: числа взяты из замороженных результатов шагов 18–22, "
        "шага 25 и шага 23; ничего не пересчитывается, новых тестов, порогов и вердиктов нет.", "",
        "## Заморозка", "", *frozen_log, "",
        "## 0. Как читать", "",
        "**Порядок работы.** Типология построена по структуре безналичных расходов (шаги 10–12). Проверки внешней валидности "
        "сформулированы после получения типологии и просмотра её профилей. До расчёта проверочных статистик на внешних данных "
        "зафиксированы правила решения, пороги и контрольные эксперименты (хэши sha256 файлов в репозитории); результаты получены "
        "по замороженным правилам. Проверки Г10–Г13 сформулированы после основных результатов и образуют отдельное семейство.", "",
        "- Уровни: «общий» — по всем МО; «внутри регионов» — значения заменены рангом внутри региона, сравниваются МО одного "
        "региона; «между регионами» — по медианам регионов (шаг 25).",
        f"- Подписи размера — условные ориентиры. η²_H: < {SUMMARY_ETA2_BINS[0]} «{SUMMARY_ETA2_LABELS[0]}», "
        f"{SUMMARY_ETA2_BINS[0]}–{SUMMARY_ETA2_BINS[1]} «{SUMMARY_ETA2_LABELS[1]}», {SUMMARY_ETA2_BINS[1]}–{SUMMARY_ETA2_BINS[2]} "
        f"«{SUMMARY_ETA2_LABELS[2]}», ≥ {SUMMARY_ETA2_BINS[2]} «{SUMMARY_ETA2_LABELS[3]}». |ρ|: < {SUMMARY_RHO_BINS[0]} "
        f"«{SUMMARY_RHO_LABELS[0]}», {SUMMARY_RHO_BINS[0]}–{SUMMARY_RHO_BINS[1]} «{SUMMARY_RHO_LABELS[1]}», "
        f"{SUMMARY_RHO_BINS[1]}–{SUMMARY_RHO_BINS[2]} «{SUMMARY_RHO_LABELS[2]}», ≥ {SUMMARY_RHO_BINS[2]} «{SUMMARY_RHO_LABELS[3]}».",
        "- Подписи размера заданы для η²_H и |ρ|; для разности η²_H шкала не задана.",
        f"- {DELTA_HEAD}.",
        "- Значения рядом с границей подписи размера читаются как соседние по размеру; подписи — условные ориентиры.",
        "- Правило текста: результат записывается как «отличимо от случайного (после поправки Холма)» или «не отличимо от "
        "случайного», затем указывается размер; при размере «ничтожный» или «едва различима» добавляется «различие обнаружимо, "
        "практически ничтожно». Для проверок вне поправки Холма в скобках указано, почему.",
        f"- Контроль «региональный шум» (шаги 18 и 21, {SUMMARY_NEG_EXPECT['n_repeats']} повторов): на общем уровне доля p < ALPHA равна "
        f"{neg[('step18', 'all_p')][0]:g} в шаге 18 и {neg[('step21', 'all_p')][0]:g} в шаге 21, положительное решение правила шага 18 даёт "
        f"{neg[('step18', 'all_pos')][0]:g}; внутри регионов доля p < ALPHA не более "
        f"{max(neg[('step18', 'in_p')][0], neg[('step21', 'in_p')][0]):g}, положительное решение правила шага 18 даёт "
        f"{neg[('step18', 'in_pos')][0]:g}: результат «отличимо от случайного» на общем уровне не показывает связи сам по себе; опора — "
        "размер эффекта и уровень «внутри регионов».", "",
        "## 1. Проверки", "",
        md(pd.DataFrame(main_rows)), "",
        "### Учёт возраста (Г7)", "", md(pd.DataFrame(delta_rows)), "",
        "### Г4 по отраслям", "", md(pd.DataFrame(g4_rows)), "",
        "### Дополнительное семейство (Г10–Г13)", "",
        "Сформулировано после основных результатов; поправка Холма — внутри семейства {Г10, Г12, Г13}; Г11 — контроль, вне поправки.", "",
        md(pd.DataFrame(sup_rows)), "", md(pd.DataFrame(sup_delta)), "", g11_rule, "",
        "## 2. Устойчивость к кодированию долей (шаг 25)", "",
        md(pd.DataFrame(comp_rows)), "", md(pd.DataFrame(ari_rows)), "",
        f"PC1 z-score пяти долей несёт {f(pc1['value'])} дисперсии; ρ(PC1, «прочее») = {f(pc1_rest['value'], '+.3f')}; "
        f"η²_H «прочего» по типам {f(eta_rest)} (общий уровень).", "", md(pd.DataFrame(rest_rows)), "",
        "Разбиение на лог-отношениях совпадает с каноническим слабо, поэтому типы рассматриваются как один из правдоподобных "
        "способов сегментации.", "",
        "## 3. Что устойчиво и что нет", "",
        "**Что устойчиво**", "",
        f"- Размер МО (Г8): общий уровень — η²_H {f(my['g8_all'])}, {g['Г8|общий|η²_H log_population_2023']['result']}; "
        f"внутри регионов — η²_H {f(my['g8_in'])}, {g['Г8|внутри регионов|η²_H log_population_2023']['result']}.",
        f"- Ось «еда ↔ общепит» (общий уровень): η²_H продовольствия {f(eta_all['Продовольствие'][0])} → {f(eta_all['Продовольствие'][1])}, общепита "
        f"{f(eta_all['Общепит'][0])} → {f(eta_all['Общепит'][1])} при переходе к лог-отношению (шаг 25).",
        f"- Внутри регионов: связь с зарплатой (Г3, разведочная; ρ {f(my['g3_in'], '+.3f')}; по лог-отношению ρ {f(g3_alr_in, '+.3f')}).", "",
        "**Что нет или зависит от записи**", "",
        f"- Г6 как проверка: S {f(my['g6_s'])} при null_mean {f(g['Г6|общий|базовый уровень нуля (null_mean)']['value'])}, но нуль "
        "нереалистичен; это описательное свойство.",
        f"- Г7: Δ {f(my['g7_all'], '+.4f')} (общий) и {f(my['g7_in'], '+.4f')} (внутри регионов) — учёт возраста различие типов по доле "
        "здравоохранения не уменьшает.",
        f"- Г9: ρ {f(my['g9'], '+.3f')} при пороге {f(g['Г9|общий|порог p95']['value'])}; {g['Г9|общий|ρ(P_t, M_t)']['result']}.",
        f"- Транспорт и здоровье (общий уровень): η²_H по типам {f(eta_all['Транспорт'][0])} → {f(eta_all['Транспорт'][1])} и "
        f"{f(eta_all['Здоровье'][0])} → {f(eta_all['Здоровье'][1])} при переходе к лог-отношению (шаг 25).",
        f"- Связь с предложением зависит от показателя: Г10 по торговой площади ρ {f(my['g10_in'], '+.3f')}, по числу магазинов ρ "
        f"{f(sens['Г10'][1][L_IN][0], '+.3f')}; Г11 по местам общепита ρ {f(my['g11_in'], '+.3f')}, по всем местам общепита ρ "
        f"{f(sens['Г11'][1][L_IN][0], '+.3f')} (внутри регионов).", "",
        "**Профили типов: σ по долям и по log(доля / «прочее») (шаг 23)**", "",
        md(pd.DataFrame(prof_rows)), "",
        f"Пометка «граница порога»: признак входит в имя типа, но меньшее из двух |σ| превышает порог менее чем на "
        f"{PROFILE_BORDER_MARGIN}σ; запас задан после просмотра профилей (вне предрегистрации).", "",
        "## 4. Ограничения", "",
        "- Доли расходов композиционны: рост одной доли сопровождается снижением других.",
        f"- Пять категорий покрывают в среднем {f(cover, '.0f')}% безналичных расходов (шаг 23).",
        "- Доля «прочего» различается по типам (шаг 25).",
        "- Декабрь 2024 — якорный месяц; сезонность не исключена.",
        "- МО пространственно зависимы: общий уровень только описательный.",
        "- Зарплата — по месту нахождения организаций, а не жителей.", "",
        "## Контрольные сверки", "", md(ctrl), "",
    ]
    text = "\n".join(lines)
    hits = {w: len(re.findall(w, text, flags=re.I)) for w in FORBIDDEN}
    nobr = re.sub(r"\[[+-]?\d+\.\d+; [+-]?\d+\.\d+\]", "", text)
    if any(hits.values()) or "[" in nobr or "]" in nobr:
        raise SystemExit(f"STOP: запрещённые слова {hits} или квадратные скобки вне интервала")
    print("запрещённые слова:", hits, "| квадратные скобки вне интервала: 0")
    check_frozen("после", frozen_log)
    if sha_neg_sources() != neg_sha_before:
        raise SystemExit("STOP: md-отчёты шагов 18 или 21 изменились во время работы")
    print("sha256 md шагов 18 и 21 до и после совпадают:", neg_sha_before)
    i = lines.index("## Заморозка") + 2 + len(FROZEN)
    lines[i:i] = frozen_log[len(FROZEN):]
    out = pd.DataFrame(ROWS)
    if out.duplicated(["block", "item", "level", "metric"]).any():
        raise SystemExit("STOP: дубликаты ключа (block, item, level, metric)")
    out["n"] = out["n"].astype("Int64")
    out.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"записано: {OUT_PATH.name} ({len(out)} строк), {REPORT_PATH.name}")
    lits = template_literals()
    print("числовые литералы в шаблонах текста (без номеров гипотез Г1–Г13; токен: строки файла):")
    for t in sorted(lits, key=lambda x: (float(x), x)):
        print(f"  {t}: {sorted(lits[t])}")


if __name__ == "__main__":
    main()
