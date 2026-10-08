"""Шаг 23. Генератор портретов типов МО (k = 6, канонические метки).

Шаг ОПИСАТЕЛЬНЫЙ: никаких p-значений, тестов, порогов и вердиктов. Все числа в отчёте считаются здесь из данных
проекта (или цитируются из замороженных результатов шагов 18–22 и из колонки fact таблицы источников); вручную
в шаблонах текста числа не набираются. Каждое числовое утверждение заканчивается ровно одной ссылкой
«(расчёт шага 23)» / «(результат шага 19c, Г8)» / «(источник S5)» / «(гипотеза, не проверена)»; проверка ссылок
(lint) и запрещённых слов выполняется до записи отчёта. Расчёты вне предрегистрации помечены
«разведочно, вне предрегистрации». Значения, которых нет в замороженных файлах, пересчитываются только здесь.
Сверка с контрольными значениями (config step23_portraits.CONTROLS): при расхождении выводятся оба числа и шаг
останавливается, значения не подгоняются.

Вход:  kmeans_labels_final, category_shares, prediction_strength_clusters_2024_12, kmeans_k6_matching (data/processed),
       data/raw/{territories, market_access}.parquet, data/external/{rosstat_mo_2023_2025, rosstat_retail_2023_2024}.parquet,
       data/external/{federal_districts, portrait_sources}.csv; hypothesis_verdicts, hypothesis_results_19a/19b/supply
       (только чтение цифр); notebooks/10b_kmeans_cluster_profiles.md (сверка профиля в σ)
Выход: data/processed/type_portraits.parquet (long: type, block, metric, value, n, source_files),
       notebooks/23_type_portraits.md
Запуск из корня проекта:  .venv/bin/python src/23_type_portraits.py
"""
import ast
import csv
import hashlib
import importlib
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (ACCESS_LABEL_RATIO_LOW, BOOT_B, CAPITAL_REGIONS, FAR_EAST_DISTRICT, FINAL_K, MIN_MO_REGION, MONTH,
                    MOSCOW_OBLAST, PORTRAIT_CONTROLS, PROFILE_BORDER_LABEL, PROFILE_BORDER_MARGIN, SEED_23, SIZE_LABEL_RATIO_CITY, SIZE_LABEL_RATIO_LARGE, SIZE_BINS_TH, SMALL_TYPES, TOP_REGIONS, TOP_REGIONS_TYPE4, YAKUTIA, YAKUTIA_SIZE_BINS_TH, ZERO_TOL)
from hypothesis_tools import spearman_rho, within_region_rank
from network_utils import SHARE_COLUMNS, month_shares, standardize_shares

step18 = importlib.import_module("18_hypothesis_thresholds")
step21 = importlib.import_module("21_supply_thresholds")
step10b = importlib.import_module("10b_kmeans_cluster_profiles")

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
EXTERNAL_DIR = PROJECT_DIR / "data" / "external"
SOURCES_PATH = EXTERNAL_DIR / "portrait_sources.csv"
FD_PATH = EXTERNAL_DIR / "federal_districts.csv"
PS_PATH = PROCESSED_DIR / "prediction_strength_clusters_2024_12.parquet"
MATCHING_PATH = PROCESSED_DIR / f"kmeans_k{FINAL_K}_matching.parquet"
VERDICTS_PATH = PROCESSED_DIR / "hypothesis_verdicts.parquet"
RESULTS_19A_PATH = PROCESSED_DIR / "hypothesis_results_19a.parquet"
RESULTS_19B_PATH = PROCESSED_DIR / "hypothesis_results_19b.parquet"
RESULTS_SUPPLY_PATH = PROCESSED_DIR / "hypothesis_results_supply.parquet"
PROFILES_10B_PATH = PROJECT_DIR / "notebooks" / "10b_kmeans_cluster_profiles.md"
PORTRAITS_PATH = PROCESSED_DIR / "type_portraits.parquet"
ROSSTAT_PATH = EXTERNAL_DIR / "rosstat_mo_2023_2025.parquet"
REPORT_PATH = PROJECT_DIR / "notebooks" / "23_type_portraits.md"

FROZEN = {
    PROCESSED_DIR / "hypothesis_thresholds.parquet": "11081afbfc7e88dbd301004e4c739e3f25e00bf8a47704cf7c74e15e8e6665f7",
    RESULTS_19A_PATH: "58e1516ccbf03b2a1fca82971d5170f581815fe18fd8d22bbf261862fdad9d8e",
    RESULTS_19B_PATH: "d66d64d68a82867a5add988d4441fb0df2d35297b8c94c1514bc9e513c615d9d",
    EXTERNAL_DIR / "rosstat_retail_2023_2024.parquet": "58f32111b48c4189811f61c6f462d9b0b8e06d3da431dd99866c8f33e1378a80",
    PROCESSED_DIR / "hypothesis_thresholds_supply.parquet": "c7d0a880d7cbac769ed7c7c8505dffea2d49eb6d3ce525b824238bc5368ddbf6",
    RESULTS_SUPPLY_PATH: "fbfd86ec33e27ad61378987ccd53e727678017cdb1f091808db8f7545bed97db",
    VERDICTS_PATH: "01177c53c539fbabded34f03e85d6a8838726df29f93ecb9d47739f96323659f",
}
DEC = step18.DEC
CATS = ["Продовольствие", "Здоровье", "Общепит", "Транспорт", "Маркетплейсы"]      # порядок SHARE_COLUMNS
CATS_11 = ["Транспорт", "Общепит", "Продовольствие", "Маркетплейсы", "Здоровье"]  # порядок таблицы 1.1
assert [f"share_{c}" for c in CATS] == list(SHARE_COLUMNS)
EXPL = "разведочно, вне предрегистрации"
SMALL_TYPES = list(SMALL_TYPES)
CI_Q = (2.5, 97.5)                      # 95% перцентильный интервал бутстрепа (тип 1)

# ссылки (единственные допустимые форматы)
R_CALC = "(расчёт шага 23)"
R_HYP = "(гипотеза, не проверена)"
MK_L, MK_R = "⟦", "⟧"        # временные метки вокруг вычисленных чисел (снимаются при записи)
CFG_L, CFG_R = "⟪", "⟫"      # метки вокруг чисел из config (не требуют ссылки)
REF_RE = re.compile(r"\((?:расчёт шага 23|результат шага [^)]*|источник S\d[^)]*|гипотеза, не проверена|разведочно, [^)]*)\)")
FORBIDDEN = ["доказано", "причина", "вызывает", "объясняется", "из-за", "следует"]
ALLOWED_INT_MAX = 24                    # номера типов, шагов, гипотез, источников, «23 месяцев», «24 месяцев»
ALLOWED_YEARS = {2004, 2012, 2023, 2024, 2025}
TABLE_EXEMPT_PREFIX = "Таблица источников"

# ── тексты гипотез (единственные допустимые формулировки, числа подставляются) ──
H_TEXT = {
    0: "Гипотеза (не проверена): в малых МО ограниченное офлайн-предложение сдвигает покупки в онлайн. Региональная "
       "литература не нашла влияния физической доступности торговых точек (источник S11). Между регионами онлайн-покупки "
       "связаны с интернет-инфраструктурой, ВРП, размером населения и федеральным округом (источник S10). Внутри регионов доля "
       "маркетплейсов обратно связана с торговой площадью на душу: ρ = {g10} (результат шага 22, Г10); после поправки "
       "на размер МО ρ = {partial} (разведочно, результат шага 22).",
    1: "Наблюдение: доля транспорта от всех расходов заметно меняется (интервал разности медиан не включает ноль) только "
       "для пар {pairs} (таблица 1.1), а отношение транспорта к «прочему» по группам размера не меняется (таблица 1.2) "
       "(расчёт шага 23): рост доли транспорта идёт вместе с ростом доли «прочего», специфического роста транспорта эти "
       "данные не показывают. Исследование использования транспорта по людности городов (источник S2) касается "
       "использования, а не доли расходов; связь с долей расходов не проверялась (гипотеза, не проверена).",
    2: "Не объяснено: высокая доля здравоохранения. Доля старших даёт {exp} п.п. из {act} п.п. разности с типом 0 "
       "(расчёт шага 23); в Г7 учёт возраста различие типов не уменьшил (результат шага 19c, Г7). Про маркетплейсы: в 2025 году "
       "Wildberries открыла или строила склады, среди них в Волгограде и Саратове, то есть после периода данных (источник S4).",
    3: "Наблюдение: общепит и транспорт (в долях от всех расходов) высоки не только в Москве и Санкт-Петербурге, но и в "
       "остальных МО типа (расчёт шага 23); относительно «прочего» устойчиво выше общепит, а транспорт и здоровье не "
       "выделяются (|σ| < {th}) (таблица признаков) (расчёт шага 23). В категорию «Транспорт» входят такси и каршеринг, в "
       "категорию «Общепит» — доставка готовой еды (источник S0). Продовольствие: городские домохозяйства тратят на питание иначе, чем сельские (источник S1).",
    4: "Гипотеза (не проверена): доля маркетплейсов зависит от доставки и пунктов выдачи. Между регионами она растёт с "
       "market_access, внутри регионов падает с размером МО, в Якутии наоборот (расчёт шага 23). В Якутске {pickup} "
       "пунктов выдачи, склады планируются (источник S5). Альтернативный канал покупок через иностранные площадки не "
       "проверялся (источник S8, S6, S7 — только контекст). Здоровье: при записи в лог-отношении ниже здоровье типа 4 "
       "сильнее (−{alr} σ) (расчёт шага 23); моложе население, но возраст различия между типами "
       "не объясняет (результат шага 19c, Г7); местное предложение медицинских услуг не проверялось (источник S9 — данные "
       "2012 года).",
    5: "Наблюдение: связь отрицательная на обоих уровнях: в МО с большей зарплатой относительно региона доля "
       "продовольствия ниже (общий ρ {g3_all}, внутри регионов ρ {g3_in}; результат шага 19c, Г3, разведочная). "
       "Продовольствие включает алкоголь и табак, зарплата относится к организациям (источник S0, S12).",
}

LINES: list = []
SOURCE_IDS: list = []
LONG: list = []


def cfg(v) -> str:
    """Число из config (без ссылки)."""
    return f"{CFG_L}{v}{CFG_R}"


def R(t, block, metric, value, n, src) -> None:
    v = np.nan if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) else float(value)
    LONG.append({"type": str(t), "block": block, "metric": metric, "value": v, "n": None if n is None else int(n),
                 "source_files": src})


def fnum(v, fmt: str) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return "—"
    out = format(v, fmt)
    return out[1:] if re.fullmatch(r"[+-]0(?:\.0+)?", out) else out      # без знака у нуля («-0.0» -> «0.0»)


def cell(v, fmt, t, block, metric, n, src) -> str:
    R(t, block, metric, v, n, src)
    return fnum(v, fmt)


def num(v, fmt, t, block, metric, n, src) -> str:
    return MK_L + cell(v, fmt, t, block, metric, n, src) + MK_R


def mk(text: str) -> str:
    """Текст с числами из внешнего источника (колонка fact) — тоже требует ссылки."""
    return MK_L + text + MK_R


def add(*lines) -> None:
    LINES.extend(lines)


def tbl(caption: str, df: pd.DataFrame, ref: str = R_CALC) -> None:
    add(f"{caption} {ref}", "", df.to_markdown(index=False, disable_numparse=True), "")


def med(s: pd.Series) -> tuple:
    s = s.dropna()
    return (float(s.median()) if len(s) else np.nan), int(len(s))


def rho(x: pd.Series, y: pd.Series) -> tuple:
    m = x.notna() & y.notna()
    return (spearman_rho(x[m].to_numpy(float), y[m].to_numpy(float)) if m.sum() > 2 else np.nan), int(m.sum())


def size_labels(bins_th) -> list:
    b = list(bins_th)
    return [f"до {b[0]}"] + [f"{b[i]}–{b[i + 1]}" for i in range(len(b) - 1)] + [f"более {b[-1]}"]


def size_group(pop: pd.Series, bins_th) -> pd.Series:
    return pd.cut(pop, [0] + [b * 1000 for b in bins_th] + [np.inf], labels=size_labels(bins_th), right=False)


def rho_between(df: pd.DataFrame, xc: str, yc: str) -> tuple:
    """ρ между регионами: медианы регионов с не менее MIN_MO_REGION МО, где есть обе величины."""
    s = df[df[xc].notna() & df[yc].notna()]
    cnt = s.groupby("region").size()
    ok = cnt[cnt >= MIN_MO_REGION].index
    m = s[s["region"].isin(ok)].groupby("region")[[xc, yc]].median()
    return spearman_rho(m[xc].to_numpy(), m[yc].to_numpy()), int(len(ok)), int(cnt.loc[ok].sum())


def rho_within(df: pd.DataFrame, xc: str, yc: str) -> tuple:
    """ρ внутри регионов: значения заменены нормированным рангом внутри региона (регионы с числом МО с данными
    < MIN_MO_REGION не участвуют), корреляция Пирсона по МО (как на уровне «внутри регионов» шага 18)."""
    s = df[df[xc].notna() & df[yc].notna()]
    cnt = s.groupby("region").size()
    ok = cnt[cnt >= MIN_MO_REGION].index
    s = s[s["region"].isin(ok)]
    reg = s["region"].to_numpy()
    rx = within_region_rank(s[xc].to_numpy(float), reg)
    ry = within_region_rank(s[yc].to_numpy(float), reg)
    return float(np.corrcoef(rx, ry)[0, 1]), int(len(ok)), int(len(s))


# ─────────────────────────── заморозка, данные ───────────────────────────
def check_frozen(when: str, log: list) -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{when}: {got}  {path.relative_to(PROJECT_DIR)} {'OK' if got == expected else 'MISMATCH'}")
        log.append(f"- {when}: `{path.relative_to(PROJECT_DIR)}` — {'совпадает' if got == expected else 'НЕ СОВПАДАЕТ'}")
        if got != expected:
            raise SystemExit(f"STOP: sha256 {path.name} = {got}, ожидается {expected}")


def load_sources() -> pd.DataFrame:
    with open(SOURCES_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    if any(len(r) != 7 for r in rows):
        raise SystemExit(f"STOP: в {SOURCES_PATH.name} не у всех строк 7 колонок")
    return pd.DataFrame(rows[1:], columns=rows[0])


def build_frame(d: dict, sup: dict) -> pd.DataFrame:
    ids = d["ids"]
    terr = d["terr"]
    D = pd.DataFrame(index=pd.Index(ids, name="territory_id"))
    D["type"] = d["parts"][DEC]
    D["region"] = terr["region_name"].to_numpy()
    D["name"] = terr["name"].to_numpy()
    D["mdt"] = terr["municipal_district_type"].to_numpy()
    for c in CATS:
        D[c] = step21.share(d, f"share_{c}", MONTH) * 100
    D["market_access"] = d["ma"].reindex(ids).to_numpy(float)
    ros = d["ros_by_year"][2023]                       # МО-годы с anomaly уже пустые во всех показателях Росстата
    D["ros_anomaly"] = ros["anomaly"].eq(True).to_numpy()
    pop = ros["population"].to_numpy(float)
    pop = np.where(pop > 0, pop, np.nan)               # население ≤ 0 или пропуск: пусты только показатели, зависящие от населения
    ok = ~np.isnan(pop)

    def r(col, scale=1.0, per_pop=False):
        v = ros[col].to_numpy(float) * scale
        return np.where(ok, v, np.nan) if per_pop else v

    D["pop"] = pop
    D["log_pop"] = np.log(pop)
    D["older"] = r("share_older", 100, per_pop=True)
    D["wage_rel"] = r("wage_rel_region")
    D["wpp"] = r("workers_per_pop", per_pop=True)
    for L in "ACGQ":
        D[f"okved_{L}"] = r(f"share_okved_{L}", 100)
    D["floor_pc"] = sup["floor_pc"]
    D["seats_pc"] = sup["seats_pc"]
    D["size_group"] = size_group(D["pop"], SIZE_BINS_TH)
    fd = pd.read_csv(FD_PATH, dtype=str, keep_default_na=False).set_index("region_name")["federal_district"]
    D["fd"] = D["region"].map(fd)
    if D["fd"].isna().any() or (D["fd"] == "").any():
        raise SystemExit("STOP: не у всех МО есть федеральный округ")
    return D


def z_profile(d: dict, D: pd.DataFrame) -> pd.DataFrame:
    """Средний z-score по типу (стандартизация как в шаге 10b); сверка с notebooks/10b_kmeans_cluster_profiles.md."""
    month = month_shares(d["shares"].drop(columns=["month"]), MONTH).reset_index(drop=True)
    if not np.array_equal(month["territory_id"].to_numpy(), d["ids"]):
        raise SystemExit("STOP: порядок МО в category_shares не совпадает с выборкой")
    X, _ = standardize_shares(month[list(SHARE_COLUMNS)].to_numpy())
    z = pd.DataFrame(X, columns=CATS).assign(type=D["type"].to_numpy()).groupby("type")[CATS].mean()
    text = PROFILES_10B_PATH.read_text(encoding="utf-8")
    heads = list(re.finditer(r"^## k = (\d+)\s*$", text, flags=re.M))
    start = next(h for h in heads if int(h.group(1)) == FINAL_K)
    end = next((h.start() for h in heads if h.start() > start.start()), len(text))
    part = text[start.end():end].split("### Отклонение от среднего (σ)")[1]
    ref = {}
    for line in part.splitlines():
        if re.match(r"\|\s*\d+\s*\|", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            ref[int(cells[0])] = [float(c) for c in cells[1:6]]
        elif ref and not line.startswith("|"):
            break
    bad = []
    for t in range(FINAL_K):
        for c, v10b in zip(CATS, ref[t]):
            if abs(round(z.loc[t, c], 2) - v10b) > 0.0051:
                bad.append((t, c, round(float(z.loc[t, c]), 2), v10b))
    print(f"сверка профиля в σ с 10b (k = {FINAL_K}): расхождений {len(bad)}")
    if bad:
        raise SystemExit(f"STOP: профиль в σ не совпадает с 10b (тип, категория, шаг 23, 10b): {bad}")
    return z


# ─────────────────────────── разделы ───────────────────────────
def legend(D, sources) -> None:
    add("## 0. Как читать", "",
        "- «(расчёт шага 23)» — число или таблица посчитаны генератором `src/23_type_portraits.py` из файлов проекта; "
        "в таблице `data/processed/type_portraits.parquet` у каждого числа указаны `source_files`.",
        "- «(результат шага 19c, Г8)» и подобные — цифра из замороженных результатов шагов 18–22 (только цитата).",
        "- «(источник S5)» — внешняя публикация, расшифровка в разделе «Источники»; числа из источников — только из колонки `fact`.",
        "- «(гипотеза, не проверена)» — предположение.",
        f"- Метка «{EXPL}» — расчёт, которого нет в предрегистрации (шаги 18–22 замороженные, здесь только описание).",
        "- База сравнения: относительно среднего по 2004 МО. Профиль типа в σ — среднее z-score МО типа; стандартизация по "
        f"2004 МО как в шаге 10b, порог подписи {cfg(step10b.PROFILE_Z_THRESHOLD)}σ; значения сверены с "
        "`notebooks/10b_kmeans_cluster_profiles.md`.",
        "- Период: доли расходов — декабрь 2024, в процентах от «Все категории»; население — на 1 января 2023 (шаг 17); "
        "показатели Росстата — 2023; розница и общепит — IV квартал 2023 (шаг 20).",
        "- Исключения Росстата: МО-годы с anomaly исключены из всех показателей Росстата. Население 2023 пусто у "
        + nopop_phrase(D, "исключения Росстата") + ". У МО без населения пусты только показатели, зависящие от населения: "
        "население, доля старших, floor_pc, seats_pc, работники на жителя; зарплата относительно региона и доли отраслей "
        f"сохраняются {R_CALC}.",
        "- «Медиана по МО» — медиана по МО типа (группы) с непустым значением, n указан рядом. «Внутри региона» — "
        f"нормированный ранг внутри региона (регионы с числом МО с данными < {cfg(MIN_MO_REGION)} не участвуют). «Между "
        f"регионами» — медианы регионов с числом МО не менее {cfg(MIN_MO_REGION)}.",
        *label_ratios_block(D),
        f"- Группы размера МО, тыс. жителей: {', '.join(cfg(x) for x in size_labels(SIZE_BINS_TH))}. Типы: канонические "
        "метки `kmeans_labels_final.parquet` (k = 6).", "")


SRC_NOPOP = "rosstat_mo_2023_2025.parquet; territories.parquet"


def nopop_phrase(D, block: str) -> str:
    """«32 МО: a без населения 2023 (пропуск или 0), b из них не аномальны (регион — n), 2 МО с anomaly (...)».
    Население берётся из исходного файла шага 17 (до обнуления МО-годов с anomaly)."""
    raw = pd.read_parquet(ROSSTAT_PATH)
    raw = raw[raw["year"] == 2023].set_index("territory_id").reindex(D.index)
    pop = raw["population"].to_numpy(float)
    if (pop < 0).any():
        raise SystemExit("STOP: в населении 2023 есть отрицательные значения (в тексте «пропуск или 0»)")
    nop_raw = ~(pop > 0)
    anom = D["ros_anomaly"].to_numpy()
    total = int(D["pop"].isna().sum())
    if total != int(nop_raw.sum()) + int((anom & ~nop_raw).sum()):
        raise SystemExit("STOP: число МО с пустым населением не равно сумме «без населения» и «с anomaly при наличии населения»")
    clean = nop_raw & ~anom
    reg = D.loc[clean, "region"].value_counts()
    ages = raw[["share_younger", "share_working_age", "share_older"]].sum(axis=1, min_count=3).to_numpy(float)
    desc = []
    for i in np.flatnonzero(anom):
        if pop[i] == 0:
            why = "население 0"
        elif np.isnan(pop[i]):
            why = "население пропущено"
        elif not np.isclose(ages[i], 1.0):
            why = "население есть, аномальна сумма возрастных долей"
        else:
            raise SystemExit(f"STOP: для МО с anomaly {D['name'].iloc[i]} не определено, что аномально")
        desc.append(f"{D['name'].iloc[i]}, {why}")
    src = SRC_NOPOP
    regs = (reg.index[0] + " — " + num(int(reg.iloc[0]), "d", "все", block, f"не аномальные МО без населения 2023: {reg.index[0]}", int(nop_raw.sum()), src)
            if len(reg) == 1 else
            "; ".join(f"{r_} — " + num(int(n_), "d", "все", block, f"не аномальные МО без населения 2023: {r_}", int(nop_raw.sum()), src) for r_, n_ in reg.items()))
    return (num(total, "d", "все", block, "МО с пустым населением 2023", len(D), src) + " МО: "
            + num(int(nop_raw.sum()), "d", "все", block, "МО без населения 2023 (пропуск или 0)", len(D), src)
            + " без населения 2023 (пропуск или 0), "
            + num(int(clean.sum()), "d", "все", block, "из них не аномальны", int(nop_raw.sum()), src)
            + f" из них не аномальны ({regs}), "
            + num(int(anom.sum()), "d", "все", block, "МО с anomaly", len(D), src)
            + " МО с anomaly (" + "; ".join(desc) + ")")


# ─────────────────────── признаки профиля, имена типов, ярлыки ───────────────────────
def alr_profile(d: dict, D: pd.DataFrame) -> pd.DataFrame:
    """Средний z-score типа по log(доля / «прочее»), «прочее» = 1 − сумма пяти долей; z-score по 2004 МО."""
    month = month_shares(d["shares"].drop(columns=["month"]), MONTH).reset_index(drop=True)
    if not np.array_equal(month["territory_id"].to_numpy(), d["ids"]):
        raise SystemExit("STOP: порядок МО в category_shares не совпадает с выборкой")
    S = month[list(SHARE_COLUMNS)].to_numpy(float)
    rest = 1.0 - S.sum(axis=1)
    if (S <= 0).any() or (rest <= 0).any():
        raise SystemExit("STOP: нулевые доли или «прочее» ≤ 0 — лог-отношение не определено")
    X, _ = standardize_shares(np.log(S / rest[:, None]))
    return pd.DataFrame(X, columns=CATS).assign(type=D["type"].to_numpy()).groupby("type")[CATS].mean()


def feature_mark(z: pd.DataFrame, za: pd.DataFrame, t: int, c: str) -> str:
    """«держится» / «зависит от записи» для признака с |σ| по долям ≥ порога 10b; иначе пусто."""
    th = step10b.PROFILE_Z_THRESHOLD
    s, a = float(z.loc[t, c]), float(za.loc[t, c])
    if abs(s) < th:
        return ""
    return "держится" if np.sign(a) == np.sign(s) and abs(a) >= th else "зависит от записи"


def label_ratios(D: pd.DataFrame) -> pd.DataFrame:
    """Отношения медиан типа к медианам выборки: население (МО с населением 2023) и market_access (2004 МО)."""
    pop_all, ma_all = float(D["pop"].median()), float(D["market_access"].median())
    rows = []
    for t in range(FINAL_K):
        sub = D[D["type"] == t]
        pr, mr = float(sub["pop"].median()) / pop_all, float(sub["market_access"].median()) / ma_all
        labs = []
        if pr >= SIZE_LABEL_RATIO_CITY:
            labs.append("крупные города")
        elif pr >= SIZE_LABEL_RATIO_LARGE:
            labs.append("крупнее среднего МО")
        if mr <= ACCESS_LABEL_RATIO_LOW:
            labs.append("низкая доступность рынков")
        rows.append({"type": t, "pop_med": float(sub["pop"].median()), "pop_n": int(sub["pop"].notna().sum()), "pop_ratio": pr,
                     "ma_med": float(sub["market_access"].median()), "ma_n": int(sub["market_access"].notna().sum()), "ma_ratio": mr,
                     "labels": labs})
    out = pd.DataFrame(rows).set_index("type")
    out.attrs["pop_all"], out.attrs["ma_all"] = pop_all, ma_all
    return out


def type_name(D: pd.DataFrame, z: pd.DataFrame, za: pd.DataFrame, t: int) -> str:
    feats = [c for c in z.columns[np.argsort(-z.loc[t].abs().to_numpy(), kind="stable")] if feature_mark(z, za, t, c) == "держится"]
    base = (", ".join(f"{c.lower()} {'выше' if z.loc[t, c] > 0 else 'ниже'}" for c in feats) + " среднего") if feats \
        else "Без выраженного профиля"
    return " · ".join([base] + label_ratios(D).loc[t, "labels"])


def border_features(z: pd.DataFrame, za: pd.DataFrame, t: int) -> list:
    """Признаки базового имени (пометка «держится»), у которых меньшее из |σ| по долям и по лог-отношению
    < порог подписи 10b + PROFILE_BORDER_MARGIN: [(категория, меньшее |σ|)] в порядке имени."""
    lim = step10b.PROFILE_Z_THRESHOLD + PROFILE_BORDER_MARGIN
    feats = [c for c in z.columns[np.argsort(-z.loc[t].abs().to_numpy(), kind="stable")] if feature_mark(z, za, t, c) == "держится"]
    out = [(c, min(abs(float(z.loc[t, c])), abs(float(za.loc[t, c])))) for c in feats]
    return [(c, m) for c, m in out if m < lim]


def full_type_name(D: pd.DataFrame, z: pd.DataFrame, za: pd.DataFrame, t: int) -> str:
    """Базовое имя (type_name, без изменений) · пометка «на границе порога: ...», если есть пограничные признаки."""
    bf = border_features(z, za, t)
    name = type_name(D, z, za, t)
    return name + (f" · {PROFILE_BORDER_LABEL}: " + ", ".join(c.lower() for c, _ in bf) if bf else "")


def gap(ratios: np.ndarray, th: float, side: str) -> tuple:
    """Ближайшие отношения по обе стороны порога: (наибольшее ниже, наименьшее не ниже) или для «≤» — (наибольшее ≤, наименьшее >)."""
    if side == "ge":
        lo, hi = ratios[ratios < th], ratios[ratios >= th]
    else:
        lo, hi = ratios[ratios <= th], ratios[ratios > th]
    if not len(lo) or not len(hi):
        raise SystemExit(f"STOP: все отношения по одну сторону порога {th}: {ratios}")
    return float(lo.max()), float(hi.min())


def label_ratios_block(D: pd.DataFrame) -> list:
    lr = label_ratios(D)
    src = "kmeans_labels_final.parquet; rosstat_mo_2023_2025.parquet; market_access.parquet"
    rows = [{"тип": str(t),
             "медиана населения, тыс.": cell(r["pop_med"] / 1000, ".1f", t, "ярлыки", "медиана населения, тыс.", r["pop_n"], src),
             "отношение к медиане МО с населением 2023": cell(r["pop_ratio"], ".2f", t, "ярлыки", "население: отношение к медиане всех МО", r["pop_n"], src),
             "медиана market_access": cell(r["ma_med"], ".1f", t, "ярлыки", "медиана market_access", r["ma_n"], src),
             "отношение к медиане всех 2004 МО": cell(r["ma_ratio"], ".2f", t, "ярлыки", "market_access: отношение к медиане всех МО", r["ma_n"], src),
             "второй ярлык": " · ".join(r["labels"]) or "—"} for t, r in lr.iterrows()]
    lo_l, hi_l = gap(lr["pop_ratio"].to_numpy(), SIZE_LABEL_RATIO_LARGE, "ge")
    lo_c, hi_c = gap(lr["pop_ratio"].to_numpy(), SIZE_LABEL_RATIO_CITY, "ge")
    lo_a, hi_a = gap(lr["ma_ratio"].to_numpy(), ACCESS_LABEL_RATIO_LOW, "le")
    LINES_0 = [
        f"- Второй ярлык имени типа: «крупнее среднего МО» — медиана населения типа не меньше {cfg(SIZE_LABEL_RATIO_LARGE)} "
        f"медиан населения всех МО выборки (МО с населением 2023: {num(lr.attrs['pop_all'] / 1000, '.1f', 'все', 'ярлыки', 'медиана населения всех МО, тыс.', int(D['pop'].notna().sum()), src)} тыс.); "
        f"«крупные города» (вместо предыдущего) — то же с множителем {cfg(SIZE_LABEL_RATIO_CITY)}; «низкая доступность рынков» — "
        f"медиана market_access типа не больше {cfg(ACCESS_LABEL_RATIO_LOW)} медианы всех 2004 МО "
        f"({num(lr.attrs['ma_all'], '.1f', 'все', 'ярлыки', 'медиана market_access всех МО', len(D), src)}) {R_CALC}.", ""]
    LINES_0 += [f"- Пометка «{PROFILE_BORDER_LABEL}»: признак входит в имя, но меньшее из двух |σ| превышает порог менее чем на "
                f"{cfg(PROFILE_BORDER_MARGIN)}σ; запас задан после просмотра профилей (вне предрегистрации).", ""]
    out = list(LINES_0)
    out += [f"Отношения медиан типа к медианам выборки {R_CALC}", "", pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True), ""]
    out += [f"- Отношения не лежат вблизи порогов: для населения между "
            f"{num(lo_l, '.2f', 'все', 'ярлыки', 'население: ближайшее отношение ниже порога крупнее среднего МО', None, src)} и "
            f"{num(hi_l, '.2f', 'все', 'ярлыки', 'население: ближайшее отношение не ниже порога крупнее среднего МО', None, src)}, для market_access между "
            f"{num(lo_a, '.2f', 'все', 'ярлыки', 'market_access: ближайшее отношение не выше порога', None, src)} и "
            f"{num(hi_a, '.2f', 'все', 'ярлыки', 'market_access: ближайшее отношение выше порога', None, src)} порог не меняет ярлыки; для порога "
            f"«крупные города» — между {num(lo_c, '.2f', 'все', 'ярлыки', 'население: ближайшее отношение ниже порога крупные города', None, src)} и "
            f"{num(hi_c, '.2f', 'все', 'ярлыки', 'население: ближайшее отношение не ниже порога крупные города', None, src)} {R_CALC}.", ""]
    return out


def type3_text(D, z, za) -> None:
    th = step10b.PROFILE_Z_THRESHOLD
    if feature_mark(z, za, 3, "Общепит") != "держится" or z.loc[3, "Общепит"] <= 0:
        raise SystemExit("STOP: общепит типа 3 не «держится» выше среднего — формулировка текста типа 3 неверна")
    for c in ["Транспорт", "Здоровье"]:
        if abs(za.loc[3, c]) >= th:
            raise SystemExit(f"STOP: {c} типа 3 по лог-отношению |σ| = {abs(za.loc[3, c]):.2f} ≥ {th} — формулировка неверна")
    add(H_TEXT[3].format(th=cfg(th)), "")


def sec_1_2_rest(D) -> dict:
    """Таблица 1.2: транспорт и общепит относительно «прочего» по группам размера; ρ с log населения."""
    src = "kmeans_labels_final.parquet; category_shares.parquet; rosstat_mo_2023_2025.parquet"
    E = D.assign(rest=100 - D[CATS].sum(axis=1))
    E["tr_r"], E["ca_r"] = E["Транспорт"] / E["rest"], E["Общепит"] / E["rest"]
    rows = []
    for g in size_labels(SIZE_BINS_TH):
        s = E[E["size_group"] == g]
        row = {"группа, тыс. жителей": g, "n МО": str(len(s))}
        for col, lab, fmt in [("Транспорт", "транспорт, % (медиана)", ".1f"), ("tr_r", "транспорт / «прочее» (медиана)", ".3f"),
                              ("rest", "«прочее», % (медиана)", ".1f"), ("ca_r", "общепит / «прочее» (медиана)", ".3f")]:
            v, n = med(s[col])
            row[lab] = cell(v, fmt, "все", "1.2 относительно «прочего»", f"{g}: {lab}", n, src)
        rows.append(row)
    add("### 1.2 Транспорт и общепит относительно «прочего»", "",
        "«Прочее» — 1 − сумма пяти долей (в процентах от «Все категории»); МО без населения не входят.", "")
    tbl(f"Медианы по группам размера — {EXPL}", pd.DataFrame(rows))
    out, rr = {}, {}
    for col, lab in [("Транспорт", "транспорт"), ("tr_r", "транспорт/«прочее»"), ("rest", "«прочее»"),
                     ("Общепит", "общепит"), ("ca_r", "общепит/«прочее»")]:
        v, n = rho(E[col], E["log_pop"])
        rr[lab] = cell(v, "+.2f", "все", "1.2 относительно «прочего»", f"ρ({lab}, log население)", n, src)
        rr[f"n {lab}"] = str(n)
        out[lab] = v
    tbl(f"ρ Спирмена с log(население 2023) — {EXPL}", pd.DataFrame([rr]))
    return {"rho_logpop_rest": out}


def sec_1_1(D) -> dict:
    labs = size_labels(SIZE_BINS_TH)
    rows, out = [], {"transport": [], "catering": []}
    src = "kmeans_labels_final.parquet; category_shares.parquet; rosstat_mo_2023_2025.parquet; rosstat_retail_2023_2024.parquet"
    for g in labs:
        s = D[D["size_group"] == g]
        row = {"группа, тыс. жителей": g, "n МО": cell(len(s), "d", "все", "1.1 группы размера", f"{g}: n МО", len(s), src)}
        for c in CATS_11:
            v, n = med(s[c])
            row[f"{c}, %"] = cell(v, ".1f", "все", "1.1 группы размера", f"{g}: медиана {c}", n, src)
            if c == "Транспорт":
                out["transport"].append(v)
            if c == "Общепит":
                out["catering"].append(v)
        for c, lab, fmt in [("floor_pc", "floor_pc, м² на 1000 жителей", ".0f"), ("seats_pc", "seats_pc, мест на 1000 жителей", ".1f")]:
            v, n = med(s[c])
            row[lab] = cell(v, fmt, "все", "1.1 группы размера", f"{g}: медиана {c}", n, src)
            row[f"n {c}"] = str(n)
        rows.append(row)
    n_all = int(D["size_group"].notna().sum())
    add("### 1.1 Группы размера МО", "")
    tbl("Медианы по группам размера (население 2023; МО без населения не входят: "
        f"{num(len(D) - n_all, 'd', 'все', '1.1 группы размера', 'МО без населения 2023', len(D), src)})", pd.DataFrame(rows))
    rr = []
    for c in CATS_11:
        row = {"категория": c}
        v, n = rho(D[c], D["log_pop"])
        row["все МО: ρ"], row["все МО: n"] = cell(v, "+.2f", "все", "1.1 ρ с log(население)", f"{c}: все МО", n, src), str(n)
        if c == "Общепит":
            out["rho_catering"] = v
        for t in SMALL_TYPES:
            v, n = rho(D.loc[D["type"] == t, c], D.loc[D["type"] == t, "log_pop"])
            row[f"тип {t}: ρ"], row[f"тип {t}: n"] = cell(v, "+.2f", t, "1.1 ρ с log(население)", f"{c}: внутри типа", n, src), str(n)
        rr.append(row)
    tbl(f"ρ Спирмена доли и log(население 2023) — {EXPL}", pd.DataFrame(rr))
    return out


def sec_1_2(D) -> dict:
    src = "category_shares.parquet; market_access.parquet; rosstat_mo_2023_2025.parquet; territories.parquet"
    rows, out = [], {}
    for lab, xc, yc in [("доля маркетплейсов ~ market_access", "Маркетплейсы", "market_access"),
                        ("доля маркетплейсов ~ log(население 2023)", "Маркетплейсы", "log_pop")]:
        b = rho_between(D, xc, yc)
        w = rho_within(D, xc, yc)
        for level, (r_, nr, nm) in [("между регионами", b), ("внутри регионов", w)]:
            rows.append({"пара": lab, "уровень": level,
                         "ρ": cell(r_, "+.2f", "все", "1.3 маркетплейсы", f"{lab}: {level}", nm, src),
                         "регионов": str(nr), "МО": str(nm)})
            R("все", "1.3 маркетплейсы", f"{lab}: {level}: регионов", nr, nm, src)
            out[(yc, level)] = r_
    add("### 1.3 Маркетплейсы: между регионами и внутри регионов", "")
    tbl(f"ρ Спирмена, регионы с числом МО с данными не менее {cfg(MIN_MO_REGION)} — {EXPL}", pd.DataFrame(rows))
    return out


def sec_1_3(D) -> None:
    src = "kmeans_labels_final.parquet; category_shares.parquet; rosstat_mo_2023_2025.parquet"
    grp = {c: D.groupby("size_group", observed=True)[c].median() for c in CATS}
    rows = []
    for t in range(FINAL_K):
        for c in CATS:
            s = D[(D["type"] == t) & D["size_group"].notna()]
            typed = float(s[c].median())
            exp = float(s["size_group"].map(grp[c]).astype(float).median())
            rows.append({"тип": str(t), "категория": c,
                         "медиана типа, %": cell(typed, ".1f", t, "1.4 сравнение с МО того же размера", f"{c}: медиана типа", len(s), src),
                         "ожидаемая по размеру, %": cell(exp, ".1f", t, "1.4 сравнение с МО того же размера", f"{c}: ожидаемая по размеру", len(s), src),
                         "разность, п.п.": cell(typed - exp, "+.1f", t, "1.4 сравнение с МО того же размера", f"{c}: разность", len(s), src),
                         "n": str(len(s))})
    add("### 1.4 Сравнение с МО того же размера", "",
        "Для каждого МО типа берётся медиана категории по всем МО той же группы размера; ожидаемая — медиана этих "
        "значений по МО типа (МО без населения не входят).", "")
    tbl(f"Медиана типа, ожидаемая по размеру и разность — {EXPL}", pd.DataFrame(rows))


def sec_1_4(sources) -> None:
    r = sources[sources["id"] == "S1"].iloc[0]
    add("### 1.5 Внешние ориентиры", "",
        f"{r['источник']}: {mk(r['fact'])} (источник S1).", "")


def small_types_table(D) -> None:
    src = "kmeans_labels_final.parquet; category_shares.parquet; rosstat_mo_2023_2025.parquet; rosstat_retail_2023_2024.parquet"
    metrics = [("здоровье, %", "Здоровье", ".1f"), ("маркетплейсы, %", "Маркетплейсы", ".1f"),
               ("транспорт, %", "Транспорт", ".1f"), ("продовольствие, %", "Продовольствие", ".1f"),
               ("доля старших, %", "older", ".1f"), ("доля занятых в здравоохранении (Q), %", "okved_Q", ".1f"),
               ("доля занятых в сельском хозяйстве (A), %", "okved_A", ".1f"),
               ("зарплата относительно медианы региона", "wage_rel", ".2f"),
               ("floor_pc, м² на 1000 жителей", "floor_pc", ".0f"), ("seats_pc, мест на 1000 жителей", "seats_pc", ".1f")]
    rows = []
    for lab, col, fmt in metrics:
        row = {"показатель (медиана по МО типа, n)": lab}
        for t in SMALL_TYPES:
            v, n = med(D.loc[D["type"] == t, col])
            row[f"тип {t}"] = cell(v, fmt, t, "2.0 малые типы", f"{lab}: медиана", n, src) + f" (n={n})"
        rows.append(row)
    add("### 2.0 Сравнительная таблица малых типов", "")
    tbl(f"Малые типы {', '.join(str(t) for t in SMALL_TYPES)}: медианы и число МО с данными", pd.DataFrame(rows))


def reliability(D, ps, matching, r19b, t) -> None:
    src_ps = "prediction_strength_clusters_2024_12.parquet"
    p = ps[(ps["k"] == FINAL_K) & (ps["доминирующий тип k=6"] == t)]
    ps_mean = float(p["доля пар"].mean())
    m = matching[(matching["cluster"] == t) & (matching["месяц"] != MONTH)]
    conf = int((m["уверенность"] == "уверенно").sum())
    g6 = r19b[(r19b["row_type"] == "g6_type") & (r19b["variant"] == "основной") & (r19b["type"] == t)]
    if len(g6) != 1:
        raise SystemExit(f"STOP: в hypothesis_results_19b нет ровно одной строки Г6 для типа {t}: {len(g6)}")
    g = g6.iloc[0]
    add("**Надёжность**", "",
        f"- Prediction strength типа: среднее «доля пар» по строкам k = 6 с этим доминирующим типом — "
        f"{num(ps_mean, '.2f', t, 'надёжность', 'PS типа (среднее доля пар)', len(p), src_ps)} {R_CALC}.",
        f"- Уверенных сопоставлений типа с якорем: {num(conf, 'd', t, 'надёжность', 'уверенных сопоставлений из 23', len(m), MATCHING_PATH.name)} "
        f"из {num(len(m), 'd', t, 'надёжность', 'месяцев без якоря', len(m), MATCHING_PATH.name)} {R_CALC}.",
        f"- Г6: доля переходов из типа в ближайшие типы (число ближайших: {num(int(g['n_nearest']), 'd', t, 'надёжность', 'Г6: N ближайших', None, RESULTS_19B_PATH.name)}) "
        f"{num(g['obs'], '.2f', t, 'надёжность', 'Г6: S по типу', int(g['n']), RESULTS_19B_PATH.name)}, базовый уровень нуля "
        f"{num(g['null_mean'], '.2f', t, 'надёжность', 'Г6: базовый уровень', int(g['n']), RESULTS_19B_PATH.name)}; переходов "
        f"{num(int(g['n']), 'd', t, 'надёжность', 'Г6: переходов', int(g['n']), RESULTS_19B_PATH.name)}, МО "
        f"{num(int(g['n_distinct_mo']), 'd', t, 'надёжность', 'Г6: МО с переходами', int(g['n']), RESULTS_19B_PATH.name)} "
        "(результат шага 19b, Г6).", "")
    return ps_mean, conf


def portrait_common(D, z, za, ps, matching, r19b, t) -> tuple:
    sub = D[D["type"] == t]
    N = len(sub)
    src = "kmeans_labels_final.parquet; territories.parquet"
    src_c = "kmeans_labels_final.parquet; category_shares.parquet"
    src_r = "kmeans_labels_final.parquet; rosstat_mo_2023_2025.parquet"
    reg = sub["region"].value_counts()
    n_reg = int(len(reg))
    top = pd.DataFrame({"регион": reg.index, "МО": reg.to_numpy()}).sort_values(["МО", "регион"], ascending=[False, True]).head(TOP_REGIONS)
    top["МО"] = [cell(v, "d", t, "состав", f"регион {r}: МО", N, src) for r, v in zip(top["регион"], top["МО"])]
    add(f"### Тип {t}", "", "**Состав**", "",
        f"- МО типа: {num(N, 'd', t, 'состав', 'МО типа', len(D), src)} "
        f"({num(100 * N / len(D), '.1f', t, 'состав', 'доля от 2004 МО, %', len(D), src)}% от 2004); регионов: "
        f"{num(n_reg, 'd', t, 'состав', 'регионов', N, src)} {R_CALC}.", "")
    tbl(f"Регионы с наибольшим числом МО типа (топ-{cfg(TOP_REGIONS)})", top)
    rows = []
    for lab, col, fmt in [("население, тыс. жителей (1 января 2023)", "pop", ".1f"), ("доля старших, %", "older", ".1f"),
                          ("зарплата относительно медианы региона", "wage_rel", ".2f"),
                          ("market_access", "market_access", ".1f"), ("работников на жителя", "wpp", ".2f")]:
        v, n = med(sub[col] / (1000 if col == "pop" else 1))
        rows.append({"показатель": lab, "медиана по МО типа": cell(v, fmt, t, "состав", f"медиана: {lab}", n, src_r), "n": str(n)})
    tbl("Состав: медианы по МО типа", pd.DataFrame(rows))
    rows = []
    for c in CATS:
        v, n = med(sub[c])
        rows.append({"категория": c, "медиана доли, %": cell(v, ".1f", t, "расходы", f"медиана {c}, %", n, src_c),
                     "n": str(n), "профиль, σ": cell(z.loc[t, c], "+.2f", t, "расходы", f"профиль {c}, σ", N, src_c)})
    add("**Расходы**", "")
    tbl("Доли расходов, декабрь 2024, и профиль в σ (среднее z-score типа)", pd.DataFrame(rows))
    sig = step10b.describe_profile(z.loc[t].rename(lambda c: f"share_{c}"))
    add(f"- Подпись профиля (как в 10b): {mk(sig)} {R_CALC}.", "")
    rows = []
    for c in CATS:
        rows.append({"категория": c, "σ по долям": cell(z.loc[t, c], "+.2f", t, "признаки профиля", f"{c}: σ по долям", N, src_c),
                     "σ по log(доля / «прочее»)": cell(za.loc[t, c], "+.2f", t, "признаки профиля", f"{c}: σ по лог-отношению", N, src_c),
                     "пометка": feature_mark(z, za, t, c) or "—"})
    tbl(f"Признаки профиля: σ по долям и по log(доля / «прочее»); пометка у признаков с |σ| по долям ≥ "
        f"{cfg(step10b.PROFILE_Z_THRESHOLD)} — {EXPL}", pd.DataFrame(rows))
    for c, m in border_features(z, za, t):
        R(t, "признаки профиля", f"{c}: {PROFILE_BORDER_LABEL}, меньшее |σ|", m, N, src_c)
    add(f"- Имя типа: {full_type_name(D, z, za, t)} {R_CALC}.", "")
    rows = []
    src_x = "kmeans_labels_final.parquet; rosstat_mo_2023_2025.parquet; rosstat_retail_2023_2024.parquet"
    for lab, col, fmt in [("доля работников в сельском хозяйстве (A), %", "okved_A", ".1f"),
                          ("доля работников в обрабатывающей промышленности (C), %", "okved_C", ".1f"),
                          ("доля работников в торговле (G), %", "okved_G", ".1f"),
                          ("доля работников в здравоохранении (Q), %", "okved_Q", ".1f"),
                          ("floor_pc, м² на 1000 жителей", "floor_pc", ".0f"), ("seats_pc, мест на 1000 жителей", "seats_pc", ".1f")]:
        v, n = med(sub[col])
        rows.append({"показатель": lab, "медиана по МО типа": cell(v, fmt, t, "росстат", f"медиана: {lab}", n, src_x), "n": str(n)})
    add("**Росстат**", "")
    tbl("Отрасли, розница и общепит: медианы по МО типа", pd.DataFrame(rows))
    ps_mean, conf = reliability(D, ps, matching, r19b, t)
    return ps_mean, conf


def type0(D, sup_res) -> None:
    add("**Сравнение с МО того же размера:** таблица 1.4, строки типа 0; сравнительная таблица малых типов — раздел 2.0.", "",
        H_TEXT[0].format(
            g10=num(sup_res["g10_within"], "+.2f", 0, "гипотеза 0", "Г10 внутри регионов: ρ", None, RESULTS_SUPPLY_PATH.name),
            partial=num(sup_res["partial_within"], "+.2f", 0, "гипотеза 0", "частная корреляция Г10 внутри регионов: ρ", None, RESULTS_SUPPLY_PATH.name)), "")


def transport_boot(D) -> list:
    """Бутстреп разности медиан доли транспорта для соседних групп размера: МО выбираются с возвращением внутри
    каждой из двух групп, BOOT_B повторов, один генератор SEED_23 для пар по порядку; 95% перцентильный интервал."""
    rng = np.random.default_rng(SEED_23)
    labs = size_labels(SIZE_BINS_TH)
    out = []
    for a, b in zip(labs, labs[1:]):
        xa = D.loc[(D["size_group"] == a) & D["Транспорт"].notna(), "Транспорт"].to_numpy(float)
        xb = D.loc[(D["size_group"] == b) & D["Транспорт"].notna(), "Транспорт"].to_numpy(float)
        ma = np.median(xa[rng.integers(0, len(xa), size=(BOOT_B, len(xa)))], axis=1)
        mb = np.median(xb[rng.integers(0, len(xb), size=(BOOT_B, len(xb)))], axis=1)
        lo, hi = np.percentile(mb - ma, CI_Q)
        out.append({"a": a, "b": b, "diff": float(np.median(xb) - np.median(xa)), "lo": float(lo), "hi": float(hi),
                    "n_a": len(xa), "n_b": len(xb), "sig": bool(lo > 0 or hi < 0)})
    return out


def type1(D, transport) -> None:
    src = "kmeans_labels_final.parquet; category_shares.parquet; rosstat_mo_2023_2025.parquet"
    labs = size_labels(SIZE_BINS_TH)
    rows = []
    for g in labs:
        s = D[D["size_group"] == g]
        v_all, n_all = med(s["Транспорт"])
        by_type = {tt: med(s.loc[s["type"] == tt, "Транспорт"]) for tt in range(FINAL_K)}
        v1, n1 = by_type[1]
        ranked = sorted([tt for tt in by_type if by_type[tt][1] >= MIN_MO_REGION], key=lambda tt: -by_type[tt][0])
        place = ranked.index(1) + 1 if 1 in ranked else np.nan
        rows.append({"группа, тыс. жителей": g,
                     "все МО: медиана транспорта, %": cell(v_all, ".1f", "все", "тип 1: транспорт по размеру", f"{g}: все МО", n_all, src),
                     "все МО: n": str(n_all),
                     "тип 1: медиана, %": cell(v1, ".1f", 1, "тип 1: транспорт по размеру", f"{g}: тип 1", n1, src),
                     "тип 1: n": str(n1),
                     "место типа 1 среди типов": (cell(place, "d", 1, "тип 1: транспорт по размеру", f"{g}: место типа 1 (1 = наибольшая медиана)", len(ranked), src)
                                                  + f" из {len(ranked)}") if ranked else "—"})
    add("**Транспорт по группам размера и место типа**", "")
    tbl(f"Медиана доли транспорта; место — среди типов с числом МО в группе не менее {cfg(MIN_MO_REGION)}", pd.DataFrame(rows))
    labs2 = size_labels(SIZE_BINS_TH)
    src_b = "category_shares.parquet; rosstat_mo_2023_2025.parquet"
    boot = transport_boot(D)
    for i, bt in enumerate(boot):
        if not np.isclose(bt["diff"], transport[i + 1] - transport[i]):
            raise SystemExit(f"STOP: разность медиан в бутстрепе не совпадает с таблицей 1.1: {bt}")
    got = [f"{bt['a']} → {bt['b']}" for bt in boot if bt["sig"]]
    expected = list(PORTRAIT_CONTROLS["transport_pairs_ci"])
    print(f"пары с интервалом без нуля: шаг 23 {got}; ожидание {expected}")
    for bt in boot:
        print(f"  {bt['a']} → {bt['b']}: {bt['diff']:+.3f} [{bt['lo']:+.3f}; {bt['hi']:+.3f}], n = {bt['n_a']}, {bt['n_b']}")
    if got != expected:
        raise SystemExit(f"STOP: пары с интервалом без нуля не совпадают с ожиданием: шаг 23 {got}, ожидание {expected}")
    add(H_TEXT[1].format(pairs=", ".join(f"{cfg(bt['a'])} → {cfg(bt['b'])}" for bt in boot if bt["sig"])), "")
    steps = ", ".join(f"{cfg(bt['a'])} → {cfg(bt['b'])}: "
                      + num(bt["diff"], "+.2f", 1, "тип 1: транспорт по размеру", f"изменение медианы {bt['a']} → {bt['b']}, п.п.", None, src_b)
                      + " [" + num(bt["lo"], "+.2f", 1, "тип 1: транспорт по размеру", f"изменение медианы {bt['a']} → {bt['b']}: нижняя граница 95% интервала", bt["n_a"] + bt["n_b"], src_b)
                      + "; " + num(bt["hi"], "+.2f", 1, "тип 1: транспорт по размеру", f"изменение медианы {bt['a']} → {bt['b']}: верхняя граница 95% интервала", bt["n_a"] + bt["n_b"], src_b)
                      + "]" for bt in boot)
    add(f"- Изменение медианы доли транспорта между соседними группами размера, п.п., в скобках "
        f"{cfg(f'{CI_Q[1] - CI_Q[0]:g}')}% интервал бутстрепа по МО "
        f"({cfg(BOOT_B)} повторов): {steps} {R_CALC}.", "")


def age_health_block(D) -> dict:
    src = "kmeans_labels_final.parquet; category_shares.parquet; rosstat_mo_2023_2025.parquet"
    s = D[D["type"].isin(SMALL_TYPES) & D["older"].notna()]
    slope, icpt = np.polyfit(s["older"].to_numpy(), s["Здоровье"].to_numpy(), 1)
    out = {"slope": float(slope)}
    med_h = {t: med(s.loc[s["type"] == t, "Здоровье"]) for t in SMALL_TYPES}
    med_o = {t: med(s.loc[s["type"] == t, "older"]) for t in SMALL_TYPES}
    rows = []
    for a, b in [(2, 0), (2, 5)]:
        act = med_h[a][0] - med_h[b][0]
        exp = slope * (med_o[a][0] - med_o[b][0])
        blk = f"тип 2: возраст и здоровье, тип {a} минус тип {b}"
        rows.append({"сравнение": f"тип {a} минус тип {b}",
                     "ожидаемая по возрасту разность медиан здоровья, п.п.": cell(exp, "+.2f", 2, blk, "ожидаемая по возрасту разность", len(s), src),
                     "фактическая разность, п.п.": cell(act, "+.2f", 2, blk, "фактическая разность", len(s), src)})
        out[(a, b)] = (exp, act)
    add("Сравнительная таблица малых типов — раздел 2.0.", "", "**Возраст и здоровье**", "",
        f"МНК доли здравоохранения (п.п.) на долю старших (п.п.) по МО малых типов с данными о возрасте: наклон "
        f"{num(slope, '+.3f', 2, 'тип 2: возраст и здоровье', 'наклон МНК, п.п. здоровья на п.п. старших', len(s), src)}, "
        f"n = {num(len(s), 'd', 2, 'тип 2: возраст и здоровье', 'n МНК', len(s), src)} — {EXPL} {R_CALC}.", "")
    tbl(f"Ожидаемая по возрасту (наклон × разность медиан доли старших) и фактическая разность медиан здоровья — {EXPL}",
        pd.DataFrame(rows))
    rr = []
    for lab, sel in [(f"тип {t}", D["type"] == t) for t in SMALL_TYPES] + [("малые типы вместе", D["type"].isin(SMALL_TYPES))]:
        s2 = D[sel]
        r1, n1 = rho(s2["Здоровье"], s2["older"])
        r2, n2 = rho(s2["Здоровье"], s2["okved_Q"])
        rr.append({"группа": lab,
                   "ρ(здоровье, доля старших)": cell(r1, "+.2f", 2, "тип 2: возраст и здоровье", f"{lab}: ρ(здоровье, старшие)", n1, src),
                   "n": str(n1),
                   "ρ(здоровье, доля занятых в здравоохранении)": cell(r2, "+.2f", 2, "тип 2: возраст и здоровье", f"{lab}: ρ(здоровье, занятые Q)", n2, src),
                   "n ": str(n2)})
    tbl(f"ρ Спирмена по типам и по малым типам — {EXPL}", pd.DataFrame(rr))
    return out


def type2(D, ah, r19a) -> None:
    exp, act = ah[(2, 0)]
    g7 = r19a[(r19a["row_type"] == "g7_delta") & (r19a["variant"] == "основной") & r19a["delta"].notna()]
    if len(g7) != 2 or (g7["delta"] > 0).any():
        raise SystemExit(f"STOP: Г7 (19a): ожидалось Δ ≤ 0 на обоих уровнях, получено\n{g7[['level', 'delta']]}")
    rows = [{"уровень": r["level"], "η²_H доли здравоохранения": fnum(r["eta2_health"], ".3f"),
             "η²_H остатка после учёта возраста": fnum(r["eta2_resid"], ".3f"), "Δ": fnum(r["delta"], "+.3f")}
            for _, r in g7.iterrows()]
    for _, r in g7.iterrows():
        for m, v in [("η²_H доли", r["eta2_health"]), ("η²_H остатка", r["eta2_resid"]), ("Δ", r["delta"])]:
            R(2, "тип 2: Г7", f"{r['level']}: {m}", v, int(r["n"]), RESULTS_19A_PATH.name)
    tbl("Г7: η²_H доли здравоохранения и остатка после учёта возраста (цитата)", pd.DataFrame(rows), "(результат шага 19a, Г7)")
    add(H_TEXT[2].format(exp=num(exp, "+.2f", 2, "гипотеза 2", "ожидаемая по возрасту разность (тип 2 − тип 0), п.п.", None, "kmeans_labels_final.parquet; category_shares.parquet; rosstat_mo_2023_2025.parquet"),
                         act=num(act, "+.2f", 2, "гипотеза 2", "фактическая разность (тип 2 − тип 0), п.п.", None, "kmeans_labels_final.parquet; category_shares.parquet; rosstat_mo_2023_2025.parquet")), "")


def type3(D) -> dict:
    src = "kmeans_labels_final.parquet; category_shares.parquet; territories.parquet"
    sub = D[D["type"] == 3]
    cap = list(CAPITAL_REGIONS)
    groups = [(cap[0], sub["region"] == cap[0]), (cap[1], sub["region"] == cap[1]),
              (MOSCOW_OBLAST, sub["region"] == MOSCOW_OBLAST), ("прочие", ~sub["region"].isin(cap + [MOSCOW_OBLAST]))]
    rows, counts = [], {}
    for lab, mask in groups:
        s = sub[mask]
        counts[lab] = len(s)
        row = {"группа МО типа 3": lab, "n": cell(len(s), "d", 3, "тип 3: состав по группам", f"{lab}: n", len(s), src)}
        for c in CATS_11:
            v, n = med(s[c])
            row[f"{c}, %"] = cell(v, ".1f", 3, "тип 3: состав по группам", f"{lab}: медиана {c}", n, src)
        rows.append(row)
    add("**Состав по группам**", "")
    tbl("МО типа 3 по группам: число МО и медианы пяти долей", pd.DataFrame(rows))
    rest = sub[~sub["region"].isin(cap)]
    allm = D
    rows = []
    for c in ["Транспорт", "Общепит"]:
        vr, nr = med(rest[c])
        va, na = med(allm[c])
        if not vr > va:
            raise SystemExit(f"STOP: медиана {c} в остальных МО типа 3 ({vr:.2f}) не выше медианы по всем МО ({va:.2f})")
        rows.append({"категория": c,
                     "остальные МО типа 3 (кроме Москвы и Санкт-Петербурга), %": cell(vr, ".1f", 3, "тип 3: остальные МО", f"{c}: остальные МО типа 3", nr, src),
                     "n": str(nr),
                     "все 2004 МО, %": cell(va, ".1f", "все", "тип 3: остальные МО", f"{c}: все МО", na, src), "n ": str(na)})
    tbl("Транспорт и общепит: остальные МО типа 3 и все МО", pd.DataFrame(rows))
    src2 = "kmeans_labels_final.parquet; category_shares.parquet; rosstat_mo_2023_2025.parquet; territories.parquet"
    rr = []
    for lab, s in [(f"{cap[0]} (все МО региона)", D[D["region"] == cap[0]]), ("тип 3 (все МО типа)", sub)]:
        r_, n_ = rho(s["Продовольствие"], s["wage_rel"])
        rr.append({"группа": lab, "ρ(продовольствие, зарплата относительно региона)": cell(r_, "+.2f", 3, "тип 3: продовольствие и зарплата", lab, n_, src2), "n": str(n_)})
    tbl(f"ρ Спирмена — {EXPL}", pd.DataFrame(rr))
    rows = []
    tot_w, tot_n = int((D["wpp"] > 1).sum()), int(D["wpp"].notna().sum())
    for lab, s in [(cap[0], D[D["region"] == cap[0]]), (cap[1], D[D["region"] == cap[1]])]:
        n_ = int(s["wpp"].notna().sum())
        k_ = int((s["wpp"] > 1).sum())
        rows.append({"регион": lab, "МО с данными": cell(n_, "d", 3, "тип 3: работников больше жителей", f"{lab}: МО с данными", n_, src2),
                     "из них работников больше жителей": cell(k_, "d", 3, "тип 3: работников больше жителей", f"{lab}: работников больше жителей", n_, src2)})
    rows.append({"регион": "все МО", "МО с данными": cell(tot_n, "d", "все", "тип 3: работников больше жителей", "все МО: с данными", tot_n, src2),
                 "из них работников больше жителей": cell(tot_w, "d", "все", "тип 3: работников больше жителей", "все МО: работников больше жителей", tot_n, src2)})
    tbl("Число МО, где численность работников (организации без малого бизнеса, по месту нахождения) больше числа жителей", pd.DataFrame(rows))
    return counts


def salary_block(sources) -> None:
    r = sources[sources["id"] == "S12"].iloc[0]
    add("**Зарплата: что измеряется**", "",
        "Зарплата и численность работников в данных шага 17 — по организациям без субъектов малого предпринимательства "
        "и по месту нахождения организаций и подразделений. Из указаний к отчётности в пересказе: "
        f"{mk(r['fact'])} (источник S12).", "",
        "Для типа 3 это значит, что зарплата относится к организациям, расположенным в МО, а не к жителям МО "
        "(гипотеза, не проверена).", "")


def type4(D, sources) -> dict:
    src = "kmeans_labels_final.parquet; category_shares.parquet; federal_districts.csv; rosstat_mo_2023_2025.parquet"
    sub = D[D["type"] == 4]
    fe = sub["fd"] == FAR_EAST_DISTRICT
    n_fe = int(fe.sum())
    add("**Дальний Восток**", "",
        f"- МО типа 4 на Дальнем Востоке: {num(n_fe, 'd', 4, 'тип 4: Дальний Восток', 'МО типа 4 в ДФО', len(sub), src)} из "
        f"{num(len(sub), 'd', 4, 'тип 4: Дальний Восток', 'МО типа 4', len(sub), src)}; вне Дальнего Востока — "
        f"{num(len(sub) - n_fe, 'd', 4, 'тип 4: Дальний Восток', 'МО типа 4 вне ДФО', len(sub), src)} {R_CALC}.", "")
    rows = []
    groups = [("все МО", D)] + [(f"тип {t}", D[D["type"] == t]) for t in range(FINAL_K)]
    skipped = []
    for lab, s in groups:
        a, b = s[s["fd"] == FAR_EAST_DISTRICT], s[s["fd"] != FAR_EAST_DISTRICT]
        if len(a) < MIN_MO_REGION or len(b) < MIN_MO_REGION:
            skipped.append(lab)
            continue
        row = {"группа": lab}
        for nm, g in [("ДФО", a), ("вне ДФО", b)]:
            row[f"{nm}: n"] = cell(len(g), "d", lab.replace("тип ", "") if lab != "все МО" else "все", "тип 4: ДФО и вне", f"{lab}: {nm}: n", len(g), src)
            for c, cl in [("Маркетплейсы", "маркетплейсы, %"), ("Здоровье", "здоровье, %"), ("older", "старшие, %")]:
                v, n = med(g[c])
                row[f"{nm}: {cl}"] = cell(v, ".1f", lab.replace("тип ", "") if lab != "все МО" else "все", "тип 4: ДФО и вне", f"{lab}: {nm}: медиана {cl}", n, src)
            row[f"{nm}: n старших"] = str(int(g["older"].notna().sum()))
        rows.append(row)
    add(f"Пропущены группы с числом МО меньше {cfg(MIN_MO_REGION)} в одной из частей: {', '.join(skipped) if skipped else 'нет'}.", "")
    tbl(f"Медианы долей маркетплейсов, здравоохранения и старших: Дальний Восток и вне его — {EXPL}", pd.DataFrame(rows))
    src_m = "kmeans_labels_final.parquet; category_shares.parquet; market_access.parquet"
    r_, n_ = rho(sub["Маркетплейсы"], sub["market_access"])
    add(f"- ρ(доля маркетплейсов, log market_access) внутри типа 4: {num(r_, '+.2f', 4, 'тип 4: маркетплейсы и market_access', 'ρ внутри типа', n_, src_m)}, "
        f"n = {num(n_, 'd', 4, 'тип 4: маркетплейсы и market_access', 'n', n_, src_m)} — {EXPL} {R_CALC}.", "")
    y = D[D["region"] == YAKUTIA].copy()
    y["yg"] = size_group(y["pop"], YAKUTIA_SIZE_BINS_TH)
    rows, ymed, yn = [], [], []
    src_y = "category_shares.parquet; rosstat_mo_2023_2025.parquet; territories.parquet"
    for g in size_labels(YAKUTIA_SIZE_BINS_TH):
        v, n = med(y.loc[y["yg"] == g, "Маркетплейсы"])
        ymed.append(v)
        yn.append(n)
        rows.append({"группа, тыс. жителей": g, "медиана доли маркетплейсов, %": cell(v, ".1f", 4, "тип 4: Якутия", f"группа {g}: медиана маркетплейсов", n, src_y), "n": str(n)})
    tbl(f"Якутия (все МО региона), группы размера — {EXPL}", pd.DataFrame(rows))
    ry, ny = rho(y["Маркетплейсы"], y["log_pop"])
    add(f"- ρ(доля маркетплейсов, log населения) внутри Якутии: {num(ry, '+.2f', 4, 'тип 4: Якутия', 'ρ внутри Якутии', ny, src_y)}, "
        f"n = {num(ny, 'd', 4, 'тип 4: Якутия', 'n Якутия', ny, src_y)} — {EXPL} {R_CALC}.", "")
    reg = sub["region"].value_counts()
    top = pd.DataFrame({"регион": reg.index, "n": reg.to_numpy()}).sort_values(["n", "регион"], ascending=[False, True]).head(TOP_REGIONS_TYPE4)
    rows = []
    for r_name, n_ in zip(top["регион"], top["n"]):
        s = sub[sub["region"] == r_name]
        row = {"регион": r_name, "n": cell(n_, "d", 4, "тип 4: регионы", f"{r_name}: n", len(sub), src)}
        for c, cl, fmt in [("Маркетплейсы", "маркетплейсы, %", ".1f"), ("Здоровье", "здоровье, %", ".1f"), ("market_access", "market_access", ".1f")]:
            v, n = med(s[c])
            row[f"медиана: {cl}"] = cell(v, fmt, 4, "тип 4: регионы", f"{r_name}: медиана {cl}", n, src_m)
        rows.append(row)
    tbl(f"Регионы типа 4 (топ-{cfg(TOP_REGIONS_TYPE4)}): число МО типа и медианы по МО типа", pd.DataFrame(rows))
    return {"yakutia_medians": ymed, "yakutia_n": yn, "rho_yakutia": ry, "n_fe": n_fe}


def type4_text(D, z, za, r19a, between, within_pop, ry, sources) -> None:
    src = "kmeans_labels_final.parquet; category_shares.parquet; market_access.parquet; rosstat_mo_2023_2025.parquet"
    if not (between[("market_access", "между регионами")] > 0 and within_pop < 0 and ry > 0):
        raise SystemExit("STOP: знаки ρ не соответствуют формулировке гипотезы 4 (между регионами с market_access +, "
                         f"внутри регионов с размером МО −, Якутия +): {between[('market_access', 'между регионами')]:.2f}, {within_pop:.2f}, {ry:.2f}")
    sub, allm = D[D["type"] == 4], D
    if not (med(sub["older"])[0] < med(allm["older"])[0]):
        raise SystemExit("STOP: медиана доли старших типа 4 не ниже медианы по всем МО (формулировка «моложе население»)")
    rows = [{"показатель": "доля старших, % (медиана по МО)", "тип 4": cell(med(sub["older"])[0], ".1f", 4, "тип 4: моложе население", "тип 4: доля старших", med(sub["older"])[1], src),
             "n": str(med(sub["older"])[1]), "все МО": cell(med(allm["older"])[0], ".1f", "все", "тип 4: моложе население", "все МО: доля старших", med(allm["older"])[1], src),
             "n ": str(med(allm["older"])[1])}]
    tbl("Доля старших: тип 4 и все МО", pd.DataFrame(rows))
    # n выборок уже записаны в parquet (блоки «1.3 маркетплейсы» и «тип 4: Якутия»), здесь только в тексте
    b_r, b_nreg, b_nmo = rho_between(D, "Маркетплейсы", "market_access")
    w_r, w_nreg, w_nmo = rho_within(D, "Маркетплейсы", "log_pop")
    y_r, y_n = rho(D.loc[D["region"] == YAKUTIA, "Маркетплейсы"], D.loc[D["region"] == YAKUTIA, "log_pop"])
    if not (np.isclose(b_r, between[("market_access", "между регионами")]) and np.isclose(w_r, within_pop) and np.isclose(y_r, ry)):
        raise SystemExit("STOP: пересчёт трёх ρ типа 4 не совпадает со значениями в тексте")
    add(f"- ρ между регионами (доля маркетплейсов ~ market_access; все МО выборки, ρ по медианам регионов с числом МО не менее "
        f"{cfg(MIN_MO_REGION)}: регионов {mk(str(b_nreg))}, МО "
        f"{mk(str(b_nmo))}): "
        f"{num(between[('market_access', 'между регионами')], '+.2f', 4, 'гипотеза 4', 'ρ между регионами с market_access', None, src)}; "
        f"внутри регионов (доля маркетплейсов ~ log населения; все МО выборки с населением 2023, ранг внутри региона, регионы "
        f"с числом МО не менее {cfg(MIN_MO_REGION)}: регионов {mk(str(w_nreg))}, МО "
        f"{mk(str(w_nmo))}): "
        f"{num(within_pop, '+.2f', 4, 'гипотеза 4', 'ρ внутри регионов с log населения', None, src)}; "
        f"в Якутии (доля маркетплейсов ~ log населения; все МО региона, не только тип 4, n = "
        f"{mk(str(y_n))}): "
        f"{num(ry, '+.2f', 4, 'гипотеза 4', 'ρ Якутия с log населения', None, src)} — {EXPL} {R_CALC}.", "")
    s5 = sources[sources["id"] == "S5"].iloc[0]["fact"]
    m = re.search(r"в Якутске (\d+) пунктов выдачи", s5)
    if not m:
        raise SystemExit("STOP: в fact источника S5 нет числа пунктов выдачи в Якутске")
    R(4, "гипотеза 4", "пунктов выдачи в Якутске (S5, fact)", int(m.group(1)), None, SOURCES_PATH.name)
    h_s, h_a = float(z.loc[4, "Здоровье"]), float(za.loc[4, "Здоровье"])
    if not (h_a < 0 and h_s < 0 and abs(h_a) > abs(h_s)):
        raise SystemExit(f"STOP: здоровье типа 4 по лог-отношению ({h_a:+.2f}σ) не ниже сильнее, чем по долям ({h_s:+.2f}σ)")
    add(H_TEXT[4].format(pickup=mk(m.group(1)),
                         alr=num(abs(h_a), ".1f", 4, "гипотеза 4", "здоровье: |σ| по лог-отношению", None,
                                 "kmeans_labels_final.parquet; category_shares.parquet")), "")


def type5(verdicts) -> None:
    g3 = verdicts[(verdicts["row_type"] == "level") & (verdicts["hypothesis"] == "Г3")].set_index("level")
    if set(g3.index) != {"общий", "внутри регионов"}:
        raise SystemExit("STOP: в hypothesis_verdicts нет строк Г3 на обоих уровнях")
    add("**Г3 (разведочная), по всем МО**", "",
        f"- ρ(доля продовольствия, зарплата относительно региона): общий уровень "
        f"{num(g3.loc['общий', 'obs'], '+.2f', 5, 'Г3 из 19c', 'ρ, общий уровень', None, VERDICTS_PATH.name)}, внутри регионов "
        f"{num(g3.loc['внутри регионов', 'obs'], '+.2f', 5, 'Г3 из 19c', 'ρ, внутри регионов', None, VERDICTS_PATH.name)} "
        "(результат шага 19c, Г3, разведочная).", "")
    if not (g3.loc["общий", "obs"] < 0 and g3.loc["внутри регионов", "obs"] < 0):
        raise SystemExit("STOP: Г3 не отрицательна на обоих уровнях — формулировка текста типа 5 неверна")
    add(H_TEXT[5].format(
        g3_all=num(g3.loc["общий", "obs"], "+.2f", 5, "Г3 из 19c", "ρ, общий уровень", None, VERDICTS_PATH.name),
        g3_in=num(g3.loc["внутри регионов", "obs"], "+.2f", 5, "Г3 из 19c", "ρ, внутри регионов", None, VERDICTS_PATH.name)), "")


def cannot_claim(D) -> None:
    src = "kmeans_labels_final.parquet; territories.parquet"
    regs = {t: int(D.loc[D["type"] == t, "region"].nunique()) for t in [0, 1, 2, 5]}
    add("## 3. Что нельзя утверждать", "",
        "- Нельзя утверждать, что различия расходов объясняются доходами или предпочтениями жителей.",
        "- Нельзя утверждать, что низкая доля маркетплейсов означает мало онлайн-покупок: специализированные "
        "интернет-магазины идут в другие категории, перечень платформ не раскрыт (источник S0).",
        "- Нельзя утверждать, что тип определяется регионом: типы 0, 1, 2, 5 разбросаны по десяткам регионов ("
        + ", ".join(num(regs[t], "d", t, "что нельзя утверждать", "регионов типа", None, src) for t in [0, 1, 2, 5])
        + f" регионов соответственно) {R_CALC}.",
        "- Нельзя утверждать, что зарплата типа 3 отражает доходы жителей.",
        "- Нельзя утверждать, что профиль типа не зависит от способа записи долей: признаки «зависят от записи» нельзя "
        "считать устойчивыми различиями типов.", "")


def limitations(D, cover_all: float, cover_dec: float) -> None:
    src = "territories.parquet; kmeans_labels_final.parquet; category_shares.parquet"
    spb = D[D["region"] == CAPITAL_REGIONS[1]]
    if spb["older"].notna().any() or spb["floor_pc"].notna().any():
        raise SystemExit("STOP: у Санкт-Петербурга есть данные о возрасте или рознице — формулировка ограничения неверна")
    n_regions = int(D["region"].nunique())
    add("## 5. Ограничения", "",
        "- Расходы — оценка модели СберИндекса по транзакциям жителей, безналичные, методика не раскрыта (источник S0).",
        f"- Выборка отобрана по порогу качества оценки (источник S0), в ней {num(n_regions, 'd', 'все', 'ограничения', 'регионов в выборке', len(D), src)} региона {R_CALC}.",
        f"- Покрытие безналичных расходов пятью категориями: в среднем {num(cover_all, '.0f', 'все', 'ограничения', 'средняя сумма пяти долей за 24 месяца, %', None, src)}% "
        f"за 24 месяца, {num(cover_dec, '.0f', 'все', 'ограничения', 'средняя сумма пяти долей, декабрь 2024, %', len(D), src)}% в декабре 2024 {R_CALC}.",
        "- Зарплата и численность работников считаются по организациям без малого бизнеса по месту нахождения организаций "
        "и подразделений (источник S12).",
        "- Население 2023 пусто у " + nopop_phrase(D, "ограничения") + ". Не аномальные МО без населения входят в медианы и ρ "
        "зарплаты относительно региона и долей отраслей, но не входят в группы размера, в расчёты с долей старших, floor_pc, "
        f"seats_pc и работниками на жителя {R_CALC}.",
        "- Типология построена по долям от всех безналичных расходов; доля «прочего» различается по типам (результат шага 25), "
        "поэтому часть различий по транспорту и здравоохранению зависит от записи долей.",
        f"- Санкт-Петербурга нет в данных возраста и розницы (число МО с данными среди {num(len(spb), 'd', 'все', 'ограничения', 'МО Санкт-Петербурга', len(spb), src)} МО региона равно нулю) {R_CALC}.",
        f"- Шаги описательных расчётов не предрегистрированы: такие расчёты помечены «{EXPL}».", "")


def sources_section(sources) -> None:
    add("## 4. Источники", "", f"{TABLE_EXEMPT_PREFIX} из `data/external/portrait_sources.csv`; числа из источников берутся только из колонки fact.", "",
        sources.to_markdown(index=False, disable_numparse=True), "")


# ─────────────────────────── контроли, lint ───────────────────────────
def controls_table(my: dict) -> pd.DataFrame:
    C = PORTRAIT_CONTROLS
    tm, tc = ZERO_TOL["median_pp"], ZERO_TOL["corr"]
    rows = []

    def add_row(name, mine, ctrl, tol):
        mine_l = list(np.atleast_1d(np.asarray(mine, dtype=float)))
        ctrl_l = list(np.atleast_1d(np.asarray(ctrl, dtype=float)))
        ok = len(mine_l) == len(ctrl_l) and all(abs(a - b) <= tol + 1e-9 for a, b in zip(mine_l, ctrl_l))
        rows.append({"контроль": name, "значение шага 23": ", ".join(f"{x:.2f}" for x in mine_l),
                     "контрольное значение": ", ".join(f"{x:.2f}" for x in ctrl_l), "допуск": f"{tol:g}", "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})

    add_row("размеры типов 0–5", my["type_sizes"], C["type_sizes"], 0)
    add_row("регионов в типах 0–5", my["type_regions"], C["type_regions"], 0)
    add_row("МО типа 4 на Дальнем Востоке", my["type4_far_east"], C["type4_far_east"], 0)
    add_row("тип 3: Москва, Санкт-Петербург, Московская область, прочие", list(my["type3_groups"].values()), list(C["type3_groups"].values()), 0)
    add_row("PS по типам 0–5 (среднее доля пар)", my["ps"], C["ps"], 0.005)
    add_row("уверенных сопоставлений из 23, типы 0–5", my["confident"], C["confident"], 0)
    add_row("транспорт по группам размера, медиана %", my["transport_by_size"], C["transport_by_size"], tm)
    add_row("общепит по группам размера, медиана %", my["catering_by_size"], C["catering_by_size"], tm)
    add_row("ρ(общепит, log населения), все МО", my["rho_catering_logpop"], C["rho_catering_logpop"], tc)
    add_row("ρ(маркетплейсы, market_access), между регионами", my["rho_market_ma_between"], C["rho_market_ma_between"], tc)
    add_row("ρ(маркетплейсы, market_access), внутри регионов", my["rho_market_ma_within"], C["rho_market_ma_within"], tc)
    add_row("Якутия по группам размера: медианы маркетплейсов, %", my["yakutia_medians"], C["yakutia_medians"], tm)
    add_row("Якутия по группам размера: n", my["yakutia_n"], C["yakutia_n"], 0)
    add_row("здоровье тип 2 минус тип 0: факт, п.п.", my["health_gap_actual"], C["health_gap_actual"], tm)
    add_row("здоровье тип 2 минус тип 0: ожидаемая по возрасту, п.п.", my["health_gap_expected"], C["health_gap_expected"], tm)
    return pd.DataFrame(rows)


def controls_table_2(my: dict) -> pd.DataFrame:
    """Дополнительные контроли шага 23 (устойчивость признаков профиля)."""
    C, T = PORTRAIT_CONTROLS, ZERO_TOL
    rows = []

    def add_row(name, mine, ctrl, tol, fmt=".2f"):
        ok = abs(float(mine) - float(ctrl)) <= tol + 1e-9
        rows.append({"контроль": name, "значение шага 23": format(float(mine), fmt), "контрольное значение": format(float(ctrl), fmt),
                     "допуск": f"{tol:g}", "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})

    for t, d_ in C["sigma_alr"].items():
        for c, v in d_.items():
            add_row(f"σ по лог-отношению, тип {t}: {c}", my["za"].loc[int(t), c], v, T["sigma"])
    for lab, v in C["rho_logpop_rest"].items():
        add_row(f"ρ({lab}, log население)", my["rho_logpop_rest"][lab], v, T["corr"])
    for t in range(FINAL_K):
        add_row(f"отношение медианы населения, тип {t}", my["pop_ratio"][t], C["pop_ratio"][t], T["ratio"])
    for t in range(FINAL_K):
        add_row(f"отношение медианы market_access, тип {t}", my["ma_ratio"][t], C["ma_ratio"][t], T["ratio"])
    for t in range(FINAL_K):
        ok = my["names"][t] == C["type_names"][t]
        rows.append({"контроль": f"имя типа {t}", "значение шага 23": my["names"][t], "контрольное значение": C["type_names"][t],
                     "допуск": "точно", "статус": "совпало" if ok else "РАСХОЖДЕНИЕ"})
    return pd.DataFrame(rows)


def strip_marks(text: str) -> str:
    for ch in (MK_L, MK_R, CFG_L, CFG_R):
        text = text.replace(ch, "")
    return text


def lint(lines: list) -> list:
    errors = []
    tok = re.compile(r"\d+(?:\.\d+)?")
    for i, line in enumerate(lines):
        if line.startswith("|") or line.startswith("#"):
            continue
        refs = list(REF_RE.finditer(line))
        if MK_L in line:
            if not refs:
                errors.append((line, "число без ссылки"))
                continue
            if MK_L in line[refs[-1].end():]:
                errors.append((line, "число после последней ссылки"))
            prev = 0
            for m in refs:
                if not line[prev:m.start()].strip(" ,;.:—-"):
                    errors.append((line, "две ссылки подряд"))
                prev = m.end()
        rest = REF_RE.sub("", line)
        rest = re.sub(f"{MK_L}.*?{MK_R}", "", rest)
        rest = re.sub(f"{CFG_L}.*?{CFG_R}", "", rest)
        rest = re.sub(r"`[^`]*`", "", rest)
        rest = re.sub(r"(?:таблиц[аы]|раздел[е]?)\s+\d+\.\d+", "", rest)     # ссылки на разделы и таблицы отчёта
        for t in tok.findall(rest):
            v = float(t)
            if not (v == int(v) and (int(v) <= ALLOWED_INT_MAX or int(v) in ALLOWED_YEARS)):
                errors.append((line, f"число {t} в шаблоне"))
    for i, line in enumerate(lines):
        if line.startswith("|") and (i == 0 or not lines[i - 1].startswith("|")):
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            cap = lines[j].rstrip() if j >= 0 else ""
            if cap.startswith(TABLE_EXEMPT_PREFIX):
                continue
            m = list(REF_RE.finditer(cap))
            if not m or m[-1].end() != len(cap):
                errors.append((cap, "таблица без ссылки в подписи"))
    text = strip_marks("\n".join(lines))
    for w in FORBIDDEN:
        if re.search(w, text, flags=re.I):
            errors.append((w, "запрещённое слово"))
    body = "\n".join(l for l in lines if not l.startswith("|"))
    cited = set(re.findall(r"\bS\d+\b", body))
    for sid in SOURCE_IDS:
        if sid not in cited:
            errors.append((sid, "источник ни разу не цитируется"))
    for sid in sorted(cited - set(SOURCE_IDS)):
        errors.append((sid, "цитируется источник, которого нет в таблице источников"))
    no_ci = re.sub(r"\[[+-]?\d+\.\d+; [+-]?\d+\.\d+\]", "", text)      # интервалы «[+0.69; +0.98]» — не метки
    if "[" in no_ci or "]" in no_ci:
        errors.append(("[ ]", "квадратные скобки-метки"))
    return errors


def template_literals() -> list:
    """Числовые литералы в шаблонах текста: строковые константы в аргументах add() и tbl() и в H_TEXT
    (без вложенных вызовов num, cell, R, cfg, mk — там подписи метрик, а не текст отчёта): токен -> строки файла."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    skip_calls = {"num", "cell", "R", "cfg", "mk", "fnum"}
    found: dict = {}

    def walk(node) -> None:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in skip_calls:
            return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for t in re.findall(r"\d+(?:\.\d+)?", node.value):
                found.setdefault(t, set()).add(node.lineno)
        for child in ast.iter_child_nodes(node):
            walk(child)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"add", "tbl"}:
            for arg in node.args:
                walk(arg)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "H_TEXT" for t in node.targets):
            walk(node.value)
    return sorted(((t, sorted(ls)) for t, ls in found.items()), key=lambda x: (float(x[0]), x[0]))


def main() -> None:
    t0 = time.time()
    frozen_log = []
    check_frozen("до", frozen_log)
    sources = load_sources()
    SOURCE_IDS[:] = sources["id"].tolist()
    d = step18.load()
    step18.clean_rosstat(d, [])
    sup, _ = step21.supply_values(d["ids"])
    D = build_frame(d, sup)
    z = z_profile(d, D)
    za = alr_profile(d, D)
    five = d["shares"][list(SHARE_COLUMNS)].sum(axis=1)
    if d["shares"]["month"].nunique() != 24:
        raise SystemExit(f"STOP: в category_shares {d['shares']['month'].nunique()} месяцев, в тексте ограничений — 24")
    cover_all = float(five.mean() * 100)                                   # все 24 месяца, все МО
    cover_dec = float(five[d["shares"]["month"] == MONTH].mean() * 100)
    ps = pd.read_parquet(PS_PATH)
    matching = pd.read_parquet(MATCHING_PATH)
    r19a, r19b = pd.read_parquet(RESULTS_19A_PATH), pd.read_parquet(RESULTS_19B_PATH)
    rsup, verdicts = pd.read_parquet(RESULTS_SUPPLY_PATH), pd.read_parquet(VERDICTS_PATH)
    g10 = rsup[(rsup["row_type"] == "corr") & (rsup["hypothesis"] == "Г10") & (rsup["variant"] == "основной") & (rsup["level"] == "внутри регионов")]
    par = rsup[(rsup["row_type"] == "partial") & (rsup["hypothesis"] == "Г10") & (rsup["level"] == "внутри регионов")]
    if len(g10) != 1 or len(par) != 1:
        raise SystemExit("STOP: в hypothesis_results_supply нет ровно одной строки Г10 (внутри регионов) и частной корреляции")
    sup_res = {"g10_within": float(g10["obs"].iloc[0]), "partial_within": float(par["obs"].iloc[0])}

    add("# 23. Портреты типов МО (k = 6)", "",
        "Сгенерировано `src/23_type_portraits.py`. Шаг описательный: без p-значений, тестов, порогов и вердиктов; "
        "замороженные результаты шагов 18–22 только цитируются.", "")
    add("## Заморозка", "", *frozen_log, "")
    legend(D, sources)
    add("## 1. Общие результаты (все МО)", "")
    my = {}
    o11 = sec_1_1(D)
    my["transport_by_size"], my["catering_by_size"], my["rho_catering_logpop"] = o11["transport"], o11["catering"], o11["rho_catering"]
    my.update(sec_1_2_rest(D))
    o12 = sec_1_2(D)
    my["rho_market_ma_between"], my["rho_market_ma_within"] = o12[("market_access", "между регионами")], o12[("market_access", "внутри регионов")]
    sec_1_3(D)
    sec_1_4(sources)
    add("## 2. Портреты типов", "")
    small_types_table(D)
    ps_list, conf_list = [], []
    for t in range(FINAL_K):
        pm, cf = portrait_common(D, z, za, ps, matching, r19b, t)
        ps_list.append(pm)
        conf_list.append(cf)
        if t == 0:
            type0(D, sup_res)
        elif t == 1:
            type1(D, o11["transport"])
        elif t == 2:
            ah = age_health_block(D)
            type2(D, ah, r19a)
            my["health_gap_expected"], my["health_gap_actual"] = ah[(2, 0)]
        elif t == 3:
            my["type3_groups"] = type3(D)
            salary_block(sources)
            type3_text(D, z, za)
        elif t == 4:
            o4 = type4(D, sources)
            my["yakutia_medians"], my["yakutia_n"] = o4["yakutia_medians"], o4["yakutia_n"]
            my["type4_far_east"] = o4["n_fe"]
            wp = rho_within(D, "Маркетплейсы", "log_pop")[0]
            type4_text(D, z, za, r19a, o12, wp, o4["rho_yakutia"], sources)
        elif t == 5:
            type5(verdicts)
            add("Сравнительная таблица малых типов — раздел 2.0.", "")
    my["type_sizes"] = [int((D["type"] == t).sum()) for t in range(FINAL_K)]
    my["type_regions"] = [int(D.loc[D["type"] == t, "region"].nunique()) for t in range(FINAL_K)]
    my["ps"], my["confident"] = ps_list, conf_list
    my["za"], my["names"] = za, [type_name(D, z, za, t) for t in range(FINAL_K)]
    border = {t: border_features(z, za, t) for t in range(FINAL_K)}
    exp_b = {int(t): v for t, v in PORTRAIT_CONTROLS["profile_border"].items()}
    got_set = {(t, c) for t, lst in border.items() for c, _ in lst}
    exp_set = {(t, c) for t, dct in exp_b.items() for c in dct}
    print(f"пограничные признаки: шаг 23 {sorted(got_set)}; ожидание {sorted(exp_set)}")
    for t, lst in border.items():
        for c, m in lst:
            print(f"  тип {t}: {c}, меньшее |σ| = {m:.4f}")
    bad_b = [(t, c, m, exp_b[t][c]) for t, lst in border.items() for c, m in lst
             if (t, c) in exp_set and abs(m - exp_b[t][c]) > ZERO_TOL["ratio"] + 1e-9]
    if got_set != exp_set or bad_b:
        raise SystemExit(f"STOP: пограничные признаки не совпадают с ожиданием: шаг 23 {sorted(got_set)}, ожидание "
                         f"{sorted(exp_set)}; значения вне допуска {bad_b}")
    lr = label_ratios(D)
    my["pop_ratio"], my["ma_ratio"] = lr["pop_ratio"].tolist(), lr["ma_ratio"].tolist()
    cannot_claim(D)
    sources_section(sources)
    limitations(D, cover_all, cover_dec)
    ctrl = controls_table(my)
    add("## 6. Контрольные сверки", "")
    tbl("Значения шага 23 и контрольные значения из config (расчёты пользователя сделаны до исключения двух аномальных МО)",
        ctrl.astype(str))
    print(ctrl.to_string(index=False))
    ctrl2 = controls_table_2(my)
    tbl("Дополнительные контроли: устойчивость признаков профиля, ρ с log населения, отношения для ярлыков и имена типов "
        "(значения шага 23 и контрольные значения из config)", ctrl2.astype(str))
    print(ctrl2.to_string(index=False))
    ctrl = pd.concat([ctrl, ctrl2], ignore_index=True)
    if (ctrl["статус"] != "совпало").any():
        raise SystemExit("STOP: расхождения с контрольными значениями (значения не подгонялись):\n"
                         + ctrl[ctrl["статус"] != "совпало"].to_string(index=False))
    errors = lint(LINES)
    if errors:
        raise SystemExit("STOP: проверка ссылок и запретных слов:\n" + "\n".join(f"  {e[1]}: {e[0][:200]}" for e in errors[:30]))
    print("lint: ссылки, запрещённые слова, квадратные скобки, числа в шаблонах — нарушений 0")

    n_before = len(frozen_log)
    check_frozen("после", frozen_log)
    at = LINES.index("## Заморозка") + 2 + n_before
    LINES[at:at] = frozen_log[n_before:]
    long = pd.DataFrame(LONG)
    dup = long.duplicated(["type", "block", "metric"], keep=False)
    for _, g in long[dup].groupby(["type", "block", "metric"]):
        if not np.allclose(g["value"].to_numpy(float), g["value"].iloc[0], equal_nan=True):
            raise SystemExit(f"STOP: одно и то же число посчитано по-разному: {g.iloc[0]['type']} / {g.iloc[0]['block']} / {g.iloc[0]['metric']}")
    long = long.drop_duplicates(["type", "block", "metric"]).reset_index(drop=True)
    long["n"] = long["n"].astype("Int64")
    long.to_parquet(PORTRAITS_PATH, engine="pyarrow", index=False)
    add(f"Время: {time.time() - t0:.1f} с", "")
    REPORT_PATH.write_text(strip_marks("\n".join(LINES)), encoding="utf-8")
    print(f"записано: {PORTRAITS_PATH.name} ({len(long)} строк), {REPORT_PATH.name}")
    lits = template_literals()
    print("числовые литералы в шаблонах генератора (токен: строки файла):")
    for tkn, ls in lits:
        print(f"  {tkn}: {ls}")
    print(f"весь шаг: {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main()
