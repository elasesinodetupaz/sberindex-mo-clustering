"""Шаг 24e-1. Каркас лендинга: один автономный HTML-файл (данные, интерактивная карта, карточки типов).

Описательный шаг: только чтение готовых файлов, новых моделей, тестов и порогов нет. Страница не делает внешних запросов
(без CDN, шрифтов и картинок по сети), открывается с диска (file://). Геометрия МО встраивается в HTML сжатой gzip и
base64 и распаковывается в браузере через DecompressionStream. Проза и подписи берутся только из site/content/texts.json
(если файла нет, он создаётся с начальными значениями и дальше скриптом не перезаписывается). Таблицы проверок
разбираются из notebooks/26_hypothesis_summary.md, их числа сверяются с data/processed/hypothesis_summary.parquet.
Параметры и ожидания — config.yaml, группа step24e_landing. Все контроли выполняются до записи выходных файлов;
при нарушении любого контроля код выхода 1 и ничего не записывается.
Вход:  data/geo/*.geojson, ATTRIBUTION.md; data/processed/{layer_data_24f, region_coverage_24f, types_outside_sample,
       types_last_month, kmeans_labels_final, category_shares, type_portraits, mirkin_rule_27, hypothesis_summary,
       composition_robustness}.parquet; docs/{type_names, interpretation}.md; notebooks/26_hypothesis_summary.md;
       notebooks/figures/24_fig*.svg|png; src/23_type_portraits.py (имена типов, доли шага 23)
Выход: site/index.html, site/content/texts.json (если нет), site/README.md, notebooks/24e_landing_check.md
Запуск из корня проекта:  .venv/bin/python src/24e_landing.py
"""
import base64
import gzip
import hashlib
import html
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

import config as C

PROJECT_DIR = Path(__file__).resolve().parents[1]
L = C.LANDING
P = lambda rel: PROJECT_DIR / rel  # noqa: E731
CATS = ["Продовольствие", "Здоровье", "Общепит", "Транспорт", "Маркетплейсы"]
CLASS_ORDER = [1, 2, 4, 3]
BROWSER_BEGIN = "<!-- browser-check:begin -->"
BROWSER_END = "<!-- browser-check:end -->"


def stop(msg: str) -> None:
    print(f"STOP: {msg}")
    sys.exit(1)


CONTROLS: list[tuple[str, str, str, bool]] = []


def ctl(name: str, got, expected, ok: bool | None = None) -> None:
    """Запись контроля; при расхождении STOP с обоими значениями (до записи любых файлов)."""
    ok = (got == expected) if ok is None else ok

    def short(v):
        t = str(v).replace("[", "(").replace("]", ")")
        return t if len(t) <= 160 else f"<{type(v).__name__} из {len(v)} элементов>" if hasattr(v, "__len__") else t[:160]
    CONTROLS.append((name, short(got), short(expected), bool(ok)))
    print(f"[{'ok' if ok else 'НАРУШЕН'}] {name}: {short(got)} (ожидалось {short(expected)})")
    if not ok:
        stop(f"{name}: получено {short(got)}, ожидалось {short(expected)}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen(tag: str) -> list[str]:
    out = []
    for rel, h in L["FROZEN_SHA256"].items():
        p = P(rel)
        if not p.exists():
            stop(f"нет замороженного файла {rel}")
        got = sha256_file(p)
        if got != h:
            stop(f"заморозка ({tag}) {rel}: {got} != {h}")
        out.append(f"- {tag}: `{rel}` — совпадает")
    return out


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def read_text(rel: str) -> str:
    p = P(rel)
    if not p.exists():
        stop(f"нет файла {rel}")
    return p.read_text(encoding="utf-8")


# ============================================================================ входные файлы
REQUIRED = [L["GEO_NATIONAL_PATH"], L["GEO_CITIES_PATH"], L["ATTRIBUTION_PATH"], L["LAYER_PATH"], L["COVERAGE_PATH"],
            L["OUTSIDE_PATH"], L["LAST_MONTH_PATH"], L["LABELS_PATH"], L["SHARES_PATH"], L["PORTRAITS_PATH"],
            L["SUMMARY_PATH"], L["TYPE_NAMES_MD_PATH"], L["INTERPRETATION_PATH"], "src/23_type_portraits.py",
            L["MIRKIN_PATH"], L["COMPOSITION_PATH"], L["NB23_PATH"], L["NB26_PATH"]]


def check_inputs() -> list[str]:
    missing = [r for r in REQUIRED if not P(r).exists()]
    if missing:
        stop(f"нет опорных файлов (предыдущий шаг не выполнен): {missing}")
    return [r for n, r in L["FIGURES"].items() if not P(r).exists()]


# ============================================================================ тексты
def load_texts() -> tuple[dict, bool]:
    """site/content/texts.json: при отсутствии создаётся (запись откладывается до конца), при наличии ключи сверяются."""
    keys = L["TEXT_KEYS"]
    p = P(L["TEXTS_PATH"])
    if len(keys) != 25 or len(set(keys)) != 25:
        stop(f"в config {len(keys)} ключей текста, ожидалось 25 уникальных")
    if not p.exists():
        d = {k: (L["HERO_TITLE_INITIAL"] if k == "hero_title" else L["TEXT_PLACEHOLDER"]) for k in keys}
        return d, True
    d = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict) or set(d) != set(keys) or any(not isinstance(v, str) for v in d.values()):
        stop(f"ключи {p} не совпадают с config: лишние {sorted(set(d) - set(keys))}, нет {sorted(set(keys) - set(d))}")
    return d, False


def texts_json_dump(d: dict) -> str:
    return json.dumps({k: d[k] for k in L["TEXT_KEYS"]}, ensure_ascii=False, indent=2) + "\n"


TEXTCHK: dict = {}
NUM_TOKEN = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?%?|\d+(?:[.,]\d+)?%?")


def token_value(s: str) -> float:
    return float(re.sub(r"[   ]", "", s).replace(",", ".").rstrip("%"))


def token_tol(s: str) -> float:
    m = re.search(r"[.,](\d+)", s)
    return max(0.002, 0.5 * 10 ** -(len(m.group(1)) if m else 0))


def check_texts_input(texts: dict) -> None:
    """Шаг 24e-2 (Б2): проверка texts_input.json и реестра чисел; результат в TEXTCHK, контроли до записи файлов."""
    import csv
    pi, pr = P(L["TEXTS_INPUT_PATH"]), P(L["REGISTRY_PATH"])
    if not pi.exists() or not pr.exists():
        stop(f"нет входных файлов текстов: {pi.name}, {pr.name}")
    ctl("texts_input.json: sha256", sha256_file(pi), L["TEXTS_INPUT_SHA256"])
    ctl("numbers_registry.csv: sha256", sha256_file(pr), L["REGISTRY_SHA256"])
    inp = json.loads(pi.read_text(encoding="utf-8"))
    keys = [k for k in L["TEXT_KEYS"] if k != "hero_title"]
    ctl("texts_input.json: ровно 24 ключа = ключи texts.json без hero_title", (len(inp), sorted(inp)), (24, sorted(keys)))
    ctl("texts_input.json: значения — непустые строки", [k for k, v in inp.items() if not isinstance(v, str) or not v.strip()], [])
    ctl("texts.json: hero_title совпадает с исходным побайтно", texts["hero_title"].encode("utf-8"), L["HERO_TITLE_INITIAL"].encode("utf-8"))
    ctl("texts.json: 24 значения совпадают с texts_input.json", [k for k in keys if texts[k] != inp[k]], [])
    ctl("тексты: маркер заглушки во входных значениях", [k for k, v in inp.items() if L["TEXT_PLACEHOLDER"] in v], [])
    low = {k: v.lower() for k, v in inp.items()}
    stems = [(k, s, inp[k][max(0, low[k].find(s) - 15):low[k].find(s) + 25]) for k in inp for s in L["FORBIDDEN_STEMS"] if s in low[k]]
    ctl("тексты: запрещённые основы (2.3) в значениях (ключ, основа, фрагмент)", stems, [])
    br = [k for k, v in inp.items() if re.sub(r"\[[+-]?\d+(?:\.\d+)?; [+-]?\d+(?:\.\d+)?\]", "", v).count("[") or re.sub(r"\[[+-]?\d+(?:\.\d+)?; [+-]?\d+(?:\.\d+)?\]", "", v).count("]")]
    ctl("тексты: квадратные скобки вне записи интервала", br, [])
    # prose(): esc() экранирует HTML, абзацы по пустой строке; ломающие вывод символы: теги и амперсанды, управляющие символы
    brk = [k for k, v in inp.items() if re.search(r"[<>&\x00-\x08\x0b\x0c\x0e-\x1f]", v)]
    ctl("тексты: символы, ломающие вывод prose() (<, >, &, управляющие)", brk, [])
    rows = list(csv.DictReader(pr.open(encoding="utf-8", newline="")))
    ctl("реестр: колонки", list(rows[0].keys()), ["key", "token", "source_file", "anchor"])
    reg: dict = {}
    for r in rows:
        reg.setdefault(r["key"], set()).add(r["token"])
    tok = {k: set(NUM_TOKEN.findall(v)) for k, v in inp.items()}
    ctl("реестр: все ключи реестра есть в текстах", sorted(set(reg) - set(inp)), [])
    ctl("числа: токены текста без строки реестра (ключ, токены)", [(k, sorted(tok[k] - reg.get(k, set()))) for k in inp if tok[k] - reg.get(k, set())], [])
    ctl("числа: строки реестра без токена в тексте (ключ, токены)", [(k, sorted(reg[k] - tok[k])) for k in reg if reg[k] - tok[k]], [])
    cache: dict = {}
    miss_file, miss_anchor, miss_num = [], [], []
    for r in rows:
        f = P(r["source_file"])
        if not f.exists():
            miss_file.append(r["source_file"])
            continue
        txt = cache.setdefault(r["source_file"], f.read_text(encoding="utf-8"))
        if r["anchor"] not in txt:
            miss_anchor.append((r["key"], r["token"], r["anchor"]))
            continue
        tv, tl = token_value(r["token"]), token_tol(r["token"])
        if not any(abs(token_value(x) - tv) <= tl for x in NUM_TOKEN.findall(r["anchor"])):
            miss_num.append((r["key"], r["token"], r["anchor"]))
    ctl("реестр: файл source_file существует", miss_file, [])
    ctl("реестр: anchor найден в файле (точное вхождение)", miss_anchor, [])
    ctl("реестр: токен совпадает с числом в anchor (допуск 2.2)", miss_num, [])
    TEXTCHK.update(keys=len(inp), tokens=sum(len(v) for v in tok.values()), rows=len(rows),
                   per_key=[(k, len(inp[k]), len(tok[k]), sum(1 for r in rows if r["key"] == k)) for k in keys])


def prose(texts: dict, key: str, tag: str = "p") -> str:
    """Проза из texts.json: абзацы по пустой строке; заглушка видна читателю."""
    t = texts[key].strip()
    cls = "ph" if t == L["TEXT_PLACEHOLDER"] else ""
    paras = [x.strip() for x in re.split(r"\n\s*\n", t) if x.strip()]
    return "".join(f'<{tag} class="{cls}" data-text-key="{esc(key)}">{esc(x)}</{tag}>' for x in paras)


def num(text, src: str) -> str:
    return f'<span class="num" title="источник: {esc(src)}">{esc(text)}</span>'


# ============================================================================ геометрия
def encode_geo(rel: str, scale: int, shift: bool) -> dict:
    """Геометрия в целых числах: контуры по часовой/против (orient как в шаге 24d), дельта-кодирование внутри колец."""
    feats = json.loads(read_text(rel))["features"]
    feats = sorted(feats, key=lambda f: int(f["properties"]["territory_id"]))
    ids, F, props, n_pts, n_rings, lons = [], [], [], 0, 0, []
    max_err = 0.0
    for f in feats:
        ids.append(int(f["properties"]["territory_id"]))
        props.append(f["properties"])
        polys = []
        for rings in f["geometry"]["coordinates"]:
            rr = []
            for ring in rings:
                a = np.array(ring, float)
                if shift:
                    a[a[:, 0] < L["LON_SHIFT_BELOW"], 0] += 360.0
                rr.append(a)
            poly = orient(Polygon(rr[0], rr[1:]), 1.0)
            enc = []
            for r in [poly.exterior, *poly.interiors]:
                c = np.asarray(r.coords)
                n_pts += len(c)
                n_rings += 1
                lons.append(c[:, 0])
                sc = c * scale
                ints = np.rint(sc).astype(np.int64)
                max_err = max(max_err, float(np.abs(sc - ints).max()))
                d = np.diff(ints, axis=0, prepend=np.zeros((1, 2), np.int64))
                enc.append(d.reshape(-1).tolist())
            polys.append(enc)
        F.append(polys)
    lon = np.concatenate(lons)
    return {"ids": ids, "F": F, "props": props, "n_pts": n_pts, "n_rings": n_rings, "max_err": max_err,
            "lon_min": float(lon.min()), "lon_max": float(lon.max()), "scale": scale}


def pack_geo(g: dict, extra: dict | None = None) -> str:
    obj = {"s": g["scale"], "ids": g["ids"], "f": g["F"]}
    if extra:
        obj.update(extra)
    raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode("ascii")


# ============================================================================ данные шага 23 (как в 24d type_names)
def step23_frame():
    s23 = importlib.import_module("23_type_portraits")
    d = s23.step18.load()
    s23.step18.clean_rosstat(d, [])
    sup, _ = s23.step21.supply_values(d["ids"])
    D = s23.build_frame(d, sup)
    z = s23.z_profile(d, D)
    za = s23.alr_profile(d, D)
    names = [s23.full_type_name(D, z, za, t) for t in range(C.FINAL_K)]
    return s23, D, names


def parse_type_names_md() -> dict:
    out = {}
    for line in read_text(L["TYPE_NAMES_MD_PATH"]).split("\n"):
        m = re.match(r"^\|\s*(\d)\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            stop(f"docs/type_names.md: ожидалось 5 колонок, получено {len(cells)}: {line[:80]}")
        out[int(cells[0])] = {"tech": cells[1], "name": cells[2], "status": cells[3], "alt": cells[4]}
    return out


def nb23_names() -> list[str]:
    out = []
    for line in read_text(L["NB23_PATH"]).split("\n"):
        m = re.match(r"- Имя типа: (.*) \(расчёт шага 23\)\.$", line)
        if m:
            out.append(m.group(1))
    return out


# ============================================================================ разбор notebooks/26
def split_row(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip())[1:-1]
    return [c.strip().replace("\\|", "|") for c in cells]


def parse_md_tables(text: str) -> list[dict]:
    """Таблицы md: {heading, header, rows}; heading — последний заголовок раздела перед таблицей."""
    lines = text.split("\n")
    tables, heading, i = [], "", 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            heading = ln.strip()
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            header = split_row(ln)
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                r = split_row(lines[j])
                if len(r) != len(header):
                    stop(f"notebooks/26: в таблице «{heading}» строка из {len(r)} ячеек при {len(header)} колонках: {lines[j][:80]}")
                rows.append(r)
                j += 1
            tables.append({"heading": heading, "header": header, "rows": rows, "line": i + 1})
            i = j
            continue
        i += 1
    return tables


def md_section_lists(text: str, heading: str) -> list[tuple[str, list[str]]]:
    """Списки раздела: [(жирный подзаголовок, [пункты])]; таблицы и текст вне списков пропускаются."""
    lines = text.split("\n")
    if heading not in lines:
        stop(f"в notebooks/26 нет раздела {heading}")
    k = lines.index(heading) + 1
    out, cur = [], None
    while k < len(lines) and not lines[k].startswith("## "):
        m = re.match(r"^\*\*(.+)\*\*$", lines[k].strip())
        if m:
            cur = (m.group(1), [])
            out.append(cur)
        elif lines[k].startswith("- ") and cur is not None:
            cur[1].append(lines[k][2:].strip())
        k += 1
    return out


TOKEN = re.compile(r"(?<![\w.])([+\-−]?\d+(?:\.\d+)?)(%?)(?![\w])")


def tokens(text: str) -> list[tuple[str, int, bool]]:
    """Числа в тексте: (нормализованная строка, число знаков после точки, процент)."""
    out = []
    for m in TOKEN.finditer(text):
        t = m.group(1).replace("−", "-")
        d = len(t.split(".")[1]) if "." in t else 0
        out.append((t, d, m.group(2) == "%"))
    return out


def parquet_numbers(rows: pd.DataFrame) -> list[tuple[float, int]]:
    """Числа строк: value, q025, q975, n; для «Δ / η²_H до учёта» ещё и в процентах (как печатает шаг 26)."""
    vals = []
    for _, r in rows.iterrows():
        for c in ("value", "q025", "q975"):
            if pd.notna(r[c]):
                vals.append((float(r[c]), 1))
                vals.append((float(r[c]) * 100.0, 100))
        if pd.notna(r["n"]):
            vals.append((float(r["n"]), 0))
    return vals


def token_matches(tok: tuple[str, int, bool], pool: list[tuple[float, int]]) -> bool:
    t, d, pct = tok
    want_pct = 100 if pct else 1
    for v, kind in pool:
        if kind == 0:                       # n и целые значения
            if d == 0 and not pct and abs(float(t) - v) < 1e-9:
                return True
            continue
        if kind != want_pct:
            continue
        if t in (f"{v:.{d}f}", f"{v:+.{d}f}"):
            return True
        if not t.startswith(("+", "-")) and f"{abs(v):.{d}f}" == t and v > 0:
            return True
    # целые значения вида 18 (value 18.0)
    if d == 0 and not pct:
        for v, kind in pool:
            if kind == 1 and abs(float(t) - v) < 1e-9:
                return True
    return False


# ============================================================================ сверка таблиц шага 26 с parquet
PRIMARY_COLS = {
    "main": ("показатель", "размер"),
    "g7": ("η²_H доли до учёта (n)", "Δ в долях η²_H до учёта"),
    "g4": ("общий: η²_H (n)", "внутри регионов: η²_H (n)"),
    "sup": ("показатель", "частная ρ с поправкой на размер МО (разведочно)"),
    "supd": ("η²_H доли до учёта (n)", "Δ в долях η²_H до учёта"),
}


def is_primary(kind: str, col: str) -> bool:
    return col in PRIMARY_COLS[kind] or (col.startswith("Δ =") and kind in ("g7", "supd"))


def row_pools(kind: str, row: list[str], header: list[str], P1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Строки parquet, с которыми сверяется строка md: по уровню (для первичных колонок) и по item (для остальных)."""
    h = dict(zip(header, row))
    if kind == "main":
        hyp, what, lev = h["проверка"], h["что проверяется"], h["уровень"]
        item = hyp
        if hyp == "Г1" and "железной дороги" in what:
            item = "Г1 железная дорога"
        elif hyp == "Г1" and "в каждом из" in what:
            item = "Г1 порядок типов"
        by_item = P1[P1["item"] == item]
        return {"level": by_item[by_item["level"] == lev], "item": by_item}
    if kind in ("g7", "sup", "supd"):
        hyp, lev = h["проверка"], h["уровень"]
        by_item = P1[P1["item"] == hyp]
        return {"level": by_item[by_item["level"] == lev], "item": by_item}
    stop(f"неизвестный вид таблицы {kind}")


def verify_main_like(kind: str, tab: dict, P1: pd.DataFrame, used: dict, rep: dict) -> None:
    header = tab["header"]
    for row in tab["rows"]:
        pools = row_pools(kind, row, header, P1)
        if pools["level"].empty:
            rep["rows_unchecked"].append((kind, row[0], row[2] if kind != "g4" else "", "нет строк parquet с таким item и уровнем"))
            continue
        pool_l, pool_i = parquet_numbers(pools["level"]), parquet_numbers(pools["item"])
        checked = 0
        for col, cell in zip(header, row):
            if col in ("проверка", "что проверяется", "уровень"):
                continue
            for tok in tokens(cell):
                prim = is_primary(kind, col)
                ok = token_matches(tok, pool_l if prim else pool_i)
                if ok:
                    checked += 1
                    rep["numbers_checked"] += 1
                    for idx, r in (pools["level"] if prim else pools["item"]).iterrows():
                        if token_matches(tok, parquet_numbers(pd.DataFrame([r]))):
                            used.setdefault(idx, "первичная" if prim else "в оговорке/результате")
                            if prim:
                                used[idx] = "первичная"
                elif tok[1] > 0 or tok[2]:
                    if prim:
                        stop(f"сверка notebooks/26 и parquet: {kind}, проверка {row[0]}, уровень {row[2]}, колонка «{col}»: число {tok[0]}"
                             f"{'%' if tok[2] else ''} в md; в parquet (item {pools['level']['item'].iloc[0]}): "
                             f"{[(r['metric'], round(r['value'], 6), r['n']) for _, r in pools['level'].iterrows()]}")
                    rep["tokens_unpaired"].append((kind, row[0], row[2], col, tok[0] + ("%" if tok[2] else "")))
                else:
                    rep["ints_unpaired"] += 1
        # объёмы выборки n в первичных колонках: «n = N» и «(N)» должны совпасть с n строк parquet точно
        for col, cell in zip(header, row):
            if is_primary(kind, col):
                for nn in re.findall(r"(?:n = |\d \()(\d+)\)", cell):
                    if int(nn) not in {int(x) for x in pools["level"]["n"].dropna()}:
                        stop(f"сверка notebooks/26 и parquet: {kind}, проверка {row[0]}, уровень {row[2]}, колонка «{col}»: n = {nn} в md; "
                             f"в parquet {sorted({int(x) for x in pools['level']['n'].dropna()})}")
                    rep["numbers_checked"] += 1
        # текстовые значения: результат и размер
        h = dict(zip(header, row))
        for col, pcol in (("результат", "result"), ("размер", "size")):
            if col not in h:
                continue
            cands = {x for x in pools["level"][pcol].tolist() if x}
            if cands and h[col] not in cands:
                stop(f"сверка notebooks/26 и parquet: {kind}, {row[0]}, {row[2]}, колонка «{col}»: md «{h[col]}», parquet {sorted(cands)}")
            if cands:
                rep["strings_checked"] += 1
        if checked == 0:
            rep["rows_unchecked"].append((kind, row[0], row[2], "ни одно число строки не найдено в parquet"))
        else:
            rep["rows_checked"] += 1
        rep["rows_total"] += 1


def verify_g4(tab: dict, P1: pd.DataFrame, used: dict, rep: dict) -> None:
    header = tab["header"]
    levels = [("общий", 1), ("внутри регионов", 4)]
    for row in tab["rows"]:
        letter = row[0]
        item = P1[P1["item"] == f"Г4 {letter}"]
        if item.empty:
            rep["rows_unchecked"].append(("g4", letter, "", "нет строк parquet"))
            continue
        for lev, c0 in levels:
            q = item[item["level"] == lev]
            if len(q) != 1:
                stop(f"сверка Г4 {letter} {lev}: в parquet {len(q)} строк, ожидалась 1")
            r = q.iloc[0]
            cell = row[c0]
            m = re.match(r"^(\d+\.\d+) \((\d+)\)$", cell)
            if not m:
                stop(f"Г4 {letter} {lev}: не разобрана ячейка «{cell}»")
            if f"{r['value']:.3f}" != m.group(1) or int(r["n"]) != int(m.group(2)):
                stop(f"сверка Г4 {letter} {lev}: md {cell}, parquet {r['value']:.6f} ({r['n']})")
            if row[c0 + 1] != r["size"] or row[c0 + 2] != r["result"]:
                stop(f"сверка Г4 {letter} {lev}: размер/результат md ({row[c0 + 1]}; {row[c0 + 2]}) и parquet ({r['size']}; {r['result']})")
            used[q.index[0]] = "первичная"
            rep["numbers_checked"] += 2
            rep["strings_checked"] += 2
        rep["rows_checked"] += 1
        rep["rows_total"] += 1


def verify_summary(tables: list[dict], H: pd.DataFrame, comp: pd.DataFrame, lists: list) -> dict:
    P1 = H[H["block"].str.startswith("1.")]
    rep = {"rows_total": 0, "rows_checked": 0, "rows_unchecked": [], "numbers_checked": 0, "strings_checked": 0,
           "tokens_unpaired": [], "ints_unpaired": 0, "tables": []}
    used: dict = {}

    def pick(heading: str, first_col: str, header_has: str | None = None) -> dict:
        c = [t for t in tables if t["heading"] == heading and t["header"][0] == first_col and (header_has is None or header_has in t["header"])]
        if len(c) != 1:
            stop(f"в notebooks/26 под заголовком «{heading}» найдено {len(c)} таблиц ({first_col}, {header_has}), ожидалась 1")
        return c[0]
    t_main = pick("## 1. Проверки", "проверка")
    t_g7 = pick("### Учёт возраста (Г7)", "проверка")
    t_g4 = pick("### Г4 по отраслям", "отрасль")
    t_sup = pick("### Дополнительное семейство (Г10–Г13)", "проверка", "показатель")
    t_supd = pick("### Дополнительное семейство (Г10–Г13)", "проверка", "η²_H доли до учёта (n)")
    for kind, tab in (("main", t_main), ("g7", t_g7), ("sup", t_sup), ("supd", t_supd)):
        before = set(used)
        verify_main_like(kind, tab, P1, used, rep)
        rep["tables"].append((kind, tab["heading"], len(tab["rows"]), len(set(used) - before)))
    before = set(used)
    verify_g4(t_g4, P1, used, rep)
    rep["tables"].insert(2, ("g4", t_g4["heading"], len(t_g4["rows"]), len(set(used) - before)))
    rep["unused"] = [(int(i), r["item"], r["level"], r["metric"], round(float(r["value"]), 6)) for i, r in P1.iterrows() if i not in used]
    rep["P1"] = len(P1)
    rep["used_primary"] = sum(1 for v in used.values() if v == "первичная")
    rep["used_folded"] = sum(1 for v in used.values() if v != "первичная")
    # раздел 3: списки; числа сверяются со всей сводкой и с шагом 25
    pool_all = parquet_numbers(H) + parquet_numbers(comp.rename(columns={}).assign(q025=np.nan, q975=np.nan)[["value", "n", "q025", "q975"]])
    s3 = {"bullets": 0, "numbers_checked": 0, "unpaired": []}
    for _, items in lists:
        for it in items:
            s3["bullets"] += 1
            for tok in tokens(it):
                if tok[1] == 0 and not tok[2]:
                    continue
                if token_matches(tok, pool_all):
                    s3["numbers_checked"] += 1
                else:
                    s3["unpaired"].append((it[:60], tok[0]))
    rep["s3"] = s3
    return rep


# ============================================================================ шаблоны страницы
CSS = r''':root{--bg:#FBFAF7;--panel:#F1EEE6;--ink:#1D2329;--muted:#4B5563;--line:#D9D4C7;--accent:#0B5A8A;--ph-bg:#FFF3D6;--ph-ink:#6B3A00;--btn:#E9E5DA;--dark:#1D2B36}
*{box-sizing:border-box}
html{scroll-behavior:smooth;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif}
a{color:var(--accent)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.9em;background:var(--panel);padding:0 .25em;border-radius:3px}
.wrap{max-width:1200px;margin:0 auto;padding:0 16px}
header.hero{padding:40px 0 24px;border-bottom:1px solid var(--line)}
h1{font-size:clamp(1.6rem,4.2vw,2.6rem);line-height:1.15;margin:0 0 12px;max-width:26ch}
h2{font-size:clamp(1.3rem,3vw,1.7rem);margin:0 0 12px;line-height:1.2}
h3{font-size:1.1rem;margin:20px 0 8px}
h4{font-size:1rem;margin:16px 0 6px}
section{padding:36px 0;border-bottom:1px solid var(--line);scroll-margin-top:52px}
p{margin:0 0 12px;max-width:75ch}
.ph{background:var(--ph-bg);color:var(--ph-ink);border:1px dashed #C9A24A;border-radius:6px;padding:8px 12px}
.muted{color:var(--muted);font-size:.92rem}
nav.top{position:sticky;top:0;z-index:20;background:var(--bg);border-bottom:1px solid var(--line)}
nav.top ul{list-style:none;margin:0;padding:8px 16px;display:flex;gap:6px 16px;overflow-x:auto;white-space:nowrap;max-width:1200px;margin:0 auto}
nav.top a{text-decoration:none;padding:0 6px;display:inline-flex;align-items:center;min-height:40px;min-width:40px}
.stats{display:flex;flex-wrap:wrap;gap:12px 32px;margin:20px 0 0;padding:0;list-style:none}
.stats li{display:flex;flex-direction:column}
.stats .big{font-size:clamp(1.8rem,5vw,2.6rem);font-weight:700;line-height:1.1}
.stats .lab{color:var(--muted);font-size:.92rem}
.num{border-bottom:1px dotted var(--muted);cursor:help}
/* карта */
.maptools{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:flex-end;margin:0 0 10px}
fieldset.layers{border:1px solid var(--line);border-radius:8px;padding:6px 10px;margin:0}
fieldset.layers legend{font-size:.85rem;color:var(--muted);padding:0 4px}
fieldset.layers{display:flex;flex-wrap:wrap;gap:4px 8px}
fieldset.layers label{position:relative;display:inline-flex;align-items:center;min-height:40px;min-width:40px;padding:0 12px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
fieldset.layers input{position:absolute;left:-1px;top:-1px;width:calc(100% + 2px);height:calc(100% + 2px);margin:0;opacity:0;cursor:pointer}
fieldset.layers label:has(input:checked){background:var(--btn);border-color:#6B6556;font-weight:700}
fieldset.layers label:has(input:focus-visible){outline:3px solid #0B5A8A;outline-offset:2px}
.btn{font:inherit;background:var(--btn);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:5px 12px;cursor:pointer;min-height:40px;min-width:40px}
.btn:hover{background:#DDD8CB}
.btn:focus-visible,input:focus-visible,summary:focus-visible,th button:focus-visible,a:focus-visible,.leg-item:focus-visible{outline:3px solid #0B5A8A;outline-offset:2px}
.search{position:relative;flex:1 1 240px;max-width:380px}
.search input{width:100%;font:inherit;padding:6px 10px;border:1px solid #8A8577;border-radius:6px;background:#fff;color:var(--ink);min-height:40px}
.search ul{position:absolute;z-index:10;left:0;right:0;top:100%;margin:2px 0 0;padding:0;list-style:none;background:#fff;border:1px solid var(--line);border-radius:6px;max-height:300px;overflow:auto}
.search li button{display:block;width:100%;min-height:40px;text-align:left;background:none;border:0;padding:6px 10px;font:inherit;color:var(--ink);cursor:pointer}
.search li button:hover{background:var(--panel)}
.search li small{display:block;color:var(--muted)}
.layer-note{margin:0 0 10px}
.mapgrid{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px;align-items:start}
.mapbox{position:relative;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#F7F5EF;aspect-ratio:2.05/1;min-height:300px}
.mapbox canvas{position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none}
#map-overlay{pointer-events:none}
.tip{position:absolute;z-index:5;pointer-events:none;background:var(--dark);color:#fff;border-radius:6px;padding:8px 10px;font-size:13px;line-height:1.4;max-width:300px;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.tip b{display:block;font-size:14px}
.tip .sub{opacity:.9}
.tip .note{margin:6px 0 0;font-size:12px;border-left:3px solid #9FB3C4;padding-left:6px}
.loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted)}
.side{display:flex;flex-direction:column;gap:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.panel h3{margin:0 0 4px;font-size:1rem}
.rows{margin:8px 0 0;font-size:.92rem}
.rows .r{display:flex;flex-wrap:wrap;gap:0 8px;padding:2px 0}
.rows .k{flex:0 1 auto;color:var(--muted)}
.rows .v{flex:1 1 9em;overflow-wrap:anywhere}
.tip .rows{margin:4px 0 0;font-size:13px}
.tip .k{color:#fff;opacity:.85}
.panel .note{margin:8px 0 0;font-size:.9rem;border-left:3px solid #8A8577;padding-left:8px}
.chip{display:inline-block;width:.9em;height:.9em;border-radius:3px;vertical-align:-1px;margin-right:6px;border:1px solid rgba(0,0,0,.4)}
.legend h3{margin:0 0 6px;font-size:.95rem}
.leg-item{display:flex;align-items:flex-start;gap:8px;width:100%;text-align:left;font:inherit;font-size:.9rem;color:var(--ink);background:none;border:1px solid transparent;border-radius:6px;padding:4px 6px;cursor:pointer;min-height:40px;min-width:40px}
.leg-item canvas{flex:0 0 auto;margin-top:2px;border:1px solid rgba(0,0,0,.4);border-radius:3px}
.leg-item[aria-pressed="false"]{color:var(--muted);text-decoration:line-through}
.leg-item[aria-pressed="false"] canvas{opacity:.35}
.leg-item:hover{background:#E6E1D4}
.leg-static{display:flex;gap:8px;align-items:flex-start;font-size:.9rem;padding:4px 6px}
.leg-static canvas{border:1px solid rgba(0,0,0,.4);border-radius:3px;margin-top:2px}
.grad{height:14px;border-radius:3px;border:1px solid rgba(0,0,0,.4);margin:4px 0}
.ticks{position:relative;height:18px;font-size:.8rem;color:var(--muted)}
.ticks span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.mapbtns{display:flex;flex-wrap:wrap;gap:6px}
.monthctl{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin:0 0 10px}
.monthctl input[type=range]{flex:1 1 220px;max-width:420px;min-height:40px}
.monthctl output{font-weight:700;min-width:5em}
.monthctl .btn:disabled,.monthctl input:disabled{opacity:.6;cursor:not-allowed}
.warn{margin-left:auto;color:#6B3A00;background:#FFF3D6;border-radius:3px;padding:0 4px;cursor:help}
#map-fallback{border:1px solid var(--line);border-radius:8px;padding:12px;background:var(--panel)}
#map-fallback img{max-width:100%;height:auto;display:block;margin:8px 0}
/* карточки */
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr));gap:16px}
.card{background:#fff;border:1px solid var(--line);border-top:6px solid var(--tc);border-radius:8px;padding:14px;min-width:0}
.card h3{margin:0 0 2px;font-size:1.1rem}
.card .tech{color:var(--muted);font-size:.9rem;margin:0 0 8px}
.card .stat{font-size:.92rem;margin:0 0 4px}
.card table{width:100%;border-collapse:collapse;font-size:.85rem;margin:8px 0}
.tablewrap{overflow-x:auto;max-width:100%}
.card th,.card td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right}
.card th:first-child,.card td:first-child{text-align:left}
.card thead th{font-weight:600;vertical-align:bottom}
.card .tablewrap table{min-width:540px}
.card th:last-child,.card td:last-child{text-align:left;min-width:8.5em}
.card .status{font-size:.85rem;color:var(--muted)}
/* таблицы */
table.dt{width:100%;border-collapse:collapse;font-size:.9rem}
table.dt th,table.dt td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
table.dt thead th{background:var(--panel);position:sticky;top:0}
table.dt td.r,table.dt th.r{text-align:right}
th button{font:inherit;font-weight:600;background:none;border:0;color:var(--ink);cursor:pointer;padding:0 6px;text-align:inherit;min-height:40px;min-width:40px}
th[aria-sort="ascending"] button::after{content:" \25B2"}
th[aria-sort="descending"] button::after{content:" \25BC"}
.bar{position:relative;min-width:130px;height:20px;background:#ECE8DD;border-radius:3px;overflow:hidden}
.bar i{position:absolute;left:0;top:0;bottom:0;background:#D2A56B}
.bar span{position:relative;padding-left:6px;font-size:.85rem;line-height:20px}
.covwrap{max-height:640px;overflow:auto;border:1px solid var(--line);border-radius:8px}
.figs figure{margin:0 0 28px}
.figs svg,.figs img{max-width:100%;height:auto;display:block;background:#fff;border:1px solid var(--line);border-radius:6px}
figcaption{margin-top:6px}
.fig-ph{border:1px dashed #C9A24A;background:var(--ph-bg);color:var(--ph-ink);padding:40px 16px;text-align:center;border-radius:6px}
ul.lim li{margin-bottom:6px;max-width:80ch}
details{margin:8px 0}
summary{cursor:pointer;min-height:40px;display:list-item;padding:8px 0}
footer{padding:24px 0;color:var(--muted);font-size:.9rem}
.srcnote{font-size:.88rem;color:var(--muted)}
@media (max-width:900px){.mapgrid{grid-template-columns:minmax(0,1fr)}.mapbox{aspect-ratio:1.15/1}}
@media (max-width:700px){
 table.rt thead{display:none}
 table.rt,table.rt tbody,table.rt tr,table.rt td{display:block;width:100%}
 table.rt tr{border:1px solid var(--line);border-radius:8px;margin:0 0 10px;padding:6px 8px;background:#fff}
 table.rt td{border:0;padding:3px 0}
 table.rt td::before{content:attr(data-label);display:block;font-size:.8rem;color:var(--muted);font-weight:600}
 table.rt td:empty{display:none}
}
@media print{nav.top{display:none}}
'''

JS = r'''(function () {
"use strict";
function $(s, r) { return (r || document).querySelector(s); }
function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined && text !== null) e.textContent = text; return e; }
function fmt(v, d) { return Number(v).toFixed(d); }

/* ---------- сортировка таблицы покрытия ---------- */
$$("table.sortable").forEach(function (t) {
  var heads = $$("th[data-type]", t), body = $("tbody", t);
  heads.forEach(function (th) {
    var b = th.querySelector("button");
    b.addEventListener("click", function () {
      var ci = Array.prototype.indexOf.call(th.parentNode.children, th);
      var cur = th.getAttribute("aria-sort"), dir = cur === "descending" ? "ascending" : (cur === "ascending" ? "descending" : (th.getAttribute("data-type") === "num" ? "descending" : "ascending"));
      heads.forEach(function (h) { h.removeAttribute("aria-sort"); });
      th.setAttribute("aria-sort", dir);
      var rows = $$("tr", body).map(function (tr, i) { return { tr: tr, i: i, v: tr.children[ci].getAttribute("data-v") }; });
      var num = th.getAttribute("data-type") === "num", sgn = dir === "ascending" ? 1 : -1;
      rows.sort(function (a, b) {
        var c = num ? (parseFloat(a.v) - parseFloat(b.v)) : a.v.localeCompare(b.v, "ru");
        return c !== 0 ? c * sgn : a.i - b.i;
      });
      rows.forEach(function (r) { body.appendChild(r.tr); });
    });
  });
});

/* ---------- карта ---------- */
var APP = JSON.parse($("#app-data").textContent);
var T = APP.types, M = APP.mo, N = M.id.length, CF = APP.cfg, STAT = APP.statusLabels, CLS = APP.classLabels, CATS = APP.cats;
var COS = Math.cos(CF.lat * Math.PI / 180);
var wrapEl = $("#map-wrap"), cv = $("#map-canvas"), ov = $("#map-overlay"), tip = $("#map-tip"), panel = $("#map-panel");
var hintHTML = panel.innerHTML;
var ctx = cv.getContext("2d"), octx = ov.getContext("2d");
var W = 0, H = 0, dpr = 1, ready = false, raf = 0, oraf = 0;
var view = { s: 1, tx: 0, ty: 0, fit: true };
var st = { layer: "type", hiddenTypes: {}, hiddenClasses: {}, sel: -1, hov: -1, hl: [], month: 23 };
var MONTHS = APP.months24, NM = MONTHS.length, LAST = NM - 1, LBL = null, playTimer = 0;
function b64bytes(b64) { var bin = atob(b64.replace(/\s+/g, "")), u = new Uint8Array(bin.length), k; for (k = 0; k < bin.length; k++) u[k] = bin.charCodeAt(k); return u; }
function monthMode() { return st.layer === "type" && st.month !== LAST; }
function labelAt(i, m) { var v = LBL[i * NM + m]; return v === APP.noLabel ? null : v; }
var natFeat = new Array(N), cityFeat = new Array(N), idIdx = {}, bbAll = null, cityBox = {};
var NN = [], i;
for (i = 0; i < N; i++) { idIdx[M.id[i]] = i; NN.push(M.name[i].toLowerCase().replace(/ё/g, "е")); }

function inflate(b64) {
  var bin = atob(b64.replace(/\s+/g, "")), u8 = new Uint8Array(bin.length), k;
  for (k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k);
  var stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).text().then(function (t) { return JSON.parse(t); });
}
function decodeGeo(js, target, cityKey) {
  var S = js.s, j, p, r, k, X, Y, x, y, ring, path, bb, poly;
  for (j = 0; j < js.ids.length; j++) {
    path = new Path2D(); bb = [Infinity, Infinity, -Infinity, -Infinity];
    for (p = 0; p < js.f[j].length; p++) {
      poly = js.f[j][p];
      for (r = 0; r < poly.length; r++) {
        ring = poly[r]; X = 0; Y = 0;
        for (k = 0; k < ring.length; k += 2) {
          X += ring[k]; Y += ring[k + 1]; x = X / S; y = -(Y / S) / COS;
          if (k === 0) path.moveTo(x, y); else path.lineTo(x, y);
          if (x < bb[0]) bb[0] = x; if (y < bb[1]) bb[1] = y; if (x > bb[2]) bb[2] = x; if (y > bb[3]) bb[3] = y;
        }
        path.closePath();
      }
    }
    var ix = idIdx[js.ids[j]];
    if (ix === undefined) throw new Error("id " + js.ids[j]);
    target[ix] = { path: path, bb: bb, city: cityKey ? js.city[j] : null };
  }
}
function unionBox(list) {
  var b = [Infinity, Infinity, -Infinity, -Infinity];
  list.forEach(function (f) { if (f.bb[0] < b[0]) b[0] = f.bb[0]; if (f.bb[1] < b[1]) b[1] = f.bb[1]; if (f.bb[2] > b[2]) b[2] = f.bb[2]; if (f.bb[3] > b[3]) b[3] = f.bb[3]; });
  return b;
}

/* ---------- цвета, ключи, стили ---------- */
function hex(c) { return [parseInt(c.substr(1, 2), 16), parseInt(c.substr(3, 2), 16), parseInt(c.substr(5, 2), 16)]; }
function ramp(stops, t) {
  t = Math.min(1, Math.max(0, t));
  var seg = (stops.length - 1) * t, a = Math.min(stops.length - 2, Math.floor(seg)), f = seg - a;
  var c0 = hex(stops[a]), c1 = hex(stops[a + 1]);
  return "rgb(" + [0, 1, 2].map(function (q) { return Math.round(c0[q] + (c1[q] - c0[q]) * f); }).join(",") + ")";
}
function maT(v) {
  var m = CF.ma, a, b, x;
  if (m.kind === "log") { a = Math.log(m.min); b = Math.log(m.max); x = Math.log(v); } else { a = m.min; b = m.max; x = v; }
  return Math.min(1, Math.max(0, (x - a) / (b - a)));
}
function keyOf(layer, i) {
  if (layer === "type" && monthMode()) return M.cls[i] === 1 ? "m:" + labelAt(i, st.month) : (M.cls[i] === 3 ? "3" : "na");
  if (layer === "type") return M.cls[i] === 3 ? "3" : M.cls[i] + ":" + M.typ[i];
  if (layer === "market") return M.ma[i] === null ? "na" : "b" + Math.round(maT(M.ma[i]) * (CF.ma.bins - 1));
  return M.cls[i] === 1 ? "s" + M.ms[i] : "na";
}
function styleOf(layer, key) {
  if (layer === "type" && key.charAt(0) === "m") return { fill: T[+key.slice(2)].color, hatch: null };
  if (key === "na") return { fill: CF.noValueFace, hatch: "horiz" };
  if (layer === "type") {
    if (key === "3") return { fill: CF.noTypeFace, hatch: "cross" };
    var p = key.split(":");
    return { fill: T[+p[1]].color, hatch: p[0] === "2" ? "diag" : (p[0] === "4" ? "dots" : null) };
  }
  if (layer === "market") return { fill: ramp(CF.maRamp, +key.slice(1) / (CF.ma.bins - 1)), hatch: null };
  return { fill: ramp(CF.stRamp, +key.slice(1) / CF.stSteps), hatch: null };
}
function hiddenKey(layer, key) {
  if (layer !== "type") return false;
  if (key === "na") return false;
  if (key.charAt(0) === "m") return !!st.hiddenClasses[1] || !!st.hiddenTypes[key.slice(2)];
  if (key === "3") return !!st.hiddenClasses[3];
  var p = key.split(":");
  return !!st.hiddenClasses[p[0]] || !!st.hiddenTypes[p[1]];
}
function orderOf(layer, key, hidden) {
  if (hidden) return 0;
  if (layer !== "type") return key === "na" ? 1 : 2;
  if (key === "3" || key === "na") return 1;
  if (key.charAt(0) === "m") return 2;
  return { "1": 2, "4": 3, "2": 4 }[key.split(":")[0]];
}
function hatchColor(kind) { return { diag: CF.assignedHatch, dots: CF.lastHatch, cross: CF.noTypeHatch, horiz: CF.noValueHatch }[kind]; }
var patCache = {};
function tile(kind, dp, scaleCss) {
  var s = Math.round(8 * dp * (scaleCss || 1)), c = document.createElement("canvas"), g;
  c.width = c.height = s; g = c.getContext("2d");
  g.strokeStyle = g.fillStyle = hatchColor(kind); g.lineWidth = Math.max(1, 0.9 * dp);
  g.beginPath();
  if (kind === "diag" || kind === "cross") {
    g.moveTo(0, s); g.lineTo(s, 0); g.moveTo(-s / 2, s / 2); g.lineTo(s / 2, -s / 2); g.moveTo(s / 2, s * 1.5); g.lineTo(s * 1.5, s / 2);
    if (kind === "cross") { g.moveTo(0, 0); g.lineTo(s, s); g.moveTo(-s / 2, s / 2); g.lineTo(s / 2, s * 1.5); g.moveTo(s / 2, -s / 2); g.lineTo(s * 1.5, s / 2); }
    g.stroke();
  } else if (kind === "dots") {
    g.arc(s / 4, s / 4, 1.1 * dp, 0, 6.2832); g.fill(); g.beginPath(); g.arc(3 * s / 4, 3 * s / 4, 1.1 * dp, 0, 6.2832); g.fill();
  } else if (kind === "horiz") {
    g.moveTo(0, s / 2); g.lineTo(s, s / 2); g.stroke();
  }
  return c;
}
function pattern(kind) {
  var key = kind + "|" + dpr;
  if (!patCache[key]) patCache[key] = ctx.createPattern(tile(kind, dpr), "repeat");
  return patCache[key];
}

/* ---------- группы путей ---------- */
var gcache = {};
function getFeat(i, city) { return city && cityFeat[i] ? cityFeat[i] : natFeat[i]; }
function buildGroups(list, city, layer) {
  var g = {}, k, key, n;
  for (n = 0; n < list.length; n++) {
    i = list[n]; key = keyOf(layer, i);
    if (!g[key]) g[key] = new Path2D();
    g[key].addPath(getFeat(i, city).path);
  }
  return Object.keys(g).map(function (k2) { return { key: k2, path: g[k2] }; });
}
function allGroups(city, layer) {
  var ck = (city ? "c" : "n") + layer + (layer === "type" && monthMode() ? "m" : "");
  if (!gcache[ck]) { var all = []; for (i = 0; i < N; i++) all.push(i); gcache[ck] = buildGroups(all, city, layer); }
  return gcache[ck];
}

/* ---------- отрисовка ---------- */
function setTransform(c, k) { c.setTransform(k, 0, 0, k, view.tx * dpr, view.ty * dpr); }
function inCityMode() { return view.s >= CF.cityPx; }
function draw() {
  raf = 0;
  if (!ready) return;
  var k = view.s * dpr, city = inCityMode();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = CF.bg; ctx.fillRect(0, 0, cv.width, cv.height);
  setTransform(ctx, k);
  var x0 = -view.tx / view.s, y0 = -view.ty / view.s, x1 = (W - view.tx) / view.s, y1 = (H - view.ty) / view.s, vis = [], b;
  for (i = 0; i < N; i++) {
    b = getFeat(i, city).bb;
    if (b[2] < x0 || b[0] > x1 || b[3] < y0 || b[1] > y1) continue;
    vis.push(i);
  }
  var groups = vis.length > 0.7 * N ? allGroups(city, st.layer) : buildGroups(vis, city, st.layer);
  var layer = st.layer, items = groups.map(function (g) {
    var hid = hiddenKey(layer, g.key);
    return { g: g, hid: hid, ord: orderOf(layer, g.key, hid) };
  });
  items.sort(function (a, b2) { return a.ord - b2.ord; });
  ctx.lineJoin = "round"; ctx.lineWidth = CF.border / view.s;
  var pt = new DOMMatrix([1 / k, 0, 0, 1 / k, 0, 0]);
  items.forEach(function (it) {
    var s = it.hid ? { fill: CF.dimFace, hatch: null } : styleOf(layer, it.g.key);
    ctx.fillStyle = s.fill; ctx.strokeStyle = s.fill;
    ctx.fill(it.g.path, "nonzero"); ctx.stroke(it.g.path);
    if (s.hatch) { var p = pattern(s.hatch); p.setTransform(pt); ctx.fillStyle = p; ctx.fill(it.g.path, "nonzero"); }
  });
  drawOverlay();
}
function requestDraw() { if (!raf) raf = requestAnimationFrame(draw); }
function strokeFeat(c, i, px, color) {
  var f = getFeat(i, inCityMode());
  c.lineWidth = px / view.s; c.strokeStyle = color; c.stroke(f.path);
}
function drawOverlay() {
  octx.setTransform(1, 0, 0, 1, 0, 0); octx.clearRect(0, 0, ov.width, ov.height);
  setTransform(octx, view.s * dpr); octx.lineJoin = "round";
  st.hl.forEach(function (i) { var f = getFeat(i, inCityMode()); octx.fillStyle = "rgba(255,235,59,0.55)"; octx.fill(f.path, "nonzero"); strokeFeat(octx, i, 3, "#FFFFFF"); strokeFeat(octx, i, 1.4, "#111111"); });
  if (st.hov >= 0) { strokeFeat(octx, st.hov, 3.5, "#FFFFFF"); strokeFeat(octx, st.hov, 1.6, "#111111"); }
  if (st.sel >= 0) { strokeFeat(octx, st.sel, 5, "#FFFFFF"); strokeFeat(octx, st.sel, 2.4, "#000000"); }
}
function requestOverlay() { if (!oraf) oraf = requestAnimationFrame(function () { oraf = 0; drawOverlay(); }); }

/* ---------- вид ---------- */
function fitScale() { var bw = bbAll[2] - bbAll[0], bh = bbAll[3] - bbAll[1]; return Math.min(W / (bw * 1.04), H / (bh * 1.04)); }
function clampView() {
  var s = view.s, lo = W * 0.05 - bbAll[2] * s, hi = W * 0.95 - bbAll[0] * s;
  view.tx = lo > hi ? (lo + hi) / 2 : Math.min(hi, Math.max(lo, view.tx));
  lo = H * 0.05 - bbAll[3] * s; hi = H * 0.95 - bbAll[1] * s;
  view.ty = lo > hi ? (lo + hi) / 2 : Math.min(hi, Math.max(lo, view.ty));
}
function resetView() {
  view.s = fitScale(); view.tx = W / 2 - (bbAll[0] + bbAll[2]) / 2 * view.s; view.ty = H / 2 - (bbAll[1] + bbAll[3]) / 2 * view.s; view.fit = true; requestDraw();
}
function zoomAt(cx, cy, f) {
  var fs = fitScale(), ns = Math.min(fs * CF.maxZoom, Math.max(fs, view.s * f)), r = ns / view.s;
  view.tx = cx - (cx - view.tx) * r; view.ty = cy - (cy - view.ty) * r; view.s = ns; view.fit = Math.abs(ns - fs) < 1e-9;
  clampView(); hideTip(); requestDraw();
}
function zoomToBox(b, pad) {
  var bw = Math.max(b[2] - b[0], 1e-6), bh = Math.max(b[3] - b[1], 1e-6), fs = fitScale();
  var s = Math.min(fs * CF.maxZoom, Math.max(fs, Math.min(W / (bw * (1 + 2 * pad)), H / (bh * (1 + 2 * pad)))));
  view.s = s; view.tx = W / 2 - (b[0] + b[2]) / 2 * s; view.ty = H / 2 - (b[1] + b[3]) / 2 * s; view.fit = Math.abs(s - fs) < 1e-9;
  clampView(); hideTip(); requestDraw();
}
function resize() {
  var r = wrapEl.getBoundingClientRect();
  var nw = Math.max(1, Math.round(r.width)), nh = Math.max(1, Math.round(r.height)), nd = window.devicePixelRatio || 1;
  if (nw === W && nh === H && nd === dpr) return;
  var wasFit = view.fit, ow = W, oh = H;
  W = nw; H = nh; dpr = nd;
  cv.width = ov.width = Math.round(W * dpr); cv.height = ov.height = Math.round(H * dpr);
  patCache = {};
  if (!ready) return;
  if (wasFit || ow === 0) resetView(); else { view.tx += (W - ow) / 2; view.ty += (H - oh) / 2; clampView(); requestDraw(); }
}

/* ---------- попадание ---------- */
function hitTest(px, py) {
  var city = inCityMode(), wx = (px - view.tx) / view.s, wy = (py - view.ty) / view.s, best = -1, ba = Infinity, f, b, a;
  setTransform(ctx, view.s * dpr);
  for (i = 0; i < N; i++) {
    f = getFeat(i, city); b = f.bb;
    if (wx < b[0] || wx > b[2] || wy < b[1] || wy > b[3]) continue;
    a = (b[2] - b[0]) * (b[3] - b[1]);
    if (a < ba && ctx.isPointInPath(f.path, px * dpr, py * dpr, "nonzero")) { best = i; ba = a; }
  }
  return best;
}
function anchorOf(i) {
  var f = getFeat(i, inCityMode()), b = f.bb, cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2, best = null, bd = Infinity, n = 24, a, c, x, y, d;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  for (a = 0; a <= n; a++) for (c = 0; c <= n; c++) {
    x = b[0] + (b[2] - b[0]) * a / n; y = b[1] + (b[3] - b[1]) * c / n;
    if (ctx.isPointInPath(f.path, x, y, "nonzero")) { d = (x - cx) * (x - cx) + (y - cy) * (y - cy); if (d < bd) { bd = d; best = [x, y]; } }
  }
  return best ? { x: best[0] * view.s + view.tx, y: best[1] * view.s + view.ty } : null;
}

/* ---------- описание МО ---------- */
function typeLabel(t) { return "Тип " + t + " · " + T[t].name; }
function describe(i, full) {
  var cls = M.cls[i], rows = [], notes = [], t = M.typ[i], c;
  rows.push(["Класс:", CLS[cls]]);
  if (cls === 1 && monthMode()) {
    var ml = labelAt(i, st.month), dl = labelAt(i, LAST);
    rows.push(["Метка месяца " + MONTHS[st.month] + ":", typeLabel(ml), ml]);
    rows.push(["Метка декабря 2024 (каноническая):", typeLabel(t), t]);
    if (full && M.dd[i] === 1) rows.push(["Метка траектории декабря 2024:", typeLabel(dl), dl]);
    for (c = 0; c < CATS.length; c++) rows.push([CATS[c] + ", дек. 2024:", fmt(M.sh[i][c], 1) + "%"]);
    if (st.conf0 === undefined) { }
    var cw = APP.conf24[st.month][ml];
    if (cw) notes.push("сопоставление типа в этом месяце неуверенное: " + APP.confWords[cw]);
  } else if (cls === 1) {
    rows.push(["Тип:", typeLabel(t), t]);
    for (c = 0; c < CATS.length; c++) rows.push([CATS[c] + ", дек. 2024:", fmt(M.sh[i][c], 1) + "%"]);
  } else if (cls === 2) {
    rows.push(["Тип:", typeLabel(t), t]);
    rows.push(["Расстояние до ближайшего центроида:", fmt(M.dn[i], 3)]);
    notes.push("тип присвоен по ближайшему центроиду, МО не входил в обучение");
  } else if (cls === 4) {
    rows.push(["Последний доступный месяц:", APP.months[M.lm[i]]]);
    rows.push(["Тип:", typeLabel(t), t]);
    rows.push(["Расстояние до ближайшего центроида:", fmt(M.dn[i], 3)]);
    rows.push(["Согласие метода с месячными метками на выборке в этом месяце:", fmt(M.mag[i], 3)]);
    notes.push("тип по последнему доступному месяцу: данных за декабрь 2024 нет, МО не входил в обучение");
  } else {
    rows.push(["Статус:", STAT[M.st[i]]]);
  }
  if (cls === 2 && monthMode()) notes.push("тип только за декабрь 2024: метки по месяцам для этого МО не считаются");
  if (cls === 4 && monthMode()) notes.push("тип по последнему доступному месяцу: " + APP.months[M.lm[i]] + "; метки по месяцам для этого МО не считаются");
  var showM = full || st.layer === "market", showS = full || st.layer === "stab";
  if (showM) rows.push(["Доступность рынков:", M.ma[i] === null ? "нет значения" : fmt(M.ma[i], 1)]);
  if (showS) rows.push(["Устойчивость типа:", cls === 1 ? M.ms[i] + " из " + CF.stSteps + " месяцев (" + fmt(M.ms[i] / CF.stSteps, 3) + ")" : "слой только для МО выборки"]);
  if (M.dd[i] === 1) notes.push(monthMode() && cls === 1 ? "метка декабря в месячном разбиении отличается от канонической: месячная " + labelAt(i, LAST) + ", каноническая " + t : "метка декабря в месячном разбиении отличается от канонической");
  return { title: M.name[i], sub: APP.regions[M.reg[i]], rows: rows, notes: notes };
}
function fillBox(box, d, withChip) {
  box.textContent = "";
  var h = el(box === tip ? "b" : "h3", null, d.title); box.appendChild(h);
  box.appendChild(el("div", "sub", d.sub));
  var dl = el("div", "rows");
  d.rows.forEach(function (r) {
    var row = el("div", "r"), dt = el("span", "k", r[0]), dd = el("span", "v");
    if (r[2] !== undefined) { var ch = el("span", "chip"); ch.style.background = T[r[2]].color; dd.appendChild(ch); }
    dd.appendChild(document.createTextNode(r[1])); row.appendChild(dt); row.appendChild(dd); dl.appendChild(row);
  });
  box.appendChild(dl);
  d.notes.forEach(function (n) { box.appendChild(el("p", "note", n)); });
}
function showTip(i, x, y) {
  fillBox(tip, describe(i, false)); tip.hidden = false;
  var tw = tip.offsetWidth, th = tip.offsetHeight, nx = x + 14, ny = y + 14;
  if (nx + tw > W - 4) nx = Math.max(4, x - tw - 14);
  if (ny + th > H - 4) ny = Math.max(4, y - th - 14);
  tip.style.left = nx + "px"; tip.style.top = ny + "px";
}
function hideTip() { tip.hidden = true; }
function select(id) {
  var ix = typeof id === "number" && idIdx[id] !== undefined ? idIdx[id] : -1;
  st.sel = ix;
  if (ix < 0) { panel.innerHTML = hintHTML; } else {
    var d = describe(ix, true); fillBox(panel, d);
    var b = el("button", "btn", "Приблизить"); b.type = "button"; b.addEventListener("click", function () { zoomToBox(getFeat(ix, true).bb, 0.5); });
    panel.appendChild(b);
  }
  requestOverlay();
  return ix < 0 ? null : { id: M.id[ix], name: M.name[ix], cls: M.cls[ix], type: M.typ[ix] };
}

/* ---------- поиск ---------- */
var resUl = $("#map-results"), inp = $("#map-search");
function norm(s) { return s.toLowerCase().replace(/ё/g, "е").replace(/\s+/g, " ").trim(); }
var RN = APP.regions.map(function (r) { return r.toLowerCase().replace(/ё/g, "е"); });
function findMatches(q) {
  q = norm(q); if (!q) return [];
  var out = [], nm, words;
  for (i = 0; i < N; i++) {
    nm = NN[i]; if (nm.indexOf(q) < 0) continue;
    words = (" " + nm).indexOf(" " + q) >= 0;
    out.push({ i: i, pr: nm === q ? 0 : (words ? 1 : 2) });
  }
  if (!out.length) { for (i = 0; i < N; i++) if (RN[M.reg[i]].indexOf(q) >= 0) out.push({ i: i, pr: 3 }); }
  out.sort(function (a, b) { return a.pr - b.pr || M.name[a.i].localeCompare(M.name[b.i], "ru") || M.id[a.i] - M.id[b.i]; });
  return out.map(function (o) { return o.i; });
}
function search(text, go) {
  var m = findMatches(text);
  st.hl = m.slice(0, 500);
  resUl.textContent = "";
  m.slice(0, CF.searchMax).forEach(function (ix) {
    var li = el("li"), b = el("button"); b.type = "button"; b.appendChild(document.createTextNode(M.name[ix])); b.appendChild(el("small", null, APP.regions[M.reg[ix]]));
    b.addEventListener("click", function () { resUl.hidden = true; select(M.id[ix]); zoomToBox(getFeat(ix, true).bb, 0.6); });
    li.appendChild(b); resUl.appendChild(li);
  });
  resUl.hidden = m.length === 0 || !text;
  if (go && m.length) {
    var byRegion = NN[m[0]].indexOf(norm(text)) < 0;
    var set = (m.length <= CF.searchZoomMax || byRegion) ? m.map(function (ix) { return getFeat(ix, true); }) : [getFeat(m[0], true)];
    zoomToBox(unionBox(set), 0.25);
    if (m.length === 1) select(M.id[m[0]]);
  }
  requestOverlay();
  return { n: m.length, ids: m.slice(0, 10).map(function (ix) { return M.id[ix]; }) };
}

/* ---------- слои и легенда ---------- */
var legEl = $("#map-legend");
function swatch(style, w, h) {
  var c = document.createElement("canvas"), g, s = 2; c.width = w * s; c.height = h * s; c.style.width = w + "px"; c.style.height = h + "px"; g = c.getContext("2d");
  g.fillStyle = style.fill; g.fillRect(0, 0, c.width, c.height);
  if (style.hatch) { var tl = tile(style.hatch, s, 0.75), pt = g.createPattern(tl, "repeat"); g.fillStyle = pt; g.fillRect(0, 0, c.width, c.height); }
  return c;
}
function count(fn) { var n = 0; for (i = 0; i < N; i++) if (fn(i)) n++; return n; }
function legendItem(style, text, pressed, onClick, title) {
  var b = el("button", "leg-item"); b.type = "button"; b.title = title || ""; b.setAttribute("aria-pressed", pressed ? "true" : "false");
  b.appendChild(swatch(style, 18, 14)); b.appendChild(el("span", null, text)); b.addEventListener("click", onClick); return b;
}
function buildLegend() {
  legEl.textContent = "";
  if (st.layer === "type" && monthMode()) {
    legEl.appendChild(el("h3", null, "Тип по метке месяца " + MONTHS[st.month] + " (клик включает и выключает тип; в скобках: МО выборки)"));
    T.forEach(function (t) {
      var n1 = count(function (k) { return M.cls[k] === 1 && labelAt(k, st.month) === t.id; }), cw = APP.conf24[st.month][t.id];
      var b = legendItem({ fill: t.color }, "Тип " + t.id + " · " + t.name + " (" + n1 + ")", !st.hiddenTypes[t.id], function () { toggleType(t.id); }, "МО выборки с этой меткой месяца; источник: kmeans_k6_trajectories.parquet");
      if (cw) { var w = el("span", "warn", "⚠"); w.title = "сопоставление типа в этом месяце неуверенное: " + APP.confWords[cw] + " (kmeans_k6_matching.parquet)"; w.setAttribute("aria-label", w.title); b.appendChild(w); }
      legEl.appendChild(b);
    });
    var wn = el("p", "muted", "⚠ — сопоставление типа в этом месяце неуверенное (умеренно или неоднозначно)"); wn.style.margin = "6px 0"; legEl.appendChild(wn);
    var st2 = el("div", "leg-static"); st2.appendChild(swatch({ fill: CF.noValueFace, hatch: "horiz" }, 18, 14));
    st2.appendChild(el("span", null, "тип только за декабрь 2024 (МО класса 2: " + count(function (k) { return M.cls[k] === 2; }) + " МО; метки по месяцам не считаются)")); legEl.appendChild(st2);
    var st4 = el("div", "leg-static"); st4.appendChild(swatch({ fill: CF.noValueFace, hatch: "horiz" }, 18, 14));
    st4.appendChild(el("span", null, "тип по последнему доступному месяцу (МО класса 4: " + count(function (k) { return M.cls[k] === 4; }) + " МО; метки по месяцам не считаются)")); legEl.appendChild(st4);
    var st3 = el("div", "leg-static"); st3.appendChild(swatch({ fill: CF.noTypeFace, hatch: "cross" }, 18, 14));
    st3.appendChild(el("span", null, "тип не присвоен (" + count(function (k) { return M.cls[k] === 3; }) + " МО): статус указан в подсказке МО")); legEl.appendChild(st3);
    if (anyHidden()) { var rb2 = el("button", "btn", "Показать все типы"); rb2.type = "button"; rb2.style.marginTop = "8px"; rb2.addEventListener("click", clearFilter); legEl.appendChild(rb2); }
  } else if (st.layer === "type") {
    var h = el("h3", null, "Тип (клик включает и выключает тип; в скобках: МО выборки + присвоено вне выборки + по последнему месяцу)"); legEl.appendChild(h);
    T.forEach(function (t) {
      var n1 = count(function (k) { return M.typ[k] === t.id && M.cls[k] === 1; }), n2 = count(function (k) { return M.typ[k] === t.id && M.cls[k] === 2; }), n4 = count(function (k) { return M.typ[k] === t.id && M.cls[k] === 4; });
      legEl.appendChild(legendItem({ fill: t.color }, "Тип " + t.id + " · " + t.name + " (" + n1 + " + " + n2 + " + " + n4 + ")", !st.hiddenTypes[t.id],
        function () { toggleType(t.id); }, "МО выборки + присвоено вне выборки + по последнему месяцу; источник: layer_data_24f.parquet, колонки class и type"));
    });
    var h2 = el("h3", null, "Классы МО"); h2.style.marginTop = "10px"; legEl.appendChild(h2);
    var cl = [
      [1, { fill: "#8C8C8C" }, "тип из выборки (декабрь 2024)"],
      [2, { fill: "#8C8C8C", hatch: "diag" }, "тип присвоен вне выборки (цвет — присвоенный тип; МО не входили в обучение)"],
      [4, { fill: "#8C8C8C", hatch: "dots" }, "тип по последнему доступному месяцу (данных за декабрь 2024 нет; МО не входили в обучение)"],
      [3, { fill: CF.noTypeFace, hatch: "cross" }, "тип не присвоен: нет данных за декабрь 2024, МО нет в наборе данных или слишком далеко от центров типов"]];
    cl.forEach(function (c) {
      legEl.appendChild(legendItem(c[1], c[2] + " (" + count(function (k) { return M.cls[k] === c[0]; }) + ")", !st.hiddenClasses[c[0]], function () { toggleClass(c[0]); }, "число МО: layer_data_24f.parquet, колонка class"));
    });
    if (anyHidden()) { var rb = el("button", "btn", "Показать все типы и классы"); rb.type = "button"; rb.style.marginTop = "8px"; rb.addEventListener("click", clearFilter); legEl.appendChild(rb); }
  } else {
    var isM = st.layer === "market", stops = isM ? CF.maRamp : CF.stRamp;
    legEl.appendChild(el("h3", null, isM ? "Доступность рынков (индекс 2024; шкала " + (CF.ma.kind === "log" ? "логарифмическая" : "линейная") + ")" : "Устойчивость типа (доля из " + CF.stSteps + " месяцев с тем же типом)"));
    var g = el("div", "grad"); g.style.background = "linear-gradient(to right," + stops.join(",") + ")"; legEl.appendChild(g);
    var tk = el("div", "ticks"), vals = isM ? [CF.ma.min, 200, 300, 500, CF.ma.max] : [0, 0.25, 0.5, 0.75, 1];
    vals.forEach(function (v) {
      if (isM && v !== CF.ma.min && v !== CF.ma.max && (v <= CF.ma.min || v >= CF.ma.max)) return;
      var sp = el("span", null, isM ? (v === CF.ma.min || v === CF.ma.max ? fmt(v, 1) : String(v)) : String(v));
      var t = isM ? maT(v) : v; sp.style.left = (t * 100) + "%"; if (t > 0.9) sp.style.transform = "translateX(-100%)"; else if (t < 0.1) sp.style.transform = "none"; tk.appendChild(sp);
    });
    legEl.appendChild(tk);
    var na = el("div", "leg-static"); na.appendChild(swatch({ fill: CF.noValueFace, hatch: "horiz" }, 18, 14));
    na.appendChild(el("span", null, isM ? "нет значения (" + count(function (k) { return M.ma[k] === null; }) + " МО)" : "слой только для МО выборки (" + count(function (k) { return M.cls[k] !== 1; }) + " МО)"));
    legEl.appendChild(na);
  }
}
function anyHidden() { return Object.keys(st.hiddenTypes).length > 0 || Object.keys(st.hiddenClasses).length > 0; }
function toggleType(t) { if (st.hiddenTypes[t]) delete st.hiddenTypes[t]; else st.hiddenTypes[t] = true; buildLegend(); requestDraw(); return hiddenState(); }
function toggleClass(c) { if (st.hiddenClasses[c]) delete st.hiddenClasses[c]; else st.hiddenClasses[c] = true; buildLegend(); requestDraw(); return hiddenState(); }
function clearFilter() { st.hiddenTypes = {}; st.hiddenClasses = {}; buildLegend(); requestDraw(); }
function hiddenState() { return { types: Object.keys(st.hiddenTypes).map(Number).sort(), classes: Object.keys(st.hiddenClasses).map(Number).sort() }; }
function setLayer(name) {
  if (["type", "market", "stab"].indexOf(name) < 0) throw new Error("layer " + name);
  st.layer = name; $$("input[name=layer]").forEach(function (r) { r.checked = r.value === name; });
  $$(".layer-note").forEach(function (n) { n.hidden = n.getAttribute("data-layer") !== name; });
  if (name !== "type") stopPlay();
  syncMonthUi(); buildLegend(); requestDraw();
  if (st.hov >= 0 || st.sel >= 0) { if (st.sel >= 0) select(M.id[st.sel]); }
  return name;
}

/* ---------- селектор месяца ---------- */
var mRange = $("#month-range"), mLabel = $("#month-label"), mNote = $("#month-note"), mPlay = $("#month-play"), mPrev = $("#month-prev"), mNext = $("#month-next");
function syncMonthUi() {
  var on = st.layer === "type";
  [mRange, mPrev, mNext, mPlay].forEach(function (e) { e.disabled = !on; });
  mRange.value = st.month; mLabel.textContent = MONTHS[st.month];
  mNote.textContent = !on ? "селектор месяца относится только к слою «Тип»; слои «Доступность рынков» и «Устойчивость типа» от месяца не зависят"
    : (st.month === LAST ? "декабрь 2024: канонические метки" : "метки траектории месяца; МО класса 2 — светлая штриховка: тип только за декабрь 2024; МО класса 4 — светлая штриховка: тип по последнему доступному месяцу; МО без типа остаются «тип не присвоен»");
}
function setMonth(v) {
  var m = typeof v === "string" ? MONTHS.indexOf(v) : v;
  if (m < 0 || m >= NM || m === undefined) throw new Error("month " + v);
  st.month = m; for (var k in gcache) if (k.slice(-1) === "m") delete gcache[k];
  syncMonthUi(); buildLegend(); hideTip(); if (st.sel >= 0) select(M.id[st.sel]); requestDraw(); return MONTHS[m];
}
function stopPlay() { if (playTimer) { clearInterval(playTimer); playTimer = 0; } mPlay.textContent = "▶ играть"; mPlay.setAttribute("aria-label", "Играть"); }
function togglePlay() {
  if (playTimer) { stopPlay(); return; }
  mPlay.textContent = "❚❚ пауза"; mPlay.setAttribute("aria-label", "Пауза");
  playTimer = setInterval(function () { setMonth((st.month + 1) % NM); }, APP.playMs);
}
mRange.addEventListener("input", function () { setMonth(+mRange.value); });
mPrev.addEventListener("click", function () { setMonth((st.month + NM - 1) % NM); });
mNext.addEventListener("click", function () { setMonth((st.month + 1) % NM); });
mPlay.addEventListener("click", togglePlay);

/* ---------- события ---------- */
var ptrs = {}, nptr = 0, moved = 0, pinch = null, hovRaf = 0, lastMouse = null;
function localXY(e) { var r = cv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
function updateHover() {
  hovRaf = 0; if (!lastMouse || nptr > 0) return;
  var ix = hitTest(lastMouse.x, lastMouse.y);
  if (ix !== st.hov) { st.hov = ix; requestOverlay(); }
  if (ix >= 0) showTip(ix, lastMouse.x, lastMouse.y); else hideTip();
}
cv.addEventListener("pointerdown", function (e) {
  cv.setPointerCapture(e.pointerId); var p = localXY(e); ptrs[e.pointerId] = p; nptr = Object.keys(ptrs).length; moved = 0; hideTip();
  if (nptr === 2) { var k = Object.keys(ptrs), a = ptrs[k[0]], b = ptrs[k[1]]; pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 }; }
});
cv.addEventListener("pointermove", function (e) {
  var p = localXY(e);
  if (ptrs[e.pointerId]) {
    var prev = ptrs[e.pointerId]; ptrs[e.pointerId] = p;
    if (nptr === 1) { view.tx += p.x - prev.x; view.ty += p.y - prev.y; moved += Math.abs(p.x - prev.x) + Math.abs(p.y - prev.y); view.fit = false; clampView(); requestDraw(); }
    else if (nptr === 2 && pinch) {
      var k = Object.keys(ptrs), a = ptrs[k[0]], b = ptrs[k[1]], d = Math.hypot(a.x - b.x, a.y - b.y), cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
      view.tx += cx - pinch.cx; view.ty += cy - pinch.cy; moved += 10; zoomAt(cx, cy, d / pinch.d); pinch = { d: d, cx: cx, cy: cy };
    }
  } else if (e.pointerType === "mouse") { lastMouse = p; if (!hovRaf) hovRaf = requestAnimationFrame(updateHover); }
});
function endPtr(e) {
  var p = ptrs[e.pointerId]; if (!p) return;
  delete ptrs[e.pointerId]; var was = nptr; nptr = Object.keys(ptrs).length; pinch = null;
  if (was === 1 && moved < 5 && e.type === "pointerup") { var ix = hitTest(p.x, p.y); select(ix >= 0 ? M.id[ix] : null); }
}
cv.addEventListener("pointerup", endPtr); cv.addEventListener("pointercancel", endPtr);
cv.addEventListener("pointerleave", function () { lastMouse = null; if (st.hov >= 0) { st.hov = -1; requestOverlay(); } hideTip(); });
cv.addEventListener("wheel", function (e) {
  e.preventDefault(); var p = localXY(e), dy = e.deltaY * (e.deltaMode === 1 ? 16 : (e.deltaMode === 2 ? 400 : 1));
  zoomAt(p.x, p.y, Math.exp(-dy * 0.0018));
}, { passive: false });
$("#btn-zoom-in").addEventListener("click", function () { zoomAt(W / 2, H / 2, 1.6); });
$("#btn-zoom-out").addEventListener("click", function () { zoomAt(W / 2, H / 2, 1 / 1.6); });
$("#btn-reset").addEventListener("click", function () { resetView(); select(null); st.hl = []; inp.value = ""; resUl.hidden = true; });
$("#btn-moscow").addEventListener("click", function () { zoomToBox(cityBox.moscow, 0.06); });
$("#btn-spb").addEventListener("click", function () { zoomToBox(cityBox.spb, 0.06); });
$$("input[name=layer]").forEach(function (r) { r.addEventListener("change", function () { if (r.checked) setLayer(r.value); }); });
inp.addEventListener("input", function () { search(inp.value, false); });
inp.addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); search(inp.value, true); resUl.hidden = true; } else if (e.key === "Escape") { resUl.hidden = true; } });
document.addEventListener("click", function (e) { if (!e.target.closest(".search")) resUl.hidden = true; });
window.addEventListener("resize", function () { resize(); });

/* ---------- запуск ---------- */
function fallback(msg) {
  $("#map-ui").hidden = true; var fb = $("#map-fallback"); fb.hidden = false; $("#map-fallback-msg").textContent = msg;
  var im = $("#fig2-img"), t = $("#map-fallback-img"); if (im && t) { t.src = im.src; t.hidden = false; $("#map-fallback-cap").hidden = false; }
}
/* ---------- CSV: типы МО выборки по месяцам ---------- */
function csvCell(v) { var t = String(v); return /[",\n\r]/.test(t) ? "\"" + t.replace(/"/g, "\"\"") + "\"" : t; }
function buildCsv() {
  var nd = 0, k, m, rows = [], lines = [], line;
  for (k = 0; k < N; k++) if (M.cls[k] === 1 && M.dd[k]) nd++;
  lines.push("# расшифровка номеров типов (технические имена):");
  T.forEach(function (t) { lines.push("# " + t.id + " = " + t.tech.replace(/,/g, " /")); });
  lines.push("# метки по месяцам получены из помесячных разбиений и сопоставлены с декабрём 2024 / у " + nd + " МО метка декабря в траекториях отличается от канонической / интерпретационные имена типов в этом файле не используются");
  lines.push(["territory_id", "name", "region_name"].concat(MONTHS).map(csvCell).join(","));
  for (k = 0; k < N; k++) {
    if (M.cls[k] !== 1) continue;
    line = [M.id[k], M.name[k], APP.regions[M.reg[k]]];
    for (m = 0; m < NM; m++) line.push(labelAt(k, m));
    rows.push(line.map(csvCell).join(","));
  }
  return "\uFEFF" + lines.concat(rows).join("\n") + "\n";
}
$("#csv-btn").addEventListener("click", function () {
  var blob = new Blob([buildCsv()], { type: "text/csv;charset=utf-8" }), url = URL.createObjectURL(blob), a = document.createElement("a");
  a.href = url; a.download = CF.csvName; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
});

window.__test = {
  ready: false,
  buildCsv: function () { return buildCsv(); },
  hitTest: function (x, y) { var ix = hitTest(x, y); return ix < 0 ? null : { id: M.id[ix], name: M.name[ix], cls: M.cls[ix], type: M.typ[ix], region: APP.regions[M.reg[ix]] }; },
  select: function (id) { return select(id); },
  search: function (text) { var r = search(text, true); resUl.hidden = true; return r; },
  applyFilter: function (type) { return toggleType(type); },
  applyClassFilter: function (c) { return toggleClass(c); },
  setLayer: function (n) { return setLayer(n); },
  zoomTo: function (id) { var ix = idIdx[id]; zoomToBox(getFeat(ix, true).bb, 0.6); return { scale: view.s, city: inCityMode() }; },
  anchor: function (id) { return anchorOf(idIdx[id]); },
  reset: function () { resetView(); },
  view: function () { return { s: view.s, tx: view.tx, ty: view.ty, W: W, H: H, dpr: dpr, city: inCityMode(), fit: fitScale() }; },
  state: function () { return { layer: st.layer, sel: st.sel >= 0 ? M.id[st.sel] : null, hidden: hiddenState(), hl: st.hl.length }; },
  paintedPixels: function () {
    var d = ctx.getImageData(0, 0, cv.width, cv.height).data, bg = hex(CF.bg), n = 0, k;
    for (k = 0; k < d.length; k += 4) if (Math.abs(d[k] - bg[0]) + Math.abs(d[k + 1] - bg[1]) + Math.abs(d[k + 2] - bg[2]) > 6) n++;
    return { painted: n, total: cv.width * cv.height };
  },
  pixelAt: function (x, y) { var d = ctx.getImageData(Math.round(x * dpr), Math.round(y * dpr), 1, 1).data; return [d[0], d[1], d[2]]; },
  tooltipText: function (id) { var d = describe(idIdx[id], false); return d.title + " | " + d.sub + " | " + d.rows.map(function (r) { return r[0] + " " + r[1]; }).join("; ") + " | " + d.notes.join("; "); },
  panelText: function () { return panel.textContent; },
  legendText: function () { return legEl.textContent; },
  setMonth: function (m) { return setMonth(m); },
  getLabel: function (id, m) { var i = idIdx[id], k = typeof m === "string" ? MONTHS.indexOf(m) : m; return { label: labelAt(i, k), canonical: M.cls[i] === 1 ? M.typ[i] : null, conf: M.cls[i] === 1 ? APP.conf24[k][labelAt(i, k)] : null }; },
  play: function () { togglePlay(); return !!playTimer; },
  month: function () { return MONTHS[st.month]; },
  keyOf: function (layer, id) { return keyOf(layer, idIdx[id]); }
};
if (typeof DecompressionStream === "undefined" || typeof Path2D === "undefined") {
  fallback("Интерактивная карта недоступна: браузер не поддерживает DecompressionStream или Path2D.");
} else {
  var loading = el("div", "loading", "Загрузка карты…"); wrapEl.appendChild(loading);
  resize();
  Promise.all([inflate($("#geo-national").textContent), inflate($("#geo-cities").textContent)]).then(function (r) {
    decodeGeo(r[0], natFeat, false); decodeGeo(r[1], cityFeat, true); LBL = b64bytes($("#month-labels").textContent); if (LBL.length !== N * NM) throw new Error("метки по месяцам");
    for (i = 0; i < N; i++) if (!natFeat[i]) throw new Error("нет геометрии МО " + M.id[i]);
    bbAll = unionBox(natFeat);
    ["moscow", "spb"].forEach(function (c) { cityBox[c] = unionBox(cityFeat.filter(function (f) { return f && f.city === c; })); });
    ready = true; loading.remove(); resize(); resetView(); buildLegend(); setLayer("type");
    draw(); window.__test.ready = true; document.documentElement.setAttribute("data-map-ready", "1");
  }).catch(function (err) { fallback("Интерактивная карта недоступна: " + err.message); });
}
})();
'''


# ============================================================================ сборка страницы: вспомогательные блоки
STATUS_LABELS = {
    "no_data_in_dataset": "МО нет в наборе данных",
    "no_dec2024_no_type": "нет данных за декабрь 2024; тип по последнему доступному месяцу не присвоен",
    "too_far": "слишком далеко от центров типов (расстояние больше максимума по выборке)",
}
CLASS_LABELS = {1: "МО выборки: тип канонического разбиения, декабрь 2024", 2: "тип присвоен вне выборки",
                4: "тип по последнему доступному месяцу", 3: "тип не присвоен"}


def lst(series: pd.Series) -> list:
    out = []
    for v in series.tolist():
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        elif isinstance(v, (np.integer,)):
            out.append(int(v))
        elif isinstance(v, (np.floating,)):
            out.append(float(v))
        elif isinstance(v, (bool, np.bool_)):
            out.append(int(v))
        else:
            out.append(v)
    return out


def contrast(fg: str, bg: str) -> float:
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    a, b = sorted([lum(fg), lum(bg)], reverse=True)
    return (a + 0.05) / (b + 0.05)


def inline_md(text: str) -> str:
    s = esc(text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", s)


def md_to_html(md: str) -> str:
    """Минимальный разбор md (заголовки, списки, абзацы, код в обратных кавычках); ссылки не создаются: адреса остаются текстом."""
    if re.search(r"\]\(", md):
        stop("в md есть ссылка в формате [текст](адрес): страница не создаёт внешних ссылок")
    out, para, ul = [], [], []

    def flush():
        nonlocal para, ul
        if para:
            out.append("<p>" + inline_md(" ".join(para)) + "</p>")
            para = []
        if ul:
            out.append("<ul>" + "".join(f"<li>{inline_md(x)}</li>" for x in ul) + "</ul>")
            ul = []
    for ln in md.split("\n"):
        if ln.startswith("# "):
            flush(); out.append(f"<h3>{inline_md(ln[2:])}</h3>")
        elif ln.startswith("## "):
            flush(); out.append(f"<h4>{inline_md(ln[3:])}</h4>")
        elif ln.startswith("- "):
            if para:
                flush()
            ul.append(ln[2:].strip())
        elif not ln.strip():
            flush()
        else:
            if ul:
                flush()
            para.append(ln.strip())
    flush()
    return "\n".join(out)


def plain_words(md: str) -> list[str]:
    t = re.sub(r"^(#+ |- )", "", md, flags=re.M).replace("`", "")
    return t.split()


def html_words(h: str) -> list[str]:
    h = re.sub(r"</?code>", "", h)
    return html.unescape(re.sub(r"<[^>]+>", " ", h)).split()


def svg_inline(text: str, n: int) -> str:
    """SVG фигуры внутрь страницы: без декларации и DOCTYPE, метаданных и пространств имён; идентификаторы получают префикс."""
    t = re.sub(r"<\?xml[^>]*\?>", "", text)
    t = re.sub(r"<!DOCTYPE[^>]*>", "", t, flags=re.S)
    t = re.sub(r"<metadata>.*?</metadata>", "", t, flags=re.S)
    m = re.search(r"<svg\b[^>]*>", t, flags=re.S)
    if not m:
        stop(f"в SVG фигуры {n} нет корневого элемента")
    vb = re.search(r'viewBox="([^"]+)"', m.group(0))
    if not vb:
        stop(f"в SVG фигуры {n} нет viewBox")
    root = (f'<svg id="fig{n}" class="fig" viewBox="{vb.group(1)}" role="img" aria-label="Фигура {n}" '
            f'preserveAspectRatio="xMidYMid meet">')
    t = t[:m.start()] + root + t[m.end():]
    pref = f"f{n}-"
    t = re.sub(r'\bid="([^"]+)"', lambda x: f'id="{pref}{x.group(1)}"', t)
    t = re.sub(r'url\(#([^)]+)\)', lambda x: f'url(#{pref}{x.group(1)})', t)
    t = re.sub(r'href="#([^"]+)"', lambda x: f'href="#{pref}{x.group(1)}"', t)
    t = t.replace("*{stroke-linejoin", f"#fig{n} *{{stroke-linejoin")
    if re.search(r"<script|https?://|xmlns", t):
        stop(f"в SVG фигуры {n} остались скрипты, внешние адреса или пространства имён")
    return t.strip()


def card_html(t: int, ctx: dict, texts: dict) -> str:
    c = ctx["cards"][t]
    nm = ctx["names"][t]
    rows = []
    for cat in CATS:
        r = c["cats"][cat]
        mark = r["mark"] or "—"
        rows.append(
            f'<tr><td>{esc(cat)}</td>'
            f'<td>{num(f"{r["median"]:.1f}", "type_portraits.parquet, блок «расходы»: медиана")}</td>'
            f'<td>{num(f"{r["mean"]:.1f}", "mirkin_rule_27.parquet: среднее по типу, %")}</td>'
            f'<td>{num(f"{r["gmean"]:.1f}", "mirkin_rule_27.parquet: общее среднее, %")}</td>'
            f'<td>{num(f"{r["sz"]:+.2f}", "type_portraits.parquet, блок «признаки профиля»: σ по долям")}</td>'
            f'<td>{num(f"{r["sa"]:+.2f}", "type_portraits.parquet, блок «признаки профиля»: σ по лог-отношению")}</td>'
            f'<td>{esc(mark)}{(" · граница порога: меньшее |σ| " + f"{r["border"]:.2f}") if r["border"] is not None else ""}</td></tr>')
    regs = "; ".join(f'{esc(n)} — {num(int(v), "type_portraits.parquet, блок «состав»: регион, МО")}' for n, v in c["regions"])
    g6 = c["g6"]
    return f'''<article class="card" id="type-{t}" style="--tc:{L["TYPE_COLORS"][t]}">
<h3>Тип {t} · {esc(nm["name"])}</h3>
<p class="tech">техническое имя: {esc(nm["tech"])}</p>
<p class="status">имя и статус: {esc(nm["status"])}</p>
{prose(texts, f"type_card_text_{t}")}
<p class="stat">МО типа: {num(c["n"], "type_portraits.parquet, блок «состав»: МО типа")} — {num(f'{c["share"]:.1f}%', "type_portraits.parquet, блок «состав»: доля от 2004 МО, %")} выборки; регионов: {num(c["regions_n"], "type_portraits.parquet, блок «состав»: регионов")}</p>
<p class="stat">Регионы с наибольшим числом МО типа: {regs}</p>
<p class="stat">в тексте карточки используются средние, в таблице показаны и средние, и медианы</p>
<div class="tablewrap"><table>
<thead><tr><th>категория, % расходов (декабрь 2024)</th><th title="шаг 23: type_portraits.parquet">медиана по МО типа (шаг 23)</th><th title="шаг 27: mirkin_rule_27.parquet">среднее по МО типа (шаг 27)</th><th title="шаг 27: mirkin_rule_27.parquet">общее среднее по 2004 МО (шаг 27)</th><th>σ по долям</th><th>σ по log(доля / «прочее»)</th><th>пометка</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p class="stat">prediction strength типа: {num(f'{c["ps"]:.2f}', "type_portraits.parquet, блок «надёжность»: PS типа")}; уверенных сопоставлений: {num(c["conf"], "type_portraits.parquet, блок «надёжность»: уверенных сопоставлений из 23")} из {num(c["conf_n"], "type_portraits.parquet, блок «надёжность»: n")} (шаг 23)</p>
<p class="stat">Г6: S по типу {num(f'{g6["s"]:.2f}', "type_portraits.parquet, блок «надёжность»: Г6: S по типу")} при базовом уровне {num(f'{g6["base"]:.2f}', "type_portraits.parquet, блок «надёжность»: Г6: базовый уровень")}; переходов {num(g6["n"], "type_portraits.parquet, блок «надёжность»: Г6: переходов")}</p>
</article>'''


def checks_html(tabs: dict, rep: dict, lists: list, ctrl_note: str, texts: dict) -> str:
    src = "notebooks/26_hypothesis_summary.md; значения сверены с data/processed/hypothesis_summary.parquet"

    def table(kind_title: str, tab: dict, cls: str = "dt rt") -> str:
        head = "".join(f"<th>{esc(h)}</th>" for h in tab["header"])
        body = []
        for r in tab["rows"]:
            tds = "".join(f'<td data-label="{esc(h)}">{esc(v)}</td>' for h, v in zip(tab["header"], r))
            body.append(f"<tr>{tds}</tr>")
        return (f'<h3>{esc(kind_title)}</h3><div class="tablewrap"><table class="{cls}" title="источник: {esc(src)}">'
                f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")
    parts = [ctrl_note,
             table("Проверки внешней валидности", tabs["main"]),
             table("Учёт возраста (Г7)", tabs["g7"]),
             table("Г4 по отраслям", tabs["g4"]),
             table("Дополнительное семейство (Г10–Г13): проверки", tabs["sup"]),
             table("Дополнительное семейство (Г10–Г13): учёт показателя предложения", tabs["supd"])]
    for title, items in lists:
        if not items:
            continue
        parts.append(f"<h3>{esc(title)}</h3><ul>" + "".join(f"<li>{esc(x)}</li>" for x in items) + "</ul>")
    parts.append(f'<p class="srcnote">Источник: {esc(src)}. Тексты таблиц и списков скопированы без изменений.</p>')
    return "\n".join(parts)


def coverage_html(cov: pd.DataFrame, empty: list[str]) -> str:
    cols = [("region_name", "region_name", "text"), ("n_mo", "n_mo", "num"), ("n_sample", "n_sample", "num"),
            ("n_assigned", "n_assigned", "num"), ("n_last_month", "n_last_month", "num"), ("n_no_type", "n_no_type", "num"),
            ("share_no_type", "share_no_type", "num")]
    head = "".join(
        f'<th data-type="{ty}"{" aria-sort=\"descending\"" if c == "share_no_type" else ""} class="{"r" if ty == "num" else ""}">'
        f'<button type="button" title="сортировка по колонке {c}">{esc(lab)}</button></th>' for c, lab, ty in cols)
    body = []
    for _, r in cov.iterrows():
        sh = float(r["share_no_type"])
        tds = [f'<td data-v="{esc(r["region_name"])}" title="region_coverage_24f.parquet: region_name">{esc(r["region_name"])}</td>']
        for c in ("n_mo", "n_sample", "n_assigned", "n_last_month", "n_no_type"):
            tds.append(f'<td class="r" data-v="{int(r[c])}" title="region_coverage_24f.parquet: {c}">{int(r[c])}</td>')
        tds.append(f'<td data-v="{sh!r}" title="region_coverage_24f.parquet: share_no_type = {sh:.6f}"><div class="bar"><i style="width:{sh * 100:.1f}%"></i><span>{sh * 100:.1f}%</span></div></td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    lst_empty = ", ".join(esc(x) for x in empty)
    return (f'<div class="covwrap"><table class="dt sortable" title="источник: data/processed/region_coverage_24f.parquet">'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
            f'<p class="srcnote">Регионы без МО в рабочей выборке (n_sample = 0), {num(len(empty), "region_coverage_24f.parquet: число строк с n_sample = 0")}: {lst_empty}.</p>')


# ============================================================================ основной расчёт
def build_context() -> dict:
    missing_figs = check_inputs()
    frozen_before = check_frozen("до")
    texts, texts_new = load_texts()
    ctx: dict = {"missing_figs": missing_figs, "frozen_before": frozen_before, "texts": texts, "texts_new": texts_new}
    th = importlib.import_module(L["PROFILE_Z_THRESHOLD_FROM"]).PROFILE_Z_THRESHOLD

    # ---------- МО: layer_data_24f ----------
    Lr = pd.read_parquet(P(L["LAYER_PATH"])).sort_values("territory_id").reset_index(drop=True)
    ctl("МО во встроенных данных (layer_data_24f.parquet)", len(Lr), L["EXPECT_ACTIVE"])
    cc = {int(k): int(v) for k, v in Lr["class"].value_counts().items()}
    exp_cc = {int(k): v for k, v in L["EXPECT_CLASSES"].items()}
    ctl("классы 1 / 2 / 4 / 3 (число МО)", {k: cc[k] for k in CLASS_ORDER}, {k: exp_cc[k] for k in CLASS_ORDER})
    ctl("сумма классов 1 + 2 + 4 + 3", sum(cc.values()), L["EXPECT_ACTIVE"])
    ctl("класс 1 равен 2004, класс 2 равен 98", (cc[1], cc[2]), (2004, 98))
    by1 = [int(((Lr["class"] == 1) & (Lr["type"] == t)).sum()) for t in range(6)]
    by2 = [int(((Lr["class"] == 2) & (Lr["type"] == t)).sum()) for t in range(6)]
    ctl("класс 1 по типам 0–5", by1, L["EXPECT_SAMPLE_BY_TYPE"])
    ctl("класс 2 по типам 0–5", by2, L["EXPECT_ASSIGNED_BY_TYPE"])
    ma_ok = int(Lr["market_access"].notna().sum())
    ctl("слой «Доступность рынков»: МО со значением", ma_ok, L["EXPECT_MA_NONNULL"])
    ctl("слой «Доступность рынков»: МО без значения", len(Lr) - ma_ok, L["EXPECT_MA_NULL"])
    lo, hi = L["EXPECT_MA_RANGE"]
    ctl("market_access: диапазон (допуск 0.05)", (round(float(Lr.market_access.min()), 1), round(float(Lr.market_access.max()), 1)), (lo, hi))
    ctl("шкала слоя «Доступность рынков» совпадает с диапазоном данных", (L["MARKET_ACCESS_SCALE"]["min"], L["MARKET_ACCESS_SCALE"]["max"]), (lo, hi))
    ctl("слой «Устойчивость типа»: МО со значением", int(Lr["stable_share"].notna().sum()), L["EXPECT_STABLE_NONNULL"])
    ctl("stable_share только у класса 1", int(Lr.loc[Lr["class"] != 1, "stable_share"].notna().sum()), 0)
    ctl("stable_share в диапазоне 0–1", (float(Lr.stable_share.min()), float(Lr.stable_share.max())), (0.0, 1.0))
    ctl("stable_share кратна 1/24 (макс. отклонение)", round(float(np.abs(Lr.stable_share.dropna() * L["STABILITY_STEPS"] - np.rint(Lr.stable_share.dropna() * L["STABILITY_STEPS"])).max()), 9), 0.0)
    ctl("dec_label_differs = true (только класс 1)", (int(Lr.dec_label_differs.fillna(False).sum()), int(Lr.loc[Lr["class"] != 1, "dec_label_differs"].notna().sum())), (L["EXPECT_DEC_DIFFERS"], 0))
    lab = pd.read_parquet(P(L["LABELS_PATH"]))
    s1 = Lr[Lr["class"] == 1].set_index("territory_id")["type"].astype(int)
    ctl("класс 1 = kmeans_labels_final.parquet (id и типы совпадают)", bool((lab.set_index("territory_id")["cluster"].sort_index().to_numpy() == s1.sort_index().to_numpy()).all()
        and set(lab["territory_id"]) == set(s1.index)), True)
    out = pd.read_parquet(P(L["OUTSIDE_PATH"]))
    a2 = out[out["status"] == "assigned"].set_index("territory_id")["type"].astype(int)
    c2 = Lr[Lr["class"] == 2].set_index("territory_id")["type"].astype(int)
    ctl("класс 2 = types_outside_sample.parquet (assigned: id и типы)", bool(set(a2.index) == set(c2.index) and (a2.sort_index().to_numpy() == c2.sort_index().to_numpy()).all()), True)
    lm = pd.read_parquet(P(L["LAST_MONTH_PATH"]))
    a4 = lm[lm["status"] == "assigned_last_month"].set_index("territory_id")
    c4 = Lr[Lr["class"] == 4].set_index("territory_id")
    ctl("класс 4 = types_last_month.parquet (assigned_last_month: id, тип, месяц, согласие)", bool(
        set(a4.index) == set(c4.index) and (a4.loc[c4.index, "type"].astype(int).to_numpy() == c4["type"].astype(int).to_numpy()).all()
        and (a4.loc[c4.index, "last_month"].to_numpy() == c4["last_month"].to_numpy()).all()
        and np.allclose(a4.loc[c4.index, "month_agreement"].to_numpy(float), c4["month_agreement"].to_numpy(float))), True)
    r409 = Lr[Lr.territory_id == 409].iloc[0]
    ctl("МО 409: Сут-Хольский район без типа (too_far)", (int(r409["class"]), r409["status"], "Сут-Хольск" in r409["name"]), (3, "too_far", True))
    ctl("статусы класса 3", {k: int(v) for k, v in Lr.loc[Lr["class"] == 3, "status"].value_counts().items()},
        {"no_data_in_dataset": 407, "no_dec2024_no_type": 10, "too_far": 1})
    ctx["Lr"] = Lr

    # ---------- поправка 3: метки по месяцам и уверенность сопоставления ----------
    from sklearn.metrics import adjusted_rand_score
    tr = pd.read_parquet(P(L["TRAJECTORIES_PATH"]))
    months = sorted(tr["month"].unique().tolist())
    ctl("месяцев в траекториях", (len(months), months[0], months[-1]), (L["EXPECT_MONTHS"], L["EXPECT_MONTH_FIRST"], C.MONTH))
    ctl("траектории: строк = 2004 МО × 24 месяца, дубликатов (МО, месяц) нет", (len(tr), int(tr.duplicated(["territory_id", "month"]).sum())), (2004 * L["EXPECT_MONTHS"], 0))
    ctl("вход CSV: строк данных = МО выборки (по входным данным, не по файлу)", len(s1), L["EXPECT_CSV_ROWS"])
    ctl("вход CSV: колонок = 3 + число месяцев (по входным данным, не по файлу)", 3 + len(months), L["EXPECT_CSV_COLS"])
    ctl("траектории: МО = класс 1", sorted(tr["territory_id"].unique().tolist()), sorted(s1.index.tolist()))
    piv = tr.pivot(index="territory_id", columns="month", values="cluster")
    ctl("метки по месяцам: 2004 МО × 24 месяца без пропусков", (piv.shape, int(piv.isna().sum().sum())), ((2004, L["EXPECT_MONTHS"]), 0))
    ctl("метки по месяцам: значения в диапазоне 0–5", (int(piv.min().min()), int(piv.max().max())), (0, 5))
    no_label = L["MONTH_NO_LABEL"]
    ml = np.full((len(Lr), len(months)), no_label, np.uint8)
    pos = {t: i for i, t in enumerate(Lr["territory_id"])}
    for t, row in piv.iterrows():
        ml[pos[t]] = row[months].to_numpy(np.uint8)
    ctl("встроенные метки: байт 255 только у МО вне выборки", (int((ml[Lr["class"].to_numpy() == 1] == no_label).sum()), int((ml[Lr["class"].to_numpy() != 1] != no_label).sum())), (0, 0))
    samp = ml[Lr["class"].to_numpy() == 1]
    ctl("число МО по типам в каждом месяце: сумма 2004", [int(sum(np.bincount(samp[:, m], minlength=6))) for m in range(len(months))], [2004] * len(months))
    canon = Lr["type"].to_numpy()
    dec_tr = ml[:, -1].astype(int)
    is1 = Lr["class"].to_numpy() == 1
    ctl("декабрь 2024: метки траекторий и канонические различаются у МО", int((dec_tr[is1] != canon[is1].astype(int)).sum()), L["EXPECT_DEC_DIFFERS"])
    ari = float(adjusted_rand_score(canon[is1].astype(int), dec_tr[is1]))
    ctl("декабрь 2024: ARI траекторий и канонических меток (допуск 0.0005)", round(ari, 4), L["EXPECT_ARI_DEC"], abs(ari - L["EXPECT_ARI_DEC"]) <= L["ARI_DEC_TOL"])
    ctl("dec_label_differs = различие меток декабря 2024 (id)", sorted(Lr.loc[is1 & (dec_tr != canon.astype(int)), "territory_id"].tolist()), sorted(Lr.loc[Lr["dec_label_differs"].fillna(False).astype(bool), "territory_id"].tolist()))
    mt = pd.read_parquet(P(L["MATCHING_PATH"]))
    ctl("сопоставление: 24 месяца × 6 типов", (len(mt), sorted(mt["месяц"].unique().tolist()) == months), (len(months) * 6, True))
    codes = {"уверенно": 0, "умеренно": 1, "неоднозначно": 2}
    ctl("сопоставление: значения колонки «уверенность»", sorted(mt["уверенность"].unique().tolist()), sorted(codes))
    conf = np.zeros((len(months), 6), np.uint8)
    for _, r in mt.iterrows():
        conf[months.index(r["месяц"]), int(r["cluster"])] = codes[r["уверенность"]]
    nd = [m for m in months if m != C.MONTH]
    conf_n = [int(sum(conf[months.index(m), k] == 0 for m in nd)) for k in range(6)]
    ctl("уверенных сопоставлений из 23 по типам 0–5", conf_n, L["EXPECT_CONFIDENT_BY_TYPE"])
    ctl("декабрь 2024: все сопоставления уверенные (якорный месяц)", int(conf[-1].sum()), 0)
    ctx.update(months24=months, ml_b64=base64.b64encode(ml.tobytes()).decode("ascii"), conf24=conf.tolist(),
               month_counts=[np.bincount(samp[:, m], minlength=6).tolist() for m in range(len(months))], ari_dec=ari)

    # ---------- геометрия ----------
    nat = encode_geo(L["GEO_NATIONAL_PATH"], L["GEO_SCALE_NATIONAL"], True)
    cit = encode_geo(L["GEO_CITIES_PATH"], L["GEO_SCALE_CITIES"], False)
    ctl("геометрия страны: число МО", len(nat["ids"]), L["EXPECT_ACTIVE"])
    ctl("геометрия страны: territory_id совпадают с layer_data_24f", nat["ids"], Lr["territory_id"].tolist())
    ctl("геометрия городов: territory_id входят в геометрию страны", sorted(set(cit["ids"]) - set(nat["ids"])), [])
    cities = {c: sum(1 for p in cit["props"] if p["city"] == c) for c in ("moscow", "spb")}
    ctl("геометрия городов: полигонов Москвы и Санкт-Петербурга", cities, L["EXPECT_CITY_POLYGONS"])
    ins = {int(p["territory_id"]): bool(p["in_sample"]) for p in nat["props"]}
    ctl("in_sample в границах = класс 1", all(ins[int(i)] == (c == 1) for i, c in zip(Lr.territory_id, Lr["class"])), True)
    ctl("название и регион в границах = layer_data_24f", all(p["name"] == n and p["region_name"] == r for p, n, r in zip(nat["props"], Lr["name"], Lr["region_name"])), True)
    for nm, g, sc in (("страны", nat, L["GEO_SCALE_NATIONAL"]), ("городов", cit, L["GEO_SCALE_CITIES"])):
        ctl(f"геометрия {nm}: координаты кратны 1/{sc} (макс. отклонение в единицах сетки)", round(g["max_err"], 3), 0.0)
    lon_lo, lon_hi = L["LON_AFTER_RANGE"]
    ctl("долготы после сдвига на +360 в допустимом диапазоне", (nat["lon_min"] >= lon_lo, nat["lon_max"] <= lon_hi), (True, True))
    raw_pts = {k: sum(len(r) for f in json.loads(read_text(p))["features"] for poly in f["geometry"]["coordinates"] for r in poly)
               for k, p in (("страны", L["GEO_NATIONAL_PATH"]), ("городов", L["GEO_CITIES_PATH"]))}
    ctl("число вершин в встроенной геометрии = в исходных geojson", (nat["n_pts"], cit["n_pts"]), (raw_pts["страны"], raw_pts["городов"]))
    ctx.update(nat=nat, cit=cit)
    ctx["geo_national_b64"] = pack_geo(nat)
    ctx["geo_cities_b64"] = pack_geo(cit, {"city": [p["city"] for p in cit["props"]]})

    # ---------- шаг 23: имена типов и доли ----------
    s23, D, tech = step23_frame()
    tn = parse_type_names_md()
    ctl("docs/type_names.md: строки типов 0–5", sorted(tn), list(range(6)))
    n23 = nb23_names()
    ctl("технические имена: full_type_name = notebooks/23_type_portraits.md", tech, n23)
    ctl("технические имена: full_type_name = docs/type_names.md", tech, [tn[t]["tech"] for t in range(6)])
    ctl("доли шага 23: МО = класс 1 (id)", D.index.tolist(), sorted(s1.index.tolist()))
    ctl("доли шага 23: тип = layer_data_24f", D["type"].astype(int).tolist(), s1.sort_index().tolist())
    sh = pd.read_parquet(P(L["SHARES_PATH"]))
    sh = sh[sh["date"] == pd.Timestamp(C.MONTH + "-01")].set_index("territory_id").sort_index()
    cols = ["share_" + c for c in CATS]
    diff = float(np.abs(sh.loc[D.index, cols].to_numpy() * 100 - D[CATS].to_numpy()).max())
    ctl("пять долей за декабрь 2024: шаг 23 (build_frame) = category_shares.parquet × 100 (макс. расхождение)", diff < 1e-9, True)
    ctx.update(D=D, names={t: {"tech": tech[t], "name": tn[t]["name"], "status": tn[t]["status"]} for t in range(6)})

    # ---------- карточки типов ----------
    Pq = pd.read_parquet(P(L["PORTRAITS_PATH"]))
    Mk = pd.read_parquet(P(L["MIRKIN_PATH"]))

    def pv(block: str, metric: str, t) -> tuple[float, object]:
        q = Pq[(Pq["type"] == str(t)) & (Pq["block"] == block) & (Pq["metric"] == metric)]
        if len(q) != 1:
            stop(f"type_portraits.parquet: тип {t}, блок «{block}», метрика «{metric}»: найдено {len(q)} строк")
        r = q.iloc[0]
        return float(r["value"]), (None if pd.isna(r["n"]) else int(r["n"]))

    def mk(t, cat: str, ind: str) -> float:
        q = Mk[(Mk["тип"] == str(t)) & (Mk["категория"] == cat) & (Mk["показатель"] == ind)]
        if len(q) != 1:
            stop(f"mirkin_rule_27.parquet: тип {t}, {cat}, «{ind}»: найдено {len(q)} строк")
        return float(q.iloc[0]["значение"])
    cards = {}
    for t in range(6):
        c = {"cats": {}}
        c["n"] = int(pv("состав", "МО типа", t)[0])
        c["share"] = pv("состав", "доля от 2004 МО, %", t)[0]
        c["regions_n"] = int(pv("состав", "регионов", t)[0])
        rg = Pq[(Pq["type"] == str(t)) & (Pq["block"] == "состав") & Pq["metric"].str.startswith("регион ") & Pq["metric"].str.endswith(": МО")]
        regs = [(m[len("регион "):-len(": МО")], float(v)) for m, v in zip(rg["metric"], rg["value"])]
        regs = [x for _, x in sorted(enumerate(regs), key=lambda p: (-p[1][1], p[0]))]
        ctl(f"тип {t}: топ регионов в type_portraits.parquet", len(regs), L["EXPECT_TOP_REGIONS"])
        c["regions"] = regs
        for cat in CATS:
            zs = pv("признаки профиля", f"{cat}: σ по долям", t)[0]
            za = pv("признаки профиля", f"{cat}: σ по лог-отношению", t)[0]
            bq = Pq[(Pq["type"] == str(t)) & (Pq["block"] == "признаки профиля") & (Pq["metric"] == f"{cat}: на границе порога, меньшее |σ|")]
            mark = "" if abs(zs) < th else ("держится" if np.sign(za) == np.sign(zs) and abs(za) >= th else "зависит от записи")
            c["cats"][cat] = {"median": pv("расходы", f"медиана {cat}, %", t)[0], "mean": mk(t, cat, "среднее по типу, %"),
                              "gmean": mk("все", cat, "общее среднее, %"), "sz": zs, "sa": za, "mark": mark,
                              "border": float(bq.iloc[0]["value"]) if len(bq) else None}
        c["ps"], _ = pv("надёжность", "PS типа (среднее доля пар)", t)
        cv_, cn_ = pv("надёжность", "уверенных сопоставлений из 23", t)
        c["conf"], c["conf_n"] = int(cv_), int(cn_)
        c["g6"] = {"s": pv("надёжность", "Г6: S по типу", t)[0], "base": pv("надёжность", "Г6: базовый уровень", t)[0],
                   "n": int(pv("надёжность", "Г6: переходов", t)[0])}
        cards[t] = c
    ctl("карточки: МО типа (type_portraits.parquet) по типам", [cards[t]["n"] for t in range(6)], L["EXPECT_SAMPLE_BY_TYPE"])
    tol = L["CARD_SHARE_TOL"]
    e0, e3 = L["EXPECT_CARDS"][0], L["EXPECT_CARDS"][3]
    ctl("тип 0: доля выборки (доля, допуск 0.006)", round(cards[0]["share"] / 100, 4), e0["share"], abs(cards[0]["share"] / 100 - e0["share"]) <= tol)
    ctl("тип 0: prediction strength (допуск 0.006)", round(cards[0]["ps"], 4), e0["ps"], abs(cards[0]["ps"] - e0["ps"]) <= tol)
    ctl("тип 0: уверенных сопоставлений из 23 (точно)", (cards[0]["conf"], cards[0]["conf_n"]), (e0["confident"], 23))
    ctl("тип 0: Г6 S и базовый уровень (допуск 0.006)", (round(cards[0]["g6"]["s"], 4), round(cards[0]["g6"]["base"], 4)), (e0["g6"], e0["g6_base"]),
        abs(cards[0]["g6"]["s"] - e0["g6"]) <= tol and abs(cards[0]["g6"]["base"] - e0["g6_base"]) <= tol)
    ctl("тип 3: prediction strength (допуск 0.006)", round(cards[3]["ps"], 4), e3["ps"], abs(cards[3]["ps"] - e3["ps"]) <= tol)
    ctl("тип 3: уверенных сопоставлений из 23 (точно)", (cards[3]["conf"], cards[3]["conf_n"]), (e3["confident"], 23))
    # независимая сверка медиан, средних и σ с данными шага 23
    Dt = D.copy()
    md_diff = max(abs(float(Dt.loc[Dt["type"] == t, cat].median()) - cards[t]["cats"][cat]["median"]) for t in range(6) for cat in CATS)
    mn_diff = max(abs(float(Dt.loc[Dt["type"] == t, cat].mean()) - cards[t]["cats"][cat]["mean"]) for t in range(6) for cat in CATS)
    gm_diff = max(abs(float(Dt[cat].mean()) - cards[0]["cats"][cat]["gmean"]) for cat in CATS)
    ctl("медианы карточек (parquet) = медианы по данным шага 23 (макс. расхождение < 1e-6)", md_diff < 1e-6, True)
    ctl("средние карточек (шаг 27) = средние по данным шага 23 (макс. расхождение < 1e-6)", mn_diff < 1e-6, True)
    ctl("общее среднее (шаг 27) = среднее по 2004 МО (макс. расхождение < 1e-6)", gm_diff < 1e-6, True)
    sig_diff = max(abs(cards[t]["cats"][cat]["sz"] - mk(t, cat, "σ по долям (шаг 23)")) for t in range(6) for cat in CATS)
    ctl("σ по долям: type_portraits.parquet = mirkin_rule_27.parquet (макс. расхождение < 1e-9)", sig_diff < 1e-9, True)
    ctx["cards"] = cards

    # ---------- проверки: notebooks/26 и hypothesis_summary.parquet ----------
    H = pd.read_parquet(P(L["SUMMARY_PATH"]))
    ctl("hypothesis_summary.parquet: строк", len(H), L["EXPECT_SUMMARY_ROWS"])
    ctl("hypothesis_summary.parquet: строк блока «0. контроль»", int((H["block"] == "0. контроль").sum()), L["EXPECT_CONTROL_ROWS"])
    comp = pd.read_parquet(P(L["COMPOSITION_PATH"]))
    text26 = read_text(L["NB26_PATH"])
    tabs_all = parse_md_tables(text26)
    prof = [t for t in tabs_all if t["header"][:3] == ["тип", "категория", "σ по долям"]]
    ctl("notebooks/26: таблица профилей типов (σ, пометки)", len(prof), 1)
    bad_prof = []
    for r in prof[0]["rows"]:
        cd = cards[int(r[0])]["cats"][r[1]]
        want = [f"{cd['sz']:+.2f}", f"{cd['sa']:+.2f}", cd["mark"] or "—", f"меньшее |σ| {cd['border']:.2f}" if cd["border"] is not None else "—"]
        if r[2:6] != want:
            bad_prof.append((r, want))
    ctl("карточки: σ, пометки и граница порога = таблица профилей notebooks/26 (30 строк)", (len(prof[0]["rows"]), bad_prof), (30, []))
    lists = md_section_lists(text26, "## 3. Что устойчиво и что нет")
    lists = [x for x in lists if x[1]]
    rep = verify_summary(tabs_all, H, comp, lists)
    ctl("notebooks/26: разобрано строк таблиц проверок (Г1–Г13)", rep["rows_total"], 46)
    ctl("notebooks/26: строк без сверки с parquet", rep["rows_unchecked"], [])
    ctl("notebooks/26: дробные числа вне первичных колонок без пары в parquet", rep["tokens_unpaired"], [])
    ctl("hypothesis_summary.parquet: строки блоков «1. …» без места в таблицах md", rep["unused"], [])
    attributed = sum(t[3] for t in rep["tables"])
    ctl("сопоставление: строки блоков «1. …» parquet = сумма строк, отнесённых к таблицам md", attributed, rep["P1"])
    ctl("notebooks/26, раздел 3: числа списков без пары в parquet", rep["s3"]["unpaired"], [])
    heads = {"main": ("## 1. Проверки", "проверка", None), "g7": ("### Учёт возраста (Г7)", "проверка", None), "g4": ("### Г4 по отраслям", "отрасль", None),
             "sup": ("### Дополнительное семейство (Г10–Г13)", "проверка", "показатель"), "supd": ("### Дополнительное семейство (Г10–Г13)", "проверка", "η²_H доли до учёта (n)")}
    tabs = {}
    for k, (hd, fc, hh) in heads.items():
        tabs[k] = [t for t in tabs_all if t["heading"] == hd and t["header"][0] == fc and (hh is None or hh in t["header"])][0]
    # блок 0 «контроль»: значения из parquet над таблицей; сверка с текстом раздела 0 md
    b0 = H[H["block"] == "0. контроль"]
    line0 = [x for x in text26.split("\n") if x.startswith("- Контроль «региональный шум»")]
    ctl("notebooks/26: строка о контроле «региональный шум»", len(line0), 1)
    have = {f"{float(v):g}" for v in b0["value"]}
    md_dec = {t[0] for t in tokens(line0[0]) if t[1] > 0}
    ctl("значения блока 0 (parquet) покрывают дробные числа строки о контроле в md", sorted(md_dec - have), [])
    items0 = "; ".join(f'{esc(r["metric"])} — {esc(r["level"])}: {float(r["value"]):.3f}' for _, r in b0.iterrows())
    src0 = "; ".join(sorted({r["source"] for _, r in b0.iterrows()}))
    ctrl_note = (f'<p class="ctrl-note" title="источник: data/processed/hypothesis_summary.parquet, блок «0. контроль» ({esc(src0)})">'
                 f"<b>Контроль «региональный шум»</b> (повторов: {int(b0['n'].iloc[0])}): {items0}.</p>")
    ctx.update(rep=rep, tabs=tabs, lists=lists, ctrl_note=ctrl_note, b0=b0)

    # ---------- покрытие ----------
    cov = pd.read_parquet(P(L["COVERAGE_PATH"]))
    ctl("таблица покрытия: строк", len(cov), L["EXPECT_COVERAGE_ROWS"])
    g = Lr.groupby("region_name")
    chk = pd.DataFrame({"n_mo": g.size(), "n_sample": g.apply(lambda d: int((d["class"] == 1).sum())),
                        "n_assigned": g.apply(lambda d: int((d["class"] == 2).sum())),
                        "n_last_month": g.apply(lambda d: int((d["class"] == 4).sum())),
                        "n_no_type": g.apply(lambda d: int((d["class"] == 3).sum()))})
    cv_ = cov.set_index("region_name")[list(chk.columns)].astype(int).sort_index()
    ctl("покрытие по регионам = пересчёт из layer_data_24f (число расхождений)", int((cv_.to_numpy() != chk.sort_index().to_numpy()).sum()), 0)
    ctl("share_no_type = n_no_type / n_mo (макс. расхождение < 1e-12)", float(np.abs(cov["share_no_type"] - cov["n_no_type"] / cov["n_mo"]).max()) < 1e-12, True)
    empty = sorted(cov.loc[cov["n_sample"] == 0, "region_name"].tolist())
    ctl("регионов с n_sample = 0", len(empty), L["EXPECT_EMPTY_SAMPLE_REGIONS"])
    ctl("регионов в выборке (из layer_data_24f)", int(Lr.loc[Lr["class"] == 1, "region_name"].nunique()), L["EXPECT_REGIONS_SAMPLE"])
    ctl("регионов в таблице покрытия с n_sample > 0 = регионов в выборке", int((cov["n_sample"] > 0).sum()), L["EXPECT_REGIONS_SAMPLE"])
    ctl("регионов в portraits (строка «регионов в выборке»)", int(Pq[(Pq["type"] == "все") & (Pq["metric"] == "регионов в выборке")]["value"].iloc[0]), L["EXPECT_REGIONS_SAMPLE"])
    cov = cov.sort_values(["share_no_type", "region_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    ctx.update(cov=cov, empty=empty)
    # присвоенные МО дальше p95 расстояний выборки: пересчёт и строка отчёта шага 24c
    md24c = read_text("notebooks/24c_types_outside_sample.md")
    m = re.search(r"p95 [\d.]+ против ([\d.]+)[^\n]*выше p95 выборки (\d+) из (\d+) присвоенных", md24c)
    if not m:
        stop("в notebooks/24c_types_outside_sample.md не найдена строка о p95 расстояний выборки")
    p95 = float(m.group(1))
    beyond = int((out.loc[out["status"] == "assigned", "dist_nearest"] > p95).sum())
    ctl("присвоенных МО дальше p95 выборки: пересчёт по types_outside_sample.parquet и строка шага 24c", (beyond, int(m.group(2))), (L["EXPECT_BEYOND_P95"], L["EXPECT_BEYOND_P95"]))
    ctx["beyond"] = (beyond, int(m.group(3)))

    # ---------- графики ----------
    figs = {}
    for n in range(1, 6):
        rel = L["FIGURES"][n]
        if not P(rel).exists():
            figs[n] = None
        elif rel.endswith(".svg"):
            figs[n] = ("svg", svg_inline(read_text(rel), n))
        else:
            figs[n] = ("png", base64.b64encode(P(rel).read_bytes()).decode("ascii"))
    ctx["figs"] = figs

    # ---------- ограничения и ATTRIBUTION ----------
    interp = read_text(L["INTERPRETATION_PATH"]).split("\n")
    if L["INTERPRETATION_LIMITS_HEADING"] not in interp:
        stop(f"в docs/interpretation.md нет раздела {L['INTERPRETATION_LIMITS_HEADING']}")
    k = interp.index(L["INTERPRETATION_LIMITS_HEADING"]) + 1
    limits = []
    while k < len(interp) and not interp[k].startswith("## "):
        if interp[k].startswith("- "):
            limits.append(interp[k][2:].strip())
        k += 1
    ctl("раздел «6. Ограничения»: пунктов", len(limits), L["EXPECT_LIMITS_ITEMS"])
    ctx["limits"] = limits
    attr = read_text(L["ATTRIBUTION_PATH"])
    ctx["attr_md"] = attr
    head = L["ATTRIBUTION_CAPTION_HEADING"]
    al = attr.split("\n")
    if head not in al:
        stop(f"в ATTRIBUTION.md нет заголовка {head}")
    j = al.index(head) + 1
    while not al[j].strip():
        j += 1
    ctx["caption"] = al[j].strip()
    return ctx


# ============================================================================ страница
def app_data(ctx: dict) -> str:
    Lr, D = ctx["Lr"], ctx["D"]
    regions = sorted(Lr["region_name"].unique().tolist())
    ridx = {r: i for i, r in enumerate(regions)}
    months = sorted(Lr["last_month"].dropna().unique().tolist())
    midx = {m: i for i, m in enumerate(months)}
    sh = []
    for tid, cls in zip(Lr["territory_id"], Lr["class"]):
        sh.append([float(D.loc[tid, c]) for c in CATS] if cls == 1 else None)
    mo = {"id": lst(Lr["territory_id"]), "name": lst(Lr["name"]), "reg": [ridx[r] for r in Lr["region_name"]],
          "cls": lst(Lr["class"]), "typ": [(-1 if pd.isna(t) else int(t)) for t in Lr["type"]], "st": lst(Lr["status"]),
          "ma": lst(Lr["market_access"]), "ms": lst(Lr["months_same"]), "dd": lst(Lr["dec_label_differs"]),
          "lm": [None if pd.isna(m) else midx[m] for m in Lr["last_month"]], "mag": lst(Lr["month_agreement"]),
          "dn": lst(Lr["dist_nearest"]), "sh": sh}
    c4 = Lr["class"].to_numpy() == 4
    cap4 = [months[mo["lm"][k]] for k in range(len(Lr)) if c4[k]]     # месяц в подписи класса 4 (JS: APP.months[M.lm[i]])
    ctl("подпись класса 4: месяц в подписи равен last_month у всех 74 МО (допуск 0)", sum(1 for a, b in zip(cap4, Lr.loc[c4, "last_month"]) if a != b), 0)
    ctl("подпись класса 4: число МО по последнему доступному месяцу", {m: cap4.count(m) for m in sorted(set(cap4))}, {k: v for k, v in L["EXPECT_LAST_MONTH_COUNTS"].items()})
    ctl("подпись класса 4: декабря 2024 нет ни у одного МО", cap4.count("2024-12"), 0)
    ma = L["MARKET_ACCESS_SCALE"]
    cfg = {"lat": L["PROJECTION_LAT_DEG"], "cityPx": L["CITY_SWITCH_PX_PER_DEG"], "maxZoom": L["MAX_ZOOM_FACTOR"],
           "border": L["BORDER_LINEWIDTH_PX"], "bg": L["MAP_BG"], "dimFace": L["DIM_FACE"], "assignedHatch": L["ASSIGNED_HATCH_COLOR"],
           "lastHatch": L["LASTMONTH_HATCH_COLOR"], "noTypeFace": L["NO_TYPE_FACE"], "noTypeHatch": L["NO_TYPE_HATCH_COLOR"],
           "noValueFace": L["NO_VALUE_FACE"], "noValueHatch": L["NO_VALUE_HATCH_COLOR"], "ma": ma, "maRamp": L["MARKET_ACCESS_RAMP"],
           "stRamp": L["STABILITY_RAMP"], "stSteps": L["STABILITY_STEPS"], "searchMax": L["SEARCH_MAX_RESULTS"],
           "searchZoomMax": L["SEARCH_ZOOM_MAX_MATCHES"], "csvName": L["CSV_FILENAME"]}
    types = [{"id": t, "color": L["TYPE_COLORS"][t], "name": ctx["names"][t]["name"], "tech": ctx["names"][t]["tech"]} for t in range(6)]
    obj = {"types": types, "mo": mo, "regions": regions, "months": months, "cats": CATS, "cfg": cfg,
           "months24": ctx["months24"], "conf24": ctx["conf24"], "confWords": ["уверенно", "умеренно", "неоднозначно"],
           "noLabel": L["MONTH_NO_LABEL"], "playMs": L["MONTH_PLAY_MS"], "statusLabels": STATUS_LABELS, "classLabels": {str(k): v for k, v in CLASS_LABELS.items()}}
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def build_html(ctx: dict) -> tuple[str, dict]:
    """Возвращает (HTML, фрагменты дословно скопированных текстов для проверки запрещённых слов)."""
    T_ = ctx["texts"]
    Lr = ctx["Lr"]
    n_sample = int((Lr["class"] == 1).sum())
    n_reg = int(Lr.loc[Lr["class"] == 1, "region_name"].nunique())
    verbatim = {}
    attr_html = md_to_html(ctx["attr_md"])
    ctl("ATTRIBUTION.md на странице дословно (слова)", html_words(attr_html), plain_words(ctx["attr_md"]))
    lim_html = '<ul class="lim">' + "".join(f"<li>{esc(x)}</li>" for x in ctx["limits"]) + "</ul>"
    ctl("раздел «6. Ограничения» на странице дословно (слова)", html_words(lim_html), " ".join(ctx["limits"]).split())
    verbatim["attribution"] = attr_html
    verbatim["limits"] = lim_html

    nav = [("map", "Карта"), ("types", "Типы"), ("checks", "Проверки"), ("figures", "Графики"), ("coverage", "Покрытие"),
           ("method", "Метод и ограничения"), ("data", "Данные и лицензии")]
    nav_html = "".join(f'<li><a href="#{a}">{esc(b)}</a></li>' for a, b in nav)

    figs_html = []
    for n in range(1, 6):
        f = ctx["figs"][n]
        if f is None:
            body = f'<div class="fig-ph">{esc(L["FIGURE_PLACEHOLDER"])} (фигура {n})</div>'
        elif f[0] == "svg":
            body = f[1]
        else:
            body = (f'<img id="fig2-img" alt="Статическая карта типов МО, декабрь 2024" src="data:image/png;base64,{f[1]}">')
        figs_html.append(f'<figure id="figure-{n}"><p class="muted">Фигура {n}</p>{body}<figcaption>{prose(T_, f"figure_caption_{n}")}</figcaption></figure>')

    cards_html = "\n".join(card_html(t, ctx, T_) for t in range(6))
    cov_defs = ("n_mo — МО в регионе; n_sample — МО рабочей выборки; n_assigned — тип присвоен вне выборки; n_last_month — тип по последнему "
                "доступному месяцу; n_no_type — МО без типа; share_no_type — доля МО без типа (определения полей: notebooks/24f_layer_data.md).")
    src_rows = [
        ("МО в выборке", f"{n_sample}", "layer_data_24f.parquet: class = 1; kmeans_labels_final.parquet"),
        ("типов", "6", "config.yaml: clustering.FINAL_K; type_portraits.parquet"),
        ("регионов в выборке", f"{n_reg}", "layer_data_24f.parquet: region_name при class = 1; type_portraits.parquet: «регионов в выборке»"),
        ("МО на карте", f"{len(Lr)}", "layer_data_24f.parquet; data/geo/mo_national.geojson"),
        ("классы 1 / 2 / 4 / 3", " / ".join(str(int((Lr['class'] == k).sum())) for k in CLASS_ORDER), "layer_data_24f.parquet: class"),
        ("доли расходов, декабрь 2024", "5 категорий", "category_shares.parquet (как в шаге 23: build_frame)"),
        ("таблицы и числа карточек типов", "6 карточек", "type_portraits.parquet (шаг 23); mirkin_rule_27.parquet (шаг 27)"),
        ("таблицы проверок", "5 таблиц", "notebooks/26_hypothesis_summary.md; hypothesis_summary.parquet"),
        ("покрытие данных", f"{len(ctx['cov'])} регионов", "region_coverage_24f.parquet"),
        ("доступность рынков", "2571 МО со значением", "layer_data_24f.parquet: market_access (источник: data/raw/market_access.parquet)"),
        ("устойчивость типа", "2004 МО", "layer_data_24f.parquet: stable_share = months_same / 24"),
    ]
    srcs = "".join(f"<tr><td data-label=\"число\">{esc(a)}</td><td data-label=\"значение\">{esc(b)}</td><td data-label=\"источник\">{esc(c)}</td></tr>" for a, b, c in src_rows)

    body = f'''<header class="hero" id="overview"><div class="wrap">
<h1 data-text-key="hero_title" class="{"ph" if T_["hero_title"].strip() == L["TEXT_PLACEHOLDER"] else ""}">{esc(T_["hero_title"].strip())}</h1>
{prose(T_, "hero_lead")}
<ul class="stats">
<li><span class="big">{num(n_sample, "layer_data_24f.parquet: class = 1; kmeans_labels_final.parquet")}</span><span class="lab">МО в рабочей выборке</span></li>
<li><span class="big">{num(6, "config.yaml: clustering.FINAL_K; type_portraits.parquet")}</span><span class="lab">типов</span></li>
<li><span class="big">{num(n_reg, "layer_data_24f.parquet: число регионов у МО выборки; type_portraits.parquet")}</span><span class="lab">регионов в выборке</span></li>
</ul></div></header>
<nav class="top" aria-label="Разделы"><ul>{nav_html}</ul></nav>
<main>
<section id="map"><div class="wrap">
<h2>Карта типов МО</h2>
{prose(T_, "section_intro_map")}
<div id="map-ui">
<div class="maptools">
<fieldset class="layers"><legend>Слой</legend>
<label><input type="radio" name="layer" value="type" checked><span>Тип</span></label>
<label><input type="radio" name="layer" value="market"><span>Доступность рынков</span></label>
<label><input type="radio" name="layer" value="stab"><span>Устойчивость типа</span></label></fieldset>
<div class="search"><label for="map-search" class="muted">Поиск по названию МО (если МО с таким названием нет, ищется регион)</label><input id="map-search" type="search" autocomplete="off" placeholder="МО или регион (Якутск)"><ul id="map-results" hidden></ul></div>
<div class="mapbtns"><button class="btn" type="button" id="btn-zoom-in" aria-label="Приблизить">+</button><button class="btn" type="button" id="btn-zoom-out" aria-label="Отдалить">−</button><button class="btn" type="button" id="btn-reset">Сброс</button><button class="btn" type="button" id="btn-moscow">Москва</button><button class="btn" type="button" id="btn-spb">Санкт-Петербург</button></div>
</div>
<div class="monthctl" id="month-ctl"><span class="muted">Месяц (слой «Тип»):</span>
<button class="btn" type="button" id="month-prev" aria-label="Месяц назад">◀</button>
<button class="btn" type="button" id="month-play" aria-label="Играть">▶ играть</button>
<button class="btn" type="button" id="month-next" aria-label="Месяц вперёд">▶|</button>
<input type="range" id="month-range" min="0" max="23" step="1" value="23" aria-label="Месяц от 2023-01 до 2024-12">
<output id="month-label" for="month-range">2024-12</output>
<button class="btn" type="button" id="csv-btn" title="файл строится в браузере из встроенных меток траекторий (kmeans_k6_trajectories.parquet)">Скачать CSV</button>
<span class="muted" id="month-note"></span></div>
<div class="layer-note" data-layer="type">{prose(T_, "layer_note_last_month")}</div>
<div class="layer-note" data-layer="market" hidden>{prose(T_, "layer_note_market_access")}</div>
<div class="layer-note" data-layer="stab" hidden>{prose(T_, "layer_note_stability")}</div>
<div class="mapgrid">
<div><div id="map-wrap" class="mapbox"><canvas id="map-canvas" role="img" aria-label="Интерактивная карта типов муниципальных образований России"></canvas><canvas id="map-overlay" aria-hidden="true"></canvas><div id="map-tip" class="tip" hidden></div></div>
<p class="srcnote">Колёсико и перетаскивание — масштаб и сдвиг; на сенсорном экране — касания и щипок.</p></div>
<div class="side"><div id="map-panel" class="panel" aria-live="polite"><p class="muted">Наведите курсор на МО, чтобы увидеть данные; клик закрепляет МО в этой панели.</p></div><div id="map-legend" class="panel legend"></div></div>
</div></div>
<div id="map-fallback" hidden><p id="map-fallback-msg"></p><img id="map-fallback-img" alt="Статическая карта типов МО, декабрь 2024" hidden><p id="map-fallback-cap" hidden>Статическая карта показывает состояние на декабрь 2024 года; на интерактивной карте дополнительно показаны типы по последнему доступному месяцу</p></div>
{prose(T_, "map_caption")}
<p class="srcnote">{esc(ctx["caption"])}</p>
</div></section>
<section id="types"><div class="wrap">
<h2>Шесть типов МО</h2>
{prose(T_, "section_intro_types")}
<div class="cards">{cards_html}</div>
</div></section>
<section id="checks"><div class="wrap">
<h2>Проверки внешней валидности</h2>
{prose(T_, "section_intro_checks")}
{checks_html(ctx["tabs"], ctx["rep"], ctx["lists"], ctx["ctrl_note"], T_)}
</div></section>
<section id="figures"><div class="wrap figs">
<h2>Графики</h2>
{prose(T_, "section_intro_figures")}
{"".join(figs_html)}
</div></section>
<section id="coverage"><div class="wrap">
<h2>Покрытие данных по регионам</h2>
{prose(T_, "section_intro_coverage")}
{coverage_html(ctx["cov"], ctx["empty"])}
<p class="srcnote">{esc(cov_defs)}</p>
</div></section>
<section id="method"><div class="wrap">
<h2>Метод и ограничения</h2>
{prose(T_, "section_intro_method")}
{prose(T_, "method_text")}
<h3>Ограничения</h3>
<!--verbatim:limits:begin-->{lim_html}<!--verbatim:limits:end-->
<p class="srcnote">Источник: docs/interpretation.md, раздел «6. Ограничения», пункты без изменений.</p>
</div></section>
<section id="data"><div class="wrap">
<h2>Данные, лицензии и воспроизведение</h2>
{prose(T_, "section_intro_data")}
<h3>Источник и лицензия границ</h3>
<!--verbatim:attribution:begin-->{attr_html}<!--verbatim:attribution:end-->
<p class="srcnote">Источник: data/geo/ATTRIBUTION.md, текст без изменений.</p>
<h3>Репозиторий и воспроизведение</h3>
<ul>
<li>Репозиторий проекта: sberindex-clustering (адрес публикации в файлах проекта не задан).</li>
<li>Описание проекта и шагов: <code>README.md</code>; решения и интерпретация: <code>docs/decisions.md</code>, <code>docs/interpretation.md</code>, <code>docs/type_names.md</code>, <code>docs/interpretation_sources.csv</code>.</li>
<li>Полный прогон: <code>bash run_all.sh</code>; пересборка этой страницы: <code>.venv/bin/python src/24e_landing.py</code>; описание: <code>site/README.md</code>, <code>notebooks/24e_landing_check.md</code>.</li>
</ul>
<details><summary>Откуда числа</summary>
<table class="dt rt"><thead><tr><th>число</th><th>значение</th><th>источник</th></tr></thead><tbody>{srcs}</tbody></table>
<p class="srcnote">Каждое число на странице снабжено подсказкой с файлом-источником (наведите курсор); таблицы проверок взяты из notebooks/26_hypothesis_summary.md и сверены с hypothesis_summary.parquet.</p></details>
</div></section>
</main>
<footer><div class="wrap">Страница собрана скриптом src/24e_landing.py; тексты — site/content/texts.json.</div></footer>'''
    title = esc(T_["hero_title"].strip())
    page = f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{CSS}</style></head>
<body>
{body}
<script id="app-data" type="application/json">{app_data(ctx)}</script>
<script id="geo-national" type="text/plain">{ctx["geo_national_b64"]}</script>
<script id="geo-cities" type="text/plain">{ctx["geo_cities_b64"]}</script>
<script id="month-labels" type="text/plain">{ctx["ml_b64"]}</script>
<script>
{JS}</script>
</body></html>
'''
    return page, verbatim


# ============================================================================ проверки готовой страницы
def blank_blobs(page: str) -> str:
    t = re.sub(r'(<script id="(?:geo-(?:national|cities)|month-labels)" type="text/plain">)[^<]*(</script>)', r"\1\2", page)
    return re.sub(r"data:image/png;base64,[A-Za-z0-9+/=\s]+", "data:image/png;base64,", t)


def page_checks(page: str, verbatim: dict, ctx: dict) -> dict:
    info = {}
    size = len(page.encode("utf-8"))
    ctl("размер site/index.html не больше MAX_HTML_BYTES", size, L["MAX_HTML_BYTES"], size <= L["MAX_HTML_BYTES"])
    info["size"] = size
    # внешние ресурсы
    vals = re.findall(r"""\b(?:src|href|xlink:href|action|poster|srcset|data|formaction)\s*=\s*(?:"([^"]*)"|'([^']*)')""", page)
    vals = [a or b for a, b in vals]
    bad = [v[:80] for v in vals if not (v.startswith("#") or v.startswith("data:"))]
    ctl("внешние ресурсы: src/href/xlink:href не ведут за пределы страницы (ссылки не на #якорь и не data:)", bad, [])
    nb = blank_blobs(page)
    urls = re.findall(r"url\(([^)]*)\)", nb)
    ctl("внешние ресурсы: url() только внутристраничные (#id или data:)", [u[:60] for u in urls if not (u.startswith("#") or u.startswith("data:"))], [])
    code_only = JS + CSS
    hits = {w: len(re.findall(w, code_only)) for w in (r"\bfetch\s*\(", r"\bimport\b", r"@import", r"XMLHttpRequest", r"WebSocket", r"sendBeacon",
                                                      r"localStorage", r"sessionStorage", r"document\.cookie", r"https?://", r"importScripts")}
    ctl("скрипт и CSS: нет fetch, import, XMLHttpRequest, WebSocket, sendBeacon, localStorage, sessionStorage, cookie и адресов http(s)", {k: v for k, v in hits.items() if v}, {})
    tags = re.findall(r"<(link|iframe|object|embed|base|meta)\b[^>]*>", nb, flags=re.I)
    ctl("в HTML нет link, iframe, object, embed, base; из meta только charset и viewport",
        sorted(t for t in tags if t.lower() != "meta"), [])
    metas = re.findall(r"<meta\b[^>]*>", nb)
    ctl("meta: charset и viewport", len(metas), 2)
    ext_text = re.findall(r"https?://[^\s<\"']+", nb)
    info["url_texts"] = len(ext_text)
    in_attr = [u for u in re.findall(r'="[^"]*https?://[^"]*"', nb)]
    ctl("адреса http(s) только в тексте, не в атрибутах", in_attr, [])
    info.update(src_n=len(re.findall(r"\bsrc=", page)), href_n=len(re.findall(r"\bhref=", page)), url_n=len(urls),
                fetch_n=len(re.findall(r"\bfetch", nb)), import_n=len(re.findall(r"\bimport\b", nb)))
    # запрещённые слова в добавленном тексте
    t = nb
    for k, v in verbatim.items():
        a, b = f"<!--verbatim:{k}:begin-->", f"<!--verbatim:{k}:end-->"
        i, j = t.index(a), t.index(b)
        t = t[:i] + t[j + len(b):]
    low = t.lower()
    fw = {w: low.count(w.lower()) for w in L["FORBIDDEN_WORDS"]}
    ctl("запрещённые слова в тексте страницы (кроме дословно скопированных ограничений и ATTRIBUTION)", {k: v for k, v in fw.items() if v}, {})
    vlow = " ".join(verbatim.values()).lower()
    info["forbidden_in_verbatim"] = {w: vlow.count(w.lower()) for w in L["FORBIDDEN_WORDS"] if vlow.count(w.lower())}
    vis = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S))).lower()
    fs = {w: vis.count(w) + JS.lower().count(w) for w in L["FORBIDDEN_STEMS"]}
    ctl("запрещённые основы (правило 2.3) в видимом тексте страницы и строках скрипта (без дословных блоков)", {k: v for k, v in fs.items() if v}, {})
    info["forbidden_stems_verbatim"] = {w: vlow.count(w) for w in L["FORBIDDEN_STEMS"] if vlow.count(w)}
    uw = {w: low.count(w) for w in L["FORBIDDEN_UNITS"]}
    ctl("площади в км² в тексте страницы", {k: v for k, v in uw.items() if v}, {})
    # состав
    ctl("карточек типов", page.count('<article class="card"'), 6)
    body = nb.split('<section id="coverage">')[1].split("</section>")[0]
    ctl("строк таблицы покрытия на странице", len(re.findall(r"<tr>", body.split("<tbody>")[1])), L["EXPECT_COVERAGE_ROWS"])
    keys = set(re.findall(r'data-text-key="([^"]+)"', nb))
    ctl("на странице используются все 25 ключей texts.json", sorted(keys), sorted(L["TEXT_KEYS"]))
    for pr in L["CONTRAST_PAIRS"]:
        r = contrast(pr["fg"], pr["bg"])
        ctl(f"контраст «{pr['name']}» {pr['fg']} на {pr['bg']} не ниже {L['CONTRAST_MIN']}", round(r, 2), L["CONTRAST_MIN"], r >= L["CONTRAST_MIN"])
    info["placeholders"] = sum(1 for v in ctx["texts"].values() if v.strip() == L["TEXT_PLACEHOLDER"])
    vis_all = html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<script.*?</script>|<style.*?</style>", " ", page, flags=re.S)))
    ctl("заглушек текста: texts.json и видимый текст страницы",
        (info["placeholders"], vis_all.count(L["TEXT_PLACEHOLDER"]), page.count('class="ph"')), (0, 0, 0))
    return info


# ============================================================================ отчёты
def md_cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(md_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def existing_browser_block() -> str:
    """Блок «Проверка в браузере»: заполненный тестом блок сохраняется; если md нет или блок пуст, пишется строка о ненайденных результатах."""
    p = P(L["REPORT_PATH"])
    if p.exists():
        t = p.read_text(encoding="utf-8")
        if BROWSER_BEGIN in t and BROWSER_END in t:
            blk = t[t.index(BROWSER_BEGIN):t.index(BROWSER_END) + len(BROWSER_END)]
            if L["BROWSER_BLOCK_EMPTY_TEXT"] not in blk:
                return blk
    return f"{BROWSER_BEGIN}\n{L['BROWSER_BLOCK_EMPTY_TEXT']}\n{BROWSER_END}"


def build_report(ctx: dict, info: dict, frozen_after: list[str], page_sha: str) -> str:
    Lr, rep = ctx["Lr"], ctx["rep"]
    ma_max = Lr.loc[Lr["market_access"].idxmax()]
    cls2_no_ma = int(((Lr["class"] == 2) & Lr["market_access"].isna()).sum())
    lm = Lr[Lr["class"] == 4]["last_month"].value_counts().sort_index()
    weak = [t for t in range(6) if ctx["cards"][t]["g6"]["s"] < ctx["cards"][t]["g6"]["base"]]
    notes = [
        "В `hypothesis_summary.parquet` нет колонки «оговорка», а колонка «результат» пуста в части строк; таблицы проверок поэтому разобраны из "
        "`notebooks/26_hypothesis_summary.md`, а числа сверены с parquet (решение автора, раздел 2).",
        f"Диапазон слоя «Доступность рынков» растянут одним МО: `{ma_max['name']}` ({ma_max['region_name']}, территория {int(ma_max['territory_id'])}) "
        f"имеет market_access {ma_max['market_access']:.1f}, медиана по МО со значением {Lr['market_access'].median():.1f}, 99-й процентиль "
        f"{Lr['market_access'].quantile(0.99):.1f}. При линейной шкале почти вся карта лежит в узком диапазоне цветов, поэтому в config задана "
        f"логарифмическая шкала (параметр `MARKET_ACCESS_SCALE.kind`, значение `linear` возвращает линейную); выбор шкалы остаётся за автором.",
        f"У {cls2_no_ma} из {int((Lr['class'] == 2).sum())} МО класса 2 нет значения market_access; всего без значения {int(Lr['market_access'].isna().sum())} МО.",
        "Г6 (type_portraits.parquet): у типов " + ", ".join(str(t) for t in weak) + " доля переходов в ближайшие типы S ниже базового уровня (" +
        "; ".join(f"тип {t}: S {ctx['cards'][t]['g6']['s']:.2f}, базовый уровень {ctx['cards'][t]['g6']['base']:.2f}" for t in weak) +
        "); в карточках значения показаны без изменений.",
        "Последний доступный месяц у МО класса 4: " + ", ".join(f"{m}: {int(n)}" for m, n in lm.items()) + "; у большинства месяц 2023-12.",
        f"Присвоенных вне выборки МО дальше 95-го процентиля расстояний выборки: {ctx['beyond'][0]} из {ctx['beyond'][1]} (пересчёт по "
        f"`types_outside_sample.parquet` совпал со строкой шага 24c).",
        "В исходных границах (`data/geo/ATTRIBUTION.md`) остаются известные ошибки: 25 пар МО с перекрытием, недопустимые полигоны после упрощения; "
        "карта заливает по правилу nonzero и не исправляет их.",
    ]
    if ctx["missing_figs"]:
        notes.append("Не найдены файлы фигур: " + ", ".join(f"`{x}`" for x in ctx["missing_figs"]) + "; на странице стоит заглушка.")
    lines = [
        "# 24e. Лендинг: каркас (данные, интерактивная карта, карточки типов)", "",
        "Сгенерировано `src/24e_landing.py`. Описательный шаг: только чтение готовых файлов, новых моделей, тестов и порогов нет. Выход: `site/index.html` "
        f"(один автономный файл, {info['size']} байт, без внешних запросов), `site/content/texts.json` (тексты автора), `site/README.md`, этот отчёт. "
        f"Параметры и ожидания: `config.yaml`, группа `step24e_landing`. sha256 страницы: `{page_sha}`.", "",
        "## 1. Что встроено", "",
        md_table(["блок страницы", "откуда данные (файлы и ключи)"], [
            ["МО (2594): класс, тип, статус, market_access, stable_share, months_same, dec_label_differs, last_month, month_agreement, dist_nearest",
             "`data/processed/layer_data_24f.parquet`; названия и регионы совпадают с `data/geo/mo_national.geojson`"],
            ["пять долей расходов за декабрь 2024 (МО выборки)", "`step23_frame()`: вызовы `step18.load`, `clean_rosstat`, `step21.supply_values`, `build_frame` из `src/23_type_portraits.py`; сверка с `category_shares.parquet`"],
            ["геометрия страны и городов", "`data/geo/mo_national.geojson` (4 знака), `data/geo/mo_cities.geojson` (5 знаков): целые координаты, дельта-кодирование, gzip, base64; долготы меньше 0 сдвинуты на +360 у страны"],
            ["технические имена типов", "`full_type_name` из `src/23_type_portraits.py` (сверка с `notebooks/23_type_portraits.md` и `docs/type_names.md`)"],
            ["интерпретационные имена и статусы", "таблица `docs/type_names.md` (разбор markdown)"],
            ["карточки типов", "`type_portraits.parquet` (блоки «состав», «расходы», «признаки профиля», «надёжность»); `mirkin_rule_27.parquet` (средние по типу и общее среднее)"],
            ["таблицы проверок", "`notebooks/26_hypothesis_summary.md` (разделы «1. Проверки», Г7, Г4, Г10–Г13), числа сверены с `hypothesis_summary.parquet`"],
            ["заметка о контроле «региональный шум»", "`hypothesis_summary.parquet`, блок «0. контроль» (6 строк)"],
            ["списки раздела 3 (что устойчиво, что нет)", "`notebooks/26_hypothesis_summary.md`, раздел 3, без профилей типов"],
            ["ограничения", "`docs/interpretation.md`, раздел «6. Ограничения», дословно"],
            ["источник и лицензия границ", "`data/geo/ATTRIBUTION.md`, дословно; подпись под картой — абзац «Подпись для карты»"],
            ["графики", "SVG из `notebooks/figures` (фигуры 1, 3, 4, 5) внутри страницы; PNG фигуры 2 в base64"],
            ["покрытие по регионам", "`data/processed/region_coverage_24f.parquet`"],
            ["тексты", "`site/content/texts.json`, 25 ключей"]]), "",
        "## 2. Решения автора и параметры", "",
        "- Лендинг: один автономный HTML-файл без внешних запросов; открывается с диска (file://); интерактивна только карта; режимов «равные клетки», слайдера и ленты по месяцам нет.",
        "- Таблица проверок: разбор `notebooks/26_hypothesis_summary.md` (разделы «1. Проверки», Г7, Г4 по отраслям, Г10–Г13) со всеми колонками, включая «оговорка» и «результат», дословно; "
        "числа сверяются с `hypothesis_summary.parquet`, при расхождении STOP. Блок «0. контроль» показан одной заметкой над таблицами. Раздел 3 показан двумя списками, "
        "профили типов не дублируются (они есть в карточках). На ширине 390 px строки таблиц проверок становятся карточками.",
        f"- Шкала слоя «Доступность рынков»: `{L['MARKET_ACCESS_SCALE']['kind']}`, диапазон {L['MARKET_ACCESS_SCALE']['min']}–{L['MARKET_ACCESS_SCALE']['max']}, {L['MARKET_ACCESS_SCALE']['bins']} оттенков (config). "
        f"Слой «Устойчивость типа»: {L['STABILITY_STEPS'] + 1} значений (доля из {L['STABILITY_STEPS']} месяцев).",
        f"- На масштабе от {L['CITY_SWITCH_PX_PER_DEG']:g} пикселей на градус долготы вместо `mo_national` рисуется `mo_cities` (257 полигонов Москвы и Санкт-Петербурга).",
        "- Пометки «держится» и «зависит от записи» в карточках вычислены по правилу шага 26 (порог |σ| из `10b_kmeans_cluster_profiles`) и сверены со строками таблицы профилей в `notebooks/26_hypothesis_summary.md`.", "",
        "## 2а. Поправка 3: селектор месяца в слое «Тип»", "",
        f"Решение автора 07.10.2026. Встроено месяцев: {len(ctx['months24'])} ({ctx['months24'][0]} … {ctx['months24'][-1]}); метки: `kmeans_k6_trajectories.parquet` (колонка cluster после сопоставления с якорем), "
        "по одному байту на МО и месяц (255 = нет метки у МО вне выборки); уверенность сопоставления: `kmeans_k6_matching.parquet`, 24 месяца × 6 типов (уверенно, умеренно, неоднозначно). "
        "По умолчанию декабрь 2024, канонические метки. В других месяцах МО выборки красятся по метке траектории месяца, МО класса 2 показаны светлой штриховкой «тип только за декабрь 2024» (тип присвоен по данным 2024-12, шаг 24c), МО класса 4 той же штриховкой с подписью «тип по последнему доступному месяцу: ГГГГ-ММ» (месяц из колонки last_month файла layer_data_24f.parquet, шаг 24g), МО класса 3 остаются «тип не присвоен» со своим пояснением. "
        "Слои «Доступность рынков» и «Устойчивость типа» от месяца не зависят, селектор для них отключён с подписью. Кнопка «играть» переключает месяц раз в секунду. "
        "Кнопка «Скачать CSV» строит в браузере (Blob) файл types_by_month_2023_2024.csv: одна строка на МО выборки (2004), колонки territory_id, name, region_name и 24 месяца; в начале файла строки комментария с техническими именами типов; кодировка UTF-8 с BOM. "
        "В легенде у типа с неуверенным сопоставлением в месяце стоит значок ⚠; подсказка показывает метку месяца и метку декабря 2024; для МО с dec_label_differs панель показывает обе метки с пометкой. "
        f"ARI меток декабря 2024 в траекториях и канонических: {ctx['ari_dec']:.4f}. Результаты контролей: раздел 3 (строки про месяцы, метки и сопоставление).",
        "- Число МО по типам по месяцам (сумма 2004 в каждом месяце): " + "; ".join(f"{m}: " + "/".join(str(x) for x in c) for m, c in zip(ctx['months24'], ctx['month_counts'])) + ".", "",
        "## 3. Контроли (выполнены до записи файлов)", "",
        md_table(["контроль", "получено", "ожидалось", "статус"], [[a, b, c, "пройден" if d else "НАРУШЕН"] for a, b, c, d in CONTROLS]), "",
        "## 4. Сверка таблиц `notebooks/26` с `hypothesis_summary.parquet`", "",
        md_table(["таблица md", "строк в md", "строк parquet, отнесённых к таблице"],
                 [[{"main": "Проверки (Г1, Г8, Г2, Г3, Г4, Г6, Г9)", "g7": "Учёт возраста (Г7)", "g4": "Г4 по отраслям", "sup": "Г10–Г13, проверки",
                    "supd": "Г10–Г13, учёт показателя предложения"}[k], n, u] for k, h, n, u in rep["tables"]]), "",
        f"- Строк разобрано: {rep['rows_total']}; сверено с parquet: {rep['rows_checked']}; строк без сверки: {len(rep['rows_unchecked'])}.",
        f"- Сверено чисел: {rep['numbers_checked']}; строковых значений («результат», «размер»): {rep['strings_checked']}; дробных чисел без пары: {len(rep['tokens_unpaired'])}; "
        f"целых чисел без пары в parquet (пороги правил, номера гипотез, порядок типов, числа месяцев): {rep['ints_unpaired']}.",
        f"- Строк блоков «1. …» в parquet: {rep['P1']}; отнесено к таблицам md: {sum(t[3] for t in rep['tables'])} (в первичных колонках: {rep['used_primary']}, "
        f"в колонках «оговорка» и «результат»: {rep['used_folded']}); без места в таблицах: {len(rep['unused'])}.",
        f"- Блок «0. контроль»: {len(ctx['b0'])} строк parquet показаны над таблицами; значения совпали с числами строки о контроле в md.",
        f"- Раздел 3 md: пунктов {rep['s3']['bullets']}, сверено чисел {rep['s3']['numbers_checked']}, без пары {len(rep['s3']['unpaired'])}.",
        "- Строки без сверки: нет.", "",
        "## 5. Заморозка", "", *ctx["frozen_before"], *frozen_after, "",
        "## 6. Проверка в браузере", "", existing_browser_block(), "",
        "## 7. Замечания по данным", "", *[f"- {x}" for x in notes], "",
        "## 8. Ограничения", "",
        "- Браузеры без `DecompressionStream` или `Path2D`: интерактивная карта не строится, показываются сообщение и статическая карта (фигура 2); подпись: "
        "статическая карта показывает состояние на декабрь 2024 года, на интерактивной карте дополнительно показаны типы по последнему доступному месяцу.",
        f"- Размер файла: {info['size']} байт (предел {L['MAX_HTML_BYTES']}); большая часть приходится на геометрию и PNG фигуры 2.",
        (f"- Заглушек в texts.json нет ({25 - info['placeholders']} из 25 ключей заполнены)." if info["placeholders"] == 0 else
         f"- Тексты автора ещё не заполнены: на странице видны заглушки ({info['placeholders']} из 25 ключей `texts.json`)."),
        f"- {ctx['beyond'][0]} из {ctx['beyond'][1]} присвоенных МО класса 2 дальше 95-го процентиля расстояний выборки до центроидов (шаг 24c): тип таких МО менее надёжен.",
        "- Режима «равные клетки» и ленты по месяцам в карточке МО нет; в слое «Тип» есть минимальный селектор месяца (поправка 3).",
        "- Проверка в браузере в этой версии выполнена в Chrome (headless); Safari и Firefox не проверялись.", "",
        "## 9. Тексты", "",
        f"- Ключей во входном файле: {TEXTCHK['keys']} (в `texts.json` всего {len(ctx['texts'])}, `hero_title` не менялся).",
        f"- Проверено числовых токенов: {TEXTCHK['tokens']}; строк реестра `numbers_registry.csv`: {TEXTCHK['rows']}.",
        "- Запрещённых слов и основ в текстах: 0; заглушек: 0; квадратных скобок вне интервалов: 0.", "",
        md_table(["ключ", "символов", "числовых токенов", "строк реестра"], [list(x) for x in TEXTCHK["per_key"]]), "",
    ]
    text = "\n".join(lines)
    nobr = re.sub(r"\[[+-]?\d+\.\d+; [+-]?\d+\.\d+\]", "", text)
    if "[" in nobr or "]" in nobr:
        stop("в отчёте остались квадратные скобки вне формата интервала")
    hits = {w: len(re.findall(w, text, flags=re.I)) for w in L["FORBIDDEN_WORDS"]}
    if any(hits.values()):
        stop(f"в отчёте запрещённые слова: {hits}")
    return text


def build_site_readme() -> str:
    text = f'''# Лендинг: каркас

Страница `site/index.html` — один автономный файл: данные, стили, скрипты и геометрия внутри, внешних запросов нет (без CDN, шрифтов и картинок по сети).

## Как пересобрать

```bash
.venv/bin/python src/24e_landing.py
```

Скрипт читает готовые файлы проекта, выполняет контроли и только потом пишет `site/index.html`, `site/README.md` и `notebooks/24e_landing_check.md`. Два прогона подряд дают побайтно одинаковый `index.html`. Предварительно должны быть выполнены шаги 23–28, 24б, 24c, 24d, 24f, 24g и `24_figures` (`bash run_all.sh`).

## Как проверить в браузере

```bash
.venv/bin/python src/24e_landing_test.py
```

Тест запускает Chrome headless (если он есть; иначе печатает «Chrome не найден»), внедряет `site/tests/landing_check.js`, проверяет карту, фильтр, метки по месяцам, размеры интерактивных элементов на ширине 390 px, горизонтальную прокрутку и CSV, печатает таблицы, пишет скриншоты в `/tmp/step24e_shots/` и результаты в `/tmp/step24e_test_results.json`, а итог подставляет в блок «Проверка в браузере» файла `notebooks/24e_landing_check.md`.

## Как открыть

Дважды щёлкните `site/index.html` или откройте файл в браузере (адрес вида `file:///…/site/index.html`). Сервер не нужен. Карте нужен современный браузер с `DecompressionStream` (Chrome, Edge, Firefox, Safari последних версий); без него показывается статическая карта.

## Где менять тексты

Все проза и подписи лежат в `site/content/texts.json` (25 ключей, обычный JSON в UTF-8). Пока ключ содержит заглушку «текст будет добавлен автором» (в квадратных скобках), на странице виден жёлтый блок. После правки файла выполните пересборку командой выше. Скрипт не перезаписывает `texts.json`; набор ключей задан в `config.yaml` (`step24e_landing.TEXT_KEYS`), лишние и недостающие ключи останавливают сборку. Абзацы разделяются пустой строкой. Числа в тексты лучше не вписывать без источника.

## Файл CSV

В Excel с русской локалью открывайте файл через «Данные, Из текста/CSV» с разделителем запятая; в pandas читайте с параметром comment="#".

## Как опубликовать на GitHub Pages

Решение за автором. Варианты:

1. Отдельная ветка (например, `gh-pages`), в корень которой копируется только `site/index.html`; в настройках репозитория: Settings → Pages → Deploy from a branch, папка `/ (root)`. Файл самодостаточен, остальные файлы страницы не нужны.
2. GitHub Actions: workflow загружает папку `site/` как артефакт Pages; ветку с копией файла вести не нужно.
3. Копия `index.html` в `docs/` и выбор папки `/docs` в настройках Pages (стандартный выбор для Deploy from a branch — корень или `/docs`; папка `/site` в этом списке не предлагается). В `docs/` лежат описания проекта, они станут доступны на сайте.

Перед публикацией проверьте тексты (`site/content/texts.json`) и подпись источника границ: её нельзя убирать (CC BY-SA 4.0, Contains data © OpenStreetMap contributors).
'''
    return text


# ============================================================================ запуск
def main() -> None:
    ctx = build_context()
    check_texts_input(ctx["texts"])
    page, verbatim = build_html(ctx)
    info = page_checks(page, verbatim, ctx)
    # во второй раз заморозка проверяется после записи; отчёты строятся по результатам
    page_bytes = page.encode("utf-8")
    page_sha = hashlib.sha256(page_bytes).hexdigest()
    frozen_after_pre = check_frozen("до записи (повторно)")
    readme = build_site_readme()
    report = build_report(ctx, info, frozen_after_pre, page_sha)
    stem_md = {w: len(re.findall(re.escape(w), report, flags=re.I)) for w in L["FORBIDDEN_STEMS"]}
    stem_rd = {w: len(re.findall(re.escape(w), readme, flags=re.I)) for w in L["FORBIDDEN_STEMS"]}
    ctl("запрещённые основы (правило 2.3) в notebooks/24e_landing_check.md целиком", {k: v for k, v in stem_md.items() if v}, {})
    ctl("запрещённые основы (правило 2.3) в site/README.md целиком", {k: v for k, v in stem_rd.items() if v}, {})
    ctl("md: блок browser-check присутствует (маркеры begin и end по одному разу)", (report.count(BROWSER_BEGIN), report.count(BROWSER_END)), (1, 1))
    report = build_report(ctx, info, frozen_after_pre, page_sha)       # повтор: таблица контролей включает две последние строки
    nobr = re.sub(r"\[[+-]?\d+\.\d+; [+-]?\d+\.\d+\]", "", readme)
    if "[" in nobr or "]" in nobr:
        stop("в site/README.md квадратные скобки вне формата интервала")
    rh = {w: len(re.findall(w, readme, flags=re.I)) for w in L["FORBIDDEN_WORDS"]}
    if any(rh.values()):
        stop(f"в site/README.md запрещённые слова: {rh}")
    # ---- все контроли пройдены: запись ----
    out = P(L["OUT_HTML_PATH"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(page_bytes)
    if ctx["texts_new"]:
        tp = P(L["TEXTS_PATH"])
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(texts_json_dump(ctx["texts"]), encoding="utf-8")
    P(L["SITE_README_PATH"]).write_text(readme, encoding="utf-8")
    after = check_frozen("после")
    report = report.replace("\n".join(frozen_after_pre), "\n".join(frozen_after_pre + after))
    P(L["REPORT_PATH"]).write_text(report, encoding="utf-8")
    print(f"записано: {out} ({info['size']} байт), {L['SITE_README_PATH']}, {L['REPORT_PATH']}"
          f"{', ' + L['TEXTS_PATH'] if ctx['texts_new'] else ''}")
    print("sha256 index.html:", page_sha)


if __name__ == "__main__":
    main()
