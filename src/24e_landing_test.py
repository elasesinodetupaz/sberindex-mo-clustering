"""Шаг 24e: проверка site/index.html в Chrome headless (протокол DevTools, только стандартная библиотека).

Внедряет site/tests/landing_check.js в страницу, прогоняет проверки на трёх размерах окна, сверяет CSV
с data/processed/kmeans_k6_trajectories.parquet, печатает таблицы, пишет результаты в /tmp (не в репозиторий)
и подставляет итог в блок «Проверка в браузере» notebooks/24e_landing_check.md (между маркерами browser-check).
Запуск: .venv/bin/python src/24e_landing_test.py (без Chrome: сообщение «Chrome не найден», код выхода 0).
"""
import base64
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pandas as pd

import config as C

PROJECT_DIR = Path(__file__).resolve().parents[1]
L = C.LANDING
P = lambda rel: PROJECT_DIR / rel  # noqa: E731
CHROME_CANDIDATES = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium", "chromium-browser"]
BEGIN, END = "<!-- browser-check:begin -->", "<!-- browser-check:end -->"
IDS = {"район Москвы": 2462, "Якутск": 309, "Новый Уренгой": 2242, "Сут-Хольский район (409, без типа)": 409,
       "МО класса 2": 64, "МО класса 4": 85, "МО с dec_label_differs": 46}
KEYS = {"район Москвы": "moscow", "Якутск": "yakutsk", "Новый Уренгой": "urengoy", "Сут-Хольский район (409, без типа)": "sut409",
        "МО класса 2": "cls2", "МО класса 4": "cls4", "МО с dec_label_differs": "differs"}


class CDP:
    """Минимальный клиент WebSocket для DevTools."""

    def __init__(self, ws_url: str):
        host, rest = ws_url[5:].lstrip("/").split("/", 1)
        h, p = host.split(":")
        self.s = socket.create_connection((h, int(p)), timeout=120)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET /{rest} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        self.buf = buf.split(b"\r\n\r\n", 1)[1]
        self.id = 0
        self.events: list[dict] = []

    def _read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.s.recv(1 << 20)
            if not chunk:
                raise ConnectionError("соединение закрыто")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _frame(self) -> str:
        data = b""
        while True:
            b1, b2 = self._read(2)
            ln = b2 & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._read(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(ln)
            op = b1 & 0x0F
            if op == 9:
                continue
            data += payload
            if b1 & 0x80:
                return data.decode("utf-8")

    def call(self, method: str, params: dict | None = None, timeout: float = 120) -> dict:
        self.id += 1
        msg = json.dumps({"id": self.id, "method": method, "params": params or {}}).encode()
        mask = os.urandom(4)
        hdr = bytes([0x81])
        if len(msg) < 126:
            hdr += bytes([0x80 | len(msg)])
        elif len(msg) < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", len(msg))
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", len(msg))
        self.s.sendall(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(msg)))
        end = time.time() + timeout
        while time.time() < end:
            m = json.loads(self._frame())
            if m.get("id") == self.id:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})
            self.events.append(m)
        raise TimeoutError(method)

    def pump(self, seconds: float) -> None:
        self.s.settimeout(0.2)
        end = time.time() + seconds
        while time.time() < end:
            try:
                self.events.append(json.loads(self._frame()))
            except (socket.timeout, TimeoutError):
                pass
        self.s.settimeout(120)


def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        p = c if os.path.isabs(c) else shutil.which(c)
        if p and os.path.exists(p):
            return p
    return None


def evaluate(cdp: CDP, expr: str, await_promise: bool = False):
    r = cdp.call("Runtime.evaluate", {"expression": expr, "awaitPromise": await_promise, "returnByValue": True})
    if "exceptionDetails" in r:
        raise RuntimeError(json.dumps(r["exceptionDetails"], ensure_ascii=False)[:500])
    return r["result"].get("value")


def console_summary(events: list[dict]) -> dict:
    errors, warns = [], []
    for e in events:
        m = e.get("method")
        if m == "Runtime.exceptionThrown":
            errors.append("исключение: " + e["params"]["exceptionDetails"].get("text", "") + " " + json.dumps(e["params"]["exceptionDetails"].get("exception", {}).get("description", ""), ensure_ascii=False)[:200])
        elif m == "Runtime.consoleAPICalled":
            txt = " ".join(str(a.get("value", a.get("description", ""))) for a in e["params"]["args"])[:200]
            if e["params"]["type"] == "error":
                errors.append("console.error: " + txt)
            elif e["params"]["type"] == "warning":
                warns.append(txt)
        elif m == "Log.entryAdded" and e["params"]["entry"]["level"] == "error":
            errors.append("журнал: " + e["params"]["entry"]["text"][:200] + " " + e["params"]["entry"].get("url", "")[:100])
    return {"errors": errors, "warnings": len(warns), "warning_kinds": sorted(set(w[:90] for w in warns))}


def run_viewport(chrome_ws: str, w: int, h: int, cfg: dict, js: str, shots: Path) -> dict:
    cdp = CDP(chrome_ws)
    for m in ("Page.enable", "Runtime.enable", "Log.enable"):
        cdp.call(m)
    mobile = w < 500
    cdp.call("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": 2 if mobile else 1, "mobile": mobile})
    if mobile:
        cdp.call("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 5})
    cdp.call("Page.navigate", {"url": P(L["OUT_HTML_PATH"]).as_uri()})
    t0 = time.time()
    while time.time() - t0 < 90:
        cdp.pump(0.3)
        if evaluate(cdp, "!!(window.__test && window.__test.ready)"):
            break
    evaluate(cdp, js)
    res = evaluate(cdp, f"window.__landingCheck({json.dumps(cfg)})", True)
    res["viewport"] = [w, h]
    evaluate(cdp, "__test.reset(); window.scrollTo(0, 0)")
    cdp.pump(0.4)
    evaluate(cdp, "window.scrollTo(0, 0)")
    cdp.pump(0.5)
    path = shots / f"{w}x{h}_top.png"
    path.write_bytes(base64.b64decode(cdp.call("Page.captureScreenshot", {"format": "png"})["data"]))
    res.setdefault("shots", []).append(str(path))
    # карта целиком: прокрутка к #map-ui, выбор МО 409 (панель), снимок области блока вместе с легендой и панелью
    evaluate(cdp, "window.scrollTo({ top: 0, behavior: 'instant' })")
    cdp.pump(0.3)
    res["mapNav"] = evaluate(cdp, "window.__clickNav('#map')", True)    # как у пользователя: клик по ссылке «Карта», меню остаётся липким
    # окно на время снимка выше блока карты: меню стоит сверху на своём месте, весь блок виден без съёмки за пределами экрана
    r0 = evaluate(cdp, "(function(){var b=document.getElementById('map-ui').getBoundingClientRect();return {top:b.top,h:b.height};})()")
    need = int(r0["top"] + r0["h"] + 12)
    if need > h:
        cdp.call("Emulation.setDeviceMetricsOverride", {"width": w, "height": need, "deviceScaleFactor": 2 if mobile else 1, "mobile": mobile})
        cdp.pump(0.6)
        res["mapNav"] = evaluate(cdp, "window.scrollTo({ top: 0, behavior: 'instant' }); window.__clickNav('#map')", True)
    evaluate(cdp, f"__test.select({cfg['ids']['sut409']})")
    cdp.pump(0.6)
    rects = evaluate(cdp, "(function(){var r=function(s){var b=document.querySelector(s).getBoundingClientRect();return {x:b.left+scrollX,y:b.top+scrollY,w:b.width,h:b.height};};"
                          "return {ui:r('#map-ui'),canvas:r('#map-canvas'),legend:r('#map-legend'),panel:r('#map-panel'),vw:innerWidth};})()")
    for tag, key in (("map", "ui"), ("legend", "legend")):
        rc = rects[key]
        clip = {"x": 0 if key == "ui" else rc["x"], "y": rc["y"], "width": rects["vw"] if key == "ui" else rc["w"], "height": rc["h"], "scale": 1}
        png = base64.b64decode(cdp.call("Page.captureScreenshot", {"format": "png", "clip": clip, "captureBeyondViewport": False})["data"])
        path = shots / f"{w}x{h}_{tag}.png"
        path.write_bytes(png)
        res["shots"].append(str(path))
        if key == "ui":
            iw, ih = struct.unpack(">II", png[16:24])
            res["mapShot"] = {"file": str(path), "px": [iw, ih], "rects": rects, "inside": all(rects[k]["y"] >= rects["ui"]["y"] - 0.5 and rects[k]["y"] + rects[k]["h"] <= rects["ui"]["y"] + rects["ui"]["h"] + 0.5 for k in ("canvas", "legend", "panel"))}
            try:
                from PIL import Image
                import io
                im = Image.open(io.BytesIO(png)).convert("RGB")
                k = iw / rects["vw"]
                cx0, cy0 = int((rects["canvas"]["x"]) * k), int((rects["canvas"]["y"] - rects["ui"]["y"]) * k)
                cr = im.crop((cx0, cy0, cx0 + int(rects["canvas"]["w"] * k), cy0 + int(rects["canvas"]["h"] * k)))
                bgc = tuple(int(L["MAP_BG"][i:i + 2], 16) for i in (1, 3, 5))
                px = list(cr.get_flattened_data() if hasattr(cr, 'get_flattened_data') else cr.getdata())
                res["mapShot"]["painted_share"] = round(sum(1 for q in px if sum(abs(q[i] - bgc[i]) for i in range(3)) > 6) / max(1, len(px)), 3)
            except ImportError:
                res["mapShot"]["painted_share"] = None
    evaluate(cdp, "__test.reset()")
    cdp.call("Emulation.setDeviceMetricsOverride", {"width": w, "height": h, "deviceScaleFactor": 2 if mobile else 1, "mobile": mobile})
    res["console"] = console_summary(cdp.events)
    res["size"] = None
    return res


def table(rows: list[list], head: list[str]) -> str:
    cols = list(zip(*([head] + [[str(c) for c in r] for r in rows])))
    wd = [max(len(c) for c in col) for col in cols]
    fm = lambda r: "| " + " | ".join(str(c).ljust(wd[i]) for i, c in enumerate(r)) + " |"  # noqa: E731
    return "\n".join([fm(head), "|" + "|".join("-" * (x + 2) for x in wd) + "|"] + [fm(r) for r in rows])


def md_table(rows: list[list], head: list[str]) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c).replace("|", "/") for c in r) + " |" for r in rows])


def main() -> int:
    chrome = find_chrome()
    if not chrome:
        print("Chrome не найден")
        return 0
    html = P(L["OUT_HTML_PATH"])
    js = (PROJECT_DIR / "site" / "tests" / "landing_check.js").read_text(encoding="utf-8")
    lay = pd.read_parquet(P(L["LAYER_PATH"]))
    traj = pd.read_parquet(P(L["TRAJECTORIES_PATH"]))
    typ0 = int(lay[(lay["class"] == 1) & (lay["type"] == 0)].sort_values("territory_id")["territory_id"].iloc[0])
    cfg = {"ids": {KEYS[k]: v for k, v in IDS.items()}, "typ0Id": typ0, "minPx": L["MIN_TOUCH_PX"], "textMarker": L["TEXT_PLACEHOLDER"], "filterTol": L["TEST_FILTER_TOL"]}
    cfg["waitMs"] = L["TEST_FILTER_WAIT_MS"]
    c4 = lay[lay["class"] == 4]
    cfg["cls4Last"] = {str(int(t)): m for t, m in zip(c4["territory_id"], c4["last_month"])}
    shots = Path(L["TEST_SHOTS_DIR"])
    shutil.rmtree(shots, ignore_errors=True)
    shots.mkdir(parents=True)
    prof = tempfile.mkdtemp(prefix="chrome24e_")
    proc = subprocess.Popen([chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", f"--user-data-dir={prof}",
                             "--remote-debugging-port=0", "--hide-scrollbars", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = []
    try:
        pf = Path(prof) / "DevToolsActivePort"
        for _ in range(100):
            if pf.exists():
                break
            time.sleep(0.2)
        port = pf.read_text().split()[0]
        for w, h in L["TEST_VIEWPORTS"]:
            tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/new?about:blank", data=b"")) if False else None
            req = urllib.request.Request(f"http://127.0.0.1:{port}/json/new?about:blank", method="PUT")
            tab = json.load(urllib.request.urlopen(req))
            results.append(run_viewport(tab["webSocketDebuggerUrl"], w, h, cfg, js, shots))
            urllib.request.urlopen(f"http://127.0.0.1:{port}/json/close/{tab['id']}").read()
    finally:
        proc.terminate()
        shutil.rmtree(prof, ignore_errors=True)

    # --- сверка CSV с траекториями
    r0 = results[0]
    piv = traj.pivot(index="territory_id", columns="month", values="cluster")
    months = list(piv.columns)
    hdr_ok = r0["csv"]["header"] == ["territory_id", "name", "region_name"] + months
    byid = {int(r[0]): r for r in r0["csv"]["rows"]}
    mism_all = sum(1 for tid, r in byid.items() if [int(x) for x in r[3:]] != [int(x) for x in piv.loc[tid].tolist()])
    names = lay.set_index("territory_id")
    meta_bad = sum(1 for tid, r in byid.items() if r[1] != names.loc[tid, "name"] or r[2] != names.loc[tid, "region_name"])
    sample_ids = sorted(byid)[::max(1, len(byid) // L["TEST_CSV_SAMPLE"])][:L["TEST_CSV_SAMPLE"]]
    sample = [[tid, byid[tid][1][:40]] + [("совпало" if [int(x) for x in byid[tid][3:]] == [int(x) for x in piv.loc[tid].tolist()] else "РАСХОЖДЕНИЕ")] for tid in sample_ids]
    bad_comments = [c for c in r0["csv"]["comments"] if any(ch in c[1:] for ch in (",", ";", '"'))]
    csv_ok = (not bad_comments and len(r0["csv"]["comments"]) == L["EXPECT_CSV_COMMENT_LINES"] and r0["csv"]["bom"] and r0["csv"]["dataRows"] == L["EXPECT_CSV_ROWS"] and r0["csv"]["cols"] == [L["EXPECT_CSV_COLS"]] and r0["csv"]["headerCols"] == L["EXPECT_CSV_COLS"]
              and hdr_ok and mism_all == 0 and meta_bad == 0 and len(set(byid)) == L["EXPECT_CSV_ROWS"]
              and r0["csvButton"].get("download") == L["CSV_FILENAME"] and r0["csvButton"].get("size", 0) > 0)

    # --- вывод
    fails = []
    map_shots = {r["viewport"][0]: r["mapShot"]["file"] for r in results}
    print("снимки карты (1440 и 390):", map_shots[1440], map_shots[390])
    print(f"Chrome: {chrome}; страница: {html}")
    for r in results:
        w, h = r["viewport"]
        print(f"\n=== окно {w}×{h} ===")
        print("консоль: ошибок", len(r["console"]["errors"]), "; предупреждений", r["console"]["warnings"], r["console"]["warning_kinds"])
        for e in r["console"]["errors"]:
            print("  ", e)
        if r["console"]["errors"]:
            fails.append(f"{w}: ошибки консоли")
        print("карта нарисована, пикселей:", r["painted"])
        if r["painted"]["painted"] <= 0:
            fails.append(f"{w}: карта пуста")
        print(table([[x["name"], x["id"], x["hit"], x["selected"], "да" if x["ok"] else "НЕТ"] for x in r["hit"]], ["МО", "id", "hitTest", "select", "совпало"]))
        fails += [f"{w}: hitTest {x['name']}" for x in r["hit"] if not x["ok"]]
        cp = r["captions"]
        bad4 = cp["cls4_bad"]
        c2_ok = cp["cls2"]["2023-06"]["has_dec_only"] and cp["cls2"]["2023-06"]["panel_has_dec_only"] and not cp["cls2"]["2024-12"]["has_dec_only"]
        c3_ok = all(cp["cls3"][m]["key"] == "3" and cp["cls3"][m]["not_assigned"] and not cp["cls3"][m]["has_dec_only"] and not cp["cls3"][m]["panel_has_dec_only"] for m in ("2023-06", "2024-12"))
        print("подписи: МО класса 4 проверено", cp["cls4_checked"], "× 2 режима, расхождений с last_month:", len(bad4), "| МО 64 (прежняя подпись в 2023-06):", c2_ok, "| МО 409 (без декабрьской подписи):", c3_ok)
        if bad4 or not c2_ok or not c3_ok:
            fails.append(f"{w}: подписи в режиме месяца (класс 4: {bad4[:3]})")
        print(table([[f["name"], q["kind"], q["before"], q["during"], q["after"], "да" if q["changed"] else "НЕТ", "да" if q["returned"] else "НЕТ"] for f in r["filter"] for q in f["points"]],
                    ["сценарий", "точка", "до клика", "после клика 1", "после клика 2", "цвет изменился", "цвет вернулся"]))
        fails += [f"{w}: фильтр {f['name']} ({q['kind']})" for f in r["filter"] for q in f["points"] if not (q["changed"] and q["returned"])]
        fails += [f"{w}: фильтр {f['name']}: заливка и штрих не различаются" for f in r["filter"] if not f.get("distinct", True)]
        ui = r["ui"]
        print("поле поиска:", ui["placeholder"], "| scrollWidth", ui["scrollWidth"], "clientWidth", ui["clientWidth"], "| текст", ui["textWidth"], "px, доступно", ui["innerWidth"], "px | виден целиком:", ui["placeholder_fits"])
        mn = ui["menu"]
        print("меню: scrollWidth", mn["scrollWidth"], "clientWidth", mn["clientWidth"], "прокручивается:", mn["scrollable"], "| последняя ссылка", mn["last"], "достижима:", mn["last_reachable"], "| страница прокручивается по горизонтали:", mn["page_scrolls_x"])
        if not ui["placeholder_fits"] or not mn["last_reachable"] or mn["page_scrolls_x"]:
            fails.append(f"{w}: плейсхолдер или меню")
        nv = r["nav"]
        print("якорная навигация (отступ заголовка секции от нижней границы меню, px):", ", ".join(f"{x['text']}: {x['headingOffset']}" for x in nv))
        ov = [x for x in nv if x["headingOffset"] < 0]
        if ov:
            fails.append(f"{w}: заголовок под меню после клика: {[x['text'] for x in ov]}")
        ms = r["mapShot"]
        print("снимок карты:", ms["file"], ms["px"], "canvas, легенда и панель внутри области:", ms["inside"], "закрашено в области canvas:", ms["painted_share"])
        if not ms["inside"] or not ms["painted_share"] or ms["painted_share"] < 0.05:
            fails.append(f"{w}: снимок карты")
        tx = r["texts"]
        tx_ok = tx["keys"] == sorted(L["TEXT_KEYS"]) and tx["markerVisible"] == 0 and tx["markerKeyed"] == 0 and tx["phElements"] == 0 and tx["emptyKeyed"] == 0
        print("тексты: ключей на странице", len(tx["keys"]), "из", len(L["TEXT_KEYS"]), "| маркер заглушки в видимом тексте", tx["markerVisible"], "| элементов .ph", tx["phElements"], "| пустых элементов с ключом", tx["emptyKeyed"], "| совпало:", tx_ok)
        if not tx_ok:
            fails.append(f"{w}: тексты (ключи или заглушки)")
        sc = r["scroll"]
        print("горизонтальная прокрутка: ширина документа", sc["docW"], "окно", sc["winW"], "нет прокрутки:", sc["ok"])
        if not sc["ok"]:
            fails.append(f"{w}: горизонтальная прокрутка")
        print("карточек", r["counts"]["cards"], "; строк покрытия", r["counts"]["coverageRows"], "; svg", r["counts"]["svg"])
        if r["counts"]["cards"] != 6 or r["counts"]["coverageRows"] != 85:
            fails.append(f"{w}: число карточек или строк покрытия")
    narrow = [r for r in results if r["viewport"][0] == L["TEST_NARROW_PX"]][0]
    groups: dict = {}
    for e in narrow["interactive"]:
        sig = e["el"].split(" «")[0]
        g = groups.setdefault(sig, {"n": 0, "w": [], "h": [], "ex": e["el"]})
        g["n"] += 1
        g["w"].append(e["w"])
        g["h"].append(e["h"])
    grows = [[sig, g["n"], f"{min(g['w']):g}–{max(g['w']):g}", f"{min(g['h']):g}–{max(g['h']):g}", g["ex"]] for sig, g in groups.items()]
    print(f"\n=== интерактивные элементы, окно {L['TEST_NARROW_PX']} px: всего {len(narrow['interactive'])}, ниже или уже {L['MIN_TOUCH_PX']} px: {len(narrow['small'])} ===")
    print(table(grows, ["элемент", "число", "ширина, px", "высота, px", "пример"]))
    for e in narrow["small"]:
        print("  МЕНЬШЕ ПОРОГА:", e)
    if narrow["small"]:
        fails.append("элементы меньше порога на узком окне")
    print("\n=== CSV ===")
    print("BOM:", r0["csv"]["bom"], "; строк данных:", r0["csv"]["dataRows"], "; колонок в строках:", r0["csv"]["cols"], "; колонок в заголовке:", r0["csv"]["headerCols"],
          "; строк комментария:", len(r0["csv"]["comments"]), "; из них с «,» «;» или кавычкой:", len(bad_comments), "; расхождений с траекториями (все 2004 МО):", mism_all, "; расхождений названий и регионов:", meta_bad)
    print("кнопка:", r0["csvButton"])
    print(table(sample, ["territory_id", "название", "24 месяца против kmeans_k6_trajectories.parquet"]))
    for c in r0["csv"]["comments"]:
        print("  ", c)
    if not csv_ok:
        fails.append("CSV")
    print("\nскриншоты:", *[s for r in results for s in r["shots"]], sep="\n  ")
    Path(L["TEST_RESULTS_PATH"]).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("результаты:", L["TEST_RESULTS_PATH"])

    # --- блок в md
    sha = hashlib.sha256(html.read_bytes()).hexdigest()
    out = [BEGIN, f"Результаты `src/24e_landing_test.py` (Chrome headless, `site/tests/landing_check.js`); проверяемая страница: sha256 `{sha}`. Скриншоты сохранены в `{L['TEST_SHOTS_DIR']}/` (вне репозитория).", ""]
    out.append("**Консоль и карта**\n")
    out.append(md_table([[f"{r['viewport'][0]}×{r['viewport'][1]}", len(r["console"]["errors"]), r["console"]["warnings"], r["painted"]["painted"], r["painted"]["total"]] for r in results],
                        ["окно", "ошибок консоли", "предупреждений консоли", "закрашено пикселей карты", "всего пикселей"]))
    out.append("\nПредупреждения консоли: " + (", ".join(sorted({k for r in results for k in r["console"]["warning_kinds"]})) or "нет") + ".\n")
    out.append("**hitTest и select по 7 МО** (`window.__test`: `zoomTo`, `anchor`, `hitTest`, `select`)\n")
    out.append(md_table([[x["name"], x["id"], *[("да" if rr["hit"][i]["ok"] else "НЕТ") for rr in results]] for i, x in enumerate(results[0]["hit"])],
                        ["МО", "id"] + [f"{r['viewport'][0]} px" for r in results]))
    out.append("\n**Метки по месяцам** (`getLabel`: метка траектории месяца / каноническая метка декабря 2024; пусто у МО вне выборки)\n")
    mrows = []
    for name in IDS:
        row = [name, IDS[name]]
        for m in ("2023-01", "2023-12", "2024-12"):
            x = [a for a in results[0]["months"] if a["month"] == m and a["id"] == IDS[name]][0]
            row.append("нет метки" if x["label"] is None else f"{x['label']} / {x['canonical']}")
        mrows.append(row)
    out.append(md_table(mrows, ["МО", "id", "2023-01", "2023-12", "2024-12"]))
    out.append("\n**Подписи МО классов 2, 4 и 3 в подсказке и панели** (подпись класса 4 сверяется с last_month из `layer_data_24f.parquet` по точному месяцу)\n")
    crow = []
    for rr in results:
        cp = rr["captions"]
        c85 = {x["mode"]: x for x in cp["cls4"]}
        crow.append([f"{rr['viewport'][0]} px", cp["cls4_checked"], len(cp["cls4_bad"]),
                     " / ".join(f"{m}: {c85[m]['last_month']}, строка {'да' if c85[m]['row_ok'] else 'НЕТ'}, подпись {'да' if c85[m]['caption_ok'] else 'НЕТ'}" for m in ("2023-06", "2024-12")),
                     "да" if cp["cls2"]["2023-06"]["has_dec_only"] and not cp["cls2"]["2024-12"]["has_dec_only"] else "НЕТ",
                     "да" if all(cp["cls3"][m]["not_assigned"] and not cp["cls3"][m]["has_dec_only"] for m in ("2023-06", "2024-12")) else "НЕТ"])
    out.append(md_table(crow, ["окно", "МО класса 4 проверено (режимы 2023-06 и 2024-12)", "расхождений с last_month", "МО 85", "МО 64: «тип только за декабрь 2024» только в режиме месяца", "МО 409: «тип не присвоен», декабрьской подписи нет"]))
    out.append(f"\n**Фильтр по типу** (цвет пикселя в точке МО после двух кадров requestAnimationFrame и ожидания {L['TEST_FILTER_WAIT_MS']} мс; допуск {L['TEST_FILTER_TOL']} из 255 по каналу; для штриховки точка на заливке и точка на штрихе сравниваются отдельно)\n")
    out.append(md_table([[f"{rr['viewport'][0]}×{rr['viewport'][1]}", f["name"], q["kind"], q["before"], q["during"], q["after"], "да" if q["changed"] else "нет", "да" if q["returned"] else "нет"]
                         for rr in results for f in rr["filter"] for q in f["points"]],
                        ["окно", "сценарий", "точка", "до клика", "после клика 1", "после клика 2", "цвет изменился", "цвет вернулся"]))
    out.append("\n**Поле поиска и меню** (placeholder сверяется по ширине текста canvas.measureText против внутренней ширины поля; меню — прокрутка внутри блока `nav.top ul`)\n")
    out.append(md_table([[f"{rr['viewport'][0]} px", rr["ui"]["placeholder"], rr["ui"]["scrollWidth"], rr["ui"]["clientWidth"], rr["ui"]["textWidth"], rr["ui"]["innerWidth"], "да" if rr["ui"]["placeholder_fits"] else "НЕТ",
                          f"{rr['ui']['menu']['scrollWidth']} / {rr['ui']['menu']['clientWidth']}", "да" if rr["ui"]["menu"]["last_reachable"] else "НЕТ", "нет" if not rr["ui"]["menu"]["page_scrolls_x"] else "есть"] for rr in results],
                        ["окно", "placeholder", "scrollWidth поля", "clientWidth поля", "ширина текста, px", "доступно, px", "виден целиком", "меню: scrollWidth / clientWidth", "последняя ссылка достижима", "прокрутка страницы по горизонтали"]))
    out.append(f"\n**Интерактивные элементы, окно {L['TEST_NARROW_PX']} px** (`getBoundingClientRect`; селектор button, a, input, select, textarea, summary, [role=button], .leg-item; порог {L['MIN_TOUCH_PX']} px). Всего видимых: {len(narrow['interactive'])}; ниже порога: {len(narrow['small'])}.\n")
    out.append(md_table([[sig, g["n"], f"{min(g['w']):g}–{max(g['w']):g}", f"{min(g['h']):g}–{max(g['h']):g}"] for sig, g in groups.items()], ["элемент", "число", "ширина, px", "высота, px"]))
    if narrow["small"]:
        out.append("\nЭлементы ниже порога:\n")
        out.append(md_table([[e["el"], e["w"], e["h"]] for e in narrow["small"]], ["элемент", "ширина, px", "высота, px"]))
    out.append("\n**Тексты на странице** (все элементы с `data-text-key`; маркер заглушки ищется в видимом тексте `innerText`)\n")
    out.append(md_table([[f"{rr['viewport'][0]} px", f"{len(rr['texts']['keys'])} из {len(L['TEXT_KEYS'])}", "да" if rr["texts"]["keys"] == sorted(L["TEXT_KEYS"]) else "НЕТ",
                          rr["texts"]["markerVisible"], rr["texts"]["phElements"], rr["texts"]["emptyKeyed"]] for rr in results],
                        ["окно", "ключей на странице", "набор ключей совпал", "маркер заглушки в видимом тексте", "элементов с классом ph", "пустых элементов с ключом"]))
    out.append("\n**Горизонтальная прокрутка страницы**\n")
    out.append(md_table([[f"{r['viewport'][0]}", r["scroll"]["docW"], r["scroll"]["winW"], "нет" if r["scroll"]["ok"] else "есть"] for r in results], ["окно, px", "ширина документа", "ширина окна", "горизонтальная прокрутка"]))
    out.append(f"\n**CSV** (`window.__test.buildCsv()` и кнопка «Скачать CSV»; сверка с `kmeans_k6_trajectories.parquet`)\n")
    out.append(md_table([["кодировка: BOM в начале", "да" if r0["csv"]["bom"] else "нет"], ["строк данных", r0["csv"]["dataRows"]], ["колонок в строках данных", r0["csv"]["cols"][0] if r0["csv"]["cols"] else "-"],
                         ["колонок в заголовке", r0["csv"]["headerCols"]], ["строк комментария (начинаются с #)", len(r0["csv"]["comments"])],
                         ["расхождений с траекториями по всем МО", mism_all], ["расхождений названий и регионов с layer_data_24f.parquet", meta_bad], ["строк комментария с «,» «;» или кавычкой", len(bad_comments)],
                         ["имя файла у кнопки", r0["csvButton"].get("download")], ["размер Blob, байт", r0["csvButton"].get("size")]], ["показатель", "значение"]))
    out.append("\nСверка 10 МО (24 месяца, точное совпадение):\n")
    out.append(md_table(sample, ["territory_id", "название", "результат"]))
    out.append("\nКомментарии в начале файла:\n")
    out += [f"    {c}" for c in r0["csv"]["comments"]]
    out.append(f"\n**Якорная навигация** (клик по ссылке меню, плавная прокрутка завершена; нижняя граница липкого меню против верха секции и её заголовка h2 через getBoundingClientRect; перекрытие: заголовок выше нижней границы меню)\n")
    out.append(md_table([[x["text"], f"{rr['viewport'][0]} px", x["navBottom"], x["secOffset"], x["headingOffset"], "да" if x["headingOffset"] < 0 else "нет"] for rr in results for x in rr["nav"]],
                        ["ссылка", "окно", "нижняя граница меню, px", "отступ секции, px", "отступ заголовка, px", "перекрытие заголовка"]))
    out.append("\n**Снимки карты** (клик по ссылке «Карта», меню липкое и не отключалось; выбран МО 409; область блока целиком: карта, панель МО и легенда; `*_legend.png` — только легенда)\n")
    out.append(md_table([[f"{rr['viewport'][0]}×{rr['viewport'][1]}", f"`{rr['mapShot']['file']}`", f"{rr['mapShot']['px'][0]}×{rr['mapShot']['px'][1]}", "да" if rr["mapShot"]["inside"] else "НЕТ", rr["mapShot"]["painted_share"]] for rr in results],
                        ["окно", "файл", "размер снимка, px", "canvas, панель и легенда внутри области", "доля закрашенных пикселей в области canvas"]))
    out.append(f"\nВсе снимки (вне репозитория): " + ", ".join(f"`{s}`" for r in results for s in r["shots"]) + ".")
    out.append(END)
    block = "\n".join(out).replace("[", "(").replace("]", ")")   # квадратные скобки в отчёте допустимы только в формате интервала
    nb = P(L["REPORT_PATH"])
    t = nb.read_text(encoding="utf-8")
    if BEGIN in t and END in t:
        t = t[:t.index(BEGIN)] + block + t[t.index(END) + len(END):]
        nb.write_text(t, encoding="utf-8")
    if fails:
        print("\nНЕ ПРОЙДЕНО:", *fails, sep="\n  ")
        return 1
    print("\nвсе проверки пройдены")
    return 0


if __name__ == "__main__":
    sys.exit(main())
