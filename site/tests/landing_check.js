/* Тест страницы site/index.html: внедряется в страницу из src/24e_landing_test.py (Chrome headless).
   Вызов: await window.__landingCheck(cfg); cfg = {ids: {имя: id}, typ0Id: id МО выборки типа 0, minPx: 40, csvName: имя файла}. */
(function () {
  "use strict";
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function raf2() { return new Promise(function (r) { requestAnimationFrame(function () { requestAnimationFrame(r); }); }); }
  async function settle(ms) { await raf2(); await sleep(ms); }
  function visible(e) { var r = e.getBoundingClientRect(), cs = getComputedStyle(e); return r.width > 0 && r.height > 0 && cs.visibility !== "hidden" && cs.display !== "none"; }
  function label(e) { return e.tagName.toLowerCase() + (e.id ? "#" + e.id : "") + (e.className && typeof e.className === "string" ? "." + e.className.split(/\s+/)[0] : "") + (e.type && e.tagName === "INPUT" ? "[" + e.type + "]" : "") + " «" + (e.textContent || e.getAttribute("aria-label") || e.value || "").trim().replace(/\s+/g, " ").slice(0, 24) + "»"; }
  function parseCsvLine(line) {
    var out = [], cur = "", q = false, k, ch;
    for (k = 0; k < line.length; k++) {
      ch = line[k];
      if (q) { if (ch === '"') { if (line[k + 1] === '"') { cur += '"'; k++; } else q = false; } else cur += ch; }
      else if (ch === '"') q = true; else if (ch === ",") { out.push(cur); cur = ""; } else cur += ch;
    }
    out.push(cur); return out;
  }
  function rgbDiff(a, b, tol) { return a.some(function (v, q) { return Math.abs(v - b[q]) > tol; }); }
  /* точки в пределах МО: заливка (самый частый цвет) и штрих (самый тёмный); для сплошной заливки обе точки совпадают */
  function pickPoints(T, id, hatched) {
    var a = T.anchor(id), pts = [], dx, dy, h, c, cnt = {}, best = null, dark = null, key, sum;
    for (dx = -14; dx <= 14; dx++) for (dy = -14; dy <= 14; dy++) {
      h = T.hitTest(a.x + dx, a.y + dy);
      if (!h || h.id !== id) continue;
      c = T.pixelAt(a.x + dx, a.y + dy); key = c.join(",");
      cnt[key] = cnt[key] || { n: 0, c: c, x: a.x + dx, y: a.y + dy }; cnt[key].n++;
      pts.push({ x: a.x + dx, y: a.y + dy, c: c });
    }
    for (key in cnt) if (!best || cnt[key].n > best.n) best = cnt[key];
    if (!hatched) return [{ kind: "заливка", x: a.x, y: a.y }];
    pts.forEach(function (p) { sum = p.c[0] + p.c[1] + p.c[2]; if (!dark || sum < dark.sum) dark = { sum: sum, x: p.x, y: p.y, c: p.c }; });
    return [{ kind: "заливка", x: best.x, y: best.y, c: best.c }, { kind: "штрих", x: dark.x, y: dark.y, c: dark.c }];
  }
  async function filterScenario(T, name, id, clickText, tol, hatched) {
    T.reset(); T.setLayer("type"); T.setMonth("2024-12"); T.zoomTo(id); await settle(150);
    var pts = pickPoints(T, id, hatched);
    var find = function () { return Array.prototype.slice.call(document.querySelectorAll(".leg-item")).filter(function (b) { return b.textContent.trim().indexOf(clickText) === 0; })[0]; };
    var btn = find();
    if (!btn) return { name: name, error: "нет элемента легенды «" + clickText + "»", points: [] };
    var before = pts.map(function (p) { return T.pixelAt(p.x, p.y); });
    btn.click(); await settle(150);
    var during = pts.map(function (p) { return T.pixelAt(p.x, p.y); }), hid1 = T.state().hidden;
    find().click(); await settle(150);
    var after = pts.map(function (p) { return T.pixelAt(p.x, p.y); }), hid2 = T.state().hidden;
    return {
      name: name, id: id, hidden_after_click1: hid1, hidden_after_click2: hid2,
      points: pts.map(function (p, q) {
        return { kind: p.kind, x: p.x, y: p.y, before: before[q], during: during[q], after: after[q], changed: rgbDiff(during[q], before[q], tol), returned: !rgbDiff(after[q], before[q], tol) };
      }),
      distinct: pts.length < 2 || rgbDiff(before[0], before[1], 30)
    };
  }
  function captionChecks(T, cfg) {
    var out = { cls4: [], cls2: {}, cls3: {} }, id, m, t, lm, row, cap, bad = [];
    ["2023-06", "2024-12"].forEach(function (mode) {
      T.setMonth(mode);
      Object.keys(cfg.cls4Last).forEach(function (sid) {
        id = +sid; lm = cfg.cls4Last[sid]; t = T.tooltipText(id); T.select(id);
        var pan = T.panelText();
        row = t.indexOf("Последний доступный месяц: " + lm + ";") >= 0;
        cap = t.indexOf("тип по последнему доступному месяцу: " + lm) >= 0;
        var capAny = /тип по последнему доступному месяцу: (\d{4}-\d{2})/g, found = [], mm;
        while ((mm = capAny.exec(t))) found.push(mm[1]);
        var monthOk = mode === "2023-06" ? (cap && found.indexOf(lm) >= 0 && found.every(function (f) { return f === lm; })) : true;
        var noDec = t.indexOf("тип только за декабрь 2024") < 0 && pan.indexOf("тип только за декабрь 2024") < 0;
        var panOk = pan.indexOf("Последний доступный месяц:" + lm) >= 0 && (mode === "2024-12" || pan.indexOf("тип по последнему доступному месяцу: " + lm) >= 0);
        var ok = row && monthOk && noDec && panOk;
        if (!ok) bad.push({ id: id, mode: mode, lm: lm });
        if (id === cfg.ids.cls4) out.cls4.push({ id: id, mode: mode, last_month: lm, row_ok: row, caption_ok: monthOk, no_dec_only: noDec, panel_ok: panOk });
      });
      id = cfg.ids.cls2; t = T.tooltipText(id); T.select(id);
      out.cls2[mode] = { has_dec_only: t.indexOf("тип только за декабрь 2024") >= 0, panel_has_dec_only: T.panelText().indexOf("тип только за декабрь 2024") >= 0 };
      id = cfg.ids.sut409; t = T.tooltipText(id); T.select(id);
      out.cls3[mode] = { key: T.keyOf("type", id), not_assigned: t.indexOf("тип не присвоен") >= 0, has_dec_only: t.indexOf("только за декабрь") >= 0, panel_has_dec_only: T.panelText().indexOf("только за декабрь") >= 0 };
    });
    out.cls4_checked = Object.keys(cfg.cls4Last).length;
    out.cls4_bad = bad;
    T.setMonth("2024-12");
    return out;
  }
  function placeholderMenu() {
    var inp = document.getElementById("map-search"), cs = getComputedStyle(inp), c = document.createElement("canvas").getContext("2d"), ph = inp.getAttribute("placeholder");
    c.font = cs.fontStyle + " " + cs.fontWeight + " " + cs.fontSize + " " + cs.fontFamily;
    var inner = inp.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    var ul = document.querySelector("nav.top ul"), links = ul.querySelectorAll("a"), last = links[links.length - 1];
    var before = ul.scrollLeft; ul.scrollLeft = ul.scrollWidth; var lr = last.getBoundingClientRect(), ur = ul.getBoundingClientRect();
    var reach = lr.right <= ur.right + 0.5 && lr.left >= ur.left - 0.5; ul.scrollLeft = before;
    return {
      placeholder: ph, scrollWidth: inp.scrollWidth, clientWidth: inp.clientWidth, textWidth: Math.round(c.measureText(ph).width * 10) / 10, innerWidth: Math.round(inner * 10) / 10,
      placeholder_fits: c.measureText(ph).width <= inner && inp.scrollWidth <= inp.clientWidth,
      menu: { scrollWidth: ul.scrollWidth, clientWidth: ul.clientWidth, scrollable: ul.scrollWidth > ul.clientWidth, last: last.textContent.trim(), last_reachable: reach, page_scrolls_x: document.documentElement.scrollWidth > window.innerWidth }
    };
  }
  /* клик по ссылке меню (обычное поведение страницы, в том числе плавная прокрутка); ждём, пока положение страницы перестанет меняться */
  async function clickNav(href) {
    var nav = document.querySelector("nav.top"), link = nav.querySelector('a[href="' + href + '"]'), last = -1, still = 0, t0 = Date.now();
    link.click();
    while ((still < 6 || Date.now() - t0 < 500) && Date.now() - t0 < 8000) { await raf2(); await sleep(50); if (Math.abs(window.scrollY - last) < 0.5) still++; else { still = 0; last = window.scrollY; } }
    var sec = document.querySelector(href), h = sec.querySelector("h2"), nb = nav.getBoundingClientRect().bottom;
    return {
      href: href, text: link.textContent.trim(), scrollY: Math.round(window.scrollY), navBottom: Math.round(nb * 10) / 10, secTop: Math.round(sec.getBoundingClientRect().top * 10) / 10,
      secOffset: Math.round((sec.getBoundingClientRect().top - nb) * 10) / 10, headingOffset: Math.round((h.getBoundingClientRect().top - nb) * 10) / 10, heading: h.textContent.trim().slice(0, 40)
    };
  }
  window.__clickNav = clickNav;
  async function navCheck() {
    var out = [], links = Array.prototype.slice.call(document.querySelectorAll("nav.top a")), k;
    for (k = 0; k < links.length; k++) { window.scrollTo({ top: 0, behavior: "instant" }); await settle(100); out.push(await clickNav(links[k].getAttribute("href"))); }
    window.scrollTo({ top: 0, behavior: "instant" }); await settle(100);
    return out;
  }
  window.__landingCheck = async function (cfg) {
    var T = window.__test, R = {}, k, id;
    R.ready = !!(T && T.ready);
    if (!R.ready) return R;
    R.painted = T.paintedPixels();
    R.hit = [];
    for (k in cfg.ids) {
      id = cfg.ids[k]; T.reset(); T.zoomTo(id); await settle(50);
      var a = T.anchor(id), h = T.hitTest(a.x, a.y), s = T.select(id);
      R.hit.push({ name: k, id: id, hit: h ? h.id : null, selected: s ? s.id : null, ok: !!h && h.id === id && !!s && s.id === id });
    }
    T.reset();
    R.months = [];
    ["2023-01", "2023-12", "2024-12"].forEach(function (m) {
      T.setMonth(m);
      for (var n in cfg.ids) { var l = T.getLabel(cfg.ids[n], m); R.months.push({ month: m, name: n, id: cfg.ids[n], label: l.label, canonical: l.canonical }); }
    });
    R.captions = captionChecks(T, cfg);
    T.setMonth("2024-12");
    R.layers = {};
    ["market", "stab", "type"].forEach(function (l) { T.setLayer(l); R.layers[l] = { layer: T.state().layer, legend: T.legendText().slice(0, 60), painted: T.paintedPixels().painted }; });
    R.search = {};
    ["Якутск", "Новый Уренгой", "Сут-Хольский"].forEach(function (q) { var r = T.search(q); R.search[q] = { n: r.n, ids: r.ids, highlighted: T.state().hl }; });
    T.search(""); T.reset();
    var tol = cfg.filterTol;
    R.filter = [];
    R.filter.push(await filterScenario(T, "тип 4 (Якутск, МО 309)", cfg.ids.yakutsk, "Тип 4 ·", tol, false));
    R.filter.push(await filterScenario(T, "тип 0 (МО выборки " + cfg.typ0Id + ")", cfg.typ0Id, "Тип 0 ·", tol, false));
    R.filter.push(await filterScenario(T, "класс 2, штриховка (МО " + cfg.ids.cls2 + ")", cfg.ids.cls2, "тип присвоен вне выборки", tol, true));
    T.reset();
    var sel = "button, a, input, select, textarea, summary, [role=button], .leg-item";
    R.interactive = Array.prototype.slice.call(document.querySelectorAll(sel)).filter(visible).map(function (e) {
      var r = e.getBoundingClientRect(); return { el: label(e), w: Math.round(r.width * 10) / 10, h: Math.round(r.height * 10) / 10 };
    });
    R.small = R.interactive.filter(function (r) { return r.w < cfg.minPx - 0.01 || r.h < cfg.minPx - 0.01; });
    R.scroll = { docW: document.documentElement.scrollWidth, bodyW: document.body.scrollWidth, winW: window.innerWidth, ok: document.documentElement.scrollWidth <= window.innerWidth };
    R.nav = await navCheck();
    R.texts = { keys: Array.from(new Set(Array.prototype.map.call(document.querySelectorAll("[data-text-key]"), function (e) { return e.getAttribute("data-text-key"); }))).sort(),
      emptyKeyed: Array.prototype.filter.call(document.querySelectorAll("[data-text-key]"), function (e) { return !e.textContent.trim(); }).length,
      phElements: document.querySelectorAll(".ph").length, markerVisible: document.body.innerText.split(cfg.textMarker).length - 1,
      markerKeyed: Array.prototype.filter.call(document.querySelectorAll("[data-text-key]"), function (e) { return e.textContent.indexOf(cfg.textMarker) >= 0; }).length };
    R.ui = placeholderMenu();
    R.counts = { cards: document.querySelectorAll("#types .card").length, coverageRows: document.querySelectorAll(".covwrap tbody tr").length, svg: document.querySelectorAll("figure svg").length, legendItems: document.querySelectorAll(".leg-item").length };
    var csv = T.buildCsv(), lines = csv.replace(/^﻿/, "").split("\n"); if (lines[lines.length - 1] === "") lines.pop();
    var com = lines.filter(function (l) { return l.charAt(0) === "#"; }), data = lines.filter(function (l) { return l.charAt(0) !== "#"; });
    var header = parseCsvLine(data[0]), rows = data.slice(1).map(parseCsvLine);
    R.csv = { bom: csv.charCodeAt(0) === 0xFEFF, comments: com, header: header, dataRows: rows.length, cols: rows.map(function (r) { return r.length; }).filter(function (n, q, arr) { return arr.indexOf(n) === q; }), headerCols: header.length, rows: rows };
    var cap = {}, origCreate = URL.createObjectURL, origClick = HTMLAnchorElement.prototype.click;
    URL.createObjectURL = function (b) { cap.size = b.size; cap.type = b.type; return origCreate.call(URL, b); };
    HTMLAnchorElement.prototype.click = function () { cap.download = this.download; cap.href = this.href.slice(0, 5); };
    document.getElementById("csv-btn").click(); await sleep(50);
    URL.createObjectURL = origCreate; HTMLAnchorElement.prototype.click = origClick;
    R.csvButton = cap;
    return R;
  };
})();
