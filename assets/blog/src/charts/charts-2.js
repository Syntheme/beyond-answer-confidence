    (() => {
  const DATA = JSON.parse(document.getElementById("blog-data-2").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
  const txt = (parent, s, attrs) => { const t = el("text", attrs, parent); t.appendChild(document.createTextNode(s)); return t; };
  const h = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const pct = v => (v * 100).toFixed(v < 0.095 ? 1 : 0) + "%";

  function frame(c, title, sub) { c.appendChild(h("p", "chart-title", title)); if (sub) c.appendChild(h("p", "chart-sub", sub)); const tip = h("div", "tip"); c.appendChild(tip); return tip; }
  function legend(c, items) {
    const lg = h("div", "legend");
    for (const it of items) { const s = h("span"); const i = h(it.square ? "b" : "i", it.square ? "sq" : null); i.style.background = it.color; s.append(i, document.createTextNode(it.name)); lg.appendChild(s); }
    c.appendChild(lg);
  }
  function showTip(tip, c, x, y, header, rows) {
    tip.replaceChildren(); const hd = h("div", null, header); hd.style.color = "var(--ink-2)"; tip.appendChild(hd);
    for (const r of rows) { const row = h("div", "row"); const i = h("i"); i.style.background = r.color || "transparent"; row.append(i); if (r.value) row.append(h("strong", null, r.value)); row.append(document.createTextNode(" " + r.name)); tip.appendChild(row); }
    const cb = c.getBoundingClientRect(); tip.classList.add("on");
    const sx = c.scrollLeft, sy = c.scrollTop, w = tip.offsetWidth;
    let left = x - cb.left + sx + 14; if (left + w > sx + cb.width - 4) left = x - cb.left + sx - w - 14;
    tip.style.left = Math.max(sx + 4, left) + "px"; tip.style.top = Math.max(4, y - cb.top + sy - 10) + "px";
  }
  const hide = tip => tip.classList.remove("on");
  function dataTable(c, caption, head, rows) {
    const d = h("details", "data"); d.appendChild(h("summary", null, "Show data table"));
    const t = h("table"); const cap = h("caption", null, caption); cap.style.textAlign = "left"; cap.style.color = "var(--ink-2)"; t.appendChild(cap);
    const thead = h("thead"), tbody = h("tbody"), tr = h("tr");
    head.forEach((x, i) => { const th = h("th", i ? "num" : null, x); th.scope = "col"; tr.appendChild(th); }); thead.appendChild(tr); t.appendChild(thead);
    for (const r of rows) { const row = h("tr"); r.forEach((x, i) => { const cell = h(i ? "td" : "th", i ? "num" : null, x); if (!i) { cell.scope = "row"; cell.style.fontWeight = "400"; cell.style.color = "var(--ink)"; } row.appendChild(cell); }); tbody.appendChild(row); }
    t.appendChild(tbody);
    d.appendChild(t); c.after(d); return d;
  }
  function segmented(c, labels, onPick, initial = 0, groupLabel = "Choose what the chart shows") {
    const seg = h("div", "seg"); seg.setAttribute("role", "group"); seg.setAttribute("aria-label", groupLabel);
    const buttons = labels.map((lab, i) => { const b = h("button", null, lab); b.type = "button"; b.setAttribute("aria-pressed", String(i === initial)); b.addEventListener("click", () => { buttons.forEach((x, j) => x.setAttribute("aria-pressed", String(j === i))); onPick(i); }); seg.appendChild(b); return b; });
    c.appendChild(seg); return seg;
  }
  // Hover/focus target for a mark: bigger than the mark itself.
  function hitTarget(svg, attrs, tip, c, header, rows, anchor) {
    const name = header + ": " + rows.map(q => [q.value, q.name].filter(Boolean).join(" ")).join("; ");
    const r = el("rect", { fill: "transparent", tabindex: 0, role: "img", "aria-label": name, "data-mark": "", ...attrs }, svg);
    const on = (x, y) => showTip(tip, c, x, y, header, rows);
    r.addEventListener("pointermove", ev => on(ev.clientX, ev.clientY));
    r.addEventListener("pointerdown", ev => on(ev.clientX, ev.clientY));
    r.addEventListener("focus", () => { const b = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal; on(b.left + anchor[0] * b.width / vb.width, b.top + anchor[1] * b.height / vb.height); });
    r.addEventListener("pointerleave", ev => { if (ev.pointerType !== "touch") hide(tip); });
    r.addEventListener("blur", () => hide(tip));
    return r;
  }



  const S1 = "var(--s1)", S2 = "var(--s2)", S3 = "var(--s3)";
  // Bars are drawn to scale: nothing for zero, a plain thin rectangle below 6 units.
  const bar = (svg, x0, y0, w, hgt, fill) => !(w > 0) ? null : w < 6 ? el("rect", { x: x0, y: y0, width: w, height: hgt, fill }, svg)
    : el("path", { d: `M${x0},${y0} h${w - 4} a4,4 0 0 1 4,4 v${Math.max(0, hgt - 8)} a4,4 0 0 1 -4,4 h${-(w - 4)} z`, fill }, svg);

  // ---- generic line chart over categorical x ----
  function lines(cId, cfg) {
    const c = document.getElementById(cId); if (!c) return;
    const tip = frame(c, cfg.title, cfg.sub);
    legend(c, cfg.series.map(s => ({ name: s.name, color: s.color })));
    const W = 720, H = cfg.height || 320, m = { l: 48, r: 96, t: 16, b: 50 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": cfg.title }, c);
    const n = cfg.X.length, lo = cfg.yMin ?? 0, hi = cfg.yMax ?? 1;
    const x = i => m.l + (n === 1 ? 0.5 : i / (n - 1)) * (W - m.l - m.r), y = v => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    for (let g = lo; g <= hi + 1e-9; g += cfg.yStep || 0.2) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(g), y2: y(g), stroke: Math.abs(g - lo) < 1e-9 ? "var(--axis)" : "var(--grid)" }, svg);
      txt(svg, Math.round(g * 100) + "%", { x: m.l - 6, y: y(g) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11 });
    }
    const every = cfg.labelEvery || 1;
    cfg.X.forEach((lab, i) => { if (i % every === 0 || (i === n - 1 && (n - 1) % every > every / 2)) txt(svg, lab, { x: x(i), y: H - m.b + 16, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10.5 }); });
    if (cfg.xLabel) txt(svg, cfg.xLabel, { x: (m.l + W - m.r) / 2, y: H - 8, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 });
    if (cfg.vline != null) {
      // cfg.vlineFrac: how far into the marked category the line falls (0 = its start)
      const vx = x(cfg.vline) + (x(1) - x(0)) * ((cfg.vlineFrac ?? 0) - 0.5);
      el("line", { x1: vx, x2: vx, y1: m.t, y2: H - m.b, stroke: "var(--muted)", "stroke-dasharray": "4 4" }, svg);
      txt(svg, cfg.vlineLabel, { x: vx - 6, y: m.t + 12, "text-anchor": "end", fill: "var(--ink-2)", "font-size": 11 });
    }
    for (const s of cfg.series) {
      const pts = cfg.rows.map((r, i) => [i, r[s.key]]).filter(p => p[1] != null);
      el("polyline", { points: pts.map(([i, v]) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": s.dash || null }, svg);
      if (cfg.markers !== false) for (const [i, v] of pts) {
        if (s.square) el("rect", { x: x(i) - 4, y: y(v) - 4, width: 8, height: 8, rx: 1.5, fill: s.color, stroke: "var(--surface)", "stroke-width": 1.5 }, svg);
        else el("circle", { cx: x(i), cy: y(v), r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 1.5 }, svg);
      }
    }
    // direct labels at the right end, de-collided
    const ends = cfg.series.map(s => { const last = [...cfg.rows].reverse().find(r => r[s.key] != null); return { s, yy: y(last[s.key]) }; }).sort((a, b) => a.yy - b.yy);
    for (let k = 1; k < ends.length; k++) if (ends[k].yy - ends[k - 1].yy < 13) ends[k].yy = ends[k - 1].yy + 13;
    for (const e of ends) txt(svg, e.s.short, { x: W - m.r + 8, y: e.yy + 4, fill: "var(--ink-2)", "font-size": 11 });
    const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--muted)", opacity: 0 }, svg);
    const hit = el("rect", { x: m.l - 16, y: m.t - 8, width: W - m.l - m.r + 32, height: H - m.t - m.b + 16, fill: "transparent", tabindex: 0, role: "img", "data-mark": "", "aria-label": `${cfg.title}; use the arrow keys to step through the points` }, svg);
    const live = h("div", "sr-live"); live.setAttribute("aria-live", "polite"); c.appendChild(live);
    let fi = 0;
    const show = (i, cx, cy) => {
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 0.8);
      const head = cfg.tipHeader(cfg.rows[i], i), vals = cfg.series.filter(s => cfg.rows[i][s.key] != null).map(s => ({ color: s.color, value: pct(cfg.rows[i][s.key]), name: s.short.toLowerCase() }));
      live.textContent = `${head}: ${vals.map(v => `${v.name} ${v.value}`).join(", ")}`;
      showTip(tip, c, cx, cy, head, vals);
    };
    const at = i => { const b = svg.getBoundingClientRect(); show(i, b.left + x(i) * b.width / W, b.top + m.t * b.height / H + 20); };
    const near = ev => { const b = svg.getBoundingClientRect(), sx = (ev.clientX - b.left) * W / b.width; let i = 0, best = 1e9; cfg.X.forEach((_, j) => { const d = Math.abs(x(j) - sx); if (d < best) { best = d; i = j; } }); fi = i; show(i, ev.clientX, ev.clientY); };
    hit.addEventListener("pointermove", near); hit.addEventListener("pointerdown", near);
    hit.addEventListener("keydown", ev => { if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") { fi = Math.max(0, Math.min(n - 1, fi + (ev.key === "ArrowRight" ? 1 : -1))); ev.preventDefault(); at(fi); } });
    hit.addEventListener("focus", () => at(fi));
    const off = () => { cross.setAttribute("opacity", 0); hide(tip); };
    hit.addEventListener("pointerleave", ev => { if (ev.pointerType !== "touch") off(); });
    hit.addEventListener("blur", off);
    dataTable(c, cfg.tableCaption, [cfg.xName, ...cfg.series.map(s => s.short)], cfg.rows.map((r, i) => [cfg.X[i], ...cfg.series.map(s => r[s.key] == null ? "—" : pct(r[s.key]))]));
  }

  // ---- dumbbells: two values per row, grouped ----
  function dumbbell(cId, cfg) {
    const c = document.getElementById(cId); if (!c) return;
    const tip = frame(c, cfg.title, cfg.sub);
    const lg = h("div", "legend");
    for (const [name, color, round] of [[cfg.aName, cfg.aColor || S1, true], [cfg.bName, cfg.bColor || S2, false]]) {
      const sp = h("span"), mk = h("b", "sq"); mk.style.background = color; if (round) mk.style.borderRadius = "50%";
      sp.append(mk, document.createTextNode(name)); lg.appendChild(sp);
    }
    c.appendChild(lg);
    const rowH = 30, gapH = 26, m = { l: cfg.left || 300, r: 30, t: 6, b: 30 };
    const nRows = cfg.groups.reduce((a, g) => a + g.rows.length, 0), W = 720;
    const H = m.t + nRows * rowH + cfg.groups.length * gapH + m.b;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": cfg.title }, c);
    const lo = cfg.xMin, hi = cfg.xMax, x = v => m.l + (v - lo) / (hi - lo) * (W - m.l - m.r);
    for (let g = lo; g <= hi + 1e-9; g += cfg.xStep || 0.1) { el("line", { x1: x(g), x2: x(g), y1: m.t, y2: H - m.b, stroke: "var(--grid)" }, svg); txt(svg, cfg.fmtAxis ? cfg.fmtAxis(g) : Math.round(g * 100) + "%", { x: x(g), y: H - 10, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 }); }
    let yy = m.t;
    const fmt = cfg.fmt || pct;
    for (const grp of cfg.groups) {
      txt(svg, grp.name, { x: 0, y: yy + 17, fill: "var(--ink-2)", "font-size": 11.5, "font-weight": 700 });
      yy += gapH;
      for (const r of grp.rows) {
        const cy = yy + rowH / 2;
        txt(svg, r.name, { x: m.l - 12, y: cy + 4, "text-anchor": "end", fill: "var(--ink)", "font-size": 12 });
        el("line", { x1: x(r.a), x2: x(r.b), y1: cy, y2: cy, stroke: "var(--axis)", "stroke-width": 2 }, svg);
        // Square first, circle on top (slightly smaller) so both stay visible when they coincide.
        el("rect", { x: x(r.b) - 5.5, y: cy - 5.5, width: 11, height: 11, rx: 1.5, fill: cfg.bColor || S2, stroke: "var(--surface)", "stroke-width": 2 }, svg);
        el("circle", { cx: x(r.a), cy, r: 4.2, fill: cfg.aColor || S1, stroke: "var(--surface)", "stroke-width": 1.5 }, svg);
        hitTarget(svg, { x: 0, y: yy, width: W, height: rowH }, tip, c, r.name + (r.detail ? ` · ${r.detail}` : ""),
          [{ color: cfg.aColor || S1, value: fmt(r.a), name: cfg.aShort }, { color: cfg.bColor || S2, value: fmt(r.b), name: cfg.bShort }, ...(r.note ? [{ name: r.note }] : [])], [x(Math.max(r.a, r.b)), cy]);
        yy += rowH;
      }
    }
    dataTable(c, cfg.tableCaption, ["", cfg.aShort, cfg.bShort], cfg.groups.flatMap(g => g.rows.map(r => [`${g.name} · ${r.name}`, fmt(r.a), fmt(r.b)])));
  }

  // ===== Finding 01: how sure vs how right =====
  (function sure() {
    const R = DATA.sure_vs_right;
    const toRow = r => ({ name: r.name, detail: r.detail, a: r.acc, b: r.conf, note: `${r.n.toLocaleString()} questions` });
    dumbbell("chartSure", {
      title: "How right vs how sure, on standard benchmarks and on new, later, made-up or misleading input",
      sub: "Accuracy (circle) and average confidence (square) on the same questions. A square to the right of its circle means overconfident.",
      aName: "Accuracy (how often right)", bName: "Average confidence (how sure)", aShort: "accuracy", bShort: "confidence",
      xMin: 0.2, xMax: 1.0, left: 320,
      groups: [{ name: "STANDARD BENCHMARKS, INFORMATION AVAILABLE", rows: R.filter(r => r.group === "familiar").map(toRow) },
               { name: "NEW, LATER, MADE-UP OR MISLEADING INPUT", rows: R.filter(r => r.group === "edge").map(toRow) }],
      tableCaption: "Accuracy and average confidence per setting",
    });
  })();

  // ===== Finding 02a: quiz clues =====
  (function quiz() {
    const Q = DATA.quizbowl;
    lines("chartQuiz", {
      title: "Quiz clues revealed one sentence at a time",
      sub: "1,055 Quizbowl questions, four options each · confidence tracks accuracy as evidence accumulates · later points cover only questions that have that many sentences (171 at eight)",
      X: Q.map(r => String(r.k)), rows: Q, xName: "Sentences shown", xLabel: "sentences of the question shown",
      series: [{ key: "acc", name: "Accuracy (how often right)", short: "Accuracy", color: S1 },
               { key: "conf", name: "Average confidence (how sure)", short: "Confidence", color: S2, square: true },
               { key: "enough", name: "\"Do the clues identify the answer for certain?\" (yes)", short: "Enough info", color: S3, dash: "5 3" }],
      yMin: 0.4, yMax: 1.0, yStep: 0.1, tipHeader: r => `${r.k} sentence${r.k === 1 ? "" : "s"} shown · ${r.n} questions`,
      tableCaption: "Quizbowl by number of sentences shown",
    });
  })();

  // ===== Finding 02b: documents, with and without the instruction =====
  (function hotpot() {
    const Hd = DATA.hotpot;
    dumbbell("chartHotpot", {
      title: "Misleading paragraphs cost accuracy, and one sentence won it back on this benchmark",
      sub: "951 HotpotQA comparison questions · accuracy (circle) vs confidence (square)",
      aName: "Accuracy", bName: "Average confidence", aShort: "accuracy", bShort: "confidence",
      xMin: 0.7, xMax: 1.0, xStep: 0.05, left: 250,
      groups: [
        { name: "AS ASKED", rows: Hd.map(r => ({ name: r.cell, a: r.acc, b: r.conf })) },
        { name: "TOLD \"THE PARAGRAPHS MAY BE IRRELEVANT\"", rows: Hd.filter(r => r.acc_inst != null).map(r => ({ name: r.cell, a: r.acc_inst, b: r.conf_inst })) },
      ],
      tableCaption: "HotpotQA accuracy and confidence, with and without the instruction",
    });
  })();

  // ===== Finding 03a: popularity and made-up subjects =====
  (function pop() {
    const P = DATA.popqa, fmtK = v => v >= 1e6 ? (v / 1e6).toFixed(0) + "M" : v >= 1e3 ? (v / 1e3).toFixed(v >= 1e4 ? 0 : 1) + "k" : String(v);
    const rows = [{ label: "made-up", acc: null, conf: P.made_up.conf, known: P.made_up.known, n: P.made_up.n },
                  ...P.quintiles.map(q => ({ label: `${fmtK(q.range[0])}–${fmtK(q.range[1])}`, ...q }))];
    lines("chartPop", {
      title: "From made-up subjects to the most famous",
      sub: "PopQA, 14,267 real questions in fifths by monthly Wikipedia views, plus 1,600 made-up twins (no option is right; 25% on each would be an equal split)",
      X: rows.map(r => r.label), rows, xName: "Subject", xLabel: "made-up subjects, then real ones from least to most viewed per month",
      series: [{ key: "acc", name: "Accuracy (how often right)", short: "Accuracy", color: S1 },
               { key: "conf", name: "Average confidence (how sure)", short: "Confidence", color: S2, square: true },
               { key: "known", name: "\"Do you know the answer for certain?\" (yes)", short: "Says it knows", color: S3, dash: "5 3" }],
      yMin: 0, yMax: 1, yStep: 0.2, tipHeader: (r, i) => i === 0 ? `Made-up subjects · ${r.n.toLocaleString()} questions` : `Real subjects, ${r.label} views/month · ${r.n.toLocaleString()} questions`,
      tableCaption: "PopQA by popularity, and made-up subjects",
    });
  })();

  // ===== Finding 03b: news by quarter =====
  (function oracle() {
    const O = DATA.oracle, qs = O.quarters;
    const [cy, cm] = O.cutoff.split("-").map(Number);
    const cutQ = `${cy} Q${Math.floor((cm - 1) / 3) + 1}`, cutFrac = ((cm - 1) % 3) / 3;
    const lastQ = qs[qs.length - 1];
    lines("chartOracle", {
      title: "Real news questions, January 2020 to July 2026",
      sub: `Daily Oracle yes/no questions, 80 per month (the last quarter, ${lastQ.q}, has July only) · each question names its own date, but today's date was not supplied · beyond the observed knowledge boundary, accuracy falls but confidence rises (the boundary is estimated from the data, not a confirmed training cutoff)`,
      X: qs.map(q => q.q.replace(" ", " ")), rows: qs, xName: "Quarter", labelEvery: 4, height: 340,
      vline: qs.findIndex(q => q.q === cutQ), vlineFrac: cutFrac, vlineLabel: `observed knowledge boundary, start of ${O.cutoff}`,
      series: [{ key: "acc", name: "Accuracy (how often right)", short: "Accuracy", color: S1 },
               { key: "conf", name: "Average confidence (how sure)", short: "Confidence", color: S2, square: true },
               { key: "known", name: "\"Do you know for certain how this turned out?\" (yes)", short: "Says it knows", color: S3, dash: "5 3" }],
      markers: false, yMin: 0, yMax: 1, yStep: 0.2, tipHeader: r => `${r.q} · ${r.n} questions · answered "no" ${pct(r.no)}`,
      tableCaption: "Daily Oracle yes/no questions by quarter",
    });
  })();

  // ===== Finding 04: the "not known" option =====
  (function unknown() {
    const c = document.getElementById("chartUnknown"); if (!c) return;
    const tip = frame(c, "Forced to choose vs allowed to say \"not known\", beyond the observed knowledge boundary", "1,680 yes/no news questions from beyond the boundary · bars are shares of all 1,680 questions, drawn to scale (values under 1% show as a thin line); the grey line under each label says how many were answered and how many of those were right");
    const S = [{ key: "confident_wrong", name: "Confidently wrong (≥ 90% on the wrong answer)", short: "confidently wrong", color: S2 },
               { key: "unknown", name: "Chose \"not known to me\"", short: "chose \"not known\"", color: S1 }];
    legend(c, S.map(s => ({ name: s.name, color: s.color, square: true })));
    const rows = DATA.unknown_option.filter(r => r.period === "post");
    const W = 720, rowH = 54, m = { l: 230, r: 60, t: 6, b: 26 }, H = m.t + rows.length * rowH + m.b;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Confident errors and abstentions beyond the observed knowledge boundary" }, c);
    const x = v => m.l + v * (W - m.l - m.r);
    for (const g of [0, 0.25, 0.5, 0.75, 1]) { el("line", { x1: x(g), x2: x(g), y1: m.t, y2: H - m.b, stroke: g === 0 ? "var(--axis)" : "var(--grid)" }, svg); txt(svg, (g * 100) + "%", { x: x(g), y: H - 8, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 }); }
    rows.forEach((r, i) => {
      const y0 = m.t + i * rowH + 8;
      txt(svg, r.cond, { x: m.l - 12, y: y0 + 14, "text-anchor": "end", fill: "var(--ink)", "font-size": 12.5 });
      txt(svg, `answered ${pct(1 - r.unknown)}, ${pct(r.acc_answered)} of those right`, { x: m.l - 12, y: y0 + 31, "text-anchor": "end", fill: "var(--ink-2)", "font-size": 11 });
      S.forEach((s, k) => {
        const yy = y0 + k * 19, w = r[s.key] > 0 ? Math.max(1.5, x(r[s.key]) - x(0)) : 0;
        bar(svg, x(0), yy, w, 16, s.color);
        txt(svg, pct(r[s.key]), { x: x(0) + w + 6, y: yy + 12, fill: "var(--ink-2)", "font-size": 11.5 });
      });
      hitTarget(svg, { x: 0, y: y0 - 4, width: W, height: rowH }, tip, c, r.cond + " · beyond the boundary",
        [...S.map(s => ({ color: s.color, value: pct(r[s.key]), name: s.short })), { value: pct(r.no), name: "answered \"no\"" }, { name: `accuracy on answered questions ${pct(r.acc_answered)}` }], [x(Math.max(r.unknown, r.confident_wrong)), y0]);
    });
    dataTable(c, "Behaviour by condition, before and beyond the boundary (shares of all questions, except the last column)", ["Condition · period", "Answered \"no\"", "Chose \"not known\"", "Answered (coverage)", "Confidently wrong", "Accuracy (answered)"],
      DATA.unknown_option.map(r => [`${r.cond} · ${r.period === "post" ? "after" : "before"}`, pct(r.no), pct(r.unknown), pct(1 - r.unknown), pct(r.confident_wrong), r.acc_answered == null ? "—" : pct(r.acc_answered)]));
  })();

  // ===== Finding 05a: stated odds by question format =====
  (function odds() {
    const c = document.getElementById("chartOdds"); if (!c) return;
    const tip = frame(c, "Stated chance vs what Jev reports, by how you ask", "480 generated future draws with the chances stated · untied chances only (tied chances are covered by the dice chart)");
    const F = [{ key: "noul", name: "One yes/no question per outcome", short: "yes/no per outcome", color: S1 },
               { key: "score", name: "A 0–100% score per outcome", short: "score per outcome", color: S3, square: true },
               { key: "instructed", name: "Multiple choice, told to give each option's probability", short: "multiple choice", color: S2 }];
    legend(c, [...F.map(f => ({ name: f.name, color: f.color })), { name: "Perfect (reported equals stated)", color: "var(--muted)" }]);
    const xs = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.7"];
    const W = 720, H = 360, m = { l: 52, r: 30, t: 14, b: 46 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Reported vs stated chance" }, c);
    const x = v => m.l + v / 0.75 * (W - m.l - m.r), y = v => m.t + (1 - v) * (H - m.t - m.b);
    for (const g of [0, 0.2, 0.4, 0.6, 0.8, 1]) { el("line", { x1: m.l, x2: W - m.r, y1: y(g), y2: y(g), stroke: g === 0 ? "var(--axis)" : "var(--grid)" }, svg); txt(svg, (g * 100) + "%", { x: m.l - 6, y: y(g) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11 }); }
    for (const v of xs) txt(svg, Math.round(+v * 100) + "%", { x: x(+v), y: H - m.b + 16, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 });
    txt(svg, "stated chance of the outcome", { x: (m.l + W - m.r) / 2, y: H - 8, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 });
    txt(svg, "probability Jev reports", { x: 12, y: (m.t + H - m.b) / 2, transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 });
    el("line", { x1: x(0), y1: y(0), x2: x(0.75), y2: y(0.75), stroke: "var(--muted)", "stroke-dasharray": "4 4" }, svg);
    for (const f of F) {
      const pts = xs.map(v => [+v, DATA.odds[f.key][v]]);
      el("polyline", { points: pts.map(([a, b]) => `${x(a)},${y(b)}`).join(" "), fill: "none", stroke: f.color, "stroke-width": 2, "stroke-linejoin": "round" }, svg);
      for (const [a, b] of pts) {
        if (f.square) el("rect", { x: x(a) - 4.5, y: y(b) - 4.5, width: 9, height: 9, rx: 1.5, fill: f.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
        else el("circle", { cx: x(a), cy: y(b), r: 4.5, fill: f.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
        hitTarget(svg, { x: x(a) - 9, y: y(b) - 9, width: 18, height: 18 }, tip, c, `Stated chance ${pct(a)}`, [{ color: f.color, value: pct(b), name: f.short }], [x(a), y(b)]);
      }
    }
    txt(svg, "multiple choice", { x: x(0.4) + 10, y: y(0.986) + 14, fill: "var(--ink)", "font-size": 11.5, "font-weight": 600 });
    txt(svg, "one outcome at a time", { x: x(0.7) - 8, y: y(0.62) + 26, "text-anchor": "end", fill: "var(--ink)", "font-size": 11.5, "font-weight": 600 });
    dataTable(c, "Mean reported probability by stated chance", ["Stated chance", ...F.map(f => f.short)], xs.map(v => [pct(+v), ...F.map(f => pct(DATA.odds[f.key][v]))]));
  })();

  // ===== Finding 05b: dice, coins, cards =====
  (function dice() {
    const c = document.getElementById("chartDice"), D = DATA.dice; if (!c) return;
    const tip = frame(c, "What Jev says vs the true odds for dice, coins and cards", "Average probability per outcome over option orders: all 720 orders for the dice, all 24 for coins and cards (5 calls each), 200 sampled orders for the sum of two dice · the outline is the true chance");
    legend(c, [{ name: "Jev (average over all orders)", color: S1, square: true }, { name: "True chance", color: "var(--muted)", square: true }]);
    const note = h("div", "dial-note"); note.setAttribute("aria-live", "polite");
    const W = 720, H = 280, m = { l: 48, r: 20, t: 16, b: 44 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Probability per outcome" });
    const y = v => m.t + (1 - v) * (H - m.t - m.b);
    function draw(i) {
      const d = D[i], k = d.labels.length; svg.replaceChildren();
      for (const g of [0, 0.25, 0.5, 0.75, 1]) { el("line", { x1: m.l, x2: W - m.r, y1: y(g), y2: y(g), stroke: g === 0 ? "var(--axis)" : "var(--grid)" }, svg); txt(svg, (g * 100) + "%", { x: m.l - 6, y: y(g) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11 }); }
      const slot = (W - m.l - m.r) / k, bw = Math.min(56, slot * 0.6);
      d.labels.forEach((lab, j) => {
        const cx = m.l + slot * (j + 0.5), v = d.jev[j], t = d.ideal[j];
        const top = y(v), hgt = y(0) - top;
        if (hgt > 0.5) el("path", { d: `M${cx - bw / 2},${y(0)} v${-Math.max(0, hgt - 4)} a4,4 0 0 1 4,-4 h${bw - 8} a4,4 0 0 1 4,4 v${Math.max(0, hgt - 4)} z`, fill: S1 }, svg);
        el("rect", { x: cx - bw / 2 - 3, y: y(t), width: bw + 6, height: y(0) - y(t), fill: "none", stroke: "var(--muted)", "stroke-width": 1.5, "stroke-dasharray": "4 3", rx: 3 }, svg);
        txt(svg, lab, { x: cx, y: H - m.b + 16, "text-anchor": "middle", fill: "var(--ink)", "font-size": 11.5 });
        txt(svg, pct(v), { x: cx, y: Math.min(top, y(t)) - 6, "text-anchor": "middle", fill: "var(--ink-2)", "font-size": 11 });
        hitTarget(svg, { x: cx - slot / 2, y: m.t, width: slot, height: H - m.t - m.b }, tip, c, `${d.name} · ${lab}`, [{ color: S1, value: pct(v), name: "Jev" }, { color: "var(--muted)", value: pct(t), name: "true chance" }], [cx, top]);
      });
      const avg1 = d.averaging["1"], avgAll = d.averaging[String(d.calls)] ?? Object.values(d.averaging).slice(-1)[0];
      note.textContent = `${d.calls.toLocaleString()} option orders. Distance from the true odds: ${avg1.toFixed(2)} for a single order, ${avgAll.toFixed(2)} after averaging over all the orders used (0 is perfect, 1 is completely wrong).`;
    }
    segmented(c, D.map(d => d.name), draw, 0, "Choose a device");
    c.appendChild(note); c.appendChild(svg); draw(0);
    dataTable(c, "Probability per outcome: Jev vs true chance", ["Device", "Outcomes (Jev / true)"], D.map(d => [d.name, d.labels.map((l, j) => `${l} ${pct(d.jev[j])}/${pct(d.ideal[j])}`).join(" · ")]));
  })();

  // ===== Finding 06: the follow-up questions =====
  (function meta() {
    const c = document.getElementById("chartMeta"), M = DATA.meta; if (!c) return;
    const tip = frame(c, "How well each follow-up question separates the original contrasts", "Original contrasts, before the shortcut controls · 1.0 is perfect separation, 0.5 is chance · bars show the original wording, whiskers the range across rewordings where tested (4–5; one wording for \"does the subject exist?\")");
    const W = 720, rowH = 50, m = { l: 290, r: 60, t: 6, b: 30 }, H = m.t + M.length * rowH + m.b;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Separation scores of follow-up questions" }, c);
    const x = v => m.l + (v - 0.5) / 0.5 * (W - m.l - m.r);
    for (const g of [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) { el("line", { x1: x(g), x2: x(g), y1: m.t, y2: H - m.b, stroke: g === 0.5 ? "var(--axis)" : "var(--grid)" }, svg); txt(svg, g.toFixed(1), { x: x(g), y: H - 10, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 }); }
    M.forEach((r, i) => {
      const yy = m.t + i * rowH + 10, bh = 18, w = Math.max(3, x(r.auroc) - x(0.5));
      txt(svg, r.name, { x: m.l - 12, y: yy + 11, "text-anchor": "end", fill: "var(--ink)", "font-size": 12.5, "font-weight": 600 });
      txt(svg, r.what, { x: m.l - 12, y: yy + 27, "text-anchor": "end", fill: "var(--ink-2)", "font-size": 11 });
      bar(svg, x(0.5), yy, w, bh, S3);
      if (r.lo != null) { el("line", { x1: x(r.lo), x2: x(r.hi), y1: yy + bh / 2, y2: yy + bh / 2, stroke: "var(--ink)", "stroke-width": 1.5 }, svg); for (const v of [r.lo, r.hi]) el("line", { x1: x(v), x2: x(v), y1: yy + 4, y2: yy + bh - 4, stroke: "var(--ink)", "stroke-width": 1.5 }, svg); }
      txt(svg, r.auroc.toFixed(2), { x: Math.max(x(r.auroc), r.hi ? x(r.hi) : 0) + 8, y: yy + 13, fill: "var(--ink-2)", "font-size": 12 });
      hitTarget(svg, { x: 0, y: yy - 6, width: W, height: rowH }, tip, c, r.name, [{ color: S3, value: r.auroc.toFixed(3), name: "original wording" }, ...(r.lo != null ? [{ value: `${r.lo.toFixed(3)}–${r.hi.toFixed(3)}`, name: `across ${r.n_wordings} wordings` }] : []), { name: r.what }], [x(r.auroc), yy]);
    });
    dataTable(c, "Separation score (AUROC) of each follow-up question", ["Question", "Original wording", "Range across wordings"], M.map(r => [r.name, r.auroc.toFixed(3), r.lo == null ? "—" : `${r.lo.toFixed(3)}–${r.hi.toFixed(3)}`]));
  })();

  // ===== Finding 07: second look =====
  (function check() {
    const K = DATA.self_check;
    dumbbell("chartCheck", {
      title: "Calibration error of the first answer vs a second-look check",
      sub: "Smooth calibration error (0 means confidence matches accuracy exactly) · lower is better",
      aName: "First answer", bName: "Second look: \"is the proposed answer correct?\"", aShort: "first answer", bShort: "second look",
      aColor: S2, bColor: S1, xMin: 0, xMax: 0.45, xStep: 0.05, left: 250, fmt: v => v.toFixed(3), fmtAxis: v => v.toFixed(2),
      groups: [{ name: "WHERE THE FIRST ANSWER WAS OVERCONFIDENT", rows: K.slice(0, 4).map(r => ({ name: r.name, a: r.ece_first, b: r.ece_check, note: `accuracy ${pct(r.acc)}; confidence ${pct(r.first)} → ${pct(r.check)}` })) },
               { name: "WHERE IT WAS ALREADY CALIBRATED", rows: K.slice(4).map(r => ({ name: r.name, a: r.ece_first, b: r.ece_check, note: `accuracy ${pct(r.acc)}; confidence ${pct(r.first)} → ${pct(r.check)}` })) }],
      tableCaption: "Calibration error before and after the second look",
    });
  })();

})();
