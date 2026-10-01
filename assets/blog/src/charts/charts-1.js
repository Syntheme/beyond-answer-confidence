    (() => {
  const DATA = JSON.parse(document.getElementById("blog-data-1").textContent);
  // ---- theme toggle: light by default; the button switches to dark and back ----
  const root = document.documentElement, btn = document.getElementById("themeToggle");
  const isDark = () => root.dataset.theme === "dark";
  const label = () => { btn.textContent = isDark() ? "Light mode" : "Dark mode"; };
  const store = { get: () => { try { return localStorage.getItem("jev-blog-theme"); } catch { return null; } }, set: v => { try { localStorage.setItem("jev-blog-theme", v); } catch { /* storage unavailable */ } } };
  if (store.get() === "dark") root.dataset.theme = "dark";
  btn.addEventListener("click", () => { root.dataset.theme = isDark() ? "light" : "dark"; store.set(root.dataset.theme); label(); });
  label();
  // printing: expand every collapsed box so data tables and notes are on paper
  let printState = null;
  window.addEventListener("beforeprint", () => {
    printState = { theme: root.dataset.theme, open: [...document.querySelectorAll("details")].map(d => [d, d.open]) };
    root.dataset.theme = "light";
    printState.open.forEach(([d]) => { d.open = true; });
  });
  window.addEventListener("afterprint", () => {
    if (!printState) return;
    printState.open.forEach(([d, o]) => { d.open = o; });
    if (printState.theme) root.dataset.theme = printState.theme; else delete root.dataset.theme;
    printState = null;
  });
  // touch: a tap on a chart mark shows its values; a tap elsewhere hides them
  document.addEventListener("pointerdown", ev => { if (!(ev.target instanceof Element) || !ev.target.closest("[data-mark]")) document.querySelectorAll(".tip.on").forEach(t => t.classList.remove("on")); });
  // narrow screens: charts scroll sideways instead of shrinking their text
  document.querySelectorAll(".chart").forEach(c => { const hint = document.createElement("p"); hint.className = "chart-hint"; hint.textContent = "Scroll the chart sideways to see all of it."; c.prepend(hint); });
  // ---- pilcrow deep links on section headings ----
  document.querySelectorAll("h2[id]").forEach(h2 => { const a = document.createElement("a"); a.className = "pil"; a.href = "#" + h2.id; a.textContent = "¶"; a.setAttribute("aria-label", "Link to this section"); h2.prepend(a); });

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

  // ===== 1. The dial: one real message, every setting =====
  (function dial() {
    const c = document.getElementById("chartDial"), D = DATA.dial; if (!c) return;
    const tip = frame(c, "One message, every setting", `"${D.text}" · right answer: ${D.gold} · nine settings from the main test, plus the no-hints test with letter codes`);
    legend(c, [{ name: "The right answer", color: "var(--s1)", square: true }, { name: "Any other option", color: "var(--ref)", square: true }]);
    const notes = {
      "No hints": "Nothing to go on, yet a third of the confidence lands on INTENT_00, the first code. The right answer isn't even in the top five.",
      "No hints, letter codes": "A separate test with codes like INTENT_DNTJ. The first-code habit disappears, but Jev still puts 19% on one arbitrary option.",
      "1 example": `Wrong, for an understandable reason: the only example for "getting virtual card" was "${D.example_lookalike}".`,
      "2 examples": "One more example per category and the right answer takes the lead.",
      "4 examples": "More sure, and right.",
      "8 examples": "96% on the right answer. Most of the rest goes to the look-alike \"card delivery estimate\".",
      "Category names": "Names alone get it right at 91%.",
      "Names + 8 examples": "The most information, the most certainty: 97%.",
      "Irrelevant filler": "Unrelated sentences attached to every code. Probability scatters onto arbitrary options, and the top pick is wrong.",
      "Swapped examples": "The examples for card arrival were moved onto another code, and Jev follows them to that code at 100%. It trusts the evidence completely.",
    };
    const W = 720, rowH = 34, m = { l: 350, r: 56, t: 4, b: 22 }, H = m.t + 5 * rowH + m.b;
    const clip = t => (t.length > 30 ? t.slice(0, 29) + "…" : t);
    const note = h("div", "dial-note"); note.setAttribute("aria-live", "polite");
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Top five options and their probabilities for the selected setting" });
    const x = v => m.l + v * (W - m.l - m.r);
    function draw(i) {
      const S = D.settings[i]; svg.replaceChildren();
      for (const g of [0, 0.25, 0.5, 0.75, 1]) { el("line", { x1: x(g), x2: x(g), y1: m.t, y2: H - m.b, stroke: g === 0 ? "var(--axis)" : "var(--grid)" }, svg); txt(svg, (g * 100) + "%", { x: x(g), y: H - 6, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 }); }
      S.top.forEach((o, k) => {
        const yy = m.t + k * rowH + 7, bh = rowH - 14, right = o.name === D.gold, color = right ? "var(--s1)" : "var(--ref)";
        txt(svg, (right ? "✓ " : "") + o.code, { x: 108, y: yy + bh / 2 + 4, "text-anchor": "end", fill: "var(--ink)", "font-size": 12, "font-family": "ui-monospace, Menlo, monospace" });
        txt(svg, clip(o.name), { x: 116, y: yy + bh / 2 + 4, fill: right ? "var(--ink)" : "var(--ink-2)", "font-size": 12, "font-weight": right ? 650 : 400 });
        const w = o.p > 0 ? Math.max(1.5, x(o.p) - x(0)) : 0;
        if (w >= 6) el("path", { d: `M${x(0)},${yy} h${w - 4} a4,4 0 0 1 4,4 v${bh - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} z`, fill: color }, svg);
        else if (w > 0) el("rect", { x: x(0), y: yy, width: w, height: bh, fill: color }, svg);
        txt(svg, pct(o.p), { x: x(0) + w + 6, y: yy + bh / 2 + 4, fill: "var(--ink-2)", "font-size": 12 });
        hitTarget(svg, { x: 0, y: yy - 6, width: W, height: bh + 12 }, tip, c, `${S.label} · ${o.code}`,
          [{ color, value: pct(o.p), name: `${o.name}${right ? " (right answer)" : ""}` }, { name: `average of ${S.reps} identical requests` }], [x(o.p), yy]);
      });
      note.textContent = notes[S.label] || "";
    }
    segmented(c, D.settings.map(s => s.label), draw, 0, "Choose what Jev is told");
    c.appendChild(note); c.appendChild(svg); draw(0);
    dataTable(c, "Top five options per setting for this message", ["Setting", "1st", "2nd", "3rd", "4th", "5th"],
      D.settings.map(S => [S.label, ...S.top.map(o => `${o.name} ${pct(o.p)}`)]));
  })();

  // ===== 2. Accuracy vs confidence by knowledge (small multiples, crosshair) =====
  (function knowledge() {
    const c = document.getElementById("chartKnow"); if (!c) return;
    const tip = frame(c, "More information: more accurate, and more confident", "How often Jev was right vs how sure it said it was · examples per category (plus no hints)");
    const S = [{ key: "acc", name: "Accuracy (how often right)", short: "Accuracy", color: "var(--s1)" }, { key: "conf", name: "Average confidence (how sure)", short: "Confidence", color: "var(--s2)" }];
    legend(c, S);
    const X = ["No hints", "1", "2", "4", "8"];
    const P = [
      { title: "Banking77 · 77 look-alike categories", acc: [0.010, 0.681, 0.796, 0.861, 0.899], conf: [0.320, 0.828, 0.879, 0.914, 0.927] },
      { title: "CLINC150 · 150 distinct categories", acc: [0.008, 0.911, 0.958, 0.969, 0.979], conf: [0.363, 0.904, 0.953, 0.964, 0.974] },
    ];
    const wrap = h("div", "panels"); c.appendChild(wrap);
    const W = 360, H = 250, m = { l: 40, r: 70, t: 28, b: 42 };
    for (const p of P) {
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": `${p.title}: accuracy and confidence` }, wrap);
      const x = i => m.l + i * (W - m.l - m.r) / (X.length - 1), y = v => m.t + (1 - v) * (H - m.t - m.b);
      txt(svg, p.title, { x: m.l, y: 14, fill: "var(--ink)", "font-size": 12, "font-weight": 650 });
      for (const g of [0, 0.25, 0.5, 0.75, 1]) { el("line", { x1: m.l, x2: W - m.r, y1: y(g), y2: y(g), stroke: g === 0 ? "var(--axis)" : "var(--grid)" }, svg); txt(svg, (g * 100) + "%", { x: m.l - 6, y: y(g) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 10 }); }
      X.forEach((lab, i) => txt(svg, lab, { x: x(i), y: H - m.b + 16, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10 }));
      txt(svg, "examples per category", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", fill: "var(--muted)", "font-size": 10 });
      for (const s of S) {
        el("polyline", { points: p[s.key].map((v, i) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round" }, svg);
        p[s.key].forEach((v, i) => el("circle", { cx: x(i), cy: y(v), r: 4.5, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg));
      }
      const ends = S.map(s => ({ s, yy: y(p[s.key][X.length - 1]) })).sort((a, b) => a.yy - b.yy);
      if (ends[1].yy - ends[0].yy < 13) { const mid = (ends[0].yy + ends[1].yy) / 2; ends[0].yy = mid - 7; ends[1].yy = mid + 7; }
      for (const e of ends) txt(svg, e.s.short, { x: W - m.r + 8, y: e.yy + 4, fill: "var(--ink-2)", "font-size": 10.5 });
      const cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--muted)", opacity: 0 }, svg);
      const hit = el("rect", { x: m.l - 16, y: m.t - 8, width: W - m.l - m.r + 32, height: H - m.t - m.b + 16, fill: "transparent", tabindex: 0, role: "img", "data-mark": "", "aria-label": `${p.title}; use the arrow keys to step through the settings` }, svg);
      const live = h("div", "sr-live"); live.setAttribute("aria-live", "polite"); c.appendChild(live);
      let fi = 0;
      const show = (i, cx, cy) => { cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 0.8);
        const head = `${p.title.split(" ·")[0]} · ${X[i] === "No hints" ? "no hints" : X[i] + " example" + (X[i] === "1" ? "" : "s")}`, vals = S.map(s => ({ color: s.color, value: pct(p[s.key][i]), name: s.short.toLowerCase() }));
        live.textContent = `${head}: ${vals.map(v => `${v.name} ${v.value}`).join(", ")}`;
        showTip(tip, c, cx, cy, head, vals); };
      const at = i => { const b = svg.getBoundingClientRect(); show(i, b.left + x(i) * b.width / W, b.top + y(p.conf[i]) * b.height / H); };
      const near = ev => { const b = svg.getBoundingClientRect(), sx = (ev.clientX - b.left) * W / b.width; let i = 0, best = 1e9; X.forEach((_, j) => { const d = Math.abs(x(j) - sx); if (d < best) { best = d; i = j; } }); fi = i; show(i, ev.clientX, ev.clientY); };
      hit.addEventListener("pointermove", near); hit.addEventListener("pointerdown", near);
      hit.addEventListener("keydown", ev => { if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") { fi = Math.max(0, Math.min(X.length - 1, fi + (ev.key === "ArrowRight" ? 1 : -1))); ev.preventDefault(); at(fi); } });
      hit.addEventListener("focus", () => at(fi));
      const off = () => { cross.setAttribute("opacity", 0); hide(tip); };
      hit.addEventListener("pointerleave", ev => { if (ev.pointerType !== "touch") off(); });
      hit.addEventListener("blur", off);
    }
    dataTable(c, "Accuracy and average confidence", ["Setting", "Banking77 accuracy", "Banking77 confidence", "CLINC150 accuracy", "CLINC150 confidence"],
      X.map((lab, i) => [lab === "No hints" ? lab : lab + " examples", pct(P[0].acc[i]), pct(P[0].conf[i]), pct(P[1].acc[i]), pct(P[1].conf[i])]));
  })();

  // ===== 3. Zero knowledge: confidence on one option (bars) =====
  (function zero() {
    const c = document.getElementById("chartZero"); if (!c) return;
    const tip = frame(c, "With no information, how much confidence goes on a single option?", "Banking77 · 77 options · no hints · average probability on the top choice (accuracy ≈ 1% in every row)");
    const rows = [
      { name: "Even split across 77 options", v: 1 / 77, color: "var(--muted)", note: "the spread a model with no information would give" },
      { name: "GLiNER2.5-Decide (open model)", v: 0.017, color: "var(--s1)", note: "same messages, same codes" },
      { name: "Jev · random letter codes", v: 0.162, color: "var(--s1)", note: "codes like INTENT_QKTV; first option chosen 0.4% of the time" },
      { name: "Jev · numbered codes", v: 0.320, color: "var(--s1)", note: "codes like INTENT_00; first option chosen 100% of the time" },
    ];
    const W = 720, rowH = 40, m = { l: 240, r: 60, t: 6, b: 26 }, H = m.t + rows.length * rowH + m.b;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Top-choice probability with no information" }, c);
    const x = v => m.l + v / 0.4 * (W - m.l - m.r);
    for (const g of [0, 0.1, 0.2, 0.3, 0.4]) { el("line", { x1: x(g), x2: x(g), y1: m.t, y2: H - m.b, stroke: g === 0 ? "var(--axis)" : "var(--grid)" }, svg); txt(svg, (g * 100) + "%", { x: x(g), y: H - 8, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 }); }
    rows.forEach((r, i) => {
      const yy = m.t + i * rowH + 8, bh = rowH - 16, w = Math.max(4, x(r.v) - x(0));
      txt(svg, r.name, { x: m.l - 10, y: yy + bh / 2 + 4, "text-anchor": "end", fill: "var(--ink)", "font-size": 12.5 });
      el("path", { d: `M${x(0)},${yy} h${w - 4} a4,4 0 0 1 4,4 v${bh - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} z`, fill: r.color }, svg);
      txt(svg, pct(r.v), { x: x(0) + w + 6, y: yy + bh / 2 + 4, fill: "var(--ink-2)", "font-size": 12 });
      hitTarget(svg, { x: 0, y: yy - 6, width: W, height: bh + 12 }, tip, c, r.name, [{ color: r.color, value: pct(r.v), name: "on the top choice" }, { name: r.note }], [x(r.v), yy]);
    });
    dataTable(c, "Top-choice probability with no information (Banking77)", ["Model / setting", "Top-choice probability"], rows.map(r => [r.name, pct(r.v)]));
  })();

  // ===== reliability helper: points + Wilson whiskers + diagonal =====
  function reliability(cId, title, sub, series, byOption, options, directLabels) {
    const c = document.getElementById(cId); if (!c) return;
    const tip = frame(c, title, sub);
    legend(c, series.map(s => ({ name: s.name, color: s.color })));
    const W = 560, H = 380, m = { l: 48, r: 24, t: 14, b: 44 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": title });
    const x = v => m.l + v * (W - m.l - m.r), y = v => m.t + (1 - v) * (H - m.t - m.b);
    function draw(i) {
      svg.replaceChildren();
      for (const g of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
        el("line", { x1: m.l, x2: W - m.r, y1: y(g), y2: y(g), stroke: g === 0 ? "var(--axis)" : "var(--grid)" }, svg);
        el("line", { x1: x(g), x2: x(g), y1: m.t, y2: H - m.b, stroke: "var(--grid)" }, svg);
        txt(svg, Math.round(g * 100) + "%", { x: m.l - 6, y: y(g) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 11 });
        txt(svg, Math.round(g * 100) + "%", { x: x(g), y: H - m.b + 16, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 });
      }
      txt(svg, "stated confidence (group average)", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 });
      txt(svg, "actually right", { x: 12, y: (m.t + H - m.b) / 2, transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})`, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 });
      el("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: "var(--muted)", "stroke-dasharray": "4 4" }, svg);
      const data = byOption(options[i]);
      // Put the diagonal's label where it is farthest from any data point.
      const allPts = series.flatMap(s => data[s.key] || []);
      const spot = [0.08, 0.2, 0.32, 0.44, 0.56, 0.68].reduce((best, t) => {
        const d = Math.min(...allPts.map(p => Math.hypot(x(p.conf) - x(t), y(p.acc) - y(t))), 1e9);
        return d > best.d ? { t, d } : best;
      }, { t: 0.08, d: -1 }).t;
      const ang = Math.atan((H - m.t - m.b) / (W - m.l - m.r)) * 180 / Math.PI;
      txt(svg, "confidence = accuracy", { x: x(spot), y: y(spot) - 6, fill: "var(--muted)", "font-size": 10.5, transform: `rotate(-${ang} ${x(spot)} ${y(spot) - 6})` });
      for (const s of series) {
        const pts = data[s.key]; if (!pts) continue;
        el("polyline", { points: pts.map(p => `${x(p.conf)},${y(p.acc)}`).join(" "), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", opacity: 0.9 }, svg);
        for (const p of pts) {
          el("line", { x1: x(p.conf), x2: x(p.conf), y1: y(p.lo), y2: y(p.hi), stroke: s.color, "stroke-width": 1.5, opacity: 0.55 }, svg);
          if (s.square) el("rect", { x: x(p.conf) - 4.5, y: y(p.acc) - 4.5, width: 9, height: 9, rx: 1.5, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
          else el("circle", { cx: x(p.conf), cy: y(p.acc), r: 4.5, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
          hitTarget(svg, { x: x(p.conf) - 9, y: y(p.acc) - 9, width: 18, height: 18 }, tip, c, `${s.name} · ${options[i]}`,
            [{ color: s.color, value: pct(p.acc), name: "actually right" }, { value: pct(p.conf), name: "stated confidence" }, { name: `95% interval ${pct(p.lo)}–${pct(p.hi)} · ${p.n} messages` }], [x(p.conf), y(p.acc)]);
        }
        if (directLabels) {
          const lowest = pts.reduce((a, b) => (b.conf < a.conf ? b : a));
          let above = lowest.acc >= lowest.conf; // above the diagonal: label above; below: label below
          if (!above && y(lowest.acc) + 18 > H - m.b - 4) above = true;
          txt(svg, s.short, { x: x(lowest.conf) + 8, y: y(lowest.acc) + (above ? -10 : 18), fill: "var(--ink)", "font-size": 11.5, "font-weight": 600 });
        }
      }
    }
    segmented(c, options, draw);
    c.appendChild(svg); draw(0);
    const rows = [];
    for (const o of options) { const d = byOption(o); for (const s of series) { if (d[s.key]) rows.push([`${o} · ${s.name}`, d[s.key].map(p => `${pct(p.conf)}→${pct(p.acc)}`).join("  ")]); } }
    dataTable(c, "Groups of similar size, lowest to highest confidence (stated → actual); groups with identical confidence are merged", ["Setting · series", "Groups"], rows);
  }

  // ===== 4. Jev reliability by knowledge setting =====
  reliability("chartRel", "Does Jev's confidence match its accuracy?", "Each dot is a group of messages sorted by confidence (up to ten groups of similar size; groups with identical confidence are merged). Vertical lines are 95% intervals. Below the diagonal means overconfident.",
    [{ key: "banking77", name: "Banking77", short: "Banking77", color: "var(--s1)" }, { key: "clinc150", name: "CLINC150", short: "CLINC150", color: "var(--s2)", square: true }],
    o => ({ banking77: DATA.reliability.banking77[o], clinc150: DATA.reliability.clinc150[o] }),
    Object.keys(DATA.reliability.banking77), true);

  // ===== 5. One example per category: accuracy vs confidence (dumbbells) =====
  (function one() {
    const c = document.getElementById("chartOne"); if (!c) return;
    const tip = frame(c, "One example per category: how right vs how sure", "Banking77 · same 770 messages · confidence to the right of accuracy = overconfident; to the left = underconfident");
    const S = [{ key: "acc", name: "Accuracy (how often right)", color: "var(--s1)" }, { key: "conf", name: "Average confidence (how sure)", color: "var(--s2)" }];
    legend(c, S);
    const rows = [
      { name: "Jev", acc: 0.681, conf: 0.828, note: "15 points overconfident" },
      { name: "Five students, same examples", acc: 0.444, conf: 0.544, note: "less accurate and less sure; 10 points overconfident" },
      { name: "GLiNER2.5-Decide", acc: 0.500, conf: 0.378, note: "12 points underconfident" },
    ];
    const W = 720, rowH = 46, m = { l: 250, r: 40, t: 8, b: 28 }, H = m.t + rows.length * rowH + m.b;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Accuracy vs confidence with one example per category" }, c);
    const x = v => m.l + (v - 0.3) / 0.6 * (W - m.l - m.r);
    for (const g of [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) { el("line", { x1: x(g), x2: x(g), y1: m.t, y2: H - m.b, stroke: "var(--grid)" }, svg); txt(svg, Math.round(g * 100) + "%", { x: x(g), y: H - 8, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 }); }
    rows.forEach((r, i) => {
      const yy = m.t + i * rowH + rowH / 2;
      txt(svg, r.name, { x: m.l - 12, y: yy + 4, "text-anchor": "end", fill: "var(--ink)", "font-size": 12.5 });
      el("line", { x1: x(r.acc), x2: x(r.conf), y1: yy, y2: yy, stroke: "var(--axis)", "stroke-width": 2 }, svg);
      for (const s of S) { el("circle", { cx: x(r[s.key]), cy: yy, r: 6, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg); txt(svg, pct(r[s.key]), { x: x(r[s.key]), y: yy - 11, "text-anchor": "middle", fill: "var(--ink-2)", "font-size": 11 }); }
      hitTarget(svg, { x: 0, y: yy - rowH / 2, width: W, height: rowH }, tip, c, r.name, [...S.map(s => ({ color: s.color, value: pct(r[s.key]), name: s.key === "acc" ? "accuracy" : "confidence" })), { name: r.note }], [x(r.conf), yy]);
    });
    dataTable(c, "Accuracy and confidence with one example per category (Banking77)", ["Model", "Accuracy", "Confidence"], rows.map(r => [r.name, pct(r.acc), pct(r.conf)]));
  })();

  // ===== 6. Jev vs GLiNER reliability =====
  reliability("chartRival", "Jev vs GLiNER2.5-Decide: whose confidence matches accuracy?", "Same 770 Banking77 messages, same codes. Above the diagonal = underconfident; below = overconfident. Squares = GLiNER, circles = Jev.",
    [{ key: "jev", name: "Jev", short: "Jev", color: "var(--s1)" }, { key: "gliner", name: "GLiNER2.5-Decide", short: "GLiNER", color: "var(--s3)", square: true }],
    o => DATA.gliner_reliability[o], Object.keys(DATA.gliner_reliability), true);

  // ===== 7. Most frequent confusions (table-like bars) =====
  (function conf() {
    const c = document.getElementById("chartConf"); if (!c) return;
    const tip = frame(c, "Jev's most frequent mistakes with full information", "Names + 8 examples · right answer → Jev's answer · count of test messages (the students' count for the same mix-up in brackets)");
    const W = 720, rowH = 28, m = { l: 470, r: 50, t: 4, b: 8 };
    const items = [...DATA.confusions.banking77.map(p => ({ ...p, ds: "Banking77" })), ...DATA.confusions.clinc150.map(p => ({ ...p, ds: "CLINC150" }))];
    const H = m.t + items.length * rowH + m.b + 22 * new Set(items.map(p => p.ds)).size;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Most frequent confusions" }, c);
    const max = Math.max(...items.map(p => p.jev));
    const x = v => m.l + v / max * (W - m.l - m.r);
    let yy = m.t, last = null;
    for (const p of items) {
      if (p.ds !== last) { txt(svg, p.ds, { x: 0, y: yy + 14, fill: "var(--ink-2)", "font-size": 11.5, "font-weight": 700 }); yy += 22; last = p.ds; }
      txt(svg, `${p.gold} → ${p.chosen}`, { x: m.l - 10, y: yy + 14, "text-anchor": "end", fill: "var(--ink)", "font-size": 12 });
      const w = Math.max(3, x(p.jev) - x(0));
      el("path", { d: `M${x(0)},${yy + 4} h${w - 4} a4,4 0 0 1 4,4 v${rowH - 16} a4,4 0 0 1 -4,4 h${-(w - 4)} z`, fill: "var(--s1)" }, svg);
      txt(svg, `${p.jev} (${p.ensemble})`, { x: x(0) + w + 6, y: yy + 14, fill: "var(--ink-2)", "font-size": 11.5 });
      hitTarget(svg, { x: 0, y: yy, width: W, height: rowH }, tip, c, `${p.ds}: ${p.gold} → ${p.chosen}`, [{ color: "var(--s1)", value: String(p.jev), name: "Jev mistakes" }, { value: String(p.ensemble), name: "students' mistakes (same mix-up)" }], [x(p.jev), yy + 8]);
      yy += rowH;
    }
    dataTable(c, "Most frequent confusions (names + 8 examples)", ["Dataset · right → chosen", "Jev", "Students"], items.map(p => [`${p.ds} · ${p.gold} → ${p.chosen}`, String(p.jev), String(p.ensemble)]));
  })();
})();
