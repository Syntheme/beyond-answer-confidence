// The story's own chart: what "Do you know?" separates, next to baselines (surface-cue controls).
(() => {
  const DATA = JSON.parse(document.getElementById("blog-data-3").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => { const n = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v); if (parent) parent.appendChild(n); return n; };
  const txt = (parent, s, attrs) => { const t = el("text", attrs, parent); t.appendChild(document.createTextNode(s)); return t; };
  const h = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
  const c = document.getElementById("chartShortcut"); if (!c) return;
  c.appendChild(h("p", "chart-title", "What \"Do you know?\" picks up, before and after the clues are removed"));
  c.appendChild(h("p", "chart-sub", "How well each signal tells made-up subjects from real ones, or later news from earlier news. A score of 1.0 is perfect and 0.5 is no better than guessing. Each score is set up so that higher means made up or later (for Jev's answer, that means 1 − its confidence). Bars show the estimate and whiskers the 95% interval."));
  const colors = { text: "var(--muted)", known: "var(--s3)", conf: "var(--s2)" };
  const lg = h("div", "legend");
  for (const [k, name] of [["text", "Baseline that never calls Jev"], ["known", "\"Do you know?\" (1 − P(yes))"], ["conf", "Jev's answer uncertainty (1 − top probability)"]]) { const sp = h("span"), b = h("b", "sq"); b.style.background = colors[k]; sp.append(b, document.createTextNode(name)); lg.appendChild(sp); }
  c.appendChild(lg);
  const tip = h("div", "tip"); c.appendChild(tip);
  const showTip = (x, y, head, lines) => {
    tip.replaceChildren(); const hd = h("div", null, head); hd.style.color = "var(--ink-2)"; tip.appendChild(hd);
    for (const line of lines) tip.appendChild(h("div", "row", line));
    const cb = c.getBoundingClientRect(), sx = c.scrollLeft; tip.classList.add("on");
    const w = tip.offsetWidth; let left = x - cb.left + sx + 14; if (left + w > sx + cb.width - 4) left = x - cb.left + sx - w - 14;
    tip.style.left = Math.max(sx + 4, left) + "px"; tip.style.top = Math.max(4, y - cb.top + c.scrollTop - 10) + "px";
  };
  const rowH = 26, gapH = 40, m = { l: 230, r: 64, t: 4, b: 30 }, W = 720;
  const nRows = DATA.reduce((a, g) => a + g.rows.length, 0), H = m.t + nRows * rowH + DATA.length * gapH + m.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Separation scores of Do you know, answer uncertainty and text-only baselines under four tests" }, c);
  const lo = 0.3, hi = 1.0, x = v => m.l + (v - lo) / (hi - lo) * (W - m.l - m.r);
  for (const g of [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) { el("line", { x1: x(g), x2: x(g), y1: m.t, y2: H - m.b, stroke: g === 0.5 ? "var(--axis)" : "var(--grid)", "stroke-dasharray": g === 0.5 ? "4 3" : null }, svg); txt(svg, g.toFixed(1), { x: x(g), y: H - 10, "text-anchor": "middle", fill: "var(--muted)", "font-size": 11 }); }
  txt(svg, "chance", { x: x(0.5) + 4, y: m.t + 10, fill: "var(--muted)", "font-size": 10.5 });
  let yy = m.t;
  for (const g of DATA) {
    txt(svg, g.name, { x: 0, y: yy + 17, fill: "var(--ink)", "font-size": 12.5, "font-weight": 700 });
    txt(svg, g.detail, { x: 0, y: yy + 32, fill: "var(--ink-2)", "font-size": 11 });
    yy += gapH;
    for (const r of g.rows) {
      const cy = yy + rowH / 2, col = colors[r.kind];
      txt(svg, r.name, { x: m.l - 12, y: cy + 4, "text-anchor": "end", fill: "var(--ink)", "font-size": 12 });
      const x0 = x(Math.min(0.5, r.auroc)), x1 = x(Math.max(0.5, r.auroc));
      if (x1 - x0 > 0) el("rect", { x: x0, y: cy - 7, width: x1 - x0, height: 14, rx: x1 - x0 >= 6 ? 3 : 0, fill: col, opacity: 0.9 }, svg);
      el("line", { x1: x(r.lo), x2: x(r.hi), y1: cy, y2: cy, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
      for (const v of [r.lo, r.hi]) el("line", { x1: x(v), x2: x(v), y1: cy - 5, y2: cy + 5, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
      txt(svg, r.auroc.toFixed(2), { x: Math.max(x(r.hi), x1) + 8, y: cy + 4, fill: "var(--ink-2)", "font-size": 11.5 });
      const label = `${g.name}, ${r.name}: ${r.auroc.toFixed(2)}, 95% interval ${r.lo.toFixed(2)} to ${r.hi.toFixed(2)}`;
      const hit = el("rect", { x: 0, y: yy, width: W, height: rowH, fill: "transparent", tabindex: 0, role: "img", "data-mark": "", "aria-label": label }, svg);
      const lines = [`${r.auroc.toFixed(3)} separation`, `95% interval ${r.lo.toFixed(3)}–${r.hi.toFixed(3)}`];
      const on = (px, py) => showTip(px, py, `${g.name} · ${r.name}`, lines);
      hit.addEventListener("pointermove", ev => on(ev.clientX, ev.clientY));
      hit.addEventListener("pointerdown", ev => on(ev.clientX, ev.clientY));
      hit.addEventListener("focus", () => { const b = svg.getBoundingClientRect(); on(b.left + x(r.auroc) * b.width / W, b.top + cy * b.height / H); });
      hit.addEventListener("pointerleave", ev => { if (ev.pointerType !== "touch") tip.classList.remove("on"); });
      hit.addEventListener("blur", () => tip.classList.remove("on"));
      yy += rowH;
    }
  }
  const d = h("details", "data"); d.appendChild(h("summary", null, "Show data table"));
  const t = h("table"), thead = h("thead"), tbody = h("tbody"), tr = h("tr");
  for (const [i, x0] of ["Test · readout", "AUROC", "95% interval"].entries()) { const th = h("th", i ? "num" : null, x0); th.scope = "col"; tr.appendChild(th); }
  thead.appendChild(tr); t.append(thead, tbody);
  for (const g of DATA) for (const r of g.rows) { const row = h("tr"); row.append(h("td", null, `${g.name} · ${r.name}`), h("td", "num", r.auroc.toFixed(3)), h("td", "num", `${r.lo.toFixed(3)}–${r.hi.toFixed(3)}`)); tbody.appendChild(row); }
  d.appendChild(t); c.after(d);
})();
