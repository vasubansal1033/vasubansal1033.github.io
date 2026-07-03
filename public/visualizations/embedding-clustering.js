(function () {
  const D3_URL = "https://cdn.jsdelivr.net/npm/d3@7/+esm";
  const EMB_DIM = 2;
  const MAX_STEPS = 4000;
  const STEPS_PER_FRAME = 8;
  const LR = 0.08;

  const initialized = new WeakSet();
  let d3Promise = null;

  function loadD3() {
    if (!d3Promise) {
      d3Promise = import(/* @vite-ignore */ D3_URL);
    }
    return d3Promise;
  }

  /* ---------- vocabulary & corpus ---------- */

  const TOKENS = [
    "the",
    "cat",
    "dog",
    "cow",
    "apple",
    "mango",
    "eat",
    "chase",
    "see",
    "red",
    "yellow",
  ];

  const TOKEN_INDEX = Object.fromEntries(TOKENS.map((t, i) => [t, i]));

  const CATEGORY = {
    the: "meta",
    cat: "animal",
    dog: "animal",
    cow: "animal",
    apple: "fruit",
    mango: "fruit",
    eat: "verb",
    chase: "verb",
    see: "verb",
    red: "attr",
    yellow: "attr",
  };

  const CATEGORY_LABEL = {
    meta: "function word",
    animal: "animal",
    fruit: "fruit",
    verb: "verb",
    attr: "fruit descriptor",
  };

  const ANIMALS = ["cat", "dog", "cow"];
  const FRUITS = ["apple", "mango"];
  const VERBS = ["eat", "chase", "see"];
  const ATTRS = ["red", "yellow"];

  function buildTrainingPairs() {
    const pairs = [];
    const pushAll = (cur, nexts) => {
      for (const nxt of nexts) pairs.push([TOKEN_INDEX[cur], TOKEN_INDEX[nxt]]);
    };

    // Cyclic toy grammar: the -> animal -> verb -> color -> fruit -> the
    // e.g. "the cat sees red apple", then back to "the ...".
    // Every token in a category shares one next-token distribution, and each
    // category's distribution is distinct from every other's, so all five
    // clusters separate instead of collapsing together.
    for (const a of ANIMALS) pushAll("the", [a]); // the   -> animals
    for (const a of ANIMALS) pushAll(a, VERBS); // animals -> verbs
    for (const v of VERBS) pushAll(v, ATTRS); // verbs   -> colors
    for (const c of ATTRS) pushAll(c, FRUITS); // colors  -> fruits
    for (const f of FRUITS) pushAll(f, ["the"]); // fruits  -> the

    return pairs;
  }

  /* ---------- tiny next-token model ---------- */

  function randn() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function makeModel() {
    const n = TOKENS.length;
    const embed = Array.from({ length: n }, () =>
      Array.from({ length: EMB_DIM }, () => randn() * 0.35)
    );
    const outW = Array.from({ length: n }, () =>
      Array.from({ length: EMB_DIM }, () => randn() * 0.35)
    );
    const bias = new Array(n).fill(0);
    return { embed, outW, bias };
  }

  function softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map(z => Math.exp(z - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }

  function forward(model, cur) {
    const logits = model.outW.map((w, j) => {
      let z = model.bias[j];
      for (let d = 0; d < EMB_DIM; d++) z += w[d] * model.embed[cur][d];
      return z;
    });
    return softmax(logits);
  }

  function trainStep(model, pairs) {
    let loss = 0;
    for (const [cur, target] of pairs) {
      const probs = forward(model, cur);
      loss -= Math.log(Math.max(probs[target], 1e-9));

      for (let j = 0; j < TOKENS.length; j++) {
        const err = probs[j] - (j === target ? 1 : 0);
        model.bias[j] -= LR * err;
        for (let d = 0; d < EMB_DIM; d++) {
          model.outW[j][d] -= LR * err * model.embed[cur][d];
          model.embed[cur][d] -= LR * err * model.outW[j][d];
        }
      }
    }
    return loss / pairs.length;
  }

  function dist2(a, b) {
    let s = 0;
    for (let d = 0; d < EMB_DIM; d++) {
      const dx = a[d] - b[d];
      s += dx * dx;
    }
    return s;
  }

  function nearestNeighbors(model, tokenIdx, k = 3) {
    const self = model.embed[tokenIdx];
    return TOKENS.map((tok, i) => ({
      tok,
      idx: i,
      dist: dist2(self, model.embed[i]),
      cat: CATEGORY[tok],
    }))
      .filter(d => d.idx !== tokenIdx)
      .sort((a, b) => a.dist - b.dist)
      .slice(0, k);
  }

  /* ---------- theme ---------- */

  const CAT_COLOR = {
    animal: "#f97316",
    fruit: "#22c55e",
    verb: "#3b82f6",
    attr: "#a855f7",
    meta: "#94a3b8",
  };

  function getColors() {
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    const styles = getComputedStyle(document.documentElement);
    return {
      isDark,
      text: styles.getPropertyValue("--foreground").trim() || "#334155",
      axis: isDark ? "rgba(148,163,184,0.7)" : "rgba(100,116,139,0.7)",
      grid: isDark ? "rgba(148,163,184,0.12)" : "rgba(100,116,139,0.12)",
      pointStroke: isDark ? "rgba(2,6,23,0.75)" : "rgba(255,255,255,0.9)",
      bg: isDark ? "rgba(15,23,42,0.35)" : "rgba(248,250,252,0.6)",
      accent:
        styles.getPropertyValue("--accent").trim() ||
        (isDark ? "#60a5fa" : "#2563eb"),
    };
  }

  /* ---------- rendering ---------- */

  function makeContext(canvas, w, h) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    return { ctx: canvas.getContext("2d"), dpr };
  }

  function embeddingExtents(model) {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (const e of model.embed) {
      x0 = Math.min(x0, e[0]);
      x1 = Math.max(x1, e[0]);
      y0 = Math.min(y0, e[1]);
      y1 = Math.max(y1, e[1]);
    }
    const padX = Math.max(0.35, (x1 - x0) * 0.15);
    const padY = Math.max(0.35, (y1 - y0) * 0.15);
    return [x0 - padX, x1 + padX, y0 - padY, y1 + padY];
  }

  const PLOT_W = 420;
  const PLOT_H = 340;
  const PLOT_M = { t: 18, r: 18, b: 36, l: 36 };

  // Map screen (drawn) coordinates back so we can hit-test the mouse.
  function computePositions(model, d3, view) {
    const [x0, x1, y0, y1] = embeddingExtents(model);
    const bx = d3.scaleLinear([x0, x1], [PLOT_M.l, PLOT_W - PLOT_M.r]);
    const by = d3.scaleLinear([y0, y1], [PLOT_H - PLOT_M.b, PLOT_M.t]);
    const positions = [];
    for (let i = 0; i < TOKENS.length; i++) {
      positions.push({
        i,
        tok: TOKENS[i],
        cat: CATEGORY[TOKENS[i]],
        px: view.tx + view.k * bx(model.embed[i][0]),
        py: view.ty + view.k * by(model.embed[i][1]),
      });
    }
    return positions;
  }

  function drawEmbeddingPlot(
    canvas,
    dpr,
    model,
    d3,
    colors,
    step,
    view,
    hover
  ) {
    const W = PLOT_W;
    const H = PLOT_H;
    const m = PLOT_M;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    ctx.save();
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(1, 1, W - 2, H - 2, 10);
    else ctx.rect(1, 1, W - 2, H - 2);
    ctx.fillStyle = colors.bg;
    ctx.fill();
    ctx.restore();

    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const gx = m.l + ((W - m.l - m.r) * i) / 4;
      const gy = m.t + ((H - m.t - m.b) * i) / 4;
      ctx.beginPath();
      ctx.moveTo(gx, m.t);
      ctx.lineTo(gx, H - m.b);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(m.l, gy);
      ctx.lineTo(W - m.r, gy);
      ctx.stroke();
    }

    ctx.fillStyle = colors.text;
    ctx.font = "10px system-ui,sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("embedding dim 0", (m.l + W - m.r) / 2, H - 8);
    ctx.save();
    ctx.translate(12, (m.t + H - m.b) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("embedding dim 1", 0, 0);
    ctx.restore();

    const positions = computePositions(model, d3, view);
    const hoverCat = hover >= 0 ? positions[hover].cat : null;

    // Only label a point when it has room (no close neighbor on screen), or
    // when it (or a same-category sibling) is being hovered. This declutters
    // the tight clusters until you zoom in or hover.
    const showLabel = new Array(positions.length).fill(false);
    for (let a = 0; a < positions.length; a++) {
      let minD = Infinity;
      for (let b = 0; b < positions.length; b++) {
        if (a === b) continue;
        const dx = positions[a].px - positions[b].px;
        const dy = positions[a].py - positions[b].py;
        minD = Math.min(minD, Math.hypot(dx, dy));
      }
      showLabel[a] = minD > 30;
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(m.l, m.t, W - m.l - m.r, H - m.t - m.b);
    ctx.clip();

    for (const p of positions) {
      const isHover = p.i === hover;
      const dim = hover >= 0 && !isHover && p.cat !== hoverCat;
      ctx.globalAlpha = dim ? 0.35 : 1;
      ctx.beginPath();
      ctx.arc(p.px, p.py, isHover ? 9 : 7, 0, Math.PI * 2);
      ctx.fillStyle = CAT_COLOR[p.cat];
      ctx.fill();
      ctx.lineWidth = isHover ? 2 : 1.2;
      ctx.strokeStyle = isHover ? colors.text : colors.pointStroke;
      ctx.stroke();

      if (showLabel[p.i] || isHover || p.cat === hoverCat) {
        ctx.globalAlpha = dim ? 0.5 : 1;
        ctx.fillStyle = colors.text;
        ctx.font = `${isHover ? "700" : "600"} 11px system-ui,sans-serif`;
        ctx.textAlign = "left";
        ctx.fillText(p.tok, p.px + 10, p.py + 4);
      }
    }
    ctx.globalAlpha = 1;
    ctx.restore();

    // Tooltip listing the hovered token plus any others stacked on top of it.
    if (hover >= 0) {
      const hp = positions[hover];
      const stacked = positions
        .filter(p => Math.hypot(p.px - hp.px, p.py - hp.py) < 12)
        .sort((a, b) => a.tok.localeCompare(b.tok));
      const lines = stacked.map(p => `${p.tok} · ${CATEGORY_LABEL[p.cat]}`);
      ctx.font = "11px system-ui,sans-serif";
      const tw = Math.max(...lines.map(l => ctx.measureText(l).width)) + 16;
      const th = lines.length * 15 + 10;
      let bx = hp.px + 14;
      let by = hp.py - th - 6;
      if (bx + tw > W - 4) bx = hp.px - tw - 14;
      if (by < 4) by = hp.py + 12;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx, by, tw, th, 6);
      else ctx.rect(bx, by, tw, th);
      ctx.fillStyle = colors.isDark
        ? "rgba(15,23,42,0.95)"
        : "rgba(255,255,255,0.97)";
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = colors.axis;
      ctx.stroke();
      ctx.textAlign = "left";
      for (let i = 0; i < stacked.length; i++) {
        ctx.fillStyle = CAT_COLOR[stacked[i].cat];
        ctx.beginPath();
        ctx.arc(bx + 8, by + 12 + i * 15, 3.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = colors.text;
        ctx.fillText(lines[i], bx + 16, by + 15 + i * 15);
      }
    }

    ctx.fillStyle = colors.text;
    ctx.font = "10px system-ui,sans-serif";
    ctx.textAlign = "right";
    ctx.globalAlpha = 0.65;
    ctx.fillText(`step ${step}`, W - m.r, m.t + 10);
    if (view.k > 1.01) ctx.fillText(`${view.k.toFixed(1)}×`, W - m.r, m.t + 24);
    ctx.globalAlpha = 1;
  }

  function drawLoss(svgSel, d3, losses, colors) {
    const W = 420;
    const H = 120;
    const m = { t: 8, r: 12, b: 28, l: 36 };
    svgSel.selectAll("*").remove();
    if (losses.length < 2) return;

    const maxL = Math.max(0.05, d3.max(losses) || 0.05);
    const x = d3.scaleLinear([0, losses.length - 1], [m.l, W - m.r]);
    const y = d3.scaleLinear([0, maxL], [H - m.b, m.t]);

    svgSel
      .append("g")
      .attr("transform", `translate(0,${H - m.b})`)
      .attr("color", colors.axis)
      .call(d3.axisBottom(x).ticks(4).tickSizeOuter(0))
      .selectAll("text")
      .attr("fill", colors.text)
      .style("font-size", "9px");

    svgSel
      .append("g")
      .attr("transform", `translate(${m.l},0)`)
      .attr("color", colors.axis)
      .call(d3.axisLeft(y).ticks(3).tickSizeOuter(0))
      .selectAll("text")
      .attr("fill", colors.text)
      .style("font-size", "9px");

    const line = d3
      .line()
      .x((d, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveMonotoneX);

    svgSel
      .append("path")
      .datum(losses)
      .attr("fill", "none")
      .attr("stroke", colors.accent)
      .attr("stroke-width", 1.6)
      .attr("d", line);
  }

  function renderNNList(container, model, token) {
    const listEl = container.querySelector('[data-out="nn"]');
    const idx = TOKEN_INDEX[token];
    const nn = nearestNeighbors(model, idx, 3);
    const cat = CATEGORY[token];
    listEl.innerHTML = nn
      .map(n => {
        const same = n.cat === cat;
        const mark = same ? "✓" : "·";
        const style = same
          ? "color:var(--accent);font-weight:600;"
          : "opacity:0.55;";
        return `<div style="${style}">${mark} ${n.tok} <span style="opacity:0.65;font-weight:400;">(${CATEGORY_LABEL[n.cat]})</span></div>`;
      })
      .join("");
  }

  /* ---------- widget ---------- */

  async function createWidget(container) {
    let d3;
    try {
      d3 = await loadD3();
    } catch {
      container.innerHTML =
        '<p style="opacity:0.7;font-style:italic;">Could not load the visualization library.</p>';
      return;
    }
    if (!document.body.contains(container)) return;

    const pairs = buildTrainingPairs();
    const state = {
      model: makeModel(),
      losses: [],
      step: 0,
      running: false,
      raf: null,
      focusToken: "cat",
      view: { k: 1, tx: 0, ty: 0 },
      hover: -1,
      lastPos: [],
      drag: null,
    };

    const styleBtn =
      "padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;";
    const styleGhost =
      "padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;";

    const legend = Object.entries(CATEGORY_LABEL)
      .filter(([k]) => k !== "meta")
      .map(
        ([k, label]) =>
          `<span style="display:inline-flex;align-items:center;gap:0.35rem;margin-right:0.85rem;">
            <span style="width:9px;height:9px;border-radius:50%;background:${CAT_COLOR[k]};display:inline-block;"></span>${label}
          </span>`
      )
      .join("");

    container.innerHTML = `
      <div class="embedding-viz" style="margin:1.5rem 0;font-family:system-ui,sans-serif;font-size:0.9rem;">
        <div style="padding:0.85rem 1rem;margin-bottom:1rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;line-height:1.55;opacity:0.9;">
          <div style="font-weight:600;margin-bottom:0.35rem;">Toy corpus (same-category tokens share continuations)</div>
          <div style="font-size:0.85rem;">
            the → <code style="opacity:0.9;">animal</code> → <code style="opacity:0.9;">verb</code> → <code style="opacity:0.9;">color</code> → <code style="opacity:0.9;">fruit</code> → the
          </div>
          <div style="font-size:0.8rem;margin-top:0.45rem;opacity:0.75;">
            e.g. “the cat sees red apple”, “the dog chases yellow mango” — each category shares one next-token distribution, and every category's is distinct.
          </div>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1.25rem;justify-content:center;align-items:flex-start;">
          <div style="flex:1 1 300px;max-width:440px;">
            <div style="font-weight:600;margin-bottom:0.35rem;text-align:center;">Learned embeddings (2D)</div>
            <canvas data-canvas="embed" width="420" height="340" style="width:100%;height:auto;border-radius:10px;border:1px solid var(--border,rgba(120,120,120,0.25));touch-action:none;cursor:grab;"></canvas>
            <div style="margin-top:0.35rem;font-size:0.72rem;opacity:0.7;text-align:center;">
              scroll to zoom · drag to pan · hover a point · double-click to reset view
            </div>
            <div style="margin-top:0.35rem;font-size:0.78rem;opacity:0.8;text-align:center;">${legend}</div>
            <svg data-svg="loss" viewBox="0 0 420 120" style="display:block;margin:0.75rem auto 0;width:100%;max-width:420px;height:auto;"></svg>
          </div>

          <div style="flex:1 1 220px;max-width:280px;padding:0.85rem 1rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;">
            <div style="font-weight:600;margin-bottom:0.5rem;">Nearest neighbors</div>
            <label style="display:flex;flex-direction:column;gap:0.3rem;margin-bottom:0.65rem;">
              <span style="opacity:0.8;font-size:0.82rem;">Focus token</span>
              <select data-ctl="focus" style="padding:0.35rem;border-radius:6px;background:transparent;color:inherit;border:1px solid var(--border,rgba(120,120,120,0.4));">
                ${TOKENS.filter(t => t !== "the")
                  .map(t => `<option value="${t}">${t}</option>`)
                  .join("")}
              </select>
            </label>
            <div data-out="nn" style="font-size:0.88rem;line-height:1.65;"></div>
            <div style="margin-top:0.75rem;font-size:0.78rem;opacity:0.7;line-height:1.45;">
              As training runs, same-category tokens should move together — and their nearest neighbors should match category, not surface form.
            </div>
          </div>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:center;justify-content:center;margin-top:1rem;">
          <button type="button" data-action="train" style="${styleBtn}">▶ Train</button>
          <button type="button" data-action="reset" style="${styleGhost}">↺ Reset</button>
          <span style="opacity:0.8;">step <b data-out="step">0</b></span>
          <span style="opacity:0.8;">loss <b data-out="loss">—</b></span>
        </div>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const embedCanvas = el('[data-canvas="embed"]');
    const trainBtn = el('[data-action="train"]');
    const embedCtx = makeContext(embedCanvas, 420, 340);
    const lossSvg = d3.select(el('[data-svg="loss"]'));

    function drawPlot() {
      const colors = getColors();
      drawEmbeddingPlot(
        embedCanvas,
        embedCtx.dpr,
        state.model,
        d3,
        colors,
        state.step,
        state.view,
        state.hover
      );
      state.lastPos = computePositions(state.model, d3, state.view);
    }

    function render() {
      const colors = getColors();
      drawPlot();
      drawLoss(lossSvg, d3, state.losses, colors);
      renderNNList(container, state.model, state.focusToken);
      el('[data-out="step"]').textContent = state.step;
      el('[data-out="loss"]').textContent =
        state.losses.length > 0
          ? state.losses[state.losses.length - 1].toFixed(3)
          : "—";
    }

    function resetModel() {
      state.model = makeModel();
      state.losses = [];
      state.step = 0;
      state.view = { k: 1, tx: 0, ty: 0 };
      state.hover = -1;
      render();
    }

    function stop() {
      state.running = false;
      if (state.raf) cancelAnimationFrame(state.raf);
      state.raf = null;
      trainBtn.textContent = "▶ Train";
    }

    function loop() {
      if (!state.running) return;
      if (!document.body.contains(container)) return stop();

      for (let k = 0; k < STEPS_PER_FRAME; k++) {
        if (state.step >= MAX_STEPS) {
          stop();
          break;
        }
        const loss = trainStep(state.model, pairs);
        if (state.step % 4 === 0) state.losses.push(loss);
        state.step++;
      }
      render();
      if (state.running) state.raf = requestAnimationFrame(loop);
    }

    function start() {
      if (state.running) return stop();
      if (state.step >= MAX_STEPS) resetModel();
      state.running = true;
      trainBtn.textContent = "⏸ Pause";
      state.raf = requestAnimationFrame(loop);
    }

    trainBtn.addEventListener("click", start);
    el('[data-action="reset"]').addEventListener("click", () => {
      stop();
      resetModel();
    });
    const focusSelect = el('[data-ctl="focus"]');
    focusSelect.addEventListener("change", e => {
      state.focusToken = e.target.value;
      render();
    });

    function toLocal(e) {
      const r = embedCanvas.getBoundingClientRect();
      return [
        (e.clientX - r.left) * (PLOT_W / r.width),
        (e.clientY - r.top) * (PLOT_H / r.height),
      ];
    }

    function pickNearest(mx, my) {
      let best = -1;
      let bd = Infinity;
      for (const p of state.lastPos) {
        const d = Math.hypot(p.px - mx, p.py - my);
        if (d < bd) {
          bd = d;
          best = p.i;
        }
      }
      return bd <= 14 ? best : -1;
    }

    embedCanvas.addEventListener("wheel", e => {
      e.preventDefault();
      const [mx, my] = toLocal(e);
      const factor = Math.exp(-e.deltaY * 0.0015);
      const kNew = Math.min(40, Math.max(1, state.view.k * factor));
      const ratio = kNew / state.view.k;
      state.view.tx = mx - ratio * (mx - state.view.tx);
      state.view.ty = my - ratio * (my - state.view.ty);
      state.view.k = kNew;
      if (kNew === 1) {
        state.view.tx = 0;
        state.view.ty = 0;
      }
      if (!state.running) drawPlot();
    });

    embedCanvas.addEventListener("mousedown", e => {
      const [mx, my] = toLocal(e);
      state.drag = { x: mx, y: my, tx: state.view.tx, ty: state.view.ty };
      embedCanvas.style.cursor = "grabbing";
    });

    window.addEventListener("mouseup", () => {
      if (state.drag) {
        state.drag = null;
        embedCanvas.style.cursor = "grab";
      }
    });

    embedCanvas.addEventListener("mousemove", e => {
      const [mx, my] = toLocal(e);
      if (state.drag) {
        state.view.tx = state.drag.tx + (mx - state.drag.x);
        state.view.ty = state.drag.ty + (my - state.drag.y);
        if (!state.running) drawPlot();
        return;
      }
      const hit = pickNearest(mx, my);
      embedCanvas.style.cursor = hit >= 0 ? "pointer" : "grab";
      if (hit !== state.hover) {
        state.hover = hit;
        if (!state.running) drawPlot();
      }
    });

    embedCanvas.addEventListener("mouseleave", () => {
      if (state.hover !== -1) {
        state.hover = -1;
        if (!state.running) drawPlot();
      }
    });

    embedCanvas.addEventListener("click", e => {
      const [mx, my] = toLocal(e);
      const hit = pickNearest(mx, my);
      if (hit >= 0 && TOKENS[hit] !== "the") {
        state.focusToken = TOKENS[hit];
        focusSelect.value = state.focusToken;
        render();
      }
    });

    embedCanvas.addEventListener("dblclick", () => {
      state.view = { k: 1, tx: 0, ty: 0 };
      if (!state.running) drawPlot();
    });

    const themeObserver = new MutationObserver(() => {
      if (!document.body.contains(container)) {
        themeObserver.disconnect();
        return;
      }
      if (!state.running) render();
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    render();
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document
      .querySelectorAll(
        '.viz-embedding-clustering[data-viz="embedding-clustering"]'
      )
      .forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
