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

  function drawEmbeddingPlot(canvas, dpr, model, d3, colors, step) {
    const W = 420;
    const H = 340;
    const m = { t: 18, r: 18, b: 36, l: 36 };
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const [x0, x1, y0, y1] = embeddingExtents(model);
    const x = d3.scaleLinear([x0, x1], [m.l, W - m.r]);
    const y = d3.scaleLinear([y0, y1], [H - m.b, m.t]);

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

    for (let i = 0; i < TOKENS.length; i++) {
      const tok = TOKENS[i];
      const px = x(model.embed[i][0]);
      const py = y(model.embed[i][1]);
      const cat = CATEGORY[tok];
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.fillStyle = CAT_COLOR[cat];
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = colors.pointStroke;
      ctx.stroke();
      ctx.fillStyle = colors.text;
      ctx.font = "600 11px system-ui,sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(tok, px + 9, py + 4);
    }

    ctx.fillStyle = colors.text;
    ctx.font = "10px system-ui,sans-serif";
    ctx.textAlign = "right";
    ctx.globalAlpha = 0.65;
    ctx.fillText(`step ${step}`, W - m.r, m.t + 10);
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
            Animals → <code style="opacity:0.9;">eat · chase · see</code> &nbsp;·&nbsp;
            Fruits → <code style="opacity:0.9;">red · yellow</code> &nbsp;·&nbsp;
            Verbs → <code style="opacity:0.9;">the</code>
          </div>
          <div style="font-size:0.8rem;margin-top:0.45rem;opacity:0.75;">
            e.g. “the cat eats”, “the dog chases”, “the apple red”, “the mango yellow”
          </div>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1.25rem;justify-content:center;align-items:flex-start;">
          <div style="flex:1 1 300px;max-width:440px;">
            <div style="font-weight:600;margin-bottom:0.35rem;text-align:center;">Learned embeddings (2D)</div>
            <canvas data-canvas="embed" width="420" height="340" style="width:100%;height:auto;border-radius:10px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
            <div style="margin-top:0.5rem;font-size:0.78rem;opacity:0.8;text-align:center;">${legend}</div>
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

    function render() {
      const colors = getColors();
      drawEmbeddingPlot(
        embedCanvas,
        embedCtx.dpr,
        state.model,
        d3,
        colors,
        state.step
      );
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
    el('[data-ctl="focus"]').addEventListener("change", e => {
      state.focusToken = e.target.value;
      render();
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
