(function () {
  const BOUNDS = 5;
  const GRID = 64;
  const MAX_EPOCHS = 600;
  const STEPS_PER_FRAME = 3;
  const D3_URL = "https://cdn.jsdelivr.net/npm/d3@7/+esm";
  const LAYER_DIMS = [2, 8, 8, 8, 8, 1];

  const initialized = new WeakSet();
  let d3Promise = null;

  function loadD3() {
    if (!d3Promise) {
      d3Promise = import(/* @vite-ignore */ D3_URL);
    }
    return d3Promise;
  }

  /* ---------- math helpers ---------- */

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  }

  const relu = z => (z > 0 ? z : 0);

  function randn() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const clamp = p => Math.min(1 - 1e-7, Math.max(1e-7, p));

  function matVec(W, v) {
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      let s = 0;
      for (let j = 0; j < v.length; j++) s += W[i][j] * v[j];
      out[i] = s;
    }
    return out;
  }

  function matMul(A, B) {
    const rows = A.length;
    const cols = B[0].length;
    const inner = B.length;
    const out = Array.from({ length: rows }, () => new Array(cols).fill(0));
    for (let i = 0; i < rows; i++) {
      for (let k = 0; k < inner; k++) {
        const aik = A[i][k];
        for (let j = 0; j < cols; j++) out[i][j] += aik * B[k][j];
      }
    }
    return out;
  }

  function vecAdd(a, b) {
    return a.map((v, i) => v + b[i]);
  }

  /* ---------- dataset (rings only) ---------- */

  function generateRings(noise, nPerClass = 150) {
    const pts = [];
    for (let c = 0; c < 2; c++) {
      const radius = c === 0 ? 1.5 : 3.4;
      for (let i = 0; i < nPerClass; i++) {
        const a = Math.random() * Math.PI * 2;
        const r = radius + randn() * (0.28 + noise * 0.6);
        pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: c });
      }
    }
    return pts;
  }

  /* ---------- model A: single linear layer ---------- */

  function makeLinear() {
    return { w0: randn() * 0.1, w1: randn() * 0.1, b: 0 };
  }

  const predictLinear = (m, x, y) => sigmoid(m.w0 * x + m.w1 * y + m.b);

  function stepLinear(m, pts, lr) {
    let gw0 = 0;
    let gw1 = 0;
    let gb = 0;
    let loss = 0;
    for (const p of pts) {
      const yhat = predictLinear(m, p.x, p.y);
      const err = yhat - p.label;
      gw0 += err * p.x;
      gw1 += err * p.y;
      gb += err;
      const pc = clamp(yhat);
      loss += -(p.label * Math.log(pc) + (1 - p.label) * Math.log(1 - pc));
    }
    const n = pts.length;
    m.w0 -= (lr * gw0) / n;
    m.w1 -= (lr * gw1) / n;
    m.b -= (lr * gb) / n;
    return loss / n;
  }

  /* ---------- models B/C: 5-layer stacks ---------- */

  function makeDeepStack(dims) {
    const weights = [];
    const biases = [];
    for (let i = 0; i < dims.length - 1; i++) {
      const out = dims[i + 1];
      const inn = dims[i];
      weights.push(
        Array.from({ length: out }, () =>
          Array.from({ length: inn }, () => randn() * 0.1)
        )
      );
      biases.push(new Array(out).fill(0));
    }
    return { weights, biases, dims };
  }

  function forwardDeep(m, x, y, useRelu) {
    const acts = [[x, y]];
    const pres = [];
    let a = [x, y];
    const n = m.weights.length;
    for (let i = 0; i < n; i++) {
      const pre = vecAdd(matVec(m.weights[i], a), m.biases[i]);
      pres.push(pre);
      a = pre;
      if (useRelu && i < n - 1) a = a.map(relu);
      acts.push(a);
    }
    return { yhat: sigmoid(a[0]), acts, pres };
  }

  function predictDeep(m, x, y, useRelu) {
    return forwardDeep(m, x, y, useRelu).yhat;
  }

  function stepDeep(m, pts, lr, useRelu) {
    const nLayers = m.weights.length;
    const gw = m.weights.map(W => W.map(row => row.map(() => 0)));
    const gb = m.biases.map(b => b.map(() => 0));
    let loss = 0;

    for (const p of pts) {
      const { yhat, acts, pres } = forwardDeep(m, p.x, p.y, useRelu);
      const err = yhat - p.label;
      let delta = [err];

      for (let i = nLayers - 1; i >= 0; i--) {
        const inp = acts[i];
        for (let r = 0; r < delta.length; r++) {
          gb[i][r] += delta[r];
          for (let c = 0; c < inp.length; c++) gw[i][r][c] += delta[r] * inp[c];
        }
        if (i === 0) break;
        const next = new Array(m.weights[i - 1][0].length).fill(0);
        for (let r = 0; r < delta.length; r++) {
          for (let c = 0; c < next.length; c++) {
            next[c] += m.weights[i][r][c] * delta[r];
          }
        }
        if (useRelu) {
          for (let c = 0; c < next.length; c++) {
            next[c] *= pres[i - 1][c] > 0 ? 1 : 0;
          }
        }
        delta = next;
      }

      const pc = clamp(yhat);
      loss += -(p.label * Math.log(pc) + (1 - p.label) * Math.log(1 - pc));
    }

    const n = pts.length;
    for (let i = 0; i < nLayers; i++) {
      for (let r = 0; r < m.weights[i].length; r++) {
        for (let c = 0; c < m.weights[i][r].length; c++) {
          m.weights[i][r][c] -= (lr * gw[i][r][c]) / n;
        }
        m.biases[i][r] -= (lr * gb[i][r]) / n;
      }
    }
    return loss / n;
  }

  /** Multiply B's five weight matrices into one 2→1 linear map. */
  function collapseDeepLinear(m) {
    let W = m.weights[0].map(row => row.slice());
    let b = m.biases[0].slice();
    for (let i = 1; i < m.weights.length; i++) {
      b = vecAdd(matVec(m.weights[i], b), m.biases[i]);
      W = matMul(m.weights[i], W);
    }
    return { W, b: b[0] };
  }

  function accuracy(pts, predict) {
    let correct = 0;
    for (const p of pts) {
      if ((predict(p.x, p.y) >= 0.5 ? 1 : 0) === p.label) correct++;
    }
    return (correct / pts.length) * 100;
  }

  /* ---------- theme ---------- */

  const C0 = "#f97316";
  const C1 = "#3b82f6";
  const CA = "#f97316";
  const CB = "#64748b";
  const CC = "#3b82f6";

  function getColors() {
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    const styles = getComputedStyle(document.documentElement);
    const accent =
      styles.getPropertyValue("--accent").trim() ||
      (isDark ? "#60a5fa" : "#3b82f6");
    return {
      isDark,
      accent,
      field: isDark
        ? ["#fb923c", "#7c2d12", "#0b1220", "#1e3a8a", "#60a5fa"]
        : ["#f97316", "#fed7aa", "#f8fafc", "#bfdbfe", "#3b82f6"],
      boundary: isDark ? "#f8fafc" : "#0f172a",
      pointStroke: isDark ? "rgba(2,6,23,0.75)" : "rgba(255,255,255,0.9)",
      text: styles.getPropertyValue("--foreground").trim() || "#334155",
      axis: isDark ? "rgba(148,163,184,0.7)" : "rgba(100,116,139,0.7)",
    };
  }

  /* ---------- d3 rendering ---------- */

  function computeField(predict) {
    const values = new Float64Array(GRID * GRID);
    for (let j = 0; j < GRID; j++) {
      const y = BOUNDS - (j / (GRID - 1)) * 2 * BOUNDS;
      for (let i = 0; i < GRID; i++) {
        const x = -BOUNDS + (i / (GRID - 1)) * 2 * BOUNDS;
        values[j * GRID + i] = predict(x, y);
      }
    }
    return values;
  }

  function makeContext(canvas, size) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    return { ctx, dpr };
  }

  function clipRoundedRect(ctx, size, radius, inset) {
    const x = inset;
    const w = size - 2 * inset;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(x, x, w, w, radius);
    } else {
      ctx.rect(x, x, w, w);
    }
    ctx.clip();
  }

  function drawPlotD3(canvas, dpr, predict, pts, d3, colors) {
    const size = 300;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    ctx.save();
    clipRoundedRect(ctx, size, 10, 1.5);

    const color = d3
      .scaleLinear()
      .domain([0, 0.25, 0.5, 0.75, 1])
      .range(colors.field)
      .interpolate(d3.interpolateRgb)
      .clamp(true);

    const values = computeField(predict);
    const thresholds = d3.range(0.05, 1, 1 / 15);
    const contours = d3.contours().size([GRID, GRID]).smooth(true);

    const scale = size / GRID;
    ctx.fillStyle = color(0);
    ctx.fillRect(0, 0, size, size);

    const path = d3.geoPath(null, ctx);
    ctx.save();
    ctx.scale(scale, scale);
    for (const t of thresholds) {
      const geo = contours.contour(values, t);
      ctx.beginPath();
      path(geo);
      ctx.fillStyle = color(t);
      ctx.fill();
    }
    ctx.restore();

    ctx.save();
    ctx.scale(scale, scale);
    const boundary = contours.contour(values, 0.5);
    ctx.beginPath();
    path(boundary);
    ctx.restore();
    ctx.lineJoin = "round";
    ctx.lineWidth = 2.25;
    ctx.strokeStyle = colors.boundary;
    ctx.stroke();

    const toX = x => ((x + BOUNDS) / (2 * BOUNDS)) * size;
    const toY = y => ((BOUNDS - y) / (2 * BOUNDS)) * size;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(toX(p.x), toY(p.y), 3, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 0 ? C0 : C1;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = colors.pointStroke;
      ctx.stroke();
    }

    ctx.restore();
  }

  function fmt(n) {
    const abs = Math.abs(n);
    if (abs >= 100 || (abs > 0 && abs < 1e-3)) return n.toExponential(3);
    return n.toFixed(4);
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

    const state = {
      noise: 0.25,
      lr: 0.5,
      pts: [],
      linear: null,
      deepLin: null,
      deepRelu: null,
      collapsed: null,
      epoch: 0,
      running: false,
      raf: null,
    };

    const styleBtn =
      "padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;";
    const styleGhost =
      "padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;";

    container.innerHTML = `
      <div class="linear-collapse-viz" style="margin:1.5rem 0;font-family:system-ui,sans-serif;font-size:0.9rem;">
        <div style="display:flex;flex-wrap:wrap;gap:1rem 1.5rem;align-items:flex-end;margin-bottom:1rem;padding:0.85rem 1rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;">
          <label style="display:flex;flex-direction:column;gap:0.25rem;min-width:140px;">
            <span style="opacity:0.8;">Noise: <b data-out="noise">0.25</b></span>
            <input data-ctl="noise" type="range" min="0" max="0.6" step="0.02" value="0.25" style="accent-color:var(--accent);" />
          </label>
          <label style="display:flex;flex-direction:column;gap:0.25rem;min-width:140px;">
            <span style="opacity:0.8;">Learning rate: <b data-out="lr">0.50</b></span>
            <input data-ctl="lr" type="range" min="0.05" max="1.5" step="0.05" value="0.5" style="accent-color:var(--accent);" />
          </label>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;justify-content:center;margin-bottom:0.75rem;">
          <div style="text-align:center;flex:1 1 220px;max-width:280px;">
            <div style="font-weight:600;margin-bottom:0.3rem;">A · 1 linear layer</div>
            <canvas data-canvas="a" width="300" height="300" style="width:100%;height:auto;border-radius:10px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
            <div style="margin-top:0.35rem;">train accuracy: <b data-acc="a" style="color:${CA};">—</b></div>
          </div>
          <div style="text-align:center;flex:1 1 220px;max-width:280px;">
            <div style="font-weight:600;margin-bottom:0.3rem;">B · 5 linear layers</div>
            <canvas data-canvas="b" width="300" height="300" style="width:100%;height:auto;border-radius:10px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
            <div style="margin-top:0.35rem;">train accuracy: <b data-acc="b" style="color:${CB};">—</b></div>
          </div>
          <div style="text-align:center;flex:1 1 220px;max-width:280px;">
            <div style="font-weight:600;margin-bottom:0.3rem;">C · 5 layers + ReLU</div>
            <canvas data-canvas="c" width="300" height="300" style="width:100%;height:auto;border-radius:10px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
            <div style="margin-top:0.35rem;">train accuracy: <b data-acc="c" style="color:${CC};">—</b></div>
          </div>
        </div>

        <div style="margin:0.75rem auto 1rem;max-width:640px;padding:0.85rem 1rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;">
          <div style="font-weight:600;margin-bottom:0.5rem;">Bonus · B collapses to one linear map</div>
          <p style="opacity:0.8;font-size:0.85rem;margin:0 0 0.6rem;line-height:1.45;">
            Multiply B's five weight matrices (2→8→8→8→8→1, no activations). The stack is exactly one matrix and one bias — same class of function as A.
          </p>
          <pre data-out="collapse" style="margin:0;padding:0.75rem;border-radius:8px;background:var(--border,rgba(120,120,120,0.12));font-size:0.78rem;line-height:1.5;overflow-x:auto;white-space:pre;">—</pre>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:center;justify-content:center;margin-bottom:0.5rem;">
          <button type="button" data-action="train" style="${styleBtn}">▶ Train all three</button>
          <button type="button" data-action="reset" style="${styleGhost}">↺ Reset weights</button>
          <button type="button" data-action="newdata" style="${styleGhost}">⤺ New data</button>
          <span style="opacity:0.8;">epoch <b data-out="epoch">0</b></span>
        </div>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const canvasA = el('[data-canvas="a"]');
    const canvasB = el('[data-canvas="b"]');
    const canvasC = el('[data-canvas="c"]');
    const trainBtn = el('[data-action="train"]');

    const ctxA = makeContext(canvasA, 300);
    const ctxB = makeContext(canvasB, 300);
    const ctxC = makeContext(canvasC, 300);

    function updateCollapse() {
      const { W, b } = collapseDeepLinear(state.deepLin);
      state.collapsed = { w0: W[0][0], w1: W[0][1], b };
      el("[data-out=collapse]").textContent =
        `W_eff = [ ${fmt(W[0][0])}   ${fmt(W[0][1])} ]\n` +
        `b_eff = ${fmt(b)}\n\n` +
        `σ(${fmt(W[0][0])}·x + ${fmt(W[0][1])}·y + ${fmt(b)})`;
    }

    function render() {
      const colors = getColors();
      const predA = (x, y) => predictLinear(state.linear, x, y);
      const predB = (x, y) => predictDeep(state.deepLin, x, y, false);
      const predC = (x, y) => predictDeep(state.deepRelu, x, y, true);

      drawPlotD3(canvasA, ctxA.dpr, predA, state.pts, d3, colors);
      drawPlotD3(canvasB, ctxB.dpr, predB, state.pts, d3, colors);
      drawPlotD3(canvasC, ctxC.dpr, predC, state.pts, d3, colors);

      el('[data-acc="a"]').textContent =
        accuracy(state.pts, predA).toFixed(1) + "%";
      el('[data-acc="b"]').textContent =
        accuracy(state.pts, predB).toFixed(1) + "%";
      el('[data-acc="c"]').textContent =
        accuracy(state.pts, predC).toFixed(1) + "%";
      el('[data-out="epoch"]').textContent = state.epoch;
      updateCollapse();
    }

    function resetWeights() {
      state.linear = makeLinear();
      state.deepLin = makeDeepStack(LAYER_DIMS);
      state.deepRelu = makeDeepStack(LAYER_DIMS);
      state.epoch = 0;
      render();
    }

    function newData() {
      state.pts = generateRings(state.noise);
      resetWeights();
    }

    function stop() {
      state.running = false;
      if (state.raf) cancelAnimationFrame(state.raf);
      state.raf = null;
      trainBtn.textContent = "▶ Train all three";
    }

    function loop() {
      if (!state.running) return;
      if (!document.body.contains(container)) return stop();
      for (let k = 0; k < STEPS_PER_FRAME; k++) {
        if (state.epoch >= MAX_EPOCHS) {
          stop();
          break;
        }
        stepLinear(state.linear, state.pts, state.lr);
        stepDeep(state.deepLin, state.pts, state.lr, false);
        stepDeep(state.deepRelu, state.pts, state.lr, true);
        state.epoch++;
      }
      render();
      if (state.running) state.raf = requestAnimationFrame(loop);
    }

    function start() {
      if (state.running) return stop();
      if (state.epoch >= MAX_EPOCHS) resetWeights();
      state.running = true;
      trainBtn.textContent = "⏸ Pause";
      state.raf = requestAnimationFrame(loop);
    }

    el('[data-ctl="noise"]').addEventListener("input", e => {
      state.noise = parseFloat(e.target.value);
      el('[data-out="noise"]').textContent = state.noise.toFixed(2);
      stop();
      newData();
    });
    el('[data-ctl="lr"]').addEventListener("input", e => {
      state.lr = parseFloat(e.target.value);
      el('[data-out="lr"]').textContent = state.lr.toFixed(2);
    });

    trainBtn.addEventListener("click", start);
    el('[data-action="reset"]').addEventListener("click", () => {
      stop();
      resetWeights();
    });
    el('[data-action="newdata"]').addEventListener("click", () => {
      stop();
      newData();
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

    newData();
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document.querySelectorAll(".viz-linear-collapse").forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
