(function () {
  const BOUNDS = 5;
  const GRID = 64;
  const DEFAULT_MAX_EPOCHS = 500;
  const STEPS_PER_FRAME = 4;
  const D3_URL = "https://cdn.jsdelivr.net/npm/d3@7/+esm";

  const initialized = new WeakSet();
  let d3Promise = null;

  function loadD3() {
    if (!d3Promise) {
      d3Promise = import(/* @vite-ignore */ D3_URL);
    }
    return d3Promise;
  }

  /* ---------- math ---------- */

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  }

  const relu = z => (z > 0 ? z : 0);
  const clamp = p => Math.min(1 - 1e-7, Math.max(1e-7, p));

  function randn() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function generateRings(noise, nPerClass = 140) {
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

  function makeModel(hidden) {
    return {
      hidden,
      w1: Array.from({ length: hidden }, () => [randn() * 0.9, randn() * 0.9]),
      b1: Array.from({ length: hidden }, () => randn() * 0.3),
      w2: Array.from({ length: hidden }, () => randn() * 0.9),
      b2: 0,
      // stable identity per neuron so the UI can show them physically swapping
      id: Array.from({ length: hidden }, (_, i) => i),
    };
  }

  function predict(m, x, y) {
    let z = m.b2;
    for (let j = 0; j < m.hidden; j++) {
      z += m.w2[j] * relu(m.w1[j][0] * x + m.w1[j][1] * y + m.b1[j]);
    }
    return sigmoid(z);
  }

  function stepModel(m, pts, lr) {
    const hidden = m.hidden;
    const gw1 = m.w1.map(() => [0, 0]);
    const gb1 = new Array(hidden).fill(0);
    const gw2 = new Array(hidden).fill(0);
    let gb2 = 0;
    let loss = 0;
    const hPre = new Array(hidden);
    const h = new Array(hidden);

    for (const p of pts) {
      for (let j = 0; j < hidden; j++) {
        hPre[j] = m.w1[j][0] * p.x + m.w1[j][1] * p.y + m.b1[j];
        h[j] = hPre[j] > 0 ? hPre[j] : 0;
      }
      let z = m.b2;
      for (let j = 0; j < hidden; j++) z += m.w2[j] * h[j];
      const yhat = sigmoid(z);
      const err = yhat - p.label;

      gb2 += err;
      for (let j = 0; j < hidden; j++) {
        gw2[j] += err * h[j];
        const dh = err * m.w2[j] * (hPre[j] > 0 ? 1 : 0);
        gw1[j][0] += dh * p.x;
        gw1[j][1] += dh * p.y;
        gb1[j] += dh;
      }
      const pc = clamp(yhat);
      loss += -(p.label * Math.log(pc) + (1 - p.label) * Math.log(1 - pc));
    }

    const n = pts.length;
    for (let j = 0; j < hidden; j++) {
      m.w1[j][0] -= (lr * gw1[j][0]) / n;
      m.w1[j][1] -= (lr * gw1[j][1]) / n;
      m.b1[j] -= (lr * gb1[j]) / n;
      m.w2[j] -= (lr * gw2[j]) / n;
    }
    m.b2 -= (lr * gb2) / n;
    return loss / n;
  }

  function accuracy(pts, m) {
    let correct = 0;
    for (const p of pts) {
      if ((predict(m, p.x, p.y) >= 0.5 ? 1 : 0) === p.label) correct++;
    }
    return (correct / pts.length) * 100;
  }

  function computeField(m) {
    const values = new Float64Array(GRID * GRID);
    for (let j = 0; j < GRID; j++) {
      const y = BOUNDS - (j / (GRID - 1)) * 2 * BOUNDS;
      for (let i = 0; i < GRID; i++) {
        const x = -BOUNDS + (i / (GRID - 1)) * 2 * BOUNDS;
        values[j * GRID + i] = predict(m, x, y);
      }
    }
    return values;
  }

  function fieldDrift(a, b) {
    let max = 0;
    for (let i = 0; i < a.length; i++) {
      const d = Math.abs(a[i] - b[i]);
      if (d > max) max = d;
    }
    return max;
  }

  /* ---------- symmetry operations ---------- */

  function permute(m, perm) {
    const reorder = arr => perm.map(p => arr[p]);
    m.w1 = reorder(m.w1);
    m.b1 = reorder(m.b1);
    m.w2 = reorder(m.w2);
    m.id = reorder(m.id);
  }

  function randomPerm(n) {
    const p = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [p[i], p[j]] = [p[j], p[i]];
    }
    return p;
  }

  // ReLU is positively homogeneous: ReLU(c*z) = c*ReLU(z) for c > 0.
  // Scale a unit's incoming weights+bias by c and its outgoing weight by 1/c
  // and the network's function is unchanged.
  function scaleUnit(m, j, c) {
    m.w1[j][0] *= c;
    m.w1[j][1] *= c;
    m.b1[j] *= c;
    m.w2[j] /= c;
  }

  function factorial(n) {
    let f = 1;
    for (let i = 2; i <= n; i++) f *= i;
    return f;
  }

  /* ---------- theme ---------- */

  const C0 = "#f97316";
  const C1 = "#3b82f6";

  function getColors() {
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    const styles = getComputedStyle(document.documentElement);
    return {
      isDark,
      field: isDark
        ? ["#fb923c", "#7c2d12", "#0b1220", "#1e3a8a", "#60a5fa"]
        : ["#f97316", "#fed7aa", "#f8fafc", "#bfdbfe", "#3b82f6"],
      boundary: isDark ? "#f8fafc" : "#0f172a",
      pointStroke: isDark ? "rgba(2,6,23,0.75)" : "rgba(255,255,255,0.9)",
      text: styles.getPropertyValue("--foreground").trim() || "#334155",
      axis: isDark ? "rgba(148,163,184,0.7)" : "rgba(100,116,139,0.7)",
      grid: isDark ? "rgba(148,163,184,0.14)" : "rgba(100,116,139,0.14)",
      accent:
        styles.getPropertyValue("--accent").trim() ||
        (isDark ? "#60a5fa" : "#2563eb"),
    };
  }

  const neuronHue = (id, total) => Math.round((id / Math.max(total, 1)) * 320);

  /* ---------- rendering ---------- */

  function makeContext(canvas, w, h) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    return { ctx: canvas.getContext("2d"), dpr };
  }

  function clipRoundedRect(ctx, size, radius, inset) {
    ctx.beginPath();
    if (ctx.roundRect)
      ctx.roundRect(inset, inset, size - 2 * inset, size - 2 * inset, radius);
    else ctx.rect(inset, inset, size - 2 * inset, size - 2 * inset);
    ctx.clip();
  }

  function drawBoundary(canvas, dpr, m, pts, d3, colors) {
    const size = 320;
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

    const values = computeField(m);
    const thresholds = d3.range(0.05, 1, 1 / 15);
    const contours = d3.contours().size([GRID, GRID]).smooth(true);
    const scale = size / GRID;

    ctx.fillStyle = color(0);
    ctx.fillRect(0, 0, size, size);

    const path = d3.geoPath(null, ctx);
    ctx.save();
    ctx.scale(scale, scale);
    for (const t of thresholds) {
      ctx.beginPath();
      path(contours.contour(values, t));
      ctx.fillStyle = color(t);
      ctx.fill();
    }
    ctx.restore();

    ctx.save();
    ctx.scale(scale, scale);
    ctx.beginPath();
    path(contours.contour(values, 0.5));
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

  function renderNeurons(container, m, selected) {
    const wrap = container.querySelector('[data-out="neurons"]');
    wrap.innerHTML = m.id
      .map((id, pos) => {
        const hue = neuronHue(id, m.hidden);
        const sel = pos === selected;
        const win = `[${m.w1[pos][0].toFixed(2)}, ${m.w1[pos][1].toFixed(2)}]`;
        return `
          <button type="button" data-neuron="${pos}" title="click to select for scaling"
            style="flex:0 0 auto;text-align:left;cursor:pointer;padding:0.4rem 0.55rem;border-radius:8px;
            border:2px solid ${sel ? `hsl(${hue} 70% 50%)` : "var(--border,rgba(120,120,120,0.3))"};
            background:hsl(${hue} 70% 50% / ${sel ? "0.22" : "0.10"});font-size:0.72rem;line-height:1.35;min-width:96px;">
            <div style="font-weight:700;display:flex;align-items:center;gap:0.35rem;">
              <span style="width:9px;height:9px;border-radius:50%;background:hsl(${hue} 70% 50%);display:inline-block;"></span>
              unit ${id}
            </div>
            <div style="opacity:0.8;">w<sub>in</sub> ${win}</div>
            <div style="opacity:0.8;">w<sub>out</sub> ${m.w2[pos].toFixed(2)}</div>
          </button>`;
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

    const state = {
      hidden: 6,
      noise: 0.25,
      lr: 0.6,
      maxEpochs: DEFAULT_MAX_EPOCHS,
      pts: [],
      model: null,
      snapshot: null,
      epoch: 0,
      running: false,
      raf: null,
      selected: 0,
      scaleVal: 1,
    };

    const styleBtn =
      "padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;";
    const styleGhost =
      "padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;";

    container.innerHTML = `
      <div class="weight-symmetry-viz" style="margin:1.5rem 0;font-family:system-ui,sans-serif;font-size:0.9rem;">
        <div style="display:flex;flex-wrap:wrap;gap:1rem 1.5rem;align-items:flex-end;margin-bottom:1rem;padding:0.85rem 1rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;">
          <label style="display:flex;flex-direction:column;gap:0.25rem;min-width:150px;">
            <span style="opacity:0.8;">Hidden ReLU units: <b data-out="hidden">6</b></span>
            <input data-ctl="hidden" type="range" min="3" max="10" step="1" value="6" style="accent-color:var(--accent);" />
          </label>
          <label style="display:flex;flex-direction:column;gap:0.25rem;min-width:150px;">
            <span style="opacity:0.8;">Epochs: <b data-out="maxepochs">500</b></span>
            <input data-ctl="maxepochs" type="range" min="100" max="1500" step="50" value="500" style="accent-color:var(--accent);" />
          </label>
          <span style="opacity:0.75;font-size:0.85rem;">permutation copies of this minimum: <b data-out="copies">720</b></span>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1.25rem;justify-content:center;align-items:flex-start;">
          <div style="flex:1 1 300px;max-width:360px;text-align:center;">
            <div style="font-weight:600;margin-bottom:0.35rem;">Decision boundary</div>
            <canvas data-canvas="boundary" width="320" height="320" style="width:100%;height:auto;border-radius:10px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
            <div style="margin-top:0.4rem;font-size:0.85rem;">
              train accuracy: <b data-out="acc">—</b>
            </div>
            <div style="margin-top:0.15rem;font-size:0.85rem;">
              function drift vs original: <b data-out="drift" style="font-variant-numeric:tabular-nums;">—</b>
            </div>
          </div>

          <div style="flex:1 1 300px;max-width:380px;">
            <div style="font-weight:600;margin-bottom:0.35rem;">Hidden neurons (identity = colour)</div>
            <div data-out="neurons" style="display:flex;flex-wrap:wrap;gap:0.45rem;margin-bottom:0.75rem;"></div>

            <button type="button" data-action="shuffle" style="${styleGhost}">⇄ Permute neuron order</button>

            <div style="margin-top:0.85rem;padding:0.75rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;">
              <div style="font-size:0.85rem;margin-bottom:0.3rem;">
                Scale selected unit — <b data-out="selname">unit 0</b>
              </div>
              <label style="display:flex;flex-direction:column;gap:0.25rem;">
                <span style="opacity:0.8;font-size:0.82rem;">c = <b data-out="scale">1.00</b> &nbsp;<span style="opacity:0.65;">(w<sub>in</sub> ×c, w<sub>out</sub> ×1/c)</span></span>
                <input data-ctl="scale" type="range" min="0.25" max="4" step="0.05" value="1" style="accent-color:var(--accent);" />
              </label>
              <div style="margin-top:0.4rem;font-size:0.78rem;opacity:0.7;line-height:1.4;">
                ReLU is positively homogeneous, so this rescaling leaves the output untouched — watch the boundary and drift stay put.
              </div>
            </div>
          </div>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:center;justify-content:center;margin:1rem 0 0.25rem;">
          <button type="button" data-action="train" style="${styleBtn}">▶ Retrain from scratch</button>
          <span style="opacity:0.8;">epoch <b data-out="epoch">0</b> / <b data-out="maxepochs-live">500</b></span>
        </div>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const boundaryCanvas = el('[data-canvas="boundary"]');
    const trainBtn = el('[data-action="train"]');
    const boundaryCtx = makeContext(boundaryCanvas, 320, 320);

    function updateReadouts() {
      const colors = getColors();
      drawBoundary(
        boundaryCanvas,
        boundaryCtx.dpr,
        state.model,
        state.pts,
        d3,
        colors
      );
      renderNeurons(container, state.model, state.selected);
      el('[data-out="acc"]').textContent =
        accuracy(state.pts, state.model).toFixed(1) + "%";
      const drift = state.snapshot
        ? fieldDrift(computeField(state.model), state.snapshot)
        : 0;
      el('[data-out="drift"]').textContent = drift.toExponential(2);
      el('[data-out="selname"]').textContent =
        `unit ${state.model.id[state.selected]}`;
      el('[data-out="copies"]').textContent = factorial(
        state.hidden
      ).toLocaleString();
      el('[data-out="epoch"]').textContent = state.epoch;
      el('[data-out="maxepochs-live"]').textContent = state.maxEpochs;
    }

    function newData() {
      state.pts = generateRings(state.noise);
    }

    function resetModel() {
      state.model = makeModel(state.hidden);
      state.snapshot = null;
      state.epoch = 0;
      state.selected = 0;
      state.scaleVal = 1;
      el('[data-ctl="scale"]').value = "1";
      el('[data-out="scale"]').textContent = "1.00";
    }

    function stop() {
      state.running = false;
      if (state.raf) cancelAnimationFrame(state.raf);
      state.raf = null;
      trainBtn.textContent = "▶ Retrain from scratch";
    }

    function loop() {
      if (!state.running) return;
      if (!document.body.contains(container)) return stop();
      for (let k = 0; k < STEPS_PER_FRAME; k++) {
        if (state.epoch >= state.maxEpochs) {
          state.snapshot = computeField(state.model);
          stop();
          break;
        }
        stepModel(state.model, state.pts, state.lr);
        state.epoch++;
      }
      updateReadouts();
      if (state.running) state.raf = requestAnimationFrame(loop);
    }

    function train() {
      if (state.running) return stop();
      newData();
      resetModel();
      state.running = true;
      trainBtn.textContent = "⏸ Pause";
      state.raf = requestAnimationFrame(loop);
    }

    el('[data-ctl="hidden"]').addEventListener("input", e => {
      state.hidden = parseInt(e.target.value, 10);
      el('[data-out="hidden"]').textContent = state.hidden;
      stop();
      newData();
      resetModel();
      updateReadouts();
    });
    el('[data-ctl="maxepochs"]').addEventListener("input", e => {
      state.maxEpochs = parseInt(e.target.value, 10);
      el('[data-out="maxepochs"]').textContent = state.maxEpochs;
      el('[data-out="maxepochs-live"]').textContent = state.maxEpochs;
    });
    el('[data-ctl="scale"]').addEventListener("input", e => {
      const v = parseFloat(e.target.value);
      const ratio = v / state.scaleVal;
      state.scaleVal = v;
      el('[data-out="scale"]').textContent = v.toFixed(2);
      scaleUnit(state.model, state.selected, ratio);
      updateReadouts();
    });

    trainBtn.addEventListener("click", train);
    el('[data-action="shuffle"]').addEventListener("click", () => {
      permute(state.model, randomPerm(state.hidden));
      state.selected = 0;
      state.scaleVal = 1;
      el('[data-ctl="scale"]').value = "1";
      el('[data-out="scale"]').textContent = "1.00";
      updateReadouts();
    });

    el('[data-out="neurons"]').addEventListener("click", e => {
      const btn = e.target.closest("[data-neuron]");
      if (!btn) return;
      state.selected = parseInt(btn.getAttribute("data-neuron"), 10);
      state.scaleVal = 1;
      el('[data-ctl="scale"]').value = "1";
      el('[data-out="scale"]').textContent = "1.00";
      updateReadouts();
    });

    const themeObserver = new MutationObserver(() => {
      if (!document.body.contains(container)) {
        themeObserver.disconnect();
        return;
      }
      if (!state.running) updateReadouts();
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    newData();
    resetModel();
    train();
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document
      .querySelectorAll('.viz-weight-symmetry[data-viz="weight-symmetry"]')
      .forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
