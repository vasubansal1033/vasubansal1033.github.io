(function () {
  const BOUNDS = 5;
  const GRID = 56;
  const MAX_EPOCHS = 600;
  const STEPS_PER_FRAME = 3;

  const initialized = new WeakSet();

  /* ---------- math helpers ---------- */

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  }

  function relu(z) {
    return z > 0 ? z : 0;
  }

  function randn() {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function clamp(p) {
    return Math.min(1 - 1e-7, Math.max(1e-7, p));
  }

  /* ---------- datasets ---------- */

  function generateData(kind, noise, nPerClass = 150) {
    const pts = [];
    const jitter = () => randn() * noise;

    if (kind === "rings") {
      for (let c = 0; c < 2; c++) {
        const radius = c === 0 ? 1.5 : 3.4;
        for (let i = 0; i < nPerClass; i++) {
          const a = Math.random() * Math.PI * 2;
          const r = radius + randn() * (0.28 + noise * 0.6);
          pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: c });
        }
      }
    } else if (kind === "moons") {
      for (let i = 0; i < nPerClass; i++) {
        const t = (Math.PI * i) / (nPerClass - 1);
        pts.push({
          x: 2.3 * Math.cos(t) - 1.15 + jitter(),
          y: 2.3 * Math.sin(t) - 1.0 + jitter(),
          label: 0,
        });
        pts.push({
          x: 2.3 * Math.cos(t) + 1.15 + jitter(),
          y: -2.3 * Math.sin(t) + 1.0 + jitter(),
          label: 1,
        });
      }
    } else if (kind === "spiral") {
      for (let c = 0; c < 2; c++) {
        for (let i = 0; i < nPerClass; i++) {
          const frac = i / nPerClass;
          const r = 0.3 + 3.6 * frac;
          const theta = 4 * Math.PI * frac + c * Math.PI;
          pts.push({
            x: r * Math.cos(theta) + jitter(),
            y: r * Math.sin(theta) + jitter(),
            label: c,
          });
        }
      }
    } else if (kind === "xor") {
      const centers = [
        [-2.2, 2.2],
        [2.2, 2.2],
        [-2.2, -2.2],
        [2.2, -2.2],
      ];
      for (let k = 0; k < centers.length; k++) {
        const [cx, cy] = centers[k];
        const label = cx * cy > 0 ? 0 : 1;
        for (let i = 0; i < Math.round((nPerClass * 2) / 4); i++) {
          pts.push({
            x: cx + randn() * (0.6 + noise),
            y: cy + randn() * (0.6 + noise),
            label,
          });
        }
      }
    }

    return pts;
  }

  /* ---------- linear model ---------- */

  function makeLinear() {
    return { w0: randn() * 0.1, w1: randn() * 0.1, b: 0 };
  }

  function predictLinear(m, x, y) {
    return sigmoid(m.w0 * x + m.w1 * y + m.b);
  }

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

  /* ---------- ReLU MLP ---------- */

  function makeRelu(hidden) {
    return {
      hidden,
      w1: Array.from({ length: hidden }, () => [randn() * 0.5, randn() * 0.5]),
      b1: new Array(hidden).fill(0),
      w2: Array.from({ length: hidden }, () => randn() * 0.5),
      b2: 0,
    };
  }

  function predictRelu(m, x, y) {
    let z = m.b2;
    for (let j = 0; j < m.hidden; j++) {
      const h = relu(m.w1[j][0] * x + m.w1[j][1] * y + m.b1[j]);
      z += m.w2[j] * h;
    }
    return sigmoid(z);
  }

  function stepRelu(m, pts, lr) {
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

  function accuracy(pts, predict) {
    let correct = 0;
    for (const p of pts) {
      if ((predict(p.x, p.y) >= 0.5 ? 1 : 0) === p.label) correct++;
    }
    return (correct / pts.length) * 100;
  }

  /* ---------- theme colors ---------- */

  function getColors() {
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    const styles = getComputedStyle(document.documentElement);
    return {
      c0: [249, 115, 22], // orange - class 0
      c1: [59, 130, 246], // blue - class 1
      boundary: isDark ? "#f8fafc" : "#0f172a",
      stroke: isDark ? "rgba(15,23,42,0.6)" : "rgba(255,255,255,0.85)",
      text: styles.getPropertyValue("--foreground").trim() || "#334155",
      accent: styles.getPropertyValue("--accent").trim() || "#617bff",
      grid: isDark ? "rgba(148,163,184,0.18)" : "rgba(100,116,139,0.18)",
      panelBg: isDark ? "#0b1220" : "#f8fafc",
    };
  }

  /* ---------- rendering ---------- */

  function drawPlot(canvas, predict, pts, colors) {
    const ctx = canvas.getContext("2d");
    const size = canvas.width;
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = colors.panelBg;
    ctx.fillRect(0, 0, size, size);

    const cell = size / GRID;
    const probs = [];

    for (let i = 0; i < GRID; i++) {
      probs[i] = [];
      for (let j = 0; j < GRID; j++) {
        const x = -BOUNDS + ((i + 0.5) / GRID) * 2 * BOUNDS;
        const y = BOUNDS - ((j + 0.5) / GRID) * 2 * BOUNDS;
        const p = predict(x, y);
        probs[i][j] = p;
        const conf = Math.abs(p - 0.5) * 2;
        const col = p >= 0.5 ? colors.c1 : colors.c0;
        const alpha = 0.1 + 0.4 * conf;
        ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${alpha})`;
        ctx.fillRect(i * cell, j * cell, cell + 1, cell + 1);
      }
    }

    // boundary (p = 0.5 contour): mark cells where predicted class flips
    ctx.fillStyle = colors.boundary;
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const c = probs[i][j] >= 0.5 ? 1 : 0;
        const right = i < GRID - 1 ? (probs[i + 1][j] >= 0.5 ? 1 : 0) : c;
        const down = j < GRID - 1 ? (probs[i][j + 1] >= 0.5 ? 1 : 0) : c;
        if (c !== right || c !== down) {
          ctx.fillRect(i * cell, j * cell, cell + 1, cell + 1);
        }
      }
    }

    // data points
    const toX = x => ((x + BOUNDS) / (2 * BOUNDS)) * size;
    const toY = y => ((BOUNDS - y) / (2 * BOUNDS)) * size;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(toX(p.x), toY(p.y), 3, 0, Math.PI * 2);
      const col = p.label === 0 ? colors.c0 : colors.c1;
      ctx.fillStyle = `rgb(${col[0]},${col[1]},${col[2]})`;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = colors.stroke;
      ctx.stroke();
    }
  }

  function drawLoss(canvas, lossLin, lossRelu, colors) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = colors.panelBg;
    ctx.fillRect(0, 0, w, h);

    const pad = 6;
    let maxLoss = 0.001;
    for (const v of lossLin) if (v > maxLoss) maxLoss = v;
    for (const v of lossRelu) if (v > maxLoss) maxLoss = v;

    const n = Math.max(lossLin.length, lossRelu.length, 2);
    const plot = (arr, color) => {
      if (arr.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      for (let k = 0; k < arr.length; k++) {
        const x = pad + (k / (n - 1)) * (w - 2 * pad);
        const y = h - pad - (arr[k] / maxLoss) * (h - 2 * pad);
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    plot(lossLin, `rgb(${colors.c0[0]},${colors.c0[1]},${colors.c0[2]})`);
    plot(lossRelu, `rgb(${colors.c1[0]},${colors.c1[1]},${colors.c1[2]})`);
  }

  /* ---------- widget ---------- */

  function createWidget(container) {
    const state = {
      dataset: "rings",
      noise: 0.25,
      lr: 0.5,
      hidden: 16,
      pts: [],
      linear: null,
      relu: null,
      lossLin: [],
      lossRelu: [],
      epoch: 0,
      running: false,
      raf: null,
    };

    const styleBtn =
      "padding:0.4rem 0.8rem;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;";
    const styleBtnGhost =
      "padding:0.4rem 0.8rem;border-radius:6px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;";

    container.innerHTML = `
      <div class="rings-viz" style="margin:1.5rem 0;font-family:system-ui,sans-serif;font-size:0.9rem;">
        <div style="display:flex;flex-wrap:wrap;gap:1rem 1.5rem;align-items:flex-end;margin-bottom:1rem;padding:0.85rem 1rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:8px;">
          <label style="display:flex;flex-direction:column;gap:0.25rem;">
            <span style="opacity:0.8;">Dataset</span>
            <select data-ctl="dataset" style="padding:0.3rem;border-radius:5px;background:transparent;color:inherit;border:1px solid var(--border,rgba(120,120,120,0.4));">
              <option value="rings">Rings</option>
              <option value="moons">Moons</option>
              <option value="spiral">Spiral</option>
              <option value="xor">XOR</option>
            </select>
          </label>
          <label style="display:flex;flex-direction:column;gap:0.25rem;">
            <span style="opacity:0.8;">Noise: <b data-out="noise">0.25</b></span>
            <input data-ctl="noise" type="range" min="0" max="0.6" step="0.02" value="0.25" />
          </label>
          <label style="display:flex;flex-direction:column;gap:0.25rem;">
            <span style="opacity:0.8;">Learning rate: <b data-out="lr">0.50</b></span>
            <input data-ctl="lr" type="range" min="0.05" max="1.5" step="0.05" value="0.5" />
          </label>
          <label style="display:flex;flex-direction:column;gap:0.25rem;">
            <span style="opacity:0.8;">ReLU hidden units: <b data-out="hidden">16</b></span>
            <input data-ctl="hidden" type="range" min="2" max="32" step="1" value="16" />
          </label>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;justify-content:center;margin-bottom:0.75rem;">
          <div style="text-align:center;flex:1 1 280px;max-width:340px;">
            <div style="font-weight:600;margin-bottom:0.3rem;">A · Linear + sigmoid</div>
            <canvas data-canvas="linear" width="300" height="300" style="width:100%;height:auto;border-radius:8px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
            <div style="margin-top:0.3rem;">train accuracy: <b data-acc="linear" style="color:rgb(249,115,22);">—</b></div>
          </div>
          <div style="text-align:center;flex:1 1 280px;max-width:340px;">
            <div style="font-weight:600;margin-bottom:0.3rem;">B · One ReLU hidden layer</div>
            <canvas data-canvas="relu" width="300" height="300" style="width:100%;height:auto;border-radius:8px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
            <div style="margin-top:0.3rem;">train accuracy: <b data-acc="relu" style="color:rgb(59,130,246);">—</b></div>
          </div>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:center;justify-content:center;margin-bottom:0.75rem;">
          <button type="button" data-action="train" style="${styleBtn}">▶ Train both</button>
          <button type="button" data-action="reset" style="${styleBtnGhost}">↺ Reset weights</button>
          <button type="button" data-action="newdata" style="${styleBtnGhost}">⤺ New data</button>
          <span style="opacity:0.8;">epoch <b data-out="epoch">0</b></span>
        </div>

        <div style="text-align:center;">
          <div style="font-size:0.8rem;opacity:0.75;margin-bottom:0.2rem;">
            training loss —
            <span style="color:rgb(249,115,22);">linear</span> vs
            <span style="color:rgb(59,130,246);">ReLU</span>
          </div>
          <canvas data-canvas="loss" width="620" height="120" style="width:100%;max-width:620px;height:auto;border-radius:8px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
        </div>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const linearCanvas = el('[data-canvas="linear"]');
    const reluCanvas = el('[data-canvas="relu"]');
    const lossCanvas = el('[data-canvas="loss"]');
    const trainBtn = el('[data-action="train"]');

    function render() {
      const colors = getColors();
      drawPlot(linearCanvas, (x, y) => predictLinear(state.linear, x, y), state.pts, colors);
      drawPlot(reluCanvas, (x, y) => predictRelu(state.relu, x, y), state.pts, colors);
      drawLoss(lossCanvas, state.lossLin, state.lossRelu, colors);
      el('[data-acc="linear"]').textContent =
        accuracy(state.pts, (x, y) => predictLinear(state.linear, x, y)).toFixed(1) + "%";
      el('[data-acc="relu"]').textContent =
        accuracy(state.pts, (x, y) => predictRelu(state.relu, x, y)).toFixed(1) + "%";
      el('[data-out="epoch"]').textContent = state.epoch;
    }

    function resetWeights() {
      state.linear = makeLinear();
      state.relu = makeRelu(state.hidden);
      state.lossLin = [];
      state.lossRelu = [];
      state.epoch = 0;
      render();
    }

    function newData() {
      state.pts = generateData(state.dataset, state.noise);
      resetWeights();
    }

    function stop() {
      state.running = false;
      if (state.raf) cancelAnimationFrame(state.raf);
      state.raf = null;
      trainBtn.textContent = "▶ Train both";
    }

    function loop() {
      if (!state.running) return;
      if (!document.body.contains(container)) return stop();
      for (let k = 0; k < STEPS_PER_FRAME; k++) {
        if (state.epoch >= MAX_EPOCHS) {
          stop();
          break;
        }
        state.lossLin.push(stepLinear(state.linear, state.pts, state.lr));
        state.lossRelu.push(stepRelu(state.relu, state.pts, state.lr));
        state.epoch++;
      }
      render();
      if (state.running) state.raf = requestAnimationFrame(loop);
    }

    function start() {
      if (state.running) {
        stop();
        return;
      }
      if (state.epoch >= MAX_EPOCHS) resetWeights();
      state.running = true;
      trainBtn.textContent = "⏸ Pause";
      state.raf = requestAnimationFrame(loop);
    }

    /* controls */
    el('[data-ctl="dataset"]').addEventListener("change", e => {
      state.dataset = e.target.value;
      stop();
      newData();
    });
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
    el('[data-ctl="hidden"]').addEventListener("input", e => {
      state.hidden = parseInt(e.target.value, 10);
      el('[data-out="hidden"]').textContent = state.hidden;
      stop();
      resetWeights();
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

    newData();
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document.querySelectorAll(".viz-rings").forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
