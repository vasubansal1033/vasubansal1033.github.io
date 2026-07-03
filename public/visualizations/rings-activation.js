(function () {
  const BOUNDS = 5;
  const GRID = 64;
  const MAX_EPOCHS = 600;
  const STEPS_PER_FRAME = 3;
  const D3_URL = "https://cdn.jsdelivr.net/npm/d3@7/+esm";

  const initialized = new WeakSet();
  let d3Promise = null;

  function loadD3() {
    if (!d3Promise) {
      d3Promise = import(/* @vite-ignore */ D3_URL).catch(() => null);
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
      for (const [cx, cy] of centers) {
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

  /* ---------- models ---------- */

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
      z += m.w2[j] * relu(m.w1[j][0] * x + m.w1[j][1] * y + m.b1[j]);
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

  /* ---------- theme ---------- */

  const C0 = "#f97316"; // class 0 (orange)
  const C1 = "#3b82f6"; // class 1 (blue)

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
      panelBg: isDark ? "#0b1220" : "#f8fafc",
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

  function drawPlotD3(canvas, dpr, predict, pts, d3, colors) {
    const size = 300;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const color = d3
      .scaleLinear()
      .domain([0, 0.25, 0.5, 0.75, 1])
      .range(colors.field)
      .interpolate(d3.interpolateRgb)
      .clamp(true);

    const values = computeField(predict);
    const thresholds = d3.range(0.05, 1, 1 / 15);
    const contours = d3.contours().size([GRID, GRID]).smooth(true);

    // scale contour (grid) coordinates to canvas pixels
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

    // crisp decision boundary at p = 0.5
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

    // data points
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
  }

  function drawLossD3(svgSel, d3, lossLin, lossRelu, colors) {
    const W = 620;
    const H = 164;
    const m = { t: 10, r: 12, b: 40, l: 40 };
    svgSel.selectAll("*").remove();

    const n = Math.max(lossLin.length, 2);
    const maxL = Math.max(0.02, d3.max([...lossLin, ...lossRelu]) || 0.02);

    const x = d3.scaleLinear([0, Math.max(n - 1, 1)], [m.l, W - m.r]);
    const y = d3.scaleLinear([0, maxL], [H - m.b, m.t]);

    const gx = svgSel
      .append("g")
      .attr("transform", `translate(0,${H - m.b})`)
      .attr("color", colors.axis)
      .call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
    const gy = svgSel
      .append("g")
      .attr("transform", `translate(${m.l},0)`)
      .attr("color", colors.axis)
      .call(d3.axisLeft(y).ticks(4).tickSizeOuter(0));
    for (const g of [gx, gy]) {
      g.selectAll("text").attr("fill", colors.text).style("font-size", "10px");
    }

    svgSel
      .append("text")
      .attr("x", (m.l + (W - m.r)) / 2)
      .attr("y", H - 6)
      .attr("text-anchor", "middle")
      .attr("fill", colors.text)
      .style("font-size", "10px")
      .style("opacity", 0.7)
      .text("epoch");

    const line = d3
      .line()
      .x((d, i) => x(i))
      .y(d => y(d))
      .curve(d3.curveMonotoneX);

    const draw = (arr, stroke) =>
      svgSel
        .append("path")
        .datum(arr)
        .attr("fill", "none")
        .attr("stroke", stroke)
        .attr("stroke-width", 1.8)
        .attr("d", line);

    draw(lossLin, C0);
    draw(lossRelu, C1);
  }

  /* ---------- fallback (no d3) ---------- */

  function drawPlotFallback(canvas, dpr, predict, pts, colors) {
    const size = 300;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = colors.panelBg;
    ctx.fillRect(0, 0, size, size);

    const N = 56;
    const cell = size / N;
    const c0 = [249, 115, 22];
    const c1 = [59, 130, 246];
    for (let i = 0; i < N; i++) {
      const x = -BOUNDS + ((i + 0.5) / N) * 2 * BOUNDS;
      for (let j = 0; j < N; j++) {
        const yy = BOUNDS - ((j + 0.5) / N) * 2 * BOUNDS;
        const p = predict(x, yy);
        const conf = Math.abs(p - 0.5) * 2;
        const col = p >= 0.5 ? c1 : c0;
        ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${0.1 + 0.4 * conf})`;
        ctx.fillRect(i * cell, j * cell, cell + 1, cell + 1);
      }
    }
    const toX = x => ((x + BOUNDS) / (2 * BOUNDS)) * size;
    const toY = y => ((BOUNDS - y) / (2 * BOUNDS)) * size;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(toX(p.x), toY(p.y), 3, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 0 ? C0 : C1;
      ctx.fill();
      ctx.strokeStyle = colors.pointStroke;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  function drawLossFallback(canvas, lossLin, lossRelu, colors) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = colors.panelBg;
    ctx.fillRect(0, 0, w, h);
    const pad = 8;
    let maxL = 0.02;
    for (const v of lossLin) if (v > maxL) maxL = v;
    for (const v of lossRelu) if (v > maxL) maxL = v;
    const n = Math.max(lossLin.length, lossRelu.length, 2);
    const plot = (arr, stroke) => {
      if (arr.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.5;
      for (let k = 0; k < arr.length; k++) {
        const x = pad + (k / (n - 1)) * (w - 2 * pad);
        const y = h - pad - (arr[k] / maxL) * (h - 2 * pad);
        k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    plot(lossLin, C0);
    plot(lossRelu, C1);
  }

  /* ---------- widget ---------- */

  async function createWidget(container) {
    const d3 = await loadD3();
    if (!document.body.contains(container)) return;

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
      "padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;";
    const styleGhost =
      "padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;";
    const lossMarkup = d3
      ? `<svg data-svg="loss" viewBox="0 0 620 164" style="display:block;margin:0 auto;width:100%;max-width:620px;height:auto;"></svg>`
      : `<canvas data-canvas="loss" width="620" height="164" style="display:block;margin:0 auto;width:100%;max-width:620px;height:auto;border-radius:8px;"></canvas>`;

    container.innerHTML = `
      <div class="rings-viz" style="margin:1.5rem 0;font-family:system-ui,sans-serif;font-size:0.9rem;">
        <div style="display:flex;flex-wrap:wrap;gap:1rem 1.5rem;align-items:flex-end;margin-bottom:1rem;padding:0.85rem 1rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;">
          <label style="display:flex;flex-direction:column;gap:0.25rem;">
            <span style="opacity:0.8;">Dataset</span>
            <select data-ctl="dataset" style="padding:0.35rem;border-radius:6px;background:transparent;color:inherit;border:1px solid var(--border,rgba(120,120,120,0.4));">
              <option value="rings">Rings</option>
              <option value="moons">Moons</option>
              <option value="spiral">Spiral</option>
              <option value="xor">XOR</option>
            </select>
          </label>
          <label style="display:flex;flex-direction:column;gap:0.25rem;min-width:140px;">
            <span style="opacity:0.8;">Noise: <b data-out="noise">0.25</b></span>
            <input data-ctl="noise" type="range" min="0" max="0.6" step="0.02" value="0.25" style="accent-color:var(--accent);" />
          </label>
          <label style="display:flex;flex-direction:column;gap:0.25rem;min-width:140px;">
            <span style="opacity:0.8;">Learning rate: <b data-out="lr">0.50</b></span>
            <input data-ctl="lr" type="range" min="0.05" max="1.5" step="0.05" value="0.5" style="accent-color:var(--accent);" />
          </label>
          <label style="display:flex;flex-direction:column;gap:0.25rem;min-width:140px;">
            <span style="opacity:0.8;">ReLU hidden units: <b data-out="hidden">16</b></span>
            <input data-ctl="hidden" type="range" min="2" max="32" step="1" value="16" style="accent-color:var(--accent);" />
          </label>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;justify-content:center;margin-bottom:0.75rem;">
          <div style="text-align:center;flex:1 1 280px;max-width:340px;">
            <div style="font-weight:600;margin-bottom:0.3rem;">A · Linear + sigmoid</div>
            <canvas data-canvas="linear" width="300" height="300" style="width:100%;height:auto;border-radius:10px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
            <div style="margin-top:0.35rem;">train accuracy: <b data-acc="linear" style="color:${C0};">—</b></div>
          </div>
          <div style="text-align:center;flex:1 1 280px;max-width:340px;">
            <div style="font-weight:600;margin-bottom:0.3rem;">B · One ReLU hidden layer</div>
            <canvas data-canvas="relu" width="300" height="300" style="width:100%;height:auto;border-radius:10px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
            <div style="margin-top:0.35rem;">train accuracy: <b data-acc="relu" style="color:${C1};">—</b></div>
          </div>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:center;justify-content:center;margin-bottom:0.75rem;">
          <button type="button" data-action="train" style="${styleBtn}">▶ Train both</button>
          <button type="button" data-action="reset" style="${styleGhost}">↺ Reset weights</button>
          <button type="button" data-action="newdata" style="${styleGhost}">⤺ New data</button>
          <span style="opacity:0.8;">epoch <b data-out="epoch">0</b></span>
        </div>

        <div style="text-align:center;">
          <div style="font-size:0.8rem;opacity:0.75;margin-bottom:0.25rem;">
            training loss —
            <span style="color:${C0};">linear</span> vs
            <span style="color:${C1};">ReLU</span>
          </div>
          ${lossMarkup}
        </div>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const linearCanvas = el('[data-canvas="linear"]');
    const reluCanvas = el('[data-canvas="relu"]');
    const trainBtn = el('[data-action="train"]');

    const linCtx = makeContext(linearCanvas, 300);
    const reluCtx = makeContext(reluCanvas, 300);
    const lossSvg = d3 ? d3.select(el('[data-svg="loss"]')) : null;
    const lossCanvas = d3 ? null : el('[data-canvas="loss"]');

    function render() {
      const colors = getColors();
      const predA = (x, y) => predictLinear(state.linear, x, y);
      const predB = (x, y) => predictRelu(state.relu, x, y);
      if (d3) {
        drawPlotD3(linearCanvas, linCtx.dpr, predA, state.pts, d3, colors);
        drawPlotD3(reluCanvas, reluCtx.dpr, predB, state.pts, d3, colors);
        drawLossD3(lossSvg, d3, state.lossLin, state.lossRelu, colors);
      } else {
        drawPlotFallback(linearCanvas, linCtx.dpr, predA, state.pts, colors);
        drawPlotFallback(reluCanvas, reluCtx.dpr, predB, state.pts, colors);
        drawLossFallback(lossCanvas, state.lossLin, state.lossRelu, colors);
      }
      el('[data-acc="linear"]').textContent = accuracy(state.pts, predA).toFixed(1) + "%";
      el('[data-acc="relu"]').textContent = accuracy(state.pts, predB).toFixed(1) + "%";
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
      if (state.running) return stop();
      if (state.epoch >= MAX_EPOCHS) resetWeights();
      state.running = true;
      trainBtn.textContent = "⏸ Pause";
      state.raf = requestAnimationFrame(loop);
    }

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

    // re-render on theme switch when idle
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
    document.querySelectorAll(".viz-rings").forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
