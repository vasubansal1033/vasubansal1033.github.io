(function () {
  const TRAIN_SIZES = [20, 200, 2000];
  const TEST_SIZE = 400;
  const HIDDEN = 64;
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

  /* ---------- data ---------- */

  function generateRings(noise, n) {
    const pts = [];
    const nPerClass = Math.ceil(n / 2);
    for (let c = 0; c < 2; c++) {
      const radius = c === 0 ? 1.5 : 3.4;
      for (let i = 0; i < nPerClass && pts.length < n; i++) {
        const a = Math.random() * Math.PI * 2;
        const r = radius + randn() * (0.28 + noise * 0.6);
        pts.push({ x: r * Math.cos(a), y: r * Math.sin(a), label: c });
      }
    }
    return pts;
  }

  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  /* ---------- model ---------- */

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

  function evalMetrics(m, pts) {
    let correct = 0;
    let loss = 0;
    for (const p of pts) {
      const yhat = predictRelu(m, p.x, p.y);
      if ((yhat >= 0.5 ? 1 : 0) === p.label) correct++;
      const pc = clamp(yhat);
      loss += -(p.label * Math.log(pc) + (1 - p.label) * Math.log(1 - pc));
    }
    return {
      acc: (correct / pts.length) * 100,
      loss: loss / pts.length,
    };
  }

  /* ---------- theme ---------- */

  const C_TRAIN = "#f97316";
  const C_TEST = "#3b82f6";
  const C_GAP = "#8b5cf6";

  function getColors() {
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    const styles = getComputedStyle(document.documentElement);
    const accent =
      styles.getPropertyValue("--accent").trim() ||
      (isDark ? "#60a5fa" : "#2563eb");
    return {
      isDark,
      text: styles.getPropertyValue("--foreground").trim() || "#334155",
      axis: isDark ? "rgba(148,163,184,0.7)" : "rgba(100,116,139,0.7)",
      grid: isDark ? "rgba(148,163,184,0.15)" : "rgba(100,116,139,0.12)",
      accent,
      barFill: isDark ? "rgba(139,92,246,0.85)" : "rgba(124,58,237,0.75)",
      barStroke: isDark ? "#c4b5fd" : "#6d28d9",
    };
  }

  /* ---------- d3 charts ---------- */

  function drawLossPanel(svgSel, d3, trainLoss, testLoss, colors, epoch) {
    const W = 280;
    const H = 120;
    const m = { t: 8, r: 8, b: 28, l: 36 };
    svgSel.selectAll("*").remove();

    const n = Math.max(trainLoss.length, 2);
    const maxL = Math.max(0.02, d3.max([...trainLoss, ...testLoss]) || 0.02);

    const x = d3.scaleLinear([0, Math.max(n - 1, 1)], [m.l, W - m.r]);
    const y = d3.scaleLinear([0, maxL * 1.05], [H - m.b, m.t]);

    const gx = svgSel
      .append("g")
      .attr("transform", `translate(0,${H - m.b})`)
      .attr("color", colors.axis)
      .call(d3.axisBottom(x).ticks(4).tickSizeOuter(0));
    const gy = svgSel
      .append("g")
      .attr("transform", `translate(${m.l},0)`)
      .attr("color", colors.axis)
      .call(d3.axisLeft(y).ticks(3).tickSizeOuter(0));
    for (const g of [gx, gy]) {
      g.selectAll("text").attr("fill", colors.text).style("font-size", "9px");
    }

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
        .attr("stroke-width", 1.6)
        .attr("d", line);

    draw(trainLoss, C_TRAIN);
    draw(testLoss, C_TEST);

    svgSel
      .append("text")
      .attr("x", W - m.r)
      .attr("y", m.t + 8)
      .attr("text-anchor", "end")
      .attr("fill", colors.text)
      .style("font-size", "8px")
      .style("opacity", 0.65)
      .text(`epoch ${epoch}`);
  }

  function drawGapChart(svgSel, d3, runs, colors) {
    const W = 620;
    const H = 200;
    const m = { t: 16, r: 16, b: 44, l: 48 };
    svgSel.selectAll("*").remove();

    const data = TRAIN_SIZES.map(size => {
      const run = runs[size];
      const gap =
        run && run.trainAcc.length
          ? run.trainAcc[run.trainAcc.length - 1] -
            run.testAcc[run.testAcc.length - 1]
          : 0;
      return { size, gap: Math.max(0, gap) };
    });

    const maxGap = Math.max(5, d3.max(data, d => d.gap) || 5);

    const x = d3
      .scaleBand()
      .domain(TRAIN_SIZES.map(String))
      .range([m.l, W - m.r])
      .padding(0.35);
    const y = d3.scaleLinear([0, maxGap * 1.1], [H - m.b, m.t]);

    svgSel
      .append("g")
      .attr("transform", `translate(0,${H - m.b})`)
      .attr("color", colors.axis)
      .call(
        d3
          .axisBottom(x)
          .tickFormat(d => `${d} pts`)
          .tickSizeOuter(0)
      )
      .selectAll("text")
      .attr("fill", colors.text)
      .style("font-size", "10px");

    svgSel
      .append("g")
      .attr("transform", `translate(${m.l},0)`)
      .attr("color", colors.axis)
      .call(
        d3
          .axisLeft(y)
          .ticks(5)
          .tickFormat(d => `${d}%`)
          .tickSizeOuter(0)
      )
      .selectAll("text")
      .attr("fill", colors.text)
      .style("font-size", "10px");

    svgSel
      .append("g")
      .attr("stroke", colors.grid)
      .selectAll("line")
      .data(y.ticks(5))
      .join("line")
      .attr("x1", m.l)
      .attr("x2", W - m.r)
      .attr("y1", d => y(d))
      .attr("y2", d => y(d));

    svgSel
      .selectAll("rect")
      .data(data)
      .join("rect")
      .attr("x", d => x(String(d.size)))
      .attr("y", d => y(d.gap))
      .attr("width", x.bandwidth())
      .attr("height", d => y(0) - y(d.gap))
      .attr("fill", colors.barFill)
      .attr("stroke", colors.barStroke)
      .attr("stroke-width", 1)
      .attr("rx", 4);

    svgSel
      .selectAll("text.gap-label")
      .data(data)
      .join("text")
      .attr("class", "gap-label")
      .attr("x", d => x(String(d.size)) + x.bandwidth() / 2)
      .attr("y", d => y(d.gap) - 5)
      .attr("text-anchor", "middle")
      .attr("fill", colors.text)
      .style("font-size", "10px")
      .style("font-weight", "600")
      .text(d => (d.gap > 0.5 ? `${d.gap.toFixed(1)}%` : ""));

    svgSel
      .append("text")
      .attr("x", (m.l + (W - m.r)) / 2)
      .attr("y", H - 8)
      .attr("text-anchor", "middle")
      .attr("fill", colors.text)
      .style("font-size", "10px")
      .style("opacity", 0.7)
      .text("training set size");

    svgSel
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -(H / 2))
      .attr("y", 14)
      .attr("text-anchor", "middle")
      .attr("fill", colors.text)
      .style("font-size", "10px")
      .style("opacity", 0.7)
      .text("generalization gap (train − test accuracy)");
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
      noise: 0.3,
      lr: 0.4,
      maxEpochs: DEFAULT_MAX_EPOCHS,
      pool: [],
      testPts: [],
      runs: {},
      epoch: 0,
      running: false,
      raf: null,
    };

    const styleBtn =
      "padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;";
    const styleGhost =
      "padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;";

    const panelHtml = size => `
      <div data-panel="${size}" style="flex:1 1 260px;max-width:320px;padding:0.75rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;">
        <div style="font-weight:600;margin-bottom:0.4rem;">n = ${size} training points</div>
        <div style="display:flex;justify-content:space-between;gap:0.5rem;margin-bottom:0.35rem;font-size:0.85rem;">
          <span>train acc: <b data-acc-train="${size}" style="color:${C_TRAIN};">—</b></span>
          <span>test acc: <b data-acc-test="${size}" style="color:${C_TEST};">—</b></span>
        </div>
        <div style="font-size:0.8rem;opacity:0.75;margin-bottom:0.2rem;">
          loss — <span style="color:${C_TRAIN};">train</span> vs <span style="color:${C_TEST};">test</span>
        </div>
        <svg data-svg-loss="${size}" viewBox="0 0 280 120" style="display:block;width:100%;height:auto;"></svg>
        <div style="margin-top:0.35rem;font-size:0.85rem;">
          gap: <b data-gap="${size}" style="color:${C_GAP};">—</b>
        </div>
      </div>
    `;

    container.innerHTML = `
      <div class="generalization-gap-viz" style="margin:1.5rem 0;font-family:system-ui,sans-serif;font-size:0.9rem;">
        <div style="display:flex;flex-wrap:wrap;gap:1rem 1.5rem;align-items:flex-end;margin-bottom:1rem;padding:0.85rem 1rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;">
          <label style="display:flex;flex-direction:column;gap:0.25rem;min-width:140px;">
            <span style="opacity:0.8;">Label noise: <b data-out="noise">0.30</b></span>
            <input data-ctl="noise" type="range" min="0.05" max="0.55" step="0.02" value="0.3" style="accent-color:var(--accent);" />
          </label>
          <label style="display:flex;flex-direction:column;gap:0.25rem;min-width:140px;">
            <span style="opacity:0.8;">Learning rate: <b data-out="lr">0.40</b></span>
            <input data-ctl="lr" type="range" min="0.05" max="1.0" step="0.05" value="0.4" style="accent-color:var(--accent);" />
          </label>
          <label style="display:flex;flex-direction:column;gap:0.25rem;min-width:140px;">
            <span style="opacity:0.8;">Epochs: <b data-out="maxepochs">500</b></span>
            <input data-ctl="maxepochs" type="range" min="100" max="2000" step="50" value="500" style="accent-color:var(--accent);" />
          </label>
          <span style="opacity:0.75;font-size:0.85rem;">64-unit ReLU net · ${TEST_SIZE} held-out test points</span>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;justify-content:center;margin-bottom:0.75rem;">
          ${TRAIN_SIZES.map(panelHtml).join("")}
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:center;justify-content:center;margin-bottom:1rem;">
          <button type="button" data-action="train" style="${styleBtn}">▶ Train all three</button>
          <button type="button" data-action="reset" style="${styleGhost}">↺ Reset</button>
          <button type="button" data-action="newdata" style="${styleGhost}">⤺ New data split</button>
          <span style="opacity:0.8;">epoch <b data-out="epoch">0</b> / <b data-out="maxepochs-live">500</b></span>
        </div>

        <div style="text-align:center;">
          <div style="font-size:0.8rem;opacity:0.75;margin-bottom:0.25rem;">
            generalization gap after training — bigger gap at small n, smaller at large n
          </div>
          <svg data-svg="gap" viewBox="0 0 620 200" style="display:block;margin:0 auto;width:100%;max-width:620px;height:auto;"></svg>
        </div>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const trainBtn = el('[data-action="train"]');
    const gapSvg = d3.select(el('[data-svg="gap"]'));
    const lossSvgs = Object.fromEntries(
      TRAIN_SIZES.map(size => [
        size,
        d3.select(el(`[data-svg-loss="${size}"]`)),
      ])
    );

    function makeRun(trainPts) {
      return {
        model: makeRelu(HIDDEN),
        trainPts,
        trainLoss: [],
        testLoss: [],
        trainAcc: [],
        testAcc: [],
      };
    }

    function initRuns() {
      state.runs = {};
      for (const size of TRAIN_SIZES) {
        const trainPts = state.pool.slice(0, size);
        state.runs[size] = makeRun(trainPts);
      }
      state.epoch = 0;
    }

    function newData() {
      state.pool = shuffle(
        generateRings(state.noise, Math.max(...TRAIN_SIZES) + TEST_SIZE + 200)
      );
      state.testPts = state.pool.splice(0, TEST_SIZE);
      initRuns();
    }

    function render() {
      const colors = getColors();

      for (const size of TRAIN_SIZES) {
        const run = state.runs[size];
        if (!run) continue;

        const trainAcc =
          run.trainAcc.length > 0
            ? run.trainAcc[run.trainAcc.length - 1]
            : evalMetrics(run.model, run.trainPts).acc;
        const testAcc =
          run.testAcc.length > 0
            ? run.testAcc[run.testAcc.length - 1]
            : evalMetrics(run.model, state.testPts).acc;
        const gap = trainAcc - testAcc;

        el(`[data-acc-train="${size}"]`).textContent =
          trainAcc.toFixed(1) + "%";
        el(`[data-acc-test="${size}"]`).textContent = testAcc.toFixed(1) + "%";
        el(`[data-gap="${size}"]`).textContent =
          (gap >= 0 ? "" : "−") + Math.abs(gap).toFixed(1) + " pp";

        drawLossPanel(
          lossSvgs[size],
          d3,
          run.trainLoss,
          run.testLoss,
          colors,
          state.epoch
        );
      }

      drawGapChart(gapSvg, d3, state.runs, colors);
      el('[data-out="epoch"]').textContent = state.epoch;
      el('[data-out="maxepochs-live"]').textContent = state.maxEpochs;
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
        if (state.epoch >= state.maxEpochs) {
          stop();
          break;
        }

        for (const size of TRAIN_SIZES) {
          const run = state.runs[size];
          run.trainLoss.push(stepRelu(run.model, run.trainPts, state.lr));
          const trainM = evalMetrics(run.model, run.trainPts);
          const testM = evalMetrics(run.model, state.testPts);
          run.testLoss.push(testM.loss);
          run.trainAcc.push(trainM.acc);
          run.testAcc.push(testM.acc);
        }
        state.epoch++;
      }

      render();
      if (state.running) state.raf = requestAnimationFrame(loop);
    }

    function start() {
      if (state.running) return stop();
      if (state.epoch >= state.maxEpochs) initRuns();
      state.running = true;
      trainBtn.textContent = "⏸ Pause";
      state.raf = requestAnimationFrame(loop);
    }

    el('[data-ctl="noise"]').addEventListener("input", e => {
      state.noise = parseFloat(e.target.value);
      el('[data-out="noise"]').textContent = state.noise.toFixed(2);
      stop();
      newData();
      render();
    });
    el('[data-ctl="lr"]').addEventListener("input", e => {
      state.lr = parseFloat(e.target.value);
      el('[data-out="lr"]').textContent = state.lr.toFixed(2);
    });
    el('[data-ctl="maxepochs"]').addEventListener("input", e => {
      state.maxEpochs = parseInt(e.target.value, 10);
      el('[data-out="maxepochs"]').textContent = state.maxEpochs;
      el('[data-out="maxepochs-live"]').textContent = state.maxEpochs;
    });

    trainBtn.addEventListener("click", start);
    el('[data-action="reset"]').addEventListener("click", () => {
      stop();
      initRuns();
      render();
    });
    el('[data-action="newdata"]').addEventListener("click", () => {
      stop();
      newData();
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

    newData();
    render();
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document.querySelectorAll(".viz-generalization-gap").forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
