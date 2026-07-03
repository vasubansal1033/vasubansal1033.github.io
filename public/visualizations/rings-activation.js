(function () {
  const initialized = new WeakSet();

  function sigmoid(z) {
    if (z >= 0) {
      const e = Math.exp(-z);
      return 1 / (1 + e);
    }
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

  function generateRings(nPerRing = 150) {
    const points = [];
    for (let c = 0; c < 2; c++) {
      const radius = c === 0 ? 1.5 : 3.5;
      for (let i = 0; i < nPerRing; i++) {
        const angle = Math.random() * Math.PI * 2;
        const r = radius + randn() * 0.35;
        points.push({
          x: r * Math.cos(angle),
          y: r * Math.sin(angle),
          label: c,
        });
      }
    }
    return points;
  }

  function accuracy(points, predict) {
    let correct = 0;
    for (const p of points) {
      const pred = predict(p.x, p.y) >= 0.5 ? 1 : 0;
      if (pred === p.label) correct++;
    }
    return (correct / points.length) * 100;
  }

  function trainLinear(points, epochs = 800, lr = 0.5) {
    let w0 = (Math.random() - 0.5) * 0.1;
    let w1 = (Math.random() - 0.5) * 0.1;
    let b = 0;

    for (let e = 0; e < epochs; e++) {
      let gw0 = 0;
      let gw1 = 0;
      let gb = 0;

      for (const p of points) {
        const z = w0 * p.x + w1 * p.y + b;
        const yhat = sigmoid(z);
        const err = yhat - p.label;
        gw0 += err * p.x;
        gw1 += err * p.y;
        gb += err;
      }

      const n = points.length;
      w0 -= (lr * gw0) / n;
      w1 -= (lr * gw1) / n;
      b -= (lr * gb) / n;
    }

    return {
      predict(x, y) {
        return sigmoid(w0 * x + w1 * y + b);
      },
    };
  }

  function trainRelu(points, hidden = 16, epochs = 1200, lr = 0.8) {
    const w1 = Array.from({ length: hidden }, () => [
      (Math.random() - 0.5) * 0.5,
      (Math.random() - 0.5) * 0.5,
    ]);
    const b1 = new Array(hidden).fill(0);
    const w2 = Array.from({ length: hidden }, () => (Math.random() - 0.5) * 0.5);
    let b2 = 0;

    for (let e = 0; e < epochs; e++) {
      const gw1 = w1.map(() => [0, 0]);
      const gb1 = new Array(hidden).fill(0);
      const gw2 = new Array(hidden).fill(0);
      let gb2 = 0;

      for (const p of points) {
        const h = new Array(hidden);
        const hPre = new Array(hidden);
        for (let j = 0; j < hidden; j++) {
          hPre[j] = w1[j][0] * p.x + w1[j][1] * p.y + b1[j];
          h[j] = relu(hPre[j]);
        }

        let z = b2;
        for (let j = 0; j < hidden; j++) z += w2[j] * h[j];
        const yhat = sigmoid(z);
        const err = yhat - p.label;
        const dz = err * yhat * (1 - yhat);

        gb2 += dz;
        for (let j = 0; j < hidden; j++) {
          gw2[j] += dz * h[j];
          const dh = dz * w2[j] * (hPre[j] > 0 ? 1 : 0);
          gw1[j][0] += dh * p.x;
          gw1[j][1] += dh * p.y;
          gb1[j] += dh;
        }
      }

      const n = points.length;
      for (let j = 0; j < hidden; j++) {
        w1[j][0] -= (lr * gw1[j][0]) / n;
        w1[j][1] -= (lr * gw1[j][1]) / n;
        b1[j] -= (lr * gb1[j]) / n;
        w2[j] -= (lr * gw2[j]) / n;
      }
      b2 -= (lr * gb2) / n;
    }

    return {
      predict(x, y) {
        let z = b2;
        for (let j = 0; j < hidden; j++) {
          const h = relu(w1[j][0] * x + w1[j][1] * y + b1[j]);
          z += w2[j] * h;
        }
        return sigmoid(z);
      },
    };
  }

  function getThemeColors() {
    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim() || "#617bff";
    const foreground =
      styles.getPropertyValue("--foreground").trim() || "#eaedf3";
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    return {
      class0: isDark ? "rgba(97, 123, 255, 0.35)" : "rgba(225, 74, 57, 0.25)",
      class1: isDark ? "rgba(234, 237, 243, 0.2)" : "rgba(1, 44, 86, 0.15)",
      point0: isDark ? "#617bff" : "#e14a39",
      point1: isDark ? "#eaedf3" : "#012c56",
      border: isDark ? "rgba(234, 237, 243, 0.2)" : "rgba(1, 44, 86, 0.15)",
      text: foreground,
      accent,
    };
  }

  function drawPlot(canvas, points, model, title, acc, colors) {
    const ctx = canvas.getContext("2d");
    const size = canvas.width;
    const padding = 28;
    const plotSize = size - padding * 2;
    const bounds = 5;

    const toCanvas = (x, y) => [
      padding + ((x + bounds) / (2 * bounds)) * plotSize,
      padding + ((bounds - y) / (2 * bounds)) * plotSize,
    ];

    ctx.clearRect(0, 0, size, size);

    const gridRes = 80;
    for (let gi = 0; gi < gridRes; gi++) {
      for (let gj = 0; gj < gridRes; gj++) {
        const x = -bounds + (gi / (gridRes - 1)) * 2 * bounds;
        const y = -bounds + (gj / (gridRes - 1)) * 2 * bounds;
        const p = model.predict(x, y);
        ctx.fillStyle = p >= 0.5 ? colors.class1 : colors.class0;
        const cellW = plotSize / gridRes;
        const cellH = plotSize / gridRes;
        const [cx, cy] = toCanvas(x, y);
        ctx.fillRect(
          cx - cellW / 2,
          cy - cellH / 2,
          cellW + 1,
          cellH + 1
        );
      }
    }

    ctx.strokeStyle = colors.border;
    ctx.lineWidth = 1;
    ctx.strokeRect(padding, padding, plotSize, plotSize);

    for (const pt of points) {
      const [cx, cy] = toCanvas(pt.x, pt.y);
      ctx.beginPath();
      ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 0 ? colors.point0 : colors.point1;
      ctx.fill();
      ctx.strokeStyle = colors.border;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    ctx.fillStyle = colors.text;
    ctx.font = "600 13px system-ui, sans-serif";
    ctx.fillText(title, padding, 18);
    ctx.font = "12px system-ui, sans-serif";
    ctx.fillStyle = colors.accent;
    ctx.fillText(`Accuracy: ${acc.toFixed(1)}%`, padding, size - 10);
  }

  function runExperiment(container) {
    const points = generateRings();
    const linear = trainLinear(points);
    const relu = trainRelu(points);
    const linearAcc = accuracy(points, linear.predict);
    const reluAcc = accuracy(points, relu.predict);
    const colors = getThemeColors();

    const linearCanvas = container.querySelector('[data-canvas="linear"]');
    const reluCanvas = container.querySelector('[data-canvas="relu"]');

    drawPlot(
      linearCanvas,
      points,
      linear,
      "Linear (no activation)",
      linearAcc,
      colors
    );
    drawPlot(
      reluCanvas,
      points,
      relu,
      "One ReLU hidden layer",
      reluAcc,
      colors
    );
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);

    container.innerHTML = `
      <div class="rings-viz" style="margin: 1.5rem 0; font-family: system-ui, sans-serif;">
        <div style="display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center; margin-bottom: 0.75rem;">
          <div style="text-align: center;">
            <canvas data-canvas="linear" width="320" height="320" style="max-width: 100%; height: auto; border-radius: 6px;"></canvas>
          </div>
          <div style="text-align: center;">
            <canvas data-canvas="relu" width="320" height="320" style="max-width: 100%; height: auto; border-radius: 6px;"></canvas>
          </div>
        </div>
        <div style="text-align: center;">
          <button type="button" data-action="regenerate" style="padding: 0.45rem 0.9rem; border-radius: 6px; border: 1px solid var(--accent); background: transparent; color: var(--accent); cursor: pointer; font-size: 0.875rem;">
            Regenerate &amp; retrain
          </button>
        </div>
      </div>
    `;

    const btn = container.querySelector('[data-action="regenerate"]');
    btn.addEventListener("click", () => runExperiment(container));
    runExperiment(container);
  }

  function initAll() {
    document.querySelectorAll(".viz-rings").forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
