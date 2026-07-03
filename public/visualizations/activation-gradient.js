(function () {
  const initialized = new WeakSet();

  /* ---------- activation functions and derivatives ---------- */

  const sig = z => 1 / (1 + Math.exp(-z));
  const tnh = z => Math.tanh(z);
  const GELU_C = Math.sqrt(2 / Math.PI);
  const gelu = z =>
    0.5 * z * (1 + Math.tanh(GELU_C * (z + 0.044715 * z * z * z)));

  const FUNCS = {
    sigmoid: {
      label: "sigmoid",
      color: "#f97316",
      f: sig,
      df: z => sig(z) * (1 - sig(z)),
      maxSlope: 0.25,
      note: "best-case slope only 0.25 — and ~0 in the tails",
    },
    tanh: {
      label: "tanh",
      color: "#a855f7",
      f: tnh,
      df: z => 1 - tnh(z) * tnh(z),
      maxSlope: 1.0,
      note: "slope 1 at the origin, but saturates to ~0 in the tails",
    },
    relu: {
      label: "ReLU",
      color: "#3b82f6",
      f: z => (z > 0 ? z : 0),
      df: z => (z > 0 ? 1 : 0),
      maxSlope: 1.0,
      note: "slope exactly 1 for every z > 0 — never saturates on the right",
    },
    leaky: {
      label: "Leaky ReLU",
      color: "#22c55e",
      f: z => (z > 0 ? z : 0.1 * z),
      df: z => (z > 0 ? 1 : 0.1),
      maxSlope: 1.0,
      note: "like ReLU, but a small slope on the left so units can't fully die",
    },
    gelu: {
      label: "GELU",
      color: "#eab308",
      f: gelu,
      df: z => {
        const h = 1e-4;
        return (gelu(z + h) - gelu(z - h)) / (2 * h);
      },
      maxSlope: 1.13,
      note: "smooth ReLU-like curve used in most modern Transformers",
    },
  };

  const ORDER = ["sigmoid", "tanh", "relu", "leaky", "gelu"];

  /* ---------- theme ---------- */

  function getColors() {
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    const styles = getComputedStyle(document.documentElement);
    return {
      isDark,
      text: styles.getPropertyValue("--foreground").trim() || "#334155",
      axis: isDark ? "rgba(148,163,184,0.9)" : "rgba(100,116,139,0.9)",
      grid: isDark ? "rgba(148,163,184,0.15)" : "rgba(100,116,139,0.15)",
      bg: isDark ? "rgba(15,23,42,0.35)" : "rgba(248,250,252,0.6)",
    };
  }

  /* ---------- canvas helpers ---------- */

  function makeContext(canvas, w, h) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    return { ctx: canvas.getContext("2d"), dpr };
  }

  function plotFrame(ctx, W, H, m, colors, xr, yr, xlabel, ylabel) {
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(1, 1, W - 2, H - 2, 8);
    else ctx.rect(1, 1, W - 2, H - 2);
    ctx.fillStyle = colors.bg;
    ctx.fill();
    ctx.restore();

    const px = x => m.l + ((x - xr[0]) / (xr[1] - xr[0])) * (W - m.l - m.r);
    const py = y => H - m.b - ((y - yr[0]) / (yr[1] - yr[0])) * (H - m.t - m.b);

    // grid + zero axes
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    for (let gx = Math.ceil(xr[0]); gx <= xr[1]; gx += 2) {
      ctx.beginPath();
      ctx.moveTo(px(gx), m.t);
      ctx.lineTo(px(gx), H - m.b);
      ctx.stroke();
    }
    ctx.strokeStyle = colors.axis;
    ctx.globalAlpha = 0.5;
    if (yr[0] <= 0 && yr[1] >= 0) {
      ctx.beginPath();
      ctx.moveTo(m.l, py(0));
      ctx.lineTo(W - m.r, py(0));
      ctx.stroke();
    }
    if (xr[0] <= 0 && xr[1] >= 0) {
      ctx.beginPath();
      ctx.moveTo(px(0), m.t);
      ctx.lineTo(px(0), H - m.b);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    ctx.fillStyle = colors.text;
    ctx.font = "10px system-ui,sans-serif";
    ctx.textAlign = "center";
    ctx.globalAlpha = 0.7;
    ctx.fillText(xlabel, (m.l + W - m.r) / 2, H - 5);
    ctx.save();
    ctx.translate(11, (m.t + H - m.b) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(ylabel, 0, 0);
    ctx.restore();
    ctx.globalAlpha = 1;

    return { px, py };
  }

  function drawFuncPlot(canvas, dpr, selected, colors, useDeriv) {
    const W = 340;
    const H = 240;
    const m = { t: 14, r: 14, b: 26, l: 30 };
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const xr = [-6, 6];
    const yr = useDeriv ? [-0.15, 1.25] : [-1.2, 3];
    const { px, py } = plotFrame(
      ctx,
      W,
      H,
      m,
      colors,
      xr,
      yr,
      "input z",
      useDeriv ? "slope f′(z)" : "output f(z)"
    );

    for (const key of ORDER) {
      if (!selected[key]) continue;
      const fn = FUNCS[key];
      const g = useDeriv ? fn.df : fn.f;
      ctx.beginPath();
      let first = true;
      for (let i = 0; i <= 240; i++) {
        const z = xr[0] + (i / 240) * (xr[1] - xr[0]);
        const X = px(z);
        const Y = py(Math.max(yr[0], Math.min(yr[1], g(z))));
        if (first) {
          ctx.moveTo(X, Y);
          first = false;
        } else ctx.lineTo(X, Y);
      }
      ctx.strokeStyle = fn.color;
      ctx.lineWidth = 2.2;
      ctx.stroke();
    }
  }

  function drawDecayPlot(canvas, dpr, selected, colors, depth) {
    const W = 340;
    const H = 240;
    const m = { t: 14, r: 14, b: 26, l: 38 };
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const xr = [1, Math.max(2, depth)];
    const yr = [0, 1.05];
    const { px, py } = plotFrame(
      ctx,
      W,
      H,
      m,
      colors,
      xr,
      yr,
      "network depth (layers)",
      "best-case gradient"
    );

    for (const key of ORDER) {
      if (!selected[key]) continue;
      const fn = FUNCS[key];
      const base = Math.min(1, fn.maxSlope);
      ctx.beginPath();
      for (let n = 1; n <= depth; n++) {
        const val = Math.pow(base, n - 1);
        const X = px(n);
        const Y = py(Math.max(yr[0], Math.min(yr[1], val)));
        if (n === 1) ctx.moveTo(X, Y);
        else ctx.lineTo(X, Y);
      }
      ctx.strokeStyle = fn.color;
      ctx.lineWidth = 2.2;
      ctx.stroke();

      const endVal = Math.pow(base, depth - 1);
      ctx.beginPath();
      ctx.arc(px(depth), py(Math.min(yr[1], endVal)), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = fn.color;
      ctx.fill();
    }
  }

  function createWidget(container) {
    const selected = {
      sigmoid: true,
      tanh: true,
      relu: true,
      leaky: false,
      gelu: false,
    };
    const state = { depth: 20 };

    const chips = ORDER.map(key => {
      const fn = FUNCS[key];
      return `<label style="display:inline-flex;align-items:center;gap:0.35rem;cursor:pointer;padding:0.25rem 0.5rem;border:1px solid var(--border,rgba(120,120,120,0.3));border-radius:999px;font-size:0.82rem;">
        <input type="checkbox" data-fn="${key}" ${selected[key] ? "checked" : ""} style="accent-color:${fn.color};" />
        <span style="width:10px;height:3px;background:${fn.color};display:inline-block;border-radius:2px;"></span>${fn.label}
      </label>`;
    }).join("");

    container.innerHTML = `
      <div class="activation-gradient-viz" style="margin:1.5rem 0;font-family:system-ui,sans-serif;font-size:0.9rem;">
        <div style="display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center;margin-bottom:0.85rem;">
          ${chips}
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:1rem;justify-content:center;">
          <div style="flex:1 1 300px;max-width:360px;text-align:center;">
            <div style="font-weight:600;margin-bottom:0.3rem;font-size:0.85rem;">The activation f(z)</div>
            <canvas data-canvas="func" width="340" height="240" style="width:100%;height:auto;border-radius:8px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
          </div>
          <div style="flex:1 1 300px;max-width:360px;text-align:center;">
            <div style="font-weight:600;margin-bottom:0.3rem;font-size:0.85rem;">Its slope f′(z) — this is what backprop multiplies</div>
            <canvas data-canvas="deriv" width="340" height="240" style="width:100%;height:auto;border-radius:8px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
          </div>
        </div>

        <div style="display:flex;flex-wrap:wrap;gap:1rem;justify-content:center;align-items:flex-start;margin-top:1rem;">
          <div style="flex:1 1 300px;max-width:360px;text-align:center;">
            <div style="font-weight:600;margin-bottom:0.3rem;font-size:0.85rem;">Gradient after N layers (best case)</div>
            <canvas data-canvas="decay" width="340" height="240" style="width:100%;height:auto;border-radius:8px;border:1px solid var(--border,rgba(120,120,120,0.25));"></canvas>
          </div>
          <div style="flex:1 1 260px;max-width:320px;">
            <label style="display:flex;flex-direction:column;gap:0.25rem;margin-bottom:0.75rem;">
              <span style="opacity:0.8;">Network depth: <b data-out="depth">20</b> layers</span>
              <input data-ctl="depth" type="range" min="2" max="40" step="1" value="20" style="accent-color:var(--accent);" />
            </label>
            <div data-out="notes" style="font-size:0.82rem;line-height:1.5;"></div>
          </div>
        </div>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const funcCanvas = el('[data-canvas="func"]');
    const derivCanvas = el('[data-canvas="deriv"]');
    const decayCanvas = el('[data-canvas="decay"]');
    const funcCtx = makeContext(funcCanvas, 340, 240);
    const derivCtx = makeContext(derivCanvas, 340, 240);
    const decayCtx = makeContext(decayCanvas, 340, 240);

    function renderNotes() {
      el('[data-out="notes"]').innerHTML = ORDER.filter(k => selected[k])
        .map(k => {
          const fn = FUNCS[k];
          const end = Math.pow(Math.min(1, fn.maxSlope), state.depth - 1);
          const endStr =
            end < 1e-4 ? end.toExponential(1) : end.toFixed(end < 1 ? 3 : 2);
          return `<div style="margin-bottom:0.5rem;">
            <span style="width:10px;height:3px;background:${fn.color};display:inline-block;border-radius:2px;vertical-align:middle;"></span>
            <b>${fn.label}</b> — ${fn.note}. After ${state.depth} layers: <b>${endStr}×</b>.
          </div>`;
        })
        .join("");
    }

    function render() {
      const colors = getColors();
      drawFuncPlot(funcCanvas, funcCtx.dpr, selected, colors, false);
      drawFuncPlot(derivCanvas, derivCtx.dpr, selected, colors, true);
      drawDecayPlot(decayCanvas, decayCtx.dpr, selected, colors, state.depth);
      renderNotes();
    }

    container.querySelectorAll("[data-fn]").forEach(cb => {
      cb.addEventListener("change", e => {
        selected[e.target.getAttribute("data-fn")] = e.target.checked;
        render();
      });
    });
    el('[data-ctl="depth"]').addEventListener("input", e => {
      state.depth = parseInt(e.target.value, 10);
      el('[data-out="depth"]').textContent = state.depth;
      render();
    });

    const themeObserver = new MutationObserver(() => {
      if (!document.body.contains(container)) {
        themeObserver.disconnect();
        return;
      }
      render();
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
        '.viz-activation-gradient[data-viz="activation-gradient"]'
      )
      .forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
