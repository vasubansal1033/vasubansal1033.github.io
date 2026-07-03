(function () {
  const initialized = new WeakSet();

  /* ---------- the symmetric loss landscape ----------
     Two hidden neurons, each summarized by one scalar parameter (theta1, theta2).
     The network's function depends only on the *set* {theta1, theta2}, so the loss
     is invariant under the swap (theta1, theta2) -> (theta2, theta1). That single
     permutation symmetry carves TWO identical valleys, mirror images across the
     diagonal theta1 = theta2. With h neurons there would be h! such valleys; we can
     only draw two parameters, so we draw the two. */

  const R = 4; // parameter range [-R, R]
  const N = 46; // grid resolution
  const TA = -2.0;
  const TB = 2.0;
  const S = 0.95;
  const HSCALE = 3.0; // vertical exaggeration of the loss

  const bump = (t, c) => {
    const d = t - c;
    return Math.exp(-(d * d) / (2 * S * S));
  };

  function lossFn(a, b) {
    const match = bump(a, TA) * bump(b, TB) + bump(a, TB) * bump(b, TA);
    const bowl = 0.12 * ((a * a + b * b) / (R * R));
    return 1 - match + bowl;
  }

  const MINIMA = [
    { a: TA, b: TB, label: "minimum" },
    { a: TB, b: TA, label: "mirror twin" },
  ];

  /* ---------- colour ---------- */

  function hexToRgb(h) {
    const n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function ramp(t, stops) {
    t = Math.max(0, Math.min(1, t));
    const seg = t * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(seg));
    const f = seg - i;
    const c0 = stops[i];
    const c1 = stops[i + 1];
    return [
      c0[0] + (c1[0] - c0[0]) * f,
      c0[1] + (c1[1] - c0[1]) * f,
      c0[2] + (c1[2] - c0[2]) * f,
    ];
  }

  function getColors() {
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    const styles = getComputedStyle(document.documentElement);
    return {
      isDark,
      text: styles.getPropertyValue("--foreground").trim() || "#334155",
      accent:
        styles.getPropertyValue("--accent").trim() ||
        (isDark ? "#60a5fa" : "#2563eb"),
      axis: isDark ? "rgba(148,163,184,0.55)" : "rgba(100,116,139,0.55)",
      stops: (isDark
        ? ["#1e3a8a", "#0e7490", "#15803d", "#ca8a04", "#b91c1c"]
        : ["#2563eb", "#06b6d4", "#22c55e", "#eab308", "#ef4444"]
      ).map(hexToRgb),
      diag: isDark ? "#f8fafc" : "#0f172a",
    };
  }

  /* ---------- widget ---------- */

  function createWidget(container) {
    const W = 660;
    const H = 440;

    container.innerHTML = `
      <div class="permutation-landscape-viz" style="margin:1.5rem 0;font-family:system-ui,sans-serif;font-size:0.9rem;">
        <div style="display:flex;flex-wrap:wrap;gap:0.6rem 1rem;align-items:center;margin-bottom:0.75rem;">
          <button type="button" data-action="swap" style="padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;">⇄ Swap the two neurons</button>
          <label style="display:inline-flex;align-items:center;gap:0.4rem;cursor:pointer;">
            <input data-ctl="rotate" type="checkbox" checked style="accent-color:var(--accent);" /> auto-rotate
          </label>
          <button type="button" data-action="reset" style="padding:0.45rem 0.9rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;">↺ Reset view</button>
          <span style="opacity:0.7;font-size:0.82rem;">drag to orbit</span>
        </div>

        <canvas data-canvas="land" width="${W}" height="${H}" style="width:100%;height:auto;max-width:${W}px;display:block;margin:0 auto;border-radius:12px;border:1px solid var(--border,rgba(120,120,120,0.25));cursor:grab;touch-action:none;"></canvas>

        <div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.5rem;justify-content:center;margin-top:0.6rem;font-size:0.85rem;">
          <span>you are in the <b data-out="which">left</b> valley</span>
          <span>loss here: <b data-out="loss" style="font-variant-numeric:tabular-nums;">—</b></span>
          <span>loss in the twin valley: <b data-out="twin" style="font-variant-numeric:tabular-nums;">—</b></span>
        </div>
        <p style="text-align:center;opacity:0.7;font-size:0.82rem;margin:0.5rem auto 0;max-width:44rem;line-height:1.5;">
          Two neurons &rarr; <b>2! = 2</b> identical valleys, reflected across the dashed diagonal (the mirror plane <span style="white-space:nowrap;">&theta;<sub>1</sub> = &theta;<sub>2</sub></span>). Swapping the neurons hops you over the ridge into the twin — same shape, same depth, same network.
        </p>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const canvas = el('[data-canvas="land"]');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d");

    // precompute vertex grid
    const xs = new Float64Array(N + 1);
    for (let i = 0; i <= N; i++) xs[i] = -R + (2 * R * i) / N;
    const zGrid = [];
    for (let i = 0; i <= N; i++) {
      const row = new Float64Array(N + 1);
      for (let j = 0; j <= N; j++) row[j] = lossFn(xs[i], xs[j]) * HSCALE;
      zGrid.push(row);
    }

    const light = (() => {
      const v = [0.45, 0.4, 0.8];
      const len = Math.hypot(v[0], v[1], v[2]);
      return [v[0] / len, v[1] / len, v[2] / len];
    })();

    const state = {
      az: -0.9,
      el: (58 * Math.PI) / 180,
      autoRotate: true,
      dragging: false,
      px: 0,
      py: 0,
      ball: { a: TA, b: TB },
      anim: null, // {t, from, to}
      raf: null,
    };

    const scale = (W * 0.5) / (R * 1.55);
    const cx = W / 2;
    const cy = H * 0.6;

    function project(x, y, z) {
      const ca = Math.cos(state.az);
      const sa = Math.sin(state.az);
      const se = Math.sin(state.el);
      const ce = Math.cos(state.el);
      const xr = x * ca - y * sa;
      const yr = x * sa + y * ca;
      return {
        sx: cx + xr * scale,
        sy: cy - (yr * se + z * ce) * scale,
        depth: yr,
      };
    }

    function draw() {
      const colors = getColors();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      // project all vertices
      const proj = [];
      for (let i = 0; i <= N; i++) {
        const row = [];
        for (let j = 0; j <= N; j++)
          row.push(project(xs[i], xs[j], zGrid[i][j]));
        proj.push(row);
      }

      // ground axes (z = 0 plane)
      ctx.lineWidth = 1;
      ctx.strokeStyle = colors.axis;
      const ax = [
        [project(-R, 0, 0), project(R, 0, 0)],
        [project(0, -R, 0), project(0, R, 0)],
      ];
      for (const [p, q] of ax) {
        ctx.beginPath();
        ctx.moveTo(p.sx, p.sy);
        ctx.lineTo(q.sx, q.sy);
        ctx.stroke();
      }

      // build + sort cells (painter's algorithm, far first)
      const cells = [];
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          const d =
            proj[i][j].depth +
            proj[i + 1][j].depth +
            proj[i][j + 1].depth +
            proj[i + 1][j + 1].depth;
          cells.push({ i, j, d });
        }
      }
      cells.sort((p, q) => q.d - p.d);

      let minL = Infinity;
      let maxL = -Infinity;
      for (let i = 0; i <= N; i++)
        for (let j = 0; j <= N; j++) {
          const v = zGrid[i][j] / HSCALE;
          if (v < minL) minL = v;
          if (v > maxL) maxL = v;
        }
      const span = maxL - minL || 1;

      for (const { i, j } of cells) {
        const a = proj[i][j];
        const b = proj[i + 1][j];
        const c = proj[i + 1][j + 1];
        const e = proj[i][j + 1];

        // world-space normal for shading
        const p0 = [xs[i], xs[j], zGrid[i][j]];
        const p1 = [xs[i + 1], xs[j], zGrid[i + 1][j]];
        const p2 = [xs[i], xs[j + 1], zGrid[i][j + 1]];
        const u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        const v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let nx = u[1] * v[2] - u[2] * v[1];
        let ny = u[2] * v[0] - u[0] * v[2];
        let nz = u[0] * v[1] - u[1] * v[0];
        const nl = Math.hypot(nx, ny, nz) || 1;
        nx /= nl;
        ny /= nl;
        nz /= nl;
        if (nz < 0) {
          nx = -nx;
          ny = -ny;
          nz = -nz;
        }
        const bright =
          0.62 +
          0.38 * Math.max(0, nx * light[0] + ny * light[1] + nz * light[2]);

        const lv =
          (zGrid[i][j] +
            zGrid[i + 1][j] +
            zGrid[i][j + 1] +
            zGrid[i + 1][j + 1]) /
          (4 * HSCALE);
        const col = ramp((lv - minL) / span, colors.stops);
        ctx.fillStyle = `rgb(${(col[0] * bright) | 0},${(col[1] * bright) | 0},${(col[2] * bright) | 0})`;
        ctx.strokeStyle = ctx.fillStyle;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(a.sx, a.sy);
        ctx.lineTo(b.sx, b.sy);
        ctx.lineTo(c.sx, c.sy);
        ctx.lineTo(e.sx, e.sy);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // mirror diagonal theta1 = theta2, riding the ridge
      ctx.setLineDash([6, 5]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = colors.diag;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      for (let k = 0; k <= 120; k++) {
        const t = -R + (2 * R * k) / 120;
        const p = project(t, t, lossFn(t, t) * HSCALE);
        if (k === 0) ctx.moveTo(p.sx, p.sy);
        else ctx.lineTo(p.sx, p.sy);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      // minima pins
      ctx.font = "600 11px system-ui,sans-serif";
      ctx.textAlign = "center";
      for (const m of MINIMA) {
        const p = project(m.a, m.b, lossFn(m.a, m.b) * HSCALE);
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, 4, 0, Math.PI * 2);
        ctx.fillStyle = colors.diag;
        ctx.fill();
      }

      // the ball (current solution)
      const bz = lossFn(state.ball.a, state.ball.b) * HSCALE;
      const bp = project(state.ball.a, state.ball.b, bz);
      const g = ctx.createRadialGradient(
        bp.sx - 3,
        bp.sy - 3,
        1,
        bp.sx,
        bp.sy,
        10
      );
      g.addColorStop(0, "#ffffff");
      g.addColorStop(1, colors.accent);
      ctx.beginPath();
      ctx.arc(bp.sx, bp.sy, 8, 0, Math.PI * 2);
      ctx.fillStyle = g;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = colors.diag;
      ctx.stroke();

      // axis labels
      ctx.fillStyle = colors.text;
      ctx.globalAlpha = 0.75;
      ctx.font = "italic 12px system-ui,sans-serif";
      const lx = project(R, 0, 0);
      const ly = project(0, R, 0);
      ctx.fillText("θ₁ (neuron A)", lx.sx, lx.sy + 14);
      ctx.fillText("θ₂ (neuron B)", ly.sx, ly.sy + 14);
      ctx.globalAlpha = 1;

      // readouts
      const cur = lossFn(state.ball.a, state.ball.b);
      el('[data-out="loss"]').textContent = cur.toFixed(4);
      el('[data-out="twin"]').textContent = lossFn(
        state.ball.b,
        state.ball.a
      ).toFixed(4);
      el('[data-out="which"]').textContent =
        state.ball.a < state.ball.b ? "left" : "right";
    }

    function loop() {
      if (!document.body.contains(container)) {
        stop();
        return;
      }
      if (state.autoRotate && !state.dragging) state.az += 0.006;
      if (state.anim) {
        state.anim.t += 0.028;
        const s = Math.min(1, state.anim.t);
        const e = s < 0.5 ? 2 * s * s : 1 - Math.pow(-2 * s + 2, 2) / 2;
        state.ball.a =
          state.anim.from.a + (state.anim.to.a - state.anim.from.a) * e;
        state.ball.b =
          state.anim.from.b + (state.anim.to.b - state.anim.from.b) * e;
        if (s >= 1) {
          state.ball = { ...state.anim.to };
          state.anim = null;
        }
      }
      draw();
      if (state.autoRotate || state.anim || state.dragging) {
        state.raf = requestAnimationFrame(loop);
      } else {
        state.raf = null;
      }
    }

    function start() {
      if (!state.raf) state.raf = requestAnimationFrame(loop);
    }
    function stop() {
      if (state.raf) cancelAnimationFrame(state.raf);
      state.raf = null;
    }

    // interactions
    canvas.addEventListener("pointerdown", e => {
      state.dragging = true;
      state.px = e.clientX;
      state.py = e.clientY;
      canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = "grabbing";
      start();
    });
    canvas.addEventListener("pointermove", e => {
      if (!state.dragging) return;
      state.az += (e.clientX - state.px) * 0.008;
      state.el = Math.max(
        0.25,
        Math.min(1.45, state.el + (e.clientY - state.py) * 0.006)
      );
      state.px = e.clientX;
      state.py = e.clientY;
      draw();
    });
    const endDrag = () => {
      state.dragging = false;
      canvas.style.cursor = "grab";
      if (state.autoRotate || state.anim) start();
    };
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    el('[data-action="swap"]').addEventListener("click", () => {
      if (state.anim) return;
      state.anim = {
        t: 0,
        from: { ...state.ball },
        to: { a: state.ball.b, b: state.ball.a },
      };
      start();
    });
    el('[data-ctl="rotate"]').addEventListener("change", e => {
      state.autoRotate = e.target.checked;
      if (state.autoRotate) start();
    });
    el('[data-action="reset"]').addEventListener("click", () => {
      state.az = -0.9;
      state.el = (58 * Math.PI) / 180;
      draw();
    });

    const themeObserver = new MutationObserver(() => {
      if (!document.body.contains(container)) {
        themeObserver.disconnect();
        return;
      }
      draw();
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    draw();
    start();
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document
      .querySelectorAll(
        '.viz-permutation-landscape[data-viz="permutation-landscape"]'
      )
      .forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
