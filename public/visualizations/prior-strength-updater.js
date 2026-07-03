(function () {
  const D3_URL = "https://cdn.jsdelivr.net/npm/d3@7/+esm";
  const initialized = new WeakSet();
  let d3Promise = null;

  function loadD3() {
    if (!d3Promise) {
      d3Promise = import(/* @vite-ignore */ D3_URL);
    }
    return d3Promise;
  }

  function bayesPosterior(prior, acc, supports = true) {
    const likeH = supports ? acc : 1 - acc;
    const likeNotH = supports ? 1 - acc : acc;
    const num = likeH * prior;
    const den = num + likeNotH * (1 - prior);
    return den > 0 ? num / den : 0;
  }

  function getColors() {
    const isDark =
      document.documentElement.getAttribute("data-theme") === "dark";
    const styles = getComputedStyle(document.documentElement);
    const accent = styles.getPropertyValue("--accent").trim() || "#3b82f6";
    return {
      isDark,
      text: styles.getPropertyValue("--foreground").trim() || "#334155",
      accent,
      muted: isDark ? "rgba(148,163,184,0.75)" : "rgba(100,116,139,0.85)",
      grid: isDark ? "rgba(148,163,184,0.15)" : "rgba(100,116,139,0.15)",
      bg: isDark ? "rgba(15,23,42,0.35)" : "rgba(248,250,252,0.6)",
      prior: "#f97316",
      posterior: accent || "#3b82f6",
      curve: "#8b5cf6",
    };
  }

  function createWidget(container) {
    const state = {
      prior: 0.2,
      acc: 0.7,
      supports: true,
      animPrior: 0.2,
      animPosterior: 0.2,
      animating: false,
      animFrame: null,
    };

    container.innerHTML = `
      <div class="prior-strength-updater" style="margin:1.5rem 0;font-family:system-ui,sans-serif;">
        <div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-bottom:0.85rem;">
          <button type="button" data-preset="pak" style="padding:0.35rem 0.75rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;">India vs Pakistan (prior 20%)</button>
          <button type="button" data-preset="nz" style="padding:0.35rem 0.75rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;">India vs New Zealand (prior 50%)</button>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:1rem;">
          <label style="font-size:0.85rem;">
            Prior P(underdog wins)
            <input type="range" data-input="prior" min="5" max="95" value="20" style="width:100%;margin-top:0.35rem;" />
            <span data-out="prior-val" style="opacity:0.8;">20%</span>
          </label>
          <label style="font-size:0.85rem;">
            Analyst accuracy
            <input type="range" data-input="acc" min="55" max="95" value="70" style="width:100%;margin-top:0.35rem;" />
            <span data-out="acc-val" style="opacity:0.8;">70%</span>
          </label>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:0.6rem;align-items:center;margin-bottom:0.75rem;font-size:0.85rem;">
          <span>Evidence direction:</span>
          <button type="button" data-dir="supports" style="padding:0.3rem 0.65rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.82rem;">Supports underdog</button>
          <button type="button" data-dir="opposes" style="padding:0.3rem 0.65rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.82rem;">Opposes underdog</button>
        </div>
        <div style="border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;padding:0.75rem;">
          <svg data-svg="curve" viewBox="0 0 520 300" style="width:100%;height:auto;display:block;"></svg>
        </div>
        <div data-out="summary" style="margin-top:0.75rem;padding:0.85rem 1rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;font-size:0.88rem;line-height:1.55;"></div>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const btnSupports = el('[data-dir="supports"]');
    const btnOpposes = el('[data-dir="opposes"]');

    function setDirection(supports) {
      state.supports = supports;
      const active =
        "padding:0.3rem 0.65rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.82rem;";
      const idle =
        "padding:0.3rem 0.65rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.82rem;";
      btnSupports.style.cssText = supports ? active : idle;
      btnOpposes.style.cssText = supports ? idle : active;
      render();
    }

    function animateUpdate() {
      if (state.animFrame) cancelAnimationFrame(state.animFrame);
      const targetPrior = state.prior;
      const targetPosterior = bayesPosterior(
        state.prior,
        state.acc,
        state.supports
      );
      const startPrior = state.animPrior;
      const startPosterior = state.animPosterior;
      const t0 = performance.now();
      const duration = 700;
      state.animating = true;

      function frame(now) {
        if (!document.body.contains(container)) {
          state.animating = false;
          return;
        }
        const t = Math.min(1, (now - t0) / duration);
        const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        state.animPrior = startPrior + (targetPrior - startPrior) * ease;
        state.animPosterior =
          startPosterior + (targetPosterior - startPosterior) * ease;
        render(false);
        if (t < 1) {
          state.animFrame = requestAnimationFrame(frame);
        } else {
          state.animating = false;
          state.animPrior = targetPrior;
          state.animPosterior = targetPosterior;
          render(false);
        }
      }
      state.animFrame = requestAnimationFrame(frame);
    }

    function render(triggerAnim = false) {
      if (!document.body.contains(container)) return;
      const colors = getColors();
      const prior = state.prior;
      const acc = state.acc;
      const posterior = bayesPosterior(prior, acc, state.supports);
      const displayPrior = state.animating ? state.animPrior : prior;
      const displayPosterior = state.animating
        ? state.animPosterior
        : posterior;

      el('[data-out="prior-val"]').textContent = `${Math.round(prior * 100)}%`;
      el('[data-out="acc-val"]').textContent = `${Math.round(acc * 100)}%`;

      const delta = posterior - prior;
      const dir = state.supports ? "supports" : "opposes";
      el('[data-out="summary"]').innerHTML = `
        Prior: <strong>${Math.round(prior * 100)}%</strong> → Posterior: <strong>${Math.round(posterior * 100)}%</strong>
        (${delta >= 0 ? "+" : ""}${Math.round(delta * 100)} pp).
        Evidence ${dir} the underdog at <strong>${Math.round(acc * 100)}%</strong> accuracy.
        ${Math.abs(prior - 0.5) > 0.25 ? "A strong prior resists the update." : "A mid prior moves more with the same evidence."}
      `;

      loadD3().then(d3 => {
        if (!document.body.contains(container)) return;

        const W = 520;
        const H = 300;
        const m = { l: 48, r: 20, t: 24, b: 44 };
        const svg = d3.select(el('[data-svg="curve"]'));
        svg.selectAll("*").remove();

        svg
          .append("rect")
          .attr("width", W)
          .attr("height", H)
          .attr("rx", 8)
          .attr("fill", colors.bg);

        const x = d3
          .scaleLinear()
          .domain([0, 1])
          .range([m.l, W - m.r]);
        const y = d3
          .scaleLinear()
          .domain([0, 1])
          .range([H - m.b, m.t]);

        const pts = d3.range(0, 101).map(i => {
          const p = i / 100;
          return {
            prior: p,
            posterior: bayesPosterior(p, acc, state.supports),
          };
        });

        const line = d3
          .line()
          .x(d => x(d.prior))
          .y(d => y(d.posterior))
          .curve(d3.curveMonotoneX);

        svg
          .append("line")
          .attr("x1", x(0))
          .attr("y1", y(0))
          .attr("x2", x(1))
          .attr("y2", y(1))
          .attr("stroke", colors.muted)
          .attr("stroke-dasharray", "5 4")
          .attr("stroke-width", 1.5)
          .attr("opacity", 0.6);

        svg
          .append("path")
          .datum(pts)
          .attr("fill", "none")
          .attr("stroke", colors.curve)
          .attr("stroke-width", 2.5)
          .attr("d", line);

        svg
          .append("circle")
          .attr("cx", x(displayPrior))
          .attr("cy", y(displayPrior))
          .attr("r", 7)
          .attr("fill", colors.prior)
          .attr("stroke", colors.isDark ? "#0f172a" : "#fff")
          .attr("stroke-width", 2);

        svg
          .append("circle")
          .attr("cx", x(displayPrior))
          .attr("cy", y(displayPosterior))
          .attr("r", 7)
          .attr("fill", colors.posterior)
          .attr("stroke", colors.isDark ? "#0f172a" : "#fff")
          .attr("stroke-width", 2);

        svg
          .append("line")
          .attr("x1", x(displayPrior))
          .attr("y1", y(displayPrior))
          .attr("x2", x(displayPrior))
          .attr("y2", y(displayPosterior))
          .attr("stroke", colors.text)
          .attr("stroke-width", 1.5)
          .attr("opacity", 0.5);

        svg
          .append("text")
          .attr("x", x(0.02))
          .attr("y", y(0.98))
          .attr("fill", colors.muted)
          .attr("font-size", 11)
          .text("no update (y = x)");

        svg
          .append("text")
          .attr("x", W / 2)
          .attr("y", H - 10)
          .attr("text-anchor", "middle")
          .attr("fill", colors.text)
          .attr("font-size", 12)
          .text("Prior P(underdog wins)");

        svg
          .append("text")
          .attr("transform", `translate(14,${H / 2}) rotate(-90)`)
          .attr("text-anchor", "middle")
          .attr("fill", colors.text)
          .attr("font-size", 12)
          .text("Posterior");

        svg
          .append("text")
          .attr("x", x(displayPrior) + 10)
          .attr("y", y(displayPrior) - 8)
          .attr("fill", colors.prior)
          .attr("font-size", 11)
          .attr("font-weight", 600)
          .text(`prior ${Math.round(displayPrior * 100)}%`);

        svg
          .append("text")
          .attr("x", x(displayPrior) + 10)
          .attr("y", y(displayPosterior) + 14)
          .attr("fill", colors.posterior)
          .attr("font-size", 11)
          .attr("font-weight", 600)
          .text(`posterior ${Math.round(displayPosterior * 100)}%`);
      });

      if (triggerAnim) animateUpdate();
    }

    el('[data-input="prior"]').addEventListener("input", e => {
      state.prior = parseInt(e.target.value, 10) / 100;
      render(true);
    });
    el('[data-input="acc"]').addEventListener("input", e => {
      state.acc = parseInt(e.target.value, 10) / 100;
      render(true);
    });
    btnSupports.addEventListener("click", () => {
      setDirection(true);
      render(true);
    });
    btnOpposes.addEventListener("click", () => {
      setDirection(false);
      render(true);
    });
    el('[data-preset="pak"]').addEventListener("click", () => {
      state.prior = 0.2;
      state.acc = 0.7;
      state.supports = true;
      el('[data-input="prior"]').value = "20";
      el('[data-input="acc"]').value = "70";
      setDirection(true);
      render(true);
    });
    el('[data-preset="nz"]').addEventListener("click", () => {
      state.prior = 0.5;
      state.acc = 0.7;
      state.supports = true;
      el('[data-input="prior"]').value = "50";
      el('[data-input="acc"]').value = "70";
      setDirection(true);
      render(true);
    });

    const themeObserver = new MutationObserver(() => {
      if (!document.body.contains(container)) {
        themeObserver.disconnect();
        return;
      }
      if (!state.animating) render(false);
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    setDirection(true);
    render(false);
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document
      .querySelectorAll(
        '.viz-prior-strength-updater[data-viz="prior-strength-updater"]'
      )
      .forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
