(function () {
  const D3_URL = "https://cdn.jsdelivr.net/npm/d3@7/+esm";
  const GRID_N = 1000;
  const initialized = new WeakSet();
  let d3Promise = null;

  function loadD3() {
    if (!d3Promise) {
      d3Promise = import(/* @vite-ignore */ D3_URL);
    }
    return d3Promise;
  }

  const SCENARIOS = {
    cab: {
      label: "Cab problem",
      rare: "Blue cab",
      common: "Green cab",
      witness: "Witness says blue",
      rareHue: 220,
      commonHue: 145,
    },
    swiggy: {
      label: "Swiggy delivery",
      rare: "Bicycle",
      common: "Bike",
      witness: "Customer says bicycle",
      rareHue: 35,
      commonHue: 200,
    },
  };

  function bayesPosterior(prior, acc) {
    const num = acc * prior;
    const den = num + (1 - acc) * (1 - prior);
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
      border: isDark ? "rgba(148,163,184,0.25)" : "rgba(100,116,139,0.25)",
      muted: isDark ? "rgba(148,163,184,0.75)" : "rgba(100,116,139,0.85)",
      bg: isDark ? "rgba(15,23,42,0.35)" : "rgba(248,250,252,0.6)",
      gut: "#f97316",
      bayes: accent || "#3b82f6",
    };
  }

  function buildPopulation(prior, acc, n) {
    const rareCount = Math.round(n * prior);
    const commonCount = n - rareCount;
    const tp = Math.round(rareCount * acc);
    const fp = Math.round(commonCount * (1 - acc));
    const cells = [];
    let i = 0;
    for (let k = 0; k < tp; k++)
      cells.push({ actual: "rare", witness: true, i: i++ });
    for (let k = 0; k < rareCount - tp; k++)
      cells.push({ actual: "rare", witness: false, i: i++ });
    for (let k = 0; k < fp; k++)
      cells.push({ actual: "common", witness: true, i: i++ });
    for (let k = 0; k < commonCount - fp; k++)
      cells.push({ actual: "common", witness: false, i: i++ });
    return { cells, rareCount, commonCount, tp, fp };
  }

  function createWidget(container) {
    const state = {
      scenario: "cab",
      prior: 0.15,
      acc: 0.8,
    };

    container.innerHTML = `
      <div class="base-rate-explorer" style="margin:1.5rem 0;font-family:system-ui,sans-serif;">
        <div style="display:flex;flex-wrap:wrap;gap:0.6rem;align-items:center;margin-bottom:0.85rem;">
          <button type="button" data-scenario="cab" style="padding:0.35rem 0.75rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;">Cab problem</button>
          <button type="button" data-scenario="swiggy" style="padding:0.35rem 0.75rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;">Swiggy delivery</button>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin-bottom:1rem;">
          <label style="font-size:0.85rem;">
            <span data-out="prior-label">Base rate (blue cabs)</span>
            <input type="range" data-input="prior" min="1" max="50" value="15" style="width:100%;margin-top:0.35rem;" />
            <span data-out="prior-val" style="opacity:0.8;">15%</span>
          </label>
          <label style="font-size:0.85rem;">
            Witness accuracy
            <input type="range" data-input="acc" min="50" max="99" value="80" style="width:100%;margin-top:0.35rem;" />
            <span data-out="acc-val" style="opacity:0.8;">80%</span>
          </label>
        </div>
        <div style="display:grid;grid-template-columns:1fr minmax(240px,320px);gap:1rem;align-items:start;">
          <div style="border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;padding:0.75rem;">
            <div data-out="grid-title" style="font-size:0.85rem;font-weight:600;margin-bottom:0.5rem;"></div>
            <svg data-svg="grid" viewBox="0 0 500 320" style="width:100%;height:auto;display:block;"></svg>
            <div data-out="legend" style="display:flex;flex-wrap:wrap;gap:0.75rem;margin-top:0.5rem;font-size:0.78rem;opacity:0.85;"></div>
          </div>
          <div style="border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;padding:0.85rem;">
            <div style="font-weight:600;font-size:0.9rem;margin-bottom:0.65rem;">Your gut vs Bayes</div>
            <svg data-svg="bars" viewBox="0 0 300 180" style="width:100%;height:auto;display:block;"></svg>
            <div data-out="counts" style="margin-top:0.65rem;font-size:0.82rem;line-height:1.5;opacity:0.9;"></div>
            <div data-out="formula" style="margin-top:0.5rem;font-size:0.8rem;opacity:0.8;font-family:ui-monospace,monospace;"></div>
          </div>
        </div>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const btnCab = el('[data-scenario="cab"]');
    const btnSwiggy = el('[data-scenario="swiggy"]');

    function setScenario(key) {
      state.scenario = key;
      const active =
        "padding:0.35rem 0.75rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;";
      const idle =
        "padding:0.35rem 0.75rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;";
      btnCab.style.cssText = key === "cab" ? active : idle;
      btnSwiggy.style.cssText = key === "swiggy" ? active : idle;
      const sc = SCENARIOS[key];
      el('[data-out="prior-label"]').textContent =
        `Base rate (${sc.rare.toLowerCase()}s)`;
      render();
    }

    function render() {
      if (!document.body.contains(container)) return;
      const colors = getColors();
      const sc = SCENARIOS[state.scenario];
      const prior = state.prior;
      const acc = state.acc;
      const posterior = bayesPosterior(prior, acc);
      const pop = buildPopulation(prior, acc, GRID_N);
      const witnessTotal = pop.tp + pop.fp;

      el('[data-out="prior-val"]').textContent = `${Math.round(prior * 100)}%`;
      el('[data-out="acc-val"]').textContent = `${Math.round(acc * 100)}%`;
      el('[data-out="grid-title"]').textContent =
        `${GRID_N} ${state.scenario === "cab" ? "cabs" : "deliveries"} — ${sc.witness}`;

      loadD3().then(d3 => {
        if (!document.body.contains(container)) return;

        const cols = 50;
        const rows = GRID_N / cols;
        const cellW = 500 / cols;
        const cellH = 300 / rows;

        const gridSvg = d3.select(el('[data-svg="grid"]'));
        gridSvg.selectAll("*").remove();

        gridSvg
          .append("rect")
          .attr("width", 500)
          .attr("height", 300)
          .attr("rx", 8)
          .attr("fill", colors.bg);

        gridSvg
          .selectAll("rect.cell")
          .data(pop.cells)
          .join("rect")
          .attr("class", "cell")
          .attr("x", d => (d.i % cols) * cellW + 0.5)
          .attr("y", d => Math.floor(d.i / cols) * cellH + 0.5)
          .attr("width", cellW - 1)
          .attr("height", cellH - 1)
          .attr("rx", 1)
          .attr("fill", d => {
            const hue = d.actual === "rare" ? sc.rareHue : sc.commonHue;
            return `hsl(${hue} 55% ${d.actual === "rare" ? "48%" : "42%"})`;
          })
          .attr("stroke", d =>
            d.witness ? (colors.isDark ? "#f8fafc" : "#0f172a") : "none"
          )
          .attr("stroke-width", d => (d.witness ? 1.2 : 0));

        gridSvg
          .append("text")
          .attr("x", 8)
          .attr("y", 16)
          .attr("fill", colors.text)
          .attr("font-size", 11)
          .attr("opacity", 0.85)
          .text(`Outlined = ${sc.witness.toLowerCase()}`);

        el('[data-out="legend"]').innerHTML = `
          <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:hsl(${sc.rareHue} 55% 48%);vertical-align:middle;margin-right:4px;"></span>${sc.rare} (actually)</span>
          <span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:hsl(${sc.commonHue} 55% 42%);vertical-align:middle;margin-right:4px;"></span>${sc.common} (actually)</span>
          <span><span style="display:inline-block;width:10px;height:10px;border:1.5px solid ${colors.isDark ? "#f8fafc" : "#0f172a"};border-radius:2px;vertical-align:middle;margin-right:4px;"></span>Witness points here</span>
        `;

        const barData = [
          { label: "Your gut", value: acc, color: colors.gut },
          { label: "Bayes", value: posterior, color: colors.bayes },
        ];

        const barsSvg = d3.select(el('[data-svg="bars"]'));
        barsSvg.selectAll("*").remove();
        const m = { l: 72, r: 16, t: 12, b: 28 };
        const W = 300;
        const H = 180;

        barsSvg
          .append("rect")
          .attr("width", W)
          .attr("height", H)
          .attr("rx", 8)
          .attr("fill", colors.bg);

        const x = d3
          .scaleBand()
          .domain(barData.map(d => d.label))
          .range([m.l, W - m.r])
          .padding(0.35);
        const y = d3
          .scaleLinear()
          .domain([0, 1])
          .range([H - m.b, m.t]);

        barsSvg
          .selectAll("rect.bar")
          .data(barData)
          .join("rect")
          .attr("x", d => x(d.label))
          .attr("y", d => y(d.value))
          .attr("width", x.bandwidth())
          .attr("height", d => y(0) - y(d.value))
          .attr("fill", d => d.color)
          .attr("rx", 4);

        barsSvg
          .selectAll("text.val")
          .data(barData)
          .join("text")
          .attr("class", "val")
          .attr("x", d => x(d.label) + x.bandwidth() / 2)
          .attr("y", d => y(d.value) - 6)
          .attr("text-anchor", "middle")
          .attr("fill", colors.text)
          .attr("font-size", 12)
          .attr("font-weight", 700)
          .text(d => `${Math.round(d.value * 100)}%`);

        barsSvg
          .selectAll("text.lbl")
          .data(barData)
          .join("text")
          .attr("class", "lbl")
          .attr("x", d => x(d.label) + x.bandwidth() / 2)
          .attr("y", H - 8)
          .attr("text-anchor", "middle")
          .attr("fill", colors.muted)
          .attr("font-size", 11)
          .text(d => d.label);

        el('[data-out="counts"]').innerHTML = `
          Of every <strong>${witnessTotal}</strong> cases where the witness says "${sc.rare.toLowerCase()}":<br/>
          <strong>${pop.tp}</strong> are truly ${sc.rare.toLowerCase()} (true positive)<br/>
          <strong>${pop.fp}</strong> are actually ${sc.common.toLowerCase()} (false positive)<br/>
          So P(actually ${sc.rare.toLowerCase()} | witness) = ${pop.tp} / ${witnessTotal} ≈ <strong>${Math.round(posterior * 100)}%</strong>
        `;

        el('[data-out="formula"]').textContent =
          `(${acc.toFixed(2)} × ${prior.toFixed(2)}) / (${acc.toFixed(2)} × ${prior.toFixed(2)} + ${(1 - acc).toFixed(2)} × ${(1 - prior).toFixed(2)})`;
      });
    }

    el('[data-input="prior"]').addEventListener("input", e => {
      state.prior = parseInt(e.target.value, 10) / 100;
      render();
    });
    el('[data-input="acc"]').addEventListener("input", e => {
      state.acc = parseInt(e.target.value, 10) / 100;
      render();
    });
    btnCab.addEventListener("click", () => setScenario("cab"));
    btnSwiggy.addEventListener("click", () => setScenario("swiggy"));

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

    setScenario("cab");
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document
      .querySelectorAll(
        '.viz-base-rate-explorer[data-viz="base-rate-explorer"]'
      )
      .forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
