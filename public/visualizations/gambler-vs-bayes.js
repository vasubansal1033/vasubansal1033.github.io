(function () {
  const D3_URL = "https://cdn.jsdelivr.net/npm/d3@7/+esm";
  const MAX_HISTORY = 40;
  const initialized = new WeakSet();
  let d3Promise = null;

  function loadD3() {
    if (!d3Promise) {
      d3Promise = import(/* @vite-ignore */ D3_URL);
    }
    return d3Promise;
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
      bg: isDark ? "rgba(15,23,42,0.35)" : "rgba(248,250,252,0.6)",
      win: "#22c55e",
      loss: "#ef4444",
      gambler: "#f97316",
      bayes: accent || "#3b82f6",
      trueP: "#8b5cf6",
    };
  }

  function gamblerDueMeter(streak) {
    if (streak <= 0) return 0.5;
    return Math.min(0.95, 0.5 + streak * 0.12);
  }

  function betaMean(wins, losses, alpha0 = 1, beta0 = 1) {
    return (alpha0 + wins) / (alpha0 + beta0 + wins + losses);
  }

  function createWidget(container) {
    const state = {
      trueP: 0.45,
      results: [],
      wins: 0,
      losses: 0,
      lossStreak: 0,
      playing: false,
      timer: null,
      history: [],
    };

    container.innerHTML = `
      <div class="gambler-vs-bayes" style="margin:1.5rem 0;font-family:system-ui,sans-serif;">
        <div style="display:flex;flex-wrap:wrap;gap:0.6rem;align-items:center;margin-bottom:0.85rem;">
          <button type="button" data-action="step" style="padding:0.35rem 0.75rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;">Step</button>
          <button type="button" data-action="play" style="padding:0.35rem 0.75rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;">▶ Play</button>
          <button type="button" data-action="reset" style="padding:0.35rem 0.75rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;">Reset</button>
        </div>
        <label style="display:block;font-size:0.85rem;margin-bottom:1rem;max-width:360px;">
          True skill (hidden) — P(team wins each independent match)
          <input type="range" data-input="truep" min="10" max="90" value="45" style="width:100%;margin-top:0.35rem;" />
          <span data-out="truep-val" style="opacity:0.8;">45%</span>
        </label>
        <div style="border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;padding:0.75rem;margin-bottom:1rem;">
          <div style="font-size:0.85rem;font-weight:600;margin-bottom:0.5rem;">Match sequence (each match is independent)</div>
          <svg data-svg="strip" viewBox="0 0 600 48" style="width:100%;height:auto;display:block;"></svg>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
          <div style="border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;padding:0.75rem;">
            <div style="font-weight:600;font-size:0.88rem;color:#f97316;margin-bottom:0.35rem;">Gambler's fallacy</div>
            <div style="font-size:0.8rem;opacity:0.85;margin-bottom:0.5rem;">"They're due for a win" — wrongly rises after a losing streak</div>
            <svg data-svg="gambler" viewBox="0 0 280 140" style="width:100%;height:auto;display:block;"></svg>
            <div data-out="gambler-val" style="margin-top:0.4rem;font-size:0.85rem;font-weight:700;color:#f97316;"></div>
          </div>
          <div style="border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;padding:0.75rem;">
            <div style="font-weight:600;font-size:0.88rem;margin-bottom:0.35rem;">Bayesian estimate</div>
            <div style="font-size:0.8rem;opacity:0.85;margin-bottom:0.5rem;">Beta posterior over stable skill — converges to true rate</div>
            <svg data-svg="bayes" viewBox="0 0 280 140" style="width:100%;height:auto;display:block;"></svg>
            <div data-out="bayes-val" style="margin-top:0.4rem;font-size:0.85rem;font-weight:700;"></div>
          </div>
        </div>
        <div data-out="note" style="margin-top:0.75rem;font-size:0.82rem;opacity:0.88;line-height:1.5;"></div>
      </div>
    `;

    const el = sel => container.querySelector(sel);
    const playBtn = el('[data-action="play"]');

    function reset() {
      state.playing = false;
      if (state.timer) clearTimeout(state.timer);
      state.timer = null;
      playBtn.textContent = "▶ Play";
      state.results = [];
      state.wins = 0;
      state.losses = 0;
      state.lossStreak = 0;
      state.history = [{ n: 0, gambler: 0.5, bayes: 0.5 }];
      render();
    }

    function step() {
      const win = Math.random() < state.trueP;
      state.results.push(win);
      if (state.results.length > MAX_HISTORY) state.results.shift();
      if (win) {
        state.wins++;
        state.lossStreak = 0;
      } else {
        state.losses++;
        state.lossStreak++;
      }
      const gambler = gamblerDueMeter(state.lossStreak);
      const bayes = betaMean(state.wins, state.losses);
      state.history.push({
        n: state.wins + state.losses,
        gambler,
        bayes,
      });
      if (state.history.length > MAX_HISTORY + 1) state.history.shift();
      render();
    }

    function stopPlay() {
      state.playing = false;
      if (state.timer) clearTimeout(state.timer);
      state.timer = null;
      playBtn.textContent = "▶ Play";
    }

    function tick() {
      if (!state.playing || !document.body.contains(container))
        return stopPlay();
      step();
      state.timer = setTimeout(tick, 450);
    }

    function render() {
      if (!document.body.contains(container)) return;
      const colors = getColors();
      el('[data-out="truep-val"]').textContent =
        `${Math.round(state.trueP * 100)}%`;

      const gamblerNow =
        state.history.length > 0
          ? state.history[state.history.length - 1].gambler
          : 0.5;
      const bayesNow =
        state.history.length > 0
          ? state.history[state.history.length - 1].bayes
          : 0.5;

      el('[data-out="gambler-val"]').textContent =
        `P(win next) ≈ ${Math.round(gamblerNow * 100)}%`;
      el('[data-out="bayes-val"]').textContent =
        `Estimated skill ≈ ${Math.round(bayesNow * 100)}%`;
      el('[data-out="note"]').textContent =
        state.lossStreak > 0
          ? `After ${state.lossStreak} loss${state.lossStreak > 1 ? "es" : ""} in a row, the gambler's meter jumps — but each match is still independent at ${Math.round(state.trueP * 100)}%. Bayes quietly updates a belief about stable skill.`
          : `Outcomes don't cause outcomes. Bayes uses history to estimate an underlying parameter (skill), not to predict that a streak must reverse.`;

      loadD3().then(d3 => {
        if (!document.body.contains(container)) return;

        const stripSvg = d3.select(el('[data-svg="strip"]'));
        stripSvg.selectAll("*").remove();
        stripSvg
          .append("rect")
          .attr("width", 600)
          .attr("height", 48)
          .attr("rx", 6)
          .attr("fill", colors.bg);

        const n = state.results.length;
        const cellW = Math.min(14, 580 / Math.max(n, 1));
        stripSvg
          .selectAll("rect.r")
          .data(state.results)
          .join("rect")
          .attr("x", (_d, i) => 10 + i * (cellW + 2))
          .attr("y", 12)
          .attr("width", cellW)
          .attr("height", 24)
          .attr("rx", 3)
          .attr("fill", d => (d ? colors.win : colors.loss));

        if (n === 0) {
          stripSvg
            .append("text")
            .attr("x", 300)
            .attr("y", 30)
            .attr("text-anchor", "middle")
            .attr("fill", colors.muted)
            .attr("font-size", 12)
            .text("Press Step or Play");
        }

        function drawMiniChart(sel, key, lineColor, showTrue) {
          const W = 280;
          const H = 140;
          const m = { l: 36, r: 10, t: 14, b: 24 };
          const svg = d3.select(sel);
          svg.selectAll("*").remove();
          svg
            .append("rect")
            .attr("width", W)
            .attr("height", H)
            .attr("rx", 6)
            .attr("fill", colors.bg);

          const hist = state.history;
          const x = d3
            .scaleLinear()
            .domain([0, Math.max(hist.length - 1, 1)])
            .range([m.l, W - m.r]);
          const y = d3
            .scaleLinear()
            .domain([0, 1])
            .range([H - m.b, m.t]);

          if (showTrue) {
            svg
              .append("line")
              .attr("x1", m.l)
              .attr("y1", y(state.trueP))
              .attr("x2", W - m.r)
              .attr("y2", y(state.trueP))
              .attr("stroke", colors.trueP)
              .attr("stroke-dasharray", "4 3")
              .attr("stroke-width", 1.5)
              .attr("opacity", 0.7);
          }

          const line = d3
            .line()
            .x((_d, i) => x(i))
            .y(d => y(d[key]))
            .curve(d3.curveMonotoneX);

          if (hist.length > 1) {
            svg
              .append("path")
              .datum(hist)
              .attr("fill", "none")
              .attr("stroke", lineColor)
              .attr("stroke-width", 2)
              .attr("d", line);
          }

          const last = hist[hist.length - 1];
          svg
            .append("circle")
            .attr("cx", x(hist.length - 1))
            .attr("cy", y(last[key]))
            .attr("r", 5)
            .attr("fill", lineColor);
        }

        drawMiniChart(
          el('[data-svg="gambler"]'),
          "gambler",
          colors.gambler,
          false
        );
        drawMiniChart(el('[data-svg="bayes"]'), "bayes", colors.bayes, true);
      });
    }

    el('[data-input="truep"]').addEventListener("input", e => {
      state.trueP = parseInt(e.target.value, 10) / 100;
      render();
    });
    el('[data-action="step"]').addEventListener("click", () => {
      stopPlay();
      step();
    });
    el('[data-action="play"]').addEventListener("click", () => {
      if (state.playing) return stopPlay();
      state.playing = true;
      playBtn.textContent = "⏸ Pause";
      tick();
    });
    el('[data-action="reset"]').addEventListener("click", reset);

    const themeObserver = new MutationObserver(() => {
      if (!document.body.contains(container)) {
        themeObserver.disconnect();
        return stopPlay();
      }
      if (!state.playing) render();
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    reset();
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document
      .querySelectorAll('.viz-gambler-vs-bayes[data-viz="gambler-vs-bayes"]')
      .forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
