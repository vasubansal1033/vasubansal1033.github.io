(function () {
  const initialized = new WeakSet();

  const MILESTONES = [
    {
      year: 1943,
      era: "roots",
      title: "The neuron as a switch",
      who: "McCulloch & Pitts",
      body: "Warren McCulloch and Walter Pitts show that a network of idealized all-or-nothing neurons — simple threshold switches — can compute any logical proposition. Thought as computation over switches: the founding idea.",
    },
    {
      year: 1948,
      era: "roots",
      title: "Cybernetics & feedback",
      who: "Norbert Wiener",
      body: "Born partly from WWII work on auto-aiming anti-aircraft guns (fire where the plane will be, and correct from error), Wiener generalizes self-correcting feedback into cybernetics — the study of control in animals and machines.",
    },
    {
      year: 1958,
      era: "spring",
      title: "The Perceptron learns",
      who: "Frank Rosenblatt",
      body: "Rosenblatt attaches adjustable weights to the switch and a rule to nudge them on every mistake — so experience reshapes the neuron. A sensation at the time. The linear model in S1-1 is essentially a modern perceptron.",
    },
    {
      year: 1969,
      era: "winter",
      title: "XOR and the first AI winter",
      who: "Minsky & Papert",
      body: "A careful proof: a single-layer perceptron cannot represent XOR, or anything not linearly separable — the exact wall the linear model hits above. Correct, influential, and devastating; funding dried up.",
    },
    {
      year: 1986,
      era: "spring",
      title: "Backpropagation thaws it",
      who: "Rumelhart, Hinton & Williams",
      body: "Send the error backward through many layers and update every weight. Combined with the smooth sigmoid, multi-layer nets can finally bend space — exactly what the ReLU model does in S1-1.",
    },
    {
      year: 1995,
      era: "winter",
      title: "Vanishing gradients & the SVM era",
      who: "Cortes & Vapnik",
      body: "Sigmoid's flat tails make gradients vanish in deep stacks, so nets stay shallow. Support Vector Machines — clean, theory-backed, convex — dominate the decade while neural nets are dismissed as 'alchemy'.",
    },
    {
      year: 2011,
      era: "boom",
      title: "ReLU breaks the dam",
      who: "Glorot, Bordes & Bengio",
      body: "Blame the activation. max(0, z): zero below, a 45° line above. Its slope on the active side is exactly 1, so gradients pass through deep stacks without shrinking. The vanishing-gradient problem largely evaporates.",
    },
    {
      year: 2012,
      era: "boom",
      title: "AlexNet wins ImageNet",
      who: "Krizhevsky, Sutskever & Hinton",
      body: "ReLU + GPUs cut the ImageNet error rate so sharply the paradigm debate ends overnight. Within a couple of years the field is deep. The switch at the heart of it all is essentially the 1940s neuron, scaled.",
    },
  ];

  const ERA_HUE = {
    roots: 210,
    spring: 150,
    winter: 25,
    boom: 265,
  };

  const X0 = 55;
  const X1 = 865;
  const Y = 122;
  const YEAR_MIN = 1940;
  const YEAR_MAX = 2015;

  const xOf = year =>
    X0 + ((year - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)) * (X1 - X0);

  function buildSvg() {
    const winterA = xOf(1969);
    const winterB = xOf(1986);
    const dots = MILESTONES.map((m, i) => {
      const x = xOf(m.year);
      const above = i % 2 === 0;
      const hue = ERA_HUE[m.era];
      const labelY = above ? 74 : 170;
      const yearY = above ? 58 : 186;
      const stemY1 = above ? 92 : 152;
      return `
        <g data-idx="${i}" style="cursor:pointer;">
          <line x1="${x}" y1="${stemY1}" x2="${x}" y2="${Y}" style="stroke:hsl(${hue} 60% 55%);stroke-width:1.5;opacity:0.5;" />
          <text x="${x}" y="${yearY}" text-anchor="middle" style="fill:var(--foreground);font-size:12px;font-weight:700;opacity:0.9;">${m.year}</text>
          <text x="${x}" y="${labelY}" text-anchor="middle" style="fill:var(--foreground);font-size:10.5px;opacity:0.75;">${m.who}</text>
          <circle data-dot="${i}" cx="${x}" cy="${Y}" r="8" style="fill:hsl(${hue} 65% 52%);stroke:var(--background,#fff);stroke-width:2;" />
        </g>`;
    }).join("");

    return `
      <svg data-svg="timeline" viewBox="0 0 920 210" style="display:block;width:100%;height:auto;overflow:visible;">
        <rect x="${winterA}" y="30" width="${winterB - winterA}" height="180" rx="6"
          style="fill:hsl(25 70% 50% / 0.10);stroke:hsl(25 70% 50% / 0.3);stroke-dasharray:4 4;" />
        <text x="${(winterA + winterB) / 2}" y="26" text-anchor="middle" style="fill:var(--foreground);font-size:10.5px;font-style:italic;opacity:0.7;">first AI winter · shallow years</text>
        <line x1="${X0}" y1="${Y}" x2="${X1}" y2="${Y}" style="stroke:var(--foreground);stroke-width:2;opacity:0.25;" />
        ${dots}
      </svg>`;
  }

  function createWidget(container) {
    container.innerHTML = `
      <div class="dl-history-timeline" style="margin:1.5rem 0;font-family:system-ui,sans-serif;">
        <div style="display:flex;flex-wrap:wrap;gap:0.6rem;align-items:center;margin-bottom:0.5rem;">
          <button type="button" data-action="prev" style="padding:0.3rem 0.7rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;">‹ Prev</button>
          <button type="button" data-action="play" style="padding:0.3rem 0.7rem;border-radius:8px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer;font-size:0.85rem;font-weight:600;">▶ Play</button>
          <button type="button" data-action="next" style="padding:0.3rem 0.7rem;border-radius:8px;border:1px solid var(--accent);background:transparent;color:var(--accent);cursor:pointer;font-size:0.85rem;">Next ›</button>
          <span style="opacity:0.7;font-size:0.8rem;">click any milestone</span>
        </div>
        <div style="border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;padding:0.75rem 0.5rem 0.5rem;">
          ${buildSvg()}
        </div>
        <div data-out="detail" style="margin-top:0.85rem;padding:0.9rem 1rem;border:1px solid var(--border,rgba(120,120,120,0.25));border-radius:10px;min-height:5.5rem;"></div>
      </div>
    `;

    const state = { active: 0, playing: false, timer: null };
    const el = sel => container.querySelector(sel);
    const svg = el('[data-svg="timeline"]');
    const playBtn = el('[data-action="play"]');

    function renderDetail() {
      const m = MILESTONES[state.active];
      const hue = ERA_HUE[m.era];
      el('[data-out="detail"]').innerHTML = `
        <div style="display:flex;align-items:baseline;gap:0.6rem;flex-wrap:wrap;margin-bottom:0.35rem;">
          <span style="font-size:1.35rem;font-weight:800;color:hsl(${hue} 65% 52%);">${m.year}</span>
          <span style="font-weight:700;">${m.title}</span>
          <span style="opacity:0.7;font-size:0.85rem;">— ${m.who}</span>
        </div>
        <div style="opacity:0.9;line-height:1.55;font-size:0.9rem;">${m.body}</div>
      `;
      svg.querySelectorAll("[data-dot]").forEach(dot => {
        const i = parseInt(dot.getAttribute("data-dot"), 10);
        dot.setAttribute("r", i === state.active ? "11" : "8");
        dot.style.filter =
          i === state.active
            ? "drop-shadow(0 0 5px hsl(0 0% 50% / 0.6))"
            : "none";
        dot.style.opacity = i === state.active ? "1" : "0.85";
      });
    }

    function setActive(i) {
      state.active = (i + MILESTONES.length) % MILESTONES.length;
      renderDetail();
    }

    function stopPlay() {
      state.playing = false;
      if (state.timer) clearTimeout(state.timer);
      state.timer = null;
      playBtn.textContent = "▶ Play";
    }

    function tick() {
      if (!state.playing) return;
      if (!document.body.contains(container)) return stopPlay();
      if (state.active >= MILESTONES.length - 1) {
        stopPlay();
        return;
      }
      setActive(state.active + 1);
      state.timer = setTimeout(tick, 2600);
    }

    function play() {
      if (state.playing) return stopPlay();
      if (state.active >= MILESTONES.length - 1) setActive(0);
      state.playing = true;
      playBtn.textContent = "⏸ Pause";
      state.timer = setTimeout(tick, 2600);
    }

    svg.addEventListener("click", e => {
      const g = e.target.closest("[data-idx]");
      if (!g) return;
      stopPlay();
      setActive(parseInt(g.getAttribute("data-idx"), 10));
    });
    el('[data-action="next"]').addEventListener("click", () => {
      stopPlay();
      setActive(state.active + 1);
    });
    el('[data-action="prev"]').addEventListener("click", () => {
      stopPlay();
      setActive(state.active - 1);
    });
    playBtn.addEventListener("click", play);

    setActive(0);
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document
      .querySelectorAll('.viz-dl-history[data-viz="dl-history"]')
      .forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
