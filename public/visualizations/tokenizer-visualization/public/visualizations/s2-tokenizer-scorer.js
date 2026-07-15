(function () {
  const initialized = new WeakSet();
  const LANG_LABELS = { en: "English", hi: "Hindi", te: "Telugu", pa: "Punjabi" };
  const PAGE_SIZES = [50, 100, 200, 500];
  const PLAYGROUND_SAMPLE = [
    "India is home to many languages and cultures.",
    "भारत एक विविधताओं वाला देश है।",
    "ਭਾਰਤ ਕਈ ਭਾਸ਼ਾਵਾਂ ਦਾ ਦੇਸ਼ ਹੈ।",
    "భారతదేశం అనేక భాషలకు నిలయం.",
  ].join("\n\n");
  const CHIP_HUES = [
    "#3b82f6",
    "#8b5cf6",
    "#ec4899",
    "#f97316",
    "#eab308",
    "#22c55e",
    "#14b8a6",
    "#06b6d4",
    "#6366f1",
    "#a855f7",
    "#f43f5e",
    "#84cc16",
  ];

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
      pass: isDark ? "#4ade80" : "#15803d",
      fail: isDark ? "#f87171" : "#b91c1c",
    };
  }

  function cardStyle(colors) {
    return `border:1px solid ${colors.border};border-radius:10px;padding:0.85rem;background:${colors.bg};`;
  }

  function escapeAttr(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function wikiLinkLabel(stats) {
    if (stats.page_title) return stats.page_title;
    try {
      const path = new URL(stats.url).pathname;
      return decodeURIComponent(path.replace(/^\/wiki\//, "")).replace(/_/g, " ");
    } catch (_err) {
      return stats.url;
    }
  }

  function wikiHref(stats) {
    if (stats.page_title) {
      try {
        const url = new URL(stats.url);
        return `${url.origin}/wiki/${stats.page_title.replace(/ /g, "_")}`;
      } catch (_err) {
        return stats.url;
      }
    }
    try {
      return decodeURI(stats.url);
    } catch (_err) {
      return stats.url;
    }
  }

  function escapeHtml(value) {
    return escapeAttr(value);
  }

  function actionButtonStyle(colors) {
    return `padding:0.35rem 0.7rem;border-radius:8px;border:1px solid ${colors.accent};background:transparent;color:${colors.accent};cursor:pointer;font-size:0.8rem;`;
  }

  function actionLinkStyle(colors) {
    return `${actionButtonStyle(colors)}text-decoration:none;display:inline-flex;align-items:center;`;
  }

  function dataFileUrl(dataBase, filename) {
    return `${String(dataBase).replace(/\/$/, "")}/${filename}`;
  }

  function downloadBlob(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function vocabToCsv(vocab) {
    const lines = ["id,lang,token"];
    for (const item of vocab) {
      const token = String(item.token).replace(/"/g, '""');
      lines.push(`${item.id},${item.lang},"${token}"`);
    }
    return lines.join("\n");
  }

  function fertilityCalculation(stats) {
    if (stats.fertility_calculation) return stats.fertility_calculation;
    const words = stats.pretoken_words ?? stats.words;
    const fertility = stats.fertility ?? 0;
    return `${stats.tokens} / ${words} = ${fertility.toFixed(4)}`;
  }

  async function loadData(base) {
    const root = base.replace(/\/$/, "");
    const reportUrl = `${root}/report.json`;
    const vocabUrl = `${root}/vocab.json`;
    const runtimeUrl = `${root}/s2_runtime.json`;
    const tokenizerUrl = `${root}/tokenizer.json`;
    const [reportRes, vocabRes, tokenizerRes] = await Promise.all([
      fetch(reportUrl),
      fetch(vocabUrl),
      fetch(tokenizerUrl),
    ]);
    if (!reportRes.ok || !vocabRes.ok) {
      throw new Error(
        `Failed to load widget data (${reportRes.status}/${vocabRes.status}). Run: uv run s2-widget`
      );
    }
    const report = await reportRes.json();
    const vocab = await vocabRes.json();
    let tokenizer = null;
    if (tokenizerRes.ok) {
      tokenizer = await tokenizerRes.json();
    }
    let s2Runtime = null;
    try {
      const runtimeRes = await fetch(runtimeUrl);
      if (runtimeRes.ok) s2Runtime = await runtimeRes.json();
    } catch (_error) {
      s2Runtime = null;
    }
    return { report, vocab, s2Runtime, tokenizer, tokenizerUrl };
  }

  function formatScore(score) {
    if (score === "inf" || score === Infinity) return "∞";
    if (typeof score === "number")
      return score.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return String(score);
  }

  function thresholdForLang(report, lang) {
    const perLang = report.fertility_thresholds;
    if (perLang && perLang[lang] != null) return perLang[lang];
    return report.fertility_threshold ?? 1.6;
  }

  function vocabCompositionLabel(report) {
    const counts = report.language_token_counts || {};
    const total = report.vocab_entries || report.vocab_size || 0;
    if (!total || !Object.keys(counts).length) return "";
    const parts = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([lang, count]) => `${LANG_LABELS[lang] || lang} ${count.toLocaleString()}`);
    return `${total.toLocaleString()} tokens — ${parts.join(" · ")}`;
  }

  function renderEvaluationSection(report, colors, reportUrl, tokenizerUrl) {
    const sorted = report.sorted || [];
    const xMax = report.x_max || {};
    const xMin = report.x_min || {};
    const xMaxLabel = LANG_LABELS[xMax.lang] || xMax.lang || "—";
    const xMinLabel = LANG_LABELS[xMin.lang] || xMin.lang || "—";
    const spreadFormula =
      report.spread_formula ||
      `X4 − X1 = ${xMax.value ?? "—"} − ${xMin.value ?? "—"} = ${report.spread}`;
    const metricNote =
      report.metric_note ||
      "Fertility = merged-vocab tokens per pretoken word (excludes whitespace gaps between pretoken words).";

    const rows = sorted
      .map(([lang, fertility], index) => {
        const stats = (report.languages || {})[lang] || {};
        const label = LANG_LABELS[lang] || lang;
        const rankLabel = `X${sorted.length - index}`;
        const threshold = stats.fertility_threshold ?? thresholdForLang(report, lang);
        const pass = stats.passes_threshold;
        const statusColor = pass ? colors.pass : colors.fail;
        const statusText = pass ? "pass" : "fail";
        const linkLabel = wikiLinkLabel(stats);
        const linkHref = wikiHref(stats);
        return `<tr>
          <td style="padding:0.4rem 0.55rem;">${rankLabel} (${label})</td>
          <td style="padding:0.4rem 0.55rem;"><a href="${escapeAttr(linkHref)}" target="_blank" rel="noopener" style="color:${colors.accent};">${escapeHtml(linkLabel)}</a></td>
          <td style="padding:0.4rem 0.55rem;text-align:right;font-family:ui-monospace,monospace;">${(stats.pretoken_words ?? stats.words ?? 0).toLocaleString()}</td>
          <td style="padding:0.4rem 0.55rem;text-align:right;font-family:ui-monospace,monospace;">${(stats.tokens ?? 0).toLocaleString()}</td>
          <td style="padding:0.4rem 0.55rem;text-align:right;font-family:ui-monospace,monospace;">${fertility.toFixed(4)}</td>
          <td style="padding:0.4rem 0.55rem;font-family:ui-monospace,monospace;font-size:0.78rem;">${escapeHtml(fertilityCalculation(stats))}</td>
          <td style="padding:0.4rem 0.55rem;text-align:right;font-family:ui-monospace,monospace;">≤ ${threshold.toFixed(1)}</td>
          <td style="padding:0.4rem 0.55rem;text-align:right;color:${statusColor};font-weight:600;">${statusText}</td>
        </tr>`;
      })
      .join("");

    return `
      <div style="${cardStyle(colors)}margin-bottom:1rem;">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:0.75rem;align-items:flex-start;margin-bottom:0.75rem;">
          <div>
            <div style="font-weight:600;font-size:0.95rem;margin-bottom:0.35rem;">Self score</div>
            <div style="font-size:2rem;font-weight:700;color:${colors.accent};line-height:1.1;">${formatScore(report.score)}</div>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:0.45rem;">
            <button type="button" data-action="download-tokenizer-json" style="${actionButtonStyle(colors)}">Download tokenizer.json</button>
            <a href="${escapeAttr(tokenizerUrl)}" download="tokenizer.json" style="${actionLinkStyle(colors)}">Link: tokenizer.json</a>
            <button type="button" data-action="download-report-json" style="${actionButtonStyle(colors)}">Download report.json</button>
            <a href="${escapeAttr(reportUrl)}" download="s2-fertility-report.json" style="${actionLinkStyle(colors)}">Link: report.json</a>
          </div>
        </div>
        <div style="font-size:0.82rem;opacity:0.85;font-family:ui-monospace,monospace;margin-bottom:0.2rem;">${report.score_formula || ""}</div>
        <div style="font-size:0.82rem;opacity:0.85;font-family:ui-monospace,monospace;margin-bottom:0.55rem;">${spreadFormula}</div>
        <div style="font-size:0.78rem;opacity:0.8;margin-bottom:0.75rem;">${metricNote}</div>
        <div style="font-size:0.82rem;opacity:0.85;display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:0.45rem;margin-bottom:1rem;">
          <div>X4 (${xMaxLabel}): <strong>${xMax.value ?? "—"}</strong></div>
          <div>X1 (${xMinLabel}): <strong>${xMin.value ?? "—"}</strong></div>
          <div>Spread: <strong>${report.spread}</strong></div>
        </div>
        <div style="font-weight:600;font-size:0.9rem;margin-bottom:0.5rem;">Per-language fertility</div>
        <div style="overflow:auto;">
          <table style="width:100%;border-collapse:collapse;font-size:0.82rem;min-width:760px;">
            <thead>
              <tr>
                <th style="text-align:left;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">Rank</th>
                <th style="text-align:left;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">Eval page</th>
                <th style="text-align:right;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">Pretoken words</th>
                <th style="text-align:right;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">Merged tokens</th>
                <th style="text-align:right;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">Fertility Xi</th>
                <th style="text-align:left;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">Calculation</th>
                <th style="text-align:right;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">Threshold</th>
                <th style="text-align:right;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">Status</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  function highlightStyle(index, colors) {
    const hue = CHIP_HUES[index % CHIP_HUES.length];
    const bg = colors.isDark ? `${hue}44` : `${hue}33`;
    return `background:${bg};border-radius:2px;cursor:help;box-decoration-break:clone;-webkit-box-decoration-break:clone;`;
  }

  function highlightTooltipStyle(colors) {
    const bg = colors.isDark ? "#1e293b" : "#0f172a";
    return `position:absolute;bottom:calc(100% + 4px);left:50%;transform:translateX(-50%);padding:0.2rem 0.45rem;border-radius:5px;background:${bg};color:#f8fafc;font-size:0.72rem;white-space:nowrap;pointer-events:none;opacity:0;transition:opacity 0.12s ease;z-index:30;box-shadow:0 2px 8px rgba(0,0,0,0.2);font-family:ui-monospace,monospace;`;
  }

  function ensureHighlightStyles() {
    if (document.getElementById("token-highlight-style")) return;
    const style = document.createElement("style");
    style.id = "token-highlight-style";
    style.textContent = `
      .playground-panels {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        align-items: stretch;
      }
      .playground-panel { display: flex; flex-direction: column; min-width: 0; }
      .playground-panel-label {
        font-size: 0.78rem;
        font-weight: 600;
        opacity: 0.75;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
      }
      .playground-input,
      .token-highlight-view {
        flex: 1;
        min-height: 11rem;
        box-sizing: border-box;
      }
      .playground-input {
        width: 100%;
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
        border: 1px solid var(--border, rgba(120,120,120,0.2));
        background: transparent;
        color: inherit;
        font-family: system-ui, sans-serif;
        font-size: 0.92rem;
        resize: vertical;
      }
      @media (max-width: 800px) {
        .playground-panels { grid-template-columns: 1fr; }
      }
      .token-highlight-view { white-space: pre-wrap; word-break: break-word; line-height: 1.75; font-size: 0.95rem; padding: 0.65rem 0.75rem; border-radius: 8px; border: 1px solid var(--border, rgba(120,120,120,0.2)); }
      .token-highlight { position: relative; }
      .token-highlight:hover .token-highlight-tip,
      .token-highlight:focus-visible .token-highlight-tip { opacity: 1 !important; }
    `;
    document.head.appendChild(style);
  }

  function spanTooltipText(span) {
    return span.collapsed
      ? `ids: ${span.ids.join(", ")} · ${JSON.stringify(span.token)}`
      : `id: ${span.id} · ${JSON.stringify(span.token)}`;
  }

  function renderHighlightedTokens(text, spans, colors, container) {
    const sorted = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
    container.replaceChildren();

    let cursor = 0;
    for (let index = 0; index < sorted.length; index += 1) {
      const span = sorted[index];
      if (span.end <= cursor || span.start >= text.length) continue;
      if (span.start > cursor) {
        container.appendChild(document.createTextNode(text.slice(cursor, span.start)));
      }
      const effectiveStart = Math.max(span.start, cursor);
      const effectiveEnd = Math.min(span.end, text.length);
      if (effectiveStart >= effectiveEnd) continue;

      const mark = document.createElement("span");
      mark.className = "token-highlight";
      mark.style.cssText = highlightStyle(index, colors);
      mark.textContent = text.slice(effectiveStart, effectiveEnd);
      mark.setAttribute("title", spanTooltipText(span));
      mark.setAttribute(
        "aria-label",
        `token id ${span.collapsed ? span.ids.join(",") : span.id}`
      );

      const tip = document.createElement("span");
      tip.className = "token-highlight-tip";
      tip.style.cssText = highlightTooltipStyle(colors);
      tip.textContent = spanTooltipText(span);
      mark.appendChild(tip);

      container.appendChild(mark);
      cursor = effectiveEnd;
    }

    if (cursor < text.length) {
      container.appendChild(document.createTextNode(text.slice(cursor)));
    }
  }

  function renderTokenizerPlaygroundShell(colors, hasRuntime) {
    if (!hasRuntime) {
      return `
        <div style="${cardStyle(colors)}margin-bottom:1rem;" data-section="tokenizer-playground">
          <div style="font-weight:600;font-size:0.95rem;margin-bottom:0.35rem;">Tokenization playground</div>
          <div style="font-size:0.82rem;opacity:0.85;">Run <code>uv run s2-widget</code> to export s2_runtime.json.</div>
        </div>
      `;
    }

    return `
      <div style="${cardStyle(colors)}margin-bottom:1rem;" data-section="tokenizer-playground">
        <div style="font-weight:600;font-size:0.95rem;margin-bottom:0.35rem;">Tokenization playground</div>
        <div style="font-size:0.82rem;opacity:0.85;margin-bottom:0.65rem;">
          All languages share one 10k merged vocab. Runtime does <strong>not</strong> call
          per-language BPE or SPM — it segments with DP (English), greedy (hi/pa), or
          Viterbi→merged map (Telugu). BPE only trains which pieces enter the vocab.
          Sample text below includes all four languages.
        </div>
        <div data-out="playground-stats" style="font-size:0.8rem;opacity:0.85;margin:0 0 0.55rem;font-family:ui-monospace,monospace;"></div>
        <div class="playground-panels">
          <div class="playground-panel">
            <div class="playground-panel-label">Input</div>
            <textarea data-input="playground-text" class="playground-input" rows="8"></textarea>
          </div>
          <div class="playground-panel">
            <div class="playground-panel-label">Tokenized</div>
            <div data-out="playground-output" class="token-highlight-view"></div>
          </div>
        </div>
      </div>
    `;
  }

  function renderTokenExplorerShell(
    vocabLength,
    colors,
    vocabLangs,
    selectedLang,
    selectedSort,
    vocabComposition,
    vocabUrl
  ) {
    const pageSizeOptions = PAGE_SIZES.map(
      size =>
        `<option value="${size}" ${size === 100 ? "selected" : ""}>${size} per page</option>`
    ).join("");

    const langOptions = [
      `<option value="all" ${selectedLang === "all" ? "selected" : ""}>All languages</option>`,
      ...vocabLangs.map(
        lang =>
          `<option value="${lang}" ${selectedLang === lang ? "selected" : ""}>${LANG_LABELS[lang] || lang}</option>`
      ),
    ].join("");

    const sortOptions = [
      ["id-asc", "ID (low → high)"],
      ["id-desc", "ID (high → low)"],
      ["lang-asc", "Language (A → Z)"],
      ["lang-desc", "Language (Z → A)"],
    ]
      .map(
        ([value, label]) =>
          `<option value="${value}" ${selectedSort === value ? "selected" : ""}>${label}</option>`
      )
      .join("");

    return `
      <div style="${cardStyle(colors)}" data-section="token-explorer">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:0.75rem;align-items:center;margin-bottom:0.65rem;">
          <div style="font-weight:600;font-size:0.95rem;">Merged tokenizer vocabulary (${vocabLength.toLocaleString()} tokens)</div>
          <div style="display:flex;flex-wrap:wrap;gap:0.45rem;">
            <button type="button" data-action="download-vocab-json" style="${actionButtonStyle(colors)}">Download JSON</button>
            <button type="button" data-action="download-vocab-csv" style="${actionButtonStyle(colors)}">Download CSV</button>
            <a href="${escapeAttr(vocabUrl)}" download="s2-merged-vocab.json" style="${actionLinkStyle(colors)}">Link: vocab.json</a>
          </div>
        </div>
        <div style="font-size:0.8rem;opacity:0.85;margin-bottom:0.65rem;">
          ${vocabComposition ? `${escapeHtml(vocabComposition)}. ` : ""}Browse below, use Download for an in-memory copy, or open
          <a href="${escapeAttr(vocabUrl)}" download="s2-merged-vocab.json" style="color:${colors.accent};">${escapeHtml(vocabUrl)}</a>
          to fetch the tokenizer vocab file directly.
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:0.6rem;margin-bottom:0.65rem;align-items:center;">
          <input type="search" data-input="token-search" placeholder="Search tokens..." style="flex:1;min-width:180px;padding:0.45rem 0.6rem;border-radius:8px;border:1px solid ${colors.border};background:transparent;color:inherit;" />
          <label style="font-size:0.82rem;display:flex;align-items:center;gap:0.35rem;">
            <span>Language</span>
            <select data-input="vocab-lang-filter" style="padding:0.35rem 0.5rem;border-radius:8px;border:1px solid ${colors.border};background:transparent;color:inherit;">${langOptions}</select>
          </label>
          <label style="font-size:0.82rem;display:flex;align-items:center;gap:0.35rem;">
            <span>Sort</span>
            <select data-input="vocab-sort" style="padding:0.35rem 0.5rem;border-radius:8px;border:1px solid ${colors.border};background:transparent;color:inherit;">${sortOptions}</select>
          </label>
          <label style="font-size:0.82rem;display:flex;align-items:center;gap:0.35rem;">
            <span>Show</span>
            <select data-input="page-size" style="padding:0.35rem 0.5rem;border-radius:8px;border:1px solid ${colors.border};background:transparent;color:inherit;">${pageSizeOptions}</select>
          </label>
        </div>
        <div data-out="token-count" style="font-size:0.78rem;opacity:0.8;margin-bottom:0.45rem;"></div>
        <div style="max-height:360px;overflow:auto;border:1px solid ${colors.border};border-radius:8px;">
          <table style="width:100%;border-collapse:collapse;font-size:0.82rem;">
            <thead style="position:sticky;top:0;background:${colors.bg};">
              <tr>
                <th style="text-align:left;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">ID</th>
                <th style="text-align:left;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">Lang</th>
                <th style="text-align:left;padding:0.4rem 0.55rem;border-bottom:1px solid ${colors.border};">Token</th>
              </tr>
            </thead>
            <tbody data-body="tokens"></tbody>
          </table>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:0.5rem;align-items:center;justify-content:space-between;margin-top:0.65rem;">
          <div data-out="page-info" style="font-size:0.82rem;opacity:0.85;"></div>
          <div style="display:flex;gap:0.4rem;">
            <button type="button" data-action="page-first" style="padding:0.35rem 0.6rem;border-radius:8px;border:1px solid ${colors.border};background:transparent;color:inherit;cursor:pointer;font-size:0.82rem;">First</button>
            <button type="button" data-action="page-prev" style="padding:0.35rem 0.6rem;border-radius:8px;border:1px solid ${colors.border};background:transparent;color:inherit;cursor:pointer;font-size:0.82rem;">Prev</button>
            <button type="button" data-action="page-next" style="padding:0.35rem 0.6rem;border-radius:8px;border:1px solid ${colors.border};background:transparent;color:inherit;cursor:pointer;font-size:0.82rem;">Next</button>
            <button type="button" data-action="page-last" style="padding:0.35rem 0.6rem;border-radius:8px;border:1px solid ${colors.border};background:transparent;color:inherit;cursor:pointer;font-size:0.82rem;">Last</button>
          </div>
        </div>
      </div>
    `;
  }

  function createWidget(container) {
    const dataBase = container.getAttribute("data-data-base") || "./public/data";
    const vocabUrl = dataFileUrl(dataBase, "vocab.json");
    const reportUrl = dataFileUrl(dataBase, "report.json");
    const tokenizerUrl = dataFileUrl(dataBase, "tokenizer.json");
    const state = {
      report: null,
      vocab: [],
      s2Runtime: null,
      tokenizer: null,
      playgroundText: PLAYGROUND_SAMPLE,
      playgroundTimer: null,
      filter: "",
      vocabLangFilter: "all",
      vocabSort: "id-asc",
      page: 1,
      pageSize: 100,
      vocabUrl,
      reportUrl,
      tokenizerUrl,
    };

    container.innerHTML = `
      <div class="s2-tokenizer-scorer" style="margin:1.5rem 0;font-family:system-ui,sans-serif;color:var(--foreground,inherit);">
        <div data-out="status" style="font-size:0.9rem;opacity:0.85;">Loading tokenizer report…</div>
        <div data-out="content" style="display:none;"></div>
      </div>
    `;

    const el = sel => container.querySelector(sel);

    function vocabLangs() {
      return [...new Set(state.vocab.map(item => item.lang).filter(Boolean))].sort();
    }

    function sortedFilteredVocab() {
      const query = state.filter.trim().toLowerCase();
      let items = state.vocab;
      if (query) {
        items = items.filter(item =>
          String(item.token).toLowerCase().includes(query)
        );
      }
      if (state.vocabLangFilter && state.vocabLangFilter !== "all") {
        items = items.filter(item => item.lang === state.vocabLangFilter);
      }
      items = [...items];
      switch (state.vocabSort) {
        case "id-desc":
          items.sort((a, b) => b.id - a.id);
          break;
        case "lang-asc":
          items.sort(
            (a, b) =>
              String(a.lang).localeCompare(String(b.lang)) || a.id - b.id
          );
          break;
        case "lang-desc":
          items.sort(
            (a, b) =>
              String(b.lang).localeCompare(String(a.lang)) || a.id - b.id
          );
          break;
        default:
          items.sort((a, b) => a.id - b.id);
          break;
      }
      return items;
    }

    function totalPages(count) {
      return Math.max(1, Math.ceil(count / state.pageSize));
    }

    function clampPage(count) {
      state.page = Math.min(state.page, totalPages(count));
      state.page = Math.max(1, state.page);
    }

    function renderTokenTable() {
      const tbody = el('[data-body="tokens"]');
      const countOut = el('[data-out="token-count"]');
      const pageInfo = el('[data-out="page-info"]');
      if (!tbody || !state.vocab.length) return;

      const filtered = sortedFilteredVocab();
      clampPage(filtered.length);
      const pages = totalPages(filtered.length);
      const start = (state.page - 1) * state.pageSize;
      const shown = filtered.slice(start, start + state.pageSize);

      tbody.innerHTML = shown
        .map(
          item => `<tr>
            <td style="padding:0.35rem 0.55rem;border-bottom:1px solid var(--border,rgba(120,120,120,0.15));font-family:ui-monospace,monospace;">${item.id}</td>
            <td style="padding:0.35rem 0.55rem;border-bottom:1px solid var(--border,rgba(120,120,120,0.15));font-family:ui-monospace,monospace;">${escapeHtml(item.lang || "—")}</td>
            <td style="padding:0.35rem 0.55rem;border-bottom:1px solid var(--border,rgba(120,120,120,0.15));word-break:break-all;">${escapeHtml(item.token)}</td>
          </tr>`
        )
        .join("");

      const rangeStart = filtered.length ? start + 1 : 0;
      const rangeEnd = start + shown.length;
      countOut.textContent =
        state.filter || state.vocabLangFilter !== "all"
          ? `${filtered.length.toLocaleString()} matches (${state.vocab.length.toLocaleString()} total tokens)`
          : `${state.vocab.length.toLocaleString()} tokens total`;

      if (pageInfo) {
        pageInfo.textContent = filtered.length
          ? `Showing ${rangeStart}–${rangeEnd} of ${filtered.length.toLocaleString()} · Page ${state.page} of ${pages}`
          : "No tokens match your search";
      }

      const disableNav = filtered.length === 0 || pages === 1;
      container.querySelectorAll('[data-action^="page-"]').forEach(button => {
        const action = button.getAttribute("data-action");
        if (action === "page-prev" || action === "page-first") {
          button.disabled = disableNav || state.page <= 1;
        } else {
          button.disabled = disableNav || state.page >= pages;
        }
        button.style.opacity = button.disabled ? "0.45" : "1";
        button.style.cursor = button.disabled ? "not-allowed" : "pointer";
      });
    }

    function updatePlaygroundChips() {
      const statsOut = el('[data-out="playground-stats"]');
      const outputOut = el('[data-out="playground-output"]');
      if (!statsOut || !outputOut || !state.s2Runtime || !window.S2Encode) return;

      ensureHighlightStyles();
      const colors = getColors();
      const text = state.playgroundText;
      const spans = window.S2Encode.displaySpans(text, state.s2Runtime);
      const tokenIds = window.S2Encode.encodeIds(text, state.s2Runtime);
      const fertilityTokens = window.S2Encode.countFertilityTokens(text, state.s2Runtime);
      const chars = [...text].length;
      const pretokenWords = window.S2Encode.countPretokenWords(
        text,
        state.s2Runtime.pretoken_pattern
      );

      statsOut.textContent = `${tokenIds.length.toLocaleString()} encode tokens · ${fertilityTokens.toLocaleString()} fertility tokens · ${pretokenWords.toLocaleString()} pretoken words · ${chars.toLocaleString()} chars · merged vocab`;

      renderHighlightedTokens(text, spans, colors, outputOut);
    }

    function renderAll() {
      if (!state.report) return;
      const existingPlayground = el('[data-input="playground-text"]');
      if (existingPlayground) state.playgroundText = existingPlayground.value;

      const colors = getColors();
      const content = el('[data-out="content"]');
      content.style.display = "block";
      content.innerHTML =
        renderEvaluationSection(state.report, colors, state.reportUrl, state.tokenizerUrl) +
        renderTokenizerPlaygroundShell(
          colors,
          Boolean(state.s2Runtime)
        ) +
        renderTokenExplorerShell(
          state.vocab.length,
          colors,
          vocabLangs(),
          state.vocabLangFilter,
          state.vocabSort,
          vocabCompositionLabel(state.report),
          state.vocabUrl
        );

      const pageSizeSelect = el('[data-input="page-size"]');
      if (pageSizeSelect) pageSizeSelect.value = String(state.pageSize);

      const langFilterSelect = el('[data-input="vocab-lang-filter"]');
      if (langFilterSelect) langFilterSelect.value = state.vocabLangFilter;

      const sortSelect = el('[data-input="vocab-sort"]');
      if (sortSelect) sortSelect.value = state.vocabSort;

      const playgroundInput = el('[data-input="playground-text"]');
      if (playgroundInput) playgroundInput.value = state.playgroundText;

      renderTokenTable();
      updatePlaygroundChips();
    }

    function schedulePlaygroundUpdate() {
      if (state.playgroundTimer) clearTimeout(state.playgroundTimer);
      state.playgroundTimer = setTimeout(() => updatePlaygroundChips(), 150);
    }

    container.addEventListener("input", event => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;

      if (target.matches('[data-input="playground-text"]')) {
        state.playgroundText = target.value;
        schedulePlaygroundUpdate();
        return;
      }

      if (target.matches('[data-input="token-search"]')) {
        state.filter = target.value;
        state.page = 1;
        renderTokenTable();
      }

      if (target.matches('[data-input="vocab-lang-filter"]')) {
        state.vocabLangFilter = target.value;
        state.page = 1;
        renderTokenTable();
      }

      if (target.matches('[data-input="vocab-sort"]')) {
        state.vocabSort = target.value;
        state.page = 1;
        renderTokenTable();
      }

      if (target.matches('[data-input="page-size"]')) {
        state.pageSize = parseInt(target.value, 10) || 100;
        state.page = 1;
        renderTokenTable();
      }
    });

    container.addEventListener("click", event => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;

      const filtered = sortedFilteredVocab();
      const pages = totalPages(filtered.length);
      const action = target.getAttribute("data-action");

      if (action === "page-first") state.page = 1;
      if (action === "page-prev") state.page = Math.max(1, state.page - 1);
      if (action === "page-next") state.page = Math.min(pages, state.page + 1);
      if (action === "page-last") state.page = pages;

      if (action === "download-tokenizer-json" && state.tokenizer) {
        downloadBlob(
          "tokenizer.json",
          JSON.stringify(state.tokenizer, null, 2),
          "application/json"
        );
        return;
      }

      if (action === "download-vocab-json") {
        downloadBlob(
          "s2-merged-vocab.json",
          JSON.stringify(state.vocab, null, 2),
          "application/json"
        );
        return;
      }

      if (action === "download-vocab-csv") {
        downloadBlob("s2-merged-vocab.csv", vocabToCsv(state.vocab), "text/csv;charset=utf-8");
        return;
      }

      if (action === "download-report-json" && state.report) {
        downloadBlob(
          "s2-fertility-report.json",
          JSON.stringify(state.report, null, 2),
          "application/json"
        );
        return;
      }

      if (action && action.startsWith("page-")) renderTokenTable();
    });

    loadData(dataBase)
      .then(({ report, vocab, s2Runtime, tokenizer, tokenizerUrl }) => {
        state.report = report;
        state.vocab = vocab;
        state.s2Runtime = s2Runtime;
        state.tokenizer = tokenizer;
        if (tokenizerUrl) state.tokenizerUrl = tokenizerUrl;
        el('[data-out="status"]').style.display = "none";
        renderAll();
      })
      .catch(err => {
        el('[data-out="status"]').textContent = err.message;
      });

    const themeObserver = new MutationObserver(() => {
      if (!document.body.contains(container)) {
        themeObserver.disconnect();
        return;
      }
      if (state.report) renderAll();
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
  }

  function mount(container) {
    if (initialized.has(container)) return;
    initialized.add(container);
    createWidget(container);
  }

  function initAll() {
    document
      .querySelectorAll('.viz-s2-tokenizer-scorer[data-viz="s2-tokenizer-scorer"]')
      .forEach(mount);
  }

  initAll();
  document.addEventListener("astro:page-load", initAll);
})();
