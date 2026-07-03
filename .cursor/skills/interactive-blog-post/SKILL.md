---
name: interactive-blog-post
description: Add or edit posts and interactive browser visualizations in this AstroPaper blog. Use when writing a new blog post, embedding an interactive/animated visualization or chart, archiving example posts, or working with drafts in src/data/blog and public/visualizations.
---

# Interactive blog posts (AstroPaper)

This repo is the AstroPaper Astro theme. Posts are Markdown in `src/data/blog/`; interactive widgets are plain JS in `public/visualizations/` and embedded via raw HTML in the Markdown.

## Drafts: dev-visible, prod-hidden

`draft: true` means a post shows in `npm run dev` but is excluded from production builds. This is how we "archive" posts (e.g. the bundled `src/data/blog/examples/`). The logic lives in [src/utils/postFilter.ts](../../../src/utils/postFilter.ts) plus inline `import.meta.env.DEV || !data.draft` filters in `src/pages/posts/[...slug]/index.astro`, `src/pages/posts/[...page].astro`, `src/pages/archives/index.astro`, and `src/pages/posts/[...slug]/index.png.ts`. To archive a post, set `draft: true`; to publish, set `draft: false`.

## Add a post

Create `src/data/blog/<slug>.md`. Frontmatter (`title`, `pubDatetime`, `description` are required; `author` defaults to `SITE.author`):

```md
---
title: Your title
pubDatetime: 2026-07-03T11:30:00Z
featured: false
draft: false
tags:
  - some-tag
description: One-sentence summary used for SEO and cards.
---
```

Markdown renders raw HTML, so a `<div>` + `<script src>` embed works directly (see LaTeX/figure usage in existing posts). Verify with `npm run build` (runs `astro check`).

## Add an interactive visualization

Create `public/visualizations/<name>.js` as a self-contained IIFE, then embed in the post:

```html
<div class="viz-myname" data-viz="myname"></div>
<script src="/visualizations/myname.js"></script>
```

Required patterns (the site uses `ClientRouter` / View Transitions, mirroring `public/toggle-theme.js`):

- Guard against double-init with a `WeakSet` of mounted containers.
- Run `initAll()` once on load AND on `document`'s `astro:page-load` event.
- In any animation loop, bail if `!document.body.contains(container)` so it stops after navigation. Disconnect any `MutationObserver` there too.
- Theme-aware colors: read `document.documentElement.getAttribute("data-theme")` and `getComputedStyle(document.documentElement).getPropertyValue("--accent" | "--foreground")`. Re-render on `data-theme` changes via a `MutationObserver` when idle.
- Hi-DPI canvases: set `canvas.width/height = cssSize * devicePixelRatio` and `ctx.setTransform(dpr,0,0,dpr,0,0)` each render.
- Asset paths are root-absolute (`/visualizations/...`); no `base` is configured.

### Using a charting/graphics library

Prefer loading via dynamic ESM import from a CDN (no bundler for `public/`):

```js
let d3Promise = null;
const loadD3 = () => (d3Promise ??= import("https://cdn.jsdelivr.net/npm/d3@7/+esm"));
```

For decision-boundary / scalar-field plots, `d3-contour` (`d3.contours().smooth(true)` + `d3.geoPath(null, ctx)`) gives smooth filled iso-bands on canvas; use `d3-scale` for the color field and `d3-axis`/`d3-shape` for charts. Clip plot drawing to a rounded rect so thick strokes don't overrun the frame. Keep a single rendering path — do not add a redundant fallback renderer unless asked.

Reference implementation: [public/visualizations/rings-activation.js](../../../public/visualizations/rings-activation.js).

## GitHub Pages base path

`SITE.website` in [src/config.ts](../../../src/config.ts) drives Astro `site` + `base` in [astro.config.ts](../../../astro.config.ts). **Must match where Pages actually serves the site.**

- User site (`username.github.io` repo) → `https://username.github.io/` → assets at `/_astro/`
- Project site (`blog` repo) → `https://username.github.io/blog/` → assets at `/blog/_astro/`

Mismatch (HTML says `/blog/_astro/…` but deploy at root) → JS/CSS 404, MIME `text/html`, broken styling.

Hardcoded public assets in layouts use `import.meta.env.BASE_URL` (favicon, toggle-theme, sitemap). Markdown embeds use root paths like `/visualizations/…` — correct when `base` is unset.

## Git workflow

Do all of this on a feature branch with small atomic commits (Conventional Commits), staging by explicit path so unrelated working-tree changes aren't swept in. One commit per logical unit (filter logic, drafts, each widget, each post).

## Verify before committing

- `npm run build` — runs `astro check` + build (use `npm`, not `pnpm`, in this environment).
- `npm run lint` — ESLint.
- `node --check public/visualizations/<name>.js` — quick syntax check for the widget.
