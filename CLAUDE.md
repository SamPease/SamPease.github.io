# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repository hosts Samantha Pease's personal website built with Astro (static output), deployed to GitHub Pages via `.github/workflows/deploy-astro.yml`.

## Commands

```bash
npm run dev      # local dev server (hot reload)
npm run build    # production build → dist/
npm run preview  # serve the dist/ build locally
```

No lint or test scripts exist. After any structural or route change, run `npm run build` and confirm the route list in the build output.

## Architecture

**Framework:** Astro 6, static output (`output: "static"`), Node ≥ 22.12.

**Content pipeline:**
- Markdown files in `src/content/blog/*.md` are the single source of truth for writeups.
- Loaded via the `blog` content collection defined in `src/content.config.ts`.
- Required frontmatter: `title`, `description`, `date`. Optional: `tags[]`, `draft` (default `false`). Posts with `draft: true` are excluded from all routes.
- Math is supported in markdown via `remark-math` + `rehype-katex`. KaTeX CSS is injected per-page via a `<slot name="head">` in the writeup detail template.

**Routes (canonical):**
- `/` — homepage, rendered by `src/pages/index.astro` using `src/components/HomeContent.astro`.
- `/project-writeups/` — writeup index, `src/pages/project-writeups/index.astro`.
- `/project-writeups/[...slug]/` — detail page, `src/pages/project-writeups/[...slug].astro`. Slug is the filename without `.md`.
- Short-form redirects (`/barselo`, etc.) are declared in `astro.config.mjs` and must be kept in sync when slugs change.

**Intentionally removed routes:** `src/pages/blog` and `src/pages/projects`. Do not reintroduce them.

**Layout system:**
- `src/layouts/BaseLayout.astro` is the single layout shell. Props: `title` (required), `description`, `fullBleed` (bool, skips the `site-shell` max-width wrapper), `navLinks` (array of `{href, label}`).
- Default `navLinks` in BaseLayout is a minimal two-link nav (Home + Project Write Ups) used on every page; no page currently overrides it.
- BaseLayout also owns the site-wide footer, the favicon (`/favicon.svg`), and Open Graph/Twitter meta tags (`og:image` points at `/DSC09430.JPG`).
- CSS design tokens are defined as CSS custom properties on `:root` in BaseLayout: `--paper`, `--ink`, `--deep-ink`, `--panel`, `--border`, `--accent`, `--accent-strong`, `--sun`. Use these for any new styled elements.
- Fonts: **Newsreader** (serif, headings via `:global(h1–h4)`), **Manrope** (sans-serif, body). Both loaded from Google Fonts in BaseLayout.

**Static assets (`public/`):**
- Only `bootstrap.min.css` and `resume.min.css` are loaded globally via BaseLayout (their utility classes — `lead`, `mb-*`, `resume-section`, etc. — are used throughout). `jquery.min.js` and `bootstrap.bundle.min.js` exist in `public/` solely because the standalone `TransAdviceAgent.html` page loads them — do not delete them or add them to layouts. Other legacy assets (Font Awesome, academicons, jquery.easing, resume.min.js, old resume PDFs) were removed.
- Use absolute paths (`/pruv.pdf`) for assets in Astro component markup — there is no `<base>` tag. URL-encode spaces in filenames (`/Calc%201%20Syllabus.doc`). Add `is:inline` on `<script>` tags that reference public assets to avoid bundling errors.
- Raw `<script>` blocks inside markdown writeups are passed through to the built HTML and do execute (e.g. the iframe auto-resizer in `trans-advice-agent.md`).

**TOC generation:** The writeup detail page builds a TOC from `h2` headings only (via `render()` headings output). Maintain `##` section headings in markdown to keep TOC links stable.

## Editing Guidance

- Prefer minimal diffs; avoid broad refactors unless requested.
- Preserve markdown content and heading text unless specifically asked to rewrite — heading text changes break TOC anchor links.
- When adding a new writeup: create `src/content/blog/<slug>.md` with required frontmatter, and add a redirect entry in `astro.config.mjs` if a short URL is desired.
- When adding new nav links, update the `navLinks` default in `BaseLayout.astro`.

## Handoff Rule

When new constraints or architectural decisions are made, update **this file** (`CLAUDE.md`) in the same change so future agents inherit current repo reality.