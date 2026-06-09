# Website Repo Instructions (Copilot Auto-Loaded)

This repository hosts Samantha Pease's personal website built with Astro (static output).

## Primary Goal

Make safe, minimal edits to the website while preserving existing behavior and content fidelity.

## Current Architecture

- Framework: Astro static site.
- Canonical content routes:
  - `/` (homepage)
  - `/project-writeups/` (writeup index)
  - `/project-writeups/[slug]/` (writeup detail pages)
- Content source: markdown files in `src/content/blog/*.md`.
- Content collection config: `src/content.config.ts`.
- Shared layout shell: `src/layouts/BaseLayout.astro`.
- Homepage content component: `src/components/HomeContent.astro`.

## Important Route Decisions

- `project-writeups` is the single canonical writeup route set.
- `src/pages/blog` has been removed on purpose to reduce redundancy.
- `src/pages/projects` legacy redirects were removed on purpose.
- Do not reintroduce duplicate route trees unless explicitly requested.

## Writeup Page Behavior

- Writeup index page: `src/pages/project-writeups/index.astro`.
- Writeup detail page: `src/pages/project-writeups/[...slug].astro`.
- Detail pages render markdown via `astro:content` `render(...)`.
- Right-side TOC is generated from `h2` headings only.
- Top nav on writeup pages is intentionally minimal: Home + Project Write Ups.

## Styling and UX Notes

- Blog/writeup heading sizes were intentionally reduced for readability.
- TOC is desktop sidebar, sticky, and hidden on small screens.
- Keep visual changes consistent with existing style language in `BaseLayout` and writeup page CSS.

## Static Assets Rules

- Static assets should live in `public/`.
- Root-level duplicate asset files were intentionally removed.
- In Astro components, use absolute public paths for static scripts/assets (example: `/resume.min.js`).
- For external `<script src="/..."></script>` in Astro component markup, use `is:inline` when needed to avoid bundling errors.

## Editing Guidance

- Prefer minimal diffs and avoid broad refactors unless requested.
- Preserve content in markdown writeups unless specifically asked to rewrite/summarize.
- Maintain section headings when possible so TOC links remain stable.
- After structural or route edits, run a full build check.

## Validation Checklist

- Run: `npm run build`
- Confirm generated routes include:
  - `/`
  - `/project-writeups/`
  - `/project-writeups/<slug>/` for each markdown file
- Confirm no accidental dependency on deleted route trees (`/blog/*`, `/projects/*`) unless intentionally reintroduced.

## Handoff Rule (Important)

When new constraints, conventions, or architectural decisions appear during future tasks, update this file (`.github/copilot-instructions.md`) in the same PR/change so future agents inherit current repo reality.
