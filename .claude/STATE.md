# State — where this site stands

Short by design: **current state and what is next.** Everything durable about
*how* to work on the site belongs in `CLAUDE.md`, not here.

Why it exists: the working container is ephemeral and long sessions get
summarized, so anything that lives only in a conversation is lost. Update this
file whenever something below changes — it costs one edit and saves a
reconstruction.

_Last updated: 2026-09-24_

## Starting a new session? Read this first

1. **Ask the owner for `.claude/redaction-denylist.md`.** It is gitignored, so a
   fresh container does not have it, and the sanitizer cannot run without it.
   She has a copy. Put it back at that exact path and confirm with
   `git check-ignore -v .claude/redaction-denylist.md` before writing anything
   sensitive.
2. **`sources/` and `drafts/` are gone** in a new container (gitignored). Every
   published post is complete in the repo; only unfinished work needs its source
   re-uploaded — see "In progress" below.
3. **Tooling is not preinstalled.** Reinstall per session as needed:
   `npm install -g agent-browser`, `npm install playwright-core` and
   `npm install mathjax-full@3` in a scratch dir, `pip install Pillow`.
   CLAUDE.md has the exact usage.

## In progress — RFM tech review (not yet published)

Source: her internal report on robot foundation models. **She wrote only its
back half**; the intro and Sections 1–3 were written by a colleague (Section 3
was internal strategy), so they were **cut entirely at her instruction**. What
remains was "Advanced-Section 4/5" and is renumbered **Section 1** (VLA
development, Era 1 → 2A → 2B → 3) and **Section 2** (training methodology:
data taxonomy, pre-training, post-training).

Done so far (in the old container):
- Cut and renumbered; internal workspace links replaced with in-document
  pointers (`→ 1-1-2`).
- The report has **both equations and dollar amounts** — the seven amounts are
  wrapped in `<span class="tex2jax_ignore">`. Verified with the real MathJax
  engine: 52 equations render, no prose swallowed. See CLAUDE.md, "Currency and
  MathJax".
- Figures: `assets/img/notes/rfm/` holds the four era diagrams (2400px, white
  ground — the originals are dense slides) and two SVGs. The data-taxonomy
  quadrant SVG had its axis labels moved to sit over their columns, with her
  approval. English copies of the two SVGs are in `en/`.
- English redraws of the four era diagrams were being made as SVG into
  `assets/img/notes/rfm/en/era*.svg`. **Check whether they exist and render
  cleanly**; if not, redraw them (see the World Model ones for the approach).
- A sanitized draft was being produced. If `drafts/rfm.md` is missing, the owner
  needs to re-upload the RFM zip; then redo the cut (keep from the line
  "# Advanced-Section 4" onward), renumbering, link replacement and currency
  wrapping before running the sanitizer.

Still to do: walk her through the sanitizer report, get her confirmation, then
post the KO/EN pair dated **2026-04-10** (the source's publication date), with
the Korean era images clickable to full size. Once it is up, link it from the
World Model post, whose intro says it follows "앞서 정리했던 RFM의 발전 흐름".

## Live

- Site: https://jiyoon0630.github.io/ — Jekyll, deployed from `main` by
  `.github/workflows/pages.yml` on every push.
- 24 notes, all ko/en. **Paper reviews** (arXiv papers): DreamZero, PLD,
  Hi Robot, Cosmos 3, π0.7. **Tech reviews**: Actuate 2026, World Model, plus
  the company-blog/report reviews GEN-0, GEN-1, GENE-26.5, ACT-2, Dyna-2.
- Her rule for `kind`: a review whose source is not an arXiv paper is a
  `tech-review` (tag `Tech-Review`), even when it analyzes a model.
- Note dates are **posting dates**, in America/Los_Angeles (`_config.yml` sets
  the timezone). A tech review is dated by its source's publication date. A
  paper's own date lives in `paper_date` and shows only on the note page.
- Some notes carry dates she chose rather than the day they were posted
  (π0.7: 2026-04-19, DreamZero: 2026-02-21, GEN-0: 2026-01-28, GEN-1: 2026-04-08, ACT-2: 2026-07-18, GENE-26.5: 2026-05-14). The 2026-07-10 "Launched this
  site" news item was removed at her request.
- Pages: home, notes, publications, projects, CV — all data-driven from `_data/`.
- The old `jiyoon0630/paper-reivew` repo is deleted; `404.html` carries the
  redirect for old `/paper-reivew/papers/<slug>/` links.

## Waiting on the owner

- **More tech-review sources** — sanitize, walk her through the report, post the
  KO/EN pair only after she confirms. CLAUDE.md lists her standing decisions.
- **The voice profile is dormant.** Korean is established from two documents;
  do not convert any post toward it until she explicitly says so. The RFM
  sanitizer run also produces voice observations — merge them when it reports.
- **Profile photo** — `assets/img/profile.svg` is a "JK" placeholder.
- **CV PDF** — `_data/cv.yml` has `pdf: ""`; filling it enables the Download
  button.
- **Dyna-2 `paper_date`** — the source gave only `2026-08`, so the day in
  `2026-08-01` is a guess.
- **Projects** — only three cards and no per-project pages yet; she wants room
  for PhySeek and small startup-tech reproduction projects.

## Known limits of this environment

- `~/.claude` does not persist. Only what is committed under `.claude/` survives.
- Egress is restricted. notion.site, super.site, arxiv.org and the live site are
  blocked; the npm registry, PyPI and raw.githubusercontent.com work, and
  WebSearch can reach papers when WebFetch cannot.
- MathJax and Google Fonts cannot load in a local browser. Check math with
  `mathjax-full` in Node instead (CLAUDE.md).
- agent-browser cannot set the viewport; use Playwright for width checks.

## Recently done

- GEN-1 review published (ko/en), dated 2026-04-08 at her request; the source
  is a company blog post, so `paper_url` points at the blog.
- ACT-2 preview review published (ko/en), dated 2026-07-18 at her request;
  `paper_date` 2026-07-17 taken from her source's date field — confirm with her.
  Kramdown drops the backslash in `\{` inside inline math; use `\lbrace`.
- Cosmos 3 paper review published (ko/en), dated 2026-06-20 at her request.
  Kramdown turns `}_{ ... }_{` inside inline `$...$` into `<em>`; escaped as `\_`.
- GEN-0 (dated 2026-01-28) and DreamZero (2026-02-21) paper reviews posted,
  KO + EN, dates at her request.
- Voice profile filled in for Korean: 12 traits from two documents.
- CLAUDE.md now records her standing redaction decisions and the currency-and-
  MathJax method.
- World Model tech review published; its six diagrams redrawn in English as SVG.
- Actuate 2026 tech review published.
- The denylist's company section judges context, not names.
- Wide tables and figures use the page margins on wide screens; CSS and JS are
  cache-busted per build.
