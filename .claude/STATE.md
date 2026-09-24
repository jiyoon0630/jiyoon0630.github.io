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
   re-uploaded. Nothing is in progress right now.
3. **Tooling is not preinstalled.** Reinstall per session as needed:
   `npm install -g agent-browser`, `npm install playwright-core` and
   `npm install mathjax-full@3` in a scratch dir, `pip install Pillow`.
   CLAUDE.md has the exact usage.

## Live

- Site: https://jiyoon0630.github.io/ — Jekyll, deployed from `main` by
  `.github/workflows/pages.yml` on every push.
- 36 notes, all ko/en. **Paper reviews** (arXiv papers): DreamZero, PLD,
  Hi Robot, Cosmos 3, π0.7, SAIL, LAPA, RTC, π*0.6 (RECAP). **Tech reviews**: RFM / VLA development (dated
  2026-04-10), Actuate 2026, World Model, plus the company-blog/report reviews
  GEN-0, GEN-1, GENE-26.5, ACT-2, Dyna-2, DYNA-1. World Model's intro links to RFM.
- Her rule for `kind`: a review whose source is not an arXiv paper is a
  `tech-review` (tag `Tech-Review`), even when it analyzes a model.
- Note dates are **posting dates**, in America/Los_Angeles (`_config.yml` sets
  the timezone). A tech review is dated by its source's publication date. A
  paper's own date lives in `paper_date` and shows only on the note page.
- Some notes carry dates she chose rather than the day they were posted
  (π0.7: 2026-04-19, DreamZero: 2026-02-21, GEN-0: 2026-01-28, GEN-1: 2026-04-08, ACT-2: 2026-07-18, GENE-26.5: 2026-05-14, DYNA-1: 2025-12-18, SAIL: 2026-04-20, LAPA: 2026-02-03, RTC: 2026-03-04, π*0.6: 2026-01-18). The 2026-07-10 "Launched this
  site" news item was removed at her request.
- Pages: home, notes, publications, projects, CV — all data-driven from `_data/`.
- The old `jiyoon0630/paper-reivew` repo is deleted; `404.html` carries the
  redirect for old `/paper-reivew/papers/<slug>/` links.

## Waiting on the owner

- **More tech-review sources** — sanitize, walk her through the report, post the
  KO/EN pair only after she confirms. CLAUDE.md lists her standing decisions.
- **The voice profile is dormant.** Korean is established from three documents;
  do not convert any post toward it until she explicitly says so.
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

- RTC paper review published (ko/en), dated 2026-03-04 at her request. It
  cites Soft RTC (arXiv:2605.25537), which postdates that date — kept as uploaded.
- LAPA paper review published (ko/en), dated 2026-02-03 at her request.
  A `|` inside inline math (`$|C|$`) is read by kramdown as a table pipe
  once the math plugin runs; write `\lvert C\rvert` instead.
- SAIL paper review published (ko/en), dated 2026-04-20 at her request.
- RFM tech review published as a KO/EN pair. Only the back half of her
  internal report (the part she wrote) was kept, renumbered Section 1–2. The
  English page uses SVG redraws of the four era diagrams
  (`assets/img/notes/rfm/en/`); the Korean one uses the originals, clickable to
  full size. Seven dollar amounts are written `&#36;` inside `tex2jax_ignore`
  spans (see CLAUDE.md, "Currency and MathJax").
- Voice profile: doc #3 added. Verdict hedging turned out to be genre-bound
  (absent from an explainer); paired contrast confirmed.
- `sources/` and `drafts/` added to `_config.yml` `exclude` after a local build
  was found copying them into `_site/` (never pushed; Actions builds from a
  clean checkout).
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
