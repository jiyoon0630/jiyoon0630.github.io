# jiyoon0630.github.io — working notes for Claude

Jiyoon Kim's personal website, published with Jekyll on **GitHub Pages** at
https://jiyoon0630.github.io/. It is both a research archive (paper and tech
reviews) and a personal profile (publications, projects, CV).

The whole site is **bilingual (English + Korean)**, switched by one toggle at
the top right of every page. Notes are separate KO and EN posts; every other
page renders both languages and shows one (see "Site-wide language" below).

## Default job: post every uploaded Markdown note in BOTH languages

When the owner uploads a Markdown note, publish it as **two paired posts — one
Korean, one English** — unless they say otherwise.

1. Pick a shared key `<ref>` (kebab-case). The date is **the day you post it**,
   not the paper's date — the lists are an archive of when she studied things.
   Put the paper's own date in `paper_date` when it is known; it shows on the
   note page and nowhere else. Write the date with her offset
   (`2026-09-17 17:00:00 -0700`); `_config.yml` sets `timezone:
   America/Los_Angeles`, without which a UTC build moves an evening post to the
   next day. The filename prefix is the posting date too
   (`:title` permalinks ignore it, so the URL is unaffected).
2. Create two files in `_posts/`:
   - Korean: `YYYY-MM-DD-<ref>.md` with `lang: ko`
   - English: `YYYY-MM-DD-<ref>-en.md` with `lang: en`
   Both carry the **same `ref`** so the KO ⇄ EN toggle links them.
3. Set `kind:` — `paper-review` (a paper) or `tech-review` (a conference
   report, technical analysis, project write-up). It drives the Notes tabs.
4. If the upload is in only one language, **translate it into the other**.
   Translate faithfully: preserve all math verbatim, translate the text inside
   tables and ASCII diagrams too, and keep the callout (`> ### 💡/⚠️/📌/🔗`)
   structure intact.
5. Give each language its own `summary` in that language.
6. Build locally to verify, then commit and push.

A single-language note is fine — on it, the header toggle's other-language
button is disabled, since there is no translation (same `ref`) to open.

## Front matter

```yaml
---
layout: paper
lang: en                      # ko | en
ref: pld-self-improving-vla   # shared key linking the KO and EN versions
kind: paper-review            # paper-review | tech-review
title: "Paper Title"
date: 2026-09-17              # the day it was POSTED — drives ordering and the
                              # date shown in every list. Not the paper's date.
paper_date: 2025-10-30        # optional: when the paper itself came out
venue: "ICLR 2026 · arXiv:2511.00091"   # optional
tags: [VLA, Reinforcement-Learning]     # click-to-filter labels
summary: "One-line description, in this file's language."
authors: "First author et al."          # optional
affiliations: "NVIDIA · CMU"            # optional
paper_url: "https://arxiv.org/abs/..."  # optional → 📄 Paper link
code_url: "https://github.com/..."      # optional → 💻 Code link
---
```

- **Math:** `$...$` inline, `$$...$$` display (MathJax on the live site; it will
  not render in a sandbox that cannot reach the CDN — verify structure instead).
  Write TeX plainly: `_plugins/protect_inline_math.rb` rewrites inline `$x$` to
  `$$x$$` before kramdown runs, so it passes through untouched (without it,
  `\{` lost its backslash, `_..._` became `<em>`, `'` a curly quote, `\%` a
  comment) and turns `<` into `\lt` (math is emitted raw, so `v_{<t}` opened
  an HTML tag). No `\_` or `\lbrace` workarounds needed — `\_` now prints a
  literal underscore. To check every formula, extract the `kdmath` spans from
  `_site` and run them through `mathjax-full`'s `tex2mml`.
  A `|` inside inline math (`$|C|$`) is still a table pipe to kramdown: a
  line holding two such spans renders as a one-row table. Write `\lvert C\rvert`
  (and `\Vert` for norms), and compare each post's table count against `_site`.
- **Currency and MathJax.** A bare `$70M ... $600M` pairs up as inline math:
  the text between them renders in italic math type and, since math does not
  wrap, stretches its table cell until the table runs off the page.
  - Post with **no equations**: set `math: false` in the front matter; MathJax
    skips the whole body.
  - Post with **both** equations and currency: write each amount as
    `<span class="tex2jax_ignore">&#36;400M</span>`. Both halves matter. The
    entity keeps a literal `$` out of the source, so `protect_inline_math.rb`
    cannot pair it with the next amount's `$` and rewrite the prose between
    them as math. The span makes MathJax skip the `$` the browser decodes, and a
    skipped element also breaks its text run, so it never pairs with one
    outside. **Do not use `\$` or `\\$`** — kramdown collapses the backslashes
    in some contexts and not others, and MathJax reads `\\` as a literal
    backslash followed by a live delimiter.
  - Korean particles are word characters, so a regex like `\$\d+M\b` misses
    `$165M을`. Use a negative lookahead `(?![A-Za-z0-9])` instead.
- **Section references: no `§`.** She finds it unfamiliar. Korean writes
  `2.3절` (and the particle follows 절: `2절은`, `4.5절이`, `3절로`,
  `5.2–5.4절`); English writes `Section 2.3`, `Sections 5.2–5.4`. Same for
  the paper's own sections (`논문 4.1절`, `paper Section 4.1`). Convert any `§`
  in an uploaded note while posting it.
- **Code fences** are highlighted (rouge); unlabeled fences (ASCII diagrams)
  render as plain monospace. **Markdown does not apply inside a fence**, so
  `**bold**` in an ASCII diagram prints its asterisks. Notion renders them, so
  strip them when converting a diagram out of a source document; indentation and
  the ▸/│ characters already carry the hierarchy.
- **Callout box** = a blockquote whose first line is `### <emoji> Title`.
  The opening quote of a note renders as a lead card.

## Tech reviews start from internal material — sanitize first

Tech-review sources are the owner's **internal work documents** and may contain
confidential information. This repo is public, so:

- **Never commit raw source material.** Put it in `sources/` and sanitized
  drafts in `drafts/` — both are gitignored. Never `git add -f` either one.
  They are also in `_config.yml`'s `exclude`: gitignore alone does not stop a
  local build from copying them into `_site/` as pages. Keep both entries.
- The sanitizer reads **`.claude/redaction-denylist.md`** — a gitignored file
  holding the specific names and business facts that must never be published.
  It is deliberately not in the repo, so a fresh clone will not have it; if it
  is missing, ask the owner for it before running the sanitizer.
- Run the **`sanitize-tech-review`** subagent on the source before doing anything
  else. It strips employer-identifying and confidential content and repairs the
  prose, then writes `drafts/<name>.md` plus a redaction report.
- The subagent is a first-pass filter, **not a clearance decision**. Surface its
  report to the owner and get their confirmation before posting. Do not tell
  them a draft is "safe".
- Only after they confirm: add front matter (`kind: tech-review`), produce the
  KO + EN pair as usual, and publish.

If the owner hands you internal material directly in chat rather than as a file,
write it to `sources/` first, then follow the same path.

### Her standing redaction decisions (from four reports — apply by default)

- **Keep:** startup funding amounts, valuations, round dates, "round under
  discussion (reported, unconfirmed)", founder acquisition amounts, and her
  employer's venture arm named as a round participant. She treats all of these
  as public.
- **Keep:** unreleased or newly announced model details presented at public
  events, and speculative reads of closed models as long as the source's own
  추정 / 비공개 marking stays.
- **Keep:** a company's name wherever it appears as ordinary published research,
  including paper-author affiliations — even for companies she works with.
- **Remove:** the internal org name in titles and image paths, the author line
  (colleague names and internal titles), the internal publication-date line,
  internal workspace links, 본 보고서 / 자사 framing, and any recommendation of
  what her employer should do.
- **Remove case by case:** a partner company listed where the information could
  only have come from working with them rather than from a public source.
- **Remove:** concrete findings from a client project even when anonymized
  (e.g. the actual errors a model made there). **Keep, generalized:** an
  architecture she built, recast as a hypothetical example ("구성할 수
  있습니다", not "설계하였습니다") with client, org and scope removed.
- **Third-party figures** (e.g. from a vendor's blog): keep them, with the
  caption crediting the source and linking the original page.
- **Facts that changed after the source date:** leave her text as written and
  add a `> ### ⚠️ 2026년 9월 기준 업데이트` callout (EN: `Update, <Month Year>`)
  next to it. Say so when a prediction of hers came true.
- **Date** a tech review by the source's own publication date.
- A report may be **co-credited internally but written by her alone** — ask
  which parts are hers before using it as a voice sample, and if a colleague
  wrote part of it, cut that part rather than publish it.

### Workflow notes from doing it

- Notion exports a page with images as a zip; ask for the zip, since pasted
  images lose their filenames and so their place in the text. Mac Safari may
  unzip it on download — re-compress the folder.
- **Extract every file in the zip, `.svg` included.** An extension filter that
  forgot SVG once made two diagrams look missing when they were there all along.
- The sanitizer cannot see images. Look at every one yourself before posting.

## The voice profile — Korean is written in her voice, always

`.claude/voice-profile.md` is a **record of observed traits** of her writing. It
exists so that what one session notices about her style is still there in the
next one. Keep it a findings file: one entry per trait, each with its evidence,
an invented neutral example, and the Claude default it displaces.

- **Tier A** = text she wrote. She supplies these documents. Read them for voice
  and add entries. Never edit them, never quote them into the profile — it is a
  public file.
- **Tier B** = text Claude wrote — every note on the site except the tech
  reviews made from her own documents (RFM, World Model, Actuate 2026, Claude
  Agent Skills), all English translations, and anything Claude drafts next.
- Tech-review sources are Tier A. The `sanitize-tech-review` subagent reports
  voice observations from the *pre-sanitization* text on every run; merge those
  into the profile. The subagent never edits the profile itself.
- Log each document in the profile's table. Prose traits are recordable from the
  first document; statistical ones need 2,000–5,000 words per language.

### Standing rule (her instruction, 2026-09-24)

**Every Korean post goes up in her voice.** When she hands over AI-written
material — a paper review, a blog-post review, any draft — polish the Korean
version to her voice before posting, without being asked. Follow
`.claude/voice-ko-checklist.md` and check with `scripts/voice-check.py` (when
converting an existing post) and `scripts/voice-stats.py --target`. This covers
paper reviews too: she wants the same 합쇼체 register there. Her own tech-review
documents are already in her voice — sanitize them, don't re-style them.

**English is not converted yet.** She will decide after the Korean pass. Until
then write English normally, and only strip the profile's anti-patterns. Do not
make the English post a translation of the converted Korean.

### The conversion pass

1. **Meaning is invariant.** No claim strengthens or weakens, no number moves,
   no hedge disappears, no terminology distinction collapses. A sentence that
   cannot be converted without changing meaning stays as it is.
2. **Structure is invariant.** Math verbatim, callout boxes, tables, code fences
   and ASCII diagrams untouched. Front matter untouched except `summary`, which
   is prose and gets converted too.
3. Strip the profile's anti-patterns — safe today, independent of any entry.
4. Apply the recorded entries. Do not chase a number at the cost of sense: a
   metric matching while the paragraph reads worse is a failed conversion.
5. Measure before and after, and say what moved.
6. Convert Korean and English **independently**. The English post is not a
   translation of the converted Korean.

### Measurement

```bash
python3 scripts/voice-stats.py --lang ko --baseline <her document>   # find differences
python3 scripts/voice-stats.py --lang ko --save-target .claude/target-ko.json sources/*.md
python3 scripts/voice-stats.py --lang ko --target .claude/target-ko.json _posts/<post>.md
```

`--baseline` compares against the Tier B numbers below; `--target` compares
against a profile saved from her own documents. Target files are gitignored —
they derive from internal material.

**Tier B fingerprint — what the site sounds like now.** English, 3 notes,
18,647 prose words: em dash 14.00/1k · "not X but Y" 1.13/1k · "It is/This is"
+ emphasis 1.66/1k · semicolon 1.34/1k · para-initial But/And 0.70/1k ·
sentences mean 23.4, median 19, p10 7, p90 45. Korean, 4 notes, 13,659 prose
words: em dash 15.23/1k · ~에 대한 1.54/1k · ~을/를 통해 0.22/1k · ~다 종결
55.71/1k · ~습니다 0.00/1k · sentences mean 13.5, median 11, p10 5, p90 24.

Note the median-vs-mean gap: length variation is already present, so uniform
sentence length is not the tell on this site. The rhetorical frames are.

## Figures in notes

- Diagrams she draws herself arrive as **transparent PNGs**. Composite them onto
  **white** before converting; PIL's `convert("RGB")` fills transparency with
  black, which hides the dark grey titles and footnotes those diagrams use.
- Resize to 1400px wide and save JPEG q84–88. Ten figures went from 3.9MB to
  925KB that way.
- A caption is an *italic line directly under the image*, with no blank line
  between, so kramdown keeps both in one paragraph and `p > img + em` styles it.
  Put the source attribution there — in the alt text it is invisible.
- **Korean labels inside a diagram do not translate.** For the English post,
  redraw the diagram as SVG under `assets/img/notes/<ref>/en/` and point the
  English post at it. SVG keeps the text sharp at any width and costs a few KB.

## Profile / publications / projects / CV are data-driven

Do **not** hand-edit the page HTML for content. Edit these instead:

| File | Drives |
|---|---|
| `_data/profile.yml` | home hero: name, role, bio, interests, links |
| `_data/news.yml` | home News list |
| `_data/publications.yml` | `/publications/`, home "Selected publications", CV |
| `_data/projects.yml` | `/projects/`, home "Selected projects", CV |
| `_data/cv.yml` | `/cv/` education, experience, skills, awards |

`selected: true` on a publication or project also surfaces it on the home page.

### Site-wide language

- Every translatable data field has a Korean sibling with a `_ko` suffix
  (`bio_ko`, `tagline_ko`, `title_ko`, `org_ko`, `period_ko`, `detail_ko`,
  `venue_ko`, `abstract_ko`, `label_ko`, `text_ko`, `group_ko`). Lists take a
  parallel list: `interests_ko`, `items_ko` (same order), and CV `bullets_ko`
  (a whole second list). A missing `_ko` falls back to the English text, so
  adding an English entry never breaks the Korean view — but add the Korean
  one too.
- Interface strings (nav, headings, buttons) live in `_data/ui.yml`, rendered
  with `{% include ui.html key="..." %}`. Page titles take `title_ko` /
  `subtitle_ko` in front matter.
- Templates print both languages through `{% include bi.html en=... ko=... %}`;
  CSS hides one based on `<html data-lang>`. That attribute is set by an inline
  script in `<head>` before paint: a note uses its own `lang` (and saves it as
  the reader's choice), other pages use the saved choice, else the browser
  language. The toggle click lives in `assets/js/filter.js`; on a note it opens
  the translation instead.
- Paper titles, author lists, tag names and proper nouns stay in English in
  both views.
Values marked `TODO` are placeholders the owner still has to fill in — leave
them until they supply real content, and never invent biographical facts,
venues, authors, or dates.

## Local verification

```bash
bundle install   # first time
JEKYLL_ENV=production bundle exec jekyll build
# If `bundle exec jekyll` cannot find the executable:
#   JEKYLL_EXE=$(bundle exec ruby -e 'require "jekyll"; print Gem.loaded_specs["jekyll"].full_gem_path + "/exe/jekyll"')
#   JEKYLL_ENV=production bundle exec ruby "$JEKYLL_EXE" build
```

Check `_site/notes/<ref>/` and `_site/notes/<ref>-en/` exist and that the
header language toggle plus the Notes filters are wired, and switch the
toggle on the home, CV and Notes pages to see both languages.

Note: `assets/js/filter.js` only runs where the filter controls exist (it bails
out unless `#tag-filters` is on the page), so note lists elsewhere stay visible.

## Start here: `.claude/STATE.md`

Current state and what is next — what is published, what is waiting on the
owner, what this environment cannot do. Read it at the start of a session and
update it when any of that changes. It is deliberately short; process belongs in
this file.

## Visual verification — actually look at the page

Structure checks miss layout bugs. The container cannot reach the live site, but
it can serve the build locally and screenshot it. Do this after any change to
CSS, layouts, or the home page.

```bash
# 1. build and serve
JEKYLL_ENV=production bundle exec jekyll build
(cd _site && python3 -m http.server 4000 &)

# 2. install the browser CLI (global — it dies with the container, so redo it)
npm install -g agent-browser

# 3. point it at the pre-installed Chromium, or it will not find a browser
export AGENT_BROWSER_EXECUTABLE_PATH=/opt/pw-browsers/chromium-1194/chrome-linux/chrome

agent-browser open http://localhost:4000/
agent-browser screenshot --full home.png     # then read the PNG
agent-browser snapshot -i                    # accessibility tree, for structure
agent-browser close
```

Known limitations in this environment:

- The proxy serves an **older agent-browser (0.27.0)** than npm's latest, so its
  flags differ from the published docs: `--full`, not `--full-page`.
- `--executable-path` is ignored once the daemon is running; the env var above
  always works. `agent-browser close` restarts it.
- **No viewport control** in this version (`viewport` is not a command and
  `--args --window-size` is ignored — `innerWidth` stays 1280). For anything
  width-dependent use Playwright instead, which drives the same pre-installed
  browser and does set the viewport:

```bash
npm install playwright-core            # the browser is already on disk
# chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' })
# newPage({ viewport: { width, height } })
```

  Check a layout change at 1440 / 1280 / 1100 / 1024 / 900 / 768 / 430 / 390 and
  assert `document.documentElement.scrollWidth <= clientWidth` at each one. A
  fixed-pixel breakout that fits at 1280 can push the page sideways at 1100.

- **Google Fonts and jsdelivr are egress-blocked**, so local screenshots use
  fallback fonts and MathJax never loads in the browser. **Check math with Node
  instead** — the npm registry is reachable, so run the real engine with the
  site's exact config over the built HTML:

```bash
npm install mathjax-full@3
# mathjax.document(html, { InputJax: new TeX({ inlineMath: [['$','$'],['\\(','\\)']],
#   displayMath: [['$$','$$'],['\\[','\\]']], processEscapes: true }),
#   ignoreHtmlClass: 'tex2jax_ignore', skipHtmlTags: [...] }).render()
# then list doc.math and flag any span containing Hangul or >120 characters —
# that is prose MathJax swallowed between two currency dollar signs.
```

  Run it on any post that has both `$` amounts and equations. A one-line test
  is not enough: kramdown's escaping behaved differently inside the full
  document than in isolation, and only the full-document run caught it.

## Design work — the `design-taste-frontend` skill

`.claude/skills/design-taste-frontend/` is a third-party design skill, committed
so it survives the container. Use it for visual work on the site's pages.

It assumes React + Tailwind + GSAP/Motion. **This site is none of those**, so
take its design judgment and ignore its code. Constraints for every use:

- **Stack:** Jekyll, hand-written CSS in `assets/css/style.css`, one small
  vanilla JS file. Do not add React, Tailwind, GSAP, Motion, or any build step.
- **Dials:** MOTION low — this is an academic site, not an agency landing page.
  DENSITY medium. VARIANCE low.
- **No placeholder media.** The skill suggests `picsum.photos` and
  `cdn.simpleicons.org`; both are wrong here. Real assets or nothing.
- **Settled decisions — do not "improve" these:** dark mode via
  `prefers-color-scheme`, the callout card treatment, the site-wide language
  toggle in the header, the notes filter controls, the data-driven page
  structure.
- Its "hard em-dash ban" applies to UI copy it writes. It does **not** override
  the voice profile or touch existing post text.

## Push policy (the owner delegated this)

- Push straight to `main`; every push triggers the deploy workflow.
- After a change reaches `main`, confirm the deploy workflow succeeded; if it
  failed, read the logs and fix it.

## Deploy

GitHub Pages builds via `.github/workflows/pages.yml` (Pages Source: GitHub
Actions, enabled by `enablement: true`). The repo is public, so Pages is free.
The `github-pages` deployment environment must allow `main`; if a fresh
environment restricts branches wrongly, deleting the `github-pages` environment
and re-running fixes it.

## Layout reference

- `index.html` home · `notes.html` → `/notes/` · `publications.html` ·
  `projects.html` · `cv.html`
- `_layouts/default.html` shell + nav + language toggle · `page.html` generic
  page · `paper.html` note page (authors, affiliations)
- `_includes/pub-item.html`, `_includes/project-card.html`, `cv-entries.html`,
  `bi.html` (both languages), `ui.html` (a string from `_data/ui.yml`)
- `assets/css/style.css` — callout cards, tags, hero, cards, CV, dark mode
- `templates/paper-template.md` — copy-paste front matter
