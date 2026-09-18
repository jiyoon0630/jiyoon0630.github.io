# jiyoon0630.github.io — working notes for Claude

Jiyoon Kim's personal website, published with Jekyll on **GitHub Pages** at
https://jiyoon0630.github.io/. It is both a research archive (paper and tech
reviews) and a personal profile (publications, projects, CV).

The site UI and all non-note pages are **English only**. Notes are
**bilingual (Korean + English)**.

## Default job: post every uploaded Markdown note in BOTH languages

When the owner uploads a Markdown note, publish it as **two paired posts — one
Korean, one English** — unless they say otherwise.

1. Pick a shared key `<ref>` (kebab-case) and a date `YYYY-MM-DD` (use the
   note's front-matter `date` if present).
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

A single-language note is fine — the toggle only appears when a translation
with the same `ref` exists.

## Front matter

```yaml
---
layout: paper
lang: en                      # ko | en
ref: pld-self-improving-vla   # shared key linking the KO and EN versions
kind: paper-review            # paper-review | tech-review
title: "Paper Title"
date: 2025-10-30              # newest-first ordering
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
- **Code fences** are highlighted (rouge); unlabeled fences (ASCII diagrams)
  render as plain monospace.
- **Callout box** = a blockquote whose first line is `### <emoji> Title`.
  The opening quote of a note renders as a lead card.

## Tech reviews start from internal material — sanitize first

Tech-review sources are the owner's **internal work documents** and may contain
confidential information. This repo is public, so:

- **Never commit raw source material.** Put it in `sources/` and sanitized
  drafts in `drafts/` — both are gitignored. Never `git add -f` either one.
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

## The voice profile — record her style now, convert posts on her signal

`.claude/voice-profile.md` is a **record of observed traits** of her writing. It
exists so that what one session notices about her style is still there in the
next one. Keep it a findings file: one entry per trait, each with its evidence,
an invented neutral example, and the Claude default it displaces.

- **Tier A** = text she wrote. She supplies these documents. Read them for voice
  and add entries. Never edit them, never quote them into the profile — it is a
  public file.
- **Tier B** = text Claude wrote — all seven notes currently on the site, both
  languages, and anything Claude drafts next.
- Tech-review sources are Tier A. The `sanitize-tech-review` subagent reports
  voice observations from the *pre-sanitization* text on every run; merge those
  into the profile. The subagent never edits the profile itself.
- Log each document in the profile's table. Prose traits are recordable from the
  first document; statistical ones need 2,000–5,000 words per language.

**Applying it is a separate job, and it is dormant.** Write notes and
translations normally until she explicitly says to start converting.

### The conversion pass, when she gives the signal

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
language toggle plus the Notes filters are wired.

Note: `assets/js/filter.js` only runs where the filter controls exist (it bails
out unless `#tag-filters` is on the page), so note lists elsewhere stay visible.

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
- `_layouts/default.html` shell + nav · `page.html` generic page ·
  `paper.html` note page (authors, affiliations, KO ⇄ EN toggle)
- `_includes/pub-item.html`, `_includes/project-card.html`
- `assets/css/style.css` — callout cards, tags, hero, cards, CV, dark mode
- `templates/paper-template.md` — copy-paste front matter
