# State — where this site stands

Short by design: **current state and what is next.** Everything durable about
*how* to work on the site belongs in `CLAUDE.md`, not here.

Why it exists: the working container is ephemeral and long sessions get
summarized, so anything that lives only in a conversation is lost. Update this
file whenever something below changes — it costs one edit and saves a
reconstruction.

_Last updated: 2026-09-18_

## Live

- Site: https://jiyoon0630.github.io/ — Jekyll, deployed from `main` by
  `.github/workflows/pages.yml` on every push.
- 6 notes: PLD (ko/en), Hi Robot (ko/en), Dyna-2 (ko/en). All paper reviews; no
  tech review published yet. The sample "Welcome" post was deleted once real
  notes existed — `templates/paper-template.md` is the format reference now.
- Note dates are **posting dates**, not paper dates; the paper's own date lives
  in `paper_date` and shows only on the note page.
- Pages: home, notes, publications, projects, CV — all data-driven from `_data/`.
- The old `jiyoon0630/paper-reivew` repo is **deleted**; `404.html` carries the
  redirect for old `/paper-reivew/papers/<slug>/` links.

## Waiting on the owner

- **Tech-review sources** — to be sanitized (subagent + denylist), confirmed by
  her, then posted as a KO/EN pair.
- **Tier A documents** for the voice profile — her own writing, to fill
  `.claude/voice-profile.md`. Applying the profile waits for her explicit signal.
- **Profile photo** — `assets/img/profile.svg` is a "JK" placeholder.
- **CV PDF** — `_data/cv.yml` has `pdf: ""`; filling it enables the Download
  button.
- **Dyna-2 `paper_date`** — the source gave only `2026-08`, so the day in
  `2026-08-01` is a guess. Confirm or correct.

## Known limits of this environment

- `~/.claude` does not persist. Only what is committed under `.claude/` survives,
  which is why the sanitizer agent, the voice profile and the design skill live
  there.
- `.claude/redaction-denylist.md` is **gitignored**, so a fresh container will
  not have it. Ask for it before running the sanitizer.
- The live site is unreachable from the container; verify visually by serving the
  build locally (see CLAUDE.md).

## Recently done

- Voice profile scaffolding — `.claude/voice-profile.md`, `scripts/voice-stats.py`,
  measured Tier B fingerprints for both languages.
- Fixed the home/notes list: a long title no longer drops below its date.
- Switched note dates to posting dates and deleted the sample Welcome post.
- Added `design-taste-frontend` and the local screenshot workflow.
