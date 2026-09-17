# jiyoon0630.github.io

Personal website — research notes, paper & tech reviews, publications, and projects.
Built with Jekyll, deployed to GitHub Pages at <https://jiyoon0630.github.io/>.

## Structure

| Path | What it is |
|---|---|
| `index.html` | Home — profile, news, selected publications/projects/notes |
| `notes.html` | `/notes/` — all reviews, filtered by kind × language × tag |
| `publications.html` | `/publications/` — my own papers |
| `projects.html` | `/projects/` — PhySeek and other builds |
| `cv.html` | `/cv/` — CV, rendered from `_data/cv.yml` |
| `_posts/` | One note per file: `YYYY-MM-DD-<ref>[-en].md` |
| `_data/` | **Edit these to update the site** (see below) |

## Editing content

Almost everything is data-driven — edit YAML, not HTML:

- `_data/profile.yml` — name, role, bio, interests, links
- `_data/news.yml` — the News list on the home page
- `_data/publications.yml` — your papers (`selected: true` also shows on home)
- `_data/projects.yml` — projects (`selected: true` also shows on home)
- `_data/cv.yml` — education, experience, skills, awards

Anything marked `TODO` is a placeholder waiting to be filled in.

## Adding a note

One paper/topic = one Markdown file in `_posts/`, named `YYYY-MM-DD-<ref>.md`
(and `-en.md` for the English version). See `templates/paper-template.md`.

Key front matter: `lang: ko|en`, `ref:` (links the two languages),
`kind: paper-review|tech-review`, `tags:`, `summary:`.

## Running locally

```bash
bundle install
bundle exec jekyll serve      # http://localhost:4000/
```

Every push to `main` builds and deploys via `.github/workflows/pages.yml`.
