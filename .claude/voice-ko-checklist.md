# Korean in her voice — the working checklist

The rules for writing or converting a Korean post in the owner's voice. The
evidence behind each rule is in `.claude/voice-profile.md`; this file is the
condensed, applicable form. First used for the 16-post conversion on
2026-09-24. When the profile gains or changes an entry, update this list too.

For the target register, read the Korean posts made from her own documents:
`_posts/2026-04-10-rfm-vla-development.md` and
`_posts/2025-11-18-claude-agent-skills.md`.

## What to change
- **Register:** every prose sentence → 합쇼체 (~습니다 / ~입니다 / ~합니다), including list items, callout bodies, the opening lead-card blockquote and the front-matter `summary`. Only direct quotations (inside quotation marks, e.g. a paper's own words or an instruction like "셔츠를 개라") and noun-phrase fragments stay as they are.
- **Table cells:** keep row and column structure exactly. A cell that is a full plain sentence (~다.) may become 개조식 (~함 / ~있음 / noun ending) or 합쇼체; short noun phrases stay.
- **Em dash:** her rate is ~1.5 per 1,000 words, and only structurally, right after a bolded label or in a heading. A mid-sentence dash becomes 즉, parentheses, a colon, or two sentences. Keep label dashes like `**⓵ 조합 불가** — …` and leave headings alone.
- **Verdict first:** where a paragraph states a conclusion, lead with it as one fully bolded 합쇼체 sentence (by reordering or bolding an existing sentence — never by inventing a claim).
- **Three verbs:** what a paper/company claims → 제시하였습니다 / 밝혔습니다 / 발표하였습니다 / 보고하였습니다; the reviewer's own evaluation → 판단됩니다 / 보입니다 / 할 수 있겠습니다, used only in the paragraph's final verdict sentence and only where the original sentence is clearly the reviewer's own judgment (not a fact reported by the paper). Never weaken or strengthen a factual claim.
- **Paired contrast:** where the text contrasts two approaches, use "A가 ~였다면, B는 ~입니다" when it fits naturally.
- **Connectives:** concede with 다만, contrast with 반면, narrow with 특히 / 즉. No sentence opens with 그래서; avoid runs of 따라서.
- **English terms** stay bare in Korean prose; glosses run 한국어(english), not english(한국어) — only reorder existing glosses, don't add new ones.
- **Remove Claude tics** (voice-profile anti-patterns): "A가 아니라 B다" as a rhetorical frame (keep it only where it states a real distinction, rephrased plainly), aphoristic one-liners, reader-address meta ("이 글은 ~한 독자를 가정하고"), decorative metaphors (keep a metaphor only when headings or later text refer back to it, e.g. a named "벽 1/벽 2"), vacuous emphasis ("핵심은 ~이다" as filler).
- **Questions (her decision, 2026-09-24):** a problem-posing question stays a
  question only inside a plain quote box (a blockquote that is not a `> ###`
  callout), in plain form: `~ㄹ까?` / `~는가?` — as in her own RFM report. Every
  other question (body, lists, callouts, bold labels) becomes a declarative
  합쇼체 sentence that keeps the setup it did: "그렇다면 X는 얼마나 클까요?" →
  "X가 얼마나 큰지 따져 보면 다음과 같습니다." Never `~ㄹ까요?`.
- Sentence length: don't chop chains into short sentences; joining two short sentences into one natural 합쇼체 sentence is fine. Don't pad.

## What must not change (`scripts/voice-check.py` enforces this)
Meaning, every number, all math ($…$ and $$…$$ byte-identical), code fences, inline code, headings (all levels, byte-identical — they are link anchors), callout titles (`> ### …` lines), images, link targets, HTML tags, table shape, front matter except `summary`. A sentence you cannot convert without changing meaning stays as it is — list it in your report.

## Checking a conversion

```bash
python3 scripts/voice-check.py _posts/<post>.md          # vs. git HEAD: must print OK
python3 scripts/voice-stats.py --lang ko --target .claude/target-ko.json _posts/<post>.md
```

`voice-check.py` fails on any change to math, numbers, headings, callout
titles, links, images, HTML, code or table shape, and lists the 해라체 endings
that remain. For a brand-new post there is no HEAD version: write it in her
voice from the start and run only `voice-stats.py`.

`.claude/target-ko.json` is gitignored (it derives from internal documents).
Rebuild it with `--save-target` from `sources/`; if `sources/` is gone, use the
published posts made from her documents (listed above) as the target instead.
