# Voice profile — Jiyoon Kim

What her writing actually looks like, built up from her own drafts so that
Claude-written translations and reviews can be brought toward it.

**Status: EMPTY — not yet derived from any sample. Do not apply it yet.**

---

## How this file is used

- It is **dormant by default.** Translations and paper reviews are written the
  normal way until the owner explicitly says to start applying this profile.
  Do not quietly start imitating a half-built profile.
- It is **built incrementally**, mainly from tech-review source documents,
  which the owner writes herself at work. The `sanitize-tech-review` subagent
  reports voice observations from the *pre-sanitization* text on every run; merge
  those in here and log the sample below. The subagent never edits this file
  itself.
- When the owner gives the signal, it becomes the style target for both new
  posts and a pass back over the existing paper reviews.

## Two hard rules

**⓵ Never quote internal source material here.** This file is in a public
repository. Record *patterns*, and illustrate them with **invented examples on
neutral topics** — never a sentence lifted from a work document. If a pattern
cannot be shown without confidential content, describe it in words instead.

**⓶ Log every sample's provenance.** A draft the owner wrote with Claude's help
carries Claude's rhythm, not hers. Mixing those in silently is how a profile
ends up describing the assistant instead of the author. If provenance is
unknown, ask before adding the sample.

---

## Provenance tiers — what each kind of sample can actually prove

The tech-review sources are **Claude-assisted drafts that the owner reviewed line
by line.** That is much stronger than an unreviewed draft and much weaker than
something she typed cold, and the two halves license different conclusions.

**Tier A — judgment-level. Trust it.** Review reliably controls these, because
a wrong one is visible to the author on sight:

- terminology, and the distinctions she refuses to collapse
- section order, what leads a paragraph, what got cut
- factual hedging: what stays marked as estimated or unverified
- claim strength — where she softened an assertion
- Korean phrasing she rejects outright (she caught `자사` unprompted)
- anything she demonstrably rewrote

**Tier B — generation-level. Do not trust it from a reviewed draft.** Line-by-
line review catches errors and awkwardness; it does not catch *statistics*.
Nobody reads a draft and thinks "too many em dashes per thousand words." So
punctuation density, sentence-length distribution, paragraph-opener habits and
rhetorical frames all survive review while still being Claude's, not hers.

**The test that separates them.** Measure the source against the site baseline:

```bash
python3 scripts/voice-stats.py --lang ko --baseline sources/<file>.md
```

Every Tier-B metric that lands near `1.00x` means review left that feature
untouched — it is the assistant's habit, and recording it here would be
circular. A metric that diverges sharply *is* editing signal and can be
promoted to Tier A with the number written down as evidence.

**Better samples, if they exist.** Anything written before an assistant touched
it is worth more than a whole reviewed document: the prompts and instructions
she wrote to Claude, raw meeting notes, outlines, Slack or messenger prose. Ask
for these — a few hundred words of them beat several thousand reviewed words.

---

## Samples this profile is built from

| # | Source (no confidential detail) | Language | Provenance | Tier | Approx. words | Date added |
|---|---|---|---|---|---|---|
| — | *none yet* | — | — | — | — | — |

Counters: **Korean 0 words · English 0 words.** A profile needs roughly
2,000–5,000 words per language before it is worth applying.

---

## Korean (한국어)

*Nothing recorded yet.*

Things to look for when the first sample arrives:

- **Sentence endings** — ~다 / ~이다 / 명사형 종결의 비율, and whether it shifts
  by section (body vs. conclusion vs. callout)
- **Sentence length and rhythm** — short declaratives vs. long chained clauses;
  where she breaks a sentence rather than joining it
- **Connectives** — which ones she actually uses (그런데 / 그래서 / 다만 / 반면)
  and which she avoids; how often a paragraph opens with one
- **Argument order** — does the claim come first (두괄식) or does she build to it
- **번역투 she avoids** — e.g. ~의 경우, ~에 다름 아니다, ~되어지다, overuse of
  ~에 대한, ~을 통해
- **English-term handling** — which technical terms stay in English, whether
  they get a Korean gloss, and how she brackets them
- **Emphasis** — bold, quotation, dashes; how heavy-handed
- **Her own tics** — repeated constructions that are recognisably hers

## English

*Nothing recorded yet.*

Things to look for:

- **Register** — how formal, how much contraction, first person or not
- **Sentence length** distribution, and whether she varies it deliberately
- **Openers** — how she starts a paragraph or a section
- **Punctuation habits** — em dash, semicolon, colon, parentheses
- **Hedging** — how she marks uncertainty, and how often
- **Terminology** — the precision distinctions she insists on
- **Constructions she avoids** — to be filled in from evidence, not assumed

---

## Anti-patterns — what to strip regardless of profile

These are general LLM tics, independent of her voice. Safe to act on now.

- "It is not X, it is Y" / "not merely X but Y" as a rhetorical frame
- Vacuous emphasis openers: "It is worth noting", "The key point is",
  "Importantly", "Crucially"
- Triads used for rhythm rather than because there are three things
- Paragraphs that open with a formal connective several times in a row
- Closing sentences that restate the paragraph without adding anything
- Vocabulary: delve, leverage (as a verb), nuanced, underscore, tapestry,
  realm, testament, landscape (figurative)
- Uniform sentence length across a whole passage

Note: an em dash is **not** on this list. Density measured across the current
posts is about 14–15 per 1,000 words in both the Korean sources and the English
translations, so it is a feature of the material, not a reliable tell here.

---

## Measurement

`scripts/voice-stats.py` prints the counts used above (character fingerprints,
phrasal tells, sentence-length distribution) for any set of files, so claims in
this profile stay evidence-based rather than impressionistic.

```bash
python3 scripts/voice-stats.py --lang en _posts/*-en.md
python3 scripts/voice-stats.py --lang ko sources/*.md
```

### Baseline: the current site, which is Claude's voice, not hers

Measured over the three English notes on the site (18,647 prose words), all
written or translated by Claude. This is the thing to move *away* from, and the
yardstick for whether a later rewrite actually moved.

| Metric | Value |
|---|---|
| em dash | 14.00 / 1k |
| "not X but Y" | 1.13 / 1k |
| "It is / This is" + emphasis | 1.66 / 1k |
| semicolon | 1.34 / 1k |
| paragraph-initial But/And | 0.70 / 1k |
| slop vocabulary | 0.05 / 1k |
| sentence length | mean 23.4 · median 19 · p10 7 · p90 45 |

And over the four Korean notes (13,659 prose words), same provenance:

| Metric | Value |
|---|---|
| em dash | 15.23 / 1k |
| 번역투 ~에 대한 | 1.54 / 1k |
| 번역투 ~을/를 통해 | 0.22 / 1k |
| ~다 종결 | 55.71 / 1k |
| ~습니다 종결 | 0.00 / 1k |
| sentence length | mean 13.5 · median 11 · p10 5 · p90 24 |

`--baseline` prints any sample as a multiple of these, which is how a Tier-B
observation earns promotion to Tier A.

Note the median-vs-mean gap: the length variation is already there, so uniform
sentence length is not the tell on this site. The rhetorical frames are.
