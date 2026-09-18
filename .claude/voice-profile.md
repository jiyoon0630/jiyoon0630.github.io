# Voice profile — Jiyoon Kim

The target her posts should sound like, and the instructions for converting
Claude-written text into it.

**Status: EMPTY — no reference documents received yet. Do not apply it yet.**

---

## What the two tiers mean

- **Tier A = text in her voice.** Documents she wrote herself. She supplies
  these; they are the reference this file is derived from, and the target.
- **Tier B = text in Claude's voice.** Everything currently on the site — all
  seven notes, both languages — plus anything Claude drafts in future.

The job is **converting Tier B into Tier A**, not sorting text into the two
buckets. The buckets only exist to name the starting point and the destination.

Two things follow:

1. **Tier A documents are reference material, never edited.** They are read to
   extract patterns and measured to set the target numbers.
2. **Every Tier B text is a conversion candidate.** When she gives the signal,
   the existing posts get a rewrite pass, and new posts are written to the
   target from the start.

## Two hard rules

**⓵ Never quote source material here.** This file is in a public repository, and
her reference documents may be internal. Record *patterns*, and illustrate them
with **invented examples on neutral topics**. If a pattern cannot be shown
without confidential content, describe it in words instead.

**⓶ It is dormant until she says otherwise.** Notes and translations are written
the normal way until she gives the signal. Do not half-apply an unfinished
profile — a partial imitation is worse than none.

---

## Tier A reference documents

| # | Document (no confidential detail) | Language | Words | Date added |
|---|---|---|---|---|
| — | *none yet* | — | — | — |

Counters: **Korean 0 words · English 0 words.** Roughly 2,000–5,000 words per
language makes the target numbers stable. Below that, the prose patterns below
are still usable; the statistics are not.

Tech-review sources arriving for sanitization also count as Tier A material —
read the original for voice **before** sanitizing, since the sanitizer's repairs
are Claude's prose, not hers.

---

## The target — Korean (한국어)

*Nothing recorded yet.* To be filled from Tier A documents:

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

## The target — English

*Nothing recorded yet.* To be filled from Tier A documents:

- **Register** — how formal, how much contraction, first person or not
- **Sentence length** distribution, and whether she varies it deliberately
- **Openers** — how she starts a paragraph or a section
- **Punctuation habits** — em dash, semicolon, colon, parentheses
- **Hedging** — how she marks uncertainty, and how often
- **Terminology** — the precision distinctions she insists on
- **Constructions she avoids** — filled in from evidence, not assumed

---

## Converting Tier B → Tier A

The pass to run on an existing post, or on a fresh draft before it is posted.

1. **Preserve meaning exactly.** This is a style conversion. No claim may get
   stronger or weaker, no number may move, no hedge may be dropped, no
   terminology distinction may be collapsed. If a rewrite cannot keep the
   meaning, leave the sentence alone.
2. **Preserve structure.** Math verbatim, callout boxes, tables, code fences and
   ASCII diagrams untouched. Front matter untouched except `summary`, which is
   prose and gets converted too.
3. **Strip the anti-patterns below.** These are safe to remove today, before any
   profile exists, because they are LLM tics rather than anything of hers.
4. **Move toward the target numbers**, once there are any — see Measurement.
   Do not chase a number at the cost of sense: a metric matching while the
   paragraph reads worse is a failed conversion.
5. **Measure before and after** and record the movement, so the claim that a
   post now sounds like her is evidence rather than assertion.
6. **Convert the Korean and the English independently.** The English post is not
   a translation of the converted Korean; both are conversions of their own
   Tier B text toward their own target.

## Anti-patterns — strip these regardless of profile

General LLM tics, independent of her voice. Safe to act on now.

- "It is not X, it is Y" / "not merely X but Y" as a rhetorical frame
- Vacuous emphasis openers: "It is worth noting", "The key point is",
  "Importantly", "Crucially"
- Triads used for rhythm rather than because there are three things
- Paragraphs that open with a formal connective several times in a row
- Closing sentences that restate the paragraph without adding anything
- Vocabulary: delve, leverage (as a verb), nuanced, underscore, tapestry,
  realm, testament, landscape (figurative)
- Uniform sentence length across a whole passage

Note: an em dash is **not** on this list on evidence so far — density is about
14–15 per 1,000 words across the current posts in both languages, which says it
is a feature of the material. If a Tier A document comes in far below that, it
moves onto the list with the number attached.

---

## Measurement

`scripts/voice-stats.py` measures character fingerprints, phrasal tells and
sentence-length distribution, so both the target and the conversion stay
evidence-based rather than impressionistic.

```bash
# 1. Set the target from her own documents (do this when Tier A arrives)
python3 scripts/voice-stats.py --lang ko --save-target .claude/target-ko.json sources/*.md

# 2. See how far a Claude-written post sits from it
python3 scripts/voice-stats.py --lang ko --target .claude/target-ko.json _posts/2026-08-01-dyna-2-scaling-law.md

# 3. Re-run after converting; the multiples should be closer to 1.00x
```

Target files are gitignored — they are derived from internal documents.

### Tier B fingerprint — where the site is starting from

Measured over the site's own posts, all Claude-written. This is the thing being
converted away from, and the yardstick for whether a conversion actually moved.

**English — 3 notes, 18,647 prose words**

| Metric | Value |
|---|---|
| em dash | 14.00 / 1k |
| "not X but Y" | 1.13 / 1k |
| "It is / This is" + emphasis | 1.66 / 1k |
| semicolon | 1.34 / 1k |
| paragraph-initial But/And | 0.70 / 1k |
| slop vocabulary | 0.05 / 1k |
| sentence length | mean 23.4 · median 19 · p10 7 · p90 45 |

**Korean — 4 notes, 13,659 prose words**

| Metric | Value |
|---|---|
| em dash | 15.23 / 1k |
| 번역투 ~에 대한 | 1.54 / 1k |
| 번역투 ~을/를 통해 | 0.22 / 1k |
| ~다 종결 | 55.71 / 1k |
| ~습니다 종결 | 0.00 / 1k |
| sentence length | mean 13.5 · median 11 · p10 5 · p90 24 |

`--baseline` prints any text as a multiple of these numbers. Useful for checking
whether a draft still reads like the old posts; `--target` is what the
conversion is actually aimed at.

Note the median-vs-mean gap: length variation is already present, so uniform
sentence length is not the tell on this site. The rhetorical frames are.
