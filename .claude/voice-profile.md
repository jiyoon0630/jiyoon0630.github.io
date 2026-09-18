# Voice profile — Jiyoon Kim

**A record of what her writing actually looks like.** Each entry is one trait
observed in a Tier A document (text she wrote), written down so it survives past
the session that noticed it. Conversion of existing posts is a separate job; the
procedure for that lives in `CLAUDE.md`. This file is the findings.

**Status: EMPTY — no Tier A document received yet.**

---

## How to add an entry

Only two rules.

**⓵ Never quote source material.** This file is in a public repository and her
documents may be internal. Describe the pattern, and illustrate it with an
**invented sentence on a neutral topic**.

**⓶ Say what the evidence was.** A trait seen once is a guess; a trait seen in
three documents is a fact. The entry says which.

Format:

```
### <short name for the trait>
- **Pattern:** what she does, stated so it can be followed.
- **Evidence:** doc #1, #3 — seen N times / throughout.
- **Example (invented):** a neutral sentence showing it.
- **Claude instead writes:** the default this displaces. ← the useful line
- **Confidence:** solid | tentative
```

The **Claude instead writes** line is what makes an entry actionable — a trait
with no contrast to the default cannot be applied to anything.

---

## Korean (한국어)

*No entries yet.*

<details>
<summary>What to watch for while reading a Tier A document</summary>

- **Sentence endings** — ~다 / ~이다 / 명사형 종결의 비율, and whether it shifts
  by section (body vs. conclusion vs. callout)
- **Sentence length and rhythm** — short declaratives vs. long chained clauses;
  where she breaks a sentence rather than joining it
- **Connectives** — which ones she actually uses (그런데 / 그래서 / 다만 / 반면)
  and which she avoids; how often a paragraph opens with one
- **Argument order** — does the claim come first (두괄식) or does she build to it
- **번역투 she avoids** — ~의 경우, ~에 다름 아니다, ~되어지다, overuse of
  ~에 대한, ~을 통해
- **English-term handling** — which technical terms stay in English, whether
  they get a Korean gloss, how she brackets them
- **Emphasis** — bold, quotation, dashes; how heavy-handed
- **Her own tics** — repeated constructions that are recognisably hers
- **Anything a Claude draft would never produce** — the most valuable find
</details>

## English

*No entries yet.*

<details>
<summary>What to watch for while reading a Tier A document</summary>

- **Register** — how formal, how much contraction, first person or not
- **Sentence length** distribution, and whether she varies it deliberately
- **Openers** — how she starts a paragraph or a section
- **Punctuation habits** — em dash, semicolon, colon, parentheses
- **Hedging** — how she marks uncertainty, and how often
- **Terminology** — the precision distinctions she insists on
- **Constructions she avoids** — recorded from evidence, never assumed
</details>

---

## Tier A documents these entries come from

| # | Document (no confidential detail) | Language | Words | Date |
|---|---|---|---|---|
| — | *none yet* | — | — | — |

Counters: **Korean 0 words · English 0 words.** Prose traits are recordable from
the first document. Statistical traits (density, sentence length) need roughly
2,000–5,000 words per language before a number means anything.

Tech-review sources count — read the original for voice **before** sanitizing,
since the sanitizer's repairs are Claude's prose, not hers.

## Measured numbers

Statistical entries carry a number from `scripts/voice-stats.py`, which is also
how a claim about her style gets checked rather than asserted:

```bash
python3 scripts/voice-stats.py --lang ko --baseline <her document>
```

`--baseline` reports each rate as a multiple of the site's current posts, all of
which Claude wrote — so a rate far from `1.00x` is a real difference between her
writing and the assistant's, and belongs in an entry above. The Tier B reference
numbers and the conversion procedure are in `CLAUDE.md`.

## Anti-patterns — already known, no evidence needed

General LLM tics rather than findings about her. Strip them whenever they appear.

- "It is not X, it is Y" / "not merely X but Y" as a rhetorical frame
- Vacuous emphasis openers: "It is worth noting", "The key point is",
  "Importantly", "Crucially"
- Triads used for rhythm rather than because there are three things
- Paragraphs that open with a formal connective several times in a row
- Closing sentences that restate the paragraph without adding anything
- Vocabulary: delve, leverage (as a verb), nuanced, underscore, tapestry,
  realm, testament, landscape (figurative)
- Uniform sentence length across a whole passage

An em dash is **not** on this list on current evidence — about 14–15 per 1,000
words across the existing posts in both languages, i.e. a feature of the
material. If a Tier A document comes in far below that, it earns an entry with
the number attached.
