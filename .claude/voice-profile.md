# Voice profile — Jiyoon Kim

**A record of what her writing actually looks like.** Each entry is one trait
observed in a Tier A document (text she wrote), written down so it survives past
the session that noticed it. Conversion of existing posts is a separate job; the
procedure for that lives in `CLAUDE.md`. This file is the findings.

**Status: Korean established from three documents (20,100 words). English: no
Tier A sample yet. Still dormant — apply only when she says so.**

Provenance of every sample so far, in her words: an AI wrote the first draft and
she did the style pass herself. So the **style layer** — register, emphasis,
hedging, punctuation, sentence shape — is hers and is what these entries record.
Factual content is not evidence of voice.

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

Traits marked **stable** held across the documents that were checked for them. Two genre notes first:
**sentence length and terminator density are genre-driven, not voice** — the
conference report ran mean 21.4 words/sentence, the tutorial 16.2, the
explainer 16.4 — so never set
a target for them. **Register, hedging and emphasis did not move**, and those are
what to convert toward.

### 합쇼체 throughout — never 해라체
- **Pattern:** every sentence ends in ~습니다 / ~입니다 / ~합니다. No plain
  declaratives (~한다 / ~이다 / ~있다) anywhere, including callouts and lists.
- **Evidence:** doc #1 — 습니다 ×90, 입니다 ×37, plain 0. Doc #2 — 입니다 ×120,
  습니다 ×107, 합니다 ×91, plain 0. voice-stats: 습니다 15.9–22.0/1k vs 0.00 in
  every Claude-written post. (Its "~다 종결" line matches both styles, so read
  the 습니다 line, not that one.) Doc #3 — 습니다 ×172, plain 0.
- **Example (invented):** 이 설정은 캐시 적중률을 높이는 데 효과적입니다.
- **Claude instead writes:** 이 설정은 캐시 적중률을 높이는 데 효과적이다.
- **Confidence:** solid — the largest single gap between her and the site.

### The verdict comes first, as a whole bolded sentence
- **Pattern:** a paragraph or section opens with its conclusion as one fully
  bolded sentence, often standing alone; plain sentences after it carry the
  evidence. Sections close with 정리하면… restating the verdict.
- **Evidence:** doc #1 — 134 bold spans in 4,100 words. Doc #2 — 472 bold markers
  in 6,747; 핵심은… opens 9 paragraphs, 정리하면… closes 4 sections. Doc #3 —
  116 bold spans in 9,253 words, every section opening on a bolded 한 줄 요약.
- **Example (invented):** **결론부터 말하면, 인덱스를 다시 만드는 편이
  빠릅니다.** 기존 인덱스는 조회 패턴이 바뀐 뒤로 한 번도 갱신되지 않았습니다.
- **Claude instead writes:** builds toward the claim, and bolds a short phrase
  inside a longer paragraph rather than the whole verdict.
- **Confidence:** solid.

### Hedging only in the verdict sentence, from a fixed set
- **Pattern:** evidence sentences are flat assertions. Only the paragraph- or
  section-final sentence softens, using 판단됩니다 / 할 수 있겠습니다 /
  생각됩니다 / 보입니다.
- **Evidence:** doc #2 — 할 수 있겠습니다 ×5, 판단됩니다 ×3, 생각됩니다 ×1, all
  in final position. Doc #1 — same set, same position. **Zero** occurrences of any
  of them in the Claude-written Korean posts. Doc #3 (an explainer) — **zero**;
  its single inference is marked 추정됩니다 instead (see the next entry).
- **Example (invented):** 따라서 배치 크기를 줄이는 편이 안전하다고 판단됩니다.
- **Claude instead writes:** hedges mid-sentence with adverbs, or ends on ~할 수
  있다 / ~로 보인다 throughout.
- **Confidence:** solid, but **genre-bound**: the hedge appears where she is
  judging (reports, surveys), not where she is explaining how something works.
  An explainer with no verdict gets no hedge — do not add one.

### Three verbs for three kinds of claim
- **Pattern:** what a company claimed → 발표하였습니다 / 밝혔습니다 /
  제시하였습니다. How it is received → 평가 받고 있습니다. Her own inference →
  판단됩니다 / 보입니다. She keeps the three apart; estimates stay marked 추정.
- **Evidence:** doc #1 throughout; doc #2 marks a closed model's internals as
  추정 / 비공개 rather than asserting them.
- **Example (invented):** 제조사는 두 배 개선을 발표하였고, 업계에서는 조건부
  결과로 평가 받고 있으나, 재현 조건이 공개되지 않은 점은 한계로 판단됩니다.
- **Claude instead writes:** collapses all three into one asserted statement, or
  hedges everything uniformly.
- **Confidence:** solid — the trait most worth protecting in any rewrite.

### Paired contrast: "A가 ~였다면, B는 ~입니다"
- **Pattern:** whenever two approaches are distinguished, one sentence with this
  scaffold, usually with both halves bolded.
- **Evidence:** doc #2 ×7; doc #3 ×10, mostly opening the paragraph that
  introduces the next model ("X가 ~였다면, Y는 …").
- **Example (invented):** **REST가 자원을 URL로 드러내는 방식이었다면, RPC는
  동작 이름을 그대로 노출하는 방식입니다.**
- **Claude instead writes:** "A와 B의 차이는 ~이다", or a comparison table.
- **Confidence:** solid — stable across #2 and #3; not counted in doc #1.

### 다만 pivots; 그래서 never opens a sentence
- **Pattern:** concedes with 다만, contrasts with 반면, narrows with 특히 / 즉.
  Almost never opens a sentence with a causal connective.
- **Evidence:** doc #1 — 다만 ×6, 반면 ×4, 특히 ×4, 즉 ×2, 그러나 ×1, 따라서 ×1,
  그래서 ×0. Doc #3 — 다만 ×10, 그래서 ×0.
- **Example (invented):** 처리량은 개선되었습니다. 다만 실패율은 함께 측정되지
  않았습니다.
- **Claude instead writes:** 그래서 / 따라서 / 그러므로 at paragraph openings.
- **Confidence:** solid — stable across #1 and #3.

### The em dash is structural, and rare
- **Pattern:** used only after a bolded label to introduce its content (headings,
  list items) — never as a mid-sentence aside. For an aside she uses 즉 or
  parentheses instead.
- **Evidence:** doc #1 — 1.71/1k, all 7 after a label. Doc #2 — 2.37/1k.
  Doc #3 — 1.19/1k.
  Claude-written posts: 14–15/1k, almost all mid-sentence — about 7× her rate.
- **Example (invented):** ### 2-3. 평가 지표 — 처리량, 지연, 실패율
- **Claude instead writes:** 이 값은 캐시에 남지 않는다 — 매 요청마다 다시
  계산된다.
- **Confidence:** solid. (This overturns the earlier note that the em dash was
  "a feature of the material": that was measured on Claude's own text.)

### English terms bare; the gloss attaches to the Korean
- **Pattern:** technical English sits in Korean prose undeclined and unitalicised,
  with particles attached directly. An acronym is expanded once at first use, then
  used bare. Definitional glosses run 한국어(english) — 명시적(explicit) — not the
  reverse.
- **Evidence:** both documents throughout.
- **Example (invented):** 요청을 비동기(asynchronous)로 처리하면 queue가 밀리지
  않습니다.
- **Claude instead writes:** translates the term, re-glosses it on later use, or
  writes English(한국어).
- **Confidence:** solid.

### Arrows compress a chain
- **Pattern:** → inside prose and headings for a transformation or progression
  that would otherwise take a clause.
- **Evidence:** doc #1 ×8; doc #2 in headings and callouts.
- **Example (invented):** 원시 로그 → 정규화 → 학습 배치 순으로 처리합니다.
- **Claude instead writes:** 원시 로그를 정규화한 뒤 학습 배치로 변환한다.
- **Confidence:** solid.

### Define before use, in a box
- **Pattern:** a term gets its own 💡 callout before or right where it is first
  needed; each model walkthrough opens with 작동 흐름은 다음과 같습니다. and
  numbered steps.
- **Evidence:** doc #2 — 24 definition callouts, the formula ×6.
- **Claude instead writes:** defines inline, parenthetically, on first use.
- **Confidence:** solid for explanatory writing; not a trait of her reports.

### ~을/를 통해 is not a tell for her
- **Pattern:** she uses ~을/를 통해 freely, so it must not be stripped as 번역투.
  The 번역투 she genuinely avoids: ~의 경우, ~되어지다, heavy ~에 대한.
- **Evidence:** doc #1 — 2.44/1k (11× the Claude posts); ~의 경우 and ~되어지 at 0.
- **Claude instead writes:** avoids ~을/를 통해 and reaches for ~로.
- **Confidence:** solid for doc #1; doc #2 uses it far less (0.59/1k).

### Terse parenthetical fragments
- **Pattern:** a verbless scope note in parentheses fences a claim without
  spending a sentence on it.
- **Example (invented):** 전처리 단계는 그대로 유지됩니다. (캐시는 여전히 필요)
- **Claude instead writes:** a full hedging sentence.
- **Confidence:** tentative — several instances, clustered in one callout.

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
| 1 | Conference report — robotics developer conference, RFM & data trends | KO | 4,100 | 2026-09-18 |
| 2 | Technical survey — world models, three trends | KO | 6,747 | 2026-09-22 |
| 3 | Technical explainer — VLA development and training methods | KO | 9,253 | 2026-09-24 |

All three: AI first draft, her style pass. Counters: **Korean 20,100 words ·
English 0 words.** Prose traits are recordable from
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

- **The mid-sentence em dash.** Her rate is 1.7–2.4 per 1,000 words and only
  after a label; Claude's Korean and English posts sit at 14–15, nearly all
  mid-sentence. See the Korean entry for the replacement (즉, parentheses).
