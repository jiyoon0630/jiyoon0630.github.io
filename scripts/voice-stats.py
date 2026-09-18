#!/usr/bin/env python3
"""Print style statistics for prose files, so voice-profile.md stays evidence-based.

Usage:
    python3 scripts/voice-stats.py _posts/*.md
    python3 scripts/voice-stats.py --lang ko sources/some-draft.md

Strips front matter, fenced code, tables and display math before counting, so
the numbers describe prose rather than markup.
"""
import argparse, re, sys, unicodedata
from pathlib import Path

CHAR_TELLS = {
    "em dash —": "—", "en dash –": "–",
    "curly ‘": "‘", "curly ’": "’",
    "curly “": "“", "curly ”": "”",
    "ellipsis …": "…", "nbsp": " ",
    "narrow nbsp": " ", "zero width": "​",
}

EN_TELLS = {
    "not X but Y": r"\bnot\s+\w[\w\s,\-]{0,40}?\s+but\s+",
    "It is/This is + emphasis": r"\b(?:It is|It's|That is|This is)\s+(?:not|precisely|exactly|the)\b",
    "the point/key is": r"\b(?:the point|the key|what matters|the real)\s+(?:is|here)\b",
    "worth -ing": r"\bworth\s+\w+ing\b",
    "in other words": r"\bin other words\b",
    "hedge adverbs": r"\b(?:essentially|fundamentally|arguably|notably|crucially|importantly)\b",
    "slop vocabulary": r"\b(?:delve|leverage|nuanced|underscore|tapestry|testament|realm|landscape)\b",
    "reader address": r"\b(?:Let's|let us|we can see|as we|note that)\b",
    "para-initial But/And": r"(?m)^\s*(?:But|And)\b",
    "semicolon": r";",
}

KO_TELLS = {
    "번역투 ~의 경우": r"의 경우",
    "번역투 ~에 대한": r"에 대한",
    "번역투 ~을/를 통해": r"[을를] 통해",
    "피동 ~되어지": r"되어지",
    "~에 다름 아니": r"에 다름 아니",
    "문두 따라서/그러므로": r"(?m)^\s*(?:따라서|그러므로|또한|하지만)",
    "명사형 종결 ~음/~함": r"[음함]\.\s",
    "~다 종결": r"다\.\s",
    "~습니다 종결": r"습니다\.",
}

def strip_markup(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        text = parts[2] if len(parts) > 2 else text
    text = re.sub(r"```.*?```", "", text, flags=re.S)      # fenced code
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)    # display math
    text = re.sub(r"(?m)^\s*\|.*$", "", text)              # table rows
    text = re.sub(r"`[^`]*`", "", text)                    # inline code
    return text

def sentences(text: str, lang: str):
    pat = r"(?<=[.!?])\s+" if lang == "en" else r"(?<=[.!?다])\s+"
    return [s for s in re.split(pat, text) if 8 < len(s) < 600]

def report(paths, lang):
    text = strip_markup("\n".join(Path(p).read_text(encoding="utf-8") for p in paths))
    words = len(text.split())
    if not words:
        print("no prose found"); return
    print(f"\n{'='*58}\nfiles: {len(paths)}   prose words: {words:,}   lang: {lang}\n{'='*58}")

    print("\n-- character fingerprints --")
    any_char = False
    for name, ch in CHAR_TELLS.items():
        n = text.count(ch)
        if n:
            any_char = True
            print(f"   {name:<16} {n:>5}   {n/words*1000:>7.2f}/1k")
    if not any_char:
        print("   (none)")

    tells = EN_TELLS if lang == "en" else KO_TELLS
    print(f"\n-- phrasal tells ({lang}) --")
    for name, pat in tells.items():
        n = len(re.findall(pat, text, flags=re.I))
        print(f"   {name:<26} {n:>5}   {n/words*1000:>7.2f}/1k")

    sents = sentences(text, lang)
    if sents:
        lens = sorted(len(s.split()) for s in sents)
        mid = lens[len(lens)//2]
        print(f"\n-- sentence length --")
        print(f"   count {len(lens)}   mean {sum(lens)/len(lens):.1f}   median {mid}"
              f"   p10 {lens[len(lens)//10]}   p90 {lens[len(lens)*9//10]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--lang", choices=["en", "ko", "auto"], default="auto")
    a = ap.parse_args()
    lang = a.lang
    if lang == "auto":
        sample = "".join(Path(p).read_text(encoding="utf-8")[:4000] for p in a.paths)
        hangul = sum(1 for c in sample if "가" <= c <= "힣")
        lang = "ko" if hangul > len(sample) * 0.08 else "en"
    report(a.paths, lang)

if __name__ == "__main__":
    main()
