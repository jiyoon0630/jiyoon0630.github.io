#!/usr/bin/env python3
"""Print style statistics for prose files, so voice-profile.md stays evidence-based.

Usage:
    python3 scripts/voice-stats.py _posts/*.md
    python3 scripts/voice-stats.py --lang ko sources/some-draft.md

Strips front matter, fenced code, tables and display math before counting, so
the numbers describe prose rather than markup.
"""
import argparse, json, re, sys, unicodedata
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

# Rates per 1,000 prose words measured over the site's own posts, all of which
# Claude wrote or translated. This is the assistant's fingerprint, not the
# owner's — `--baseline` reports how far a sample sits from it. A reviewed
# Claude draft that lands on these numbers tells you review did not touch the
# generation-level features; a sharp divergence is real editing signal.
BASELINE = {
    "en": {  # 18,647 prose words, 3 English notes
        "em dash —": 14.00, "en dash –": 0.64,
        "not X but Y": 1.13, "It is/This is + emphasis": 1.66,
        "the point/key is": 0.05, "worth -ing": 0.11, "in other words": 0.16,
        "hedge adverbs": 0.16, "slop vocabulary": 0.05, "reader address": 0.27,
        "para-initial But/And": 0.70, "semicolon": 1.34,
        "__mean__": 23.4, "__median__": 19,
    },
    "ko": {  # 13,659 prose words, 4 Korean notes
        "em dash —": 15.23, "en dash –": 0.37,
        "\ubc88\uc5ed\ud22c ~\uc758 \uacbd\uc6b0": 0.07,
        "\ubc88\uc5ed\ud22c ~\uc5d0 \ub300\ud55c": 1.54,
        "\ubc88\uc5ed\ud22c ~\uc744/\ub97c \ud1b5\ud574": 0.22,
        "\ud53c\ub3d9 ~\ub418\uc5b4\uc9c0": 0.00,
        "~\uc5d0 \ub2e4\ub984 \uc544\ub2c8": 0.00,
        "\ubb38\ub450 \ub530\ub77c\uc11c/\uadf8\ub7ec\ubbc0\ub85c": 0.00,
        "\uba85\uc0ac\ud615 \uc885\uacb0 ~\uc74c/~\ud568": 0.00,
        "~\ub2e4 \uc885\uacb0": 55.71, "~\uc2b5\ub2c8\ub2e4 \uc885\uacb0": 0.00,
        "__mean__": 13.5, "__median__": 11,
    },
}

def delta(name, rate, ref):
    """Render the gap to a reference profile, or nothing if there is none."""
    if ref is None:
        return ""
    base = ref.get(name)
    if base is None:
        return "        —"
    if base == 0:
        return "   new" if rate else "   ="
    return f"   {rate / base:>5.2f}x"


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

def report(paths, lang, ref=None, ref_label="", save=None):
    text = strip_markup("\n".join(Path(p).read_text(encoding="utf-8") for p in paths))
    words = len(text.split())
    if not words:
        print("no prose found"); return
    print(f"\n{'='*58}\nfiles: {len(paths)}   prose words: {words:,}   lang: {lang}\n{'='*58}")

    rates = {}
    if ref_label:
        print(f"(x = multiple of {ref_label})")
    print("\n-- character fingerprints --")
    any_char = False
    for name, ch in CHAR_TELLS.items():
        n = text.count(ch)
        if n:
            any_char = True
            r = n / words * 1000
            print(f"   {name:<16} {n:>5}   {r:>7.2f}/1k{delta(name, r, ref)}")
            rates[name] = round(r, 2)
    if not any_char:
        print("   (none)")

    tells = EN_TELLS if lang == "en" else KO_TELLS
    print(f"\n-- phrasal tells ({lang}) --")
    for name, pat in tells.items():
        n = len(re.findall(pat, text, flags=re.I))
        r = n / words * 1000
        print(f"   {name:<26} {n:>5}   {r:>7.2f}/1k{delta(name, r, ref)}")
        rates[name] = round(r, 2)

    sents = sentences(text, lang)
    if sents:
        lens = sorted(len(s.split()) for s in sents)
        mid = lens[len(lens)//2]
        print(f"\n-- sentence length --")
        mean = sum(lens) / len(lens)
        print(f"   count {len(lens)}   mean {mean:.1f}   median {mid}"
              f"   p10 {lens[len(lens)//10]}   p90 {lens[len(lens)*9//10]}")
        rates["__mean__"], rates["__median__"] = round(mean, 1), mid
        if ref:
            print(f"   reference  mean {ref.get('__mean__')}   median {ref.get('__median__')}"
                  f"   ({ref_label})")
    if save:
        Path(save).write_text(json.dumps(rates, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"\ntarget written to {save}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--lang", choices=["en", "ko", "auto"], default="auto")
    ap.add_argument("--baseline", action="store_true",
                    help="compare against the Claude-written site posts (Tier B)")
    ap.add_argument("--target", metavar="FILE",
                    help="compare against a saved target profile (Tier A)")
    ap.add_argument("--save-target", metavar="FILE", dest="save_target",
                    help="write these measurements out as a target profile")
    a = ap.parse_args()
    lang = a.lang
    if lang == "auto":
        sample = "".join(Path(p).read_text(encoding="utf-8")[:4000] for p in a.paths)
        hangul = sum(1 for c in sample if "가" <= c <= "힣")
        lang = "ko" if hangul > len(sample) * 0.08 else "en"
    ref, label = None, ""
    if a.target:
        ref = json.loads(Path(a.target).read_text(encoding="utf-8"))
        label = f"target {a.target}"
    elif a.baseline:
        ref, label = BASELINE.get(lang, {}), "Claude-written site posts"
    report(a.paths, lang, ref, label, a.save_target)

if __name__ == "__main__":
    main()
