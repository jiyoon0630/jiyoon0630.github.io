#!/usr/bin/env python3
"""Invariant check for a voice conversion: compares a post's committed version
(git HEAD) with the working-tree version and fails on any structural or factual
drift. Usage: python3 scripts/voice-check.py _posts/<post>.md   (run from repo root)"""
import re, subprocess, sys
from collections import Counter
path = sys.argv[1]
old = subprocess.run(['git', 'show', f'HEAD:{path}'], capture_output=True, text=True, check=True).stdout
new = open(path, encoding='utf-8').read()

def split_fm(t):
    _, fm, body = t.split('---', 2)
    return fm, body
ofm, ob = split_fm(old); nfm, nb = split_fm(new)
problems = []
def cmp(name, a, b):
    if a != b:
        ca, cb = Counter(a), Counter(b)
        problems.append(f"{name}: removed {list((ca-cb).elements())[:8]} added {list((cb-ca).elements())[:8]}")

# front matter: identical except summary
strip_sum = lambda fm: re.sub(r'(?m)^summary:.*$', '', fm)
if strip_sum(ofm) != strip_sum(nfm):
    problems.append("front matter changed outside `summary`")

fence = r'```.*?```'
cmp('code fences', re.findall(fence, ob, re.S), re.findall(fence, nb, re.S))
ob_nf, nb_nf = re.sub(fence, '', ob, flags=re.S), re.sub(fence, '', nb, flags=re.S)
cmp('display math', re.findall(r'\$\$.*?\$\$', ob_nf, re.S), re.findall(r'\$\$.*?\$\$', nb_nf, re.S))
inl = lambda t: re.findall(r'(?<![\\$])\$([^$\n]+?)\$(?!\$)', re.sub(r'\$\$.*?\$\$', '', t, flags=re.S))
cmp('inline math', sorted(inl(ob_nf)), sorted(inl(nb_nf)))
cmp('inline code', sorted(re.findall(r'`[^`\n]+`', ob_nf)), sorted(re.findall(r'`[^`\n]+`', nb_nf)))
cmp('headings', re.findall(r'(?m)^#{1,6} .*$', ob_nf), re.findall(r'(?m)^#{1,6} .*$', nb_nf))
cmp('callout titles', re.findall(r'(?m)^>\s*#{2,6} .*$', ob_nf), re.findall(r'(?m)^>\s*#{2,6} .*$', nb_nf))
cmp('images', sorted(re.findall(r'!\[[^\]]*\]\([^)]*\)', ob_nf)), sorted(re.findall(r'!\[[^\]]*\]\([^)]*\)', nb_nf)))
cmp('link targets', sorted(re.findall(r'\]\(([^)]+)\)', ob_nf)), sorted(re.findall(r'\]\(([^)]+)\)', nb_nf)))
cmp('html tags', sorted(re.findall(r'<[^>]+>', ob_nf)), sorted(re.findall(r'<[^>]+>', nb_nf)))
# tables: same number of rows and columns per row
trows = lambda t: [l.count('|') for l in t.splitlines() if l.lstrip().lstrip('>').lstrip().startswith('|')]
if trows(ob_nf) != trows(nb_nf):
    problems.append(f"table shape changed: {len(trows(ob_nf))} rows -> {len(trows(nb_nf))} rows (or column counts differ)")
# numbers in prose (math excluded): multiset must match
nomath = lambda t: re.sub(r'\$\$.*?\$\$|\$[^$\n]+\$', '', t, flags=re.S)
num = lambda t: sorted(re.findall(r'\d+(?:[.,]\d+)*%?', nomath(t)))
cmp('numbers', num(ob_nf), num(nb_nf))
# blockquote / list skeleton
cmp('callout count', [len(re.findall(r'(?m)^>\s*### ', ob_nf))], [len(re.findall(r'(?m)^>\s*### ', nb_nf))])

# register
body = re.sub(r'(?m)^\s*>?\s*\|.*$', '', nb_nf)          # tables excluded
body = re.sub(r'(?m)^#{1,6} .*$', '', body)              # headings excluded
plain = re.findall(r'[^\s]*(?<!니)다[.!?](?=\s|\*|$)', body)
polite = re.findall(r'니다[.!?]', body)
emd = new.count('—'); words = len(nb.split())
print(f"합쇼체 {len(polite)} / 해라체 {len(plain)}   em dash {emd} ({1000*emd/words:.1f}/1k, was {1000*old.count('—')/len(ob.split()):.1f}/1k)")
if plain:
    print("remaining 해라체 endings (check each is a quotation or intentional):", plain[:15])
if problems:
    print("FAIL"); [print(" -", p) for p in problems]; sys.exit(1)
print("OK: structure, math, numbers, links and headings unchanged")
