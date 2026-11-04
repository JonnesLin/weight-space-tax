"""Summarise test B. usage: python score_loop.py loop_gen.jsonl"""
import json, sys
from collections import defaultdict
from math import comb

import gzip
opener = gzip.open if sys.argv[1].endswith(".gz") else open
rows = [json.loads(l) for l in opener(sys.argv[1], "rt")]
by = defaultdict(dict)
for r in rows:
    by[r["dose"]][r["pid"]] = r
doses = sorted(by)
base = by.get(0.0, {})


def sign_p(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


print(f"{'dose':>5s} {'n':>3s} | {'finished':>8s} {'loop':>5s} {'cap':>4s} | {'correct':>7s} | {'mean tok':>8s} "
      f"{'median tok (finished)':>21s} | correct vs c=0 (lost/gained, p)")
for c in doses:
    R = list(by[c].values())
    n = len(R)
    fin = sum(r["reason"] == "eos" for r in R)
    lp = sum(r["reason"] == "loop" for r in R)
    cap = sum(r["reason"] == "cap" for r in R)
    cor = sum(r["correct"] for r in R)
    ft = sorted(r["n_tokens"] for r in R if r["reason"] == "eos")
    med = ft[len(ft) // 2] if ft else 0
    mean = sum(r["n_tokens"] for r in R) / n
    paired = ""
    if c != 0.0 and base:
        common = [p for p in by[c] if p in base]
        lost = sum(base[p]["correct"] and not by[c][p]["correct"] for p in common)
        gain = sum(by[c][p]["correct"] and not base[p]["correct"] for p in common)
        paired = f"-{lost}/+{gain}, p={sign_p(lost, gain):.3g} (n={len(common)})"
    print(f"{c:>5g} {n:>3d} | {fin:>8d} {lp:>5d} {cap:>4d} | {cor:>7d} | {mean:>8.0f} {med:>21d} | {paired}")
