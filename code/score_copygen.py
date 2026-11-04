"""Score test A (copy/repetition generation): base vs edited, paired per item.
usage: python score_copygen.py copygen_greedy.jsonl [--show N]   (N = print N diffs where the two models differ)"""
import difflib, json, re, sys
from collections import defaultdict

path = sys.argv[1]
show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0
rows = [json.loads(l) for l in open(path)]


def norm(s):
    s = s.strip()
    lines = s.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    lines = [l.rstrip() for l in lines if l.strip() not in ("TEXT START", "TEXT END")]
    return "\n".join(lines).strip()


def toks(s):
    return re.findall(r"[^\s]+|\n", s)


def diff_stats(exp, out):
    a, b = toks(exp), toks(out)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ops = [o for o in sm.get_opcodes() if o[0] != "equal"]
    cost = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
    return len(ops), cost, len(a)


def score(r):
    m = r["meta"] if "meta" in r else r
    task, exp = m["task"], m["expected"]
    out = norm(r.get("answer_text") or r.get("output_text") or "")
    exp = "\n".join(l.rstrip() for l in exp.split("\n")).strip()
    spans, cost, n = diff_stats(exp, out)
    res = {"exact": out == exp, "spans": spans, "cost": cost, "n_exp_tokens": n, "hit_max": bool(r.get("hit_max")),
           "out": out, "exp": exp}
    if task == "repeat_word":
        res["count"] = len(re.findall(rf"(?<!\S){re.escape(m['word'])}(?!\S)", out))
        res["count_ok"] = res["count"] == m["n"]
    if task == "repeat_sentence":
        res["count"] = sum(1 for l in out.split("\n") if l.strip() == m["sentence"])
        res["count_ok"] = res["count"] == m["n"] and len([l for l in out.split("\n") if l.strip()]) == m["n"]
    if task == "count_list":
        nums = [int(x) for x in re.findall(r"-?\d+", out)]
        want = list(range(m["a"], m["b"] + 1))
        k = 0
        while k < min(len(nums), len(want)) and nums[k] == want[k]:
            k += 1
        res["correct_prefix"] = k / len(want)
    if task == "loop_sim":
        got = re.findall(r"step\s+(\d+)\s*:\s*x\s*=\s*(-?\d+)", out)
        want = re.findall(r"step (\d+): x = (\d+)", exp)
        res["steps"] = len(got)
        res["steps_ok"] = len(got) == m["n"]
        wd = dict(want)
        res["values_right"] = sum(1 for s, v in got if wd.get(s) == v) / m["n"]
    if task == "fewshot_format":
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        res["lines_right"] = sum(1 for w in m["queries"] if f"{w} -> «{w}»::{w}" in lines) / len(m["queries"])
    return task, m["cfg"], r["id"], res


by = defaultdict(dict)  # (task, cfg, id) -> model -> res
for r in rows:
    task, cfg, iid, res = score(r)
    by[(task, cfg, iid)][r["model"]] = res

groups = defaultdict(list)
for (task, cfg, iid), d in by.items():
    if "base" in d and "edited" in d:
        groups[(task, cfg)].append(d)

print(f"{'task':16s} {'cfg':11s} {'n':>3s} | {'exact b/e':>9s} {'lost/gained':>11s} | {'err spans b/e':>13s} "
      f"{'err tokens b/e':>15s} | {'hit max b/e':>11s} | extra")
tot = defaultdict(int)
for (task, cfg) in sorted(groups):
    L = groups[(task, cfg)]
    eb = sum(d["base"]["exact"] for d in L); ee = sum(d["edited"]["exact"] for d in L)
    lost = sum(d["base"]["exact"] and not d["edited"]["exact"] for d in L)
    gained = sum(d["edited"]["exact"] and not d["base"]["exact"] for d in L)
    sb = sum(d["base"]["spans"] for d in L); se = sum(d["edited"]["spans"] for d in L)
    cb = sum(d["base"]["cost"] for d in L); ce = sum(d["edited"]["cost"] for d in L)
    hb = sum(d["base"]["hit_max"] for d in L); he = sum(d["edited"]["hit_max"] for d in L)
    extra = ""
    if task in ("repeat_word", "repeat_sentence"):
        extra = (f"count ok {sum(d['base']['count_ok'] for d in L)}/{sum(d['edited']['count_ok'] for d in L)}; counts b="
                 f"{[d['base']['count'] for d in L]} e={[d['edited']['count'] for d in L]}")
    if task == "count_list":
        extra = (f"correct prefix b={[round(d['base']['correct_prefix'], 2) for d in L]} "
                 f"e={[round(d['edited']['correct_prefix'], 2) for d in L]}")
    if task == "loop_sim":
        extra = (f"steps b={[d['base']['steps'] for d in L]} e={[d['edited']['steps'] for d in L]}; values right "
                 f"b={sum(d['base']['values_right'] for d in L) / len(L):.2f} e={sum(d['edited']['values_right'] for d in L) / len(L):.2f}")
    if task == "fewshot_format":
        extra = (f"lines right b={sum(d['base']['lines_right'] for d in L) / len(L):.2f} "
                 f"e={sum(d['edited']['lines_right'] for d in L) / len(L):.2f}")
    print(f"{task:16s} {cfg:11s} {len(L):>3d} | {eb:>4d}/{ee:<4d} {lost:>5d}/{gained:<5d} | {sb:>6d}/{se:<6d} "
          f"{cb:>7d}/{ce:<7d} | {hb:>5d}/{he:<5d} | {extra}")
    for k, v in (("n", len(L)), ("eb", eb), ("ee", ee), ("lost", lost), ("gained", gained), ("sb", sb), ("se", se),
                 ("cb", cb), ("ce", ce), ("hb", hb), ("he", he)):
        tot[k] += v
print(f"{'TOTAL':28s} {tot['n']:>3d} | {tot['eb']:>4d}/{tot['ee']:<4d} {tot['lost']:>5d}/{tot['gained']:<5d} | "
      f"{tot['sb']:>6d}/{tot['se']:<6d} {tot['cb']:>7d}/{tot['ce']:<7d} | {tot['hb']:>5d}/{tot['he']:<5d}")

if show:
    k = 0
    for (task, cfg, iid), d in sorted(by.items()):
        if k >= show or "edited" not in d or "base" not in d:
            continue
        if d["base"]["exact"] != d["edited"]["exact"] or d["base"]["spans"] != d["edited"]["spans"]:
            k += 1
            print(f"\n===== {iid} ({task} {cfg})  base spans {d['base']['spans']}  edited spans {d['edited']['spans']}")
            for name in ("base", "edited"):
                a, b = toks(d[name]["exp"]), toks(d[name]["out"])
                sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
                ds = [f"{op}: {' '.join(a[i1:i2])[:80]!r} -> {' '.join(b[j1:j2])[:80]!r}"
                      for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"][:6]
                print(f"  [{name}] " + (" | ".join(ds) if ds else "exact"))
