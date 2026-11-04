"""Score the dose sweep. usage: python score_dose.py dose_gen.jsonl [--ref copygen_greedy.jsonl] [--show N]
Per task/cfg and dose: items exactly right, error spans (word-level diff), lost/gained vs c=0.
--ref: for the original test-A items, check c0 == base and c6 == edited outputs (sanity)."""
import difflib, json, re, sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]
ref = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else None
show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 0


def norm(s):
    lines = s.strip().split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(l.rstrip() for l in lines if l.strip() not in ("TEXT START", "TEXT END")).strip()


def toks(s):
    return re.findall(r"[^\s]+|\n", s)


def sign_p(lost, gained):
    """two-sided exact sign test (McNemar) on discordant pairs"""
    from math import comb
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def spans(exp, out):
    sm = difflib.SequenceMatcher(None, toks(exp), toks(out), autojunk=False)
    return [o for o in sm.get_opcodes() if o[0] != "equal"]


res = defaultdict(dict)  # id -> dose -> dict
meta = {}
for r in rows:
    m = r["meta"]; meta[r["id"]] = m
    exp = "\n".join(l.rstrip() for l in m["expected"].split("\n")).strip()
    out = norm(r["answer_text"])
    d = {"exact": out == exp, "spans": len(spans(exp, out)), "hit": r["hit_max"], "out": out, "exp": exp}
    if m["task"] == "repeat_sentence":
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        d["count"] = sum(1 for l in lines if l == m["sentence"])
    res[r["id"]][r["dose"]] = d
doses = sorted({r["dose"] for r in rows})

groups = defaultdict(list)
for iid, d in res.items():
    if all(c in d for c in doses):
        m = meta[iid]
        groups[(m["task"], m["cfg"] if m["task"] != "template_code" or not m.get("orig") else "pytest(orig)")].append(iid)
        groups[(m["task"], "ALL")].append(iid)

print("exact-match items per dose (error spans in brackets); lost/gained = paired vs c=0")
print(f"{'task':16s} {'cfg':13s} {'n':>3s} | " + " | ".join(f"{'c=' + format(c, 'g'):>24s}" for c in doses))
for key in sorted(groups):
    ids = groups[key]
    cells = []
    for c in doses:
        ex = sum(res[i][c]["exact"] for i in ids); sp = sum(res[i][c]["spans"] for i in ids)
        if c == doses[0]:
            cells.append(f"{ex:>3d} [{sp:>4d}]       ")
        else:
            lost = sum(res[i][doses[0]]["exact"] and not res[i][c]["exact"] for i in ids)
            gain = sum(res[i][c]["exact"] and not res[i][doses[0]]["exact"] for i in ids)
            p = sign_p(lost, gain)
            cells.append(f"{ex:>3d} [{sp:>4d}] -{lost}/+{gain}" + (f" p={p:.3g}" if key[1] == "ALL" else ""))
    print(f"{key[0]:16s} {key[1]:13s} {len(ids):>3d} | " + " | ".join(f"{s:>24s}" for s in cells))
    if key[0] == "repeat_sentence" and key[1] == "ALL":
        for c in doses:
            cnt = [res[i][c]["count"] for i in ids]
            print(f"{'':34s} c={c:g}: count==20 in {sum(x == 20 for x in cnt)}/{len(ids)}, mean count "
                  f"{sum(cnt) / len(cnt):.1f}, wrote it <=2 times in {sum(x <= 2 for x in cnt)}")
print("hit cap per dose:", {f"c{c:g}": sum(res[i][c]["hit"] for i in res if c in res[i]) for c in doses})

if ref:
    refo = defaultdict(dict)
    for l in open(ref):
        r = json.loads(l)
        refo[r["id"]][r["model"]] = norm(r.get("answer_text") or r.get("output_text") or "")
    for c, name in ((0.0, "base"), (6.0, "edited")):
        both = [i for i in res if i in refo and c in res[i] and name in refo[i]]
        same = sum(res[i][c]["out"] == refo[i][name] for i in both)
        print(f"sanity: c={c:g} output identical to test-A {name} on {same}/{len(both)} original items")

k = 0
for iid in sorted(res):
    if k >= show:
        break
    d = res[iid]
    if d.get(doses[0], {}).get("exact") and any(not d[c]["exact"] for c in doses if c in d):
        k += 1
        print(f"\n===== {iid} ({meta[iid]['task']} {meta[iid]['cfg']})")
        for c in doses:
            ops = spans(d[c]["exp"], d[c]["out"])
            a, b = toks(d[c]["exp"]), toks(d[c]["out"])
            txt = " | ".join(f"{' '.join(a[i1:i2])[:50]!r}->{' '.join(b[j1:j2])[:50]!r}" for _, i1, i2, j1, j2 in ops[:4])
            print(f"  c={c:g}: {'exact' if not ops else txt}")
