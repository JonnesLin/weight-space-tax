"""Expanded suite for the dose sweep: the three task types where test A showed a tax.
usage: python build_dose_suite.py ORIG_SUITE.jsonl OUT.jsonl
  - copy_code: stdlib functions of 30-60 lines, every identifier renamed to random pseudo-words (30 new)
  - template_code: 4 patterns (pytest / config dict / SQL insert / HTML list) x k in {10, 20} x 5 (40 new)
  - repeat_sentence: one sentence 20 times (12 new)
plus the original test-A items of those types (copy_code 30-60lines, template_code, repeat_sentence n20), tagged orig."""
import ast, builtins, glob, json, keyword, random, re, sys, textwrap

orig_path, out_path = sys.argv[1], sys.argv[2]
rng = random.Random(777)
SYL = ["ka", "ri", "mo", "vel", "tun", "sor", "an", "qui", "dre", "lo", "pha", "zen", "mir", "bo", "ste", "gal",
       "nu", "tra", "ix", "ol", "per", "wyn", "cas", "dun"]


def pw(k=3):
    return "".join(rng.choice(SYL) for _ in range(k))


files = sorted(glob.glob("/usr/lib/python3.14/*.py"))
sentences, funcs = set(), []
for f in files:
    try:
        src = open(f, encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception:
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                for s in re.split(r"(?<=[.!?])\s+", " ".join(doc.split())):
                    if 8 <= len(s.split()) <= 30 and re.fullmatch(r"[A-Z][A-Za-z0-9 ,;:'()\-]*\.", s):
                        sentences.add(s)
        if isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(src, node, padded=True)
            if seg and seg.isascii() and "\t" not in seg:
                seg = textwrap.dedent(seg)
                if seg.startswith("def ") and 30 <= seg.count("\n") + 1 <= 60:
                    funcs.append(seg)
sentences = sorted(sentences); rng.shuffle(sentences)
rng.shuffle(funcs)
KEEP = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}


def rename_code(seg):
    lines = seg.split("\n")
    fn = ast.parse(seg).body[0]
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        lines = lines[:fn.body[0].lineno - 1] + lines[fn.body[0].end_lineno:]
    lines = [l.rstrip() for l in lines if not l.strip().startswith("#") and l.strip() != ""]
    mapping = {}

    def sub(m):
        w = m.group(0)
        if w in KEEP:
            return w
        if w not in mapping:
            mapping[w] = pw(2) + "_" + pw(2)
        return mapping[w]

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sub, "\n".join(lines))


COPY = ("Copy the following text exactly, character for character. Output only the copied text (without the TEXT START "
        "/ TEXT END lines) and nothing else.\n\nTEXT START\n{t}\nTEXT END")
items = []


def emit(task, cfg, prompt, expected, **extra):
    items.append({"id": f"d_{task}_{len(items):03d}", "prompt": prompt, "task": task, "cfg": cfg,
                  "expected": expected, **extra})


n_code = 0
for seg in funcs:
    if n_code == 30:
        break
    try:
        t = rename_code(seg)
    except SyntaxError:
        continue
    if 30 <= t.count("\n") + 1 <= 60:
        emit("copy_code", "30-60lines", COPY.format(t=t), t)
        n_code += 1

PATTERNS = {
    "pytest": ("\n\n", lambda i, C: f"def test_{C['op']}_{i}():\n    assert {C['op']}({i}, {C['c']}) == {C['op']}_ref({i}, {C['c']})",
               lambda: {"op": rng.choice(["add", "scale", "merge", "clip", "blend"]), "c": rng.randint(2, 97)}),
    "config": ("\n", lambda i, C: f'CONFIG_{i} = {{"name": "{C["name"]}", "retries": {C["c"]}, "timeout": {C["t"]}}}',
               lambda: {"name": pw(3), "c": rng.randint(2, 9), "t": rng.choice([15, 30, 45, 60, 90])}),
    "sql": ("\n", lambda i, C: f"INSERT INTO {C['table']} (id, status, owner) VALUES ({i}, '{C['status']}', '{C['owner']}');",
            lambda: {"table": pw(2) + "_log", "status": rng.choice(["pending", "active", "archived"]), "owner": pw(3)}),
    "html": ("\n", lambda i, C: f'<li class="{C["cls"]}" data-id="{i}">{C["label"]}</li>',
             lambda: {"cls": pw(2) + "-item", "label": pw(3)}),
}
for name, (sep, f, mk) in PATTERNS.items():
    for k in (10, 20):
        for _ in range(5):
            C = mk()
            emit("template_code", name,
                 f"Continue this pattern for i = 1 to {k}, including the two items shown. Output only the {k} items and "
                 f"nothing else.\n\n{f(1, C)}{sep}{f(2, C)}", sep.join(f(i, C) for i in range(1, k + 1)), k=k)

for _ in range(12):
    s = sentences.pop()
    emit("repeat_sentence", "n20",
         f"Repeat the following sentence exactly 20 times, one per line. Output nothing else.\n\n{s}",
         "\n".join([s] * 20), n=20, sentence=s)

for l in open(orig_path):
    r = json.loads(l)
    if (r["task"] == "copy_code" and r["cfg"] == "30-60lines") or r["task"] == "template_code" or \
            (r["task"] == "repeat_sentence" and r["cfg"] == "n20"):
        r["orig"] = True
        items.append(r)

with open(out_path, "w") as fo:
    for it in items:
        fo.write(json.dumps(it) + "\n")
from collections import Counter
print(len(items), "items", dict(Counter((it["task"], it.get("orig", False)) for it in items)), file=sys.stderr)
