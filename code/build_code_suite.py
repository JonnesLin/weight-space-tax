"""Confirmation suite (option 2): a fresh, larger set of code-type copy tasks, independent of the exploration sets.
usage: python build_code_suite.py OUT.jsonl
  copy_code      50  stdlib functions of 30-60 lines, identifiers renamed to random pseudo-words
  template_code  48  8 patterns x k in {10,15,20} x 2; constants must stay constant while i varies
  code_edit      51  insert a line / rename a variable / delete a line in a 20-45-line function; rest unchanged
  copy_json      48  pretty-printed JSON list; some fields hold the same value in every object
Every item has one exact expected output."""
import ast, builtins, glob, json, keyword, random, re, sys, textwrap

rng = random.Random(20261103)
SYL = ["ka", "ri", "mo", "vel", "tun", "sor", "an", "qui", "dre", "lo", "pha", "zen", "mir", "bo", "ste", "gal",
       "nu", "tra", "ix", "ol", "per", "wyn", "cas", "dun"]


def pw(k=3):
    return "".join(rng.choice(SYL) for _ in range(k))


funcs = []
for f in sorted(glob.glob("/usr/lib/python3.14/*.py")):
    try:
        src = open(f, encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(src, node, padded=True)
            if seg and seg.isascii() and "\t" not in seg:
                seg = textwrap.dedent(seg)
                if seg.startswith("def "):
                    funcs.append(seg)
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

    out = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sub, "\n".join(lines))
    ast.parse(out)
    return out


def take_funcs(lo, hi, n):
    got = []
    while len(got) < n and funcs:
        seg = funcs.pop()
        try:
            t = rename_code(seg)
        except SyntaxError:
            continue
        if lo <= t.count("\n") + 1 <= hi:
            got.append(t)
    return got


items = []


def emit(task, cfg, prompt, expected, **extra):
    items.append({"id": f"v_{task}_{len(items):03d}", "prompt": prompt, "task": task, "cfg": cfg,
                  "expected": expected, **extra})


COPY = ("Copy the following text exactly, character for character. Output only the copied text (without the TEXT START "
        "/ TEXT END lines) and nothing else.\n\nTEXT START\n{t}\nTEXT END")

# ---- copy_code
for t in take_funcs(30, 60, 50):
    emit("copy_code", "30-60lines", COPY.format(t=t), t)

# ---- code_edit
EDIT = ("{instr} Keep everything else exactly the same. Output only the full modified function and nothing else.\n\n"
        "```python\n{t}\n```")
pool = take_funcs(20, 45, 120)
kinds = ["insert", "rename", "delete"]
counts = {k: 0 for k in kinds}
for t in pool:
    kind = min(kinds, key=lambda k: counts[k])
    if counts[kind] >= 17:
        break
    lines = t.split("\n")
    fn = ast.parse(t).body[0]
    if kind == "insert":
        i0 = fn.body[0].lineno - 1
        indent = lines[i0][:len(lines[i0]) - len(lines[i0].lstrip())]
        new = f"{indent}log_call('{fn.name}')"
        exp = "\n".join(lines[:i0] + [new] + lines[i0:])
        instr = (f"Insert the line `log_call('{fn.name}')` as the first statement of the function body, with the same "
                 f"indentation as the body.")
    elif kind == "rename":
        names = [w for w in set(re.findall(r"\b[a-z]+_[a-z]+\b", t)) if w != fn.name
                 and len(re.findall(rf"\b{w}\b", t)) >= 3]
        if not names:
            continue
        old = rng.choice(sorted(names))
        new = pw(2) + "_" + pw(2)
        exp = re.sub(rf"\b{old}\b", new, t)
        instr = f"Rename `{old}` to `{new}` everywhere it appears."
    else:
        cands = [n for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AugAssign, ast.Expr)) and n.lineno == n.end_lineno
                 and n.lineno > fn.body[0].lineno - 1]
        cands = [n for n in cands if lines.count(lines[n.lineno - 1]) == 1 and len(lines[n.lineno - 1].strip()) > 10]
        if not cands:
            continue
        n = rng.choice(cands)
        line = lines[n.lineno - 1]
        exp = "\n".join(lines[:n.lineno - 1] + lines[n.lineno:])
        instr = f"Delete the line `{line.strip()}`."
    counts[kind] += 1
    emit("code_edit", kind, EDIT.format(instr=instr, t=t), exp)

# ---- template_code
PATTERNS = {
    "pytest": ("\n\n", lambda i, C: f"def test_{C['op']}_{i}():\n    assert {C['op']}({i}, {C['c']}) == {C['op']}_ref({i}, {C['c']})",
               lambda: {"op": rng.choice(["add", "scale", "merge", "clip", "blend"]), "c": rng.randint(2, 97)}),
    "config": ("\n", lambda i, C: f'CONFIG_{i} = {{"name": "{C["name"]}", "retries": {C["c"]}, "timeout": {C["t"]}}}',
               lambda: {"name": pw(3), "c": rng.randint(2, 9), "t": rng.choice([15, 30, 45, 60, 90])}),
    "sql": ("\n", lambda i, C: f"INSERT INTO {C['table']} (id, status, owner) VALUES ({i}, '{C['status']}', '{C['owner']}');",
            lambda: {"table": pw(2) + "_log", "status": rng.choice(["pending", "active", "archived"]), "owner": pw(3)}),
    "html": ("\n", lambda i, C: f'<li class="{C["cls"]}" data-id="{i}">{C["label"]}</li>',
             lambda: {"cls": pw(2) + "-item", "label": pw(3)}),
    "yaml": ("\n", lambda i, C: f"- name: {C['svc']}-{i}\n  image: {C['img']}:{C['ver']}\n  replicas: {C['c']}",
             lambda: {"svc": pw(2), "img": pw(3), "ver": f"{rng.randint(1, 9)}.{rng.randint(0, 20)}", "c": rng.randint(2, 9)}),
    "switch": ("\n", lambda i, C: f"    case {i}: return {C['fn']}({i}, {C['c']}, {C['flag']});",
               lambda: {"fn": pw(2) + "_" + pw(2), "c": rng.randint(2, 97), "flag": rng.choice(["MODE_FAST", "MODE_SAFE", "MODE_RAW"])}),
    "css": ("\n", lambda i, C: f".{C['cls']}-{i} {{ margin: {C['c']}px; color: {C['color']}; }}",
            lambda: {"cls": pw(2), "c": rng.randint(2, 40), "color": rng.choice(["#3a7bd5", "#e94e77", "#2f9e44", "#7048e8"])}),
    "js": ("\n", lambda i, C: f"const {C['var']}{i} = new {C['cls']}({{ id: {i}, mode: '{C['mode']}', retry: {C['c']} }});",
           lambda: {"var": pw(2), "cls": pw(2).capitalize() + "Client", "mode": rng.choice(["sync", "batch", "stream"]), "c": rng.randint(2, 9)}),
}
for name, (sep, f, mk) in PATTERNS.items():
    for k in (10, 15, 20):
        for _ in range(2):
            C = mk()
            emit("template_code", name,
                 f"Continue this pattern for i = 1 to {k}, including the two items shown. Output only the {k} items and "
                 f"nothing else.\n\n{f(1, C)}{sep}{f(2, C)}", sep.join(f(i, C) for i in range(1, k + 1)), k=k)

# ---- copy_json
for k in (6, 10, 14):
    for _ in range(16):
        status, region, retries, tag = rng.choice(["active", "paused", "draft"]), rng.choice(
            ["eu-west", "ap-south", "us-east", "sa-east"]), rng.randint(2, 9), pw(2)
        objs = [{"id": f"{rng.choice('ABKMRTX')}-{rng.randint(1000, 9999)}", "owner": pw(3), "status": status,
                 "region": region, "retries": retries, "tags": [tag, pw(2)]} for _ in range(k)]
        t = json.dumps(objs, indent=2)
        emit("copy_json", f"{k}objs", COPY.format(t=t), t)

with open(sys.argv[1], "w") as fo:
    for it in items:
        fo.write(json.dumps(it) + "\n")
from collections import Counter
print(len(items), "items", dict(Counter((it["task"], it["cfg"]) for it in items)), file=sys.stderr)
