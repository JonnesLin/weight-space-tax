"""Build the copy/repetition generation suite (test A). Every item has an exact expected output.
usage: python build_copygen.py OUT.jsonl
Sources are made novel where possible: shuffled docstring sentences, random records, stdlib functions with every
identifier renamed to random pseudo-words, random pseudo-words for few-shot."""
import ast, builtins, glob, json, keyword, random, re, sys, textwrap

rng = random.Random(1234)
out = open(sys.argv[1], "w")
n_items = 0


def emit(task, cfg, prompt, expected, **extra):
    global n_items
    n_items += 1
    out.write(json.dumps({"id": f"{task}_{n_items:03d}", "prompt": prompt, "task": task, "cfg": cfg,
                          "expected": expected, **extra}) + "\n")


SYL = ["ka", "ri", "mo", "vel", "tun", "sor", "an", "qui", "dre", "lo", "pha", "zen", "mir", "bo", "ste", "gal",
       "nu", "tra", "ix", "ol", "per", "wyn", "cas", "dun"]


def pseudo_word(k=3):
    return "".join(rng.choice(SYL) for _ in range(k))


# ---------- sources
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
                    w = s.split()
                    if 8 <= len(w) <= 30 and re.fullmatch(r"[A-Z][A-Za-z0-9 ,;:'()\-]*\.", s):
                        sentences.add(s)
        if isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(src, node, padded=True)
            if seg and seg.isascii() and "\t" not in seg:
                seg = textwrap.dedent(seg)
                if seg.startswith("def "):  # skip functions whose strings break the common indentation
                    funcs.append(seg)
sentences = sorted(sentences)
rng.shuffle(sentences)
print(f"{len(sentences)} sentences, {len(funcs)} functions", file=sys.stderr)
sent_iter = iter(sentences)


def passage(k):
    return " ".join(next(sent_iter) for _ in range(k))


KEEP = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls"}


def rename_code(seg):
    lines = seg.split("\n")
    tree = ast.parse("\n".join(lines))
    fn = tree.body[0]
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        d0, d1 = fn.body[0].lineno - 1, fn.body[0].end_lineno
        lines = lines[:d0] + lines[d1:]
    lines = [l for l in lines if not l.strip().startswith("#") and l.strip() != ""]
    code = "\n".join(l.rstrip() for l in lines)
    mapping = {}

    def sub(m):
        w = m.group(0)
        if w in KEEP:
            return w
        if w not in mapping:
            mapping[w] = pseudo_word(2) + "_" + pseudo_word(2)
        return mapping[w]

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", sub, code)


small = [f for f in funcs if 10 <= f.count("\n") + 1 <= 20]
large = [f for f in funcs if 30 <= f.count("\n") + 1 <= 60]
rng.shuffle(small); rng.shuffle(large)

COPY = ("Copy the following text exactly, character for character. Output only the copied text (without the TEXT START "
        "/ TEXT END lines) and nothing else.\n\nTEXT START\n{t}\nTEXT END")

# T1 prose copy
for k in (5, 12, 25):
    for _ in range(8):
        t = passage(k)
        emit("copy_prose", f"{k}sent", COPY.format(t=t), t)

# T2 records copy
FIRST = ["Mara", "Tomasz", "Aiko", "Bruno", "Leila", "Oskar", "Priya", "Jonas", "Ines", "Kwame", "Sofia", "Dmitri",
         "Hana", "Rafael", "Elif", "Nils"]
LAST = ["Lindqvist", "Okafor", "Nakamura", "Ferreira", "Haddad", "Novak", "Iyer", "Berger", "Costa", "Mensah",
        "Rossi", "Volkov", "Sato", "Mendes", "Yilmaz", "Dahl"]
CITY = ["Tallinn", "Porto", "Osaka", "Lagos", "Quito", "Gdansk", "Leeds", "Pune", "Bergen", "Accra", "Turin", "Graz"]


def record():
    return (f"{rng.choice('ABCDEFGHKMNPRTX')}-{rng.randint(1000, 9999)} | {rng.choice(FIRST)} {rng.choice(LAST)} | "
            f"{rng.choice(CITY)} | qty {rng.randint(1, 48)} | {rng.randint(1, 999)}.{rng.randint(0, 99):02d} | "
            f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}")


for n in (10, 25, 50):
    for _ in range(6):
        t = "\n".join(record() for _ in range(n))
        emit("copy_records", f"{n}lines", COPY.format(t=t), t)

# T3 renamed code copy
for name, pool in (("10-20lines", small), ("30-60lines", large)):
    for seg in pool[:8]:
        t = rename_code(seg)
        emit("copy_code", name, COPY.format(t=t), t)

# T4 copy with exactly one change
for _ in range(10):
    t = passage(10)
    words = [w for w in re.findall(r"\b[a-z]{5,}\b", t) if len(re.findall(rf"\b{w}\b", t)) == 1]
    old = rng.choice(words)
    new = pseudo_word(3)
    exp = re.sub(rf"\b{old}\b", new, t)
    emit("copy_one_change", "10sent",
         f"Copy the following text exactly, but replace the word \"{old}\" with \"{new}\" (it appears once). Change "
         f"nothing else. Output only the resulting text (without the TEXT START / TEXT END lines).\n\nTEXT START\n{t}\n"
         f"TEXT END", exp)

# T5 repeat a word N times
for n in (10, 30, 60, 100):
    for w in ("apple", "blue sky", "zorvane"):
        emit("repeat_word", f"n{n}",
             f"Write \"{w}\" exactly {n} times, separated by single spaces. Output nothing else.",
             " ".join([w] * n), n=n, word=w)

# T6 repeat a sentence N times
for n in (10, 20, 40):
    for _ in range(3):
        s = next(sent_iter)
        emit("repeat_sentence", f"n{n}",
             f"Repeat the following sentence exactly {n} times, one per line. Output nothing else.\n\n{s}",
             "\n".join([s] * n), n=n, sentence=s)

# T7 count list
for span in (50, 150, 300):
    for _ in range(3):
        a = rng.randint(1, 900)
        b = a + span - 1
        emit("count_list", f"{span}nums",
             f"Write all integers from {a} to {b} inclusive, in increasing order, separated by \", \". Output nothing else.",
             ", ".join(str(i) for i in range(a, b + 1)), a=a, b=b)

# T8 templated code
for k in (10, 25):
    for op in ("add", "scale", "merge"):
        c = rng.randint(2, 9)
        tmpl = "\n\n".join(f"def test_{op}_{i}():\n    assert {op}({i}, {c}) == {op}_ref({i}, {c})" for i in (1, 2))
        exp = "\n\n".join(f"def test_{op}_{i}():\n    assert {op}({i}, {c}) == {op}_ref({i}, {c})" for i in range(1, k + 1))
        emit("template_code", f"k{k}",
             f"Write {k} Python test functions that follow exactly this pattern, for i = 1 to {k}. Output only the code.\n\n"
             f"{tmpl}", exp, k=k)

# T9 few-shot format copying with novel words
for _ in range(6):
    ex = [pseudo_word(3) for _ in range(6)]
    qs = [pseudo_word(3) for _ in range(10)]
    shots = "\n".join(f"{w} -> «{w}»::{w}" for w in ex)
    emit("fewshot_format", "10q",
         f"Follow the pattern exactly.\n\n{shots}\n\nNow apply the same pattern to each of these words, one per line, "
         f"and output only those lines:\n" + "\n".join(qs), "\n".join(f"{w} -> «{w}»::{w}" for w in qs), queries=qs)

# T10 legitimate loop: simulate an iteration
for n in (10, 20, 40):
    for _ in range(3):
        x0, a, m = rng.randint(1, 9), rng.randint(2, 5), rng.choice([7, 11, 13, 17])
        x, vals = x0, []
        for step in range(1, n + 1):
            x = (x * a + step) % m
            vals.append(x)
        emit("loop_sim", f"n{n}",
             f"Let x = {x0}. For step = 1 to {n}: set x = (x * {a} + step) mod {m}. Write one line per step in the form "
             f"\"step k: x = value\", and output nothing else.",
             "\n".join(f"step {i}: x = {v}" for i, v in enumerate(vals, 1)), n=n)

print(f"{n_items} items", file=sys.stderr)
