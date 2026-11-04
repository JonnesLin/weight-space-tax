"""Copy-tax test: teacher-forced NLL of base vs edited (and dose variants) on copyable vs novel tokens.

Variants: W_c = W_base + (c/6) * (W_edited - W_base) on the two edited down_proj tensors
(c=0 base, c=6 the published edit; c=1 is the 'remove the u component' the blog describes).
Data (no sampling, nothing the edit was tuned on):
  A. random-token sequences repeated twice (pure induction/copy)
  B. Python standard-library source files (/usr/lib/python3.14)
  C. English prose: license texts in /usr/share/common-licenses
Token classes for B/C: copy4 = the 4-gram ending at this token occurred earlier in the context; novel = its bigram
(prev, tok) never occurred earlier. Output: results/copytest.json + printed table.
usage: python copytest.py [--doses 0,1,2,3,6]
"""
import os, sys, glob, json, random, argparse
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "code"))
import torch
from safetensors import safe_open
from wsp_common import load_model, load_tokenizer, MODEL_PATHS

ap = argparse.ArgumentParser()
ap.add_argument("--doses", default="0,1,2,3,6")
ap.add_argument("--out", default=os.path.expanduser("results/copytest.json"))
args = ap.parse_args()
doses = [float(x) for x in args.doses.split(",")]

tok = load_tokenizer("base")
model = load_model("base")
LAYERS = [15, 19]
base_w, delta = {}, {}
st = safe_open(os.path.join(MODEL_PATHS["edited"], [f for f in os.listdir(MODEL_PATHS["edited"]) if f.endswith(".safetensors")][0]), "pt")
for L in LAYERS:
    p = model.model.layers[L].mlp.down_proj.weight
    base_w[L] = p.data.clone()
    ed = st.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight").to(p.device)
    delta[L] = ed.float() - base_w[L].float()
    print(f"layer {L}: entries changed {(delta[L] != 0).sum().item():,}")

def set_dose(c):
    for L in LAYERS:
        model.model.layers[L].mlp.down_proj.weight.data.copy_((base_w[L].float() + (c / 6.0) * delta[L]).to(torch.bfloat16))

# ---------------- data
rng = random.Random(0)
seqs = []  # (group, token ids)
specials = set(tok.all_special_ids)
pool = [i for i in range(1000, 150000) if i not in specials]
for k in range(64):
    half = [rng.choice(pool) for _ in range(256)]
    seqs.append(("A_random_repeat", half + half))
py = sorted(f for f in glob.glob(os.path.join(os.environ.get("STDLIB_DIR", "/usr/lib/python3.14"), "*.py")) if os.path.getsize(f) > 20000)
py = py[:: max(1, len(py) // 48)][:48]
for f in py:
    ids = tok(open(f, encoding="utf-8", errors="replace").read()).input_ids[:2048]
    if len(ids) >= 512:
        seqs.append(("B_python", ids))
for f in sorted(glob.glob(os.path.join(os.environ.get("LICENSE_DIR", "/usr/share/common-licenses"), "*"))):
    if os.path.isfile(f) and os.path.getsize(f) > 8000:
        ids = tok(open(f, encoding="utf-8", errors="replace").read()).input_ids
        for s in range(0, min(len(ids), 8192) - 511, 2048):
            seqs.append(("C_prose", ids[s:s + 2048]))
print({g: sum(1 for x in seqs if x[0] == g) for g in sorted({x[0] for x in seqs})}, "sequences")

def token_classes(ids):
    """per target position i (predicting ids[i], i>=1): 'copy4', 'novel', or 'other'."""
    seen2, seen4, cls = set(), set(), []
    for i in range(1, len(ids)):
        bg = (ids[i - 1], ids[i])
        fg = tuple(ids[i - 3:i + 1]) if i >= 3 else None
        if fg is not None and fg in seen4:
            cls.append("copy4")
        elif bg not in seen2:
            cls.append("novel")
        else:
            cls.append("other")
        seen2.add(bg)
        if fg is not None:
            seen4.add(fg)
    return cls

meta = []
for g, ids in seqs:
    if g == "A_random_repeat":
        n = len(ids) // 2
        cls = ["first_half"] * (n - 1) + ["second_half"] * n
    else:
        cls = token_classes(ids)
    meta.append(cls)

@torch.no_grad()
def score_all():
    """returns list of (nll array, top1-correct array) per sequence"""
    res = []
    for g, ids in seqs:
        x = torch.tensor([ids], device="cuda")
        logits = model(x).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1)
        tgt = x[0, 1:]
        nll = -lp.gather(-1, tgt[:, None])[:, 0]
        top1 = (logits.argmax(-1) == tgt)
        res.append((nll.cpu(), top1.cpu()))
    return res

results = {}
for c in doses:
    set_dose(c)
    results[c] = score_all()
    print(f"dose {c}: done")
set_dose(0.0)

# ---------------- aggregate
summary = {}
groups = sorted({g for g, _ in seqs})
for g in groups:
    idx = [i for i, (gg, _) in enumerate(seqs) if gg == g]
    classes = sorted({c for i in idx for c in meta[i]})
    for cl in classes:
        key = f"{g}/{cl}"
        row = {}
        for c in doses:
            nll = torch.cat([results[c][i][0][torch.tensor([m == cl for m in meta[i]])] for i in idx])
            t1 = torch.cat([results[c][i][1][torch.tensor([m == cl for m in meta[i]])] for i in idx])
            row[c] = {"n": int(nll.numel()), "nll": float(nll.mean()), "top1": float(t1.float().mean())}
        # paired: top-1 flips vs base
        if 0.0 in doses:
            for c in doses:
                if c == 0.0:
                    continue
                lost = gained = 0
                for i in idx:
                    m = torch.tensor([mm == cl for mm in meta[i]])
                    b, e = results[0.0][i][1][m], results[c][i][1][m]
                    lost += int((b & ~e).sum()); gained += int((~b & e).sum())
                row[c]["top1_lost"] = lost; row[c]["top1_gained"] = gained
        summary[key] = row

print("\nmean NLL (nats/token) and top-1 accuracy; diff vs base in brackets")
hdr = f"{'group/class':28s} {'n':>7s} " + " ".join(f"{'c=' + format(c, 'g'):>22s}" for c in doses)
print(hdr)
for key, row in summary.items():
    b = row.get(0.0)
    cells = []
    for c in doses:
        r = row[c]
        d = f"{r['nll'] - b['nll']:+.3f}" if b and c != 0.0 else "     "
        fl = f" -{r['top1_lost']}/+{r['top1_gained']}" if "top1_lost" in r else ""
        cells.append(f"{r['nll']:.3f}[{d}] {100 * r['top1']:.1f}%")
    print(f"{key:28s} {row[doses[0]]['n']:>7d} " + " ".join(f"{s:>22s}" for s in cells))
    flips = "  ".join(f"c={c:g}: lost {row[c]['top1_lost']} gained {row[c]['top1_gained']}" for c in doses if "top1_lost" in row[c])
    if flips:
        print(f"{'':28s} top-1 flips vs base: {flips}")
json.dump({"doses": doses, "summary": {k: {str(c): v for c, v in r.items()} for k, r in summary.items()},
           "files": {"python": py}}, open(args.out, "w"), indent=1)
print("saved", args.out)
