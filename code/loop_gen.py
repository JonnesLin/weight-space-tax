"""Test B: loop rate and accuracy vs edit dose, thinking ON, authors' sampling, GSM8K *train* problems.
Dose c: W = bf16(W_base + (c/6) * (W_edited - W_base)) on layers 15/19 down_proj.
Each row stops at EOS, at max-new-tokens ("cap"), or early when the last WINDOW tokens are an obvious loop
("loop": fewer than THRESH of their 16-grams are distinct). Same seeds for every dose -> paired comparison.
usage: python loop_gen.py OUT.jsonl [--doses 0,6,3,1,2,4.5] [--n 40] [--max-new-tokens 16384] [--batch-size 40]"""
import argparse, json, os, random, re, sys, time, urllib.request
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "code"))
import torch
from safetensors import safe_open
from transformers import LogitsProcessor, StoppingCriteria, StoppingCriteriaList
from wsp_common import MODEL_PATHS, chat_prompt, load_model, load_tokenizer, sampling_kwargs, stop_token_ids

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--doses", default="0,6,3,1,2,4.5")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--max-new-tokens", type=int, default=16384)
ap.add_argument("--batch-size", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--window", type=int, default=2048)
ap.add_argument("--thresh", type=float, default=0.2)
ap.add_argument("--resume", action="store_true", help="append to OUT, skipping batches already written there")
args = ap.parse_args()
import importlib.util
IMPL = "fla" if importlib.util.find_spec("fla") else "torch"
print(f"linear-attention implementation: {IMPL}", flush=True)
doses = [float(x) for x in args.doses.split(",")]

data = os.path.join(_REPO, "data/gsm8k_train.jsonl")
if not os.path.exists(data):
    os.makedirs(os.path.dirname(data), exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl", data)
probs = [json.loads(l) for l in open(data)]
pick = random.Random(12345).sample(range(len(probs)), args.n)
items = [{"pid": f"gsm8k_train_{i}", "question": probs[i]["question"],
          "gold": probs[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in pick]

tok = load_tokenizer("base")
tok.padding_side = "left"
model = load_model("base")
stops = stop_token_ids(tok, model)
pad = tok.pad_token_id if tok.pad_token_id is not None else stops[0]
LAYERS = [15, 19]
st = safe_open(os.path.join(MODEL_PATHS["edited"], [f for f in os.listdir(MODEL_PATHS["edited"]) if f.endswith(".safetensors")][0]), "pt")
base_w, delta = {}, {}
for L in LAYERS:
    p = model.model.layers[L].mlp.down_proj.weight
    base_w[L] = p.data.clone()
    delta[L] = st.get_tensor(f"model.language_model.layers.{L}.mlp.down_proj.weight").to(p.device).float() - base_w[L].float()


def set_dose(c):
    for L in LAYERS:
        model.model.layers[L].mlp.down_proj.weight.data.copy_((base_w[L].float() + (c / 6.0) * delta[L]).to(torch.bfloat16))


class LoopStop(StoppingCriteria):
    def __init__(self, prompt_len, batch, every=128, n=16):
        self.prompt_len, self.every, self.n = prompt_len, every, n
        self.looped = torch.zeros(batch, dtype=torch.bool)
        self.at = [None] * batch
        self.frac = [None] * batch
        self.t0 = time.time()

    def __call__(self, input_ids, scores, **kw):
        g = input_ids.shape[1] - self.prompt_len
        if g > 0 and g % 512 == 0:
            el = time.time() - self.t0
            print(f"    step {g}: {g / el:.1f} steps/s, {el / 60:.1f} min, looped so far {int(self.looped.sum())}",
                  flush=True)
        if g >= args.window and g % self.every == 0:
            tail = input_ids[:, -args.window:].tolist()
            for b, row in enumerate(tail):
                if self.looped[b] or any(t in stops for t in row[-self.every:]):
                    continue
                grams = {tuple(row[i:i + self.n]) for i in range(len(row) - self.n + 1)}
                frac = len(grams) / (len(row) - self.n + 1)
                if frac < args.thresh:
                    self.looped[b], self.at[b], self.frac[b] = True, g, round(frac, 4)
        return self.looped.to(input_ids.device).clone()


class NanWatch(LogitsProcessor):
    """records, per row, the first decoding step whose logits contain NaN (numerical-fault guard)"""

    def __init__(self, B):
        self.first, self.step = [None] * B, 0

    def __call__(self, input_ids, scores):
        for b in torch.isnan(scores).any(-1).nonzero().flatten().tolist():
            if self.first[b] is None:
                self.first[b] = self.step
        self.step += 1
        return scores


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    for k in range(j, len(s)):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[j:k]
    return None


def num(s):
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}|\\[a-z]+|[$,\s]|dollars?", "", s)
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else None


SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
prompts = [chat_prompt(tok, it["question"] + SUFFIX, thinking=True) for it in items]
think_end = tok.convert_tokens_to_ids("</think>")
t_all = time.time()
done = set()
if args.resume and os.path.exists(args.out):
    for l in open(args.out):
        r = json.loads(l)
        done.add((r["dose"], r["pid"]))
    print(f"resume: {len(done)} rows already in {args.out}", flush=True)
with open(args.out, "a" if args.resume else "w") as f:
    for c in doses:
        set_dose(c)
        for bi, s in enumerate(range(0, len(items), args.batch_size)):
            idx = list(range(s, min(s + args.batch_size, len(items))))
            if all((c, items[i]["pid"]) in done for i in idx):
                print(f"dose {c:g} batch {bi}: already done, skipped", flush=True)
                continue
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            plen = enc.input_ids.shape[1]
            ls = LoopStop(plen, len(idx))
            torch.manual_seed(args.seed * 1000 + bi)
            t = time.time()
            nw = NanWatch(len(idx))
            kw = sampling_kwargs(False, plen)
            kw["logits_processor"].insert(0, nw)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens, eos_token_id=stops, pad_token_id=pad,
                                     stopping_criteria=StoppingCriteriaList([ls]), **kw)
            gen = out[:, plen:].tolist()
            counts = {"eos": 0, "loop": 0, "cap": 0}
            for j, i in enumerate(idx):
                ids = gen[j]
                e = next((k for k, x in enumerate(ids) if x in stops), None)
                if e is not None and (ls.at[j] is None or e < ls.at[j]):
                    ids, reason = ids[:e], "eos"
                elif ls.looped[j]:
                    ids, reason = ids[:ls.at[j]], "loop"
                else:
                    reason = "cap"
                counts[reason] += 1
                text = tok.decode(ids, skip_special_tokens=False)
                closed = think_end in ids
                ans = text.split("</think>")[-1] if closed else ""
                pred = num(last_boxed(ans))
                gold = float(items[i]["gold"])
                f.write(json.dumps({"dose": c, "pid": items[i]["pid"], "reason": reason, "n_tokens": len(ids),
                                    "think_closed": closed, "pred": pred, "gold": gold,
                                    "correct": pred is not None and abs(pred - gold) < 1e-6,
                                    "loop_at": ls.at[j], "loop_frac": ls.frac[j], "nan_step": nw.first[j],
                                    "impl": IMPL, "batch_size": args.batch_size,
                                    "text": text}) + "\n")
            f.flush()
            print(f"dose {c:g} batch {bi}: {len(idx)} rows, {len(gen[0])} steps, {counts}, "
                  f"NaN rows {sum(x is not None for x in nw.first)}, {time.time() - t:.0f}s", flush=True)
print(f"total {time.time() - t_all:.0f}s", flush=True)
