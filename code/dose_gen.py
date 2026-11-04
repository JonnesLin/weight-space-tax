"""Greedy generation (thinking off) for a list of edit doses, loading the model once.
Dose c: W = bf16(W_base + (c/6) * (W_edited - W_base)) on layers 15/19 down_proj (c=0 base, c=6 published edit).
Batches are sorted by expected length; each batch's max_new_tokens = max over its items of 1.3 x expected + 32.
usage: python dose_gen.py ITEMS.jsonl OUT.jsonl [--doses 0,1,2,3,6] [--batch-size 64]"""
import argparse, json, os, sys, time
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "code"))
import torch
from safetensors import safe_open
from wsp_common import MODEL_PATHS, chat_prompt, load_model, load_tokenizer, stop_token_ids

ap = argparse.ArgumentParser()
ap.add_argument("items"); ap.add_argument("out")
ap.add_argument("--doses", default="0,1,2,3,6")
ap.add_argument("--batch-size", type=int, default=64)
args = ap.parse_args()
doses = [float(x) for x in args.doses.split(",")]
items = [json.loads(l) for l in open(args.items)]

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


for it in items:
    it["_prompt"] = chat_prompt(tok, it["prompt"], thinking=False)
    it["_cap"] = int(1.3 * len(tok(it["expected"]).input_ids)) + 32
order = sorted(range(len(items)), key=lambda i: items[i]["_cap"])
t_all = time.time()
with open(args.out, "w") as f:
    for c in doses:
        set_dose(c)
        for b, s in enumerate(range(0, len(order), args.batch_size)):
            idx = order[s:s + args.batch_size]
            enc = tok([items[i]["_prompt"] for i in idx], return_tensors="pt", padding=True).to("cuda")
            cap = max(items[i]["_cap"] for i in idx)
            t = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=cap, do_sample=False, temperature=None, top_p=None,
                                     top_k=None, eos_token_id=stops, pad_token_id=pad)
            gen = out[:, enc.input_ids.shape[1]:].tolist()
            ntok = 0
            for j, i in enumerate(idx):
                ids = gen[j]
                n = next((k for k, x in enumerate(ids) if x in stops), None)
                ids = ids if n is None else ids[:n]
                ntok += len(ids)
                meta = {k: v for k, v in items[i].items() if not k.startswith("_") and k != "prompt"}
                f.write(json.dumps({"dose": c, "model": f"c{c:g}", "id": items[i]["id"],
                                    "answer_text": tok.decode(ids, skip_special_tokens=True), "n_tokens": len(ids),
                                    "hit_max": n is None, "cap": cap, "meta": meta}) + "\n")
            f.flush()
            dt = time.time() - t
            print(f"dose {c:g} batch {b}: {len(idx)} seqs, {gen and len(gen[0])} steps (cap {cap}), {ntok} tokens, "
                  f"{dt:.1f}s = {ntok / dt:.0f} tok/s", flush=True)
print(f"total {time.time() - t_all:.0f}s")
