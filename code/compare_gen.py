# Usage: python code/compare_gen.py PROMPTS.jsonl OUT.jsonl \
#            [--models base,edited] [--thinking on|off|both] [--decode sampled|greedy|both] [--n 1] \
#            [--max-new-tokens 4096] [--seed 0] [--batch-size 8] [--temperature 0.6 --top-p 0.95 --top-k 20 --presence-penalty 1.5]
# PROMPTS.jsonl: one object per line, {"id": "...", "prompt": "user text"} or {"id": ..., "messages": [{role, content}, ...]};
# any other fields are copied to the output under "meta". Models are loaded one at a time (about 2 GB GPU each).
# Sampled mode = authors' vLLM sampling (presence penalty on generated tokens -> T -> top-k -> top-p). The same seed
# schedule is used for both models, so the two models share random numbers and differences come from the weights.
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from wsp_common import AUTHOR_SAMPLING, chat_prompt, load_model, load_tokenizer, sampling_kwargs, stop_token_ids


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(s)):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[j:k]
    return None


def rep_ngram(ids, n=4):
    """Fraction of n-grams that are repeats of an earlier n-gram (0 = no repetition, ->1 = looping)."""
    if len(ids) < n + 1:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--models", default="base,edited")
    ap.add_argument("--thinking", choices=["on", "off", "both"], default="on")
    ap.add_argument("--decode", choices=["sampled", "greedy", "both"], default="sampled")
    ap.add_argument("--n", type=int, default=1, help="samples per prompt (sampled mode only; greedy always 1)")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=AUTHOR_SAMPLING["temperature"])
    ap.add_argument("--top-p", type=float, default=AUTHOR_SAMPLING["top_p"])
    ap.add_argument("--top-k", type=int, default=AUTHOR_SAMPLING["top_k"])
    ap.add_argument("--presence-penalty", type=float, default=AUTHOR_SAMPLING["presence_penalty"])
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.prompts) if l.strip()]
    for i, it in enumerate(items):
        it.setdefault("id", str(i))
    thinkings = {"on": [True], "off": [False], "both": [True, False]}[args.thinking]
    decodes = {"sampled": [False], "greedy": [True], "both": [False, True]}[args.decode]
    tok = load_tokenizer()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fout = open(args.out, "w")
    summary = []

    for mname in args.models.split(","):
        t0 = time.time()
        model = load_model(mname)
        stops = stop_token_ids(tok, model)
        print(f"[{mname}] loaded in {time.time() - t0:.1f}s; stop ids {stops}", flush=True)
        for thinking in thinkings:
            for greedy in decodes:
                n = 1 if greedy else args.n
                jobs = [(it, s) for it in items for s in range(n)]
                tot_tok, tot_time, nhit = 0, 0.0, 0
                for b0 in range(0, len(jobs), args.batch_size):
                    batch = jobs[b0:b0 + args.batch_size]
                    texts = [chat_prompt(tok, it.get("messages", it.get("prompt")), thinking) for it, _ in batch]
                    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                    plen = enc["input_ids"].shape[1]
                    seed = args.seed + b0  # same schedule for every model -> common random numbers
                    torch.manual_seed(seed)
                    kw = sampling_kwargs(greedy, plen, args.temperature, args.top_p, args.top_k, args.presence_penalty)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    with torch.inference_mode():
                        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, eos_token_id=stops,
                                             pad_token_id=tok.pad_token_id, **kw)
                    torch.cuda.synchronize()
                    dt = time.time() - t1
                    gen = out[:, plen:].tolist()
                    steps = len(gen[0])
                    btok = 0
                    for j, ((it, s), text, ids) in enumerate(zip(batch, texts, gen)):
                        stop_at = next((i for i, t in enumerate(ids) if t in stops), None)
                        hit_max = stop_at is None
                        ids = ids if hit_max else ids[:stop_at]
                        btok += len(ids) + (0 if hit_max else 1)
                        nhit += hit_max
                        otext = tok.decode(ids, skip_special_tokens=False)
                        rec = dict(model=mname, id=it["id"], thinking=thinking, decode="greedy" if greedy else "sampled",
                                   sample=s, seed=seed, batch_index=b0, prompt_text=text,
                                   prompt_tokens=int(enc["attention_mask"][j].sum()),
                                   output_text=otext, output_ids=ids, n_tokens=len(ids), hit_max=hit_max,
                                   finish="length" if hit_max else tok.convert_ids_to_tokens(gen[j][stop_at]),
                                   max_new_tokens=args.max_new_tokens, rep4=round(rep_ngram(ids), 4))
                        if thinking:
                            closed = "</think>" in otext
                            rec["think_closed"] = closed
                            rec["thinking_text"] = otext.split("</think>")[0] if closed else otext
                            rec["answer_text"] = otext.split("</think>", 1)[1].strip() if closed else ""
                            tid = tok.convert_tokens_to_ids("</think>")
                            rec["thinking_tokens"] = ids.index(tid) if tid in ids else len(ids)
                        else:
                            rec["answer_text"] = otext.strip()
                        rec["boxed"] = last_boxed(rec["answer_text"]) if rec["answer_text"] else None
                        if not greedy:
                            rec["sampling"] = dict(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                                   presence_penalty=args.presence_penalty)
                        rec["meta"] = {k: v for k, v in it.items() if k not in ("id", "prompt", "messages")}
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    tot_tok += btok
                    tot_time += dt
                    print(f"[{mname}] thinking={thinking} {'greedy' if greedy else 'sampled'} batch {b0 // args.batch_size}: "
                          f"{len(batch)} seqs, {steps} steps, {btok} tokens in {dt:.1f}s = {btok / dt:.1f} tok/s "
                          f"({steps / dt:.1f} steps/s)", flush=True)
                summary.append((mname, thinking, "greedy" if greedy else "sampled", len(jobs), nhit, tot_tok, tot_time))
        del model
        torch.cuda.empty_cache()

    fout.close()
    print("\nmodel   thinking decode   seqs hit_max tokens  tok/s")
    for m, th, d, nj, nh, tt, tm in summary:
        print(f"{m:7s} {str(th):8s} {d:8s} {nj:4d} {nh:7d} {tt:6d} {tt / max(tm, 1e-9):6.1f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
