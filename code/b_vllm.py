"""Test B with vLLM: one dose per call. Thinking ON, the authors' sampling, GSM8K *train* problems (same selection as
loop_gen.py: the first 40 of any n >= 40 are the same problems). vLLM generates to EOS or max-tokens; afterwards the
loop rule of loop_gen.py is applied to the token ids (every 128 steps, last 2048 tokens, < 20% distinct 16-grams ->
'loop' at that step, as if generation had been stopped there), so rows are comparable with the HF run.
usage: python b_vllm.py MODEL_DIR DOSE OUT.jsonl [--n 40] [--max-tokens 16384] [--seed 0]"""
import argparse, json, os, random, re, time
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def loop_point(ids, W, thresh, n=16):
    for g in range(W, len(ids) + 1, 128):
        row = ids[g - W:g]
        frac = len({tuple(row[i:i + n]) for i in range(len(row) - n + 1)}) / (len(row) - n + 1)
        if frac < thresh:
            return g, round(frac, 4)
    return None, None


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("dose", type=float); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--thresh", type=float, default=0.2)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    probs = [json.loads(l) for l in open(os.path.join(_REPO, "data/gsm8k_train.jsonl"))]
    pick = random.Random(12345).sample(range(len(probs)), args.n)
    items = [{"pid": f"gsm8k_train_{i}", "question": probs[i]["question"],
              "gold": float(probs[i]["answer"].split("####")[-1].strip().replace(",", ""))} for i in pick]
    SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["question"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for it in items]
    think_end = tok.convert_tokens_to_ids("</think>")

    llm = LLM(model=args.model_dir, dtype="bfloat16", max_model_len=args.max_tokens + 1024,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=args.max_num_seqs, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0})
    params = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5,
                             max_tokens=args.max_tokens, seed=args.seed * 100000 + i) for i in range(len(items))]
    t = time.time()
    outs = llm.generate(prompts, params)
    dt = time.time() - t

    ntok = 0
    counts = {"eos": 0, "loop": 0, "cap": 0}
    with open(args.out, "a") as f:
        for it, o in zip(items, outs):
            c = o.outputs[0]
            ids = list(c.token_ids)
            ntok += len(ids)
            loop_at, frac = loop_point(ids, args.window, args.thresh)
            finished = c.finish_reason == "stop"
            reason = "loop" if loop_at is not None else ("eos" if finished else "cap")
            counts[reason] += 1
            text = c.text
            closed = think_end in ids
            ans = text.split("</think>")[-1] if closed else ""
            pred = num(last_boxed(ans))
            right = pred is not None and abs(pred - it["gold"]) < 1e-6
            f.write(json.dumps({"dose": args.dose, "pid": it["pid"], "reason": reason,
                                "n_tokens": loop_at if reason == "loop" else len(ids), "gen_tokens": len(ids),
                                "finish_reason": c.finish_reason, "think_closed": closed, "pred": pred,
                                "gold": it["gold"], "correct": right and reason == "eos",
                                "correct_if_not_stopped": right and finished, "loop_at": loop_at, "loop_frac": frac,
                                "impl": "vllm", "max_tokens": args.max_tokens, "text": text}) + "\n")
    print(f"dose {args.dose:g}: {len(items)} problems, {counts}, {ntok} tokens in {dt:.0f}s = {ntok / dt:.0f} tok/s",
          flush=True)


if __name__ == "__main__":
    main()
