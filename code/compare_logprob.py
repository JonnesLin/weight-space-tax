# Usage: python code/compare_logprob.py INPUT.jsonl OUT.jsonl [--append-eos] [--chunk 256] [--max-tokens N]
#        python code/compare_logprob.py --text "raw text to score" [--text ...] OUT.jsonl
# INPUT lines (any mix):
#   {"id":..., "text": "..."}                                          raw text; every token after the first is scored
#   {"id":..., "prompt": "..." | "messages": [...], "continuation": "...", "thinking": false}
#        chat format: prompt rendered with the chat template + generation prompt, only the continuation is scored.
#        With thinking=true the generation prompt ends in "<think>\n", so the continuation should contain the
#        reasoning, "</think>\n\n", then the answer. With thinking=false it ends in "<think>\n\n</think>\n\n".
#   a compare_gen.py output record: scores its exact output_ids after its prompt_text
#        (select records with e.g. --where thinking=True --where decode=greedy --where model=base).
# Teacher-forces each sequence through both models (both on GPU, about 4 GB) and saves per scored token the log-prob of
# the actual token under base and edited, KL(base||edited) over the full vocab, top-1 agreement, and an is_copy flag
# (the bigram (prev token, token) already occurred earlier in the context, i.e. an induction-style copy).
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from wsp_common import chat_prompt, load_model, load_tokenizer


def build(tok, it, append_eos, im_end):
    if "output_ids" in it and "prompt_text" in it:
        p = tok(it["prompt_text"], add_special_tokens=False)["input_ids"]
        c = list(it["output_ids"])
        if append_eos and not it.get("hit_max", False):
            c.append(im_end)
        return "gen", p, c
    if "continuation" in it:
        p = tok(chat_prompt(tok, it.get("messages", it.get("prompt")), it.get("thinking", False)),
                add_special_tokens=False)["input_ids"]
        c = tok(it["continuation"], add_special_tokens=False)["input_ids"]
        if append_eos:
            c.append(im_end)
        return "chat", p, c
    ids = tok(it["text"], add_special_tokens=False)["input_ids"]
    return "text", ids[:1], ids[1:]


@torch.inference_mode()
def score(models, ids, P, chunk):
    x = torch.tensor([ids], device="cuda")
    hs = {m: mod.model(input_ids=x).last_hidden_state[0] for m, mod in models.items()}
    L = x.shape[1]
    out = {k: [] for k in ("lp_base", "lp_edited", "kl", "top1_agree", "ent_base")}
    for s in range(P - 1, L - 1, chunk):
        e = min(s + chunk, L - 1)
        t = x[0, s + 1:e + 1, None]
        lb = models["base"].lm_head(hs["base"][s:e]).float().log_softmax(-1)
        le = models["edited"].lm_head(hs["edited"][s:e]).float().log_softmax(-1)
        pb = lb.exp()
        out["lp_base"].append(lb.gather(1, t)[:, 0])
        out["lp_edited"].append(le.gather(1, t)[:, 0])
        out["kl"].append((pb * (lb - le)).sum(-1).clamp_min(0))
        out["top1_agree"].append(lb.argmax(-1) == le.argmax(-1))
        out["ent_base"].append(-(pb * lb).sum(-1))
        del lb, le, pb
    return {k: torch.cat(v).cpu() for k, v in out.items()}


def copy_flags(ids, P):
    seen, flags = set(), []
    for i in range(1, len(ids)):
        bg = (ids[i - 1], ids[i])
        if i >= P:
            flags.append(bg in seen)
        seen.add(bg)
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="INPUT.jsonl OUT.jsonl, or just OUT.jsonl when --text is used")
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--append-eos", action="store_true", help="also score <|im_end|> after chat/gen continuations")
    ap.add_argument("--chunk", type=int, default=256, help="positions per lm_head chunk (memory: ~4 GB per 1000)")
    ap.add_argument("--max-tokens", type=int, default=0, help="truncate each sequence to this many tokens (0 = no)")
    ap.add_argument("--no-token-strings", action="store_true")
    ap.add_argument("--where", action="append", default=[],
                    help="keep only input lines whose field equals value, e.g. --where thinking=True --where decode=greedy")
    args = ap.parse_args()
    if args.text:
        assert len(args.paths) == 1, "with --text give only OUT.jsonl"
        items, outp = [{"id": f"text{i}", "text": t} for i, t in enumerate(args.text)], args.paths[0]
    else:
        assert len(args.paths) == 2, "give INPUT.jsonl OUT.jsonl"
        items = [json.loads(l) for l in open(args.paths[0]) if l.strip()]
        for w in args.where:
            k, v = w.split("=", 1)
            items = [it for it in items if str(it.get(k)) == v]
        print(f"{len(items)} items after filters {args.where}")
        outp = args.paths[1]

    tok = load_tokenizer()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    models = {"base": load_model("base"), "edited": load_model("edited")}

    # self-check: chunked hidden->lm_head path == model(...).logits on a short prefix
    with torch.inference_mode():
        chk = torch.tensor([tok("The quick brown fox jumps over the lazy dog. " * 4, add_special_tokens=False)["input_ids"]], device="cuda")
        full = models["base"](input_ids=chk).logits[0].float()
        mine = models["base"].lm_head(models["base"].model(input_ids=chk).last_hidden_state[0]).float()
        print(f"self-check max|logit diff| = {(full - mine).abs().max().item():.3e}")

    os.makedirs(os.path.dirname(os.path.abspath(outp)) or ".", exist_ok=True)
    fout = open(outp, "w")
    agg = {"n": 0, "nll_b": 0.0, "nll_e": 0.0, "kl": 0.0, "agree": 0, "copy_n": 0, "copy_d": 0.0, "copy_kl": 0.0,
           "nc_n": 0, "nc_d": 0.0, "nc_kl": 0.0}
    print(f"{'id':>14s} {'src':>6s} {'kind':>4s} {'n':>6s} {'NLL_base':>9s} {'NLL_edit':>9s} {'dNLL(e-b)':>10s} {'KL(b||e)':>9s} "
          f"{'top1=':>6s} {'dNLL copy':>10s} {'dNLL other':>10s}")
    for i, it in enumerate(items):
        kind, p, c = build(tok, it, args.append_eos, im_end)
        ids = p + c
        if args.max_tokens and len(ids) > args.max_tokens:
            ids = ids[:args.max_tokens]
        if len(ids) <= len(p):
            print(f"skip {it.get('id', i)}: nothing to score")
            continue
        r = score(models, ids, len(p), args.chunk)
        flags = torch.tensor(copy_flags(ids, len(p)))
        nll_b, nll_e = -r["lp_base"], -r["lp_edited"]
        d = nll_e - nll_b
        n = len(d)
        cm = flags
        rec = dict(id=it.get("id", str(i)), source_model=it.get("model"), thinking=it.get("thinking"), kind=kind,
                   n_prompt_tokens=len(p), n_scored=n,
                   mean_nll_base=nll_b.mean().item(), mean_nll_edited=nll_e.mean().item(), mean_dnll=d.mean().item(),
                   mean_kl=r["kl"].mean().item(), top1_agree=r["top1_agree"].float().mean().item(),
                   n_copy=int(cm.sum()), mean_dnll_copy=d[cm].mean().item() if cm.any() else None,
                   mean_dnll_noncopy=d[~cm].mean().item() if (~cm).any() else None,
                   mean_kl_copy=r["kl"][cm].mean().item() if cm.any() else None,
                   mean_kl_noncopy=r["kl"][~cm].mean().item() if (~cm).any() else None,
                   token_ids=ids[len(p):],
                   lp_base=[round(v, 5) for v in r["lp_base"].tolist()],
                   lp_edited=[round(v, 5) for v in r["lp_edited"].tolist()],
                   kl=[round(v, 6) for v in r["kl"].tolist()],
                   ent_base=[round(v, 4) for v in r["ent_base"].tolist()],
                   top1_agree_tok=r["top1_agree"].int().tolist(), is_copy=cm.int().tolist())
        if not args.no_token_strings:
            rec["tokens"] = [tok.decode([t]) for t in ids[len(p):]]
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        agg["n"] += n; agg["nll_b"] += nll_b.sum().item(); agg["nll_e"] += nll_e.sum().item()
        agg["kl"] += r["kl"].sum().item(); agg["agree"] += int(r["top1_agree"].sum())
        agg["copy_n"] += int(cm.sum()); agg["copy_d"] += d[cm].sum().item(); agg["copy_kl"] += r["kl"][cm].sum().item()
        agg["nc_n"] += int((~cm).sum()); agg["nc_d"] += d[~cm].sum().item(); agg["nc_kl"] += r["kl"][~cm].sum().item()
        fmt = lambda v: f"{v:10.4f}" if v is not None else f"{'-':>10s}"
        print(f"{str(rec['id'])[:14]:>14s} {str(rec['source_model'] or '-')[:6]:>6s} {kind:>4s} {n:6d} {rec['mean_nll_base']:9.4f} "
              f"{rec['mean_nll_edited']:9.4f} {rec['mean_dnll']:10.4f} {rec['mean_kl']:9.5f} {rec['top1_agree']:6.3f} "
              f"{fmt(rec['mean_dnll_copy'])} {fmt(rec['mean_dnll_noncopy'])}", flush=True)
    fout.close()
    N = max(agg["n"], 1)
    print(f"\nALL ({agg['n']} scored tokens; token-weighted): mean NLL base={agg['nll_b'] / N:.4f}  edited={agg['nll_e'] / N:.4f}  "
          f"diff(edited-base)={(agg['nll_e'] - agg['nll_b']) / N:+.4f}  mean KL(base||edited)={agg['kl'] / N:.5f}  "
          f"top1 agreement={agg['agree'] / N:.4f}")
    if agg["copy_n"]:
        print(f"  copy tokens    ({agg['copy_n']}): dNLL={agg['copy_d'] / agg['copy_n']:+.4f}  KL={agg['copy_kl'] / agg['copy_n']:.5f}")
    if agg["nc_n"]:
        print(f"  non-copy tokens({agg['nc_n']}): dNLL={agg['nc_d'] / agg['nc_n']:+.4f}  KL={agg['nc_kl'] / agg['nc_n']:.5f}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
