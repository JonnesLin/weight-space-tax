"""Shared helpers for comparing base Qwen3.5-0.8B vs the weight-space-programming edit.

Not a script. Imported by compare_gen.py / compare_logprob.py.
Safety: never uses trust_remote_code; loads safetensors only; no hub kernels (the `kernels` package is not installed).
"""
import os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

MODELS_DIR = os.path.join(_REPO, "models")
MODEL_PATHS = {"base": os.path.join(MODELS_DIR, "base"), "edited": os.path.join(MODELS_DIR, "edited")}

# Authors' eval sampling (thinking ON): T=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_new_tokens=32768
AUTHOR_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)


def load_tokenizer(name="base"):
    tok = AutoTokenizer.from_pretrained(MODEL_PATHS[name], trust_remote_code=False)
    return tok


def load_model(name, device="cuda", dtype=torch.bfloat16):
    """Load the language model only (Qwen3_5ForCausalLM; vision tower + MTP head are dropped).
    Raises if any weight would be left randomly initialised."""
    path = MODEL_PATHS.get(name, name)
    model, info = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, trust_remote_code=False, use_safetensors=True, output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "lm_head" not in k]  # lm_head is tied to embed_tokens
    if missing or info.get("mismatched_keys"):
        raise RuntimeError(f"{name}: missing={missing[:10]} mismatched={info.get('mismatched_keys')}")
    if model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
        raise RuntimeError(f"{name}: lm_head is not tied to embed_tokens")
    model.to(device).eval()
    model.generation_config.pad_token_id = model.generation_config.pad_token_id or model.config.eos_token_id
    return model


def chat_prompt(tok, messages, thinking):
    """messages: str (single user turn) or list of {role, content}. Returns the prompt string ending with the
    assistant generation prompt ('<think>\\n' when thinking, '<think>\\n\\n</think>\\n\\n' when not)."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking))


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """OpenAI/vLLM-style presence penalty: subtract `penalty` from the logit of every token that already appears in
    the *generated* part (prompt tokens are not penalised, as in vLLM). Must run before temperature/top-k/top-p."""

    def __init__(self, penalty, prompt_len):
        self.penalty = float(penalty)
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len:]
        if self.penalty == 0 or gen.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, gen, True)
        return scores - self.penalty * seen.to(scores.dtype)


def sampling_kwargs(greedy, prompt_len, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5):
    """generate() kwargs. Sampled mode reproduces vLLM order: presence penalty -> temperature -> top-k -> top-p.
    HF's own warpers are disabled (temperature=1, top_k=0, top_p=1) and ours are passed explicitly."""
    if greedy:
        return dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    procs = []
    if presence_penalty:
        procs.append(PresencePenaltyLogitsProcessor(presence_penalty, prompt_len))
    procs.append(TemperatureLogitsWarper(temperature))
    if top_k:
        procs.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        procs.append(TopPLogitsWarper(top_p))
    return dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, logits_processor=LogitsProcessorList(procs))


def stop_token_ids(tok, model):
    ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    eos = model.generation_config.eos_token_id
    ids |= set(eos if isinstance(eos, (list, tuple)) else [eos])
    return sorted(i for i in ids if i is not None)
