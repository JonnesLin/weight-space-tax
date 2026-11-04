# Usage: HF_HUB_DISABLE_XET=1 python code/download_models.py
# Downloads base Qwen3.5-0.8B (pinned rev) and the weight-edited release into models/{base,edited},
# serially, allow-listed files only (no .py), then verifies sha256 of the safetensors against the Hub LFS hashes.
import hashlib, os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import HfApi, hf_hub_download

ROOT = os.path.join(_REPO, "models")
REPOS = {
    "base":   ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "edited": ("RikiyaT/Qwen3.5-0.8B-weight-space-programming", "d1671a178705bcab7644806d23c535517f35723b"),
}
COMMON = ["config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "model.safetensors.index.json", "LICENSE"]
EXTRA = {"edited": ["README.md"]}
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"

api = HfApi()

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()

for name, (repo, rev) in REPOS.items():
    info = api.model_info(repo, revision=rev, files_metadata=True)
    sib = {s.rfilename: s for s in info.siblings}
    bad = [f for f in sib if f.endswith(".py")]
    if bad:
        print(f"[{name}] NOTE: repo ships python files (not downloaded): {bad}")
    local = os.path.join(ROOT, name)
    os.makedirs(local, exist_ok=True)
    for fn in COMMON + EXTRA.get(name, []) + [WEIGHTS]:
        if fn not in sib:
            print(f"[{name}] missing on hub: {fn}"); continue
        for attempt in range(5):
            try:
                p = hf_hub_download(repo, fn, revision=rev, local_dir=local)
                break
            except Exception as e:
                print(f"[{name}] {fn} attempt {attempt} failed: {e!r}", flush=True)
        else:
            sys.exit(f"[{name}] giving up on {fn}")
        print(f"[{name}] {fn} -> {p} ({os.path.getsize(p)} bytes)", flush=True)
    want = sib[WEIGHTS].lfs.sha256
    got = sha256(os.path.join(local, WEIGHTS))
    print(f"[{name}] {WEIGHTS} sha256 {got} {'OK' if got == want else 'MISMATCH (hub: ' + want + ')'}", flush=True)
