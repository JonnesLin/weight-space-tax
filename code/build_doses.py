"""Write one full checkpoint folder per dose for vLLM: W_c = bf16(W + (c/6)(W_edited - W)) on the two edited tensors,
everything else byte-identical to the base release. Non-weight files are symlinked from the base folder.
usage: python build_doses.py OUT_DIR 0,1,2,3,4.5,6"""
import hashlib, os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors import safe_open
from safetensors.torch import save_file

out_root, doses = sys.argv[1], [float(x) for x in sys.argv[2].split(",")]
BASE = os.path.join(_REPO, "models/base")
EDIT = os.path.join(_REPO, "models/edited")
WF = "model.safetensors-00001-of-00001.safetensors"
NAMES = [f"model.language_model.layers.{L}.mlp.down_proj.weight" for L in (15, 19)]

base = {}
with safe_open(os.path.join(BASE, WF), "pt") as f:
    meta = f.metadata()
    for k in f.keys():
        base[k] = f.get_tensor(k)
with safe_open(os.path.join(EDIT, WF), "pt") as f:
    edited = {k: f.get_tensor(k) for k in NAMES}
delta = {k: edited[k].float() - base[k].float() for k in NAMES}
sha = lambda t: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]

for c in doses:
    d = os.path.join(out_root, f"c{c:g}")
    os.makedirs(d, exist_ok=True)
    for fn in os.listdir(BASE):
        if fn != WF and not fn.startswith(".") and not os.path.exists(os.path.join(d, fn)):
            os.symlink(os.path.join(BASE, fn), os.path.join(d, fn))
    sd = dict(base)
    for k in NAMES:
        sd[k] = (base[k].float() + (c / 6.0) * delta[k]).to(torch.bfloat16)
    save_file(sd, os.path.join(d, WF), metadata=meta or {"format": "pt"})
    checks = []
    for k in NAMES:
        checks.append("=base" if torch.equal(sd[k], base[k]) else ("=edited" if torch.equal(sd[k], edited[k]) else "new"))
    print(f"c={c:g}: {d}  layers 15/19: {checks}  sha {[sha(sd[k]) for k in NAMES]}", flush=True)
