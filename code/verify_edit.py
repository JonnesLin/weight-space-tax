# Usage: python code/verify_edit.py [--base DIR] [--edited DIR] [--out NPZ]
# Bitwise-diffs every tensor of base vs edited safetensors, checks published sha256s, reports which tensors/columns
# changed, checks each layer's delta is rank-1 (w <- w - c u u^T w), recovers u and the strength c per layer, and saves
# u, c and the edited column indices to an .npz. Output of the reference run: results/verify_edit.log
import argparse, glob, json, os, struct
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=os.path.join(_REPO, "models/base"))
ap.add_argument("--edited", default=os.path.join(_REPO, "models/edited"))
ap.add_argument("--out", default=os.path.expanduser("results/edit_direction.npz"))
args = ap.parse_args()


def st_file(d):
    fs = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    assert len(fs) == 1, fs
    return fs[0]


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), n


fa, fb = st_file(args.base), st_file(args.edited)
ha, na = st_header(fa)
hb, nb = st_header(fb)
print(f"header bytes: base={na} edited={nb}")
print("base __metadata__:", ha.get("__metadata__"))
print("edited __metadata__:", hb.get("__metadata__"))

A = safe_open(fa, framework="pt")
B = safe_open(fb, framework="pt")
ka, kb = set(A.keys()), set(B.keys())
print(f"tensors: base={len(ka)} edited={len(kb)} only_base={sorted(ka - kb)[:10]} only_edited={sorted(kb - ka)[:10]}")
prefixes = sorted({k.split(".layers.")[0] if ".layers." in k else ".".join(k.split(".")[:2]) for k in ka})
print("key prefixes:", prefixes)

total_params = 0
changed = {}
for k in sorted(ka & kb):
    a, b = A.get_tensor(k), B.get_tensor(k)
    total_params += a.numel()
    if a.dtype != b.dtype or a.shape != b.shape:
        print(f"!! {k}: dtype/shape differ {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}")
        changed[k] = None
        continue
    # bitwise comparison (robust to NaN / -0.0)
    ai = a.contiguous().view(torch.int16) if a.element_size() == 2 else a.contiguous().view(torch.uint8)
    bi = b.contiguous().view(torch.int16) if b.element_size() == 2 else b.contiguous().view(torch.uint8)
    neq = ai != bi
    if neq.any():
        changed[k] = (a, b, neq)
print(f"total params compared: {total_params:,}; dtypes: {sorted({str(A.get_slice(k).get_dtype()) for k in ka})}")
print(f"tensors that differ: {len(changed)}")

res = {}
for k, v in changed.items():
    if v is None:
        continue
    a, b, neq = v
    n = int(neq.sum())
    print(f"\n== {k}  shape={tuple(a.shape)} dtype={a.dtype}  differing entries: {n:,} / {a.numel():,}")
    if a.ndim != 2:
        continue
    col_counts = neq.sum(0)  # per input-feature column (W[:, j] = vector written to residual by neuron j)
    row_counts = neq.sum(1)
    cols = torch.nonzero(col_counts).flatten()
    rows = torch.nonzero(row_counts).flatten()
    print(f"   columns (dim1) with any diff: {len(cols)} / {a.shape[1]};  rows (dim0) with any diff: {len(rows)} / {a.shape[0]}")
    cc = col_counts[cols]
    print(f"   per-column differing entries: min={int(cc.min())} median={int(cc.median())} max={int(cc.max())} (column length {a.shape[0]})")
    print(f"   entries in those columns: {len(cols) * a.shape[0]:,}; differing: {n:,}")
    D = (b.float() - a.float())
    print(f"   |delta| max={D.abs().max():.3e}  ||delta||_F={D.norm():.4e}  ||W_cols||_F={a.float()[:, cols].norm():.4e}")
    res[k] = dict(a=a.float(), b=b.float(), cols=cols)

if not res:
    raise SystemExit("no 2D diffs")

# sha256 of raw bf16 bytes, as published in the edited repo's README / exp8_math/runs/p50c/edit/manifest.json
import hashlib
EXPECTED_SHA = {
    "model.language_model.layers.15.mlp.down_proj.weight": "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
    "model.language_model.layers.19.mlp.down_proj.weight": "8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
}
EXPECTED_BASE_SHA = {  # manifest.json "base_sha256"
    "model.language_model.layers.15.mlp.down_proj.weight": "ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca",
    "model.language_model.layers.19.mlp.down_proj.weight": "638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e",
}
print("\n== sha256 of raw bf16 bytes of changed tensors")
for k in changed:
    for tag, F, ref in (("base  ", A, EXPECTED_BASE_SHA), ("edited", B, EXPECTED_SHA)):
        h = hashlib.sha256(F.get_tensor(k).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        exp = ref.get(k)
        print(f"   {tag} {k}: {h}  {'OK (matches published hash)' if h == exp else ('MISMATCH, published ' + exp if exp else 'no published hash')}")
print(f"   changed set == published set: {set(changed) == set(EXPECTED_SHA)}")


def svd_report(name, D):
    s = torch.linalg.svdvals(D)
    print(f"   {name}: shape={tuple(D.shape)} top singular values: {[f'{x:.4e}' for x in s[:8].tolist()]}")
    print(f"      s2/s1={s[1] / s[0]:.3e}  energy in s1: {(s[0] ** 2 / (s ** 2).sum()):.6f}")
    return s


def fix_sign(u):
    return u * torch.sign(u[u.abs().argmax()])


print("\n== rank-1 check (deltas restricted to the edited columns)")
us, svals, colsd, coefs, names, strengths = {}, {}, {}, {}, {}, {}
deltas = []
for k, r in res.items():
    layer = int(k.split(".layers.")[1].split(".")[0])
    D = (r["b"] - r["a"])[:, r["cols"]]
    deltas.append(D)
    s = svd_report(f"layer {layer}", D)
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    u = fix_sign(U[:, 0])
    us[layer], svals[layer], colsd[layer], names[layer] = u, s, r["cols"], k

    # fit the edit strength c in  w <- w - c u (u^T w)   (c = 1 would be a plain projection-out)
    Wc, Wn = r["a"][:, r["cols"]], r["b"][:, r["cols"]]
    proj = u @ Wc
    c = float(-(u @ D) @ proj / (proj @ proj))
    ratio = -(u @ D) / proj
    print(f"      least-squares strength c in w <- w - c u(u^T w): c = {c:.5f}; per-column -delta_u/(u^T w): "
          f"min={ratio.min():.4f} median={ratio.median():.4f} max={ratio.max():.4f}")
    print(f"      u^T w after/before (median over cols) = {((u @ Wn) / proj).median():.4f}  (1 - c; 0 would mean 'removed')")
    for cc in (1.0, round(c), c):
        resid = D + cc * torch.outer(u, proj)
        sim = (Wc - cc * torch.outer(u, proj)).to(torch.bfloat16)
        exact = (sim.float() == Wn).float().mean()
        ulp = (sim.view(torch.int16).int() - Wn.to(torch.bfloat16).view(torch.int16).int()).abs()
        print(f"      model c={cc:.4f}: ||delta + c u u^T w||/||delta|| = {resid.norm() / D.norm():.3e}; "
              f"re-simulated bf16 == edited for {exact:.2%} of entries, within 1 ulp {(ulp <= 1).float().mean():.2%}")
    strengths[layer] = c
    before, after = proj.abs(), (u @ Wn).abs()
    print(f"      |u^T w| on edited cols: before mean={before.mean():.4e} max={before.max():.4e}; after mean={after.mean():.3e} max={after.max():.3e}")
    # were these the top-64 columns by |u^T w| (or signed)?
    proj_all = u @ r["a"]
    k64 = len(r["cols"])
    top_abs = set(torch.topk(proj_all.abs(), k64).indices.tolist())
    top_pos = set(torch.topk(proj_all, k64).indices.tolist())
    top_neg = set(torch.topk(-proj_all, k64).indices.tolist())
    ec = set(r["cols"].tolist())
    print(f"      edited cols vs top-{k64} by |u^T w|: overlap {len(ec & top_abs)}; by +u^T w: {len(ec & top_pos)}; by -u^T w: {len(ec & top_neg)}")
    rank = torch.argsort(proj_all.abs(), descending=True)
    pos = {c: i for i, c in enumerate(rank.tolist())}
    print(f"      rank of edited cols by |u^T w| (0=largest): max rank {max(pos[c] for c in ec)}")
    coefs[layer] = (u @ Wc)

layers = sorted(us)
if len(layers) == 2:
    l0, l1 = layers
    cos = float(us[l0] @ us[l1])
    print(f"\n== cos(u_{l0}, u_{l1}) = {cos:.6f}  (|cos|={abs(cos):.6f}) -> {'SAME direction (up to sign/rounding)' if abs(cos) > 0.999 else 'DIFFERENT directions'}")
    Dall = torch.cat(deltas, 1)
    s_all = svd_report("both layers stacked", Dall)
    U, S, Vh = torch.linalg.svd(Dall, full_matrices=False)
    u_joint = fix_sign(U[:, 0])
else:
    cos, s_all, u_joint = float("nan"), torch.zeros(1), us[layers[0]]

np.savez(args.out,
         layers=np.array(layers),
         tensor_names=np.array([names[l] for l in layers]),
         **{f"u_layer{l}": us[l].numpy() for l in layers},
         **{f"cols_layer{l}": colsd[l].numpy() for l in layers},
         **{f"svals_layer{l}": svals[l].numpy() for l in layers},
         **{f"coef_layer{l}": coefs[l].numpy() for l in layers},  # u^T w_j (base) for the edited columns, same order as cols
         **{f"strength_layer{l}": np.array(strengths[l]) for l in layers},
         u_joint=u_joint.numpy(), svals_joint=s_all.numpy(), cos_between_layers=np.array(cos),
         note=np.array("u: unit vector in residual space (dim0 of down_proj weight), sign arbitrary (fixed so largest |entry| > 0); "
                       "cols: indices into dim1 (intermediate neurons) whose output columns were edited; "
                       "edit is w <- w - c u (u^T w) with c = strength_layerL (fitted, ~6.0: the u-component is mapped to -5x, not removed); "
                       "u_joint: top left singular vector of both layers' deltas stacked (the two layers use different u's)."))
print(f"\nsaved {args.out}")
