import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _save_heatmap(mat: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(6, 5), dpi=160)
    vmax = float(np.max(np.abs(mat))) if mat.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    plt.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _save_spectrum(svals: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(6, 4), dpi=160)
    plt.plot(svals)
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("singular value")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _try_svd(mat: np.ndarray, max_dim: int = 1024) -> Optional[np.ndarray]:
    if mat.ndim != 2:
        return None
    if mat.shape[0] > max_dim or mat.shape[1] > max_dim:
        mat = mat[:max_dim, :max_dim]
    try:
        s = np.linalg.svd(mat, compute_uv=False)
        return s
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="viz")
    p.add_argument("--max_prims", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    prim_A = sd.get("prim_A")
    prim_B = sd.get("prim_B")
    prim_bias = sd.get("prim_bias")
    tok_emb = sd.get("tok_emb.weight")
    bit_proj = sd.get("bit_proj.weight")
    out_w = sd.get("out.weight")

    meta_path = os.path.join(args.out_dir, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"ckpt={args.ckpt}\n")
        if isinstance(ckpt, dict) and "cfg" in ckpt:
            f.write(f"cfg={ckpt['cfg']}\n")
        for k in ["prim_A", "prim_B", "prim_bias", "tok_emb.weight", "bit_proj.weight", "out.weight"]:
            t = sd.get(k)
            if t is None:
                f.write(f"{k}=MISSING\n")
            else:
                f.write(f"{k}={tuple(t.shape)}\n")

    if tok_emb is not None:
        _save_heatmap(tok_emb.detach().cpu().float().numpy(), os.path.join(args.out_dir, "tok_emb.png"), "tok_emb.weight (256 x d_model)")

    if bit_proj is not None:
        bp = bit_proj.detach().cpu().float().numpy()
        _save_heatmap(bp, os.path.join(args.out_dir, "bit_proj.png"), f"bit_proj.weight {bp.shape}")

    if out_w is not None:
        _save_heatmap(out_w.detach().cpu().float().numpy(), os.path.join(args.out_dir, "out_weight.png"), "out.weight (256 x (d_model+d_state))")

    if prim_bias is not None:
        _save_heatmap(prim_bias.detach().cpu().float().numpy(), os.path.join(args.out_dir, "prim_bias.png"), "prim_bias (n_primitives x d_state)")

    if prim_A is not None:
        A = prim_A.detach().cpu().float().numpy()
        for i in range(min(A.shape[0], args.max_prims)):
            mat = A[i]
            _save_heatmap(mat, os.path.join(args.out_dir, f"prim_A_{i}.png"), f"prim_A[{i}] (d_state x d_state)")
            s = _try_svd(mat)
            if s is not None:
                _save_spectrum(s, os.path.join(args.out_dir, f"prim_A_{i}_sv.png"), f"SVD spectrum prim_A[{i}]")

    if prim_B is not None:
        B = prim_B.detach().cpu().float().numpy()
        for i in range(min(B.shape[0], args.max_prims)):
            mat = B[i]
            _save_heatmap(mat, os.path.join(args.out_dir, f"prim_B_{i}.png"), f"prim_B[{i}] (d_state x d_model)")
            s = _try_svd(mat)
            if s is not None:
                _save_spectrum(s, os.path.join(args.out_dir, f"prim_B_{i}_sv.png"), f"SVD spectrum prim_B[{i}]")

    print(f"Wrote visualizations to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
