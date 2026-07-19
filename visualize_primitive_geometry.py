import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def _load_prim_a(ckpt_path: str) -> tuple[torch.Tensor, int | None]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        prim_a = ckpt["state_dict"]["prim_A"].float()
        step = ckpt.get("step", None)
        return prim_a, step
    prim_a = ckpt["prim_A"].float()
    return prim_a, None


def _render_primitive_page(matrix: np.ndarray, prim_idx: int, ckpt_label: str, out_path: str) -> None:
    sym = 0.5 * (matrix + matrix.T)
    skew = 0.5 * (matrix - matrix.T)
    eigvals = np.linalg.eigvals(matrix)
    svals = np.linalg.svd(matrix, compute_uv=False)

    vmax_full = max(float(np.abs(matrix).max()), 1e-6)
    vmax_sym = max(float(np.abs(sym).max()), 1e-6)
    vmax_skew = max(float(np.abs(skew).max()), 1e-6)

    fig = plt.figure(figsize=(15, 10), dpi=180)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 0.9])

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(matrix, cmap="coolwarm", vmin=-vmax_full, vmax=vmax_full, aspect="auto")
    ax0.set_title(f"P{prim_idx} full A")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(sym, cmap="coolwarm", vmin=-vmax_sym, vmax=vmax_sym, aspect="auto")
    ax1.set_title(f"P{prim_idx} symmetric part")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(skew, cmap="coolwarm", vmin=-vmax_skew, vmax=vmax_skew, aspect="auto")
    ax2.set_title(f"P{prim_idx} skew part")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(np.real(eigvals), np.imag(eigvals), s=18, alpha=0.8, color="#7b2cbf")
    ax3.axhline(0.0, color="#888888", linewidth=0.8)
    ax3.axvline(0.0, color="#888888", linewidth=0.8)
    unit = plt.Circle((0.0, 0.0), 1.0, color="#aaaaaa", fill=False, linestyle="--", linewidth=1.0)
    ax3.add_patch(unit)
    ax3.set_aspect("equal", adjustable="box")
    ax3.set_title(f"P{prim_idx} eigenvalues")
    ax3.set_xlabel("real")
    ax3.set_ylabel("imag")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(np.arange(len(svals)), svals, color="#1d3557", linewidth=1.8)
    ax4.set_title(f"P{prim_idx} singular values")
    ax4.set_xlabel("index")
    ax4.set_ylabel("σ")

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    sym_norm = float(np.linalg.norm(sym))
    skew_norm = float(np.linalg.norm(skew))
    mag = np.abs(eigvals)
    complex_frac = float((np.abs(np.imag(eigvals)) > 1e-4).mean())
    stable_frac = float((mag < 1.0).mean())
    summary = "\n".join(
        [
            f"checkpoint: {ckpt_label}",
            f"primitive: P{prim_idx}",
            f"A norm: {np.linalg.norm(matrix):.4f}",
            f"sym norm: {sym_norm:.4f}",
            f"skew norm: {skew_norm:.4f}",
            f"skew/sym: {skew_norm / (sym_norm + 1e-8):.4f}",
            f"complex eig frac: {complex_frac:.4f}",
            f"|eig|<1 frac: {stable_frac:.4f}",
            f"top σ: {', '.join(f'{v:.3f}' for v in svals[:6])}",
        ]
    )
    ax5.text(0.02, 0.98, summary, ha="left", va="top", fontsize=11, family="monospace")

    fig.suptitle(f"Primitive geometry autopsy | {ckpt_label} | P{prim_idx}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints1/b.pt")
    p.add_argument("--out_dir", type=str, default="viz/primitive_geometry")
    p.add_argument("--primitives", type=int, nargs="+", default=[2, 6])
    args = p.parse_args()

    prim_a, step = _load_prim_a(args.ckpt)
    ckpt_label = os.path.basename(args.ckpt)
    if step is not None:
        ckpt_label = f"{ckpt_label} @ step {step}"

    os.makedirs(args.out_dir, exist_ok=True)
    for prim_idx in args.primitives:
        if int(prim_idx) < 0 or int(prim_idx) >= prim_a.shape[0]:
            continue
        matrix = prim_a[int(prim_idx)].cpu().numpy()
        out_path = os.path.join(args.out_dir, f"primitive_P{int(prim_idx)}_geometry.png")
        _render_primitive_page(matrix, int(prim_idx), ckpt_label, out_path)
        print(f"Wrote: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
