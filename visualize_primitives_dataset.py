import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from raplib.data import iter_qa_samples
from raplib.model import ModelConfig, PrimitiveManifoldLM


def _step_and_get_prim(model: PrimitiveManifoldLM, tok: int, st: torch.Tensor):
    x = torch.tensor([tok], device=st.device, dtype=torch.long)
    # Use internal _get_emb to handle both embedding and bit-linear modes
    x_emb = model._get_emb(x)
    ctrl_in = torch.cat([x_emb, st], dim=-1)
    w = torch.softmax(model.controller(ctrl_in), dim=-1)

    sA = torch.einsum("b d, k d e -> b k e", st, model.prim_A)
    xB = torch.einsum("b m, k d m -> b k d", x_emb, model.prim_B)
    cand = torch.tanh(sA + xB + model.prim_bias.unsqueeze(0))
    mixed = torch.einsum("b k, b k d -> b d", w, cand)

    gate = model.slice_gate(ctrl_in)
    st2 = gate * mixed + (1.0 - gate) * st

    prim = int(torch.argmax(w, dim=-1).item())
    return prim, st2


def _save_heatmap(mat: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(6, 5), dpi=160)
    vmax = float(np.max(mat)) if mat.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    plt.imshow(mat, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, default="natural_reasoning_train.jsonl")
    p.add_argument("--out_dir", type=str, default="viz/prism")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--max_prompt_bytes", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["cfg"]) if isinstance(ckpt, dict) and "cfg" in ckpt else ModelConfig()

    model = PrimitiveManifoldLM(cfg)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(args.device)
    model.eval()

    K = cfg.n_primitives
    usage = np.zeros((K,), dtype=np.int64)
    trans = np.zeros((K, K), dtype=np.int64)

    total_steps = 0

    samples_it = iter_qa_samples(args.data, include_answer_prefix=True)

    with torch.no_grad():
        for i in range(args.num_samples):
            s = next(samples_it)
            prompt = s.prompt_bytes[: args.max_prompt_bytes]
            ids = list(prompt)
            if not ids:
                continue

            st = model.init_state(batch_size=1, device=torch.device(args.device))

            prev = None
            for tok in ids:
                prim, st = _step_and_get_prim(model, tok, st)
                usage[prim] += 1
                if prev is not None:
                    trans[prev, prim] += 1
                prev = prim
                total_steps += 1

    usage_norm = usage / max(int(usage.sum()), 1)
    trans_norm = trans.astype(np.float64)
    row_sums = trans_norm.sum(axis=1, keepdims=True)
    trans_norm = np.divide(trans_norm, np.maximum(row_sums, 1.0))

    plt.figure(figsize=(7, 3), dpi=160)
    plt.bar(np.arange(K), usage_norm)
    plt.title(f"Primitive usage (normalized) over {args.num_samples} prompts")
    plt.xlabel("primitive id")
    plt.ylabel("fraction")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "usage_hist.png"))
    plt.close()

    _save_heatmap(trans_norm, os.path.join(args.out_dir, "transition_heatmap.png"), "Primitive transition probabilities (row-normalized)")

    meta_path = os.path.join(args.out_dir, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"ckpt={args.ckpt}\n")
        f.write(f"data={args.data}\n")
        f.write(f"num_samples={args.num_samples}\n")
        f.write(f"max_prompt_bytes={args.max_prompt_bytes}\n")
        f.write(f"total_prompt_steps={total_steps}\n")
        f.write(f"n_primitives={K}\n")

    print(f"Wrote dataset primitive visuals to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
