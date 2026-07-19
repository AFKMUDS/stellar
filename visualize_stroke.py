import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from raplib.model import ModelConfig, PrimitiveManifoldLM


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    comp = vt[:2].T
    y = x @ comp
    return y.astype(np.float32)


def _run_trace(model: PrimitiveManifoldLM, ids: list[int], device: str):
    model.eval()
    st = model.init_state(batch_size=1, device=torch.device(device))

    states = []
    prim_ids = []

    with torch.no_grad():
        for tok in ids:
            x = torch.tensor([tok], device=device, dtype=torch.long)
            # Use internal _get_emb to handle both embedding and bit-linear modes
            x_emb = model._get_emb(x)

            ctrl_in = torch.cat([x_emb, st], dim=-1)
            w = torch.softmax(model.controller(ctrl_in), dim=-1)

            sA = torch.einsum("b d, k d e -> b k e", st, model.prim_A)
            xB = torch.einsum("b m, k d m -> b k d", x_emb, model.prim_B)
            cand = torch.tanh(sA + xB + model.prim_bias.unsqueeze(0))
            mixed = torch.einsum("b k, b k d -> b d", w, cand)

            gate = model.slice_gate(ctrl_in)
            st = gate * mixed + (1.0 - gate) * st

            states.append(st[0].detach().cpu().float().numpy())
            prim_ids.append(int(torch.argmax(w, dim=-1).item()))

    return np.stack(states, axis=0), np.array(prim_ids, dtype=np.int32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=str, default="viz/stroke.png")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--color_by_primitive", action="store_true")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["cfg"]) if isinstance(ckpt, dict) and "cfg" in ckpt else ModelConfig()

    model = PrimitiveManifoldLM(cfg)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(args.device)

    text = "Question:\n" + args.prompt + "\n\nAnswer:\n"
    ids = list(text.encode("utf-8", errors="replace"))[: args.max_len]

    states, prim_ids = _run_trace(model, ids, device=args.device)
    xy = _pca_2d(states)

    plt.figure(figsize=(7, 7), dpi=180)

    if args.color_by_primitive:
        n = xy.shape[0]
        cmap = plt.get_cmap("tab10")
        for i in range(n - 1):
            c = cmap(int(prim_ids[i]) % 10)
            plt.plot([xy[i, 0], xy[i + 1, 0]], [xy[i, 1], xy[i + 1, 1]], color=c, linewidth=1.0)
    else:
        plt.plot(xy[:, 0], xy[:, 1], color="black", linewidth=1.0)

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(args.out)
    plt.close()

    print(f"Wrote stroke visualization to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
