import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
from raplib.model import ModelConfig, PrimitiveManifoldLM

def _pca_3d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    comp = vt[:3].T
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
    p.add_argument("--out", type=str, default="viz/stroke_3d.png") # Or .gif? Static 3D plot for now
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--color_by_primitive", action="store_true")
    p.add_argument("--interactive", action="store_true", help="Show plot instead of saving")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    
    # Handle both full checkpoint dict and state_dict only
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        cfg_dict = ckpt["cfg"]
        # Filter for forward compatibility
        valid_keys = ModelConfig.__annotations__.keys()
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = ModelConfig(**filtered_cfg)
        state_dict = ckpt["state_dict"]
    else:
        # Fallback or raw state dict
        print("Warning: No config found, using defaults.")
        cfg = ModelConfig()
        state_dict = ckpt

    model = PrimitiveManifoldLM(cfg)
    model.load_state_dict(state_dict)
    model.to(args.device)

    text = "Question:\n" + args.prompt + "\n\nAnswer:\n"
    ids = list(text.encode("utf-8", errors="replace"))[: args.max_len]

    print(f"Tracing trajectory for {len(ids)} bytes...")
    states, prim_ids = _run_trace(model, ids, device=args.device)
    xyz = _pca_3d(states)

    fig = plt.figure(figsize=(10, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    if args.color_by_primitive:
        n = xyz.shape[0]
        cmap = plt.get_cmap("tab10")
        # Plot segments
        for i in range(n - 1):
            c = cmap(int(prim_ids[i]) % 10)
            ax.plot(
                [xyz[i, 0], xyz[i + 1, 0]],
                [xyz[i, 1], xyz[i + 1, 1]],
                [xyz[i, 2], xyz[i + 1, 2]],
                color=c, linewidth=1.5
            )
    else:
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="black", linewidth=1.0, alpha=0.7)
        
    # Mark start and end
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], c='green', s=100, label='Start')
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], c='red', s=100, label='End')
    
    ax.set_title(f"3D Manifold Trajectory: '{args.prompt}'")
    ax.set_axis_off()
    plt.legend()
    plt.tight_layout()

    if args.interactive:
        plt.show()
    else:
        plt.savefig(args.out)
        print(f"Wrote 3D stroke visualization to: {os.path.abspath(args.out)}")
    
    plt.close()

if __name__ == "__main__":
    main()
