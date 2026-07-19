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
    
    with torch.no_grad():
        for tok in ids:
            x = torch.tensor([tok], device=device, dtype=torch.long)
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

    return np.stack(states, axis=0)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=str, default="viz/jump_trace.png")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_len", type=int, default=100)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        cfg_dict = ckpt["cfg"]
        valid_keys = ModelConfig.__annotations__.keys()
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = ModelConfig(**filtered_cfg)
        state_dict = ckpt["state_dict"]
    else:
        print("Warning: No config found, using defaults.")
        cfg = ModelConfig()
        state_dict = ckpt

    model = PrimitiveManifoldLM(cfg)
    model.load_state_dict(state_dict)
    model.to(args.device)

    # Prepare input
    # We want to visualize the processing of the prompt
    input_bytes = args.prompt.encode("utf-8", errors="replace")
    ids = list(input_bytes)[: args.max_len]

    print(f"Tracing trajectory for {len(ids)} bytes...")
    states = _run_trace(model, ids, device=args.device)
    xyz = _pca_3d(states)

    fig = plt.figure(figsize=(12, 10), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # Calculate pseudo-depth for sizing
    # Assume default view (elev=30, azim=-60)
    # Camera vector roughly: x+, y-, z+
    # Let's just use the projection on the view vector (-1, 1, 1) normalized
    # Farther points (smaller projection) -> smaller size
    # This is a rough heuristic
    view_vec = np.array([-1, -1, 1]) 
    view_vec = view_vec / np.linalg.norm(view_vec)
    
    # Project points onto view vector to get "depth"
    # Closer points have higher value
    depths = xyz @ view_vec
    
    # Normalize depths to 0..1
    d_min, d_max = depths.min(), depths.max()
    if d_max - d_min > 1e-6:
        depths_norm = (depths - d_min) / (d_max - d_min)
    else:
        depths_norm = np.zeros_like(depths) + 0.5
        
    # Size mapping: 20 to 200
    sizes = 20 + 180 * depths_norm

    # Plot the path (thicker line)
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="black", linewidth=1.5, alpha=0.6, zorder=1)

    # Plot nodes
    # Color gradient from start to end
    n = len(ids)
    colors = plt.cm.plasma(np.linspace(0, 1, n))
    
    # Scatter with dynamic sizes and depth shading
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=sizes, depthshade=True, zorder=2)

    # Add labels
    for i in range(n):
        # Label
        byte_val = ids[i]
        try:
            char = chr(byte_val)
            if byte_val == 32: # Space
                label = "_"
            elif not char.isprintable():
                label = f"x{byte_val:02x}"
            else:
                label = f"'{char}'"
        except:
            label = "?"
            
        # Offset label slightly
        ax.text(xyz[i, 0], xyz[i, 1], xyz[i, 2], label, fontsize=10, fontweight='bold', zorder=10)

    ax.set_title(f"Thought Jumps: '{args.prompt}'\n(Size/Color = Time/Depth)")
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Wrote jump trace to: {os.path.abspath(args.out)}")
    plt.close()

if __name__ == "__main__":
    main()
