import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch

from raplib.model import ModelConfig, PrimitiveManifoldLM


def _get_primitive_at_step(model: PrimitiveManifoldLM, tok: int, st: torch.Tensor):
    # Determine which primitive is dominant for this step
    x = torch.tensor([tok], device=st.device, dtype=torch.long)
    # Use internal _get_emb to handle both embedding and bit-linear modes
    x_emb = model._get_emb(x)
    ctrl_in = torch.cat([x_emb, st], dim=-1)
    w = torch.softmax(model.controller(ctrl_in), dim=-1)
    prim_idx = int(torch.argmax(w, dim=-1).item())
    
    # Also return the next state
    sA = torch.einsum("b d, k d e -> b k e", st, model.prim_A)
    xB = torch.einsum("b m, k d m -> b k d", x_emb, model.prim_B)
    cand = torch.tanh(sA + xB + model.prim_bias.unsqueeze(0))
    mixed = torch.einsum("b k, b k d -> b d", w, cand)
    gate = model.slice_gate(ctrl_in)
    st2 = gate * mixed + (1.0 - gate) * st
    
    return prim_idx, st2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=str, default="viz/flow_graph.png")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Load Model
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["cfg"]) if isinstance(ckpt, dict) and "cfg" in ckpt else ModelConfig()
    model = PrimitiveManifoldLM(cfg)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(args.device)
    model.eval()

    # Prepare Data
    # Just the prompt for this visualization to show the 'reading' flow
    text = args.prompt
    ids = list(text.encode("utf-8", errors="replace"))

    # Build Graph
    G = nx.DiGraph()
    pos = {}
    
    st = model.init_state(batch_size=1, device=torch.device(args.device))
    
    # Colors for primitives (0-7)
    cmap = plt.get_cmap("Set2")
    
    print(f"Tracing prompt: '{text}'")
    print(f"Bytes: {[bin(b) for b in ids]}")

    node_colors = []
    labels = {}

    with torch.no_grad():
        for i, tok in enumerate(ids):
            prim_idx, st = _get_primitive_at_step(model, tok, st)
            
            # Node ID
            node_id = i
            G.add_node(node_id)
            
            # Node Label: Char + Bits
            # Handle non-printable chars
            char_repr = chr(tok)
            if not char_repr.isprintable():
                char_repr = f"\\x{tok:02x}"
            if tok == 32: char_repr = "' '"
            
            bit_repr = f"{tok:08b}"
            labels[node_id] = f"{char_repr}\n{bit_repr}"
            
            # Layout: simple horizontal chain
            pos[node_id] = (i, 0)
            
            # Color by primitive
            node_colors.append(cmap(prim_idx % 8))
            
            if i > 0:
                G.add_edge(i-1, i)

    # Draw
    plt.figure(figsize=(max(10, len(ids) * 0.8), 4), dpi=100)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family="monospace")
    
    # Legend for primitives
    patches = [plt.Line2D([0], [0], marker='o', color='w', label=f'Prim {i}',
                          markerfacecolor=cmap(i % 8), markersize=10) for i in range(cfg.n_primitives)]
    plt.legend(handles=patches, loc='upper right', title="Active Primitive")

    plt.title(f"RAP v0 Byte-Level Flow: \"{text}\"\n(Nodes = Single Bytes. No Token Chunking.)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()

    print(f"Wrote flow graph to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
