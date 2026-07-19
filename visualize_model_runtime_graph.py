import argparse
import os
from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import torch

from raplib.model import ModelConfig, PrimitiveManifoldLM


def _load_model(ckpt_path: str, device: str) -> PrimitiveManifoldLM:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ModelConfig(**ckpt["cfg"]) if isinstance(ckpt, dict) and "cfg" in ckpt else ModelConfig()
    model = PrimitiveManifoldLM(cfg)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()
    return model


def _iter_leaf_named_modules(model: torch.nn.Module):
    for name, m in model.named_modules():
        if name == "":
            continue
        if len(list(m.children())) == 0:
            yield name, m


def _iter_all_named_modules(model: torch.nn.Module):
    for name, m in model.named_modules():
        if name == "":
            continue
        yield name, m


def _trace_one_step(
    model: PrimitiveManifoldLM,
    *,
    device: str,
    leaf_only: bool,
    tok: int,
) -> List[str]:
    call_order: List[str] = []

    def hook(name: str):
        def _fn(_m, _inp, _out):
            call_order.append(name)
        return _fn

    handles = []
    it = _iter_leaf_named_modules(model) if leaf_only else _iter_all_named_modules(model)
    for name, m in it:
        handles.append(m.register_forward_hook(hook(name)))

    try:
        with torch.no_grad():
            b = 1
            idx = torch.tensor([tok], device=device, dtype=torch.long)
            st = model.init_state(b, device=torch.device(device))
            # Call step with raw indices to include embedding path.
            _logits, _st2 = model.step(idx, st)
    finally:
        for h in handles:
            h.remove()

    # Remove immediate repeats (common with module reuse or wrapper modules)
    dedup: List[str] = []
    for n in call_order:
        if len(dedup) == 0 or dedup[-1] != n:
            dedup.append(n)
    return dedup


def _build_graph(call_order: List[str]) -> Tuple[nx.DiGraph, Counter]:
    G = nx.DiGraph()
    edge_counts: Counter = Counter()

    for n in call_order:
        G.add_node(n)

    for a, b in zip(call_order, call_order[1:]):
        edge_counts[(a, b)] += 1

    for (a, b), c in edge_counts.items():
        G.add_edge(a, b, weight=c)

    return G, edge_counts


def _draw_graph(G: nx.DiGraph, out_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Layered-ish layout: use topological order along x axis, spread by index.
    try:
        topo = list(nx.topological_sort(G))
    except Exception:
        topo = list(G.nodes())

    pos = {n: (i, 0) for i, n in enumerate(topo)}

    plt.figure(figsize=(max(12, len(topo) * 0.8), 5), dpi=140)

    nx.draw_networkx_nodes(G, pos, node_size=2200, node_color="#e8eef7", edgecolors="#2d3a4a", linewidths=1.0)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=18, width=2.0, edge_color="#2d3a4a")

    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family="monospace")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="viz/model_runtime_graph.png")
    p.add_argument("--order_out", type=str, default="viz/model_runtime_order.txt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tok", type=int, default=65, help="Single byte/token to run through one step (0-255).")
    p.add_argument("--leaf_only", action="store_true", help="Only record leaf nn.Modules to reduce noise")
    args = p.parse_args()

    model = _load_model(args.ckpt, args.device)
    call_order = _trace_one_step(model, device=args.device, leaf_only=bool(args.leaf_only), tok=int(args.tok) % 256)

    G, _edge_counts = _build_graph(call_order)

    title = f"PrimitiveManifoldLM runtime module order (one step)\nckpt={os.path.basename(args.ckpt)} tok={int(args.tok)%256} leaf_only={bool(args.leaf_only)}"
    _draw_graph(G, args.out, title)

    os.makedirs(os.path.dirname(args.order_out) or ".", exist_ok=True)
    with open(args.order_out, "w", encoding="utf-8") as f:
        for i, n in enumerate(call_order):
            f.write(f"{i:03d} {n}\n")

    print(f"Wrote: {os.path.abspath(args.out)}")
    print(f"Wrote: {os.path.abspath(args.order_out)}")


if __name__ == "__main__":
    main()
