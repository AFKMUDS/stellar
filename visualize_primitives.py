import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from probe_latent_step import build_retrieval, load_model
from visualize_byte_pages import _byte_label, _fit_pca_transform


def trace_with_primitives(model, ids: list[int], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    state = model.init_state(batch_size=1, device=device)
    states = []
    weights = []
    last_pos = None
    state_hist = None
    if bool(getattr(model, "retrieval_enabled", False)):
        last_pos = torch.full((1, model.cfg.vocab_size), -1, device=device, dtype=torch.long)
        state_hist = []
    with torch.no_grad():
        for step_idx, tok in enumerate(ids):
            x = torch.tensor([int(tok)], device=device, dtype=torch.long)
            x_emb = model._get_emb(x)
            retrieval = build_retrieval(model, int(tok), state, last_pos, state_hist)
            if retrieval is not None:
                ctrl_in = torch.cat([x_emb, state, retrieval], dim=-1)
            else:
                ctrl_in = torch.cat([x_emb, state], dim=-1)
            w = torch.softmax(model.controller(ctrl_in), dim=-1)
            _logits, state = model.step(x_emb, state, retrieval=retrieval)
            if last_pos is not None and state_hist is not None:
                state_hist.append(state.detach().clone())
                last_pos.scatter_(1, x.unsqueeze(1), torch.full((1, 1), step_idx, device=device, dtype=torch.long))
            states.append(state[0].detach().cpu().float().numpy())
            weights.append(w[0].detach().cpu().float().numpy())
    return np.stack(states, axis=0), np.stack(weights, axis=0)


def render_single_primitive(coords: np.ndarray, weights: np.ndarray, ids: list[int], primitive_idx: int, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    strength = weights[:, primitive_idx]
    sizes = 30.0 + 260.0 * strength
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=strength, s=sizes, cmap="magma", alpha=0.9)
    ax.plot(coords[:, 0], coords[:, 1], color="#777777", linewidth=0.8, alpha=0.45)
    for i, tok in enumerate(ids):
        ax.text(coords[i, 0], coords[i, 1], _byte_label(tok), fontsize=7, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("State PC1")
    ax.set_ylabel("State PC2")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def render_primitive_grid(coords: np.ndarray, weights: np.ndarray, ids: list[int], out_path: str, title: str) -> None:
    n_primitives = weights.shape[1]
    cols = 4
    rows = int(np.ceil(n_primitives / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(18, 8 * rows / 2), dpi=180)
    axes = np.asarray(axes).reshape(-1)
    vmax = max(float(weights.max()), 1e-6)
    for prim_idx, ax in enumerate(axes):
        if prim_idx >= n_primitives:
            ax.axis("off")
            continue
        strength = weights[:, prim_idx]
        sizes = 18.0 + 220.0 * strength
        ax.scatter(coords[:, 0], coords[:, 1], c=strength, s=sizes, cmap="viridis", vmin=0.0, vmax=vmax, alpha=0.9)
        ax.plot(coords[:, 0], coords[:, 1], color="#999999", linewidth=0.7, alpha=0.35)
        top_k = np.argsort(-strength)[: min(6, len(strength))]
        for idx in top_k.tolist():
            ax.text(coords[idx, 0], coords[idx, 1], f"{idx}:{_byte_label(ids[idx])}", fontsize=6.5, alpha=0.9)
        ax.set_title(f"P{prim_idx}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path)
    plt.close(fig)


def render_argmax_space(coords: np.ndarray, weights: np.ndarray, ids: list[int], out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    argmax = np.argmax(weights, axis=1)
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=argmax, s=44, cmap="tab10", alpha=0.92)
    ax.plot(coords[:, 0], coords[:, 1], color="#aaaaaa", linewidth=0.8, alpha=0.3)
    for i, tok in enumerate(ids):
        ax.text(coords[i, 0], coords[i, 1], f"{i}:{_byte_label(tok)}", fontsize=6.5, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("State PC1")
    ax.set_ylabel("State PC2")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(weights.shape[1]))
    cbar.set_ticklabels([f"P{i}" for i in range(weights.shape[1])])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints1/b.pt")
    p.add_argument("--out_dir", type=str, default="primitive_viz")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--primitive", type=int, default=-1)
    p.add_argument("--trace_response", action="store_true")
    p.add_argument("--max_new_bytes", type=int, default=48)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=32)
    return p.parse_args()


def sample_next_id(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(float(temperature), 1e-6)
    if int(top_k) > 0 and int(top_k) < logits.shape[-1]:
        vals, idx = torch.topk(logits, k=int(top_k), dim=-1)
        probs = torch.softmax(vals, dim=-1)
        pick = torch.multinomial(probs, num_samples=1)
        return int(idx.gather(-1, pick).item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_ids(model, prompt_ids: list[int], device: torch.device, max_new_bytes: int, temperature: float, top_k: int) -> list[int]:
    if int(max_new_bytes) <= 0:
        return []
    model.eval()
    ids = list(prompt_ids)
    with torch.no_grad():
        if ids:
            x = torch.tensor([ids], device=device, dtype=torch.long)
            out = model(x, state=None)
            state = out[1]
            last_id = int(ids[-1])
        else:
            state = model.init_state(batch_size=1, device=device)
            last_id = 10
        generated = []
        for _ in range(int(max_new_bytes)):
            x = torch.tensor([[last_id]], device=device, dtype=torch.long)
            out = model(x, state=state)
            logits, state = out[0], out[1]
            next_id = sample_next_id(logits[:, -1, :], temperature=float(temperature), top_k=int(top_k))
            generated.append(int(next_id))
            last_id = int(next_id)
    return generated


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(args.ckpt, args.device)
    ids = list(str(args.prompt).encode("utf-8", errors="replace"))
    if bool(args.trace_response):
        ids = ids + generate_ids(model, ids, device, int(args.max_new_bytes), float(args.temperature), int(args.top_k))
    states, weights = trace_with_primitives(model, ids, device)
    coords, _mean, _comp = _fit_pca_transform(states, n_components=2)

    seq_text = "prompt+response" if bool(args.trace_response) else "prompt"
    render_argmax_space(
        coords,
        weights,
        ids,
        os.path.join(args.out_dir, "primitive_argmax_space.png"),
        f"Primitive argmax over runtime state space | {seq_text}",
    )
    render_primitive_grid(
        coords,
        weights,
        ids,
        os.path.join(args.out_dir, "primitive_grid.png"),
        f"Primitive-separated runtime state space | {seq_text}",
    )
    if 0 <= int(args.primitive) < weights.shape[1]:
        render_single_primitive(
            coords,
            weights,
            ids,
            int(args.primitive),
            os.path.join(args.out_dir, f"primitive_{int(args.primitive)}.png"),
            f"Runtime state space weighted by P{int(args.primitive)} | {seq_text}",
        )

    print(f"Loaded: {args.ckpt}")
    print(f"Prompt bytes: {len(ids)}")
    print(f"Wrote: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
