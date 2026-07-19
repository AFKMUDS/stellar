from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from raplib.model import ModelConfig, PrimitiveManifoldLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-ckpt", required=True)
    parser.add_argument("--after-ckpt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="primitive_training_comparison.gif")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-bytes", type=int, default=96)
    parser.add_argument("--fps", type=int, default=3)
    return parser.parse_args()


def load_standard_model(path: str, device: torch.device) -> PrimitiveManifoldLM:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_kind", "standard") != "standard":
        raise ValueError(f"{path} is not a standard PrimitiveManifoldLM checkpoint")
    valid = {field.name for field in fields(ModelConfig)}
    cfg = ModelConfig(**{key: value for key, value in checkpoint["cfg"].items() if key in valid})
    model = PrimitiveManifoldLM(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def trace_model(model: PrimitiveManifoldLM, ids: list[int], device: torch.device) -> dict[str, np.ndarray]:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        _, _, trace = model(x, return_trace=True, trace_detail=True)
    candidates = trace["primitive_candidates"].float()
    state_before = trace["state_before"].float().unsqueeze(-2)
    deltas = candidates - state_before
    if candidates.shape[-2] == 2:
        effect_cosine = F.cosine_similarity(deltas[:, :, 0], deltas[:, :, 1], dim=-1)
        effect_distance = torch.linalg.vector_norm(candidates[:, :, 0] - candidates[:, :, 1], dim=-1)
    else:
        effect_cosine = torch.zeros(x.shape[1], device=device)
        effect_distance = torch.zeros(x.shape[1], device=device)
    return {
        "weights": trace["prim_weights"][0].float().cpu().numpy(),
        "chosen": trace["chosen_primitive"][0].cpu().numpy(),
        "entropy": trace["routing_entropy"][0].float().cpu().numpy(),
        "gate": trace["gate_mean"][0].float().cpu().numpy(),
        "effect_cosine": effect_cosine[0].float().cpu().numpy(),
        "effect_distance": effect_distance[0].float().cpu().numpy(),
    }


def byte_label(value: int) -> str:
    if 32 <= value <= 126:
        return chr(value)
    return f"\\x{value:02x}"


def format_weights(weights: np.ndarray) -> str:
    return "  ".join(f"P{i}: {value:.2f}" for i, value in enumerate(weights))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ids = list(args.prompt.encode("utf-8", errors="replace"))[: max(1, int(args.max_bytes))]
    before = trace_model(load_standard_model(args.before_ckpt, device), ids, device)
    after = trace_model(load_standard_model(args.after_ckpt, device), ids, device)
    if before["weights"].shape[1] != 2 or after["weights"].shape[1] != 2:
        raise ValueError("This comparison animation currently requires two-primitive checkpoints")

    figure = plt.figure(figsize=(15, 9), dpi=130)
    grid = figure.add_gridspec(3, 2, height_ratios=[1.15, 1.15, 0.7])
    before_axis = figure.add_subplot(grid[0, 0])
    after_axis = figure.add_subplot(grid[0, 1])
    effect_axis = figure.add_subplot(grid[1, :])
    summary_axis = figure.add_subplot(grid[2, :])
    colors = ["#3776ab", "#e07a5f"]
    positions = np.arange(len(ids))

    def setup_axis(axis: plt.Axes, title: str) -> None:
        axis.set_title(title, fontsize=13, weight="bold")
        axis.set_ylim(0, 1.05)
        axis.set_xlim(-0.5, max(0.5, len(ids) - 0.5))
        axis.set_ylabel("routing weight")
        axis.set_xlabel("byte position")
        axis.grid(axis="y", alpha=0.22)
        axis.set_xticks(positions)
        axis.set_xticklabels([byte_label(value) for value in ids], fontsize=8)

    setup_axis(before_axis, "Before: CE trains primitives indirectly")
    setup_axis(after_axis, "After: CE + optional primitive-focused losses")
    before_lines = [before_axis.plot([], [], color=colors[i], linewidth=2, label=f"P{i}")[0] for i in range(2)]
    after_lines = [after_axis.plot([], [], color=colors[i], linewidth=2, label=f"P{i}")[0] for i in range(2)]
    before_dot = before_axis.scatter([], [], s=100, color="#111111", zorder=3)
    after_dot = after_axis.scatter([], [], s=100, color="#111111", zorder=3)
    before_axis.legend(loc="upper right")
    after_axis.legend(loc="upper right")

    effect_axis.set_title("Candidate transition separation at the current byte", fontsize=13, weight="bold")
    effect_axis.set_xlim(-1.05, 1.05)
    effect_axis.set_ylim(0, max(1.0, float(np.max(np.r_[before["effect_distance"], after["effect_distance"]])) * 1.25))
    effect_axis.set_xlabel("cosine(P0(state, byte) - state, P1(state, byte) - state)")
    effect_axis.set_ylabel("candidate-state distance")
    effect_axis.grid(alpha=0.22)
    before_effect = effect_axis.scatter([], [], s=150, color="#3776ab", label="before")
    after_effect = effect_axis.scatter([], [], s=150, color="#e07a5f", label="after")
    effect_axis.legend(loc="upper right")

    summary_axis.axis("off")
    summary = summary_axis.text(0.01, 0.96, "", va="top", family="monospace", fontsize=11)
    title = figure.suptitle("Primitive transition comparison", fontsize=18, weight="bold")

    def update(frame: int):
        frame = min(frame, len(ids) - 1)
        visible = slice(0, frame + 1)
        for primitive, line in enumerate(before_lines):
            line.set_data(positions[visible], before["weights"][visible, primitive])
        for primitive, line in enumerate(after_lines):
            line.set_data(positions[visible], after["weights"][visible, primitive])
        before_dot.set_offsets([[frame, before["weights"][frame, before["chosen"][frame]]]])
        after_dot.set_offsets([[frame, after["weights"][frame, after["chosen"][frame]]]])
        before_effect.set_offsets([[before["effect_cosine"][frame], before["effect_distance"][frame]]])
        after_effect.set_offsets([[after["effect_cosine"][frame], after["effect_distance"][frame]]])
        token = byte_label(ids[frame])
        title.set_text(f"Primitive transition comparison | step {frame} | byte {token!r}")
        summary.set_text(
            "Before: routing and primitive transitions receive gradients only from next-byte CE.\n"
            "After: architecture is unchanged; optional losses can sharpen routing, stabilize chains, balance usage, and separate effects.\n\n"
            f"Before  {format_weights(before['weights'][frame])} | entropy {before['entropy'][frame]:.3f} | gate {before['gate'][frame]:.3f} | chosen P{before['chosen'][frame]}\n"
            f"After   {format_weights(after['weights'][frame])} | entropy {after['entropy'][frame]:.3f} | gate {after['gate'][frame]:.3f} | chosen P{after['chosen'][frame]}\n"
            f"Effect: cosine closer to 0 means more distinct delta directions; larger distance means more separated candidate states.\n"
            f"Before cosine {before['effect_cosine'][frame]:.3f}, distance {before['effect_distance'][frame]:.3f} | "
            f"After cosine {after['effect_cosine'][frame]:.3f}, distance {after['effect_distance'][frame]:.3f}"
        )
        return [*before_lines, *after_lines, before_dot, after_dot, before_effect, after_effect, summary, title]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(figure, update, frames=len(ids), interval=1000 / max(1, int(args.fps)), blit=False)
    movie.save(output, writer="pillow", fps=max(1, int(args.fps)))
    plt.close(figure)
    print(f"Saved animation: {output.resolve()}")


if __name__ == "__main__":
    main()
