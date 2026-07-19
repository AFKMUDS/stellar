from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from raplib.model import ModelConfig, PrimitiveManifoldLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", default="primitive_process.gif")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-bytes", type=int, default=64)
    parser.add_argument("--state-dims", type=int, default=64)
    parser.add_argument("--fps", type=int, default=2)
    return parser.parse_args()


def load_model(path: str, device: torch.device) -> PrimitiveManifoldLM:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_kind", "standard") != "standard":
        raise ValueError("This visualizer requires a standard PrimitiveManifoldLM checkpoint")
    valid = {field.name for field in fields(ModelConfig)}
    cfg = ModelConfig(**{key: value for key, value in checkpoint["cfg"].items() if key in valid})
    if cfg.n_primitives != 2:
        raise ValueError("This visualizer currently requires exactly two primitives")
    model = PrimitiveManifoldLM(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def trace(model: PrimitiveManifoldLM, ids: list[int], device: torch.device) -> dict[str, np.ndarray]:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        _, _, values = model(x, return_trace=True, trace_detail=True)
    return {key: value[0].float().cpu().numpy() for key, value in values.items()}


def byte_label(value: int) -> str:
    if 32 <= value <= 126:
        return repr(chr(value))
    return f"0x{value:02x}"


def add_box(axis: plt.Axes, x: float, y: float, width: float, height: float, label: str, color: str) -> None:
    axis.add_patch(patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", facecolor=color, edgecolor="#263238", linewidth=1.4))
    axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10, weight="bold", wrap=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ids = list(args.prompt.encode("utf-8", errors="replace"))[: max(1, int(args.max_bytes))]
    values = trace(load_model(args.ckpt, device), ids, device)
    dims = min(int(args.state_dims), values["state_before"].shape[-1])
    state_before = values["state_before"][:, :dims]
    candidates = values["primitive_candidates"][:, :, :dims]
    mixed = values["mixed_candidate"][:, :dims]
    state_after = values["state_after"][:, :dims]
    gates = values["slice_gate"][:, :dims]
    all_values = np.concatenate([state_before.ravel(), candidates.ravel(), mixed.ravel(), state_after.ravel()])
    state_limit = max(0.1, float(np.quantile(np.abs(all_values), 0.99)))

    figure = plt.figure(figsize=(15, 10), dpi=130)
    grid = figure.add_gridspec(3, 2, height_ratios=[0.9, 1.8, 1.0])
    flow_axis = figure.add_subplot(grid[0, :])
    state_axis = figure.add_subplot(grid[1, 0])
    candidate_axis = figure.add_subplot(grid[1, 1])
    result_axis = figure.add_subplot(grid[2, :])
    flow_axis.set_xlim(0, 1)
    flow_axis.set_ylim(0, 1)
    flow_axis.axis("off")
    add_box(flow_axis, 0.02, 0.25, 0.13, 0.45, "current\nbyte", "#f6bd60")
    add_box(flow_axis, 0.21, 0.25, 0.15, 0.45, "router\nsoftmax", "#84a59d")
    add_box(flow_axis, 0.43, 0.25, 0.19, 0.45, "P0 and P1\ncandidates", "#90caf9")
    add_box(flow_axis, 0.69, 0.25, 0.12, 0.45, "weighted\nmix", "#ce93d8")
    add_box(flow_axis, 0.86, 0.25, 0.12, 0.45, "slice gate\nnext state", "#ef9a9a")
    for start, end in [(0.15, 0.21), (0.36, 0.43), (0.62, 0.69), (0.81, 0.86)]:
        flow_axis.annotate("", xy=(end, 0.475), xytext=(start, 0.475), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#37474f"})

    state_image = state_axis.imshow(np.vstack([state_before[0], mixed[0], gates[0], state_after[0]]), aspect="auto", cmap="coolwarm", vmin=-state_limit, vmax=state_limit)
    state_axis.set_title("Actual recurrent state update", weight="bold")
    state_axis.set_yticks([0, 1, 2, 3], labels=["old state", "weighted mix", "slice gate", "new state"])
    state_axis.set_xlabel(f"first {dims} state dimensions")
    figure.colorbar(state_image, ax=state_axis, fraction=0.046, pad=0.04)

    candidate_image = candidate_axis.imshow(candidates[0], aspect="auto", cmap="coolwarm", vmin=-state_limit, vmax=state_limit)
    candidate_axis.set_title("Atomic primitive transition candidates", weight="bold")
    candidate_axis.set_yticks([0, 1], labels=["P0 candidate", "P1 candidate"])
    candidate_axis.set_xlabel(f"first {dims} state dimensions")
    figure.colorbar(candidate_image, ax=candidate_axis, fraction=0.046, pad=0.04)

    result_axis.set_xlim(-0.6, 1.6)
    result_axis.set_ylim(0, 1.05)
    result_axis.set_xticks([0, 1], labels=["P0", "P1"])
    result_axis.set_ylabel("routing weight")
    result_axis.set_title("Router decision: candidates are mixed before the slice gate applies the recurrent update", weight="bold")
    result_axis.grid(axis="y", alpha=0.25)
    bars = result_axis.bar([0, 1], values["prim_weights"][0], color=["#3776ab", "#e07a5f"], width=0.55)
    info = result_axis.text(0.02, 0.93, "", transform=result_axis.transAxes, va="top", family="monospace", fontsize=10)
    title = figure.suptitle("Primitive process", fontsize=18, weight="bold")

    def update(frame: int):
        current = int(frame)
        state_image.set_data(np.vstack([state_before[current], mixed[current], gates[current], state_after[current]]))
        candidate_image.set_data(candidates[current])
        weights = values["prim_weights"][current]
        for index, bar in enumerate(bars):
            bar.set_height(weights[index])
            bar.set_alpha(1.0 if index == int(values["chosen_primitive"][current]) else 0.55)
        entropy = float(values["routing_entropy"][current])
        candidate_distance = float(np.linalg.norm(candidates[current, 0] - candidates[current, 1]))
        title.set_text(f"Primitive process | byte {current}: {byte_label(ids[current])}")
        info.set_text(
            f"weights: P0={weights[0]:.3f}, P1={weights[1]:.3f} | chosen=P{int(values['chosen_primitive'][current])} | routing entropy={entropy:.3f}\n"
            f"candidate distance={candidate_distance:.3f} | mean gate={float(values['gate_mean'][current]):.3f}\n"
            "The heatmaps show the actual tensors used for this byte: old state, each primitive's candidate, weighted mix, gate, and new state."
        )
        return [state_image, candidate_image, *bars, info, title]

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(figure, update, frames=len(ids), interval=1000 / max(1, int(args.fps)), blit=False)
    movie.save(output, writer="pillow", fps=max(1, int(args.fps)))
    plt.close(figure)
    print(f"Saved process animation: {output.resolve()}")


if __name__ == "__main__":
    main()
