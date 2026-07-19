from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TRACE_PATH = Path(__file__).with_name("generation_trace.json")
OUTPUT_PATH = Path(__file__).with_name("computation_trace.png")
OUTPUT_PATH_ALT = Path(__file__).with_name("computation_trace_timeline.png")


def load_trace(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    trace = obj.get("trace", [])
    if not isinstance(trace, list):
        raise RuntimeError("generation_trace.json has no valid 'trace' list")
    return trace


def byte_label(b: int) -> str:
    if b == 32:
        return "sp"
    if 32 <= b <= 126:
        return chr(b)
    return f"x{b:02x}"


def main() -> None:
    trace = load_trace(TRACE_PATH)
    if not trace:
        raise RuntimeError("Trace is empty")

    steps = np.array([int(row["step"]) for row in trace], dtype=np.int32)
    primitive_weights = np.array([row["primitive_weights"] for row in trace], dtype=np.float32)
    routing_entropy = np.array([float(row.get("routing_entropy_norm", 0.0)) for row in trace], dtype=np.float32)
    state_norm = np.array([float(row.get("state_norm", 0.0)) for row in trace], dtype=np.float32)
    retrieval_hit = np.array([1.0 if bool(row.get("retrieval_hit", False)) else 0.0 for row in trace], dtype=np.float32)
    chosen_bytes = np.array([int(row["chosen_byte"]) for row in trace], dtype=np.int32)

    active_primitive = np.argmax(primitive_weights, axis=1)
    primitive_confidence = np.max(primitive_weights, axis=1)

    n_prims = primitive_weights.shape[1]

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[3.2, 1.2, 1.2, 1.2])

    ax0 = fig.add_subplot(gs[0])
    heat = primitive_weights.T
    im = ax0.imshow(
        heat,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        extent=[steps[0] - 0.5, steps[-1] + 0.5, -0.5, n_prims - 0.5],
        vmin=0.0,
        vmax=1.0,
    )
    ax0.plot(steps, active_primitive, color="cyan", linewidth=1.5, label="argmax primitive")

    hit_steps = steps[retrieval_hit > 0.5]
    hit_prims = active_primitive[retrieval_hit > 0.5]
    if len(hit_steps) > 0:
        ax0.scatter(hit_steps, hit_prims, s=50, color="lime", edgecolors="black", linewidths=0.5, label="retrieval hit", zorder=5)

    ax0.set_title("RAP computation trace: primitive routing over generation steps")
    ax0.set_ylabel("Primitive ID")
    ax0.set_yticks(range(n_prims))
    ax0.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax0, pad=0.01)
    cbar.set_label("Routing weight")

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(steps, primitive_confidence, color="tab:blue", label="primitive confidence")
    ax1.plot(steps, 1.0 - routing_entropy, color="tab:orange", label="1 - routing entropy")
    if len(hit_steps) > 0:
        ax1.vlines(hit_steps, ymin=0.0, ymax=1.0, color="lime", alpha=0.25, linewidth=1)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Confidence")
    ax1.legend(loc="lower right")

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax2.plot(steps, state_norm, color="tab:red", label="state norm")
    if len(hit_steps) > 0:
        ax2.scatter(hit_steps, state_norm[retrieval_hit > 0.5], color="lime", edgecolors="black", linewidths=0.5, s=35, zorder=5)
    ax2.set_ylabel("State norm")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax3.step(steps, retrieval_hit, where="mid", color="tab:green", label="retrieval hit")
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_ylabel("Hit")
    ax3.set_xlabel("Generation step")
    ax3.legend(loc="upper right")

    sample_every = max(1, len(steps) // 24)
    tick_idx = np.arange(0, len(steps), sample_every)
    ax3.set_xticks(steps[tick_idx])
    ax3.set_xticklabels([byte_label(int(chosen_bytes[i])) for i in tick_idx], rotation=0)

    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)

    fig2, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, constrained_layout=True)

    axes[0].scatter(steps, active_primitive, c=primitive_confidence, cmap="viridis", s=70)
    if len(hit_steps) > 0:
        axes[0].scatter(hit_steps, hit_prims, s=120, facecolors="none", edgecolors="red", linewidths=1.5)
    axes[0].set_ylabel("Active primitive")
    axes[0].set_title("Primitive switching timeline (color = routing confidence, red ring = retrieval hit)")

    axes[1].plot(steps, routing_entropy, color="tab:purple", label="routing entropy")
    axes[1].plot(steps, primitive_confidence, color="tab:blue", label="primitive confidence")
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("Router stats")

    axes[2].bar(steps, chosen_bytes, color=np.where(retrieval_hit > 0.5, "tab:green", "tab:gray"))
    axes[2].set_ylabel("Chosen byte")
    axes[2].set_xlabel("Generation step")

    fig2.savefig(OUTPUT_PATH_ALT, dpi=180)
    plt.close(fig2)

    switch_points = [0]
    for i in range(1, len(active_primitive)):
        if int(active_primitive[i]) != int(active_primitive[i - 1]):
            switch_points.append(i)

    segments = []
    start = 0
    for i in range(1, len(active_primitive) + 1):
        end_segment = i == len(active_primitive) or int(active_primitive[i]) != int(active_primitive[i - 1])
        if end_segment:
            seg = {
                "start_step": int(steps[start]),
                "end_step": int(steps[i - 1]),
                "primitive": int(active_primitive[start]),
                "length": int(i - start),
                "mean_confidence": float(np.mean(primitive_confidence[start:i])),
                "retrieval_hits": int(np.sum(retrieval_hit[start:i])),
                "bytes": [int(x) for x in chosen_bytes[start:i].tolist()],
                "byte_labels": [byte_label(int(x)) for x in chosen_bytes[start:i].tolist()],
            }
            segments.append(seg)
            start = i

    summary = {
        "steps": int(len(steps)),
        "primitive_usage_counts": {str(i): int(np.sum(active_primitive == i)) for i in range(n_prims)},
        "mean_confidence": float(np.mean(primitive_confidence)),
        "mean_routing_entropy": float(np.mean(routing_entropy)),
        "retrieval_hit_rate": float(np.mean(retrieval_hit)),
        "switch_count": int(len(segments) - 1),
        "segments": segments,
        "output_files": [str(OUTPUT_PATH.name), str(OUTPUT_PATH_ALT.name)],
    }

    with open(Path(__file__).with_name("computation_trace_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {OUTPUT_PATH_ALT}")
    print("Primitive usage counts:", summary["primitive_usage_counts"])
    print("Retrieval hit rate:", f"{summary['retrieval_hit_rate']:.3f}")
    print("Switch count:", summary["switch_count"])


if __name__ == "__main__":
    main()
