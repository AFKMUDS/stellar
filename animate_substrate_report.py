from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = [
    "state_entropy",
    "primitive_entropy",
    "expected_state_mass",
    "chain_dominance",
    "chain_reuse_ratio",
]


METRIC_TITLES = {
    "state_entropy": "State Entropy",
    "primitive_entropy": "Primitive Entropy",
    "expected_state_mass": "Expected State Mass",
    "chain_dominance": "Chain Dominance",
    "chain_reuse_ratio": "Chain Reuse Ratio",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Animate a substrate experiment report.")
    p.add_argument("--report", type=str, required=True)
    p.add_argument("--out", type=str, default="substrate_animation.gif")
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--title", type=str, default="Substrate Dynamics")
    return p.parse_args()


def load_report(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_series(history: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for item in history:
        if key in item:
            vals.append(float(item[key]))
        else:
            vals.append(float("nan"))
    return vals


def text_block(lines: list[str], limit: int) -> str:
    trimmed = lines[:limit]
    if not trimmed:
        return "-"
    return "\n".join(trimmed)


def proto_lines(frame: dict[str, Any], limit: int = 6) -> list[str]:
    rows = frame.get("proto_words", []) or []
    out = []
    for row in rows[:limit]:
        chain = str(row.get("chain", "?"))
        strength = float(row.get("strength", 0.0))
        out.append(f"{chain} | {strength:.2f}")
    return out


def primitive_lines(frame: dict[str, Any], limit: int = 6) -> list[str]:
    rows = frame.get("top_primitives", []) or []
    out = []
    for row in rows[:limit]:
        name = str(row.get("primitive", "?"))
        usage = float(row.get("usage", 0.0))
        out.append(f"{name} | {usage:.0f}")
    return out


def macro_lines(frame: dict[str, Any], limit: int = 6) -> list[str]:
    rows = frame.get("macro_sequences", []) or []
    out = []
    for row in rows[:limit]:
        if isinstance(row, list):
            out.append(" -> ".join(str(x) for x in row))
        else:
            out.append(str(row))
    return out


def render_frame(history: list[dict[str, Any]], idx: int, report_title: str) -> np.ndarray:
    frame = history[idx]
    steps = [int(x.get("step", i + 1)) for i, x in enumerate(history)]

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    metric_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    metric_axes2 = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    text_ax = fig.add_subplot(gs[1, 2])
    all_axes = metric_axes + metric_axes2

    for ax, key in zip(all_axes, METRIC_KEYS):
        ys = metric_series(history, key)
        arr = np.array(ys, dtype=float)
        valid = arr[np.isfinite(arr)]
        ymin = float(valid.min()) if valid.size else 0.0
        ymax = float(valid.max()) if valid.size else 1.0
        if abs(ymax - ymin) < 1e-9:
            ymin -= 0.1
            ymax += 0.1
        pad = 0.08 * (ymax - ymin)
        ax.plot(steps[: idx + 1], arr[: idx + 1], color="#7aa2f7", linewidth=2.5)
        ax.scatter([steps[idx]], [arr[idx]], color="#f7768e", s=50, zorder=3)
        ax.set_title(METRIC_TITLES[key])
        ax.set_xlim(steps[0], steps[-1])
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(alpha=0.25)

    text_ax.axis("off")
    primitives = primitive_lines(frame)
    protos = proto_lines(frame)
    macros = macro_lines(frame)

    summary_lines = [
        f"step: {int(frame.get('step', idx + 1))}",
        f"active_states: {int(frame.get('active_state_count', 0))}",
        f"active_edges: {int(frame.get('active_edge_count', 0))}",
        f"active_trigrams: {int(frame.get('active_trigram_count', 0))}",
        f"macro_count: {int(frame.get('macro_count', 0))}",
        f"locality: {float(frame.get('support_locality_mean_hamming', 0.0)):.3f}",
        "",
        "top primitives:",
        text_block(primitives, 6),
        "",
        "proto-words:",
        text_block(protos, 6),
        "",
        "macros:",
        text_block(macros, 6),
    ]
    text_ax.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.suptitle(f"{report_title} | frame {idx + 1}/{len(history)}", fontsize=16)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def main() -> None:
    args = parse_args()
    payload = load_report(args.report)
    history = payload.get("history", []) or []
    if not history:
        raise ValueError("Report has no history to animate.")

    frames = [render_frame(history, idx, args.title) for idx in range(len(history))]
    imageio.mimsave(args.out, frames, fps=max(1, int(args.fps)), loop=0)
    print(f"saved_animation={args.out}")


if __name__ == "__main__":
    main()
