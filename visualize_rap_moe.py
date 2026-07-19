import argparse
import ast
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


def _load_ckpt_meta(path: str) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return {"step": None, "cfg": {}}
    cfg = ckpt.get("cfg", {})
    step = ckpt.get("step", None)
    return {"step": step, "cfg": cfg if isinstance(cfg, dict) else {}}


def _read_last_metrics(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    last: dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                try:
                    obj = ast.literal_eval(line)
                except Exception:
                    obj = None
            if isinstance(obj, dict):
                last = obj
    return last


def _fmt_mass(mass: list[float], top_n: int = 3) -> str:
    if not mass:
        return ""
    pairs = sorted([(i, float(v)) for i, v in enumerate(mass)], key=lambda x: x[1], reverse=True)[:top_n]
    return ", ".join([f"P{i}={v:.2f}" for i, v in pairs])


def _box(ax, xy, w, h, text, fc="#f6f6f6", ec="#333333", lw=1.5, fontsize=12, rounded=True):
    if rounded:
        patch = patches.FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
    else:
        patch = patches.Rectangle(xy, w, h, linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2.0, xy[1] + h / 2.0, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _arrow(ax, p0, p1, text=None, color="#333333", lw=1.8, style="->", fontsize=10):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle=style, lw=lw, color=color, shrinkA=0, shrinkB=0),
    )
    if text:
        mx = (p0[0] + p1[0]) / 2.0
        my = (p0[1] + p1[1]) / 2.0
        ax.text(mx, my + 0.025, text, ha="center", va="bottom", fontsize=fontsize, color=color)


def _draw_standard_moe(ax):
    ax.text(0.5, 0.96, "Standard MoE", ha="center", va="top", fontsize=16, weight="bold")
    _box(ax, (0.10, 0.79), 0.22, 0.09, "Token / hidden\nrepresentation", fc="#e8f0ff")
    _box(ax, (0.39, 0.79), 0.22, 0.09, "Router", fc="#fff1cc")
    _box(ax, (0.68, 0.79), 0.22, 0.09, "Combine", fc="#e9f7ef")

    for i, y in enumerate([0.61, 0.47, 0.33]):
        _box(ax, (0.36, y), 0.28, 0.09, f"Expert {i}\\nFFN block", fc="#f8e8ff")

    _arrow(ax, (0.32, 0.835), (0.39, 0.835))
    _arrow(ax, (0.50, 0.79), (0.50, 0.70), text="top-k / soft weights")
    _arrow(ax, (0.50, 0.70), (0.50, 0.655))
    _arrow(ax, (0.50, 0.70), (0.50, 0.515))
    _arrow(ax, (0.50, 0.70), (0.50, 0.375))
    _arrow(ax, (0.64, 0.655), (0.79, 0.79))
    _arrow(ax, (0.64, 0.515), (0.79, 0.79))
    _arrow(ax, (0.64, 0.375), (0.79, 0.79))

    ax.text(
        0.5,
        0.14,
        "Experts mostly transform the current token representation.\nRouting decides which feedforward experts process this step.",
        ha="center",
        va="center",
        fontsize=11,
    )


def _draw_rap_moe(ax, *, n_primitives: int, step: int | None, top_mass_text: str):
    ax.text(0.5, 0.96, "RAP Primitive-Routed MoE", ha="center", va="top", fontsize=16, weight="bold")
    _box(ax, (0.05, 0.80), 0.20, 0.09, "Input byte\nembedding", fc="#e8f0ff")
    _box(ax, (0.28, 0.80), 0.20, 0.09, "Prev state", fc="#e8f0ff")
    _box(ax, (0.51, 0.80), 0.17, 0.09, "Retrieval\n(optional)", fc="#e8f7ff")
    _box(ax, (0.73, 0.80), 0.18, 0.09, "Controller\n(router)", fc="#fff1cc")

    y_positions = np.linspace(0.61, 0.23, n_primitives)
    for i, y in enumerate(y_positions):
        _box(ax, (0.33, float(y)), 0.32, 0.055, f"Primitive {i}: tanh(Aᵢ·state + Bᵢ·x + biasᵢ)", fc="#f8e8ff", fontsize=9)
        _arrow(ax, (0.82, 0.80), (0.65, float(y) + 0.0275), text="wᵢ" if i == 0 else None, color="#aa6f00", fontsize=9)

    _box(ax, (0.70, 0.50), 0.21, 0.08, "Weighted mix\nΣ wᵢ · candᵢ", fc="#e9f7ef")
    _box(ax, (0.70, 0.37), 0.21, 0.08, "Slice gate", fc="#ffe9e9")
    _box(ax, (0.70, 0.22), 0.21, 0.09, "New state", fc="#e8f0ff")
    _box(ax, (0.70, 0.08), 0.21, 0.08, "Next-byte logits", fc="#f0f0f0")

    _arrow(ax, (0.25, 0.845), (0.73, 0.845), text="x")
    _arrow(ax, (0.48, 0.845), (0.73, 0.845), text="state")
    _arrow(ax, (0.68, 0.845), (0.73, 0.845), text="retrieval")

    _arrow(ax, (0.25, 0.845), (0.33, 0.637))
    _arrow(ax, (0.48, 0.845), (0.33, 0.637))
    _arrow(ax, (0.25, 0.845), (0.33, 0.257))
    _arrow(ax, (0.48, 0.845), (0.33, 0.257))

    for y in y_positions:
        _arrow(ax, (0.65, float(y) + 0.0275), (0.70, 0.54), color="#555555", lw=1.2)

    _arrow(ax, (0.80, 0.50), (0.80, 0.45))
    _arrow(ax, (0.80, 0.37), (0.80, 0.31), text="gate ⊙ mix + (1-gate) ⊙ state")
    _arrow(ax, (0.80, 0.22), (0.80, 0.16))

    subtitle = f"n_primitives={n_primitives}"
    if step is not None:
        subtitle += f" | checkpoint step={step}"
    ax.text(0.5, 0.16, subtitle, ha="center", va="center", fontsize=10, color="#444444")

    if top_mass_text:
        ax.text(0.5, 0.11, f"latest mean primitive mass: {top_mass_text}", ha="center", va="center", fontsize=10, color="#444444")

    ax.text(
        0.5,
        0.03,
        "RAP experts are state-transition operators. The router mixes primitive candidates,\nthen a gate decides how much of the mixed update overwrites the old state.",
        ha="center",
        va="bottom",
        fontsize=11,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints1/b.pt")
    p.add_argument("--metrics", type=str, default="")
    p.add_argument("--out", type=str, default="viz/rap_moe_diagram.png")
    args = p.parse_args()

    meta = _load_ckpt_meta(args.ckpt)
    cfg = meta.get("cfg", {}) or {}
    n_primitives = int(cfg.get("n_primitives", 8))
    step = meta.get("step", None)

    metrics = _read_last_metrics(args.metrics)
    top_mass_text = ""
    mass = metrics.get("mean_primitive_mass", None)
    if isinstance(mass, (list, tuple)) and len(mass) > 0:
        top_mass_text = _fmt_mass([float(x) for x in mass])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=180)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    _draw_standard_moe(axes[0])
    _draw_rap_moe(axes[1], n_primitives=n_primitives, step=step, top_mass_text=top_mass_text)

    fig.suptitle("Why RAP is MoE-like, but not a standard MoE", fontsize=20, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out)
    plt.close(fig)
    print(f"Wrote: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
