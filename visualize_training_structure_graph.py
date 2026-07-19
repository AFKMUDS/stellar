import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def _box(ax, xy: Tuple[float, float], w: float, h: float, text: str) -> None:
    x, y = xy
    rect = Rectangle((x, y), w, h, linewidth=1.2, edgecolor="#2d3a4a", facecolor="#eef3fb")
    ax.add_patch(rect)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=9, family="monospace")


def _arrow(ax, src: Tuple[float, float], dst: Tuple[float, float], text: str = "", rad: float = 0.0, color: str = "#2d3a4a") -> None:
    a = FancyArrowPatch(
        src,
        dst,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.5,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    if text:
        mx = (src[0] + dst[0]) / 2.0
        my = (src[1] + dst[1]) / 2.0
        ax.text(mx, my + 0.18, text, ha="center", va="center", fontsize=8, family="monospace", color=color)


def draw_training_pipeline(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=140)
    ax.set_axis_off()

    W = 3.6
    H = 1.0

    pos: Dict[str, Tuple[float, float]] = {
        "cfg": (0.5, 6.2),
        "stream": (0.5, 4.8),
        "batch": (0.5, 3.4),
        "forward": (5.0, 3.4),
        "lossmask": (9.5, 4.6),
        "xe": (9.5, 3.4),
        "observer": (9.5, 2.2),
        "aux": (9.5, 1.0),
        "total": (5.0, 1.0),
        "backward": (5.0, -0.4),
        "opt": (5.0, -1.8),
        "log": (9.5, -1.8),
        "metrics": (9.5, -0.4),
        "ckpt": (9.5, -3.2),
        "loop": (0.5, -1.8),
    }

    _box(ax, pos["cfg"], W, H, "(1) build/load cfg\nresume? load ckpt+opt")
    _box(ax, pos["stream"], W, H, "(2) data stream\niter_train_sequences / raw")
    _box(ax, pos["batch"], W, H, "(3) make batch\npad/truncate\ncreate x,y")

    _box(ax, pos["forward"], W, H, "(4) model(x)\nforward unroll over t")

    _box(ax, pos["lossmask"], W, H, "(5) loss_mask\n(prompt vs target)")
    _box(ax, pos["xe"], W, H, "(6) cross-entropy\nloss_per_token")
    _box(ax, pos["observer"], W, H, "(7) reweight loss\nmeta observer / portal\ntiny observer mult")
    _box(ax, pos["aux"], W, H, "(8) aux losses\n(digit/nibbles)")

    _box(ax, pos["total"], W, H, "(9) total loss")
    _box(ax, pos["backward"], W, H, "(10) backward\nloss.backward()")
    _box(ax, pos["opt"], W, H, "(11) optimizer step\nclip_grad\nAdamW.step()")

    _box(ax, pos["metrics"], W, H, "(12) periodic eval\nmetrics (entropy/usage)")
    _box(ax, pos["log"], W, H, "(13) log train events\n.events.jsonl")
    _box(ax, pos["ckpt"], W, H, "(14) checkpoint\n(out.pt + step*.pt)")

    _box(ax, pos["loop"], W, H, "repeat for step")

    def right_mid(k: str):
        x, y = pos[k]
        return (x + W, y + H / 2.0)

    def left_mid(k: str):
        x, y = pos[k]
        return (x, y + H / 2.0)

    def top_mid(k: str):
        x, y = pos[k]
        return (x + W / 2.0, y + H)

    def bot_mid(k: str):
        x, y = pos[k]
        return (x + W / 2.0, y)

    _arrow(ax, right_mid("cfg"), left_mid("stream"))
    _arrow(ax, bot_mid("stream"), top_mid("batch"))
    _arrow(ax, right_mid("batch"), left_mid("forward"))

    _arrow(ax, right_mid("forward"), left_mid("xe"))
    _arrow(ax, right_mid("batch"), left_mid("lossmask"), text="y + prompt_lens")
    _arrow(ax, bot_mid("lossmask"), top_mid("xe"), text="mask")

    _arrow(ax, bot_mid("xe"), top_mid("observer"), text="loss_tok")
    _arrow(ax, bot_mid("observer"), top_mid("aux"), text="optional")

    _arrow(ax, left_mid("xe"), right_mid("total"), text="masked sum", rad=0.15)
    _arrow(ax, left_mid("observer"), right_mid("total"), text="reweight", rad=-0.15)
    _arrow(ax, left_mid("aux"), bot_mid("total"), text="+ aux", rad=0.0)

    _arrow(ax, bot_mid("total"), top_mid("backward"))
    _arrow(ax, bot_mid("backward"), top_mid("opt"), text="grads")

    _arrow(ax, right_mid("opt"), left_mid("metrics"), text="every N", rad=-0.08)
    _arrow(ax, right_mid("opt"), left_mid("log"), text="every N", rad=0.08)
    _arrow(ax, right_mid("opt"), left_mid("ckpt"), text="save_every", rad=-0.12)

    _arrow(ax, left_mid("opt"), right_mid("loop"), text="step++", rad=0.12)
    _arrow(ax, bot_mid("loop"), top_mid("batch"), text="next batch", rad=0.0)

    ax.set_xlim(0, 14.2)
    ax.set_ylim(-3.8, 7.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def draw_unroll_with_back_edges(out_path: str, t: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), dpi=160)
    ax.set_axis_off()

    w = 1.6
    h = 0.9
    y_main = 1.8
    y_mem = 0.3

    xs = [0.6 + i * 2.0 for i in range(max(2, int(t)))]

    for i, x in enumerate(xs):
        _box(ax, (x, y_main), w, h, f"t={i}\nstep()")
        _box(ax, (x, y_mem), w, h, f"state_hist[{i}]\nlast_pos")

        if i > 0:
            _arrow(ax, (xs[i - 1] + w, y_main + h / 2.0), (x, y_main + h / 2.0), text="forward")

        _arrow(ax, (x + w / 2.0, y_main), (x + w / 2.0, y_mem + h), text="write state")

    for i in range(1, len(xs)):
        x_src = xs[i]
        x_dst = xs[max(0, i - 2)]
        _arrow(
            ax,
            (x_src + w / 2.0, y_main + h),
            (x_dst + w / 2.0, y_mem + h),
            text="retrieval: inv/rev lookup\n(go back if seen)",
            rad=0.35,
            color="#7a2e2e",
        )

    x0 = xs[0]
    xN = xs[-1]
    _arrow(ax, (xN + w, y_main + h / 2.0), (x0, y_main + h / 2.0), text="backprop (BPTT)\nloss gradients", rad=-0.35, color="#1f5f2a")

    ax.text(0.6, 3.25, "Forward unroll over time + 'go back' symmetry retrieval edges + gradient flow", fontsize=10, family="monospace")

    ax.set_xlim(0, xs[-1] + 3.2)
    ax.set_ylim(-0.2, 3.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="viz")
    p.add_argument("--t", type=int, default=6, help="Number of time steps to show in the unroll diagram")
    args = p.parse_args()

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    draw_training_pipeline(os.path.join(out_dir, "training_pipeline.png"))
    draw_unroll_with_back_edges(os.path.join(out_dir, "training_unroll.png"), int(args.t))

    print(os.path.abspath(os.path.join(out_dir, "training_pipeline.png")))
    print(os.path.abspath(os.path.join(out_dir, "training_unroll.png")))


if __name__ == "__main__":
    main()
