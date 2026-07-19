import argparse
import ast
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _read_history_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    rows = payload.get("history", [])
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = [r for r in rows if isinstance(r, dict)]
    out.sort(key=lambda r: int(r.get("step", 0)))
    return out


def _read_metrics_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # train_rap.py writes python dict repr via str(rec), not strict JSON.
            # Use ast.literal_eval to parse safely.
            try:
                rec = ast.literal_eval(s)
            except Exception:
                continue

            if isinstance(rec, dict):
                rows.append(rec)

    rows.sort(key=lambda r: int(r.get("step", 0)))
    return rows


def _read_rows(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        rows = _read_history_json(path)
        if rows:
            return rows
    return _read_metrics_jsonl(path)


def _extract_series(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]:
    steps: List[int] = []
    series: Dict[str, List[float]] = {
        "loss": [],
        "mean_entropy": [],
        "mean_maxprob": [],
    }
    counts_list: List[List[int]] = []

    for r in rows:
        if "step" not in r:
            continue
        try:
            step = int(r["step"])
        except Exception:
            continue

        steps.append(step)

        for k in list(series.keys()):
            v = r.get(k, None)
            series[k].append(float("nan") if v is None else float(v))

        c = r.get("argmax_counts", None)
        if isinstance(c, (list, tuple)) and len(c) > 0:
            try:
                counts_list.append([int(x) for x in c])
            except Exception:
                counts_list.append([])
        else:
            counts_list.append([])

    steps_arr = np.asarray(steps, dtype=np.int64)
    out_series = {k: np.asarray(v, dtype=np.float64) for k, v in series.items()}

    counts_arr: Optional[np.ndarray] = None
    if len(counts_list) > 0 and any(len(x) > 0 for x in counts_list):
        k_max = max((len(x) for x in counts_list), default=0)
        tmp = np.full((len(counts_list), k_max), np.nan, dtype=np.float64)
        for i, c in enumerate(counts_list):
            if len(c) == 0:
                continue
            tmp[i, : len(c)] = np.asarray(c, dtype=np.float64)
        counts_arr = tmp

    return steps_arr, out_series, counts_arr


def _extract_compute_and_speak_series(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    keys = [
        "loss",
        "total_loss",
        "perplexity",
        "mean_route_direct",
        "mean_route_copy_recent",
        "mean_route_bit_op",
        "top_route_direct_rate",
        "top_route_copy_recent_rate",
        "top_route_bit_op_rate",
        "target_reachable_by_ops_rate",
        "target_in_copy_window_rate",
        "mean_op_target_mass",
        "mean_copy_choice_target_mass",
        "top_op_keep_rate",
        "top_op_inc_rate",
        "top_op_dec_rate",
        "top_op_flip_bit_0_rate",
        "top_op_flip_bit_1_rate",
        "top_op_flip_bit_2_rate",
        "top_op_flip_bit_3_rate",
        "top_op_flip_bit_4_rate",
        "bit_identity_mean_loss",
        "bit_flip1_mean_loss",
        "bit_flip2plus_mean_loss",
    ]
    steps: List[int] = []
    series: Dict[str, List[float]] = {k: [] for k in keys}
    for r in rows:
        if "step" not in r:
            continue
        try:
            step = int(r["step"])
        except Exception:
            continue
        steps.append(step)
        for k in keys:
            v = r.get(k, None)
            series[k].append(float("nan") if v is None else float(v))
    return np.asarray(steps, dtype=np.int64), {k: np.asarray(v, dtype=np.float64) for k, v in series.items()}


def _maybe_smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    if y.size < window:
        return y

    # Simple moving average; keep same length by padding endpoints.
    kernel = np.ones(window, dtype=np.float64) / float(window)
    ys = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), kernel, mode="same")
    return ys


def _plot(path: str, out: str, smooth: int, show: bool) -> bool:
    rows = _read_rows(path)
    if len(rows) == 0:
        return False

    if any("mean_route_direct" in r for r in rows):
        return _plot_compute_and_speak(path, out, smooth, show, rows)

    steps, series, counts = _extract_series(rows)
    if steps.size == 0:
        return False

    fig = plt.figure(figsize=(12, 8), dpi=120)

    ax1 = plt.subplot(2, 1, 1)
    for k, y in series.items():
        ys = _maybe_smooth(y, smooth)
        ax1.plot(steps, ys, label=k)
    ax1.set_xlabel("step")
    ax1.set_title(os.path.basename(path))
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2 = plt.subplot(2, 1, 2)
    if counts is not None and counts.shape[1] > 0:
        # Convert per-log counts to proportions (more comparable across runs).
        denom = np.nansum(counts, axis=1, keepdims=True)
        denom = np.where(denom <= 0, np.nan, denom)
        props = counts / denom

        for i in range(props.shape[1]):
            ax2.plot(steps, props[:, i], label=f"prim_{i}")

        ax2.set_ylabel("argmax proportion")
        ax2.set_ylim(0.0, 1.0)
        ax2.legend(ncol=min(8, props.shape[1]), fontsize=8, loc="upper right")
    else:
        ax2.text(0.5, 0.5, "No argmax_counts in metrics.", ha="center", va="center", transform=ax2.transAxes)

    ax2.set_xlabel("step")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out)

    if show:
        plt.show()
    plt.close(fig)
    return True


def _plot_compute_and_speak(path: str, out: str, smooth: int, show: bool, rows: List[Dict[str, Any]]) -> bool:
    steps, series = _extract_compute_and_speak_series(rows)
    if steps.size == 0:
        return False

    fig = plt.figure(figsize=(14, 10), dpi=120)

    ax1 = plt.subplot(2, 2, 1)
    for k in ["loss", "total_loss", "perplexity"]:
        y = series.get(k)
        if y is not None:
            ax1.plot(steps, _maybe_smooth(y, smooth), label=k)
    ax1.set_title("Loss / Perplexity")
    ax1.set_xlabel("step")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2 = plt.subplot(2, 2, 2)
    for k in ["mean_route_direct", "mean_route_copy_recent", "mean_route_bit_op"]:
        y = series.get(k)
        if y is not None:
            ax2.plot(steps, _maybe_smooth(y, smooth), label=k)
    for k in ["top_route_direct_rate", "top_route_copy_recent_rate", "top_route_bit_op_rate"]:
        y = series.get(k)
        if y is not None:
            ax2.plot(steps, _maybe_smooth(y, smooth), linestyle="--", alpha=0.8, label=k)
    ax2.set_title("Route Usage")
    ax2.set_xlabel("step")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)

    ax3 = plt.subplot(2, 2, 3)
    for k in ["target_reachable_by_ops_rate", "target_in_copy_window_rate", "mean_op_target_mass", "mean_copy_choice_target_mass"]:
        y = series.get(k)
        if y is not None:
            ax3.plot(steps, _maybe_smooth(y, smooth), label=k)
    ax3.set_title("Tool Opportunity / Target Mass")
    ax3.set_xlabel("step")
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best", fontsize=8)

    ax4 = plt.subplot(2, 2, 4)
    for k in [
        "top_op_keep_rate",
        "top_op_inc_rate",
        "top_op_dec_rate",
        "top_op_flip_bit_0_rate",
        "top_op_flip_bit_1_rate",
        "top_op_flip_bit_2_rate",
        "top_op_flip_bit_3_rate",
        "top_op_flip_bit_4_rate",
    ]:
        y = series.get(k)
        if y is not None:
            ax4.plot(steps, _maybe_smooth(y, smooth), label=k.replace("top_op_", "").replace("_rate", ""))
    ax4.set_title("Top Op Trends")
    ax4.set_xlabel("step")
    ax4.set_ylim(0.0, 1.0)
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle(os.path.basename(path))
    plt.tight_layout()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out)

    if show:
        plt.show()
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=str, required=True, help="Path to metrics JSONL or a trainer JSON file with history")
    p.add_argument("--out", type=str, default="viz/training_metrics.png")
    p.add_argument("--smooth", type=int, default=1, help="Moving average window (in points).")
    p.add_argument("--show", action="store_true", help="Display the figure window")

    p.add_argument("--follow", action="store_true", help="Keep updating the plot while metrics file grows")
    p.add_argument("--refresh", type=float, default=2.0, help="Seconds between refreshes when --follow")

    args = p.parse_args()

    last_size = -1
    while True:
        size = os.path.getsize(args.metrics) if os.path.exists(args.metrics) else -1
        if size != last_size:
            ok = _plot(args.metrics, args.out, args.smooth, show=args.show and not args.follow)
            if ok:
                print(f"Wrote: {os.path.abspath(args.out)}")
            else:
                print("No valid metrics records yet.")
            last_size = size

        if not args.follow:
            break
        time.sleep(float(args.refresh))


if __name__ == "__main__":
    main()
