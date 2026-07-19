from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

TRACE_PATH = Path(__file__).with_name("generation_trace.json")
SUMMARY_PATH = Path(__file__).with_name("program_lifetime_summary.json")
PLOT_PATH = Path(__file__).with_name("program_lifetimes.png")
GRAPH_PATH = Path(__file__).with_name("program_transition_graph.png")


def byte_label(b: int) -> str:
    if b == 32:
        return "sp"
    if 32 <= b <= 126:
        return chr(b)
    return f"x{b:02x}"


def load_trace(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    trace = obj.get("trace", [])
    if not isinstance(trace, list) or not trace:
        raise RuntimeError("generation_trace.json does not contain a non-empty trace")
    return trace


def segment_trace(trace: list[dict]) -> list[dict]:
    primitive_ids = [int(np.argmax(row["primitive_weights"])) for row in trace]
    segments = []
    start = 0

    for i in range(1, len(trace) + 1):
        if i == len(trace) or primitive_ids[i] != primitive_ids[i - 1]:
            rows = trace[start:i]
            prim = primitive_ids[start]
            state_norms = [float(r.get("state_norm", 0.0)) for r in rows]
            confidences = [float(max(r.get("primitive_weights", [0.0]))) for r in rows]
            entropies = [float(r.get("routing_entropy_norm", 0.0)) for r in rows]
            retrieval_hits = [1 if bool(r.get("retrieval_hit", False)) else 0 for r in rows]
            bytes_out = [int(r.get("chosen_byte", 0)) for r in rows]

            top10_first = rows[0].get("top10", []) or []
            chosen_prob_first = None
            for opt in top10_first:
                if int(opt.get("byte", -1)) == int(rows[0].get("chosen_byte", -2)):
                    chosen_prob_first = float(opt.get("prob", 0.0))
                    break

            top10_last = rows[-1].get("top10", []) or []
            chosen_prob_last = None
            for opt in top10_last:
                if int(opt.get("byte", -1)) == int(rows[-1].get("chosen_byte", -2)):
                    chosen_prob_last = float(opt.get("prob", 0.0))
                    break

            seg = {
                "segment_id": len(segments),
                "primitive": int(prim),
                "start_step": int(rows[0]["step"]),
                "end_step": int(rows[-1]["step"]),
                "duration": int(len(rows)),
                "start_state_norm": float(state_norms[0]),
                "end_state_norm": float(state_norms[-1]),
                "delta_state_norm": float(state_norms[-1] - state_norms[0]),
                "mean_state_norm": float(np.mean(state_norms)),
                "start_entropy": float(entropies[0]),
                "end_entropy": float(entropies[-1]),
                "entropy_drop": float(entropies[0] - entropies[-1]),
                "mean_entropy": float(np.mean(entropies)),
                "avg_confidence": float(np.mean(confidences)),
                "max_confidence": float(np.max(confidences)),
                "retrieval_hits": int(sum(retrieval_hits)),
                "retrieval_hit_rate": float(np.mean(retrieval_hits)),
                "bytes": bytes_out,
                "byte_labels": [byte_label(b) for b in bytes_out],
                "text_preview": " ".join(byte_label(b) for b in bytes_out[:24]),
                "chosen_prob_first": chosen_prob_first,
                "chosen_prob_last": chosen_prob_last,
                "next_primitive": None,
            }
            segments.append(seg)
            start = i

    for i in range(len(segments) - 1):
        segments[i]["next_primitive"] = int(segments[i + 1]["primitive"])

    return segments


def transition_summary(segments: list[dict]) -> dict:
    edge_counts: Counter[tuple[int, int]] = Counter()
    node_counts: Counter[int] = Counter()
    durations_by_prim: defaultdict[int, list[int]] = defaultdict(list)
    retrieval_by_prim: defaultdict[int, list[int]] = defaultdict(list)
    entropy_drop_by_prim: defaultdict[int, list[float]] = defaultdict(list)

    for seg in segments:
        p = int(seg["primitive"])
        node_counts[p] += 1
        durations_by_prim[p].append(int(seg["duration"]))
        retrieval_by_prim[p].append(int(seg["retrieval_hits"]))
        entropy_drop_by_prim[p].append(float(seg["entropy_drop"]))
        if seg["next_primitive"] is not None:
            edge_counts[(p, int(seg["next_primitive"]))] += 1

    program_types = []
    for p in sorted(node_counts):
        program_types.append(
            {
                "primitive": int(p),
                "segment_count": int(node_counts[p]),
                "mean_duration": float(np.mean(durations_by_prim[p])),
                "max_duration": int(np.max(durations_by_prim[p])),
                "mean_retrieval_hits": float(np.mean(retrieval_by_prim[p])),
                "mean_entropy_drop": float(np.mean(entropy_drop_by_prim[p])),
            }
        )

    transitions = [
        {"src": int(src), "dst": int(dst), "count": int(count)}
        for (src, dst), count in sorted(edge_counts.items())
    ]

    return {
        "program_types": program_types,
        "transitions": transitions,
    }


def save_plots(segments: list[dict], transitions: dict) -> None:
    x = np.arange(len(segments))
    primitives = np.array([int(s["primitive"]) for s in segments])
    durations = np.array([int(s["duration"]) for s in segments])
    start_norms = np.array([float(s["start_state_norm"]) for s in segments])
    end_norms = np.array([float(s["end_state_norm"]) for s in segments])
    confidences = np.array([float(s["avg_confidence"]) for s in segments])
    retrieval_hits = np.array([int(s["retrieval_hits"]) for s in segments])
    entropy_drop = np.array([float(s["entropy_drop"]) for s in segments])

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, constrained_layout=True)

    scatter = axes[0].scatter(x, primitives, c=confidences, s=80 + durations * 25, cmap="viridis")
    axes[0].plot(x, primitives, color="gray", alpha=0.4)
    axes[0].set_ylabel("Program primitive")
    axes[0].set_title("Program lifetimes (segment = contiguous argmax-primitive run)")
    cbar = fig.colorbar(scatter, ax=axes[0], pad=0.01)
    cbar.set_label("Avg confidence")

    axes[1].bar(x, durations, color="tab:blue", alpha=0.75, label="duration")
    axes[1].set_ylabel("Duration")
    axes[1].legend(loc="upper right")

    axes[2].plot(x, start_norms, label="start norm", color="tab:orange")
    axes[2].plot(x, end_norms, label="end norm", color="tab:red")
    axes[2].scatter(x, end_norms, s=25 + retrieval_hits * 30, c=np.where(retrieval_hits > 0, "lime", "black"), alpha=0.8)
    axes[2].set_ylabel("State norm")
    axes[2].legend(loc="upper right")

    axes[3].bar(x, entropy_drop, color=np.where(retrieval_hits > 0, "tab:green", "tab:gray"), alpha=0.8)
    axes[3].set_ylabel("Entropy drop")
    axes[3].set_xlabel("Program segment index")

    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    G = nx.DiGraph()
    for program in transitions["program_types"]:
        p = int(program["primitive"])
        G.add_node(
            p,
            segment_count=int(program["segment_count"]),
            mean_duration=float(program["mean_duration"]),
        )
    for edge in transitions["transitions"]:
        G.add_edge(int(edge["src"]), int(edge["dst"]), weight=int(edge["count"]))

    pos = nx.spring_layout(G, seed=7)
    plt.figure(figsize=(9, 7))
    node_sizes = [700 + 180 * G.nodes[n]["segment_count"] for n in G.nodes]
    node_colors = [G.nodes[n]["mean_duration"] for n in G.nodes]
    edges = list(G.edges(data=True))
    widths = [1.0 + 1.5 * e[2]["weight"] for e in edges]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.plasma)
    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: f"P{n}\nseg={G.nodes[n]['segment_count']}\nmean_dur={G.nodes[n]['mean_duration']:.1f}" for n in G.nodes},
        font_size=9,
    )
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowstyle="-|>", arrowsize=18, edge_color="gray")
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels={(u, v): d["weight"] for u, v, d in edges},
        font_size=9,
    )
    plt.title("Program transition graph")
    plt.axis("off")
    plt.savefig(GRAPH_PATH, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    trace = load_trace(TRACE_PATH)
    segments = segment_trace(trace)
    transitions = transition_summary(segments)

    summary = {
        "segment_count": int(len(segments)),
        "program_types": transitions["program_types"],
        "transitions": transitions["transitions"],
        "segments": segments,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_plots(segments, transitions)

    print(f"Saved: {SUMMARY_PATH}")
    print(f"Saved: {PLOT_PATH}")
    print(f"Saved: {GRAPH_PATH}")
    print("Program types:")
    for p in summary["program_types"]:
        print(
            f"  P{p['primitive']}: segments={p['segment_count']} mean_duration={p['mean_duration']:.2f} "
            f"mean_retrieval_hits={p['mean_retrieval_hits']:.2f} mean_entropy_drop={p['mean_entropy_drop']:.4f}"
        )


if __name__ == "__main__":
    main()
