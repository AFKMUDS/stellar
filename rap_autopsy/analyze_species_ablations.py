from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from segment_species_analysis import cluster_segments, pca

BASE = Path(__file__).with_name("segment_species_1000")
FEATURES_PATH = BASE / "segment_features.npz"
SEGMENTS_PATH = BASE / "segments_dataset.jsonl"
OUTPUT_PATH = BASE / "feature_ablation_report.json"
PLOT_PATH = BASE / "feature_ablation_pca.png"


def select_indices(names: list[str], mode: str) -> list[int]:
    if mode == "full":
        return list(range(len(names)))
    if mode == "no_routing_context":
        return [i for i, n in enumerate(names) if not n.startswith("previous_primitive_") and not n.startswith("next_primitive_")]
    if mode == "dynamics_only":
        prefixes = (
            "log_duration", "state_drift", "state_norm_delta", "entropy_drift", "mean_entropy",
            "mean_confidence", "min_confidence", "retrieval_hits", "retrieval_count", "retrieval_hit_rate",
            "mean_retrieval_norm", "mean_graph_distance", "state_direction_rp_",
        )
        return [i for i, n in enumerate(names) if n.startswith(prefixes)]
    if mode == "state_only":
        prefixes = ("state_drift", "state_norm_delta", "state_direction_rp_")
        return [i for i, n in enumerate(names) if n.startswith(prefixes)]
    if mode == "control_only":
        prefixes = (
            "log_duration", "entropy_drift", "mean_entropy", "mean_confidence", "min_confidence",
            "retrieval_hits", "retrieval_count", "retrieval_hit_rate", "mean_retrieval_norm", "mean_graph_distance",
        )
        return [i for i, n in enumerate(names) if n.startswith(prefixes)]
    if mode == "symbol_only":
        return [i for i, n in enumerate(names) if n.startswith("byte_class_") or n == "mean_chosen_probability"]
    raise ValueError(mode)


def standardize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(0)
    std = X.std(0)
    keep = std >= 1e-6
    X = X[:, keep]
    mean = mean[keep]
    std = std[keep]
    return (X - mean) / std


def normalized_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    a_values, a_inv = np.unique(a, return_inverse=True)
    b_values, b_inv = np.unique(b, return_inverse=True)
    joint = np.zeros((len(a_values), len(b_values)), dtype=np.float64)
    for x, y in zip(a_inv, b_inv):
        joint[x, y] += 1.0
    joint /= joint.sum()
    pa = joint.sum(1)
    pb = joint.sum(0)
    mi = 0.0
    for i in range(len(pa)):
        for j in range(len(pb)):
            if joint[i, j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (pa[i] * pb[j]))
    ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    return float(mi / max(np.sqrt(ha * hb), 1e-12))


def main() -> None:
    data = np.load(FEATURES_PATH)
    raw = data["raw"]
    names = [str(x) for x in data["feature_names"].tolist()]
    primitives = []
    with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            primitives.append(int(obj["primitive"]))
    primitives = np.asarray(primitives, dtype=np.int32)

    modes = ["full", "no_routing_context", "dynamics_only", "state_only", "control_only", "symbol_only"]
    results = []
    plot_rows = []

    for mode_index, mode in enumerate(modes):
        indices = select_indices(names, mode)
        X = standardize(raw[:, indices])
        best, candidates = cluster_segments(X, 2, 8, 100 + mode_index)
        coords, _, explained = pca(X, 2)
        labels = best["labels"]
        nmi = normalized_mutual_information(labels, primitives)
        result = {
            "mode": mode,
            "feature_count": int(X.shape[1]),
            "selected_k": int(best["k"]),
            "silhouette": float(best["silhouette"]),
            "nmi_with_primitive": nmi,
            "pca_explained_variance_ratio": [float(x) for x in explained],
            "cluster_sizes": {str(c): int(np.sum(labels == c)) for c in np.unique(labels)},
            "candidates": [
                {"k": int(c["k"]), "silhouette": float(c["silhouette"]), "inertia": float(c["inertia"])}
                for c in candidates
            ],
            "features": [names[i] for i in indices],
        }
        results.append(result)
        plot_rows.append((mode, coords, labels, best["silhouette"], nmi))
        print(f"{mode}: k={best['k']} silhouette={best['silhouette']:.4f} nmi_primitive={nmi:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for ax, (mode, coords, labels, score, nmi) in zip(axes.flat, plot_rows):
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=5, alpha=0.45)
        ax.set_title(f"{mode}\nk={len(np.unique(labels))} sil={score:.3f} NMIprim={nmi:.3f}")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    report = {
        "purpose": "Test whether computation species persist after removing primitive-context and symbol features.",
        "segment_count": int(len(raw)),
        "results": results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"saved={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
