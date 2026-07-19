from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_species_ablations import select_indices, standardize
from segment_species_analysis import kmeans

BASE = Path(__file__).with_name("segment_species_1000")
FEATURES_PATH = BASE / "segment_features.npz"
SEGMENTS_PATH = BASE / "segments_dataset.jsonl"
OUTPUT_PATH = BASE / "discovered_species_profiles.json"


def load_segments() -> list[dict]:
    fields = {
        "segment_id", "prompt_id", "primitive", "previous_primitive", "next_primitive", "duration",
        "text", "byte_labels", "state_drift", "state_norm_delta", "entropy_drift", "mean_entropy",
        "mean_confidence", "retrieval_hits", "retrieval_hit_rate", "mean_retrieval_norm",
        "mean_chosen_probability", "byte_class_histogram",
    }
    out = []
    with open(SEGMENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append({k: obj.get(k) for k in fields})
    return out


def profile_mode(mode: str, k: int, seed: int, raw: np.ndarray, names: list[str], segments: list[dict]) -> dict:
    indices = select_indices(names, mode)
    X = standardize(raw[:, indices])
    _, labels, centers = kmeans(X, k, seed, restarts=12)
    profiles = []

    for cluster in range(k):
        idx = np.where(labels == cluster)[0]
        members = [segments[i] for i in idx]
        distances = np.linalg.norm(X[idx] - centers[cluster], axis=1)
        exemplar_idx = idx[np.argsort(distances)[:12]]
        primitive_counts = Counter(int(s["primitive"]) for s in members)
        previous_counts = Counter("none" if s["previous_primitive"] is None else str(s["previous_primitive"]) for s in members)
        next_counts = Counter("none" if s["next_primitive"] is None else str(s["next_primitive"]) for s in members)
        byte_hist = np.mean(np.asarray([s["byte_class_histogram"] for s in members], dtype=np.float32), axis=0)
        profiles.append({
            "species": cluster,
            "count": len(members),
            "primitive_distribution": {str(a): int(b) for a, b in primitive_counts.most_common()},
            "previous_primitive_distribution": dict(previous_counts.most_common()),
            "next_primitive_distribution": dict(next_counts.most_common()),
            "mean_duration": float(np.mean([s["duration"] for s in members])),
            "mean_state_drift": float(np.mean([s["state_drift"] for s in members])),
            "mean_state_norm_delta": float(np.mean([s["state_norm_delta"] for s in members])),
            "mean_entropy_drift": float(np.mean([s["entropy_drift"] for s in members])),
            "mean_entropy": float(np.mean([s["mean_entropy"] for s in members])),
            "mean_confidence": float(np.mean([s["mean_confidence"] for s in members])),
            "mean_retrieval_hits": float(np.mean([s["retrieval_hits"] for s in members])),
            "mean_retrieval_hit_rate": float(np.mean([s["retrieval_hit_rate"] for s in members])),
            "mean_retrieval_norm": float(np.mean([s["mean_retrieval_norm"] for s in members])),
            "mean_chosen_probability": float(np.mean([s["mean_chosen_probability"] for s in members])),
            "byte_class_distribution": [float(x) for x in byte_hist],
            "exemplars": [
                {
                    "segment_id": int(segments[i]["segment_id"]),
                    "prompt_id": int(segments[i]["prompt_id"]),
                    "primitive": int(segments[i]["primitive"]),
                    "duration": int(segments[i]["duration"]),
                    "text": segments[i]["text"],
                    "byte_labels": segments[i]["byte_labels"],
                }
                for i in exemplar_idx
            ],
        })

    return {
        "mode": mode,
        "k": k,
        "features": [names[i] for i in indices],
        "profiles": profiles,
    }


def main() -> None:
    data = np.load(FEATURES_PATH)
    raw = data["raw"]
    names = [str(x) for x in data["feature_names"].tolist()]
    segments = load_segments()
    report = {
        "byte_class_order": ["whitespace", "digit", "letter", "punctuation", "reserved_tag", "control", "high_byte"],
        "modes": [
            profile_mode("no_routing_context", 4, 211, raw, names, segments),
            profile_mode("control_only", 4, 212, raw, names, segments),
            profile_mode("symbol_only", 5, 213, raw, names, segments),
        ],
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved={OUTPUT_PATH}")
    for mode in report["modes"]:
        print(mode["mode"])
        for p in mode["profiles"]:
            print(
                f"  S{p['species']} n={p['count']} prim={p['primitive_distribution']} duration={p['mean_duration']:.2f} "
                f"retrieval={p['mean_retrieval_hit_rate']:.3f} entropy={p['mean_entropy']:.3f}"
            )


if __name__ == "__main__":
    main()
