from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(ROOT))

from raplib.model import ModelConfig, PrimitiveManifoldLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints1" / "rap_v0_step009000.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "natural_reasoning_train.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("segment_species"))
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--max-gen", type=int, default=64)
    parser.add_argument("--max-prompt-bytes", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def load_model(path: Path, device: str) -> PrimitiveManifoldLM:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "cfg" in checkpoint:
        raw_cfg = checkpoint["cfg"]
        valid = ModelConfig.__annotations__.keys()
        cfg = ModelConfig(**{k: v for k, v in raw_cfg.items() if k in valid})
        state_dict = checkpoint["state_dict"]
    else:
        cfg = ModelConfig()
        state_dict = checkpoint
    model = PrimitiveManifoldLM(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def iter_prompts(path: Path, limit: int, max_prompt_bytes: int):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if count >= limit:
                return
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = str(obj.get("question", "")).strip()
            if not question:
                continue
            text = f"Question: {question}\n\nAnswer:"
            raw = text.encode("utf-8")
            if max_prompt_bytes > 0 and len(raw) > max_prompt_bytes:
                raw = raw[-max_prompt_bytes:]
            yield count, line_no, question, raw
            count += 1


def byte_class(b: int) -> int:
    if b in (9, 10, 13, 32):
        return 0
    if 48 <= b <= 57:
        return 1
    if 65 <= b <= 90 or 97 <= b <= 122:
        return 2
    if 33 <= b <= 126:
        return 3
    if b in (128, 129, 130):
        return 4
    if b < 32 or b == 127:
        return 5
    return 6


def byte_label(b: int) -> str:
    if b == 32:
        return "sp"
    if b == 10:
        return "\\n"
    if b == 9:
        return "\\t"
    if 32 <= b <= 126:
        return chr(b)
    return f"x{b:02x}"


def retrieval_for_token(model, tok, state, last_pos, state_hist, tpos):
    retrieval = torch.zeros_like(state)
    details = {"hit": False, "count": 0, "mean_distance": None, "retrieved_bytes": []}
    if tpos <= 0 or last_pos is None or state_hist is None:
        return retrieval, details
    is_reserved = (tok == 128) | (tok == 129) | (tok == 130)
    if bool(is_reserved.any().item()):
        return retrieval, details

    sources = []
    if bool(getattr(model, "path_retrieval_enabled", False)):
        neigh = model.path_neigh_lut[tok]
        dist = model.path_dist_lut[tok].to(dtype=state.dtype)
        pos = last_pos.gather(1, neigh.clamp(min=0))
        valid = (neigh >= 0) & (pos >= 0)
        if bool(valid.any().item()):
            got = torch.gather(state_hist, 1, pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, state_hist.shape[-1]))
            alpha = torch.tensor(float(getattr(model, "path_alpha", 1.0)), device=state.device, dtype=state.dtype)
            weights = torch.exp(-alpha * dist) * valid.to(state.dtype)
            value = (got * weights.unsqueeze(-1)).sum(1) / torch.clamp(weights.sum(1, keepdim=True), min=1.0)
            sources.append(value)
            details["count"] += int(valid.sum().item())
            details["mean_distance"] = float(((weights * dist).sum() / torch.clamp(weights.sum(), min=1e-6)).item())
            details["retrieved_bytes"].extend([int(x) for x in neigh[valid].detach().cpu().tolist()])

    if bool(getattr(model, "word_map_retrieval_enabled", False)):
        neigh = model.word_neigh_lut[tok]
        weights0 = model.word_w_lut[tok].to(dtype=state.dtype)
        pos = last_pos.gather(1, neigh.clamp(min=0))
        valid = (neigh >= 0) & (pos >= 0)
        if bool(valid.any().item()):
            got = torch.gather(state_hist, 1, pos.clamp(min=0).unsqueeze(-1).expand(-1, -1, state_hist.shape[-1]))
            weights = weights0 * valid.to(state.dtype)
            value = (got * weights.unsqueeze(-1)).sum(1) / torch.clamp(weights.sum(1, keepdim=True), min=1e-6)
            sources.append(value)
            details["count"] += int(valid.sum().item())
            details["retrieved_bytes"].extend([int(x) for x in neigh[valid].detach().cpu().tolist()])

    if bool(getattr(model, "symmetry_retrieval_enabled", False)):
        inv = tok ^ 0xFF
        rev = model.bit_reverse_lut[tok]
        values = []
        for candidate in (inv, rev):
            pos = last_pos.gather(1, candidate.unsqueeze(1)).squeeze(1)
            if int(pos.item()) >= 0:
                values.append(state_hist[0, int(pos.item())].unsqueeze(0))
                details["count"] += 1
                details["retrieved_bytes"].append(int(candidate.item()))
        if values:
            sources.append(torch.stack(values, dim=0).mean(dim=0))

    if sources:
        retrieval = torch.stack(sources, dim=0).mean(dim=0)
        details["hit"] = True
    return retrieval, details


def route_metrics(model, tok, state_before, retrieval):
    x_emb = model._get_emb(tok)
    if bool(getattr(model, "retrieval_enabled", False)):
        ctrl_in = torch.cat([x_emb, state_before, retrieval], dim=-1)
    else:
        ctrl_in = torch.cat([x_emb, state_before], dim=-1)
    weights = torch.softmax(model.controller(ctrl_in), dim=-1)
    entropy = -(weights * (weights + 1e-12).log()).sum(-1)
    entropy_norm = entropy / math.log(model.cfg.n_primitives)
    return x_emb, ctrl_in, weights, float(entropy_norm.item())


def trace_prompt(model, prompt_bytes, max_gen, temperature, top_k, repetition_penalty, repetition_window, greedy, generator):
    device = next(model.parameters()).device
    state = model.init_state(1, device)
    retrieval_enabled = bool(getattr(model, "retrieval_enabled", False))
    capacity = len(prompt_bytes) + max_gen + 2
    last_pos = torch.full((1, 256), -1, device=device, dtype=torch.long) if retrieval_enabled else None
    state_hist = torch.zeros((1, capacity, model.cfg.d_state), device=device, dtype=state.dtype) if retrieval_enabled else None
    tpos = 0
    next_logits = torch.zeros((1, 256), device=device, dtype=state.dtype)

    with torch.inference_mode():
        for b in prompt_bytes:
            tok = torch.tensor([int(b)], device=device, dtype=torch.long)
            retrieval, _ = retrieval_for_token(model, tok, state, last_pos, state_hist, tpos) if retrieval_enabled else (None, {})
            next_logits, state = model.step(tok, state, retrieval=retrieval)
            if retrieval_enabled:
                state_hist[:, tpos] = state
                last_pos.scatter_(1, tok.unsqueeze(1), torch.full((1, 1), tpos, device=device, dtype=torch.long))
            tpos += 1

        rows = []
        generated = []
        for step in range(max_gen):
            scored = next_logits.clone()
            if repetition_penalty > 1.0:
                for token_id in set(generated[-repetition_window:]):
                    scored[:, token_id] = scored[:, token_id] / repetition_penalty
            scored = scored / max(temperature, 1e-6)
            if top_k > 0:
                values, _ = torch.topk(scored, min(top_k, scored.shape[-1]))
                scored[scored < values[:, [-1]]] = -float("inf")
            probs = torch.softmax(scored, dim=-1)
            if greedy:
                chosen = int(torch.argmax(probs, dim=-1).item())
            else:
                chosen = int(torch.multinomial(probs, 1, generator=generator).item())
            if chosen == 10:
                break

            tok = torch.tensor([chosen], device=device, dtype=torch.long)
            state_before = state.clone()
            retrieval, retrieval_details = retrieval_for_token(model, tok, state, last_pos, state_hist, tpos) if retrieval_enabled else (None, {"hit": False, "count": 0, "mean_distance": None, "retrieved_bytes": []})
            retrieval_for_route = retrieval if retrieval is not None else torch.zeros_like(state)
            _, _, weights, route_entropy = route_metrics(model, tok, state_before, retrieval_for_route)
            next_logits, state = model.step(tok, state, retrieval=retrieval)

            rows.append({
                "step": step,
                "byte": chosen,
                "chosen_probability": float(probs[0, chosen].item()),
                "primitive": int(torch.argmax(weights, dim=-1).item()),
                "confidence": float(torch.max(weights).item()),
                "routing_entropy": route_entropy,
                "retrieval_hit": bool(retrieval_details["hit"]),
                "retrieval_count": int(retrieval_details["count"]),
                "retrieval_mean_distance": retrieval_details["mean_distance"],
                "retrieved_bytes": retrieval_details["retrieved_bytes"],
                "retrieval_norm": float(torch.linalg.norm(retrieval_for_route).item()),
                "state_before": state_before[0].float().cpu().numpy(),
                "state_after": state[0].float().cpu().numpy(),
            })
            generated.append(chosen)
            if retrieval_enabled:
                state_hist[:, tpos] = state
                last_pos.scatter_(1, tok.unsqueeze(1), torch.full((1, 1), tpos, device=device, dtype=torch.long))
            tpos += 1

    return rows, generated


def make_segments(prompt_id, line_no, question, rows, generated):
    if not rows:
        return []
    segments = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["primitive"] != rows[i - 1]["primitive"]:
            part = rows[start:i]
            bytes_out = [r["byte"] for r in part]
            state_start = part[0]["state_before"]
            state_end = part[-1]["state_after"]
            delta = state_end - state_start
            delta_norm = float(np.linalg.norm(delta))
            direction = delta / max(delta_norm, 1e-8)
            hist = np.zeros(7, dtype=np.float32)
            for b in bytes_out:
                hist[byte_class(b)] += 1.0
            hist /= max(1, len(bytes_out))
            hit_distances = [r["retrieval_mean_distance"] for r in part if r["retrieval_mean_distance"] is not None]
            seg = {
                "segment_id": -1,
                "prompt_id": prompt_id,
                "dataset_line": line_no,
                "question": question,
                "primitive": int(part[0]["primitive"]),
                "previous_primitive": None,
                "next_primitive": None,
                "start_step": int(part[0]["step"]),
                "end_step": int(part[-1]["step"]),
                "duration": len(part),
                "bytes": bytes_out,
                "byte_labels": [byte_label(b) for b in bytes_out],
                "text": bytes(bytes_out).decode("utf-8", errors="replace"),
                "start_state_norm": float(np.linalg.norm(state_start)),
                "end_state_norm": float(np.linalg.norm(state_end)),
                "state_drift": delta_norm,
                "state_norm_delta": float(np.linalg.norm(state_end) - np.linalg.norm(state_start)),
                "state_direction": direction.astype(np.float32),
                "start_entropy": float(part[0]["routing_entropy"]),
                "end_entropy": float(part[-1]["routing_entropy"]),
                "entropy_drift": float(part[-1]["routing_entropy"] - part[0]["routing_entropy"]),
                "mean_entropy": float(np.mean([r["routing_entropy"] for r in part])),
                "mean_confidence": float(np.mean([r["confidence"] for r in part])),
                "min_confidence": float(np.min([r["confidence"] for r in part])),
                "retrieval_hits": int(sum(r["retrieval_hit"] for r in part)),
                "retrieval_count": int(sum(r["retrieval_count"] for r in part)),
                "retrieval_hit_rate": float(np.mean([r["retrieval_hit"] for r in part])),
                "mean_retrieval_norm": float(np.mean([r["retrieval_norm"] for r in part])),
                "mean_graph_distance": float(np.mean(hit_distances)) if hit_distances else None,
                "mean_chosen_probability": float(np.mean([r["chosen_probability"] for r in part])),
                "byte_class_histogram": hist,
            }
            segments.append(seg)
            start = i

    for i, seg in enumerate(segments):
        if i > 0:
            seg["previous_primitive"] = int(segments[i - 1]["primitive"])
        if i + 1 < len(segments):
            seg["next_primitive"] = int(segments[i + 1]["primitive"])
    return segments


def build_features(segments, projection_dim, seed, n_primitives):
    rng = np.random.default_rng(seed)
    state_dim = len(segments[0]["state_direction"])
    projection = rng.normal(size=(state_dim, projection_dim)).astype(np.float32) / math.sqrt(projection_dim)
    names = [
        "log_duration", "state_drift", "state_norm_delta", "entropy_drift", "mean_entropy",
        "mean_confidence", "min_confidence", "retrieval_hits", "retrieval_count", "retrieval_hit_rate",
        "mean_retrieval_norm", "mean_graph_distance", "mean_chosen_probability",
    ]
    names += [f"byte_class_{i}" for i in range(7)]
    names += [f"state_direction_rp_{i}" for i in range(projection_dim)]
    names += [f"previous_primitive_{i}" for i in range(n_primitives + 1)]
    names += [f"next_primitive_{i}" for i in range(n_primitives + 1)]
    rows = []
    for seg in segments:
        base = [
            math.log1p(seg["duration"]), seg["state_drift"], seg["state_norm_delta"], seg["entropy_drift"],
            seg["mean_entropy"], seg["mean_confidence"], seg["min_confidence"], seg["retrieval_hits"],
            seg["retrieval_count"], seg["retrieval_hit_rate"], seg["mean_retrieval_norm"],
            0.0 if seg["mean_graph_distance"] is None else seg["mean_graph_distance"], seg["mean_chosen_probability"],
        ]
        base.extend(seg["byte_class_histogram"].tolist())
        base.extend((seg["state_direction"] @ projection).tolist())
        prev = np.zeros(n_primitives + 1, dtype=np.float32)
        nxt = np.zeros(n_primitives + 1, dtype=np.float32)
        prev[n_primitives if seg["previous_primitive"] is None else seg["previous_primitive"]] = 1.0
        nxt[n_primitives if seg["next_primitive"] is None else seg["next_primitive"]] = 1.0
        base.extend(prev.tolist())
        base.extend(nxt.tolist())
        rows.append(base)
    X = np.asarray(rows, dtype=np.float32)
    return X, names, projection


def standardize(X):
    mean = X.mean(0)
    std = X.std(0)
    std[std < 1e-6] = 1.0
    return (X - mean) / std, mean, std


def pca(X, dims=2):
    centered = X - X.mean(0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:dims].T
    variance = singular ** 2
    ratio = variance[:dims] / max(float(variance.sum()), 1e-12)
    return coords, vt[:dims], ratio


def kmeans(X, k, seed, restarts=8, max_iter=200):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        centers = [X[rng.integers(len(X))]]
        while len(centers) < k:
            d2 = np.min(((X[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(2), axis=1)
            total = d2.sum()
            idx = rng.integers(len(X)) if total <= 0 else rng.choice(len(X), p=d2 / total)
            centers.append(X[idx])
        centers = np.asarray(centers).copy()
        labels = np.zeros(len(X), dtype=np.int32)
        for _ in range(max_iter):
            new_labels = np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(2), axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            for j in range(k):
                members = X[labels == j]
                centers[j] = members.mean(0) if len(members) else X[rng.integers(len(X))]
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy(), centers.copy())
    return best


def silhouette_score(X, labels):
    unique = np.unique(labels)
    if len(unique) < 2 or len(X) > 20000:
        return 0.0
    sample_idx = np.arange(len(X))
    if len(X) > 4000:
        sample_idx = np.random.default_rng(1).choice(len(X), 4000, replace=False)
    scores = []
    for idx in sample_idx:
        same = np.where(labels == labels[idx])[0]
        same = same[same != idx]
        a = float(np.linalg.norm(X[same] - X[idx], axis=1).mean()) if len(same) else 0.0
        b = min(float(np.linalg.norm(X[labels == c] - X[idx], axis=1).mean()) for c in unique if c != labels[idx])
        scores.append((b - a) / max(a, b, 1e-8))
    return float(np.mean(scores))


def cluster_segments(X, min_k, max_k, seed):
    candidates = []
    max_k = min(max_k, max(2, len(X) - 1))
    for k in range(min_k, max_k + 1):
        inertia, labels, centers = kmeans(X, k, seed + k)
        score = silhouette_score(X, labels)
        candidates.append({"k": k, "inertia": inertia, "silhouette": score, "labels": labels, "centers": centers})
        print(f"k={k} silhouette={score:.4f} inertia={inertia:.1f}")
    return max(candidates, key=lambda row: row["silhouette"]), candidates


def json_safe_segment(seg):
    out = dict(seg)
    out["state_direction"] = [float(x) for x in seg["state_direction"]]
    out["byte_class_histogram"] = [float(x) for x in seg["byte_class_histogram"]]
    return out


def summarize_clusters(segments, labels, X, centers):
    summaries = []
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        members = [segments[i] for i in idx]
        distances = np.linalg.norm(X[idx] - centers[c], axis=1)
        exemplar_order = idx[np.argsort(distances)[:8]]
        primitive_counts = Counter(int(s["primitive"]) for s in members)
        byte_classes = np.mean([s["byte_class_histogram"] for s in members], axis=0)
        summaries.append({
            "cluster": int(c),
            "count": len(members),
            "primitive_distribution": {str(k): int(v) for k, v in sorted(primitive_counts.items())},
            "mean_duration": float(np.mean([s["duration"] for s in members])),
            "mean_state_drift": float(np.mean([s["state_drift"] for s in members])),
            "mean_state_norm_delta": float(np.mean([s["state_norm_delta"] for s in members])),
            "mean_entropy_drift": float(np.mean([s["entropy_drift"] for s in members])),
            "mean_confidence": float(np.mean([s["mean_confidence"] for s in members])),
            "mean_retrieval_hit_rate": float(np.mean([s["retrieval_hit_rate"] for s in members])),
            "mean_retrieval_norm": float(np.mean([s["mean_retrieval_norm"] for s in members])),
            "byte_class_distribution": [float(x) for x in byte_classes],
            "exemplars": [{
                "segment_id": int(segments[i]["segment_id"]),
                "prompt_id": int(segments[i]["prompt_id"]),
                "primitive": int(segments[i]["primitive"]),
                "duration": int(segments[i]["duration"]),
                "text": segments[i]["text"],
                "byte_labels": segments[i]["byte_labels"],
            } for i in exemplar_order],
        })
    return summaries


def save_visualizations(output, coords, labels, segments, cluster_summaries):
    output.mkdir(parents=True, exist_ok=True)
    primitives = np.array([s["primitive"] for s in segments])
    retrieval = np.array([s["retrieval_hit_rate"] for s in segments])
    duration = np.array([s["duration"] for s in segments])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    axes[0].scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=12 + duration * 2, alpha=0.7)
    axes[0].set_title("Behavioral computation species")
    axes[1].scatter(coords[:, 0], coords[:, 1], c=primitives, cmap="tab10", s=12 + duration * 2, alpha=0.7)
    axes[1].set_title("Same space colored by primitive")
    sc = axes[2].scatter(coords[:, 0], coords[:, 1], c=retrieval, cmap="viridis", s=12 + duration * 2, alpha=0.7)
    axes[2].set_title("Same space colored by retrieval hit rate")
    fig.colorbar(sc, ax=axes[2], label="retrieval hit rate")
    for ax in axes:
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    fig.savefig(output / "segment_species_pca.png", dpi=180)
    plt.close(fig)

    G = nx.DiGraph()
    for summary in cluster_summaries:
        c = summary["cluster"]
        G.add_node(c, count=summary["count"], duration=summary["mean_duration"])
    edge_counts = Counter()
    by_prompt = defaultdict(list)
    for seg, label in zip(segments, labels):
        by_prompt[seg["prompt_id"]].append((seg["start_step"], int(label)))
    for sequence in by_prompt.values():
        sequence.sort()
        for (_, a), (_, b) in zip(sequence, sequence[1:]):
            edge_counts[(a, b)] += 1
    for (a, b), count in edge_counts.items():
        G.add_edge(a, b, weight=count)

    pos = nx.spring_layout(G, seed=12, weight="weight")
    plt.figure(figsize=(12, 10))
    sizes = [300 + 4 * G.nodes[n]["count"] for n in G.nodes]
    colors = [G.nodes[n]["duration"] for n in G.nodes]
    widths = [0.5 + math.log1p(d["weight"]) for _, _, d in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, cmap=plt.cm.plasma)
    nx.draw_networkx_labels(G, pos, labels={n: f"S{n}\nn={G.nodes[n]['count']}" for n in G.nodes}, font_size=9)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.55, arrows=True, arrowsize=15)
    labels_edge = {(a, b): d["weight"] for a, b, d in G.edges(data=True) if d["weight"] >= max(2, len(segments) // 500)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_edge, font_size=7)
    plt.title("Discovered computation-species transition ecology")
    plt.axis("off")
    plt.savefig(output / "species_transition_graph.png", dpi=180, bbox_inches="tight")
    plt.close()
    return edge_counts


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)

    model = load_model(args.checkpoint, args.device)
    all_segments = []
    prompt_records = []

    for prompt_id, line_no, question, prompt_bytes in iter_prompts(args.dataset, args.num_prompts, args.max_prompt_bytes):
        rows, generated = trace_prompt(
            model, prompt_bytes, args.max_gen, args.temperature, args.top_k,
            args.repetition_penalty, args.repetition_window, args.greedy, generator,
        )
        segments = make_segments(prompt_id, line_no, question, rows, generated)
        for seg in segments:
            seg["segment_id"] = len(all_segments)
            all_segments.append(seg)
        prompt_records.append({
            "prompt_id": prompt_id,
            "dataset_line": line_no,
            "question": question,
            "generated_bytes": generated,
            "generated_text": bytes(generated).decode("utf-8", errors="replace"),
            "step_count": len(rows),
            "segment_count": len(segments),
        })
        if (prompt_id + 1) % args.progress_every == 0:
            print(f"prompts={prompt_id + 1} segments={len(all_segments)}", flush=True)

    if len(all_segments) < 3:
        raise RuntimeError("Not enough segments to cluster")

    X_raw, feature_names, projection = build_features(all_segments, args.projection_dim, args.seed, model.cfg.n_primitives)
    X, mean, std = standardize(X_raw)
    coords, pca_components, explained = pca(X, 2)
    best, candidates = cluster_segments(X, args.min_clusters, args.max_clusters, args.seed)
    labels = best["labels"]

    for seg, label in zip(all_segments, labels):
        seg["species"] = int(label)

    cluster_summaries = summarize_clusters(all_segments, labels, X, best["centers"])
    edge_counts = save_visualizations(args.output, coords, labels, all_segments, cluster_summaries)

    with open(args.output / "segments_dataset.jsonl", "w", encoding="utf-8") as f:
        for seg in all_segments:
            f.write(json.dumps(json_safe_segment(seg), ensure_ascii=False) + "\n")
    with open(args.output / "prompt_generations.jsonl", "w", encoding="utf-8") as f:
        for row in prompt_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    np.savez_compressed(
        args.output / "segment_features.npz",
        raw=X_raw,
        standardized=X,
        feature_names=np.asarray(feature_names),
        labels=labels,
        pca_coords=coords,
        pca_components=pca_components,
        standardization_mean=mean,
        standardization_std=std,
        state_projection=projection,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "num_prompts": len(prompt_records),
        "num_segments": len(all_segments),
        "generation": {
            "max_gen": args.max_gen,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "greedy": args.greedy,
            "seed": args.seed,
        },
        "clustering": {
            "selected_k": int(best["k"]),
            "selected_silhouette": float(best["silhouette"]),
            "candidates": [{"k": int(c["k"]), "inertia": float(c["inertia"]), "silhouette": float(c["silhouette"])} for c in candidates],
            "current_primitive_excluded_from_features": True,
            "feature_names": feature_names,
            "pca_explained_variance_ratio": [float(x) for x in explained],
        },
        "species": cluster_summaries,
        "species_transitions": [{"src": int(a), "dst": int(b), "count": int(n)} for (a, b), n in edge_counts.most_common()],
        "primitive_segment_counts": {str(k): int(v) for k, v in Counter(s["primitive"] for s in all_segments).items()},
    }
    with open(args.output / "species_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"selected_k={best['k']} silhouette={best['silhouette']:.4f}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
