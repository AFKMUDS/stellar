from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from raplib.data import iter_qa_samples, iter_raw_byte_sequences
from raplib.pathfinder import build_k_hop_tables


DEFAULT_CHARSET = " \nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"-()[]{}+/\\@#$%^&*_=`~<>|"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize an English-focused byte codec against RAP bitwise move costs")
    p.add_argument("--data", type=str, default="natural_reasoning_train.jsonl")
    p.add_argument("--mode", type=str, default="jsonl", choices=["jsonl", "bin"])
    p.add_argument("--charset", type=str, default=DEFAULT_CHARSET)
    p.add_argument("--max_samples", type=int, default=50000)
    p.add_argument("--max_total_bytes", type=int, default=4096)
    p.add_argument("--move_weight", type=float, default=1.0)
    p.add_argument("--fallback_cost", type=float, default=8.0)
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_chars", type=int, default=0)
    p.add_argument("--reserved_bytes", type=str, default="128,129,130")
    p.add_argument("--out_json", type=str, default="codec_search_result.json")
    p.add_argument("--sample_text", type=str, default="Question:\nWhat is gravity?\n\nAnswer:\nGravity is the attraction between masses.")
    return p.parse_args()


def parse_reserved(text: str) -> Tuple[int, ...]:
    vals: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part) & 0xFF)
    return tuple(sorted(set(vals)))


def iter_text_sequences(data: str, mode: str, max_total_bytes: int) -> Iterable[str]:
    if mode == "bin":
        for seq_bytes, _ in iter_raw_byte_sequences(data, max_total_bytes=max_total_bytes):
            yield seq_bytes.decode("utf-8", errors="replace")
        return
    for sample in iter_qa_samples(data, include_answer_prefix=True):
        text = (sample.prompt_bytes + sample.target_bytes).decode("utf-8", errors="replace")
        yield text


def collect_transition_stats(
    data: str,
    mode: str,
    allowed_chars: Sequence[str],
    max_samples: int,
    max_total_bytes: int,
) -> Tuple[List[str], Counter[str], Counter[Tuple[str, str]]]:
    allowed = set(allowed_chars)
    char_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    seen = 0
    for text in iter_text_sequences(data, mode, max_total_bytes):
        filtered = [ch for ch in text if ch in allowed]
        if not filtered:
            continue
        seen += 1
        char_counts.update(filtered)
        pair_counts.update(zip(filtered, filtered[1:]))
        if seen >= int(max_samples):
            break
    chars = [ch for ch, _count in char_counts.most_common()]
    return chars, char_counts, pair_counts


def build_move_distance_matrix(reserved: Sequence[int], fallback_cost: float) -> List[List[float]]:
    neigh, dist = build_k_hop_tables(k=4, reserved=reserved)
    neigh_np = neigh.cpu().tolist()
    dist_np = dist.cpu().tolist()
    n = 256
    matrix = [[float(fallback_cost) for _ in range(n)] for _ in range(n)]
    for src in range(n):
        matrix[src][src] = 0.0
        frontier = [src]
        seen = {src: 0.0}
        while frontier:
            cur = frontier.pop(0)
            row = neigh_np[cur]
            drow = dist_np[cur]
            for nb, step_d in zip(row, drow):
                nb = int(nb)
                if nb < 0:
                    continue
                nd = seen[cur] + float(step_d)
                if nb not in seen or nd < seen[nb]:
                    seen[nb] = nd
                    frontier.append(nb)
        for dst, val in seen.items():
            matrix[src][dst] = min(matrix[src][dst], float(val))
    return matrix


def ascii_initial_mapping(chars: Sequence[str], available_bytes: Sequence[int]) -> Dict[str, int]:
    remaining = set(int(b) for b in available_bytes)
    mapping: Dict[str, int] = {}
    for ch in chars:
        code = ord(ch)
        if code in remaining:
            mapping[ch] = code
            remaining.remove(code)
    for ch in chars:
        if ch in mapping:
            continue
        chosen = min(remaining)
        mapping[ch] = int(chosen)
        remaining.remove(chosen)
    return mapping


def greedy_frequency_seed(
    chars: Sequence[str],
    char_counts: Counter[str],
    pair_counts: Counter[Tuple[str, str]],
    move_dist: Sequence[Sequence[float]],
    available_bytes: Sequence[int],
) -> Dict[str, int]:
    ordered_chars = sorted(chars, key=lambda ch: (-char_counts[ch], ch))
    centrality = Counter[str]()
    for (a, b), cnt in pair_counts.items():
        centrality[a] += cnt
        centrality[b] += cnt
    byte_scores = []
    for b in available_bytes:
        score = 0.0
        for other in available_bytes:
            score += move_dist[b][other]
        byte_scores.append((score, int(b)))
    byte_scores.sort(key=lambda item: (item[0], item[1]))
    mapping: Dict[str, int] = {}
    for ch, (_score, b) in zip(ordered_chars, byte_scores):
        mapping[ch] = int(b)
    return mapping


def mapping_cost(
    chars: Sequence[str],
    pair_counts: Counter[Tuple[str, str]],
    mapping: Dict[str, int],
    move_dist: Sequence[Sequence[float]],
    move_weight: float,
) -> float:
    total = 0.0
    for (a, b), cnt in pair_counts.items():
        total += float(cnt) * float(move_weight) * float(move_dist[mapping[a]][mapping[b]])
    return total


def optimize_mapping(
    chars: Sequence[str],
    pair_counts: Counter[Tuple[str, str]],
    start_mapping: Dict[str, int],
    move_dist: Sequence[Sequence[float]],
    move_weight: float,
    iterations: int,
    seed: int,
) -> Tuple[Dict[str, int], float, List[Dict[str, float]]]:
    rng = random.Random(seed)
    current = dict(start_mapping)
    best = dict(current)
    current_cost = mapping_cost(chars, pair_counts, current, move_dist, move_weight)
    best_cost = current_cost
    history: List[Dict[str, float]] = [{"iter": 0.0, "cost": float(current_cost)}]
    if len(chars) < 2:
        return best, best_cost, history
    char_list = list(chars)
    for it in range(1, int(iterations) + 1):
        a, b = rng.sample(char_list, 2)
        current[a], current[b] = current[b], current[a]
        trial_cost = mapping_cost(chars, pair_counts, current, move_dist, move_weight)
        accept = False
        if trial_cost <= current_cost:
            accept = True
        else:
            temp = max(0.01, 1.0 - (it / max(1, int(iterations))))
            if rng.random() < math.exp((current_cost - trial_cost) / temp):
                accept = True
        if accept:
            current_cost = trial_cost
            if trial_cost < best_cost:
                best_cost = trial_cost
                best = dict(current)
        else:
            current[a], current[b] = current[b], current[a]
        if it == 1 or it % max(1, int(iterations) // 20) == 0 or it == int(iterations):
            history.append({"iter": float(it), "cost": float(current_cost), "best_cost": float(best_cost)})
    return best, best_cost, history


def encode_text(text: str, mapping: Dict[str, int]) -> List[int]:
    out: List[int] = []
    for ch in text:
        if ch in mapping:
            out.append(int(mapping[ch]))
    return out


def summarize_top_pairs(pair_counts: Counter[Tuple[str, str]], mapping: Dict[str, int], move_dist: Sequence[Sequence[float]], limit: int = 20) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (a, b), cnt in pair_counts.most_common(limit):
        rows.append(
            {
                "pair": a + b,
                "count": int(cnt),
                "src_byte": int(mapping[a]),
                "dst_byte": int(mapping[b]),
                "move_cost": float(move_dist[mapping[a]][mapping[b]]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    reserved = parse_reserved(args.reserved_bytes)
    charset = list(dict.fromkeys(list(str(args.charset))))
    chars, char_counts, pair_counts = collect_transition_stats(
        args.data,
        args.mode,
        charset,
        int(args.max_samples),
        int(args.max_total_bytes),
    )
    if int(args.top_chars) > 0:
        chars = chars[: int(args.top_chars)]
        keep = set(chars)
        char_counts = Counter({ch: cnt for ch, cnt in char_counts.items() if ch in keep})
        pair_counts = Counter({pair: cnt for pair, cnt in pair_counts.items() if pair[0] in keep and pair[1] in keep})
    available_bytes = [b for b in range(256) if b not in reserved]
    if len(chars) > len(available_bytes):
        raise SystemExit(f"charset too large for available bytes: {len(chars)} chars > {len(available_bytes)} bytes")
    move_dist = build_move_distance_matrix(reserved, float(args.fallback_cost))
    ascii_map = ascii_initial_mapping(chars, available_bytes)
    greedy_map = greedy_frequency_seed(chars, char_counts, pair_counts, move_dist, available_bytes)
    ascii_cost = mapping_cost(chars, pair_counts, ascii_map, move_dist, float(args.move_weight))
    greedy_cost = mapping_cost(chars, pair_counts, greedy_map, move_dist, float(args.move_weight))
    seed_map = greedy_map if greedy_cost < ascii_cost else ascii_map
    best_map, best_cost, history = optimize_mapping(
        chars,
        pair_counts,
        seed_map,
        move_dist,
        float(args.move_weight),
        int(args.iterations),
        int(args.seed),
    )
    sample_ascii = encode_text(str(args.sample_text), ascii_map)
    sample_best = encode_text(str(args.sample_text), best_map)
    result = {
        "data": str(args.data),
        "mode": str(args.mode),
        "charset_size": int(len(chars)),
        "reserved_bytes": [int(x) for x in reserved],
        "ascii_cost": float(ascii_cost),
        "greedy_cost": float(greedy_cost),
        "optimized_cost": float(best_cost),
        "improvement_vs_ascii": float((ascii_cost - best_cost) / max(ascii_cost, 1e-9)),
        "top_chars": [{"char": ch, "count": int(char_counts[ch]), "byte": int(best_map[ch])} for ch in chars[:40]],
        "ascii_top_pairs": summarize_top_pairs(pair_counts, ascii_map, move_dist, limit=20),
        "optimized_top_pairs": summarize_top_pairs(pair_counts, best_map, move_dist, limit=20),
        "ascii_mapping": {ch: int(ascii_map[ch]) for ch in chars},
        "optimized_mapping": {ch: int(best_map[ch]) for ch in chars},
        "history": history,
        "sample_text": str(args.sample_text),
        "sample_ascii_bytes": sample_ascii,
        "sample_optimized_bytes": sample_best,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"charset_size={len(chars)}")
    print(f"ascii_cost={ascii_cost:.3f}")
    print(f"greedy_cost={greedy_cost:.3f}")
    print(f"optimized_cost={best_cost:.3f}")
    print(f"improvement_vs_ascii={(ascii_cost - best_cost) / max(ascii_cost, 1e-9):.3%}")
    print(f"wrote_json={out_path}")


if __name__ == "__main__":
    main()
