import argparse
import glob
import json
import os
from collections import defaultdict

from probe_latent_step import decode_ids, load_model, rollout_from_state, capture_state


DEFAULT_PROMPTS = [
    "What is gravity?",
    "Explain time.",
    "How does electricity work?",
    "Why do objects fall?",
    "What is a cell?",
]


def parse_list(spec: str) -> list[str]:
    vals = [x.strip() for x in str(spec).split("|")]
    return [v for v in vals if v]


def text_metrics(text: str) -> dict[str, float]:
    if not text:
        return {
            "len": 0.0,
            "printable_ratio": 0.0,
            "alpha_ratio": 0.0,
            "space_ratio": 0.0,
            "unique_char_ratio": 0.0,
            "repeat_penalty": 1.0,
            "score": -1.0,
        }
    n = len(text)
    printable = sum(1 for ch in text if ch.isprintable()) / n
    alpha = sum(1 for ch in text if ch.isalpha()) / n
    space = sum(1 for ch in text if ch.isspace()) / n
    unique = len(set(text)) / n
    repeats = 0
    for i in range(2, n):
        if text[i] == text[i - 1] == text[i - 2]:
            repeats += 1
    repeat_penalty = repeats / n
    score = (1.8 * printable) + (1.2 * alpha) + (0.4 * unique) + (0.2 * min(space, 0.25)) - (2.4 * repeat_penalty)
    return {
        "len": float(n),
        "printable_ratio": float(printable),
        "alpha_ratio": float(alpha),
        "space_ratio": float(space),
        "unique_char_ratio": float(unique),
        "repeat_penalty": float(repeat_penalty),
        "score": float(score),
    }


def summarize_case(text: str) -> dict[str, float | str]:
    out = text_metrics(text)
    out["text"] = text
    return out


def aggregate_rows(rows: list[dict]) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        p = int(row["primitive"])
        grouped[p]["normal_score"].append(float(row["normal_score"]))
        grouped[p]["forced_score"].append(float(row["forced_score"]))
        grouped[p]["forbidden_score"].append(float(row["forbidden_score"]))
        grouped[p]["forced_delta"].append(float(row["forced_delta"]))
        grouped[p]["forbidden_delta"].append(float(row["forbidden_delta"]))
    summary = {}
    for p, vals in grouped.items():
        summary[p] = {k: float(sum(v) / max(len(v), 1)) for k, v in vals.items()}
        summary[p]["helpfulness_index"] = summary[p]["forbidden_delta"] - abs(min(summary[p]["forced_delta"], 0.0))
        summary[p]["harmfulness_index"] = -summary[p]["forced_delta"] + max(summary[p]["forbidden_delta"], 0.0)
    return summary


def resolve_checkpoints(spec: str) -> list[str]:
    parts = parse_list(spec)
    if not parts:
        parts = [str(spec)]
    found: list[str] = []
    for part in parts:
        matches = sorted(glob.glob(part))
        if matches:
            found.extend(matches)
            continue
        if os.path.exists(part):
            found.append(part)
            continue
        raise FileNotFoundError(f"No checkpoints match: {part}")
    deduped = []
    seen = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", type=str, default="checkpoints1\\b.pt")
    p.add_argument("--prompts", type=str, default="")
    p.add_argument("--capture_step", type=int, default=-1)
    p.add_argument("--max_new", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="primitive_report.json")
    args = p.parse_args()

    ckpts = resolve_checkpoints(args.ckpts)
    prompts = parse_list(args.prompts) if str(args.prompts).strip() else list(DEFAULT_PROMPTS)
    report_rows: list[dict] = []

    for ckpt in ckpts:
        model = load_model(ckpt, args.device)
        for prompt in prompts:
            prompt_ids = list(prompt.encode("utf-8", errors="replace"))
            capture_step = int(args.capture_step)
            if capture_step < 0:
                capture_step = max(len(prompt_ids) - 1, 0)
            cap_state, cap_last_id, _cap_weights, cap_last_pos, cap_state_hist = capture_state(model, prompt_ids, capture_step, args.device)
            normal_ids, _normal_w = rollout_from_state(
                model,
                cap_state.clone(),
                cap_last_id,
                last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
                state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
                max_new=int(args.max_new),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
            )
            normal_text = decode_ids(normal_ids)
            normal_case = summarize_case(normal_text)

            for prim in range(model.cfg.n_primitives):
                forced_ids, _forced_w = rollout_from_state(
                    model,
                    cap_state.clone(),
                    cap_last_id,
                    last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
                    state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
                    max_new=int(args.max_new),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    force_primitive=int(prim),
                    force_strength=1.0,
                )
                forbidden_ids, _forbidden_w = rollout_from_state(
                    model,
                    cap_state.clone(),
                    cap_last_id,
                    last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
                    state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
                    max_new=int(args.max_new),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    forbid_primitive=int(prim),
                )
                forced_case = summarize_case(decode_ids(forced_ids))
                forbidden_case = summarize_case(decode_ids(forbidden_ids))
                report_rows.append({
                    "checkpoint": ckpt,
                    "prompt": prompt,
                    "primitive": int(prim),
                    "normal_score": float(normal_case["score"]),
                    "forced_score": float(forced_case["score"]),
                    "forbidden_score": float(forbidden_case["score"]),
                    "forced_delta": float(forced_case["score"] - normal_case["score"]),
                    "forbidden_delta": float(normal_case["score"] - forbidden_case["score"]),
                    "normal_text": normal_case["text"],
                    "forced_text": forced_case["text"],
                    "forbidden_text": forbidden_case["text"],
                })
        del model

    summary = aggregate_rows(report_rows)
    helpful = sorted(summary.items(), key=lambda kv: kv[1]["helpfulness_index"], reverse=True)
    harmful = sorted(summary.items(), key=lambda kv: kv[1]["harmfulness_index"], reverse=True)
    report = {
        "checkpoints": ckpts,
        "prompts": prompts,
        "summary": {str(k): v for k, v in summary.items()},
        "helpful_ranking": [int(k) for k, _v in helpful],
        "harmful_ranking": [int(k) for k, _v in harmful],
        "rows": report_rows,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote: {os.path.abspath(args.out)}")
    print("Helpful ranking:", report["helpful_ranking"])
    print("Harmful ranking:", report["harmful_ranking"])
    for prim, stats in helpful:
        print(
            f"P{prim}: helpful={stats['helpfulness_index']:.3f} harmful={stats['harmfulness_index']:.3f} "
            f"forced_delta={stats['forced_delta']:.3f} forbidden_delta={stats['forbidden_delta']:.3f}"
        )


if __name__ == "__main__":
    main()
