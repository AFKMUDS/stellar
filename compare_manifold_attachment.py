import argparse
import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from raplib.model import ModelConfig, PrimitiveManifoldLM


@dataclass
class RunCfg:
    ckpt: str
    manifold_pt: str
    basis_tier: str
    max_groups: int
    words_per_group: int
    max_word_bytes: int
    prompt_template: str
    device: str
    out_json: str


def parse_args() -> RunCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--manifold_pt", type=str, required=True)
    p.add_argument("--basis_tier", type=str, default="deep_operators")
    p.add_argument("--max_groups", type=int, default=12)
    p.add_argument("--words_per_group", type=int, default=8)
    p.add_argument("--max_word_bytes", type=int, default=32)
    p.add_argument("--prompt_template", type=str, default="{word}")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", type=str, default="manifold_attachment_compare.json")
    a = p.parse_args()
    return RunCfg(
        ckpt=a.ckpt,
        manifold_pt=a.manifold_pt,
        basis_tier=a.basis_tier,
        max_groups=a.max_groups,
        words_per_group=a.words_per_group,
        max_word_bytes=a.max_word_bytes,
        prompt_template=a.prompt_template,
        device=a.device,
        out_json=a.out_json,
    )


def load_model(ckpt_path: str, device: str) -> PrimitiveManifoldLM:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("cfg") if isinstance(ckpt, dict) else None
    if isinstance(cfg_dict, dict):
        valid_keys = ModelConfig.__annotations__.keys()
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = ModelConfig(**filtered)
    else:
        cfg = ModelConfig()
    model = PrimitiveManifoldLM(cfg)
    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else ckpt
    if state_dict is None:
        state_dict = ckpt.get("student_state_dict") if isinstance(ckpt, dict) else None
    if state_dict is None:
        raise KeyError("Checkpoint must contain either 'state_dict' or 'student_state_dict'.")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def encode_text(text: str, max_word_bytes: int) -> list[int]:
    raw = list(str(text).encode("utf-8", errors="replace"))
    return raw[: max(1, int(max_word_bytes))]


def analyze_word(
    model: PrimitiveManifoldLM,
    word: str,
    device: str,
    max_word_bytes: int,
    prompt_template: str,
) -> dict[str, Any]:
    text = str(prompt_template).replace("{word}", str(word))
    ids = encode_text(text, max_word_bytes=max_word_bytes)
    x = torch.tensor([ids], device=device, dtype=torch.long)
    with torch.no_grad():
        logits, state, aux = model(x, state=None, return_trace=True)
    prim_weights = aux["prim_weights"][0]
    mean_prim = prim_weights.mean(dim=0).detach().cpu()
    final_state = state[0].detach().cpu()
    final_logits = logits[0, -1].detach().cpu()
    top_prim = int(torch.argmax(mean_prim).item())
    top_logit_ids = torch.topk(final_logits, k=min(8, final_logits.numel())).indices.tolist()
    return {
        "word": word,
        "prompt": text,
        "ids": ids,
        "top_primitive": top_prim,
        "primitive_mean": mean_prim,
        "state": final_state,
        "top_logit_ids": top_logit_ids,
    }


def mean_pairwise_cos(items: list[torch.Tensor]) -> float:
    if len(items) < 2:
        return 1.0
    vals: list[float] = []
    for i in range(len(items)):
        a = items[i].reshape(1, -1)
        for j in range(i + 1, len(items)):
            b = items[j].reshape(1, -1)
            vals.append(float(F.cosine_similarity(a, b).item()))
    return sum(vals) / max(1, len(vals))


def cross_group_cos(a_items: list[torch.Tensor], b_items: list[torch.Tensor]) -> float:
    if not a_items or not b_items:
        return 0.0
    vals: list[float] = []
    for a in a_items:
        aa = a.reshape(1, -1)
        for b in b_items:
            bb = b.reshape(1, -1)
            vals.append(float(F.cosine_similarity(aa, bb).item()))
    return sum(vals) / max(1, len(vals))


def build_word_lookup(manifold: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in manifold["words"]["records"]:
        out[str(row["word"])] = row
    return out


def choose_groups(manifold: dict[str, Any], basis_tier: str, max_groups: int, words_per_group: int) -> list[dict[str, Any]]:
    word_lookup = build_word_lookup(manifold)
    tier_rows = list(manifold["operator_basis"].get(str(basis_tier), []))
    chosen: list[dict[str, Any]] = []
    for row in tier_rows:
        words = [w for w in row.get("example_words", []) if str(w) in word_lookup]
        if len(words) < 2:
            continue
        chosen.append(
            {
                "grounding": row["grounding"],
                "primitive_score": float(row.get("primitive_score", 0.0)),
                "words": words[: max(2, int(words_per_group))],
            }
        )
        if len(chosen) >= int(max_groups):
            break
    return chosen


def main() -> None:
    args = parse_args()
    model = load_model(args.ckpt, args.device)
    manifold = torch.load(args.manifold_pt, map_location="cpu")
    groups = choose_groups(manifold, basis_tier=args.basis_tier, max_groups=int(args.max_groups), words_per_group=int(args.words_per_group))

    analyzed_groups: list[dict[str, Any]] = []
    for group in groups:
        analyzed_words = [
            analyze_word(
                model,
                word=w,
                device=args.device,
                max_word_bytes=int(args.max_word_bytes),
                prompt_template=args.prompt_template,
            )
            for w in group["words"]
        ]
        prim_vecs = [item["primitive_mean"] for item in analyzed_words]
        state_vecs = [item["state"] for item in analyzed_words]
        top_prims = [int(item["top_primitive"]) for item in analyzed_words]
        analyzed_groups.append(
            {
                "grounding": group["grounding"],
                "primitive_score": group["primitive_score"],
                "words": [
                    {
                        "word": item["word"],
                        "prompt": item["prompt"],
                        "top_primitive": item["top_primitive"],
                        "top_logit_ids": item["top_logit_ids"],
                    }
                    for item in analyzed_words
                ],
                "within_primitive_cos": mean_pairwise_cos(prim_vecs),
                "within_state_cos": mean_pairwise_cos(state_vecs),
                "top_primitive_hist": {str(k): top_prims.count(k) for k in sorted(set(top_prims))},
                "_prim_vecs": prim_vecs,
                "_state_vecs": state_vecs,
            }
        )

    cross_primitive_vals: list[float] = []
    cross_state_vals: list[float] = []
    for i in range(len(analyzed_groups)):
        for j in range(i + 1, len(analyzed_groups)):
            cross_primitive_vals.append(cross_group_cos(analyzed_groups[i]["_prim_vecs"], analyzed_groups[j]["_prim_vecs"]))
            cross_state_vals.append(cross_group_cos(analyzed_groups[i]["_state_vecs"], analyzed_groups[j]["_state_vecs"]))

    payload = {
        "ckpt": args.ckpt,
        "manifold_pt": args.manifold_pt,
        "basis_tier": args.basis_tier,
        "prompt_template": args.prompt_template,
        "group_count": len(analyzed_groups),
        "mean_within_primitive_cos": sum(g["within_primitive_cos"] for g in analyzed_groups) / max(1, len(analyzed_groups)),
        "mean_within_state_cos": sum(g["within_state_cos"] for g in analyzed_groups) / max(1, len(analyzed_groups)),
        "mean_cross_primitive_cos": sum(cross_primitive_vals) / max(1, len(cross_primitive_vals)),
        "mean_cross_state_cos": sum(cross_state_vals) / max(1, len(cross_state_vals)),
        "groups": [
            {
                k: v
                for k, v in group.items()
                if not k.startswith("_")
            }
            for group in analyzed_groups
        ],
    }

    out_path = args.out_json
    if not out_path.lower().endswith(".json"):
        out_path = out_path + ".json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"checkpoint={args.ckpt}")
    print(f"manifold_pt={args.manifold_pt}")
    print(f"basis_tier={args.basis_tier}")
    print(f"prompt_template={args.prompt_template}")
    print(f"groups={payload['group_count']}")
    print(f"mean_within_primitive_cos={payload['mean_within_primitive_cos']:.4f}")
    print(f"mean_cross_primitive_cos={payload['mean_cross_primitive_cos']:.4f}")
    print(f"mean_within_state_cos={payload['mean_within_state_cos']:.4f}")
    print(f"mean_cross_state_cos={payload['mean_cross_state_cos']:.4f}")
    print(f"out_json={out_path}")


if __name__ == "__main__":
    main()
