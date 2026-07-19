import argparse
import math
from dataclasses import dataclass
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F

from raplib.data import ids_to_bytes
from raplib.model import ModelConfig, PrimitiveManifoldLM


ASCII_ALLOWED = [9, 10, 13] + list(range(32, 127))


@dataclass
class SearchResult:
    ids: list[int]
    score: float
    teacher_score: float
    rollout_text: str
    rollout_lcp: int
    rollout_ratio: float


def load_model(path: str, device: str) -> PrimitiveManifoldLM:
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt.get("cfg", {})
    valid = ModelConfig.__annotations__.keys()
    cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    model = PrimitiveManifoldLM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def fmt(ids: list[int]) -> str:
    return ids_to_bytes(ids).decode("utf-8", errors="replace")


def ascii_mask(logits: torch.Tensor) -> torch.Tensor:
    allowed = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    allowed[ASCII_ALLOWED] = True
    out = logits.clone()
    out[:, ~allowed] = -float("inf")
    return out


@torch.no_grad()
def next_logits_after_prompt(model: PrimitiveManifoldLM, prompt_ids: list[int]) -> torch.Tensor:
    device = next(model.parameters()).device
    state = model.init_state(1, device)
    if prompt_ids:
        x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    else:
        x = torch.tensor([[0]], device=device, dtype=torch.long)
    out = model(x, state)
    return out[0][:, -1, :]


@torch.no_grad()
def teacher_logprob(model: PrimitiveManifoldLM, prompt_ids: list[int], target_ids: list[int]) -> float:
    device = next(model.parameters()).device
    state = model.init_state(1, device)
    if prompt_ids:
        x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    else:
        x = torch.tensor([[0]], device=device, dtype=torch.long)
    out = model(x, state)
    next_logits, state = out[0][:, -1, :], out[1]

    total = 0.0
    for tid in target_ids:
        logp = F.log_softmax(next_logits, dim=-1)
        total += float(logp[0, int(tid)].item())
        x = torch.tensor([[int(tid)]], device=device, dtype=torch.long)
        out = model(x, state)
        next_logits, state = out[0][:, -1, :], out[1]
    return total


@torch.no_grad()
def greedy_rollout(
    model: PrimitiveManifoldLM,
    prompt_ids: list[int],
    length: int,
    *,
    ascii_only: bool,
    top_k: int,
) -> list[int]:
    device = next(model.parameters()).device
    state = model.init_state(1, device)
    if prompt_ids:
        x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    else:
        x = torch.tensor([[0]], device=device, dtype=torch.long)
    out = model(x, state)
    next_logits, state = out[0][:, -1, :], out[1]

    out_ids: list[int] = []
    for _ in range(length):
        logits = next_logits.clone()
        if ascii_only:
            logits = ascii_mask(logits)
        if top_k > 0 and top_k < logits.shape[-1]:
            values, idx = torch.topk(logits, top_k)
            kept = torch.full_like(logits, -float("inf"))
            kept.scatter_(1, idx, values)
            logits = kept
        nxt = int(torch.argmax(logits, dim=-1).item())
        out_ids.append(nxt)
        x = torch.tensor([[nxt]], device=device, dtype=torch.long)
        out = model(x, state)
        next_logits, state = out[0][:, -1, :], out[1]
    return out_ids


def lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def rollout_score(rollout: str, target: str) -> tuple[int, float]:
    return lcp(rollout, target), SequenceMatcher(None, rollout, target).ratio()


def candidate_score(
    model: PrimitiveManifoldLM,
    ids: list[int],
    target_text: str,
    target_ids: list[int],
    *,
    rollout_len: int,
    ascii_only: bool,
    rollout_top_k: int,
    teacher_weight: float,
    lcp_weight: float,
    ratio_weight: float,
    prefix_len: int,
    score_prompt_tail: bool,
) -> SearchResult:
    tscore = teacher_logprob(model, ids, target_ids)
    rollout_ids = greedy_rollout(model, ids, rollout_len, ascii_only=ascii_only, top_k=rollout_top_k)
    rtext = fmt(rollout_ids)
    surface = rtext
    if score_prompt_tail:
        surface = fmt(ids[prefix_len:]) + rtext
    rlcp, rratio = rollout_score(surface, target_text)
    score = teacher_weight * tscore + lcp_weight * rlcp + ratio_weight * rratio
    return SearchResult(ids, score, tscore, rtext, rlcp, rratio)


def search(args: argparse.Namespace) -> SearchResult:
    model = load_model(args.ckpt, args.device)
    target_ids = list(args.target.encode("utf-8"))[: args.target_prefix_len]
    prefix_ids = list(args.prefix.encode("utf-8"))
    raw_prefix_len = len(prefix_ids)
    if len(prefix_ids) > args.prompt_len:
        prefix_ids = prefix_ids[: args.prompt_len]

    seed = candidate_score(
        model,
        prefix_ids,
        args.target,
        target_ids,
        rollout_len=args.rollout_len,
        ascii_only=args.ascii_only,
        rollout_top_k=args.rollout_top_k,
        teacher_weight=args.teacher_weight,
        lcp_weight=args.lcp_weight,
        ratio_weight=args.ratio_weight,
        prefix_len=raw_prefix_len,
        score_prompt_tail=args.score_prompt_tail,
    )
    beams = [seed]

    # Add target bytes as exploration hints so rollout-aware search is not trapped
    # inside only the model's most habitual next prompt bytes.
    hint_bytes = list(dict.fromkeys(list(args.target.encode("utf-8")) + list(args.hints.encode("utf-8"))))
    hint_bytes = [b for b in hint_bytes if (not args.ascii_only) or b in ASCII_ALLOWED]

    while len(beams[0].ids) < args.prompt_len:
        all_cand: list[SearchResult] = []
        for beam in beams:
            logits = next_logits_after_prompt(model, beam.ids)
            if args.ascii_only:
                logits = ascii_mask(logits)
            _, idx = torch.topk(logits, k=min(args.expand_topk, logits.shape[-1]))
            next_bytes = list(dict.fromkeys([int(x) for x in idx[0].tolist()] + hint_bytes))
            for b in next_bytes:
                ids = beam.ids + [b]
                all_cand.append(
                    candidate_score(
                        model,
                        ids,
                        args.target,
                        target_ids,
                        rollout_len=args.rollout_len,
                        ascii_only=args.ascii_only,
                        rollout_top_k=args.rollout_top_k,
                        teacher_weight=args.teacher_weight,
                        lcp_weight=args.lcp_weight,
                        ratio_weight=args.ratio_weight,
                        prefix_len=raw_prefix_len,
                        score_prompt_tail=args.score_prompt_tail,
                    )
                )
        all_cand.sort(key=lambda r: r.score, reverse=True)
        beams = all_cand[: args.beam_width]
        best = beams[0]
        print(
            f"len={len(best.ids):03d} score={best.score:.3f} teacher={best.teacher_score:.3f} "
            f"lcp={best.rollout_lcp} ratio={best.rollout_ratio:.3f} prompt={fmt(best.ids)!r} "
            f"rollout={best.rollout_text[:80]!r}",
            flush=True,
        )
    return beams[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints1/b.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target", default="Gravity is a force that pulls objects with mass toward each other.")
    p.add_argument("--prefix", default="Answer:")
    p.add_argument("--prompt_len", type=int, default=14)
    p.add_argument("--target_prefix_len", type=int, default=32)
    p.add_argument("--rollout_len", type=int, default=64)
    p.add_argument("--beam_width", type=int, default=4)
    p.add_argument("--expand_topk", type=int, default=6)
    p.add_argument("--rollout_top_k", type=int, default=0)
    p.add_argument("--ascii_only", action="store_true", default=True)
    p.add_argument("--teacher_weight", type=float, default=0.03)
    p.add_argument("--lcp_weight", type=float, default=4.0)
    p.add_argument("--ratio_weight", type=float, default=12.0)
    p.add_argument("--hints", default=" gravity force mass pulls objects")
    p.add_argument("--score_prompt_tail", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    best = search(args)
    print("\nBEST")
    print(f"prompt:  {fmt(best.ids)!r}")
    print(f"score:   {best.score:.3f}")
    print(f"teacher: {best.teacher_score:.3f}")
    print(f"lcp:     {best.rollout_lcp}")
    print(f"ratio:   {best.rollout_ratio:.3f}")
    print(f"rollout: {best.rollout_text!r}")


if __name__ == "__main__":
    main()
