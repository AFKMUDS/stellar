from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from raplib.data import bytes_to_ids, iter_raw_byte_sequences, iter_train_sequences
from raplib.model import ModelConfig, PrimitiveManifoldLM
from raplib.persistent_model import PersistentRAPConfig, PersistentTwoProgramLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B", "C", "D"], default="D")
    parser.add_argument("--data", default="natural_reasoning_train.jsonl")
    parser.add_argument("--val-data", default="")
    parser.add_argument("--out", default="RobinWilliams/D/persistent2.pt")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-prompt", type=int, default=384)
    parser.add_argument("--max-target", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-symmetry-retrieval", action="store_true")
    parser.add_argument("--use-path-retrieval", action="store_true")
    parser.add_argument("--path-k", type=int, default=2)
    parser.add_argument("--path-alpha", type=float, default=1.0)
    parser.add_argument("--use-word-map-retrieval", action="store_true")
    parser.add_argument("--word-map-path", default="")
    parser.add_argument("--word-map-topk", type=int, default=32)
    parser.add_argument("--word-map-min-count", type=int, default=1)
    parser.add_argument("--switch-cost", type=float, default=0.01)
    parser.add_argument("--switch-cost-end", type=int, default=9000)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--balance-end", type=int, default=3000)
    parser.add_argument("--carrier-target", type=float, default=0.8)
    parser.add_argument("--phase1-end", type=int, default=2000)
    parser.add_argument("--phase2-end", type=int, default=9000)
    parser.add_argument("--release-end", type=int, default=14000)
    parser.add_argument("--phase1-seq-len", type=int, default=192)
    parser.add_argument("--phase1-max-prompt", type=int, default=128)
    parser.add_argument("--phase2-seq-len", type=int, default=384)
    parser.add_argument("--phase2-max-prompt", type=int, default=256)
    parser.add_argument("--hard-routing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--primitive-sharpness-weight", type=float, default=0.0)
    parser.add_argument("--primitive-stability-weight", type=float, default=0.0)
    parser.add_argument("--primitive-balance-weight", type=float, default=0.0)
    parser.add_argument("--primitive-separation-weight", type=float, default=0.0)
    parser.add_argument("--val-batches", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stream_factory(path: str, args: argparse.Namespace, seq_len: int, max_prompt: int):
    if str(path).lower().endswith(".bin"):
        return iter_raw_byte_sequences(path, max_total_bytes=int(seq_len) + 1)
    return iter_train_sequences(
        jsonl_path=path,
        max_prompt_bytes=int(max_prompt),
        max_target_bytes=int(args.max_target),
        max_total_bytes=int(seq_len) + 1,
        include_answer_prefix=True,
    )


def endless_stream(path: str, args: argparse.Namespace, seq_len: int, max_prompt: int):
    while True:
        yielded = False
        for row in stream_factory(path, args, seq_len, max_prompt):
            yielded = True
            yield row
        if not yielded:
            raise RuntimeError(f"No training sequences found in {path}")


def make_batch(stream: Iterator, args: argparse.Namespace, device: torch.device, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = []
    prompt_lengths = []
    sequence_lengths = []
    while len(rows) < int(args.batch_size):
        sequence, prompt_length = next(stream)
        ids = list(bytes_to_ids(sequence))
        if len(ids) < 2:
            continue
        effective = min(len(ids), int(seq_len) + 1)
        ids = ids[: int(seq_len) + 1]
        ids += [0] * (int(seq_len) + 1 - len(ids))
        rows.append(ids)
        prompt_lengths.append(min(int(prompt_length), int(seq_len)))
        sequence_lengths.append(effective)

    x = torch.tensor([row[:-1] for row in rows], device=device, dtype=torch.long)
    y = torch.tensor([row[1:] for row in rows], device=device, dtype=torch.long)
    mask = torch.zeros_like(y, dtype=torch.float32)
    for i, (prompt_length, sequence_length) in enumerate(zip(prompt_lengths, sequence_lengths)):
        end = max(0, sequence_length - 1)
        if end > prompt_length:
            mask[i, prompt_length:end] = 1.0
    mask = mask / mask.sum().clamp_min(1.0)
    return x, y, mask


def linear_decay(step: int, start_value: float, end_step: int) -> float:
    if end_step <= 0 or step >= end_step:
        return 0.0
    return float(start_value) * (1.0 - float(step) / float(end_step))


def schedule(step: int, args: argparse.Namespace) -> dict:
    experiment = str(args.experiment)
    persistent = experiment in ("C", "D")
    retrieval_requested = bool(
        args.use_symmetry_retrieval or args.use_path_retrieval or args.use_word_map_retrieval
    )
    if not retrieval_requested:
        retrieval_scale = 0.0
    elif experiment == "D":
        if step < int(args.phase1_end):
            retrieval_scale = 0.0
        elif step < int(args.phase2_end):
            width = max(1, int(args.phase2_end) - int(args.phase1_end))
            retrieval_scale = float(step - int(args.phase1_end)) / float(width)
        else:
            retrieval_scale = 1.0
    else:
        retrieval_scale = 1.0

    if experiment == "D" and step < int(args.phase2_end):
        carrier_gate_strength = 1.0
    elif experiment == "D" and step < int(args.release_end):
        width = max(1, int(args.release_end) - int(args.phase2_end))
        carrier_gate_strength = 1.0 - float(step - int(args.phase2_end)) / float(width)
    else:
        carrier_gate_strength = 0.0

    if experiment == "D" and step < int(args.phase1_end):
        active_seq_len = min(int(args.seq_len), int(args.phase1_seq_len))
        active_max_prompt = min(int(args.max_prompt), int(args.phase1_max_prompt))
        phase = 1
    elif experiment == "D" and step < int(args.phase2_end):
        active_seq_len = min(int(args.seq_len), int(args.phase2_seq_len))
        active_max_prompt = min(int(args.max_prompt), int(args.phase2_max_prompt))
        phase = 2
    else:
        active_seq_len = int(args.seq_len)
        active_max_prompt = int(args.max_prompt)
        phase = 3

    return {
        "persistent": persistent,
        "phase": phase,
        "active_seq_len": active_seq_len,
        "active_max_prompt": active_max_prompt,
        "retrieval_scale": retrieval_scale,
        "carrier_gate_strength": carrier_gate_strength,
        "switch_cost_weight": linear_decay(step, args.switch_cost, args.switch_cost_end) if persistent else 0.0,
        "balance_weight": linear_decay(step, args.balance_weight, args.balance_end) if persistent else 0.0,
    }


def build_model(args: argparse.Namespace):
    retrieval = {
        "use_symmetry_retrieval": bool(args.use_symmetry_retrieval),
        "use_path_retrieval": bool(args.use_path_retrieval),
        "use_word_map_retrieval": bool(args.use_word_map_retrieval),
        "path_k": int(args.path_k),
        "path_alpha": float(args.path_alpha),
        "word_map_path": str(args.word_map_path),
        "word_map_topk": int(args.word_map_topk),
        "word_map_min_count": int(args.word_map_min_count),
    }
    if args.experiment in ("C", "D"):
        cfg = PersistentRAPConfig(
            d_model=int(args.d_model),
            d_state=int(args.d_state),
            initial_carrier_probability=float(args.carrier_target),
            hard_routing=bool(args.hard_routing),
            **retrieval,
        )
        return PersistentTwoProgramLM(cfg), cfg, "persistent2"
    cfg = ModelConfig(
        d_model=int(args.d_model),
        d_state=int(args.d_state),
        n_primitives=8 if args.experiment == "A" else 2,
        **retrieval,
    )
    return PrimitiveManifoldLM(cfg), cfg, "standard"


def forward_model(
    model,
    model_kind: str,
    x: torch.Tensor,
    current_schedule: dict,
    return_trace: bool,
    trace_detail: bool = False,
):
    if model_kind == "persistent2":
        return model(
            x,
            retrieval_scale=float(current_schedule["retrieval_scale"]),
            carrier_gate_strength=float(current_schedule["carrier_gate_strength"]),
            return_trace=return_trace,
        )
    return model(x, return_trace=return_trace, trace_detail=trace_detail)


def cross_entropy_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    return (token_loss.reshape_as(y) * mask).sum()


def persistent_losses(trace: dict, mask: torch.Tensor, args: argparse.Namespace, current_schedule: dict):
    token_mask = mask / mask.sum().clamp_min(1e-8)
    switch_loss = (trace["switch_probability"].float() * token_mask).sum()
    carrier_mass = (trace["program_probability"][:, :, 1].float() * token_mask).sum()
    balance_loss = torch.abs(carrier_mass - float(args.carrier_target))
    total = (
        float(current_schedule["switch_cost_weight"]) * switch_loss
        + float(current_schedule["balance_weight"]) * balance_loss
    )
    return total, switch_loss, balance_loss, carrier_mass


def primitive_regularization_losses(trace: dict, mask: torch.Tensor, args: argparse.Namespace):
    weights = trace["prim_weights"].float()
    token_mask = mask.float()
    denom = token_mask.sum().clamp_min(1e-8)
    primitive_count = weights.shape[-1]
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1) / math.log(float(primitive_count))
    sharpness_loss = (entropy * token_mask).sum() / denom
    usage = (weights * token_mask.unsqueeze(-1)).sum(dim=(0, 1)) / denom
    balance_loss = ((usage - 1.0 / float(primitive_count)) ** 2).mean()

    if weights.shape[1] > 1:
        pair_mask = token_mask[:, 1:] * token_mask[:, :-1]
        pair_denom = pair_mask.sum().clamp_min(1e-8)
        routing_change = (weights[:, 1:] - weights[:, :-1]).abs().sum(dim=-1) / 2.0
        stability_loss = (routing_change * pair_mask).sum() / pair_denom
    else:
        stability_loss = torch.zeros((), device=weights.device)

    separation_loss = torch.zeros((), device=weights.device)
    if "primitive_candidates" in trace and primitive_count > 1:
        deltas = trace["primitive_candidates"].float() - trace["state_before"].float().unsqueeze(-2)
        normalized = F.normalize(deltas, dim=-1, eps=1e-8)
        cosine = torch.einsum("btkd,btjd->btkj", normalized, normalized)
        pair_select = torch.triu(
            torch.ones(primitive_count, primitive_count, device=weights.device, dtype=torch.bool), diagonal=1
        )
        pair_similarity = cosine[..., pair_select].square().mean(dim=-1)
        separation_loss = (pair_similarity * token_mask).sum() / denom

    total = (
        float(args.primitive_sharpness_weight) * sharpness_loss
        + float(args.primitive_stability_weight) * stability_loss
        + float(args.primitive_balance_weight) * balance_loss
        + float(args.primitive_separation_weight) * separation_loss
    )
    return total, sharpness_loss, stability_loss, balance_loss, separation_loss


def trace_metrics(trace: dict, mask: torch.Tensor, model_kind: str) -> dict:
    active = mask > 0
    if model_kind == "persistent2":
        switch_probability = trace["switch_probability"].float()[active]
        switch_event = trace["switch_event"].float()[active]
        program = trace["active_program"][active]
        retrieval_hit = trace["retrieval_hit"][active]
        effective_norm = trace["retrieval_effective_norm"].float()[active]
        counts = torch.bincount(program, minlength=2)
        return {
            "program_counts": [int(x) for x in counts.detach().cpu().tolist()],
            "mean_switch_probability": float(switch_probability.mean().item()) if switch_probability.numel() else 0.0,
            "switch_rate": float(switch_event.mean().item()) if switch_event.numel() else 0.0,
            "mean_program_duration": float(1.0 / max(float(switch_event.mean().item()), 1e-8)) if switch_event.numel() else 0.0,
            "retrieval_hit_rate": float(retrieval_hit.float().mean().item()) if retrieval_hit.numel() else 0.0,
            "mean_effective_retrieval_norm": float(effective_norm.mean().item()) if effective_norm.numel() else 0.0,
            "mean_state_norm": float(trace["state_norm"].float()[active].mean().item()) if active.any() else 0.0,
            "mean_state_delta_norm": float(trace["state_delta_norm"].float()[active].mean().item()) if active.any() else 0.0,
        }
    weights = trace["prim_weights"].float()[active]
    chosen = trace["chosen_primitive"][active]
    counts = torch.bincount(chosen, minlength=weights.shape[-1])
    usage = weights.mean(dim=0)
    entropy = trace["routing_entropy"].float()[active] / math.log(weights.shape[-1])
    switches = chosen[1:] != chosen[:-1] if chosen.numel() > 1 else torch.empty(0, device=chosen.device, dtype=torch.bool)
    result = {
        "program_counts": [int(x) for x in counts.detach().cpu().tolist()],
        "mean_primitive_usage": [float(x) for x in usage.detach().cpu().tolist()],
        "primitive_usage_variance": float(usage.var(unbiased=False).item()),
        "mean_routing_entropy": float(entropy.mean().item()),
        "mean_max_probability": float(weights.max(dim=-1).values.mean().item()),
        "near_hard_fraction": float((weights.max(dim=-1).values >= 0.9).float().mean().item()),
        "argmax_switch_rate": float(switches.float().mean().item()) if switches.numel() else 0.0,
        "mean_run_length": float(chosen.numel() / max(int(switches.sum().item()), 1)),
        "mean_gate": float(trace["gate_mean"].float()[active].mean().item()),
    }
    if "primitive_candidates" in trace and "state_before" in trace:
        candidates = trace["primitive_candidates"].float()
        deltas = candidates - trace["state_before"].float().unsqueeze(-2)
        normalized = F.normalize(deltas, dim=-1, eps=1e-8)
        cosine = torch.einsum("btkd,btjd->btkj", normalized, normalized)
        distance = torch.linalg.vector_norm(candidates.unsqueeze(-2) - candidates.unsqueeze(-3), dim=-1)
        pair_mask = torch.triu(torch.ones(weights.shape[-1], weights.shape[-1], device=weights.device, dtype=torch.bool), diagonal=1)
        result["mean_candidate_delta_cosine"] = float(cosine[..., pair_mask][active].mean().item())
        result["mean_candidate_state_distance"] = float(distance[..., pair_mask][active].mean().item())
    return result


def evaluate(model, model_kind: str, path: str, args: argparse.Namespace, device: torch.device, current_schedule: dict) -> float:
    if not path:
        return float("nan")
    stream = endless_stream(path, args, int(args.seq_len), int(args.max_prompt))
    losses = []
    model.eval()
    with torch.inference_mode():
        for _ in range(int(args.val_batches)):
            x, y, mask = make_batch(stream, args, device, int(args.seq_len))
            output = forward_model(model, model_kind, x, current_schedule, return_trace=False)
            losses.append(float(cross_entropy_loss(output[0], y, mask).item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def save_checkpoint(path: Path, model, optimizer, scaler, cfg, args, step: int, model_kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": model_kind,
            "experiment": str(args.experiment),
            "cfg": asdict(cfg),
            "args": vars(args),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": None if scaler is None else scaler.state_dict(),
            "step": int(step),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(args.device)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path.with_suffix(".metrics.jsonl")
    model, cfg, model_kind = build_model(args)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and amp_dtype == torch.float16))
    start_step = 0

    if args.resume and output_path.exists():
        checkpoint = torch.load(output_path, map_location=device, weights_only=False)
        if checkpoint.get("model_kind") != model_kind or checkpoint.get("experiment") != args.experiment:
            raise RuntimeError("Checkpoint model kind or experiment does not match the requested run")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = int(checkpoint["step"]) + 1

    train_streams = {}
    milestones = {200, 500, 1000, 2000, 5000, 9000, 20000}
    progress = tqdm(range(start_step, int(args.steps)), initial=start_step, total=int(args.steps), desc=f"run {args.experiment}")

    for step in progress:
        current_schedule = schedule(step, args)
        stream_key = (int(current_schedule["active_seq_len"]), int(current_schedule["active_max_prompt"]))
        if stream_key not in train_streams:
            train_streams[stream_key] = endless_stream(args.data, args, stream_key[0], stream_key[1])
        x, y, mask = make_batch(train_streams[stream_key], args, device, stream_key[0])
        optimizer.zero_grad(set_to_none=True)

        collect_detailed_trace = bool(
            model_kind == "standard"
            and (
                float(args.primitive_separation_weight) != 0.0
                or (int(args.metrics_every) > 0 and (step + 1) % int(args.metrics_every) == 0)
            )
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            output = forward_model(
                model,
                model_kind,
                x,
                current_schedule,
                return_trace=True,
                trace_detail=collect_detailed_trace,
            )
            logits = output[0]
            trace = output[2]
            ce_loss = cross_entropy_loss(logits, y, mask)
            structural_loss = torch.zeros((), device=device)
            switch_loss = torch.zeros((), device=device)
            balance_loss = torch.zeros((), device=device)
            carrier_mass = torch.zeros((), device=device)
            primitive_loss = torch.zeros((), device=device)
            primitive_sharpness_loss = torch.zeros((), device=device)
            primitive_stability_loss = torch.zeros((), device=device)
            primitive_balance_loss = torch.zeros((), device=device)
            primitive_separation_loss = torch.zeros((), device=device)
            if model_kind == "persistent2":
                structural_loss, switch_loss, balance_loss, carrier_mass = persistent_losses(trace, mask, args, current_schedule)
            else:
                (
                    primitive_loss,
                    primitive_sharpness_loss,
                    primitive_stability_loss,
                    primitive_balance_loss,
                    primitive_separation_loss,
                ) = primitive_regularization_losses(trace, mask, args)
            loss = ce_loss + structural_loss + primitive_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        scaler.step(optimizer)
        scaler.update()

        progress.set_postfix(loss=float(loss.detach()), ce=float(ce_loss.detach()))

        if int(args.metrics_every) > 0 and (step + 1) % int(args.metrics_every) == 0:
            metrics = trace_metrics(trace, mask, model_kind)
            val_loss = evaluate(model, model_kind, args.val_data, args, device, current_schedule)
            record = {
                "step": step + 1,
                "experiment": args.experiment,
                "model_kind": model_kind,
                "loss": float(loss.detach()),
                "ce_loss": float(ce_loss.detach()),
                "val_loss": None if math.isnan(val_loss) else val_loss,
                "switch_loss": float(switch_loss.detach()),
                "balance_loss": float(balance_loss.detach()),
                "carrier_mass": float(carrier_mass.detach()),
                "primitive_loss": float(primitive_loss.detach()),
                "primitive_sharpness_loss": float(primitive_sharpness_loss.detach()),
                "primitive_stability_loss": float(primitive_stability_loss.detach()),
                "primitive_balance_loss": float(primitive_balance_loss.detach()),
                "primitive_separation_loss": float(primitive_separation_loss.detach()),
                **current_schedule,
                **metrics,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        checkpoint_number = step + 1
        should_save = checkpoint_number in milestones or (
            int(args.save_every) > 0 and checkpoint_number % int(args.save_every) == 0
        )
        if should_save:
            save_checkpoint(output_path, model, optimizer, scaler, cfg, args, step, model_kind)
            step_path = output_path.with_name(f"{output_path.stem}_step{checkpoint_number:06d}.pt")
            save_checkpoint(step_path, model, optimizer, scaler, cfg, args, step, model_kind)

    final_step = int(args.steps) - 1
    save_checkpoint(output_path, model, optimizer, scaler, cfg, args, final_step, model_kind)


if __name__ == "__main__":
    main()
