from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from raplib.data import bytes_to_ids, iter_train_sequences


OP_NAMES = [
    "keep",
    "flip_bit_0",
    "flip_bit_1",
    "flip_bit_2",
    "flip_bit_3",
    "flip_bit_4",
    "flip_bit_5",
    "flip_bit_6",
    "flip_bit_7",
    "inc",
    "dec",
    "xor20",
    "rotl1",
    "rotr1",
]
ROUTE_NAMES = ["direct", "copy_recent", "bit_op"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compute-and-speak byte model with copy/direct/bit-op routes.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "probe", "generate", "chat", "dump"])
    parser.add_argument("--data", type=str, default="natural_reasoning_train.jsonl")
    parser.add_argument("--out", type=str, default="checkpoints/compute_and_speak_byte_trainer.json")
    parser.add_argument("--save-model", type=str, default="checkpoints/compute_and_speak_byte_trainer.pt")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-prompt", type=int, default=384)
    parser.add_argument("--max-target", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--d-state", type=int, default=384)
    parser.add_argument("--recent-window", type=int, default=8)
    parser.add_argument("--route-temp", type=float, default=1.0)
    parser.add_argument("--op-supervision-scale", type=float, default=0.25)
    parser.add_argument("--copy-supervision-scale", type=float, default=0.15)
    parser.add_argument("--route-supervision-scale", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--probe-rows", type=int, default=3)
    parser.add_argument("--probe-steps", type=int, default=64)
    parser.add_argument("--gen-bytes", type=int, default=160)
    parser.add_argument("--sample-temp", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--dump-text", type=str, default="")
    parser.add_argument("--dump-tail", type=int, default=24)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_stream(args: argparse.Namespace):
    return iter_train_sequences(
        jsonl_path=str(args.data),
        max_prompt_bytes=int(args.max_prompt),
        max_target_bytes=int(args.max_target),
        max_total_bytes=int(args.seq_len) + 1,
        include_answer_prefix=True,
    )


def next_batch(stream, args: argparse.Namespace, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch: List[Tuple[int, ...]] = []
    prompt_lens: List[int] = []
    seq_lens: List[int] = []
    while len(batch) < int(args.batch_size):
        seq_bytes, prompt_len = next(stream)
        ids = bytes_to_ids(seq_bytes)
        if len(ids) < 2:
            continue
        eff_len = min(len(ids), int(args.seq_len) + 1)
        ids = ids[: int(args.seq_len) + 1]
        if len(ids) < int(args.seq_len) + 1:
            ids = ids + (0,) * (int(args.seq_len) + 1 - len(ids))
        batch.append(ids)
        prompt_lens.append(min(int(prompt_len), int(args.seq_len)))
        seq_lens.append(min(int(eff_len), int(args.seq_len) + 1))

    x = torch.tensor([row[: int(args.seq_len)] for row in batch], device=device, dtype=torch.long)
    y = torch.tensor([row[1 : int(args.seq_len) + 1] for row in batch], device=device, dtype=torch.long)
    loss_mask = torch.zeros_like(y, dtype=torch.float32)
    for i, (pl, sl) in enumerate(zip(prompt_lens, seq_lens)):
        start = min(int(pl), int(args.seq_len))
        end = max(0, int(sl) - 1)
        if end > start:
            loss_mask[i, start:end] = 1.0
    loss_mask = loss_mask / loss_mask.sum().clamp_min(1.0)
    return x, y, loss_mask


def apply_ops(src: torch.Tensor) -> torch.Tensor:
    src = src.to(dtype=torch.long)
    outs = [src]
    for bit in range(8):
        outs.append(src ^ (1 << bit))
    outs.append((src + 1) & 0xFF)
    outs.append((src - 1) & 0xFF)
    outs.append(src ^ 0x20)
    outs.append(((src << 1) | (src >> 7)) & 0xFF)
    outs.append(((src >> 1) | ((src & 1) << 7)) & 0xFF)
    return torch.stack(outs, dim=-1)


def classify_transitions(src: torch.Tensor, dst: torch.Tensor) -> Dict[str, torch.Tensor]:
    xorv = torch.bitwise_xor(src, dst)
    pop = torch.zeros_like(xorv)
    for bit in range(8):
        pop = pop + ((xorv >> bit) & 1)
    rotl1 = ((src << 1) | (src >> 7)) & 0xFF
    rotr1 = ((src >> 1) | ((src & 1) << 7)) & 0xFF
    return {
        "identity": (dst == src),
        "inc": (dst == ((src + 1) & 0xFF)),
        "dec": (dst == ((src - 1) & 0xFF)),
        "xor20": (xorv == 0x20),
        "flip1": (pop == 1),
        "flip2plus": (pop >= 2),
        "rotl1": (dst == rotl1),
        "rotr1": (dst == rotr1),
    }


def compute_target_op_mask(op_targets: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return op_targets.eq(y.unsqueeze(-1))


def build_recent_bytes(x: torch.Tensor, window: int) -> torch.Tensor:
    batch, steps = x.shape
    recent: List[torch.Tensor] = []
    for offset in range(1, int(window) + 1):
        if offset < steps:
            pad = x[:, :1]
            shifted = torch.cat([pad.expand(-1, offset), x[:, :-offset]], dim=1)
        else:
            shifted = x[:, :1].expand(-1, steps)
        recent.append(shifted.unsqueeze(-1))
    return torch.cat(recent, dim=-1)


def printable_byte(v: int) -> str:
    if v == 10:
        return "\\n"
    if v == 13:
        return "\\r"
    if v == 9:
        return "\\t"
    if 32 <= int(v) <= 126:
        return chr(int(v))
    return f"\\x{int(v):02x}"


def decode_bytes(values: List[int]) -> str:
    return bytes(int(v) for v in values).decode("utf-8", errors="replace")


def format_bits(value: int) -> str:
    return format(int(value) & 0xFF, "08b")


class ComputeAndSpeakByteLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_state: int, recent_window: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.recent_window = int(recent_window)
        self.num_ops = len(OP_NAMES)
        self.num_routes = len(ROUTE_NAMES)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.prev_emb = nn.Embedding(vocab_size, d_model)
        self.state_cell = nn.GRUCell(d_model * 2, d_state)
        self.control = nn.Sequential(
            nn.Linear(d_model * 2 + d_state, d_state),
            nn.GELU(),
            nn.Linear(d_state, d_state),
            nn.GELU(),
        )
        self.direct_head = nn.Linear(d_state + d_model, vocab_size)
        self.route_head = nn.Linear(d_state + d_model, self.num_routes)
        self.op_head = nn.Linear(d_state + d_model, self.num_ops)
        self.copy_head = nn.Linear(d_state + d_model, self.recent_window)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_state, device=device)

    def forward(self, x: torch.Tensor, route_temp: float) -> Dict[str, torch.Tensor]:
        batch, steps = x.shape
        tok = self.token_emb(x)
        prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        prev_tok = self.prev_emb(prev)
        recent_bytes = build_recent_bytes(x, self.recent_window)
        state = self.init_state(batch, x.device)

        final_logprob_seq: List[torch.Tensor] = []
        route_prob_seq: List[torch.Tensor] = []
        op_prob_seq: List[torch.Tensor] = []
        copy_choice_prob_seq: List[torch.Tensor] = []
        direct_logprob_seq: List[torch.Tensor] = []
        reachable_seq: List[torch.Tensor] = []
        copy_targets_seq: List[torch.Tensor] = []

        for t in range(steps):
            tok_t = tok[:, t, :]
            prev_t = prev_tok[:, t, :]
            state = self.state_cell(torch.cat([tok_t, prev_t], dim=-1), state)
            ctrl = self.control(torch.cat([tok_t, prev_t, state], dim=-1))
            fused = torch.cat([ctrl, tok_t], dim=-1)

            direct_logprob = F.log_softmax(self.direct_head(fused), dim=-1)
            route_prob = F.softmax(self.route_head(fused) / max(float(route_temp), 1e-4), dim=-1)
            op_prob = F.softmax(self.op_head(fused), dim=-1)
            copy_choice_prob = F.softmax(self.copy_head(fused), dim=-1)

            prev_byte = prev[:, t]
            op_targets = apply_ops(prev_byte)
            op_byte_prob = torch.zeros(batch, self.vocab_size, device=x.device)
            op_byte_prob.scatter_add_(1, op_targets, op_prob)

            copy_candidates = recent_bytes[:, t, :]
            copy_prob = torch.zeros(batch, self.vocab_size, device=x.device)
            copy_prob.scatter_add_(1, copy_candidates, copy_choice_prob)

            direct_prob = direct_logprob.exp()
            final_prob = (
                route_prob[:, 0:1] * direct_prob
                + route_prob[:, 1:2] * copy_prob
                + route_prob[:, 2:3] * op_byte_prob
            ).clamp_min(1e-9)

            final_logprob_seq.append(final_prob.log().unsqueeze(1))
            route_prob_seq.append(route_prob.unsqueeze(1))
            op_prob_seq.append(op_prob.unsqueeze(1))
            copy_choice_prob_seq.append(copy_choice_prob.unsqueeze(1))
            direct_logprob_seq.append(direct_logprob.unsqueeze(1))
            reachable_seq.append(op_targets.unsqueeze(1))
            copy_targets_seq.append(copy_candidates.unsqueeze(1))

        return {
            "final_logprob": torch.cat(final_logprob_seq, dim=1),
            "route_prob": torch.cat(route_prob_seq, dim=1),
            "op_prob": torch.cat(op_prob_seq, dim=1),
            "copy_choice_prob": torch.cat(copy_choice_prob_seq, dim=1),
            "direct_logprob": torch.cat(direct_logprob_seq, dim=1),
            "reachable_targets": torch.cat(reachable_seq, dim=1),
            "copy_targets": torch.cat(copy_targets_seq, dim=1),
            "prev_bytes": prev,
        }


def compute_metrics(
    x: torch.Tensor,
    y: torch.Tensor,
    loss_tok: torch.Tensor,
    loss_mask: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    valid = loss_mask > 0
    if not bool(valid.any().item()):
        return {}

    prev = outputs["prev_bytes"][valid]
    dst = y[valid]
    losses = loss_tok[valid]
    route_prob = outputs["route_prob"][valid]
    op_prob = outputs["op_prob"][valid]
    copy_choice_prob = outputs["copy_choice_prob"][valid]
    reachable = outputs["reachable_targets"][valid]
    copy_targets = outputs["copy_targets"][valid]
    target_reachable = reachable.eq(dst.unsqueeze(-1)).any(dim=-1)
    copy_target_mask = copy_targets.eq(dst.unsqueeze(-1))
    copy_target = copy_target_mask.any(dim=-1)
    copy_mass_for_target = (copy_choice_prob * copy_target_mask.float()).sum(dim=-1)
    op_mass_for_target = (op_prob * reachable.eq(dst.unsqueeze(-1)).float()).sum(dim=-1)

    out: Dict[str, float] = {
        "bit_overall_mean_loss": float(losses.mean().item()),
        "target_reachable_by_ops_rate": float(target_reachable.float().mean().item()),
        "target_in_copy_window_rate": float(copy_target.float().mean().item()),
        "mean_route_direct": float(route_prob[:, 0].mean().item()),
        "mean_route_copy_recent": float(route_prob[:, 1].mean().item()),
        "mean_route_bit_op": float(route_prob[:, 2].mean().item()),
        "mean_copy_choice_target_mass": float(copy_mass_for_target.mean().item()),
        "mean_op_target_mass": float(op_mass_for_target.mean().item()),
    }

    if bool(target_reachable.any().item()):
        out["mean_bit_op_route_when_target_reachable"] = float(route_prob[target_reachable, 2].mean().item())
        out["mean_op_target_mass_when_target_reachable"] = float(op_mass_for_target[target_reachable].mean().item())
    if bool((~target_reachable).any().item()):
        out["mean_direct_route_when_target_not_reachable"] = float(route_prob[~target_reachable, 0].mean().item())
    if bool(copy_target.any().item()):
        out["mean_copy_route_when_target_in_copy_window"] = float(route_prob[copy_target, 1].mean().item())
        out["mean_copy_choice_target_mass_when_target_in_copy_window"] = float(copy_mass_for_target[copy_target].mean().item())

    top_routes = route_prob.argmax(dim=-1)
    out["top_route_direct_rate"] = float((top_routes == 0).float().mean().item())
    out["top_route_copy_recent_rate"] = float((top_routes == 1).float().mean().item())
    out["top_route_bit_op_rate"] = float((top_routes == 2).float().mean().item())

    top_ops = op_prob.argmax(dim=-1)
    for i, name in enumerate(OP_NAMES):
        out[f"top_op_{name}_rate"] = float((top_ops == i).float().mean().item())

    classes = classify_transitions(prev.to(dtype=torch.long), dst.to(dtype=torch.long))
    for name, mask in classes.items():
        count = int(mask.sum().item())
        if count <= 0:
            continue
        out[f"bit_{name}_count"] = float(count)
        out[f"bit_{name}_mean_loss"] = float(losses[mask].mean().item())
        out[f"bit_{name}_reachable_rate"] = float(target_reachable[mask].float().mean().item())
    return out


def load_model_from_checkpoint(args: argparse.Namespace, device: torch.device) -> ComputeAndSpeakByteLM:
    checkpoint_path = str(args.checkpoint or args.save_model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = dict(payload.get("config", {}))
    model = ComputeAndSpeakByteLM(
        vocab_size=256,
        d_model=int(cfg.get("d_model", args.d_model)),
        d_state=int(cfg.get("d_state", args.d_state)),
        recent_window=int(cfg.get("recent_window", args.recent_window)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def build_probe_examples(args: argparse.Namespace) -> List[Tuple[str, List[int]]]:
    if str(args.prompt):
        prompt_bytes = list(str(args.prompt).encode("utf-8", errors="replace"))
        return [("prompt", prompt_bytes)]

    stream = build_stream(args)
    examples: List[Tuple[str, List[int]]] = []
    limit = max(1, int(args.probe_rows))
    while len(examples) < limit:
        seq_bytes, _prompt_len = next(stream)
        ids = list(bytes_to_ids(seq_bytes))
        if len(ids) < 2:
            continue
        ids = ids[: max(2, int(args.probe_steps) + 1)]
        examples.append((f"dataset_row_{len(examples) + 1}", ids))
    return examples


def summarize_probe_row(name: str, seq_ids: List[int], outputs: Dict[str, torch.Tensor], limit: int) -> Dict[str, object]:
    x = torch.tensor(seq_ids[:-1], dtype=torch.long, device=outputs["route_prob"].device).unsqueeze(0)
    y = torch.tensor(seq_ids[1:], dtype=torch.long, device=outputs["route_prob"].device).unsqueeze(0)
    steps = min(int(limit), y.shape[1])
    route_prob = outputs["route_prob"][0, :steps]
    op_prob = outputs["op_prob"][0, :steps]
    copy_choice_prob = outputs["copy_choice_prob"][0, :steps]
    direct_logprob = outputs["direct_logprob"][0, :steps]
    reachable = outputs["reachable_targets"][0, :steps]
    copy_targets = outputs["copy_targets"][0, :steps]
    final_logprob = outputs["final_logprob"][0, :steps]

    rows: List[Dict[str, object]] = []
    for t in range(steps):
        target = int(y[0, t].item())
        route_idx = int(route_prob[t].argmax().item())
        op_idx = int(op_prob[t].argmax().item())
        copy_idx = int(copy_choice_prob[t].argmax().item())
        pred_byte = int(final_logprob[t].argmax().item())
        direct_prob_t = direct_logprob[t].exp()
        op_target_mask = reachable[t].eq(target)
        copy_target_mask = copy_targets[t].eq(target)
        direct_target_mass = float(direct_prob_t[target].item())
        op_mass = float((op_prob[t] * op_target_mask.float()).sum().item())
        copy_mass = float((copy_choice_prob[t] * copy_target_mask.float()).sum().item())
        top_direct_byte = int(direct_prob_t.argmax().item())
        top_copy_byte = int(copy_targets[t, copy_idx].item())
        top_op_byte = int(reachable[t, op_idx].item())
        rows.append(
            {
                "t": int(t),
                "src": printable_byte(int(x[0, t].item())),
                "target": printable_byte(target),
                "pred": printable_byte(pred_byte),
                "route": ROUTE_NAMES[route_idx],
                "route_probs": {ROUTE_NAMES[i]: round(float(route_prob[t, i].item()), 4) for i in range(len(ROUTE_NAMES))},
                "top_direct_pred": printable_byte(top_direct_byte),
                "top_op": OP_NAMES[op_idx],
                "top_op_result": printable_byte(top_op_byte),
                "top_copy_offset": int(copy_idx + 1),
                "top_copy_byte": printable_byte(top_copy_byte),
                "target_reachable_by_op": bool(op_target_mask.any().item()),
                "target_in_copy_window": bool(copy_target_mask.any().item()),
                "direct_target_mass": round(direct_target_mass, 4),
                "op_target_mass": round(op_mass, 4),
                "copy_target_mass": round(copy_mass, 4),
            }
        )

    return {
        "name": name,
        "text": decode_bytes(seq_ids[: steps + 1]),
        "steps": rows,
    }


def sample_from_probs(prob: torch.Tensor, temperature: float, top_k: int) -> int:
    scaled = prob.clamp_min(1e-9)
    if float(temperature) != 1.0:
        scaled = scaled.log().div(max(float(temperature), 1e-4)).softmax(dim=-1)
    if int(top_k) > 0 and int(top_k) < scaled.shape[-1]:
        values, indices = torch.topk(scaled, k=int(top_k), dim=-1)
        filtered = torch.zeros_like(scaled)
        filtered.scatter_(0, indices, values)
        scaled = filtered / filtered.sum().clamp_min(1e-9)
    return int(torch.multinomial(scaled, num_samples=1).item())


def generate_bytes(model: ComputeAndSpeakByteLM, prompt_ids: List[int], gen_bytes: int, route_temp: float, sample_temp: float, top_k: int) -> Tuple[List[int], List[Dict[str, object]]]:
    device = next(model.parameters()).device
    generated = list(prompt_ids)
    trace: List[Dict[str, object]] = []
    max_context = 512
    fallback_prompt = [ord(" ")]

    with torch.no_grad():
        for step in range(int(gen_bytes)):
            context = generated[-max_context:]
            if not context:
                context = fallback_prompt
            x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            outputs = model(x, route_temp=float(route_temp))
            route_prob = outputs["route_prob"][0, -1]
            op_prob = outputs["op_prob"][0, -1]
            copy_choice_prob = outputs["copy_choice_prob"][0, -1]
            direct_prob = outputs["direct_logprob"][0, -1].exp()
            final_prob = outputs["final_logprob"][0, -1].exp()
            reachable = outputs["reachable_targets"][0, -1]
            copy_targets = outputs["copy_targets"][0, -1]

            next_byte = sample_from_probs(final_prob, temperature=float(sample_temp), top_k=int(top_k))
            generated.append(next_byte)

            route_idx = int(route_prob.argmax().item())
            op_idx = int(op_prob.argmax().item())
            copy_idx = int(copy_choice_prob.argmax().item())
            trace.append(
                {
                    "step": int(step),
                    "byte": printable_byte(next_byte),
                    "route": ROUTE_NAMES[route_idx],
                    "route_probs": {ROUTE_NAMES[i]: round(float(route_prob[i].item()), 4) for i in range(len(ROUTE_NAMES))},
                    "top_direct_pred": printable_byte(int(direct_prob.argmax().item())),
                    "top_op": OP_NAMES[op_idx],
                    "top_op_result": printable_byte(int(reachable[op_idx].item())),
                    "top_copy_offset": int(copy_idx + 1),
                    "top_copy_byte": printable_byte(int(copy_targets[copy_idx].item())),
                }
            )
    return generated, trace


def run_probe(args: argparse.Namespace) -> None:
    device = torch.device(str(args.device))
    model = load_model_from_checkpoint(args, device)
    examples = build_probe_examples(args)
    summaries: List[Dict[str, object]] = []

    with torch.no_grad():
        for name, seq_ids in examples:
            x = torch.tensor(seq_ids[:-1], device=device, dtype=torch.long).unsqueeze(0)
            outputs = model(x, route_temp=float(args.route_temp))
            summaries.append(summarize_probe_row(name, seq_ids, outputs, int(args.probe_steps)))

    print(json.dumps({"checkpoint": str(args.checkpoint or args.save_model), "probes": summaries}, indent=2))


def run_generate(args: argparse.Namespace) -> None:
    device = torch.device(str(args.device))
    model = load_model_from_checkpoint(args, device)
    prompt_bytes = list(str(args.prompt).encode("utf-8", errors="replace")) if str(args.prompt) else []
    generated, trace = generate_bytes(
        model,
        prompt_ids=prompt_bytes,
        gen_bytes=int(args.gen_bytes),
        route_temp=float(args.route_temp),
        sample_temp=float(args.sample_temp),
        top_k=int(args.top_k),
    )
    prompt_text = decode_bytes(prompt_bytes)
    full_text = decode_bytes(generated)
    payload = {
        "checkpoint": str(args.checkpoint or args.save_model),
        "prompt": prompt_text,
        "completion": full_text[len(prompt_text) :],
        "text": full_text,
        "trace_tail": trace[-min(len(trace), 32) :],
    }
    print(json.dumps(payload, indent=2))


def run_chat(args: argparse.Namespace) -> None:
    device = torch.device(str(args.device))
    model = load_model_from_checkpoint(args, device)
    prompt_text = str(args.prompt)
    if not prompt_text:
        prompt_text = "User: Hello\nAssistant:"
    elif "Assistant:" not in prompt_text:
        prompt_text = f"User: {prompt_text}\nAssistant:"
    prompt_bytes = list(prompt_text.encode("utf-8", errors="replace"))
    generated, _trace = generate_bytes(
        model,
        prompt_ids=prompt_bytes,
        gen_bytes=int(args.gen_bytes),
        route_temp=float(args.route_temp),
        sample_temp=float(args.sample_temp),
        top_k=int(args.top_k),
    )
    full_text = decode_bytes(generated)
    reply = full_text[len(prompt_text) :]
    for stop in ["\nUser:", "\nAssistant:"]:
        if stop in reply:
            reply = reply.split(stop, 1)[0]
    print(reply.strip())


def build_dump_rows(seq_ids: List[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    prev = None
    for idx, value in enumerate(seq_ids):
        row: Dict[str, object] = {
            "i": int(idx),
            "char": printable_byte(int(value)),
            "byte": int(value),
            "hex": f"0x{int(value):02x}",
            "bits": format_bits(int(value)),
        }
        if prev is not None:
            xorv = int(prev) ^ int(value)
            row["xor_from_prev"] = f"0x{xorv:02x}"
            row["xor_bits"] = format_bits(xorv)
        rows.append(row)
        prev = int(value)
    return rows


def build_continuation_report(model: ComputeAndSpeakByteLM, seq_ids: List[int], route_temp: float, top_k: int) -> Dict[str, object]:
    device = next(model.parameters()).device
    context = seq_ids[-512:] if seq_ids else [ord(" ")]
    x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x, route_temp=float(route_temp))
    route_prob = outputs["route_prob"][0, -1]
    op_prob = outputs["op_prob"][0, -1]
    copy_choice_prob = outputs["copy_choice_prob"][0, -1]
    direct_prob = outputs["direct_logprob"][0, -1].exp()
    final_prob = outputs["final_logprob"][0, -1].exp()
    reachable = outputs["reachable_targets"][0, -1]
    copy_targets = outputs["copy_targets"][0, -1]
    k = min(max(int(top_k), 8), int(final_prob.shape[0]))
    vals, idxs = torch.topk(final_prob, k=k, dim=-1)
    candidates: List[Dict[str, object]] = []
    for rank, (value, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        target = int(idx)
        op_target_mask = reachable.eq(target)
        copy_target_mask = copy_targets.eq(target)
        candidates.append(
            {
                "rank": int(rank),
                "char": printable_byte(target),
                "byte": int(target),
                "hex": f"0x{target:02x}",
                "bits": format_bits(target),
                "prob": round(float(value), 6),
                "direct_mass": round(float(direct_prob[target].item()), 6),
                "op_mass": round(float((op_prob * op_target_mask.float()).sum().item()), 6),
                "copy_mass": round(float((copy_choice_prob * copy_target_mask.float()).sum().item()), 6),
                "reachable_by_op": bool(op_target_mask.any().item()),
                "in_copy_window": bool(copy_target_mask.any().item()),
            }
        )
    return {
        "route_probs": {ROUTE_NAMES[i]: round(float(route_prob[i].item()), 4) for i in range(len(ROUTE_NAMES))},
        "top_op": OP_NAMES[int(op_prob.argmax().item())],
        "top_op_result": printable_byte(int(reachable[int(op_prob.argmax().item())].item())),
        "top_copy_offset": int(copy_choice_prob.argmax().item() + 1),
        "top_copy_byte": printable_byte(int(copy_targets[int(copy_choice_prob.argmax().item())].item())),
        "top_candidates": candidates,
    }


def run_dump(args: argparse.Namespace) -> None:
    source_text = str(args.dump_text or args.prompt)
    if not source_text:
        raise SystemExit("Provide --dump-text or --prompt for --mode dump")
    seq_ids = list(source_text.encode("utf-8", errors="replace"))
    rows = build_dump_rows(seq_ids)
    payload: Dict[str, object] = {
        "text": source_text,
        "utf8_len": int(len(seq_ids)),
        "tail_rows": rows[-max(1, int(args.dump_tail)) :],
    }
    checkpoint_path = str(args.checkpoint or args.save_model)
    if checkpoint_path and os.path.exists(checkpoint_path):
        device = torch.device(str(args.device))
        model = load_model_from_checkpoint(args, device)
        payload["continuation_report"] = build_continuation_report(model, seq_ids, route_temp=float(args.route_temp), top_k=int(args.top_k))
    print(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    if str(args.mode) == "probe":
        run_probe(args)
        return
    if str(args.mode) == "generate":
        run_generate(args)
        return
    if str(args.mode) == "chat":
        run_chat(args)
        return
    if str(args.mode) == "dump":
        run_dump(args)
        return
    device = torch.device(str(args.device))
    os.makedirs(os.path.dirname(str(args.out)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(str(args.save_model)) or ".", exist_ok=True)

    model = ComputeAndSpeakByteLM(
        vocab_size=256,
        d_model=int(args.d_model),
        d_state=int(args.d_state),
        recent_window=int(args.recent_window),
    ).to(device)
    opt = AdamW(model.parameters(), lr=float(args.lr))
    stream = build_stream(args)
    history: List[dict] = []
    last_metrics: Dict[str, float] = {}

    for step in range(int(args.steps)):
        x, y, loss_mask = next_batch(stream, args, device)
        outputs = model(x, route_temp=float(args.route_temp))
        final_logprob = outputs["final_logprob"]
        nll = -final_logprob.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        loss = (nll * loss_mask).sum()

        prev = outputs["prev_bytes"]
        reachable_mask = compute_target_op_mask(outputs["reachable_targets"], y)
        op_prob = outputs["op_prob"].clamp_min(1e-9)
        op_mass_for_target = (op_prob * reachable_mask.float()).sum(dim=-1)
        op_aux = -(op_mass_for_target.clamp_min(1e-9).log())
        op_aux = (op_aux * loss_mask * reachable_mask.any(dim=-1).float()).sum()

        copy_choice_prob = outputs["copy_choice_prob"].clamp_min(1e-9)
        copy_target_mask = outputs["copy_targets"].eq(y.unsqueeze(-1))
        copy_target_any = copy_target_mask.any(dim=-1).float()
        copy_choice_mass = (copy_choice_prob * copy_target_mask.float()).sum(dim=-1).clamp_min(1e-9)
        copy_aux = -copy_choice_mass.log() * copy_target_any * loss_mask
        copy_aux = copy_aux.sum()

        route_target = torch.zeros_like(outputs["route_prob"])
        route_target[..., 0] = ((1.0 - reachable_mask.any(dim=-1).float()) * (1.0 - copy_target_any)).clamp_min(0.0)
        route_target[..., 1] = copy_target_any
        route_target[..., 2] = reachable_mask.any(dim=-1).float()
        route_target = route_target / route_target.sum(dim=-1, keepdim=True).clamp_min(1.0)
        route_logprob = outputs["route_prob"].clamp_min(1e-9).log()
        route_aux = -(route_target * route_logprob).sum(dim=-1)
        route_aux = (route_aux * loss_mask).sum()

        total_loss = (
            loss
            + float(args.op_supervision_scale) * op_aux
            + float(args.copy_supervision_scale) * copy_aux
            + float(args.route_supervision_scale) * route_aux
        )

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if int(args.metrics_every) > 0 and ((step + 1) % int(args.metrics_every) == 0 or step == 0):
            with torch.no_grad():
                last_metrics = compute_metrics(x, y, nll.detach(), loss_mask, outputs)

        if (step + 1) % int(args.log_every) == 0 or step == 0:
            row = {
                "step": int(step + 1),
                "loss": float(loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "perplexity": float(math.exp(min(20.0, float(loss.detach().cpu())))),
            }
            row.update(last_metrics)
            history.append(row)
            print(json.dumps(row))

        if int(args.save_every) > 0 and (step + 1) % int(args.save_every) == 0:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": vars(args),
                    "step": int(step + 1),
                },
                str(args.save_model),
            )

    payload = {
        "config": vars(args),
        "history": history,
        "saved_model": str(args.save_model),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args),
            "step": int(args.steps),
        },
        str(args.save_model),
    )
    print(str(Path(args.out).resolve()))


if __name__ == "__main__":
    main()
