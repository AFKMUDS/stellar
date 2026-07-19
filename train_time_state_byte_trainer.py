from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from raplib.data import bytes_to_ids, iter_train_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level model with explicit time and state weighting.")
    parser.add_argument("--data", type=str, default="natural_reasoning_train.jsonl")
    parser.add_argument("--out", type=str, default="checkpoints/time_state_byte_trainer.json")
    parser.add_argument("--save-model", type=str, default="checkpoints/time_state_byte_trainer.pt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-prompt", type=int, default=384)
    parser.add_argument("--max-target", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--d-state", type=int, default=384)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--state-weight-scale", type=float, default=1.0)
    parser.add_argument("--time-weight-scale", type=float, default=1.0)
    parser.add_argument("--bit-metrics-every", type=int, default=100)
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


class TimeStateByteLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_state: int, seq_len: int, time_dim: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.seq_len = int(seq_len)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.time_emb = nn.Embedding(seq_len, time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim + 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cand_proj = nn.Sequential(
            nn.Linear(d_model * 2 + d_state, d_state),
            nn.GELU(),
            nn.Linear(d_state, d_state),
            nn.Tanh(),
        )
        self.mix_gate = nn.Linear(d_model * 2 + d_state, d_state)
        self.state_strength = nn.Linear(d_model * 2 + d_state, 1)
        self.time_strength = nn.Linear(d_model * 2 + d_state, 1)
        self.out = nn.Linear(d_model + d_state, vocab_size)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_state, device=device)

    def forward(self, x: torch.Tensor, state_weight_scale: float, time_weight_scale: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, steps = x.shape
        tok = self.token_emb(x)
        pos = torch.arange(steps, device=x.device, dtype=torch.long)
        pos_norm = pos.to(dtype=torch.float32) / max(1, steps - 1)
        pos_feats = torch.stack(
            [
                pos_norm,
                torch.sin(pos_norm * math.pi),
                torch.cos(pos_norm * math.pi),
            ],
            dim=-1,
        )
        time_in = torch.cat([self.time_emb(pos), pos_feats], dim=-1)
        time_hidden = self.time_proj(time_in).unsqueeze(0).expand(batch, -1, -1)

        state = self.init_state(batch, x.device)
        logits_seq: List[torch.Tensor] = []
        state_strength_seq: List[torch.Tensor] = []
        time_strength_seq: List[torch.Tensor] = []
        state_norm_seq: List[torch.Tensor] = []
        for t in range(steps):
            tok_t = tok[:, t, :]
            time_t = time_hidden[:, t, :]
            ctrl_in = torch.cat([tok_t, time_t, state], dim=-1)
            cand = self.cand_proj(ctrl_in)
            mix = torch.sigmoid(self.mix_gate(ctrl_in))
            state_strength = torch.sigmoid(self.state_strength(ctrl_in)) * float(state_weight_scale)
            time_strength = torch.sigmoid(self.time_strength(ctrl_in)) * float(time_weight_scale)
            state = (1.0 - mix) * state + mix * cand
            projected_time = time_t[:, : self.d_state] if time_t.shape[-1] >= self.d_state else F.pad(time_t, (0, self.d_state - time_t.shape[-1]))
            state = state * state_strength + projected_time * time_strength
            logits = self.out(torch.cat([tok_t, state], dim=-1))
            logits_seq.append(logits.unsqueeze(1))
            state_strength_seq.append(state_strength)
            time_strength_seq.append(time_strength)
            state_norm_seq.append(state.norm(dim=-1, keepdim=True))
        aux = {
            "state_strength": torch.cat(state_strength_seq, dim=1),
            "time_strength": torch.cat(time_strength_seq, dim=1),
            "state_norm": torch.cat(state_norm_seq, dim=1),
        }
        return torch.cat(logits_seq, dim=1), aux


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


def compute_bit_metrics(x: torch.Tensor, y: torch.Tensor, loss_tok: torch.Tensor, loss_mask: torch.Tensor) -> Dict[str, float]:
    valid = loss_mask > 0
    if not bool(valid.any().item()):
        return {}
    src = x[valid].to(dtype=torch.long)
    dst = y[valid].to(dtype=torch.long)
    losses = loss_tok[valid]
    classes = classify_transitions(src, dst)
    out: Dict[str, float] = {}
    for name, mask in classes.items():
        count = int(mask.sum().item())
        if count <= 0:
            continue
        out[f"bit_{name}_count"] = float(count)
        out[f"bit_{name}_mean_loss"] = float(losses[mask].mean().item())
    out["bit_overall_mean_loss"] = float(losses.mean().item())
    return out


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(str(args.device))
    os.makedirs(os.path.dirname(str(args.out)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(str(args.save_model)) or ".", exist_ok=True)

    model = TimeStateByteLM(vocab_size=256, d_model=int(args.d_model), d_state=int(args.d_state), seq_len=int(args.seq_len), time_dim=int(args.time_dim)).to(device)
    opt = AdamW(model.parameters(), lr=float(args.lr))
    stream = build_stream(args)
    history: List[dict] = []
    last_bit_metrics: Dict[str, float] = {}

    for step in range(int(args.steps)):
        x, y, loss_mask = next_batch(stream, args, device)
        logits, aux = model(
            x,
            state_weight_scale=float(args.state_weight_scale),
            time_weight_scale=float(args.time_weight_scale),
        )
        loss_per_tok = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="none").reshape_as(y)
        loss = (loss_per_tok * loss_mask).sum()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if int(args.bit_metrics_every) > 0 and (step + 1) % int(args.bit_metrics_every) == 0:
            with torch.no_grad():
                last_bit_metrics = compute_bit_metrics(x, y, loss_per_tok.detach(), loss_mask)

        if (step + 1) % int(args.log_every) == 0 or step == 0:
            row = {
                "step": int(step + 1),
                "loss": float(loss.detach().cpu()),
                "perplexity": float(math.exp(min(20.0, float(loss.detach().cpu())))),
                "mean_state_strength": float(aux["state_strength"].mean().detach().cpu()),
                "mean_time_strength": float(aux["time_strength"].mean().detach().cpu()),
                "mean_state_norm": float(aux["state_norm"].mean().detach().cpu()),
            }
            row.update(last_bit_metrics)
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
