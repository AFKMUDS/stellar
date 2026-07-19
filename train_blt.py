from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Iterator, List, Sequence, Tuple

import torch
from torch.optim import AdamW
from tqdm import tqdm

from raplib.blt_model import BLTConfig, ByteLatentTransformer
from raplib.data import bytes_to_ids, iter_raw_byte_sequences, iter_train_sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a minimal BLT-style byte latent transformer")
    p.add_argument("--data", type=str, default="natural_reasoning_train.jsonl")
    p.add_argument("--out", type=str, default="checkpoints/blt_proto.pt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--max_prompt", type=int, default=384)
    p.add_argument("--max_target", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--raw_bytes", action="store_true", help="Treat --data as a raw binary/text corpus instead of JSONL QA")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--metrics_path", type=str, default="")

    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--d_local", type=int, default=192)
    p.add_argument("--d_patch", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--local_layers", type=int, default=2)
    p.add_argument("--global_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--min_patch_len", type=int, default=4)
    p.add_argument("--max_patch_len", type=int, default=32)
    p.add_argument("--entropy_threshold", type=float, default=4.0)
    p.add_argument("--patch_budget", type=int, default=128)
    p.add_argument("--no_bit_embed", action="store_true")
    p.add_argument("--diffusion_noise_std", type=float, default=0.0)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> BLTConfig:
    return BLTConfig(
        d_model=int(args.d_model),
        d_local=int(args.d_local),
        d_patch=int(args.d_patch),
        n_heads=int(args.n_heads),
        local_layers=int(args.local_layers),
        global_layers=int(args.global_layers),
        dropout=float(args.dropout),
        max_seq_len=int(args.seq_len),
        min_patch_len=int(args.min_patch_len),
        max_patch_len=int(args.max_patch_len),
        entropy_threshold=float(args.entropy_threshold),
        patch_budget=int(args.patch_budget),
        use_bit_embed=not bool(args.no_bit_embed),
        diffusion_noise_std=float(args.diffusion_noise_std),
    )


def build_stream(args: argparse.Namespace) -> Iterator[Tuple[bytes, int]]:
    if bool(args.raw_bytes):
        return iter_raw_byte_sequences(args.data, max_total_bytes=int(args.seq_len) + 1)
    return iter_train_sequences(
        jsonl_path=args.data,
        max_prompt_bytes=int(args.max_prompt),
        max_target_bytes=int(args.max_target),
        max_total_bytes=int(args.seq_len) + 1,
    )


def make_batch(stream: Iterator[Tuple[bytes, int]], batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[List[int]] = []
    ys: List[List[int]] = []
    while len(xs) < batch_size:
        seq_bytes, _prompt_len = next(stream)
        ids = list(bytes_to_ids(seq_bytes))
        if len(ids) < 2:
            continue
        ids = ids[: seq_len + 1]
        if len(ids) < seq_len + 1:
            ids = ids + [0] * (seq_len + 1 - len(ids))
        x = ids[:-1]
        y = ids[1:]
        if len(x) < seq_len:
            pad = seq_len - len(x)
            x = x + [0] * pad
            y = y + [-100] * pad
        xs.append(x[:seq_len])
        ys.append(y[:seq_len])
    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )


def cycle_stream(factory_args: argparse.Namespace) -> Iterator[Tuple[bytes, int]]:
    while True:
        for row in build_stream(factory_args):
            yield row


def save_checkpoint(path: str, model: ByteLatentTransformer, optimizer: AdamW, step: int, cfg: BLTConfig) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "cfg": asdict(cfg),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = build_config(args)
    model = ByteLatentTransformer(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    start_step = 0

    if args.resume and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", 0))
        if "cfg" in ckpt:
            valid_keys = BLTConfig.__annotations__.keys()
            filtered_cfg = {k: v for k, v in ckpt["cfg"].items() if k in valid_keys}
            cfg = BLTConfig(**filtered_cfg)
            model.cfg = cfg

    metrics_path = str(args.metrics_path).strip() or f"{args.out}.metrics.jsonl"
    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    stream = cycle_stream(args)
    progress = tqdm(range(start_step, int(args.steps)), initial=start_step, total=int(args.steps))
    for step in progress:
        x, y = make_batch(stream, int(args.batch_size), int(args.seq_len), device)
        model.train()
        out = model(x, y)
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        patch_count = float(out["patch_count"].detach().cpu()) if "patch_count" in out else 0.0
        progress.set_postfix(loss=loss_value, patches=patch_count)

        if (step + 1) % int(args.log_every) == 0 or step == start_step:
            rec = {
                "step": int(step + 1),
                "loss": loss_value,
                "main_loss": float(out["main_loss"].detach().cpu()),
                "prelim_loss": float(out["prelim_loss"].detach().cpu()),
                "patch_count": patch_count,
                "entropy_mean": float(out["entropy"].mean().detach().cpu()),
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        if (step + 1) % int(args.save_every) == 0:
            save_checkpoint(args.out, model, optimizer, step + 1, cfg)

    save_checkpoint(args.out, model, optimizer, int(args.steps), cfg)


if __name__ == "__main__":
    main()
