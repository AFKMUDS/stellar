from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

from raplib.model import ModelConfig, PrimitiveManifoldLM
from raplib.persistent_model import PersistentRAPConfig, PersistentTwoProgramLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="natural_reasoning_train.jsonl")
    parser.add_argument("--root", default="RobinWilliams")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-prompt", type=int, default=384)
    parser.add_argument("--max-target", type=int, default=192)
    parser.add_argument("--tail-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="RobinWilliams/comparison.json")
    return parser.parse_args()


def tail_records(path: str, byte_count: int, count: int) -> list[dict]:
    size = os.path.getsize(path)
    with open(path, "rb") as stream:
        stream.seek(max(0, size - int(byte_count)))
        if stream.tell() > 0:
            stream.readline()
        lines = stream.read().splitlines()
    records = []
    for line in reversed(lines):
        if not line.strip():
            continue
        records.append(json.loads(line.decode("utf-8")))
        if len(records) >= count:
            break
    records.reverse()
    if len(records) < count:
        raise RuntimeError(f"Only found {len(records)} tail records")
    return records


def encode_record(record: dict, seq_len: int, max_prompt: int, max_target: int):
    prompt = f"Question:\n{record.get('question', '')}\n\nAnswer:\n".encode("utf-8", errors="replace")[:max_prompt]
    target = str(record.get("reference_answer", "")).encode("utf-8", errors="replace")[:max_target]
    full = prompt + target
    ids = list(full[: seq_len + 1])
    effective = len(ids)
    ids += [0] * (seq_len + 1 - len(ids))
    x = ids[:-1]
    y = ids[1:]
    mask = [0.0] * seq_len
    start = min(len(prompt), seq_len)
    end = max(0, effective - 1)
    for position in range(start, end):
        mask[position] = 1.0
    return x, y, mask


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_kind = checkpoint.get("model_kind", "standard")
    if model_kind == "persistent2":
        valid = {field.name for field in fields(PersistentRAPConfig)}
        cfg = PersistentRAPConfig(**{key: value for key, value in checkpoint["cfg"].items() if key in valid})
        model = PersistentTwoProgramLM(cfg)
    else:
        valid = {field.name for field in fields(ModelConfig)}
        cfg = ModelConfig(**{key: value for key, value in checkpoint["cfg"].items() if key in valid})
        model = PrimitiveManifoldLM(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, model_kind, checkpoint


def evaluate(model, model_kind: str, examples: list, batch_size: int, device: torch.device):
    token_loss_sum = 0.0
    token_count = 0.0
    token_correct = 0.0
    elapsed = 0.0
    trace_totals = {
        "switches": 0.0,
        "program_tokens": [0.0, 0.0],
        "retrieval_hits": 0.0,
        "retrieval_norm": 0.0,
        "state_delta_norm": 0.0,
    }

    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            x = torch.tensor([row[0] for row in chunk], device=device, dtype=torch.long)
            y = torch.tensor([row[1] for row in chunk], device=device, dtype=torch.long)
            mask = torch.tensor([row[2] for row in chunk], device=device, dtype=torch.float32)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            if model_kind == "persistent2":
                logits, _, trace = model(x, retrieval_scale=1.0, carrier_gate_strength=0.0, return_trace=True)
            else:
                logits, _, trace = model(x, return_trace=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started

            token_loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").reshape_as(y)
            token_loss_sum += float((token_loss * mask).sum().item())
            token_count += float(mask.sum().item())
            token_correct += float(((logits.argmax(dim=-1) == y).float() * mask).sum().item())

            active = mask > 0
            if model_kind == "persistent2":
                trace_totals["switches"] += float(trace["switch_event"][active].sum().item())
                program = trace["active_program"][active]
                counts = torch.bincount(program, minlength=2)
                trace_totals["program_tokens"][0] += float(counts[0].item())
                trace_totals["program_tokens"][1] += float(counts[1].item())
                trace_totals["retrieval_hits"] += float(trace["retrieval_hit"][active].float().sum().item())
                trace_totals["retrieval_norm"] += float(trace["retrieval_effective_norm"][active].sum().item())
                trace_totals["state_delta_norm"] += float(trace["state_delta_norm"][active].sum().item())
            else:
                program = trace["prim_weights"][active].argmax(dim=-1)
                counts = [float(value) for value in torch.bincount(program, minlength=model.cfg.n_primitives).cpu().tolist()]
                if len(trace_totals["program_tokens"]) != len(counts):
                    trace_totals["program_tokens"] = [0.0] * len(counts)
                trace_totals["program_tokens"] = [total + value for total, value in zip(trace_totals["program_tokens"], counts)]

    result = {
        "cross_entropy": token_loss_sum / max(token_count, 1.0),
        "perplexity": float(torch.exp(torch.tensor(token_loss_sum / max(token_count, 1.0))).item()),
        "accuracy": token_correct / max(token_count, 1.0),
        "answer_tokens": int(token_count),
        "seconds": elapsed,
        "tokens_per_second": token_count / max(elapsed, 1e-8),
        "program_counts": [int(value) for value in trace_totals["program_tokens"]],
    }
    if model_kind == "persistent2":
        result.update(
            {
                "switch_rate": trace_totals["switches"] / max(token_count, 1.0),
                "mean_program_duration": token_count / max(trace_totals["switches"], 1.0),
                "retrieval_hit_rate": trace_totals["retrieval_hits"] / max(token_count, 1.0),
                "mean_retrieval_norm": trace_totals["retrieval_norm"] / max(token_count, 1.0),
                "mean_state_delta_norm": trace_totals["state_delta_norm"] / max(token_count, 1.0),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_paths = {
        "A": Path(args.root) / "A" / "rap8.pt",
        "B": Path(args.root) / "B" / "rap2.pt",
        "C": Path(args.root) / "C" / "persistent2.pt",
        "D": Path(args.root) / "D" / "persistent2_staged.pt",
    }
    records = tail_records(args.data, args.tail_bytes, args.samples)
    examples = [encode_record(record, args.seq_len, args.max_prompt, args.max_target) for record in records]
    report = {
        "evaluation_source": "tail of training JSONL, outside the prefix consumed by the 2500-step runs",
        "sample_count": len(examples),
        "seq_len": args.seq_len,
        "results": {},
    }

    for name, path in checkpoint_paths.items():
        model, model_kind, checkpoint = load_model(path, device)
        result = evaluate(model, model_kind, examples, args.batch_size, device)
        result.update(
            {
                "checkpoint": str(path),
                "step": int(checkpoint["step"]) + 1,
                "model_kind": model_kind,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
        )
        report["results"][name] = result
        print(name, json.dumps(result, sort_keys=True))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
