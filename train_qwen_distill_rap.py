from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from raplib.data import iter_qa_samples
from raplib.model import ModelConfig, PrimitiveManifoldLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-model", type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--data", type=str, default="natural_reasoning_train.jsonl")
    p.add_argument("--out", type=str, default="checkpoints/rap_qwen_distill.pt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=768)
    p.add_argument("--max_prompt", type=int, default=512)
    p.add_argument("--max_target", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--teacher-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--teacher-dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--teacher-layer", type=int, default=-1)
    p.add_argument("--teacher-hidden-loss", type=float, default=1.0)
    p.add_argument("--teacher-logit-loss", type=float, default=0.0)
    p.add_argument("--ce-loss", type=float, default=1.0)
    p.add_argument("--hidden-loss-kind", type=str, default="cosine_mse", choices=["mse", "cosine", "cosine_mse"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_state", type=int, default=512)
    p.add_argument("--n_primitives", type=int, default=8)
    p.add_argument("--use_bit_linear", action="store_true")
    p.add_argument("--use_symmetry_expansion", action="store_true")
    p.add_argument("--use_symmetry_retrieval", action="store_true")
    p.add_argument("--use_path_retrieval", action="store_true")
    p.add_argument("--path_k", type=int, default=2)
    p.add_argument("--path_alpha", type=float, default=1.0)
    p.add_argument("--use_word_map_retrieval", action="store_true")
    p.add_argument("--word_map_path", type=str, default="")
    p.add_argument("--word_map_topk", type=int, default=32)
    p.add_argument("--word_map_min_count", type=int, default=1)
    p.add_argument("--use_hypercube", action="store_true")
    p.add_argument("--use_numeric_features", action="store_true")
    p.add_argument("--use_aux_numeric", action="store_true")
    p.add_argument("--aux_digit_weight", type=float, default=0.1)
    p.add_argument("--aux_nibble_weight", type=float, default=0.1)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def resolve_dtype(name: str, device: str) -> torch.dtype:
    lowered = str(name).strip().lower()
    if lowered == "float32":
        return torch.float32
    if lowered == "float16":
        return torch.float16
    if lowered == "bfloat16":
        return torch.bfloat16
    if str(device).startswith("cuda"):
        return torch.float16
    return torch.float32


def byte_prefix_offsets(text: str) -> List[int]:
    out = [0]
    total = 0
    for ch in text:
        total += len(ch.encode("utf-8", errors="replace"))
        out.append(total)
    return out


def span_bytes(prefix: Sequence[int], start_char: int, end_char: int) -> Tuple[int, int]:
    start = prefix[max(0, min(int(start_char), len(prefix) - 1))]
    end = prefix[max(0, min(int(end_char), len(prefix) - 1))]
    return int(start), int(end)


def teacher_targets_for_text(
    text: str,
    tokenizer,
    teacher,
    device: str,
    layer: int,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max(2, int(seq_len)),
    )
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = teacher(**encoded, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states
    layer_count = len(hidden_states)
    layer_idx = int(layer if layer >= 0 else layer_count + int(layer))
    layer_idx = max(0, min(layer_count - 1, layer_idx))
    token_hidden = out.hidden_states[layer_idx][0].detach().to("cpu", dtype=torch.float32)
    token_logits = out.logits[0].detach().to("cpu", dtype=torch.float32)
    prefix = byte_prefix_offsets(text)
    text_bytes = text.encode("utf-8", errors="replace")
    max_bytes = min(len(text_bytes), int(seq_len) + 1)
    hidden_dim = int(token_hidden.shape[-1])
    vocab_dim = int(token_logits.shape[-1])
    byte_hidden = torch.zeros((int(seq_len), hidden_dim), dtype=torch.float32)
    byte_logits = torch.zeros((int(seq_len), vocab_dim), dtype=torch.float32)
    byte_mask = torch.zeros((int(seq_len),), dtype=torch.float32)
    for tok_idx, (start_char, end_char) in enumerate(offset_mapping):
        start_b, end_b = span_bytes(prefix, int(start_char), int(end_char))
        if end_b <= start_b:
            continue
        end_b = min(end_b, max_bytes)
        if start_b >= end_b:
            continue
        fill_start = max(0, start_b - 1)
        fill_end = min(int(seq_len), end_b - 1)
        if fill_end <= fill_start:
            continue
        byte_hidden[fill_start:fill_end] = token_hidden[tok_idx]
        byte_logits[fill_start:fill_end] = token_logits[tok_idx]
        byte_mask[fill_start:fill_end] = 1.0
    valid_steps = max(0, min(int(seq_len), max_bytes - 1))
    if valid_steps < int(seq_len):
        byte_mask[valid_steps:] = 0.0
    return byte_hidden, byte_logits, byte_mask, valid_steps


def build_batch(
    samples: Sequence[Tuple[bytes, int, str]],
    seq_len: int,
    tokenizer,
    teacher,
    teacher_device: str,
    teacher_layer: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xs: List[List[int]] = []
    ys: List[List[int]] = []
    ce_masks: List[List[float]] = []
    teacher_hid: List[torch.Tensor] = []
    teacher_log: List[torch.Tensor] = []
    teacher_masks: List[torch.Tensor] = []
    for seq_bytes, prompt_len, text in samples:
        ids = list(seq_bytes[: int(seq_len) + 1])
        eff_len = len(ids)
        if eff_len < 2:
            continue
        if len(ids) < int(seq_len) + 1:
            ids = ids + ([0] * (int(seq_len) + 1 - len(ids)))
        x = ids[: int(seq_len)]
        y = ids[1 : int(seq_len) + 1]
        mask = [0.0] * int(seq_len)
        start = min(int(prompt_len), int(seq_len))
        end = max(0, min(int(seq_len), eff_len - 1))
        for i in range(start, end):
            mask[i] = 1.0
        hid, logit, teacher_mask, _ = teacher_targets_for_text(
            text=text,
            tokenizer=tokenizer,
            teacher=teacher,
            device=teacher_device,
            layer=teacher_layer,
            seq_len=int(seq_len),
        )
        xs.append(x)
        ys.append(y)
        ce_masks.append(mask)
        teacher_hid.append(hid)
        teacher_log.append(logit)
        teacher_masks.append(teacher_mask)
    if not xs:
        raise RuntimeError("batch construction produced no valid samples")
    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(ce_masks, dtype=torch.float32),
        torch.stack(teacher_hid, dim=0),
        torch.stack(teacher_log, dim=0),
        torch.stack(teacher_masks, dim=0),
    )


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def hidden_loss_fn(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, kind: str) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    if kind in {"mse", "cosine_mse"}:
        mse = ((pred - target) ** 2).mean(dim=-1)
        losses.append(masked_mean(mse, mask))
    if kind in {"cosine", "cosine_mse"}:
        cos = 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=1e-8)
        losses.append(masked_mean(cos, mask))
    return sum(losses)


def student_forward_with_states(student: PrimitiveManifoldLM, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = idx.shape
    state = student.init_state(batch_size, idx.device)
    logits_seq: List[torch.Tensor] = []
    state_seq: List[torch.Tensor] = []
    for pos in range(seq_len):
        tok = idx[:, pos]
        x_emb = student._get_emb(tok)
        logits, state = student.step(x_emb, state)
        logits_seq.append(logits.unsqueeze(1))
        state_seq.append(state.unsqueeze(1))
    return torch.cat(logits_seq, dim=1), torch.cat(state_seq, dim=1)


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg = ModelConfig(
        d_model=int(args.d_model),
        d_state=int(args.d_state),
        n_primitives=int(args.n_primitives),
        use_bit_linear=bool(args.use_bit_linear),
        use_symmetry_expansion=bool(args.use_symmetry_expansion),
        use_symmetry_retrieval=bool(args.use_symmetry_retrieval),
        use_path_retrieval=bool(args.use_path_retrieval),
        path_k=int(args.path_k),
        path_alpha=float(args.path_alpha),
        use_word_map_retrieval=bool(args.use_word_map_retrieval),
        word_map_path=str(args.word_map_path),
        word_map_topk=int(args.word_map_topk),
        word_map_min_count=int(args.word_map_min_count),
        use_hypercube=bool(args.use_hypercube),
        use_numeric_features=bool(args.use_numeric_features),
        use_aux_numeric=bool(args.use_aux_numeric),
        aux_digit_weight=float(args.aux_digit_weight),
        aux_nibble_weight=float(args.aux_nibble_weight),
    )

    student = PrimitiveManifoldLM(cfg).to(args.device)
    proj_hidden = None
    proj_logits = None
    opt = None
    start_step = 0

    if bool(args.resume) and os.path.exists(args.out):
        ckpt = torch.load(args.out, map_location=args.device)
        student.load_state_dict(ckpt["student_state_dict"])
        proj_hidden = torch.nn.Linear(int(cfg.d_state), int(ckpt["teacher_hidden_dim"]), bias=False).to(args.device)
        proj_hidden.load_state_dict(ckpt["proj_hidden_state_dict"])
        if "proj_logits_state_dict" in ckpt and ckpt["proj_logits_state_dict"] is not None:
            proj_logits = torch.nn.Linear(int(cfg.vocab_size), int(ckpt["teacher_vocab_dim"]), bias=False).to(args.device)
            proj_logits.load_state_dict(ckpt["proj_logits_state_dict"])
        params: List[torch.nn.Parameter] = list(student.parameters()) + list(proj_hidden.parameters())
        if proj_logits is not None:
            params += list(proj_logits.parameters())
        opt = AdamW(params, lr=float(args.lr))
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", -1)) + 1
    
    teacher_dtype = resolve_dtype(args.teacher_dtype, str(args.teacher_device))
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, trust_remote_code=True, torch_dtype=teacher_dtype)
    teacher.to(args.teacher_device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    sample_stream = iter_qa_samples(jsonl_path=str(args.data), include_answer_prefix=True)

    if proj_hidden is None:
        teacher_hidden_dim = int(teacher.config.hidden_size)
        proj_hidden = torch.nn.Linear(int(cfg.d_state), teacher_hidden_dim, bias=False).to(args.device)
        params = list(student.parameters()) + list(proj_hidden.parameters())
        if float(args.teacher_logit_loss) != 0.0:
            teacher_vocab_dim = int(getattr(teacher.config, "vocab_size", tokenizer.vocab_size))
            proj_logits = torch.nn.Linear(int(cfg.vocab_size), teacher_vocab_dim, bias=False).to(args.device)
            params += list(proj_logits.parameters())
        opt = AdamW(params, lr=float(args.lr))

    assert opt is not None
    assert proj_hidden is not None

    pbar = tqdm(range(start_step, int(args.steps)), desc="distill_qwen_rap")
    for step in pbar:
        batch_rows: List[Tuple[bytes, int, str]] = []
        while len(batch_rows) < int(args.batch_size):
            sample = next(sample_stream)
            prompt = sample.prompt_bytes[: int(args.max_prompt)]
            target = sample.target_bytes[: int(args.max_target)]
            full = (prompt + target)[: int(args.seq_len) + 1]
            prompt_len = min(len(prompt), len(full))
            if len(full) < 2:
                continue
            text = full.decode("utf-8", errors="replace")
            batch_rows.append((full, prompt_len, text))

        x_cpu, y_cpu, ce_mask_cpu, teacher_hidden_cpu, teacher_logits_cpu, teacher_mask_cpu = build_batch(
            samples=batch_rows,
            seq_len=int(args.seq_len),
            tokenizer=tokenizer,
            teacher=teacher,
            teacher_device=str(args.teacher_device),
            teacher_layer=int(args.teacher_layer),
        )

        x = x_cpu.to(args.device)
        y = y_cpu.to(args.device)
        ce_mask = ce_mask_cpu.to(args.device)
        teacher_hidden = teacher_hidden_cpu.to(args.device)
        teacher_logits = teacher_logits_cpu.to(args.device)
        teacher_mask = (teacher_mask_cpu.to(args.device) * ce_mask).clamp(0.0, 1.0)

        student_logits, student_states = student_forward_with_states(student, x)
        student_hidden = proj_hidden(student_states)

        ce_per_tok = F.cross_entropy(student_logits.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="none").reshape_as(y)
        ce_loss = masked_mean(ce_per_tok, ce_mask)

        hid_loss = hidden_loss_fn(student_hidden, teacher_hidden, teacher_mask, str(args.hidden_loss_kind))

        total_loss = float(args.ce_loss) * ce_loss + float(args.teacher_hidden_loss) * hid_loss
        logit_loss_value = None
        if proj_logits is not None and float(args.teacher_logit_loss) != 0.0:
            student_teacher_logits = proj_logits(student_logits)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            student_log_probs = F.log_softmax(student_teacher_logits, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
            logit_loss_value = masked_mean(kl, teacher_mask)
            total_loss = total_loss + (float(args.teacher_logit_loss) * logit_loss_value)

        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(proj_hidden.parameters()) + ([] if proj_logits is None else list(proj_logits.parameters())), 1.0)
        opt.step()

        stats = {
            "loss": float(total_loss.detach().cpu().item()),
            "ce": float(ce_loss.detach().cpu().item()),
            "hidden": float(hid_loss.detach().cpu().item()),
        }
        if logit_loss_value is not None:
            stats["teacher_kl"] = float(logit_loss_value.detach().cpu().item())
        pbar.set_postfix(stats)

        if int(args.save_every) > 0 and (((step + 1) % int(args.save_every)) == 0 or (step + 1) == int(args.steps)):
            torch.save(
                {
                    "step": int(step),
                    "cfg": asdict(cfg),
                    "student_state_dict": student.state_dict(),
                    "proj_hidden_state_dict": proj_hidden.state_dict(),
                    "proj_logits_state_dict": None if proj_logits is None else proj_logits.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "teacher_model": str(args.teacher_model),
                    "teacher_hidden_dim": int(proj_hidden.out_features),
                    "teacher_vocab_dim": None if proj_logits is None else int(proj_logits.out_features),
                    "args": vars(args),
                },
                args.out,
            )


if __name__ == "__main__":
    main()
