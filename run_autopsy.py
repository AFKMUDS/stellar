from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import numpy as np

CHECKPOINT_PATH = Path(r"C:\Programming\Programming_Projects\stellar\checkpoints1\rap_v0_step009000.pt")
OUTPUT_DIR = Path("rap_autopsy")
PROMPT = "Question: What is gravity?\n\nAnswer:"
MAX_LEN = 128
TEMPERATURE = 0.8
TOP_K = 50
ASCII_ONLY = False
REPETITION_PENALTY = 1.1
REPETITION_WINDOW = 64
STOP_ON_NEWLINE = True

OUTPUT_DIR.mkdir(exist_ok=True)


def safely_load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# 1. Checkpoint structure
# ---------------------------------------------------------------------------
checkpoint = safely_load_checkpoint(CHECKPOINT_PATH)
print("Checkpoint type:", type(checkpoint))
if isinstance(checkpoint, dict):
    print("Top-level keys:")
    for key in checkpoint:
        print(" ", key, type(checkpoint[key]))
else:
    print("Checkpoint is not a dictionary.")


def find_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()

    if isinstance(obj, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
            "weights",
            "net",
        ):
            value = obj.get(key)
            if isinstance(value, torch.nn.Module):
                return value.state_dict()
            if isinstance(value, dict) and any(
                isinstance(v, torch.Tensor) for v in value.values()
            ):
                return value

        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            return {
                str(k): v
                for k, v in obj.items()
                if isinstance(v, torch.Tensor)
            }

    raise RuntimeError("Could not locate a state_dict in this checkpoint.")


state_dict = find_state_dict(checkpoint)

tensor_report = []
total_parameters = 0

for name, tensor in state_dict.items():
    if not isinstance(tensor, torch.Tensor):
        continue

    detached = tensor.detach().cpu()
    total_parameters += detached.numel()

    row = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": detached.numel(),
    }

    if detached.numel() and (
        detached.is_floating_point() or detached.is_complex()
    ):
        numeric = detached.float()
        row.update(
            {
                "mean": numeric.mean().item(),
                "std": numeric.std(unbiased=False).item(),
                "min": numeric.min().item(),
                "max": numeric.max().item(),
                "zero_fraction": (numeric == 0).float().mean().item(),
            }
        )

    tensor_report.append(row)

report = {
    "checkpoint_path": str(CHECKPOINT_PATH),
    "checkpoint_type": str(type(checkpoint)),
    "top_level_keys": (
        list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    ),
    "total_parameters": total_parameters,
    "tensor_count": len(tensor_report),
    "tensors": tensor_report,
}

with open(OUTPUT_DIR / "checkpoint_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

with open(OUTPUT_DIR / "checkpoint_keys.txt", "w", encoding="utf-8") as f:
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            f.write(f"{key}: {type(value)}\n")

print("Total parameters:", f"{total_parameters:,}")
print("Tensor count:", len(tensor_report))

# ---------------------------------------------------------------------------
# 2. Model definition and config
# ---------------------------------------------------------------------------
shutil.copy("raplib/model.py", OUTPUT_DIR / "model.py")
shutil.copy("raplib/hypercube.py", OUTPUT_DIR / "hypercube.py")
shutil.copy("raplib/pathfinder.py", OUTPUT_DIR / "pathfinder.py")
shutil.copy("raplib/word_map.py", OUTPUT_DIR / "word_map.py")
shutil.copy("raplib/data.py", OUTPUT_DIR / "data.py")
shutil.copy("app.py", OUTPUT_DIR / "generation.py")
shutil.copy("train_rap.py", OUTPUT_DIR / "training_config.py")

# Save config from checkpoint if present
cfg_dict = {}
if isinstance(checkpoint, dict) and "cfg" in checkpoint:
    cfg = checkpoint["cfg"]
    if isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        cfg_dict = {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}

with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(cfg_dict, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# 3. Byte vocabulary
# ---------------------------------------------------------------------------
byte_mapping = {
    "type": "direct_0_255",
    "vocab_size": 256,
    "reserved_bytes": [128, 129, 130],
    "reserved_labels": {
        "128": "<NOUN>",
        "129": "<VERB>",
        "130": "<PRON>",
    },
    "id_to_byte": [int(b) for b in range(256)],
    "byte_to_id": {str(b): b for b in range(256)},
}

with open(OUTPUT_DIR / "byte_mapping.json", "w", encoding="utf-8") as f:
    json.dump(byte_mapping, f, indent=2)

# ---------------------------------------------------------------------------
# 4. Full generation trace
# ---------------------------------------------------------------------------
from raplib.model import ModelConfig, PrimitiveManifoldLM

ckpt = safely_load_checkpoint(CHECKPOINT_PATH)
if isinstance(ckpt, dict) and "cfg" in ckpt:
    cfg_raw = ckpt["cfg"]
    if isinstance(cfg_raw, dict):
        valid_keys = ModelConfig.__annotations__.keys()
        cfg = ModelConfig(**{k: v for k, v in cfg_raw.items() if k in valid_keys})
    else:
        cfg = cfg_raw
    state_dict = ckpt["state_dict"]
else:
    cfg = ModelConfig()
    state_dict = ckpt

model = PrimitiveManifoldLM(cfg)
model.load_state_dict(state_dict)
model.eval()

device = "cpu"
model.to(device)


def _apply_printable_ascii_mask(logits: torch.Tensor) -> torch.Tensor:
    allowed = torch.zeros(256, dtype=torch.bool, device=logits.device)
    allowed[9] = True
    allowed[10] = True
    allowed[13] = True
    allowed[32:127] = True
    masked = logits.clone()
    masked[:, ~allowed] = -float("inf")
    return masked


def _apply_repetition_penalty(logits: torch.Tensor, recent_ids, penalty: float) -> torch.Tensor:
    if not recent_ids or penalty is None or penalty <= 1.0:
        return logits
    out = logits.clone()
    for tid in set(int(x) for x in recent_ids):
        out[:, tid] = out[:, tid] / penalty
    return out


def _sample_next_token(next_logits: torch.Tensor, temperature: float, top_k: int):
    logits = next_logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
        logits = logits.clone()
        logits[logits < v[:, [-1]]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _is_reserved(tok: torch.Tensor) -> torch.Tensor:
    return (tok == 128) | (tok == 129) | (tok == 130)


def generate_trace(
    model: PrimitiveManifoldLM,
    prompt_text: str,
    max_len: int,
    temperature: float,
    top_k: int,
    ascii_only: bool,
    repetition_penalty: float,
    repetition_window: int,
    stop_on_newline: bool,
):
    prompt_ids = list(prompt_text.encode("utf-8"))
    device = next(model.parameters()).device
    state = model.init_state(batch_size=1, device=device)

    retrieval_enabled = bool(getattr(model, "retrieval_enabled", False))
    last_pos = None
    state_hist = None
    if retrieval_enabled:
        last_pos = torch.full((1, 256), -1, device=device, dtype=torch.long)
        state_hist = torch.zeros((1, len(prompt_ids) + int(max_len) + 1, int(model.cfg.d_state)), device=device, dtype=state.dtype)

    telemetry = {
        "retrieval_enabled": bool(retrieval_enabled),
        "retrieval_used": 0,
        "hit_steps": 0,
        "avg_dist_sum": 0.0,
        "hit_seq": [],
        "avg_dist_seq": [],
    }

    trace_rows = []
    out_ids: list[int] = []
    tpos = 0
    next_logits = torch.zeros((1, 256), device=device, dtype=state.dtype)

    with torch.no_grad():
        # Prefill
        for b in prompt_ids:
            tok = torch.tensor([int(b)], device=device, dtype=torch.long)
            retrieval = None
            if retrieval_enabled:
                retrieval = torch.zeros_like(state)
                if tpos > 0 and last_pos is not None and state_hist is not None:
                    is_res = _is_reserved(tok)
                    if bool(getattr(model, "path_retrieval_enabled", False)) and bool((~is_res).any().item()):
                        neigh = model.path_neigh_lut[tok]
                        dist = model.path_dist_lut[tok].to(dtype=state.dtype)
                        neigh_clamped = neigh.clamp(min=0)
                        pos = last_pos.gather(1, neigh_clamped)
                        valid = (neigh >= 0) & (pos >= 0) & (~is_res).unsqueeze(1)
                        if bool(valid.any().item()):
                            pos_clamped = pos.clamp(min=0)
                            got = torch.gather(
                                state_hist,
                                dim=1,
                                index=pos_clamped.unsqueeze(-1).expand(-1, -1, state_hist.shape[-1]),
                            )
                            alpha = torch.tensor(float(getattr(model, "path_alpha", 1.0)), device=device, dtype=state.dtype)
                            w = torch.exp(-alpha * dist) * valid.to(dtype=state.dtype)
                            num = (got * w.unsqueeze(-1)).sum(dim=1)
                            den = torch.clamp(w.sum(dim=1, keepdim=True), min=1.0)
                            retrieval = num / den
                    elif bool(getattr(model, "symmetry_retrieval_enabled", False)) and bool((~is_res).any().item()):
                        inv = tok ^ 0xFF
                        rev = model.bit_reverse_lut[tok]
                        pos_inv = last_pos.gather(1, inv.unsqueeze(1)).squeeze(1)
                        pos_rev = last_pos.gather(1, rev.unsqueeze(1)).squeeze(1)
                        cnt = torch.zeros((1, 1), device=device, dtype=state.dtype)
                        for pos1 in (pos_inv, pos_rev):
                            m = (pos1 >= 0) & (~is_res)
                            if bool(m.any().item()):
                                p = pos1.clamp(min=0)
                                got = state_hist[0, p]
                                retrieval = retrieval + got * m.to(dtype=state.dtype).unsqueeze(1)
                                cnt = cnt + m.to(dtype=state.dtype).unsqueeze(1)
                        retrieval = retrieval / torch.clamp(cnt, min=1.0)

                logits, state = model.step(tok, state, retrieval=retrieval)
                next_logits = logits

                if last_pos is not None and state_hist is not None:
                    state_hist[:, tpos, :] = state.detach()
                    last_pos.scatter_(1, tok.unsqueeze(1), torch.full((1, 1), int(tpos), device=device, dtype=torch.long))
                tpos += 1

        # Generate
        for step in range(int(max_len)):
            logits = next_logits
            if ascii_only:
                logits = _apply_printable_ascii_mask(logits)
            if repetition_penalty and repetition_penalty > 1.0:
                logits = _apply_repetition_penalty(logits, out_ids[-int(repetition_window):], float(repetition_penalty))

            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            topv, topi = torch.topk(probs, k=min(10, probs.size(-1)), dim=-1)
            top10 = [
                {"byte": int(topi[0, j].item()), "prob": float(topv[0, j].item())}
                for j in range(topi.size(1))
            ]

            next_token = _sample_next_token(logits, temperature=temperature, top_k=top_k)
            token_int = int(next_token.item())

            entropy = float((-probs * (probs + 1e-12).log()).sum(dim=-1).item())

            if token_int == 10 and stop_on_newline:
                break

            out_ids.append(token_int)

            tok = torch.tensor([token_int], device=device, dtype=torch.long)
            retrieval = None
            step_hit = False
            step_avg_dist = None
            retrieved_bytes = []
            retrieved_positions = []
            graph_distances = []
            top_neighbors = []

            if retrieval_enabled:
                retrieval = torch.zeros_like(state)
                telemetry["retrieval_used"] += 1
                is_res = _is_reserved(tok)

                if tpos > 0 and last_pos is not None and state_hist is not None and bool((~is_res).any().item()):
                    if bool(getattr(model, "path_retrieval_enabled", False)):
                        neigh = model.path_neigh_lut[tok]
                        dist = model.path_dist_lut[tok].to(dtype=state.dtype)
                        neigh_clamped = neigh.clamp(min=0)
                        pos = last_pos.gather(1, neigh_clamped)
                        valid = (neigh >= 0) & (pos >= 0) & (~is_res).unsqueeze(1)
                        if bool(valid.any().item()):
                            step_hit = True
                            telemetry["hit_steps"] += 1
                            pos_clamped = pos.clamp(min=0)
                            got = torch.gather(
                                state_hist,
                                dim=1,
                                index=pos_clamped.unsqueeze(-1).expand(-1, -1, state_hist.shape[-1]),
                            )
                            alpha = torch.tensor(float(getattr(model, "path_alpha", 1.0)), device=device, dtype=state.dtype)
                            w = torch.exp(-alpha * dist) * valid.to(dtype=state.dtype)
                            num = (got * w.unsqueeze(-1)).sum(dim=1)
                            den = torch.clamp(w.sum(dim=1, keepdim=True), min=1.0)
                            retrieval = num / den

                            wd = (w * dist).sum(dim=1)
                            ws = torch.clamp(w.sum(dim=1), min=1e-6)
                            step_avg_dist = float((wd / ws).item())
                            telemetry["avg_dist_sum"] += float(step_avg_dist)

                            w0 = w[0].detach().float().cpu().numpy()
                            neigh0 = neigh[0].detach().cpu().numpy()
                            pos0 = pos[0].detach().cpu().numpy()
                            top_ix = np.argsort(-w0)
                            for j in top_ix:
                                if w0[j] <= 0.0:
                                    continue
                                nb = int(neigh0[j])
                                if nb < 0:
                                    continue
                                top_neighbors.append({
                                    "byte": nb,
                                    "weight": float(w0[j]),
                                    "graph_distance": float(dist[0, j].item()),
                                    "position": int(pos0[j]),
                                })
                            for j in range(len(pos0)):
                                if pos0[j] >= 0:
                                    nb = int(neigh0[j])
                                    if nb >= 0:
                                        retrieved_bytes.append(nb)
                                        retrieved_positions.append(int(pos0[j]))
                                        graph_distances.append(float(dist[0, j].item()))

                    elif bool(getattr(model, "symmetry_retrieval_enabled", False)):
                        inv = tok ^ 0xFF
                        rev = model.bit_reverse_lut[tok]
                        pos_inv = last_pos.gather(1, inv.unsqueeze(1)).squeeze(1)
                        pos_rev = last_pos.gather(1, rev.unsqueeze(1)).squeeze(1)
                        cnt = torch.zeros((1, 1), device=device, dtype=state.dtype)
                        hit = False
                        for pos1, name in ((pos_inv, "invert"), (pos_rev, "reverse")):
                            m = (pos1 >= 0) & (~is_res)
                            if bool(m.any().item()):
                                hit = True
                                p = pos1.clamp(min=0)
                                got = state_hist[0, p]
                                retrieval = retrieval + got * m.to(dtype=state.dtype).unsqueeze(1)
                                cnt = cnt + m.to(dtype=state.dtype).unsqueeze(1)
                                retrieved_positions.append(int(p.item()))
                                retrieved_bytes.append(int((tok ^ 0xFF).item() if name == "invert" else int(model.bit_reverse_lut[tok].item())))
                                graph_distances.append(0.0)
                        retrieval = retrieval / torch.clamp(cnt, min=1.0)
                        if hit:
                            step_hit = True
                            telemetry["hit_steps"] += 1

            telemetry["hit_seq"].append(bool(step_hit))
            telemetry["avg_dist_seq"].append(None if step_avg_dist is None else float(step_avg_dist))

            logits, state = model.step(tok, state, retrieval=retrieval)
            next_logits = logits

            state_norm = float(torch.linalg.norm(state, dim=-1).item())

            # primitive weights
            x_emb = model._get_emb(tok)
            ctrl_in = torch.cat([x_emb, state, retrieval], dim=-1) if retrieval is not None else torch.cat([x_emb, state], dim=-1)
            w = torch.softmax(model.controller(ctrl_in), dim=-1)
            sA = torch.einsum("b d, k d e -> b k e", state, model.prim_A)
            xB = torch.einsum("b m, k d m -> b k d", x_emb, model.prim_B)
            cand = torch.tanh(sA + xB + model.prim_bias.unsqueeze(0))
            mixed = torch.einsum("b k, b k d -> b d", w, cand)
            gate = model.slice_gate(ctrl_in)
            primitive_weights = [float(x) for x in w[0].detach().cpu().tolist()]
            candidate_norms = [float(x) for x in torch.linalg.norm(cand[0], dim=-1).detach().cpu().tolist()]
            gate_mean = float(gate.mean().item())

            if getattr(model, "delta_enabled", False):
                eps = 1e-8
                H = -(w * (w + eps).log()).sum(dim=-1, keepdim=True)
                logK = torch.log(torch.tensor(float(model.cfg.n_primitives), device=w.device, dtype=w.dtype))
                Hn = H / (logK + eps)
                entropy_norm = float(Hn[0].item())
            else:
                eps = 1e-8
                H = -(w * (w + eps).log()).sum(dim=-1)
                logK = torch.log(torch.tensor(float(model.cfg.n_primitives), device=w.device, dtype=w.dtype))
                entropy_norm = float((H / (logK + eps)).item())

            trace_rows.append({
                "step": step,
                "position": tpos,
                "chosen_byte": token_int,
                "chosen_char": chr(token_int) if 32 <= token_int <= 126 else f"x{token_int:02x}",
                "top10": top10,
                "entropy": entropy,
                "state_norm": state_norm,
                "retrieval_hit": step_hit,
                "retrieved_bytes": retrieved_bytes,
                "retrieved_positions": retrieved_positions,
                "graph_distances": graph_distances,
                "top_neighbors": top_neighbors,
                "primitive_weights": primitive_weights,
                "candidate_norms": candidate_norms,
                "gate_mean": gate_mean,
                "routing_entropy_norm": entropy_norm,
                "source": "retrieval" if step_hit else "generation",
            })

            if retrieval_enabled and last_pos is not None and state_hist is not None:
                state_hist[:, tpos, :] = state.detach()
                last_pos.scatter_(1, tok.unsqueeze(1), torch.full((1, 1), int(tpos), device=device, dtype=torch.long))
            tpos += 1

    return bytes(out_ids), telemetry, trace_rows


prompt_ids = list(PROMPT.encode("utf-8"))

reply_bytes, telemetry, trace_rows = generate_trace(
    model,
    PROMPT,
    max_len=MAX_LEN,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    ascii_only=ASCII_ONLY,
    repetition_penalty=REPETITION_PENALTY,
    repetition_window=REPETITION_WINDOW,
    stop_on_newline=STOP_ON_NEWLINE,
)

generation_report = {
    "prompt": PROMPT,
    "prompt_bytes": prompt_ids,
    "generated_text": reply_bytes.decode("utf-8", errors="replace"),
    "generated_bytes": list(reply_bytes),
    "temperature": TEMPERATURE,
    "top_k": TOP_K,
    "ascii_only": ASCII_ONLY,
    "repetition_penalty": REPETITION_PENALTY,
    "repetition_window": REPETITION_WINDOW,
    "stop_on_newline": STOP_ON_NEWLINE,
    "telemetry": telemetry,
    "trace": trace_rows,
}

with open(OUTPUT_DIR / "generation_trace.json", "w", encoding="utf-8") as f:
    json.dump(generation_report, f, indent=2)

print("Generated text:", reply_bytes.decode("utf-8", errors="replace"))
print("Generated bytes:", len(reply_bytes))
print("Retrieval used:", telemetry["retrieval_used"])
print("Hit steps:", telemetry["hit_steps"])
print("Saved autopsy bundle to:", OUTPUT_DIR)
