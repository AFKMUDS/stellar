import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from raplib.model import ModelConfig, PrimitiveManifoldLM
from raplib.hypercube import get_cube_permutations
from raplib.pathfinder import build_k_hop_tables
import os
import re
import json
import ast
import time
import html


def _sample_next_token(next_logits, temperature: float, top_k: int):
    logits = next_logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
        logits = logits.clone()
        logits[logits < v[:, [-1]]] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _format_bytes_for_display(b: bytes, mode: str) -> str:
    if mode == "UTF-8 (replace errors)":
        return b.decode("utf-8", errors="replace")

    out = []
    for x in b:
        if x == 10:
            out.append("\\n")
        elif x == 9:
            out.append("\\t")
        elif x == 13:
            out.append("\\r")
        elif 32 <= x <= 126:
            out.append(chr(x))
        else:
            out.append(f"\\x{x:02x}")
    return "".join(out)


def _byte_label(b: int) -> str:
    x = int(b) & 0xFF
    if x == 32:
        return "(sp)"
    if 32 <= x <= 126:
        return f"'{chr(x)}'"
    return f"x{x:02x}"


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


_TAG_LABELS = {
    128: "<NOUN>",
    129: "<VERB>",
    130: "<PRON>",
}


_TAG_BYTES_BY_TEXT = {
    "<NOUN>": 128,
    "<VERB>": 129,
    "<PRON>": 130,
}


def _apply_tag_display(b: bytes, tag_display: str) -> bytes:
    if tag_display == "Hide tags":
        return bytes(x for x in b if int(x) not in _TAG_LABELS)
    if tag_display == "Render <TAGS>":
        out = bytearray()
        for x in b:
            lab = _TAG_LABELS.get(int(x))
            if lab is not None:
                out.extend(lab.encode("ascii", errors="replace"))
            else:
                out.append(int(x) & 0xFF)
        return bytes(out)
    return b


_ESC_BYTE_RE = re.compile(r"\\x([0-9a-fA-F]{2})")


def _parse_escaped_or_tagged_bytes(s: str) -> list[int]:
    # Supports:
    # - Literal text (ASCII recommended)
    # - Hex escapes like \x81
    # - Symbolic tags like <VERB>
    ids: list[int] = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 3 < len(s) and s[i + 1] == "x":
            m = _ESC_BYTE_RE.match(s, i)
            if m:
                ids.append(int(m.group(1), 16))
                i = m.end()
                continue

        if s[i] == "<":
            j = s.find(">", i + 1)
            if j != -1:
                tok = s[i : j + 1]
                if tok in _TAG_BYTES_BY_TEXT:
                    ids.append(int(_TAG_BYTES_BY_TEXT[tok]))
                    i = j + 1
                    continue

        # Fallback: treat as UTF-8 text and add the single byte if possible.
        # This is mainly for quick tests; for arbitrary unicode, UTF-8 may emit multi-byte sequences.
        b = s[i].encode("utf-8", errors="replace")
        ids.extend(list(b))
        i += 1
    return ids

# Page Config
st.set_page_config(page_title="RAP Hypercube Explorer", layout="wide")

# --- Helper Functions ---

@st.cache_resource
def load_model(ckpt_path, device="cpu", *, cfg_override: dict | None = None):
    if not os.path.exists(ckpt_path):
        return None
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if isinstance(ckpt, dict) and "cfg" in ckpt:
        cfg_dict = ckpt["cfg"]
        # Filter for forward compatibility
        valid_keys = ModelConfig.__annotations__.keys()
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = ModelConfig(**filtered_cfg)
        state_dict = ckpt["state_dict"]
    else:
        cfg = ModelConfig()
        state_dict = ckpt

    cfg_base = cfg
    cfg_effective = cfg_base
    if cfg_override:
        cfg_effective = ModelConfig(**{k: getattr(cfg_base, k) for k in ModelConfig.__annotations__.keys()})
        for k, v in dict(cfg_override).items():
            if k in ModelConfig.__annotations__:
                setattr(cfg_effective, k, v)

    try:
        model = PrimitiveManifoldLM(cfg_effective)
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Fallback to checkpoint-native config when runtime overrides changed architecture.
        # This avoids hard-failing app startup when toggles are incompatible with a ckpt.
        model = PrimitiveManifoldLM(cfg_base)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def _build_chat_prompt(messages):
    parts = []
    for m in messages:
        if m.get("role") == "user":
            parts.append(f"Question: {m.get('content','')}\n")
        elif m.get("role") == "assistant":
            parts.append(f"Answer: {m.get('content','')}\n")
    return "".join(parts)


def generate_chat_turn(
    model,
    prompt_text: str,
    temperature: float,
    top_k: int,
    max_len: int,
    *,
    ascii_only: bool = False,
    repetition_penalty: float = 1.0,
    repetition_window: int = 64,
    stop_on_newline: bool = True,
    enable_retrieval: bool = True,
    enable_backspace_solver: bool = False,
    solver_lookahead: int = 12,
    solver_topk: int = 8,
    solver_min_commit: int = 12,
    solver_commit_score: float = -2.0,
    solver_drop_delta: float = 0.8,
    solver_fail_threshold: int = 3,
    solver_hint_len: int = 4,
    solver_hint_beam_width: int = 8,
    solver_hint_expand_topk: int = 8,
    solver_hint_target_prefix_len: int = 24,
    solver_max_rewinds: int = 64,
    solver_max_hints: int = 6,
    stream_callback=None,
    stream_every: int = 1,
    stream_min_interval_s: float = 0.05,
    stream_sleep_s: float = 0.0,
    stream_backspace_sleep_s: float = 0.0,
):
    device = next(model.parameters()).device
    prompt_ids = list(prompt_text.encode("utf-8"))

    state = model.init_state(batch_size=1, device=device)
    # Solver mode recomputes prefix state via model.forward() and expects deterministic
    # scoring; disable retrieval inside the solver to avoid mismatch with app-level
    # retrieval buffers.
    retrieval_enabled = (
        (not bool(enable_backspace_solver))
        and bool(enable_retrieval)
        and bool(getattr(model, "retrieval_enabled", False))
    )
    last_pos = None
    state_hist = None
    if retrieval_enabled:
        last_pos = torch.full((1, 256), -1, device=device, dtype=torch.long)
        state_hist = torch.zeros((1, len(prompt_ids) + int(max_len) + 1, int(model.cfg.d_state)), device=device, dtype=state.dtype)

    def _is_reserved(tok: torch.Tensor) -> torch.Tensor:
        return (tok == 128) | (tok == 129) | (tok == 130)

    telemetry = {
        "retrieval_enabled": bool(retrieval_enabled),
        "steps": 0,
        "stopped_immediately": False,
        "stop_reason": "",
        "retrieval_used": 0,
        "hit_steps": 0,
        "avg_dist_sum": 0.0,
        "top_neighbors": [],
        "hit_seq": [],
        "avg_dist_seq": [],
        "solver_enabled": bool(enable_backspace_solver),
        "solver_rewinds": 0,
        "solver_commits": 0,
        "solver_hints": 0,
        "solver_trace": [],
    }

    def _trace(s: str) -> None:
        if not bool(enable_backspace_solver):
            return
        tr = telemetry.get("solver_trace")
        if isinstance(tr, list):
            if len(tr) < 5000:
                tr.append(str(s))

    last_stream_t = 0.0
    stream_steps = 0

    def _maybe_stream(*, out_bytes: bytes, force: bool = False) -> None:
        nonlocal last_stream_t, stream_steps
        cb = stream_callback
        if cb is None:
            return
        stream_steps += 1
        if not force:
            if int(stream_every) > 1 and (stream_steps % int(stream_every)) != 0:
                return
            now = time.time()
            if float(stream_min_interval_s) > 0 and (now - float(last_stream_t)) < float(stream_min_interval_s):
                return
            last_stream_t = float(now)
        try:
            cb(bytes(out_bytes))
            if float(stream_sleep_s) > 0:
                time.sleep(float(stream_sleep_s))
        except Exception:
            return

    def _hash_anchor(prefix_ids: list[int]) -> int:
        # Cheap deterministic hash for an anchor (bytes matter, order matters).
        # 64-bit FNV-1a.
        h = 1469598103934665603
        for b in prefix_ids:
            h ^= int(b) & 0xFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return int(h)

    def _rollout_greedy_score(*, prefix_ids: list[int], steps: int) -> float:
        # Deterministic greedy rollout score: average logprob of chosen tokens.
        # Uses model.step() with explicit zero retrieval (if the model expects retrieval)
        # so solver behavior is deterministic and shape-consistent.
        if steps <= 0:
            return 0.0

        st = model.init_state(batch_size=1, device=device)
        next_logits = None
        with torch.no_grad():
            if len(prefix_ids) == 0:
                # Seed: feed a 0 byte.
                tok0 = torch.tensor([0], device=device, dtype=torch.long)
                if bool(getattr(model, "retrieval_enabled", False)):
                    retrieval0 = torch.zeros_like(st)
                    next_logits, st = model.step(tok0, st, retrieval=retrieval0)
                else:
                    next_logits, st = model.step(tok0, st)
            else:
                for b in prefix_ids:
                    tok = torch.tensor([int(b)], device=device, dtype=torch.long)
                    if bool(getattr(model, "retrieval_enabled", False)):
                        retrieval0 = torch.zeros_like(st)
                        next_logits, st = model.step(tok, st, retrieval=retrieval0)
                    else:
                        next_logits, st = model.step(tok, st)

            if next_logits is None:
                return 0.0

            total = 0.0
            for _ in range(int(steps)):
                logp = torch.log_softmax(next_logits, dim=-1)
                tid = int(torch.argmax(logp, dim=-1).item())
                total += float(logp[0, tid].item())
                tok = torch.tensor([int(tid)], device=device, dtype=torch.long)
                if bool(getattr(model, "retrieval_enabled", False)):
                    retrieval0 = torch.zeros_like(st)
                    next_logits, st = model.step(tok, st, retrieval=retrieval0)
                else:
                    next_logits, st = model.step(tok, st)

        return float(total) / float(max(1, int(steps)))

    def _recompute_state_after_prefix(prefix_ids: list[int]):
        # Deterministically recompute (next_logits, state) for the given prefix.
        st = model.init_state(batch_size=1, device=device)
        next_logits = torch.zeros((1, 256), device=device, dtype=st.dtype)
        with torch.no_grad():
            if len(prefix_ids) == 0:
                tok0 = torch.tensor([0], device=device, dtype=torch.long)
                if bool(getattr(model, "retrieval_enabled", False)):
                    retrieval0 = torch.zeros_like(st)
                    next_logits, st = model.step(tok0, st, retrieval=retrieval0)
                else:
                    next_logits, st = model.step(tok0, st)
            else:
                for b in prefix_ids:
                    tok = torch.tensor([int(b)], device=device, dtype=torch.long)
                    if bool(getattr(model, "retrieval_enabled", False)):
                        retrieval0 = torch.zeros_like(st)
                        next_logits, st = model.step(tok, st, retrieval=retrieval0)
                    else:
                        next_logits, st = model.step(tok, st)
        return next_logits, st

    def _mini_inverse_hint(*, prefix_bytes: bytes, target_text: str) -> bytes:
        # Use existing inverse search to find a short bridge (ASCII-only) that makes
        # target_text likely after prefix_bytes.
        try:
            prefix_text = prefix_bytes.decode("utf-8", errors="replace")
            best_ids, _best_score = beam_search_prompt_for_target(
                model,
                target_text,
                prefix_text=prefix_text,
                prompt_len=int(len(prefix_text) + int(solver_hint_len)),
                beam_width=int(solver_hint_beam_width),
                expand_topk=int(solver_hint_expand_topk),
                ascii_only=True,
                target_prefix_len=int(solver_hint_target_prefix_len),
            )
            # Extract the suffix beyond prefix_text.
            prefix_ids = list(prefix_text.encode("utf-8"))
            if len(best_ids) <= len(prefix_ids):
                return b""
            bridge = bytes(int(b) & 0xFF for b in best_ids[len(prefix_ids) :])
            return bridge[: max(0, int(solver_hint_len))]
        except Exception:
            return b""

    with torch.no_grad():
        # Prefill prompt incrementally so retrieval memory is populated.
        tpos = 0
        next_logits = torch.zeros((1, 256), device=device, dtype=state.dtype)
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

                if retrieval_enabled and last_pos is not None and state_hist is not None:
                    state_hist[:, tpos, :] = state.detach()
                    last_pos.scatter_(1, tok.unsqueeze(1), torch.full((1, 1), int(tpos), device=device, dtype=torch.long))
                tpos += 1

            # Generate
            out_ids: list[int] = []

            # Stream the empty prefix so the UI placeholder is created consistently.
            _maybe_stream(out_bytes=b"", force=True)

            committed: list[int] = []
            draft: list[int] = []
            banned_next: dict[int, set[int]] = {}
            fail_count: dict[int, int] = {}
            committed_best_health = -1e9
            health = -1e9
            hints_used = 0
            rewinds = 0

            for _ in range(int(max_len)):
                logits = next_logits
                if ascii_only:
                    logits = _apply_printable_ascii_mask(logits)
                if repetition_penalty and repetition_penalty > 1.0:
                    logits = _apply_repetition_penalty(logits, out_ids[-int(repetition_window) :], float(repetition_penalty))

                token_int: int
                if not bool(enable_backspace_solver):
                    next_token = _sample_next_token(logits, temperature=temperature, top_k=top_k)
                    token_int = int(next_token.item())
                else:
                    prefix_ids = prompt_ids + committed + draft
                    anchor_id = _hash_anchor(prefix_ids)
                    banned = banned_next.setdefault(anchor_id, set())

                    _trace(f"anchor={anchor_id:016x} len={len(prefix_ids)} banned={len(banned)}")

                    logp = torch.log_softmax(logits, dim=-1)
                    k = max(1, int(solver_topk))
                    _v, idx = torch.topk(logp, k=min(k, logp.size(-1)))
                    candidates = [int(x) for x in idx[0].tolist() if int(x) not in banned]

                    chosen = None
                    chosen_score = None
                    best_since_commit = float(committed_best_health)

                    for cand in candidates:
                        score = _rollout_greedy_score(prefix_ids=prefix_ids + [int(cand)], steps=int(solver_lookahead))
                        if committed_best_health <= -1e8:
                            committed_best_health = float(score)
                            best_since_commit = float(score)
                        drift = bool(float(score) < float(best_since_commit) - float(solver_drop_delta))
                        if not drift:
                            chosen = int(cand)
                            chosen_score = float(score)
                            _trace(f"choose {chosen}({_byte_label(chosen)}) score={float(score):.3f}")
                            break
                        _trace(f"ban {int(cand)}({_byte_label(int(cand))}) score={float(score):.3f}")
                        banned.add(int(cand))

                    if chosen is None:
                        fc = int(fail_count.get(anchor_id, 0)) + 1
                        fail_count[anchor_id] = fc
                        _trace(f"fail anchor={anchor_id:016x} count={fc}")

                        if fc >= int(solver_fail_threshold) and hints_used < int(solver_max_hints):
                            try:
                                x = torch.tensor([prefix_ids], device=device, dtype=torch.long)
                                out = model(x, state=None)
                                logits_seq, st = out[0], out[1]
                                nl = logits_seq[:, -1, :]
                                tgt_ids = []
                                for _i in range(int(solver_hint_target_prefix_len)):
                                    lp = torch.log_softmax(nl, dim=-1)
                                    tid = int(torch.argmax(lp, dim=-1).item())
                                    tgt_ids.append(tid)
                                    xt = torch.tensor([[tid]], device=device, dtype=torch.long)
                                    out = model(xt, st)
                                    logits_seq, st = out[0], out[1]
                                    nl = logits_seq[:, -1, :]
                                target_text = bytes(int(b) & 0xFF for b in tgt_ids).decode("utf-8", errors="replace")
                            except Exception:
                                target_text = ""

                            bridge = _mini_inverse_hint(prefix_bytes=bytes(prefix_ids), target_text=target_text)
                            if bridge:
                                hints_used += 1
                                telemetry["solver_hints"] = int(hints_used)
                                _trace(f"hint bytes={list(bridge)}")
                                for bb in bridge:
                                    out_ids.append(int(bb))
                                    draft.append(int(bb))
                                    _maybe_stream(out_bytes=bytes(out_ids), force=True)
                                    if float(stream_sleep_s) > 0:
                                        time.sleep(float(stream_sleep_s))
                                    tok = torch.tensor([int(bb)], device=device, dtype=torch.long)
                                    logits2, state2 = model.step(tok, state)
                                    state = state2
                                    next_logits = logits2
                                    telemetry["steps"] += 1
                                fail_count[anchor_id] = 0
                                continue

                        if draft and rewinds < int(solver_max_rewinds):
                            bad_byte = int(draft.pop())
                            out_ids.pop()
                            _maybe_stream(out_bytes=bytes(out_ids), force=True)
                            if float(stream_backspace_sleep_s) > 0:
                                time.sleep(float(stream_backspace_sleep_s))
                            rewinds += 1
                            telemetry["solver_rewinds"] = int(rewinds)
                            _trace(f"rewind bad={bad_byte}({_byte_label(bad_byte)}) rewinds={rewinds}")
                            prev_prefix = prompt_ids + committed + draft
                            prev_anchor = _hash_anchor(prev_prefix)
                            banned_next.setdefault(prev_anchor, set()).add(int(bad_byte))
                            next_logits, state = _recompute_state_after_prefix(prev_prefix)
                            continue

                        chosen = int(torch.argmax(logp, dim=-1).item())
                        chosen_score = float(_rollout_greedy_score(prefix_ids=prefix_ids + [chosen], steps=int(solver_lookahead)))

                    token_int = int(chosen)
                    if chosen_score is not None:
                        health = float(chosen_score) if health <= -1e8 else (0.8 * float(health) + 0.2 * float(chosen_score))
                        committed_best_health = max(float(committed_best_health), float(chosen_score))

                if token_int == 10 and bool(stop_on_newline):
                    if not out_ids:
                        telemetry["stopped_immediately"] = True
                        telemetry["stop_reason"] = "sampled_newline_first"
                    break
                out_ids.append(token_int)
                _maybe_stream(out_bytes=bytes(out_ids), force=False)

                if bool(enable_backspace_solver):
                    draft.append(int(token_int))
                    if len(draft) >= int(solver_min_commit):
                        if float(committed_best_health) >= float(solver_commit_score):
                            committed.extend(draft)
                            draft.clear()
                            telemetry["solver_commits"] = int(telemetry.get("solver_commits", 0)) + 1
                            _trace(f"commit bytes={len(committed)} commits={int(telemetry.get('solver_commits', 0))}")

                tok = torch.tensor([token_int], device=device, dtype=torch.long)
                retrieval = None
                step_hit = False
                step_avg_dist = None
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
                                top_ix = np.argsort(-w0)[:5]
                                top = []
                                for j in top_ix:
                                    if w0[j] <= 0.0:
                                        continue
                                    nb = int(neigh0[j])
                                    if nb < 0:
                                        continue
                                    top.append((nb, float(w0[j]), float(dist[0, j].item())))
                                telemetry["top_neighbors"].append(top)

                        elif bool(getattr(model, "symmetry_retrieval_enabled", False)):
                            inv = tok ^ 0xFF
                            rev = model.bit_reverse_lut[tok]
                            pos_inv = last_pos.gather(1, inv.unsqueeze(1)).squeeze(1)
                            pos_rev = last_pos.gather(1, rev.unsqueeze(1)).squeeze(1)
                            cnt = torch.zeros((1, 1), device=device, dtype=state.dtype)
                            hit = False
                            for pos1 in (pos_inv, pos_rev):
                                m = (pos1 >= 0) & (~is_res)
                                if bool(m.any().item()):
                                    hit = True
                                    p = pos1.clamp(min=0)
                                    got = state_hist[0, p]
                                    retrieval = retrieval + got * m.to(dtype=state.dtype).unsqueeze(1)
                                    cnt = cnt + m.to(dtype=state.dtype).unsqueeze(1)
                            retrieval = retrieval / torch.clamp(cnt, min=1.0)
                            if hit:
                                step_hit = True
                                telemetry["hit_steps"] += 1

                telemetry["hit_seq"].append(bool(step_hit))
                telemetry["avg_dist_seq"].append(None if step_avg_dist is None else float(step_avg_dist))

                logits, state = model.step(tok, state, retrieval=retrieval)
                next_logits = logits
                telemetry["steps"] += 1

                if retrieval_enabled and last_pos is not None and state_hist is not None:
                    state_hist[:, tpos, :] = state.detach()
                    last_pos.scatter_(1, tok.unsqueeze(1), torch.full((1, 1), int(tpos), device=device, dtype=torch.long))
                tpos += 1

            return bytes(out_ids), telemetry


def get_cube_vertices(byte_val):
    # Map index 0..7 to (x,y,z) in {-1, 1}
    # i & 4 -> x, i & 2 -> y, i & 1 -> z
    vertices = []
    colors = []
    
    # Standard mapping used in hypercube.py
    # 0: 000, 1: 001, ..., 7: 111
    # But usually bits are read LSB or MSB.
    # hypercube.py used: x=(i&4), y=(i&2), z=(i&1)
    
    for i in range(8):
        x = 1 if (i & 4) else -1
        y = 1 if (i & 2) else -1
        z = 1 if (i & 1) else -1
        vertices.append([x, y, z])
        
        # Is the i-th bit set?
        is_set = (byte_val >> i) & 1
        colors.append(1 if is_set else 0)
        
    return np.array(vertices), np.array(colors)

def get_cube_edges():
    edges = []
    vertices = []
    for i in range(8):
        x = 1 if (i & 4) else -1
        y = 1 if (i & 2) else -1
        z = 1 if (i & 1) else -1
        vertices.append([x, y, z])
    vertices = np.array(vertices)
    
    for i in range(8):
        for j in range(i + 1, 8):
            dist = np.sum(np.abs(vertices[i] - vertices[j]))
            if dist == 2: # Connected
                edges.append((i, j))
    return edges, vertices


@st.cache_resource
def _cube_symmetry_signature_maps():
    perms = get_cube_permutations().cpu().numpy()  # (48, 8)

    def orbit(byte_val: int) -> tuple[int, ...]:
        bits = [(byte_val >> b) & 1 for b in range(8)]
        outs = set()
        for p in perms:
            v = 0
            for j in range(8):
                v |= (bits[int(p[j])] & 1) << j
            outs.add(int(v) & 0xFF)
        return tuple(sorted(outs))

    sig_by_byte: list[str] = [""] * 256
    orbit_by_byte: list[tuple[int, ...]] = [tuple()] * 256
    groups: dict[str, list[int]] = {}

    for b in range(256):
        ob = orbit(b)
        orbit_by_byte[b] = ob
        sig = "-".join(f"{x:02x}" for x in ob)
        sig_by_byte[b] = sig
        groups.setdefault(sig, []).append(b)

    return sig_by_byte, orbit_by_byte, groups

def run_trace(model, ids):
    device = next(model.parameters()).device
    st_vec = model.init_state(batch_size=1, device=device)
    states = []
    prim_acts = []
    candidates_list = []
    last_pos = None
    state_hist = None
    if bool(getattr(model, "symmetry_retrieval_enabled", False)):
        last_pos = torch.full((1, 256), -1, device=device, dtype=torch.long)
        state_hist = []
    
    with torch.no_grad():
        for i, tok in enumerate(ids):
            x = torch.tensor([tok], device=device, dtype=torch.long)
            x_emb = model._get_emb(x)

            if last_pos is not None and state_hist is not None and i > 0:
                inv = x ^ 0xFF
                rev = model.bit_reverse_lut[x]

                pos_inv = last_pos.gather(1, inv.view(1, 1)).view(-1)
                pos_rev = last_pos.gather(1, rev.view(1, 1)).view(-1)

                retrieval = torch.zeros_like(st_vec)
                cnt = torch.zeros((1, 1), device=device, dtype=st_vec.dtype)
                for pos in (pos_inv, pos_rev):
                    if int(pos.item()) >= 0:
                        retrieval = retrieval + state_hist[int(pos.item())]
                        cnt = cnt + 1.0
                retrieval = retrieval / torch.clamp(cnt, min=1.0)

                ctrl_in = torch.cat([x_emb, st_vec, retrieval], dim=-1)
            elif bool(getattr(model, "symmetry_retrieval_enabled", False)):
                retrieval = torch.zeros_like(st_vec)
                ctrl_in = torch.cat([x_emb, st_vec, retrieval], dim=-1)
            else:
                ctrl_in = torch.cat([x_emb, st_vec], dim=-1)
            w = torch.softmax(model.controller(ctrl_in), dim=-1)

            sA = torch.einsum("b d, k d e -> b k e", st_vec, model.prim_A)
            xB = torch.einsum("b m, k d m -> b k d", x_emb, model.prim_B)
            cand = torch.tanh(sA + xB + model.prim_bias.unsqueeze(0)) # [1, 8, dim]
            mixed = torch.einsum("b k, b k d -> b d", w, cand)

            gate = model.slice_gate(ctrl_in)
            st_vec = gate * mixed + (1.0 - gate) * st_vec

            if last_pos is not None and state_hist is not None:
                state_hist.append(st_vec.clone())
                last_pos.scatter_(1, x.view(1, 1), torch.full((1, 1), i, device=device, dtype=torch.long))

            states.append(st_vec[0].detach().cpu().numpy())
            prim_acts.append(w[0].detach().cpu().numpy())
            candidates_list.append(cand[0].detach().cpu().numpy())

    return np.stack(states, axis=0), np.stack(prim_acts, axis=0), np.stack(candidates_list, axis=0)


def _ascii_allowed_mask(device, dtype):
    allowed = torch.zeros(256, dtype=torch.bool, device=device)
    allowed[9] = True
    allowed[10] = True
    allowed[13] = True
    allowed[32:127] = True
    return allowed


def _next_logits_after_prompt(model, prompt_ids):
    device = next(model.parameters()).device
    state = model.init_state(batch_size=1, device=device)
    if len(prompt_ids) == 0:
        x0 = torch.tensor([[0]], device=device, dtype=torch.long)
        with torch.no_grad():
            out = model(x0, state)
            logits_seq, state = out[0], out[1]
            return logits_seq[:, -1, :], state

    x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(x, state)
        logits_seq, state = out[0], out[1]
        return logits_seq[:, -1, :], state


def _logprob_target_given_prompt(model, prompt_ids, target_ids):
    device = next(model.parameters()).device
    state = model.init_state(batch_size=1, device=device)

    with torch.no_grad():
        if len(prompt_ids) > 0:
            x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
            out = model(x, state)
            logits_seq, state = out[0], out[1]
            next_logits = logits_seq[:, -1, :]
        else:
            x0 = torch.tensor([[0]], device=device, dtype=torch.long)
            out = model(x0, state)
            logits_seq, state = out[0], out[1]
            next_logits = logits_seq[:, -1, :]

        total = 0.0
        for tid in target_ids:
            logp = torch.log_softmax(next_logits, dim=-1)
            total += float(logp[0, int(tid)].item())
            xt = torch.tensor([[int(tid)]], device=device, dtype=torch.long)
            out = model(xt, state)
            logits_seq, state = out[0], out[1]
            next_logits = logits_seq[:, -1, :]

    return total


def beam_search_prompt_for_target(
    model,
    target_text: str,
    *,
    prefix_text: str = "",
    prompt_len: int = 32,
    beam_width: int = 16,
    expand_topk: int = 16,
    ascii_only: bool = True,
    target_prefix_len: int = 32,
):
    target_ids_all = list(target_text.encode("utf-8"))
    target_ids = target_ids_all[: max(1, int(target_prefix_len))]
    prefix_ids = list(prefix_text.encode("utf-8"))
    if prompt_len < len(prefix_ids):
        prefix_ids = prefix_ids[:prompt_len]

    beams = [(prefix_ids, _logprob_target_given_prompt(model, prefix_ids, target_ids))]
    while len(beams[0][0]) < prompt_len:
        all_cand = []
        for ids, _score in beams:
            next_logits, _st = _next_logits_after_prompt(model, ids)
            logits = next_logits.clone()
            if ascii_only:
                allowed = _ascii_allowed_mask(logits.device, logits.dtype)
                logits[:, ~allowed] = -float("inf")
            k = max(1, int(expand_topk))
            v, idx = torch.topk(logits, k=min(k, logits.size(-1)))
            for b in idx[0].tolist():
                new_ids = ids + [int(b)]
                s = _logprob_target_given_prompt(model, new_ids, target_ids)
                all_cand.append((new_ids, s))

        all_cand.sort(key=lambda x: x[1], reverse=True)
        beams = all_cand[: max(1, int(beam_width))]
        if len(beams) == 0:
            break

    best_ids, best_score = beams[0]
    return best_ids, best_score

def get_pca_transform(x):
    # Returns a function that projects data into the PCA space of x
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    comp = vt[:3].T
    
    def transform(data):
        # Handle single vector or batch
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return (data - mean) @ comp
        
    return transform

# --- UI Layout ---

st.title("🧠 RAP Hypercube Explorer")
st.markdown("Interactive visualization of the **Geometric Language Model**.")

# Sidebar
ckpt_path = st.sidebar.text_input("Checkpoint Path", "checkpoints1/b.pt")
device_choice = st.sidebar.selectbox(
    "Device",
    ["cuda" if torch.cuda.is_available() else "cpu", "cpu"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Retrieval (runtime overrides)**")
ov_word_map_mode = st.sidebar.selectbox(
    "Word-map retrieval",
    ["Auto (checkpoint)", "Force ON", "Force OFF"],
    index=0,
)
ov_word_map_path = st.sidebar.text_input("Word map path (for Force ON)", "word_chains.txt")
ov_word_map_topk = int(st.sidebar.number_input("Word map top-k (for Force ON)", min_value=1, max_value=256, value=32, step=1))
ov_word_map_min_count = int(st.sidebar.number_input("Word map min count (for Force ON)", min_value=1, max_value=100, value=1, step=1))

cfg_override = None
if ov_word_map_mode == "Force ON":
    cfg_override = {
        "use_word_map_retrieval": True,
        "word_map_path": str(ov_word_map_path).strip(),
        "word_map_topk": int(ov_word_map_topk),
        "word_map_min_count": int(ov_word_map_min_count),
    }
elif ov_word_map_mode == "Force OFF":
    cfg_override = {
        "use_word_map_retrieval": False,
    }

model = load_model(ckpt_path, device=device_choice, cfg_override=cfg_override)

if model is None:
    st.error(f"Could not load checkpoint at `{ckpt_path}`.")
    st.stop()
else:
    st.sidebar.success("Model Loaded!")

mode = st.sidebar.radio(
    "Mode",
    ["Cube Viewer", "Decision Tree Explorer", "Trajectory Explorer", "Checkpoint Viewer", "Training Monitor", "Training Visualizer", "Mirror Lab", "Chat"],
)


def _corrupt_prompt_ids(prompt_ids: list[int], *, prob: float, mode: str) -> list[int]:
    if prob <= 0.0 or len(prompt_ids) == 0:
        return list(prompt_ids)
    out = list(prompt_ids)
    t = len(out)
    m = (torch.rand((t,), device="cpu") < float(prob)).tolist()
    if mode == "xor_ff":
        for i, mm in enumerate(m):
            if mm:
                out[i] = int(out[i]) ^ 0xFF
        return out
    for i, mm in enumerate(m):
        if not mm:
            continue
        r = int(torch.randint(0, 256, (1,), device="cpu").item())
        if r == int(out[i]):
            r = (r + 1) % 256
        out[i] = r
    return out


def _greedy_generate_ids(
    model,
    prompt_ids: list[int],
    *,
    gen_len: int,
    temperature: float,
    top_k: int,
    ascii_only: bool,
) -> list[int]:
    device = next(model.parameters()).device
    state = model.init_state(batch_size=1, device=device)
    with torch.no_grad():
        if len(prompt_ids) > 0:
            x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
            out = model(x, state)
            logits_seq, state = out[0], out[1]
            next_logits = logits_seq[:, -1, :]
        else:
            x0 = torch.tensor([[0]], device=device, dtype=torch.long)
            out = model(x0, state)
            logits_seq, state = out[0], out[1]
            next_logits = logits_seq[:, -1, :]

        out_ids: list[int] = []
        for _ in range(int(gen_len)):
            logits = next_logits.clone()
            if bool(ascii_only):
                allowed = _ascii_allowed_mask(logits.device, logits.dtype)
                logits[:, ~allowed] = -float("inf")
            if float(temperature) != 1.0:
                logits = logits / max(1e-6, float(temperature))
            k = int(top_k)
            if k > 0 and k < logits.shape[-1]:
                v, idx = torch.topk(logits, k=k)
                mask = torch.full_like(logits, -float("inf"))
                mask.scatter_(1, idx, v)
                logits = mask
            nxt = int(torch.argmax(logits, dim=-1).item())
            out_ids.append(nxt)
            xt = torch.tensor([[nxt]], device=device, dtype=torch.long)
            out = model(xt, state)
            logits_seq, state = out[0], out[1]
            next_logits = logits_seq[:, -1, :]
    return out_ids


def _trace_generation_decisions(
    model,
    prompt_ids: list[int],
    *,
    gen_len: int,
    temperature: float,
    top_k: int,
    ascii_only: bool,
    repetition_penalty: float,
    repetition_window: int,
    stop_on_newline: bool,
    enable_retrieval: bool,
    inspect_topk: int,
    branch_preview_len: int,
) -> tuple[list[int], list[dict]]:
    device = next(model.parameters()).device
    st_vec = model.init_state(batch_size=1, device=device)
    last_pos = None
    state_hist = None
    retrieval_enabled = bool(enable_retrieval) and bool(getattr(model, "symmetry_retrieval_enabled", False))
    if retrieval_enabled:
        last_pos = torch.full((1, 256), -1, device=device, dtype=torch.long)
        state_hist = []

    def _forward_one(tok: int):
        nonlocal st_vec, last_pos, state_hist
        x = torch.tensor([int(tok)], device=device, dtype=torch.long)
        x_emb = model._get_emb(x)

        retrieval = None
        retrieval_hit = False
        if last_pos is not None and state_hist is not None:
            retrieval = torch.zeros_like(st_vec)
            cnt = torch.zeros((1, 1), device=device, dtype=st_vec.dtype)
            inv = x ^ 0xFF
            rev = model.bit_reverse_lut[x]
            for pos in (
                last_pos.gather(1, inv.view(1, 1)).view(-1),
                last_pos.gather(1, rev.view(1, 1)).view(-1),
            ):
                if int(pos.item()) >= 0:
                    retrieval = retrieval + state_hist[int(pos.item())]
                    cnt = cnt + 1.0
                    retrieval_hit = True
            retrieval = retrieval / torch.clamp(cnt, min=1.0)
            ctrl_in = torch.cat([x_emb, st_vec, retrieval], dim=-1)
        else:
            ctrl_in = torch.cat([x_emb, st_vec], dim=-1)

        w = torch.softmax(model.controller(ctrl_in), dim=-1)
        sA = torch.einsum("b d, k d e -> b k e", st_vec, model.prim_A)
        xB = torch.einsum("b m, k d m -> b k d", x_emb, model.prim_B)
        cand = torch.tanh(sA + xB + model.prim_bias.unsqueeze(0))
        mixed = torch.einsum("b k, b k d -> b d", w, cand)
        gate = model.slice_gate(ctrl_in)
        if bool(getattr(model, "delta_enabled", False)):
            eps = 1e-8
            logK = torch.log(torch.tensor(float(model.cfg.n_primitives), device=w.device, dtype=w.dtype))
            H = -(w * (w + eps).log()).sum(dim=-1, keepdim=True)
            Hn = H / (logK + eps)
            lo = torch.tensor(float(getattr(model, "delta_entropy_lo", 0.55)), device=w.device, dtype=w.dtype)
            hi = torch.tensor(float(getattr(model, "delta_entropy_hi", 0.95)), device=w.device, dtype=w.dtype)
            beta_max = torch.tensor(float(getattr(model, "delta_beta_max", 1.0)), device=w.device, dtype=w.dtype)
            alpha = ((Hn - lo) / (hi - lo + eps)).clamp(0.0, 1.0)
            beta = beta_max * alpha
            eff = gate * beta
            st_vec = (1.0 - eff) * st_vec + eff * mixed
            entropy_norm = float(Hn.item())
        else:
            st_vec = gate * mixed + (1.0 - gate) * st_vec
            eps = 1e-8
            logK = torch.log(torch.tensor(float(model.cfg.n_primitives), device=w.device, dtype=w.dtype))
            H = -(w * (w + eps).log()).sum(dim=-1)
            entropy_norm = float((H / (logK + eps)).item())

        logits = model.out(torch.cat([x_emb, st_vec], dim=-1))

        if last_pos is not None and state_hist is not None:
            state_hist.append(st_vec.clone())
            last_pos.scatter_(1, x.view(1, 1), torch.full((1, 1), len(state_hist) - 1, device=device, dtype=torch.long))

        return {
            "logits": logits,
            "weights": w[0].detach().cpu().tolist(),
            "candidate_norms": torch.linalg.norm(cand[0], dim=-1).detach().cpu().tolist(),
            "gate_mean": float(gate.mean().item()),
            "retrieval_hit": bool(retrieval_hit),
            "entropy_norm": float(entropy_norm),
        }

    with torch.no_grad():
        last_info = None
        for tok in prompt_ids:
            last_info = _forward_one(int(tok))

        if last_info is None:
            last_info = _forward_one(0)

        generated = []
        rows = []
        for step_idx in range(int(gen_len)):
            logits = last_info["logits"].clone()
            if bool(ascii_only):
                logits = _apply_printable_ascii_mask(logits)
            if repetition_penalty and repetition_penalty > 1.0:
                logits = _apply_repetition_penalty(logits, generated[-int(repetition_window) :], float(repetition_penalty))
            scored_logits = logits / max(1e-6, float(temperature))
            probs = torch.softmax(scored_logits, dim=-1)

            k = int(top_k)
            if k > 0 and k < probs.shape[-1]:
                topv, topi = torch.topk(probs, k=k, dim=-1)
                masked = torch.zeros_like(probs)
                masked.scatter_(1, topi, topv)
                probs = masked / torch.clamp(masked.sum(dim=-1, keepdim=True), min=1e-8)

            chosen = int(torch.multinomial(probs, num_samples=1).item())
            if int(chosen) == 10 and bool(stop_on_newline):
                break
            inspect_k = max(1, min(int(inspect_topk), probs.shape[-1]))
            topv, topi = torch.topk(probs, k=inspect_k, dim=-1)
            options = []
            for prob_v, tok_i in zip(topv[0].tolist(), topi[0].tolist()):
                tok_i = int(tok_i)
                preview = generated + [tok_i]
                options.append(
                    {
                        "byte": tok_i,
                        "label": _byte_label(tok_i),
                        "prob": float(prob_v),
                        "preview": _format_bytes_for_display(bytes(preview[-max(1, int(branch_preview_len)) :]), "UTF-8 (replace errors)"),
                    }
                )

            generated.append(chosen)
            rows.append(
                {
                    "step": int(step_idx),
                    "chosen_byte": int(chosen),
                    "chosen_label": _byte_label(chosen),
                    "prefix_preview": _format_bytes_for_display(bytes((prompt_ids + generated[:-1])[-80:]), "UTF-8 (replace errors)"),
                    "options": options,
                    "primitive_weights": last_info["weights"],
                    "candidate_norms": last_info["candidate_norms"],
                    "gate_mean": float(last_info["gate_mean"]),
                    "retrieval_hit": bool(last_info["retrieval_hit"]),
                    "entropy_norm": float(last_info["entropy_norm"]),
                }
            )
            last_info = _forward_one(chosen)

    return generated, rows


def _kl_divergence_per_pos(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits_a, dim=-1)
    p = logp.exp()
    logq = F.log_softmax(logits_b, dim=-1)
    return (p * (logp - logq)).sum(dim=-1)


def _render_byte_blocks(ids: list[int], scores: list[float], *, thr: float, title: str) -> None:
    block_w = 12
    block_h = 16
    gap = 2
    items = []
    for b, s in zip(ids, scores):
        good = float(s) <= float(thr)
        col = "#2e8b57" if good else "#b22222"
        lab = _byte_label(int(b))
        tip = f"{int(b)} {lab} | KL={float(s):.4f}"
        items.append(
            f"<span title='{html.escape(tip)}' "
            f"style='display:inline-block;width:{block_w}px;height:{block_h}px;"
            f"margin-right:{gap}px;background:{col};border:1px solid #00000055;'></span>"
        )
    st.markdown(f"**{title}**")
    st.markdown("<div style='line-height:0.9;white-space:nowrap;overflow-x:auto;padding:6px 0;'>" + "".join(items) + "</div>", unsafe_allow_html=True)

def _read_records(path: str, *, max_lines: int = 20000) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= int(max_lines):
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
                    continue
            except Exception:
                pass
            try:
                obj = ast.literal_eval(line)
                if isinstance(obj, dict):
                    out.append(obj)
                    continue
            except Exception:
                continue
    return out


def _safe_float(x, default: float | None = None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

# --- MODE: Cube Viewer ---
if mode == "Cube Viewer":
    st.header("Token Cube Geometry")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        char_input = st.text_input("Enter a character", "A", max_chars=1)
        if not char_input: char_input = " "
        
        byte_val = ord(char_input)
        st.metric("Character", f"'{char_input}'")

elif mode == "Decision Tree Explorer":
    st.header("Decision Tree Explorer")
    st.markdown("Inspect the realized generation path step by step, including alternatives and primitive routing.")

    colA, colB = st.columns([2, 1])
    with colA:
        prompt_text = st.text_area("Prompt Text", "Question: What is gravity?\n\nAnswer:", height=140)
    with colB:
        gen_len = int(st.slider("Generate length (bytes)", min_value=8, max_value=1024, value=128, step=8, key="tree_gen_len"))
        inspect_topk = int(st.slider("Show top-k options per step", min_value=2, max_value=16, value=5, step=1, key="tree_topk_view"))
        decode_topk = int(st.slider("Decoder top-k", min_value=0, max_value=256, value=32, step=1, key="tree_decode_topk"))
        temperature = float(st.slider("Temperature", min_value=0.05, max_value=2.0, value=0.8, step=0.05, key="tree_temperature"))
        ascii_only = st.checkbox("ASCII-only generation", value=False, key="tree_ascii_only")
        repetition_penalty = float(st.slider("Repetition penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.05, key="tree_rep_pen"))
        repetition_window = int(st.slider("Repetition window (bytes)", min_value=8, max_value=256, value=64, step=8, key="tree_rep_win"))
        stop_on_newline = st.checkbox("Stop on newline", value=True, key="tree_stop_newline")
        enable_retrieval = st.checkbox("Enable retrieval", value=True, key="tree_enable_retrieval")
        branch_preview_len = int(st.slider("Alternative preview bytes", min_value=1, max_value=64, value=16, step=1, key="tree_preview_len"))

    run_tree = st.button("Generate decision trace")
    if not run_tree:
        st.info("Click 'Generate decision trace' to inspect the step-by-step decoded path.")
        st.stop()

    prompt_ids = list(prompt_text.encode("utf-8"))
    with st.spinner("Tracing generation decisions..."):
        gen_ids, decision_rows = _trace_generation_decisions(
            model,
            prompt_ids,
            gen_len=int(gen_len),
            temperature=float(temperature),
            top_k=int(decode_topk),
            ascii_only=bool(ascii_only),
            repetition_penalty=float(repetition_penalty),
            repetition_window=int(repetition_window),
            stop_on_newline=bool(stop_on_newline),
            enable_retrieval=bool(enable_retrieval),
            inspect_topk=int(inspect_topk),
            branch_preview_len=int(branch_preview_len),
        )

    st.markdown("**Prompt**")
    st.code(prompt_text)
    st.markdown("**Generated continuation**")
    st.code(bytes(gen_ids).decode("utf-8", errors="replace"))
    st.markdown(f"**Generated bytes**: `{len(gen_ids)}`")

    if decision_rows:
        chosen_probs = []
        entropy_vals = []
        retrieval_vals = []
        for row in decision_rows:
            options = row.get("options") or []
            chosen_p = 0.0
            for opt in options:
                if int(opt.get("byte", -1)) == int(row.get("chosen_byte", -2)):
                    chosen_p = float(opt.get("prob", 0.0))
                    break
            chosen_probs.append(chosen_p)
            entropy_vals.append(float(row.get("entropy_norm", 0.0)))
            retrieval_vals.append(1.0 if bool(row.get("retrieval_hit", False)) else 0.0)

        fig_tree = go.Figure()
        fig_tree.add_trace(go.Scatter(x=list(range(len(decision_rows))), y=chosen_probs, mode="lines", name="chosen_prob"))
        fig_tree.add_trace(go.Scatter(x=list(range(len(decision_rows))), y=entropy_vals, mode="lines", name="routing_entropy_norm"))
        fig_tree.add_trace(go.Scatter(x=list(range(len(decision_rows))), y=retrieval_vals, mode="lines", name="retrieval_hit"))
        fig_tree.update_layout(height=300, margin=dict(l=0, r=0, b=0, t=30), title="Decision Trace Summary")
        st.plotly_chart(fig_tree, use_container_width=True)

        step_sel = int(st.slider("Inspect decision step", min_value=0, max_value=max(0, len(decision_rows) - 1), value=0, step=1, key="tree_step_sel"))
        row = decision_rows[step_sel]
        st.markdown(
            f"**step**=`{row['step']}`  **chosen**=`{row['chosen_byte']}` {row['chosen_label']}  **entropy**=`{row['entropy_norm']:.3f}`  **retrieval_hit**=`{row['retrieval_hit']}`"
        )
        st.markdown("**Prefix before this decision**")
        st.code(str(row.get("prefix_preview", "")))

        options = row.get("options") or []
        if options:
            fig_opts = go.Figure(
                data=[
                    go.Bar(
                        x=[f"{int(opt['byte'])} {opt['label']}" for opt in options],
                        y=[float(opt["prob"]) for opt in options],
                    )
                ]
            )
            fig_opts.update_layout(height=260, margin=dict(l=0, r=0, b=0, t=30), title="Top next-byte options")
            st.plotly_chart(fig_opts, use_container_width=True)

            st.markdown("**Alternative branches**")
            for rank, opt in enumerate(options, start=1):
                st.markdown(
                    f"- **#{rank}** `{int(opt['byte'])}` {opt['label']}  `p={float(opt['prob']):.4f}`  preview=`{opt['preview']}`"
                )

        fig_pw = go.Figure(
            data=[go.Bar(x=[f"Prim {i}" for i in range(len(row["primitive_weights"]))], y=row["primitive_weights"])]
        )
        fig_pw.update_layout(height=260, margin=dict(l=0, r=0, b=0, t=30), title="Primitive weights at this step")
        st.plotly_chart(fig_pw, use_container_width=True)

        fig_cn = go.Figure(
            data=[go.Bar(x=[f"Prim {i}" for i in range(len(row["candidate_norms"]))], y=row["candidate_norms"])]
        )
        fig_cn.update_layout(height=260, margin=dict(l=0, r=0, b=0, t=30), title="Candidate state norms at this step")
        st.plotly_chart(fig_cn, use_container_width=True)

        with st.expander("All decision rows"):
            st.json(decision_rows)

elif mode == "Training Visualizer":
    st.header("Training Visualizer")

    colA, colB = st.columns([1, 1])
    with colA:
        prompt_text = st.text_area("Prompt (UTF-8)", value="What is gravity?\n", height=120)
        gen_len = int(st.slider("Generate length (bytes)", min_value=8, max_value=4096, value=64, step=8))
        ascii_only = st.checkbox("ASCII-only generation", value=True)
    with colB:
        noise_prob = float(st.slider("Prompt noise prob", min_value=0.0, max_value=0.2, value=0.01, step=0.005))
        noise_mode = st.selectbox("Noise mode", ["random", "xor_ff"], index=0)
        thr = float(st.slider("Green/Red KL threshold", min_value=0.0, max_value=2.0, value=0.15, step=0.01))
        temperature = float(st.slider("Temperature", min_value=0.05, max_value=2.0, value=1.0, step=0.05))
        top_k = int(st.slider("Top-k", min_value=0, max_value=256, value=0, step=1))

    run = st.button("Run visualizer")
    if not run:
        st.stop()

    prompt_ids = list(prompt_text.encode("utf-8"))
    noisy_prompt_ids = _corrupt_prompt_ids(prompt_ids, prob=float(noise_prob), mode=str(noise_mode))

    gen_ids = _greedy_generate_ids(
        model,
        prompt_ids,
        gen_len=int(gen_len),
        temperature=float(temperature),
        top_k=int(top_k),
        ascii_only=bool(ascii_only),
    )

    gen_ids_noisy = _greedy_generate_ids(
        model,
        noisy_prompt_ids,
        gen_len=int(gen_len),
        temperature=float(temperature),
        top_k=int(top_k),
        ascii_only=bool(ascii_only),
    )

    clean_full = prompt_ids + gen_ids
    noisy_full = noisy_prompt_ids + gen_ids

    device = next(model.parameters()).device
    with torch.no_grad():
        x_clean = torch.tensor([clean_full], device=device, dtype=torch.long)
        x_noisy = torch.tensor([noisy_full], device=device, dtype=torch.long)
        logits_clean = model(x_clean)[0][0]
        logits_noisy = model(x_noisy)[0][0]
        kl = _kl_divergence_per_pos(logits_clean, logits_noisy).detach().to("cpu")

    kl_list = [float(x) for x in kl.tolist()]
    if len(kl_list) < len(clean_full):
        kl_list = kl_list + [kl_list[-1] if kl_list else 0.0] * (len(clean_full) - len(kl_list))

    st.subheader("Clean vs Noised Prompt")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Clean prompt**")
        st.code(bytes(prompt_ids).decode("utf-8", errors="replace"))
        _render_byte_blocks(prompt_ids, kl_list[: len(prompt_ids)], thr=float(thr), title="Prompt blocks (clean vs noised KL)")
    with col2:
        st.markdown("**Noised prompt**")
        st.code(bytes(noisy_prompt_ids).decode("utf-8", errors="replace"))

    st.subheader("Continuation")
    col3, col4 = st.columns([1, 1])
    cont_kl = kl_list[len(prompt_ids) : len(prompt_ids) + len(gen_ids)]
    with col3:
        st.markdown("**Output (clean prompt)**")
        st.code(bytes(gen_ids).decode("utf-8", errors="replace"))
        _render_byte_blocks(gen_ids, cont_kl, thr=float(thr), title="Output blocks (clean vs noised KL)")
    with col4:
        st.markdown("**Output (noised prompt)**")
        st.code(bytes(gen_ids_noisy).decode("utf-8", errors="replace"))

# --- MODE: Trajectory Explorer ---
elif mode == "Trajectory Explorer":
    st.header("Manifold Trajectory & Byte Logic")

    traj_mode = st.radio("Trajectory Mode", ["Forward", "Generated Response", "Inverse (find input for target output)"])
    show_swarm = st.checkbox("Show Primitive Swarm (Ghost States)", value=False)

    ids = None
    states = None
    acts = None
    candidates = None
    pca_func = None
    xyz = None
    prompt_trace_len = None

    if traj_mode == "Forward":
        input_mode = st.selectbox(
            "Input mode",
            ["UTF-8 text", "Escaped/Tagged bytes"],
            index=0,
        )
        if input_mode == "Escaped/Tagged bytes":
            text_input = st.text_area(
                "Input (supports \\xNN and <NOUN>/<VERB>/<PRON>)",
                "The <NOUN> runs <VERB> .",
            )
        else:
            text_input = st.text_area("Input Text", "The quick brown fox jumps over the lazy dog")
        if not text_input:
            st.stop()
        if input_mode == "Escaped/Tagged bytes":
            ids = _parse_escaped_or_tagged_bytes(text_input)
        else:
            ids = list(text_input.encode("utf-8"))
        with st.spinner("Tracing neural pathway..."):
            states, acts, candidates = run_trace(model, ids)
            pca_func = get_pca_transform(states)
            xyz = pca_func(states)
    elif traj_mode == "Generated Response":
        colA, colB = st.columns([2, 1])
        with colA:
            prompt_text = st.text_area("Prompt Text", "Question: What is gravity?\n\nAnswer:")
        with colB:
            gen_len = int(st.slider("Generate length (bytes)", min_value=8, max_value=4096, value=96, step=8, key="traj_gen_len"))
            temperature = float(st.slider("Temperature", min_value=0.05, max_value=2.0, value=0.8, step=0.05, key="traj_temperature"))
            top_k = int(st.slider("Top-k", min_value=0, max_value=256, value=32, step=1, key="traj_topk"))
            ascii_only = st.checkbox("ASCII-only generation", value=False, key="traj_ascii_only")

        run_gen_trace = st.button("Generate and trace")
        if not run_gen_trace:
            st.info("Click 'Generate and trace' to sample a continuation and visualize its trajectory.")
            st.stop()

        prompt_ids = list(prompt_text.encode("utf-8"))
        with st.spinner("Generating continuation and tracing neural pathway..."):
            gen_ids = _greedy_generate_ids(
                model,
                prompt_ids,
                gen_len=int(gen_len),
                temperature=float(temperature),
                top_k=int(top_k),
                ascii_only=bool(ascii_only),
            )
            ids = list(prompt_ids) + list(gen_ids)
            prompt_trace_len = len(prompt_ids)
            states, acts, candidates = run_trace(model, ids)
            pca_func = get_pca_transform(states)
            xyz = pca_func(states)

        st.markdown("**Prompt**")
        st.code(prompt_text)
        st.markdown("**Generated continuation**")
        st.code(bytes(gen_ids).decode("utf-8", errors="replace"))
        st.markdown(
            f"**Trace bytes**: total=`{len(ids)}`  prompt=`{len(prompt_ids)}`  generated=`{len(gen_ids)}`"
        )
    else:
        colA, colB = st.columns([2, 1])
        with colA:
            target_text = st.text_area("Target Output (what you want it to say)", "Gravity is")
            prefix_text = st.text_input("Prefix (optional)", "Answer:")
        with colB:
            prompt_len = st.number_input("Prompt length (bytes)", min_value=1, max_value=256, value=16, step=1)
            target_prefix_len = st.number_input("Target prefix length (bytes)", min_value=1, max_value=256, value=24, step=1)
            beam_width = st.number_input("Beam width", min_value=1, max_value=64, value=8, step=1)
            expand_topk = st.number_input("Expand top-k", min_value=1, max_value=64, value=8, step=1)
            ascii_only = st.checkbox("ASCII-only prompt", value=True)

        run_inverse = st.button("Run inverse search")

        if not target_text:
            st.stop()

        cache_key = "inverse_result"
        cache_sig = {
            "target_text": target_text,
            "prefix_text": prefix_text,
            "prompt_len": int(prompt_len),
            "target_prefix_len": int(target_prefix_len),
            "beam_width": int(beam_width),
            "expand_topk": int(expand_topk),
            "ascii_only": bool(ascii_only),
        }

        if run_inverse:
            with st.spinner("Searching for a prompt that makes this output likely..."):
                best_ids, best_score = beam_search_prompt_for_target(
                    model,
                    target_text,
                    prefix_text=prefix_text,
                    prompt_len=int(prompt_len),
                    beam_width=int(beam_width),
                    expand_topk=int(expand_topk),
                    ascii_only=bool(ascii_only),
                    target_prefix_len=int(target_prefix_len),
                )
                st.session_state[cache_key] = {
                    "sig": cache_sig,
                    "best_ids": best_ids,
                    "best_score": float(best_score),
                }

        cached = st.session_state.get(cache_key)
        if cached and cached.get("sig") == cache_sig:
            best_ids = cached.get("best_ids")
            best_score = cached.get("best_score")
            st.markdown("**Best prompt (escaped bytes):**")
            st.code(_format_bytes_for_display(bytes(best_ids), "Escaped bytes (recommended)"))
            st.markdown(f"**Score (log P(target|prompt))**: `{float(best_score):.2f}`")

            rollout_len = max(8, min(256, len(target_text.encode("utf-8", errors="replace")) + 24))
            rollout_ids = _greedy_generate_ids(
                model,
                list(best_ids),
                gen_len=int(rollout_len),
                temperature=1.0,
                top_k=0,
                ascii_only=bool(ascii_only),
            )
            st.markdown("**Free-run continuation from best prompt:**")
            st.code(_format_bytes_for_display(bytes(rollout_ids), "Escaped bytes (recommended)"))

            with st.spinner("Tracing inverse prompt trajectory..."):
                ids = list(best_ids)
                states, acts, candidates = run_trace(model, ids)
                pca_func = get_pca_transform(states)
                xyz = pca_func(states)
        else:
            st.info("Click 'Run inverse search' to start. Forward mode will remain responsive.")
            st.stop()

    if ids is None or states is None or xyz is None or candidates is None:
        st.stop()

    # Plotly 3D Path
    fig = go.Figure()
    
    # Line
    fig.add_trace(go.Scatter3d(
        x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
        mode='lines',
        line=dict(color='gray', width=3),
        name='Path'
    ))
    
    # Nodes
    labels = []
    for b in ids:
        if int(b) in _TAG_LABELS:
            labels.append(_TAG_LABELS[int(b)])
            continue
        c = chr(b)
        if b == 32:
            labels.append("(sp)")
        elif 32 <= b <= 126:
            labels.append(f"'{c}'")
        else:
            labels.append(f"x{b:02x}")
        
    t_sel = int(st.slider("Inspect timestep", min_value=0, max_value=max(0, len(ids) - 1), value=0, step=1))
    st.markdown(f"**t={t_sel}**  byte=`{int(ids[t_sel])}`  label=`{labels[t_sel]}`")
    if prompt_trace_len is not None:
        phase = "prompt" if int(t_sel) < int(prompt_trace_len) else "generated"
        rel_t = int(t_sel) if phase == "prompt" else int(t_sel) - int(prompt_trace_len)
        st.markdown(f"**phase**=`{phase}`  **phase_t**=`{rel_t}`")
    fig_ins = go.Figure(data=[go.Bar(x=[f"Prim {i}" for i in range(acts.shape[1])], y=acts[t_sel])])
    fig_ins.update_layout(height=260, margin=dict(l=0, r=0, b=0, t=20), title="Primitive Weights at t")
    st.plotly_chart(fig_ins, use_container_width=True)

    cand_norms = np.linalg.norm(candidates[t_sel], axis=-1)
    fig_norm = go.Figure(data=[go.Bar(x=[f"Prim {i}" for i in range(cand_norms.shape[0])], y=cand_norms)])
    fig_norm.update_layout(height=260, margin=dict(l=0, r=0, b=0, t=20), title="Candidate State Norms at t")
    st.plotly_chart(fig_norm, use_container_width=True)
        
    state_hover = [
        f"t={i}<br>byte={int(ids[i])}<br>label={labels[i]}" + ("<br><b>TAG</b>" if int(ids[i]) in _TAG_LABELS else "")
        for i in range(len(ids))
    ]

    tag_mask = np.array([1 if int(b) in _TAG_LABELS else 0 for b in ids], dtype=np.int32)

    idx_all = np.arange(len(ids))
    idx_tags = idx_all[tag_mask == 1]
    idx_norm = idx_all[tag_mask == 0]

    # Normal byte states
    fig.add_trace(go.Scatter3d(
        x=xyz[idx_norm, 0], y=xyz[idx_norm, 1], z=xyz[idx_norm, 2],
        mode='markers+text',
        marker=dict(
            size=8,
            color=idx_norm,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Time"),
            symbol='circle',
        ),
        text=[labels[i] for i in idx_norm.tolist()],
        textposition="top center",
        hovertext=[state_hover[i] for i in idx_norm.tolist()],
        hoverinfo="text",
        name='States'
    ))

    # Tag byte states (highlighted)
    if idx_tags.size > 0:
        fig.add_trace(go.Scatter3d(
            x=xyz[idx_tags, 0], y=xyz[idx_tags, 1], z=xyz[idx_tags, 2],
            mode='markers+text',
            marker=dict(
                size=12,
                color=idx_tags,
                colorscale='Viridis',
                showscale=False,
                symbol='diamond',
                line=dict(color="#ff2d55", width=3),
            ),
            text=[labels[i] for i in idx_tags.tolist()],
            textposition="top center",
            hovertext=[state_hover[i] for i in idx_tags.tolist()],
            hoverinfo="text",
            name='Tags'
        ))

    fig.add_trace(go.Scatter3d(
        x=[float(xyz[t_sel, 0])], y=[float(xyz[t_sel, 1])], z=[float(xyz[t_sel, 2])],
        mode='markers',
        marker=dict(size=14, color='#ff4d4d', opacity=1.0),
        hovertext=[f"SELECTED<br>t={t_sel}<br>byte={int(ids[t_sel])} ({labels[t_sel]})"],
        hoverinfo="text",
        name='Selected State'
    ))

    # --- Primitive Swarm ---
    if show_swarm:
        # candidates shape: (T, 8, Dim)
        T, P, D = candidates.shape
        
        # Flatten to project: (T*8, Dim)
        flat_cand = candidates.reshape(-1, D)
        xyz_cand = pca_func(flat_cand) # (T*8, 3)
        
        # Reshape back to handle by primitive type
        xyz_cand = xyz_cand.reshape(T, P, 3)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for p in range(8):
            # Extract trajectory of Primitive p (if it had 100% weight)
            p_xyz = xyz_cand[:, p, :]

            p_w = acts[:, p]
            p_norm = np.linalg.norm(candidates[:, p, :], axis=-1)
            p_hover = [
                f"t={i}<br>byte={int(ids[i])} ({labels[i]})<br>prim={p}<br>w={float(p_w[i]):.3f}<br>||cand||={float(p_norm[i]):.3f}"
                for i in range(int(T))
            ]
            
            fig.add_trace(go.Scatter3d(
                x=p_xyz[:, 0], y=p_xyz[:, 1], z=p_xyz[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors[p],
                    opacity=0.4
                ),
                hovertext=p_hover,
                hoverinfo="text",
                name=f'Ghost Prim {p}'
            ))

        ghost_sel = xyz_cand[t_sel]  # (P, 3)
        ghost_hover = [
            f"SELECTED GHOST<br>t={t_sel}<br>byte={int(ids[t_sel])} ({labels[t_sel]})<br>prim={p}<br>w={float(acts[t_sel, p]):.3f}<br>||cand||={float(cand_norms[p]):.3f}"
            for p in range(int(P))
        ]
        fig.add_trace(go.Scatter3d(
            x=ghost_sel[:, 0], y=ghost_sel[:, 1], z=ghost_sel[:, 2],
            mode='markers',
            marker=dict(
                size=7,
                color=np.arange(int(P)),
                colorscale='Viridis',
                opacity=1.0,
            ),
            hovertext=ghost_hover,
            hoverinfo='text',
            name='Selected Ghosts'
        ))
    
    fig.update_layout(
        title="3D Thought Trajectory",
        scene=dict(
            xaxis_title="PCA 1",
            yaxis_title="PCA 2",
            zaxis_title="PCA 3"
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap of Primitives
    st.subheader("Byte Bot Logic (Primitive Activations)")
    st.markdown("This heatmap shows **which logic unit (Primitive)** handles the connection for each character.")
    
    fig2 = go.Figure(data=go.Heatmap(
        z=acts.T,
        x=labels,
        y=[f"Prim {i}" for i in range(8)],
        colorscale='Magma',
        hoverongaps=False
    ))
    
    fig2.update_layout(
        title="Active Primitives per Character",
        xaxis_title="Input Sequence",
        yaxis_title="Primitive ID",
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

    tail_n = st.number_input("Tail (records)", min_value=100, max_value=200000, value=5000, step=100)
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_s = st.slider("Refresh seconds", 0.5, 10.0, 2.0, 0.5)
    refresh_btn = st.button("Refresh now")

    if auto_refresh and not refresh_btn:
        time.sleep(float(refresh_s))
        st.rerun()

    metrics_path = os.path.splitext(str(ckpt_path))[0] + ".metrics.jsonl"
    events_path = os.path.splitext(str(ckpt_path))[0] + ".events.jsonl"
    max_lines = int(tail_n)
    mp = os.path.abspath(str(metrics_path))
    ep = os.path.abspath(str(events_path))

    def _stat_line(p: str) -> str:
        if not p:
            return "(empty)"
        if not os.path.exists(p):
            return "missing"
        try:
            sz = os.path.getsize(p)
            mt = os.path.getmtime(p)
            return f"exists  size={sz}B  mtime={mt:.0f}"
        except Exception:
            return "exists"

    st.markdown("**Resolved paths:**")
    st.code(
        "\n".join(
            [
                f"cwd: {os.getcwd()}",
                f"metrics: {mp}  ({_stat_line(mp)})",
                f"events:  {ep}  ({_stat_line(ep)})",
            ]
        )
    )

    metrics = _read_records(mp, max_lines=int(max_lines))
    events = _read_records(ep, max_lines=int(max_lines))

    st.markdown(f"**Loaded records:** metrics={len(metrics)} events={len(events)}")

    if int(tail_n) > 0:
        if len(metrics) > int(tail_n):
            metrics = metrics[-int(tail_n):]
        if len(events) > int(tail_n):
            events = events[-int(tail_n):]

    if not metrics and not events:
        st.info("No log records found yet. Enable --log_metrics and/or --log_train_events during training.")
        st.stop()

    cols = st.columns(4)
    if events:
        last_e = events[-1]
        cols[0].metric("Last step (events)", int(last_e.get("step", 0)))
        cols[1].metric("Loss", _safe_float(last_e.get("loss", 0.0)))
        cols[2].metric("Spike", bool(last_e.get("is_spike", False)))
        cols[3].metric("Tiny mult", _safe_float(last_e.get("tiny_mult", 1.0)))
    elif metrics:
        last_m = metrics[-1]
        cols[0].metric("Last step (metrics)", int(last_m.get("step", 0)))
        cols[1].metric("Loss", _safe_float(last_m.get("loss", 0.0)))
        cols[2].metric("Mean entropy", _safe_float(last_m.get("mean_entropy", 0.0)))
        cols[3].metric("Mean maxprob", _safe_float(last_m.get("mean_maxprob", 0.0)))

    if events:
        xs = [int(r.get("step", 0)) for r in events]
        ys_loss = [_safe_float(r.get("loss", None)) for r in events]
        ys_ema = [_safe_float(r.get("ema_loss", None)) for r in events]
        ys_mult = [_safe_float(r.get("tiny_mult", 1.0)) for r in events]
        ys_spike = [1.0 if bool(r.get("is_spike", False)) else 0.0 for r in events]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys_loss, mode="lines", name="loss"))
        fig.add_trace(go.Scatter(x=xs, y=ys_ema, mode="lines", name="ema_loss"))
        fig.update_layout(height=320, margin=dict(l=0, r=0, b=0, t=30), title="Training Loss (events)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=xs, y=ys_mult, mode="lines", name="tiny_mult"))
        fig2.add_trace(go.Scatter(x=xs, y=ys_spike, mode="lines", name="spike"))
        fig2.update_layout(height=280, margin=dict(l=0, r=0, b=0, t=30), title="Buff Mode Signals")
        st.plotly_chart(fig2, use_container_width=True)

    if metrics:
        xs = [int(r.get("step", 0)) for r in metrics]
        ent = [_safe_float(r.get("mean_entropy", None)) for r in metrics]
        maxp = [_safe_float(r.get("mean_maxprob", None)) for r in metrics]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=xs, y=ent, mode="lines", name="mean_entropy"))
        fig3.add_trace(go.Scatter(x=xs, y=maxp, mode="lines", name="mean_maxprob"))
        fig3.update_layout(height=320, margin=dict(l=0, r=0, b=0, t=30), title="Controller Summary (metrics)")
        st.plotly_chart(fig3, use_container_width=True)

        counts = [r.get("argmax_counts") for r in metrics if isinstance(r.get("argmax_counts"), (list, tuple))]
        if counts:
            last = counts[-1]
            fig4 = go.Figure(data=[go.Bar(x=[f"Prim {i}" for i in range(len(last))], y=list(last))])
            fig4.update_layout(height=280, margin=dict(l=0, r=0, b=0, t=30), title="Argmax primitive counts (latest metrics window)")
            st.plotly_chart(fig4, use_container_width=True)


elif mode == "Chat":
    st.header("Chat")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    with st.sidebar:
        st.subheader("Chat Settings")
        st.subheader("Delta / Eraser")
        delta_enabled = st.checkbox("Enable Delta overwrite (entropy eraser)", value=False)
        delta_entropy_lo = st.slider("Eraser entropy low", 0.0, 1.0, 0.55, 0.01)
        delta_entropy_hi = st.slider("Eraser entropy high", 0.0, 1.0, 0.95, 0.01)
        delta_beta_max = st.slider("Eraser beta max", 0.0, 1.0, 1.0, 0.05)

        display_mode = st.selectbox(
            "Byte display mode",
            ["Escaped bytes (recommended)", "UTF-8 (replace errors)"],
            index=0,
        )
        tag_display = st.selectbox(
            "POS tag display",
            ["Keep raw", "Hide tags", "Render <TAGS>"],
            index=1,
        )
        show_live_panel = st.checkbox("Show live retrieval panel", value=True)
        live_move_k = int(st.slider("Live move-graph k", min_value=1, max_value=4, value=2, step=1))
        temperature = st.slider("Temperature", 0.01, 2.0, 0.7, 0.01)
        top_k = st.slider("Top-k", 0, 200, 50, 1)
        max_len = st.slider("Max response bytes", 16, 4096, 256, 16)
        ascii_only = st.checkbox("Restrict to printable ASCII", value=False)
        repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.1, 0.05)
        repetition_window = st.slider("Repetition window (bytes)", 8, 256, 64, 8)
        stop_on_newline = st.checkbox("Stop on newline", value=True)
        enable_retrieval = st.checkbox("Enable retrieval", value=True)

        st.subheader("Backspace Solver")
        enable_solver = st.checkbox("Enable backspace solver", value=False)
        solver_lookahead = int(st.slider("Solver lookahead (bytes)", 4, 32, 12, 1))
        solver_topk = int(st.slider("Solver candidate top-k", 2, 64, 8, 1))
        solver_min_commit = int(st.slider("Min commit length", 1, 64, 12, 1))
        solver_commit_score = float(st.slider("Commit score threshold", -10.0, 0.0, -2.0, 0.1))
        solver_drop_delta = float(st.slider("Drift delta", 0.0, 5.0, 0.8, 0.1))
        solver_fail_threshold = int(st.slider("Fail threshold (same anchor)", 1, 10, 3, 1))
        solver_hint_len = int(st.slider("Hint bytes", 0, 16, 4, 1))
        solver_hint_target_prefix_len = int(st.slider("Hint target prefix len", 4, 64, 24, 1))
        solver_hint_beam_width = int(st.slider("Hint beam width", 2, 64, 8, 1))
        solver_hint_expand_topk = int(st.slider("Hint expand top-k", 2, 64, 8, 1))
        solver_max_rewinds = int(st.slider("Max rewinds", 0, 256, 64, 1))
        solver_max_hints = int(st.slider("Max hints", 0, 32, 6, 1))

        st.subheader("Live Streaming")
        stream_live = st.checkbox("Stream output live", value=False)
        stream_every = int(st.slider("Stream every N bytes", 1, 32, 1, 1))
        stream_min_interval_s = float(st.slider("Stream min interval (s)", 0.0, 0.50, 0.05, 0.01))
        stream_show_backspace = st.checkbox("Show backspace flashes", value=True)
        stream_sleep_s = float(st.slider("Visual delay per update (s)", 0.0, 0.20, 0.00, 0.01))
        stream_backspace_sleep_s = float(st.slider("Extra delay on backspace (s)", 0.0, 0.50, 0.10, 0.01))

        # Quick visibility into whether the loaded model can actually do retrieval.
        st.code(
            "\n".join(
                [
                    f"model.retrieval_enabled={bool(getattr(model, 'retrieval_enabled', False))}",
                    f"model.path_retrieval_enabled={bool(getattr(model, 'path_retrieval_enabled', False))}",
                    f"model.symmetry_retrieval_enabled={bool(getattr(model, 'symmetry_retrieval_enabled', False))}",
                    f"effective_retrieval={bool(enable_retrieval) and bool(getattr(model, 'retrieval_enabled', False))}",
                ]
            )
        )
        if st.button("Clear chat"):
            st.session_state.chat_messages = []

    def _render_live_retrieval_panel(*, telemetry: dict, reply_bytes: bytes, key_suffix: str) -> None:
        steps = int(telemetry.get("steps", 0))
        used = int(telemetry.get("retrieval_used", 0))
        hits = int(telemetry.get("hit_steps", 0))
        hit_rate = (float(hits) / float(used)) if used > 0 else 0.0
        avg_dist = (float(telemetry.get("avg_dist_sum", 0.0)) / float(hits)) if hits > 0 else 0.0
        enabled = bool(telemetry.get("retrieval_enabled", False))
        solver_enabled = bool(telemetry.get("solver_enabled", False))
        solver_rewinds = int(telemetry.get("solver_rewinds", 0))
        solver_commits = int(telemetry.get("solver_commits", 0))
        solver_hints = int(telemetry.get("solver_hints", 0))

        with st.expander("Live retrieval telemetry (this response)", expanded=False):
            if steps == 0:
                if bool(telemetry.get("stopped_immediately")):
                    st.info("No generation steps were taken (the model sampled a newline immediately). Try increasing temperature or disabling stop-on-newline.")
                else:
                    st.info("No generation steps were taken.")

            c0, c1 = st.columns([2, 1])
            with c0:
                st.code(
                    "\n".join(
                        [
                            f"retrieval_enabled={enabled}",
                            f"solver_enabled={solver_enabled}",
                            f"gen_steps={steps}",
                            f"retrieval_used={used}",
                            f"hit_steps={hits}",
                            f"hit_rate={hit_rate:.3f}",
                            f"avg_hit_dist={avg_dist:.3f}",
                            f"solver_rewinds={solver_rewinds}",
                            f"solver_commits={solver_commits}",
                            f"solver_hints={solver_hints}",
                        ]
                    )
                )

            with c1:
                plot_mode = st.checkbox("Plot", value=False, key=f"tele_plot_{key_suffix}")
                plot_3d = st.checkbox("3D", value=False, key=f"tele_3d_{key_suffix}")

            if bool(plot_mode) and steps > 0:
                hit_seq = telemetry.get("hit_seq")
                dist_seq = telemetry.get("avg_dist_seq")

                xs = list(range(int(steps)))
                hs = []
                ds = []
                for i in range(int(steps)):
                    h = False
                    d = None
                    if isinstance(hit_seq, list) and i < len(hit_seq):
                        h = bool(hit_seq[i])
                    if isinstance(dist_seq, list) and i < len(dist_seq):
                        d = dist_seq[i]
                    hs.append(1.0 if h else 0.0)
                    ds.append(None if d is None else float(d))

                if bool(plot_3d):
                    pts_x = []
                    pts_y = []
                    pts_z = []
                    pts_c = []
                    for i in range(int(steps)):
                        if ds[i] is None:
                            continue
                        pts_x.append(float(xs[i]))
                        pts_y.append(float(ds[i]))
                        pts_z.append(float(hs[i]))
                        pts_c.append("hit" if hs[i] > 0.5 else "miss")

                    fig3 = go.Figure()
                    fig3.add_trace(
                        go.Scatter3d(
                            x=pts_x,
                            y=pts_y,
                            z=pts_z,
                            mode="markers",
                            marker=dict(size=4, color=[1 if c == "hit" else 0 for c in pts_c], colorscale=[[0, "#dd6666"], [1, "#44dd88"]]),
                            name="steps",
                        )
                    )
                    fig3.update_layout(
                        height=260,
                        margin=dict(l=0, r=0, b=0, t=0),
                        scene=dict(
                            xaxis_title="step",
                            yaxis_title="dist",
                            zaxis_title="hit",
                        ),
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    cc0, cc1 = st.columns(2)
                    with cc0:
                        fig_t = go.Figure()
                        fig_t.add_trace(
                            go.Scatter(
                                x=xs,
                                y=ds,
                                mode="lines+markers",
                                name="avg_dist",
                                line=dict(color="#66ccff"),
                            )
                        )
                        fig_t.update_layout(height=220, margin=dict(l=0, r=0, b=0, t=20), title="Avg dist")
                        st.plotly_chart(fig_t, use_container_width=True)
                    with cc1:
                        fig_b = go.Figure(
                            data=[
                                go.Bar(
                                    x=["hit", "miss"],
                                    y=[float(hits), float(max(0, used - hits))],
                                    marker=dict(color=["#44dd88", "#dd6666"]),
                                )
                            ]
                        )
                        fig_b.update_layout(height=220, margin=dict(l=0, r=0, b=0, t=20), title="Hit/miss")
                        st.plotly_chart(fig_b, use_container_width=True)

        if reply_bytes:
            last_b = int(reply_bytes[-1])
            if last_b in _TAG_LABELS:
                st.markdown("**Last byte is a POS tag (hidden). Live move-graph disabled for tag bytes.**")
            else:
                reserved = {128, 129, 130}

                @st.cache_resource
                def _live_move_tables(k: int):
                    neigh, dist = build_k_hop_tables(k=int(k), reserved=tuple(sorted(reserved)))
                    return neigh.cpu().numpy(), dist.cpu().numpy()

                neigh_np, dist_np = _live_move_tables(int(live_move_k))
                neigh_row = neigh_np[last_b]
                dist_row = dist_np[last_b]
                pairs = []
                for nb, dd in zip(neigh_row.tolist(), dist_row.tolist()):
                    if int(nb) < 0:
                        continue
                    pairs.append((int(nb), float(dd)))
                pairs.sort(key=lambda x: (x[1], x[0]))

                if pairs:
                    st.markdown(f"**Live move-graph neighbors for last byte={last_b} ({_byte_label(last_b)})**")
                    st.code("\n".join([f"{nb:3d}  {_byte_label(nb):>6s}  dist={dd:.0f}" for nb, dd in pairs[:64]]))

        tops = telemetry.get("top_neighbors")
        if isinstance(tops, list) and tops:
            last_top = tops[-1]
            if isinstance(last_top, list) and last_top:
                st.markdown("**Top weighted neighbors (last step):**")
                st.code(
                    "\n".join(
                        [
                            f"{int(nb):3d}  {_byte_label(int(nb)):>6s}  w={float(w):.4f}  dist={float(d):.0f}"
                            for (nb, w, d) in last_top
                        ]
                    )
                )

    for mi, m in enumerate(st.session_state.chat_messages):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and isinstance(m.get("raw_bytes"), (bytes, bytearray)):
                b0 = bytes(m["raw_bytes"])
                b1 = _apply_tag_display(b0, tag_display)
                st.markdown(_format_bytes_for_display(b1, display_mode))

                if show_live_panel and isinstance(m.get("telemetry"), dict):
                    _render_live_retrieval_panel(
                        telemetry=dict(m["telemetry"]),
                        reply_bytes=b0,
                        key_suffix=str(mi),
                    )
            else:
                st.markdown(m["content"]) 

    user_text = st.chat_input("Type a question...")
    if user_text:
        st.session_state.chat_messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        prompt_text = _build_chat_prompt(st.session_state.chat_messages) + "Answer:"

        if hasattr(model, "set_delta_erase"):
            model.set_delta_erase(
                delta_enabled,
                entropy_lo=float(delta_entropy_lo),
                entropy_hi=float(delta_entropy_hi),
                beta_max=float(delta_beta_max),
            )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                live_ph = st.empty()
                _live_state = {"last_len": 0, "flash_until": 0.0}

                def _render_live(b: bytes) -> None:
                    shown = _apply_tag_display(bytes(b), tag_display)
                    txt = _format_bytes_for_display(shown, display_mode)
                    now = time.time()
                    if bool(stream_show_backspace) and len(shown) < int(_live_state["last_len"]):
                        _live_state["flash_until"] = float(now) + 0.20
                    _live_state["last_len"] = len(shown)
                    prefix = "⌫ " if bool(stream_show_backspace) and float(now) < float(_live_state["flash_until"]) else ""
                    # A lightweight "typewriter" view; backspaces show as real deletions.
                    live_ph.markdown(
                        "<div style='opacity:0.65;font-family:monospace;white-space:pre-wrap;'>"
                        + html.escape(prefix + txt + ("▌" if bool(stream_live) else ""))
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                if bool(stream_live):
                    _render_live(b"")

                reply_bytes, telemetry = generate_chat_turn(
                    model,
                    prompt_text=prompt_text,
                    temperature=float(temperature),
                    top_k=int(top_k),
                    max_len=int(max_len),
                    ascii_only=bool(ascii_only),
                    repetition_penalty=float(repetition_penalty),
                    repetition_window=int(repetition_window),
                    stop_on_newline=bool(stop_on_newline),
                    enable_retrieval=bool(enable_retrieval),
                    enable_backspace_solver=bool(enable_solver),
                    solver_lookahead=int(solver_lookahead),
                    solver_topk=int(solver_topk),
                    solver_min_commit=int(solver_min_commit),
                    solver_commit_score=float(solver_commit_score),
                    solver_drop_delta=float(solver_drop_delta),
                    solver_fail_threshold=int(solver_fail_threshold),
                    solver_hint_len=int(solver_hint_len),
                    solver_hint_beam_width=int(solver_hint_beam_width),
                    solver_hint_expand_topk=int(solver_hint_expand_topk),
                    solver_hint_target_prefix_len=int(solver_hint_target_prefix_len),
                    solver_max_rewinds=int(solver_max_rewinds),
                    solver_max_hints=int(solver_max_hints),
                    stream_callback=_render_live if bool(stream_live) else None,
                    stream_every=int(stream_every),
                    stream_min_interval_s=float(stream_min_interval_s),
                    stream_sleep_s=float(stream_sleep_s),
                    stream_backspace_sleep_s=float(stream_backspace_sleep_s),
                )
            # Clear the live placeholder after final render so only one final answer stays.
            live_ph.empty()
            shown_bytes = _apply_tag_display(reply_bytes, tag_display)
            st.markdown(_format_bytes_for_display(shown_bytes, display_mode))

            if show_live_panel and isinstance(telemetry, dict):
                _render_live_retrieval_panel(
                    telemetry=dict(telemetry),
                    reply_bytes=bytes(reply_bytes),
                    key_suffix=str(len(st.session_state.chat_messages)),
                )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": _format_bytes_for_display(shown_bytes, "UTF-8 (replace errors)"),
                "raw_bytes": reply_bytes,
                "telemetry": telemetry,
            }
        )


elif mode == "Mirror Lab":
    st.header("Mirror Lab")
    st.markdown("Session-only adaptation: updates a copy of the checkpoint in RAM based on your example. Nothing is saved unless you manually save a checkpoint.")

    if "mirror_base_sd" not in st.session_state:
        st.session_state.mirror_base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if "mirror_sd_sig" not in st.session_state:
        st.session_state.mirror_sd_sig = ckpt_path

    # If user changes checkpoint path, reset base.
    if st.session_state.mirror_sd_sig != ckpt_path:
        st.session_state.mirror_base_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        st.session_state.mirror_sd_sig = ckpt_path
        st.session_state.pop("mirror_adapted_sd", None)

    colA, colB = st.columns([2, 1])
    with colA:
        prompt_text = st.text_area("Prompt", "The cat is orange.\nAnswer:")
        target_text = st.text_area("Your preferred continuation (gold)", "Orange cats are known to be wild cards.")
    with colB:
        adapt_steps = int(st.slider("Adapt steps", min_value=0, max_value=200, value=25, step=1))
        adapt_lr = float(st.number_input("Adapt LR", min_value=1e-6, max_value=1e-1, value=1e-3, step=1e-3, format="%.6f"))
        adapt_which = st.selectbox(
            "Adapt parameters",
            ["out only", "controller+out", "all"],
            index=0,
        )
        ascii_only = st.checkbox("Restrict to printable ASCII (generation)", value=False)
        temperature = float(st.slider("Temperature", 0.01, 2.0, 0.7, 0.01))
        top_k = int(st.slider("Top-k", 0, 200, 50, 1))
        max_len = int(st.slider("Max response bytes", 8, 4096, 256, 8))
        run_adapt = st.button("Adapt")
        reset_adapt = st.button("Reset to base")

    if reset_adapt:
        st.session_state.pop("mirror_adapted_sd", None)

    # Build a fresh model for adaptation/generation each rerun.
    device = next(model.parameters()).device
    mirror_model = PrimitiveManifoldLM(model.cfg).to(device)
    base_sd = st.session_state.mirror_base_sd
    mirror_model.load_state_dict({k: v.to(device) for k, v in base_sd.items()}, strict=True)

    if "mirror_adapted_sd" in st.session_state:
        mirror_model.load_state_dict({k: v.to(device) for k, v in st.session_state.mirror_adapted_sd.items()}, strict=True)

    def _to_ids(s: str) -> list:
        return list(s.encode("utf-8"))

    if run_adapt and adapt_steps > 0:
        if not prompt_text:
            st.warning("Prompt is empty.")
            st.stop()
        if not target_text:
            st.warning("Gold continuation is empty.")
            st.stop()

        prompt_ids = _to_ids(prompt_text)
        target_ids = _to_ids(target_text)

        # Select parameters to adapt (existing weights only).
        for p in mirror_model.parameters():
            p.requires_grad_(False)

        params = []
        for name, p in mirror_model.named_parameters():
            if adapt_which == "out only":
                if name.startswith("out."):
                    p.requires_grad_(True)
                    params.append(p)
            elif adapt_which == "controller+out":
                if name.startswith("out.") or name.startswith("controller."):
                    p.requires_grad_(True)
                    params.append(p)
            else:
                p.requires_grad_(True)
                params.append(p)

        opt = torch.optim.AdamW(params, lr=adapt_lr)

        x = torch.tensor([prompt_ids + target_ids[:-1]], device=device, dtype=torch.long)
        y = torch.tensor([target_ids], device=device, dtype=torch.long)

        with st.spinner("Adapting in RAM..."):
            mirror_model.train()
            for _ in range(int(adapt_steps)):
                opt.zero_grad(set_to_none=True)
                logits, _ = mirror_model(x)
                pred = logits[:, -len(target_ids):, :]
                loss = torch.nn.functional.cross_entropy(
                    pred.reshape(-1, 256),
                    y.reshape(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

        st.success(f"Adapted. Final loss={float(loss.detach().cpu()):.4f}")
        st.session_state.mirror_adapted_sd = {k: v.detach().cpu().clone() for k, v in mirror_model.state_dict().items()}

    st.subheader("Generate")
    if st.button("Generate with current (adapted if any)"):
        mirror_model.eval()
        reply_bytes, _telemetry = generate_chat_turn(
            mirror_model,
            prompt_text=prompt_text,
            temperature=float(temperature),
            top_k=int(top_k),
            max_len=int(max_len),
            ascii_only=bool(ascii_only),
            repetition_penalty=1.1,
            repetition_window=64,
        )
        st.markdown(_format_bytes_for_display(reply_bytes, "Escaped bytes (recommended)"))


elif mode == "Checkpoint Viewer":
    st.header("Checkpoint Viewer")

    st.subheader("Byte Mind Map")
    colA, colB = st.columns([1, 2])
    with colA:
        byte_val = int(st.number_input("Byte (0-255)", min_value=0, max_value=255, value=97, step=1))
        char_in = st.text_input("Char (optional)", "a", max_chars=1)
        if char_in:
            byte_val = int(ord(char_in[0]))
        nn_k = int(st.slider("Nearest neighbors (k)", min_value=0, max_value=50, value=10, step=1))
        show_labels = st.checkbox("Show labels", value=False)
        max_pts = int(st.slider("Max points", min_value=64, max_value=256, value=256, step=1))

        st.markdown("---")
        st.subheader("Byte Move Graph")
        show_move_graph = st.checkbox("Overlay move-graph neighbors", value=True)
        move_k = int(st.slider("Move-graph radius (k)", min_value=1, max_value=4, value=2, step=1))

    with torch.no_grad():
        ids_all = torch.arange(256, device=next(model.parameters()).device, dtype=torch.long)
        emb_all = model._get_emb(ids_all).detach().float().cpu().numpy()  # (256, d_model)

    if max_pts < 256:
        # deterministic downsample but always include selected byte
        keep = np.linspace(0, 255, num=max_pts, dtype=int)
        if byte_val not in keep:
            keep[0] = byte_val
        id_view = np.unique(keep)
    else:
        id_view = np.arange(256)

    # PCA to 3D (stable: computed on all 256 bytes)
    mu = emb_all.mean(axis=0, keepdims=True)
    x_all = emb_all - mu
    u, s, vt = np.linalg.svd(x_all, full_matrices=False)
    coords_all = x_all @ vt[:3].T  # (256, 3)
    coords = coords_all[id_view]

    labels = [_byte_label(int(b)) for b in id_view]
    hover = [f"byte={int(b)}<br>label={_byte_label(int(b))}" for b in id_view]

    color = id_view.astype(float)

    figm = go.Figure()
    figm.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers" + ("+text" if show_labels else ""),
        marker=dict(size=4, color=color, colorscale="Viridis", opacity=0.6),
        text=labels if show_labels else None,
        hovertext=hover,
        hoverinfo="text",
        name="Bytes",
    ))

    # Selected byte highlight (separate trace so we can control styling)
    figm.add_trace(go.Scatter3d(
        x=[float(coords_all[byte_val, 0])],
        y=[float(coords_all[byte_val, 1])],
        z=[float(coords_all[byte_val, 2])],
        mode="markers+text",
        marker=dict(size=10, color="#ff4d4d", opacity=1.0),
        text=[_byte_label(byte_val)] if show_labels else None,
        textposition="top center",
        hovertext=[f"SELECTED<br>byte={int(byte_val)}<br>label={_byte_label(int(byte_val))}"],
        hoverinfo="text",
        name="Selected",
    ))

    # Overlay move-graph neighborhood (rot/inv/rev/inc/dec BFS) in embedding PCA space.
    reserved = {128, 129, 130}

    @st.cache_resource
    def _move_tables(k: int):
        neigh, dist = build_k_hop_tables(k=int(k), reserved=tuple(sorted(reserved)))
        return neigh.cpu().numpy(), dist.cpu().numpy()

    move_neighbors_text = None
    if show_move_graph:
        if int(byte_val) in reserved:
            move_neighbors_text = "Selected byte is a reserved POS tag; move-graph neighbors are disabled for tag bytes."
        else:
            neigh_np, dist_np = _move_tables(int(move_k))
            neigh_row = neigh_np[int(byte_val)]
            dist_row = dist_np[int(byte_val)]

            pairs = []
            for nb, dd in zip(neigh_row.tolist(), dist_row.tolist()):
                if int(nb) < 0:
                    continue
                pairs.append((int(nb), float(dd)))
            pairs.sort(key=lambda x: (x[1], x[0]))

            if pairs:
                move_neighbors_text = "\n".join(
                    [f"{nb:3d}  {_byte_label(nb):>6s}  dist={dd:.0f}" for nb, dd in pairs]
                )

                nb_ids = np.array([nb for nb, _dd in pairs], dtype=int)
                nb_coords = coords_all[nb_ids]
                nb_colors = np.array([dd for _nb, dd in pairs], dtype=float)

                figm.add_trace(
                    go.Scatter3d(
                        x=nb_coords[:, 0],
                        y=nb_coords[:, 1],
                        z=nb_coords[:, 2],
                        mode="markers+text",
                        marker=dict(
                            size=8,
                            color=nb_colors,
                            colorscale="Blues",
                            cmin=1.0,
                            cmax=float(max(nb_colors.max(), 1.0)),
                            opacity=1.0,
                            line=dict(color="#111111", width=1),
                        ),
                        text=[_byte_label(int(b)) for b in nb_ids] if show_labels else None,
                        textposition="top center",
                        hovertext=[
                            f"MOVE-NB<br>byte={int(b)}<br>label={_byte_label(int(b))}<br>dist={float(d):.0f}"
                            for (b, d) in pairs
                        ],
                        hoverinfo="text",
                        name="Move neighbors",
                    )
                )

                # Draw edges from selected -> neighbor
                xs = []
                ys = []
                zs = []
                x0, y0, z0 = coords_all[int(byte_val)]
                for nb, _dd in pairs:
                    x1, y1, z1 = coords_all[int(nb)]
                    xs.extend([float(x0), float(x1), None])
                    ys.extend([float(y0), float(y1), None])
                    zs.extend([float(z0), float(z1), None])

                figm.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line=dict(color="#99c2ff", width=3),
                        hoverinfo="skip",
                        name="Move edges",
                    )
                )
            else:
                move_neighbors_text = "No move-graph neighbors found (unexpected)."

    # Optional nearest neighbors in embedding space (full 256, not downsampled)
    nn_text = None
    if nn_k > 0:
        v = emb_all[byte_val]
        d = np.linalg.norm(emb_all - v[None, :], axis=1)
        order = np.argsort(d)
        order = [int(i) for i in order if int(i) != byte_val][:nn_k]
        nn_text = "\n".join([f"{i:3d}  {_byte_label(i):>6s}  dist={float(d[i]):.4f}" for i in order])

        nn_ids = np.array(order, dtype=int)
        nn_coords = coords_all[nn_ids]
        figm.add_trace(go.Scatter3d(
            x=nn_coords[:, 0], y=nn_coords[:, 1], z=nn_coords[:, 2],
            mode="markers+text",
            marker=dict(size=7, color="#ffcc00", opacity=1.0),
            text=[_byte_label(int(b)) for b in nn_ids],
            textposition="top center",
            hovertext=[f"NN<br>byte={int(b)}<br>label={_byte_label(int(b))}<br>dist={float(d[int(b)]):.4f}" for b in nn_ids],
            hoverinfo="text",
            name="Nearest",
        ))

    figm.update_layout(
        height=600,
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"Byte embedding PCA (selected {byte_val} {_byte_label(byte_val)})",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    )

    with colB:
        st.plotly_chart(figm, use_container_width=True)
        if nn_text is not None:
            st.markdown("**Nearest neighbors (by embedding distance):**")
            st.code(nn_text)
        if move_neighbors_text is not None:
            st.markdown("**Move-graph neighbors (rot/inv/rev/inc/dec) within k:**")
            st.code(move_neighbors_text)

    st.subheader("Symmetry Graph")
    sg_col1, sg_col2 = st.columns([1, 2])

    def _bit_reverse_u8(v: int) -> int:
        x = int(v) & 0xFF
        r = 0
        for _ in range(8):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r

    def _rotl_u8(v: int, r: int) -> int:
        x = int(v) & 0xFF
        rr = int(r) % 8
        return ((x << rr) | (x >> (8 - rr))) & 0xFF

    with sg_col1:
        base_chars = st.text_input("Base chars (comma-separated)", "a,e,t, ")
        base_bytes_csv = st.text_input("Base bytes (comma-separated ints; optional)", "")

        include_invert = st.checkbox("Include invert (x ^ 0xFF)", value=True)
        include_bit_reverse = st.checkbox("Include bit-reverse", value=True)
        include_rotations = st.checkbox("Include bit-rotations", value=False)
        rot_max = int(st.slider("Rotations to include (1..N)", min_value=1, max_value=7, value=3, step=1))

        show_context = st.checkbox("Show reference printable ASCII cloud", value=True)
        context_pts = int(st.slider("Context points", min_value=32, max_value=95, value=95, step=1))

    def _parse_bytes_csv(s: str):
        out = []
        for part in s.split(","):
            p = part.strip()
            if not p:
                continue
            try:
                v = int(p)
            except Exception:
                continue
            if 0 <= v <= 255:
                out.append(v)
        return out

    base_ids = []
    for part in base_chars.split(","):
        p = part
        if p == "":
            continue
        # preserve user ability to specify a literal comma/space etc.
        p = p.strip("\n\r")
        if p == "":
            continue
        ch = p[0]
        base_ids.append(int(ord(ch)) & 0xFF)
    base_ids.extend(_parse_bytes_csv(base_bytes_csv))
    base_ids = [int(x) for x in dict.fromkeys(base_ids)]  # stable unique
    base_ids = base_ids[:16]

    # Build symmetry edges and node groups
    node_types = {}  # byte -> set(types)
    edges = []  # (src, dst, kind)
    for b0 in base_ids:
        node_types.setdefault(int(b0), set()).add("base")
        if include_invert:
            bi = int(b0) ^ 0xFF
            node_types.setdefault(int(bi), set()).add("invert")
            edges.append((int(b0), int(bi), "invert"))
        if include_bit_reverse:
            br = _bit_reverse_u8(int(b0))
            node_types.setdefault(int(br), set()).add("bit_reverse")
            edges.append((int(b0), int(br), "bit_reverse"))
        if include_rotations:
            for r in range(1, int(rot_max) + 1):
                bt = _rotl_u8(int(b0), r)
                node_types.setdefault(int(bt), set()).add(f"rot{r}")
                edges.append((int(b0), int(bt), f"rot{r}"))

    node_ids = sorted(node_types.keys())

    # Context bytes for reference
    if show_context:
        printable = np.arange(32, 127, dtype=int)
        if context_pts < len(printable):
            printable = printable[np.linspace(0, len(printable) - 1, num=context_pts, dtype=int)]
        ctx_ids = [int(x) for x in printable.tolist() if int(x) not in node_types]
    else:
        ctx_ids = []

    # Build mapping table
    rows = []
    for src, dst, kind in edges:
        rows.append(
            {
                "base_byte": int(src),
                "base_label": _byte_label(int(src)),
                "kind": str(kind),
                "sym_byte": int(dst),
                "sym_label": _byte_label(int(dst)),
                "sym_hex": f"0x{int(dst):02x}",
            }
        )

    # Plot graph in PCA embedding space
    figg = go.Figure()
    if len(ctx_ids) > 0:
        ctx_coords = coords_all[np.array(ctx_ids, dtype=int)]
        figg.add_trace(go.Scatter3d(
            x=ctx_coords[:, 0], y=ctx_coords[:, 1], z=ctx_coords[:, 2],
            mode="markers",
            marker=dict(size=3, color="#888888", opacity=0.15),
            hoverinfo="skip",
            name="Printable ASCII (context)",
        ))

    # Edge traces
    edge_colors = {
        "invert": "#4ea1ff",
        "bit_reverse": "#30c46a",
    }
    for src, dst, kind in edges:
        c = edge_colors.get(kind, "#b074ff")
        a = coords_all[int(src)]
        b = coords_all[int(dst)]
        figg.add_trace(go.Scatter3d(
            x=[float(a[0]), float(b[0])],
            y=[float(a[1]), float(b[1])],
            z=[float(a[2]), float(b[2])],
            mode="lines",
            line=dict(color=c, width=5),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Node traces by type priority
    def _add_nodes(kind: str, color: str, size: int):
        ids = [b for b in node_ids if kind in node_types.get(int(b), set())]
        if len(ids) == 0:
            return
        cc = coords_all[np.array(ids, dtype=int)]
        figg.add_trace(go.Scatter3d(
            x=cc[:, 0], y=cc[:, 1], z=cc[:, 2],
            mode="markers+text",
            marker=dict(size=size, color=color, opacity=1.0),
            text=[_byte_label(int(b)) for b in ids],
            textposition="top center",
            hovertext=[f"byte={int(b)}<br>label={_byte_label(int(b))}<br>kind={kind}" for b in ids],
            hoverinfo="text",
            name=kind,
        ))

    _add_nodes("base", "#ff4d4d", 10)
    if include_invert:
        _add_nodes("invert", "#4ea1ff", 8)
    if include_bit_reverse:
        _add_nodes("bit_reverse", "#30c46a", 8)
    if include_rotations:
        _add_nodes("rot1", "#b074ff", 7)

    figg.update_layout(
        height=650,
        margin=dict(l=0, r=0, b=0, t=30),
        title="Symmetry Graph in byte-embedding PCA space",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
    )

    with sg_col2:
        if len(base_ids) == 0:
            st.info("Enter at least one base char/byte to render the symmetry graph.")
        else:
            st.plotly_chart(figg, use_container_width=True)
            if len(rows) > 0:
                st.markdown("**Mappings:**")
                st.dataframe(rows, use_container_width=True)

    sd = model.state_dict()
    names = list(sd.keys())
    if len(names) == 0:
        st.info("No parameters found in model state_dict().")
        st.stop()

    sel = st.selectbox("Tensor", names, index=0)
    w = sd[sel].detach().to("cpu")
    st.markdown(f"**Shape**: `{tuple(w.shape)}`")
    st.markdown(f"**Dtype**: `{w.dtype}`")

    wf = w.float()
    st_cols = st.columns(5)
    st_cols[0].metric("min", float(wf.min().item()))
    st_cols[1].metric("max", float(wf.max().item()))
    st_cols[2].metric("mean", float(wf.mean().item()))
    st_cols[3].metric("std", float(wf.std(unbiased=False).item()))
    st_cols[4].metric("abs mean", float(wf.abs().mean().item()))

    flat = wf.reshape(-1)
    max_hist = int(st.slider("Histogram sample (elements)", min_value=512, max_value=200000, value=20000, step=512))
    if flat.numel() > max_hist:
        idx = torch.linspace(0, flat.numel() - 1, steps=max_hist).long()
        flat_hist = flat[idx]
    else:
        flat_hist = flat

    fig_hist = go.Figure(data=[go.Histogram(x=flat_hist.numpy(), nbinsx=120)])
    fig_hist.update_layout(height=260, margin=dict(l=0, r=0, b=0, t=20), title="Value Histogram")
    st.plotly_chart(fig_hist, use_container_width=True)

    max_view = int(st.slider("Max elements to render", min_value=256, max_value=200000, value=20000, step=256))

    if w.ndim == 0:
        st.code(str(w.item()))
        st.stop()

    view = w
    if w.ndim > 2:
        fixed = []
        for d in range(w.ndim - 2):
            fixed.append(
                st.slider(
                    f"Index dim {d}",
                    min_value=0,
                    max_value=int(w.shape[d]) - 1,
                    value=0,
                    step=1,
                )
            )
        view = w[tuple(int(x) for x in fixed)]

    if view.ndim == 1:
        n = int(view.shape[0])
        if n > max_view:
            start = int(st.slider("Slice start", min_value=0, max_value=max(0, n - max_view), value=0, step=1))
            end = min(n, start + max_view)
            v = view[start:end].float().numpy()
            st.markdown(f"Showing `[{start}:{end}]` of `{n}`")
        else:
            v = view.float().numpy()
        fig_line = go.Figure(data=[go.Scatter(y=v, mode="lines")])
        fig_line.update_layout(height=260, margin=dict(l=0, r=0, b=0, t=20), title="1D Slice")
        st.plotly_chart(fig_line, use_container_width=True)
        st.stop()

    if view.ndim == 2:
        r, c = int(view.shape[0]), int(view.shape[1])
        max_side = int(max(8, np.sqrt(max_view)))
        rmax = min(r, max_side)
        cmax = min(c, max_side)

        row_start = int(st.slider("Row start", min_value=0, max_value=max(0, r - rmax), value=0, step=1))
        col_start = int(st.slider("Col start", min_value=0, max_value=max(0, c - cmax), value=0, step=1))
        row_end = min(r, row_start + rmax)
        col_end = min(c, col_start + cmax)

        sub = view[row_start:row_end, col_start:col_end].float().numpy()
        st.markdown(f"Showing rows `[{row_start}:{row_end}]` / `{r}`, cols `[{col_start}:{col_end}]` / `{c}`")
        fig_hm = go.Figure(data=go.Heatmap(z=sub, colorscale="RdBu", zmid=0.0))
        fig_hm.update_layout(height=520, margin=dict(l=0, r=0, b=0, t=20), title="2D Slice Heatmap")
        st.plotly_chart(fig_hm, use_container_width=True)

        show_3d = st.checkbox("Show 3D PCA view", value=False)
        if show_3d:
            st.markdown("This projects the selected 2D slice into 3D using PCA (SVD) and plots points.")
            axis = st.selectbox("Point axis", ["Rows as points", "Cols as points"], index=0)
            max_points = int(st.slider("Max points", min_value=16, max_value=4096, value=512, step=16))

            m = view[row_start:row_end, col_start:col_end].float()
            if axis == "Cols as points":
                m = m.t()

            n_pts = int(m.shape[0])
            n_dim = int(m.shape[1])
            if n_pts > max_points:
                idx = torch.linspace(0, n_pts - 1, steps=max_points).long()
                m = m[idx]
                n_pts = int(m.shape[0])

            x = m.detach().cpu().numpy()
            x = x - x.mean(axis=0, keepdims=True)
            u, s, vt = np.linalg.svd(x, full_matrices=False)
            coords = x @ vt[:3].T

            fig3 = go.Figure(data=[go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(size=3, color=np.arange(n_pts), colorscale="Viridis", opacity=0.8),
            )])
            fig3.update_layout(
                height=600,
                margin=dict(l=0, r=0, b=0, t=30),
                title=f"3D PCA of slice ({axis.lower()}, points={n_pts}, dim={n_dim})",
                scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.stop()

    st.info("Unsupported tensor rank for rendering.")
