import argparse
from dataclasses import dataclass
from typing import Optional

import torch

from raplib.data import decode_bytes, ids_to_bytes, strip_pos_tag_bytes
from raplib.model import ModelConfig, PrimitiveManifoldLM


@dataclass
class RunCfg:
    ckpt: str
    prompt: str
    capture_step: int
    max_new: int
    temperature: float
    top_k: int
    force_primitive: int
    force_strength: float
    forbid_primitive: int
    pulse_primitive: int
    pulse_strength: float
    pulse_steps: int
    primitive_order: str
    no_primitives: bool
    mix_primitives: str
    mix_weights: str
    mix_steps: int
    device: str


def parse_args() -> RunCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints1/b.pt")
    p.add_argument("--prompt", type=str, default="What is gravity?")
    p.add_argument("--capture_step", type=int, default=8)
    p.add_argument("--max_new", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--force_primitive", type=int, default=-1)
    p.add_argument("--force_strength", type=float, default=1.0)
    p.add_argument("--forbid_primitive", type=int, default=-1)
    p.add_argument("--pulse_primitive", type=int, default=-1)
    p.add_argument("--pulse_strength", type=float, default=1.0)
    p.add_argument("--pulse_steps", type=int, default=0)
    p.add_argument("--primitive_order", type=str, default="")
    p.add_argument("--no_primitives", action="store_true")
    p.add_argument("--mix_primitives", type=str, default="")
    p.add_argument("--mix_weights", type=str, default="")
    p.add_argument("--mix_steps", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    return RunCfg(
        ckpt=a.ckpt,
        prompt=a.prompt,
        capture_step=a.capture_step,
        max_new=a.max_new,
        temperature=a.temperature,
        top_k=a.top_k,
        force_primitive=a.force_primitive,
        force_strength=a.force_strength,
        forbid_primitive=a.forbid_primitive,
        pulse_primitive=a.pulse_primitive,
        pulse_strength=a.pulse_strength,
        pulse_steps=a.pulse_steps,
        primitive_order=a.primitive_order,
        no_primitives=bool(a.no_primitives),
        mix_primitives=a.mix_primitives,
        mix_weights=a.mix_weights,
        mix_steps=a.mix_steps,
        device=a.device,
    )


def parse_primitive_order(spec: str, n_primitives: int) -> Optional[list[int]]:
    spec = str(spec).strip()
    if not spec:
        return None
    parts = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) != int(n_primitives):
        raise ValueError(f"primitive_order must list exactly {int(n_primitives)} entries")
    want = list(range(int(n_primitives)))
    if sorted(parts) != want:
        raise ValueError(f"primitive_order must be a permutation of {want}")
    return parts


def parse_mix(spec_prims: str, spec_weights: str, n_primitives: int) -> Optional[tuple[list[int], list[float]]]:
    prim_spec = str(spec_prims).strip()
    weight_spec = str(spec_weights).strip()
    if not prim_spec:
        return None
    prims = [int(x.strip()) for x in prim_spec.split(",") if x.strip()]
    if any(p < 0 or p >= int(n_primitives) for p in prims):
        raise ValueError(f"mix_primitives entries must be in [0, {int(n_primitives) - 1}]")
    if weight_spec:
        weights = [float(x.strip()) for x in weight_spec.split(",") if x.strip()]
    else:
        weights = [1.0 for _ in prims]
    if len(weights) != len(prims):
        raise ValueError("mix_weights must have the same number of entries as mix_primitives")
    s = sum(max(float(w), 0.0) for w in weights)
    if s <= 0.0:
        raise ValueError("mix_weights must contain at least one positive value")
    weights = [max(float(w), 0.0) / s for w in weights]
    return prims, weights



def sample_next_id(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(float(temperature), 1e-6)
    if int(top_k) > 0 and int(top_k) < logits.shape[-1]:
        vals, idx = torch.topk(logits, k=int(top_k), dim=-1)
        probs = torch.softmax(vals, dim=-1)
        pick = torch.multinomial(probs, num_samples=1)
        return int(idx.gather(-1, pick).item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())



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



def decode_ids(ids: list[int]) -> str:
    raw = ids_to_bytes(ids)
    raw = strip_pos_tag_bytes(raw)
    return decode_bytes(raw)


def build_retrieval(
    model: PrimitiveManifoldLM,
    tok: int,
    state: torch.Tensor,
    last_pos: Optional[torch.Tensor],
    state_hist: Optional[list[torch.Tensor]],
) -> Optional[torch.Tensor]:
    if not bool(getattr(model, "retrieval_enabled", False)):
        return None
    retrieval = torch.zeros_like(state)
    cnt = torch.zeros((state.shape[0], 1), device=state.device, dtype=state.dtype)
    if last_pos is None or state_hist is None or len(state_hist) == 0:
        return retrieval

    x = torch.tensor([int(tok)], device=state.device, dtype=torch.long)
    is_reserved = (x == model.reserved_bytes[0]) | (x == model.reserved_bytes[1]) | (x == model.reserved_bytes[2])
    batch_ix = torch.arange(state.shape[0], device=state.device)
    hist_tensor = torch.stack(state_hist, dim=1)

    if bool(getattr(model, "path_retrieval_enabled", False)) and bool((~is_reserved).any().item()):
        neigh = model.path_neigh_lut[x]
        dist = model.path_dist_lut[x].to(dtype=state.dtype)
        neigh_clamped = neigh.clamp(min=0)
        pos = last_pos.gather(1, neigh_clamped)
        valid = (neigh >= 0) & (pos >= 0) & (~is_reserved).unsqueeze(1)
        if bool(valid.any().item()):
            pos_clamped = pos.clamp(min=0)
            got = torch.gather(
                hist_tensor,
                dim=1,
                index=pos_clamped.unsqueeze(-1).expand(-1, -1, hist_tensor.shape[-1]),
            )
            alpha = torch.tensor(float(getattr(model, "path_alpha", 1.0)), device=state.device, dtype=state.dtype)
            w = torch.exp(-alpha * dist) * valid.to(dtype=state.dtype)
            num = (got * w.unsqueeze(-1)).sum(dim=1)
            den = torch.clamp(w.sum(dim=1, keepdim=True), min=1.0)
            retrieval = retrieval + (num / den)
            cnt = cnt + 1.0

    if bool(getattr(model, "word_map_retrieval_enabled", False)) and bool((~is_reserved).any().item()):
        neigh = model.word_neigh_lut[x]
        ww = model.word_w_lut[x].to(dtype=state.dtype)
        neigh_clamped = neigh.clamp(min=0)
        pos = last_pos.gather(1, neigh_clamped)
        valid = (neigh >= 0) & (pos >= 0) & (~is_reserved).unsqueeze(1)
        if bool(valid.any().item()):
            pos_clamped = pos.clamp(min=0)
            got = torch.gather(
                hist_tensor,
                dim=1,
                index=pos_clamped.unsqueeze(-1).expand(-1, -1, hist_tensor.shape[-1]),
            )
            w = ww * valid.to(dtype=state.dtype)
            num = (got * w.unsqueeze(-1)).sum(dim=1)
            den = torch.clamp(w.sum(dim=1, keepdim=True), min=1e-6)
            retrieval = retrieval + (num / den)
            cnt = cnt + 1.0

    if bool(getattr(model, "symmetry_retrieval_enabled", False)) and bool((~is_reserved).any().item()):
        inv = x ^ 0xFF
        rev = model.bit_reverse_lut[x]
        pos_inv = last_pos.gather(1, inv.unsqueeze(1)).squeeze(1)
        pos_rev = last_pos.gather(1, rev.unsqueeze(1)).squeeze(1)
        for pos1 in (pos_inv, pos_rev):
            m = (pos1 >= 0) & (~is_reserved)
            if bool(m.any().item()):
                p = pos1.clamp(min=0)
                got = hist_tensor[batch_ix, p]
                retrieval = retrieval + got * m.to(dtype=state.dtype).unsqueeze(1)
                cnt = cnt + m.to(dtype=state.dtype).unsqueeze(1)

    retrieval = retrieval / torch.clamp(cnt, min=1.0)
    return retrieval


def step_once(
    model: PrimitiveManifoldLM,
    x_id: int,
    state: torch.Tensor,
    *,
    last_pos: Optional[torch.Tensor] = None,
    state_hist: Optional[list[torch.Tensor]] = None,
    force_primitive: Optional[int] = None,
    force_strength: float = 1.0,
    forbid_primitive: Optional[int] = None,
    primitive_order: Optional[list[int]] = None,
    no_primitives: bool = False,
    mix_override: Optional[tuple[list[int], list[float]]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor([int(x_id)], device=state.device, dtype=torch.long)
    x_emb = model._get_emb(x)
    retrieval = build_retrieval(model, int(x_id), state, last_pos, state_hist)
    if retrieval is not None:
        ctrl_in = torch.cat([x_emb, state, retrieval], dim=-1)
    else:
        ctrl_in = torch.cat([x_emb, state], dim=-1)
    w = torch.softmax(model.controller(ctrl_in), dim=-1)

    if forbid_primitive is not None and 0 <= int(forbid_primitive) < model.cfg.n_primitives:
        w[:, int(forbid_primitive)] = 0.0
        sums = w.sum(dim=-1, keepdim=True)
        fallback = torch.full_like(w, 1.0 / float(w.shape[-1]))
        w = torch.where(sums > 1e-8, w / torch.clamp(sums, min=1e-8), fallback)

    if force_primitive is not None and 0 <= int(force_primitive) < model.cfg.n_primitives:
        one_hot = torch.zeros_like(w)
        one_hot[:, int(force_primitive)] = 1.0
        alpha = max(0.0, min(float(force_strength), 1.0))
        w = (1.0 - alpha) * w + alpha * one_hot
        w = w / torch.clamp(w.sum(dim=-1, keepdim=True), min=1e-8)

    if mix_override is not None:
        prims, weights = mix_override
        mix_w = torch.zeros_like(w)
        for prim, weight in zip(prims, weights):
            mix_w[:, int(prim)] = float(weight)
        w = mix_w / torch.clamp(mix_w.sum(dim=-1, keepdim=True), min=1e-8)

    sA = torch.einsum("b d, k d e -> b k e", state, model.prim_A)
    xB = torch.einsum("b m, k d m -> b k d", x_emb, model.prim_B)
    cand = torch.tanh(sA + xB + model.prim_bias.unsqueeze(0))
    if primitive_order is not None:
        order = torch.tensor(primitive_order, device=cand.device, dtype=torch.long)
        cand = cand.index_select(1, order)
    if bool(no_primitives):
        mixed = torch.zeros_like(state)
    else:
        mixed = torch.einsum("b k, b k d -> b d", w, cand)
    gate = model.slice_gate(ctrl_in)
    new_state = gate * mixed + (1.0 - gate) * state
    logits = model.out(torch.cat([x_emb, new_state], dim=-1))
    return logits, new_state, w


def capture_state(model: PrimitiveManifoldLM, ids: list[int], capture_step: int, device: str) -> tuple[torch.Tensor, int, list[torch.Tensor]]:
    if not ids:
        state = model.init_state(batch_size=1, device=torch.device(device))
        return state, 10, [], None, []

    state = model.init_state(batch_size=1, device=torch.device(device))
    weights = []
    last_id = int(ids[0])
    cap = max(0, min(int(capture_step), len(ids) - 1))
    last_pos = None
    state_hist = None
    if bool(getattr(model, "retrieval_enabled", False)):
        last_pos = torch.full((1, model.cfg.vocab_size), -1, device=state.device, dtype=torch.long)
        state_hist = []
    with torch.no_grad():
        for i, tok in enumerate(ids):
            logits, state, w = step_once(model, int(tok), state, last_pos=last_pos, state_hist=state_hist)
            _ = logits
            weights.append(w.detach().cpu())
            last_id = int(tok)
            if last_pos is not None and state_hist is not None:
                state_hist.append(state.detach().clone())
                x = torch.tensor([[int(tok)]], device=state.device, dtype=torch.long)
                last_pos.scatter_(1, x, torch.full((1, 1), i, device=state.device, dtype=torch.long))
            if i >= cap:
                break
    if state_hist is None:
        state_hist = []
    return state, last_id, weights, last_pos, state_hist


def rollout_from_state(
    model: PrimitiveManifoldLM,
    state: torch.Tensor,
    last_id: int,
    *,
    last_pos: Optional[torch.Tensor] = None,
    state_hist: Optional[list[torch.Tensor]] = None,
    max_new: int,
    temperature: float,
    top_k: int,
    force_primitive: Optional[int] = None,
    force_strength: float = 1.0,
    forbid_primitive: Optional[int] = None,
    pulse_primitive: Optional[int] = None,
    pulse_strength: float = 1.0,
    pulse_steps: int = 0,
    primitive_order: Optional[list[int]] = None,
    no_primitives: bool = False,
    mix_override: Optional[tuple[list[int], list[float]]] = None,
    mix_steps: int = 0,
) -> tuple[list[int], list[torch.Tensor]]:
    out_ids: list[int] = []
    weights: list[torch.Tensor] = []
    with torch.no_grad():
        for step_idx in range(int(max_new)):
            active_force_primitive = force_primitive
            active_force_strength = force_strength
            active_mix_override = None
            if int(pulse_steps) > 0 and pulse_primitive is not None and 0 <= int(pulse_primitive) < model.cfg.n_primitives and step_idx < int(pulse_steps):
                active_force_primitive = int(pulse_primitive)
                active_force_strength = float(pulse_strength)
            if mix_override is not None and int(mix_steps) > 0 and step_idx < int(mix_steps):
                active_mix_override = mix_override
            logits, state, w = step_once(
                model,
                last_id,
                state,
                last_pos=last_pos,
                state_hist=state_hist,
                force_primitive=active_force_primitive,
                force_strength=float(active_force_strength),
                forbid_primitive=forbid_primitive,
                primitive_order=primitive_order,
                no_primitives=bool(no_primitives),
                mix_override=active_mix_override,
            )
            next_id = sample_next_id(logits, temperature=float(temperature), top_k=int(top_k))
            out_ids.append(int(next_id))
            weights.append(w.detach().cpu())
            if last_pos is not None and state_hist is not None:
                state_hist.append(state.detach().clone())
                pos_ix = len(state_hist) - 1
                x = torch.tensor([[int(last_id)]], device=state.device, dtype=torch.long)
                last_pos.scatter_(1, x, torch.full((1, 1), pos_ix, device=state.device, dtype=torch.long))
            last_id = int(next_id)
            if next_id == 10 and len(out_ids) >= 8:
                break
    return out_ids, weights



def summarize_weights(weights: list[torch.Tensor]) -> str:
    if not weights:
        return "(no weights)"
    mat = torch.cat(weights, dim=0)
    mean_w = mat.mean(dim=0)
    argmax = torch.argmax(mat, dim=-1)
    counts = torch.bincount(argmax, minlength=mat.shape[1])
    parts = []
    for i in range(mat.shape[1]):
        parts.append(f"P{i}: mean={float(mean_w[i]):.3f} argmax_count={int(counts[i])}")
    return " | ".join(parts)



def main() -> None:
    args = parse_args()
    model = load_model(args.ckpt, args.device)
    primitive_order = parse_primitive_order(args.primitive_order, model.cfg.n_primitives)
    mix_override = parse_mix(args.mix_primitives, args.mix_weights, model.cfg.n_primitives)

    prompt_text = args.prompt
    prompt_ids = list(prompt_text.encode("utf-8", errors="replace"))
    cap_state, cap_last_id, cap_weights, cap_last_pos, cap_state_hist = capture_state(model, prompt_ids, args.capture_step, args.device)

    normal_ids, normal_weights = rollout_from_state(
        model,
        cap_state.clone(),
        cap_last_id,
        last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
        state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
        max_new=int(args.max_new),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        force_primitive=None,
        force_strength=0.0,
        forbid_primitive=None,
        pulse_primitive=None,
        pulse_strength=0.0,
        pulse_steps=0,
        primitive_order=primitive_order,
        no_primitives=bool(args.no_primitives),
        mix_override=mix_override,
        mix_steps=int(args.mix_steps),
    )

    print(f"Checkpoint: {args.ckpt}")
    print(f"Prompt: {prompt_text}")
    print(f"Capture step: {min(max(int(args.capture_step), 0), max(len(prompt_ids) - 1, 0))}")
    print(f"Captured last input byte: {cap_last_id} ({repr(bytes([cap_last_id]).decode('utf-8', errors='replace'))})")
    print(f"Primitive order: {primitive_order if primitive_order is not None else 'identity'}")
    print(f"No primitives: {bool(args.no_primitives)}")
    print(f"Mixed launch: {mix_override if mix_override is not None else 'none'} for {int(args.mix_steps)} steps")
    print()
    print("[capture phase primitive summary]")
    print(summarize_weights(cap_weights))
    print()
    print("[normal rollout]")
    print(decode_ids(normal_ids))
    print("[normal rollout primitive summary]")
    print(summarize_weights(normal_weights))

    if 0 <= int(args.forbid_primitive) < model.cfg.n_primitives:
        forbidden_ids, forbidden_weights = rollout_from_state(
            model,
            cap_state.clone(),
            cap_last_id,
            last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
            state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
            max_new=int(args.max_new),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            force_primitive=None,
            force_strength=0.0,
            forbid_primitive=int(args.forbid_primitive),
            pulse_primitive=None,
            pulse_strength=0.0,
            pulse_steps=0,
            primitive_order=primitive_order,
            no_primitives=bool(args.no_primitives),
            mix_override=mix_override,
            mix_steps=int(args.mix_steps),
        )
        print()
        print(f"[forbidden rollout: primitive={int(args.forbid_primitive)}]")
        print(decode_ids(forbidden_ids))
        print("[forbidden rollout primitive summary]")
        print(summarize_weights(forbidden_weights))

    if 0 <= int(args.force_primitive) < model.cfg.n_primitives:
        forced_ids, forced_weights = rollout_from_state(
            model,
            cap_state.clone(),
            cap_last_id,
            last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
            state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
            max_new=int(args.max_new),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            force_primitive=int(args.force_primitive),
            force_strength=float(args.force_strength),
            forbid_primitive=None,
            pulse_primitive=None,
            pulse_strength=0.0,
            pulse_steps=0,
            primitive_order=primitive_order,
            no_primitives=bool(args.no_primitives),
            mix_override=mix_override,
            mix_steps=int(args.mix_steps),
        )
        print()
        print(f"[forced rollout: primitive={int(args.force_primitive)} strength={float(args.force_strength):.2f}]")
        print(decode_ids(forced_ids))
        print("[forced rollout primitive summary]")
        print(summarize_weights(forced_weights))

    if 0 <= int(args.pulse_primitive) < model.cfg.n_primitives and int(args.pulse_steps) > 0:
        pulsed_ids, pulsed_weights = rollout_from_state(
            model,
            cap_state.clone(),
            cap_last_id,
            last_pos=None if cap_last_pos is None else cap_last_pos.clone(),
            state_hist=None if cap_state_hist is None else [s.clone() for s in cap_state_hist],
            max_new=int(args.max_new),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            force_primitive=None,
            force_strength=0.0,
            forbid_primitive=None,
            pulse_primitive=int(args.pulse_primitive),
            pulse_strength=float(args.pulse_strength),
            pulse_steps=int(args.pulse_steps),
            primitive_order=primitive_order,
            no_primitives=bool(args.no_primitives),
            mix_override=mix_override,
            mix_steps=int(args.mix_steps),
        )
        print()
        print(f"[pulsed rollout: primitive={int(args.pulse_primitive)} strength={float(args.pulse_strength):.2f} steps={int(args.pulse_steps)}]")
        print(decode_ids(pulsed_ids))
        print("[pulsed rollout primitive summary]")
        print(summarize_weights(pulsed_weights))


if __name__ == "__main__":
    main()
