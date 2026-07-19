from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch


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

MOTIF_NAMES = [
    "inc_chain",
    "dec_chain",
    "xor_toggle",
    "rotate_left",
    "flip0_pulse",
    "noise",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an anchored reinforcement/decay substrate dynamics experiment over byte states.")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--agents", type=int, default=32768)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--decay", type=float, default=0.9995)
    p.add_argument("--edge-lr", type=float, default=0.08)
    p.add_argument("--state-lr", type=float, default=0.02)
    p.add_argument("--pair-lr", type=float, default=0.05)
    p.add_argument("--motif-switch-prob", type=float, default=0.002)
    p.add_argument("--motif-reset-prob", type=float, default=0.01)
    p.add_argument("--anchor-span", type=int, default=16)
    p.add_argument("--expected-horizon", type=int, default=8)
    p.add_argument("--noise-frac", type=float, default=0.25)
    p.add_argument("--report-every", type=int, default=200)
    p.add_argument("--out", type=str, default="substrate_dynamics_report_v3.json")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def sample_structured_motifs(num: int, device: torch.device) -> torch.Tensor:
    return torch.randint(low=0, high=len(MOTIF_NAMES) - 1, size=(num,), device=device, dtype=torch.long)


def motif_anchor_starts(device: torch.device) -> torch.Tensor:
    return torch.tensor([32, 80, 128, 160, 192], device=device, dtype=torch.long)


def sample_states_for_motifs(motifs: torch.Tensor, anchor_span: int) -> torch.Tensor:
    device = motifs.device
    states = torch.randint(low=0, high=256, size=motifs.shape, device=device, dtype=torch.long)
    structured = motifs != len(MOTIF_NAMES) - 1
    if bool(structured.any().item()):
        starts = motif_anchor_starts(device=device)
        base = starts[motifs[structured]]
        offset = torch.randint(low=0, high=max(1, int(anchor_span)), size=(int(structured.sum().item()),), device=device, dtype=torch.long)
        states[structured] = (base + offset) & 0xFF
    return states


def choose_ops(states: torch.Tensor, motifs: torch.Tensor, step: int, device: torch.device) -> torch.Tensor:
    ops = torch.zeros_like(motifs)

    mask = motifs == 0
    ops[mask] = 9

    mask = motifs == 1
    ops[mask] = 10

    mask = motifs == 2
    ops[mask] = 11

    mask = motifs == 3
    ops[mask] = 12

    mask = motifs == 4
    phase = (states & 1) == 0
    ops[mask & phase] = 1
    ops[mask & (~phase)] = 9

    mask = motifs == 5
    if bool(mask.any().item()):
        ops[mask] = torch.randint(low=0, high=len(OP_NAMES), size=(int(mask.sum().item()),), device=device, dtype=torch.long)

    if step % 97 == 0:
        pulse = motifs != 5
        ops[pulse] = torch.where((states[pulse] % 3) == 0, torch.full_like(ops[pulse], 12), ops[pulse])
    return ops


def hamming_distance_matrix() -> torch.Tensor:
    vals = torch.arange(256, dtype=torch.long)
    xorv = vals.unsqueeze(1) ^ vals.unsqueeze(0)
    pop = torch.zeros_like(xorv)
    for bit in range(8):
        pop = pop + ((xorv >> bit) & 1)
    return pop.to(dtype=torch.float32)


def build_expected_mask(device: torch.device, anchor_span: int, expected_horizon: int) -> torch.Tensor:
    starts = motif_anchor_starts(device=device)
    mask = torch.zeros(256, device=device, dtype=torch.bool)
    for motif in range(len(MOTIF_NAMES) - 1):
        seeds = (starts[motif] + torch.arange(max(1, int(anchor_span)), device=device, dtype=torch.long)) & 0xFF
        active = torch.zeros(256, device=device, dtype=torch.bool)
        active.scatter_(0, seeds, True)
        seen = active.clone()
        for step_idx in range(max(1, int(expected_horizon))):
            src = torch.nonzero(active, as_tuple=False).squeeze(-1)
            if src.numel() == 0:
                break
            if motif == 0:
                nxt = (src + 1) & 0xFF
            elif motif == 1:
                nxt = (src - 1) & 0xFF
            elif motif == 2:
                nxt = src ^ 0x20
            elif motif == 3:
                nxt = ((src << 1) | (src >> 7)) & 0xFF
            else:
                nxt = torch.where((src & 1) == 0, src ^ 0x01, (src + 1) & 0xFF)
            active = torch.zeros(256, device=device, dtype=torch.bool)
            active.scatter_(0, nxt, True)
            seen |= active
        mask |= seen
    return mask


def summarize(
    step: int,
    state_energy: torch.Tensor,
    edge_strength: torch.Tensor,
    pair_strength: torch.Tensor,
    motif_hist: torch.Tensor,
    expected_mask: torch.Tensor,
    hamming_lut: torch.Tensor,
) -> Dict[str, float | int | list | dict]:
    state_mass = state_energy.sum().clamp_min(1e-9)
    edge_mass = edge_strength.sum().clamp_min(1e-9)
    pair_mass = pair_strength.sum().clamp_min(1e-9)

    top_state_vals, top_state_idx = torch.topk(state_energy, k=16)
    top_edge_vals, top_edge_idx = torch.topk(edge_strength.reshape(-1), k=16)
    top_pair_vals, top_pair_idx = torch.topk(pair_strength.reshape(-1), k=12)

    energy_p = (state_energy / state_mass).clamp_min(1e-12)
    edge_by_state = edge_strength.sum(dim=-1)
    edge_top = edge_strength.max(dim=-1).values
    active_states = edge_by_state > 1e-6
    dominance = (edge_top[active_states] / edge_by_state[active_states].clamp_min(1e-9)).mean() if bool(active_states.any().item()) else torch.tensor(0.0, device=state_energy.device)

    support = torch.nonzero(state_energy > (state_energy.mean() + state_energy.std()), as_tuple=False).squeeze(-1)
    if support.numel() >= 2:
        sub = hamming_lut[support][:, support]
        locality = float(sub.mean().item())
    else:
        locality = 0.0

    expected_mass = float(state_energy[expected_mask].sum().item() / state_mass.item())

    top_states = [
        {"state": int(idx.item()), "energy": float(val.item())}
        for idx, val in zip(top_state_idx, top_state_vals)
    ]
    top_edges: List[Dict[str, float | int | str]] = []
    for flat_idx, val in zip(top_edge_idx, top_edge_vals):
        src = int((flat_idx // len(OP_NAMES)).item())
        op = int((flat_idx % len(OP_NAMES)).item())
        top_edges.append({"src": src, "op": OP_NAMES[op], "strength": float(val.item())})

    top_pairs: List[Dict[str, float | str]] = []
    for flat_idx, val in zip(top_pair_idx, top_pair_vals):
        prev = int((flat_idx // len(OP_NAMES)).item())
        cur = int((flat_idx % len(OP_NAMES)).item())
        top_pairs.append({"prev_op": OP_NAMES[prev], "cur_op": OP_NAMES[cur], "strength": float(val.item())})

    motif_dist = {
        MOTIF_NAMES[i]: float(motif_hist[i].item() / motif_hist.sum().clamp_min(1.0).item())
        for i in range(len(MOTIF_NAMES))
    }

    return {
        "step": int(step),
        "state_entropy": float((-(energy_p * energy_p.log()).sum()).item()),
        "edge_dominance": float(dominance.item()),
        "expected_state_mass": expected_mass,
        "state_mass": float(state_mass.item()),
        "edge_mass": float(edge_mass.item()),
        "pair_mass": float(pair_mass.item()),
        "active_state_count": int((state_energy > 1e-5).sum().item()),
        "active_edge_count": int((edge_strength > 1e-5).sum().item()),
        "support_locality_mean_hamming": locality,
        "motif_distribution": motif_dist,
        "top_states": top_states,
        "top_edges": top_edges,
        "top_macro_pairs": top_pairs,
    }


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(str(args.device))

    num_agents = int(args.agents)
    noise_count = int(round(num_agents * float(args.noise_frac)))
    structured_count = max(0, num_agents - noise_count)

    motifs = torch.empty(num_agents, device=device, dtype=torch.long)
    if structured_count > 0:
        motifs[:structured_count] = sample_structured_motifs(structured_count, device=device)
    if noise_count > 0:
        motifs[structured_count:] = len(MOTIF_NAMES) - 1
    states = sample_states_for_motifs(motifs, anchor_span=int(args.anchor_span))

    prev_ops = torch.full((num_agents,), -1, device=device, dtype=torch.long)
    state_energy = torch.zeros(256, device=device, dtype=torch.float32)
    edge_strength = torch.zeros(256, len(OP_NAMES), device=device, dtype=torch.float32)
    pair_strength = torch.zeros(len(OP_NAMES), len(OP_NAMES), device=device, dtype=torch.float32)

    expected_states = build_expected_mask(
        device=device,
        anchor_span=int(args.anchor_span),
        expected_horizon=int(args.expected_horizon),
    )
    hamming_lut = hamming_distance_matrix().to(device)
    history: List[Dict[str, float | int | list | dict]] = []

    for step in range(1, int(args.steps) + 1):
        state_energy.mul_(float(args.decay))
        edge_strength.mul_(float(args.decay))
        pair_strength.mul_(float(args.decay))

        switch_mask = torch.rand(num_agents, device=device) < float(args.motif_switch_prob)
        structured_switch = switch_mask & (motifs != len(MOTIF_NAMES) - 1)
        if bool(structured_switch.any().item()):
            motifs[structured_switch] = sample_structured_motifs(int(structured_switch.sum().item()), device=device)
            states[structured_switch] = sample_states_for_motifs(motifs[structured_switch], anchor_span=int(args.anchor_span))

        reset_mask = (torch.rand(num_agents, device=device) < float(args.motif_reset_prob)) & (motifs != len(MOTIF_NAMES) - 1)
        if bool(reset_mask.any().item()):
            states[reset_mask] = sample_states_for_motifs(motifs[reset_mask], anchor_span=int(args.anchor_span))

        ops = choose_ops(states, motifs, step=step, device=device)
        next_states = apply_ops(states).gather(1, ops.unsqueeze(1)).squeeze(1)

        edge_delta = torch.zeros_like(edge_strength)
        edge_delta.index_put_((states, ops), torch.full_like(ops, float(args.edge_lr), dtype=torch.float32), accumulate=True)
        edge_strength.add_(edge_delta)

        state_delta = torch.zeros_like(state_energy)
        state_delta.index_put_((next_states,), torch.full_like(next_states, float(args.state_lr), dtype=torch.float32), accumulate=True)
        state_energy.add_(state_delta)

        valid_pairs = prev_ops >= 0
        if bool(valid_pairs.any().item()):
            pair_delta = torch.zeros_like(pair_strength)
            pair_delta.index_put_(
                (prev_ops[valid_pairs], ops[valid_pairs]),
                torch.full((int(valid_pairs.sum().item()),), float(args.pair_lr), device=device, dtype=torch.float32),
                accumulate=True,
            )
            pair_strength.add_(pair_delta)

        prev_ops = ops
        states = next_states

        if step == 1 or step % int(args.report_every) == 0 or step == int(args.steps):
            motif_hist = torch.bincount(motifs, minlength=len(MOTIF_NAMES)).to(dtype=torch.float32)
            snapshot = summarize(
                step=step,
                state_energy=state_energy.detach().clone(),
                edge_strength=edge_strength.detach().clone(),
                pair_strength=pair_strength.detach().clone(),
                motif_hist=motif_hist,
                expected_mask=expected_states,
                hamming_lut=hamming_lut,
            )
            history.append(snapshot)
            print(json.dumps({
                "step": snapshot["step"],
                "state_entropy": round(float(snapshot["state_entropy"]), 4),
                "edge_dominance": round(float(snapshot["edge_dominance"]), 4),
                "expected_state_mass": round(float(snapshot["expected_state_mass"]), 4),
                "active_state_count": int(snapshot["active_state_count"]),
                "active_edge_count": int(snapshot["active_edge_count"]),
                "support_locality_mean_hamming": round(float(snapshot["support_locality_mean_hamming"]), 4),
            }))

    payload = {
        "config": vars(args),
        "final": history[-1] if history else {},
        "history": history,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved_report={args.out}")


if __name__ == "__main__":
    main()
