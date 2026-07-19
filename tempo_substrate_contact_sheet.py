from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
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

BASE_OP_COUNT = len(OP_NAMES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a contact sheet of tempo substrate process snapshots.")
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--agents", type=int, default=8192)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--decay", type=float, default=0.9992)
    p.add_argument("--state-lr", type=float, default=0.02)
    p.add_argument("--edge-lr", type=float, default=0.06)
    p.add_argument("--pair-lr", type=float, default=0.04)
    p.add_argument("--trigram-lr", type=float, default=0.03)
    p.add_argument("--tempo-lr", type=float, default=0.002)
    p.add_argument("--tempo-gain", type=float, default=1.2)
    p.add_argument("--coupling-gain", type=float, default=0.7)
    p.add_argument("--resonance-gain", type=float, default=0.7)
    p.add_argument("--motif-gain", type=float, default=1.6)
    p.add_argument("--novelty-gain", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--macro-threshold", type=float, default=5000.0)
    p.add_argument("--macro-gain", type=float, default=1.5)
    p.add_argument("--max-macros", type=int, default=8)
    p.add_argument("--motif-switch-prob", type=float, default=0.001)
    p.add_argument("--noise-frac", type=float, default=0.15)
    p.add_argument("--anchor-span", type=int, default=16)
    p.add_argument("--shots", type=int, default=9)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--out", type=str, default="tempo_substrate_contact_sheet.png")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def apply_base_ops(src: torch.Tensor) -> torch.Tensor:
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


def motif_bias_table(device: torch.device) -> torch.Tensor:
    bias = torch.full((len(MOTIF_NAMES), BASE_OP_COUNT), -0.55, device=device, dtype=torch.float32)
    bias[0, 9] = 2.0
    bias[0, 1] = 0.5
    bias[0, 12] = 0.3
    bias[1, 10] = 2.0
    bias[1, 1] = 0.5
    bias[1, 13] = 0.3
    bias[2, 11] = 2.0
    bias[2, 1] = 0.6
    bias[2, 2] = 0.6
    bias[3, 12] = 2.0
    bias[3, 13] = 0.8
    bias[3, 11] = 0.3
    bias[4, 1] = 1.9
    bias[4, 9] = 1.3
    bias[4, 10] = 0.4
    bias[5, :] = 0.0
    return bias


def primitive_periods(device: torch.device) -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 4.0, 6.0, 8.0, 9.0, 12.0], device=device)


def popcount(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for bit in range(8):
        out = out + ((x >> bit) & 1)
    return out


def primitive_resonance(states: torch.Tensor, periods: torch.Tensor, phase_offsets: torch.Tensor) -> torch.Tensor:
    states_f = states.to(dtype=torch.float32).unsqueeze(1)
    periods_f = periods.unsqueeze(0)
    phase_f = phase_offsets.unsqueeze(0)
    mod = torch.remainder(states_f + phase_f, periods_f)
    centered = (mod / periods_f) * (2.0 * math.pi)
    resonance = torch.cos(centered)
    bit_drive = (popcount(states).to(dtype=torch.float32).unsqueeze(1) / 8.0) * 2.0 - 1.0
    return 0.7 * resonance + 0.3 * bit_drive


def tempo_scores(step: int, periods: torch.Tensor, phase_offsets: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
    t = torch.tensor(float(step), device=periods.device, dtype=torch.float32)
    base = torch.sin((2.0 * math.pi * t / periods) + phase_offsets)
    overtone = torch.sin((4.0 * math.pi * t / periods) + (phase_offsets * 0.5))
    return amplitudes * (0.75 * base + 0.25 * overtone)


def apply_macro_sequences(states: torch.Tensor, macro_sequences: List[List[int]]) -> torch.Tensor:
    if not macro_sequences:
        return torch.zeros((states.shape[0], 0), device=states.device, dtype=torch.long)
    outs = []
    for seq in macro_sequences:
        cur = states
        for op in seq:
            cur = apply_base_ops(cur).gather(1, torch.full((cur.shape[0], 1), int(op), device=cur.device, dtype=torch.long)).squeeze(1)
        outs.append(cur)
    return torch.stack(outs, dim=1)


def top_macro_candidates(trigram_strength: torch.Tensor, threshold: float, limit: int) -> List[List[int]]:
    vals, idx = torch.topk(trigram_strength.reshape(-1), k=min(limit * 8, trigram_strength.numel()))
    out: List[List[int]] = []
    seen = set()
    for val, flat_idx in zip(vals, idx):
        if float(val.item()) < float(threshold):
            break
        a = int((flat_idx // (BASE_OP_COUNT * BASE_OP_COUNT)).item())
        rem = int((flat_idx % (BASE_OP_COUNT * BASE_OP_COUNT)).item())
        b = rem // BASE_OP_COUNT
        c = rem % BASE_OP_COUNT
        key = (a, b, c)
        if key in seen:
            continue
        seen.add(key)
        out.append([a, b, c])
        if len(out) >= limit:
            break
    return out


def state_grid(states: torch.Tensor) -> np.ndarray:
    counts = torch.bincount(states, minlength=256).to(dtype=torch.float32)
    arr = counts.reshape(16, 16).cpu().numpy()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def primitive_histogram(ops: torch.Tensor, total_ops: int) -> np.ndarray:
    counts = torch.bincount(ops, minlength=total_ops).to(dtype=torch.float32).cpu().numpy()
    denom = counts.sum() if counts.sum() > 0 else 1.0
    return counts / denom


def motif_histogram(motifs: torch.Tensor) -> np.ndarray:
    counts = torch.bincount(motifs, minlength=len(MOTIF_NAMES)).to(dtype=torch.float32).cpu().numpy()
    denom = counts.sum() if counts.sum() > 0 else 1.0
    return counts / denom


def top_chain_strings(trigram_strength: torch.Tensor, limit: int = 3) -> List[str]:
    vals, idx = torch.topk(trigram_strength.reshape(-1), k=min(limit, trigram_strength.numel()))
    out = []
    for val, flat_idx in zip(vals, idx):
        if float(val.item()) <= 0:
            continue
        a = int((flat_idx // (BASE_OP_COUNT * BASE_OP_COUNT)).item())
        rem = int((flat_idx % (BASE_OP_COUNT * BASE_OP_COUNT)).item())
        b = rem // BASE_OP_COUNT
        c = rem % BASE_OP_COUNT
        out.append(f"{OP_NAMES[a]}->{OP_NAMES[b]}->{OP_NAMES[c]}")
    return out or ["-"]


def snapshot_steps(total_steps: int, shots: int) -> List[int]:
    raw = np.linspace(1, total_steps, num=max(1, shots))
    vals = sorted({int(round(x)) for x in raw})
    vals[0] = 1
    vals[-1] = total_steps
    return vals


def collect_snapshots(args: argparse.Namespace) -> List[dict]:
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

    motif_bias = motif_bias_table(device=device)
    periods = primitive_periods(device=device)
    phase_offsets = torch.rand(BASE_OP_COUNT, device=device, dtype=torch.float32) * (2.0 * math.pi)
    amplitudes = torch.ones(BASE_OP_COUNT, device=device, dtype=torch.float32)
    macro_sequences: List[List[int]] = []

    prev_ops = torch.full((num_agents,), -1, device=device, dtype=torch.long)
    prev_prev_ops = torch.full((num_agents,), -1, device=device, dtype=torch.long)

    state_energy = torch.zeros(256, device=device, dtype=torch.float32)
    edge_strength = torch.zeros(256, BASE_OP_COUNT, device=device, dtype=torch.float32)
    pair_strength = torch.zeros(BASE_OP_COUNT, BASE_OP_COUNT, device=device, dtype=torch.float32)
    trigram_strength = torch.zeros(BASE_OP_COUNT, BASE_OP_COUNT, BASE_OP_COUNT, device=device, dtype=torch.float32)
    primitive_usage = torch.zeros(BASE_OP_COUNT + int(args.max_macros), device=device, dtype=torch.float32)

    targets = set(snapshot_steps(int(args.steps), int(args.shots)))
    snapshots: List[dict] = []

    for step in range(1, int(args.steps) + 1):
        state_energy.mul_(float(args.decay))
        edge_strength.mul_(float(args.decay))
        pair_strength.mul_(float(args.decay))
        trigram_strength.mul_(float(args.decay))
        primitive_usage.mul_(float(args.decay))

        switch_mask = torch.rand(num_agents, device=device) < float(args.motif_switch_prob)
        structured_switch = switch_mask & (motifs != len(MOTIF_NAMES) - 1)
        if bool(structured_switch.any().item()):
            motifs[structured_switch] = sample_structured_motifs(int(structured_switch.sum().item()), device=device)
            states[structured_switch] = sample_states_for_motifs(motifs[structured_switch], anchor_span=int(args.anchor_span))

        macro_sequences = top_macro_candidates(trigram_strength, threshold=float(args.macro_threshold), limit=int(args.max_macros))

        tempo_vec = tempo_scores(step=step, periods=periods, phase_offsets=phase_offsets, amplitudes=amplitudes)
        resonance = primitive_resonance(states=states, periods=periods, phase_offsets=phase_offsets)
        base_scores = float(args.tempo_gain) * tempo_vec.unsqueeze(0) + float(args.resonance_gain) * resonance + float(args.motif_gain) * motif_bias[motifs]

        usage_norm = primitive_usage[:BASE_OP_COUNT] / primitive_usage[:BASE_OP_COUNT].sum().clamp_min(1e-6)
        novelty_penalty = float(args.novelty_gain) * torch.log1p(usage_norm * float(num_agents) + 1.0)
        base_scores = base_scores - novelty_penalty.unsqueeze(0)

        valid_prev = prev_ops >= 0
        if bool(valid_prev.any().item()):
            safe_prev = prev_ops[valid_prev].clamp(max=BASE_OP_COUNT - 1)
            base_scores[valid_prev] = base_scores[valid_prev] + float(args.coupling_gain) * pair_strength[safe_prev]

        macro_scores = None
        if macro_sequences:
            macro_score_cols = []
            for seq in macro_sequences:
                seq_score = base_scores[:, seq].mean(dim=1)
                if bool(valid_prev.any().item()):
                    prev_match = (prev_ops == seq[0]).to(dtype=torch.float32)
                    seq_score = seq_score + float(args.macro_gain) * prev_match
                macro_score_cols.append(seq_score)
            macro_scores = torch.stack(macro_score_cols, dim=1)

        full_scores = base_scores if macro_scores is None else torch.cat([base_scores, macro_scores], dim=1)

        noise_mask = motifs == len(MOTIF_NAMES) - 1
        if bool(noise_mask.any().item()):
            full_scores[noise_mask] = full_scores[noise_mask] + 0.35 * torch.randn_like(full_scores[noise_mask])

        probs = torch.softmax(full_scores / max(1e-4, float(args.temperature)), dim=1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)

        base_targets = apply_base_ops(states)
        next_states = torch.empty_like(states)
        base_mask = sampled < BASE_OP_COUNT
        if bool(base_mask.any().item()):
            next_states[base_mask] = base_targets[base_mask].gather(1, sampled[base_mask].unsqueeze(1)).squeeze(1)
        if bool((~base_mask).any().item()):
            macro_idx = sampled[~base_mask] - BASE_OP_COUNT
            macro_states = apply_macro_sequences(states[~base_mask], macro_sequences)
            next_states[~base_mask] = macro_states.gather(1, macro_idx.unsqueeze(1)).squeeze(1)

        if bool(base_mask.any().item()):
            edge_strength.index_put_(
                (states[base_mask], sampled[base_mask]),
                torch.full((int(base_mask.sum().item()),), float(args.edge_lr), device=device, dtype=torch.float32),
                accumulate=True,
            )
        state_energy.index_put_(
            (next_states,),
            torch.full((num_agents,), float(args.state_lr), device=device, dtype=torch.float32),
            accumulate=True,
        )
        primitive_usage.index_put_((sampled,), torch.ones(num_agents, device=device, dtype=torch.float32), accumulate=True)

        valid_pair = (prev_ops >= 0) & (sampled < BASE_OP_COUNT)
        if bool(valid_pair.any().item()):
            pair_strength.index_put_(
                (prev_ops[valid_pair].clamp(max=BASE_OP_COUNT - 1), sampled[valid_pair]),
                torch.full((int(valid_pair.sum().item()),), float(args.pair_lr), device=device, dtype=torch.float32),
                accumulate=True,
            )

        valid_tri = (prev_prev_ops >= 0) & (prev_ops >= 0) & (sampled < BASE_OP_COUNT)
        if bool(valid_tri.any().item()):
            trigram_strength.index_put_(
                (prev_prev_ops[valid_tri].clamp(max=BASE_OP_COUNT - 1), prev_ops[valid_tri].clamp(max=BASE_OP_COUNT - 1), sampled[valid_tri]),
                torch.full((int(valid_tri.sum().item()),), float(args.trigram_lr), device=device, dtype=torch.float32),
                accumulate=True,
            )

        centered_usage = primitive_usage[:BASE_OP_COUNT] / primitive_usage[:BASE_OP_COUNT].sum().clamp_min(1e-6)
        amplitudes = torch.clamp(0.97 * amplitudes + float(args.tempo_lr) * (1.0 - centered_usage * BASE_OP_COUNT), min=0.35, max=3.0)
        phase_offsets = torch.remainder(phase_offsets + float(args.tempo_lr) * (amplitudes - amplitudes.mean()), 2.0 * math.pi)

        prev_prev_ops = torch.where(sampled < BASE_OP_COUNT, prev_ops, prev_prev_ops)
        prev_ops = torch.where(sampled < BASE_OP_COUNT, sampled, prev_ops)
        states = next_states

        if step in targets:
            snapshots.append(
                {
                    "step": step,
                    "state_grid": state_grid(states),
                    "primitive_hist": primitive_histogram(sampled, BASE_OP_COUNT + len(macro_sequences)),
                    "motif_hist": motif_histogram(motifs),
                    "top_chains": top_chain_strings(trigram_strength, limit=3),
                    "macro_sequences": ["->".join(OP_NAMES[s] for s in seq) for seq in macro_sequences[:3]] or ["-"],
                }
            )
            print(f"captured_snapshot_step={step}")

    return snapshots


def render_contact_sheet(snapshots: List[dict], cols: int, out_path: str) -> None:
    cols = max(1, int(cols))
    rows = math.ceil(len(snapshots) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), constrained_layout=True)
    axes_arr = np.array(axes, dtype=object).reshape(rows, cols)

    for ax in axes_arr.flat:
        ax.axis("off")

    for ax, snap in zip(axes_arr.flat, snapshots):
        grid = snap["state_grid"]
        ax.imshow(grid, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"step {snap['step']}")
        ax.set_xticks([])
        ax.set_yticks([])
        motif_text = " ".join(f"{MOTIF_NAMES[i]}:{snap['motif_hist'][i]:.2f}" for i in range(len(MOTIF_NAMES)))
        chain_text = " | ".join(snap["top_chains"])
        macro_text = " | ".join(snap["macro_sequences"])
        ax.text(
            0.02,
            -0.12,
            f"chains: {chain_text}\nmacros: {macro_text}\n{motif_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            family="monospace",
        )

        inset = ax.inset_axes([1.02, 0.05, 0.28, 0.5])
        hist = snap["primitive_hist"]
        x = np.arange(len(hist))
        colors = ["#7aa2f7"] * BASE_OP_COUNT + ["#9ece6a"] * max(0, len(hist) - BASE_OP_COUNT)
        inset.bar(x, hist, color=colors)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("firing", fontsize=8)
        inset.set_ylim(0.0, max(0.12, float(np.max(hist)) * 1.15))

    fig.suptitle("Tempo substrate contact sheet", fontsize=18)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    snaps = collect_snapshots(args)
    render_contact_sheet(snaps, cols=int(args.cols), out_path=args.out)
    print(f"saved_contact_sheet={args.out}")


if __name__ == "__main__":
    main()
