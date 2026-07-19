from __future__ import annotations

import argparse
import io
import math
import random
from dataclasses import dataclass
from typing import List

import imageio.v2 as imageio
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


@dataclass
class Regime:
    name: str
    tempo_mode: str
    phase_mode: str
    amplitude_mode: str
    temperature: float
    novelty_gain: float
    color: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Animate multiple tempo populations interacting in one shared byte-space.")
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--agents", type=int, default=16384)
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
    p.add_argument("--macro-threshold", type=float, default=5000.0)
    p.add_argument("--macro-gain", type=float, default=1.5)
    p.add_argument("--max-macros", type=int, default=8)
    p.add_argument("--motif-switch-prob", type=float, default=0.001)
    p.add_argument("--noise-frac", type=float, default=0.15)
    p.add_argument("--anchor-span", type=int, default=16)
    p.add_argument("--capture-every", type=int, default=60)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", type=str, default="tempo_shared_space.gif")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def regimes() -> List[Regime]:
    return [
        Regime("weird-spiky", "weird", "staggered", "spiky", 0.88, 1.00, "#f7768e"),
        Regime("clustered-mirror", "clustered", "mirrored", "alternating", 0.82, 1.15, "#7aa2f7"),
        Regime("detuned-chaos", "detuned", "random", "spiky", 0.95, 1.25, "#9ece6a"),
        Regime("default-flat", "default", "random", "flat", 0.90, 1.00, "#e0af68"),
    ]


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


def build_periods(device: torch.device, mode: str) -> torch.Tensor:
    if mode == "weird":
        vals = [1.0, 2.0, 2.5, 3.5, 5.5, 6.5, 10.5, 14.5, 15.5, 21.5, 22.5, 26.5, 34.5, 35.5]
    elif mode == "clustered":
        vals = [2.0, 2.125, 2.25, 2.5, 4.0, 4.125, 4.25, 4.5, 8.0, 8.25, 8.5, 9.0, 16.0, 16.5]
    elif mode == "detuned":
        vals = [1.618, 2.414, 2.618, 3.141, 4.236, 5.083, 6.283, 7.389, 8.314, 9.472, 10.873, 12.112, 13.579, 15.173]
    else:
        vals = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 4.0, 6.0, 8.0, 9.0, 12.0]
    return torch.tensor(vals, device=device, dtype=torch.float32)


def init_phase_offsets(device: torch.device, mode: str) -> torch.Tensor:
    if mode == "staggered":
        return torch.linspace(0.0, 2.0 * math.pi * (1.0 - 1.0 / BASE_OP_COUNT), steps=BASE_OP_COUNT, device=device, dtype=torch.float32)
    if mode == "mirrored":
        base = torch.linspace(0.0, math.pi, steps=(BASE_OP_COUNT // 2) + 1, device=device, dtype=torch.float32)
        return torch.cat([base[:-1], torch.flip(base[:-1], dims=[0])], dim=0)[:BASE_OP_COUNT]
    return torch.rand(BASE_OP_COUNT, device=device, dtype=torch.float32) * (2.0 * math.pi)


def init_amplitudes(device: torch.device, mode: str) -> torch.Tensor:
    if mode == "alternating":
        vals = [1.8 if i % 2 == 0 else 0.55 for i in range(BASE_OP_COUNT)]
    elif mode == "spiky":
        vals = [2.4, 0.6, 1.7, 0.5, 2.1, 0.7, 1.3, 0.45, 2.3, 0.8, 1.9, 0.65, 1.4, 0.5]
    else:
        vals = [1.0 for _ in range(BASE_OP_COUNT)]
    return torch.tensor(vals, device=device, dtype=torch.float32)


def popcount(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    for bit in range(8):
        out = out + ((x >> bit) & 1)
    return out


def primitive_resonance(states: torch.Tensor, periods: torch.Tensor, phase_offsets: torch.Tensor) -> torch.Tensor:
    states_f = states.to(dtype=torch.float32).unsqueeze(1)
    if periods.ndim == 1:
        periods_f = periods.unsqueeze(0)
        phase_f = phase_offsets.unsqueeze(0)
    else:
        periods_f = periods
        phase_f = phase_offsets
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
    outs: List[torch.Tensor] = []
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


def occupancy_grid(states: torch.Tensor) -> np.ndarray:
    counts = torch.bincount(states, minlength=256).to(dtype=torch.float32)
    arr = counts.reshape(16, 16).cpu().numpy()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def group_overlay(states: torch.Tensor, group_ids: torch.Tensor, num_groups: int) -> np.ndarray:
    layers = []
    for gid in range(num_groups):
        mask = group_ids == gid
        counts = torch.bincount(states[mask], minlength=256).to(dtype=torch.float32).reshape(16, 16).cpu().numpy()
        if counts.max() > 0:
            counts = counts / counts.max()
        layers.append(counts)
    rgb = np.zeros((16, 16, 3), dtype=np.float32)
    palette = np.array([
        [247, 118, 142],
        [122, 162, 247],
        [158, 206, 106],
        [224, 175, 104],
    ], dtype=np.float32) / 255.0
    for gid, layer in enumerate(layers):
        rgb = np.maximum(rgb, layer[..., None] * palette[gid % len(palette)])
    return np.clip(rgb, 0.0, 1.0)


def op_counts_for_group(ops: torch.Tensor, mask: torch.Tensor, total_ops: int) -> np.ndarray:
    counts = torch.bincount(ops[mask], minlength=total_ops).to(dtype=torch.float32).cpu().numpy()
    denom = counts.sum() if counts.sum() > 0 else 1.0
    return counts / denom


def top_chain_strings(trigram_strength: torch.Tensor, limit: int = 4) -> List[str]:
    vals, idx = torch.topk(trigram_strength.reshape(-1), k=min(limit, trigram_strength.numel()))
    out = []
    for val, flat_idx in zip(vals, idx):
        if float(val.item()) <= 0:
            continue
        a = int((flat_idx // (BASE_OP_COUNT * BASE_OP_COUNT)).item())
        rem = int((flat_idx % (BASE_OP_COUNT * BASE_OP_COUNT)).item())
        b = rem // BASE_OP_COUNT
        c = rem % BASE_OP_COUNT
        out.append(f"{OP_NAMES[a]}->{OP_NAMES[b]}->{OP_NAMES[c]} | {float(val.item()):.1f}")
    return out or ["-"]


def render_frame(step: int, occ: np.ndarray, overlay: np.ndarray, delta: np.ndarray, per_group_hists: List[np.ndarray], chain_lines: List[str], regimes_list: List[Regime]) -> np.ndarray:
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_occ = fig.add_subplot(gs[:, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_delta = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[:, 2])

    ax_occ.imshow(occ, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_occ.set_title("Shared occupancy")
    ax_occ.set_xticks(range(0, 16, 2))
    ax_occ.set_yticks(range(0, 16, 2))

    ax_overlay.imshow(overlay, interpolation="nearest")
    ax_overlay.set_title("Per-group footprint overlay")
    ax_overlay.set_xticks([])
    ax_overlay.set_yticks([])

    im = ax_delta.imshow(delta, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest")
    ax_delta.set_title("Shared flow / change")
    ax_delta.set_xticks([])
    ax_delta.set_yticks([])
    fig.colorbar(im, ax=ax_delta, fraction=0.046, pad=0.04)

    ax_text.axis("off")
    lines = [f"step: {step}", "", "groups:"]
    for regime, hist in zip(regimes_list, per_group_hists):
        top_idx = np.argsort(hist)[::-1][:3]
        top_names = ", ".join(OP_NAMES[i] if i < BASE_OP_COUNT else f"M{i-BASE_OP_COUNT}" for i in top_idx)
        lines.append(f"{regime.name}: {top_names}")
    lines.extend(["", "shared top chains:", *chain_lines[:6]])
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")

    bar_base_y = 0.02
    height = 0.17
    for idx, (regime, hist) in enumerate(zip(regimes_list, per_group_hists)):
        inset = ax_text.inset_axes([0.02, bar_base_y + idx * 0.19, 0.95, height])
        x = np.arange(len(hist))
        colors = [regime.color] * len(hist)
        inset.bar(x, hist, color=colors)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(regime.name, fontsize=8, loc="left")
        inset.set_ylim(0.0, max(0.12, float(np.max(hist)) * 1.15))

    fig.suptitle("Shared-space tempo interaction", fontsize=18)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return imageio.imread(buf)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(str(args.device))
    regs = regimes()
    num_groups = len(regs)

    num_agents = int(args.agents)
    group_ids = torch.arange(num_agents, device=device, dtype=torch.long) % num_groups

    motifs = torch.empty(num_agents, device=device, dtype=torch.long)
    noise_mask = torch.rand(num_agents, device=device) < float(args.noise_frac)
    motifs[noise_mask] = len(MOTIF_NAMES) - 1
    if bool((~noise_mask).any().item()):
        motifs[~noise_mask] = sample_structured_motifs(int((~noise_mask).sum().item()), device=device)
    states = sample_states_for_motifs(motifs, anchor_span=int(args.anchor_span))

    motif_bias = motif_bias_table(device=device)
    periods_bank = torch.stack([build_periods(device, r.tempo_mode) for r in regs], dim=0)
    phase_bank = torch.stack([init_phase_offsets(device, r.phase_mode) for r in regs], dim=0)
    amp_bank = torch.stack([init_amplitudes(device, r.amplitude_mode) for r in regs], dim=0)
    temp_by_group = torch.tensor([r.temperature for r in regs], device=device, dtype=torch.float32)
    novelty_by_group = torch.tensor([r.novelty_gain for r in regs], device=device, dtype=torch.float32)

    prev_ops = torch.full((num_agents,), -1, device=device, dtype=torch.long)
    prev_prev_ops = torch.full((num_agents,), -1, device=device, dtype=torch.long)

    state_energy = torch.zeros(256, device=device, dtype=torch.float32)
    edge_strength = torch.zeros(256, BASE_OP_COUNT, device=device, dtype=torch.float32)
    pair_strength = torch.zeros(BASE_OP_COUNT, BASE_OP_COUNT, device=device, dtype=torch.float32)
    trigram_strength = torch.zeros(BASE_OP_COUNT, BASE_OP_COUNT, BASE_OP_COUNT, device=device, dtype=torch.float32)
    primitive_usage = torch.zeros(BASE_OP_COUNT + int(args.max_macros), device=device, dtype=torch.float32)
    macro_sequences: List[List[int]] = []

    frames = []
    prev_occ: np.ndarray | None = None

    for step in range(1, int(args.steps) + 1):
        state_energy.mul_(float(args.decay))
        edge_strength.mul_(float(args.decay))
        pair_strength.mul_(float(args.decay))
        trigram_strength.mul_(float(args.decay))
        primitive_usage.mul_(float(args.decay))

        switch_mask = (torch.rand(num_agents, device=device) < float(args.motif_switch_prob)) & (motifs != len(MOTIF_NAMES) - 1)
        if bool(switch_mask.any().item()):
            motifs[switch_mask] = sample_structured_motifs(int(switch_mask.sum().item()), device=device)
            states[switch_mask] = sample_states_for_motifs(motifs[switch_mask], anchor_span=int(args.anchor_span))

        macro_sequences = top_macro_candidates(trigram_strength, threshold=float(args.macro_threshold), limit=int(args.max_macros))

        periods = periods_bank[group_ids]
        phases = phase_bank[group_ids]
        amps = amp_bank[group_ids]

        t = torch.tensor(float(step), device=device, dtype=torch.float32)
        base = torch.sin((2.0 * math.pi * t / periods) + phases)
        overtone = torch.sin((4.0 * math.pi * t / periods) + (phases * 0.5))
        tempo_term = amps * (0.75 * base + 0.25 * overtone)
        resonance = primitive_resonance(states=states, periods=periods, phase_offsets=phases)
        base_scores = float(args.tempo_gain) * tempo_term + float(args.resonance_gain) * resonance + float(args.motif_gain) * motif_bias[motifs]

        usage_norm = primitive_usage[:BASE_OP_COUNT] / primitive_usage[:BASE_OP_COUNT].sum().clamp_min(1e-6)
        novelty_penalty = torch.log1p(usage_norm * float(num_agents) + 1.0)
        base_scores = base_scores - novelty_penalty.unsqueeze(0) * novelty_by_group[group_ids].unsqueeze(1)

        valid_prev = prev_ops >= 0
        if bool(valid_prev.any().item()):
            safe_prev = prev_ops[valid_prev].clamp(max=BASE_OP_COUNT - 1)
            base_scores[valid_prev] = base_scores[valid_prev] + float(args.coupling_gain) * pair_strength[safe_prev]

        macro_scores = None
        if macro_sequences:
            macro_cols = []
            for seq in macro_sequences:
                seq_score = base_scores[:, seq].mean(dim=1)
                if bool(valid_prev.any().item()):
                    seq_score = seq_score + float(args.macro_gain) * (prev_ops == seq[0]).to(dtype=torch.float32)
                macro_cols.append(seq_score)
            macro_scores = torch.stack(macro_cols, dim=1)

        full_scores = base_scores if macro_scores is None else torch.cat([base_scores, macro_scores], dim=1)
        if bool(noise_mask.any().item()):
            full_scores[noise_mask] = full_scores[noise_mask] + 0.35 * torch.randn_like(full_scores[noise_mask])

        sampled = torch.empty(num_agents, device=device, dtype=torch.long)
        for gid in range(num_groups):
            mask = group_ids == gid
            probs = torch.softmax(full_scores[mask] / max(1e-4, float(temp_by_group[gid].item())), dim=1)
            sampled[mask] = torch.multinomial(probs, num_samples=1).squeeze(1)

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
            edge_strength.index_put_((states[base_mask], sampled[base_mask]), torch.full((int(base_mask.sum().item()),), float(args.edge_lr), device=device, dtype=torch.float32), accumulate=True)
        state_energy.index_put_((next_states,), torch.full((num_agents,), float(args.state_lr), device=device, dtype=torch.float32), accumulate=True)
        primitive_usage.index_put_((sampled,), torch.ones(num_agents, device=device, dtype=torch.float32), accumulate=True)

        valid_pair = (prev_ops >= 0) & (sampled < BASE_OP_COUNT)
        if bool(valid_pair.any().item()):
            pair_strength.index_put_((prev_ops[valid_pair].clamp(max=BASE_OP_COUNT - 1), sampled[valid_pair]), torch.full((int(valid_pair.sum().item()),), float(args.pair_lr), device=device, dtype=torch.float32), accumulate=True)

        valid_tri = (prev_prev_ops >= 0) & (prev_ops >= 0) & (sampled < BASE_OP_COUNT)
        if bool(valid_tri.any().item()):
            trigram_strength.index_put_((prev_prev_ops[valid_tri].clamp(max=BASE_OP_COUNT - 1), prev_ops[valid_tri].clamp(max=BASE_OP_COUNT - 1), sampled[valid_tri]), torch.full((int(valid_tri.sum().item()),), float(args.trigram_lr), device=device, dtype=torch.float32), accumulate=True)

        for gid in range(num_groups):
            mask = group_ids == gid
            group_usage = primitive_usage[:BASE_OP_COUNT] / primitive_usage[:BASE_OP_COUNT].sum().clamp_min(1e-6)
            amp_bank[gid] = torch.clamp(0.97 * amp_bank[gid] + float(args.tempo_lr) * (1.0 - group_usage * BASE_OP_COUNT), min=0.35, max=3.0)
            phase_bank[gid] = torch.remainder(phase_bank[gid] + float(args.tempo_lr) * (amp_bank[gid] - amp_bank[gid].mean()), 2.0 * math.pi)
            _ = mask

        prev_prev_ops = torch.where(sampled < BASE_OP_COUNT, prev_ops, prev_prev_ops)
        prev_ops = torch.where(sampled < BASE_OP_COUNT, sampled, prev_ops)
        states = next_states

        if step == 1 or step % int(args.capture_every) == 0 or step == int(args.steps):
            occ = occupancy_grid(states)
            overlay = group_overlay(states, group_ids, num_groups)
            delta = np.zeros_like(occ) if prev_occ is None else occ - prev_occ
            group_hists = [op_counts_for_group(sampled, group_ids == gid, BASE_OP_COUNT + len(macro_sequences)) for gid in range(num_groups)]
            frames.append(render_frame(step, occ, overlay, delta, group_hists, top_chain_strings(trigram_strength), regs))
            prev_occ = occ.copy()
            print(f"captured_frame_step={step}")

    imageio.mimsave(args.out, frames, fps=max(1, int(args.fps)), loop=0)
    print(f"saved_animation={args.out}")


if __name__ == "__main__":
    main()
