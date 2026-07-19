from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pathfinder import build_k_hop_tables
from .word_map import build_word_map_tables


@dataclass
class PersistentRAPConfig:
    vocab_size: int = 256
    d_model: int = 256
    d_state: int = 512
    n_programs: int = 2
    use_symmetry_retrieval: bool = False
    use_path_retrieval: bool = False
    use_word_map_retrieval: bool = False
    path_k: int = 2
    path_alpha: float = 1.0
    word_map_path: str = ""
    word_map_topk: int = 32
    word_map_min_count: int = 1
    initial_carrier_probability: float = 0.8
    hard_routing: bool = True


class PersistentProgramState(NamedTuple):
    hidden: torch.Tensor
    program_probability: torch.Tensor


class PersistentTwoProgramLM(nn.Module):
    def __init__(self, cfg: PersistentRAPConfig):
        super().__init__()
        if int(cfg.n_programs) != 2:
            raise ValueError("PersistentTwoProgramLM requires exactly two programs")
        self.cfg = cfg
        self.reserved_bytes = (128, 129, 130)
        self.symmetry_retrieval_enabled = bool(cfg.use_symmetry_retrieval)
        self.path_retrieval_enabled = bool(cfg.use_path_retrieval)
        self.word_map_retrieval_enabled = bool(cfg.use_word_map_retrieval)
        self.retrieval_enabled = bool(
            self.symmetry_retrieval_enabled or self.path_retrieval_enabled or self.word_map_retrieval_enabled
        )
        self.path_k = int(cfg.path_k)
        self.path_alpha = float(cfg.path_alpha)

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if self.symmetry_retrieval_enabled:
            reverse = torch.empty(256, dtype=torch.long)
            for i in range(256):
                value = 0
                for bit in range(8):
                    value = (value << 1) | ((i >> bit) & 1)
                reverse[i] = value
            self.register_buffer("bit_reverse_lut", reverse)

        if self.path_retrieval_enabled:
            neighbors, distances = build_k_hop_tables(k=self.path_k, reserved=self.reserved_bytes)
            self.register_buffer("path_neigh_lut", neighbors)
            self.register_buffer("path_dist_lut", distances)

        if self.word_map_retrieval_enabled:
            if not str(cfg.word_map_path).strip():
                raise ValueError("use_word_map_retrieval requires word_map_path")
            neighbors, weights = build_word_map_tables(
                path=cfg.word_map_path,
                topk=int(cfg.word_map_topk),
                min_count=int(cfg.word_map_min_count),
            )
            self.register_buffer("word_neigh_lut", neighbors)
            self.register_buffer("word_w_lut", weights)

        ctrl_dim = cfg.d_model + cfg.d_state + (cfg.d_state if self.retrieval_enabled else 0) + 2
        self.switch_controller = nn.Sequential(
            nn.Linear(ctrl_dim, cfg.d_state),
            nn.GELU(),
            nn.Linear(cfg.d_state, 1),
        )
        self.prim_A = nn.Parameter(torch.randn(2, cfg.d_state, cfg.d_state) * 0.02)
        self.prim_B = nn.Parameter(torch.randn(2, cfg.d_state, cfg.d_model) * 0.02)
        self.prim_bias = nn.Parameter(torch.zeros(2, cfg.d_state))
        self.slice_gate = nn.Sequential(nn.Linear(ctrl_dim, cfg.d_state), nn.Sigmoid())
        self.out = nn.Linear(cfg.d_model + cfg.d_state, cfg.vocab_size)

    def init_state(self, batch_size: int, device: torch.device) -> PersistentProgramState:
        hidden = torch.zeros(batch_size, self.cfg.d_state, device=device)
        carrier = float(max(0.0, min(1.0, self.cfg.initial_carrier_probability)))
        initial = torch.tensor([1.0 - carrier, carrier], device=device, dtype=hidden.dtype)
        probability = initial.unsqueeze(0).expand(batch_size, -1).clone()
        if self.cfg.hard_routing:
            hard = F.one_hot(probability.argmax(dim=-1), num_classes=2).to(probability.dtype)
            probability = hard + probability - probability.detach()
        return PersistentProgramState(hidden, probability)

    def _get_emb(self, idx: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(idx)

    def _retrieval(
        self,
        tok: torch.Tensor,
        last_pos: torch.Tensor,
        state_hist: torch.Tensor,
        position: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = tok.shape[0]
        device = tok.device
        dtype = state_hist.dtype
        retrieval = torch.zeros(batch, self.cfg.d_state, device=device, dtype=dtype)
        source_count = torch.zeros(batch, 1, device=device, dtype=dtype)
        hit = torch.zeros(batch, device=device, dtype=torch.bool)
        distance_sum = torch.zeros(batch, device=device, dtype=dtype)
        distance_weight = torch.zeros(batch, device=device, dtype=dtype)
        is_reserved = (tok == 128) | (tok == 129) | (tok == 130)
        batch_index = torch.arange(batch, device=device)

        if position > 0 and self.path_retrieval_enabled:
            neighbors = self.path_neigh_lut[tok]
            distances = self.path_dist_lut[tok].to(dtype=dtype)
            positions = last_pos.gather(1, neighbors.clamp(min=0))
            valid = (neighbors >= 0) & (positions >= 0) & (~is_reserved).unsqueeze(1)
            gathered = torch.gather(
                state_hist,
                1,
                positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.cfg.d_state),
            )
            weights = torch.exp(-self.path_alpha * distances) * valid.to(dtype)
            denominator = weights.sum(1, keepdim=True)
            value = (gathered * weights.unsqueeze(-1)).sum(1) / denominator.clamp(min=1.0)
            present = valid.any(dim=1)
            retrieval = retrieval + value * present.to(dtype).unsqueeze(1)
            source_count = source_count + present.to(dtype).unsqueeze(1)
            hit = hit | present
            distance_sum = distance_sum + (weights * distances).sum(1)
            distance_weight = distance_weight + weights.sum(1)

        if position > 0 and self.word_map_retrieval_enabled:
            neighbors = self.word_neigh_lut[tok]
            base_weights = self.word_w_lut[tok].to(dtype=dtype)
            positions = last_pos.gather(1, neighbors.clamp(min=0))
            valid = (neighbors >= 0) & (positions >= 0) & (~is_reserved).unsqueeze(1)
            gathered = torch.gather(
                state_hist,
                1,
                positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.cfg.d_state),
            )
            weights = base_weights * valid.to(dtype)
            denominator = weights.sum(1, keepdim=True)
            value = (gathered * weights.unsqueeze(-1)).sum(1) / denominator.clamp(min=1e-6)
            present = valid.any(dim=1)
            retrieval = retrieval + value * present.to(dtype).unsqueeze(1)
            source_count = source_count + present.to(dtype).unsqueeze(1)
            hit = hit | present

        if position > 0 and self.symmetry_retrieval_enabled:
            inverse = tok ^ 0xFF
            reverse = self.bit_reverse_lut[tok]
            symmetry_value = torch.zeros_like(retrieval)
            symmetry_count = torch.zeros_like(source_count)
            for candidate in (inverse, reverse):
                positions = last_pos.gather(1, candidate.unsqueeze(1)).squeeze(1)
                valid = (positions >= 0) & (~is_reserved)
                gathered = state_hist[batch_index, positions.clamp(min=0)]
                symmetry_value = symmetry_value + gathered * valid.to(dtype).unsqueeze(1)
                symmetry_count = symmetry_count + valid.to(dtype).unsqueeze(1)
                hit = hit | valid
            present = symmetry_count.squeeze(1) > 0
            symmetry_value = symmetry_value / symmetry_count.clamp(min=1.0)
            retrieval = retrieval + symmetry_value * present.to(dtype).unsqueeze(1)
            source_count = source_count + present.to(dtype).unsqueeze(1)

        retrieval = retrieval / source_count.clamp(min=1.0)
        mean_distance = distance_sum / distance_weight.clamp(min=1e-6)
        mean_distance = torch.where(distance_weight > 0, mean_distance, torch.full_like(mean_distance, -1.0))
        return retrieval, {
            "retrieval_hit": hit,
            "retrieval_source_count": source_count.squeeze(1),
            "retrieval_mean_distance": mean_distance,
            "retrieval_raw_norm": torch.linalg.norm(retrieval, dim=-1),
        }

    def step(
        self,
        idx_or_embedding: torch.Tensor,
        state: PersistentProgramState,
        retrieval: Optional[torch.Tensor] = None,
        retrieval_scale: float = 1.0,
        carrier_gate_strength: float = 0.0,
        return_trace: bool = False,
    ):
        x_emb = self._get_emb(idx_or_embedding) if idx_or_embedding.dtype == torch.long else idx_or_embedding
        hidden, current_program = state
        raw_retrieval = torch.zeros_like(hidden) if retrieval is None else retrieval
        carrier_probability = current_program[:, 1:2]
        retrieval_gate = (1.0 - float(carrier_gate_strength)) + float(carrier_gate_strength) * carrier_probability
        effective_retrieval = raw_retrieval * retrieval_gate * float(retrieval_scale)

        parts = [x_emb, hidden]
        if self.retrieval_enabled:
            parts.append(effective_retrieval)
        parts.append(current_program)
        ctrl_in = torch.cat(parts, dim=-1)
        switch_probability = torch.sigmoid(self.switch_controller(ctrl_in))
        stay_probability = 1.0 - switch_probability
        next_soft = torch.cat(
            [
                current_program[:, 0:1] * stay_probability + current_program[:, 1:2] * switch_probability,
                current_program[:, 1:2] * stay_probability + current_program[:, 0:1] * switch_probability,
            ],
            dim=-1,
        )
        if self.cfg.hard_routing:
            next_hard = F.one_hot(next_soft.argmax(dim=-1), num_classes=2).to(next_soft.dtype)
            next_program = next_hard + next_soft - next_soft.detach()
        else:
            next_program = next_soft

        state_candidates = torch.einsum("b d, k d e -> b k e", hidden, self.prim_A)
        input_candidates = torch.einsum("b m, k d m -> b k d", x_emb, self.prim_B)
        candidates = torch.tanh(state_candidates + input_candidates + self.prim_bias.unsqueeze(0))
        selected = torch.einsum("b k, b k d -> b d", next_program, candidates)
        gate = self.slice_gate(ctrl_in)
        new_hidden = gate * selected + (1.0 - gate) * hidden
        logits = self.out(torch.cat([x_emb, new_hidden], dim=-1))
        new_state = PersistentProgramState(new_hidden, next_program)

        if not return_trace:
            return logits, new_state
        trace = {
            "program_probability": next_program,
            "program_probability_soft": next_soft,
            "active_program": next_program.argmax(dim=-1),
            "switch_probability": switch_probability.squeeze(-1),
            "switch_event": (next_program.argmax(dim=-1) != current_program.argmax(dim=-1)).to(new_hidden.dtype),
            "state_norm": torch.linalg.norm(new_hidden, dim=-1),
            "state_delta_norm": torch.linalg.norm(new_hidden - hidden, dim=-1),
            "retrieval_gate": retrieval_gate.squeeze(-1),
            "retrieval_effective_norm": torch.linalg.norm(effective_retrieval, dim=-1),
            "gate_mean": gate.mean(dim=-1),
        }
        return logits, new_state, trace

    def forward(
        self,
        idx: torch.Tensor,
        state: Optional[PersistentProgramState] = None,
        retrieval_scale: float = 1.0,
        carrier_gate_strength: float = 0.0,
        return_trace: bool = False,
    ):
        batch, length = idx.shape
        device = idx.device
        if state is None:
            state = self.init_state(batch, device)
        logits_sequence = []
        trace_sequence: Dict[str, list[torch.Tensor]] = {}

        last_pos = None
        state_hist = None
        if self.retrieval_enabled:
            last_pos = torch.full((batch, self.cfg.vocab_size), -1, device=device, dtype=torch.long)
            state_hist = torch.zeros(batch, length, self.cfg.d_state, device=device, dtype=state.hidden.dtype)

        for position in range(length):
            tok = idx[:, position]
            if self.retrieval_enabled:
                retrieval, retrieval_trace = self._retrieval(tok, last_pos, state_hist, position)
            else:
                retrieval = None
                retrieval_trace = {
                    "retrieval_hit": torch.zeros(batch, device=device, dtype=torch.bool),
                    "retrieval_source_count": torch.zeros(batch, device=device),
                    "retrieval_mean_distance": torch.full((batch,), -1.0, device=device),
                    "retrieval_raw_norm": torch.zeros(batch, device=device),
                }
            logits, state, step_trace = self.step(
                tok,
                state,
                retrieval=retrieval,
                retrieval_scale=retrieval_scale,
                carrier_gate_strength=carrier_gate_strength,
                return_trace=True,
            )
            logits_sequence.append(logits.unsqueeze(1))
            if return_trace:
                combined = dict(retrieval_trace)
                combined.update(step_trace)
                for key, value in combined.items():
                    trace_sequence.setdefault(key, []).append(value.unsqueeze(1))
            if self.retrieval_enabled:
                state_hist[:, position] = state.hidden.detach()
                last_pos.scatter_(1, tok.unsqueeze(1), torch.full((batch, 1), position, device=device, dtype=torch.long))

        logits_out = torch.cat(logits_sequence, dim=1)
        if not return_trace:
            return logits_out, state
        trace_out = {key: torch.cat(values, dim=1) for key, values in trace_sequence.items()}
        return logits_out, state, trace_out
