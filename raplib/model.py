from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .hypercube import get_cube_permutations
from .pathfinder import build_k_hop_tables
from .word_map import build_word_map_tables


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 256
    d_state: int = 512
    n_primitives: int = 8
    use_bit_linear: bool = False
    use_symmetry_expansion: bool = False
    use_symmetry_retrieval: bool = False
    use_path_retrieval: bool = False
    use_word_map_retrieval: bool = False
    path_k: int = 2
    path_alpha: float = 1.0
    use_hypercube: bool = False

    word_map_path: str = ""
    word_map_topk: int = 32
    word_map_min_count: int = 1

    # Delta/Eraser (entropy-driven overwrite) controls
    use_delta_erase: bool = False
    delta_entropy_lo: float = 0.55
    delta_entropy_hi: float = 0.95
    delta_beta_max: float = 1.0

    # Numeric inductive bias / auxiliary objectives
    use_numeric_features: bool = False
    use_aux_numeric: bool = False
    aux_digit_weight: float = 0.1
    aux_nibble_weight: float = 0.1


class PrimitiveManifoldLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.symmetry_retrieval_enabled: bool = bool(getattr(cfg, "use_symmetry_retrieval", False))
        self.path_retrieval_enabled: bool = bool(getattr(cfg, "use_path_retrieval", False))
        self.word_map_retrieval_enabled: bool = bool(getattr(cfg, "use_word_map_retrieval", False))
        self.path_k: int = int(getattr(cfg, "path_k", 2))
        self.path_alpha: float = float(getattr(cfg, "path_alpha", 1.0))

        # Treat tag bytes as atomic (never transformed/retrieved via neighbor graph).
        self.reserved_bytes = (128, 129, 130)

        self.retrieval_enabled: bool = bool(
            self.symmetry_retrieval_enabled or self.path_retrieval_enabled or self.word_map_retrieval_enabled
        )

        # Inference-time "Delta/Eraser" controls. These do not add trainable weights and
        # are safe for old checkpoints.
        self.delta_enabled: bool = bool(getattr(cfg, "use_delta_erase", False))
        self.delta_entropy_lo: float = float(getattr(cfg, "delta_entropy_lo", 0.55))
        self.delta_entropy_hi: float = float(getattr(cfg, "delta_entropy_hi", 0.95))
        self.delta_beta_max: float = float(getattr(cfg, "delta_beta_max", 1.0))

        if cfg.use_bit_linear or cfg.use_symmetry_expansion or cfg.use_hypercube:
            # Base 8 bits
            base_bits = torch.zeros(256, 8, dtype=torch.float32)
            for i in range(256):
                for b in range(8):
                    if (i >> b) & 1:
                        base_bits[i, b] = 1.0
            
            # Map to +/- 1
            base_bits = base_bits * 2.0 - 1.0

            if cfg.use_hypercube:
                # 48 symmetries of the cube
                perms = get_cube_permutations() # (48, 8)
                n_sym = perms.shape[0]
                
                # Apply permutations
                # base_bits: (256, 8)
                # We want (256, 48, 8)
                # Gather matches the index logic
                # bits[i, p, j] = base_bits[i, perms[p, j]]
                
                # Expand base_bits to (256, 1, 8)
                # Expand perms to (1, 48, 8)
                # Actually, simple indexing:
                # We can just do base_bits[:, perms] ? 
                # base_bits[:, perms] would be (256, 48, 8) if pytorch supports it directly like numpy.
                # Yes, torch indexing works.
                
                expanded = base_bits[:, perms] # (256, 48, 8)
                final_bits = expanded.reshape(256, -1) # (256, 384)
                self.input_dim = 384
                
            elif cfg.use_symmetry_expansion:
                # Expand to: Original(8) + Inverted(8) + Reversed(8) + 7 Rotations(56) = 80 bits
                expanded = [base_bits] # Original
                
                # Inverted (-x)
                expanded.append(-base_bits)
                
                # Reversed
                expanded.append(torch.flip(base_bits, dims=[1]))
                
                # Rotations (1 to 7)
                for r in range(1, 8):
                    expanded.append(torch.roll(base_bits, shifts=r, dims=1))
                
                final_bits = torch.cat(expanded, dim=1) # (256, 80)
                self.input_dim = 80
            else:
                final_bits = base_bits
                self.input_dim = 8

            # Make reserved tag bytes atomic under symmetry expansion:
            # overwrite their expanded rows so every symmetry-derived block equals the base bits.
            rep = self.input_dim // 8
            if rep > 1:
                for rb in self.reserved_bytes:
                    final_bits[rb] = base_bits[rb].repeat(rep)

            self.register_buffer("bit_lookup", final_bits)
            self.bit_proj = nn.Linear(self.input_dim, cfg.d_model)
            self.tok_emb = None
        else:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if self.symmetry_retrieval_enabled:
            rev_lut = torch.empty(256, dtype=torch.long)
            for i in range(256):
                v = 0
                for b in range(8):
                    v = (v << 1) | ((i >> b) & 1)
                rev_lut[i] = v
            self.register_buffer("bit_reverse_lut", rev_lut)

        if self.path_retrieval_enabled:
            neigh, dist = build_k_hop_tables(k=self.path_k, reserved=self.reserved_bytes)
            self.register_buffer("path_neigh_lut", neigh)
            self.register_buffer("path_dist_lut", dist)

        if self.word_map_retrieval_enabled:
            wm_path = str(getattr(cfg, "word_map_path", "")).strip()
            if not wm_path:
                raise ValueError("use_word_map_retrieval requires word_map_path to be set")
            neigh_wm, w_wm = build_word_map_tables(
                path=wm_path,
                topk=int(getattr(cfg, "word_map_topk", 32)),
                min_count=int(getattr(cfg, "word_map_min_count", 1)),
            )
            self.register_buffer("word_neigh_lut", neigh_wm)
            self.register_buffer("word_w_lut", w_wm)

        self.numeric_features_enabled: bool = bool(getattr(cfg, "use_numeric_features", False))
        self.aux_numeric_enabled: bool = bool(getattr(cfg, "use_aux_numeric", False))

        if self.numeric_features_enabled:
            # 3 features: is_digit, digit_value_norm, byte_value_norm
            self.numeric_proj = nn.Linear(3, cfg.d_model)
        else:
            self.numeric_proj = None

        if self.aux_numeric_enabled:
            self.aux_digit_head = nn.Linear(cfg.d_state, 10)
            self.aux_hi_nibble_head = nn.Linear(cfg.d_state, 16)
            self.aux_lo_nibble_head = nn.Linear(cfg.d_state, 16)
        else:
            self.aux_digit_head = None
            self.aux_hi_nibble_head = None
            self.aux_lo_nibble_head = None

        ctrl_in_dim = cfg.d_model + cfg.d_state + (cfg.d_state if self.retrieval_enabled else 0)
        self.controller = nn.Sequential(
            nn.Linear(ctrl_in_dim, cfg.d_state),
            nn.GELU(),
            nn.Linear(cfg.d_state, cfg.n_primitives),
        )

        self.prim_A = nn.Parameter(torch.randn(cfg.n_primitives, cfg.d_state, cfg.d_state) * 0.02)
        self.prim_B = nn.Parameter(torch.randn(cfg.n_primitives, cfg.d_state, cfg.d_model) * 0.02)
        self.prim_bias = nn.Parameter(torch.zeros(cfg.n_primitives, cfg.d_state))

        self.slice_gate = nn.Sequential(
            nn.Linear(ctrl_in_dim, cfg.d_state),
            nn.Sigmoid(),
        )

        self.out = nn.Linear(cfg.d_model + cfg.d_state, cfg.vocab_size)

    def set_delta_erase(
        self,
        enabled: bool,
        *,
        entropy_lo: Optional[float] = None,
        entropy_hi: Optional[float] = None,
        beta_max: Optional[float] = None,
    ) -> None:
        self.delta_enabled = bool(enabled)
        if entropy_lo is not None:
            self.delta_entropy_lo = float(entropy_lo)
        if entropy_hi is not None:
            self.delta_entropy_hi = float(entropy_hi)
        if beta_max is not None:
            self.delta_beta_max = float(beta_max)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.cfg.d_state, device=device)

    def _numeric_features(self, idx: torch.Tensor) -> torch.Tensor:
        x = idx.to(dtype=torch.float32)
        byte_norm = (x / 255.0) * 2.0 - 1.0
        is_digit = ((idx >= 48) & (idx <= 57)).to(dtype=torch.float32)
        digit_val = (x - 48.0) / 9.0
        digit_norm = digit_val * 2.0 - 1.0
        digit_norm = digit_norm * is_digit
        return torch.stack([is_digit, digit_norm, byte_norm], dim=-1)

    def _get_emb(self, idx: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_bit_linear or self.cfg.use_symmetry_expansion or self.cfg.use_hypercube:
            bits = torch.nn.functional.embedding(idx, self.bit_lookup)
            x_emb = self.bit_proj(bits)
        else:
            x_emb = self.tok_emb(idx)

        if self.numeric_features_enabled and self.numeric_proj is not None:
            nf = self._numeric_features(idx).to(dtype=x_emb.dtype)
            x_emb = x_emb + self.numeric_proj(nf)
        return x_emb

    def step(
        self,
        x_emb_or_idx: torch.Tensor,
        state: torch.Tensor,
        retrieval: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        trace_detail: bool = False,
    ):
        # Support passing either pre-computed embeddings or raw indices
        if x_emb_or_idx.dtype == torch.long:
             x_emb = self._get_emb(x_emb_or_idx)
        else:
             x_emb = x_emb_or_idx

        if self.retrieval_enabled:
            if retrieval is None:
                retrieval = torch.zeros_like(state)
            ctrl_in = torch.cat([x_emb, state, retrieval], dim=-1)
        else:
            ctrl_in = torch.cat([x_emb, state], dim=-1)
        w = torch.softmax(self.controller(ctrl_in), dim=-1)

        sA = torch.einsum("b d, k d e -> b k e", state, self.prim_A)
        xB = torch.einsum("b m, k d m -> b k d", x_emb, self.prim_B)
        cand = torch.tanh(sA + xB + self.prim_bias.unsqueeze(0))

        mixed = torch.einsum("b k, b k d -> b d", w, cand)

        gate = self.slice_gate(ctrl_in)

        if getattr(self, "delta_enabled", False):
            # Entropy of primitive selection (higher entropy = more uncertainty/conflict).
            # Normalize by log(K) to bring into ~[0,1].
            eps = 1e-8
            H = -(w * (w + eps).log()).sum(dim=-1, keepdim=True)

            # Avoid dtype/device surprises
            logK = torch.log(torch.tensor(float(self.cfg.n_primitives), device=H.device, dtype=H.dtype))
            Hn = H / (logK + eps)

            lo = torch.tensor(float(getattr(self, "delta_entropy_lo", 0.55)), device=H.device, dtype=H.dtype)
            hi = torch.tensor(float(getattr(self, "delta_entropy_hi", 0.95)), device=H.device, dtype=H.dtype)
            beta_max = torch.tensor(float(getattr(self, "delta_beta_max", 1.0)), device=H.device, dtype=H.dtype)
            denom = torch.clamp(hi - lo, min=1e-6)
            beta = torch.clamp((Hn - lo) / denom, 0.0, 1.0) * beta_max

            # Delta overwrite: when beta is high, overwrite more cleanly.
            # Equivalent to: new_state = state + (gate*beta) * (mixed - state)
            eff = gate * beta
            new_state = (1.0 - eff) * state + eff * mixed
        else:
            new_state = gate * mixed + (1.0 - gate) * state

        logits = self.out(torch.cat([x_emb, new_state], dim=-1))
        if not return_trace:
            return logits, new_state

        entropy = -(w * w.clamp_min(1e-8).log()).sum(dim=-1)
        trace: Dict[str, torch.Tensor] = {
            "prim_weights": w,
            "chosen_primitive": w.argmax(dim=-1),
            "routing_entropy": entropy,
            "gate_mean": gate.mean(dim=-1),
        }
        if trace_detail:
            trace.update(
                {
                    "state_before": state,
                    "primitive_candidates": cand,
                    "mixed_candidate": mixed,
                    "slice_gate": gate,
                    "state_after": new_state,
                }
            )
        return logits, new_state, trace

    def forward(
        self,
        idx: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        return_trace: bool = False,
        trace_detail: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b, t = idx.shape
        device = idx.device

        if state is None:
            state = self.init_state(b, device)

        logits_seq = []
        trace_seq: Dict[str, list[torch.Tensor]] = {}
        if self.aux_numeric_enabled:
            digit_seq = []
            hi_seq = []
            lo_seq = []
        if self.retrieval_enabled:
            last_pos = torch.full((b, self.cfg.vocab_size), -1, device=device, dtype=torch.long)
            state_hist = torch.zeros(b, t, self.cfg.d_state, device=device, dtype=state.dtype)
            batch_ix = torch.arange(b, device=device)
            for i in range(t):
                tok = idx[:, i]

                retrieval = torch.zeros_like(state)
                cnt = torch.zeros((b, 1), device=device, dtype=state.dtype)
                if i > 0:
                    # Exclude reserved tag bytes from any retrieval logic.
                    is_reserved = (tok == self.reserved_bytes[0]) | (tok == self.reserved_bytes[1]) | (tok == self.reserved_bytes[2])

                    if self.path_retrieval_enabled and bool((~is_reserved).any().item()):
                        neigh = self.path_neigh_lut[tok]  # (b, M)
                        dist = self.path_dist_lut[tok].to(dtype=state.dtype)  # (b, M)

                        neigh_clamped = neigh.clamp(min=0)
                        pos = last_pos.gather(1, neigh_clamped)

                        valid = (neigh >= 0) & (pos >= 0) & (~is_reserved).unsqueeze(1)
                        if bool(valid.any().item()):
                            pos_clamped = pos.clamp(min=0)
                            st_b = state_hist  # (b, t, d)
                            got = torch.gather(
                                st_b,
                                dim=1,
                                index=pos_clamped.unsqueeze(-1).expand(-1, -1, st_b.shape[-1]),
                            )  # (b, M, d)

                            alpha = torch.tensor(self.path_alpha, device=device, dtype=state.dtype)
                            w = torch.exp(-alpha * dist) * valid.to(dtype=state.dtype)
                            num = (got * w.unsqueeze(-1)).sum(dim=1)
                            den = torch.clamp(w.sum(dim=1, keepdim=True), min=1.0)
                            retrieval = retrieval + (num / den)
                            cnt = cnt + 1.0

                    if self.word_map_retrieval_enabled and bool((~is_reserved).any().item()):
                        neigh = self.word_neigh_lut[tok]  # (b, K)
                        ww = self.word_w_lut[tok].to(dtype=state.dtype)  # (b, K)
                        neigh_clamped = neigh.clamp(min=0)
                        pos = last_pos.gather(1, neigh_clamped)
                        valid = (neigh >= 0) & (pos >= 0) & (~is_reserved).unsqueeze(1)
                        if bool(valid.any().item()):
                            pos_clamped = pos.clamp(min=0)
                            st_b = state_hist
                            got = torch.gather(
                                st_b,
                                dim=1,
                                index=pos_clamped.unsqueeze(-1).expand(-1, -1, st_b.shape[-1]),
                            )
                            w = ww * valid.to(dtype=state.dtype)
                            num = (got * w.unsqueeze(-1)).sum(dim=1)
                            den = torch.clamp(w.sum(dim=1, keepdim=True), min=1e-6)
                            retrieval = retrieval + (num / den)
                            cnt = cnt + 1.0

                    if self.symmetry_retrieval_enabled and bool((~is_reserved).any().item()):
                        inv = tok ^ 0xFF
                        rev = self.bit_reverse_lut[tok]

                        pos_inv = last_pos.gather(1, inv.unsqueeze(1)).squeeze(1)
                        pos_rev = last_pos.gather(1, rev.unsqueeze(1)).squeeze(1)

                        for pos1 in (pos_inv, pos_rev):
                            m = (pos1 >= 0) & (~is_reserved)
                            if bool(m.any().item()):
                                p = pos1.clamp(min=0)
                                got = state_hist[batch_ix, p]
                                retrieval = retrieval + got * m.to(dtype=state.dtype).unsqueeze(1)
                                cnt = cnt + m.to(dtype=state.dtype).unsqueeze(1)

                    retrieval = retrieval / torch.clamp(cnt, min=1.0)

                x_emb = self._get_emb(tok)
                step_output = self.step(
                    x_emb,
                    state,
                    retrieval=retrieval,
                    return_trace=bool(return_trace),
                    trace_detail=bool(trace_detail),
                )
                if bool(return_trace):
                    logits, state, step_trace = step_output
                    for key, value in step_trace.items():
                        trace_seq.setdefault(key, []).append(value.unsqueeze(1))
                else:
                    logits, state = step_output
                logits_seq.append(logits.unsqueeze(1))

                if self.aux_numeric_enabled:
                    digit_seq.append(self.aux_digit_head(state).unsqueeze(1))
                    hi_seq.append(self.aux_hi_nibble_head(state).unsqueeze(1))
                    lo_seq.append(self.aux_lo_nibble_head(state).unsqueeze(1))

                # Retrieval history is a non-differentiable memory buffer.
                # Storing detached states avoids autograd inplace-version errors.
                state_hist[:, i, :] = state.detach()
                last_pos.scatter_(1, tok.unsqueeze(1), torch.full((b, 1), i, device=device, dtype=torch.long))
        else:
            for i in range(t):
                x_emb = self._get_emb(idx[:, i])
                step_output = self.step(
                    x_emb,
                    state,
                    return_trace=bool(return_trace),
                    trace_detail=bool(trace_detail),
                )
                if bool(return_trace):
                    logits, state, step_trace = step_output
                    for key, value in step_trace.items():
                        trace_seq.setdefault(key, []).append(value.unsqueeze(1))
                else:
                    logits, state = step_output
                logits_seq.append(logits.unsqueeze(1))

                if self.aux_numeric_enabled:
                    digit_seq.append(self.aux_digit_head(state).unsqueeze(1))
                    hi_seq.append(self.aux_hi_nibble_head(state).unsqueeze(1))
                    lo_seq.append(self.aux_lo_nibble_head(state).unsqueeze(1))

        logits_out = torch.cat(logits_seq, dim=1)
        if self.aux_numeric_enabled or bool(return_trace):
            aux: Dict[str, torch.Tensor] = {}
            if self.aux_numeric_enabled:
                aux.update(
                    {
                        "digit_logits": torch.cat(digit_seq, dim=1),
                        "hi_nibble_logits": torch.cat(hi_seq, dim=1),
                        "lo_nibble_logits": torch.cat(lo_seq, dim=1),
                    }
                )
            if bool(return_trace):
                aux.update({key: torch.cat(values, dim=1) for key, values in trace_seq.items()})
            return logits_out, state, aux
        return logits_out, state
