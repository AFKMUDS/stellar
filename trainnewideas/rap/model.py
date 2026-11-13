"""RAP recurrence plus a separately learned, reversible byte expression layer.

The learned float recurrence is NOT claimed to be invertible. Only the eight
discrete byte code maps are bijective when their contexts are fixed.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .tokens import BOS, VOCAB_SIZE

PATH_NAMES = ["identity", "xor_previous", "rotate_left", "rotate_right",
              "bit_reverse", "permutation_1", "permutation_2", "xor_lag2"]

def make_maps(seed=701, random_paths=False):
    values = torch.arange(VOCAB_SIZE, dtype=torch.long)
    raw = values[:256]
    reverse = torch.tensor([int(f"{b:08b}"[::-1], 2) for b in range(256)])
    generator = torch.Generator().manual_seed(seed)
    maps = values.repeat(8, 1)
    if random_paths:
        for k in range(8):
            maps[k, :256] = torch.randperm(256, generator=generator)
    else:
        maps[2, :256] = ((raw << 1) & 255) | (raw >> 7)
        maps[3, :256] = (raw >> 1) | ((raw & 1) << 7)
        maps[4, :256] = reverse
        maps[5, :256] = torch.randperm(256, generator=generator)
        maps[6, :256] = torch.randperm(256, generator=generator)
    return maps, reverse

@dataclass
class State:
    h: torch.Tensor
    memory: torch.Tensor
    seen: torch.Tensor
    written_at: torch.Tensor
    position: torch.Tensor
    last_byte: torch.Tensor
    previous_byte: torch.Tensor
    recent: torch.Tensor

    def detached(self):
        return State(**{k: v.detach() for k, v in vars(self).items()})

    def to(self, device):
        return State(**{k: v.to(device) for k, v in vars(self).items()})

    def packed(self):
        return {k: v.detach().cpu().clone() for k, v in vars(self).items()}

    @classmethod
    def unpack(cls, raw, device):
        return cls(**{k: v.to(device) for k, v in raw.items()})

class RAP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m, s, k = cfg.embedding_size, cfg.state_size, cfg.primitives
        self.embedding = nn.Embedding(VOCAB_SIZE, m)
        self.numeric_projection = nn.Linear(3, m) if cfg.numeric_features else None
        self.prim_A = nn.Parameter(torch.randn(k, s, s) * 0.02)
        self.prim_B = nn.Parameter(torch.randn(k, s, m) * 0.02)
        self.prim_bias = nn.Parameter(torch.zeros(k, s))
        # Two ordered memory channels, with presence and age information.
        context_size = m + s + 2*s + 4
        self.controller = nn.Sequential(nn.Linear(context_size, cfg.controller_size),
                                        nn.GELU(), nn.Linear(cfg.controller_size, k))
        self.slice_gate = nn.Linear(context_size, s)
        self.readout = nn.Sequential(nn.Linear(context_size, cfg.readout_size),
                                     nn.GELU(), nn.Linear(cfg.readout_size, VOCAB_SIZE))
        self.expression_gate = None
        self.motif = None
        if cfg.expression_head != "direct":
            self.expression_gate = nn.Linear(context_size, 8)
            self.motif = nn.Embedding(VOCAB_SIZE, 8)
            with torch.no_grad():
                b = torch.arange(VOCAB_SIZE).float().unsqueeze(1).clamp(max=255)/255
                self.motif.weight.copy_(0.001*torch.cos(b*torch.arange(1, 9)*math.pi))
        if cfg.byte_operator_seeds:
            self.seed_weights = nn.Parameter(torch.randn(8, 8, s)*0.002)
            self.seed_scale = nn.Parameter(torch.tensor(0.05))
        else:
            self.register_parameter("seed_weights", None)
            self.register_parameter("seed_scale", None)
        native, reverse = make_maps(cfg.permutation_seed)
        expression, _ = make_maps(cfg.permutation_seed,
                                  random_paths=cfg.expression_head == "random")
        self.register_buffer("native_maps", native)
        self.register_buffer("expression_maps", expression)
        self.register_buffer("bit_reverse", reverse)
        self.register_buffer("byte_values", torch.arange(256))
        self.register_buffer("bit_features", (((torch.arange(256)[:, None] >> torch.arange(8)) & 1).float()*2-1))
        generator = torch.Generator().manual_seed(cfg.permutation_seed+900)
        self.register_buffer("random_keys", torch.stack([torch.randperm(256, generator=generator),
                                                         torch.randperm(256, generator=generator)], dim=-1))
        self.future_heads = nn.ModuleDict({str(h): nn.Linear(s, 256) for h in cfg.future_horizons})
        self.past_heads = nn.ModuleDict({str(h): nn.Linear(s, 256) for h in cfg.past_lags})
        self.boundary_head = nn.Linear(s, 256)
        self.stop_head = nn.Linear(s, 1)

    def initial_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        b, s = batch_size, self.cfg.state_size
        recent_size = max(self.cfg.past_lags+[1])
        return State(torch.zeros(b, s, device=device),
                     torch.zeros(b, 256, s, device=device),
                     torch.zeros(b, 256, dtype=torch.bool, device=device),
                     torch.full((b, 256), -1, dtype=torch.long, device=device),
                     torch.zeros(b, dtype=torch.long, device=device),
                     torch.zeros(b, dtype=torch.long, device=device),
                     torch.zeros(b, dtype=torch.long, device=device),
                     torch.full((b, recent_size), -1, dtype=torch.long, device=device))

    def embed(self, ids):
        e = self.embedding(ids)
        if self.numeric_projection is not None:
            raw = ids < 256
            x = ids.clamp(max=255).float()
            digit = (ids >= 48) & (ids <= 57)
            numeric = torch.stack([digit.float(),
                ((x-48)/9*2-1)*digit, x/255*2-1], dim=-1)
            e = e + self.numeric_projection(numeric)*raw.unsqueeze(-1)
        return e

    def map_indices(self, last, lag2, *, native=False):
        maps = self.native_maps if native else self.expression_maps
        shape = list(last.shape) + [8, VOCAB_SIZE]
        indices = maps.expand(shape).clone()
        if native or self.cfg.expression_head == "reversible":
            indices[..., 1, :256] = self.byte_values ^ last.unsqueeze(-1)
            indices[..., 7, :256] = self.byte_values ^ lag2.unsqueeze(-1)
        return indices

    def _reset(self, state, mask):
        state.h = state.h * (~mask).unsqueeze(-1)
        with torch.no_grad():
            state.memory[mask] = 0
            state.seen[mask] = False
            state.written_at[mask] = -1
            state.position[mask] = 0
            state.last_byte[mask] = 0
            state.previous_byte[mask] = 0
            state.recent[mask] = -1

    def _retrieve(self, idx, state, disabled=False):
        raw = idx < 256
        key_byte = idx.clamp(max=255)
        keys = self.random_keys[key_byte] if self.cfg.retrieval == "random" else torch.stack(
            [key_byte ^ 255, self.bit_reverse[key_byte]], dim=-1)
        rows = torch.arange(len(idx), device=idx.device)
        valid = state.seen[rows[:, None], keys] & raw[:, None]
        if disabled or self.cfg.retrieval == "none":
            valid = torch.zeros_like(valid)
        # Advanced indexing copies the detached values before the table is mutated.
        reads = state.memory[rows[:, None], keys] * valid.unsqueeze(-1)
        age = (state.position[:, None]-state.written_at[rows[:, None], keys]).clamp_min(0)
        age = age * valid
        age_features = torch.log1p(age.float())/math.log(1000001)
        if self.cfg.retrieval == "mean":
            mean = reads.sum(dim=1)/valid.sum(dim=1).clamp_min(1).unsqueeze(-1)
            reads = torch.stack([mean, torch.zeros_like(mean)], dim=1)
            presence = torch.stack([valid.any(dim=1), torch.zeros_like(raw)], dim=1)
            age_features = torch.stack([age_features.sum(1)/valid.sum(1).clamp_min(1),
                                         torch.zeros_like(age_features[:, 0])], dim=1)
        else:
            presence = valid
        metadata = torch.cat([presence.float(), age_features], dim=-1)
        return reads, metadata, valid, age

    def _advance(self, idx, state, e, projected, *, reset=False,
                 transition_policy="native", retrieval_disabled=False):
        if reset:
            self._reset(state, idx == BOS)
        reads, metadata, valid, age = self._retrieve(idx, state, retrieval_disabled)
        context = torch.cat([e, state.h, reads.flatten(1), metadata], dim=-1)
        weights = torch.softmax(self.controller(context).float(), dim=-1)
        if transition_policy == "uniform":
            weights = torch.ones_like(weights)/self.cfg.primitives
        elif transition_policy == "hard":
            weights = F.one_hot(weights.argmax(-1), self.cfg.primitives).float()
        elif transition_policy != "native":
            raise ValueError("Unknown transition policy.")
        recurrent = torch.einsum("bd,kde->bke", state.h, self.prim_A)
        proposal = recurrent + projected + self.prim_bias
        if self.seed_weights is not None:
            byte = idx.clamp(max=255)
            codes = self.native_maps[:, byte].T.clone()
            codes[:, 1] = byte ^ state.last_byte
            codes[:, 7] = byte ^ state.previous_byte
            bits = self.bit_features[codes] * (idx < 256)[:, None, None]
            seed = torch.einsum("bki,kis->bks", bits, self.seed_weights)
            proposal = proposal + self.seed_scale*seed
        candidates = torch.tanh(proposal.float())
        mixed = (weights.unsqueeze(-1)*candidates).sum(dim=1)
        gate = torch.sigmoid(self.slice_gate(context).float())
        h = state.h + gate*(mixed-state.h)
        # Readout is trained jointly and receives the two retrieved values directly.
        readout_features = torch.cat([e, h, reads.flatten(1), metadata], dim=-1)
        past = torch.stack([state.recent[:, lag-1].clone() for lag in self.cfg.past_lags], dim=-1) if self.cfg.past_lags else idx.new_empty((len(idx), 0))
        raw = idx < 256
        next_last = torch.where(raw, idx, state.last_byte)
        next_previous = torch.where(raw, state.last_byte, state.previous_byte)
        state.h = h
        with torch.no_grad():
            rows = torch.arange(len(idx), device=idx.device)
            address = idx.clamp(max=255)
            previous_value = state.memory[rows, address]
            state.memory[rows, address] = torch.where(raw[:, None], h.detach(), previous_value)
            state.seen[rows, address] |= raw
            state.written_at[rows, address] = torch.where(raw, state.position,
                                                        state.written_at[rows, address])
            state.position += 1
            state.last_byte = next_last
            state.previous_byte = next_previous
            state.recent = torch.cat([idx[:, None], state.recent[:, :-1]], dim=1)
        return h, readout_features, weights, gate.mean(-1).detach(), valid, age, past, next_last, next_previous

    def forward(self, ids, state=None, *, transition_policy="native",
                retrieval_disabled=False, expression_policy="native"):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("ids must be a nonempty [batch, time] tensor.")
        if state is None:
            state = self.initial_state(ids.shape[0], ids.device)
        embeddings = self.embed(ids)
        projected = F.linear(embeddings, self.prim_B.flatten(0, 1)).reshape(
            ids.shape[0], ids.shape[1], self.cfg.primitives, self.cfg.state_size)
        reset_columns = (ids == BOS).any(dim=0).detach().cpu().tolist()
        outputs = [[] for _ in range(9)]
        for t in range(ids.shape[1]):
            row = self._advance(ids[:, t], state, embeddings[:, t], projected[:, t],
                reset=reset_columns[t], transition_policy=transition_policy,
                retrieval_disabled=retrieval_disabled)
            for destination, value in zip(outputs, row):
                destination.append(value)
        h, features, route, gates, hits, ages, past, last, lag2 = [torch.stack(x, dim=1) for x in outputs]
        code_logp = F.log_softmax(self.readout(features).float(), dim=-1)
        if self.expression_gate is None:
            head_weights = torch.ones(*ids.shape, 1, device=ids.device)
            indices = torch.arange(VOCAB_SIZE, device=ids.device).expand(*ids.shape, 1, VOCAB_SIZE)
            logp = code_logp
        else:
            gate_logits = self.expression_gate(features).float() + self.motif(ids).float()
            head_weights = torch.softmax(gate_logits, dim=-1)
            if expression_policy == "hard":
                head_weights = F.one_hot(head_weights.argmax(-1), 8).float()
            elif expression_policy == "uniform":
                head_weights = torch.ones_like(head_weights)/8
            elif expression_policy != "native":
                raise ValueError("Unknown expression policy.")
            indices = self.map_indices(last, lag2)
            expert_logp = torch.gather(code_logp.unsqueeze(-2).expand_as(indices), -1, indices)
            logp = torch.logsumexp(head_weights.clamp_min(1e-30).log().unsqueeze(-1)+expert_logp, dim=-2)
        return dict(log_probs=logp, state=state, hidden=h, transition_probs=route,
                    expression_probs=head_weights, code_log_probs=code_logp, map_indices=indices,
                    gate_mean=gates, retrieval_hits=hits, retrieval_ages=ages,
                    past_targets=past, stop_logits=self.stop_head(h).squeeze(-1))

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

