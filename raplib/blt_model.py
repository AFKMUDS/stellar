from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BLTConfig:
    vocab_size: int = 256
    d_model: int = 192
    d_local: int = 192
    d_patch: int = 256
    n_heads: int = 4
    local_layers: int = 2
    global_layers: int = 4
    dropout: float = 0.1
    max_seq_len: int = 1024
    min_patch_len: int = 4
    max_patch_len: int = 32
    entropy_threshold: float = 4.0
    patch_budget: int = 128
    use_bit_embed: bool = True
    diffusion_noise_std: float = 0.0


class ByteEmbedding(nn.Module):
    def __init__(self, cfg: BLTConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.use_bit_embed:
            bits = torch.zeros(cfg.vocab_size, 8, dtype=torch.float32)
            for i in range(cfg.vocab_size):
                for b in range(8):
                    if (i >> b) & 1:
                        bits[i, b] = 1.0
            bits = bits * 2.0 - 1.0
            self.register_buffer("bit_lookup", bits)
            self.bit_proj = nn.Linear(8, cfg.d_model)
            self.token_emb = None
        else:
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.bit_proj = None
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = idx.shape
        pos = torch.arange(seqlen, device=idx.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        if self.token_emb is not None:
            x = self.token_emb(idx)
        else:
            bits = F.embedding(idx, self.bit_lookup)
            x = self.bit_proj(bits)
        return x + self.pos_emb(pos)


class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqlen = x.shape[1]
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device, dtype=torch.bool), diagonal=1)
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class PatchCrossBlock(nn.Module):
    def __init__(self, byte_dim: int, patch_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.byte_to_patch = nn.MultiheadAttention(patch_dim, n_heads, dropout=dropout, batch_first=True, kdim=byte_dim, vdim=byte_dim)
        self.patch_to_byte = nn.MultiheadAttention(byte_dim, n_heads, dropout=dropout, batch_first=True, kdim=patch_dim, vdim=patch_dim)
        self.patch_ln = nn.LayerNorm(patch_dim)
        self.byte_ln = nn.LayerNorm(byte_dim)
        self.patch_mlp = nn.Sequential(
            nn.Linear(patch_dim, patch_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_dim * 4, patch_dim),
            nn.Dropout(dropout),
        )
        self.byte_mlp = nn.Sequential(
            nn.Linear(byte_dim, byte_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(byte_dim * 4, byte_dim),
            nn.Dropout(dropout),
        )

    def forward(self, byte_states: torch.Tensor, patch_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_norm = self.patch_ln(patch_states)
        byte_norm = self.byte_ln(byte_states)
        patch_update, _ = self.byte_to_patch(patch_norm, byte_norm, byte_norm, need_weights=False)
        patch_states = patch_states + patch_update
        patch_states = patch_states + self.patch_mlp(self.patch_ln(patch_states))
        byte_update, _ = self.patch_to_byte(byte_norm, self.patch_ln(patch_states), self.patch_ln(patch_states), need_weights=False)
        byte_states = byte_states + byte_update
        byte_states = byte_states + self.byte_mlp(self.byte_ln(byte_states))
        return byte_states, patch_states


class ByteLatentTransformer(nn.Module):
    def __init__(self, cfg: BLTConfig):
        super().__init__()
        self.cfg = cfg
        self.byte_embed = ByteEmbedding(cfg)
        self.local_in = nn.Linear(cfg.d_model, cfg.d_local)
        self.local_blocks = nn.ModuleList(
            [CausalSelfAttentionBlock(cfg.d_local, cfg.n_heads, cfg.dropout) for _ in range(cfg.local_layers)]
        )
        self.local_ln = nn.LayerNorm(cfg.d_local)
        self.local_head = nn.Linear(cfg.d_local, cfg.vocab_size)
        self.patch_proj = nn.Linear(cfg.d_local + 2, cfg.d_patch)
        self.patch_pos_emb = nn.Embedding(cfg.patch_budget, cfg.d_patch)
        self.global_blocks = nn.ModuleList(
            [CausalSelfAttentionBlock(cfg.d_patch, cfg.n_heads, cfg.dropout) for _ in range(cfg.global_layers)]
        )
        self.cross_blocks = nn.ModuleList(
            [PatchCrossBlock(cfg.d_local, cfg.d_patch, cfg.n_heads, cfg.dropout) for _ in range(max(1, cfg.global_layers // 2))]
        )
        self.byte_fuse = nn.Sequential(
            nn.LayerNorm(cfg.d_local + cfg.d_patch),
            nn.Linear(cfg.d_local + cfg.d_patch, cfg.d_local),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.final_head = nn.Linear(cfg.d_local, cfg.vocab_size)

    def _build_patch_segments(self, entropy: torch.Tensor) -> List[List[Tuple[int, int]]]:
        bsz, seqlen = entropy.shape
        threshold = float(self.cfg.entropy_threshold)
        min_patch = max(1, int(self.cfg.min_patch_len))
        max_patch = max(min_patch, int(self.cfg.max_patch_len))
        budget = max(1, int(self.cfg.patch_budget))
        all_segments: List[List[Tuple[int, int]]] = []
        for b in range(bsz):
            segs: List[Tuple[int, int]] = []
            start = 0
            last_cut = 0
            for idx in range(seqlen):
                length = idx - start + 1
                should_cut = False
                if idx + 1 == seqlen:
                    should_cut = True
                elif length >= max_patch:
                    should_cut = True
                elif length >= min_patch and float(entropy[b, idx].item()) >= threshold:
                    should_cut = True
                if should_cut:
                    segs.append((start, idx + 1))
                    start = idx + 1
                    last_cut = idx + 1
                    if len(segs) >= budget:
                        break
            if last_cut < seqlen and len(segs) < budget:
                segs.append((last_cut, seqlen))
            if not segs:
                segs = [(0, seqlen)]
            all_segments.append(segs)
        return all_segments

    def _pool_patches(self, local_states: torch.Tensor, entropy: torch.Tensor, prelim_conf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, int]]]]:
        segments = self._build_patch_segments(entropy)
        bsz, _seqlen, _ = local_states.shape
        max_patches = min(max(len(seg) for seg in segments), int(self.cfg.patch_budget))
        patch_tensor = torch.zeros(bsz, max_patches, self.cfg.d_patch, device=local_states.device, dtype=local_states.dtype)
        patch_mask = torch.zeros(bsz, max_patches, device=local_states.device, dtype=torch.bool)
        for b in range(bsz):
            segs = segments[b][:max_patches]
            for p, (start, end) in enumerate(segs):
                span = local_states[b, start:end]
                span_entropy = entropy[b, start:end].mean().unsqueeze(0)
                span_conf = prelim_conf[b, start:end].mean().unsqueeze(0)
                pooled = span.mean(dim=0)
                features = torch.cat([pooled, span_entropy, span_conf], dim=0)
                patch_tensor[b, p] = self.patch_proj(features)
                patch_mask[b, p] = True
        pos = torch.arange(max_patches, device=local_states.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        patch_tensor = patch_tensor + self.patch_pos_emb(pos)
        return patch_tensor, patch_mask, segments

    def _scatter_patch_context(
        self,
        patch_states: torch.Tensor,
        segments: List[List[Tuple[int, int]]],
        seqlen: int,
    ) -> torch.Tensor:
        bsz = len(segments)
        out = torch.zeros(bsz, seqlen, patch_states.shape[-1], device=patch_states.device, dtype=patch_states.dtype)
        for b, segs in enumerate(segments):
            for p, (start, end) in enumerate(segs[: patch_states.shape[1]]):
                out[b, start:end] = patch_states[b, p]
        return out

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        diffusion_noise_std: Optional[float] = None,
    ) -> Dict[str, torch.Tensor | List[List[Tuple[int, int]]]]:
        x = self.byte_embed(idx)
        x = self.local_in(x)
        for block in self.local_blocks:
            x = block(x)
        local_states = self.local_ln(x)
        prelim_logits = self.local_head(local_states)
        prelim_probs = torch.softmax(prelim_logits.detach(), dim=-1)
        prelim_entropy = -(prelim_probs * torch.log(prelim_probs.clamp_min(1e-8))).sum(dim=-1)
        prelim_conf = prelim_probs.max(dim=-1).values

        patch_states, patch_mask, segments = self._pool_patches(local_states, prelim_entropy, prelim_conf)
        key_padding_mask = ~patch_mask
        for idx_block, block in enumerate(self.global_blocks):
            patch_states = block(patch_states)
            if idx_block < len(self.cross_blocks):
                local_states, patch_states = self.cross_blocks[idx_block](local_states, patch_states)
        if diffusion_noise_std is None:
            diffusion_noise_std = float(self.cfg.diffusion_noise_std)
        if diffusion_noise_std > 0.0 and self.training:
            patch_states = patch_states + torch.randn_like(patch_states) * float(diffusion_noise_std)
        patch_context = self._scatter_patch_context(patch_states, segments, idx.shape[1])
        fused = self.byte_fuse(torch.cat([local_states, patch_context], dim=-1))
        logits = self.final_head(fused)

        out: Dict[str, torch.Tensor | List[List[Tuple[int, int]]]] = {
            "logits": logits,
            "prelim_logits": prelim_logits,
            "patch_mask": patch_mask,
            "entropy": prelim_entropy,
            "segments": segments,
        }
        if targets is not None:
            main_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
            prelim_loss = F.cross_entropy(prelim_logits.reshape(-1, prelim_logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
            patch_count = patch_mask.sum(dim=-1).to(dtype=logits.dtype)
            patch_penalty = patch_count.mean() / max(1.0, float(self.cfg.patch_budget))
            out["loss"] = main_loss + 0.25 * prelim_loss + 0.01 * patch_penalty
            out["main_loss"] = main_loss
            out["prelim_loss"] = prelim_loss
            out["patch_count"] = patch_count.mean()
        return out

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        device: Optional[torch.device] = None,
    ) -> List[int]:
        if device is None:
            device = next(self.parameters()).device
        generated = list(int(x) & 0xFF for x in prompt_ids)
        self.eval()
        for _ in range(max_new_tokens):
            context = generated[-int(self.cfg.max_seq_len) :]
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = self.forward(x)["logits"]
            next_logits = logits[:, -1, :] / max(float(temperature), 1e-6)
            if int(top_k) > 0:
                values, _ = torch.topk(next_logits, min(int(top_k), next_logits.shape[-1]))
                next_logits = next_logits.masked_fill(next_logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_id)
        return generated

    def config_dict(self) -> Dict[str, object]:
        return asdict(self.cfg)
