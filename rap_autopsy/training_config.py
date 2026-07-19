import argparse
import json
import math
import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from raplib.data import bytes_to_ids, iter_raw_byte_sequences, iter_train_sequences
from raplib.model import ModelConfig, PrimitiveManifoldLM


class SymmetryScout:
    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 512,
        verbose: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "llama-cpp-python is required for --use_tiny_observer. Install it with: pip install llama-cpp-python"
            ) from e

        self._llm = Llama(model_path=model_path, n_ctx=int(n_ctx), n_gpu_layers=0, verbose=bool(verbose))

    def score(
        self,
        *,
        prompt: str,
        temperature: float = 0.1,
        top_p: float = 0.95,
        max_tokens: int = 16,
    ) -> float:
        res = self._llm(prompt, max_tokens=int(max_tokens), temperature=float(temperature), top_p=float(top_p))
        txt = ""
        try:
            txt = str(res["choices"][0]["text"]).strip()
        except Exception:
            txt = ""

        import re

        m = re.search(r"(-?\d+(?:\.\d+)?)", txt)
        if not m:
            return 5.0
        try:
            v = float(m.group(1))
        except ValueError:
            return 5.0
        return float(max(1.0, min(10.0, v)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="natural_reasoning_train.jsonl")
    p.add_argument("--out", type=str, default="checkpoints/rap_v0.pt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=768)
    p.add_argument("--max_prompt", type=int, default=512)
    p.add_argument("--max_target", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_state", type=int, default=512)
    p.add_argument("--n_primitives", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--amp", action="store_true", help="Enable mixed-precision training with autocast (CUDA only)")
    p.add_argument(
        "--amp_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16"],
        help="Autocast dtype when --amp is enabled (bf16 requires GPU support)",
    )

    p.add_argument("--use_bit_linear", action="store_true", help="Use bit-level linear projection inputs")
    p.add_argument("--use_symmetry_expansion", action="store_true", help="Expand inputs with geometric symmetries (rotate, flip, invert)")
    p.add_argument("--use_symmetry_retrieval", action="store_true", help="Enable causal Ctrl+F retrieval over bit-invert/bit-reverse transformed byte IDs")
    p.add_argument("--use_path_retrieval", action="store_true", help="Enable causal retrieval over k-hop neighbors in the byte-move graph (rot/inv/rev/inc/dec)")
    p.add_argument("--path_k", type=int, default=2, help="Neighborhood radius k for path retrieval (k-hop BFS)")
    p.add_argument("--path_alpha", type=float, default=1.0, help="Distance decay alpha for path retrieval weights: w=exp(-alpha*dist)")
    p.add_argument("--use_word_map_retrieval", action="store_true", help="Enable causal retrieval over immediate neighbors from a precomputed word-chain map")
    p.add_argument("--word_map_path", type=str, default="", help="Path to word_chains.txt used for word-map retrieval")
    p.add_argument("--word_map_topk", type=int, default=32, help="Top-K outgoing neighbors per byte from the word map")
    p.add_argument("--word_map_min_count", type=int, default=1, help="Minimum edge count to keep in the word map adjacency")
    p.add_argument("--use_hypercube", action="store_true", help="Expand inputs with 48 rigid symmetries of a 3D hypercube")

    p.add_argument("--use_numeric_features", action="store_true", help="Inject explicit numeric features (is_digit, digit_value, byte_value) into embeddings")
    p.add_argument("--use_aux_numeric", action="store_true", help="Add auxiliary digit/nibble prediction losses")
    p.add_argument("--aux_digit_weight", type=float, default=0.1, help="Weight for digit classification auxiliary loss")
    p.add_argument("--aux_nibble_weight", type=float, default=0.1, help="Weight for high/low nibble auxiliary loss")

    p.add_argument("--use_delta_erase", action="store_true", help="Enable entropy-driven Delta/Eraser overwrite during state updates")
    p.add_argument("--delta_entropy_lo", type=float, default=0.55, help="Normalized entropy threshold where erasing begins")
    p.add_argument("--delta_entropy_hi", type=float, default=0.95, help="Normalized entropy threshold where erasing is maximal")
    p.add_argument("--delta_beta_max", type=float, default=1.0, help="Maximum erase strength (beta)")

    p.add_argument("--save_every", type=int, default=200, help="Save checkpoint every N steps")
    p.add_argument("--save_steps", action="store_true", help="Also write step-stamped checkpoints next to --out")
    p.add_argument("--log_metrics", action="store_true", help="Log primitive usage/entropy metrics during training")
    p.add_argument("--metrics_every", type=int, default=200, help="Log metrics every N steps (requires --log_metrics)")
    p.add_argument("--metrics_samples", type=int, default=16, help="Number of eval samples for metrics")
    p.add_argument("--metrics_seq_len", type=int, default=128, help="Max bytes per eval sample for metrics")

    p.add_argument("--no_mask_padding", action="store_true", help="Do not mask loss on padded tail bytes (not recommended)")
    p.add_argument("--probe_nulls", action="store_true", help="Print null/transform statistics during training")
    p.add_argument("--probe_every", type=int, default=200, help="Probe every N steps (requires --probe_nulls)")

    p.add_argument("--use_meta_observer", action="store_true", help="Enable heuristic meta-observer loss reweighting")
    p.add_argument("--meta_ema_beta", type=float, default=0.98, help="EMA beta for tracking baseline loss")
    p.add_argument("--meta_loss_spike_mult", type=float, default=3.0, help="Trigger when loss > EMA * mult")
    p.add_argument("--meta_spike_boost", type=float, default=1.0, help="Extra multiplier when a spike triggers")
    p.add_argument(
        "--meta_portal_bytes",
        type=str,
        default="",
        help="Comma-separated list of portal byte IDs (e.g., '137,176'). Empty disables portal weighting.",
    )
    p.add_argument("--meta_portal_window", type=int, default=10, help="How many downstream tokens to boost after a portal byte")
    p.add_argument("--meta_portal_boost", type=float, default=1.0, help="Extra multiplier applied to downstream window")
    p.add_argument(
        "--meta_portal_requires_spike",
        action="store_true",
        help="Only apply portal-byte downstream boosting when a loss spike triggers",
    )
    p.add_argument("--meta_max_weight", type=float, default=8.0, help="Clamp max observer weight per token")

    p.add_argument("--use_tiny_observer", action="store_true", help="Enable CPU-only tiny GGUF observer via llama-cpp-python")
    p.add_argument("--tiny_observer_model", type=str, default="", help="Path to GGUF model file")
    p.add_argument("--tiny_observer_every", type=int, default=0, help="If >0, run tiny observer every N steps")
    p.add_argument("--tiny_observer_ctx_bytes", type=int, default=256, help="How many bytes to show the observer")
    p.add_argument("--tiny_observer_on_spike", action="store_true", help="Run tiny observer when a loss spike triggers")
    p.add_argument("--tiny_observer_on_portal", action="store_true", help="Run tiny observer when a portal byte is present")
    p.add_argument("--tiny_observer_max_mult", type=float, default=2.0, help="Clamp max loss multiplier from tiny observer")
    p.add_argument("--tiny_observer_temperature", type=float, default=0.1)
    p.add_argument("--tiny_observer_top_p", type=float, default=0.95)
    p.add_argument(
        "--tiny_observer_system_prompt",
        type=str,
        default=(
            "You are a structural complexity scout for a byte-level language model. "
            "You will receive a short byte snippet shown as hex bytes and a lossy UTF-8 text view. "
            "Return ONLY a single number 1-10: 1 means simple/natural, 10 means highly structured/complex "
            "(math, code, tables, dense symbols, irregular byte patterns)."
        ),
    )

    p.add_argument("--log_train_events", action="store_true", help="Write per-step training event JSONL for visualization")
    p.add_argument("--train_events_path", type=str, default="", help="Path to write train events JSONL (default: <out>.events.jsonl)")
    p.add_argument("--train_events_every", type=int, default=10, help="Write train event record every N steps")

    p.add_argument("--use_consistency", action="store_true", help="Enable noise-augmented consistency loss (stability under small prompt perturbations)")
    p.add_argument("--consistency_weight", type=float, default=0.1, help="Weight for consistency KL loss")
    p.add_argument("--consistency_noise_prob", type=float, default=0.01, help="Per-byte probability of corrupting prompt bytes in the noised view")
    p.add_argument(
        "--consistency_noise_mode",
        type=str,
        default="random",
        choices=["random", "xor_ff"],
        help="How to corrupt bytes in the noised view: random replacement, or XOR 0xFF",
    )

    p.add_argument("--primitive_balance_weight", type=float, default=0.0, help="Weight for encouraging balanced primitive usage over supervised tokens")
    p.add_argument("--primitive_entropy_weight", type=float, default=0.0, help="Weight for encouraging non-collapsed primitive routing entropy")
    p.add_argument("--primitive_target_entropy", type=float, default=0.75, help="Target normalized routing entropy (0-1) for primitive entropy regularization")

    p.add_argument("--resume", action="store_true", help="Resume from the checkpoint if it exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ckpt = None
    start_step = 0

    # Build a CLI cfg (used when starting fresh; also used for mismatch warnings).
    cfg_cli = ModelConfig(
        d_model=args.d_model,
        d_state=args.d_state,
        n_primitives=args.n_primitives,
        use_bit_linear=args.use_bit_linear,
        use_symmetry_expansion=args.use_symmetry_expansion,
        use_symmetry_retrieval=args.use_symmetry_retrieval,
        use_path_retrieval=args.use_path_retrieval,
        path_k=args.path_k,
        path_alpha=args.path_alpha,
        use_word_map_retrieval=args.use_word_map_retrieval,
        word_map_path=str(getattr(args, "word_map_path", "")),
        word_map_topk=int(getattr(args, "word_map_topk", 32)),
        word_map_min_count=int(getattr(args, "word_map_min_count", 1)),
        use_hypercube=args.use_hypercube,

        use_numeric_features=args.use_numeric_features,
        use_aux_numeric=args.use_aux_numeric,
        aux_digit_weight=args.aux_digit_weight,
        aux_nibble_weight=args.aux_nibble_weight,

        use_delta_erase=args.use_delta_erase,
        delta_entropy_lo=args.delta_entropy_lo,
        delta_entropy_hi=args.delta_entropy_hi,
        delta_beta_max=args.delta_beta_max,
    )

    if args.resume and os.path.exists(args.out):
        print(f"Resuming from {args.out}...")
        ckpt = torch.load(args.out, map_location=args.device)

        if isinstance(ckpt, dict) and "cfg" in ckpt:
            ckpt_cfg_dict = ckpt["cfg"]
            valid_keys = ModelConfig.__annotations__.keys()
            filtered = {k: v for k, v in ckpt_cfg_dict.items() if k in valid_keys}
            cfg = ModelConfig(**filtered)
        else:
            cfg = cfg_cli

        # Warn if CLI differs from checkpoint on architecture-affecting fields.
        arch_keys = [
            "d_model",
            "d_state",
            "n_primitives",
            "use_bit_linear",
            "use_symmetry_expansion",
            "use_hypercube",
            "use_symmetry_retrieval",
            "use_path_retrieval",
            "use_word_map_retrieval",
            "path_k",
            "path_alpha",
            "word_map_path",
            "word_map_topk",
            "word_map_min_count",
            "use_numeric_features",
            "use_aux_numeric",
        ]
        mism = []
        for k in arch_keys:
            if getattr(cfg_cli, k, None) != getattr(cfg, k, None):
                mism.append((k, getattr(cfg_cli, k, None), getattr(cfg, k, None)))
        if mism:
            print("[resume warning] CLI flags differ from checkpoint cfg; using checkpoint cfg:")
            for k, a, b in mism:
                print(f"  - {k}: cli={a}  ckpt={b}")

        model = PrimitiveManifoldLM(cfg).to(args.device)
        opt = AdamW(model.parameters(), lr=args.lr)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)

        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])

        if isinstance(ckpt, dict) and "step" in ckpt:
            start_step = int(ckpt["step"]) + 1
            print(f"Resumed at step {start_step}")
        else:
            print("Checkpoint did not contain step info; starting loop from 0 (with loaded weights).")
    else:
        cfg = cfg_cli
        model = PrimitiveManifoldLM(cfg).to(args.device)
        opt = AdamW(model.parameters(), lr=args.lr)

    if str(args.data).lower().endswith(".bin"):
        stream = iter_raw_byte_sequences(
            bin_path=args.data,
            max_total_bytes=args.seq_len + 1,
        )
    else:
        stream = iter_train_sequences(
            jsonl_path=args.data,
            max_prompt_bytes=args.max_prompt,
            max_target_bytes=args.max_target,
            max_total_bytes=args.seq_len + 1,
            include_answer_prefix=True,
        )

    metrics_stream = None
    if args.log_metrics:
        if str(args.data).lower().endswith(".bin"):
            metrics_stream = iter_raw_byte_sequences(
                bin_path=args.data,
                max_total_bytes=args.metrics_seq_len + 1,
            )
        else:
            metrics_stream = iter_train_sequences(
                jsonl_path=args.data,
                max_prompt_bytes=args.max_prompt,
                max_target_bytes=args.max_target,
                max_total_bytes=args.metrics_seq_len + 1,
                include_answer_prefix=True,
            )

    metrics_path = os.path.splitext(args.out)[0] + ".metrics.jsonl"
    if args.log_metrics:
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)

    events_path = None
    if bool(getattr(args, "log_train_events", False)):
        raw = str(getattr(args, "train_events_path", "")).strip()
        if raw:
            events_path = raw
        else:
            events_path = os.path.splitext(args.out)[0] + ".events.jsonl"
        os.makedirs(os.path.dirname(events_path) or ".", exist_ok=True)

    model.train()

    use_amp = bool(getattr(args, "amp", False)) and str(args.device).startswith("cuda") and torch.cuda.is_available()
    if use_amp:
        if str(getattr(args, "amp_dtype", "fp16")).lower() == "bf16":
            amp_dtype = torch.bfloat16
            scaler = None
        else:
            amp_dtype = torch.float16
            scaler = GradScaler()
    else:
        amp_dtype = None
        scaler = None

    portal_ids = []
    if bool(getattr(args, "use_meta_observer", False)) and str(getattr(args, "meta_portal_bytes", "")).strip():
        for part in str(args.meta_portal_bytes).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                v = int(part)
            except ValueError:
                continue
            if 0 <= v <= 255:
                portal_ids.append(v)

    ema_loss = None
    tiny = None

    rev_lut = torch.empty(256, dtype=torch.long)
    for i in range(256):
        v = 0
        for b in range(8):
            v = (v << 1) | ((i >> b) & 1)
        rev_lut[i] = v
    rev_lut = rev_lut.to(device=args.device)

    if start_step >= int(args.steps):
        print(
            f"Nothing to do: checkpoint start_step={int(start_step)} but --steps={int(args.steps)}. "
            "Increase --steps to continue training."
        )
        return

    pbar = tqdm(range(start_step, args.steps), desc="train")
    for step in pbar:
        batch = []
        prompt_lens = []
        seq_lens = []
        while len(batch) < args.batch_size:
            seq_bytes, prompt_len = next(stream)
            ids = bytes_to_ids(seq_bytes)
            if len(ids) < 2:
                continue
            eff_len = min(len(ids), args.seq_len + 1)
            ids = ids[: args.seq_len + 1]
            if len(ids) < args.seq_len + 1:
                ids = ids + (0,) * (args.seq_len + 1 - len(ids))
            batch.append(ids)
            prompt_lens.append(min(prompt_len, args.seq_len))
            seq_lens.append(min(eff_len, args.seq_len + 1))

        x = torch.tensor([b[: args.seq_len] for b in batch], device=args.device, dtype=torch.long)
        y = torch.tensor([b[1 : args.seq_len + 1] for b in batch], device=args.device, dtype=torch.long)

        with autocast(enabled=bool(use_amp), dtype=amp_dtype):
            need_prim_trace = float(getattr(args, "primitive_balance_weight", 0.0)) != 0.0 or float(getattr(args, "primitive_entropy_weight", 0.0)) != 0.0
            out = model(x, return_trace=bool(need_prim_trace))
            logits = out[0]
            aux = out[2] if len(out) > 2 else None

        loss_mask = torch.zeros_like(y, dtype=torch.float32)
        if args.no_mask_padding:
            for i, pl in enumerate(prompt_lens):
                loss_mask[i, pl:] = 1.0
        else:
            for i, (pl, sl) in enumerate(zip(prompt_lens, seq_lens)):
                end = max(0, int(sl) - 1)
                start = min(int(pl), args.seq_len)
                if end > start:
                    loss_mask[i, start:end] = 1.0
        loss_mask = loss_mask / (loss_mask.sum() + 1e-8)

        if args.probe_nulls and args.probe_every > 0 and (step + 1) % int(args.probe_every) == 0:
            with torch.no_grad():
                m = (loss_mask > 0)
                x_m = x[m]
                if x_m.numel() > 0:
                    inv_m = (x_m ^ 0xFF)
                    rev_m = rev_lut[x_m]
                    n = int(x_m.numel())
                    z_raw = int((x_m == 0).sum().item())
                    z_inv = int((inv_m == 0).sum().item())
                    z_rev = int((rev_m == 0).sum().item())
                    z_any = int(((inv_m == 0) | (rev_m == 0)).sum().item())
                    p_raw = z_raw / max(n, 1)
                    p_inv = z_inv / max(n, 1)
                    p_rev = z_rev / max(n, 1)
                    p_any = z_any / max(n, 1)
                    print(
                        f"[probe_nulls] step={step} supervised_tokens={n} "
                        f"raw0={z_raw}({p_raw:.3f}) inv0={z_inv}({p_inv:.3f}) "
                        f"rev0={z_rev}({p_rev:.3f}) inv_or_rev0={z_any}({p_any:.3f})"
                    )

        logits_for_loss = logits.float() if logits.dtype != torch.float32 else logits
        loss_per_tok = F.cross_entropy(logits_for_loss.reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="none")
        loss_tok = loss_per_tok.reshape_as(y)

        consistency_loss = None
        if bool(getattr(args, "use_consistency", False)) and float(getattr(args, "consistency_weight", 0.0)) != 0.0:
            # Build a noised view of the prompt (only within prompt_lens). The goal is that
            # small prompt perturbations do not cause large downstream distribution drift.
            x_noisy = x.detach().clone()
            prob = float(getattr(args, "consistency_noise_prob", 0.01))
            mode = str(getattr(args, "consistency_noise_mode", "random"))
            if prob > 0.0:
                bsz, tlen = x_noisy.shape
                for i, pl in enumerate(prompt_lens):
                    pl_i = int(max(0, min(int(pl), tlen)))
                    if pl_i <= 0:
                        continue
                    m = (torch.rand((pl_i,), device=x_noisy.device) < prob)
                    if not bool(m.any().item()):
                        continue
                    if mode == "xor_ff":
                        x_noisy[i, :pl_i][m] = (x_noisy[i, :pl_i][m] ^ 0xFF)
                    else:
                        r = torch.randint(0, cfg.vocab_size, (int(m.sum().item()),), device=x_noisy.device, dtype=x_noisy.dtype)
                        cur = x_noisy[i, :pl_i][m]
                        r = torch.where(r == cur, (r + 1) % cfg.vocab_size, r)
                        x_noisy[i, :pl_i][m] = r

            with autocast(enabled=bool(use_amp), dtype=amp_dtype):
                out_noisy = model(x_noisy)
                logits_noisy = out_noisy[0]

            with torch.no_grad():
                logp = F.log_softmax(logits.detach(), dim=-1)
                p_clean = logp.exp()
            logq = F.log_softmax(logits_noisy.float() if logits_noisy.dtype != torch.float32 else logits_noisy, dim=-1)
            kl_tok = (p_clean * (logp - logq)).sum(dim=-1)  # (b, t)
            consistency_loss = (kl_tok * loss_mask).sum()

        observer_w = None
        is_spike = False
        if bool(getattr(args, "use_meta_observer", False)):
            with torch.no_grad():
                obs = torch.ones_like(loss_mask)

                denom = float(loss_mask.sum().detach().cpu().item())
                cur = float(((loss_tok.detach() * loss_mask).sum() / (loss_mask.sum() + 1e-8)).detach().cpu().item())

                if ema_loss is None:
                    ema_loss = cur
                else:
                    beta = float(getattr(args, "meta_ema_beta", 0.98))
                    ema_loss = beta * float(ema_loss) + (1.0 - beta) * cur

                spike_mult = float(getattr(args, "meta_loss_spike_mult", 3.0))
                spike_boost = float(getattr(args, "meta_spike_boost", 1.0))
                is_spike = float(ema_loss) > 0.0 and cur > float(ema_loss) * spike_mult
                if spike_boost != 0.0 and is_spike:
                    obs = obs * (1.0 + spike_boost)

                portal_boost = float(getattr(args, "meta_portal_boost", 1.0))
                portal_window = int(getattr(args, "meta_portal_window", 10))
                portal_requires_spike = bool(getattr(args, "meta_portal_requires_spike", False))
                allow_portal = (not portal_requires_spike) or bool(is_spike)
                if allow_portal and portal_ids and portal_window > 0 and portal_boost != 0.0:
                    is_portal = torch.zeros_like(x, dtype=torch.bool)
                    for pid in portal_ids:
                        is_portal = is_portal | (x == int(pid))

                    for off in range(1, int(portal_window) + 1):
                        obs[:, off:] = obs[:, off:] + is_portal[:, :-off].to(dtype=obs.dtype) * portal_boost

                max_w = float(getattr(args, "meta_max_weight", 8.0))
                if max_w > 0:
                    obs = torch.clamp(obs, min=0.0, max=max_w)

                if denom > 0.0:
                    obs = obs / ((obs * loss_mask).sum() / (loss_mask.sum() + 1e-8) + 1e-8)

                observer_w = obs

        tiny_mult = 1.0
        if bool(getattr(args, "use_tiny_observer", False)):
            need_model = bool(str(getattr(args, "tiny_observer_model", "")).strip())
            if not need_model:
                raise ValueError("--use_tiny_observer requires --tiny_observer_model to be set to a GGUF path")

            run_every = int(getattr(args, "tiny_observer_every", 0) or 0)
            run_on_spike = bool(getattr(args, "tiny_observer_on_spike", False))
            run_on_portal = bool(getattr(args, "tiny_observer_on_portal", False))

            has_portal = False
            if run_on_portal and portal_ids:
                with torch.no_grad():
                    has_portal = False
                    for pid in portal_ids:
                        if bool((x == int(pid)).any().item()):
                            has_portal = True
                            break

            do_run = False
            if run_every > 0 and (step % run_every) == 0:
                do_run = True
            if run_on_spike and bool(is_spike):
                do_run = True
            if run_on_portal and bool(has_portal):
                do_run = True

            if do_run:
                if tiny is None:
                    tiny = SymmetryScout(model_path=str(getattr(args, "tiny_observer_model")), n_ctx=512, verbose=False)

                ctx_n = int(getattr(args, "tiny_observer_ctx_bytes", 256))
                ctx_n = max(16, min(int(args.seq_len), ctx_n))
                ids0 = x[0, :ctx_n].detach().cpu().to(dtype=torch.uint8).numpy().tobytes()
                hex_view = " ".join([f"{b:02X}" for b in ids0])
                text_view = ids0.decode("utf-8", errors="replace")

                sys_p = str(getattr(args, "tiny_observer_system_prompt", "")).strip()
                prompt = (
                    f"{sys_p}\n\n"
                    f"HEX:\n{hex_view}\n\n"
                    f"TEXT(utf-8, errors=replace):\n{text_view}\n\n"
                    f"Score 1-10:"
                )

                s = tiny.score(
                    prompt=prompt,
                    temperature=float(getattr(args, "tiny_observer_temperature", 0.1)),
                    top_p=float(getattr(args, "tiny_observer_top_p", 0.95)),
                    max_tokens=16,
                )

                max_mult = float(getattr(args, "tiny_observer_max_mult", 2.0))
                max_mult = max(1.0, max_mult)
                tiny_mult = float(min(max_mult, max(1.0, s / 5.0)))

        if observer_w is not None:
            loss = (loss_tok * loss_mask * observer_w).sum()
        else:
            loss = (loss_tok * loss_mask).sum()

        if consistency_loss is not None:
            loss = loss + float(getattr(args, "consistency_weight", 0.1)) * consistency_loss

        if aux is not None and "prim_weights" in aux:
            prim_w = aux["prim_weights"].float()
            mask_w = loss_mask.unsqueeze(-1)
            prim_mass = (prim_w * mask_w).sum(dim=(0, 1))
            prim_mass = prim_mass / torch.clamp(prim_mass.sum(), min=1e-8)

            balance_weight = float(getattr(args, "primitive_balance_weight", 0.0))
            if balance_weight != 0.0:
                target = torch.full_like(prim_mass, 1.0 / float(cfg.n_primitives))
                balance_loss = ((prim_mass - target) ** 2).mean()
                loss = loss + balance_weight * balance_loss

            entropy_weight = float(getattr(args, "primitive_entropy_weight", 0.0))
            if entropy_weight != 0.0:
                eps = 1e-8
                H_tok = -(prim_w * (prim_w + eps).log()).sum(dim=-1)
                H_mean = (H_tok * loss_mask).sum() / (loss_mask.sum() + eps)
                H_norm = H_mean / math.log(float(cfg.n_primitives))
                target_H = float(getattr(args, "primitive_target_entropy", 0.75))
                entropy_loss = torch.relu(torch.tensor(target_H, device=H_norm.device, dtype=H_norm.dtype) - H_norm)
                loss = loss + entropy_weight * entropy_loss

        if tiny_mult != 1.0:
            loss = loss * float(tiny_mult)

        if events_path is not None:
            every = int(getattr(args, "train_events_every", 10) or 0)
            if every <= 0:
                every = 1
            if (step + 1) % every == 0:
                rec = {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "is_spike": bool(is_spike),
                    "ema_loss": None if ema_loss is None else float(ema_loss),
                    "tiny_mult": float(tiny_mult),
                }
                with open(events_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

        if aux is not None and bool(getattr(cfg, "use_aux_numeric", False)):
            mask_flat = loss_mask.reshape(-1)

            # Digit classification on input tokens (x)
            x_flat = x.reshape(-1)
            digit_mask = ((x_flat >= 48) & (x_flat <= 57)).to(dtype=torch.float32, device=x.device) * mask_flat
            if float(digit_mask.sum().item()) > 0.0:
                digit_targets = (x_flat - 48).clamp(min=0, max=9)
                digit_logits = aux["digit_logits"].reshape(-1, 10)
                digit_loss_tok = F.cross_entropy(digit_logits, digit_targets, reduction="none")
                digit_loss = (digit_loss_tok * digit_mask).sum() / (digit_mask.sum() + 1e-8)
            else:
                digit_loss = torch.zeros((), device=x.device)

            # Nibble prediction on input tokens (x)
            hi_t = (x_flat >> 4).clamp(min=0, max=15)
            lo_t = (x_flat & 15).clamp(min=0, max=15)
            hi_logits = aux["hi_nibble_logits"].reshape(-1, 16)
            lo_logits = aux["lo_nibble_logits"].reshape(-1, 16)
            hi_loss_tok = F.cross_entropy(hi_logits, hi_t, reduction="none")
            lo_loss_tok = F.cross_entropy(lo_logits, lo_t, reduction="none")
            nib_loss = ((hi_loss_tok + lo_loss_tok) * mask_flat).sum() / (mask_flat.sum() + 1e-8)

            loss = loss + float(getattr(cfg, "aux_digit_weight", 0.1)) * digit_loss + float(getattr(cfg, "aux_nibble_weight", 0.1)) * nib_loss

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        pbar.set_postfix(loss=float(loss.detach().cpu()))

        do_save = args.save_every > 0 and (step + 1) % args.save_every == 0
        do_metrics = args.log_metrics and args.metrics_every > 0 and (step + 1) % args.metrics_every == 0

        if do_metrics and metrics_stream is not None:
            model.eval()
            with torch.no_grad():
                counts = torch.zeros(cfg.n_primitives, dtype=torch.long)
                mass_sum = torch.zeros(cfg.n_primitives, dtype=torch.float64)
                ent_sum = 0.0
                maxp_sum = 0.0
                n = 0
                for _ in range(int(args.metrics_samples)):
                    seq_bytes, _pl = next(metrics_stream)
                    ids = bytes_to_ids(seq_bytes)
                    if len(ids) < 2:
                        continue
                    ids = ids[: int(args.metrics_seq_len)]
                    x_eval = torch.tensor([ids], device=args.device, dtype=torch.long)

                    st = model.init_state(batch_size=1, device=torch.device(args.device))
                    last_pos = None
                    state_hist = None
                    batch_ix = None
                    if bool(getattr(model, "retrieval_enabled", False)):
                        bsz, tlen = x_eval.shape
                        last_pos = torch.full((bsz, cfg.vocab_size), -1, device=x_eval.device, dtype=torch.long)
                        state_hist = torch.zeros(bsz, tlen, cfg.d_state, device=x_eval.device, dtype=st.dtype)
                        batch_ix = torch.arange(bsz, device=x_eval.device)
                    for i in range(x_eval.shape[1]):
                        tok = x_eval[:, i]
                        x_emb = model._get_emb(tok)

                        retrieval = None
                        if last_pos is not None and state_hist is not None and batch_ix is not None:
                            retrieval = torch.zeros_like(st)
                            if i > 0:
                                # Exclude reserved tag bytes from any retrieval logic.
                                is_reserved = (tok == 128) | (tok == 129) | (tok == 130)

                                if bool(getattr(model, "path_retrieval_enabled", False)) and bool((~is_reserved).any().item()):
                                    neigh = model.path_neigh_lut[tok]  # (b, M)
                                    dist = model.path_dist_lut[tok].to(dtype=st.dtype)  # (b, M)
                                    neigh_clamped = neigh.clamp(min=0)
                                    pos = last_pos.gather(1, neigh_clamped)
                                    valid = (neigh >= 0) & (pos >= 0) & (~is_reserved).unsqueeze(1)
                                    if bool(valid.any().item()):
                                        pos_clamped = pos.clamp(min=0)
                                        got = torch.gather(
                                            state_hist,
                                            dim=1,
                                            index=pos_clamped.unsqueeze(-1).expand(-1, -1, state_hist.shape[-1]),
                                        )
                                        alpha = torch.tensor(float(getattr(model, "path_alpha", 1.0)), device=st.device, dtype=st.dtype)
                                        w = torch.exp(-alpha * dist) * valid.to(dtype=st.dtype)
                                        num = (got * w.unsqueeze(-1)).sum(dim=1)
                                        den = torch.clamp(w.sum(dim=1, keepdim=True), min=1.0)
                                        retrieval = num / den

                                elif bool(getattr(model, "symmetry_retrieval_enabled", False)) and bool((~is_reserved).any().item()):
                                    inv = tok ^ 0xFF
                                    rev = model.bit_reverse_lut[tok]

                                    pos_inv = last_pos.gather(1, inv.unsqueeze(1)).squeeze(1)
                                    pos_rev = last_pos.gather(1, rev.unsqueeze(1)).squeeze(1)

                                    cnt = torch.zeros((tok.shape[0], 1), device=st.device, dtype=st.dtype)
                                    for pos1 in (pos_inv, pos_rev):
                                        m = (pos1 >= 0) & (~is_reserved)
                                        if bool(m.any().item()):
                                            p = pos1.clamp(min=0)
                                            got = state_hist[batch_ix, p]
                                            retrieval = retrieval + got * m.to(dtype=st.dtype).unsqueeze(1)
                                            cnt = cnt + m.to(dtype=st.dtype).unsqueeze(1)
                                    retrieval = retrieval / torch.clamp(cnt, min=1.0)

                            ctrl_in = torch.cat([x_emb, st, retrieval], dim=-1)
                        else:
                            ctrl_in = torch.cat([x_emb, st], dim=-1)

                        w = torch.softmax(model.controller(ctrl_in), dim=-1)
                        eps = 1e-8
                        H = -(w * (w + eps).log()).sum(dim=-1)
                        ent_sum += float(H.item())
                        maxp_sum += float(w.max(dim=-1).values.item())
                        counts[int(w.argmax(dim=-1).item())] += 1
                        mass_sum += w[0].detach().cpu().to(dtype=torch.float64)

                        sA = torch.einsum("b d, k d e -> b k e", st, model.prim_A)
                        xB = torch.einsum("b m, k d m -> b k d", x_emb, model.prim_B)
                        cand = torch.tanh(sA + xB + model.prim_bias.unsqueeze(0))
                        mixed = torch.einsum("b k, b k d -> b d", w, cand)
                        gate = model.slice_gate(ctrl_in)
                        if getattr(model, "delta_enabled", False):
                            logK = torch.log(torch.tensor(float(cfg.n_primitives), device=w.device, dtype=w.dtype))
                            Hn = H.view(1, 1) / (logK + eps)
                            lo = torch.tensor(float(getattr(model, "delta_entropy_lo", 0.55)), device=w.device, dtype=w.dtype)
                            hi = torch.tensor(float(getattr(model, "delta_entropy_hi", 0.95)), device=w.device, dtype=w.dtype)
                            beta_max = torch.tensor(float(getattr(model, "delta_beta_max", 1.0)), device=w.device, dtype=w.dtype)
                            denom = torch.clamp(hi - lo, min=1e-6)
                            beta = torch.clamp((Hn - lo) / denom, 0.0, 1.0) * beta_max
                            eff = gate * beta
                            st = (1.0 - eff) * st + eff * mixed
                        else:
                            st = gate * mixed + (1.0 - gate) * st
                        n += 1

                        if state_hist is not None and last_pos is not None:
                            state_hist[:, i, :] = st
                            last_pos.scatter_(1, tok.unsqueeze(1), torch.full((tok.shape[0], 1), i, device=tok.device, dtype=torch.long))

                if n > 0:
                    rec = {
                        "step": int(step),
                        "loss": float(loss.detach().cpu()),
                        "mean_entropy": float(ent_sum / n),
                        "mean_maxprob": float(maxp_sum / n),
                        "argmax_counts": [int(x) for x in counts.tolist()],
                        "mean_primitive_mass": [float(x) for x in (mass_sum / max(n, 1)).tolist()],
                    }
                    with open(metrics_path, "a", encoding="utf-8") as f:
                        f.write(str(rec) + "\n")
            model.train()

        if do_save:
            ckpt_obj = {
                "cfg": asdict(cfg),
                "state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "step": step,
            }
            torch.save(ckpt_obj, args.out)

            if args.save_steps:
                out_dir = os.path.dirname(args.out) or "."
                base = os.path.splitext(os.path.basename(args.out))[0]
                step_path = os.path.join(out_dir, f"{base}_step{(step+1):06d}.pt")
                torch.save(ckpt_obj, step_path)

    final_step = int(args.steps) - 1
    torch.save(
        {
            "cfg": asdict(cfg),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "step": final_step,
        },
        args.out,
    )


if __name__ == "__main__":
    main()
