"""Four-way checkpoint-B experiment. Run from repo root: python -m rap_training_rebuild.experiment."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from raplib.model import ModelConfig, PrimitiveManifoldLM
from .data import DELAYS, batch, encode, load_examples, memory_example, operation_example
from .flags import add_flags, validate_flags
from .online_teacher import QwenTeacher, append_teacher_example, load_teacher_examples, weakest_delay


MIX = {"language": 35, "conversation": 30, "memory": 20, "operations": 10, "stability": 5}


class Readout(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.d_model + cfg.d_state * 2),
            nn.Linear(cfg.d_model + cfg.d_state * 2, cfg.d_state),
            nn.GELU(),
            nn.Linear(cfg.d_state, cfg.vocab_size),
        )

    def forward(self, emb, state, retrieval):
        return self.net(torch.cat((emb, state, retrieval), dim=-1))


def load_base(path, device):
    # Local trusted project checkpoint only: weights_only=False loads Python pickle.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in saved["cfg"].items() if k in ModelConfig.__annotations__})
    if saved.get("step") != 87199:
        raise ValueError(f"Expected checkpoint B step 87199, found {saved.get('step')}")
    if not cfg.use_symmetry_retrieval or cfg.use_path_retrieval or cfg.use_word_map_retrieval:
        raise ValueError("Pilot supports only B's symmetry retrieval")
    model = PrimitiveManifoldLM(cfg).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model


class ByteStream:
    """Match the symmetry retrieval in PrimitiveManifoldLM.forward, including detached history."""

    def __init__(self, model, readout, batch_size, device):
        self.model, self.readout = model, readout
        self.state = model.init_state(batch_size, device)
        self.seen = torch.zeros(batch_size, 256, dtype=torch.bool, device=device)
        self.table = torch.zeros(batch_size, 256, model.cfg.d_state, device=device)
        self.rows = torch.arange(batch_size, device=device)

    def step(self, byte):
        hits = (byte ^ 255, self.model.bit_reverse_lut[byte])
        retrieval = torch.zeros_like(self.state)
        count = torch.zeros(byte.shape[0], 1, device=byte.device)
        active = ~((byte == 128) | (byte == 129) | (byte == 130))
        for hit in hits:
            valid = self.seen[self.rows, hit] & active
            retrieval = retrieval + self.table[self.rows, hit] * valid[:, None]
            count = count + valid[:, None]
        retrieval = retrieval / count.clamp_min(1)
        emb = self.model._get_emb(byte)
        native, self.state = self.model.step(emb, self.state, retrieval=retrieval)
        logits = self.readout(emb, self.state, retrieval) if self.readout is not None else native
        self.table[self.rows, byte] = self.state.detach()
        self.seen[self.rows, byte] = True
        return logits


def forward(model, readout, x):
    stream = ByteStream(model, readout, x.shape[0], x.device)
    return torch.stack([stream.step(x[:, i]) for i in range(x.shape[1])], dim=1)


def sample(category, split, rng, index, pools, max_delay):
    if category == "memory":
        if split == "train" and pools.get("online_memory") and rng.random() < 0.5:
            return rng.choice(pools["online_memory"])
        return memory_example(split, index, max_delay)
    if category == "operations":
        return operation_example(split, index)
    return rng.choice(pools[category])


def available_pools(args, split):
    pools = {
        "language": load_examples(args.language, split, "language"),
        "conversation": load_examples(args.conversations, split, "conversation"),
        "stability": load_examples(args.stability, split, "stability"),
    }
    return pools


def choose_category(rng, pools):
    names = [name for name in MIX if name in ("memory", "operations") or pools[name]]
    return rng.choices(names, weights=[MIX[name] for name in names])[0]


def describe_data(args):
    train = available_pools(args, "train")
    val = available_pools(args, "val")
    active = {key: weight for key, weight in MIX.items()
              if key in ("memory", "operations") or train[key]}
    print("Training mix (renormalized):", json.dumps(
        {key: round(weight / sum(active.values()), 3) for key, weight in active.items()}))
    print("Optional data train/val:", json.dumps(
        {key: [len(train[key]), len(val[key])] for key in train}))
    return train


@torch.inference_mode()
def generate(model, readout, prompt, max_bytes, device):
    stream = ByteStream(model, readout, 1, device)
    logits = None
    for byte in prompt.encode("utf-8"):
        logits = stream.step(torch.tensor([byte], device=device))
    output = []
    for _ in range(max_bytes):
        byte = int(logits.argmax(dim=-1).item())
        output.append(byte)
        logits = stream.step(torch.tensor([byte], device=device))
    return bytes(output)


def show_dialogue(model, readout, example, max_bytes, device):
    """Show a training example and an independent RAP rollout, never a validation probe."""
    prompt = example.prompt
    if example.category == "memory" and len(prompt) > 320:
        prompt = prompt[:130] + " ... [context omitted] ... " + prompt[-130:]
    print(f"\n[Teacher challenge: {example.category}, delay={example.delay}]\n{prompt}", flush=True)
    source = "Qwen answer" if example.category == "conversation" else "Verified memory answer"
    print(f"{source}: {example.answer!r}", flush=True)
    reply = generate(model, readout, example.prompt, max_bytes, device)
    print(f"RAP greedy reply ({max_bytes} bytes): {reply.decode('utf-8', errors='replace')!r}\n", flush=True)


def degeneration(data):
    if not data:
        return 0.0
    tabs = data.count(b"\t") / len(data)
    repeated = max((data.count(bytes([byte])) for byte in set(data)), default=0) / len(data)
    return max(tabs, repeated)


@torch.inference_mode()
def evaluate(model, readout, args, device):
    model.eval()
    if readout:
        readout.eval()
    pools = available_pools(args, "val")
    groups = {"operations": [operation_example("val", i) for i in range(args.eval_examples)]}
    for delay in DELAYS["val"]:
        if delay > args.max_delay:
            continue
        rows = []
        for i in range(args.eval_examples * 40):
            example = memory_example("val", i, args.max_delay)
            if example.delay == delay:
                rows.append(example)
                if len(rows) == args.eval_examples:
                    break
        groups[f"memory_{delay}"] = rows
    for category, rows in pools.items():
        if rows:
            groups[category] = rows[:args.eval_examples]
    results = {}
    for category, rows in groups.items():
        total_loss = total_tokens = correct = exact = 0
        generations = []
        for example in rows:
            x, y, mask = batch([example], args.seq_len, device)
            # Only process occupied positions, not padding.
            length = len(encode(example, args.seq_len)[0]) - 1
            logits = forward(model, readout, x[:, :length])
            losses = F.cross_entropy(logits.reshape(-1, 256), y[:, :length].reshape(-1), reduction="none")
            active = mask[:, :length].reshape(-1).bool()
            predictions = logits.argmax(-1).reshape(-1)[active]
            targets = y[:, :length].reshape(-1)[active]
            total_loss += float(losses[active].sum())
            total_tokens += int(active.sum())
            correct += int((predictions == targets).sum())
            exact += int(torch.equal(predictions, targets))
            if (category.startswith("memory_") or category == "conversation") and len(generations) < args.rollouts:
                actual = generate(model, readout, example.prompt, min(args.rollout_bytes, 64), device)
                generations.append((actual, example.answer.encode("utf-8")))
        result = {"loss": total_loss / total_tokens, "byte_accuracy": correct / total_tokens,
                  "teacher_forced_exact": exact / len(rows), "examples": len(rows)}
        if generations:
            result["free_exact"] = sum(g.startswith(target) for g, target in generations) / len(generations)
            result["degeneration"] = sum(degeneration(g) for g, _ in generations) / len(generations)
        results[category] = result
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--base", default="rap_B_latest (1).pt")
    p.add_argument("--out", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--conversations", default="")
    p.add_argument("--language", default="")
    p.add_argument("--stability", default="")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-delay", type=int, default=256)
    p.add_argument("--eval-examples", type=int, default=8)
    p.add_argument("--rollouts", type=int, default=2)
    p.add_argument("--rollout-bytes", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--online-teacher", action="store_true",
                   help="Generate fresh Qwen challenges during training; not a matched-data A-D comparison")
    p.add_argument("--show-dialogue", action="store_true",
                   help="Print teacher training challenges and RAP sample replies at refreshes")
    p.add_argument("--teacher-every", type=int, default=50, help="Optimization steps between teacher batches")
    p.add_argument("--teacher-probes", type=int, default=2, help="Dev probes per delay, not final validation")
    p.add_argument("--teacher-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--teacher-device", default="cpu")
    p.add_argument("--teacher-max-tokens", type=int, default=96)
    add_flags(p)
    args = p.parse_args()
    validate_flags(args)
    if args.seq_len < args.max_delay + 300:
        raise ValueError("--seq-len must be at least --max-delay + 300 (no silent truncation)")
    if args.max_delay < 48:
        raise ValueError("--max-delay must be at least 48 to include held-out memory probes")
    if args.steps < 1 or args.batch_size < 1 or args.eval_examples < 1 or args.save_every < 1:
        raise ValueError("steps, batch-size, eval-examples and save-every must be positive")
    if args.rollouts < 0 or args.rollout_bytes < 1:
        raise ValueError("rollouts must be nonnegative and rollout-bytes positive")
    if args.online_teacher and (args.variant == "A" or args.eval_only):
        raise ValueError("Online teacher requires training variant B, C, or D")
    if args.show_dialogue and not args.online_teacher:
        raise ValueError("--show-dialogue requires --online-teacher")
    if min(args.teacher_every, args.teacher_probes, args.teacher_max_tokens) < 1:
        raise ValueError("teacher-every, teacher-probes and teacher-max-tokens must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = load_base(args.base, device)
    readout = Readout(model.cfg).to(device) if args.variant in ("B", "D") else None
    pools = describe_data(args)
    if args.variant == "B":
        for param in model.parameters():
            param.requires_grad_(False)
    if args.variant == "A":
        print(json.dumps(evaluate(model, None, args, device), indent=2))
        return
    params = [param for param in list(model.parameters()) + list(readout.parameters() if readout else [])
              if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    out = Path(args.out or f"rap_training_rebuild/outputs/{args.variant}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    teacher_log = out.with_suffix(".teacher.jsonl")
    start = 0
    trained_bytes = 0
    train_seconds = 0.0
    teacher_seconds = 0.0
    teacher_index = 0
    teacher_examples = []
    if args.resume or args.eval_only:
        saved = torch.load(out, map_location=device, weights_only=False)
        if saved["variant"] != args.variant or saved["base_step"] != 87199:
            raise ValueError("Resume checkpoint variant/base mismatch")
        model.load_state_dict(saved["model"])
        if readout:
            readout.load_state_dict(saved["readout"])
        if args.eval_only:
            print(json.dumps(evaluate(model, readout, args, device), indent=2))
            return
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        if bool(saved.get("online_teacher", False)) != args.online_teacher:
            raise ValueError("Resume must keep the online teacher setting")
        if args.online_teacher:
            fixed = ("base", "conversations", "language", "stability", "batch_size", "seq_len",
                     "max_delay", "seed", "lr", "teacher_every", "teacher_probes",
                     "teacher_model", "teacher_max_tokens")
            if any(saved["args"][key] != getattr(args, key) for key in fixed):
                raise ValueError("Online teacher resume requires unchanged data and curriculum settings")
        trained_bytes = saved.get("trained_answer_bytes", 0)
        train_seconds = saved.get("train_seconds", 0.0)
        teacher_seconds = saved.get("teacher_seconds", 0.0)
        rng.setstate(saved["rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if args.online_teacher:
            teacher_index = saved["teacher_index"]
            teacher_examples = load_teacher_examples(teacher_log)
            if len(teacher_examples) != saved["teacher_count"]:
                raise ValueError("Teacher log differs from saved checkpoint; use a new output path")
    elif args.online_teacher and teacher_log.exists():
        raise ValueError(f"Teacher log already exists at {teacher_log}; resume or choose another output")
    pools["online_memory"] = [row for row in teacher_examples if row.category == "memory"]
    pools["conversation"].extend(row for row in teacher_examples if row.category == "conversation")
    seen = {row.prompt for row in teacher_examples}
    seen.update(row.prompt for row in available_pools(args, "val")["conversation"])
    if args.show_dialogue and args.resume and teacher_examples:
        show_dialogue(model, readout, teacher_examples[-1], args.rollout_bytes, device)
    teacher = None
    for step in range(start, args.steps):
        refreshed = False
        fresh_examples = []
        if args.online_teacher and step % args.teacher_every == 0:
            teaching_started = time.perf_counter()
            if teacher is None:
                teacher = QwenTeacher(args.teacher_model, args.teacher_device, args.teacher_max_tokens)
            dev_delay, scores = weakest_delay(
                model, readout, args.max_delay, args.seq_len, device, args.teacher_probes)
            print(f"Teacher dev loss by delay: {scores}; targeting {dev_delay} bytes", flush=True)
            for focus in ("memory", "conversation"):
                example = teacher.challenge(
                    teacher_index, focus, args.max_delay, args.seq_len, seen, focus_delay=dev_delay)
                teacher_index += 1
                if example is None:
                    print(f"Teacher rejected all {focus} attempts; no sample added", flush=True)
                    continue
                append_teacher_example(teacher_log, example)
                if focus == "memory":
                    pools["online_memory"].append(example)
                else:
                    pools["conversation"].append(example)
                teacher_examples.append(example)
                fresh_examples.append(example)
                print(f"Teacher accepted {focus} example ({example.delay} byte delay)", flush=True)
            teacher_seconds += time.perf_counter() - teaching_started
            refreshed = True
        started = time.perf_counter()
        model.train()
        if readout:
            readout.train()
        examples = [sample(choose_category(rng, pools), "train", rng,
                           step * args.batch_size + i, pools, args.max_delay)
                    for i in range(args.batch_size)]
        x, y, mask = batch(examples, args.seq_len, device)
        length = max(len(encode(e, args.seq_len)[0]) - 1 for e in examples)
        logits = forward(model, readout, x[:, :length])
        losses = F.cross_entropy(logits.reshape(-1, 256), y[:, :length].reshape(-1), reduction="none")
        loss = (losses * mask[:, :length].reshape(-1)).sum() / mask[:, :length].sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        trained_bytes += int(mask.sum().item())
        train_seconds += time.perf_counter() - started
        if step == start or (step + 1) % 10 == 0:
            print(f"{args.variant} step {step+1}/{args.steps} loss {loss.item():.4f}", flush=True)
        if args.show_dialogue and fresh_examples:
            for example in fresh_examples:
                show_dialogue(model, readout, example, args.rollout_bytes, device)
        if refreshed or (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            torch.save({"variant": args.variant, "base_step": 87199, "step": step + 1,
                        "online_teacher": args.online_teacher, "teacher_index": teacher_index,
                        "teacher_count": len(teacher_examples), "teacher_seconds": teacher_seconds,
                        "trained_answer_bytes": trained_bytes, "train_seconds": train_seconds,
                        "model": model.state_dict(), "readout": readout.state_dict() if readout else None,
                        "optimizer": optimizer.state_dict(), "rng": rng.getstate(),
                        "torch_rng": torch.get_rng_state(), "args": vars(args)}, out)
            print(f"Saved {out}: {trained_bytes} target bytes, {train_seconds:.1f}s RAP, "
                  f"{teacher_seconds:.1f}s teacher", flush=True)
    print(json.dumps(evaluate(model, readout, args, device), indent=2))


if __name__ == "__main__":
    main()
