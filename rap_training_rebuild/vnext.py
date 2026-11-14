"""Guarded original/fresh training launcher. Unsupported vNext features fail before training."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .flags import CHOICES, add_flags


ROOT = Path(__file__).resolve().parent.parent
FRESH = ROOT / "trainnewideas"


def validate(args):
    choices = {key: getattr(args, key) for key in CHOICES}
    core = choices["core"]
    if core is None:
        raise ValueError("--core original or --core fresh is required")
    defaults = {
        "original": dict(primitive_mode="original", encoding="original", head="original",
                         retrieval="original", memory_readout="off", state_persistence="record"),
        "fresh": dict(primitive_mode="fresh", encoding="bitviews", head="expression",
                      retrieval="ordered", memory_readout="joint", state_persistence="stream"),
    }
    resolved = {k: v if v is not None else defaults[core].get(k, "basic" if k == "diagnostics" else "off")
                for k, v in choices.items()}
    expected = dict(defaults[core])
    if core == "original" and resolved["head"] == "joint":
        expected.update(head="joint", memory_readout="joint")
    if core == "fresh":
        if resolved["head"] == "direct":
            expected["head"] = "direct"
        if resolved["retrieval"] in ("none", "random"):
            expected["retrieval"] = resolved["retrieval"]
            expected["memory_readout"] = "off" if resolved["retrieval"] == "none" else "joint"
    expected.update(expression_loss="off", phase_clock="off", route_regularization="off",
                    curriculum="off", diagnostics="basic")
    unsupported = [f"--{key.replace('_', '-')}={value} (supported: {expected[key]})"
                   for key, value in resolved.items() if key != "core" and value != expected[key]]
    if unsupported:
        raise ValueError("Incompatible or unimplemented feature flags: " + ", ".join(unsupported))
    if core == "original" and (args.resume or args.initialize_from):
        raise ValueError("Use the original pilot's --resume; fresh checkpoints cannot initialize the original core")
    if core == "original" and not args.base:
        raise ValueError("Original core requires --base pointing at the trusted step-87199 checkpoint")
    if core == "fresh" and args.base:
        raise ValueError("Fresh core does not accept original --base weights")
    return resolved


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_flags(p)
    p.add_argument("--dataset", nargs="+", help="Existing JSONL data (fresh: disk-backed; original: small pilot only)")
    p.add_argument("--mix", help="Validated, bounded source mixture from rap_training_rebuild.mix")
    p.add_argument("--base", help="Trusted original checkpoint, only for --core original")
    p.add_argument("--config", default=str(FRESH / "configs" / "cloud.json"))
    p.add_argument("--prepared", default=str(FRESH / "data" / "prepared"))
    p.add_argument("--run", default=str(FRESH / "runs" / "vnext"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--initialize-from", help="Fresh-format checkpoint for a new fresh training stage")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--chunk-length", type=int, default=128)
    p.add_argument("--gradient-accumulation", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1024, help="Original pilot only")
    p.add_argument("--max-delay", type=int, default=256, help="Original pilot only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("auto", "float32", "bf16", "fp16"), default="auto")
    p.add_argument("--evaluate-every", type=int, default=1)
    p.add_argument("--validation-target-tokens", type=int, default=64)
    p.add_argument("--sample-every", type=int, default=0)
    return p.parse_args(argv)


def original(args, flags):
    if args.mix:
        raise ValueError("The original eager-loading pilot does not support weighted --mix")
    if not args.dataset or len(args.dataset) != 1:
        raise ValueError("Original pilot requires exactly one small JSONL --dataset")
    path = Path(args.dataset[0])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Original pilot eagerly loads JSONL; use train_rap.py for the large original corpus")
    variant = "D" if flags["head"] == "joint" else "C"
    command = [sys.executable, "-m", "rap_training_rebuild.experiment",
               "--variant", variant, "--base", args.base, "--conversations", str(path),
               "--out", str(Path(args.run) / f"original_{variant}.pt"),
               "--steps", str(args.steps), "--batch-size", str(args.batch_size),
               "--seq-len", str(args.seq_len), "--max-delay", str(args.max_delay),
               "--device", args.device, "--eval-examples", "2", "--rollouts", "0"]
    subprocess.run(command, cwd=ROOT, check=True)


def fresh(args, flags):
    from trainnewideas.rap.config import as_dict, from_dict, load_config
    from trainnewideas.rap.data import prepare_dataset, sha256_file
    from trainnewideas.rap.engine import train
    from trainnewideas.rap.runtime import load_payload

    mapping = {"direct": "direct", "expression": "reversible"}
    overrides = {
        "model.expression_head": mapping[flags["head"]],
        "model.retrieval": {"ordered": "tagged", "none": "none", "random": "random"}[flags["retrieval"]],
        "model.byte_operator_seeds": True,
        "train.steps": args.steps,
        "train.batch_size": args.batch_size,
        "train.chunk_length": args.chunk_length,
        "train.gradient_accumulation": args.gradient_accumulation,
        "train.device": args.device,
        "train.precision": args.precision,
        "train.evaluate_every": args.evaluate_every,
        "train.validation_target_tokens": args.validation_target_tokens,
        "train.sample_every": args.sample_every,
        # Identical primary objective for the first controlled comparisons.
        "train.future_weight": 0.0, "train.past_weight": 0.0,
        "train.boundary_weight": 0.0, "train.stop_weight": 0.0,
    }
    mix = None
    if args.mix:
        if args.dataset:
            raise ValueError("Choose --dataset or --mix, not both")
        directory = Path(args.mix).resolve()
        mix = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if mix.get("format") != "rap-weighted-mix-v1":
            raise ValueError("Unsupported weighted mixture manifest")
        for name, digest in mix["files_sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Mix source checksum mismatch: {name}")
        overrides["data.source_weights"] = mix["weights"]
    config = load_config(args.config, overrides)
    latest = Path(args.run) / "latest.pt"
    if args.resume and not latest.exists():
        raise FileNotFoundError(f"Resume requested but no fresh checkpoint at {latest}")
    source = args.initialize_from
    if source and args.resume:
        raise ValueError("Choose --resume or --initialize-from")
    if source:
        if Path(source).resolve().parent == Path(args.run).resolve():
            raise ValueError("New-stage --run must differ from the source checkpoint directory")
        if Path(args.run).exists() and any(Path(args.run).iterdir()):
            raise ValueError("New-stage --run must be an empty/new directory; never overwrite another run")
        previous = from_dict(load_payload(source)["config"])
        if as_dict(previous)["model"] != as_dict(config)["model"]:
            raise ValueError("Fresh new-stage initialization requires identical model architecture")
    prepared = Path(args.prepared)
    if mix:
        names = list(mix["weights"])
        train_files = [directory / f"{name}.train.jsonl" for name in names]
        val_files = [directory / f"{name}.validation.jsonl" for name in names]
        prepared = prepare_dataset(train_files, prepared, config.data, validation_path=val_files)
    elif args.dataset:
        prepared = prepare_dataset(args.dataset, prepared, config.data)
    elif not (prepared / "manifest.json").exists():
        raise ValueError("Pass --dataset for first preparation or an existing --prepared")
    if mix and (prepared / "manifest.json").exists():
        recorded = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        if recorded["data_config"]["source_weights"] != mix["weights"]:
            raise ValueError("Prepared source weights do not match the mixture")
    path, _ = train(config, prepared, args.run, resume="auto" if args.resume else None,
                    initialize=source)
    print("Fresh checkpoint:", path)


def main(argv=None):
    args = parse_args(argv)
    flags = validate(args)
    print("vNext resolved flags:", json.dumps(flags, sort_keys=True), flush=True)
    if flags["core"] == "original":
        original(args, flags)
    else:
        fresh(args, flags)


if __name__ == "__main__":
    main()
