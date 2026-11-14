"""Free-generation synthetic recall / reversible-byte probe on untouched validation rows."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from trainnewideas.rap.runtime import Session, choose_precision, load_model


def evaluate_probes(checkpoint, validation_file, *, device="cuda", count=12, max_new_bytes=32):
    model, cfg, _, selected = load_model(checkpoint, device)
    precision = choose_precision(cfg.train.precision, selected)
    groups = defaultdict(list)
    with Path(validation_file).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            messages = record.get("messages", [])
            if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
                continue
            prompt, answer = messages[0]["content"], messages[1]["content"]
            kind = ("memory" if "Remember " in prompt else
                    "byte_inverse" if "Bitwise invert " in prompt else "operation")
            if len(groups[kind]) >= count:
                continue
            session = Session(model, selected, precision=precision)
            session.prompt(prompt)
            raw, info = session.generate(max_bytes=max(max_new_bytes, len(answer.encode("utf-8"))),
                                         seed=cfg.train.seed, temperature=0)
            groups[kind].append({
                "prompt_bytes": len(prompt.encode("utf-8")),
                "expected": answer, "generated": raw.decode("utf-8", errors="replace"),
                "exact_prefix": raw.startswith(answer.encode("utf-8")),
                "ended_on_eos": info["ended_on_eos"],
            })
            if all(len(groups[name]) >= count for name in ("memory", "byte_inverse", "operation")):
                break
    return {kind: {"exact_prefix": sum(row["exact_prefix"] for row in rows) / len(rows),
                   "examples": rows} for kind, rows in groups.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--validation", required=True, help="mix/synthetic.validation.jsonl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--max-new-bytes", type=int, default=32)
    args = p.parse_args(argv)
    if args.count < 1 or args.max_new_bytes < 1:
        p.error("--count and --max-new-bytes must be positive")
    print(json.dumps(evaluate_probes(args.checkpoint, args.validation, device=args.device,
                                     count=args.count, max_new_bytes=args.max_new_bytes), indent=2))


if __name__ == "__main__":
    main()
