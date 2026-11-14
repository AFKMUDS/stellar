"""Compare compatible fresh checkpoints on the exact same held-out mixed corpus."""

import argparse
import json
from pathlib import Path

from trainnewideas.rap.data import Corpus
from trainnewideas.rap.evaluation import evaluate
from trainnewideas.rap.model import parameter_count
from trainnewideas.rap.runtime import Session, choose_precision, load_model


def compare(paths, prepared, *, budget=256, device="cuda", prompt="What is gravity?"):
    corpus = Corpus(prepared, "validation")
    result = {}
    for path in paths:
        model, cfg, saved, selected_device = load_model(path, device)
        precision = choose_precision(cfg.train.precision, selected_device)
        scores = {
            source: evaluate(model, Corpus(prepared, "validation", source=source),
                             cfg.train, selected_device, precision, target_budget=budget)
            for source in corpus.manifest.get("source_names", [])
        }
        session = Session(model, selected_device, precision=precision)
        session.prompt(prompt)
        response, _ = session.generate(max_bytes=64, seed=cfg.train.seed, temperature=0)
        result[str(path)] = dict(
            step=saved["step"], target_tokens=saved.get("target_tokens"),
            target_raw_bytes=saved.get("metrics", {}).get("target_raw_bytes"),
            parameters=parameter_count(model),
            source_scores=scores,
            generated=response.decode("utf-8", errors="replace"),
            generated_hex=response.hex(),
            supervised_bytes_per_second=saved.get("metrics", {}).get("supervised_bytes_per_second"),
            peak_vram_bytes=saved.get("metrics", {}).get("peak_vram_bytes"),
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs=2, required=True, help="Baseline and new-stage fresh checkpoints")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-target-tokens", type=int, default=256)
    parser.add_argument("--prompt", default="What is gravity?")
    args = parser.parse_args(argv)
    print(json.dumps(compare([Path(p) for p in args.checkpoints], args.prepared,
                             budget=args.validation_target_tokens, device=args.device,
                             prompt=args.prompt), indent=2))


if __name__ == "__main__":
    main()
