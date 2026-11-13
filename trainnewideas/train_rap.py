"""Prepare a dataset and train a fresh RAP, or resume this folder's own checkpoint."""
import argparse
import json
from pathlib import Path
from rap.config import load_config
from rap.data import prepare_dataset, SUFFIXES
from rap.engine import train

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT/"configs/laptop.json"))
    p.add_argument("--dataset", nargs="+", help="JSONL, messages JSON, text, optional Parquet, or dataset folder(s)")
    p.add_argument("--validation", help="Optional separate validation dataset")
    p.add_argument("--prepared", default=str(ROOT/"data/prepared"))
    p.add_argument("--run", default=str(ROOT/"runs/rap"))
    p.add_argument("--resume", nargs="?", const="auto", help="'auto' or a checkpoint from this new trainer")
    p.add_argument('--initialize-from', help='Start a new stage from this trainer\'s weights; reset optimizer, carrier and data cursors.')
    p.add_argument("--force-prepare", action="store_true")
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--chunk-length", type=int)
    p.add_argument("--gradient-accumulation", type=int)
    p.add_argument("--device")
    p.add_argument("--precision", choices=["auto", "float32", "bf16", "fp16"])
    p.add_argument("--head", choices=["reversible", "direct", "random"])
    p.add_argument("--retrieval", choices=["tagged", "mean", "none", "random"])
    args = p.parse_args()
    overrides = {}
    for name in ["steps", "batch_size", "chunk_length", "gradient_accumulation", "device", "precision"]:
        if getattr(args, name) is not None:
            overrides["train."+name] = getattr(args, name)
    if args.head: overrides["model.expression_head"] = args.head
    if args.retrieval: overrides["model.retrieval"] = args.retrieval
    config = load_config(args.config, overrides)
    dataset = args.dataset
    if not dataset and not (Path(args.prepared)/'manifest.json').exists() and (ROOT/'data').is_dir():
        dataset = sorted(str(x) for x in (ROOT/"data").iterdir()
                         if x.is_file() and str(x).lower().endswith(SUFFIXES))
    if dataset:
        prepared = prepare_dataset(dataset, args.prepared, config.data,
            validation_path=args.validation, force=args.force_prepare)
    elif (Path(args.prepared)/"manifest.json").exists():
        prepared = Path(args.prepared)
    else:
        p.error("Put a dataset file in data/ or pass --dataset FILE. No prepared corpus was found.")
    checkpoint, _ = train(config, prepared, args.run, resume=args.resume, initialize=args.initialize_from)
    print("Saved:", checkpoint, flush=True)

if __name__ == "__main__":
    main()
