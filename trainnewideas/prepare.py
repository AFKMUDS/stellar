"""Build a disk-backed, document-split corpus without starting training."""
import argparse
from pathlib import Path
from rap.config import load_config
from rap.data import prepare_dataset

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", nargs="+", required=True)
    p.add_argument("--validation")
    p.add_argument("--config", default=str(ROOT/"configs/laptop.json"))
    p.add_argument("--prepared", default=str(ROOT/"data/prepared"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    print(prepare_dataset(args.dataset, args.prepared, cfg.data,
                           validation_path=args.validation, force=args.force))

if __name__ == "__main__":
    main()

