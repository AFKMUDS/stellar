import argparse
import os
from typing import Any

from datasets import load_dataset
from tqdm import tqdm



def _extract_text(example: dict[str, Any]) -> str:
    for key in ("text", "content", "sentence", "document", "story"):
        value = example.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    for value in example.values():
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="bookcorpus_clean.bin")
    ap.add_argument("--dataset", type=str, default="kd13/bookcorpus-clean")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--separator", type=str, default="\n\n")
    args = ap.parse_args()

    if str(args.config).strip():
        ds = load_dataset(args.dataset, args.config, split=args.split)
    else:
        ds = load_dataset(args.dataset, split=args.split)

    iterator = ds
    if int(args.max_samples) > 0:
        iterator = ds.select(range(min(int(args.max_samples), len(ds))))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    kept = 0
    total = 0
    sep = str(args.separator).encode("utf-8", errors="replace")
    with open(args.out, "wb") as f:
        for ex in tqdm(iterator, desc="Converting BookCorpus"):
            total += 1
            text = _extract_text(ex)
            if not text:
                continue
            f.write(text.encode("utf-8", errors="replace"))
            f.write(sep)
            kept += 1

    print(f"Wrote {kept} / {total} documents to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
