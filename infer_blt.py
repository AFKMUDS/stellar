from __future__ import annotations

import argparse

import torch

from raplib.blt_model import BLTConfig, ByteLatentTransformer
from raplib.data import decode_bytes, ids_to_bytes, strip_pos_tag_bytes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from the BLT prototype")
    p.add_argument("--ckpt", type=str, default="checkpoints/blt_proto.pt")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg_dict = ckpt.get("cfg", {})
    valid_keys = BLTConfig.__annotations__.keys()
    filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    cfg = BLTConfig(**filtered_cfg)
    model = ByteLatentTransformer(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    prompt_bytes = args.prompt.encode("utf-8", errors="replace")
    generated_ids = model.generate(
        list(prompt_bytes),
        int(args.max_new),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        device=device,
    )
    out_bytes = ids_to_bytes(generated_ids)
    out_bytes = strip_pos_tag_bytes(out_bytes)
    print(decode_bytes(out_bytes))


if __name__ == "__main__":
    main()
