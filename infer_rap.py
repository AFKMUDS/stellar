import argparse
from dataclasses import dataclass

import torch

from raplib.data import decode_bytes, ids_to_bytes, strip_pos_tag_bytes
from raplib.model import ModelConfig, PrimitiveManifoldLM


@dataclass
class RunCfg:
    ckpt: str
    prompt: str
    max_new: int
    temperature: float
    device: str


def parse_args() -> RunCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/rap_v0.pt")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max_new", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    return RunCfg(ckpt=a.ckpt, prompt=a.prompt, max_new=a.max_new, temperature=a.temperature, device=a.device)


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["cfg"])

    model = PrimitiveManifoldLM(cfg)
    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        state_dict = ckpt.get("student_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint must contain either 'state_dict' or 'student_state_dict'.")
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    prompt_text = "Question:\n" + args.prompt + "\n\nAnswer:\n"
    prompt_bytes = prompt_text.encode("utf-8", errors="replace")
    ids = list(prompt_bytes)

    state = None
    with torch.no_grad():
        if len(ids) > 0:
            x = torch.tensor([ids], device=args.device, dtype=torch.long)
            out = model(x, state=None)
            state = out[1]

        last_id = ids[-1] if ids else 0
        for _ in range(args.max_new):
            x = torch.tensor([[last_id]], device=args.device, dtype=torch.long)
            out = model(x, state=state)
            logits, state = out[0], out[1]
            logits = logits[:, -1, :] / max(args.temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)
            last_id = next_id

    out_bytes = ids_to_bytes(ids)
    full_text = decode_bytes(out_bytes)
    answer_bytes = out_bytes[len(prompt_bytes):]
    answer_bytes = strip_pos_tag_bytes(answer_bytes)
    answer_text = decode_bytes(answer_bytes)

    print(prompt_text, end="")
    print(answer_text)

    if "\ufffd" in answer_text:
        print("\n[debug] answer contains invalid utf-8 bytes; raw bytes (latin-1 view) below:")
        print(answer_bytes.decode("latin-1", errors="replace"))


if __name__ == "__main__":
    main()
