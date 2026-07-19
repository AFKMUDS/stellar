import argparse
from dataclasses import dataclass

import torch

from raplib.data import decode_bytes, ids_to_bytes, strip_pos_tag_bytes
from raplib.model import ModelConfig, PrimitiveManifoldLM


@dataclass
class RunCfg:
    ckpt: str
    device: str
    turns: int
    max_new: int
    temperature: float
    top_k: int
    speaker_a: str
    speaker_b: str


def parse_args() -> RunCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints1/b.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--turns", type=int, default=8)
    p.add_argument("--max_new", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--speaker_a", type=str, default="A")
    p.add_argument("--speaker_b", type=str, default="B")
    a = p.parse_args()
    return RunCfg(
        ckpt=a.ckpt,
        device=a.device,
        turns=a.turns,
        max_new=a.max_new,
        temperature=a.temperature,
        top_k=a.top_k,
        speaker_a=a.speaker_a,
        speaker_b=a.speaker_b,
    )


def sample_next_id(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(float(temperature), 1e-6)
    if int(top_k) > 0 and int(top_k) < logits.shape[-1]:
        vals, idx = torch.topk(logits, k=int(top_k), dim=-1)
        probs = torch.softmax(vals, dim=-1)
        pick = torch.multinomial(probs, num_samples=1)
        return int(idx.gather(-1, pick).item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_turn(
    model: PrimitiveManifoldLM,
    state: torch.Tensor,
    last_id: int,
    *,
    max_new: int,
    temperature: float,
    top_k: int,
) -> tuple[list[int], torch.Tensor, int]:
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(int(max_new)):
            x = torch.tensor([[int(last_id)]], device=state.device, dtype=torch.long)
            out = model(x, state=state)
            logits, state = out[0], out[1]
            next_id = sample_next_id(logits[:, -1, :], temperature=float(temperature), top_k=int(top_k))
            generated.append(int(next_id))
            last_id = int(next_id)
            if next_id == 10 and len(generated) >= 8:
                break
    return generated, state, last_id


def decode_turn(ids: list[int]) -> str:
    raw = ids_to_bytes(ids)
    raw = strip_pos_tag_bytes(raw)
    return decode_bytes(raw).strip()



def prime_from_prefix(model: PrimitiveManifoldLM, prefix: bytes, device: str) -> tuple[torch.Tensor, int]:
    state = model.init_state(batch_size=1, device=torch.device(device))
    ids = list(prefix)
    last_id = 10
    if not ids:
        return state, last_id
    with torch.no_grad():
        for tok in ids:
            x = torch.tensor([[int(tok)]], device=device, dtype=torch.long)
            out = model(x, state=state)
            _logits, state = out[0], out[1]
            last_id = int(tok)
    return state, last_id



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

    transcript = b""
    state, last_id = prime_from_prefix(model, transcript, args.device)

    speakers = [args.speaker_a, args.speaker_b]
    print("RAP self-chat starting...\n")
    for turn_idx in range(int(args.turns)):
        speaker = speakers[turn_idx % 2]
        prefix = f"{speaker}: ".encode("utf-8", errors="replace")
        with torch.no_grad():
            for tok in prefix:
                x = torch.tensor([[int(tok)]], device=args.device, dtype=torch.long)
                out = model(x, state=state)
                _logits, state = out[0], out[1]
                last_id = int(tok)
        gen_ids, state, last_id = generate_turn(
            model,
            state,
            last_id,
            max_new=int(args.max_new),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
        )
        line = decode_turn(gen_ids)
        print(f"{speaker}: {line}")
        transcript += prefix + ids_to_bytes(gen_ids) + b"\n"
        with torch.no_grad():
            x = torch.tensor([[10]], device=args.device, dtype=torch.long)
            out = model(x, state=state)
            _logits, state = out[0], out[1]
            last_id = 10


if __name__ == "__main__":
    main()
