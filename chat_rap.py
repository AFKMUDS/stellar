import argparse

import torch

from raplib.data import decode_bytes, ids_to_bytes, strip_pos_tag_bytes
from raplib.model import ModelConfig, PrimitiveManifoldLM


def _generate(model: PrimitiveManifoldLM, prompt_text: str, max_new: int, temperature: float, device: str) -> str:
    prompt_bytes = prompt_text.encode("utf-8", errors="replace")
    ids = list(prompt_bytes)

    state = None
    with torch.no_grad():
        if ids:
            x = torch.tensor([ids], device=device, dtype=torch.long)
            _, state = model(x, state=None)

        last_id = ids[-1] if ids else 0
        for _ in range(max_new):
            x = torch.tensor([[last_id]], device=device, dtype=torch.long)
            logits, state = model(x, state=state)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)
            last_id = next_id

    out_bytes = ids_to_bytes(ids)
    answer_bytes = out_bytes[len(prompt_bytes) :]
    answer_bytes = strip_pos_tag_bytes(answer_bytes)
    return decode_bytes(answer_bytes).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/rap_v0.pt")
    p.add_argument("--max_new", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--system", type=str, default="You are RAP-v0. Answer concisely.")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["cfg"])

    model = PrimitiveManifoldLM(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    history = []

    print("RAP chat. Type /reset to clear, /quit to exit.")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user.lower() in {"/q", "/quit", "quit", "exit"}:
            break
        if user.lower() in {"/r", "/reset", "reset"}:
            history = []
            print("[reset]")
            continue

        history.append(("user", user))

        convo = args.system + "\n"
        for role, msg in history[-8:]:
            if role == "user":
                convo += f"User: {msg}\n"
            else:
                convo += f"Assistant: {msg}\n"

        prompt_text = "Question:\n" + convo + "\n\nAnswer:\n"
        ans = _generate(model, prompt_text, max_new=args.max_new, temperature=args.temperature, device=args.device)

        history.append(("assistant", ans))
        print(f"rap> {ans}\n")


if __name__ == "__main__":
    main()
