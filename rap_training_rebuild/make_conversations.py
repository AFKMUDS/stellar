"""Generate supervised practice conversations with local Qwen3-0.6B."""

import argparse
import json
from pathlib import Path

import torch


SEEDS = (
    "Explain why a shadow changes length during the day in two short sentences.",
    "A user asks for a recipe without eggs. Suggest a simple breakfast.",
    "A user says they are learning Python. Explain a for loop with an example.",
    "A user asks to compare a bicycle and a bus for a two-mile commute.",
    "A user says their pet's name is Miso. Ask a follow-up using that name.",
    "A user says they prefer concise answers. Describe evaporation briefly.",
    "A user asks for help writing a polite thank-you message.",
    "A user asks how to check whether a source is trustworthy.",
    "A user wants three ideas for a rainy afternoon at home.",
    "A user is confused by fractions. Explain one half plus one quarter.",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--out", default="rap_training_rebuild/conversations.jsonl")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True).to(args.device).eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for seed in SEEDS:
            messages = [{"role": "user", "content": seed}]
            transcript = ""
            for turn in range(2):
                encoded = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    enable_thinking=False, return_tensors="pt",
                ).to(args.device)
                input_ids = encoded["input_ids"] if hasattr(encoded, "get") else encoded
                attention_mask = encoded.get("attention_mask") if hasattr(encoded, "get") else None
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                result = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                answer = tokenizer.decode(result[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
                if not answer:
                    break
                user_text = messages[-1]["content"]
                # Group both turns into one split, and train RAP on plain UTF-8 role markers.
                prompt = transcript + f"User: {user_text}\nAssistant: "
                row = {"prompt": prompt, "answer": answer, "teacher": args.model, "group": seed}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                transcript = prompt + answer + "\n"
                messages.extend(({"role": "assistant", "content": answer},
                                 {"role": "user", "content": "Can you restate the main point more briefly?"}))
    print(f"Wrote practice examples to {out}")


if __name__ == "__main__":
    main()
