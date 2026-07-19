import argparse
import json
import os
from typing import Any

from datasets import load_dataset
from tqdm import tqdm


USER_ROLES = {"user", "human", "prompt"}
ASSISTANT_ROLES = {"assistant", "gpt", "bot", "model", "assistant_response"}


def _stringify_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, list):
        parts = []
        for item in x:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content") or item.get("value") or ""
                if txt:
                    parts.append(str(txt).strip())
            else:
                parts.append(str(item).strip())
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(x, dict):
        txt = x.get("text") or x.get("content") or x.get("value") or ""
        return str(txt).strip()
    return str(x).strip()


def _extract_pair_from_messages(messages: list[Any]) -> tuple[str, str] | None:
    cleaned: list[tuple[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or msg.get("from") or msg.get("speaker") or "").strip().lower()
        content = _stringify_text(
            msg.get("content") or msg.get("text") or msg.get("value") or msg.get("utterance") or ""
        )
        if role and content:
            cleaned.append((role, content))

    for i in range(len(cleaned) - 1):
        role_a, text_a = cleaned[i]
        role_b, text_b = cleaned[i + 1]
        if role_a in USER_ROLES and role_b in ASSISTANT_ROLES and text_a and text_b:
            return text_a, text_b
    return None


def _extract_pair(example: dict[str, Any]) -> tuple[str, str] | None:
    for key in ("messages", "conversation", "conversations"):
        value = example.get(key)
        if isinstance(value, list):
            pair = _extract_pair_from_messages(value)
            if pair is not None:
                return pair

    prompt = _stringify_text(example.get("question") or example.get("prompt") or example.get("instruction") or example.get("input"))
    answer = _stringify_text(example.get("reference_answer") or example.get("response") or example.get("answer") or example.get("output") or example.get("completion"))
    if prompt and answer:
        return prompt, answer
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="ultrachat_train.jsonl")
    ap.add_argument("--dataset", type=str, default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--split", type=str, default="train_sft")
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    load_kwargs = {}
    if str(args.config).strip():
        ds = load_dataset(args.dataset, args.config, split=args.split, **load_kwargs)
    else:
        ds = load_dataset(args.dataset, split=args.split, **load_kwargs)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    total = 0
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        iterator = ds
        if int(args.max_samples) > 0:
            iterator = ds.select(range(min(int(args.max_samples), len(ds))))
        for ex in tqdm(iterator, desc="Converting UltraChat"):
            total += 1
            pair = _extract_pair(ex)
            if pair is None:
                continue
            question, reference_answer = pair
            question = question.strip()
            reference_answer = reference_answer.strip()
            if not question or not reference_answer:
                continue
            rec = {
                "question": question,
                "reference_answer": reference_answer,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} / {total} samples to {args.out}")


if __name__ == "__main__":
    main()
