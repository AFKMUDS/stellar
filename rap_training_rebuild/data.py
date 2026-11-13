"""Deterministic, disjoint synthetic probes and Qwen-generated conversation loader."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str
    category: str
    delay: int = 0


KEYS = {
    "train": ("animal", "color", "place", "tool"),
    "dev": ("season", "metal", "vehicle", "fruit"),
    "val": ("instrument", "fabric", "planet", "flower"),
}
VALUES = {
    "train": ("otter", "copper", "meadow", "ladder", "sparrow", "indigo"),
    "dev": ("autumn", "silver", "canoe", "peach", "winter", "bronze"),
    "val": ("cello", "linen", "saturn", "iris", "violin", "velvet"),
}
DELAYS = {
    "train": (32, 64, 128, 256, 1024, 4096),
    "dev": (40, 80, 160, 384, 768, 2048),
    "val": (48, 96, 192, 512, 1536, 4096),
}


def memory_example(split: str, index: int, max_delay: int = 4096, delay_override: int | None = None) -> Example:
    if split not in KEYS:
        raise ValueError(split)
    rng = random.Random(13007 + index * 7717 + {"train": 0, "dev": 2, "val": 1}[split])
    key = rng.choice(KEYS[split])
    value = rng.choice(VALUES[split])
    choices = [d for d in DELAYS[split] if d <= max_delay]
    if delay_override is not None and delay_override not in choices:
        raise ValueError("Requested delay is outside the training curriculum")
    delay = delay_override if delay_override is not None else rng.choice(choices)
    filler = (" the river flows gently." * (delay // 24 + 2))[:delay]
    mode = index % 4
    prefix = f"User: Remember {key} = {value}.\n"
    if mode == 1:
        replacement = rng.choice([v for v in VALUES[split] if v != value])
        prefix += f"User: Update {key} = {replacement}.\n"
        value = replacement
    if mode == 2:
        other = rng.choice([k for k in KEYS[split] if k != key])
        prefix += f"User: Remember {other} = {rng.choice(VALUES[split])}.\n"
    if mode == 3:
        query = rng.choice([k for k in KEYS[split] if k != key])
        value = "unknown"
    else:
        query = key
    prompt = prefix + f"Context: {filler}\nUser: What is {query}? Answer with one word.\nAssistant: "
    return Example(prompt, value, "memory", delay)


def operation_example(split: str, index: int) -> Example:
    rng = random.Random(index * 313 + (0 if split == "train" else 19))
    alphabet = "abcdefghijklm" if split == "train" else "nopqrstuvwxyz"
    text = "".join(rng.sample(alphabet, 4))
    op = ("copy", "reverse", "uppercase")[index % 3]
    answer = {"copy": text, "reverse": text[::-1], "uppercase": text.upper()}[op]
    return Example(f"User: {op} the string {text}. Output only the result.\nAssistant: ", answer, "operations")


def load_examples(path: str, split: str, category: str) -> list[Example]:
    """Split by stable prompt hash so no prompt appears in both sets."""
    if not path:
        return []
    examples = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            answer = row.get("answer")
            if prompt is None and "question" in row:
                prompt = "Question:\n" + row["question"] + "\n\nAnswer:\n"
                answer = row.get("reference_answer")
            if prompt is None and "instruction" in row:
                prompt = "User: " + row["instruction"] + "\nAssistant: "
                answer = row.get("output")
            if prompt is None and "text" in row:
                text = row["text"]
                if isinstance(text, str) and len(text.encode("utf-8")) >= 2:
                    prompt, answer = text[0], text[1:]
            if not isinstance(prompt, str) or not isinstance(answer, str) or not answer:
                raise ValueError(f"Expected nonempty prompt/answer strings in {path}")
            group = row.get("group", prompt)
            if not isinstance(group, str):
                raise ValueError("group must be a string")
            bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest(), 16) % 10
            if (bucket == 0) == (split == "val"):
                examples.append(Example(prompt, answer, category))
    return examples


def encode(example: Example, seq_len: int) -> tuple[bytes, int]:
    prompt = example.prompt.encode("utf-8")
    answer = example.answer.encode("utf-8")
    if len(prompt) + len(answer) > seq_len + 1:
        raise ValueError(f"{example.category} sample exceeds seq_len={seq_len}; increase --seq-len")
    return prompt + answer, len(prompt)


def batch(examples: list[Example], seq_len: int, device):
    import torch

    x = torch.zeros(len(examples), seq_len, device=device, dtype=torch.long)
    y = torch.zeros_like(x)
    mask = torch.zeros(len(examples), seq_len, device=device)
    for row, example in enumerate(examples):
        raw, boundary = encode(example, seq_len)
        n = len(raw) - 1
        x[row, :n] = torch.tensor(list(raw[:-1]), device=device)
        y[row, :n] = torch.tensor(list(raw[1:]), device=device)
        mask[row, boundary - 1:n] = 1
    return x, y, mask
