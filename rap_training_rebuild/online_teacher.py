"""Optional bounded, adaptive local Qwen teacher; never writes evaluation data."""

import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import DELAYS, KEYS, VALUES, Example, batch, encode, memory_example


TOPICS = ("weather", "gardening", "music", "baking", "travel", "photography",
          "reading", "architecture", "the ocean", "mountains", "astronomy", "crafts")
BASE_QUESTIONS = (
    "Why does rain form in clouds?",
    "What helps a tomato plant grow well?",
    "What is musical tempo?",
    "Why does bread dough rise?",
    "What should someone check before a train trip?",
    "How does light affect a photograph?",
    "How can a reader find a book's main idea?",
    "What helps a building stay stable?",
    "Why do ocean tides change?",
    "How do mountains form?",
    "Why does the moon appear to change shape?",
    "How can glue be used safely in a paper craft?",
)
FORBIDDEN = (set(KEYS["train"] + KEYS["dev"] + KEYS["val"]
                 + VALUES["train"] + VALUES["dev"] + VALUES["val"])
             | {"remember", "update", "unknown"})


def acceptable(text, min_chars=16, max_chars=280):
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    return (min_chars <= len(text) <= max_chars
            and not words.intersection(FORBIDDEN)
            and not any(x in lowered for x in ("assistant:", "user:", "<think>", "```", "\t", "\x00"))
            and len(set(words)) >= 4)


def complete_sentence(text, terminators=".!?"):
    return bool(text and text.rstrip().endswith(tuple(terminators)))


def objective_question(text):
    return not re.search(
        r"\b(best|favorite|favourite|you|your|i|me|my|most|like|recently|ever|prefer)\b",
        text.lower(),
    )


def non_personal_answer(text):
    return not re.search(r"\b(i|me|my|we|our)\b", text.lower())


def train_bucket(group):
    return int(hashlib.sha256(group.encode("utf-8")).hexdigest(), 16) % 10 != 0


def target_train_delay(dev_delay, max_delay):
    if dev_delay is None:
        raise ValueError("Development delay is required")
    return min((d for d in DELAYS["train"] if dev_delay <= d <= max_delay),
               default=max(d for d in DELAYS["train"] if d <= max_delay))


@torch.inference_mode()
def weakest_delay(model, readout, max_delay, seq_len, device, probes=2):
    """Only development probes influence the curriculum; final val is untouched."""
    from .experiment import forward

    model.eval()
    if readout is not None:
        readout.eval()
    scores = {}
    for delay in DELAYS["dev"]:
        if delay > max_delay:
            continue
        rows = []
        for index in range(200):
            example = memory_example("dev", index, max_delay)
            if example.delay == delay:
                rows.append(example)
                if len(rows) == probes:
                    break
        losses = []
        for example in rows:
            x, y, mask = batch([example], seq_len, device)
            length = len(encode(example, seq_len)[0]) - 1
            logits = forward(model, readout, x[:, :length])
            loss = F.cross_entropy(logits.reshape(-1, 256), y[:, :length].reshape(-1), reduction="none")
            losses.append(float((loss * mask[:, :length].reshape(-1)).sum() / mask.sum()))
        if losses:
            scores[delay] = sum(losses) / len(losses)
    if not scores:
        raise ValueError("No development probes fit --max-delay")
    return max(scores, key=scores.get), scores


class QwenTeacher:
    def __init__(self, name="Qwen/Qwen3-0.6B", device="cpu", max_tokens=64):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True).to(device).eval()

    @torch.inference_mode()
    def ask(self, instruction):
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
        ).to(self.device)
        input_ids = encoded["input_ids"] if hasattr(encoded, "get") else encoded
        attention_mask = encoded.get("attention_mask") if hasattr(encoded, "get") else None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        result = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            do_sample=False, max_new_tokens=self.max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(result[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

    def challenge(self, index, focus, max_delay, seq_len, seen, focus_delay=None):
        """Memory answers come from verified bindings; conversation answers are filtered Qwen text."""
        for attempt in range(3):
            topic_index = (index + attempt) % len(TOPICS)
            topic = TOPICS[topic_index]
            if focus == "memory":
                delay = target_train_delay(focus_delay, max_delay)
                source = memory_example("train", 100000 + index * 3 + attempt,
                                        max_delay, delay_override=delay)
                context = self.ask(
                    f"Write one neutral sentence of at most 20 words about {topic}. "
                    "No names, numbers, role labels, stored facts, questions, or code. Sentence only."
                )
                context = context.encode("ascii", errors="ignore").decode("ascii")
                if not acceptable(context) or not complete_sentence(context):
                    continue
                filler = ((" " + context) * (delay // (len(context) + 1) + 2))[:delay]
                prefix, suffix = source.prompt.split("Context: ", 1)
                _, query = suffix.split("\nUser: What is ", 1)
                example = Example(prefix + "Context: " + filler + "\nUser: What is " + query,
                                  source.answer, "memory", delay)
            else:
                audience = ("a beginner", "a neighbor", "a student", "a visitor", "a parent",
                            "a coworker", "a hobbyist", "a traveler")[((index // len(TOPICS)) + attempt) % 8]
                seed = BASE_QUESTIONS[topic_index]
                question = self.ask(
                    f"Rewrite this factual question in plain language for {audience}. "
                    f"Keep the same topic, ask for no personal opinions, output only a question: {seed}"
                )
                if (not acceptable(question, min_chars=18, max_chars=160)
                        or not complete_sentence(question, "?") or not objective_question(question)
                        or not set(re.findall(r"[a-z]+", topic)).intersection(
                            re.findall(r"[a-z]+", question.lower()))):
                    question = seed
                prompt = f"User: {question}\nAssistant: "
                answer = self.ask(
                    f"Answer this user question accurately in at most 35 words and end with a period: {question}"
                )
                if not acceptable(answer) or not complete_sentence(answer) or not non_personal_answer(answer):
                    continue
                example = Example(prompt, answer, "conversation")
            group = example.prompt
            if group in seen or not train_bucket(group):
                continue
            try:
                encode(example, seq_len)
            except ValueError:
                continue
            seen.add(group)
            return example
        return None


def load_teacher_examples(path):
    rows = []
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                rows.append(Example(**row))
    return rows


def append_teacher_example(path, example):
    from dataclasses import asdict

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
