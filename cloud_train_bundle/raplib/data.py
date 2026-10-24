import json
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple


@dataclass(frozen=True)
class Sample:
    prompt_bytes: bytes
    target_bytes: bytes

def _to_bytes(text: str) -> bytes:
    return text.encode("utf-8", errors="replace")


def iter_qa_samples(
    jsonl_path: str,
    include_answer_prefix: bool = True,
) -> Iterator[Sample]:
    answer_prefix = "\n\nAnswer:\n" if include_answer_prefix else "\n"
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question", "")
            a = obj.get("reference_answer", "")
            prompt = f"Question:\n{q}{answer_prefix}"
            yield Sample(prompt_bytes=_to_bytes(prompt), target_bytes=_to_bytes(a))


def iter_train_sequences(
    jsonl_path: str,
    max_prompt_bytes: int,
    max_target_bytes: int,
    max_total_bytes: int,
    include_answer_prefix: bool = True,
) -> Iterator[Tuple[bytes, int]]:
    for s in iter_qa_samples(jsonl_path=jsonl_path, include_answer_prefix=include_answer_prefix):
        prompt = s.prompt_bytes[:max_prompt_bytes] if max_prompt_bytes > 0 else s.prompt_bytes
        target = s.target_bytes[:max_target_bytes] if max_target_bytes > 0 else s.target_bytes
        full = prompt + target
        prompt_boundary = len(prompt)

        if max_total_bytes <= 0 or len(full) <= max_total_bytes:
            prompt_len = min(prompt_boundary, len(full))
            yield full, prompt_len
            continue

        start = 0
        while start < len(full):
            chunk = full[start : start + max_total_bytes]
            if not chunk:
                break
            if start >= prompt_boundary:
                prompt_len = 0
            else:
                prompt_len = min(prompt_boundary - start, len(chunk))
            yield chunk, prompt_len
            start += max_total_bytes


def iter_raw_byte_sequences(
    bin_path: str,
    max_total_bytes: int,
) -> Iterator[Tuple[bytes, int]]:
    if max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be > 0 for raw byte corpora")

    buf = bytearray()
    with open(bin_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            buf.extend(chunk)

            while len(buf) >= max_total_bytes:
                seq = bytes(buf[:max_total_bytes])
                del buf[:max_total_bytes]
                yield seq, 0

    if buf:
        yield bytes(buf), 0


def bytes_to_ids(b: bytes) -> Tuple[int, ...]:
    return tuple(b)


def ids_to_bytes(ids: Iterable[int]) -> bytes:
    return bytes(int(i) & 0xFF for i in ids)


def strip_pos_tag_bytes(b: bytes) -> bytes:
    return bytes(x for x in b if int(x) not in (128, 129, 130))


def decode_bytes(b: bytes) -> str:
    return b.decode("utf-8", errors="replace")
