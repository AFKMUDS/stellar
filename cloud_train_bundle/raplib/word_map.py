from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Tuple

import torch


def _token_to_byte(tok: str) -> int:
    tok = tok.strip()
    if not tok:
        raise ValueError("empty token")

    if tok.startswith("0x") or tok.startswith("0X"):
        v = int(tok, 16)
        if not (0 <= v <= 255):
            raise ValueError(f"hex token out of range: {tok}")
        return int(v)

    # Control pictures: U+2400..U+241F map to 0x00..0x1F
    if len(tok) == 1:
        cp = ord(tok)
        if 0x2400 <= cp <= 0x241F:
            return int(cp - 0x2400)
        # U+2421 '␡' is DEL (0x7F)
        if cp == 0x2421:
            return 0x7F
        return int(cp) & 0xFF

    raise ValueError(f"unrecognized token form: {tok!r}")


def iter_word_chains_bytes(path: str) -> Iterable[Tuple[str, list[int]]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            if "\t" not in line:
                continue
            lemma, rhs = line.split("\t", 1)
            rhs = rhs.strip()
            if not rhs:
                continue
            toks = [t for t in rhs.split(" ") if t]
            seq: list[int] = []
            ok = True
            for t in toks:
                try:
                    seq.append(_token_to_byte(t))
                except Exception:
                    ok = False
                    break
            if not ok or len(seq) < 2:
                continue
            yield lemma, seq


def build_word_map_tables(
    *,
    path: str,
    topk: int = 32,
    device: torch.device | None = None,
    min_count: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build immediate-transition adjacency tables from word_chains.txt.

    For each chain c0,c1,...,cL, we add directed edges c_i -> c_{i+1}.

    Returns:
      - neigh: (256, K) long tensor of neighbor byte IDs (padded with -1)
      - w:     (256, K) float tensor of normalized weights per neighbor
    """

    k = int(topk)
    if k <= 0:
        raise ValueError("topk must be >= 1")

    counters = [Counter() for _ in range(256)]

    for _lemma, seq in iter_word_chains_bytes(path):
        for a, b in zip(seq, seq[1:]):
            a = int(a) & 0xFF
            b = int(b) & 0xFF
            counters[a][b] += 1

    neigh = torch.full((256, k), -1, dtype=torch.long, device=device)
    w = torch.zeros((256, k), dtype=torch.float32, device=device)

    for src in range(256):
        items = [(dst, cnt) for dst, cnt in counters[src].items() if int(cnt) >= int(min_count)]
        if not items:
            continue

        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:k]

        dsts = [int(d) for d, _c in items]
        cnts = [int(c) for _d, c in items]

        # Smooth + normalize. log1p makes very common edges not dominate too hard.
        ww = [math.log1p(float(c)) for c in cnts]
        s = float(sum(ww))
        if s <= 0.0:
            ww = [1.0 for _ in ww]
            s = float(len(ww))
        ww = [float(x) / s for x in ww]

        n = len(dsts)
        neigh[src, :n] = torch.tensor(dsts, dtype=torch.long, device=device)
        w[src, :n] = torch.tensor(ww, dtype=torch.float32, device=device)

    return neigh, w
