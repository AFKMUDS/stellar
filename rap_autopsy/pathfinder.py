from __future__ import annotations

from collections import deque
from typing import Iterable, Tuple

import torch


def _rotl(x: int) -> int:
    x &= 0xFF
    return ((x << 1) | (x >> 7)) & 0xFF


def _rotr(x: int) -> int:
    x &= 0xFF
    return ((x >> 1) | ((x & 1) << 7)) & 0xFF


def _inv(x: int) -> int:
    return (~x) & 0xFF


def _revbits(x: int) -> int:
    x &= 0xFF
    v = 0
    for b in range(8):
        v = (v << 1) | ((x >> b) & 1)
    return v & 0xFF


def _inc(x: int) -> int:
    return (x + 1) & 0xFF


def _dec(x: int) -> int:
    return (x - 1) & 0xFF


_MOVES = (_rotl, _rotr, _inv, _revbits, _inc, _dec)


def build_k_hop_tables(
    *,
    k: int,
    reserved: Iterable[int] = (128, 129, 130),
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute k-hop neighborhood tables for bytes 0..255.

    Returns:
      - neigh: (256, M) long tensor of neighbor byte IDs (padded with -1)
      - dist:  (256, M) float tensor of graph distances (same padding)

    Reserved bytes are treated as *atomic*: they have no neighbors.
    """

    if k <= 0:
        raise ValueError("k must be >= 1")

    reserved_set = {int(x) & 0xFF for x in reserved}

    neigh_lists: list[list[int]] = [[] for _ in range(256)]
    dist_lists: list[list[int]] = [[] for _ in range(256)]

    for src in range(256):
        if src in reserved_set:
            continue

        q = deque([src])
        dist = {src: 0}

        while q:
            u = q.popleft()
            du = dist[u]
            if du >= k:
                continue

            for mv in _MOVES:
                v = mv(u)
                if v in reserved_set:
                    continue
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)

        # Exclude self; only keep nodes at 1..k.
        for v, dv in dist.items():
            if v == src:
                continue
            if 1 <= dv <= k:
                neigh_lists[src].append(int(v))
                dist_lists[src].append(int(dv))

    max_m = max((len(x) for x in neigh_lists), default=0)
    if max_m == 0:
        max_m = 1

    neigh = torch.full((256, max_m), -1, dtype=torch.long, device=device)
    d = torch.zeros((256, max_m), dtype=torch.float32, device=device)

    for b in range(256):
        xs = neigh_lists[b]
        ds = dist_lists[b]
        if not xs:
            continue
        n = len(xs)
        neigh[b, :n] = torch.tensor(xs, dtype=torch.long, device=device)
        d[b, :n] = torch.tensor(ds, dtype=torch.float32, device=device)

    return neigh, d
