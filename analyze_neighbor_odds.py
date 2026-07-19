import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from raplib.data import iter_raw_byte_sequences, iter_train_sequences
from raplib.pathfinder import build_k_hop_tables


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help=".jsonl QA file or raw .bin byte corpus")
    p.add_argument("--mode", type=str, default="auto", choices=["auto", "jsonl", "bin"])
    p.add_argument("--seq_len", type=int, default=768)
    p.add_argument("--max_prompt", type=int, default=512)
    p.add_argument("--max_target", type=int, default=256)

    p.add_argument("--k", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--reserved", type=str, default="128,129,130")

    p.add_argument("--limit_seqs", type=int, default=0, help="Stop after N sequences (0 = no limit)")
    p.add_argument("--out", type=str, default="neighbor_odds.txt")
    p.add_argument("--per_token", action="store_true", help="Also write per-token hit rates")
    p.add_argument("--top_n", type=int, default=40)
    p.add_argument("--plot", action="store_true", help="Write PNG visualizations to --plot_dir")
    p.add_argument("--plot_dir", type=str, default="neighbor_odds_viz")
    return p.parse_args()


def _byte_label(b: int) -> str:
    b = int(b) & 0xFF
    if b == 10:
        return "\\n"
    if b == 9:
        return "\\t"
    if 32 <= b <= 126:
        return chr(b)
    if b in {128, 129, 130}:
        return "<TAG>"
    return f"x{b:02x}"


def main() -> None:
    args = parse_args()

    reserved = {int(x.strip()) & 0xFF for x in str(args.reserved).split(",") if x.strip()}

    neigh_t, _dist_t = build_k_hop_tables(k=int(args.k), reserved=tuple(sorted(reserved)))
    neigh = neigh_t.cpu().numpy()  # (256, M)

    mode = str(args.mode).lower()
    if mode == "auto":
        mode = "bin" if str(args.data).lower().endswith(".bin") else "jsonl"

    if mode == "bin":
        stream = iter_raw_byte_sequences(bin_path=str(args.data), max_total_bytes=int(args.seq_len))
    else:
        stream = iter_train_sequences(
            jsonl_path=str(args.data),
            max_prompt_bytes=int(args.max_prompt),
            max_target_bytes=int(args.max_target),
            max_total_bytes=int(args.seq_len),
            include_answer_prefix=True,
        )

    total_tokens = 0
    total_usable = 0

    # Per-token (current token) stats: how often token has any usable neighbor in history.
    tok_count = np.zeros(256, dtype=np.int64)
    tok_usable = np.zeros(256, dtype=np.int64)

    # Range breakdown
    range_count = defaultdict(int)
    range_usable = defaultdict(int)

    def bucket(b: int) -> str:
        if b in reserved:
            return "reserved"
        if 0 <= b <= 127:
            return "ascii_0_127"
        return "ext_128_255"

    n_seqs = 0
    for seq_bytes, _prompt_len in stream:
        n_seqs += 1
        if int(args.limit_seqs) > 0 and n_seqs > int(args.limit_seqs):
            break

        ids = [int(x) & 0xFF for x in seq_bytes]
        if not ids:
            continue

        # last seen position of each byte (within this sequence)
        last_pos = np.full(256, -1, dtype=np.int32)

        for t, tok in enumerate(ids):
            if tok in reserved:
                # Treat reserved bytes as atomic and non-retrievable.
                last_pos[tok] = t
                continue

            nb_row = neigh[tok]
            usable = False
            for nb in nb_row.tolist():
                nb = int(nb)
                if nb < 0:
                    continue
                if last_pos[nb] >= 0:
                    usable = True
                    break

            total_tokens += 1
            tok_count[tok] += 1
            range_count[bucket(tok)] += 1

            if usable:
                total_usable += 1
                tok_usable[tok] += 1
                range_usable[bucket(tok)] += 1

            last_pos[tok] = t

    overall_p = (float(total_usable) / float(total_tokens)) if total_tokens > 0 else 0.0

    lines = []
    lines.append(f"data={args.data}")
    lines.append(f"mode={mode}")
    lines.append(f"k={int(args.k)}")
    lines.append(f"reserved={sorted(reserved)}")
    lines.append(f"seqs={n_seqs}")
    lines.append(f"tokens_scored={total_tokens}")
    lines.append(f"usable_hits={total_usable}")
    lines.append(f"P(usable_neighbor)={overall_p:.4f}")

    for key in ["ascii_0_127", "ext_128_255", "reserved"]:
        c = int(range_count.get(key, 0))
        u = int(range_usable.get(key, 0))
        p = (float(u) / float(c)) if c > 0 else 0.0
        lines.append(f"P(usable_neighbor | {key})={p:.4f}  (hits={u} / n={c})")

    out_path = str(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Optional visualizations
    if bool(args.plot):
        out_dir = str(args.plot_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Range bar chart
        keys = ["ascii_0_127", "ext_128_255", "reserved"]
        ps = []
        ns = []
        for key in keys:
            c = int(range_count.get(key, 0))
            u = int(range_usable.get(key, 0))
            pval = (float(u) / float(c)) if c > 0 else 0.0
            ps.append(pval)
            ns.append(c)

        plt.figure(figsize=(6, 4), dpi=160)
        plt.bar(keys, ps, color=["#66ccff", "#4477aa", "#ff6666"])
        plt.ylim(0.0, 1.0)
        plt.ylabel("P(usable neighbor)")
        plt.title(f"Usable neighbor probability by range (k={int(args.k)})")
        for i, (pval, c) in enumerate(zip(ps, ns)):
            plt.text(i, min(0.98, pval + 0.03), f"{pval:.3f}\n(n={c})", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "range_probs.png"))
        plt.close()

        # Per-token hit-rate scatter (only tokens observed)
        observed = tok_count > 0
        p_tok = np.zeros(256, dtype=np.float64)
        p_tok[observed] = tok_usable[observed].astype(np.float64) / tok_count[observed].astype(np.float64)

        xs = np.arange(256, dtype=np.int32)
        plt.figure(figsize=(10, 4), dpi=160)
        plt.scatter(xs[observed], p_tok[observed], s=10, alpha=0.7, c="#66ccff")
        for rb in sorted(reserved):
            plt.axvline(rb, color="#ff6666", alpha=0.35, linewidth=1)
        plt.ylim(0.0, 1.0)
        plt.xlim(0, 255)
        plt.xlabel("byte id")
        plt.ylabel("P(usable neighbor | token)")
        plt.title(f"Per-token usable neighbor probability (observed tokens only) | k={int(args.k)}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_token_scatter.png"))
        plt.close()

        # 16x16 heatmap by byte id
        grid = p_tok.reshape(16, 16)
        plt.figure(figsize=(6, 5), dpi=160)
        plt.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"P(usable neighbor | token) heatmap (byte id grid) | k={int(args.k)}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "per_token_heatmap.png"))
        plt.close()

    if bool(args.per_token):
        rows = []
        for b in range(256):
            if b in reserved:
                continue
            c = int(tok_count[b])
            if c <= 0:
                continue
            u = int(tok_usable[b])
            p = float(u) / float(c)
            rows.append((p, c, b, u))

        rows.sort(reverse=True)
        top = rows[: int(args.top_n)]
        bot = list(reversed(rows[-int(args.top_n) :])) if rows else []

        pt_path = os.path.splitext(out_path)[0] + ".per_token.txt"
        with open(pt_path, "w", encoding="utf-8") as f:
            f.write("Top tokens by P(usable_neighbor)\n")
            for p, c, b, u in top:
                f.write(f"{b:3d}  {_byte_label(b):>6s}  p={p:.4f}  hits={u}  n={c}\n")
            f.write("\nBottom tokens by P(usable_neighbor)\n")
            for p, c, b, u in bot:
                f.write(f"{b:3d}  {_byte_label(b):>6s}  p={p:.4f}  hits={u}  n={c}\n")

    print(f"Wrote: {os.path.abspath(out_path)}")
    if bool(args.per_token):
        print(f"Wrote: {os.path.abspath(os.path.splitext(out_path)[0] + '.per_token.txt')}")
    if bool(args.plot):
        print(f"Wrote: {os.path.abspath(str(args.plot_dir))}")


if __name__ == "__main__":
    main()
