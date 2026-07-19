import argparse
import glob
import os
from dataclasses import asdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from raplib.model import ModelConfig, PrimitiveManifoldLM
from raplib.pathfinder import build_k_hop_tables


def _find_latest_ckpt(ckpt_glob: str) -> str:
    paths = glob.glob(ckpt_glob)
    if not paths:
        raise FileNotFoundError(f"No checkpoints matched: {ckpt_glob}")
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def _load_model(ckpt_path: str, device: torch.device) -> tuple[PrimitiveManifoldLM, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("cfg") if isinstance(ckpt, dict) else None

    if isinstance(cfg_dict, dict):
        valid_keys = ModelConfig.__annotations__.keys()
        filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        cfg = ModelConfig(**filtered)
    else:
        cfg = ModelConfig()

    model = PrimitiveManifoldLM(cfg).to(device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()

    meta = {
        "ckpt": ckpt_path,
        "cfg": asdict(cfg),
        "step": int(ckpt.get("step", -1)) if isinstance(ckpt, dict) else -1,
    }
    return model, meta


def _pca(x: np.ndarray, n: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return (u[:, :n] * s[:n])


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


_MOVES: Tuple[Tuple[str, callable], ...] = (
    ("rotl", _rotl),
    ("rotr", _rotr),
    ("inv", _inv),
    ("rev", _revbits),
    ("inc", _inc),
    ("dec", _dec),
)


def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a - b) ** 2).sum(axis=-1))


def _build_graph_dist_lookup(neigh: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Dense lookup table D[src, dst] = shortest dist within k hops, else 0.

    Note: 0 means either src==dst or unreachable-within-k.
    """
    D = np.zeros((256, 256), dtype=np.int16)
    for src in range(256):
        row_n = neigh[src]
        row_d = dist[src]
        for nb, dd in zip(row_n.tolist(), row_d.tolist()):
            nb = int(nb)
            if nb < 0:
                continue
            D[src, nb] = int(dd)
    return D


def _run_neighbor_probe(
    *,
    out_dir: str,
    emb: np.ndarray,
    reserved: set[int],
    neigh: np.ndarray,
    dist: np.ndarray,
    focus: int,
    sample_pairs: int,
    seed: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    D = _build_graph_dist_lookup(neigh, dist)

    # Per-move embedding distance stats.
    move_lines = []
    for name, fn in _MOVES:
        ds = []
        for b in range(256):
            if b in reserved:
                continue
            bb = int(fn(b))
            if bb in reserved:
                continue
            ds.append(float(np.linalg.norm(emb[b] - emb[bb])))
        if ds:
            move_lines.append(
                f"{name:>4s}  mean={np.mean(ds):.4f}  med={np.median(ds):.4f}  p90={np.quantile(ds, 0.90):.4f}  n={len(ds)}"
            )

    with open(os.path.join(out_dir, "move_distance_stats.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(move_lines) + ("\n" if move_lines else ""))

    # Graph dist vs embedding dist scatter (sampled).
    rng = np.random.default_rng(int(seed))
    valid_bytes = np.array([b for b in range(256) if b not in reserved], dtype=np.int32)
    if valid_bytes.size >= 2 and int(sample_pairs) > 0:
        a_ix = rng.integers(0, valid_bytes.size, size=int(sample_pairs), dtype=np.int32)
        b_ix = rng.integers(0, valid_bytes.size, size=int(sample_pairs), dtype=np.int32)
        srcs = valid_bytes[a_ix]
        dsts = valid_bytes[b_ix]

        gds = D[srcs, dsts].astype(np.int32)
        eds = _pairwise_l2(emb[srcs], emb[dsts]).astype(np.float64)

        # Only keep pairs where graph distance is known (>0) within k.
        m = gds > 0
        gds = gds[m]
        eds = eds[m]

        if gds.size > 0:
            # Correlation on (dist, emb_dist)
            corr = float(np.corrcoef(gds.astype(np.float64), eds)[0, 1]) if gds.size > 1 else float("nan")

            plt.figure(figsize=(6, 4), dpi=160)
            plt.scatter(gds, eds, s=8, alpha=0.35, c="#66ccff")
            plt.title(f"Graph dist vs embedding L2 (k-hop known pairs)\nPearson r={corr:.3f}")
            plt.xlabel("move-graph shortest dist (within k)")
            plt.ylabel("embedding L2")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "graphdist_vs_embdist.png"))
            plt.close()

            with open(os.path.join(out_dir, "graphdist_vs_embdist.txt"), "w", encoding="utf-8") as f:
                f.write(f"k_hop_max={int(dist.max()) if dist.size else 0}\n")
                f.write(f"pairs_sampled={int(sample_pairs)}\n")
                f.write(f"pairs_kept={int(gds.size)}\n")
                f.write(f"pearson_r={corr}\n")

    # Nearest embedding neighbors for focus byte, annotated with move-graph distance.
    focus = int(focus) & 0xFF
    if focus not in reserved:
        v = emb[focus]
        d = np.linalg.norm(emb - v[None, :], axis=1)
        order = np.argsort(d)
        order = [int(i) for i in order if int(i) != focus and int(i) not in reserved][:64]
        lines = []
        for i in order:
            gd = int(D[focus, i])
            lines.append(f"{i:3d}  {_byte_label(i):>6s}  emb_dist={float(d[i]):.4f}  graph_dist={gd if gd > 0 else -1}")
        with open(os.path.join(out_dir, f"nearest_neighbors_focus{focus}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="", help="Path to a checkpoint .pt file")
    p.add_argument(
        "--ckpt_glob",
        type=str,
        default="*.pt",
        help="Glob used to auto-pick the newest checkpoint when --ckpt is not set",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="byte_manifold.png")
    p.add_argument("--dims", type=int, default=2, choices=[2, 3])
    p.add_argument("--move_k", type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--edges", type=str, default="focus", choices=["none", "focus", "all"]) 
    p.add_argument("--focus", type=int, default=58, help="Byte ID to focus (used when --edges=focus)")
    p.add_argument("--annotate", action="store_true", help="Annotate points with labels (can be cluttered)")
    p.add_argument("--max_labels", type=int, default=128)
    p.add_argument("--probe", action="store_true", help="Write neighborhood probe outputs (PNG/TXT) to --probe_dir")
    p.add_argument("--probe_dir", type=str, default="byte_probe", help="Output directory for --probe artifacts")
    p.add_argument("--probe_pairs", type=int, default=20000, help="Random pairs to sample for graphdist-vs-embdist")
    p.add_argument("--probe_seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = args.ckpt
    if not ckpt_path:
        ckpt_path = _find_latest_ckpt(args.ckpt_glob)

    device = torch.device(args.device)
    model, meta = _load_model(ckpt_path, device=device)

    with torch.no_grad():
        idx = torch.arange(256, device=device, dtype=torch.long)
        emb = model._get_emb(idx).detach().float().cpu().numpy()  # (256, d_model)

    coords = _pca(emb, n=int(args.dims))

    reserved = {128, 129, 130}
    printable = np.array([(32 <= i <= 126) for i in range(256)], dtype=bool)
    is_reserved = np.array([(i in reserved) for i in range(256)], dtype=bool)

    colors = np.full(256, "#999999", dtype=object)
    colors[printable] = "#66ccff"
    colors[~printable] = "#aaaaaa"
    colors[is_reserved] = "#ff6666"

    neigh = None
    dist = None
    if args.edges != "none":
        neigh_t, dist_t = build_k_hop_tables(k=int(args.move_k), reserved=tuple(sorted(reserved)), device=torch.device("cpu"))
        neigh = neigh_t.cpu().numpy()
        dist = dist_t.cpu().numpy()
    elif bool(args.probe):
        # Probe needs neighborhood tables even if we don't draw edges.
        neigh_t, dist_t = build_k_hop_tables(k=int(args.move_k), reserved=tuple(sorted(reserved)), device=torch.device("cpu"))
        neigh = neigh_t.cpu().numpy()
        dist = dist_t.cpu().numpy()

    if bool(args.probe) and neigh is not None and dist is not None:
        _run_neighbor_probe(
            out_dir=str(args.probe_dir),
            emb=emb,
            reserved=reserved,
            neigh=neigh,
            dist=dist,
            focus=int(args.focus),
            sample_pairs=int(args.probe_pairs),
            seed=int(args.probe_seed),
        )

    if int(args.dims) == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors.tolist(), s=26, alpha=0.95)

        if args.edges != "none" and neigh is not None and dist is not None:
            if args.edges == "focus":
                srcs = [int(args.focus) & 0xFF]
            else:
                srcs = [i for i in range(256) if i not in reserved]

            for src in srcs:
                row_n = neigh[src]
                row_d = dist[src]
                for nb, dd in zip(row_n.tolist(), row_d.tolist()):
                    if int(nb) < 0:
                        continue
                    nb = int(nb)
                    if nb in reserved:
                        continue
                    x0, y0, z0 = coords[src]
                    x1, y1, z1 = coords[nb]
                    a = 0.22 if args.edges == "all" else 0.55
                    lw = 0.6 if float(dd) > 1.0 else 1.2
                    ax.plot([x0, x1], [y0, y1], [z0, z1], color="#444444", alpha=a, linewidth=lw)

        if bool(args.annotate):
            shown = 0
            for i in range(256):
                if shown >= int(args.max_labels):
                    break
                if i in reserved or (32 <= i <= 126):
                    ax.text(coords[i, 0], coords[i, 1], coords[i, 2], _byte_label(i), fontsize=7)
                    shown += 1

        ax.set_title(f"Byte manifold PCA-{args.dims} | step={meta['step']} | ckpt={os.path.basename(meta['ckpt'])}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.tight_layout()
        plt.savefig(args.out)
        plt.close()

    else:
        fig = plt.figure(figsize=(10, 8), dpi=160)
        ax = fig.add_subplot(111)
        ax.scatter(coords[:, 0], coords[:, 1], c=colors.tolist(), s=26, alpha=0.95)

        if args.edges != "none" and neigh is not None and dist is not None:
            if args.edges == "focus":
                srcs = [int(args.focus) & 0xFF]
            else:
                srcs = [i for i in range(256) if i not in reserved]

            for src in srcs:
                row_n = neigh[src]
                row_d = dist[src]
                for nb, dd in zip(row_n.tolist(), row_d.tolist()):
                    if int(nb) < 0:
                        continue
                    nb = int(nb)
                    if nb in reserved:
                        continue
                    x0, y0 = coords[src]
                    x1, y1 = coords[nb]
                    a = 0.18 if args.edges == "all" else 0.6
                    lw = 0.5 if float(dd) > 1.0 else 1.0
                    ax.plot([x0, x1], [y0, y1], color="#444444", alpha=a, linewidth=lw)

        if bool(args.annotate):
            shown = 0
            for i in range(256):
                if shown >= int(args.max_labels):
                    break
                if i in reserved or (32 <= i <= 126):
                    ax.text(coords[i, 0], coords[i, 1], _byte_label(i), fontsize=7)
                    shown += 1

        ax.set_title(f"Byte manifold PCA-{args.dims} | step={meta['step']} | ckpt={os.path.basename(meta['ckpt'])}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.tight_layout()
        plt.savefig(args.out)
        plt.close()

    print(f"Loaded: {meta['ckpt']}")
    print(f"Wrote:  {os.path.abspath(args.out)}")
    if bool(args.probe):
        print(f"Probe:  {os.path.abspath(str(args.probe_dir))}")


if __name__ == "__main__":
    main()
