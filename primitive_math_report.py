import argparse
import json
import math
import os

import numpy as np
import torch

from probe_latent_step import load_model



def flatten_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))



def matrix_stats(mat: np.ndarray) -> dict[str, float]:
    s = np.linalg.svd(mat.astype(np.float64), compute_uv=False)
    fro = float(np.linalg.norm(mat))
    spec = float(s[0]) if len(s) else 0.0
    stable_rank = float((fro * fro) / max(spec * spec, 1e-12))
    return {
        "fro_norm": fro,
        "spectral_norm": spec,
        "stable_rank": stable_rank,
        "sv_top1": float(s[0]) if len(s) > 0 else 0.0,
        "sv_top2": float(s[1]) if len(s) > 1 else 0.0,
        "sv_top3": float(s[2]) if len(s) > 2 else 0.0,
        "mean_singular": float(np.mean(s)) if len(s) else 0.0,
    }



def build_similarity_matrix(mats: list[np.ndarray]) -> list[list[float]]:
    n = len(mats)
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = flatten_cosine(mats[i], mats[j])
    return out



def primitive_report(model) -> dict:
    A = model.prim_A.detach().cpu().float().numpy()
    B = model.prim_B.detach().cpu().float().numpy()
    bias = model.prim_bias.detach().cpu().float().numpy()
    n_primitives = int(A.shape[0])

    rows = []
    A_mats = []
    B_mats = []
    for k in range(n_primitives):
        Ak = A[k]
        Bk = B[k]
        bk = bias[k]
        A_mats.append(Ak)
        B_mats.append(Bk)

        a_stats = matrix_stats(Ak)
        b_stats = matrix_stats(Bk)
        bias_norm = float(np.linalg.norm(bk))
        dominance_ratio = float(a_stats["fro_norm"] / max(b_stats["fro_norm"], 1e-12))
        thinker_score = float(a_stats["spectral_norm"] / max(b_stats["spectral_norm"], 1e-12))
        state_energy = float(np.mean(np.sum(Ak * Ak, axis=-1)))
        input_energy = float(np.mean(np.sum(Bk * Bk, axis=0)))
        rows.append(
            {
                "primitive": k,
                "A": a_stats,
                "B": b_stats,
                "bias_norm": bias_norm,
                "dominance_ratio_A_over_B": dominance_ratio,
                "thinker_score": thinker_score,
                "state_energy_mean": state_energy,
                "input_energy_mean": input_energy,
                "bias_to_A_ratio": float(bias_norm / max(a_stats["fro_norm"], 1e-12)),
                "bias_to_B_ratio": float(bias_norm / max(b_stats["fro_norm"], 1e-12)),
            }
        )

    sim_A = build_similarity_matrix(A_mats)
    sim_B = build_similarity_matrix(B_mats)
    ghost_candidates = []
    for i in range(n_primitives):
        for j in range(i + 1, n_primitives):
            pair_score = 0.5 * (sim_A[i][j] + sim_B[i][j])
            ghost_candidates.append(
                {
                    "pair": [i, j],
                    "A_cosine": sim_A[i][j],
                    "B_cosine": sim_B[i][j],
                    "ghost_score": float(pair_score),
                }
            )
    ghost_candidates.sort(key=lambda x: x["ghost_score"], reverse=True)

    thinker_ranking = sorted(rows, key=lambda r: r["dominance_ratio_A_over_B"], reverse=True)
    reactor_ranking = sorted(rows, key=lambda r: r["dominance_ratio_A_over_B"])
    amplifier_ranking = sorted(rows, key=lambda r: r["A"]["spectral_norm"], reverse=True)
    stabilizer_ranking = sorted(rows, key=lambda r: r["A"]["spectral_norm"])
    bias_ranking = sorted(rows, key=lambda r: r["bias_norm"], reverse=True)

    return {
        "n_primitives": n_primitives,
        "rows": rows,
        "A_similarity": sim_A,
        "B_similarity": sim_B,
        "ghost_candidates": ghost_candidates,
        "rankings": {
            "thinker_like": [int(r["primitive"]) for r in thinker_ranking],
            "reactor_like": [int(r["primitive"]) for r in reactor_ranking],
            "amplifier_like": [int(r["primitive"]) for r in amplifier_ranking],
            "stabilizer_like": [int(r["primitive"]) for r in stabilizer_ranking],
            "bias_drift_like": [int(r["primitive"]) for r in bias_ranking],
        },
    }



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints1/b.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="primitive_math_report.json")
    args = p.parse_args()

    model = load_model(args.ckpt, args.device)
    report = primitive_report(model)
    report["checkpoint"] = args.ckpt

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote: {os.path.abspath(args.out)}")
    print("Thinker-like:", report["rankings"]["thinker_like"])
    print("Reactor-like:", report["rankings"]["reactor_like"])
    print("Amplifier-like:", report["rankings"]["amplifier_like"])
    print("Stabilizer-like:", report["rankings"]["stabilizer_like"])
    print("Bias-drift-like:", report["rankings"]["bias_drift_like"])
    print("Top ghost candidates:")
    for row in report["ghost_candidates"][:5]:
        print(
            f"  P{row['pair'][0]}<->P{row['pair'][1]} ghost={row['ghost_score']:.4f} "
            f"A={row['A_cosine']:.4f} B={row['B_cosine']:.4f}"
        )


if __name__ == "__main__":
    main()
