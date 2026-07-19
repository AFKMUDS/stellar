import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_symmetry_magnitudes(ckpt_path):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    bit_proj = sd.get("bit_proj.weight")
    
    if bit_proj is None:
        raise ValueError("bit_proj.weight not found")
        
    weights = bit_proj.detach().float().numpy()
    magnitudes = []
    for i in range(48):
        start_idx = i * 8
        slice_w = weights[:, start_idx : start_idx+8]
        magnitudes.append(np.mean(np.abs(slice_w)))
    return np.array(magnitudes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt1", type=str, required=True, help="Path to earlier checkpoint")
    parser.add_argument("--ckpt2", type=str, required=True, help="Path to later checkpoint")
    parser.add_argument("--label1", type=str, default="Early")
    parser.add_argument("--label2", type=str, default="Late")
    parser.add_argument("--out", type=str, default="viz/symmetry_comparison.png")
    args = parser.parse_args()

    m1 = get_symmetry_magnitudes(args.ckpt1)
    m2 = get_symmetry_magnitudes(args.ckpt2)
    
    # Calculate change
    delta = m2 - m1
    
    # Sort by magnitude in the LATE checkpoint
    indices = np.argsort(m2)[::-1]
    
    plt.figure(figsize=(14, 8))
    
    x = np.arange(48)
    width = 0.35
    
    plt.bar(x - width/2, m1[indices], width, label=args.label1, color='#1f77b4', alpha=0.7)
    plt.bar(x + width/2, m2[indices], width, label=args.label2, color='#ff7f0e', alpha=0.9)
    
    plt.title(f"Symmetry Importance Evolution: {args.label1} vs {args.label2}")
    plt.xlabel("Symmetry Index (Sorted by Late importance)")
    plt.ylabel("Mean Absolute Weight")
    plt.xticks(x, indices, rotation=90, fontsize=8)
    plt.legend()
    
    # Add delta text
    for i in range(48):
        d = delta[indices[i]]
        color = 'green' if d > 0 else 'red'
        # plt.text(x[i], max(m1[indices[i]], m2[indices[i]]) + 0.001, 
        #          f"{d:+.3f}", rotation=90, ha='center', va='bottom', fontsize=6, color=color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out)
    print(f"Comparison plot saved to: {args.out}")

if __name__ == "__main__":
    main()
