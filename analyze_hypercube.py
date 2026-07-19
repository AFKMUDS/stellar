import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint with hypercube expansion")
    parser.add_argument("--out", type=str, default="viz/hypercube_analysis.png")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    
    bit_proj = sd.get("bit_proj.weight") # Shape: (d_model, 384)
    
    if bit_proj is None:
        print("Error: 'bit_proj.weight' not found in checkpoint.")
        return

    if bit_proj.shape[1] != 384:
        print(f"Error: Expected input dimension 384 for hypercube expansion, found {bit_proj.shape[1]}.")
        return

    # Calculate magnitude (importance) of each group
    # Groups are 8 bits each, corresponding to the 48 symmetries.
    
    weights = bit_proj.detach().float().numpy() # (d_model, 384)
    magnitudes = []
    
    for i in range(48):
        start_idx = i * 8
        slice_w = weights[:, start_idx : start_idx+8]
        # Mean absolute weight
        mag = np.mean(np.abs(slice_w))
        magnitudes.append(mag)

    # Sort to see top symmetries
    sorted_indices = np.argsort(magnitudes)[::-1]
    
    print("\nHypercube Symmetry Importance Analysis:")
    print("-" * 40)
    print(f"{'Rank':<5} {'SymID':<8} {'Magnitude':<10}")
    
    for rank, idx in enumerate(sorted_indices):
        print(f"{rank+1:<5} {idx:<8} {magnitudes[idx]:.4f}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(48), magnitudes, color='#1f77b4')
    
    plt.title("Importance of 48 Hypercube Symmetries")
    plt.ylabel("Average Weight Magnitude")
    plt.xlabel("Symmetry Index (0-47)")
    plt.xlim(-1, 48)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out)
    print(f"\nAnalysis plot saved to: {args.out}")

if __name__ == "__main__":
    main()
