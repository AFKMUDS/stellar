import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint with symmetry expansion")
    parser.add_argument("--out", type=str, default="viz/symmetry_analysis.png")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    
    bit_proj = sd.get("bit_proj.weight") # Shape: (d_model, 80)
    
    if bit_proj is None:
        print("Error: 'bit_proj.weight' not found in checkpoint.")
        return

    if bit_proj.shape[1] != 80:
        print(f"Error: Expected input dimension 80 for symmetry expansion, found {bit_proj.shape[1]}.")
        print("This script is specifically for the 'Expanded Symmetry' model (Original + Invert + Reverse + 7 Rotations).")
        return

    # Calculate magnitude (importance) of each group
    # Groups are 8 bits each.
    # 0: Original
    # 1: Inverted
    # 2: Reversed
    # 3-9: Rotations 1-7
    
    weights = bit_proj.detach().float().numpy() # (d_model, 80)
    magnitudes = []
    labels = []
    
    # Helper to calculate mean absolute weight for a slice
    def get_mag(start_idx, name):
        slice_w = weights[:, start_idx : start_idx+8]
        # Frobenius norm of the slice, normalized by size
        mag = np.mean(np.abs(slice_w))
        magnitudes.append(mag)
        labels.append(name)
        return mag

    print("\nSymmetry Importance Analysis:")
    print("-" * 30)
    
    # 1. Original
    get_mag(0, "Original")
    
    # 2. Inverted
    get_mag(8, "Inverted")
    
    # 3. Reversed
    get_mag(16, "Reversed")
    
    # 4. Rotations
    for r in range(1, 8):
        get_mag(16 + r*8, f"Rot-{r}")

    # Print values
    for l, m in zip(labels, magnitudes):
        print(f"{l:10s}: {m:.4f}")

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, magnitudes, color=['#1f77b4', '#ff7f0e', '#2ca02c'] + ['#d62728']*7)
    
    plt.title("Importance of Geometric Symmetries (Mean Absolute Weight)")
    plt.ylabel("Average Weight Magnitude")
    plt.xlabel("Transformation Type")
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out)
    print(f"\nAnalysis plot saved to: {args.out}")

if __name__ == "__main__":
    main()
