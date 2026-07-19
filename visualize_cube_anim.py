import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def get_vertices():
    # Map index 0..7 to (x,y,z) in {-1, 1}
    # i & 4 -> x, i & 2 -> y, i & 1 -> z
    vertices = []
    for i in range(8):
        x = 1 if (i & 4) else -1
        y = 1 if (i & 2) else -1
        z = 1 if (i & 1) else -1
        vertices.append([x, y, z])
    return np.array(vertices)

def get_edges(vertices):
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            # Connect if Hamming distance is 1 (differ by 1 bit)
            # dist = sum of abs diffs / 2
            dist = np.sum(np.abs(vertices[i] - vertices[j]))
            if dist == 2: # in {-1, 1} coords, diff is 2 means one coordinate flipped
                edges.append((i, j))
    return edges

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Stellar", help="Text to visualize")
    parser.add_argument("--out", type=str, default="viz/cube_anim.gif")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    # Data prep
    text_bytes = [int(b) for b in args.text.encode("utf-8")]
    # Pad with a pause at the end
    text_bytes += [text_bytes[-1]] * 2 
    
    vertices = get_vertices()
    edges = get_edges(vertices)

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Cube settings
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_axis_off()
    
    # Static elements
    # Draw edges
    for (i, j) in edges:
        ax.plot3D(
            [vertices[i, 0], vertices[j, 0]],
            [vertices[i, 1], vertices[j, 1]],
            [vertices[i, 2], vertices[j, 2]],
            'gray', alpha=0.3
        )

    # Dynamic elements: Nodes
    # We will update colors/sizes based on bits
    # Store reference to scatter to remove it later
    scat = [ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=200, c='blue', depthshade=False)]
    
    # Title
    title_text = ax.text2D(0.5, 0.95, "", transform=ax.transAxes, ha="center", fontsize=16)

    # Animation function
    # Frames per character
    frames_per_char = 10
    total_frames = len(text_bytes) * frames_per_char
    
    def update(frame):
        # Remove previous scatter
        if scat[0] is not None:
            scat[0].remove()
        
        char_idx = frame // frames_per_char
        if char_idx >= len(text_bytes):
            char_idx = len(text_bytes) - 1
            
        # Interpolation factor (0 to 1)
        t = (frame % frames_per_char) / frames_per_char
        
        # Current and Next Byte
        curr_b = text_bytes[char_idx]
        next_b = text_bytes[min(char_idx + 1, len(text_bytes) - 1)]
        
        # Bits (0 or 1)
        curr_bits = np.array([(curr_b >> k) & 1 for k in range(8)])
        next_bits = np.array([(next_b >> k) & 1 for k in range(8)])
        
        # Interpolate "activation"
        interp_bits = curr_bits * (1 - t) + next_bits * t
        
        # Map to colors
        colors = []
        sizes = []
        for val in interp_bits:
            # Blue (#1f77b4) to Red (#d62728)
            r = 0.12 * (1 - val) + 0.84 * val
            g = 0.47 * (1 - val) + 0.15 * val
            b = 0.71 * (1 - val) + 0.16 * val
            colors.append((r, g, b))
            sizes.append(200 + 100 * val)
            
        # Redraw scatter
        scat[0] = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=sizes, c=colors, depthshade=False)
        
        # Rotate camera slowly
        ax.view_init(elev=20, azim=frame * 2)
        
        # Update title
        if t < 0.5:
            char_code = curr_b
        else:
            char_code = next_b
        
        try:
            char_str = chr(char_code) if 32 <= char_code <= 126 else f"x{char_code:02x}"
        except:
            char_str = "?"
            
        title_text.set_text(f"Char: '{char_str}' ({char_code})")
        
        # Return list of artists to update (though for 3D anim with blit=False it matters less)
        return scat[0], title_text

    print(f"Generating animation ({total_frames} frames)...")
    # Turn off blit because we are removing artists
    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/args.fps, blit=False)
    anim.save(args.out, writer='pillow', fps=args.fps)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
