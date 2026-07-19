import torch
import itertools

def get_cube_permutations():
    """
    Generates the 48 permutations of indices 0..7 corresponding to the 
    rigid symmetries (rotations + reflections) of a cube.
    
    Mapping indices 0..7 to coordinates (x,y,z) in {0,1}^3:
    0: 000
    1: 001
    2: 010
    ...
    7: 111
    
    A symmetry is an affine transform on the coordinates:
    v' = M v + b  (mod 2)
    where M is a signed permutation matrix and b is a translation.
    
    Actually, simpler view:
    The group of symmetries of the cube (Oh) acts on the vertices.
    We just need to enumerate them.
    
    Vertices can be viewed as {-1, 1}^3 centered at 0.
    Symmetries are signed permutations of coordinates:
    (x, y, z) -> (+/- x_pi, +/- y_pi, +/- z_pi)
    There are 3! = 6 permutations of axes.
    There are 2^3 = 8 sign choices.
    Total 48 symmetries.
    """
    
    # Base vertices in {-1, 1}
    # Index i corresponds to binary(i) mapped to -1/1
    # 0 -> (-1, -1, -1)
    # 7 -> ( 1,  1,  1)
    vertices = []
    for i in range(8):
        x = 1.0 if (i & 4) else -1.0
        y = 1.0 if (i & 2) else -1.0
        z = 1.0 if (i & 1) else -1.0
        vertices.append(torch.tensor([x, y, z]))
    
    vertices = torch.stack(vertices) # (8, 3)
    
    permutations = []
    
    # 1. Axis permutations (3! = 6)
    for axis_order in itertools.permutations([0, 1, 2]):
        # 2. Sign flips (2^3 = 8)
        for signs in itertools.product([1.0, -1.0], repeat=3):
            # Construct the transform function
            # v' = (s0 * v[a0], s1 * v[a1], s2 * v[a2])
            
            perm_indices = []
            for i in range(8):
                v = vertices[i]
                v_new = torch.tensor([
                    signs[0] * v[axis_order[0]],
                    signs[1] * v[axis_order[1]],
                    signs[2] * v[axis_order[2]]
                ])
                
                # Find which original vertex v_new matches
                # We can compute dot products or just integer ID reconstruction
                # Reconstruct ID:
                # x>0 -> 4, y>0 -> 2, z>0 -> 1
                id_new = 0
                if v_new[0] > 0: id_new += 4
                if v_new[1] > 0: id_new += 2
                if v_new[2] > 0: id_new += 1
                perm_indices.append(id_new)
            
            permutations.append(torch.tensor(perm_indices, dtype=torch.long))
            
    return torch.stack(permutations) # (48, 8)

if __name__ == "__main__":
    perms = get_cube_permutations()
    print(f"Generated {len(perms)} permutations.")
    print("Identity:", perms[0])
    # Check uniqueness
    unique = torch.unique(perms, dim=0)
    print(f"Unique permutations: {len(unique)}")
