import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def prune_volume(volume, block_h=4, block_w=4):
    """
    Pruning of voxels in a 3D volume accelerated using Numba.
    
    For each slice in the 3D boolean volume (shape: [n_slices, height, width]), the function:
      - Computes the mid-point of the slice as (x=width//2, y=height//2).
      - Splits the slice into non-overlapping 4x4 blocks by default.
      - Within each block, finds the True pixel furthest from the mid-point (using squared Euclidean distance).
      - Prunes the block by setting all True pixels to False except the selected one.
    
    Parameters:
        volume (np.ndarray): 3D numpy array with boolean values.
        block_h (int): Height of each block.
        block_w (int): Width of each block.
        
    Returns:
        volume (np.ndarray): The modified volume with pruned voxels.
    """
    n_slices, height, width = volume.shape
    block_h, block_w = 4, 4
    # mid-point for each slice (same for all slices)
    mid_y = height // 2
    mid_x = width // 2

    # Compute the number of blocks along each dimension.
    grid_y = (height + block_h - 1) // block_h
    grid_x = (width + block_w - 1) // block_w

    # Process each slice in parallel.
    for s in prange(n_slices):
        for by in range(grid_y):
            for bx in range(grid_x):
                start_y = by * block_h
                start_x = bx * block_w
                max_dist = -1.0
                max_r = -1
                max_c = -1
                # First pass: Find the True pixel with the maximum distance.
                for i in range(block_h):
                    r = start_y + i
                    if r >= height:
                        break
                    for j in range(block_w):
                        c = start_x + j
                        if c >= width:
                            break
                        if volume[s, r, c]:
                            dx = c - mid_x
                            dy = r - mid_y
                            dist = dx * dx + dy * dy
                            if dist > max_dist:
                                max_dist = dist
                                max_r = r
                                max_c = c
                # Second pass: Clear all True pixels except the one with max distance.
                if max_r != -1:  # if at least one True pixel was found
                    for i in range(block_h):
                        r = start_y + i
                        if r >= height:
                            break
                        for j in range(block_w):
                            c = start_x + j
                            if c >= width:
                                break
                            if volume[s, r, c] and not (r == max_r and c == max_c):
                                volume[s, r, c] = False
    return volume