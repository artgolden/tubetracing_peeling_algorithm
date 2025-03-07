import numpy as np
from numba import njit, prange

@njit(parallel=True)
def remove_outliers(volume, k_param_for_outliers=3, outer_cube_size=41, local_cube_size=7):
    """
    Remove outlier positive voxels from a 3D boolean volume.

    For each positive voxel (True) in the input volume, the function:
      1. Considers the 41x41x41 cube centered on the voxel,
         excluding the inner 7x7x7 cube.
      2. For the outer positive voxels in that region, it computes the mean 
         and standard deviation of their Euclidean distances to the image midpoint.
      3. If the voxel’s distance to the midpoint is greater than mean + 3*std,
         the voxel and its local 7x7x7 neighborhood are set to False.

    The image midpoint is defined as the bottom middle voxel (ZYX order):
      (z = volume.shape[0]-1, y = volume.shape[1]//2, x = volume.shape[2]//2).

    Parameters:
      volume (numpy.ndarray): A 3D boolean array representing the volume.

    Returns:
      numpy.ndarray: A 3D boolean array with outlier voxels and their local neighborhoods removed.
    """
    outer_off = outer_cube_size - 1 // 2
    local_off = local_cube_size - 1 // 2
    z_dim, y_dim, x_dim = volume.shape
    # Create a copy for the output so we can mark outliers
    output = volume.copy()

    # Define image_midpoint: bottom middle in ZYX (z at last index, y and x at half)
    mid_z = z_dim - 1
    mid_y = y_dim // 2
    mid_x = x_dim // 2

    # Iterate over all voxels in parallel over the z-axis
    for z in prange(z_dim):
        for y in range(y_dim):
            for x in range(x_dim):
                if volume[z, y, x]:
                    # Define bounds for a cube around the voxel (±outer_off)
                    z0 = z - outer_off if z - outer_off >= 0 else 0
                    z1 = z + outer_off if z + outer_off < z_dim else z_dim - 1
                    y0 = y - outer_off if y - outer_off >= 0 else 0
                    y1 = y + outer_off if y + outer_off < y_dim else y_dim - 1
                    x0 = x - outer_off if x - outer_off >= 0 else 0
                    x1 = x + outer_off if x + outer_off < x_dim else x_dim - 1

                    # Accumulators for outer positive voxels distances
                    n = 0
                    sum_dist = 0.0
                    sum_sq_dist = 0.0

                    # Iterate over the outer cube, skipping the inner (±local_off)
                    for zz in range(z0, z1 + 1):
                        for yy in range(y0, y1 + 1):
                            for xx in range(x0, x1 + 1):
                                if abs(zz - z) <= local_off and abs(yy - y) <= local_off and abs(xx - x) <= local_off:
                                    continue
                                if volume[zz, yy, xx]:
                                    dz = zz - mid_z
                                    dy = yy - mid_y
                                    dx = xx - mid_x
                                    dist = (dz * dz + dy * dy + dx * dx) ** 0.5
                                    n += 1
                                    sum_dist += dist
                                    sum_sq_dist += dist * dist

                    if n > 0:
                        mean = sum_dist / n
                        variance = (sum_sq_dist / n) - (mean * mean)
                        if variance < 0.0:
                            variance = 0.0
                        std = variance ** 0.5

                        # Compute the distance of the voxel under consideration to the midpoint
                        dz = z - mid_z
                        dy = y - mid_y
                        dx = x - mid_x
                        dist_voxel = (dz * dz + dy * dy + dx * dx) ** 0.5

                        # Check if the voxel is an outlier
                        if dist_voxel > mean + k_param_for_outliers * std:
                            # Mark the voxel and its local neighborhood (±local_off) as False
                            z0_local = z - local_off if z - local_off >= 0 else 0
                            z1_local = z + local_off if z + local_off < z_dim else z_dim - 1
                            y0_local = y - local_off if y - local_off >= 0 else 0
                            y1_local = y + local_off if y + local_off < y_dim else y_dim - 1
                            x0_local = x - local_off if x - local_off >= 0 else 0
                            x1_local = x + local_off if x + local_off < x_dim else x_dim - 1
                            for zz in range(z0_local, z1_local + 1):
                                for yy in range(y0_local, y1_local + 1):
                                    for xx in range(x0_local, x1_local + 1):
                                        output[zz, yy, xx] = False
    return output