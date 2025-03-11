import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection

# -----------------------------------------------------------------------------
# Cylindrical projection for distortion computation:
# Given each point in (z,y,x) order, compute an unwrapped projection that maps:
#   (y, x) --> (s, x) with s = max_r * (theta + π/2)
# where theta = arctan2(y - origin_y, x - origin_x) (with B = x-origin_x checked to be >=0).
# Z is left unchanged for the distortion map interpolation.
# -----------------------------------------------------------------------------
@njit(parallel=True)
def cylindrical_projection(volume_points, origin_yx, max_r):
    N = volume_points.shape[0]
    # projected coordinates: column 0: s (angular coordinate unwrapped), column 1: x (preserved)
    projected = np.empty((N, 2), dtype=np.float64)
    for i in prange(N):
        # volume_points in z,y,x order; note z is preserved later in visualization
        y = volume_points[i, 1]
        x = volume_points[i, 2]
        A = y - origin_yx[0]
        B = x - origin_yx[1]
        if B < 0:
            theta = 0.0
        else:
            theta = math.atan2(A, B)
        theta_shifted = theta + math.pi / 2  # shift so that theta spans [0, π]
        s = max_r * theta_shifted
        projected[i, 0] = s
        projected[i, 1] = x
    return projected

# -----------------------------------------------------------------------------
# Compute distortion vectors.
# For the mapping F: (y, x) -> (s, x) with
#    s = max_r*(arctan2(y - origin_y, x - origin_x) + π/2)
# the analytical derivatives are:
#    ds/dy = max_r*(B/(A^2+B^2))   and   ds/dx = max_r*(-A/(A^2+B^2))
# We define the 2D distortion vector as:
#    [ ds/dy - 1,  ds/dx ]
# -----------------------------------------------------------------------------
@njit(parallel=True)
def compute_distortion_vectors(volume_points, origin_yx, max_r):
    N = volume_points.shape[0]
    distortions = np.empty((N, 2), dtype=np.float64)
    for i in prange(N):
        y = volume_points[i, 1]
        x = volume_points[i, 2]
        A = y - origin_yx[0]
        B = x - origin_yx[1]
        denom = A * A + B * B
        if denom == 0:
            distortions[i, 0] = 0.0
            distortions[i, 1] = 0.0
        else:
            dtheta_dy = B / denom
            dtheta_dx = -A / denom
            ds_dy = max_r * dtheta_dy
            ds_dx = max_r * dtheta_dx
            distortions[i, 0] = ds_dy - 1.0  # deviation from identity in y direction
            distortions[i, 1] = ds_dx
    return distortions

# -----------------------------------------------------------------------------
# Interpolate distortion vectors onto a regular 2D grid.
# The final distortion map has shape:
#    (n_theta, x_size, 2)
# where n_theta = int(π * max_r) and x_size comes from volume_shape_zyx.
# For each projected point we determine its grid cell (using s and x) and average.
# -----------------------------------------------------------------------------
def interpolate_to_grid(projected, distortions, volume_shape_zyx, max_r):
    n_theta = int(np.pi * max_r)
    x_size = volume_shape_zyx[2]
    grid = np.zeros((n_theta, x_size, 2), dtype=np.float64)
    count = np.zeros((n_theta, x_size), dtype=np.int32)
    
    N = projected.shape[0]
    for i in range(N):
        s = projected[i, 0]
        x = projected[i, 1]
        theta_idx = int(s)
        if theta_idx < 0 or theta_idx >= n_theta:
            continue
        x_idx = int(x)
        if x_idx < 0 or x_idx >= x_size:
            continue
        grid[theta_idx, x_idx, 0] += distortions[i, 0]
        grid[theta_idx, x_idx, 1] += distortions[i, 1]
        count[theta_idx, x_idx] += 1

    # Average distortion vectors in each grid cell
    for i in range(n_theta):
        for j in range(x_size):
            if count[i, j] > 0:
                grid[i, j, 0] /= count[i, j]
                grid[i, j, 1] /= count[i, j]
    return grid

# -----------------------------------------------------------------------------
# Main distortion map computation.
# Given volume_points (in zyx), origin_yx, max_r, and volume_shape_zyx,
# this function computes the cylindrical projection, the local distortion vectors,
# and then interpolates them into a 2D grid.
# -----------------------------------------------------------------------------
def compute_distortion_map(volume_points_zyx, origin_yx, max_r, volume_shape_zyx):
    projected = cylindrical_projection(volume_points_zyx, origin_yx, max_r)
    distortions = compute_distortion_vectors(volume_points_zyx, origin_yx, max_r)
    grid = interpolate_to_grid(projected, distortions, volume_shape_zyx, max_r)
    return grid

# -----------------------------------------------------------------------------
# For visualization: Project points onto the cylindrical surface.
# For each volume point (z,y,x), compute the corresponding point on the surface of
# a cylinder with fixed radius max_r (centered at origin_x,origin_y) by computing:
#   theta = arctan2(y-origin_y, x-origin_x) (with B>=0 check),
#   theta_shifted = theta + π/2,
# and then convert to Cartesian coordinates:
#   x_proj = origin_x + max_r*cos(theta_shifted)
#   y_proj = origin_y + max_r*sin(theta_shifted)
# The z coordinate remains unchanged.
# The returned array is in (x, y, z) order for plotting.
# -----------------------------------------------------------------------------
def project_to_cylinder_surface(volume_points, origin_yx, max_r):
    N = volume_points.shape[0]
    projected_3d = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        z = volume_points[i, 0]
        y = volume_points[i, 1]
        x = volume_points[i, 2]
        A = y - origin_yx[0]
        B = x - origin_yx[1]
        if B < 0:
            theta = 0.0
        else:
            theta = math.atan2(A, B)
        theta_shifted = theta + math.pi / 2
        # Compute projected coordinates on the cylinder surface:
        x_proj = origin_yx[1] + max_r * math.cos(theta_shifted)
        y_proj = origin_yx[0] + max_r * math.sin(theta_shifted)
        projected_3d[i, 0] = x_proj
        projected_3d[i, 1] = y_proj
        projected_3d[i, 2] = z
    return projected_3d

# -----------------------------------------------------------------------------
# Visualization: 3D scatter plot of original and cylinder-projected points.
# The original points (given in z,y,x) are converted to (x,y,z) for display.
# -----------------------------------------------------------------------------
def visualize_projection(volume_points, projected_points):
    # Convert original points from (z,y,x) to (x,y,z)
    pts_original = np.empty_like(projected_points)
    pts_original[:, 0] = volume_points[:, 2]  # x
    pts_original[:, 1] = volume_points[:, 1]  # y
    pts_original[:, 2] = volume_points[:, 0]  # z

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_original[:, 0], pts_original[:, 1], pts_original[:, 2],
               c='b', marker='o', label='Original Points', s=10, alpha=0.6)
    ax.scatter(projected_points[:, 0], projected_points[:, 1], projected_points[:, 2],
               c='r', marker='^', label='Projected Points', s=10, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Original Points vs. Cylinder Projected Points")
    plt.show()

# -----------------------------------------------------------------------------
# Visualization: Heatmap of the distortion map magnitude.
# The distortion map is a (n_theta, x_size, 2) array. The magnitude is computed as:
#    magnitude = sqrt(dist_x^2 + dist_y^2)
# -----------------------------------------------------------------------------
def visualize_distortion_map(distortion_map):
    mag = np.sqrt(distortion_map[:, :, 0]**2 + distortion_map[:, :, 1]**2)
    plt.figure()
    plt.imshow(mag, aspect='auto', origin='lower', cmap='hot')
    plt.colorbar(label='Distortion Magnitude')
    plt.title("Distortion Map Magnitude")
    plt.xlabel("X index")
    plt.ylabel("Angular Theta index")
    plt.show()

# -----------------------------------------------------------------------------
# Test example:
#
# Generate points on a half sphere (hemisphere) of radius 50 that sits at the
# bottom middle of a volume with shape ZYX = (90, 70, 200). Here, we choose the
# hemisphere to be the set of points on a sphere with center at (z, y, x) = (89, 35, 100)
# (i.e. bottom middle) that lie above the center (i.e. with z <= 89).
# -----------------------------------------------------------------------------
def test_example():
    # Volume dimensions: ZYX = (90, 70, 200)
    volume_shape_zyx = (90, 70, 200)
    num_points = 1000
    # For a hemisphere that “sits” on the bottom, we set the center at the bottom middle.
    # Here we use center = (z, y, x) with z = last slice, y and x at half the size.
    center = np.array([volume_shape_zyx[0] - 1, volume_shape_zyx[1] // 2, volume_shape_zyx[2] // 2], dtype=np.float64)
    radius = 50.0

    # Sample points uniformly on the hemisphere.
    # Standard spherical coordinates: for a full sphere, z = center_z + radius*cos(theta).
    # To sample only the hemisphere with z <= center_z, require cos(theta) <= 0.
    # Let u be uniform in [0, 1] and set cos(theta) = -u, so theta in [π/2, π].
    u = np.random.rand(num_points)
    theta = np.arccos(-u)  # theta in [pi/2, pi]
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    
    # Spherical to Cartesian conversion.
    # Standard formulas: 
    #   x_offset = radius * sin(theta)*cos(phi)
    #   y_offset = radius * sin(theta)*sin(phi)
    #   z_offset = radius * cos(theta)
    x_offsets = radius * np.sin(theta) * np.cos(phi)
    y_offsets = radius * np.sin(theta) * np.sin(phi)
    z_offsets = radius * np.cos(theta)
    
    # Compute volume points in (z, y, x) order.
    volume_points = np.empty((num_points, 3), dtype=np.float64)
    volume_points[:, 0] = center[0] + z_offsets   # z
    volume_points[:, 1] = center[1] + y_offsets   # y
    volume_points[:, 2] = center[2] + x_offsets   # x

    # Define projection origin as (origin_y, origin_x) using center y and x.
    origin_yx = (center[1], center[2])
    
    # Compute the distortion map (using the unwrapped projection functions).
    distortion_map = compute_distortion_map(volume_points, origin_yx, radius, volume_shape_zyx)
    
    # Compute the 3D projected points on the cylinder surface (for visualization).
    projected_points = project_to_cylinder_surface(volume_points, origin_yx, radius)
    
    # Visualize the original and projected points.
    visualize_projection(volume_points, projected_points)
    
    # Visualize the distortion map as a heatmap.
    visualize_distortion_map(distortion_map)

# -----------------------------------------------------------------------------
# Run the test example if this module is executed as the main script.
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    test_example()
