import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@njit(parallel=True)
def project_points_to_cylinder(volume_points_zyx, origin_yx, max_r):
    n_points = volume_points_zyx.shape[0]
    projected_points = np.zeros_like(volume_points_zyx)
    
    for i in prange(n_points):
        z = volume_points_zyx[i, 0]
        y = volume_points_zyx[i, 1]
        x = volume_points_zyx[i, 2]
        dy = y - origin_yx[0]
        dx = x - origin_yx[1]
        theta = np.arctan2(dy, dx)
        projected_y = np.float32(max_r * np.sin(theta) + origin_yx[0])
        projected_x = np.float32(max_r * np.cos(theta) + origin_yx[1])
        projected_points[i, 0] = z
        projected_points[i, 1] = projected_y
        projected_points[i, 2] = projected_x
    return projected_points

@njit(parallel=True)
def calculate_distortion_map(volume_points_zyx, projected_points_zyx, volume_shape_zyx, max_r, origin_yx):
    num_theta = int(np.pi * max_r)
    x_size = volume_shape_zyx[2]
    distortion_map = np.zeros((num_theta, x_size, 2), dtype=np.float32)
    count_map = np.zeros((num_theta, x_size), dtype=np.int32)

    for i in prange(len(volume_points_zyx)):
        z0 = volume_points_zyx[i, 0]
        y0 = volume_points_zyx[i, 1]
        x0 = volume_points_zyx[i, 2]
        z1 = projected_points_zyx[i, 0]
        y1 = projected_points_zyx[i, 1]
        x1 = projected_points_zyx[i, 2]

        dy = y1 - y0
        dx = x1 - x0

        angle = np.arctan2(y0 - origin_yx[0], x0 - origin_yx[1])
        theta_index = int(((angle + np.pi) / (2 * np.pi)) * num_theta)

        if 0 <= theta_index < num_theta and 0 <= int(x0) < x_size:
            distortion_map[theta_index, int(x0), 0] += dx
            distortion_map[theta_index, int(x0), 1] += dy
            count_map[theta_index, int(x0)] += 1

    for i in prange(num_theta):
        for j in range(x_size):
            if count_map[i, j] > 0:
                distortion_map[i, j, 0] /= count_map[i, j]
                distortion_map[i, j, 1] /= count_map[i, j]

    return distortion_map

def compute_distortion_module(volume_points_zyx, origin_yx, max_r, volume_shape_zyx):
    volume_points_zyx = np.array(volume_points_zyx, dtype=np.float32)
    projected_points = project_points_to_cylinder(volume_points_zyx, origin_yx, max_r)
    distortion_map = calculate_distortion_map(volume_points_zyx, projected_points, volume_shape_zyx, max_r, origin_yx)
    return distortion_map

def visualize_distortion(distortion_map):
    """
    Visualizes distortion map as:
    - Heatmaps for X and Y distortion magnitude.
    - Vector field (quiver plot) for X and Y directions.
    """
    theta_size, x_size, _ = distortion_map.shape
    x = np.arange(x_size)
    theta = np.linspace(0, 2 * np.pi, theta_size)

    X, Theta = np.meshgrid(x, theta)

    distortion_x = distortion_map[:, :, 0]
    distortion_y = distortion_map[:, :, 1]
    magnitude = np.sqrt(distortion_x**2 + distortion_y**2)

    # Plot heatmap of magnitude
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Distortion Magnitude Heatmap")
    plt.imshow(magnitude, aspect='auto', origin='lower', extent=[0, x_size, 0, 2*np.pi])
    plt.xlabel("X axis")
    plt.ylabel("Theta (radians)")
    plt.colorbar(label="Distortion Magnitude")

    # Plot quiver vector field
    plt.subplot(1, 2, 2)
    plt.title("Distortion Vector Field")
    skip = (slice(None, None, theta_size // 30), slice(None, None, x_size // 30))  # downsample for readability
    plt.quiver(X[skip], Theta[skip], distortion_x[skip], distortion_y[skip], angles='xy', scale_units='xy', scale=1)
    plt.xlabel("X axis")
    plt.ylabel("Theta (radians)")
    plt.tight_layout()
    plt.show()

def generate_half_sphere_points(radius=50, num_points=1000, center=(0, 0, 0)):
    """
    Generate sparse points on a half sphere (flat side parallel to XY plane, pointing in +Z).
    """
    points = []
    for _ in range(num_points):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(np.pi / 2, np.pi)  # upper hemisphere only
        x = radius * np.sin(phi) * np.cos(theta) + center[2]
        y = radius * np.sin(phi) * np.sin(theta) + center[1]
        z = radius * np.cos(phi) + center[0]
        points.append((z, y, x))  # ZYX format
    return np.array(points, dtype=np.float32)

def visualize_3d_points(volume_points_zyx, volume_shape_zyx=None, title="Original 3D Points"):
    """
    Visualize the original 3D surface points inside the 3D volume space.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    z = volume_points_zyx[:, 0]
    y = volume_points_zyx[:, 1]
    x = volume_points_zyx[:, 2]

    ax.scatter(x, y, z, c='blue', s=1, alpha=0.5)

    if volume_shape_zyx is not None:
        max_lim = max(volume_shape_zyx)
        ax.set_xlim([0, max_lim])
        ax.set_ylim([0, max_lim])
        ax.set_zlim([0, max_lim])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Update test function to include 3D visualization
def test_distortion_on_half_sphere():
    volume_shape_zyx = (70, 70, 200)
    max_r = 50
    origin_yx = (volume_shape_zyx[1] // 2, volume_shape_zyx[2] // 2)

    # Generate half sphere and embed at the bottom of the volume
    half_sphere_points = generate_half_sphere_points(radius=max_r, num_points=1000,
                                                     center=(volume_shape_zyx[0], origin_yx[0], origin_yx[1]))

    # Visualize original points
    visualize_3d_points(half_sphere_points, volume_shape_zyx)

    # # Compute distortion map
    # distortion_map = compute_distortion_module(half_sphere_points, origin_yx, max_r, volume_shape_zyx)

    # # Visualize distortion maps
    # visualize_distortion(distortion_map)

# Re-run updated test
test_distortion_on_half_sphere()
