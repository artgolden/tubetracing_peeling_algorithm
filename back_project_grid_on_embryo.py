import numpy as np
import trimesh
import torch
import matplotlib.pyplot as plt
import tifffile as tiff

from pytorch3d.structures import Meshes
from trimesh.ray.ray_pyembree import RayMeshIntersector


def visualize_3d_points(volume_points_zyx, volume_shape_zyx=None, highlighted_points_idx=None, extra_points_zyx=None, mesh=None, title="Original 3D Points"):
    """
    Visualize the original 3D surface points inside the 3D volume space.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    z = volume_points_zyx[:, 0]
    y = volume_points_zyx[:, 1]
    x = volume_points_zyx[:, 2]

    ax.scatter(x, y, z, c='purple', s=3, alpha=0.5)
    if highlighted_points_idx is not None:
        ax.scatter(x[highlighted_points_idx], 
                y[highlighted_points_idx], 
                z[highlighted_points_idx], 
                c='red', s=10, alpha=0.9)
    if extra_points_zyx is not None:
        ax.scatter(extra_points_zyx[:, 2], extra_points_zyx[:, 1], extra_points_zyx[:, 0], c='green', s=3, alpha=0.7)
    

    if volume_shape_zyx is not None:
        ax.set_xlim([0, volume_shape_zyx[2] - 1])
        ax.set_ylim([0, volume_shape_zyx[1] - 1])
        ax.set_zlim([0, volume_shape_zyx[0] - 1])

    ax.set_aspect('equal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if mesh is not None:
        ax.plot_trisurf(mesh.vertices[:, 2], mesh.vertices[:, 1], mesh.vertices[:, 0], triangles=mesh.faces, color='gray', alpha=0.3)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def load_mesh_from_point_cloud(npy_path, vol_shape):
    """
    Load a point cloud from a .npy file and compute the convex hull mesh.
    Returns a Trimesh mesh.
    """
    points = np.load(npy_path)[:, [2, 1, 0]]
    points[:,[0]] = vol_shape[0] - points[:,[0]] -1 # Flip z axis
    cloud = trimesh.points.PointCloud(points)
    mesh = cloud.convex_hull
    return mesh, points


def perform_ray_mesh_intersection(mesh, ray_origins, ray_directions):
    """
    Use trimesh's RayMeshIntersector to compute ray-mesh intersections.
    Returns an array of hit points with same shape as ray_origins, with NaNs for no-hit rays.
    """
    intersector = RayMeshIntersector(mesh)
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins.cpu().numpy(), ray_directions.cpu().numpy(), multiple_hits=False
    )

    result = np.full((ray_origins.shape[0], 3), np.nan, dtype=np.float32)
    result[index_ray] = locations
    return result


def sparse_grid_on_half_cylinder(
    image_shape,
    spacing_x=7,
    spacing_theta=7,
    radius=1.0,
    origin_yz=(0.0, 0.0)
):
    height, width = image_shape
    y0, z0 = origin_yz

    num_points_theta = max(1, int(width) // spacing_theta)
    num_points_x = max(1, int((height) // spacing_x))

    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points_theta)
    x = np.arange(0, num_points_x * spacing_x, spacing_x)

    u_coords = np.round(np.linspace(0, width, num_points_theta))
    uu, vv = np.meshgrid(u_coords, x)
    uv_grid = np.stack((uu.ravel(), vv.ravel()), axis=-1)


    theta_grid, x_grid = np.meshgrid(theta, x)

    z = radius * np.cos(theta_grid) + z0
    y = radius * np.sin(theta_grid) + y0

    points_3d = np.column_stack([z.ravel(), y.ravel(), x_grid.ravel()])
    uv_grid_shape = (num_points_theta, num_points_x)
    return points_3d, uv_grid, uv_grid_shape

def visualize_uv_grid(uv_grid, uv_grid_highlighted=None, title="UV Grid Visualization"):
    """
    Plot the UV grid with optional highlighted points.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(uv_grid[:, 0], uv_grid[:,   1], c='blue', s=5, label='UV Grid')
    if uv_grid_highlighted is not None:
        plt.scatter(uv_grid_highlighted[:, 0], uv_grid_highlighted[:, 1], c='red', s=20, label='Highlighted')
    plt.xlabel('U')
    plt.ylabel('V')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def compute_avg_neighbor_distances(hit_points_3d: np.ndarray, shape_2d: tuple) -> tuple:
    """
    Given a 1D array of 3D points and a desired 2D shape (rows, columns),
    reshape the points into a grid and compute two 2D matrices representing the average
    distances to vertical (top and bottom) and horizontal (left and right) neighbors.

    For each point in the internal grid (excluding borders), the average distance is:
       - Vertical: 0.5 * (distance to top neighbor + distance to bottom neighbor)
       - Horizontal: 0.5 * (distance to left neighbor + distance to right neighbor)
    If the point or any of its required neighbors contains NaN, the result is set to NaN.

    Parameters:
        hit_points_3d (np.ndarray): Array of 3D points of shape (N, 3).
        shape_2d (tuple): Tuple (rows, cols) that specifies the 2D arrangement.
    
    Returns:
        vertical_avg (np.ndarray): 2D array with vertical average distances.
        horizontal_avg (np.ndarray): 2D array with horizontal average distances.
    """
    rows, cols = shape_2d
    if hit_points_3d.shape[0] != rows * cols:
        raise ValueError("The number of 3D points does not match the provided 2D shape.")

    # Reshape to (rows, cols, 3)
    grid = hit_points_3d.reshape(rows, cols, 3)

    # Initialize output arrays with NaNs
    vertical_avg = np.full((rows, cols), np.nan, dtype=np.float64)
    horizontal_avg = np.full((rows, cols), np.nan, dtype=np.float64)

    # Compute vertical average distances for internal rows (exclude first and last row)
    if rows > 2:
        # Slicing for central elements and their neighbors
        center_vertical = grid[1:-1, :, :]
        top_neighbors = grid[0:-2, :, :]
        bottom_neighbors = grid[2:, :, :]

        # Create valid masks: ensure that none of the points have any NaNs
        valid_mask_vertical = (~np.isnan(center_vertical).any(axis=2) &
                                ~np.isnan(top_neighbors).any(axis=2) &
                                ~np.isnan(bottom_neighbors).any(axis=2))

        # Compute Euclidean distances
        diff_top = np.linalg.norm(center_vertical - top_neighbors, axis=2)
        diff_bottom = np.linalg.norm(center_vertical - bottom_neighbors, axis=2)
        avg_vert = 0.5 * (diff_top + diff_bottom)

        # Assign computed averages only where all involved points are valid
        vertical_avg[1:-1, :] = np.where(valid_mask_vertical, avg_vert, np.nan)

    # Compute horizontal average distances for internal columns (exclude first and last column)
    if cols > 2:
        center_horizontal = grid[:, 1:-1, :]
        left_neighbors = grid[:, 0:-2, :]
        right_neighbors = grid[:, 2:, :]

        valid_mask_horizontal = (~np.isnan(center_horizontal).any(axis=2) &
                                  ~np.isnan(left_neighbors).any(axis=2) &
                                  ~np.isnan(right_neighbors).any(axis=2))

        diff_left = np.linalg.norm(center_horizontal - left_neighbors, axis=2)
        diff_right = np.linalg.norm(center_horizontal - right_neighbors, axis=2)
        avg_horiz = 0.5 * (diff_left + diff_right)

        horizontal_avg[:, 1:-1] = np.where(valid_mask_horizontal, avg_horiz, np.nan)

    return vertical_avg, horizontal_avg


def visualize_distance_heatmaps(vertical_avg: np.ndarray, horizontal_avg: np.ndarray,
                                title_vertical: str = "Vertical Neighbor Avg Distance",
                                title_horizontal: str = "Horizontal Neighbor Avg Distance"):
    """
    Visualize the two 2D matrices (vertical and horizontal average distances) as heatmaps.

    Parameters:
        vertical_avg (np.ndarray): 2D array with vertical average distances.
        horizontal_avg (np.ndarray): 2D array with horizontal average distances.
        title_vertical (str): Title for the vertical heatmap.
        title_horizontal (str): Title for the horizontal heatmap.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    im_v = axs[0].imshow(vertical_avg, interpolation='nearest', aspect='auto')
    axs[0].set_title(title_vertical)
    fig.colorbar(im_v, ax=axs[0])
    
    im_h = axs[1].imshow(horizontal_avg, interpolation='nearest', aspect='auto')
    axs[1].set_title(title_horizontal)
    fig.colorbar(im_h, ax=axs[1])
    
    plt.tight_layout()
    plt.show()


# Get image volume shape
vol_shape = tiff.imread("outs/down_cropped_tp_300.tif").shape
print(f"Volume shape: {vol_shape}")
max_r = round(vol_shape[1] / 2.0 * 1.15)

# Load mesh from point cloud .npy file
mesh, point_cloud = load_mesh_from_point_cloud("outs/hull_embryo_surface_points.npy", vol_shape)


# Generate grid points on half-cylinder surface
image_shape = (vol_shape[2], round(np.pi * max_r + 1))
cylinder_radius = max_r
spacing_x = 10
spacing_theta = 10
cylinder_points_zyx, uv_grid, uv_grid_shape = sparse_grid_on_half_cylinder(
    image_shape=image_shape,
    spacing_x=spacing_x,
    spacing_theta=spacing_theta,
    radius=cylinder_radius,
    origin_yz = (vol_shape[1]//2, 0)
)

# Convert ZYX to XYZ for processing
surface_points = torch.tensor(cylinder_points_zyx, dtype=torch.float32)

# Ray directions and origins
ray_origins = surface_points.clone()
ray_origins[:, 1] = vol_shape[1] // 2  # Set Y to center of volume
ray_origins[:, 0] = 0  # Set Z to 0
ray_directions = torch.nn.functional.normalize(surface_points - ray_origins, dim=1)

# Perform ray-mesh intersection
hit_points = perform_ray_mesh_intersection(mesh, ray_origins, ray_directions)

# Print or export intersection points
# print("Total Intersections:", hit_points.shape[0])
print("Surface Points: ", surface_points.shape[0])
hit_points_3d = np.array(hit_points)


visualize_3d_points(hit_points_3d[1000:1002], extra_points_zyx=cylinder_points_zyx[1000:1002], mesh=mesh, volume_shape_zyx=vol_shape) # #FFFFFFFFFFFFFF flip vol_shape coord order or something??

print(f"uv_grid shape: {uv_grid_shape} uv_grid num points: {uv_grid.shape[0]}")
uv_grid_highlighted = uv_grid[1000:1002]  # example highlighted indices
visualize_uv_grid(uv_grid, uv_grid_highlighted)

# ---- New Code for Neighbor Distance Computations and Visualization ---- #

# Example: Arrange hit_points_3d into a 2D matrix with dimensions matching the uv_grid shape.
# It is assumed that the number of hit points is equal to rows * cols (uv_grid shape)
cols, rows = uv_grid_shape
# Alternatively, set shape_2d explicitly if known; for example:
# shape_2d = (num_rows, num_cols)
# Here we assume the original grid dimensions are known (for example, from the sparse grid generation).
# As a simple example, let's use the dimensions from the uv_grid if it was constructed as a mesh grid:
# For the sparse grid generated above, uv_grid is of shape (num_points, 2); you may need to determine the actual rows and cols.
# For demonstration purposes, assume rows = number of x positions and cols = number of theta positions.
# Since x positions were generated using:
#    x = np.arange(0, num_points_x * spacing_x, spacing_x)
# and theta positions using:
#    theta = np.linspace(-np.pi/2, np.pi/2, num_points_theta)
# Their meshgrid produces a uv_grid of shape (num_points_x, num_points_theta).
# We can extract these dimensions from the sparse grid generation:
shape_2d = (rows, cols)

# Compute the average neighbor distances
vertical_avg, horizontal_avg = compute_avg_neighbor_distances(hit_points_3d, shape_2d)

# Visualize the computed heatmaps for average distances
visualize_distance_heatmaps(vertical_avg, horizontal_avg)