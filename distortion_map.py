"""
A module for distortion map calculation of the mapping of points 
on embryo surface mesh to the 2D surface of half a cylinder.

This module provides functions to:
  - Generate a mesh from a point cloud.
  - Compute ray-mesh intersections.
  - Generate a sparse grid on a half-cylinder surface.
  - Compute average neighbor distances and interpolate missing data.
  - Resize distortion maps using linear interpolation.

It also defines a wrapper function `process_embryo_data` that,
given the input parameters, runs the full processing pipeline and reports the total execution time.
"""

from typing import Tuple, Dict, Any
import numpy as np
import trimesh
from scipy.interpolate import griddata
from trimesh.ray.ray_pyembree import RayMeshIntersector


def mesh_from_point_cloud(points: np.ndarray, vol_shape: Tuple[int, int, int]) -> trimesh.Trimesh:
    """
    Convert a point cloud to a convex hull mesh.
    
    Parameters:
        points (np.ndarray): Array of 3D points with shape (N, 3).
        vol_shape (tuple): The volume shape as (Z, Y, X).
    
    Returns:
        trimesh.Trimesh: The convex hull mesh generated from the point cloud.
    """
    points[:, [0]] = vol_shape[0] - points[:, [0]] - 1  # Flip the Z-axis.
    cloud = trimesh.points.PointCloud(points)
    mesh = cloud.convex_hull
    return mesh


def perform_ray_mesh_intersection(mesh: trimesh.Trimesh,
                                  ray_origins: np.ndarray,
                                  ray_directions: np.ndarray) -> np.ndarray:
    """
    Compute ray-mesh intersections using trimesh's RayMeshIntersector.
    
    Parameters:
        mesh (trimesh.Trimesh): The mesh to intersect rays with.
        ray_origins (np.ndarray): Array of ray origin points with shape (N, 3).
        ray_directions (np.ndarray): Array of ray direction vectors with shape (N, 3).
    
    Returns:
        np.ndarray: Array of hit points with shape (N, 3); rays with no hit are marked with NaNs.
    """
    intersector = RayMeshIntersector(mesh)
    locations, index_ray, _ = intersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False
    )
    result = np.full((ray_origins.shape[0], 3), np.nan, dtype=np.float32)
    result[index_ray] = locations
    return result


def sparse_grid_on_half_cylinder(
    image_shape: Tuple[int, int],
    num_points_theta: int,
    num_points_x: int,
    radius: float = 1.0,
    origin_yz: Tuple[float, float] = (0.0, 0.0)
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Generate a sparse grid on a half-cylinder surface.
    
    Parameters:
        image_shape (tuple): The shape (height, width) of the projection image.
        num_points_theta (int): Number of angular (theta) samples along the half-cylinder.
        num_points_x (int): Number of samples along the x-axis.
        radius (float): Radius of the cylinder.
        origin_yz (tuple): The (y, z) coordinates of the cylinder's origin.
    
    Returns:
        Tuple containing:
          - points_3d (np.ndarray): Array of 3D points on the cylinder surface in ZYX order.
          - uv_grid (np.ndarray): Array of corresponding UV grid coordinates.
          - uv_grid_shape (tuple): The shape (num_points_theta, num_points_x) of the UV grid.
    """
    height, width = image_shape
    y0, z0 = origin_yz

    # Angular samples along the half-cylinder.
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points_theta)
    # X-axis samples evenly along the image height.
    x = np.linspace(0, height, num_points_x)

    # Create a grid for the UV coordinates.
    u_coords = np.linspace(0, width, num_points_theta)
    uu, vv = np.meshgrid(u_coords, x)
    uv_grid = np.stack((uu.ravel(), vv.ravel()), axis=-1)

    # Create grid of theta and x values for generating 3D points.
    theta_grid, x_grid = np.meshgrid(theta, x)
    z = radius * np.cos(theta_grid) + z0
    y = radius * np.sin(theta_grid) + y0

    points_3d = np.column_stack([z.ravel(), y.ravel(), x_grid.ravel()])
    uv_grid_shape = (num_points_theta, num_points_x)
    
    return points_3d, uv_grid, uv_grid_shape


def compute_avg_neighbor_distances(hit_points_3d: np.ndarray,
                                   shape_2d: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average distances to vertical and horizontal neighbors on a grid.
    
    For each internal grid point (excluding borders):
      - Vertical average = 0.5 * (distance to top neighbor + distance to bottom neighbor)
      - Horizontal average = 0.5 * (distance to left neighbor + distance to right neighbor)
    
    If the point or any of its neighbors contains NaN, the result is set to NaN.
    
    Parameters:
        hit_points_3d (np.ndarray): Array of 3D points with shape (N, 3).
        shape_2d (tuple): The 2D grid shape as (rows, cols).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 2D arrays (vertical_avg, horizontal_avg) with computed distances.
    """
    rows, cols = shape_2d
    if hit_points_3d.shape[0] != rows * cols:
        raise ValueError("The number of 3D points does not match the provided 2D shape.")

    grid = hit_points_3d.reshape(rows, cols, 3)
    vertical_avg = np.full((rows, cols), np.nan, dtype=np.float64)
    horizontal_avg = np.full((rows, cols), np.nan, dtype=np.float64)

    # Compute vertical averages (excluding the first and last rows).
    if rows > 2:
        center_vertical = grid[1:-1, :, :]
        top_neighbors = grid[0:-2, :, :]
        bottom_neighbors = grid[2:, :, :]
        valid_mask_vertical = (~np.isnan(center_vertical).any(axis=2) &
                                 ~np.isnan(top_neighbors).any(axis=2) &
                                 ~np.isnan(bottom_neighbors).any(axis=2))
        diff_top = np.linalg.norm(center_vertical - top_neighbors, axis=2)
        diff_bottom = np.linalg.norm(center_vertical - bottom_neighbors, axis=2)
        avg_vert = 0.5 * (diff_top + diff_bottom)
        vertical_avg[1:-1, :] = np.where(valid_mask_vertical, avg_vert, np.nan)

    # Compute horizontal averages (excluding the first and last columns).
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


def interpolate_nan_elements(matrix: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN elements in a 2D matrix using linear interpolation.
    
    Parameters:
        matrix (np.ndarray): 2D array containing numeric values and NaNs.
    
    Returns:
        np.ndarray: A 2D array with NaN elements replaced by interpolated values.
    
    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input is not 2-dimensional or contains no valid data.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")

    valid_mask = ~np.isnan(matrix)
    if not np.any(valid_mask):
        raise ValueError("The input matrix contains no valid data for interpolation.")

    rows, cols = matrix.shape
    grid_x, grid_y = np.mgrid[0:rows, 0:cols]
    valid_coords = np.array(np.nonzero(valid_mask)).T
    valid_values = matrix[valid_mask]

    interpolated_matrix = griddata(valid_coords, valid_values, (grid_x, grid_y), method='linear')
    return interpolated_matrix


def interpolate_nans_horizontally(matrix: np.ndarray, max_interpolation_distance: int = 5) -> np.ndarray:
    """
    Interpolate NaN elements row-wise using linear interpolation based on horizontal neighbors.
    
    Only contiguous segments of NaNs that are bounded by valid numbers and whose gap size is
    less than or equal to max_interpolation_distance are interpolated.
    
    Parameters:
        matrix (np.ndarray): 2D array of floats, possibly containing NaNs.
        max_interpolation_distance (int): Maximum number of consecutive NaNs to interpolate (default is 5).
    
    Returns:
        np.ndarray: A new 2D array with eligible NaNs replaced by interpolated values.
    """
    interpolated = matrix.copy()
    n_rows, n_cols = interpolated.shape

    for row_idx in range(n_rows):
        row = interpolated[row_idx]
        valid_indices = np.where(~np.isnan(row))[0]
        if valid_indices.size < 2:
            continue
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            gap_size = end_idx - start_idx - 1
            if gap_size > 0 and gap_size <= max_interpolation_distance:
                start_value = row[start_idx]
                end_value = row[end_idx]
                interpolated_values = np.linspace(start_value, end_value, num=gap_size + 2)[1:-1]
                row[start_idx + 1:end_idx] = interpolated_values
    return interpolated


def resize_distortion_map(matrix: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize a 2D distortion map to a target shape using linear interpolation.
    
    Points in the target grid falling outside the convex hull of valid data remain NaN.
    
    Parameters:
        matrix (np.ndarray): 2D array representing the distortion map.
        target_shape (tuple): Desired shape as (target_rows, target_cols).
    
    Returns:
        np.ndarray: Resized 2D array with interpolated values.
    """
    orig_rows, orig_cols = matrix.shape
    target_rows, target_cols = target_shape
    grid_orig = np.mgrid[0:orig_rows, 0:orig_cols]
    coords_orig = np.stack((grid_orig[0].ravel(), grid_orig[1].ravel()), axis=-1)
    values = matrix.ravel()

    valid_mask = ~np.isnan(values)
    coords_valid = coords_orig[valid_mask]
    values_valid = values[valid_mask]

    target_x = np.linspace(0, orig_rows - 1, target_rows)
    target_y = np.linspace(0, orig_cols - 1, target_cols)
    grid_target = np.meshgrid(target_x, target_y, indexing='ij')
    coords_target = np.stack((grid_target[0].ravel(), grid_target[1].ravel()), axis=-1)

    interpolated_values = griddata(coords_valid, values_valid, coords_target, method='linear')
    resized_matrix = interpolated_values.reshape(target_shape)
    return resized_matrix

def calculate_distortion_map(
    embryo_vol_shape: Tuple[int, int, int],
    cylinder_radius: float,
    approx_spacing_x: int,
    approx_spacing_theta: int,
    point_cloud: np.ndarray,
    full_size_projection_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculating distortion map of cylindrical cartography projection.
    Calculate distortion maps along vertical and horizontal directions for the mapping of points 
    on embryo surface mesh to the surface of half of a cylinder above the embryo.
    
    This function performs the following steps:
      1. Generates a sparse grid on a half-cylinder surface.
      2. Creates a convex hull mesh from the given point cloud.
      3. Computes ray origins, directions, and performs ray-mesh intersections.
      4. Computes average neighbor distances on the grid.
      5. Interpolates missing data in the distance matrices.
      6. Resizes the distortion maps based on UV grid sampling rates.
      7. Reports the total execution time.
    
    Parameters:
        embryo_vol_shape (tuple): The embryo volume shape (Z, Y, X).
        cylinder_radius (float): The cylinder radius for projection.
        approx_spacing_x (int): Approximate spacing along the X-axis.
        approx_spacing_theta (int): Approximate angular spacing.
        point_cloud (np.ndarray): Array of 3D points representing the embryo surface.
        full_size_projection_shape (tuple): Projection image shape as (width, height).
    
    Returns:
        tuple:
            - vertical_distortion (np.ndarray): A matrix with distortion factors along vertical axis. Calculated where possible otherwise NaN.
            - horizontal_distortion (np.ndarray): A matrix with distortion factors along horizontal axis. Calculated where possible otherwise NaN.
    """

    # Determine grid resolution.
    num_points_theta = full_size_projection_shape[1] // approx_spacing_theta
    num_points_x = full_size_projection_shape[0] // approx_spacing_x
    spacing_u = full_size_projection_shape[1] / num_points_theta
    spacing_v = full_size_projection_shape[0] / num_points_x

    # Generate sparse grid on the half-cylinder surface.
    cylinder_points_zyx, uv_grid, uv_grid_shape = sparse_grid_on_half_cylinder(
        image_shape=full_size_projection_shape,
        num_points_theta=num_points_theta,
        num_points_x=num_points_x,
        radius=cylinder_radius,
        origin_yz=(embryo_vol_shape[1] // 2, 0)
    )

    # Create a mesh from the point cloud.
    mesh = mesh_from_point_cloud(point_cloud, embryo_vol_shape)

    # Set up ray origins and directions.
    surface_points = cylinder_points_zyx.astype(np.float32)
    ray_origins = surface_points.copy()
    ray_origins[:, 1] = embryo_vol_shape[1] // 2  # Center Y
    ray_origins[:, 0] = 0  # Set Z to 0

    vecs = surface_points - ray_origins
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    ray_directions = vecs / norms

    hit_points = perform_ray_mesh_intersection(mesh, ray_origins, ray_directions)

    # Prepare grid shape for neighbor calculations.
    cols, rows = uv_grid_shape
    shape_2d = (rows, cols)

    vertical_avg, horizontal_avg = compute_avg_neighbor_distances(hit_points, shape_2d)
    vertical_avg = interpolate_nans_horizontally(vertical_avg)
    vertical_avg = interpolate_nan_elements(vertical_avg)
    horizontal_avg = interpolate_nans_horizontally(horizontal_avg)
    horizontal_avg = interpolate_nan_elements(horizontal_avg)

    horizontal_distortion = resize_distortion_map(spacing_u / horizontal_avg, full_size_projection_shape)
    vertical_distortion = resize_distortion_map(spacing_v / vertical_avg, full_size_projection_shape)

    return vertical_distortion, horizontal_distortion
