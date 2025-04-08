#!/usr/bin/env python
"""
Main script to test distortion map calculation and visualize distortion maps.

This script reads the embryo volume and point cloud data from files, then uses the core module to process
the data and generate distortion maps. Finally, the visualization module is used to display heatmaps of
the distortion factors. All log output is sent to stdout.
"""

from typing import Tuple, Dict, Any
import logging
import sys
import time
import numpy as np
import tifffile as tiff

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import distortion_map as core
import distortion_map_vis as viz


def calculate_distortion_map_all_outputs(
    embryo_vol_shape: Tuple[int, int, int],
    cylinder_radius: float,
    approx_spacing_x: int,
    approx_spacing_theta: int,
    point_cloud: np.ndarray,
    full_size_projection_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """
    Process embryo data to generate distortion maps from a point cloud.
    
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
        dict: A dictionary containing:
            - uv_grid (np.ndarray): The UV grid coordinates.
            - uv_grid_shape (tuple): Shape of the UV grid (num_points_theta, num_points_x).
            - hit_points (np.ndarray): 3D hit points from ray-mesh intersections.
            - vertical_distortion (np.ndarray): Vertical distortion factor map.
            - horizontal_distortion (np.ndarray): Horizontal distortion factor map.
            - mesh (trimesh.Trimesh): The generated mesh.
            - elapsed_time (float): Total execution time in seconds.
    """
    start_time = time.time()

    # Determine grid resolution.
    num_points_theta = full_size_projection_shape[1] // approx_spacing_theta
    num_points_x = full_size_projection_shape[0] // approx_spacing_x
    spacing_u = full_size_projection_shape[1] / num_points_theta
    spacing_v = full_size_projection_shape[0] / num_points_x

    # Generate sparse grid on the half-cylinder surface.
    cylinder_points_zyx, uv_grid, uv_grid_shape = core.sparse_grid_on_half_cylinder(
        image_shape=full_size_projection_shape,
        num_points_theta=num_points_theta,
        num_points_x=num_points_x,
        radius=cylinder_radius,
        origin_yz=(embryo_vol_shape[1] // 2, 0)
    )

    # Create a mesh from the point cloud.
    mesh = core.mesh_from_point_cloud(point_cloud, embryo_vol_shape)

    # Set up ray origins and directions.
    surface_points = cylinder_points_zyx.astype(np.float32)
    ray_origins = surface_points.copy()
    ray_origins[:, 1] = embryo_vol_shape[1] // 2  # Center Y
    ray_origins[:, 0] = 0  # Set Z to 0

    vecs = surface_points - ray_origins
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    ray_directions = vecs / norms

    hit_points = core.perform_ray_mesh_intersection(mesh, ray_origins, ray_directions)

    # Prepare grid shape for neighbor calculations.
    cols, rows = uv_grid_shape
    shape_2d = (rows, cols)

    vertical_avg, horizontal_avg = core.compute_avg_neighbor_distances(hit_points, shape_2d)
    vertical_avg = core.interpolate_nans_horizontally(vertical_avg)
    vertical_avg = core.interpolate_nan_elements(vertical_avg)
    horizontal_avg = core.interpolate_nans_horizontally(horizontal_avg)
    horizontal_avg = core.interpolate_nan_elements(horizontal_avg)

    horizontal_distortion = core.resize_distortion_map(spacing_u / horizontal_avg, full_size_projection_shape)
    vertical_distortion = core.resize_distortion_map(spacing_v / vertical_avg, full_size_projection_shape)

    elapsed_time = time.time() - start_time

    return {
        "uv_grid": uv_grid,
        "uv_grid_shape": uv_grid_shape,
        "hit_points": hit_points,
        "vertical_distortion": vertical_distortion,
        "horizontal_distortion": horizontal_distortion,
        "mesh": mesh,
        "cylinder_points_zyx": cylinder_points_zyx,
        "elapsed_time": elapsed_time
    }

def main() -> None:
    """
    Main function that executes the data processing and visualization pipeline.
    """
    # Set up logging to output to stdout.
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    start_time = time.time()

    # ===== Input Data =====
    embryo_vol = tiff.imread("outs/down_cropped_tp_300.tif")
    embryo_vol_shape = embryo_vol.shape
    logging.info(f"Volume shape: {embryo_vol_shape}")

    max_r = round(embryo_vol_shape[1] / 2.0 * 1.15)
    cylinder_radius = max_r
    approx_spacing_x = 2
    approx_spacing_theta = 5

    point_cloud = np.load("outs/hull_embryo_surface_points.npy")[:, [2, 1, 0]]
    full_size_projection_shape = (embryo_vol_shape[2], round(np.pi * max_r + 1))

    num_points_theta = full_size_projection_shape[1] // approx_spacing_theta
    num_points_x = full_size_projection_shape[0] // approx_spacing_x
    spacing_u = full_size_projection_shape[1] / num_points_theta
    spacing_v = full_size_projection_shape[0] / num_points_x
    logging.info(f"Spacing u: {spacing_u}, Spacing v: {spacing_v}")

    # Process embryo data using the core module.
    results = calculate_distortion_map_all_outputs(
        embryo_vol_shape=embryo_vol_shape,
        cylinder_radius=cylinder_radius,
        approx_spacing_x=approx_spacing_x,
        approx_spacing_theta=approx_spacing_theta,
        point_cloud=point_cloud,
        full_size_projection_shape=full_size_projection_shape
    )

    hit_points = results["hit_points"]
    uv_grid = results["uv_grid"]
    uv_grid_shape = results["uv_grid_shape"]
    vertical_distortion = results["vertical_distortion"]
    horizontal_distortion = results["horizontal_distortion"]
    elapsed_time = results["elapsed_time"]
    mesh = results["mesh"]
    cylinder_points_zyx = results["cylinder_points_zyx"]

    logging.info(f"Surface Points: {hit_points.shape[0]}")
    logging.info(f"Full size cylindrical projection shape: {full_size_projection_shape}")
    logging.info(f"Vertical distortion matrix shape: {vertical_distortion.shape}")
    logging.info(f"Horizontal distortion matrix shape: {horizontal_distortion.shape}")

    elapsed_total = time.time() - start_time
    logging.info(f"Distortion map calculations done in {elapsed_total:.4f} seconds")

    half_of_cylinder_points = cylinder_points_zyx.shape[0] // 2
    viz.visualize_3d_points(hit_points, 
                            mesh=mesh, 
                            extra_points_zyx=cylinder_points_zyx[:half_of_cylinder_points], 
                            volume_shape_zyx=embryo_vol_shape)
    # uv_grid_highlighted = uv_grid[1000:1002]  # example highlighted indices
    viz.visualize_uv_grid(uv_grid)

    # Visualize the distance heatmaps.
    viz.visualize_distance_heatmaps(
        vertical_avg=vertical_distortion,
        horizontal_avg=horizontal_distortion,
        title_vertical="Vertical distortion factor\n of embryo to cylinder mapping",
        title_horizontal="Horizontal distortion factor\n of embryo to cylinder mapping"
    )



if __name__ == "__main__":
    main()
