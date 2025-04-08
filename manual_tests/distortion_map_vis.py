"""
Visualization module for intermediate and final outputs from the module 
for distortion map calculation of the mapping of points 
on embryo surface mesh to the 2D surface of half a cylinder.

This module provides functions to visualize:
  - 3D point clouds and meshes within the 3D volume.
  - The UV grid with optional highlighted points.
  - Heatmaps of vertical and horizontal distortion factors.
These visualization routines aid in debugging and data presentation.
"""

import matplotlib.pyplot as plt
import numpy as np
import trimesh


def visualize_3d_points(
    volume_points_zyx: np.ndarray,
    volume_shape_zyx: np.ndarray = None,
    highlighted_points_idx: np.ndarray = None,
    extra_points_zyx: np.ndarray = None,
    mesh: trimesh.Trimesh = None,
    title: str = "Original 3D Points"
) -> None:
    """
    Visualize the 3D point cloud within the volume space.
    
    Parameters:
        volume_points_zyx (np.ndarray): Array of 3D points in ZYX order.
        volume_shape_zyx (np.ndarray, optional): The shape (Z, Y, X) of the volume.
        highlighted_points_idx (np.ndarray, optional): Indices of points to highlight.
        extra_points_zyx (np.ndarray, optional): Extra 3D points to visualize.
        mesh (trimesh.Trimesh, optional): Mesh to overlay on the point cloud.
        title (str): Title for the plot.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    z = volume_points_zyx[:, 0]
    y = volume_points_zyx[:, 1]
    x = volume_points_zyx[:, 2]

    ax.scatter(x, y, z, c="purple", s=3, alpha=0.5)
    if highlighted_points_idx is not None:
        ax.scatter(x[highlighted_points_idx],
                   y[highlighted_points_idx],
                   z[highlighted_points_idx],
                   c="red", s=10, alpha=0.9)
    if extra_points_zyx is not None:
        ax.scatter(extra_points_zyx[:, 2],
                   extra_points_zyx[:, 1],
                   extra_points_zyx[:, 0],
                   c="green", s=3, alpha=0.7)

    if volume_shape_zyx is not None:
        ax.set_xlim([0, volume_shape_zyx[2] - 1])
        ax.set_ylim([0, volume_shape_zyx[1] - 1])
        ax.set_zlim([0, volume_shape_zyx[0] - 1])

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if mesh is not None:
        ax.plot_trisurf(mesh.vertices[:, 2], mesh.vertices[:, 1], mesh.vertices[:, 0],
                        triangles=mesh.faces, color="gray", alpha=0.3)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def visualize_uv_grid(
    uv_grid: np.ndarray,
    uv_grid_highlighted: np.ndarray = None,
    title: str = "UV Grid Visualization"
) -> None:
    """
    Visualize the UV grid and optionally highlight selected points.
    
    Parameters:
        uv_grid (np.ndarray): Array of UV coordinates.
        uv_grid_highlighted (np.ndarray, optional): Subset of UV coordinates to highlight.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(uv_grid[:, 0], uv_grid[:, 1], c="blue", s=5, label="UV Grid")
    if uv_grid_highlighted is not None:
        plt.scatter(uv_grid_highlighted[:, 0], uv_grid_highlighted[:, 1],
                    c="red", s=20, label="Highlighted")
    plt.xlabel("U")
    plt.ylabel("V")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def visualize_distance_heatmaps(
    vertical_avg: np.ndarray,
    horizontal_avg: np.ndarray,
    title_vertical: str = "Vertical Neighbor Avg Distance",
    title_horizontal: str = "Horizontal Neighbor Avg Distance",
    xlim: tuple = None,
    ylim: tuple = None
) -> None:
    """
    Visualize vertical and horizontal distance matrices as heatmaps, using color scales that
    exclude the extreme top and bottom 2% of values (only used for scaling) and handling NaNs.

    Parameters:
        vertical_avg (np.ndarray): 2D array representing vertical average distances.
        horizontal_avg (np.ndarray): 2D array representing horizontal average distances.
        title_vertical (str): Title for the vertical heatmap.
        title_horizontal (str): Title for the horizontal heatmap.
        xlim (tuple, optional): Limits for cropping along the x-axis.
        ylim (tuple, optional): Limits for cropping along the y-axis.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    v_data = vertical_avg.copy()
    h_data = horizontal_avg.copy()

    # Apply axis limits if provided
    if ylim:
        v_data = v_data[ylim[0]:ylim[1], :]
        h_data = h_data[ylim[0]:ylim[1], :]
    if xlim:
        v_data = v_data[:, xlim[0]:xlim[1]]
        h_data = h_data[:, xlim[0]:xlim[1]]

    # Calculate percentile bounds excluding NaNs
    vmin_v = np.nanpercentile(v_data, 2)
    vmax_v = np.nanpercentile(v_data, 98)
    vmin_h = np.nanpercentile(h_data, 2)
    vmax_h = np.nanpercentile(h_data, 98)

    # Create heatmap for vertical data with adjusted color scale
    im_v = axs[0].imshow(
        v_data,
        interpolation="nearest",
        aspect="auto",
        origin="lower",
        vmin=vmin_v,
        vmax=vmax_v
    )
    axs[0].set_title(title_vertical)
    axs[0].set_xlabel("U (embryo width)")
    axs[0].set_ylabel("V (embryo length)")
    fig.colorbar(im_v, ax=axs[0])

    # Create heatmap for horizontal data with adjusted color scale
    im_h = axs[1].imshow(
        h_data,
        interpolation="nearest",
        aspect="auto",
        origin="lower",
        vmin=vmin_h,
        vmax=vmax_h
    )
    axs[1].set_title(title_horizontal)
    axs[1].set_xlabel("U (embryo width)")
    axs[1].set_ylabel("V (embryo length)")
    fig.colorbar(im_h, ax=axs[1])

    plt.tight_layout()
    plt.show()
