import time
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.interpolate import griddata

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

def mesh_from_point_cloud(points, vol_shape) -> trimesh.Trimesh:
    """
    Point cloud to convex hull mesh.
    Returns a Trimesh mesh.
    """
    points[:, [0]] = vol_shape[0] - points[:, [0]] - 1  # Flip z axis
    cloud = trimesh.points.PointCloud(points)
    mesh = cloud.convex_hull
    return mesh

def perform_ray_mesh_intersection(mesh, ray_origins, ray_directions):
    """
    Use trimesh's RayMeshIntersector to compute ray-mesh intersections.
    Returns an array of hit points with same shape as ray_origins, with NaNs for no-hit rays.
    """
    intersector = RayMeshIntersector(mesh)
    locations, index_ray, index_tri = intersector.intersects_location(
        ray_origins, ray_directions, multiple_hits=False
    )

    result = np.full((ray_origins.shape[0], 3), np.nan, dtype=np.float32)
    result[index_ray] = locations
    return result

def sparse_grid_on_half_cylinder(
    image_shape,
    num_points_theta,
    num_points_x,
    radius=1.0,
    origin_yz=(0.0, 0.0)
):
    height, width = image_shape
    y0, z0 = origin_yz

    # Generate angular samples along the half-cylinder
    theta = np.linspace(-np.pi/2, np.pi/2, num_points_theta)
    # Generate x samples evenly along the image height.
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

def visualize_uv_grid(uv_grid, uv_grid_highlighted=None, title="UV Grid Visualization"):
    """
    Plot the UV grid with optional highlighted points.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(uv_grid[:, 0], uv_grid[:, 1], c='blue', s=5, label='UV Grid')
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

def interpolate_nan_elements(matrix: np.ndarray) -> np.ndarray:
    """
    Interpolates NaN elements in a 2D matrix using linear interpolation when there is
    sufficient information from surrounding non-NaN elements.

    This function expects a 2D numpy array (for example, the vertical or horizontal average
    distance matrix produced by compute_avg_neighbor_distances). It identifies valid (non-NaN)
    elements, and then computes a linear interpolation across the grid. Elements that lie
    outside the convex hull of the valid points are not interpolated and will remain NaN.

    Parameters:
        matrix (np.ndarray): A 2D numpy array containing numeric values and NaNs.

    Returns:
        np.ndarray: A 2D numpy array of the same shape as `matrix` with some or all NaN
                    elements replaced by linearly interpolated values.

    Raises:
        TypeError: If the input is not a numpy array.
        ValueError: If the input is not 2-dimensional or if there are no valid data points
                    to interpolate from.

    Example:
        >>> import numpy as np
        >>> test_matrix = np.array([[1.0, 2.0, 3.0],
        ...                         [4.0, np.nan, 6.0],
        ...                         [7.0, 8.0, 9.0]])
        >>> interpolated = interpolate_nan_elements(test_matrix)
        >>> print(interpolated)
        [[1. 2. 3.]
         [4. 5. 6.]
         [7. 8. 9.]]
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
    Interpolates NaN elements in the input matrix along each row using linear interpolation
    based solely on horizontal neighbors. Only contiguous segments of NaNs that are bounded
    by valid numbers at both ends and whose gap size is less than or equal to
    max_interpolation_distance are interpolated.
    
    Parameters:
        matrix (np.ndarray): 2D array of float values which may contain np.nan.
        max_interpolation_distance (int): Maximum number of consecutive NaN elements in a gap
                                          that will be interpolated (default is 5).
    
    Returns:
        np.ndarray: A new 2D array (of the same shape as the input) with eligible NaN values
                    replaced by their linearly interpolated values.
    """
    interpolated = matrix.copy()
    n_rows, n_cols = interpolated.shape

    for row_idx in range(n_rows):
        row = interpolated[row_idx]
        valid_indices = np.where(~np.isnan(row))[0]
        if valid_indices.size < 2:
            continue

        # Process each gap between consecutive valid indices.
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

def resize_distortion_map(matrix: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resizes a 2D distortion map matrix to a new target shape using linear interpolation.
    This function tolerates NaN values by excluding them from the interpolation dataset.
    Points in the target grid falling outside the convex hull of valid data will remain NaN.

    Parameters:
        matrix (np.ndarray): 2D numpy array representing the distortion map, possibly containing NaNs.
        target_shape (tuple): Tuple (target_rows, target_cols) specifying the desired shape.

    Returns:
        np.ndarray: A 2D numpy array of shape target_shape with values interpolated from the input matrix.
    """
    from scipy.interpolate import griddata

    orig_rows, orig_cols = matrix.shape
    target_rows, target_cols = target_shape

    # Create a grid of coordinates for the original matrix.
    grid_orig = np.mgrid[0:orig_rows, 0:orig_cols]
    # Stack coordinates into (num_points, 2) and flatten the matrix values.
    coords_orig = np.stack((grid_orig[0].ravel(), grid_orig[1].ravel()), axis=-1)
    values = matrix.ravel()

    # Exclude NaN values from the interpolation dataset.
    valid_mask = ~np.isnan(values)
    coords_valid = coords_orig[valid_mask]
    values_valid = values[valid_mask]

    # Create the target grid coordinates.
    target_x = np.linspace(0, orig_rows - 1, target_rows)
    target_y = np.linspace(0, orig_cols - 1, target_cols)
    grid_target = np.meshgrid(target_x, target_y, indexing='ij')
    coords_target = np.stack((grid_target[0].ravel(), grid_target[1].ravel()), axis=-1)

    # Perform linear interpolation; points outside the convex hull will be set to NaN.
    interpolated_values = griddata(coords_valid, values_valid, coords_target, method='linear')
    resized_matrix = interpolated_values.reshape(target_shape)

    return resized_matrix

def visualize_distance_heatmaps(vertical_avg: np.ndarray, horizontal_avg: np.ndarray,
                                title_vertical: str = "Vertical Neighbor Avg Distance",
                                title_horizontal: str = "Horizontal Neighbor Avg Distance",
                                xlim: tuple = None,
                                ylim: tuple = None):
    """
    Visualize the two 2D matrices (vertical and horizontal average distances) as heatmaps.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    v_data = vertical_avg.copy()
    h_data = horizontal_avg.copy()

    if ylim:
        v_data = v_data[ylim[0]:ylim[1], :]
        h_data = h_data[ylim[0]:ylim[1], :]
    if xlim:
        v_data = v_data[:, xlim[0]:xlim[1]]
        h_data = h_data[:, xlim[0]:xlim[1]]

    im_v = axs[0].imshow(v_data, interpolation='nearest', aspect='auto', origin='lower')
    axs[0].set_title(title_vertical)
    axs[0].set_xlabel("U (embryo width)")
    axs[0].set_ylabel("V (embryo length)")
    fig.colorbar(im_v, ax=axs[0])

    im_h = axs[1].imshow(h_data, interpolation='nearest', aspect='auto', origin='lower')
    axs[1].set_title(title_horizontal)
    axs[1].set_xlabel("U (embryo width)")
    axs[1].set_ylabel("V (embryo length)")
    fig.colorbar(im_h, ax=axs[1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_time = time.time()

    # ===== Inputs =====

    # Get image volume shape
    embryo_vol_shape = tiff.imread("outs/down_cropped_tp_300.tif").shape
    print(f"Volume shape: {embryo_vol_shape}")
    max_r = round(embryo_vol_shape[1] / 2.0 * 1.15)
    cylinder_radius = max_r
    approx_spacing_x = 2
    approx_spacing_theta = 5

    point_cloud = np.load("outs/hull_embryo_surface_points.npy")[:, [2, 1, 0]]

    # Generate grid points on half-cylinder surface
    full_size_projection_shape = (embryo_vol_shape[2], round(np.pi * max_r + 1))

    # ===== Main script =====
    num_points_theta = full_size_projection_shape[1] // approx_spacing_theta
    num_points_x = full_size_projection_shape[0] // approx_spacing_x
    spacing_u = full_size_projection_shape[1] / num_points_theta
    spacing_v = full_size_projection_shape[0] / num_points_x
    print(f"Spacing u: {spacing_u}, Spacing v: {spacing_v}")


    cylinder_points_zyx, uv_grid, uv_grid_shape = sparse_grid_on_half_cylinder(
        image_shape=full_size_projection_shape,
        num_points_theta=num_points_theta,
        num_points_x=num_points_x,
        radius=cylinder_radius,
        origin_yz=(embryo_vol_shape[1] // 2, 0)
    )

    mesh = mesh_from_point_cloud(point_cloud, embryo_vol_shape)

    # Convert ZYX to XYZ for processing using NumPy.
    # Since cylinder_points_zyx is already a NumPy array, just ensure the type is float32.
    surface_points = cylinder_points_zyx.astype(np.float32)

    # Ray origins and directions using NumPy.
    ray_origins = surface_points.copy()
    ray_origins[:, 1] = embryo_vol_shape[1] // 2  # Set Y to center of volume
    ray_origins[:, 0] = 0                 # Set Z to 0

    # Compute ray directions by normalizing the vector differences.
    vecs = surface_points - ray_origins
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    ray_directions = vecs / norms

    # Perform ray-mesh intersection
    hit_points = perform_ray_mesh_intersection(mesh, ray_origins, ray_directions)

    print("Surface Points: ", surface_points.shape[0])
    hit_points_3d = np.array(hit_points)

    # Example visualization calls (uncomment to run):
    #visualize_3d_points(hit_points_3d, extra_points_zyx=cylinder_points_zyx[1000:1002], mesh=mesh, volume_shape_zyx=vol_shape)
    # print(f"uv_grid shape: {uv_grid_shape} uv_grid num points: {uv_grid.shape[0]}")
    uv_grid_highlighted = uv_grid[1000:1002]  # example highlighted indices
    #visualize_uv_grid(uv_grid, uv_grid_highlighted)

    cols, rows = uv_grid_shape
    shape_2d = (rows, cols)

    # Compute the average neighbor distances
    vertical_avg, horizontal_avg = compute_avg_neighbor_distances(hit_points_3d, shape_2d)
    vertical_avg = interpolate_nans_horizontally(vertical_avg)
    vertical_avg = interpolate_nan_elements(vertical_avg)
    horizontal_avg = interpolate_nans_horizontally(horizontal_avg)
    horizontal_avg = interpolate_nan_elements(horizontal_avg)

    print(f"Full size cylindrical projection shape: {full_size_projection_shape}")
    horizontal_distortion = resize_distortion_map(spacing_u / horizontal_avg, full_size_projection_shape)
    vertical_distortion = resize_distortion_map(spacing_v / vertical_avg, full_size_projection_shape)



    print(f"Vertical distortion matrix shape: {vertical_distortion.shape}")
    print(f"Horizontal distortion matrix shape: {horizontal_distortion.shape}")


    end_time = time.time()
    # Visualize the computed heatmaps for average distances
    visualize_distance_heatmaps(vertical_distortion,
                                horizontal_distortion,
                                # ylim=(45*2, 300*2),
                                title_vertical="Vertical distortion factor\n of embryo to cylinder mapping",
                                title_horizontal="Horizontal distortion factor\n of embryo to cylinder mapping")

    elapsed_time = end_time - start_time
    print(f"Done in {elapsed_time:.4f} seconds")

# TODO:
# + interpolate the distortion map?
# + convert distances to distortion factors by 1/(uv_grid sampling rates)
# + rescale distortion maps to defined size