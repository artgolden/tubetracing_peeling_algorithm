import datetime
import warnings
import tifffile as tiff
import os
import shutil
import subprocess
import re
import argparse
import logging
import contextlib
import io
from tqdm import tqdm
import numpy as np
from skimage import measure
from skimage import filters
from skimage import morphology
from skimage import transform
from skimage import draw
import cv2
from vedo import Points, ConvexHull, Volume
from dexp.utils import xpArray
from dexp.utils.backends import Backend, BestBackend, CupyBackend, NumpyBackend
import importlib
from scipy import ndimage as cpu_ndimage
from numba import njit
from typing import Optional
from joblib import Parallel, delayed
import torch
import torch.nn.functional as F

DEBUG_MODE = True
RATIO_FOR_EXPANDING_THE_CROPPED_REGION_AROUND_THE_EMBRYO = 1.15

def logging_broadcast(message):
    logging.info(message)
    print(message)

def load_3d_volume(file_path):
    """
    Loads a 3D volume from a TIFF file using the tifffile library.

    Args:
        file_path (str): The path to the TIFF file.

    Returns:
        numpy.ndarray: An 8-bit or 16-bit numpy array representing the 3D volume.
                       Returns None if the file cannot be loaded or if the data type is unsupported.
    """
    try:
        volume = tiff.imread(file_path)

        if volume.dtype == np.uint8 or volume.dtype == np.uint16:
            return volume
        else:
            print(
                f"Unsupported data type: {volume.dtype}.  Only uint8 and uint16 are supported."
            )
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading TIFF file: {e}")
        return None

def crop_rotated_3d_torch(image, center, size, rotation_matrix, device=None):
    """
    High-performance GPU 3D crop with rotation using PyTorch.

    :param image: 3D numpy array (Z, Y, X), float32
    :param center: (z, y, x) center of the crop region
    :param size: (depth, height, width) of the desired output
    :param rotation_matrix: 3x3 rotation matrix (numpy or torch)
    :param device: 'cuda' or 'cpu'
    :return: Cropped 3D region as numpy array (float32)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rotation_matrix = np.linalg.inv(rotation_matrix)

    def convert_numpy_dtype_to_torch_dtype(numpy_dtype):
        if numpy_dtype == np.float32:
            return torch.float32
        elif numpy_dtype == np.uint64:
            return torch.uint64
        elif numpy_dtype == np.uint32:
            return torch.uint32
        elif numpy_dtype == np.uint16:
            return torch.uint16
        elif numpy_dtype == np.uint8:
            return torch.uint8
        elif numpy_dtype == np.int8:
            return torch.int8
        elif numpy_dtype == np.bool_:
            return torch.bool
        else:
            return None 

    input_dtype = convert_numpy_dtype_to_torch_dtype(image.dtype)
    # Move everything to torch
    image = image.astype(np.float32)
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, D, H, W)
    center = torch.tensor(center, dtype=torch.float32, device=device)
    rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32, device=device)

    D, H, W = size
    out_z, out_y, out_x = D, H, W
    img_z, img_y, img_x = image.shape

    # IMPORTANT: Construct target grid in pixel coordinates (not normalized!)
    dz = torch.arange(D, dtype=torch.float32, device=device) - D // 2
    dy = torch.arange(H, dtype=torch.float32, device=device) - H // 2
    dx = torch.arange(W, dtype=torch.float32, device=device) - W // 2
    zz, yy, xx = torch.meshgrid(dz, dy, dx, indexing='ij')  # (D, H, W)

    local_coords = torch.stack([zz, yy, xx], dim=-1)  # (D, H, W, 3)
    world_coords = torch.einsum('ij,dhwj->dhwi', rotation_matrix, local_coords) + center

    # Map world_coords (z,y,x) to PyTorch-normalized grid_sample coordinates (x_norm, y_norm, z_norm)
    # align_corners=True: normalized_coord = 2 * index / (size - 1) - 1
    grid = torch.empty_like(world_coords)
    grid[..., 2] = 2.0 * world_coords[..., 0] / max(img_z - 1, 1) - 1.0  # Z → norm
    grid[..., 1] = 2.0 * world_coords[..., 1] / max(img_y - 1, 1) - 1.0  # Y → norm
    grid[..., 0] = 2.0 * world_coords[..., 2] / max(img_x - 1, 1) - 1.0  # X → norm

    # Reshape grid for grid_sample
    grid = grid.unsqueeze(0)  # (1, D, H, W, 3)

    # Sample with NEAREST interpolation (matches order=0)
    sampled = F.grid_sample(
        image_tensor,
        grid,
        mode='nearest',           # true voxel sampling
        padding_mode='border',    # match scipy.mode='nearest'
        align_corners=True
    )
    result_uint8 = torch.round(sampled).clamp(0, torch.iinfo(input_dtype).max).to(input_dtype)
    return result_uint8.squeeze().cpu().numpy()

def crop_rotated_3d(image, center, size, rotation_matrix, n_jobs=-1):
    """
    High-performance cropped rotated 3D region from an image using parallel processing.
    
    :param image: 3D numpy array (Z, Y, X)
    :param center: (z, y, x) center of the crop region
    :param size: (depth, height, width) of the desired output
    :param rotation_matrix: 3x3 rotation matrix
    :param n_jobs: Number of parallel jobs (-1 uses all available cores)
    :return: Cropped 3D region
    """
    inv_rot = np.linalg.inv(rotation_matrix)

    # Precompute local coordinates for one 2D slice
    dz = np.arange(size[0]) - size[0] // 2
    dy = np.arange(size[1]) - size[1] // 2
    dx = np.arange(size[2]) - size[2] // 2
    grid_y, grid_x = np.meshgrid(dy, dx, indexing='ij')
    base_coords = np.stack([np.zeros_like(grid_x), grid_y, grid_x], axis=-1)  # Shape: (H, W, 3)

    def process_z_slice(i):
        z_offset = dz[i]
        coords = base_coords.copy()
        coords[..., 0] = z_offset  # insert Z-offset into all coordinates
        world_coords = np.einsum('ij,hwj->hwi', inv_rot, coords) + center
        sampled = cpu_ndimage.map_coordinates(
            image,
            [world_coords[..., 0], world_coords[..., 1], world_coords[..., 2]],
            order=1,
            mode='nearest'
        )
        return sampled

    cropped_volume = Parallel(n_jobs=n_jobs)(
        delayed(process_z_slice)(i) for i in range(size[0])
    )

    return np.stack(cropped_volume, axis=0)


def get_matrix_with_circle(radius, shape=None, center=None):
    """
    Creates a NumPy array with a filled circle of 1s, using skimage.draw.disk.

    Args:
        radius (int): The radius of the circle.
        shape (tuple of ints): The shape of the 2D array (rows, cols). Default square of radius*2+1.
        center (tuple of ints, optional): The center of the circle (row, col).
                                            If None, defaults to the center of the array.

    Returns:
        numpy.ndarray: A NumPy array representing the circle.
    """
    if shape is None:
        shape = (radius * 2 + 1, radius * 2 + 1)
    img = np.zeros(shape, dtype=np.uint8)

    if center is None:
        center = (shape[0] // 2, shape[1] // 2)  # Integer division for center

    rr, cc = draw.disk(center, radius, shape=shape) # Get circle indices within bounds
    img[rr, cc] = 1

    return img

def crop_around_embryo(image_3d, mask, target_crop_shape=None) -> Optional[np.ndarray]:
    """
    Detects the largest object in a boolean image mask, fits an ellipse (RotatedRect),
    crops the corresponding region from a 3D image, and saves the cropped region
    as a separate TIF file, along with ellipse parameters in a JSON file.

    Args:
        image_3d (numpy.ndarray): The 3D image (Z, Y, X) to crop from.
        mask (numpy.ndarray): Boolean image mask (Y, X).
    """
    


    # 1. Input validation
    if len(image_3d.shape) != 3:
        raise ValueError("Input 'image_3d' must be a 3D array (Z, Y, X).")
    if mask.dtype != bool:
        raise ValueError("Input 'mask' must be a boolean array (dtype=bool).")
    if len(mask.shape) != 2:
        raise ValueError("Input 'mask' must be a 2D array (grayscale).")
    if image_3d.shape[1] != mask.shape[0] or image_3d.shape[2] != mask.shape[1]:
        raise ValueError("The dimensions of 'image_3d' (Y, X) must match the dimensions of 'mask'.")

    # Convert boolean mask to uint8 - necessary for OpenCV
    binary_image = mask.astype(np.uint8) * 255

    # 2. Find connected components (objects)
    labels = measure.label(binary_image)
    regions = measure.regionprops(labels)

    if not regions:
        print("Error: No objects found in the image.")
        return None

    largest_region = max(regions, key=lambda region: region.area)

    # 3. Extract largest object and find its edges using Canny
    largest_object_mask = (labels == largest_region.label).astype(np.uint8) * 255  # Create a mask of the largest object

    edges = cv2.Canny(largest_object_mask, 100, 200) # Apply Canny edge detection

    y, x = np.where(edges == 255)  # (row, col) for white pixels
    points = np.column_stack((x, y))  # fitEllipse expects (x, y) order

    # 2. Check that we have enough points to fit an ellipse
    if len(points) < 5:
        print("Error: Not enough edge points to fit an ellipse for embryo segmentation.")
        return None

    # 3. Fit the ellipse
    rotated_rect = cv2.fitEllipse(points)

    full_depth = image_3d.shape[0]
    # 4. Extract ellipse properties from RotatedRect
    (center_x, center_y), (width, height), angle_deg = rotated_rect
    center = (full_depth/2, center_y, center_x)  
    expand_r = RATIO_FOR_EXPANDING_THE_CROPPED_REGION_AROUND_THE_EMBRYO
    size = (full_depth, int(width*expand_r), int(height*expand_r))  
    logging.info(f"Cropping embryo with center: {center}, size: {size}, angle: {angle_deg}")
    if target_crop_shape is not None:
        size = target_crop_shape
        logging.info(f"Target crop shape was provided: {target_crop_shape}")

    # Convert angle to radians
    theta = np.radians(angle_deg - 90)

    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)],
    ])

    cropped = crop_rotated_3d(image_3d, center, size, rotation_matrix)
    return cropped
    
def cartesian_to_polar(volume: xpArray, origin, rho_res=1, theta_res=360, phi_res=180, phi_max=np.pi / 2) -> xpArray:
    """
    Converts a 3D numpy array from Cartesian coordinates to polar coordinates,
    with the origin of the polar coordinates at the specified point, and interpolates
    the original volume's values onto the new polar grid.
    Assumes the input volume array has axes in the following order: Z, Y, X.

    Args:
        volume (xpArray): A 3D numpy array representing the volume in Cartesian coordinates (Z, Y, X).
        origin (tuple): A tuple (z0, y0, x0) representing the origin of the polar coordinate system
                        within the volume.
        rho_res (int): Number of samples along radial direction.
        theta_res (int): Number of samples along azimuthal direction.
        phi_res (int): Number of samples along polar direction.

    Returns:
        xpArray: A 3D numpy array representing the volume in polar coordinates (rho, theta, phi)
                       with interpolated values from the original volume.
    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()
    interpolate = importlib.import_module(".interpolate", sp.__name__) # Had to use a workaround, since sp.interpolate gives an error: 
                                                                       # module 'cupyx.scipy' has no attribute 'interpolate' when backend is CuPy
    volume = Backend.to_backend(volume)
    z0, y0, x0 = origin

    # Define the polar grid
    max_dist = np.sqrt(
        max(z0, volume.shape[0] - z0) ** 2
        + max(y0, volume.shape[1] - y0) ** 2
        + max(x0, volume.shape[2] - x0) ** 2
    )
    rho = xp.linspace(0, max_dist, round(max_dist))
    theta = xp.linspace(0, 2 * xp.pi, theta_res)
    phi = xp.linspace(0, phi_max, phi_res) # xp.pi / 2 default: taking only positive Z values, since we assume only half of the embryo inside the image and origing with Z=0 or max Z value

    theta_grid, phi_grid, rho_grid = xp.meshgrid(theta, phi, rho, indexing="ij")

    # Convert polar coordinates back to Cartesian coordinates
    x = rho_grid * xp.sin(phi_grid) * xp.cos(theta_grid) + x0
    y = rho_grid * xp.sin(phi_grid) * xp.sin(theta_grid) + y0
    z = rho_grid * xp.cos(phi_grid) + z0

    # Interpolate the original volume onto the polar grid
    points = (
        xp.arange(volume.shape[0]),
        xp.arange(volume.shape[1]),
        xp.arange(volume.shape[2]),
    )  # Z, Y, X
    xi = xp.stack((z, y, x), axis=-1)  # Stack z,y,x to create interpolation points

    polar_volume = interpolate.interpn(
        points, volume, xi, method="linear", bounds_error=False, fill_value=0
    )
    return polar_volume

def find_background_std(volume: xpArray) -> float:
    """
    Find the standard deviation of the background in a 3D volume from a 50x50x50 corner cube.
    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()
    volume = Backend.to_backend(volume)
    # Calculate full image min and max
    vmin = volume.min()
    vmax = volume.max()
    full_range = vmax - vmin
    # Background threshold: values must not exceed 20% above the minimum
    threshold = vmin + 0.2 * full_range

    cube_found = None
    # Define the eight corners (z, y, x) for a 50x50x50 cube.
    corners = [
        (0, 0, 0),
        (0, 0, volume.shape[2] - 50),
        (0, volume.shape[1] - 50, 0),
        (0, volume.shape[1] - 50, volume.shape[2] - 50),
        (volume.shape[0] - 50, 0, 0),
        (volume.shape[0] - 50, 0, volume.shape[2] - 50),
        (volume.shape[0] - 50, volume.shape[1] - 50, 0),
        (volume.shape[0] - 50, volume.shape[1] - 50, volume.shape[2] - 50)
    ]

    for corner in corners:
        z, y, x = corner
        cube = volume[z:z+50, y:y+50, x:x+50]
        # Check if no voxel in cube exceeds the threshold.
        if cube.max() <= threshold:
            cube_found = cube
            break

    if cube_found is None:
        cube_found = volume[0:50, 0:50, 0:50]
        print(f"Warning: Could not find a background cube with values below threshold; using default first corner. Volume min: {vmin}, max: {vmax}")

    bkg_std = float(xp.std(cube_found))
    print(f"Background standard deviation: {bkg_std}")
    return bkg_std

def detect_signal_start(intensity, bkg_std, smoothing_window=4, threshold_factor=2):
    intensity = np.trim_zeros(intensity, trim='b')
    
    # Smooth the intensity values to reduce noise
    smoothed_intensity = cpu_ndimage.uniform_filter1d(intensity, size=smoothing_window)

    # Calculate the first derivative
    derivative = np.diff(smoothed_intensity)

    # Determine the threshold for significant rise
    threshold = threshold_factor * bkg_std

    # Find the index where the derivative exceeds the threshold
    signal_start_index = np.where(derivative > threshold)[-1] # taking the most outer step in the signal

    if len(signal_start_index) == 0:
        return None  # No signal detected
    else:
        return signal_start_index[0]

def raytracing_in_polar(polar_volume, bkg_std, patch_size=1, tubetracing_density="dense"):
    """
    Performs ray tracing in the polar volume to detect the start of a signal along each ray.
    Instead of processing each (theta, phi) ray individually, the function collects intensity
    profiles from a patch centered at (theta, phi) with a radius 'patch_size' in both directions,
    averages them, and then determines the signal start.

    Args:
        polar_volume : A 3D numpy array representing the volume in polar coordinates
                                      (theta, phi, rho).
        bkg_std (float): The standard deviation of the background noise.
        patch_size (int): The radius of the patch to average in theta and phi dimensions.
                          For example, patch_size=1 will average a 3x3 patch.

    Returns:
        numpy.ndarray: A 2D numpy array with dimensions (theta, phi) containing the rho index where
                       the signal starts. Returns -1 if no signal is detected along a given ray.
    """

    theta_res, phi_res, rho_res = polar_volume.shape
    signal_starts = np.zeros((theta_res, phi_res))

    ray_step = 1
    match tubetracing_density:
        case "sparse":
            ray_step = 1 + 2 * patch_size
        case "dense":
            ray_step = 1
        case _:
            raise ValueError("Invalid tubetracing_density value. Choose from 'sparse', 'dense'.")

    for theta_idx in range(0, theta_res, ray_step):
        for phi_idx in range(0, phi_res, ray_step):
            # Define patch boundaries ensuring indices remain within valid limits
            theta_min = max(0, theta_idx - patch_size)
            theta_max = min(theta_res - 1, theta_idx + patch_size + 1)
            phi_min = max(0, phi_idx - patch_size)
            phi_max = min(phi_res - 1, phi_idx + patch_size + 1)

            # Extract the patch and average the intensity profiles along theta and phi dimensions
            patch_profiles = polar_volume[theta_min:theta_max, phi_min:phi_max, :]
            averaged_intensity = np.mean(patch_profiles, axis=(0, 1))

            # Detect the signal start using the averaged profile
            start_index = detect_signal_start(averaged_intensity, bkg_std)

            if start_index is not None:
                signal_starts[theta_idx, phi_idx] = start_index
            else:
                signal_starts[theta_idx, phi_idx] = -1  # Mark no signal detected

    return signal_starts

def filter_high_rho_outliers(signal_starts, threshold_factor=1.2, min_neighbors=5):
    """
    Filters out outlier rho values (high values) from a signal_starts array.  A value
    is considered an outlier if it is significantly higher than its neighbors, suggesting
    a spurious signal detection.

    Args:
        signal_starts (numpy.ndarray): A 2D numpy array (theta x phi) containing rho indices where signals start.
                                        Values of -1 (or similar) are assumed to represent no signal.
        threshold_factor (float):  Factor by which a rho value must exceed its neighbors' median to be considered an outlier.
        min_neighbors (int): Minimum number of valid neighbors required for outlier detection.  Helps avoid
                             incorrectly flagging edge points or areas with sparse signals.

    Returns:
        numpy.ndarray: A copy of the signal_starts array with outlier rho values replaced by -1 (no signal).
    """

    filtered_signal_starts = (
        signal_starts.copy()
    )  # Create a copy to avoid modifying the original array

    rows, cols = signal_starts.shape

    for row in range(rows):
        for col in range(cols):
            rho_value = signal_starts[row, col]

            # Skip processing if the current value represents 'no signal'
            if rho_value == -1:
                continue

            # Collect valid neighbors (excluding -1 values)
            neighbors = []
            for i in range(max(0, row - 1), min(rows, row + 2)):
                for j in range(max(0, col - 1), min(cols, col + 2)):
                    if (i, j) != (row, col) and signal_starts[i, j] != -1:
                        neighbors.append(signal_starts[i, j])

            # Check if enough valid neighbors exist
            if len(neighbors) < min_neighbors:
                continue  # Not enough neighbors to make a reliable determination

            # Calculate the median of the neighbors
            median_neighbor_rho = np.median(neighbors)

            # Check if the current value is significantly higher than its neighbors
            if rho_value > median_neighbor_rho * threshold_factor:
                filtered_signal_starts[row, col] = -1  # Mark as no signal

    return filtered_signal_starts

@njit(cache=True)
def export_signal_points(signal_starts, origin, theta_res, phi_res, orig_vol_shape, phi_max):
    """
    Exports the 3D coordinates of detected signals (in ZYX order) to a CSV file.
    The resulting file can be loaded as a point cloud in napari and overlaid on the original image.

    Args:
        signal_starts (numpy.ndarray): A 2D numpy array (theta x phi) containing the rho indices where signals start.
                                       Points with no signal should have the value -1.
        origin (tuple): A tuple (z0, y0, x0) representing the origin used for polar coordinate conversion.
        theta_res (int): The number of samples along the azimuthal (theta) direction.
        phi_res (int): The number of samples along the polar (phi) direction.
        output_file (str): Path to the output CSV file.
    """
    z0, y0, x0 = origin

    # Create theta and phi arrays.
    # Note: Using the same linspace parameters as in the polar conversion.
    theta = np.linspace(0, 2 * np.pi, theta_res)
    phi = np.linspace(0, phi_max, phi_res)

    points = []

    # Loop over the (theta, phi) grid.
    for i in range(theta_res):
        for j in range(phi_res):
            rho = signal_starts[i, j]
            if rho == -1:
                continue  # Skip points with no detected signal

            # Convert polar coordinates to Cartesian coordinates
            x = rho * np.sin(phi[j]) * np.cos(theta[i]) + x0
            y = rho * np.sin(phi[j]) * np.sin(theta[i]) + y0
            z = rho * np.cos(phi[j]) + z0

            x = round(x, 3)
            y = round(y, 3)
            z = round(orig_vol_shape[0] - 1 - z, 3) # flipping the z axis to match the original image, since we flipped the original volume for processing

            # Append the point in ZYX order
            points.append([z, y, x])

    return np.array(points)

def add_projected_embryo_outline_points(volume_shape_zyx, points) -> np.ndarray:
    """    
    Adds projected embryo outline points to the given set of points.    
    The function takes the volume shape and a set of points, and creates a new set of points by projecting the points with a z-coordinate greater than 80% of the maximum z-coordinate onto the maximum z-plane. These projected points are then concatenated to the original set of points.
    This is done to ensure that convex hull later extends to the volume boundary.
    
    Args:
        volume_shape (tuple): The shape of the volume.
        points (numpy.ndarray): The set of points to which the projected points will be added.
    Returns:
        numpy.ndarray: The updated set of points with the projected points added.
    """        
    max_z = volume_shape_zyx[0] - 1
    p_proj = points.copy()
    p_proj = p_proj[p_proj[:,0] > max_z * 0.8]
    p_proj[:,0] = max_z
    more_points = np.concatenate((points, p_proj))
    return more_points

def points_to_convex_hull_volume_mask(points, volume_shape_zyx, dilation_radius=3) -> Volume:
    """
    Converts a set of 3D points to a binary volume mask of the inner part of the embryo using a convex hull.

    This function takes a set of 3D points and a volume shape, constructs a convex hull from the points,
    binarizes the convex hull into a volume mask, and then erodes/dilates the mask. 

    Args:
        points (numpy.ndarray): A numpy array of shape (N, 3) representing the 3D points in ZYX order.
        volume_shape_zyx (tuple): A tuple (z, y, x) representing the shape of the volume.
        dilation_radius (int): The radius of the dilation applied to the volume mask.  This expands the mask
            outwards, useful for ensuring complete coverage of the structure represented by the points.

    Returns:
        vedo.Volume: A vedo.Volume object representing the binary volume mask.  The mask has values of 255 inside
            the convex hull and 0 outside.
    """
    points_raw = points[:, [2, 1, 0]]
    pts = Points(points_raw)
    logging.debug("Creating convex hull from points")
    hull = ConvexHull(pts)

    vol_shape_xyz = volume_shape_zyx[::-1]
    logging.debug("Binarizing convex hull into volume mask")
    vol_mask = hull.binarize(values=(255,0),dims=vol_shape_xyz,spacing=[1,1,1], origin=(0,0,0))
    if dilation_radius > 0:
        logging.debug(f"Dilating with radius of {dilation_radius}")
        modified = vol_mask.clone().dilate(neighbours=(dilation_radius,dilation_radius,dilation_radius))
    else:
        erosion_radius = abs(dilation_radius)
        logging.debug(f"Eroding with erosion radius of {erosion_radius}")
        modified = vol_mask.clone().erode(neighbours=(erosion_radius,erosion_radius,erosion_radius))
    return modified

def substract_mask_from_embryo_volume(volume_zyx: np.ndarray, mask_xyz) -> np.ndarray:
    """
    Subtracts a mask from an embryo volume.

    This function takes a 3D numpy array representing the embryo volume and a vedo.Volume object representing the mask.
    It transposes the volume to XYZ order, converts it to a vedo.Volume, thresholds the mask to create a binary mask,
    and then performs an element-wise multiplication of the embryo volume with the mask. Finally, it transposes the
    result back to ZYX order.

    Args:
        volume_zyx (numpy.ndarray): A 3D numpy array representing the embryo volume in ZYX order.
        mask (vedo.Volume): A vedo.Volume object representing the mask in XYZ order.

    Returns:
        numpy.ndarray: A 3D numpy array representing the embryo volume with the mask applied, in ZYX order.
    """
    # data_matrix = np.transpose(volume_zyx, (2, 1, 0))
    embryo = Volume(volume_zyx)
    if isinstance(mask_xyz, np.ndarray):
        mask_xyz = Volume(mask_xyz)
    mask_xyz = mask_xyz.threshold(above=1, replace_value=1)
    mask_xyz = mask_xyz.threshold(below=254, replace_value=0)
    diff = embryo.clone().operation("*",mask_xyz)
    return diff.tonumpy()

def cylindrical_cartography_projection(volume: xpArray, origin: tuple[int, int, int], num_r=None, num_theta=None, num_z=None, 
                                       reduce_r_sampling_in_cylindrical_coords=True) -> np.ndarray:
    """
    Converts the embryo volume to cylindrical coordinates with the cylinder axis
    parallel to the original image's X axis. The cylindrical coordinate system is centered at
    the given origin. Linear interpolation is used to sample the volume onto the cylindrical grid.
    Then, a maximum intensity projection along the radial (r) axis is computed, resulting in a 2D cartography projection.
    
    The cylindrical volume is arranged with the radial coordinate as the 0th axis.
    
    Args:
        volume (array): 3D volume (backend array) in Z, Y, X order.
        origin (tuple): A tuple (origin_z, origin_y, origin_x) representing the center of the cylindrical coordinate system.
        num_r (int, optional): Number of radial samples. If None, defaults to int(orig_img_y_max/2).
        num_theta (int, optional): Number of angular (theta) samples. If None, defaults to int(np.pi * max_r).
        num_z (int, optional): Number of samples along the cylinder's z axis (parallel to original X).
                               If None, defaults to the number of x slices in volume.
    
    Returns:
        array: A 2D array (theta x z) corresponding to the maximum intensity projection along the radial direction.
    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()
    interpolate = importlib.import_module(".interpolate", sp.__name__) # Had to use a workaround, since sp.interpolate gives an error: 
                                                                       # module 'cupyx.scipy' has no attribute 'interpolate' when backend is CuPy
    volume = Backend.to_backend(volume)

    origin_z, origin_y, origin_x = origin

    # Determine the maximum radius as half the size of the original y-dimension.
    orig_img_y_max = volume.shape[1]
    max_r = round(orig_img_y_max / 2.0 * 1.15)
    # Set default sampling resolutions if not provided.
    if num_r is None:
        num_r = max_r  # approximately one sample per pixel
    if num_theta is None:
        num_theta = int(xp.pi * max_r)  # based on the half-circumference
    if num_z is None:
        num_z = volume.shape[2]
    
    r_min = 0
    if reduce_r_sampling_in_cylindrical_coords: # it looks like such reduction in sampling in r does not change final max projection almost at all
        num_r = round(max_r - max_r*0.6)
        # r_min = max_r*0.6
    logging.debug(f"num_r: {num_r}, max_r - max_r*0.6: {max_r - max_r*0.6}, r_min: {r_min}, r_max: {max_r}, num_theta: {num_theta}, xp.pi * max_r: {xp.pi * max_r}")

    # Define the cylindrical grid.
    r_vals = xp.linspace(0, max_r, num_r)
    theta_vals = xp.linspace(0, xp.pi, num_theta, endpoint=False)
    z_vals = xp.linspace(-origin_x, volume.shape[2] - 1 - origin_x, num_z)
    
    # Create a 3D meshgrid in cylindrical coordinates (r, theta, z).
    r_grid, theta_grid, z_grid = xp.meshgrid(r_vals, theta_vals, z_vals, indexing="ij")
    
    # Map cylindrical (r, theta, z) back to original Cartesian coordinates.
    orig_y = origin_y + r_grid * xp.cos(theta_grid)
    orig_z = origin_z + r_grid * xp.sin(theta_grid)
    orig_x = z_grid + origin_x  # recover the original x coordinate
    
    # Prepare points for interpolation (volume is in Z, Y, X order).
    xi = xp.stack((orig_z, orig_y, orig_x), axis=-1).astype(xp.float16)
    
    # Define the grid for the original volume.
    grid_z = xp.arange(volume.shape[0], dtype=xp.float16)
    grid_y = xp.arange(volume.shape[1], dtype=xp.float16)
    grid_x = xp.arange(volume.shape[2], dtype=xp.float16)
    points = (grid_z, grid_y, grid_x)
    
    # Interpolate using linear interpolation.
    volume = volume.astype(xp.float16)
    cylindrical_volume = interpolate.interpn(points, volume, xi, method="linear", bounds_error=False, fill_value=0)
    logging.debug(f"Cylindrical volume shape: {cylindrical_volume.shape}")
    
    # Compute the maximum intensity projection along the radial (r) axis.
    projection = xp.max(cylindrical_volume, axis=0)
    
    return projection[::-1,:]

def upscale_mask(mask: np.ndarray, full_res_shape: tuple[int,int,int]) -> np.ndarray:
    return transform.resize(
        mask,
        full_res_shape,  #Explicitly specify shape
        order=0,
        anti_aliasing=False
    )

def get_origin(volume) -> tuple[int, int, int]:
    origin_z = 0
    origin_y = volume.shape[1] // 2  # Middle of Y
    origin_x = volume.shape[2] // 2  # Middle of X
    return origin_z, origin_y, origin_x

def detect_embryo_surface_tubetracing(volume: np.ndarray) -> np.ndarray:
    origin = get_origin(volume)
    bkg_std = find_background_std(volume)

    phi_max = np.pi / 2
    polar_volume = cartesian_to_polar(volume, origin, phi_max = phi_max)
    signals = raytracing_in_polar(Backend.to_numpy(polar_volume), bkg_std, tubetracing_density="sparse")
    # filtered_signals = filter_high_rho_outliers(signals)
    points = export_signal_points(signals, origin, polar_volume.shape[0], polar_volume.shape[1], volume.shape, phi_max)
    return points

def detect_embryo_surface_wbns(volume: np.ndarray, threshold: str=None):
    """
    Detect points of the embryo surface by using WBNS wavelet based background substraction then thresholding and eroding the mask. 
    This leaves sparse points on the embryo outline.
    """
    from wbns import substract_background
    from skimage import filters

    volume = Backend.to_numpy(volume)
    volume = np.transpose(volume, (2,1,0))
    substracted_bkg = substract_background(volume, 4, 1)
    if threshold is None:
        threshold = "isodata"
    match threshold:
        case "otsu":
            th = filters.threshold_otsu(substracted_bkg)
        case "yen":
            th = filters.threshold_yen(substracted_bkg)
        case "li":
            th = filters.threshold_li(substracted_bkg)
        case "isodata":
            th = filters.threshold_isodata(substracted_bkg)
        case "minimum":
            th = filters.threshold_minimum(substracted_bkg)
        case "yen":
            th = filters.threshold_yen(substracted_bkg)
        case "triangle":
            th = filters.threshold_triangle(substracted_bkg)
        case "mean":
            th = filters.threshold_mean(substracted_bkg)
        case "sauvola":
            th = filters.threshold_sauvola(substracted_bkg)
    mask = substracted_bkg >= th
    mask = np.transpose(mask, (2,1,0))

    structuring_element = np.ones((3,3,3))
    eroded_mask = cpu_ndimage.binary_erosion(mask, structure=structuring_element).astype(mask.dtype)  # Keep original datatype
    # Zerroing out the border to remove artifacts that wbns generates
    zero_y = int(eroded_mask.shape[1] * (RATIO_FOR_EXPANDING_THE_CROPPED_REGION_AROUND_THE_EMBRYO - 1) / 2) 
    zero_x = int(eroded_mask.shape[2] * (RATIO_FOR_EXPANDING_THE_CROPPED_REGION_AROUND_THE_EMBRYO - 1) / 2)
    eroded_mask[:,-zero_y:,:] = False
    eroded_mask[:,:zero_y,:] = False
    eroded_mask[:,:,-zero_x:] = False
    eroded_mask[:,:,:zero_x] = False

    return (eroded_mask, substracted_bkg)

def load_and_merge_illuminations(ill_file_paths: list[str]):
    images = [load_3d_volume(f) for f in ill_file_paths]
    assert len(images) < 3 # There should be no more than 2 illuminations possible for each volume
    if not isinstance(images[0], np.ndarray):
        print(f"Error loading first volume from files: {ill_file_paths}")
        return None
    if len(images) == 1:
        return images[0]
    else:
        return np.mean(np.stack(images, axis=0), axis=0).astype(images[0].dtype)
    
def get_downsampled_and_isotropic(full_res: np.ndarray, voxel_size_zyx: tuple[float, float, float]=(2.34, .586, .586), xy_downsample_factor: int=2):
    z_um, y_um, _ = voxel_size_zyx
    anisotropy_factor = z_um/y_um
    aniso_down = anisotropy_factor/xy_downsample_factor

    downsampled = measure.block_reduce(full_res, block_size=(1, xy_downsample_factor, xy_downsample_factor), func=np.mean)  # ZYX order

    scaled_image = transform.resize(
        downsampled,
        (round(downsampled.shape[0]*aniso_down), downsampled.shape[1], downsampled.shape[2]),  #Explicitly specify shape
        order=1,
        anti_aliasing=False
    )

    return scaled_image.astype(full_res.dtype)

def get_isotropic_volume(full_res: np.ndarray, voxel_size_zyx: tuple[float, float, float]=(2.34, .586, .586)):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()
    full_res = Backend.to_backend(full_res)
    z_um, y_um, _ = voxel_size_zyx
    anisotropy_factor = z_um/y_um
    aniso_down = anisotropy_factor
    zoom_factors = (round(aniso_down), 1, 1)
    scaled_image = sp.ndimage.zoom(full_res.astype(xp.float32), zoom=zoom_factors, order=1)
    return Backend.to_numpy(scaled_image, dtype=full_res.dtype)

def threshold_image_xy(volume: np.ndarray):
    max_projection = np.max(volume, axis=0)
    img_median = filters.median(max_projection, morphology.disk(5)) # TODO: need to replace with something based on pixel size in um

    th = filters.threshold_triangle(img_median)
    mask = img_median >= th

    # Clean up the mask
    cleaning_circle_radius = round(mask.shape[1] * 0.014)
    structuring_element = get_matrix_with_circle(cleaning_circle_radius)
    mask = cpu_ndimage.binary_opening(mask, structure=structuring_element, iterations=5).astype(mask.dtype)
    return mask

def parse_filename(filepath: str):
    """
    Parse a TIF file name of the form:
    timelapseID-20241008-143038_SPC-0001_TP-0870_ILL-0_CAM-1_CH-01_PL-(ZS)-outOf-0073.tif
    Returns:
        timeseries_key: a string identifying the time series (all parts except the TP and ILL parts)
        timepoint: integer value parsed after _TP-
        illumination: integer value parsed after _ILL-
    """
    base = os.path.basename(filepath)
    tp_match = re.search(r'_TP-(\d+)', base)
    ill_match = re.search(r'_ILL-(\d+)', base)
    if not (tp_match and ill_match):
        raise ValueError(f"Filename {base} does not match expected pattern.")
    timepoint = int(tp_match.group(1))
    illumination = int(ill_match.group(1))
    # Remove TP and ILL parts to form the timeseries key
    timeseries_key = re.sub(r'_TP-\d+', '', base)
    timeseries_key = re.sub(r'_ILL-\d+', '', timeseries_key)
    # Remove file extension
    timeseries_key = os.path.splitext(timeseries_key)[0]
    return timeseries_key, timepoint, illumination

def group_files(file_list):
    """
    Group files by timeseries key and then by timepoint.
    Returns a dictionary:
       { timeseries_key: { timepoint: [list of file paths for this timepoint] } }
    """
    series_dict = {}
    for f in file_list:
        try:
            key, tp, ill = parse_filename(f)
        except ValueError as e:
            logging.warning(f"Failed to parse filename {f}: {e}")
            continue
        if key not in series_dict:
            series_dict[key] = {}
        series_dict[key].setdefault(tp, []).append(f)
    return series_dict

def get_git_commit_hash(script_path):
    """
    Retrieves the current Git commit hash for the given script.

    Args:
        script_path:  The path to the Python script.  This is used to
                      determine the repository's root directory.

    Returns:
        The Git commit hash as a string, or None if not in a Git repository
        or if there's an error.
    """
    try:
        # Use git rev-parse --short HEAD to get the short commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path),  # Important: Run command in script's dir!
            check=True  # Raise exception on non-zero return code
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("Warning: Not a git repository or git command failed.")
        return None
    except FileNotFoundError:
        print("Warning: Git command not found.  Make sure Git is installed and in your PATH.")
        return None
    except Exception as e:
        print(f"Warning: Could not get git commit hash: {e}")
        return None

def copy_script_with_commit_hash(output_dir):
    """
    Copies the script to the output directory, adding the commit hash to the filename.

    Args:
        script_path: The path to the Python script to copy.
        output_dir: The directory to copy the script to.
        commit_hash: Optional. The commit hash to include in the filename.
                     If None, the script's original name is used.
    """
    try:
        script_path = os.path.abspath(__file__)
    except Exception as e:
        print(f"Warning: could not get this script path: {e}")
        return
    if script_path is None:
        print("Warning: could not get this script path.")
        return
    
    script_name = os.path.basename(script_path)
    name, ext = os.path.splitext(script_name)

    commit_hash = get_git_commit_hash(script_path)

    if commit_hash:
        new_script_name = f"{name}_{commit_hash}{ext}"
    else:
        new_script_name = script_name

    output_path = os.path.join(output_dir, new_script_name)

    try:
        shutil.copy2(script_path, output_path)  # copy2 preserves metadata
        print(f"Script copied to: {output_path}")
    except Exception as e:
        print(f"Warning: could not copy source code of the script: {e}")

def peel_embryo_with_cartography(full_res_zyx: np.ndarray, 
                                 downsampled_zyx: np.ndarray, 
                                 output_dir: str, 
                                 timepoint: int,
                                 peeling_mask: np.ndarray = None,
                                 surface_detection_mode: str = "wbns",
                                 thresholding_after_wbns: str = None,
                                 do_cylindrical_cartography=True,
                                 prune_voxels_after_wbns=True,
                                 remove_outliers_after_wbns=True,
                                 do_save_points=True, 
                                 do_save_peeled_volume=True,
                                 do_save_z_max_projection=True,
                                 do_save_mask=True,
                                 do_save_wbns_output=True,
                                 load_surface_voxels_from_file: bool = False) -> bool:
    logging.info("Starting peel_embryo_with_cartography")
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if not isinstance(peeling_mask, np.ndarray):
        volume_mask_dilation_radius = 3
        match surface_detection_mode:
            case "tubetracing":
                downsampled_zyx = Backend.to_backend(downsampled_zyx)
                downsampled_zyx = downsampled_zyx[::-1, :, :]
                logging.info("Peeling: Using tubetracing")
                volume_mask_dilation_radius = 3
                points = detect_embryo_surface_tubetracing(downsampled_zyx)
            case "wbns":
                volume_mask_dilation_radius = -9
                if load_surface_voxels_from_file:
                    # Check if the file with the precomputed surface voxels exists.
                    points_dir = os.path.join(output_dir, "surface_voxels_mask")
                    file_path = os.path.join(points_dir, f"tp_{timepoint}_wbns_surface_voxels_true.tif")
                    if os.path.exists(file_path):
                        logging.info(f"Peeling: Loading sparce_voxels_on_embryo_surface from file: {file_path}")
                        sparce_voxels_on_embryo_surface = tiff.imread(file_path)
                        points = np.transpose(np.where(sparce_voxels_on_embryo_surface))
                    else:
                        logging.warning(f"Peeling: File {file_path} not found. Falling back to WBNS detection.")
                        sparce_voxels_on_embryo_surface, only_structures_wbns = detect_embryo_surface_wbns(downsampled_zyx, threshold=thresholding_after_wbns)
                        if prune_voxels_after_wbns:
                            from prune_volume import prune_volume
                            logging.info("Pruning volume after WBNS")
                            sparce_voxels_on_embryo_surface = prune_volume(sparce_voxels_on_embryo_surface)
                        if remove_outliers_after_wbns:
                            from remove_outliers_volume_mask import remove_outliers
                            logging.info("Removing outliers (expected dirt) after WBNS")
                            sparce_voxels_on_embryo_surface = remove_outliers(sparce_voxels_on_embryo_surface)
                        points = np.transpose(np.where(sparce_voxels_on_embryo_surface))
                        if do_save_wbns_output:
                            output_dir_wbns = os.path.join(output_dir, "wbns_output")
                            os.makedirs(output_dir_wbns, exist_ok=True)
                            tiff.imwrite(os.path.join(output_dir_wbns, f"tp_{timepoint}_wbns_background_removed.tif"), only_structures_wbns)
                        points_dir = os.path.join(output_dir, "surface_voxels_mask")
                        os.makedirs(points_dir, exist_ok=True)
                        tiff.imwrite(os.path.join(points_dir, f"tp_{timepoint}_wbns_surface_voxels_true.tif"), sparce_voxels_on_embryo_surface)
                else:
                    logging.info("Peeling: Using WBNS wavelet based background subtraction for detecting only embryo structures.")
                    sparce_voxels_on_embryo_surface, only_structures_wbns = detect_embryo_surface_wbns(downsampled_zyx, threshold=thresholding_after_wbns)
                    if prune_voxels_after_wbns:
                        from prune_volume import prune_volume
                        logging.info("Pruning volume after WBNS")
                        sparce_voxels_on_embryo_surface = prune_volume(sparce_voxels_on_embryo_surface)
                    if remove_outliers_after_wbns:
                        from remove_outliers_volume_mask import remove_outliers
                        logging.info("Removing outliers (expected dirt) after WBNS")
                        sparce_voxels_on_embryo_surface = remove_outliers(sparce_voxels_on_embryo_surface)
                    points = np.transpose(np.where(sparce_voxels_on_embryo_surface))
                    if do_save_wbns_output:
                        output_dir_wbns = os.path.join(output_dir, "wbns_output")
                        os.makedirs(output_dir_wbns, exist_ok=True)
                        tiff.imwrite(os.path.join(output_dir_wbns, f"tp_{timepoint}_wbns_background_removed.tif"), only_structures_wbns)
                if do_save_points:
                    # Only save the points file if it was computed (not loaded)
                    if not load_surface_voxels_from_file:
                        points_dir = os.path.join(output_dir, "surface_voxels_mask")
                        os.makedirs(points_dir, exist_ok=True)
                        tiff.imwrite(os.path.join(points_dir, f"tp_{timepoint}_wbns_surface_voxels_true.tif"), sparce_voxels_on_embryo_surface)
                    do_save_points = False

        print(f"Number of detected points: {len(points)}")
        logging.debug(f"Number of detected points: {len(points)}")
        if len(points) > 300000:
            logging.error("Too many points detected. This is likely due to a bad surface segmentation. Embryo peeling failed.")
            return False, False

        points = add_projected_embryo_outline_points(downsampled_zyx.shape, points)
        if do_save_points:
            points_dir = os.path.join(output_dir, "surface_points")
            os.makedirs(points_dir, exist_ok=True)
            np.save(os.path.join(points_dir, f"tp_{timepoint}_surface_points.npy"), points)

        # Create a volume mask from the points
        logging.info("Peeling: Converting points to mask")
        mask = points_to_convex_hull_volume_mask(points, downsampled_zyx.shape, dilation_radius=volume_mask_dilation_radius)

        logging.debug(f"Mask shape: {mask.shape}, converting mask to numpy and transposing.")
        mask_np = np.transpose(mask.tonumpy(), (2, 1, 0))
        logging.debug(f"Upscaling mask")
        mask_upscaled = upscale_mask(mask_np, full_res_zyx.shape)
        if do_save_mask:
            logging.debug(f"Saving masks to disk")
            mask_dir = os.path.join(output_dir, "substraction_embryo_mask")
            os.makedirs(mask_dir, exist_ok=True)
            tiff.imwrite(os.path.join(mask_dir, f"tp_{timepoint}_mask.tif"), mask_np)
            np.save(os.path.join(mask_dir, f"tp_{timepoint}_upscaled_mask.npy"), mask_upscaled)
    else: 
        logging.info("Peeling: Using provided mask")
        print("Peeling: Using provided mask")
        mask_upscaled = peeling_mask

    # Subtract the mask from the embryo volume
    logging.info("Peeling: Substracting mask from embryo volume")
    peeled_volume = substract_mask_from_embryo_volume(full_res_zyx, mask_upscaled)
    if do_save_peeled_volume:
        peeled_volume_dir = os.path.join(output_dir, "peeled_volume")
        os.makedirs(peeled_volume_dir, exist_ok=True)
        tiff.imwrite(os.path.join(peeled_volume_dir, f"tp_{timepoint}_peeled_volume.tif"), peeled_volume.astype(np.uint8))

    if do_save_z_max_projection:
        z_max_projection_dir = os.path.join(output_dir, "z_max_projection")
        os.makedirs(z_max_projection_dir, exist_ok=True)
        tiff.imwrite(os.path.join(z_max_projection_dir, f"tp_{timepoint}_z_max_projection.tif"), np.max(peeled_volume[:-12], axis=0).astype(np.uint8))

    if do_cylindrical_cartography:
        logging.info("Starting cylindrical cartography projection")
        reduce_ratio = 1 / RATIO_FOR_EXPANDING_THE_CROPPED_REGION_AROUND_THE_EMBRYO
        _, y_size, x_size = peeled_volume.shape
        x_crop = int(x_size - x_size * reduce_ratio) // 2
        y_crop = int(y_size - y_size * reduce_ratio) // 2
        peeled_volume = Backend.to_backend(peeled_volume, dtype=xp.float16)[::-1, :, :]
        peeled_volume = peeled_volume[:, y_crop:-y_crop, x_crop:-x_crop]
        logging.debug(f"Peeled volume origin coordinates: {get_origin(peeled_volume)}, x and y crop: {x_crop}, {y_crop}, shape: {peeled_volume.shape}")
        cylindrical_projection = cylindrical_cartography_projection(peeled_volume, get_origin(peeled_volume))
        projection_cpu = Backend.to_numpy(cylindrical_projection, dtype=np.uint8)

        cartography_dir = os.path.join(output_dir, "cylindrical_cartography")
        os.makedirs(cartography_dir, exist_ok=True)
        cartography_file = os.path.join(cartography_dir, f"tp_{timepoint}_cyl_proj.tif")
        tiff.imwrite(cartography_file, projection_cpu)
    return True, mask_upscaled


def process_timepoint(ill_file_paths: list, 
                      output_dir: str, 
                      timepoint: int,
                      compute_backend: Backend,
                      target_crop_shape=None,
                      peeling_mask=None,
                      thresholding_after_wbns=None,
                      do_save_thresholding_mask=True,
                      do_save_down_cropped=True,
                      do_save_cropped_iso=True,
                      load_surface_voxels_from_file: bool = False):
    logging.info(f"Processing timepoint with files: {ill_file_paths}")
    print(f"Processing timepoint {timepoint}")
    
    # Merge illuminations for the timepoint
    merged_volume = load_and_merge_illuminations(ill_file_paths)
    if merged_volume is None:
        logging.error(f"Error loading merged volume for files: {ill_file_paths}")
        return None

    # Threshold to get mask
    mask = threshold_image_xy(merged_volume)
    if do_save_thresholding_mask:
        mask_dir = os.path.join(output_dir, "thresholding_mask")
        os.makedirs(mask_dir, exist_ok=True)
        tiff.imwrite(os.path.join(mask_dir, f"thresholding_mask_tp_{timepoint}.tif"), mask)
    # Crop around embryo. For the first timepoint, we call without target_crop_shape.
    # For subsequent timepoints, crop_around_embryo should use the provided target shape.
    if target_crop_shape is None:
        cropped_volume = crop_around_embryo(merged_volume, mask)
    else:
        cropped_volume = crop_around_embryo(merged_volume, mask, target_crop_shape)
    
    if cropped_volume is None:
        logging.error(f"Error segmenting and cropping around embryo for files: {ill_file_paths}")
        return None
    cropped_vol_shape = cropped_volume.shape   
    logging.info(f"TP: {timepoint} Cropped volume shape: {cropped_vol_shape}")
    print(f"TP: {timepoint} Cropped volume shape: {cropped_vol_shape}")
    
    down_cropped = get_downsampled_and_isotropic(cropped_volume)
    if do_save_down_cropped:
        down_dir = os.path.join(output_dir, "downsampled_cropped")
        os.makedirs(down_dir, exist_ok=True)
        tiff.imwrite(os.path.join(down_dir, f"down_cropped_tp_{timepoint}.tif"), down_cropped)
    
    full_res_iso = get_isotropic_volume(cropped_volume)
    if do_save_cropped_iso:
        iso_dir = os.path.join(output_dir, "cropped_isotropic_embryo")
        os.makedirs(iso_dir, exist_ok=True)
        tiff.imwrite(os.path.join(iso_dir, f"cropped_isotropic_tp_{timepoint}.tif"), full_res_iso)
    
    peel_success, peeling_mask_curr_tp = peel_embryo_with_cartography(
        full_res_iso, 
        down_cropped, 
        output_dir, 
        timepoint, 
        peeling_mask=peeling_mask,
        thresholding_after_wbns=thresholding_after_wbns,
        load_surface_voxels_from_file=load_surface_voxels_from_file
    )
    if not peel_success:
        logging.error(f"Error peeling embryo. Aborting processing this timepoint")
        compute_backend.clear_memory_pool()
        return None, None
    with contextlib.redirect_stdout(io.StringIO()) as captured_output:
        compute_backend.clear_memory_pool()
    logging.info(captured_output.getvalue())
    return cropped_vol_shape, peeling_mask_curr_tp


def process_time_series(timeseries_key: str, 
                        timepoints_dict: dict, 
                        base_out_dir: str, 
                        compute_backend: Backend,
                        reuse_peeling_mask: bool,
                        thresholding_after_wbns=None,
                        load_surface_voxels_from_file: bool = False):
    """
    Process a single time series.
    
    Parameters:
      timeseries_key: unique identifier for the time series
      timepoints_dict: dictionary with keys as timepoint integers and values as list of file paths
      base_out_dir: base output folder where time series folders are created
    """
    series_out_dir = os.path.join(base_out_dir, timeseries_key)
    os.makedirs(series_out_dir, exist_ok=True)
    logging.info(f"Processing time series: {timeseries_key}")
    print(f"Processing time series: {timeseries_key}")
    
    # Process timepoints in ascending order
    sorted_timepoints = sorted(timepoints_dict.keys())
    target_crop_shape = None  # will be set by the first processed timepoint
    peeling_mask_whole_series = None

    # Use tqdm to show progress for this time series
    for tp in tqdm(sorted_timepoints, desc=f"Series {timeseries_key}", unit="timepoint"):
        print("")
        tp_files = timepoints_dict[tp]
        crop_shape, peeling_mask_curr_tp = process_timepoint(
            tp_files, 
            series_out_dir, 
            tp, 
            compute_backend, 
            target_crop_shape=target_crop_shape, 
            peeling_mask=peeling_mask_whole_series,
            thresholding_after_wbns=thresholding_after_wbns,
            load_surface_voxels_from_file=load_surface_voxels_from_file
        )
        if crop_shape is None:
            logging.error(f"Error processing timepoint {tp}. Aborting processing this time series")
            return
        if reuse_peeling_mask and peeling_mask_whole_series is None:
            logging.info(f"Reusing peeling mask from first timepoint in the series.")
            print(f"Reusing peeling mask from first timepoint in the series.")
            peeling_mask_whole_series = peeling_mask_curr_tp
        # Save crop shape from first timepoint and use for subsequent timepoints
        if target_crop_shape is None and crop_shape is not None:
            target_crop_shape = crop_shape


def main():
    parser = argparse.ArgumentParser(
        description="Process a folder of 3D embryo TIF images: group them into time series, "
                    "apply peeling and cylindrical cartographic projection to each timepoint."
    )
    parser.add_argument("input_folder", type=str, help="Folder containing TIF files")
    parser.add_argument("--output_folder", type=str, default=None, 
                        help="Output folder (default: <input_folder>/outs)")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, etc.)")
    parser.add_argument("--wbns_threshold", type=str, default=None, help="Thresholding method used after WBNS structure detection. Options: 'otsu', 'yen', 'li', 'isodata', 'minimum', 'yen', 'triangle', 'mean', 'sauvola'.")
    parser.add_argument("--reuse_peeling", action="store_true", help="Reuse embryo peeling mask from the first timepoint in time series.")
    parser.add_argument("--force_cpu", action="store_true", help="Force execution on CPU only.")
    parser.add_argument("--skip_patterns", type=str, nargs='*', default=[], 
                        help="List of patterns; time series whose keys contain any of these will be skipped.")
    parser.add_argument("--load_surface_voxels", action="store_true", 
                        help="If set, load sparce_voxels_on_embryo_surface from file if available and skip WBNS detection and saving of only_structures_wbns.")
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder if args.output_folder else os.path.join(input_folder, "outs")
    os.makedirs(output_folder, exist_ok=True)
    
    copy_script_with_commit_hash(output_folder)
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_folder, f"process_{timestamp}.log")
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting processing of time series")
    print("Starting processing...")
    
    device_id = 0
    devices = CupyBackend.available_devices("memory")
    if len(devices) == 0:
        logging.error("Could not find available CUDA devices.")
    else:
        device_id = devices[0]
        logging_broadcast(f"Using CUDA device: {device_id}")

    if args.force_cpu:
        logging_broadcast("--force_cpu argument is set: Forcing execution only on CPU.")
     
    with (NumpyBackend() if args.force_cpu else BestBackend(device_id=device_id)) as compute_backend:
        # Find all TIF files in the input folder
        tif_files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(".tif")
        ]
        logging.info(f"Found {len(tif_files)} TIF files in {input_folder}")
        if not tif_files:
            logging.error("No TIF files found in input folder.")
            print("No TIF files found. Exiting.")
            return
        
        # Group files into time series and timepoints
        timeseries_dict = group_files(tif_files)
        logging.info(f"Found {len(timeseries_dict)} time series")

        if args.reuse_peeling:
            print("Reusing peeling masks from the first timepoint in each time series")
        else:
            print("Generating peeling masks for each timepoint in each time series")
        
        # Process each time series
        for series_key, tp_dict in timeseries_dict.items():
            if any(pattern in series_key for pattern in args.skip_patterns):
                logging_broadcast(f"Skipping time series '{series_key}' due to matching skip pattern {args.skip_patterns}")
                continue
            process_time_series(
                series_key, 
                tp_dict, 
                output_folder, 
                compute_backend, 
                args.reuse_peeling, 
                args.wbns_threshold,
                args.load_surface_voxels
            )
        
        logging.info("Processing complete")
        print("Processing complete.")


if __name__ == "__main__":
    main()
