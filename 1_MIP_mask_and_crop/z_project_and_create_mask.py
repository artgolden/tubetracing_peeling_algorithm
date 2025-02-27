import numpy as np
import tifffile as tiff
from skimage import filters
from skimage import morphology
import cv2
import json
from skimage import measure
import os
import scipy.ndimage

def crop_rotated_3d(image, center, size, rotation_matrix):
    """
    Crops a rotated 3D region from an image.
    
    :param image: 3D numpy array (Z, Y, X)
    :param center: (z, y, x) center of the crop region
    :param size: (depth, height, width) of the desired output
    :param rotation_matrix: 3x3 rotation matrix
    :return: Cropped 3D region
    """
    # Get the inverse transformation (map crop space to original space)
    inv_rot_matrix = np.linalg.inv(rotation_matrix)
    
    # Generate target coordinates in the cropped space
    dz, dy, dx = np.meshgrid(
        np.arange(size[0]) - size[0] // 2,
        np.arange(size[1]) - size[1] // 2,
        np.arange(size[2]) - size[2] // 2,
        indexing='ij'
    )

    # Stack coordinates and apply inverse transformation
    target_coords = np.stack([dz, dy, dx], axis=-1)
    source_coords = np.dot(target_coords, inv_rot_matrix.T) + center

    # Interpolate from original image using scipy.ndimage
    cropped_region = scipy.ndimage.map_coordinates(
        image, 
        [source_coords[..., 0], 
         source_coords[..., 1], 
         source_coords[..., 2]], 
        order=1,
        mode='nearest'
    )
    
    return cropped_region


def crop_and_save_ellipse_region(image_3d, mask, outs_dir="outs", output_json_filename="ellipse_data.json", cropped_tiff_filename="cropped_region.tif", diagnostic_image_filename="ellipse_overlay.png"):
    """
    Detects the largest object in a boolean image mask, fits an ellipse (RotatedRect),
    crops the corresponding region from a 3D image, and saves the cropped region
    as a separate TIF file, along with ellipse parameters in a JSON file.

    Args:
        image_3d (numpy.ndarray): The 3D image (Z, Y, X) to crop from.
        mask (numpy.ndarray): Boolean image mask (Y, X).
        outs_dir (str): Path to the output directory. Defaults to "outs".
        output_json_filename (str): Filename for the JSON output. Defaults to "ellipse_data.json".
        cropped_tiff_filename (str): Filename for the cropped TIFF output. Defaults to "cropped_region.tif".
    """

    # 1. Input validation
    if not isinstance(image_3d, np.ndarray):
        raise TypeError("Input 'image_3d' must be a NumPy array.")
    if len(image_3d.shape) != 3:
        raise ValueError("Input 'image_3d' must be a 3D array (Z, Y, X).")
    if not isinstance(mask, np.ndarray):
        raise TypeError("Input 'mask' must be a NumPy array.")
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
        print("Warning: No objects found in the image.")
        ellipse_data = {"error": "No objects found"}
        output_json_path = os.path.join(outs_dir, output_json_filename)
        os.makedirs(outs_dir, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(ellipse_data, f, indent=4)
        return

    largest_region = max(regions, key=lambda region: region.area)

    # 3. Extract largest object and find its edges using Canny
    largest_object_mask = (labels == largest_region.label).astype(np.uint8) * 255  # Create a mask of the largest object

    edges = cv2.Canny(largest_object_mask, 100, 200) # Apply Canny edge detection

    y, x = np.where(edges == 255)  # (row, col) for white pixels
    points = np.column_stack((x, y))  # fitEllipse expects (x, y) order

    # 2. Check that we have enough points to fit an ellipse
    if len(points) < 5:
        print("Warning: Not enough edge points to fit an ellipse.")
        ellipse_data = {"error": "Not enough edge points to fit an ellipse"}
        output_json_path = os.path.join(outs_dir, output_json_filename)
        os.makedirs(outs_dir, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(ellipse_data, f, indent=4)
        return

    # 3. Fit the ellipse
    rotated_rect = cv2.fitEllipse(points)

    full_depth = image_3d.shape[0]
    # 4. Extract ellipse properties from RotatedRect
    (center_x, center_y), (width, height), angle_deg = rotated_rect
    center = (full_depth/2, center_y, center_x)  
    size = (full_depth, int(width)*1.15, int(height)*1.15)  

    # Convert angle to radians
    theta = np.radians(angle_deg - 90)

    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)],
    ])

    cropped_volume = crop_rotated_3d(image_3d, center, size, rotation_matrix)
    
    # Save the cropped volume as a TIF file
    output_tiff_path = os.path.join(outs_dir, cropped_tiff_filename)
    tiff.imwrite(output_tiff_path, cropped_volume)

    ellipse_data = {
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
        "angle_deg": angle_deg
    }

    # Save ellipse data to JSON file
    output_json_path = os.path.join(outs_dir, output_json_filename)
    os.makedirs(outs_dir, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(ellipse_data, f, indent=4)



if __name__ == '__main__':
    merged = tiff.imread("outs/merged_ill.tif") #Replace dummy data load with the actual merged file
    max_proj = np.max(merged, axis=0)
    img_median = filters.median(max_proj, morphology.disk(5)) # TODO: need to replace with something based on pixel size in um

    th = filters.threshold_triangle(img_median)
    mask = img_median >= th

    tiff.imwrite("outs/z_proj_mask.tif", mask)

    try:
        crop_and_save_ellipse_region(merged, mask)  # Pass the 3D image 'merged' to the function
    except ValueError as e:
        print(e)
    except TypeError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")