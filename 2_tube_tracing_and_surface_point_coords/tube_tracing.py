import tifffile
import os
import numpy as np
from scipy.interpolate import interpn
import scipy.ndimage as ndimage
from vedo import Plotter, Points, ConvexHull, Volume


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
        volume = tifffile.imread(file_path)

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


def cartesian_to_polar(volume, origin, rho_res=1, theta_res=360, phi_res=180, phi_max=np.pi / 2):
    """
    Converts a 3D numpy array from Cartesian coordinates to polar coordinates,
    with the origin of the polar coordinates at the specified point, and interpolates
    the original volume's values onto the new polar grid.
    Assumes the input volume array has axes in the following order: Z, Y, X.

    Args:
        volume (numpy.ndarray): A 3D numpy array representing the volume in Cartesian coordinates (Z, Y, X).
        origin (tuple): A tuple (z0, y0, x0) representing the origin of the polar coordinate system
                        within the volume.
        rho_res (int): Number of samples along radial direction.
        theta_res (int): Number of samples along azimuthal direction.
        phi_res (int): Number of samples along polar direction.

    Returns:
        numpy.ndarray: A 3D numpy array representing the volume in polar coordinates (rho, theta, phi)
                       with interpolated values from the original volume.
    """
    z0, y0, x0 = origin

    # Define the polar grid
    max_dist = np.sqrt(
        max(z0, volume.shape[0] - z0) ** 2
        + max(y0, volume.shape[1] - y0) ** 2
        + max(x0, volume.shape[2] - x0) ** 2
    )
    rho = np.linspace(0, max_dist, round(max_dist))
    theta = np.linspace(0, 2 * np.pi, theta_res)
    phi = np.linspace(0, phi_max, phi_res) # np.pi / 2 default: taking only positive Z values, since we assume only half of the embryo inside the image and origing with Z=0 or max Z value

    theta_grid, phi_grid, rho_grid = np.meshgrid(theta, phi, rho, indexing="ij")

    # Convert polar coordinates back to Cartesian coordinates
    x = rho_grid * np.sin(phi_grid) * np.cos(theta_grid) + x0
    y = rho_grid * np.sin(phi_grid) * np.sin(theta_grid) + y0
    z = rho_grid * np.cos(phi_grid) + z0

    # Interpolate the original volume onto the polar grid
    points = (
        np.arange(volume.shape[0]),
        np.arange(volume.shape[1]),
        np.arange(volume.shape[2]),
    )  # Z, Y, X
    xi = np.stack((z, y, x), axis=-1)  # Stack z,y,x to create interpolation points

    polar_volume = interpn(
        points, volume, xi, method="linear", bounds_error=False, fill_value=0
    )

    return polar_volume


def find_background_std(volume):
    """
    Find the standard deviation of the background in a 3D volume from a 50x50x50 corner cube.
    """
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
        print("Warning: Could not find a background cube with values below threshold; using default first corner.")

    bkg_std = np.std(cube_found)
    print(f"Background standard deviation: {bkg_std}")
    return bkg_std


def detect_signal_start(intensity, bkg_std, smoothing_window=4, threshold_factor=2.5):
    intensity = intensity.copy()
    intensity = np.trim_zeros(intensity, trim='b')
    # Smooth the intensity values to reduce noise
    smoothed_intensity = ndimage.uniform_filter1d(intensity, size=smoothing_window)

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


def raytracing_in_polar(polar_volume, bkg_std, patch_size=1):
    """
    Performs ray tracing in the polar volume to detect the start of a signal along each ray.
    Instead of processing each (theta, phi) ray individually, the function collects intensity
    profiles from a patch centered at (theta, phi) with a radius 'patch_size' in both directions,
    averages them, and then determines the signal start.

    Args:
        polar_volume (numpy.ndarray): A 3D numpy array representing the volume in polar coordinates
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

    for theta_idx in range(theta_res):
        for phi_idx in range(phi_res):
            # Define patch boundaries ensuring indices remain within valid limits
            theta_min = max(0, theta_idx - patch_size)
            theta_max = min(theta_res, theta_idx + patch_size + 1)
            phi_min = max(0, phi_idx - patch_size)
            phi_max = min(phi_res, phi_idx + patch_size + 1)

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



def save_points_to_csv(points, output_file):
    """
    Save the given points to a CSV file.
    """
        # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the points to a CSV file with a header
    np.savetxt(output_file, points, delimiter=",", header="z,y,x", comments="", fmt='%.3f')

    print(f"Exported {points.shape[0]} signal points to {output_file}")


def points_to_convex_hull_volume_mask(points, volume_shape_zyx, dilation_radius=3) -> Volume:
    """
    Converts a set of 3D points to a binary volume mask of the inner part of the embryo using a convex hull.

    This function takes a set of 3D points and a volume shape, constructs a convex hull from the points,
    binarizes the convex hull into a volume mask, and then dilates the mask. 

    Args:
        points (numpy.ndarray): A numpy array of shape (N, 3) representing the 3D points in ZYX order.
        volume_shape_zyx (tuple): A tuple (z, y, x) representing the shape of the volume.
        dilation_radius (int): The radius of the dilation applied to the volume mask.  This expands the mask
            outwards, useful for ensuring complete coverage of the structure represented by the points.

    Returns:
        vedo.Volume: A vedo.Volume object representing the binary volume mask.  The mask has values of 255 inside
            the dilated convex hull and 0 outside.
    """
    points_raw = points[:, [2, 1, 0]]
    pts = Points(points_raw)
    hull = ConvexHull(pts).alpha(0.2)

    vol_shape_xyz = volume_shape_zyx[::-1]
    vol_mask = hull.binarize(values=(255,0),dims=vol_shape_xyz,spacing=[1,1,1], origin=(0,0,0))
    dilated = vol_mask.clone().dilate(neighbours=(dilation_radius,dilation_radius,dilation_radius))
    return dilated


def substract_mask_from_embryo_volume(volume_zyx: np.ndarray, mask_xyz: Volume) -> np.ndarray:
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
    data_matrix = np.transpose(volume_zyx, (2, 1, 0))
    embryo = Volume(data_matrix)
    mask_xyz = mask_xyz.threshold(above=1, replace_value=1)
    mask_xyz = mask_xyz.threshold(below=254, replace_value=0)
    diff = embryo.clone().operation("*",mask_xyz)
    return np.transpose(diff.tonumpy(), (2, 1, 0))


def cylindrical_cartography_projection(volume, origin, num_r=None, num_theta=360, num_z=None):
    """
    Converts the embryo volume to cylindrical coordinates with the cylinder axis
    parallel to the original image's X axis. The cylindrical coordinate system is centered at
    the given origin. Linear interpolation is used to sample the volume onto the cylindrical grid.
    Then, a maximum intensity projection along the radial (r) axis (within a cylinder of maximum 
    r = orig_img_y_max/2) is computed, resulting in a 2D cartography projection.
    
    Args:
        volume (numpy.ndarray): 3D volume in Z, Y, X order.
        origin (tuple): A tuple (origin_z, origin_y, origin_x) representing the center of the cylindrical
                        coordinate system.
        num_r (int, optional): Number of radial samples. If None, defaults to int(orig_img_y_max/2).
        num_theta (int, optional): Number of angular (theta) samples. Default is 360.
        num_z (int, optional): Number of samples along the cylinder’s z axis (parallel to original X).
                               If None, defaults to the number of x slices in volume.
    
    Returns:
        numpy.ndarray: A 2D numpy array (theta x z) corresponding to the maximum intensity projection
                       along the radial direction.
    """
    # Unpack the origin
    origin_z, origin_y, origin_x = origin

    # Determine the maximum radius as half the size of the original y-dimension.
    orig_img_y_max = volume.shape[1]
    max_r = orig_img_y_max / 2.0

    # Set default sampling resolutions if not provided.
    if num_r is None:
        num_r = int(max_r)  # one sample per pixel (approximately)
    if num_z is None:
        num_z = volume.shape[2]
    
    # Define the cylindrical grid:
    # r: from 0 to max_r
    r_vals = np.linspace(0, max_r, num_r)
    # theta: from 0 to 2*pi (without repeating the endpoint)
    theta_vals = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
    # z: along the cylinder axis (which is the original X axis, shifted by origin_x)
    # Here we define the new z coordinate to range from -origin_x to (x_max - origin_x)
    z_vals = np.linspace(-origin_x, volume.shape[2] - 1 - origin_x, num_z)
    
    # Create a 3D meshgrid in cylindrical coordinates.
    # The ordering will be (theta, z, r) so that after sampling we have a grid
    # over which we can take a maximum over r.
    theta_grid, z_grid, r_grid = np.meshgrid(theta_vals, z_vals, r_vals, indexing="ij")
    
    # Map cylindrical (r, theta, z) back to original Cartesian coordinates.
    # Note that the cylinder’s axis (z in cylindrical coordinates) corresponds to the original X axis.
    # The radial plane is the (y,z) plane of the original image, with the origin at (origin_y, origin_z).
    #
    # Using the standard mapping:
    #    original_y = origin_y + r * cos(theta)
    #    original_z = origin_z + r * sin(theta)
    #    original_x = z + origin_x   (since z here is the offset along original X)
    orig_z = origin_z + r_grid * np.sin(theta_grid)
    orig_y = origin_y + r_grid * np.cos(theta_grid)
    orig_x = z_grid + origin_x  # recovering the original x coordinate
    
    # Prepare points for interpolation. The volume is in (Z, Y, X) order.
    xi = np.stack((orig_z, orig_y, orig_x), axis=-1)
    
    # Define the grid for the original volume.
    grid_z = np.arange(volume.shape[0])
    grid_y = np.arange(volume.shape[1])
    grid_x = np.arange(volume.shape[2])
    points = (grid_z, grid_y, grid_x)
    
    # Interpolate using linear interpolation.
    cylindrical_volume = interpn(points, volume, xi, method="linear", bounds_error=False, fill_value=0)
    
    # Compute the maximum intensity projection along the radial (r) axis.
    projection = np.max(cylindrical_volume, axis=2)
    
    return projection


if __name__ == "__main__":
    file_path = "outs/down_cropped.tif"
    volume_orig = load_3d_volume(file_path)
    volume = volume_orig[::-1, :, :]

    if volume is not None:
        # Crop the volume
        # cropped_volume = volume[:10, :20, :20]

        # Define the origin
        origin_z = 0
        origin_y = volume.shape[1] // 2  # Middle of Y
        origin_x = volume.shape[2] // 2  # Middle of X
        origin = (origin_z, origin_y, origin_x)
        phi_max = np.pi / 2

        # Find the background standard deviation
        bkg_std = find_background_std(volume)

        # Convert to polar coordinates
        polar_volume = cartesian_to_polar(volume, origin, phi_max = phi_max)
        signals = raytracing_in_polar(polar_volume, bkg_std)
        filtered_signals = filter_high_rho_outliers(signals)
        points = export_signal_points(filtered_signals, origin, polar_volume.shape[0], polar_volume.shape[1], volume.shape, phi_max)
        points = add_projected_embryo_outline_points(volume.shape, points)
        save_points_to_csv(points, "outs/surface_points.csv")

        # Create a volume mask from the points
        mask = points_to_convex_hull_volume_mask(points, volume.shape, dilation_radius=3)
        mask_np = np.transpose(mask.tonumpy(), (2, 1, 0))
        np.save("outs/down_cropped_hull_mask.npy", mask_np)

        # Subtract the mask from the embryo volume
        peeled_volume = substract_mask_from_embryo_volume(volume_orig, mask)
        np.save("outs/down_cropped_minus_hull.npy", peeled_volume)

        print("Original volume shape:", volume.shape)
        print("Polar volume shape:", polar_volume.shape)
        # print("Polar volume:", polar_volume)
    else:
        print("Failed to load the volume.")
