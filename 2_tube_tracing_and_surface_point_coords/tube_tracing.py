import tifffile
import os
import numpy as np
from scipy.interpolate import interpn
import scipy.ndimage as ndimage
import pyvista as pv



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


def create_surface_mesh(signal_starts, origin, theta_res, phi_res, max_dist):
    """
    Creates a surface mesh object in Cartesian coordinates from the signal_starts array,
    representing the detected signal start locations in 3D space.

    Args:
        signal_starts (numpy.ndarray): A 2D numpy array (theta x phi) containing rho indices where signals start.
                                        -1 or similar values indicate no signal detected.
        origin (tuple):  (z0, y0, x0) origin of the polar coordinate system in Cartesian coordinates.
        theta_res (int): Number of samples along the azimuthal (theta) direction.
        phi_res (int): Number of samples along the polar (phi) direction.
        max_dist (float): The maximum radial distance used when converting to polar coordinates. This is important for scaling the rho values back to cartesian.

    Returns:
        pyvista.PolyData: A pyvista PolyData object representing the surface mesh.
    """

    z0, y0, x0 = origin

    # Create theta and phi arrays
    theta = np.linspace(0, 2 * np.pi, theta_res)
    phi = np.linspace(0, np.pi, phi_res)

    # Create a meshgrid of theta and phi values
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='xy')  # Use 'xy' indexing for consistent array shapes

    # Prepare arrays to store Cartesian coordinates
    x = np.zeros_like(theta_grid)
    y = np.zeros_like(theta_grid)
    z = np.zeros_like(theta_grid)

    # Convert polar coordinates to Cartesian coordinates
    for i in range(theta_res):
        for j in range(phi_res):
            rho_index = signal_starts[i, j]

            if rho_index != -1:  # Only process points with detected signals
                rho = rho_index

                x[i, j] = rho * np.sin(phi_grid[i, j]) * np.cos(theta_grid[i, j]) + x0
                y[i, j] = rho * np.sin(phi_grid[i, j]) * np.sin(theta_grid[i, j]) + y0
                z[i, j] = rho * np.cos(phi_grid[i, j]) + z0
            else:
                # Assign NaN values to points with no signal
                x[i, j] = np.nan
                y[i, j] = np.nan
                z[i, j] = np.nan

    # Stack the coordinates to create the points array
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Remove NaN values from the points array (points with no signal)
    points = points[~np.isnan(points).any(axis=1)]


    # Create the pyvista PolyData object (surface)
    cloud = pv.PolyData(points)

    # Option 1: Create a surface using a Delaunay 2D filter (better for non-uniform data)
    surface = cloud.delaunay_2d()

    # Option 2: Create a structured grid (requires regular spacing and is less flexible).
    # Ensure points are organized into a regular grid for this to work. Requires a complete surface, with no NaN points.
    #structured_grid = pv.StructuredGrid(x, y, z) #This will error if you have NaN values


    return surface


def export_signal_points(signal_starts, origin, theta_res, phi_res, orig_vol_shape, phi_max, output_file="outs/surface_points.csv"):
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
            z = round(orig_vol_shape[0] - z, 3) # flipping the z axis to match the original image, since we flipped the original volume for processing

            # Append the point in ZYX order
            points.append([z, y, x])

    points = np.array(points)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the points to a CSV file with a header
    np.savetxt(output_file, points, delimiter=",", header="z,y,x", comments="", fmt='%.3f')

    print(f"Exported {points.shape[0]} signal points to {output_file}")

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

if __name__ == "__main__":
    file_path = "outs/down_cropped.tif"
    volume = load_3d_volume(file_path)
    volume = volume[::-1, :, :]

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
        export_signal_points(filtered_signals, origin, polar_volume.shape[0], polar_volume.shape[1], volume.shape, phi_max)
    
        print("Original volume shape:", volume.shape)
        print("Polar volume shape:", polar_volume.shape)
        # print("Polar volume:", polar_volume)
    else:
        print("Failed to load the volume.")
