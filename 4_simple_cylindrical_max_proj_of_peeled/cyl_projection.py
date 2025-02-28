import numpy as np
from scipy.interpolate import interpn
# import scalene
# import scalene.profile
from line_profiler import profile

@profile
def cylindrical_cartography_projection(volume, origin, num_r=None, num_theta=None, num_z=None):
    """
    Converts the embryo volume to cylindrical coordinates with the cylinder axis
    parallel to the original image's X axis. The cylindrical coordinate system is centered at
    the given origin. Linear interpolation is used to sample the volume onto the cylindrical grid.
    Then, a maximum intensity projection along the radial (r) axis is computed, resulting in a 2D cartography projection.
    
    The cylindrical volume is arranged with the radial coordinate as the 0th axis.
    
    Args:
        volume (numpy.ndarray): 3D volume in Z, Y, X order.
        origin (tuple): A tuple (origin_z, origin_y, origin_x) representing the center of the cylindrical
                        coordinate system.
        num_r (int, optional): Number of radial samples. If None, defaults to int(orig_img_y_max/2).
        num_theta (int, optional): Number of angular (theta) samples. If None, defaults to int(np.pi * max_r),
                                   i.e. the half-circumference of a circle with radius max_r.
        num_z (int, optional): Number of samples along the cylinder's z axis (parallel to original X).
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
        num_r = int(max_r)  # approximately one sample per pixel
    if num_theta is None:
        num_theta = int(np.pi * max_r)  # samples based on the half-circumference of a circle with radius max_r
    if num_z is None:
        num_z = volume.shape[2]
    
    # Define the cylindrical grid:
    # r: from 0 to max_r
    r_vals = np.linspace(0, max_r, num_r)
    # theta: from 0 to pi (assuming only half of an embryo in the volume)
    theta_vals = np.linspace(0, np.pi, num_theta, endpoint=False)
    # z: along the cylinder axis (which corresponds to the original X axis, shifted by origin_x)
    z_vals = np.linspace(-origin_x, volume.shape[2] - 1 - origin_x, num_z)
    
    # Create a 3D meshgrid in cylindrical coordinates with ordering (r, theta, z)
    r_grid, theta_grid, z_grid = np.meshgrid(r_vals, theta_vals, z_vals, indexing="ij")
    
    # Map cylindrical (r, theta, z) back to original Cartesian coordinates.
    # The radial plane is the (y, z) plane with the center at (origin_y, origin_z).
    orig_y = origin_y + r_grid * np.cos(theta_grid)
    orig_z = origin_z + r_grid * np.sin(theta_grid)
    orig_x = z_grid + origin_x  # recover the original x coordinate
    
    # Prepare points for interpolation. The volume is in (Z, Y, X) order.
    xi = np.stack((orig_z, orig_y, orig_x), axis=-1)
    
    # Define the grid for the original volume.
    grid_z = np.arange(volume.shape[0])
    grid_y = np.arange(volume.shape[1])
    grid_x = np.arange(volume.shape[2])
    points = (grid_z, grid_y, grid_x)
    
    # Interpolate using linear interpolation.
    cylindrical_volume = interpn(points, volume, xi, method="linear", bounds_error=False, fill_value=0)
    
    # Print out the shape of the cylindrical volume.
    # The expected shape is (num_r, num_theta, num_z)
    print("Cylindrical volume shape:", cylindrical_volume.shape)
    
    # Compute the maximum intensity projection along the radial (r) axis, which is axis 0.
    projection = np.max(cylindrical_volume, axis=0)
    
    # Print out the shape of the resulting projection.
    # The resulting shape should be (num_theta, num_z)
    print("Projection shape:", projection.shape)
    
    return projection



if __name__ == "__main__":
    volume = np.load("outs/down_cropped_minus_hull.npy")
    volume = volume[::-1, :, :]
    origin_z = 0
    origin_y = volume.shape[1] // 2  # Middle of Y
    origin_x = volume.shape[2] // 2  # Middle of X
    origin = (origin_z, origin_y, origin_x)

    # with scalene.profile.Scalene():
    projection = cylindrical_cartography_projection(volume, origin)
    np.save("outs/cylindrical_projection.npy", projection)
    np.save("outs/cylindrical_projection.npy", projection)