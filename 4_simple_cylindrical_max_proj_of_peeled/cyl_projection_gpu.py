from line_profiler import profile
import numpy as np 
from dexp.utils import xpArray
from dexp.utils.backends import Backend, BestBackend
import importlib

@profile
def cylindrical_cartography_projection(volume: xpArray, origin: tuple[int, int, int], num_r=None, num_theta=None, num_z=None) -> np.ndarray:
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
    # Unpack the origin
    origin_z, origin_y, origin_x = origin

    # Determine the maximum radius as half the size of the original y-dimension.
    orig_img_y_max = volume.shape[1]
    max_r = orig_img_y_max / 2.0 * 1.1

    # Set default sampling resolutions if not provided.
    if num_r is None:
        num_r = int(max_r)  # approximately one sample per pixel
    if num_theta is None:
        num_theta = int(xp.pi * max_r)  # based on the half-circumference
    if num_z is None:
        num_z = volume.shape[2]
    num_r = round(max_r - max_r*0.6)
    # Define the cylindrical grid.
    r_vals = xp.linspace(max_r*0.6, max_r, num_r)
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
    cylindrical_volume = interpolate.interpn(points, volume, xi, method="nearest", bounds_error=False, fill_value=0)
    print("Cylindrical volume shape:", cylindrical_volume.shape)
    
    # Compute the maximum intensity projection along the radial (r) axis.
    projection = xp.max(cylindrical_volume, axis=0)
    print("Projection shape:", projection.shape)
    
    return projection[::-1,:]

if __name__ == "__main__":
    # Load the volume using NumPy (file I/O remains on CPU).
    volume = np.load("outs/down_cropped_minus_hull.npy")
    

    with BestBackend():
        volume = Backend.to_backend(volume)
        volume = volume[::-1, :, :] 
        volume = volume[:,50:-50, 90:-90]
        # Define the cylindrical coordinate origin.
        origin_z = 0
        origin_y = volume.shape[1] // 2  # Middle of Y
        origin_x = volume.shape[2] // 2  # Middle of X
        origin = (origin_z, origin_y, origin_x)
        projection = cylindrical_cartography_projection(volume, origin)
        
        projection_cpu = Backend.to_numpy(projection)
        np.save("outs/cylindrical_projection.npy", projection_cpu)
