import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3d plotting
from scipy.interpolate import griddata
import numba
from numba import njit, prange

# =============================================================================
# Projection and Distortion Calculation (using numba for speed and parallelism)
# =============================================================================
@njit(parallel=True)
def project_points_njit(volume_points, center_y, center_x, R):
    """
    For each input point (in z,y,x order) project the (y,x) onto the surface of a
    half-cylinder (semicircle) of fixed radius R. The half-cylinder is defined by:
       - Center in (y,x): (center_y, center_x)
       - Only angles in [0, π] (i.e. the curved part)
    Also computes the distortion vector in (y,x) as the difference between the
    projected and original (y,x) coordinates.
    """
    n = volume_points.shape[0]
    projected = np.empty_like(volume_points)
    distortions = np.empty((n, 2))
    for i in prange(n):
        # Keep z coordinate unchanged (cylinder axis)
        z = volume_points[i, 0]
        y = volume_points[i, 1]
        x = volume_points[i, 2]
        # Compute relative vector from center (in y,x)
        dy = y - center_y
        dx = x - center_x
        theta = np.arctan2(dy, dx)
        # Force angle into [0, π] for the half-cylinder
        if theta < 0:
            theta += np.pi
        # Project onto circle of radius R (in (y,x) plane)
        proj_y = center_y + R * np.sin(theta)
        proj_x = center_x + R * np.cos(theta)
        projected[i, 0] = z
        projected[i, 1] = proj_y
        projected[i, 2] = proj_x
        # Compute distortion (difference in y and x)
        distortions[i, 0] = proj_y - y
        distortions[i, 1] = proj_x - x
    return projected, distortions

def project_points(volume_points, volume_shape_zyx):
    """
    Given a set of 3D points (in z,y,x) and a volume shape,
    project the points onto a half-cylinder surface.
    
    Assumptions:
      - Cylinder axis is along the Z–axis.
      - Cylinder cross-section is in the (Y,X) plane.
      - Cylinder radius is half of volume width (using the X dimension).
      - The half-cylinder uses only angles in [0, π].
    """
    # Compute cylinder parameters
    R = volume_shape_zyx[2] / 2.0         # radius = half of volume's width (X dimension)
    center_y = volume_shape_zyx[1] / 2.0    # center of (Y,X) circle (Y coordinate)
    center_x = volume_shape_zyx[2] / 2.0    # center of (Y,X) circle (X coordinate)
    # Use numba-compiled function for projection and distortion computation.
    projected, distortions = project_points_njit(volume_points, center_y, center_x, R)
    return projected, distortions

# =============================================================================
# Distortion Map Interpolation
# =============================================================================
def compute_distortion_map(volume_points, distortions, volume_shape_zyx, grid_resolution=1):
    """
    Interpolates the (y,x) distortion vectors (computed per point) onto a full 2D grid.
    
    The grid is defined in the (Y,X) plane spanning:
        Y: 0 to volume_shape[1]-1
        X: 0 to volume_shape[2]-1
    
    Returns:
      grid_y, grid_x : meshgrid coordinates for the 2D map
      distortion_y : 2D array of Y–component distortions
      distortion_x : 2D array of X–component distortions
    """
    # Define grid limits based on volume shape.
    y_min, y_max = 0, volume_shape_zyx[1] - 1
    x_min, x_max = 0, volume_shape_zyx[2] - 1
    grid_y, grid_x = np.mgrid[y_min:y_max+1:grid_resolution, x_min:x_max+1:grid_resolution]
    
    # Use the original (Y,X) locations from volume_points as interpolation points.
    points = volume_points[:, 1:3]  # (y,x) for each point
    
    # Interpolate each distortion component.
    distortion_y = griddata(points, distortions[:, 0],
                            (grid_y, grid_x), method='linear', fill_value=0)
    distortion_x = griddata(points, distortions[:, 1],
                            (grid_y, grid_x), method='linear', fill_value=0)
    return grid_y, grid_x, distortion_y, distortion_x

# =============================================================================
# Visualization Functions
# =============================================================================
def visualize_points(volume_points, projected_points):
    """
    Creates a 3D scatter plot showing original volume points (blue) and projected
    points (red). Note: Coordinates are displayed as (X, Y, Z) for plotting.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Note: For visualization we swap the order to (x,y,z)
    ax.scatter(volume_points[:, 2], volume_points[:, 1], volume_points[:, 0],
               c='b', marker='o', label='Original')
    ax.scatter(projected_points[:, 2], projected_points[:, 1], projected_points[:, 0],
               c='r', marker='^', label='Projected')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("Original and Projected Points")
    plt.show()

def visualize_distortion_map(grid_y, grid_x, distortion_y, distortion_x):
    """
    Visualizes the distortion map as a heatmap.
    The magnitude of the (Y,X) distortion is computed (sqrt(dy^2+dx^2)).
    """
    distortion_magnitude = np.sqrt(distortion_y**2 + distortion_x**2)
    plt.figure()
    plt.imshow(distortion_magnitude, origin='lower', aspect='auto',
               extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()))
    plt.colorbar(label='Distortion Magnitude')
    plt.title("Distortion Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# =============================================================================
# Test Function
# =============================================================================
def test_module():
    """
    Test the module by creating a half-sphere (a dome) placed with its flat side
    on the bottom (Y=0) and centered in Z and X. The volume has dimensions (90, 70, 200)
    corresponding to (Z, Y, X). The dome is generated using spherical coordinates
    (only the top half, i.e. theta from 0 to π/2 so that the flat side touches Y=0).
    It then projects these points onto the half-cylinder, computes and visualizes the
    distortion map, and displays both the original and projected points in 3D.
    """
    volume_shape = (90, 70, 200)  # (Z, Y, X)
    
    # Parameters for the half-sphere (dome)
    r = 25.0
    center_z = volume_shape[0] / 2.0  # middle in Z
    center_x = volume_shape[2] / 2.0  # middle in X
    center_y = r  # so that the flat side touches Y=0
    
    # Generate points on a dome (half-sphere):
    num_theta = 50   # polar angle (from vertical Y)
    num_phi = 100    # azimuthal angle in the (Z,X) plane
    theta_vals = np.linspace(0, np.pi/2, num_theta)  # only top half (dome)
    phi_vals = np.linspace(0, 2*np.pi, num_phi)
    
    points_list = []
    for theta in theta_vals:
        for phi in phi_vals:
            # Spherical coordinate conversion:
            # Here theta is measured from the vertical (Y) axis.
            y = center_y + r * np.cos(theta)
            z = center_z + r * np.sin(theta) * np.cos(phi)
            x = center_x + r * np.sin(theta) * np.sin(phi)
            points_list.append([z, y, x])
    volume_points = np.array(points_list)
    
    # Project points onto the half-cylinder
    projected_points, distortions = project_points(volume_points, volume_shape)
    
    # Compute the distortion map by interpolating the per-point distortion vectors.
    grid_y, grid_x, distortion_y, distortion_x = compute_distortion_map(volume_points, distortions, volume_shape)
    
    # Visualize the original and projected points in one 3D plot.
    visualize_points(volume_points, projected_points)
    
    # Visualize the distortion map (showing the magnitude as a heatmap).
    visualize_distortion_map(grid_y, grid_x, distortion_y, distortion_x)

# =============================================================================
# Run Test When Executed as Main Script
# =============================================================================
if __name__ == '__main__':
    test_module()
