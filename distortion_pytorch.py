import numpy as np
import trimesh
import torch
import faiss
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
import tifffile as tiff
import open3d as o3d
from scipy.spatial import ConvexHull


def project_to_cylinder(vertices, radius=1.0, origin_yz=(0, 0)):
    y0,z0 = origin_yz
    x, y, z = vertices[:, 2], vertices[:, 1], vertices[:, 0]
    theta = np.arctan2(y - y0, z - z0)
    # u = theta[np.abs(theta) <= np.pi/2]
    u = (theta + np.pi) / (2*np.pi) * np.pi * radius # only half cylinder
    v = x
    uv = np.column_stack([u, v])
    P_cyl = np.column_stack([
        radius * np.cos(theta) + z0,  # Z on cylinder
        radius * np.sin(theta) + y0, # Y on cylinder
        x,                       # Vertical coordinate (unchanged)
    ])
    return uv, P_cyl


def gpu_knn_search(vertices, k):
    vertices = vertices.astype(np.float32)
    index = faiss.IndexFlatL2(3)
    index.add(vertices)
    distances, indices = index.search(vertices, k)
    return indices


def estimate_local_distortion_gpu(vertices, uv_coords, neighbors):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = torch.tensor(vertices, dtype=torch.float32, device=device)  # [N, 3]
    UV = torch.tensor(uv_coords, dtype=torch.float32, device=device)  # [N, 2]

    N, K = neighbors.shape
    i = torch.arange(N).unsqueeze(1).expand(-1, K).to(device)
    neighbor_idx = torch.tensor(neighbors, dtype=torch.int64, device=device)

    d3d = V[neighbor_idx] - V[i]        # [N, K, 3]
    duv = UV[neighbor_idx] - UV[i]      # [N, K, 2]

    A = d3d                             # [N, K, 3]
    B = duv                             # [N, K, 2]

    # Compute ATA and ATB in batch
    AT = A.transpose(1, 2)              # [N, 3, K]
    ATA = AT.bmm(A)                    # [N, 3, 3]
    ATB = AT.bmm(B)                    # [N, 3, 2]

    # Check rank of ATA using SVD
    U, S, Vh = torch.linalg.svd(ATA)
    rank_mask = (S[:, 1] > 1e-6) & (S[:, 2] > 1e-6)  # requires full rank (3 non-zero singular values)

    # Initialize output
    J = torch.zeros((N, 2, 3), device=device)

    # Use pseudo-inverse fallback for safe solution even when ATA is nearly singular
    ATA_inv = torch.linalg.pinv(ATA)  # [N, 3, 3]
    J_all = ATA_inv.bmm(ATB).transpose(1, 2)  # [N, 2, 3]
    J[rank_mask] = J_all[rank_mask]

    # Compute stretch per UV axis: norm of Jacobian row vectors
    stretch_u = torch.linalg.norm(J[:, 0, :], dim=1)  # [N]
    stretch_v = torch.linalg.norm(J[:, 1, :], dim=1)  # [N]

    dist = torch.stack([stretch_u, stretch_v], dim=1)  # [N, 2]

    # Mask out low-confidence results (non-full-rank entries)
    dist[~rank_mask.cpu()] = float('nan')

    return dist.cpu().numpy()


def rasterize_distortion_map(uv_coords, distortions, resolution=(512, 512)):
    u = uv_coords[:, 0]
    v = uv_coords[:, 1]
    grid_u, grid_v = np.mgrid[0:1:complex(resolution[0]), 0:1:complex(resolution[1])]

    distortion_x = griddata((u, v), distortions[:, 0], (grid_u, grid_v), method='linear', fill_value=0)
    distortion_y = griddata((u, v), distortions[:, 1], (grid_u, grid_v), method='linear', fill_value=0)

    return distortion_x, distortion_y


def visualize_distortion_map(distortion_x, distortion_y):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(distortion_x.T, origin='lower', cmap='RdBu')
    axs[0].set_title('Distortion X')
    axs[1].imshow(distortion_y.T, origin='lower', cmap='RdBu')
    axs[1].set_title('Distortion Y')
    plt.tight_layout()
    plt.show()

def visualize_distortion_scatter(uv_coords, distortions, distortion_mag_factor=5, cutoff=10):
    u_filtered_ids = distortions[:, 0] < cutoff
    v_filtered_ids = distortions[:, 1] < cutoff
    filtered_ids = u_filtered_ids & v_filtered_ids
    u = uv_coords[:, 0][filtered_ids]
    v = uv_coords[:, 1][filtered_ids]
    stretch_u = distortions[:, 0][filtered_ids]
    stretch_v = distortions[:, 1][filtered_ids]

    umin, umax = u.min(), u.max()
    vmin, vmax = v.min(), v.max()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].scatter(u, v, s=stretch_u * distortion_mag_factor, alpha=0.6, c='blue')
    axs[0].set_title('U Stretch Factor (Size Encoded)')
    axs[0].set_xlim(umin, umax)
    axs[0].set_ylim(vmin, vmax)
    axs[0].set_aspect('equal')
    axs[0].grid(True)

    axs[1].scatter(u, v, s=stretch_v * distortion_mag_factor, alpha=0.6, c='green')
    axs[1].set_title('V Stretch Factor (Size Encoded)')
    axs[1].set_xlim(umin, umax)
    axs[1].set_ylim(vmin, vmax)
    axs[1].set_aspect('equal')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def visualize_uv_projection(uv_coords, heatmap=False, cmap='viridis'):
    """
    Visualizes UV coordinates with uniform axis scaling and limits
    suitable for cylindrical unwrapping. Optionally includes a point density heatmap.

    Args:
        uv_coords (np.ndarray): An array of UV coordinates with shape (N, 2).
                                Assumes U is in the range [0, 2*pi] and V is in [0, 1].
        heatmap (bool): If True, colors the points based on 2D density using KDE.
        cmap (str): Colormap to use for the heatmap if enabled.
    """
    plt.figure(figsize=(6, 6))

    if heatmap:
        # Estimate density using Gaussian KDE
        kde = gaussian_kde(uv_coords.T)
        density = kde(uv_coords.T)

        # Sort points by density (low to high) for better visualization layering
        idx = density.argsort()
        uv_sorted = uv_coords[idx]
        density_sorted = density[idx]

        plt.scatter(uv_sorted[:, 0], uv_sorted[:, 1], c=density_sorted, s=5, cmap=cmap, alpha=0.7)
    else:
        plt.scatter(uv_coords[:, 0], uv_coords[:, 1], s=1, alpha=0.6)

    plt.title('Projected UV Coordinates (Uniform Scaling)')
    plt.xlabel('U')
    plt.ylabel('V')

    # Set axis limits to cover the entire UV space
    plt.xlim(np.min(uv_coords[:, 0]), np.max(uv_coords[:, 0]))
    plt.ylim(np.min(uv_coords[:, 1]), np.max(uv_coords[:, 1]))

    # Equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_3d_points(volume_points_zyx, volume_shape_zyx=None, highlighted_points_idx=None, extra_points_zyx=None, title="Original 3D Points"):
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
        max_lim = max(volume_shape_zyx)
        ax.set_xlim([0, max_lim])
        ax.set_ylim([0, max_lim])
        ax.set_zlim([0, max_lim])

    ax.set_aspect('equal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def random_points_on_sphere_normal(n_points=1000, radius=0.8):
    """Generates random points on a sphere using a normal distribution.
       This is a simpler method, but the distribution is not perfectly uniform.
    """
    points = np.random.normal(size=(n_points, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True)  # Normalize to unit vectors
    points = points * radius # Scale to the desired radius
    return points

def generate_meridian_points_vectorized(radius=0.8, num_meridians=20, points_per_meridian=20):
    """
    Generates points on a sphere arranged to mimic meridians (lines of longitude),
    using vectorized NumPy operations for efficiency.

    Args:
        radius: The radius of the sphere.
        num_meridians: The number of meridians to generate.
        points_per_meridian: The number of points to generate along each meridian.

    Returns:
        A numpy array of shape (num_meridians * points_per_meridian, 3) containing the
        3D coordinates of the points.
    """

    # Create arrays of meridian and latitude angles
    meridian_indices = np.arange(num_meridians)
    phi = 2 * np.pi * meridian_indices / num_meridians  # Shape: (num_meridians,)

    if points_per_meridian > 1:
        latitude_indices = np.arange(points_per_meridian)
        theta = np.pi * latitude_indices / (points_per_meridian - 1)  # Shape: (points_per_meridian,)
    else:
        theta = np.array([np.pi/2]) #Special case

    # Create a meshgrid of angles
    phi, theta = np.meshgrid(phi, theta)  # Shape: (points_per_meridian, num_meridians)

    # Flatten the angle arrays
    phi = phi.flatten()  # Shape: (num_meridians * points_per_meridian,)
    theta = theta.flatten()  # Shape: (num_meridians * points_per_meridian,)

    # Convert to Cartesian coordinates using vectorized operations
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Stack the coordinates into a single array
    points = np.stack((x, y, z), axis=-1)  # Shape: (num_meridians * points_per_meridian, 3)

    return points

def sample_surface_points_from_convex_hull(
    points: np.ndarray,
    n_samples: int = 10000,
    method: str = "uniform"
) -> np.ndarray:
    """
    Uniformly sample points from the surface of a convex hull mesh.

    Parameters:
    -----------
    points : np.ndarray
        Input 3D point cloud, shape (N, 3).
    n_samples : int
        Number of surface points to sample.
    method : str
        Sampling method: "uniform" or "poisson".

    Returns:
    --------
    sampled_points : np.ndarray
        Sampled surface points, shape (n_samples, 3).
    """
    # Step 1: Compute convex hull
    hull = ConvexHull(points)

    # Step 2: Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(hull.simplices)
    mesh.compute_vertex_normals()

    # Step 3: Sample surface points
    if method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=n_samples)
    elif method == "poisson":
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_samples, init_factor=5)
    else:
        raise ValueError("Invalid method. Choose 'uniform' or 'poisson'.")

    sampled_points = np.asarray(pcd.points)
    return sampled_points

if __name__ == "__main__":
    # Generate test mesh: small sphere
    # mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

    # points = mesh.vertices[:, [2, 1, 0]]
    points = np.load("outs/hull_embryo_surface_points.npy")[:, [2, 1, 0]]
    print(f"Points z: {min(points[:, 0])} to {max(points[:, 0])}, y: {min(points[:, 1])} to {max(points[:, 1])}, x: {min(points[:, 2])} to {max(points[:, 2])}")
    print("Loaded points: ", len(points))
    points = sample_surface_points_from_convex_hull(points)
    if False:
        from scipy.spatial import ConvexHull
        import trimesh
        hull = ConvexHull(points)
        hull_vertices = points[hull.vertices]
        hull_faces = hull.simplices

        # Step 2: Create initial mesh from convex hull
        mesh = trimesh.Trimesh(vertices=points, faces=hull_faces)
        # Step 3: Subdivide the mesh to increase triangle density
        # Subdivide using trimesh remeshing (loop subdivision)
        mesh = mesh.subdivide()
        # mesh = mesh.subdivide()
        # mesh = mesh.subdivide_to_size(max_edge=10)

        # Optional: Further subdivision if needed
        # mesh = mesh.subdivide()

        # Step 4: Get dense vertices (passed further down)
        dense_vertices = mesh.vertices
        points = dense_vertices
    print("Generated dense hull mesh: ", len(points), "vertices")
    
    
    vol_shape = tiff.imread("outs/down_cropped_tp_300.tif").shape
    points[:,[0]] = vol_shape[0] - points[:,[0]] -1 # Flip z axis
    points = points[points[:, 0] > min(points[:, 0])]
    print(f"Volume shape: {vol_shape}")
    # points = random_points_on_sphere_normal(n_points=6000)
    # points = generate_meridian_points_vectorized()
    # half_sphere_points = points[points[:, 0] >= 0]
    max_r = round(vol_shape[1] / 2.0 * 1.15)
    uv_coords, points_on_cyl = project_to_cylinder(points, radius=max_r, origin_yz=(vol_shape[1]//2, 0))
    # neighbors = gpu_knn_search(points, k=7)
    # distortions = estimate_local_distortion_gpu(points, uv_coords, neighbors)
    # distortion_x, distortion_y = rasterize_distortion_map(uv_coords, distortions)
    # visualize_distortion_map(distortion_x, distortion_y)
    # visualize_distortion_scatter(uv_coords, distortions, distortion_mag_factor=5)
    visualize_uv_projection(uv_coords, heatmap=True)
    # visualize_3d_points(points, highlighted_points_idx=neighbors[1])
    visualize_3d_points(points, extra_points_zyx=points_on_cyl)
