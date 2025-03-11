import numpy as np
import trimesh
import torch
import faiss
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def project_to_cylinder(vertices, radius=1.0):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    theta = np.arctan2(x, z)
    u = (theta + np.pi) #/ (2 * np.pi)  # Normalize to [0, 1]
    v = (y - y.min()) #/ (y.max() - y.min())  # Normalize height
    uv = np.column_stack([u, v])
    P_cyl = np.column_stack([y, radius * np.sin(theta), radius * np.cos(theta)])
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

    # Solve least squares per vertex: A * J^T = B  => J = (A^T A)^-1 A^T B
    AT = A.transpose(1, 2)              # [N, 3, K]
    ATA = AT.bmm(A)                     # [N, 3, 3]
    ATB = AT.bmm(B)                     # [N, 3, 2]

    J = torch.linalg.solve(ATA, ATB)    # [N, 3, 2]
    J = J.transpose(1, 2)               # [N, 2, 3]

    ideal_J = torch.tensor([[1, 0, 0], [0, 1, 0]], device=device).expand(N, -1, -1)
    dist = (J - ideal_J)[:, :, :2].mean(dim=2)  # [N, 2]

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


def visualize_uv_projection(uv_coords):
    """
    Visualizes UV coordinates with uniform axis scaling and limits
    suitable for cylindrical unwrapping.

    Args:
        uv_coords (np.ndarray): An array of UV coordinates with shape (N, 2).
                                 Assumes U is in the range [0, 2*pi] and V is in [0, 1].
    """

    plt.figure(figsize=(6, 6))
    plt.scatter(uv_coords[:, 0], uv_coords[:, 1], s=1, alpha=0.6)
    plt.title('Projected UV Coordinates (Uniform Scaling)')
    plt.xlabel('U')
    plt.ylabel('V')

    # Set axis limits to cover the entire UV space
    plt.xlim(0, 2 * np.pi)
    plt.ylim(np.min(uv_coords[:, 1]), np.max(uv_coords[:, 1]))

    # Ensure equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')  # 'box' ensures scaling based on visible data

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

    ax.scatter(x, y, z, c='blue', s=1, alpha=0.5)
    if highlighted_points_idx is not None:
        ax.scatter(x[highlighted_points_idx], 
                y[highlighted_points_idx], 
                z[highlighted_points_idx], 
                c='red', s=10, alpha=0.9)
    if extra_points_zyx is not None:
        ax.scatter(extra_points_zyx[:, 2], extra_points_zyx[:, 1], extra_points_zyx[:, 0], c='green', s=1, alpha=0.7)
    

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

def random_points_on_sphere_normal(n_points=10000, radius=0.8):
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

if __name__ == "__main__":
    # Generate test mesh: small sphere
    # mesh = trimesh.creation.icosphere(subdivisions=3, radius=.8)
    # points = mesh.vertices
    # points = random_points_on_sphere_normal()
    points = generate_meridian_points_vectorized()
    print("Generated test mesh: sphere with", len(points), "vertices")

    uv_coords, points_on_cyl = project_to_cylinder(points)
    neighbors = gpu_knn_search(points, k=7)
    # distortions = estimate_local_distortion_gpu(points, uv_coords, neighbors)
    # distortion_x, distortion_y = rasterize_distortion_map(uv_coords, distortions)
    # visualize_distortion_map(distortion_x, distortion_y)
    visualize_uv_projection(uv_coords)
    # visualize_3d_points(points, highlighted_points_idx=neighbors[1])
    visualize_3d_points(points, extra_points_zyx=points_on_cyl)
