import numpy as np
import trimesh
import torch
import matplotlib.pyplot as plt
import tifffile as tiff

from pytorch3d.structures import Meshes
from pytorch3d.ops import ray_mesh_intersect



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

def load_mesh_from_point_cloud(npy_path):
    """
    Load a point cloud from a .npy file and compute the convex hull mesh.
    Returns a PyTorch3D Mesh.
    """
    points = np.load(npy_path)[:, [2, 1, 0]]
    cloud = trimesh.points.PointCloud(points)
    mesh = cloud.convex_hull
    verts = torch.tensor(mesh.vertices, dtype=torch.float32).unsqueeze(0)
    faces = torch.tensor(mesh.faces, dtype=torch.int64).unsqueeze(0)
    return Meshes(verts=verts, faces=faces).to("cuda")


def perform_ray_mesh_intersection(mesh_p3d, ray_origins, ray_directions, max_hits=1):
    ray_lengths, hit_faces, bary_coords = ray_mesh_intersect(
        meshes=mesh_p3d,
        rays_origins=ray_origins,
        rays_directions=ray_directions,
        max_hits=max_hits
    )
    mask = ray_lengths[0][:, 0] > 0
    hit_lengths = ray_lengths[0][mask]
    hit_dirs = ray_directions[0][mask]
    hit_origins = ray_origins[0][mask]
    hit_points = hit_origins + hit_dirs * hit_lengths
    return hit_points


def sparse_grid_on_half_cylinder(
    image_shape,
    spacing_x=7,
    spacing_theta=7,
    radius=1.0,
    origin_yz=(0.0, 0.0)
):
    height, width = image_shape
    y0, z0 = origin_yz

    num_points_theta = max(1, int(width) // spacing_theta)
    num_points_x = max(1, int((height) // spacing_x))

    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points_theta)
    x = np.arange(0, num_points_x * spacing_x, spacing_x)

    theta_grid, x_grid = np.meshgrid(theta, x)

    z = radius * np.cos(theta_grid) + z0
    y = radius * np.sin(theta_grid) + y0

    points_3d = np.column_stack([z.ravel(), y.ravel(), x_grid.ravel()])
    return points_3d


# Load mesh from point cloud .npy file
mesh_p3d = load_mesh_from_point_cloud("outs/hull_embryo_surface_points.npy")

# Get image volume shape
vol_shape = tiff.imread("outs/down_cropped_tp_300.tif").shape
print(f"Volume shape: {vol_shape}")
max_r = round(vol_shape[1] / 2.0 * 1.15)

# Generate grid points on half-cylinder surface
image_shape = (vol_shape[2], round(np.pi * max_r + 1))
cylinder_radius = max_r
spacing_x = 4
spacing_theta = 4
cylinder_points_zyx = sparse_grid_on_half_cylinder(
    image_shape=image_shape,
    spacing_x=spacing_x,
    spacing_theta=spacing_theta,
    radius=cylinder_radius,
    origin_yz = (vol_shape[1]//2, 0)
)

# Convert ZYX to XYZ for processing
surface_points = torch.tensor(cylinder_points_zyx[:, [2, 1, 0]], dtype=torch.float32)

# Ray directions and origins
ray_origins = torch.zeros_like(surface_points)  # All rays from center axis (0, 0, 0)
ray_directions = torch.nn.functional.normalize(surface_points - ray_origins, dim=1)

# Reshape for batch ray casting
ray_origins = ray_origins.unsqueeze(0).to("cuda")       # (1, N, 3)
ray_directions = ray_directions.unsqueeze(0).to("cuda") # (1, N, 3)

# Perform ray-mesh intersection
hit_points = perform_ray_mesh_intersection(mesh_p3d, ray_origins, ray_directions)

# Print or export intersection points
print("Total Intersections:", hit_points.shape[0])
hit_points = hit_points.cpu().numpy()

visualize_3d_points(hit_points, extra_points_zyx=cylinder_points_zyx)
