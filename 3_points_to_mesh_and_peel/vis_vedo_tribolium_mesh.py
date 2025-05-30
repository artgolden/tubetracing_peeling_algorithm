import numpy as np
import tifffile
import os
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

import numpy as np
import pandas as pd  # Recommended for cleaner handling of CSV files

def read_points_from_csv(filepath):
    """
    Reads 3D point coordinates from a CSV file into a NumPy array.

    Args:
        filepath (str): The path to the CSV file.  The CSV should have at least
                      three columns representing X, Y, and Z coordinates, respectively.

    Returns:
        np.ndarray: A NumPy array of shape (N, 3), where N is the number of points,
                    and each row contains the (X, Y, Z) coordinates of a point.
                    Returns None if there is an error reading the file.
    """
    try:
        # Use pandas for efficient and flexible CSV reading
        df = pd.read_csv(filepath)

        # Ensure there are at least 3 columns
        if df.shape[1] < 3:
            print("Error: CSV file must have at least 3 columns (X, Y, Z).")
            return None

        # Extract the first 3 columns as X, Y, and Z coordinates
        points = df.iloc[:, :3].to_numpy()  # Efficient conversion to NumPy array

        return points

    except FileNotFoundError:
        print(f"Error: File not found at path: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {filepath}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse CSV file: {filepath}.  Check the format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

points_raw = read_points_from_csv("../outs/surface_points.csv")
points_raw = points_raw[:, [2, 1, 0]]
points = Points(points_raw)
points = points.subsample(0.005)
print("points", points.bounds())

data_matrix = load_3d_volume("../outs/down_cropped.tif")
print("data_matrix", data_matrix.shape)
data_matrix = np.transpose(data_matrix, (2, 1, 0))
print("data_matrix", data_matrix.shape)
embryo = Volume(data_matrix)
embryo = embryo.threshold(below=17)
print("embryo", embryo.bounds())

plt = Plotter(shape=(1,5), axes=9)


plt.at(0).show(points)

hull = ConvexHull(points).alpha(0.2)
print("hull", hull.bounds())
plt.at(1).show(hull)

vol = hull.binarize(values=(255,0),dims=embryo.shape,spacing=[1,1,1], origin=(0,0,0))
eroded = vol.clone().dilate(neighbours=(3,3,3))


eroded.alpha([0,0.75]).cmap('blue5')
print("vol", eroded.bounds())
print("vol shape", eroded.shape)
plt.at(2).show(eroded)
eroded = eroded.threshold(above=1, replace_value=1)
eroded = eroded.threshold(below=254, replace_value=0)

iso = vol.isosurface().color("blue5")
print("iso", iso.bounds())
plt.at(3).show("..the volume is isosurfaced:", iso)

diff = embryo.clone().operation("*",eroded)
plt.at(4).show(diff)

np.save("../outs/down_cropped_minus_hull.npy", np.transpose(diff.tonumpy(), (2, 1, 0)))

plt.interactive().close()
