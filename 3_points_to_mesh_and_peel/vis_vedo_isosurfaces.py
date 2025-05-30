import numpy as np
import tifffile
import os
from vedo import dataurl, Volume, Text2D
from vedo.applications import IsosurfaceBrowser

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

data_matrix = load_3d_volume("../outs/down_cropped.tif")

vol = Volume(data_matrix)
# vol.cmap(['white','b','g','r']).mode(1)
# vol.add_scalarbar()

# IsosurfaceBrowser(Plotter) instance:
plt = IsosurfaceBrowser(vol, use_gpu=True, c='gold')

plt.show(axes=7, bg2='lb').close()
