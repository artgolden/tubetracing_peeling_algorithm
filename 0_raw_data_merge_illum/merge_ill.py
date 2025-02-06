import numpy as np
import tifffile as tiff

# Load the two TIFF images
image1 = tiff.imread("timelapseID-20240926-211701_SPC-0002_TP-0400_ILL-0_CAM-1_CH-00_PL-(ZS)-outOf-0072.tif")
image2 = tiff.imread("timelapseID-20240926-211701_SPC-0002_TP-0400_ILL-1_CAM-1_CH-00_PL-(ZS)-outOf-0072.tif")

# Compute the average image
average_image = np.mean([image1, image2], axis=0).astype(image1.dtype)

# Save the averaged image
tiff.imwrite("../outs/merged_ill.tif", average_image)
