import tifffile as tiff
import numpy as np

with tiff.TiffFile("/nfs5/zyh/MUF-Clust/single_image_output/segmentation/11_Scan1/features/features_matrix_mean_cluster_cluster_map.tiff") as tif:
    for i, level in enumerate(tif.series[0].levels):
        img = level.asarray()
        non_zero = np.count_nonzero(img)
        print(f"Level {i}: shape={img.shape}, non_zero_pixels={non_zero}")