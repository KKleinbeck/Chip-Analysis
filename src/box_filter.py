import numpy as np
import cv2
import tifffile as tiff
import scipy.ndimage as nd
import matplotlib.pyplot as plt

from pathlib import Path

# Open a TIFF stack
data = Path(r"L:\AG data\20250415 chip\exported\Line 1_sfGFP-PopZ 20nM\Cleaned\test\Cleaned")
file = data / "20250415 Chip 10-1min long incubation_sfGFP-PopZ_sfGFP-IDP40G_TXTL assembly-dissassembly_TileScan 1 - synthesis and inactive lys incubation_Line 1_0_Merged_BF.tif"
file1 = data / "20250425_diff DNA amount test_sfGFP_sfGFP-PopZ with mScarletI in bckgr_TileScan 1_Line 1_0_Merged_BF.tif"
file2 = data / "20250429 chip_sfGFP-PopZ 20-5nM with without mScarletI bckgrd 2nM_TileScan 1_Line 2, sfGFP-PopZ 20nM_250_Merged_BF.tif"

if __name__ == "__main__":
    # Open a TIFF stack
    imgs = tiff.imread(file2)

    # Extract Data
    img = imgs[4,:,:] # image index, bounds

    # Filter the data
   # from scipy.ndimage import gaussian_filter
    #img = gaussian_filter(img, sigma=1)

    # 1. Remove outliers
    #quantiles 0.075 to 0.25 normal
    #quantiles very low
    quantiles = [0.075, 0.25]
    thresholds = np.quantile(img, quantiles)
    img[img < thresholds[0]] = thresholds[0]
    img[img > thresholds[1]] = thresholds[1]
    # 2. Normalise data
    img = img - thresholds[0]
    img = img / np.max(img)
    # 3. Black and White mapping
    bwThreshold = 0.3
    img[img <  bwThreshold] = 0
    img[img >= bwThreshold] = 1
    img = 1 - img # invert
    # 4. Coarse filtering - sets the pixels around the edge to 0. 100 pixels per direction are removed
    img[:400,:] = 0 # 400 from top
    img[1500:,:] = 0 # 700ish from bottom
    img[:,:200] = 0 # remove from the right
    img[:,-200:] = 0 # remove from the left
    bwImg = img.copy()
    # 5. Remove pixel island
    #img = nd.binary_opening(img, iterations=1, mask=img)
    labelled, nlabels = nd.label(img)
    thresholdSmall = 2_500
    thresholdLarge = 50_000
    for label, indices in nd.value_indices(labelled).items():
        # Remove regions larger or smaller than a pixel threshold
        if indices[0].size < thresholdSmall or indices[0].size > thresholdLarge:
            img[indices] = 0
    img[img > 1] = 1
    # 5. Filter based on aspect ratio
    labelled, nlabels = nd.label(img)
    for label, indices in nd.value_indices(labelled).items():
        dX = np.max(indices[0]) - np.min(indices[0])
        dY = np.max(indices[1]) - np.min(indices[1])
        print(dX, dY, dX / dY)
        if dX / dY > 2. or dY / dX > 2.0 or 210 > min(dX,dY) or 350 < max(dX,dY):
            img[indices] = 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
    axes[0].imshow(bwImg, cmap='gray')
    axes[0].set_title('Quantile Filtered')
    axes[0].axis('off')

    axes[1].imshow(labelled)
    axes[1].set_title('Pixel Area filtered - labelled')
    axes[1].axis('off')

    axes[2].imshow(img, cmap='gray')
    axes[2].set_title('Aspect Area filtered')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()