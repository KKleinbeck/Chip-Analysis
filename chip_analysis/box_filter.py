import numpy as np
import tifffile as tiff
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from pathlib import Path

def mark_dirty(method):
    def wrapper(self, *args, **kwargs):
        self._isClean = False
        return method(self, *args, **kwargs)
    return wrapper

@dataclass
class BoxFilterConfig:
    outlierQuantile: float = 0.1
    invert: bool = True
    boundaryPixelRemoval: list[int] = field(default_factory=lambda: [400, 200])
    pixelSizeThresholds: list[int] = field(default_factory=lambda: [2_500, 50_000])
    targetSize: list[float] = field(default_factory=lambda: [210., 350.])
    targetAspect: float = 2.0

class BoxFilter:
    def __init__(self, tiffFile: Path, boxFilterConfig: BoxFilterConfig = BoxFilterConfig()):
        self.imgs = tiff.imread(tiffFile)
        self.config = boxFilterConfig
        self._isclean = False

        # Intermediate step containers
        self._outlierFree = None
        self._denoised = None
        self._culled = None
        self._aspectRatioFiltered = None

        self.result = None
    
    def filter(self):
        print("Commencing image processing...")

        print("  1/4 - Removing outliers...")
        self.removeOutliers()
        print("  2/4 - Removing outliers...")
        self.denoise()
        print("  3/4 - Culling boundary pixel...")
        self.cullBoundary()
        print("  4/4 - Aspect ratio filter...")
        self.aspectRatioFilter()

        self.result = np.copy(self._aspectRatioFiltered)
        self._isclean = True

    @mark_dirty
    def removeOutliers(self):
        """
        Removes outliers from the image.
        For this every pixel value below or above the quantile specified in the config parameter
        `outlierQuantiles` is set to the respective quantile values. Then the image is normalised
        to a [0,1] intensity range.
        """
        imgs = np.copy(self.imgs)
        threshold = np.quantile(imgs, self.config.outlierQuantile, [1, 2])
        imgs[imgs < threshold[:,None,None]] = 0
        imgs[imgs > 0] = 1
        if self.config.invert:
            imgs = 1 - imgs # invert

        self._outlierFree = imgs
    
    @mark_dirty
    def denoise(self):
        imgs = np.copy(self._outlierFree)
        for n in range(imgs.shape[0]):
            imgs[n,:,:] = nd.binary_opening(
                imgs[n,:,:], structure = nd.generate_binary_structure(2, 2), iterations=1
            )
        self._denoised = imgs
    
    @mark_dirty
    def cullBoundary(self):
        """
        Removes a fixed amount of pixels from each side of the image.
        The amount of removed pixels are specified in the config parameter `boundaryPixelRemoval`,
        in either left-right and top-bottom, or left, top, right, bottom order.
        """
        imgs = np.copy(self._denoised)
        if len(self.config.boundaryPixelRemoval) == 2:
            l, t = self.config.boundaryPixelRemoval
            r, b = l, t
        elif len(self.config.boundaryPixelRemoval) == 4:
            l, t, r, b = self.config.boundaryPixelRemoval
        else:
            raise ValueError("Config parameter `bounadryPixelRemoval` shall be an array of size 2 or 4.")
        imgs[:,:l,:] = 0
        imgs[:,-r:,:] = 0
        imgs[:,:,:t] = 0
        imgs[:,:,-b:] = 0
        self._culled = imgs.copy()

    @mark_dirty
    def aspectRatioFilter(self):
        """
        Filter features on their physical dimensions.
        This checks whether the remaining features have an aspect ratio as specified by the
        config parameter `targetAspect` or whether they fall outside of the pixel size limits
        (per coordinate axis) defined by `targetSize`.
        """
        imgs = np.copy(self._culled)
        sLow, sHigh = self.config.targetSize
        maxAspect = self.config.targetAspect
        for n in range(imgs.shape[0]):
            labelled, _ = nd.label(imgs[n])
            for indices in nd.value_indices(labelled).values():
                dX = np.max(indices[0]) - np.min(indices[0])
                dY = np.max(indices[1]) - np.min(indices[1])
                if not maxAspect > dX / (dY + 1e-6) > (1/maxAspect) or sLow > min(dX, dY) or sHigh < max(dX, dY):
                    imgs[n,*indices] = 0
        self._aspectRatioFiltered = imgs

    def visualise(self):
        if self.result is None:
            raise ValueError("Result is not ready yet. Run the filter method first")
        if not self._isclean:
            RuntimeWarning(
                "Result is not clean. This implies some intermediate results have been tempered with." +
                "Plotting results regardless. Run the `filter` method again to guarantee reproducible results"
            )
        
        nRows = self.imgs.shape[0]
        _, axes = plt.subplots(nRows, 3, figsize=(12, 4*nRows)) 
        for index in range(nRows):
            axes[index,0].imshow(self._outlierFree[index,:,:], cmap='gray')
            axes[index,0].set_title('Quantile Filtered')
            axes[index,0].axis('off')
            
            axes[index,1].imshow(self._culled[index,:,:], cmap='gray')
            axes[index,1].set_title('Denoised and culled')
            axes[index,1].axis('off')

            axes[index,2].imshow(self.result[index,:,:], cmap='gray')
            axes[index,2].set_title('Aspect ratio filtered')
            axes[index,2].axis('off')

        plt.tight_layout()
        plt.show()