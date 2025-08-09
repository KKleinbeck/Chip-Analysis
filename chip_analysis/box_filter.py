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
    outlierQuantiles: list[float] = field(default_factory=lambda: [0.075, 0.25])
    bwThreshold: float = 0.3
    bwInvert: bool = True
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
        self._0_outlierFree = None
        self._1_bw = None
        self._2_culled = None
        self._3_noIslands = None
        self._4_aspectRatioFiltered = None

        self.result = None
    
    def filter(self):
        print("Commencing image processing...")

        print("  1/5 - Removing outliers...")
        self.removeOutliers()
        print("  2/5 - Black-white filtering...")
        self.blackWhiteFilter()
        print("  3/5 - Culling boundary pixel...")
        self.cullBoundary()
        print("  4/5 - Removing pixel islands...")
        self.removeIslands()
        print("  5/5 - Aspect ratio filter...")
        self.aspectRatioFilter()

        self.result = np.copy(self._4_aspectRatioFiltered)
        self._isclean = True

    @mark_dirty
    def removeOutliers(self):
        """
        Removes outliers from the image.
        For this every pixel value below or above the quantile specified in the config parameter
        `outlierQuantiles` is set to the respective quantile values. Then the image is normalised
        to a [0,1] intensity range.
        """
        # Clamp everything into the threshold range
        imgs = np.copy(self.imgs)
        thresholds = np.quantile(imgs, self.config.outlierQuantiles, [1, 2])
        lower = thresholds[0,:,None,None]
        upper = thresholds[1,:,None,None]
        imgs = np.clip(imgs, lower, upper)

        # Normalise data to [0, 1] range
        imgs = imgs - lower
        imgs = imgs / (upper - lower)
        
        self._0_outlierFree = imgs
    
    @mark_dirty
    def blackWhiteFilter(self):
        """
        Creates a black and white map.
        Sets all pixel below the config parameter `bwThreshold` to 0 and to 1 otherwise.
        If the config parameter `bwInvert` is set to true, the images are inverted afterwards.
        """
        imgs = np.copy(self._0_outlierFree)
        imgs[imgs <  self.config.bwThreshold] = 0
        imgs[imgs >= self.config.bwThreshold] = 1
        if self.config.bwInvert:
            imgs = 1 - imgs # invert

        self._1_bw = imgs
    
    @mark_dirty
    def cullBoundary(self):
        """
        Removes a fixed amount of pixels from each side of the image.
        The amount of removed pixels are specified in the config parameter `boundaryPixelRemoval`,
        in either left-right and top-bottom, or left, top, right, bottom order.
        """
        imgs = np.copy(self._1_bw)
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
        self._2_culled = imgs.copy()
    
    @mark_dirty
    def removeIslands(self):
        """
        Removes pixel island below and above certain thresholds.
        At this point we expect that the features we want to extract is given by a feature with a
        roughly known pixel count. Every other feature with less or more pixel than specified in
        the config parameter `pixelSizeThreshold` will thusly be removed.
        """
        imgs = np.copy(self._2_culled)
        tSmall, tLarge = self.config.pixelSizeThresholds
        for n in range(imgs.shape[0]):
            s = nd.generate_binary_structure(2,2)
            labelled, _ = nd.label(imgs[n,:,:], structure=s)
            for indices in nd.value_indices(labelled).values():
                # Remove regions larger or smaller than a pixel threshold
                if indices[0].size < tSmall or indices[0].size > tLarge:
                    imgs[n,indices] = 0
        self._3_noIslands = imgs

    @mark_dirty
    def aspectRatioFilter(self):
        """
        Filter features on their physical dimensions.
        This checks whether the remaining features have an aspect ratio as specified by the
        config parameter `targetAspect`.
        """
        imgs = np.copy(self._3_noIslands)
        # sLow, sHigh = self.config.targetSize
        maxAspect = self.config.targetAspect
        for n in range(imgs.shape[0]):
            labelled, _ = nd.label(imgs[n])
            for indices in nd.value_indices(labelled).values():
                dX = np.max(indices[0]) - np.min(indices[0])
                dY = np.max(indices[1]) - np.min(indices[1])
                if maxAspect > dX / dY > (1/maxAspect): #or sLow > min(dX,dY) or sHigh < max(dX,dY):
                    imgs[n,indices] = 0
        self._4_aspectRatioFiltered = imgs

    def visualise(self):
        if self.result is None:
            raise ValueError("Result is not ready yet. Run the filter method first")
        if not self._isclean:
            RuntimeWarning(
                "Result is not clean. This implies some intermediate results have been tempered with." +
                "Plotting results regardless. Run the `filter` method again to guarantee reproducible results"
            )
        
        nRows = self.imgs.shape[0]
        _, axes = plt.subplots(nRows, 4, figsize=(16, 4*nRows)) 
        for index in range(nRows):
            axes[index,0].imshow(self._1_bw[index,:,:], cmap='gray')
            axes[index,0].set_title('Quantile Filtered')
            axes[index,0].axis('off')
            
            labelled, _ = nd.label(self._2_culled[index,:,:])
            axes[index,1].imshow(labelled)
            axes[index,1].set_title('Black white and outlier filtered - labelled')
            axes[index,1].axis('off')

            axes[index,2].imshow(self._3_noIslands[index,:,:])
            axes[index,2].set_title('Removed islands')
            axes[index,2].axis('off')

            axes[index,3].imshow(self.result[index,:,:], cmap='gray')
            axes[index,3].set_title('Aspect Area filtered')
            axes[index,3].axis('off')

        plt.tight_layout()
        plt.show()