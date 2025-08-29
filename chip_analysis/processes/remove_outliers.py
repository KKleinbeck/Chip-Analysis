import numpy as np

from chip_analysis.pipeline_framework.process_step import AbstractProcessStep, process_steps

class RemoveOutliers(AbstractProcessStep):
  inputs = {"input_stack": np.ndarray,}
  deliverables = {"filtered_stack": np.ndarray,}

  options = {"filtered_stack": ([float, tuple[float, float]], 0.1),}

  def _configurate(self):
     self.outlier_quantile = 0.1
     self.invert = False

  def _execute(self):
    """
    Removes outliers from the image stack.

    For this every pixel value below or above the quantile specified in the config parameter
    `outlierQuantiles` is set to the respective quantile values. Then the image is normalised
    to a [0,1] intensity range.
    """
    threshold = np.quantile(self.input_stack, self.outlier_quantile, [1, 2])
    self.input_stack[self.input_stack < threshold[:,None,None]] = 0
    self.input_stack[self.input_stack > 0] = 1
    if self.invert:
        self.input_stack = 1 - self.input_stack

    self.filtered_stack = self.input_stack

process_steps["RemoveOutliers"] = RemoveOutliers