"""Fixed depth/height colours shared by the live pipeline and diagnostics."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Renderer:
    low: float
    high: float
    bright: bool = False
    ascending: bool = False

    def __post_init__(self):
        if not np.isfinite([self.low, self.high]).all() or self.high <= self.low:
            raise ValueError('Colour limits must be finite and increasing')

    def colorize(self, values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        normalized = np.subtract(values, self.low)
        normalized /= self.high - self.low
        np.clip(normalized, 0, 1, out=normalized)
        if not self.ascending:
            np.subtract(1, normalized, out=normalized)
        # The clipped values are nonnegative, so OpenCV can combine scaling,
        # rounding and the uint8 conversion in one pass. Invalid pixels are
        # masked in the final colour image.
        indices = (cv2.convertScaleAbs(normalized, alpha=204, beta=25.5) if self.bright
                   else cv2.convertScaleAbs(normalized, alpha=255))
        colors = cv2.applyColorMap(indices, cv2.COLORMAP_TURBO)
        colors[~valid] = 0
        return colors
