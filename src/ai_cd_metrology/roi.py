"""Normalized ROI conversion shared by diagnostic and metrology code."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


PixelBounds = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class NormalizedROI:
    """Half-open ROI bounds expressed in normalized image coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("normalized ROI values must be finite")
        if not (0.0 <= self.x_min < self.x_max <= 1.0):
            raise ValueError("normalized ROI must satisfy 0 <= x_min < x_max <= 1")
        if not (0.0 <= self.y_min < self.y_max <= 1.0):
            raise ValueError("normalized ROI must satisfy 0 <= y_min < y_max <= 1")


def normalized_roi_to_pixel_bounds(
    image_shape: tuple[int, ...],
    roi: NormalizedROI,
) -> PixelBounds:
    """Convert a normalized ROI to half-open ``(x0, y0, x1, y1)`` bounds."""

    if len(image_shape) < 2:
        raise ValueError("image_shape must include positive height and width")
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")

    x0 = math.floor(roi.x_min * width)
    y0 = math.floor(roi.y_min * height)
    x1 = math.ceil(roi.x_max * width)
    y1 = math.ceil(roi.y_max * height)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("normalized ROI converts outside the image bounds")
    return x0, y0, x1, y1


def crop_normalized_roi(
    image: NDArray[np.generic],
    roi: NormalizedROI,
) -> tuple[NDArray[np.generic], PixelBounds]:
    """Return an isolated ROI copy and its half-open pixel bounds."""

    x0, y0, x1, y1 = normalized_roi_to_pixel_bounds(image.shape, roi)
    return image[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

