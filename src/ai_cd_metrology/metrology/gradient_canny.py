"""Gradient/Canny metrology placeholder for the next implementation phase."""

import numpy as np
from numpy.typing import NDArray

from ..schemas import CalibrationRecord, ImageRecord, MeasurementResult, MetrologyMethod
from .base import MetrologyAlgorithm


class GradientCannyMetrology(MetrologyAlgorithm):
    """Skeleton only; ROI, edge detection, and edge pairing are still TBD."""

    @property
    def method(self) -> MetrologyMethod:
        return MetrologyMethod.GRADIENT_CANNY

    def measure(
        self,
        image: NDArray[np.generic],
        image_record: ImageRecord,
        calibration: CalibrationRecord | None = None,
    ) -> list[MeasurementResult]:
        raise NotImplementedError(
            "Gradient/Canny ROI selection, edge detection, edge pairing, and "
            "width/gap calculation are not implemented in PHASE A."
        )

