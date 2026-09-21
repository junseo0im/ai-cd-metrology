"""Method-independent interface for width/gap metrology."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..schemas import CalibrationRecord, ImageRecord, MeasurementResult, MetrologyMethod


class MetrologyAlgorithm(ABC):
    """Contract shared by threshold, Gradient/Canny, and U-Net methods."""

    @property
    @abstractmethod
    def method(self) -> MetrologyMethod:
        """Return the method identifier used in measurement results."""

    @abstractmethod
    def measure(
        self,
        image: NDArray[np.generic],
        image_record: ImageRecord,
        calibration: CalibrationRecord | None = None,
    ) -> list[MeasurementResult]:
        """Measure line widths and gaps without method-specific output types."""


