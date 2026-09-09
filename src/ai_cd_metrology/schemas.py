"""Shared records used by every metrology implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path


class MetrologyMethod(str, Enum):
    """Width/gap measurement methods planned for comparison."""

    INTENSITY_THRESHOLD = "INTENSITY_THRESHOLD"
    GRADIENT_CANNY = "GRADIENT_CANNY"
    UNET = "UNET"


class MeasurementStatus(str, Enum):
    """Validity state of a measurement result."""

    VALID = "VALID"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """Identity and acquisition metadata for one image."""

    image_id: str
    file_path: Path
    wafer_id: str
    die_id: str
    pattern_position: str
    capture_region: str
    defect_label: str | None = None


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """Pixel scales associated with one microscope calibration."""

    calibration_id: str
    objective: str
    scale_x_um_per_px: float
    scale_y_um_per_px: float

    def __post_init__(self) -> None:
        for field_name in ("scale_x_um_per_px", "scale_y_um_per_px"):
            scale = getattr(self, field_name)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f"{field_name} must be a finite positive value")


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    """Method-independent representation of one outer/inner/gap measurement."""

    image_id: str
    method: MetrologyMethod
    measurement_id: str
    outer_width_px: float | None
    inner_width_px: float | None
    gap_px: float | None
    outer_width_um: float | None
    inner_width_um: float | None
    gap_um: float | None
    calibration_id: str | None = None
    edge_coordinates: list[tuple[float, float]] = field(default_factory=list)
    status: MeasurementStatus = MeasurementStatus.VALID
    failure_reason: str | None = None
    runtime_ms: float | None = None
