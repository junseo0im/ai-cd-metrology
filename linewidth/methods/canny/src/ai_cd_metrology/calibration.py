"""Conversion helpers kept separate from image measurement algorithms."""

from typing import Literal

from .schemas import CalibrationRecord

MeasurementAxis = Literal["x", "y"]


def pixels_to_micrometers(
    value_px: float | None,
    calibration: CalibrationRecord | None,
    *,
    axis: MeasurementAxis,
) -> float | None:
    """Convert a pixel distance along one image axis to micrometers.

    A missing pixel value or calibration intentionally produces ``None`` so
    pixel-only measurements remain representable.
    """

    if value_px is None or calibration is None:
        return None

    if axis == "x":
        scale = calibration.scale_x_um_per_px
    elif axis == "y":
        scale = calibration.scale_y_um_per_px
    else:
        raise ValueError("axis must be 'x' or 'y'")

    return value_px * scale


def convert_pixel_measurements(
    outer_width_px: float | None,
    inner_width_px: float | None,
    gap_px: float | None,
    calibration: CalibrationRecord | None,
    *,
    axis: MeasurementAxis,
) -> tuple[float | None, float | None, float | None]:
    """Convert outer width, inner width, and gap along one explicit axis."""

    return (
        pixels_to_micrometers(outer_width_px, calibration, axis=axis),
        pixels_to_micrometers(inner_width_px, calibration, axis=axis),
        pixels_to_micrometers(gap_px, calibration, axis=axis),
    )

