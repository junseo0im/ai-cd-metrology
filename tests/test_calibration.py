import math

import pytest

from ai_cd_metrology.calibration import (
    convert_pixel_measurements,
    pixels_to_micrometers,
)
from ai_cd_metrology.schemas import CalibrationRecord


def test_pixels_to_micrometers_uses_selected_axis() -> None:
    calibration = CalibrationRecord(
        calibration_id="cal-001",
        objective="TBD-objective",
        scale_x_um_per_px=0.25,
        scale_y_um_per_px=0.5,
    )

    assert pixels_to_micrometers(8.0, calibration, axis="x") == pytest.approx(2.0)
    assert pixels_to_micrometers(8.0, calibration, axis="y") == pytest.approx(4.0)


def test_missing_calibration_keeps_physical_values_none() -> None:
    outer_width_um, inner_width_um, gap_um = convert_pixel_measurements(
        outer_width_px=12.0,
        inner_width_px=8.0,
        gap_px=6.0,
        calibration=None,
        axis="x",
    )

    assert outer_width_um is None
    assert inner_width_um is None
    assert gap_um is None


def test_calibration_converts_all_three_measurements() -> None:
    calibration = CalibrationRecord(
        calibration_id="cal-001",
        objective="TBD-objective",
        scale_x_um_per_px=0.25,
        scale_y_um_per_px=0.5,
    )

    outer_width_um, inner_width_um, gap_um = convert_pixel_measurements(
        outer_width_px=12.0,
        inner_width_px=8.0,
        gap_px=6.0,
        calibration=calibration,
        axis="x",
    )

    assert outer_width_um == pytest.approx(3.0)
    assert inner_width_um == pytest.approx(2.0)
    assert gap_um == pytest.approx(1.5)


@pytest.mark.parametrize("invalid_scale", [0.0, -0.1, math.inf, -math.inf, math.nan])
def test_calibration_rejects_invalid_scale(invalid_scale: float) -> None:
    with pytest.raises(ValueError):
        CalibrationRecord(
            calibration_id="invalid",
            objective="TBD-objective",
            scale_x_um_per_px=invalid_scale,
            scale_y_um_per_px=0.5,
        )
