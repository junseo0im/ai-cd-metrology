from pathlib import Path

from ai_cd_metrology.schemas import (
    ImageRecord,
    MeasurementResult,
    MeasurementStatus,
    MetrologyMethod,
)


def test_enum_values() -> None:
    assert MetrologyMethod.INTENSITY_THRESHOLD.value == "INTENSITY_THRESHOLD"
    assert MetrologyMethod.GRADIENT_CANNY.value == "GRADIENT_CANNY"
    assert MetrologyMethod.UNET.value == "UNET"
    assert {status.value for status in MeasurementStatus} == {
        "VALID",
        "WARNING",
        "FAIL",
    }


def test_schema_creation_supports_pixel_only_result() -> None:
    image_record = ImageRecord(
        image_id="w1-11-c",
        file_path=Path("data/raw/w1_11_c.png"),
        wafer_id="w1",
        die_id="11",
        pattern_position="center",
        capture_region="standard",
    )
    result = MeasurementResult(
        image_id=image_record.image_id,
        method=MetrologyMethod.GRADIENT_CANNY,
        measurement_id="measurement-001",
        outer_width_px=10.0,
        inner_width_px=7.0,
        gap_px=5.0,
        outer_width_um=None,
        inner_width_um=None,
        gap_um=None,
        edge_coordinates=[(10.0, 20.0), (20.0, 20.0)],
        status=MeasurementStatus.WARNING,
        failure_reason="Calibration is not available.",
        runtime_ms=1.5,
    )

    assert image_record.defect_label is None
    assert result.outer_width_px == 10.0
    assert result.inner_width_px == 7.0
    assert result.outer_width_um is None
    assert result.inner_width_um is None
    assert result.status is MeasurementStatus.WARNING
