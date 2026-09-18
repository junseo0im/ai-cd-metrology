from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ai_cd_metrology.metrology.gradient_canny import (
    DarkBandPairingDiagnostic,
    EdgeDiagnosticResult,
    GradientCannyMeasurementDiagnostic,
    PairingValidationResult,
    SignedGradientProfile,
)
from ai_cd_metrology.roi import PixelBounds
from ai_cd_metrology.schemas import (
    ImageRecord,
    MeasurementResult,
    MeasurementStatus,
    MetrologyMethod,
)
from dashboard.adapters import (
    DashboardAnalysis,
    build_evaluation_record,
    load_dashboard_image,
    method_comparison_records,
)
from dashboard.data_loader import (
    canonical_pattern_position,
    catalog_for_scope,
    filter_catalog,
)


def test_canonical_pattern_position_maps_filename_aliases() -> None:
    assert canonical_pattern_position("c") == "center"
    assert canonical_pattern_position("lt") == "left-top"
    assert canonical_pattern_position("rb") == "right-bottom"
    assert canonical_pattern_position("missing") == "unknown"


def test_filter_catalog_uses_exact_identity_values() -> None:
    catalog = pd.DataFrame(
        [
            {"image_id": "a", "source": "S", "wafer_id": "w1", "die_id": "1", "pattern_position": "center", "status_label": "PASS"},
            {"image_id": "b", "source": "S", "wafer_id": "w2", "die_id": "2", "pattern_position": "left-top", "status_label": "WARNING"},
        ]
    )

    result = filter_catalog(catalog, wafer_id="w2", pattern_position="left-top")

    assert result["image_id"].tolist() == ["b"]


def test_catalog_for_scope_preserves_curated_default_and_full_inventory() -> None:
    inventory = pd.DataFrame(
        [
            {"image_id": "curated", "is_curated": True},
            {"image_id": "evaluation", "is_curated": False},
        ]
    )

    assert catalog_for_scope(inventory, "Curated only")["image_id"].tolist() == [
        "curated"
    ]
    assert catalog_for_scope(inventory, "Full inventory")["image_id"].tolist() == [
        "curated",
        "evaluation",
    ]


def test_dashboard_image_loader_supports_unicode_path(tmp_path: Path) -> None:
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    image_path = tmp_path / "결함" / "sample.png"
    image_path.parent.mkdir()
    success, encoded = cv2.imencode(".png", image)
    assert success
    encoded.tofile(image_path)

    loaded = load_dashboard_image(
        {
            "image_id": "sample",
            "file_path": str(image_path),
            "wafer_id": "w1",
            "die_id": "11",
            "pattern_position": "right-bottom",
            "capture_region": "standard",
            "defect_label": "unknown",
        }
    )

    np.testing.assert_array_equal(loaded, image)


def _analysis() -> DashboardAnalysis:
    import numpy as np

    image = np.zeros((10, 20), dtype=np.uint8)
    empty_profile = SignedGradientProfile(
        values=np.zeros(20, dtype=np.float32),
        positive_peak_indices=(),
        negative_peak_indices=(),
        positive_threshold=0.0,
        negative_threshold=0.0,
    )
    diagnostic = GradientCannyMeasurementDiagnostic(
        roi_bounds=PixelBounds((0, 0, 20, 10)),
        edge_diagnostics=EdgeDiagnosticResult(
            grayscale_roi=image,
            x_gradient=np.zeros((10, 20), dtype=np.float32),
            x_gradient_visualization=image,
            canny_edges=image,
        ),
        profile=empty_profile,
        pairing=DarkBandPairingDiagnostic(doublets=(), rejected_peaks=()),
        candidates=(),
        validation=PairingValidationResult(
            status="VALID",
            reasons=(),
            outer_greater_than_inner_count=0,
            catastrophic_mispairing=False,
        ),
    )
    record = ImageRecord(
        image_id="w1_11_c",
        file_path=Path("unused.png"),
        wafer_id="w1",
        die_id="11",
        pattern_position="center",
        capture_region="standard",
    )
    result = MeasurementResult(
        image_id="w1_11_c",
        method=MetrologyMethod.GRADIENT_CANNY,
        measurement_id="w1_11_c:gradient-canny",
        outer_width_px=240.0,
        inner_width_px=120.0,
        gap_px=110.0,
        outer_width_um=None,
        inner_width_um=None,
        gap_um=None,
        status=MeasurementStatus.VALID,
        runtime_ms=1.5,
    )
    return DashboardAnalysis(
        image=image,
        image_record=record,
        result=result,
        diagnostic=diagnostic,
        source="CURATED_IMAGES::PASS_normal",
    )


def test_evaluation_record_preserves_pending_physical_and_accuracy_values() -> None:
    result = build_evaluation_record(_analysis())

    assert result["outer_px"] == 240.0
    assert result["outer_um"] is None
    assert result["manual_outer_um"] is None
    assert result["outer_absolute_error_um"] is None
    assert result["calibration_id"] is None
    assert result["measurement_status"] == "VALID"


def test_method_comparison_has_only_real_method2_measurement() -> None:
    rows = method_comparison_records(build_evaluation_record(_analysis()))

    assert [row["Method"] for row in rows] == [
        "INTENSITY_THRESHOLD",
        "GRADIENT_CANNY",
        "UNET",
    ]
    assert rows[0]["Outer px"] is None
    assert rows[1]["Outer px"] == 240.0
    assert rows[2]["Outer px"] is None
