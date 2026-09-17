"""Adapters between dashboard views and the frozen production interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
import yaml

from ai_cd_metrology.image_io import load_image
from ai_cd_metrology.metrology.gradient_canny import (
    GradientCannyMeasurementDiagnostic,
    GradientCannyMetrology,
)
from ai_cd_metrology.roi import NormalizedROI
from ai_cd_metrology.schemas import ImageRecord, MeasurementResult


METHOD2_VERSION = "gradient-canny-frozen-baseline"


@dataclass(frozen=True, slots=True)
class DashboardAnalysis:
    """Selected-image outputs retained by the Streamlit session."""

    image: NDArray[np.generic]
    image_record: ImageRecord
    result: MeasurementResult
    diagnostic: GradientCannyMeasurementDiagnostic
    source: str


def _text(row: Mapping[str, Any], key: str, default: str = "unknown") -> str:
    value = row.get(key, default)
    text = str(value).strip() if value is not None else ""
    return text or default


def image_record_from_row(row: Mapping[str, Any]) -> ImageRecord:
    """Build the production image identity from one catalog record."""

    defect_label = _text(row, "defect_label", "unknown")
    return ImageRecord(
        image_id=_text(row, "image_id"),
        file_path=Path(_text(row, "file_path")),
        wafer_id=_text(row, "wafer_id"),
        die_id=_text(row, "die_id"),
        pattern_position=_text(row, "pattern_position"),
        capture_region=_text(row, "capture_region", "standard"),
        defect_label=None if defect_label == "unknown" else defect_label,
    )


def build_method2(project_root: Path) -> GradientCannyMetrology:
    """Construct Method 2 from the frozen repository ROI configuration."""

    config_path = project_root / "configs/metrology.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    roi_config = config["metrology"]["roi"]
    common_x = roi_config["common_x_range"]
    roi_by_position = {
        position: NormalizedROI(
            x_min=float(common_x["x_min"]),
            x_max=float(common_x["x_max"]),
            y_min=float(bounds["y_min"]),
            y_max=float(bounds["y_max"]),
        )
        for position, bounds in roi_config["by_pattern_position"].items()
    }
    return GradientCannyMetrology(roi_by_pattern_position=roi_by_position)


def run_method2_analysis(
    row: Mapping[str, Any],
    algorithm: GradientCannyMetrology,
) -> DashboardAnalysis:
    """Run public frozen measurement and diagnostic APIs for one image only."""

    image_record = image_record_from_row(row)
    image = load_image(image_record)
    result = algorithm.measure(image, image_record, calibration=None)[0]
    diagnostic = algorithm.analyze(image, image_record)
    return DashboardAnalysis(
        image=image,
        image_record=image_record,
        result=result,
        diagnostic=diagnostic,
        source=_text(row, "source"),
    )


def build_evaluation_record(analysis: DashboardAnalysis) -> dict[str, object]:
    """Map one production result into the common evaluation/report contract."""

    result = analysis.result
    return {
        "image_id": result.image_id,
        "wafer_id": analysis.image_record.wafer_id,
        "die_id": analysis.image_record.die_id,
        "pattern_position": analysis.image_record.pattern_position,
        "source": analysis.source,
        "method_id": result.method.value,
        "method_version": METHOD2_VERSION,
        "outer_px": result.outer_width_px,
        "inner_px": result.inner_width_px,
        "gap_px": result.gap_px,
        "outer_um": result.outer_width_um,
        "inner_um": result.inner_width_um,
        "gap_um": result.gap_um,
        "manual_outer_um": None,
        "manual_inner_um": None,
        "manual_gap_um": None,
        "outer_error_um": None,
        "inner_error_um": None,
        "gap_error_um": None,
        "outer_absolute_error_um": None,
        "inner_absolute_error_um": None,
        "gap_absolute_error_um": None,
        "outer_relative_error_percent": None,
        "inner_relative_error_percent": None,
        "gap_relative_error_percent": None,
        "measurement_status": result.status.value,
        "failure_reason": result.failure_reason,
        "calibration_id": result.calibration_id,
        "runtime_ms": result.runtime_ms,
        "review_required": None,
        "critical_flag": None,
        "inspection_timestamp": None,
        "algorithm_version": None,
        "acquisition_id": None,
    }


def diagnostic_summary(
    diagnostic: GradientCannyMeasurementDiagnostic,
) -> dict[str, int | str]:
    """Keep reliability diagnostics separate from representative CD values."""

    return {
        "raw_candidate_count": len(diagnostic.candidates),
        "accepted_candidate_count": len(diagnostic.accepted_candidates),
        "rejected_peak_count": len(diagnostic.pairing.rejected_peaks),
        "long_pair_candidate_count": len(diagnostic.long_pairing_candidates),
        "pairing_validation": diagnostic.validation.status,
    }


def local_band_records(
    diagnostic: GradientCannyMeasurementDiagnostic,
) -> list[dict[str, object]]:
    """Return five-band diagnostic rows without changing image-level results."""

    records: list[dict[str, object]] = []
    for band in diagnostic.local_bands:
        outer, inner, gap = band.medians if band.medians is not None else (None, None, None)
        records.append(
            {
                "band_index": band.band_index,
                "y_start_px": band.y_bounds_px[0],
                "y_end_px": band.y_bounds_px[1],
                "status": band.status.value,
                "raw_candidate_count": len(band.candidates),
                "accepted_candidate_count": len(band.accepted_candidates),
                "rejected_peak_count": len(band.pairing.rejected_peaks),
                "long_pair_candidate_count": len(band.long_pairing_candidates),
                "outer_median_px": outer,
                "inner_median_px": inner,
                "gap_median_px": gap,
            }
        )
    return records


def method_comparison_records(
    method2_record: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    """Build a no-fake-value comparison table for the three planned methods."""

    rows = []
    for method_id in ("INTENSITY_THRESHOLD", "GRADIENT_CANNY", "UNET"):
        if method_id == "GRADIENT_CANNY" and method2_record is not None:
            rows.append(
                {
                    "Method": method_id,
                    "Outer px": method2_record["outer_px"],
                    "Inner px": method2_record["inner_px"],
                    "Gap px": method2_record["gap_px"],
                    "Status": method2_record["measurement_status"],
                    "Runtime ms": method2_record["runtime_ms"],
                    "Calibration": "Pending",
                    "Manual reference": "Pending",
                    "Absolute error": "Pending",
                }
            )
        else:
            rows.append(
                {
                    "Method": method_id,
                    "Outer px": None,
                    "Inner px": None,
                    "Gap px": None,
                    "Status": "NOT AVAILABLE" if method_id != "GRADIENT_CANNY" else "NOT RUN",
                    "Runtime ms": None,
                    "Calibration": "Pending" if method_id == "GRADIENT_CANNY" else "Not available",
                    "Manual reference": "Pending",
                    "Absolute error": "Pending",
                }
            )
    return rows
