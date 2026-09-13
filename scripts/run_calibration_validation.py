"""Validate calibration plumbing, never establish a production calibration."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import median

import numpy as np

from run_gradient_canny_pilot import PROJECT_ROOT, _load_roi_config
from run_pairing_reference import ANALYSIS_Y_RANGE, _candidate_edges, _read_image

from ai_cd_metrology.calibration import convert_pixel_measurements
from ai_cd_metrology.image_io import load_image
from ai_cd_metrology.metrology.gradient_canny import GradientCannyMetrology
from ai_cd_metrology.roi import NormalizedROI
from ai_cd_metrology.schemas import CalibrationRecord, ImageRecord, MeasurementResult


# Approximate manual annotation spans supplied for this validation only.
MANUAL_REFERENCE = {
    "outer_width": (11.67, 146),
    "inner_width": (10.39, 130),
    "gap": (9.11, 114),
}
SMOKE_IMAGES = (
    ("w1_11_c.png", "11", "center"),
    ("w1_13_lt.png", "13", "left-top"),
    ("w1_11_rb.png", "11", "right-bottom"),
)


def _check_plumbing(
    pixel: MeasurementResult,
    physical: MeasurementResult,
    calibration: CalibrationRecord,
) -> dict[str, bool]:
    fields = ("outer_width", "inner_width", "gap")
    pixels_unchanged = all(
        getattr(pixel, f"{field}_px") == getattr(physical, f"{field}_px")
        for field in fields
    )
    conversion_matches = all(
        (
            getattr(physical, f"{field}_um") is None
            if getattr(pixel, f"{field}_px") is None
            else np.isclose(
                getattr(physical, f"{field}_um"),
                getattr(pixel, f"{field}_px") * calibration.scale_x_um_per_px,
            )
        )
        for field in fields
    )
    expected_id = (
        calibration.calibration_id if physical.outer_width_um is not None else None
    )
    checks = {
        "pixels_unchanged": pixels_unchanged,
        "status_unchanged": pixel.status is physical.status,
        "reason_unchanged": pixel.failure_reason == physical.failure_reason,
        "edges_unchanged": pixel.edge_coordinates == physical.edge_coordinates,
        "conversion_matches_x_scale": bool(conversion_matches),
        "calibration_id_matches": physical.calibration_id == expected_id,
        "uncalibrated_fields_none": pixel.calibration_id is None
        and all(getattr(pixel, f"{field}_um") is None for field in fields),
    }
    if not all(checks.values()):
        raise AssertionError(f"calibration plumbing mismatch: {checks}")
    return checks


def main() -> None:
    consistency = [
        {
            "measurement": field,
            "manual_um": manual_um,
            "approximate_manual_span_px": span_px,
            "implied_scale_um_per_px": manual_um / span_px,
        }
        for field, (manual_um, span_px) in MANUAL_REFERENCE.items()
    ]
    scales = [row["implied_scale_um_per_px"] for row in consistency]
    scale = median(scales)
    calibration = CalibrationRecord(
        calibration_id="optiview_reference_validation",
        objective="unknown-validation-only",
        scale_x_um_per_px=scale,
        # Existing schema requires y scale; no y conversion is validated here.
        scale_y_um_per_px=scale,
    )

    reference_path = PROJECT_ROOT / "data/raw/optiview_ref_image.png"
    reference_image = _read_image(reference_path)
    y0, y1 = ANALYSIS_Y_RANGE
    height = reference_image.shape[0]
    reference_roi = NormalizedROI(0.0, y0 / height, 1.0, y1 / height)
    record = ImageRecord(
        image_id=reference_path.stem,
        file_path=reference_path,
        wafer_id="unknown",
        die_id="unknown",
        pattern_position="reference",
        capture_region="reference-annotation-free-band",
    )
    algorithm = GradientCannyMetrology({"reference": reference_roi})
    diagnostic = algorithm.analyze(reference_image, record)
    if diagnostic.roi_bounds != (0, y0, reference_image.shape[1], y1):
        raise ValueError("reference ROI must match the existing analysis band")
    pixel = algorithm.measure(reference_image, record)
    physical = algorithm.measure(reference_image, record, calibration)
    reference_checks = _check_plumbing(pixel[0], physical[0], calibration)
    reference_pixel, reference_physical = pixel[0], physical[0]

    # Reuse the previously inspected manual reference interval, not a new ROI
    # or nearest-peak heuristic. Require an exact match to an existing candidate.
    edges = _candidate_edges(diagnostic.profile)
    expected_x = tuple(x for x, _ in edges)
    if tuple(polarity for _, polarity in edges) != (
        "negative", "positive", "negative", "positive", "negative", "positive"
    ):
        raise ValueError("reference E1-E6 polarity sequence changed")
    matches = [
        candidate for candidate in diagnostic.candidates
        if (
            candidate.e1_x, candidate.e2_x, candidate.e3_x,
            candidate.e4_x, candidate.e5_x, candidate.e6_x,
        ) == expected_x
    ]
    if len(matches) != 1:
        raise ValueError("existing reference E1-E6 candidate is unavailable")
    candidate = matches[0]
    px_values = (candidate.outer_width_px, candidate.inner_width_px, candidate.gap_px)
    um_values = convert_pixel_measurements(*px_values, calibration, axis="x")
    comparison = []
    for field, value_px, value_um in zip(
        MANUAL_REFERENCE, px_values, um_values, strict=True
    ):
        manual_um, _ = MANUAL_REFERENCE[field]
        absolute_error = abs(value_um - manual_um)
        comparison.append({
            "measurement": field,
            "automatic_candidate_px": value_px,
            "automatic_candidate_um": value_um,
            "manual_um": manual_um,
            "absolute_error_um": absolute_error,
            "relative_error_percent": absolute_error / manual_um * 100.0,
        })

    smoke = []
    config = PROJECT_ROOT / "configs/metrology.yaml"
    for filename, die_id, position in SMOKE_IMAGES:
        path = PROJECT_ROOT / "data/raw/images/images/PASS_normal" / filename
        record = ImageRecord(path.stem, path, "w1", die_id, position, "standard", "normal")
        image = load_image(record)
        algorithm = GradientCannyMetrology({position: _load_roi_config(config, position)})
        pixel = algorithm.measure(image, record)[0]
        physical = algorithm.measure(image, record, calibration)[0]
        if any(getattr(physical, f"{field}_um") is None for field in MANUAL_REFERENCE):
            raise AssertionError(f"smoke image did not produce physical values: {filename}")
        smoke.append({
            "filename": filename,
            "pixel_result": asdict(pixel),
            "calibrated_result": asdict(physical),
            "checks": _check_plumbing(pixel, physical, calibration),
        })

    report = {
        "purpose": "validation-only; not official ruler or production calibration",
        "manual_source": "user-supplied OptiView annotations and approximate pixel endpoints",
        "scale_consistency": consistency,
        "median_validation_scale_um_per_px": scale,
        "scale_range_um_per_px": max(scales) - min(scales),
        "calibration": asdict(calibration),
        "y_scale_note": "required existing field only; x-axis conversion is the sole validation",
        "reference_measure": {
            "roi_bounds": diagnostic.roi_bounds,
            "raw_candidates": [asdict(item) for item in diagnostic.candidates],
            "pixel_result": asdict(reference_pixel),
            "calibrated_result": asdict(reference_physical),
            "checks": reference_checks,
        },
        "reference_candidate_comparison": {
            "note": "individual manual-region candidate, NOT a successful measure() median result; no error-based PASS/FAIL",
            "edges": edges,
            "candidate": asdict(candidate),
            "comparison": comparison,
        },
        "pass_normal_smoke": smoke,
        "limitations": [
            "reference measure() needs >=3 candidates; existing reference band contains only 2",
            "manual annotations and integer-pixel spans are approximate",
            "Outer manual starting endpoint differs from Sobel extremum by several pixels; no offset applied",
            "reference-derived scale is not an independent accuracy or official calibration validation",
            "PASS_normal physical values validate plumbing, not physical ground truth",
        ],
    }
    output_dir = PROJECT_ROOT / "data/derived/calibration_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibration_validation.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
