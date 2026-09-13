from pathlib import Path
from dataclasses import asdict

import numpy as np
import pytest

from ai_cd_metrology.metrology.gradient_canny import (
    GradientCannyMetrology,
    DEFAULT_LOCAL_BAND_COUNT,
    SignedGradientProfile,
    build_pixel_pairing_candidates,
    characterize_signed_x_gradient,
    compute_local_band_bounds,
    extract_edge_diagnostics,
    pair_dark_band_doublets,
    validate_pairing_candidates,
)
from ai_cd_metrology.roi import NormalizedROI
from ai_cd_metrology.schemas import (
    CalibrationRecord,
    ImageRecord,
    MeasurementStatus,
    MetrologyMethod,
)


def test_method_is_gradient_canny() -> None:
    algorithm = GradientCannyMetrology()

    assert algorithm.method == MetrologyMethod.GRADIENT_CANNY


def _image_record(*, pattern_position: str = "center") -> ImageRecord:
    return ImageRecord(
        image_id="synthetic-1",
        file_path=Path("unused.png"),
        wafer_id="w1",
        die_id="1",
        pattern_position=pattern_position,
        capture_region="standard",
    )


def _full_image_algorithm(
    band_count: int = DEFAULT_LOCAL_BAND_COUNT,
) -> GradientCannyMetrology:
    return GradientCannyMetrology(
        {
            "center": NormalizedROI(
                x_min=0.0,
                y_min=0.0,
                x_max=1.0,
                y_max=1.0,
            )
        },
        band_count=band_count,
    )


def _vertical_dark_bands(
    bands: tuple[tuple[int, int], ...],
    *,
    width: int = 240,
) -> np.ndarray:
    image = np.full((64, width), 255, dtype=np.uint8)
    for start, end in bands:
        image[:, start:end] = 0
    return image


def test_measure_runs_on_synthetic_vertical_bands_and_uses_medians() -> None:
    image = _vertical_dark_bands(
        ((10, 20), (40, 50), (70, 80), (100, 110), (130, 140), (160, 170), (190, 200))
    )
    algorithm = _full_image_algorithm()

    diagnostic = algorithm.analyze(image, _image_record())
    result = algorithm.measure(image, _image_record())[0]

    assert len(diagnostic.candidates) == 3
    assert result.status is MeasurementStatus.VALID
    assert result.outer_width_px == pytest.approx(
        np.median([candidate.outer_width_px for candidate in diagnostic.candidates])
    )
    assert result.inner_width_px == pytest.approx(
        np.median([candidate.inner_width_px for candidate in diagnostic.candidates])
    )
    assert result.gap_px == pytest.approx(
        np.median([candidate.gap_px for candidate in diagnostic.candidates])
    )
    assert len(result.edge_coordinates) == 6
    assert result.runtime_ms is not None and result.runtime_ms >= 0.0


def test_measure_warns_when_rejected_peak_remains_with_enough_candidates() -> None:
    image = _vertical_dark_bands(
        ((0, 5), (20, 30), (50, 60), (80, 90), (110, 120), (140, 150), (170, 180), (200, 210))
    )

    diagnostic = _full_image_algorithm().analyze(image, _image_record())
    result = _full_image_algorithm().measure(image, _image_record())[0]

    assert len(diagnostic.candidates) >= 3
    assert diagnostic.pairing.rejected_peaks
    assert result.status is MeasurementStatus.WARNING
    assert result.failure_reason is not None


def test_measure_fails_when_candidate_count_is_insufficient() -> None:
    image = _vertical_dark_bands(((20, 30), (50, 60), (80, 90)))

    result = _full_image_algorithm().measure(image, _image_record())[0]

    assert result.status is MeasurementStatus.FAIL
    assert result.outer_width_px is None
    assert result.inner_width_px is None
    assert result.gap_px is None
    assert result.edge_coordinates == []
    assert result.failure_reason is not None


def test_measure_without_calibration_keeps_physical_values_none() -> None:
    image = _vertical_dark_bands(
        ((10, 20), (40, 50), (70, 80), (100, 110), (130, 140), (160, 170), (190, 200))
    )

    result = _full_image_algorithm().measure(
        image,
        _image_record(),
    )[0]

    assert result.outer_width_um is None
    assert result.inner_width_um is None
    assert result.gap_um is None
    assert result.calibration_id is None


@pytest.mark.parametrize("band_count", [3, 7])
@pytest.mark.parametrize("partial_left", [False, True])
def test_calibration_preserves_pixels_and_measurement_status(
    band_count: int, partial_left: bool,
) -> None:
    bands = tuple((20 + 30 * i, 30 + 30 * i) for i in range(band_count))
    if partial_left:
        bands = ((0, 5),) + bands
    image = _vertical_dark_bands(bands)
    calibration = CalibrationRecord(
        calibration_id="synthetic-calibration",
        objective="synthetic",
        scale_x_um_per_px=0.1,
        scale_y_um_per_px=0.9,
    )
    algorithm = _full_image_algorithm()
    pixel = algorithm.measure(image, _image_record())[0]
    physical = algorithm.measure(image, _image_record(), calibration)[0]

    assert physical.status is pixel.status
    assert physical.failure_reason == pixel.failure_reason
    assert physical.edge_coordinates == pixel.edge_coordinates
    for field in ("outer_width", "inner_width", "gap"):
        value_px = getattr(pixel, f"{field}_px")
        assert getattr(physical, f"{field}_px") == value_px
        value_um = getattr(physical, f"{field}_um")
        if value_px is None:
            assert value_um is None
        else:
            assert value_um == pytest.approx(value_px * 0.1)
    if band_count < 7:
        assert physical.status is MeasurementStatus.FAIL
        assert physical.calibration_id is None
    else:
        expected_status = (
            MeasurementStatus.WARNING if partial_left else MeasurementStatus.VALID
        )
        assert physical.status is expected_status
        assert physical.calibration_id == calibration.calibration_id


def test_signed_gradient_profile_uses_median_and_keeps_polarity() -> None:
    row = np.array([0, 1, 4, 1, 0, -2, -5, -2, 0], dtype=np.float32)
    x_gradient = np.vstack((row, row + 1, row - 1))

    profile = characterize_signed_x_gradient(x_gradient)

    np.testing.assert_array_equal(profile.values, row)
    assert profile.positive_peak_indices == (2,)
    assert profile.negative_peak_indices == (6,)
    assert profile.positive_threshold == pytest.approx(1.4)
    assert profile.negative_threshold == pytest.approx(-1.75)


def test_edge_diagnostics_reject_invalid_canny_thresholds() -> None:
    image = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="0 <= low < high"):
        extract_edge_diagnostics(
            image,
            canny_threshold_low=60,
            canny_threshold_high=20,
        )


def _profile_with_peaks(
    *,
    negative: tuple[int, ...],
    positive: tuple[int, ...],
) -> SignedGradientProfile:
    return SignedGradientProfile(
        values=np.zeros(160, dtype=np.float32),
        positive_peak_indices=positive,
        negative_peak_indices=negative,
        positive_threshold=1.0,
        negative_threshold=-1.0,
    )


def test_dark_band_doublets_and_reference_pixel_candidates() -> None:
    profile = _profile_with_peaks(
        negative=(10, 40, 70, 100, 130),
        positive=(20, 50, 80, 110, 140),
    )

    pairing = pair_dark_band_doublets(profile)
    candidates = build_pixel_pairing_candidates(pairing)

    assert len(pairing.doublets) == 5
    assert pairing.doublets[0].dark_band_width_px == 10
    assert pairing.rejected_peaks == ()
    assert [candidate.outer_width_px for candidate in candidates] == [40, 40]
    assert [candidate.inner_width_px for candidate in candidates] == [20, 20]
    assert [candidate.gap_px for candidate in candidates] == [20, 20]


def test_abnormal_peaks_are_rejected_without_bridging_candidates() -> None:
    profile = _profile_with_peaks(
        negative=(10, 40, 70, 75),
        positive=(20, 30, 50, 80),
    )

    pairing = pair_dark_band_doublets(profile)

    assert [(peak.x, peak.polarity) for peak in pairing.rejected_peaks] == [
        (30, "positive"),
        (70, "negative"),
    ]
    assert build_pixel_pairing_candidates(pairing) == ()


def test_partial_boundary_peaks_are_rejected() -> None:
    profile = _profile_with_peaks(
        negative=(10, 30),
        positive=(5, 20),
    )

    pairing = pair_dark_band_doublets(profile)

    assert [(peak.x, peak.reason) for peak in pairing.rejected_peaks] == [
        (5, "partial_left_positive"),
        (30, "partial_right_negative"),
    ]


def test_pairing_validation_passes_with_three_valid_candidates() -> None:
    profile = _profile_with_peaks(
        negative=(10, 40, 70, 100, 130, 160, 190),
        positive=(20, 50, 80, 110, 140, 170, 200),
    )
    pairing = pair_dark_band_doublets(profile)
    candidates = build_pixel_pairing_candidates(pairing)

    validation = validate_pairing_candidates(pairing, candidates)

    assert validation.status == "PASS"
    assert validation.reasons == ()
    assert validation.outer_greater_than_inner_count == 3
    assert not validation.catastrophic_mispairing


def test_pairing_validation_fails_with_empty_candidates() -> None:
    profile = _profile_with_peaks(negative=(), positive=())
    pairing = pair_dark_band_doublets(profile)

    validation = validate_pairing_candidates(pairing, ())

    assert validation.status == "FAIL"
    assert validation.reasons == (
        "candidate_count_below_3: 0",
        "no_detected_peaks",
    )


@pytest.mark.parametrize('height,count', [(10, 5), (13, 5), (10, 1)])
def test_local_band_bounds_cover_height_without_gaps(height: int, count: int) -> None:
    bounds = compute_local_band_bounds(height, count)

    assert len(bounds) == count
    assert bounds[0][0] == 0
    assert bounds[-1][1] == height
    assert all(start < end for start, end in bounds)
    assert all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
    assert sum(end - start for start, end in bounds) == height
    assert bounds == compute_local_band_bounds(height, count)


@pytest.mark.parametrize('height,count', [(0, 5), (-1, 5), (10, 0), (10, -1), (3, 5)])
def test_invalid_local_band_bounds_rejected(height: int, count: int) -> None:
    with pytest.raises(ValueError):
        compute_local_band_bounds(height, count)


def _local_test_image() -> np.ndarray:
    return _vertical_dark_bands(
        ((20, 30), (50, 60), (80, 90), (110, 120), (140, 150), (170, 180), (200, 210))
    )


def test_single_band_matches_whole_roi_and_extracts_once(monkeypatch) -> None:
    import ai_cd_metrology.metrology.gradient_canny as gradient_module

    calls = []
    original_extract = gradient_module.extract_edge_diagnostics

    def counted_extract(roi):
        calls.append(roi.shape)
        return original_extract(roi)

    monkeypatch.setattr(gradient_module, 'extract_edge_diagnostics', counted_extract)
    image = _local_test_image()
    # Nonzero original-image y offset verifies local metadata coordinates.
    roi = NormalizedROI(0.0, 0.25, 1.0, 0.75)
    algorithm = GradientCannyMetrology({'center': roi}, band_count=1)
    diagnostic = algorithm.analyze(image, _image_record())
    band = diagnostic.local_bands[0]

    assert len(calls) == 1
    assert len(diagnostic.local_bands) == 1
    assert band.band_index == 0
    assert band.y_bounds_px == (16, 48)
    assert band.candidates == diagnostic.candidates
    assert band.medians == tuple(
        float(np.median([getattr(candidate, field) for candidate in diagnostic.candidates]))
        for field in ('outer_width_px', 'inner_width_px', 'gap_px')
    )
    np.testing.assert_array_equal(band.profile.values, diagnostic.profile.values)
    assert band.validation == diagnostic.validation
    whole_result = algorithm.measure(image, _image_record())[0]
    assert band.status is whole_result.status


def test_local_status_is_independent_and_gradient_is_sliced() -> None:
    image = _local_test_image()
    image[:13] = 255
    image[51:, :5] = 0
    algorithm = _full_image_algorithm()
    diagnostic = algorithm.analyze(image, _image_record())
    statuses = {band.status for band in diagnostic.local_bands}

    assert len(diagnostic.local_bands) == DEFAULT_LOCAL_BAND_COUNT
    assert MeasurementStatus.VALID in statuses
    assert MeasurementStatus.FAIL in statuses
    assert MeasurementStatus.WARNING in statuses
    for band in diagnostic.local_bands:
        y0, y1 = band.y_bounds_px
        np.testing.assert_array_equal(
            band.profile.values,
            np.median(diagnostic.edge_diagnostics.x_gradient[y0:y1], axis=0),
        )
    # A local FAIL is not rolled up to image-level FAIL.
    assert algorithm.measure(image, _image_record())[0].status is MeasurementStatus.VALID


def test_band_count_and_calibration_do_not_change_image_level_path() -> None:
    image = _local_test_image()
    image[:13] = 255
    calibration = CalibrationRecord('local-test', 'synthetic', 0.1, 0.2)
    baseline = None
    for band_count in (1, 3, 5):
        algorithm = _full_image_algorithm(band_count)
        before = algorithm.analyze(image, _image_record())
        pixel = algorithm.measure(image, _image_record())[0]
        physical = algorithm.measure(image, _image_record(), calibration)[0]
        after = algorithm.analyze(image, _image_record())
        values = asdict(pixel)
        values.pop('runtime_ms')
        if baseline is None:
            baseline = values
        assert values == baseline
        assert physical.status is pixel.status
        assert physical.failure_reason == pixel.failure_reason
        assert physical.edge_coordinates == pixel.edge_coordinates
        for field in ('outer_width', 'inner_width', 'gap'):
            assert getattr(physical, f'{field}_px') == getattr(pixel, f'{field}_px')
            assert getattr(physical, f'{field}_um') == pytest.approx(
                getattr(pixel, f'{field}_px') * 0.1
            )
        for left, right in zip(before.local_bands, after.local_bands, strict=True):
            assert left.candidates == right.candidates
            assert left.medians == right.medians
            assert left.status is right.status
            np.testing.assert_array_equal(left.profile.values, right.profile.values)
            assert not hasattr(left, 'calibration_id')
