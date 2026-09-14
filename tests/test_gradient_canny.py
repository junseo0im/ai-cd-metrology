from pathlib import Path
from dataclasses import asdict

import numpy as np
import pytest

from ai_cd_metrology.metrology.gradient_canny import (
    GradientCannyMetrology,
    DEFAULT_LOCAL_BAND_COUNT,
    SignedGradientProfile,
    PixelPairingCandidate,
    flag_long_pairing_candidates,
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


def _guard_candidate(index: int, outer: int, inner: int, gap: int = 118) -> PixelPairingCandidate:
    x = index * 500
    return PixelPairingCandidate(index, x, x + 16, x + 16 + inner,
                                 x + outer, x + outer + gap, x + outer + gap + 16,
                                 outer, inner, gap)


def test_long_pairing_guard_flags_only_upward_long_candidate() -> None:
    candidates = tuple(_guard_candidate(i, outer, outer - 32)
                       for i, outer in enumerate((130, 132, 131, 270, 133, 131)))
    assert flag_long_pairing_candidates(candidates) == (candidates[3],)
    gap_candidates = tuple(_guard_candidate(i, 130, 98, 270 if i == 3 else 118) for i in range(6))
    assert flag_long_pairing_candidates(gap_candidates) == (gap_candidates[3],)


def test_long_pairing_guard_preserves_normal_like_candidates() -> None:
    candidates = tuple(_guard_candidate(i, outer, outer - 32)
                       for i, outer in enumerate((130, 132, 131, 133, 131)))
    assert flag_long_pairing_candidates(candidates) == ()


def test_long_pairing_guard_preserves_smaller_narrowing_candidate() -> None:
    candidates = tuple(_guard_candidate(i, outer, outer - 32)
                       for i, outer in enumerate((130, 132, 131, 100, 133, 131)))
    assert flag_long_pairing_candidates(candidates) == ()


def test_long_pairing_guard_handles_empty_candidates() -> None:
    assert flag_long_pairing_candidates(()) == ()


def test_long_pairing_guard_ceiling_is_strictly_exceeded() -> None:
    normal = tuple(_guard_candidate(i, 200, 168) for i in range(4))
    assert flag_long_pairing_candidates(normal + (_guard_candidate(4, 320, 168),), relative_ceiling=1.6) == ()
    upward = _guard_candidate(4, 321, 168)
    assert flag_long_pairing_candidates(normal + (upward,), relative_ceiling=1.6) == (upward,)


@pytest.mark.parametrize('ceiling', [1.0, 0.0, float('nan'), float('inf')])
def test_long_pairing_guard_rejects_invalid_ceiling(ceiling: float) -> None:
    with pytest.raises(ValueError, match='finite and greater than 1'):
        flag_long_pairing_candidates((), relative_ceiling=ceiling)


def test_missing_band_long_pair_warns_in_whole_and_local_chains() -> None:
    image = _vertical_dark_bands(tuple((10 + 30 * i, 20 + 30 * i) for i in range(15) if i != 7), width=480)
    algorithm = _full_image_algorithm()
    diagnostic = algorithm.analyze(image, _image_record())
    result = algorithm.measure(image, _image_record())[0]
    assert len(diagnostic.candidates) == 6  # Backward-compatible raw windows.
    assert len(diagnostic.long_pairing_candidates) == 1
    assert len(diagnostic.accepted_candidates) == 5
    assert not set(diagnostic.accepted_candidates).intersection(diagnostic.long_pairing_candidates)
    assert set(diagnostic.candidates) == set(diagnostic.accepted_candidates) | set(diagnostic.long_pairing_candidates)
    assert result.status is MeasurementStatus.WARNING
    assert result.failure_reason == 'long_pairing_candidates: 1'
    for field in ('outer_width', 'inner_width', 'gap'):
        expected = float(np.median([getattr(c, field + '_px') for c in diagnostic.accepted_candidates]))
        assert getattr(result, field + '_px') == expected
    for band in diagnostic.local_bands:
        assert band.candidates == diagnostic.candidates
        assert band.long_pairing_candidates == diagnostic.long_pairing_candidates
        assert band.accepted_candidates == diagnostic.accepted_candidates
        assert band.status is MeasurementStatus.WARNING
        assert band.failure_reason == 'long_pairing_candidates: 1'
    assert result.outer_width_um is None and result.calibration_id is None


def test_long_pairing_guard_reuses_existing_insufficient_candidate_fail() -> None:
    image = _vertical_dark_bands(tuple((10 + 30 * i, 20 + 30 * i) for i in range(8) if i != 3), width=280)
    algorithm = _full_image_algorithm()
    diagnostic = algorithm.analyze(image, _image_record())
    result = algorithm.measure(image, _image_record())[0]
    assert len(diagnostic.candidates) == 3
    assert len(diagnostic.accepted_candidates) == 2
    assert len(diagnostic.long_pairing_candidates) == 1
    assert result.status is MeasurementStatus.FAIL
    assert 'candidate_count_below_3: 2' in result.failure_reason
    assert (result.outer_width_px, result.inner_width_px, result.gap_px) == (None, None, None)
    assert result.edge_coordinates == []
    assert all(b.status is MeasurementStatus.FAIL and len(b.accepted_candidates) == 2 for b in diagnostic.local_bands)
