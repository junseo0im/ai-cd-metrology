"""Gradient/Canny diagnostics and pixel-only metrology."""

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

from ..calibration import convert_pixel_measurements
from ..roi import NormalizedROI, PixelBounds, crop_normalized_roi
from ..schemas import (
    CalibrationRecord,
    ImageRecord,
    MeasurementResult,
    MeasurementStatus,
    MetrologyMethod,
)
from .base import MetrologyAlgorithm


PILOT_GAUSSIAN_KERNEL_SIZE = 5
PILOT_GAUSSIAN_SIGMA = 0.0
PILOT_SOBEL_KERNEL_SIZE = 3
PILOT_CANNY_THRESHOLD_LOW = 50
PILOT_CANNY_THRESHOLD_HIGH = 150
CHARACTERIZATION_CANNY_THRESHOLD_LOW = 20
CHARACTERIZATION_CANNY_THRESHOLD_HIGH = 60
PROFILE_PEAK_RELATIVE_THRESHOLD = 0.35
DEFAULT_LOCAL_BAND_COUNT = 5


@dataclass(frozen=True, slots=True)
class EdgeDiagnosticResult:
    """Display-oriented outputs only; no edge pairing or measurements."""

    grayscale_roi: NDArray[np.uint8]
    x_gradient: NDArray[np.float32]
    x_gradient_visualization: NDArray[np.uint8]
    canny_edges: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class SignedGradientProfile:
    """Diagnostic 1D profile and unpaired signed peak locations."""

    values: NDArray[np.float32]
    positive_peak_indices: tuple[int, ...]
    negative_peak_indices: tuple[int, ...]
    positive_threshold: float
    negative_threshold: float


@dataclass(frozen=True, slots=True)
class DarkBandDoublet:
    """Accepted adjacent negative/positive peaks for one optical dark band."""

    sequence_index: int
    negative_peak_x: int
    positive_peak_x: int
    dark_band_width_px: int
    segment_index: int


@dataclass(frozen=True, slots=True)
class RejectedGradientPeak:
    """Unpaired peak retained for pilot diagnostics."""

    x: int
    polarity: str
    reason: str


@dataclass(frozen=True, slots=True)
class DarkBandPairingDiagnostic:
    """Dark-band doublets and peaks rejected without forced correction."""

    doublets: tuple[DarkBandDoublet, ...]
    rejected_peaks: tuple[RejectedGradientPeak, ...]


@dataclass(frozen=True, slots=True)
class PixelPairingCandidate:
    """Reference-validated E1-E6 pixel distances; not a MeasurementResult."""

    sequence_index: int
    e1_x: int
    e2_x: int
    e3_x: int
    e4_x: int
    e5_x: int
    e6_x: int
    outer_width_px: int
    inner_width_px: int
    gap_px: int


@dataclass(frozen=True, slots=True)
class PairingValidationResult:
    """Pilot-only reproducibility decision, separate from production QC."""

    status: str
    reasons: tuple[str, ...]
    outer_greater_than_inner_count: int
    catastrophic_mispairing: bool


@dataclass(frozen=True, slots=True)
class LocalBandMeasurement:
    '''Pixel-only local diagnostic; y bounds are original-image half-open bounds.

    Candidate x positions remain ROI-relative. Medians are Outer/Inner/Gap px;
    raw medians may remain available on FAIL and must be read with status.
    '''

    band_index: int
    y_bounds_px: tuple[int, int]
    profile: SignedGradientProfile
    pairing: DarkBandPairingDiagnostic
    candidates: tuple[PixelPairingCandidate, ...]
    validation: PairingValidationResult
    medians: tuple[float, float, float] | None
    status: MeasurementStatus
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class GradientCannyMeasurementDiagnostic:
    """Raw pixel-pairing detail retained outside the scalar result schema."""

    roi_bounds: PixelBounds
    edge_diagnostics: EdgeDiagnosticResult
    profile: SignedGradientProfile
    pairing: DarkBandPairingDiagnostic
    candidates: tuple[PixelPairingCandidate, ...]
    validation: PairingValidationResult
    local_bands: tuple[LocalBandMeasurement, ...] = ()


def _to_grayscale_uint8(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    if image.dtype != np.uint8:
        raise ValueError("Gradient/Canny pilot expects a uint8 ROI")
    if image.ndim == 2:
        return image.copy()
    if image.ndim != 3:
        raise ValueError("ROI must be a grayscale, BGR, or BGRA image")
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError("ROI must have 1, 3, or 4 channels")


def extract_edge_diagnostics(
    roi: NDArray[np.generic],
    *,
    canny_threshold_low: int = PILOT_CANNY_THRESHOLD_LOW,
    canny_threshold_high: int = PILOT_CANNY_THRESHOLD_HIGH,
) -> EdgeDiagnosticResult:
    """Run the single PHASE pilot baseline without deriving measurements."""

    if roi.size == 0:
        raise ValueError("ROI must not be empty")
    if not 0 <= canny_threshold_low < canny_threshold_high:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")

    grayscale = _to_grayscale_uint8(roi)
    blurred = cv2.GaussianBlur(
        grayscale,
        (PILOT_GAUSSIAN_KERNEL_SIZE, PILOT_GAUSSIAN_KERNEL_SIZE),
        PILOT_GAUSSIAN_SIGMA,
    )
    x_gradient = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        1,
        0,
        ksize=PILOT_SOBEL_KERNEL_SIZE,
    )
    max_absolute_gradient = float(np.max(np.abs(x_gradient)))
    if max_absolute_gradient == 0.0:
        gradient_visualization = np.full(grayscale.shape, 128, dtype=np.uint8)
    else:
        gradient_visualization = np.clip(
            127.5 + (127.5 * x_gradient / max_absolute_gradient),
            0,
            255,
        ).astype(np.uint8)

    canny_edges = cv2.Canny(
        blurred,
        canny_threshold_low,
        canny_threshold_high,
        apertureSize=PILOT_SOBEL_KERNEL_SIZE,
        L2gradient=True,
    )
    return EdgeDiagnosticResult(
        grayscale_roi=grayscale,
        x_gradient=x_gradient,
        x_gradient_visualization=gradient_visualization,
        canny_edges=canny_edges,
    )


def _strongest_index_per_run(
    profile: NDArray[np.float32],
    mask: NDArray[np.bool_],
    *,
    positive: bool,
) -> tuple[int, ...]:
    """Choose one extremum from each contiguous above-threshold run."""

    padded = np.pad(mask, (1, 1), constant_values=False)
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    peak_indices: list[int] = []
    for start, end in zip(starts, ends, strict=True):
        run = profile[start:end]
        offset = int(np.argmax(run) if positive else np.argmin(run))
        peak_indices.append(int(start + offset))
    return tuple(peak_indices)


def characterize_signed_x_gradient(
    x_gradient: NDArray[np.generic],
) -> SignedGradientProfile:
    """Median-aggregate Sobel-x over y and locate unpaired signed peaks."""

    if x_gradient.ndim != 2 or x_gradient.size == 0:
        raise ValueError("x_gradient must be a non-empty 2D array")

    profile = np.median(x_gradient, axis=0).astype(np.float32, copy=False)
    positive_threshold = PROFILE_PEAK_RELATIVE_THRESHOLD * max(
        float(np.max(profile)), 0.0
    )
    negative_threshold = PROFILE_PEAK_RELATIVE_THRESHOLD * min(
        float(np.min(profile)), 0.0
    )

    positive_mask = (
        profile >= positive_threshold
        if positive_threshold > 0.0
        else np.zeros(profile.shape, dtype=bool)
    )
    negative_mask = (
        profile <= negative_threshold
        if negative_threshold < 0.0
        else np.zeros(profile.shape, dtype=bool)
    )
    return SignedGradientProfile(
        values=profile,
        positive_peak_indices=_strongest_index_per_run(
            profile, positive_mask, positive=True
        ),
        negative_peak_indices=_strongest_index_per_run(
            profile, negative_mask, positive=False
        ),
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold,
    )


def pair_dark_band_doublets(
    profile: SignedGradientProfile,
) -> DarkBandPairingDiagnostic:
    """Accept only adjacent negative-to-positive peak events as doublets."""

    events = [(x, "positive") for x in profile.positive_peak_indices]
    events.extend((x, "negative") for x in profile.negative_peak_indices)
    events.sort()

    doublets: list[DarkBandDoublet] = []
    rejected: list[RejectedGradientPeak] = []
    segment_index = 0
    event_index = 0
    while event_index < len(events):
        x, polarity = events[event_index]
        if polarity == "negative" and event_index + 1 < len(events):
            next_x, next_polarity = events[event_index + 1]
            if next_polarity == "positive":
                doublets.append(
                    DarkBandDoublet(
                        sequence_index=len(doublets),
                        negative_peak_x=x,
                        positive_peak_x=next_x,
                        dark_band_width_px=next_x - x,
                        segment_index=segment_index,
                    )
                )
                event_index += 2
                continue

        if event_index == 0 and polarity == "positive":
            reason = "partial_left_positive"
        elif event_index == len(events) - 1 and polarity == "negative":
            reason = "partial_right_negative"
        else:
            reason = f"unexpected_{polarity}"
        rejected.append(RejectedGradientPeak(x=x, polarity=polarity, reason=reason))
        segment_index += 1
        event_index += 1

    return DarkBandPairingDiagnostic(
        doublets=tuple(doublets),
        rejected_peaks=tuple(rejected),
    )


def build_pixel_pairing_candidates(
    pairing: DarkBandPairingDiagnostic,
) -> tuple[PixelPairingCandidate, ...]:
    """Apply the reference E1-E6 rule to uninterrupted doublet triplets."""

    candidates: list[PixelPairingCandidate] = []
    doublets = pairing.doublets
    for start in range(0, len(doublets) - 2, 2):
        first, second, third = doublets[start : start + 3]
        if len({first.segment_index, second.segment_index, third.segment_index}) != 1:
            continue
        candidates.append(
            PixelPairingCandidate(
                sequence_index=len(candidates),
                e1_x=first.negative_peak_x,
                e2_x=first.positive_peak_x,
                e3_x=second.negative_peak_x,
                e4_x=second.positive_peak_x,
                e5_x=third.negative_peak_x,
                e6_x=third.positive_peak_x,
                outer_width_px=second.positive_peak_x - first.negative_peak_x,
                inner_width_px=second.negative_peak_x - first.positive_peak_x,
                gap_px=third.negative_peak_x - second.positive_peak_x,
            )
        )
    return tuple(candidates)


def validate_pairing_candidates(
    pairing: DarkBandPairingDiagnostic,
    candidates: tuple[PixelPairingCandidate, ...],
    *,
    minimum_candidate_count: int = 3,
) -> PairingValidationResult:
    """Apply only the requested pilot reproducibility criteria."""

    if minimum_candidate_count <= 0:
        raise ValueError("minimum_candidate_count must be positive")

    reasons: list[str] = []
    if len(candidates) < minimum_candidate_count:
        reasons.append(
            f"candidate_count_below_{minimum_candidate_count}: {len(candidates)}"
        )

    detected_peak_count = 2 * len(pairing.doublets) + len(pairing.rejected_peaks)
    rejected_peak_count = len(pairing.rejected_peaks)
    if detected_peak_count == 0:
        reasons.append("no_detected_peaks")
    elif rejected_peak_count * 2 >= detected_peak_count:
        reasons.append("rejected_peaks_not_a_minority")

    nonpositive_count = sum(
        candidate.outer_width_px <= 0
        or candidate.inner_width_px <= 0
        or candidate.gap_px <= 0
        for candidate in candidates
    )
    if nonpositive_count:
        reasons.append(f"nonpositive_distance_candidates: {nonpositive_count}")

    outer_greater_count = sum(
        candidate.outer_width_px > candidate.inner_width_px
        for candidate in candidates
    )
    if candidates and outer_greater_count * 2 <= len(candidates):
        reasons.append(
            "outer_not_greater_than_inner_for_majority: "
            f"{outer_greater_count}/{len(candidates)}"
        )

    catastrophic = bool(nonpositive_count) or bool(
        candidates and outer_greater_count * 2 <= len(candidates)
    )
    return PairingValidationResult(
        status="PASS" if not reasons else "FAIL",
        reasons=tuple(reasons),
        outer_greater_than_inner_count=outer_greater_count,
        catastrophic_mispairing=catastrophic,
    )


def _candidate_medians(
    candidates: tuple[PixelPairingCandidate, ...],
) -> tuple[float, float, float] | None:
    if not candidates:
        return None
    return (
        float(np.median([candidate.outer_width_px for candidate in candidates])),
        float(np.median([candidate.inner_width_px for candidate in candidates])),
        float(np.median([candidate.gap_px for candidate in candidates])),
    )


def _measurement_status(
    pairing: DarkBandPairingDiagnostic,
    candidates: tuple[PixelPairingCandidate, ...],
    validation: PairingValidationResult,
    medians: tuple[float, float, float] | None,
) -> tuple[MeasurementStatus, str | None]:
    """Map pilot pairing validity to the existing measurement status enum."""

    reasons = list(validation.reasons)
    if medians is None:
        if "representative_median_unavailable" not in reasons:
            reasons.append("representative_median_unavailable")
    else:
        outer_width_px, inner_width_px, gap_px = medians
        if min(outer_width_px, inner_width_px, gap_px) <= 0:
            reasons.append("representative_median_not_positive")
        if outer_width_px <= inner_width_px:
            reasons.append("representative_outer_not_greater_than_inner")

    if reasons or validation.catastrophic_mispairing:
        return MeasurementStatus.FAIL, "; ".join(dict.fromkeys(reasons))
    if pairing.rejected_peaks:
        return (
            MeasurementStatus.WARNING,
            f"rejected_gradient_peaks: {len(pairing.rejected_peaks)}",
        )
    return MeasurementStatus.VALID, None


def compute_local_band_bounds(
    roi_height_px: int,
    band_count: int,
) -> tuple[tuple[int, int], ...]:
    '''Partition [0, H) into deterministic contiguous half-open local bands.'''

    if roi_height_px <= 0:
        raise ValueError('roi_height_px must be positive')
    if band_count < 1:
        raise ValueError('band_count must be at least 1')
    if roi_height_px < band_count:
        raise ValueError('roi_height_px must be at least band_count')
    return tuple(
        (index * roi_height_px // band_count, (index + 1) * roi_height_px // band_count)
        for index in range(band_count)
    )


def measure_local_band(
    x_gradient_band: NDArray[np.generic],
    *,
    band_index: int,
    y_bounds_px: tuple[int, int],
) -> LocalBandMeasurement:
    '''Wrap the existing pairing chain without recomputing blur or Sobel.'''

    profile = characterize_signed_x_gradient(x_gradient_band)
    pairing = pair_dark_band_doublets(profile)
    candidates = build_pixel_pairing_candidates(pairing)
    validation = validate_pairing_candidates(pairing, candidates)
    medians = _candidate_medians(candidates)
    status, failure_reason = _measurement_status(pairing, candidates, validation, medians)
    return LocalBandMeasurement(
        band_index=band_index,
        y_bounds_px=y_bounds_px,
        profile=profile,
        pairing=pairing,
        candidates=candidates,
        validation=validation,
        medians=medians,
        status=status,
        failure_reason=failure_reason,
    )


def _representative_candidate(
    candidates: tuple[PixelPairingCandidate, ...],
    medians: tuple[float, float, float],
) -> PixelPairingCandidate:
    """Choose an actual candidate closest to the reported pixel median triplet.

    Reported distances are medians across valid candidates; this candidate's
    distances need not equal them, especially for an even candidate count.
    """

    outer_median, inner_median, gap_median = medians
    return min(
        candidates,
        key=lambda candidate: (
            abs(candidate.outer_width_px - outer_median)
            + abs(candidate.inner_width_px - inner_median)
            + abs(candidate.gap_px - gap_median),
            candidate.sequence_index,
        ),
    )


_POSITION_ALIASES = {
    "c": "center",
    "lt": "left-top",
    "rb": "right-bottom",
}


class GradientCannyMetrology(MetrologyAlgorithm):
    """Measure from signed Sobel-x; Canny is diagnostic only, not used in pairing."""

    def __init__(
        self,
        roi_by_pattern_position: Mapping[str, NormalizedROI] | None = None,
        band_count: int = DEFAULT_LOCAL_BAND_COUNT,
    ) -> None:
        self._roi_by_pattern_position = dict(roi_by_pattern_position or {})
        if band_count < 1:
            raise ValueError('band_count must be at least 1')
        self._band_count = band_count

    @property
    def method(self) -> MetrologyMethod:
        return MetrologyMethod.GRADIENT_CANNY

    def _roi_for(self, pattern_position: str) -> NormalizedROI:
        roi = self._roi_by_pattern_position.get(pattern_position)
        if roi is None:
            canonical_position = _POSITION_ALIASES.get(pattern_position)
            if canonical_position is not None:
                roi = self._roi_by_pattern_position.get(canonical_position)
        if roi is None:
            raise ValueError(
                f"no normalized ROI configured for pattern position: "
                f"{pattern_position!r}"
            )
        return roi

    def analyze(
        self,
        image: NDArray[np.generic],
        image_record: ImageRecord,
    ) -> GradientCannyMeasurementDiagnostic:
        """Run the pixel pipeline while preserving every raw candidate."""

        roi, roi_bounds = crop_normalized_roi(
            image,
            self._roi_for(image_record.pattern_position),
        )
        edge_diagnostics = extract_edge_diagnostics(roi)
        profile = characterize_signed_x_gradient(edge_diagnostics.x_gradient)
        pairing = pair_dark_band_doublets(profile)
        candidates = build_pixel_pairing_candidates(pairing)
        validation = validate_pairing_candidates(pairing, candidates)
        band_bounds = compute_local_band_bounds(
            edge_diagnostics.x_gradient.shape[0], self._band_count
        )
        roi_y0 = roi_bounds[1]
        local_bands = tuple(
            measure_local_band(
                edge_diagnostics.x_gradient[y0:y1],
                band_index=index,
                y_bounds_px=(roi_y0 + y0, roi_y0 + y1),
            )
            for index, (y0, y1) in enumerate(band_bounds)
        )
        return GradientCannyMeasurementDiagnostic(
            roi_bounds=roi_bounds,
            edge_diagnostics=edge_diagnostics,
            profile=profile,
            pairing=pairing,
            candidates=candidates,
            validation=validation,
            local_bands=local_bands,
        )

    def measure(
        self,
        image: NDArray[np.generic],
        image_record: ImageRecord,
        calibration: CalibrationRecord | None = None,
    ) -> list[MeasurementResult]:
        """Return one median-aggregated measurement, optionally calibrated on x.

        Calibration affects only the final representative distances. Raw
        per-window pixel candidates remain available through :meth:`analyze`.
        """

        started_at = perf_counter()
        diagnostic = self.analyze(image, image_record)
        medians = _candidate_medians(diagnostic.candidates)
        status, failure_reason = _measurement_status(
            diagnostic.pairing,
            diagnostic.candidates,
            diagnostic.validation,
            medians,
        )

        edge_coordinates: list[tuple[float, float]] = []
        outer_width_px: float | None = None
        inner_width_px: float | None = None
        gap_px: float | None = None
        if status is not MeasurementStatus.FAIL and medians is not None:
            outer_width_px, inner_width_px, gap_px = medians
            representative = _representative_candidate(
                diagnostic.candidates,
                medians,
            )
            x0, y0, _, y1 = diagnostic.roi_bounds
            # E1-E6 x positions come from the median-y 1D profile. The ROI-center
            # y below is a display placeholder, not a measured edge y position.
            representative_y = y0 + (y1 - y0 - 1) / 2.0
            edge_coordinates = [
                (float(x0 + edge_x), float(representative_y))
                for edge_x in (
                    representative.e1_x,
                    representative.e2_x,
                    representative.e3_x,
                    representative.e4_x,
                    representative.e5_x,
                    representative.e6_x,
                )
            ]

        outer_width_um, inner_width_um, gap_um = convert_pixel_measurements(
            outer_width_px, inner_width_px, gap_px, calibration, axis="x"
        )
        applied_calibration_id = (
            calibration.calibration_id
            if calibration is not None and outer_width_um is not None
            else None
        )
        runtime_ms = (perf_counter() - started_at) * 1000.0
        return [
            MeasurementResult(
                image_id=image_record.image_id,
                method=self.method,
                measurement_id=f"{image_record.image_id}:gradient-canny",
                outer_width_px=outer_width_px,
                inner_width_px=inner_width_px,
                gap_px=gap_px,
                outer_width_um=outer_width_um,
                inner_width_um=inner_width_um,
                gap_um=gap_um,
                calibration_id=applied_calibration_id,
                edge_coordinates=edge_coordinates,
                status=status,
                failure_reason=failure_reason,
                runtime_ms=runtime_ms,
            )
        ]
