"""Generate frozen Method 2 outputs under the shared team protocol."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import sys

import cv2
import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEAM_PROTOCOL_DIRECTORY = PROJECT_ROOT / "team protocol"
for import_path in (PROJECT_ROOT / "src", TEAM_PROTOCOL_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from ai_cd_metrology.metrology.gradient_canny import (  # noqa: E402
    characterize_signed_x_gradient,
    extract_edge_diagnostics,
    pair_dark_band_doublets,
)
from team_protocol import (  # noqa: E402
    MEASUREMENT_ORDER,
    TARGETS_PER_ROW,
    UM_PER_NATIVE_PX,
    measurement_rows,
    parse_measurement_types,
    save_prediction_csv,
    select_central_targets,
)


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
METHOD_ID = "method2_gradient_team_adapter"
PAIR_CONSISTENCY_MINIMUM = 0.95
MISSING_SLOT_MESSAGE = (
    "missing physical target slot; "
    "continuity-preserving selection: no candidate"
)
AMBIGUOUS_SLOT_MESSAGE = (
    "ambiguous physical target slot; "
    "continuity-preserving selection: multiple candidates"
)
TARGET_SCALE_NOTE = "scale not yet confirmed against acquisition-native metadata"
OVERLAY_COLORS = {
    "line": (255, 255, 0),
    "gap": (255, 0, 255),
    "sidewall": (0, 255, 0),
}
DISPLAY_PREFIX = {"line": "L", "gap": "G", "sidewall": "S"}


@dataclass(frozen=True, slots=True)
class NativeRecovery:
    image: NDArray[np.generic]
    input_shape: tuple[int, ...]
    native_shape: tuple[int, ...]
    duplicate_factor: int
    row_pair_consistency: float
    column_pair_consistency: float
    offset_x: int = 0
    offset_y: int = 0


@dataclass(frozen=True, slots=True)
class ImageComparison:
    image_path: Path
    recovery: NativeRecovery
    rows: list[dict[str, object]]
    measurements_by_y: dict[float, dict[str, list[dict[str, object]]]]
    runtime_ms: float
    overlay_path: Path
    all_measurements_overlay_path: Path
    summary_path: Path


def load_image_unicode(path: Path) -> NDArray[np.generic]:
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def recover_native_pixels(
    image: NDArray[np.generic],
    duplicate_factor: int | None = None,
) -> NativeRecovery:
    """Remove nearest-neighbor repetition; detect 2x duplication by default."""

    if duplicate_factor is None:
        try:
            return recover_native_pixels(image, 2)
        except ValueError:
            return recover_native_pixels(image, 1)

    if duplicate_factor not in (1, 2):
        raise ValueError("duplicate_factor must be 1 or 2")
    input_shape = tuple(image.shape)
    if duplicate_factor == 1:
        return NativeRecovery(image, input_shape, input_shape, 1, 1.0, 1.0)

    def pair_consistency(axis: int, offset: int) -> float:
        if axis == 0:
            paired = ((image.shape[0] - offset) // 2) * 2
            source = image[offset : offset + paired]
            pairs = source.reshape(paired // 2, 2, *image.shape[1:])
            equal = pairs[:, 0] == pairs[:, 1]
        else:
            paired = ((image.shape[1] - offset) // 2) * 2
            source = image[:, offset : offset + paired]
            pairs = source.reshape(image.shape[0], paired // 2, 2, *image.shape[2:])
            equal = pairs[:, :, 0] == pairs[:, :, 1]
        if equal.ndim == 3:
            equal = np.all(equal, axis=2)
        return float(np.mean(equal))

    offset_y, row_consistency = max(
        ((offset, pair_consistency(0, offset)) for offset in (0, 1)),
        key=lambda item: item[1],
    )
    offset_x, column_consistency = max(
        ((offset, pair_consistency(1, offset)) for offset in (0, 1)),
        key=lambda item: item[1],
    )
    if min(row_consistency, column_consistency) < PAIR_CONSISTENCY_MINIMUM:
        raise ValueError(
            "Declared 2x duplicate pixels are not supported by the image: "
            f"row={row_consistency:.4f}, column={column_consistency:.4f}"
        )
    native = image[offset_y::2, offset_x::2].copy()
    return NativeRecovery(
        native,
        input_shape,
        tuple(native.shape),
        2,
        row_consistency,
        column_consistency,
        offset_x,
        offset_y,
    )


def _measurement(
    measurement_type: str,
    left_x: int,
    right_x: int,
    *,
    brightness: float | None = None,
) -> dict[str, object]:
    return {
        "measurement_type": measurement_type,
        "left_x_px": float(left_x),
        "right_x_px": float(right_x),
        "center_x_px": (left_x + right_x) / 2.0,
        "width_px": float(right_x - left_x),
        "brightness": brightness,
        "status": "ok",
    }


def measurements_at_native_y(
    grayscale_row: NDArray[np.uint8],
    gradient_row: NDArray[np.float32],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    """Build team line/gap/sidewall definitions at one exact native Y row."""

    profile = characterize_signed_x_gradient(gradient_row[np.newaxis, :])
    pairing = pair_dark_band_doublets(profile)
    sidewalls = []
    for physical_index, doublet in enumerate(pairing.doublets):
        sidewalls.append(
            {
                **_measurement(
                    "sidewall",
                    doublet.negative_peak_x,
                    doublet.positive_peak_x,
                ),
                "physical_index": physical_index,
            }
        )

    intervals: list[dict[str, object]] = []
    for interval_index, (left_band, right_band) in enumerate(
        zip(pairing.doublets, pairing.doublets[1:], strict=False)
    ):
        if left_band.segment_index != right_band.segment_index:
            continue
        left_x = left_band.positive_peak_x
        right_x = right_band.negative_peak_x
        if right_x <= left_x:
            continue
        intervals.append(
            {
                **_measurement(
                    "interval",
                    left_x,
                    right_x,
                    brightness=float(np.mean(grayscale_row[left_x:right_x])),
                ),
                "parity": interval_index % 2,
                "physical_index": interval_index,
            }
        )

    parity_brightness = {
        parity: float(
            np.median(
                [
                    float(interval["brightness"])
                    for interval in intervals
                    if interval["parity"] == parity
                ]
            )
        )
        for parity in (0, 1)
        if any(interval["parity"] == parity for interval in intervals)
    }
    if len(parity_brightness) != 2:
        raise ValueError("Both alternating interval parities are required")
    line_parity = max(parity_brightness, key=parity_brightness.get)
    gap_parity = 1 - line_parity
    diagnostics = {
        "negative_peak_count": len(profile.negative_peak_indices),
        "positive_peak_count": len(profile.positive_peak_indices),
        "doublet_count": len(pairing.doublets),
        "rejected_peak_count": len(pairing.rejected_peaks),
        "line_parity": line_parity,
        "gap_parity": gap_parity,
        "parity_brightness": parity_brightness,
    }
    return {"interval": intervals, "sidewall": sidewalls}, diagnostics


def _typed_candidates(
    candidates: dict[str, list[dict[str, object]]],
    line_parity: int,
) -> dict[str, list[dict[str, object]]]:
    intervals = candidates["interval"]
    return {
        "line": [
            {**interval, "measurement_type": "line"}
            for interval in intervals
            if interval["parity"] == line_parity
        ],
        "gap": [
            {**interval, "measurement_type": "gap"}
            for interval in intervals
            if interval["parity"] != line_parity
        ],
        "sidewall": candidates["sidewall"],
    }


def visualization_measurements(
    candidates: list[dict[str, object]],
    measurement_type: str,
) -> list[dict[str, object]]:
    """Assign left-to-right display IDs that are independent of CSV target IDs."""

    ordered = sorted(candidates, key=lambda item: float(item["center_x_px"]))
    return [
        {
            **candidate,
            "display_id": f"{DISPLAY_PREFIX[measurement_type]}{index}",
        }
        for index, candidate in enumerate(ordered, start=1)
    ]


def visualization_physical_intervals(
    candidates: dict[str, list[dict[str, object]]],
    line_parity: int,
) -> dict[str, list[dict[str, object]]]:
    """Represent every inter-sidewall slot, including invalid missing slots."""

    sidewalls = sorted(
        candidates["sidewall"],
        key=lambda item: float(item["center_x_px"]),
    )
    valid_by_slot = {
        int(interval["physical_index"]): interval
        for interval in candidates["interval"]
    }
    displayed: dict[str, list[dict[str, object]]] = {
        "line": [],
        "gap": [],
        "missing": [],
    }
    display_counts = {"line": 0, "gap": 0, "missing": 0}
    for slot_index, (left_sidewall, right_sidewall) in enumerate(
        zip(sidewalls, sidewalls[1:], strict=False)
    ):
        interval = valid_by_slot.get(slot_index)
        if interval is None:
            display_counts["missing"] += 1
            left_x = float(left_sidewall["right_x_px"])
            right_x = float(right_sidewall["left_x_px"])
            displayed["missing"].append(
                {
                    "measurement_type": "missing",
                    "display_id": f"X{display_counts['missing']}",
                    "physical_index": slot_index,
                    "left_x_px": left_x,
                    "right_x_px": right_x,
                    "center_x_px": (left_x + right_x) / 2.0,
                    "width_px": None,
                    "status": "error",
                }
            )
            continue
        measurement_type = (
            "line" if int(interval["parity"]) == line_parity else "gap"
        )
        display_counts[measurement_type] += 1
        displayed[measurement_type].append(
            {
                **interval,
                "measurement_type": measurement_type,
                "display_id": (
                    f"{DISPLAY_PREFIX[measurement_type]}"
                    f"{display_counts[measurement_type]}"
                ),
            }
        )
    return displayed


def continuity_preserving_selection(
    candidates: list[dict[str, object]],
    reference_candidates: list[dict[str, object]],
    measurement_type: str,
) -> list[dict[str, object]]:
    """Map one Y row onto fixed physical slots without replacement targets."""

    ordered_reference = sorted(
        reference_candidates,
        key=lambda item: float(item["center_x_px"]),
    )
    if len(ordered_reference) < TARGETS_PER_ROW:
        raise ValueError(
            f"Reference row has fewer than {TARGETS_PER_ROW} {measurement_type} targets"
        )
    selected_reference = select_central_targets(
        ordered_reference,
        measurement_type,
    )
    ordered_candidates = sorted(
        candidates,
        key=lambda item: float(item["center_x_px"]),
    )
    selected: list[dict[str, object]] = []
    for target_id, reference in zip(
        (item["target_id"] for item in selected_reference),
        selected_reference,
        strict=True,
    ):
        reference_center = float(reference["center_x_px"])
        reference_index = next(
            index
            for index, item in enumerate(ordered_reference)
            if float(item["center_x_px"]) == reference_center
        )
        lower_bound = float("-inf")
        upper_bound = float("inf")
        if reference_index > 0:
            lower_bound = (
                float(ordered_reference[reference_index - 1]["center_x_px"])
                + reference_center
            ) / 2.0
        if reference_index + 1 < len(ordered_reference):
            upper_bound = (
                reference_center
                + float(ordered_reference[reference_index + 1]["center_x_px"])
            ) / 2.0
        matches = [
            item
            for item in ordered_candidates
            if lower_bound <= float(item["center_x_px"]) < upper_bound
        ]
        common = {
            "measurement_type": measurement_type,
            "target_id": target_id,
            "slot_center_x_px": reference_center,
            "reference_physical_index": reference["physical_index"],
        }
        if len(matches) == 1:
            selected.append(
                {
                    **matches[0],
                    **common,
                    "selection_message": "continuity-preserving physical slot match",
                }
            )
            continue
        selected.append(
            {
                **common,
                "left_x_px": None,
                "right_x_px": None,
                "center_x_px": None,
                "width_px": None,
                "brightness": None,
                "status": "error",
                "selection_message": (
                    MISSING_SLOT_MESSAGE if not matches else AMBIGUOUS_SLOT_MESSAGE
                ),
            }
        )
    return selected


def _reference_fraction(
    scanlines: dict[float, tuple[int, dict[str, list[dict[str, object]]], dict[str, object]]],
) -> float:
    """Choose the most complete scanline, preferring fewer rejections and Y50."""

    return min(
        scanlines,
        key=lambda fraction: (
            -len(scanlines[fraction][1]["interval"]),
            int(scanlines[fraction][2]["rejected_peak_count"]),
            abs(fraction - 0.5),
            fraction,
        ),
    )


def compare_image(
    image_path: Path,
    measurement_types: list[str],
    *,
    duplicate_factor: int,
    output_root: Path,
    runtime_is_warmup: bool,
) -> ImageComparison:
    started_at = perf_counter()
    input_image = load_image_unicode(image_path)
    recovery = recover_native_pixels(input_image, duplicate_factor)
    edge_diagnostics = extract_edge_diagnostics(recovery.image)
    image_id = image_path.stem
    rows: list[dict[str, object]] = []
    measurements_by_y: dict[float, dict[str, list[dict[str, object]]]] = {}
    all_measurements_by_y: dict[float, dict[str, list[dict[str, object]]]] = {}
    scanlines: dict[
        float,
        tuple[int, dict[str, list[dict[str, object]]], dict[str, object]],
    ] = {}
    for y_fraction, native_y in measurement_rows(recovery.image.shape[0]):
        candidates, diagnostics = measurements_at_native_y(
            edge_diagnostics.grayscale_roi[native_y],
            edge_diagnostics.x_gradient[native_y],
        )
        scanlines[y_fraction] = (native_y, candidates, diagnostics)

    reference_fraction = _reference_fraction(scanlines)
    _, reference_raw, reference_diagnostics = scanlines[reference_fraction]
    reference_line_parity = int(reference_diagnostics["line_parity"])
    reference_candidates = _typed_candidates(
        reference_raw,
        reference_line_parity,
    )
    scale_note = TARGET_SCALE_NOTE if image_id == "target_01" else ""

    for y_fraction, (native_y, candidates, diagnostics) in scanlines.items():
        typed_candidates = _typed_candidates(candidates, reference_line_parity)
        displayed_intervals = visualization_physical_intervals(
            candidates,
            reference_line_parity,
        )
        all_measurements_by_y[y_fraction] = {
            **displayed_intervals,
            "sidewall": visualization_measurements(
                typed_candidates["sidewall"],
                "sidewall",
            ),
        }
        selected = {
            measurement_type: continuity_preserving_selection(
                typed_candidates[measurement_type],
                reference_candidates[measurement_type],
                measurement_type,
            )
            for measurement_type in MEASUREMENT_ORDER
        }
        measurements_by_y[y_fraction] = selected
        message = (
            f"native_y_px={native_y}; duplicate_factor={recovery.duplicate_factor}; "
            f"reference_y_fraction={reference_fraction}; "
            f"reference_line_parity={reference_line_parity}; "
            f"observed_line_parity={diagnostics['line_parity']}; "
            f"rejected_peaks={diagnostics['rejected_peak_count']}; "
            "continuity-preserving physical slots"
        )
        if scale_note:
            message = f"{message}; {scale_note}"
        for measurement_type in measurement_types:
            for measurement in selected[measurement_type]:
                width_px = measurement["width_px"]
                rows.append(
                    {
                        "image_id": image_id,
                        "measurement_type": measurement_type,
                        "y_fraction": y_fraction,
                        "target_id": measurement["target_id"],
                        "width_px": width_px,
                        "width_um": (
                            None
                            if width_px is None
                            else float(width_px) * UM_PER_NATIVE_PX
                        ),
                        "center_x_px": measurement["center_x_px"],
                        "left_x_px": measurement["left_x_px"],
                        "right_x_px": measurement["right_x_px"],
                        "status": measurement["status"],
                        "message": (
                            f"{message}; {measurement['selection_message']}"
                        ),
                    }
                )

    runtime_ms = (perf_counter() - started_at) * 1000.0
    for row in rows:
        row["runtime_ms"] = "" if runtime_is_warmup else round(runtime_ms, 6)
    overlay_directory = output_root / "method2_overlay"
    overlay_directory.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_directory / f"{image_id}_method2_team_protocol_overlay.png"
    render_overlay(
        recovery.image,
        image_id,
        measurements_by_y,
        measurement_types,
        overlay_path,
        scale_note=scale_note,
    )
    all_measurements_overlay_path = (
        overlay_directory / f"{image_id}_method2_all_measurements_overlay.png"
    )
    render_all_measurements_overlay(
        recovery.image,
        image_id,
        all_measurements_by_y,
        all_measurements_overlay_path,
        scale_note=scale_note,
    )
    summary_directory = output_root / "summaries"
    summary_path = summary_directory / f"{image_id}_method2_team_protocol_summary.txt"
    write_summary(
        summary_path,
        image_id,
        measurements_by_y,
        measurement_types,
        reference_fraction=reference_fraction,
        scale_note=scale_note,
    )
    return ImageComparison(
        image_path=image_path,
        recovery=recovery,
        rows=rows,
        measurements_by_y=measurements_by_y,
        runtime_ms=runtime_ms,
        overlay_path=overlay_path,
        all_measurements_overlay_path=all_measurements_overlay_path,
        summary_path=summary_path,
    )


def _draw_dashed_line(
    canvas: NDArray[np.uint8],
    start_x: int,
    end_x: int,
    y: int,
) -> None:
    for x in range(start_x, end_x, 24):
        cv2.line(canvas, (x, y), (min(x + 12, end_x), y), (255, 255, 255), 1)


def render_overlay(
    native_image: NDArray[np.generic],
    image_id: str,
    measurements_by_y: dict[float, dict[str, list[dict[str, object]]]],
    measurement_types: list[str],
    output_path: Path,
    *,
    scale_note: str = "",
) -> None:
    """Render one native-resolution panel per selected measurement type."""

    if native_image.ndim == 2:
        base = cv2.cvtColor(native_image, cv2.COLOR_GRAY2BGR)
    elif native_image.shape[2] == 4:
        base = cv2.cvtColor(native_image, cv2.COLOR_BGRA2BGR)
    else:
        base = native_image.copy()
    height, width = base.shape[:2]
    left_margin = 62
    panel_header = 35
    title_height = 91 if scale_note else 72
    bottom_height = 58
    panel_height = panel_header + height
    canvas = np.full(
        (
            title_height + panel_height * len(measurement_types) + bottom_height,
            left_margin + width + 12,
            3,
        ),
        248,
        dtype=np.uint8,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "Frozen Method 2 - Team Protocol Comparison",
        (left_margin + 180, 25),
        font,
        0.68,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{image_id}; scale = {UM_PER_NATIVE_PX} um/native px; native coordinates",
        (left_margin + 205, 51),
        font,
        0.48,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    if scale_note:
        cv2.putText(
            canvas,
            scale_note,
            (left_margin + 205, 70),
            font,
            0.42,
            (0, 0, 180),
            1,
            cv2.LINE_AA,
        )

    for panel_index, measurement_type in enumerate(measurement_types):
        panel_top = title_height + panel_index * panel_height
        image_top = panel_top + panel_header
        canvas[image_top : image_top + height, left_margin : left_margin + width] = base
        color = OVERLAY_COLORS[measurement_type]
        cv2.putText(
            canvas,
            measurement_type.upper(),
            (left_margin, panel_top + 23),
            font,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        for y_fraction, native_y in measurement_rows(height):
            canvas_y = image_top + native_y
            _draw_dashed_line(
                canvas,
                left_margin,
                left_margin + width - 1,
                canvas_y,
            )
            y_label = f"Y{round(y_fraction * 100):02d} y={native_y}px"
            cv2.putText(
                canvas,
                y_label,
                (3, canvas_y + 4),
                font,
                0.34,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            for target_index, measurement in enumerate(
                measurements_by_y[y_fraction][measurement_type]
            ):
                if measurement["status"] != "ok":
                    slot_x = int(round(float(measurement["slot_center_x_px"])))
                    marker_x = left_margin + slot_x
                    error_color = (0, 0, 255)
                    cv2.line(
                        canvas,
                        (marker_x - 6, canvas_y - 6),
                        (marker_x + 6, canvas_y + 6),
                        error_color,
                        2,
                    )
                    cv2.line(
                        canvas,
                        (marker_x - 6, canvas_y + 6),
                        (marker_x + 6, canvas_y - 6),
                        error_color,
                        2,
                    )
                    label = f"{measurement['target_id']} missing"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, 0.32, 1
                    )
                    text_x = marker_x - text_width // 2
                    text_y = canvas_y + 19
                    cv2.rectangle(
                        canvas,
                        (text_x - 2, text_y - text_height - 2),
                        (text_x + text_width + 2, text_y + baseline + 2),
                        (250, 250, 250),
                        -1,
                    )
                    cv2.putText(
                        canvas,
                        label,
                        (text_x, text_y),
                        font,
                        0.32,
                        error_color,
                        1,
                        cv2.LINE_AA,
                    )
                    continue
                left_x = int(round(float(measurement["left_x_px"])))
                right_x = int(round(float(measurement["right_x_px"])))
                start = left_margin + left_x
                end = left_margin + right_x
                cv2.line(canvas, (start, canvas_y), (end, canvas_y), color, 2)
                cv2.line(canvas, (start, canvas_y - 5), (start, canvas_y + 5), color, 2)
                cv2.line(canvas, (end, canvas_y - 5), (end, canvas_y + 5), color, 2)
                width_px = float(measurement["width_px"])
                label = (
                    f"{measurement['target_id']} {width_px:g}px/"
                    f"{width_px * UM_PER_NATIVE_PX:.4f}um"
                )
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, 0.28, 1
                )
                center = (start + end) // 2
                text_x = center - text_width // 2
                if measurement_type == "sidewall":
                    text_y = canvas_y - 9 if target_index % 2 == 0 else canvas_y + 19
                else:
                    text_y = canvas_y + 17
                cv2.rectangle(
                    canvas,
                    (text_x - 2, text_y - text_height - 2),
                    (text_x + text_width + 2, text_y + baseline + 2),
                    (250, 250, 250),
                    -1,
                )
                cv2.putText(
                    canvas,
                    label,
                    (text_x, text_y),
                    font,
                    0.28,
                    (15, 15, 15),
                    1,
                    cv2.LINE_AA,
                )

    footer_y = title_height + panel_height * len(measurement_types) + 30
    cv2.putText(
        canvas,
        "Adapter output only: production Outer/Inner/Gap semantics are unchanged.",
        (left_margin + 180, footer_y),
        font,
        0.42,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    encoded_ok, encoded = cv2.imencode(".png", canvas)
    if not encoded_ok:
        raise RuntimeError("Could not encode overlay PNG")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(output_path)


def render_all_measurements_overlay(
    native_image: NDArray[np.generic],
    image_id: str,
    measurements_by_y: dict[float, dict[str, list[dict[str, object]]]],
    output_path: Path,
    *,
    scale_note: str = "",
) -> None:
    """Render every detected candidate; display IDs never enter the team CSV."""

    if native_image.ndim == 2:
        base = cv2.cvtColor(native_image, cv2.COLOR_GRAY2BGR)
    elif native_image.shape[2] == 4:
        base = cv2.cvtColor(native_image, cv2.COLOR_BGRA2BGR)
    else:
        base = native_image.copy()
    height, width = base.shape[:2]
    left_margin = 62
    title_height = 94
    bottom_height = 42
    canvas = np.full(
        (title_height + height + bottom_height, left_margin + width + 12, 3),
        248,
        dtype=np.uint8,
    )
    image_top = title_height
    canvas[image_top : image_top + height, left_margin : left_margin + width] = base
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "Frozen Method 2 - All Detected Measurements",
        (left_margin + 180, 25),
        font,
        0.68,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{image_id}; scale = {UM_PER_NATIVE_PX} um/current px",
        (left_margin + 205, 51),
        font,
        0.48,
        (55, 55, 55),
        1,
        cv2.LINE_AA,
    )
    warning = scale_note or "provisional: acquisition-native scale not yet confirmed"
    cv2.putText(
        canvas,
        warning,
        (left_margin + 205, 71),
        font,
        0.42,
        (0, 0, 180),
        1,
        cv2.LINE_AA,
    )
    legend_x = left_margin + 205
    for measurement_type in MEASUREMENT_ORDER:
        color = OVERLAY_COLORS[measurement_type]
        cv2.putText(
            canvas,
            measurement_type.upper(),
            (legend_x, 88),
            font,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
        legend_x += 95
    cv2.putText(
        canvas,
        "MISSING",
        (legend_x, 88),
        font,
        0.36,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    segment_offsets = {"line": -5, "gap": 5, "sidewall": 0}
    for y_fraction, native_y in measurement_rows(height):
        canvas_y = image_top + native_y
        _draw_dashed_line(
            canvas,
            left_margin,
            left_margin + width - 1,
            canvas_y,
        )
        cv2.putText(
            canvas,
            f"Y{round(y_fraction * 100):02d} y={native_y}px",
            (3, canvas_y + 4),
            font,
            0.34,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        for measurement_type in MEASUREMENT_ORDER:
            color = OVERLAY_COLORS[measurement_type]
            for display_index, measurement in enumerate(
                measurements_by_y[y_fraction][measurement_type]
            ):
                left_x = int(round(float(measurement["left_x_px"])))
                right_x = int(round(float(measurement["right_x_px"])))
                segment_y = canvas_y + segment_offsets[measurement_type]
                start = left_margin + left_x
                end = left_margin + right_x
                cv2.line(canvas, (start, segment_y), (end, segment_y), color, 2)
                cv2.line(
                    canvas,
                    (start, segment_y - 4),
                    (start, segment_y + 4),
                    color,
                    1,
                )
                cv2.line(
                    canvas,
                    (end, segment_y - 4),
                    (end, segment_y + 4),
                    color,
                    1,
                )
                width_um = float(measurement["width_px"]) * UM_PER_NATIVE_PX
                label = f"{measurement['display_id']} {width_um:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    font,
                    0.26,
                    1,
                )
                center = (start + end) // 2
                text_x = max(
                    left_margin,
                    min(
                        center - text_width // 2,
                        left_margin + width - text_width - 1,
                    ),
                )
                if measurement_type == "line":
                    text_y = canvas_y - 12
                elif measurement_type == "gap":
                    text_y = canvas_y + 21
                else:
                    text_y = (
                        canvas_y - 28
                        if display_index % 2 == 0
                        else canvas_y + 37
                    )
                cv2.rectangle(
                    canvas,
                    (text_x - 1, text_y - text_height - 1),
                    (text_x + text_width + 1, text_y + baseline + 1),
                    (248, 248, 248),
                    -1,
                )
                cv2.putText(
                    canvas,
                    label,
                    (text_x, text_y),
                    font,
                    0.26,
                    (15, 15, 15),
                    1,
                    cv2.LINE_AA,
                )
        for missing_index, measurement in enumerate(
            measurements_by_y[y_fraction]["missing"]
        ):
            left_x = int(round(float(measurement["left_x_px"])))
            right_x = int(round(float(measurement["right_x_px"])))
            start = left_margin + left_x
            end = left_margin + right_x
            center = (start + end) // 2
            error_color = (0, 0, 255)
            for dash_start in range(start, end, 12):
                cv2.line(
                    canvas,
                    (dash_start, canvas_y),
                    (min(dash_start + 6, end), canvas_y),
                    error_color,
                    2,
                )
            cv2.line(
                canvas,
                (center - 6, canvas_y - 6),
                (center + 6, canvas_y + 6),
                error_color,
                2,
            )
            cv2.line(
                canvas,
                (center - 6, canvas_y + 6),
                (center + 6, canvas_y - 6),
                error_color,
                2,
            )
            label = f"{measurement['display_id']} missing"
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                font,
                0.28,
                1,
            )
            text_x = max(
                left_margin,
                min(
                    center - text_width // 2,
                    left_margin + width - text_width - 1,
                ),
            )
            text_y = canvas_y + 53 + (missing_index % 2) * 13
            cv2.rectangle(
                canvas,
                (text_x - 1, text_y - text_height - 1),
                (text_x + text_width + 1, text_y + baseline + 1),
                (248, 248, 248),
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (text_x, text_y),
                font,
                0.28,
                error_color,
                1,
                cv2.LINE_AA,
            )

    footer_y = title_height + height + 27
    cv2.putText(
        canvas,
        "Display IDs are visualization-only; CSV IDs remain center-five physical slots.",
        (left_margin + 180, footer_y),
        font,
        0.42,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    encoded_ok, encoded = cv2.imencode(".png", canvas)
    if not encoded_ok:
        raise RuntimeError("Could not encode all-measurements overlay PNG")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(output_path)


def write_summary(
    output_path: Path,
    image_id: str,
    measurements_by_y: dict[float, dict[str, list[dict[str, object]]]],
    measurement_types: list[str],
    *,
    reference_fraction: float,
    scale_note: str,
) -> None:
    """Write a concise comparison-only continuity summary."""

    lines = [
        f"image_id: {image_id}",
        f"scale_um_per_native_px: {UM_PER_NATIVE_PX}",
        f"reference_y_fraction: {reference_fraction}",
        f"scale_note: {scale_note or 'none'}",
        "selection: continuity-preserving physical slots",
    ]
    total_ok = 0
    total_error = 0
    for y_fraction, measurements in measurements_by_y.items():
        for measurement_type in measurement_types:
            selected = measurements[measurement_type]
            missing = [
                str(item["target_id"])
                for item in selected
                if item["status"] != "ok"
            ]
            ok_count = len(selected) - len(missing)
            total_ok += ok_count
            total_error += len(missing)
            missing_text = ",".join(missing) if missing else "none"
            lines.append(
                f"Y{round(y_fraction * 100):02d} {measurement_type}: "
                f"ok={ok_count} error={len(missing)} missing={missing_text}"
            )
    lines.append(f"total_ok: {total_ok}")
    lines.append(f"total_error: {total_error}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def discover_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def run(
    input_path: Path,
    *,
    measurement_selection: str,
    duplicate_factor: int | None,
    output_root: Path,
) -> tuple[list[ImageComparison], Path]:
    measurement_types = parse_measurement_types(measurement_selection)
    images = discover_images(input_path)
    if not images:
        raise ValueError(f"No supported images found under {input_path}")
    folder_batch = input_path.is_dir()
    comparisons = [
        compare_image(
            image_path,
            measurement_types,
            duplicate_factor=duplicate_factor,
            output_root=output_root,
            runtime_is_warmup=folder_batch and index == 0,
        )
        for index, image_path in enumerate(images)
    ]
    prediction_filename = (
        "method2_predictions.csv"
        if folder_batch
        else f"{images[0].stem}_method2_predictions.csv"
    )
    prediction_path = output_root / "predictions" / prediction_filename
    save_prediction_csv(
        prediction_path,
        METHOD_ID,
        [row for comparison in comparisons for row in comparison.rows],
    )
    return comparisons, prediction_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--measurement-types", default="123")
    parser.add_argument(
        "--duplicate-factor",
        type=int,
        choices=(1, 2),
        default=None,
        help="Auto-detect exact 2x duplicate pixels by default; use 1 or 2 to override.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data/derived/team_protocol_compare",
    )
    args = parser.parse_args()
    comparisons, prediction_path = run(
        args.input_path,
        measurement_selection=args.measurement_types,
        duplicate_factor=args.duplicate_factor,
        output_root=args.output_root,
    )
    for comparison in comparisons:
        print(f"image={comparison.image_path}")
        print(
            f"input_shape={comparison.recovery.input_shape}; "
            f"native_shape={comparison.recovery.native_shape}; "
            f"duplicate_factor={comparison.recovery.duplicate_factor}; "
            f"runtime_ms={comparison.runtime_ms:.3f}; "
            f"overlay={comparison.overlay_path}; "
            f"all_measurements_overlay={comparison.all_measurements_overlay_path}; "
            f"summary={comparison.summary_path}"
        )
    print(f"prediction_csv={prediction_path}")


if __name__ == "__main__":
    main()
