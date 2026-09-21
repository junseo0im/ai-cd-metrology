"""Select a stable sidewall from every detected Canny/gradient sidewall track.

This is the production-selection path.  The frozen team-protocol benchmark
remains in ``run_method2_team_protocol_compare.py`` so missing detections stay
visible during accuracy evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
TEAM_PROTOCOL_DIRECTORY = PROJECT_ROOT / "team protocol"
for import_path in (SCRIPT_DIRECTORY, PROJECT_ROOT / "src", TEAM_PROTOCOL_DIRECTORY):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_method2_team_protocol_compare as benchmark  # noqa: E402
from team_protocol import UM_PER_NATIVE_PX, measurement_rows  # noqa: E402


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
CHOICE_NAMES = {"1": "min", "2": "median", "3": "max", "min": "min", "median": "median", "max": "max"}


def _sample(candidate: dict[str, object], row: dict[str, object], source: str) -> dict[str, object]:
    width_px = float(candidate["width_px"])
    return {
        "row_index": int(row["row_index"]),
        "y_fraction": float(row["y_fraction"]),
        "y_native": int(row["y_native"]),
        "left_x_native": float(candidate["left_x_px"]),
        "right_x_native": float(candidate["right_x_px"]),
        "center_x_native": float(candidate["center_x_px"]),
        "width_px_native": width_px,
        "width_um": width_px * UM_PER_NATIVE_PX,
        "detection_source": source,
    }


def track_sidewalls(
    rows: list[dict[str, object]],
    center_tolerance_px: float | None = None,
) -> tuple[list[dict[str, object]], float]:
    """Associate detected sidewalls across Y rows without fixed central slots."""

    populated = [row for row in rows if row["candidates"]]
    if not populated:
        return [], float(center_tolerance_px or 20.0)
    anchor = max(populated, key=lambda row: len(row["candidates"]))
    anchor_candidates = sorted(anchor["candidates"], key=lambda item: float(item["center_x_px"]))
    centers = np.asarray([float(item["center_x_px"]) for item in anchor_candidates])
    if center_tolerance_px is None:
        pitches = np.diff(centers)
        center_tolerance_px = max(3.0, float(np.median(pitches)) * 0.35) if len(pitches) else 20.0
    if center_tolerance_px <= 0:
        raise ValueError("center_tolerance_px must be positive")

    tracks = [{"samples": [_sample(candidate, anchor, "global")]} for candidate in anchor_candidates]
    other_rows = sorted(
        (row for row in rows if row is not anchor),
        key=lambda row: len(row["candidates"]),
        reverse=True,
    )
    for row in other_rows:
        candidates = sorted(row["candidates"], key=lambda item: float(item["center_x_px"]))
        possible: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            reference = float(np.mean([item["center_x_native"] for item in track["samples"]]))
            for candidate_index, candidate in enumerate(candidates):
                distance = abs(reference - float(candidate["center_x_px"]))
                if distance <= center_tolerance_px:
                    possible.append((distance, track_index, candidate_index))
        used_tracks: set[int] = set()
        used_candidates: set[int] = set()
        for _, track_index, candidate_index in sorted(possible):
            if track_index in used_tracks or candidate_index in used_candidates:
                continue
            tracks[track_index]["samples"].append(
                _sample(candidates[candidate_index], row, "global")
            )
            used_tracks.add(track_index)
            used_candidates.add(candidate_index)
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index not in used_candidates:
                tracks.append({"samples": [_sample(candidate, row, "global")]})
    return tracks, float(center_tolerance_px)


def _local_retry_candidate(
    gradient_row: np.ndarray,
    expected_center: float,
    expected_width: float,
) -> dict[str, object] | None:
    """Retry one missing row in a narrow local window with a relaxed peak cutoff."""

    profile = np.asarray(gradient_row, dtype=float).ravel()
    center = int(round(expected_center))
    radius = max(4, int(round(expected_width * 1.8)))
    left_start, left_end = max(0, center - radius), min(len(profile), center + 1)
    right_start, right_end = max(0, center), min(len(profile), center + radius + 1)
    if left_end - left_start < 2 or right_end - right_start < 2:
        return None
    negative_x = left_start + int(np.argmin(profile[left_start:left_end]))
    positive_x = right_start + int(np.argmax(profile[right_start:right_end]))
    negative_value = float(profile[negative_x])
    positive_value = float(profile[positive_x])
    global_negative = abs(min(float(np.min(profile)), 0.0))
    global_positive = max(float(np.max(profile)), 0.0)
    width = positive_x - negative_x
    if negative_value >= 0.0 or positive_value <= 0.0:
        return None
    if global_negative == 0.0 or global_positive == 0.0:
        return None
    if abs(negative_value) < 0.20 * global_negative or positive_value < 0.20 * global_positive:
        return None
    if not 0.45 * expected_width <= width <= 1.80 * expected_width:
        return None
    return {
        "measurement_type": "sidewall",
        "left_x_px": float(negative_x),
        "right_x_px": float(positive_x),
        "center_x_px": (negative_x + positive_x) / 2.0,
        "width_px": float(width),
        "status": "ok",
    }


def retry_missing_rows(tracks: list[dict[str, object]], rows: list[dict[str, object]]) -> None:
    """Add only genuine local measurements; never interpolate a width."""

    for track in tracks:
        samples = track["samples"]
        if not samples:
            continue
        expected_center = float(np.mean([item["center_x_native"] for item in samples]))
        expected_width = float(np.median([item["width_px_native"] for item in samples]))
        present = {int(item["row_index"]) for item in samples}
        for row in rows:
            row_index = int(row["row_index"])
            if row_index in present:
                continue
            candidate = _local_retry_candidate(
                np.asarray(row["gradient_row"]), expected_center, expected_width
            )
            if candidate is not None:
                samples.append(_sample(candidate, row, "local_retry"))


def summarize_tracks(
    tracks: list[dict[str, object]],
    expected_rows: int,
    min_valid_rows: int,
    max_deviation_um: float,
) -> list[dict[str, object]]:
    if not 1 <= min_valid_rows <= expected_rows:
        raise ValueError("min_valid_rows must be between 1 and the number of Y rows")
    if max_deviation_um < 0:
        raise ValueError("max_deviation_um must be non-negative")
    ordered = sorted(
        tracks,
        key=lambda track: float(np.mean([item["center_x_native"] for item in track["samples"]])),
    )
    summaries = []
    for track_id, track in enumerate(ordered, start=1):
        samples = sorted(track["samples"], key=lambda item: int(item["row_index"]))
        widths = np.asarray([float(item["width_px_native"]) for item in samples])
        mean_px = float(np.mean(widths))
        std_px = float(np.std(widths))
        std_um = std_px * UM_PER_NATIVE_PX
        reasons = []
        if len(samples) < min_valid_rows:
            reasons.append(f"coverage {len(samples)}/{expected_rows} below {min_valid_rows}")
        if std_um > max_deviation_um:
            reasons.append(f"std {std_um:.4f} um exceeds {max_deviation_um:.4f} um")
        summaries.append(
            {
                "track_id": f"T{track_id}",
                "center_x_native": round(float(np.mean([item["center_x_native"] for item in samples])), 4),
                "sample_count": len(samples),
                "expected_sample_count": expected_rows,
                "coverage_ratio": round(len(samples) / expected_rows, 4),
                "mean_width_px_native": round(mean_px, 4),
                "std_width_px_native": round(std_px, 4),
                "mean_width_um": round(mean_px * UM_PER_NATIVE_PX, 4),
                "std_width_um": round(std_um, 4),
                "usable": not reasons,
                "rejection_reason": "; ".join(reasons) if reasons else None,
                "samples": samples,
                "selected": False,
            }
        )
    return summaries


def select_track(summaries: list[dict[str, object]], choice: str) -> dict[str, object] | None:
    usable = sorted(
        (item for item in summaries if item["usable"]),
        key=lambda item: float(item["mean_width_px_native"]),
    )
    if not usable:
        return None
    normalized = CHOICE_NAMES[choice]
    if normalized == "min":
        selected = usable[0]
    elif normalized == "max":
        selected = usable[-1]
    else:
        selected = usable[len(usable) // 2]
    selected["selected"] = True
    selected["selection_choice"] = normalized
    return selected


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _save_overlay(
    native: np.ndarray,
    rows: list[dict[str, object]],
    summaries: list[dict[str, object]],
    selected: dict[str, object] | None,
    path: Path,
    choice: str,
) -> None:
    if native.ndim == 2:
        canvas = cv2.cvtColor(native, cv2.COLOR_GRAY2BGR)
    elif native.shape[2] == 4:
        canvas = cv2.cvtColor(native, cv2.COLOR_BGRA2BGR)
    else:
        canvas = native[:, :, :3].copy()
    for row in rows:
        y = int(row["y_native"])
        for x in range(0, canvas.shape[1], 16):
            cv2.line(canvas, (x, y), (min(x + 8, canvas.shape[1] - 1), y), (255, 255, 255), 1)
    for summary in summaries:
        if not summary["usable"]:
            continue
        is_selected = bool(summary["selected"])
        color = (0, 255, 0) if is_selected else (255, 255, 0)
        thickness = 3 if is_selected else 1
        for sample in summary["samples"]:
            y = int(sample["y_native"])
            left = int(round(float(sample["left_x_native"])))
            right = int(round(float(sample["right_x_native"])))
            cv2.line(canvas, (left, y), (right, y), color, thickness)
            cv2.line(canvas, (left, y - 4), (left, y + 4), color, thickness)
            cv2.line(canvas, (right, y - 4), (right, y + 4), color, thickness)
            if is_selected:
                label = f"{summary['track_id']} {sample['width_px_native']:.1f}px/{sample['width_um']:.3f}um"
                cv2.putText(canvas, label, (left + 5, max(14, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    title = "No usable sidewall" if selected is None else (
        f"selected {selected['track_id']} ({CHOICE_NAMES[choice]}): "
        f"mean {selected['mean_width_px_native']:.2f}px/{selected['mean_width_um']:.3f}um, "
        f"std {selected['std_width_um']:.3f}um"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 24), (255, 255, 255), -1)
    cv2.putText(canvas, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Could not encode usable-selection overlay")
    encoded.tofile(path)


def process_image(
    image_path: Path,
    output_root: Path,
    choice: str,
    min_valid_rows: int,
    max_deviation_um: float,
    center_tolerance_px: float | None,
    side_margin_ratio: float,
) -> dict[str, object]:
    if not 0.0 <= side_margin_ratio < 0.5:
        raise ValueError("side_margin_ratio must be in [0, 0.5)")
    image = benchmark.load_image_unicode(image_path)
    recovery = benchmark.recover_native_pixels(image)
    diagnostics = benchmark.extract_edge_diagnostics(recovery.image)
    rows = []
    left_margin = recovery.image.shape[1] * side_margin_ratio
    right_margin = recovery.image.shape[1] * (1.0 - side_margin_ratio)
    for row_index, (fraction, native_y) in enumerate(measurement_rows(recovery.image.shape[0])):
        candidates, detail = benchmark.measurements_at_native_y(
            diagnostics.grayscale_roi[native_y], diagnostics.x_gradient[native_y]
        )
        rows.append(
            {
                "row_index": row_index,
                "y_fraction": fraction,
                "y_native": native_y,
                "candidates": [
                    item for item in candidates["sidewall"]
                    if left_margin <= float(item["center_x_px"]) <= right_margin
                ],
                "gradient_row": diagnostics.x_gradient[native_y],
                "rejected_peak_count": detail["rejected_peak_count"],
            }
        )
    tracks, tolerance = track_sidewalls(rows, center_tolerance_px)
    retry_missing_rows(tracks, rows)
    summaries = summarize_tracks(tracks, len(rows), min_valid_rows, max_deviation_um)
    selected = select_track(summaries, choice)

    image_output = output_root / image_path.stem
    overlay_path = image_output / f"{image_path.stem}_canny_usable_{CHOICE_NAMES[choice]}_overlay.png"
    tracks_path = image_output / "sidewall_tracks.csv"
    selected_path = image_output / "selected_sidewall_rows.csv"
    report_path = image_output / "selection_report.json"
    _save_overlay(recovery.image, rows, summaries, selected, overlay_path, choice)
    summary_fields = [
        "track_id", "center_x_native", "sample_count", "expected_sample_count",
        "coverage_ratio", "mean_width_px_native", "std_width_px_native",
        "mean_width_um", "std_width_um", "usable", "rejection_reason", "selected",
    ]
    _write_csv(tracks_path, summaries, summary_fields)
    selected_rows = [] if selected is None else [
        {"track_id": selected["track_id"], **sample} for sample in selected["samples"]
    ]
    _write_csv(
        selected_path,
        selected_rows,
        ["track_id", "row_index", "y_fraction", "y_native", "left_x_native",
         "right_x_native", "center_x_native", "width_px_native", "width_um",
         "detection_source"],
    )
    report = {
        "image": str(image_path.resolve()),
        "native_shape": list(recovery.native_shape),
        "duplicate_factor": recovery.duplicate_factor,
        "duplicate_offset_x": recovery.offset_x,
        "duplicate_offset_y": recovery.offset_y,
        "um_per_native_px": UM_PER_NATIVE_PX,
        "choice": CHOICE_NAMES[choice],
        "min_valid_rows": min_valid_rows,
        "max_deviation_um": max_deviation_um,
        "center_tolerance_px": tolerance,
        "side_margin_ratio": side_margin_ratio,
        "track_count": len(summaries),
        "usable_track_count": sum(bool(item["usable"]) for item in summaries),
        "selected_track": selected,
        "overlay": str(overlay_path.resolve()),
        "tracks_csv": str(tracks_path.resolve()),
        "selected_rows_csv": str(selected_path.resolve()),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(item for item in path.iterdir() if item.suffix.lower() in SUPPORTED_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--choice", choices=tuple(CHOICE_NAMES), default="2", help="1=min, 2=median, 3=max")
    parser.add_argument("--min-valid-rows", type=int, default=4)
    parser.add_argument("--max-deviation-um", type=float, default=0.10)
    parser.add_argument("--center-tolerance-px", type=float, default=None)
    parser.add_argument("--side-margin-ratio", type=float, default=0.03)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "canny_usable")
    args = parser.parse_args()

    reports = [
        process_image(
            image, args.output_root, args.choice, args.min_valid_rows,
            args.max_deviation_um, args.center_tolerance_px, args.side_margin_ratio,
        )
        for image in collect_images(args.input_path)
    ]
    rows = []
    for report in reports:
        selected = report["selected_track"]
        rows.append(
            {
                "image": report["image"],
                "choice": report["choice"],
                "usable_track_count": report["usable_track_count"],
                "selected_track_id": selected["track_id"] if selected else None,
                "mean_width_px_native": selected["mean_width_px_native"] if selected else None,
                "std_width_px_native": selected["std_width_px_native"] if selected else None,
                "mean_width_um": selected["mean_width_um"] if selected else None,
                "std_width_um": selected["std_width_um"] if selected else None,
                "overlay": report["overlay"],
            }
        )
    _write_csv(
        args.output_root / "usable_selection_summary.csv",
        rows,
        ["image", "choice", "usable_track_count", "selected_track_id",
         "mean_width_px_native", "std_width_px_native", "mean_width_um",
         "std_width_um", "overlay"],
    )
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
