"""Characterize unpaired signed-gradient peaks for the six pilot images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from run_gradient_canny_pilot import (
    PILOT_IMAGES,
    PROJECT_ROOT,
    _labeled_panel,
    _load_roi_config,
    _write_png,
)

from ai_cd_metrology.image_io import load_image
from ai_cd_metrology.metrology.gradient_canny import (
    CHARACTERIZATION_CANNY_THRESHOLD_HIGH,
    CHARACTERIZATION_CANNY_THRESHOLD_LOW,
    PILOT_GAUSSIAN_KERNEL_SIZE,
    PILOT_GAUSSIAN_SIGMA,
    PILOT_SOBEL_KERNEL_SIZE,
    PROFILE_PEAK_RELATIVE_THRESHOLD,
    SignedGradientProfile,
    characterize_signed_x_gradient,
    extract_edge_diagnostics,
)
from ai_cd_metrology.roi import crop_normalized_roi
from ai_cd_metrology.schemas import ImageRecord


POSITIVE_COLOR = (30, 30, 240)
NEGATIVE_COLOR = (240, 220, 30)


def _overlay_peak_guides(
    grayscale: np.ndarray,
    profile: SignedGradientProfile,
) -> np.ndarray:
    overlay = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)
    for x in profile.positive_peak_indices:
        cv2.line(overlay, (x, 0), (x, overlay.shape[0] - 1), POSITIVE_COLOR, 2)
    for x in profile.negative_peak_indices:
        cv2.line(overlay, (x, 0), (x, overlay.shape[0] - 1), NEGATIVE_COLOR, 2)
    return overlay


def _profile_plot(profile: SignedGradientProfile) -> np.ndarray:
    values = profile.values
    width = values.size
    height = 420
    top_margin = 35
    bottom_margin = 45
    center_y = (top_margin + height - bottom_margin) // 2
    amplitude = (height - top_margin - bottom_margin) / 2
    max_absolute = max(float(np.max(np.abs(values))), 1.0)

    plot = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.line(plot, (0, center_y), (width - 1, center_y), (125, 125, 125), 1)

    def value_to_y(value: float) -> int:
        return round(center_y - (value / max_absolute) * amplitude)

    cv2.line(
        plot,
        (0, value_to_y(profile.positive_threshold)),
        (width - 1, value_to_y(profile.positive_threshold)),
        (80, 80, 150),
        1,
    )
    cv2.line(
        plot,
        (0, value_to_y(profile.negative_threshold)),
        (width - 1, value_to_y(profile.negative_threshold)),
        (150, 130, 70),
        1,
    )

    points = np.column_stack(
        (
            np.arange(width, dtype=np.int32),
            np.rint(center_y - values * amplitude / max_absolute).astype(np.int32),
        )
    ).reshape((-1, 1, 2))
    cv2.polylines(plot, [points], False, (230, 230, 230), 2, cv2.LINE_AA)

    for x in profile.positive_peak_indices:
        y = value_to_y(float(values[x]))
        cv2.line(plot, (x, top_margin), (x, height - bottom_margin), POSITIVE_COLOR, 1)
        cv2.circle(plot, (x, y), 5, POSITIVE_COLOR, -1, cv2.LINE_AA)
    for x in profile.negative_peak_indices:
        y = value_to_y(float(values[x]))
        cv2.line(plot, (x, top_margin), (x, height - bottom_margin), NEGATIVE_COLOR, 1)
        cv2.circle(plot, (x, y), 5, NEGATIVE_COLOR, -1, cv2.LINE_AA)

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = min(round(fraction * width), width - 1)
        cv2.line(plot, (x, height - bottom_margin), (x, height - bottom_margin + 8), (180, 180, 180), 1)
        cv2.putText(
            plot,
            str(x),
            (max(0, x - 18), height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return plot


def _format_positions(indices: tuple[int, ...]) -> str:
    return ",".join(str(index) for index in indices)


def _build_characterization(
    filename: str,
    pattern_position: str,
    grayscale: np.ndarray,
    profile: SignedGradientProfile,
    canny_edges: np.ndarray,
) -> np.ndarray:
    panels = [
        _labeled_panel(grayscale, "Grayscale ROI"),
        _labeled_panel(
            _overlay_peak_guides(grayscale, profile),
            "Peak guides: positive=red, negative=cyan",
        ),
        _labeled_panel(
            _profile_plot(profile),
            "Median-y signed Sobel-x profile with unpaired peaks",
        ),
        _labeled_panel(
            canny_edges,
            "Canny edges "
            f"({CHARACTERIZATION_CANNY_THRESHOLD_LOW}/"
            f"{CHARACTERIZATION_CANNY_THRESHOLD_HIGH})",
        ),
    ]
    grid = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    header_height = 108
    output = np.full((grid.shape[0] + header_height, grid.shape[1], 3), 18, dtype=np.uint8)
    output[header_height:] = grid
    lines = (
        f"{filename} | position={pattern_position} | peaks are diagnostic only (not paired)",
        f"positive x: {_format_positions(profile.positive_peak_indices)}",
        f"negative x: {_format_positions(profile.negative_peak_indices)}",
    )
    for row, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (16, 28 + 32 * row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return output


def _signed_sequence(profile: SignedGradientProfile) -> str:
    peaks = [(x, "+") for x in profile.positive_peak_indices]
    peaks.extend((x, "-") for x in profile.negative_peak_indices)
    return "".join(sign for _, sign in sorted(peaks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/images/images/PASS_normal"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/metrology.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/gradient_canny_characterization"),
    )
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    raw_root = (PROJECT_ROOT / "data" / "raw").resolve()
    try:
        output_dir.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "characterization: median-y aggregation, "
        f"relative peak threshold={PROFILE_PEAK_RELATIVE_THRESHOLD:.2f}, "
        f"Gaussian={PILOT_GAUSSIAN_KERNEL_SIZE}x{PILOT_GAUSSIAN_KERNEL_SIZE}, "
        f"sigma={PILOT_GAUSSIAN_SIGMA}, Sobel-x={PILOT_SOBEL_KERNEL_SIZE}, "
        f"Canny={CHARACTERIZATION_CANNY_THRESHOLD_LOW}/"
        f"{CHARACTERIZATION_CANNY_THRESHOLD_HIGH}"
    )
    for filename, wafer_id, die_id, pattern_position in PILOT_IMAGES:
        path = image_dir / filename
        image = load_image(
            ImageRecord(
                image_id=path.stem,
                file_path=path,
                wafer_id=wafer_id,
                die_id=die_id,
                pattern_position=pattern_position,
                capture_region="standard",
                defect_label="normal",
            )
        )
        roi = _load_roi_config(args.config, pattern_position)
        cropped, _ = crop_normalized_roi(image, roi)
        diagnostics = extract_edge_diagnostics(
            cropped,
            canny_threshold_low=CHARACTERIZATION_CANNY_THRESHOLD_LOW,
            canny_threshold_high=CHARACTERIZATION_CANNY_THRESHOLD_HIGH,
        )
        profile = characterize_signed_x_gradient(diagnostics.x_gradient)
        output = _build_characterization(
            filename,
            pattern_position,
            diagnostics.grayscale_roi,
            profile,
            diagnostics.canny_edges,
        )
        output_path = output_dir / f"{path.stem}_gradient_characterization.png"
        _write_png(output_path, output)
        canny_pixels = int(np.count_nonzero(diagnostics.canny_edges))
        print(
            f"{output_path} | positive={len(profile.positive_peak_indices)}, "
            f"negative={len(profile.negative_peak_indices)}, "
            f"sequence={_signed_sequence(profile)}, canny_pixels={canny_pixels}"
        )


if __name__ == "__main__":
    main()
