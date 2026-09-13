"""Validate reference E1-E6 pixel pairing on the six normal pilot images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from run_gradient_canny_pilot import (
    PILOT_IMAGES,
    PROJECT_ROOT,
    _load_roi_config,
    _write_png,
)

from ai_cd_metrology.image_io import load_image
from ai_cd_metrology.metrology.gradient_canny import (
    DarkBandPairingDiagnostic,
    PixelPairingCandidate,
    SignedGradientProfile,
    build_pixel_pairing_candidates,
    characterize_signed_x_gradient,
    extract_edge_diagnostics,
    pair_dark_band_doublets,
)
from ai_cd_metrology.roi import crop_normalized_roi
from ai_cd_metrology.schemas import ImageRecord


NEGATIVE_COLOR = (255, 255, 0)
POSITIVE_COLOR = (0, 0, 255)
REJECTED_COLOR = (255, 0, 255)
OUTER_COLOR = (0, 220, 0)
INNER_COLOR = (0, 220, 255)
GAP_COLOR = (255, 120, 0)
DISPLAY_ROI_HEIGHT = 500
PROFILE_HEIGHT = 400


def _put_outlined_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_segment(
    image: np.ndarray,
    x0: int,
    x1: int,
    y: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    cv2.line(image, (x0, y), (x1, y), color, 3, cv2.LINE_AA)
    cv2.line(image, (x0, y - 9), (x0, y + 9), color, 2, cv2.LINE_AA)
    cv2.line(image, (x1, y - 9), (x1, y + 9), color, 2, cv2.LINE_AA)
    _put_outlined_text(image, label, ((x0 + x1) // 2 - 55, y - 12), color)


def _pairing_overlay(
    grayscale: np.ndarray,
    pairing: DarkBandPairingDiagnostic,
    candidates: tuple[PixelPairingCandidate, ...],
) -> np.ndarray:
    width = grayscale.shape[1]
    resized = cv2.resize(
        grayscale,
        (width, DISPLAY_ROI_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    output = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    for doublet in pairing.doublets:
        cv2.line(
            output,
            (doublet.negative_peak_x, 0),
            (doublet.negative_peak_x, DISPLAY_ROI_HEIGHT - 1),
            NEGATIVE_COLOR,
            1,
        )
        cv2.line(
            output,
            (doublet.positive_peak_x, 0),
            (doublet.positive_peak_x, DISPLAY_ROI_HEIGHT - 1),
            POSITIVE_COLOR,
            1,
        )
        midpoint = (doublet.negative_peak_x + doublet.positive_peak_x) // 2
        cv2.line(
            output,
            (doublet.negative_peak_x, 18),
            (doublet.positive_peak_x, 18),
            (255, 255, 255),
            2,
        )
        _put_outlined_text(
            output,
            f"D{doublet.sequence_index}",
            (midpoint - 12, 42),
            (255, 255, 255),
            0.42,
        )

    for rejected in pairing.rejected_peaks:
        cv2.line(
            output,
            (rejected.x, 0),
            (rejected.x, DISPLAY_ROI_HEIGHT - 1),
            REJECTED_COLOR,
            3,
        )
        _put_outlined_text(
            output,
            f"REJECT x={rejected.x} {rejected.reason}",
            (max(5, rejected.x - 105), DISPLAY_ROI_HEIGHT - 18),
            REJECTED_COLOR,
            0.46,
        )

    if candidates:
        representative = candidates[0]
        _draw_segment(
            output,
            representative.e1_x,
            representative.e4_x,
            105,
            f"Outer {representative.outer_width_px}px",
            OUTER_COLOR,
        )
        _draw_segment(
            output,
            representative.e2_x,
            representative.e3_x,
            155,
            f"Inner {representative.inner_width_px}px",
            INNER_COLOR,
        )
        _draw_segment(
            output,
            representative.e4_x,
            representative.e5_x,
            205,
            f"Gap {representative.gap_px}px",
            GAP_COLOR,
        )
    return output


def _profile_plot(
    profile: SignedGradientProfile,
    pairing: DarkBandPairingDiagnostic,
) -> np.ndarray:
    values = profile.values
    width = values.size
    height = PROFILE_HEIGHT
    top = 35
    bottom = 42
    center_y = (top + height - bottom) // 2
    amplitude = (height - top - bottom) / 2
    max_absolute = max(float(np.max(np.abs(values))), 1.0)
    plot = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.line(plot, (0, center_y), (width - 1, center_y), (120, 120, 120), 1)

    points = np.column_stack(
        (
            np.arange(width, dtype=np.int32),
            np.rint(center_y - values * amplitude / max_absolute).astype(np.int32),
        )
    ).reshape((-1, 1, 2))
    cv2.polylines(plot, [points], False, (235, 235, 235), 2, cv2.LINE_AA)

    for doublet in pairing.doublets:
        for x, color in (
            (doublet.negative_peak_x, NEGATIVE_COLOR),
            (doublet.positive_peak_x, POSITIVE_COLOR),
        ):
            y = round(center_y - float(values[x]) * amplitude / max_absolute)
            cv2.circle(plot, (x, y), 5, color, -1, cv2.LINE_AA)
    for rejected in pairing.rejected_peaks:
        x = rejected.x
        y = round(center_y - float(values[x]) * amplitude / max_absolute)
        cv2.drawMarker(
            plot,
            (x, y),
            REJECTED_COLOR,
            cv2.MARKER_TILTED_CROSS,
            18,
            3,
            cv2.LINE_AA,
        )

    cv2.putText(
        plot,
        "cyan=accepted negative; red=accepted positive; magenta X=rejected peak",
        (15, height - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return plot


def _format_values(
    candidates: tuple[PixelPairingCandidate, ...],
    attribute: str,
) -> str:
    return ", ".join(str(getattr(candidate, attribute)) for candidate in candidates)


def _build_diagnostic(
    filename: str,
    pattern_position: str,
    grayscale: np.ndarray,
    profile: SignedGradientProfile,
    pairing: DarkBandPairingDiagnostic,
    candidates: tuple[PixelPairingCandidate, ...],
) -> np.ndarray:
    overlay = _pairing_overlay(grayscale, pairing, candidates)
    profile_plot = _profile_plot(profile, pairing)
    header_height = 145
    panel_header = 38
    width = grayscale.shape[1]
    output = np.full(
        (
            header_height
            + panel_header
            + overlay.shape[0]
            + panel_header
            + profile_plot.shape[0],
            width,
            3,
        ),
        18,
        dtype=np.uint8,
    )

    header_lines = (
        f"{filename} | position={pattern_position} | pixel-only pairing diagnostic",
        (
            f"peaks: negative={len(profile.negative_peak_indices)}, "
            f"positive={len(profile.positive_peak_indices)} | "
            f"accepted doublets={len(pairing.doublets)} | "
            f"rejected peaks={len(pairing.rejected_peaks)} | "
            f"valid candidates={len(candidates)}"
        ),
        f"Outer px: [{_format_values(candidates, 'outer_width_px')}]",
        f"Inner px: [{_format_values(candidates, 'inner_width_px')}]",
        f"Gap px: [{_format_values(candidates, 'gap_px')}]",
    )
    for row, line in enumerate(header_lines):
        cv2.putText(
            output,
            line,
            (15, 25 + row * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    y = header_height
    cv2.putText(
        output,
        "Accepted dark-band doublets and first valid E1-E6 candidate",
        (15, y + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    y += panel_header
    output[y : y + overlay.shape[0]] = overlay
    y += overlay.shape[0]
    cv2.putText(
        output,
        "Median-y signed Sobel-x profile",
        (15, y + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    y += panel_header
    output[y : y + profile_plot.shape[0]] = profile_plot
    return output


def _mean_std(values: list[int]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


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
        default=Path("data/derived/pairing_pilot"),
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

    csv_path = output_dir / "pairing_pilot_candidates.csv"
    csv_rows: list[dict[str, str | int]] = []
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
        diagnostics = extract_edge_diagnostics(cropped)
        profile = characterize_signed_x_gradient(diagnostics.x_gradient)
        pairing = pair_dark_band_doublets(profile)
        candidates = build_pixel_pairing_candidates(pairing)

        diagnostic_image = _build_diagnostic(
            filename,
            pattern_position,
            diagnostics.grayscale_roi,
            profile,
            pairing,
            candidates,
        )
        output_path = output_dir / f"{path.stem}_pairing_pilot.png"
        _write_png(output_path, diagnostic_image)

        for candidate in candidates:
            csv_rows.append(
                {
                    "image": filename,
                    "candidate_index": candidate.sequence_index,
                    "e1_x": candidate.e1_x,
                    "e2_x": candidate.e2_x,
                    "e3_x": candidate.e3_x,
                    "e4_x": candidate.e4_x,
                    "e5_x": candidate.e5_x,
                    "e6_x": candidate.e6_x,
                    "outer_width_px": candidate.outer_width_px,
                    "inner_width_px": candidate.inner_width_px,
                    "gap_px": candidate.gap_px,
                }
            )

        outer = [candidate.outer_width_px for candidate in candidates]
        inner = [candidate.inner_width_px for candidate in candidates]
        gap = [candidate.gap_px for candidate in candidates]
        rejected = [
            f"{peak.x}:{peak.polarity}:{peak.reason}"
            for peak in pairing.rejected_peaks
        ]
        print(
            f"{filename} | neg={len(profile.negative_peak_indices)} "
            f"pos={len(profile.positive_peak_indices)} "
            f"doublets={len(pairing.doublets)} rejected={rejected} "
            f"candidates={len(candidates)}"
        )
        print(f"  outer={outer} mean/std={_mean_std(outer)}")
        print(f"  inner={inner} mean/std={_mean_std(inner)}")
        print(f"  gap={gap} mean/std={_mean_std(gap)}")
        print(f"  diagnostic={output_path}")

    fieldnames = [
        "image",
        "candidate_index",
        "e1_x",
        "e2_x",
        "e3_x",
        "e4_x",
        "e5_x",
        "e6_x",
        "outer_width_px",
        "inner_width_px",
        "gap_px",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"candidate_csv={csv_path}")


if __name__ == "__main__":
    main()
