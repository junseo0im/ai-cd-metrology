"""Build a manual OptiView-to-gradient pairing reference diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_cd_metrology.metrology.gradient_canny import (  # noqa: E402
    SignedGradientProfile,
    characterize_signed_x_gradient,
    extract_edge_diagnostics,
)


# Image-specific manual inspection bounds. These do not change production ROI config.
ANALYSIS_Y_RANGE = (80, 360)
REFERENCE_X_RANGE = (800, 1140)
RULER_DETAIL_Y_RANGE = (400, 590)

# Approximate integer-pixel endpoints read directly from the embedded OptiView rulers.
MANUAL_RULER_ENDPOINTS = {
    "Outer 11.67 um": (840, 986),
    "Inner 10.39 um": (848, 978),
    "Gap 9.11 um": (986, 1100),
}

NEGATIVE_COLOR = (255, 255, 0)
POSITIVE_COLOR = (0, 0, 255)
RULER_COLORS = ((255, 0, 255), (0, 255, 0), (255, 180, 0))


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read reference image: {path}")
    return image


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"OpenCV could not encode diagnostic image: {path}")
    encoded.tofile(path)


def _candidate_edges(
    profile: SignedGradientProfile,
) -> list[tuple[int, str]]:
    x_min, x_max = REFERENCE_X_RANGE
    edges = [
        (x, "positive")
        for x in profile.positive_peak_indices
        if x_min <= x <= x_max
    ]
    edges.extend(
        (x, "negative")
        for x in profile.negative_peak_indices
        if x_min <= x <= x_max
    )
    edges.sort()
    if len(edges) != 6:
        raise ValueError(
            "reference interval must contain exactly six signed peak candidates"
        )
    return edges


def _draw_candidate_guides(
    image: np.ndarray,
    edges: list[tuple[int, str]],
    *,
    x_offset: int = 0,
) -> np.ndarray:
    output = _to_bgr(image)
    for number, (global_x, polarity) in enumerate(edges, start=1):
        x = global_x - x_offset
        color = POSITIVE_COLOR if polarity == "positive" else NEGATIVE_COLOR
        cv2.line(output, (x, 0), (x, output.shape[0] - 1), color, 2)
        sign = "+" if polarity == "positive" else "-"
        label = f"E{number}{sign} x={global_x}"
        label_x = x + 5 if polarity == "positive" else x - 78
        label_y = 57 if polarity == "positive" else 28
        cv2.putText(
            output,
            label,
            (max(2, label_x), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (15, 15, 15),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (max(2, label_x), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def _draw_manual_endpoints(detail: np.ndarray, y_offset: int) -> np.ndarray:
    output = detail.copy()
    annotation_rows = (459, 475, 523)
    for row, ((_, (x0, x1)), color) in enumerate(
        zip(MANUAL_RULER_ENDPOINTS.items(), RULER_COLORS, strict=True)
    ):
        y = annotation_rows[row] - y_offset
        cv2.circle(output, (x0, y), 7, color, 2, cv2.LINE_AA)
        cv2.circle(output, (x1, y), 7, color, 2, cv2.LINE_AA)
    return output


def _profile_plot(
    profile: SignedGradientProfile,
    edges: list[tuple[int, str]],
    width: int,
) -> np.ndarray:
    height = 330
    top = 40
    bottom = 45
    center_y = (top + height - bottom) // 2
    amplitude = (height - top - bottom) / 2
    values = profile.values
    max_absolute = max(float(np.max(np.abs(values))), 1.0)
    plot = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.line(plot, (0, center_y), (width - 1, center_y), (120, 120, 120), 1)

    points = np.column_stack(
        (
            np.arange(values.size, dtype=np.int32),
            np.rint(center_y - values * amplitude / max_absolute).astype(np.int32),
        )
    ).reshape((-1, 1, 2))
    cv2.polylines(plot, [points], False, (235, 235, 235), 2, cv2.LINE_AA)
    for number, (x, polarity) in enumerate(edges, start=1):
        color = POSITIVE_COLOR if polarity == "positive" else NEGATIVE_COLOR
        y = round(center_y - float(values[x]) * amplitude / max_absolute)
        cv2.line(plot, (x, top), (x, height - bottom), color, 1)
        cv2.circle(plot, (x, y), 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            plot,
            f"E{number}",
            (x - 10, max(22, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        plot,
        "median-y signed Sobel-x profile; candidates are unpaired labels only",
        (15, height - 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return plot


def _add_title(image: np.ndarray, title: str) -> np.ndarray:
    header = 42
    output = np.full((image.shape[0] + header, image.shape[1], 3), 18, dtype=np.uint8)
    output[header:] = _to_bgr(image)
    cv2.putText(
        output,
        title,
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/raw/optiview_ref_image.png"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/pairing_reference"),
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    expected_path = (PROJECT_ROOT / "data" / "raw" / "optiview_ref_image.png").resolve()
    if image_path != expected_path:
        raise ValueError("this diagnostic accepts only data/raw/optiview_ref_image.png")

    output_dir = args.output_dir.resolve()
    raw_root = (PROJECT_ROOT / "data" / "raw").resolve()
    try:
        output_dir.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside data/raw")

    image = _read_image(image_path)
    y0, y1 = ANALYSIS_Y_RANGE
    if not 0 <= y0 < y1 <= image.shape[0]:
        raise ValueError("manual analysis band is outside the reference image")
    optical_band = image[y0:y1].copy()
    diagnostics = extract_edge_diagnostics(optical_band)
    profile = characterize_signed_x_gradient(diagnostics.x_gradient)
    edges = _candidate_edges(profile)

    original = _to_bgr(image)
    cv2.rectangle(
        original,
        (0, y0),
        (image.shape[1] - 1, y1 - 1),
        (255, 255, 255),
        2,
    )

    analysis_overlay = _draw_candidate_guides(optical_band, edges)
    detail_y0, detail_y1 = RULER_DETAIL_Y_RANGE
    ruler_detail = _draw_candidate_guides(image[detail_y0:detail_y1], edges)
    ruler_detail = _draw_manual_endpoints(ruler_detail, detail_y0)
    profile_image = _profile_plot(profile, edges, image.shape[1])

    panels = [
        _add_title(
            original,
            "Original reference; embedded OptiView rulers retained; white box=y[80,360)",
        ),
        _add_title(
            analysis_overlay,
            "Annotation-free optical band; E1-E6 are signed-gradient candidates",
        ),
        _add_title(profile_image, "Signed-gradient 1D profile"),
        _add_title(
            ruler_detail,
            "Ruler detail; endpoint circles: magenta=Outer, green=Inner, orange=Gap",
        ),
    ]
    overview_path = output_dir / "optiview_pairing_reference_overview.png"
    detail_path = output_dir / "optiview_pairing_reference_detail.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_png(overview_path, panels[0])
    _write_png(detail_path, np.vstack(panels[1:]))

    print(f"reference={image_path}")
    print(f"shape={image.shape}, dtype={image.dtype}")
    print(f"analysis_y=[{y0},{y1})")
    for number, (x, polarity) in enumerate(edges, start=1):
        print(f"E{number}: x={x}, polarity={polarity}")
    for name, endpoints in MANUAL_RULER_ENDPOINTS.items():
        print(f"manual {name}: approximate endpoints={endpoints}")
    print(overview_path)
    print(detail_path)


if __name__ == "__main__":
    main()
