"""Generate six ROI edge-extraction diagnostics without measuring width or gap."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_cd_metrology.image_io import load_image  # noqa: E402
from ai_cd_metrology.metrology.gradient_canny import (  # noqa: E402
    PILOT_CANNY_THRESHOLD_HIGH,
    PILOT_CANNY_THRESHOLD_LOW,
    PILOT_GAUSSIAN_KERNEL_SIZE,
    PILOT_GAUSSIAN_SIGMA,
    PILOT_SOBEL_KERNEL_SIZE,
    extract_edge_diagnostics,
)
from ai_cd_metrology.roi import NormalizedROI, crop_normalized_roi  # noqa: E402
from ai_cd_metrology.schemas import ImageRecord  # noqa: E402


PILOT_IMAGES = [
    ("w1_32_c.png", "w1", "32", "center"),
    ("w2_32_c.png", "w2", "32", "center"),
    ("w1_36_lt.png", "w1", "36", "left-top"),
    ("w2_35_lt.png", "w2", "35", "left-top"),
    ("w1_38_rb.png", "w1", "38", "right-bottom"),
    ("w2_22_rb.png", "w2", "22", "right-bottom"),
]

PANEL_WIDTH = 900
PANEL_HEIGHT = 440
PANEL_LABEL_HEIGHT = 42


def _load_roi_config(config_path: Path, pattern_position: str) -> NormalizedROI:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    roi_config = config["metrology"]["roi"]
    if roi_config["strategy"] != "position_specific_normalized":
        raise ValueError("pilot requires position_specific_normalized ROI strategy")
    if roi_config["coordinate_space"] != "normalized":
        raise ValueError("pilot requires normalized ROI coordinates")

    common_x = roi_config["common_x_range"]
    position_y = roi_config["by_pattern_position"][pattern_position]
    return NormalizedROI(
        x_min=float(common_x["x_min"]),
        y_min=float(position_y["y_min"]),
        x_max=float(common_x["x_max"]),
        y_max=float(position_y["y_max"]),
    )


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    bgr = _to_bgr(image)
    available_height = PANEL_HEIGHT - PANEL_LABEL_HEIGHT
    scale = min(PANEL_WIDTH / bgr.shape[1], available_height / bgr.shape[0])
    width = max(1, round(bgr.shape[1] * scale))
    height = max(1, round(bgr.shape[0] * scale))
    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 24, dtype=np.uint8)
    x = (PANEL_WIDTH - width) // 2
    y = PANEL_LABEL_HEIGHT + (available_height - height) // 2
    panel[y : y + height, x : x + width] = resized
    cv2.putText(
        panel,
        label,
        (15, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return panel


def _build_diagnostic(
    filename: str,
    pattern_position: str,
    roi: NormalizedROI,
    bounds: tuple[int, int, int, int],
    original_roi: np.ndarray,
    grayscale: np.ndarray,
    x_gradient_visualization: np.ndarray,
    canny_edges: np.ndarray,
) -> np.ndarray:
    panels = [
        _labeled_panel(original_roi, "Original ROI"),
        _labeled_panel(grayscale, "Grayscale ROI"),
        _labeled_panel(x_gradient_visualization, "Signed x-gradient (Sobel)"),
        _labeled_panel(
            canny_edges,
            f"Canny edges ({PILOT_CANNY_THRESHOLD_LOW}/{PILOT_CANNY_THRESHOLD_HIGH})",
        ),
    ]
    grid = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    header_height = 72
    output = np.full((grid.shape[0] + header_height, grid.shape[1], 3), 18, dtype=np.uint8)
    output[header_height:] = grid
    title = (
        f"{filename} | position={pattern_position} | "
        f"normalized ROI=({roi.x_min:.2f},{roi.y_min:.2f})-({roi.x_max:.2f},{roi.y_max:.2f})"
    )
    pixel_bounds = f"pixel bounds (half-open): x=[{bounds[0]},{bounds[2]}), y=[{bounds[1]},{bounds[3]})"
    cv2.putText(
        output,
        title,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        pixel_bounds,
        (16, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return output


def _write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"OpenCV could not encode diagnostic image: {path}")
    encoded.tofile(path)


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
        default=Path("data/derived/gradient_canny_pilot"),
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
        "baseline: "
        f"Gaussian={PILOT_GAUSSIAN_KERNEL_SIZE}x{PILOT_GAUSSIAN_KERNEL_SIZE}, "
        f"sigma={PILOT_GAUSSIAN_SIGMA}, Sobel-x={PILOT_SOBEL_KERNEL_SIZE}, "
        f"Canny={PILOT_CANNY_THRESHOLD_LOW}/{PILOT_CANNY_THRESHOLD_HIGH}, "
        "L2gradient=True"
    )
    for filename, wafer_id, die_id, pattern_position in PILOT_IMAGES:
        path = image_dir / filename
        record = ImageRecord(
            image_id=path.stem,
            file_path=path,
            wafer_id=wafer_id,
            die_id=die_id,
            pattern_position=pattern_position,
            capture_region="standard",
            defect_label="normal",
        )
        image = load_image(record)
        roi = _load_roi_config(args.config, pattern_position)
        cropped, bounds = crop_normalized_roi(image, roi)
        diagnostics = extract_edge_diagnostics(cropped)
        diagnostic_image = _build_diagnostic(
            filename,
            pattern_position,
            roi,
            bounds,
            cropped,
            diagnostics.grayscale_roi,
            diagnostics.x_gradient_visualization,
            diagnostics.canny_edges,
        )
        output_path = output_dir / f"{path.stem}_edge_diagnostic.png"
        _write_png(output_path, diagnostic_image)
        print(output_path)


if __name__ == "__main__":
    main()

