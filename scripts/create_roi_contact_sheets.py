"""Create read-only ROI review contact sheets from selected raw images."""

from __future__ import annotations

import argparse
import math
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd


PASS_SAMPLES = [
    "images/images/PASS_normal/w1_32_c.png",
    "images/images/PASS_normal/w1_31_c.png",
    "images/images/PASS_normal/w1_36_lt.png",
    "images/images/PASS_normal/w1_32_lt.png",
    "images/images/PASS_normal/w1_38_rb.png",
    "images/images/PASS_normal/w1_11_rb.png",
    "images/images/PASS_normal/w2_32_c.png",
    "images/images/PASS_normal/w2_26_c.png",
    "images/images/PASS_normal/w2_35_lt.png",
    "images/images/PASS_normal/w2_23_lt.png",
    "images/images/PASS_normal/w2_22_rb.png",
    "images/images/PASS_normal/w2_35_rb.png",
    "images/images/PASS_normal/w3_33_c.png",
    "images/images/PASS_normal/w3_26_c.png",
    "images/images/PASS_normal/w3_15_lt.png",
    "images/images/PASS_normal/w3_16_lt.png",
    "images/images/PASS_normal/w3_37_rb.png",
    "images/images/PASS_normal/w3_11_rb.png",
]

ROBUSTNESS_SAMPLES = [
    "images/images/WARNING_nc/w2_16_lt_nc.png",
    "images/images/FAIL/particle/w1_18_c_p.png",
    "images/images/FAIL/pattern_narrowing/w3_31_c_pn.png",
    "images/images/FAIL/pattern_die/w2_18_rb_pd.png",
    "images/images/FAIL/open/open.png",
    "images/images/FAIL/short/w2_35_c_s.png",
    "웨이퍼 사진/노광 70/wafer1/w1_25_c.png",
]

TILE_WIDTH = 660
TILE_HEIGHT = 470
IMAGE_BOX_WIDTH = 600
IMAGE_BOX_HEIGHT = 330
GRID_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def _write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"OpenCV could not encode contact sheet: {path}")
    encoded.tofile(path)


def _display_value(value: object) -> str:
    return "unknown" if pd.isna(value) or value in (None, "") else str(value)


def _fit_image(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    scale = min(IMAGE_BOX_WIDTH / width, IMAGE_BOX_HEIGHT / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized, resized_width, resized_height


def _put_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.46,
    color: tuple[int, int, int] = (235, 235, 235),
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_normalized_guide(
    tile: np.ndarray,
    image_x: int,
    image_y: int,
    image_width: int,
    image_height: int,
) -> None:
    overlay = tile.copy()
    for fraction in GRID_FRACTIONS:
        x = image_x + round(fraction * (image_width - 1))
        y = image_y + round(fraction * (image_height - 1))
        cv2.line(
            overlay,
            (x, image_y),
            (x, image_y + image_height - 1),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            overlay,
            (image_x, y),
            (image_x + image_width - 1, y),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.55, tile, 0.45, 0.0, dst=tile)

    for fraction in GRID_FRACTIONS:
        label = f"{fraction:.2g}"
        x = image_x + round(fraction * (image_width - 1))
        y = image_y + round(fraction * (image_height - 1))
        label_x = min(max(x - 9, image_x + 2), image_x + image_width - 27)
        label_y = min(max(y + 5, image_y + 12), image_y + image_height - 3)
        _put_text(tile, f"x={label}", (label_x, image_y + 15), scale=0.32)
        _put_text(tile, f"y={label}", (image_x + 3, label_y), scale=0.32)


def _make_tile(raw_root: Path, row: pd.Series, sequence: int) -> np.ndarray:
    relative_path = str(row["relative_path"])
    image = _read_image(raw_root / PurePosixPath(relative_path))
    original_height, original_width = image.shape[:2]
    resized, resized_width, resized_height = _fit_image(image)

    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), 38, dtype=np.uint8)
    filename = PurePosixPath(relative_path).name
    _put_text(tile, f"{sequence:02d}. {filename}", (18, 27), scale=0.52)
    metadata = (
        f"{original_width}x{original_height} | wafer={_display_value(row['wafer_id'])} | "
        f"pos={_display_value(row['pattern_position'])}"
    )
    labels = (
        f"status={_display_value(row['status'])} | defect={_display_value(row['defect_label'])}"
    )
    _put_text(tile, metadata, (18, 53), scale=0.42)
    _put_text(tile, labels, (18, 77), scale=0.42)

    image_x = (TILE_WIDTH - resized_width) // 2
    image_y = 105 + (IMAGE_BOX_HEIGHT - resized_height) // 2
    tile[image_y : image_y + resized_height, image_x : image_x + resized_width] = resized
    _draw_normalized_guide(tile, image_x, image_y, resized_width, resized_height)
    return tile


def create_contact_sheet(
    raw_root: Path,
    audit: pd.DataFrame,
    relative_paths: list[str],
    *,
    columns: int = 3,
) -> np.ndarray:
    indexed_audit = audit.set_index("relative_path", drop=False)
    missing = [path for path in relative_paths if path not in indexed_audit.index]
    if missing:
        raise ValueError(f"Selected paths missing from audit CSV: {missing}")

    rows = math.ceil(len(relative_paths) / columns)
    sheet = np.full((rows * TILE_HEIGHT, columns * TILE_WIDTH, 3), 20, dtype=np.uint8)
    for index, relative_path in enumerate(relative_paths):
        tile = _make_tile(raw_root, indexed_audit.loc[relative_path], index + 1)
        row, column = divmod(index, columns)
        y = row * TILE_HEIGHT
        x = column * TILE_WIDTH
        sheet[y : y + TILE_HEIGHT, x : x + TILE_WIDTH] = tile
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("data/derived/audit/image_audit.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/roi_pilot"),
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("output-dir must not be inside data/raw")

    audit = pd.read_csv(args.audit_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "PASS_normal": (
            PASS_SAMPLES,
            output_dir / "pass_normal_roi_comparison.png",
        ),
        "WARNING_FAIL_UI": (
            ROBUSTNESS_SAMPLES,
            output_dir / "warning_fail_ui_robustness.png",
        ),
    }
    for label, (samples, output_path) in outputs.items():
        sheet = create_contact_sheet(raw_root, audit, samples)
        _write_png(output_path, sheet)
        print(f"{label}: {output_path}")


if __name__ == "__main__":
    main()

