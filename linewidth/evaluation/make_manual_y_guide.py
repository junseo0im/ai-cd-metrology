"""Create native-pixel Y guides and blank manual-reference CSV templates.

The generated guide is only for locating the common measurement rows.  Make
the actual ImageJ measurement on the untouched ``*_native.png`` image and
enter the left/right edge coordinates in the CSV template.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from project_paths import UNET_METHOD_DIR

if str(UNET_METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_METHOD_DIR))

from shared import load_image
from team_protocol import TARGET_PREFIX, TARGETS_PER_ROW, measurement_rows, parse_measurement_types


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
GUIDE_COLOR = (0, 255, 255)
# Vertical helpers for choosing the central targets; they show image geometry only,
# never any method's detected edges, so the manual reference stays independent.
CENTER_COLOR = (255, 0, 255)
SIDE_MARGIN_COLOR = (255, 255, 255)
SIDE_MARGIN_RATIO = 0.03
TEXT_COLOR = (255, 255, 255)
LABEL_BACKGROUND = (0, 0, 0)


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    images = sorted(
        path for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images in: {input_path}")
    return images


def draw_dashed_vertical(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    height: int,
    dash: int,
    gap: int,
    color: tuple[int, int, int],
) -> None:
    y = 0
    while y < height:
        draw.line((x, y, x, min(y + dash - 1, height - 1)), fill=color, width=1)
        y += dash + gap


def draw_dashed_horizontal(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    width: int,
    dash: int,
    gap: int,
    color: tuple[int, int, int],
) -> None:
    x = 0
    while x < width:
        draw.line((x, y, min(x + dash - 1, width - 1), y), fill=color, width=1)
        x += dash + gap


def write_rows_csv(path: Path, image_name: str, rows: list[tuple[float, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_name", "y_fraction", "y_percent", "y_px_native"),
        )
        writer.writeheader()
        for fraction, y_px in rows:
            writer.writerow(
                {
                    "image_name": image_name,
                    "y_fraction": f"{fraction:.2f}",
                    "y_percent": int(round(fraction * 100)),
                    "y_px_native": y_px,
                }
            )


def write_template_csv(
    path: Path,
    image_name: str,
    rows: list[tuple[float, int]],
    target_count: int,
    measurement_types: list[str],
) -> None:
    fields = (
        "image_name",
        "measurement_type",
        "y_fraction",
        "y_px_native",
        "target_id",
        "left_edge_x_native",
        "right_edge_x_native",
        "width_px_native",
        "notes",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for kind in measurement_types:
            for fraction, y_px in rows:
                for target_index in range(1, target_count + 1):
                    writer.writerow(
                        {
                            "image_name": image_name,
                            "measurement_type": kind,
                            "y_fraction": f"{fraction:.2f}",
                            "y_px_native": y_px,
                            "target_id": f"{TARGET_PREFIX[kind]}{target_index}",
                            "left_edge_x_native": "",
                            "right_edge_x_native": "",
                            "width_px_native": "",
                            "notes": "",
                        }
                    )


def process_image(
    image_path: Path,
    output_dir: Path,
    target_count: int,
    measurement_types: list[str],
    write_template: bool = True,
) -> None:
    loaded = load_image(str(image_path))
    native = Image.fromarray(loaded.native_arr, mode="RGB")
    rows = measurement_rows(native.height)

    stem = image_path.stem
    native_path = output_dir / f"{stem}_native.png"
    guide_path = output_dir / f"{stem}_y_guide.png"
    rows_path = output_dir / f"{stem}_y_rows.csv"
    template_path = output_dir / f"{stem}_manual_reference_template.csv"

    native.save(native_path)
    guide = native.copy()
    draw = ImageDraw.Draw(guide)
    font = ImageFont.load_default()
    dash = max(6, native.width // 115)
    gap = max(4, dash // 2)

    margin = max(3, int(round(native.width * SIDE_MARGIN_RATIO)))
    for x, color in (
        (native.width // 2, CENTER_COLOR),
        (margin, SIDE_MARGIN_COLOR),
        (native.width - 1 - margin, SIDE_MARGIN_COLOR),
    ):
        draw_dashed_vertical(draw, x=x, height=native.height, dash=dash, gap=gap, color=color)

    for fraction, y_px in rows:
        draw_dashed_horizontal(
            draw,
            y=y_px,
            width=native.width,
            dash=dash,
            gap=gap,
            color=GUIDE_COLOR,
        )
        label = f"Y{int(round(fraction * 100))}% = {y_px}px (native)"
        box = draw.textbbox((0, 0), label, font=font)
        label_width = box[2] - box[0]
        label_height = box[3] - box[1]
        top = max(0, min(native.height - label_height - 4, y_px - label_height - 5))
        draw.rectangle((2, top, label_width + 8, top + label_height + 4), fill=LABEL_BACKGROUND)
        draw.text((5, top + 2), label, fill=TEXT_COLOR, font=font)

    guide.save(guide_path)
    write_rows_csv(rows_path, image_path.name, rows)
    if write_template:
        write_template_csv(template_path, image_path.name, rows, target_count, measurement_types)

    print(f"image={image_path.name}")
    print(
        f"capture={loaded.capture_arr.shape[1]}x{loaded.capture_arr.shape[0]}; "
        f"native={native.width}x{native.height}; "
        f"duplicate_factor={loaded.scale.factor_x}x{loaded.scale.factor_y}"
    )
    print("rows=" + ", ".join(f"{int(f*100)}%:{y}px" for f, y in rows))
    print(f"guide={guide_path}")
    print(f"native={native_path}")
    if write_template:
        print(f"template={template_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create dashed native-Y guides and manual line/gap/sidewall CSV templates."
    )
    parser.add_argument("input", type=Path, help="One image or a folder of images")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--targets", type=int, default=TARGETS_PER_ROW, help="Manual targets per Y row")
    parser.add_argument(
        "--measurement-types", nargs="+", default=["line", "gap"],
        help="line gap sidewall 중 이름으로 지정 (default: line gap)",
    )
    parser.add_argument(
        "--no-template", action="store_true",
        help="Skip per-image CSV templates (e.g. when filling evaluate.py's ground-truth CSV)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.targets <= 0:
        raise ValueError("--targets must be positive")
    output_dir = args.output or (
        Path(__file__).resolve().parents[2]
        / "analysis"
        / "testing"
        / "manual_reference_guides"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    measurement_types = parse_measurement_types(args.measurement_types)
    for image_path in collect_images(args.input):
        if image_path.stem.lower().endswith("_i"):
            continue  # annotated reference screenshots are not measurement inputs
        process_image(
            image_path, output_dir, args.targets, measurement_types, not args.no_template
        )
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
