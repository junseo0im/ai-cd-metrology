"""U-Net을 이용해 선폭을 측정하는 '돌리는' 파일: 모델 정의부터 단일/배치 측정까지 전부 포함한다.

Pipeline:
    capture image -> shared.load_image/native_arr -> U-Net probability
    -> minimal component filtering -> mask runs at selected rows
    -> left/right edges -> pixel width -> provisional micrometre width
    -> JSON report and annotated overlay

구성 (옛 파일 -> 이 파일 안의 절):
    1) unet_model.py       -> U-Net 모델 정의
    2) unet_runtime.py     -> 체크포인트 로드 및 타일 추론
    3) unet_postprocess.py -> 마스크 후처리(작은 성분 제거)
    4) unet_measure.py     -> 단일 이미지 측정 (measure_unet_image, main_single)
    5) batch_measure.py    -> 배치 측정 (main_batch)

이 폴더(unet_inference.py, shared.py, team_protocol.py, models/)만 있으면 어디서든 실행된다:
    python unet_inference.py single "사진.png"
    python unet_inference.py batch "사진폴더" --measurement-type 3 --device cpu

모델은 models/best_unet_manual.pt를 쓰며, --checkpoint 또는 환경변수
LINEWIDTH_UNET_CHECKPOINT로 바꿀 수 있다. 결과는 기본적으로 이 폴더의 outputs/에 저장된다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Import NumPy before PyTorch for the current Anaconda OpenMP runtime.
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from torch import nn

UNET_DIR = Path(__file__).resolve().parent
if str(UNET_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_DIR))

from shared import (
    LineDetection,
    build_width_measurements_by_type,
    compute_bands,
    default_measurement_rows,
    extract_row_profile,
    load_image,
    native_x_to_capture,
    native_y_to_capture,
    normalize_measurement_types,
    px_to_um,
    render_measurement_overlay,
    resolve_calibration,
    summarize_measurement_widths,
)
from team_protocol import TARGET_PREFIX, TARGETS_PER_ROW, Y_FRACTIONS, save_prediction_csv

_configured_checkpoint = os.environ.get("LINEWIDTH_UNET_CHECKPOINT")
DEFAULT_CHECKPOINT = (
    Path(_configured_checkpoint).expanduser()
    if _configured_checkpoint
    else UNET_DIR / "models" / "best_unet_manual.pt"
)
OUTPUTS_ROOT = UNET_DIR / "outputs"


# ---------------------------------------------------------------------------
# 1. U-Net 모델 정의 (옛 unet_model.py)
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Binary U-Net. Returns raw logits; sigmoid is applied outside the model."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        b = base_channels

        self.input_block = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)

        self.up1 = Up(b * 16, b * 8, b * 8)
        self.up2 = Up(b * 8, b * 4, b * 4)
        self.up3 = Up(b * 4, b * 2, b * 2)
        self.up4 = Up(b * 2, b, b)
        self.output = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.output(x)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


# ---------------------------------------------------------------------------
# 2. 체크포인트 로드 및 타일 추론 (옛 unet_runtime.py)
# ---------------------------------------------------------------------------

PATCH_SIZE = 256
STRIDE = 128
DEFAULT_TILE_BATCH_SIZE = 32


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_unet(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: torch.device | None = None,
) -> tuple[UNet, dict, torch.device]:
    """Load a checkpoint with its recorded architecture settings."""
    selected_device = device or resolve_device()
    checkpoint = torch.load(checkpoint_path, map_location=selected_device)
    model = UNet(
        in_channels=int(checkpoint.get("in_channels", 3)),
        out_channels=int(checkpoint.get("out_channels", 1)),
        base_channels=int(checkpoint["base_channels"]),
    ).to(selected_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # Convolution-heavy U-Net inference is faster in channels-last memory
    # layout on the tested CPU. This does not alter model weights.
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, checkpoint, selected_device


def patch_starts(length: int) -> list[int]:
    if length < PATCH_SIZE:
        raise ValueError(f"Image dimension {length} is smaller than {PATCH_SIZE}")
    starts = list(range(0, length - PATCH_SIZE + 1, STRIDE))
    final = length - PATCH_SIZE
    if starts[-1] != final:
        starts.append(final)
    return starts


def predict_probability(
    model: UNet,
    image: np.ndarray,
    device: torch.device,
    tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
) -> np.ndarray:
    """Predict a full-resolution probability map with batched overlapping tiles."""
    if tile_batch_size < 1:
        raise ValueError("tile_batch_size must be at least 1")
    height, width = image.shape[:2]
    probability_sum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    normalized = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
    normalized /= 255.0
    coordinates = [
        (y, x)
        for y in patch_starts(height)
        for x in patch_starts(width)
    ]

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(coordinates), tile_batch_size):
            batch_coordinates = coordinates[start : start + tile_batch_size]
            batch = np.stack(
                [
                    normalized[:, y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                    for y, x in batch_coordinates
                ]
            )
            tensor = torch.from_numpy(batch).to(
                device,
                non_blocking=device.type == "cuda",
            )
            tensor = tensor.contiguous(memory_format=torch.channels_last)
            probabilities = torch.sigmoid(model(tensor))[:, 0].cpu().numpy()
            for (y, x), probability in zip(batch_coordinates, probabilities):
                probability_sum[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += probability
                weight_sum[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1.0

    return probability_sum / np.maximum(weight_sum, 1.0)


# ---------------------------------------------------------------------------
# 3. 마스크 후처리 (옛 unet_postprocess.py)
# ---------------------------------------------------------------------------


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 30,
    min_height_ratio: float = 0.10,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Remove connected foreground components that are too small or too short.

    Eight-neighbour connectivity keeps slightly slanted vertical lines as one
    component.  Retained components are copied without changing their pixels.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be HxW, got shape={mask.shape}")
    if min_area < 0:
        raise ValueError("min_area must be at least 0")
    if not 0.0 <= min_height_ratio <= 1.0:
        raise ValueError("min_height_ratio must be between 0 and 1")

    binary = mask.astype(bool, copy=False)
    structure = ndimage.generate_binary_structure(rank=2, connectivity=2)
    labels, component_count = ndimage.label(binary, structure=structure)
    objects = ndimage.find_objects(labels)
    minimum_height = max(1, math.ceil(binary.shape[0] * min_height_ratio))

    cleaned = np.zeros_like(binary)
    kept_components = 0
    removed_components = 0

    for component_id, component_slice in enumerate(objects, start=1):
        if component_slice is None:
            continue
        component = labels[component_slice] == component_id
        area = int(component.sum())
        height = int(component_slice[0].stop - component_slice[0].start)
        if area >= min_area and height >= minimum_height:
            target = cleaned[component_slice]
            target[component] = True
            kept_components += 1
        else:
            removed_components += 1

    statistics: dict[str, int | float] = {
        "component_count_before": int(component_count),
        "component_count_after": kept_components,
        "removed_components": removed_components,
        "removed_pixels": int(binary.sum() - cleaned.sum()),
        "min_area": int(min_area),
        "min_height_ratio": float(min_height_ratio),
        "min_height_pixels": minimum_height,
    }
    return cleaned, statistics


# ---------------------------------------------------------------------------
# 4. 단일 이미지 측정 (옛 unet_measure.py)
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = OUTPUTS_ROOT / "measurements"
CYAN = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:\\Windows\\Fonts\\malgun.ttf", size)
    except Exception:
        return ImageFont.load_default()


def lines_from_mask_at_row(
    mask: np.ndarray,
    probability: np.ndarray,
    y_native: int,
    row_band: int = 2,
    row_vote: float = 0.5,
    edge_threshold: float = 0.5,
    min_run_width: int = 1,
) -> list[LineDetection]:
    """Convert foreground runs near one y position into left/right line edges.

    A majority vote across neighbouring rows suppresses one-pixel protrusions
    without applying erosion or dilation to the full mask.
    """
    height, width = mask.shape
    if not 0 <= y_native < height:
        raise ValueError(f"y_native={y_native} is outside 0..{height - 1}")
    y0 = max(0, y_native - row_band)
    y1 = min(height, y_native + row_band + 1)
    profile = np.mean(mask[y0:y1], axis=0) >= row_vote
    probability_profile = np.mean(probability[y0:y1], axis=0)

    padded = np.pad(profile.astype(np.int8), (1, 1), mode="constant")
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    lines: list[LineDetection] = []
    for start, end in zip(starts, ends):
        run_width = int(end - start)
        if run_width < min_run_width:
            continue

        # Interpolate where the probability crosses the threshold.  The mask
        # still decides which components exist, while this step avoids widths
        # being restricted to whole-pixel multiples.
        if start > 0:
            p0 = float(probability_profile[start - 1])
            p1 = float(probability_profile[start])
            if p1 != p0:
                left_edge = (start - 1) + (edge_threshold - p0) / (p1 - p0)
            else:
                left_edge = start - 0.5
            left_edge = float(np.clip(left_edge, start - 1, start))
        else:
            left_edge = 0.0

        if end < width:
            p0 = float(probability_profile[end - 1])
            p1 = float(probability_profile[end])
            if p1 != p0:
                right_edge = (end - 1) + (edge_threshold - p0) / (p1 - p0)
            else:
                right_edge = end - 0.5
            right_edge = float(np.clip(right_edge, end - 1, end))
        else:
            right_edge = float(width)

        confidence = float(np.mean(probability[y0:y1, start:end]))
        lines.append(
            LineDetection(
                center_x=(left_edge + right_edge) / 2.0,
                left_edge_x=left_edge,
                right_edge_x=right_edge,
                prominence=confidence,
            )
        )
    return lines


def _default_rows(height: int) -> list[int]:
    """Backward-compatible name for the shared five-row rule."""
    return default_measurement_rows(height)


def _draw_dashed_horizontal(
    draw: ImageDraw.ImageDraw,
    y: float,
    width: int,
    dash: int = 12,
    gap: int = 8,
) -> None:
    x = 0
    while x < width:
        draw.line([(x, y), (min(width - 1, x + dash), y)], fill=WHITE, width=2)
        x += dash + gap


def _centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill=BLACK,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    draw.text((center_x - text_width / 2.0, y), text, font=font, fill=fill)


def render_width_overlay(
    capture_arr: np.ndarray,
    scale,
    rows: list[dict],
    calibration,
    checkpoint: dict,
) -> Image.Image:
    """Draw numbered lines and cyan width markers similar to the reference."""
    source = Image.fromarray(capture_arr).convert("RGB")
    top_margin = 42
    canvas = Image.new("RGB", (source.width, source.height + top_margin), WHITE)
    canvas.paste(source, (0, top_margin))
    draw = ImageDraw.Draw(canvas)

    title_font = _font(16)
    number_font = _font(15)
    value_font = _font(12)
    unit = "µm" if calibration.um_per_native_px is not None else "native px"
    title = (
        f"U-Net line width; checkpoint epoch {checkpoint.get('epoch', '?')}; "
        f"labels in {unit}; cyan = measured width"
    )
    _centered_text(draw, source.width / 2.0, 10, title, title_font)

    if rows:
        number_y = top_margin + max(28, round(source.height * 0.035))
        for index, line in enumerate(rows[0]["line_objects"], start=1):
            center = native_x_to_capture(line.center_x, scale)
            text = str(index)
            box = draw.textbbox((0, 0), text, font=number_font)
            tw = box[2] - box[0]
            th = box[3] - box[1]
            x0 = center - tw / 2.0 - 4
            y0 = number_y
            draw.rectangle((x0, y0, x0 + tw + 8, y0 + th + 6), fill=(255, 245, 240))
            _centered_text(draw, center, y0 + 2, text, number_font)

    for row in rows:
        y_capture = native_y_to_capture(row["y_native"], scale) + top_margin
        _draw_dashed_horizontal(draw, y_capture, source.width)
        for line, line_data in zip(row["line_objects"], row["lines"]):
            left = native_x_to_capture(line.left_edge_x, scale)
            right = native_x_to_capture(line.right_edge_x, scale)
            center = (left + right) / 2.0
            tick = 7
            draw.line([(left, y_capture), (right, y_capture)], fill=CYAN, width=3)
            draw.line([(left, y_capture - tick), (left, y_capture + tick)], fill=CYAN, width=2)
            draw.line([(right, y_capture - tick), (right, y_capture + tick)], fill=CYAN, width=2)
            value = line_data["width_um_provisional"]
            label = f"{value:.3f}" if value is not None else f"{line_data['width_px_native']:.2f}"
            _centered_text(draw, center, y_capture + 10, label, value_font)

    return canvas


def _line_summary(rows: list[dict], calibration) -> list[dict]:
    maximum = max((len(row["measurements"]) for row in rows), default=0)
    summary = []
    for index in range(maximum):
        widths = [
            row["measurements"][index]["width_px_native"]
            for row in rows
            if index < len(row["measurements"])
        ]
        mean_px = float(np.mean(widths))
        std_px = float(np.std(widths))
        mean_um = px_to_um(mean_px, calibration)
        std_um = px_to_um(std_px, calibration)
        summary.append(
            {
                "measurement_index": index + 1,
                "sample_count": len(widths),
                "mean_width_px_native": round(mean_px, 4),
                "std_width_px_native": round(std_px, 4),
                "mean_width_um_provisional": round(mean_um, 4) if mean_um is not None else None,
                "std_width_um_provisional": round(std_um, 4) if std_um is not None else None,
            }
        )
    return summary


def measure_unet_image(
    image_path: Path,
    model: torch.nn.Module,
    checkpoint: dict,
    device: torch.device,
    *,
    y_rows: list[int] | None = None,
    row_band: int = 2,
    row_vote: float = 0.5,
    threshold: float | None = None,
    min_area: int = 30,
    min_height_ratio: float = 0.50,
    side_margin_ratio: float = 0.03,
    measurement_types="line",
    tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
) -> dict:
    """Measure one image and return serializable report plus image artifacts."""
    total_started_at = time.perf_counter()
    kinds = normalize_measurement_types(measurement_types)
    selected_threshold = (
        float(threshold)
        if threshold is not None
        else float(checkpoint.get("threshold", 0.5))
    )
    loaded = load_image(str(image_path))
    inference_started_at = time.perf_counter()
    probability = predict_probability(
        model,
        loaded.native_arr,
        device,
        tile_batch_size=tile_batch_size,
    )
    inference_ms = (time.perf_counter() - inference_started_at) * 1000.0
    raw_mask = probability >= selected_threshold
    if not 0.0 <= side_margin_ratio < 0.5:
        raise ValueError("side_margin_ratio must be between 0 and 0.5")

    measurement_mask = raw_mask.copy()
    side_margin_pixels = int(round(measurement_mask.shape[1] * side_margin_ratio))
    side_removed_pixels = 0
    if side_margin_pixels > 0:
        side_removed_pixels = int(
            measurement_mask[:, :side_margin_pixels].sum()
            + measurement_mask[:, -side_margin_pixels:].sum()
        )
        measurement_mask[:, :side_margin_pixels] = False
        measurement_mask[:, -side_margin_pixels:] = False

    mask, postprocess_stats = remove_small_components(
        measurement_mask,
        min_area=min_area,
        min_height_ratio=min_height_ratio,
    )
    postprocess_stats["side_margin_ratio"] = float(side_margin_ratio)
    postprocess_stats["side_margin_pixels"] = side_margin_pixels
    postprocess_stats["side_removed_pixels"] = side_removed_pixels
    calibration = resolve_calibration(loaded.scale)
    selected_rows = y_rows if y_rows else _default_rows(mask.shape[0])

    report_rows: list[dict] = []
    for y_native in selected_rows:
        line_objects = lines_from_mask_at_row(
            mask,
            probability,
            y_native=y_native,
            row_band=row_band,
            row_vote=row_vote,
            edge_threshold=selected_threshold,
        )
        lines = []
        for index, line in enumerate(line_objects, start=1):
            width_um = px_to_um(line.thickness_px, calibration)
            lines.append(
                {
                    "line_index": index,
                    "center_x_native": round(line.center_x, 3),
                    "left_edge_x_native": round(line.left_edge_x, 3),
                    "right_edge_x_native": round(line.right_edge_x, 3),
                    "width_px_native": round(line.thickness_px, 3),
                    "width_um_provisional": round(width_um, 4) if width_um is not None else None,
                    "mean_probability": round(line.prominence, 4),
                }
            )
        profile = extract_row_profile(loaded.native_arr, int(y_native), band=row_band)
        band_objects = compute_bands(line_objects, profile=profile)
        bands = []
        for band in band_objects:
            width_um = px_to_um(band.width_px, calibration)
            item = band.to_dict()
            item["width_um_provisional"] = round(width_um, 4) if width_um is not None else None
            bands.append(item)
        measurements_by_type = build_width_measurements_by_type(
            line_objects, band_objects, kinds
        )
        for measurements in measurements_by_type.values():
            for item in measurements:
                width_um = px_to_um(item["width_px_native"], calibration)
                item["left_x_native"] = round(item["left_x_native"], 3)
                item["right_x_native"] = round(item["right_x_native"], 3)
                item["center_x_native"] = round(item["center_x_native"], 3)
                item["width_px_native"] = round(item["width_px_native"], 3)
                item["width_um_provisional"] = (
                    round(width_um, 4) if width_um is not None else None
                )
        measurements = [
            item for kind in kinds for item in measurements_by_type[kind]
        ]
        report_rows.append(
            {
                "y_native": int(y_native),
                "line_count": len(lines),
                "lines": lines,
                "bands": bands,
                "measurements": measurements,
                "measurements_by_type": measurements_by_type,
                "line_objects": line_objects,
            }
        )

    overlay = render_measurement_overlay(
        loaded.capture_arr,
        loaded.scale,
        report_rows,
        kinds,
        calibration,
        title_note=f"U-Net epoch {checkpoint.get('epoch', '?')}",
    )
    serializable_rows = [
        {key: value for key, value in row.items() if key != "line_objects"}
        for row in report_rows
    ]
    total_ms = (time.perf_counter() - total_started_at) * 1000.0
    report = {
        "method": "U-Net mask -> row majority vote -> connected runs -> left/right edges",
        "measurement_types": kinds,
        "measurement_definitions": {
            "line": "bright band between adjacent sidewall edges",
            "gap": "darker alternating band between adjacent sidewall edges",
            "sidewall": "thickness between the left and right edges of one orange boundary",
        },
        "measurement_rows": "native image height at 20%, 35%, 50%, 65%, and 80% unless --rows is supplied",
        "image": str(image_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_dice": checkpoint.get("val_dice"),
        "coordinate_system": "native_arr",
        "capture_size": {"width": loaded.capture_arr.shape[1], "height": loaded.capture_arr.shape[0]},
        "native_size": {"width": loaded.native_arr.shape[1], "height": loaded.native_arr.shape[0]},
        "scale": loaded.scale.to_dict(),
        "calibration": calibration.to_dict(),
        "threshold": selected_threshold,
        "row_band": row_band,
        "row_vote": row_vote,
        "runtime": {
            "device": str(device),
            "tile_batch_size": int(tile_batch_size),
            "inference_ms": round(inference_ms, 3),
            "measurement_ms_excluding_file_save": round(total_ms, 3),
        },
        "postprocessing": postprocess_stats,
        "rows": serializable_rows,
        "measurement_summary_across_rows": {
            kind: _line_summary(
                [
                    {**row, "measurements": row["measurements_by_type"][kind]}
                    for row in serializable_rows
                ],
                calibration,
            )
            for kind in kinds
        },
    }
    return {
        "loaded": loaded,
        "report_rows": report_rows,
        "report": report,
        "overlay": overlay,
        "raw_mask": raw_mask,
        "mask": mask,
        "probability": probability,
        "calibration": calibration,
    }


def main_single() -> None:
    """단일 이미지에 대해 U-Net 측정을 실행한다 (옛 unet_measure.main)."""
    parser = argparse.ArgumentParser(description="Measure line/gap/sidewall widths from a trained U-Net mask")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, nargs="*", default=None, help="Native y coordinates")
    parser.add_argument("--row-band", type=int, default=2, help="Use y +/- this many rows")
    parser.add_argument("--row-vote", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument(
        "--min-height-ratio",
        type=float,
        default=0.50,
        help="Measurement lines must span at least this fraction of image height.",
    )
    parser.add_argument(
        "--side-margin-ratio",
        type=float,
        default=0.03,
        help="Ignore ruler/frame components inside this fraction of each side.",
    )
    parser.add_argument(
        "--measurement-type",
        nargs="+",
        default=None,
        help="1=line, 2=gap, 3=sidewall. Multiple selections: 1 2 3 or 123.",
    )
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=DEFAULT_TILE_BATCH_SIZE,
        help="U-Net tiles per inference call. Lower this if memory is limited.",
    )
    args = parser.parse_args()

    choice = args.measurement_type
    while choice is None:
        value = input("측정 종류 [1=line, 2=gap, 3=sidewall]: ").strip()
        try:
            choice = normalize_measurement_types(value)
        except ValueError:
            print("1, 2, 3 중 하나 이상을 입력하세요. 예: 1 2 3")
    measurement_types = normalize_measurement_types(choice)

    torch.set_num_threads(args.threads)
    model, checkpoint, device = load_unet(args.checkpoint)
    measurement = measure_unet_image(
        args.image,
        model,
        checkpoint,
        device,
        y_rows=args.rows,
        row_band=args.row_band,
        row_vote=args.row_vote,
        threshold=args.threshold,
        min_area=args.min_area,
        min_height_ratio=args.min_height_ratio,
        side_margin_ratio=args.side_margin_ratio,
        measurement_types=measurement_types,
        tile_batch_size=args.tile_batch_size,
    )
    loaded = measurement["loaded"]
    report = measurement["report"]
    report["checkpoint"] = str(args.checkpoint)
    serializable_rows = report["rows"]
    overlay = measurement["overlay"]
    raw_mask = measurement["raw_mask"]
    mask = measurement["mask"]
    probability = measurement["probability"]
    calibration = measurement["calibration"]

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    kind_slug = "-".join(measurement_types)
    report_path = args.output / f"{stem}_unet_{kind_slug}_measurement.json"
    overlay_path = args.output / f"{stem}_unet_{kind_slug}_width_overlay.png"
    raw_mask_path = args.output / f"{stem}_unet_mask_raw.png"
    mask_path = args.output / f"{stem}_unet_mask.png"
    probability_path = args.output / f"{stem}_unet_probability.png"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    overlay.save(overlay_path)
    Image.fromarray(raw_mask.astype(np.uint8) * 255, mode="L").save(raw_mask_path)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    Image.fromarray(np.clip(probability * 255, 0, 255).astype(np.uint8), mode="L").save(probability_path)

    print("Device:", device)
    print("Coordinate system: native_arr")
    print("Capture size:", loaded.capture_arr.shape[1], "x", loaded.capture_arr.shape[0])
    print("Native size:", loaded.native_arr.shape[1], "x", loaded.native_arr.shape[0])
    print("Calibration:", calibration.status, calibration.um_per_native_px)
    print("Measurement types:", ", ".join(measurement_types))
    print(
        "Runtime:",
        f"inference={report['runtime']['inference_ms']:.1f} ms,",
        f"measurement={report['runtime']['measurement_ms_excluding_file_save']:.1f} ms,",
        f"tile_batch_size={report['runtime']['tile_batch_size']}",
    )
    for row in serializable_rows:
        for kind in measurement_types:
            widths = [
                item["width_um_provisional"]
                for item in row["measurements_by_type"][kind]
            ]
            print(f"y_native={row['y_native']}: {kind}s={len(widths)}, widths={widths}")
    print("Report:", report_path)
    print("Overlay:", overlay_path)
    print("Mask:", mask_path)


# ---------------------------------------------------------------------------
# 5. 배치 측정 (옛 batch_measure.py)
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_OUTPUT_ROOT = OUTPUTS_ROOT / "batch_measurements"


def _central_targets(items: list[dict], count: int = TARGETS_PER_ROW) -> list[dict]:
    """Return the middle features, sorted from left to right.

    This keeps partial features at the image edges out of the shared benchmark
    and gives every method the same target IDs (L1-L5, G1-G5, S1-S5).
    """
    ordered = sorted(items, key=lambda item: float(item["center_x_native"]))
    if len(ordered) <= count:
        return ordered
    start = (len(ordered) - count) // 2
    return ordered[start : start + count]


def _collect_images(inputs: list[Path], recursive: bool) -> list[Path]:
    images: list[Path] = []
    for entry in inputs:
        if entry.is_file():
            if entry.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image type: {entry}")
            images.append(entry.resolve())
        elif entry.is_dir():
            iterator = entry.rglob("*") if recursive else entry.iterdir()
            images.extend(
                item.resolve()
                for item in iterator
                if item.is_file()
                and item.suffix.lower() in IMAGE_SUFFIXES
                and not item.stem.lower().endswith("_i")
            )
        else:
            raise FileNotFoundError(entry)
    return sorted(set(images), key=lambda path: str(path).lower())


def _read_measurement_types(choice) -> list[str]:
    if choice is not None:
        return normalize_measurement_types(choice)
    while True:
        value = input("측정 종류 [1=line, 2=gap, 3=sidewall]: ").strip()
        try:
            return normalize_measurement_types(value)
        except ValueError:
            print("1, 2, 3 중 하나 이상을 입력하세요. 예: 1 2 3")


def _save_artifacts(measurement: dict, output_dir: Path, stem: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": output_dir / f"{stem}_measurement.json",
        "overlay": output_dir / f"{stem}_width_overlay.png",
        "raw_mask": output_dir / f"{stem}_mask_raw.png",
        "mask": output_dir / f"{stem}_mask.png",
        "probability": output_dir / f"{stem}_probability.png",
    }
    paths["report"].write_text(
        json.dumps(measurement["report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    measurement["overlay"].save(paths["overlay"])
    Image.fromarray(measurement["raw_mask"].astype(np.uint8) * 255, mode="L").save(
        paths["raw_mask"]
    )
    Image.fromarray(measurement["mask"].astype(np.uint8) * 255, mode="L").save(
        paths["mask"]
    )
    Image.fromarray(
        np.clip(measurement["probability"] * 255, 0, 255).astype(np.uint8),
        mode="L",
    ).save(paths["probability"])
    return {name: str(path) for name, path in paths.items()}


def main_batch() -> None:
    """여러 이미지에 대해 U-Net 측정을 실행한다 (옛 batch_measure.main).

    The number keys are reserved exclusively for measurement types:
    1=line, 2=gap, 3=sidewall.  They never select minimum/median/maximum lines.
    """
    parser = argparse.ArgumentParser(
        description="Measure all stable line/gap/sidewall features in a batch"
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="One folder containing images, or individual image paths.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search subfolders too.")
    parser.add_argument(
        "--image-id-root",
        type=Path,
        default=None,
        help="Use paths relative to this folder as unique image IDs.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=15,
        help="Require this many images. Use 0 to disable the count check.",
    )
    parser.add_argument(
        "--measurement-type",
        nargs="+",
        default=None,
        help="1=line, 2=gap, 3=sidewall. Multiple selections: 1 2 3 or 123.",
    )
    parser.add_argument(
        "--max-deviation-um",
        type=float,
        default=0.05,
        help="A line is usable when its width standard deviation is at or below this value.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        help="Required fraction of measurement rows containing the same line.",
    )
    parser.add_argument(
        "--center-tolerance-px",
        type=float,
        default=None,
        help="Maximum x-coordinate movement used to match the same line across rows.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use cpu for the cross-laptop accuracy protocol; auto is faster for normal runs.",
    )
    parser.add_argument(
        "--method-name",
        default="unet",
        help="Method label written to the standardized benchmark prediction CSV.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rows", type=int, nargs="*", default=None)
    parser.add_argument("--row-band", type=int, default=2)
    parser.add_argument("--row-vote", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--min-height-ratio", type=float, default=0.50)
    parser.add_argument("--side-margin-ratio", type=float, default=0.03)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--tile-batch-size",
        type=int,
        default=DEFAULT_TILE_BATCH_SIZE,
        help="U-Net tiles per inference call. Lower this if memory is limited.",
    )
    args = parser.parse_args()

    images = _collect_images(args.inputs, args.recursive)
    image_id_root = args.image_id_root
    if image_id_root is None and args.recursive and len(args.inputs) == 1 and args.inputs[0].is_dir():
        image_id_root = args.inputs[0]
    if args.expected_count < 0:
        parser.error("--expected-count must be zero or a positive integer")
    if args.expected_count and len(images) != args.expected_count:
        parser.error(
            f"Expected {args.expected_count} images, but found {len(images)}. "
            "Use --expected-count 0 only for a trial run."
        )
    if not images:
        parser.error("No image files were found.")
    if args.max_deviation_um < 0:
        parser.error("--max-deviation-um must be non-negative")

    measurement_types = _read_measurement_types(args.measurement_type)
    output_dir = (
        args.output
        if args.output is not None
        else DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    selected_device = None if args.device == "auto" else torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is not available")
    model, checkpoint, device = load_unet(args.checkpoint, device=selected_device)
    records: list[dict] = []
    prediction_rows: list[dict] = []

    print(
        f"Images: {len(images)} | measurements: {','.join(measurement_types)}"
    )
    print(
        f"Usable condition: std <= {args.max_deviation_um:.4f} um, "
        f"coverage >= {args.min_coverage:.0%}"
    )

    for index, image_path in enumerate(images, start=1):
        if image_id_root is None:
            image_id = image_path.stem
        else:
            relative = image_path.resolve().relative_to(image_id_root.resolve()).with_suffix("")
            image_id = "__".join(relative.parts)
        item_dir = output_dir / f"{index:02d}_{image_path.stem}"
        try:
            started_at = time.perf_counter()
            measurement = measure_unet_image(
                image_path,
                model,
                checkpoint,
                device,
                y_rows=args.rows,
                row_band=args.row_band,
                row_vote=args.row_vote,
                threshold=args.threshold,
                min_area=args.min_area,
                min_height_ratio=args.min_height_ratio,
                side_margin_ratio=args.side_margin_ratio,
                measurement_types=measurement_types,
                tile_batch_size=args.tile_batch_size,
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            # The first image warms up the model and is deliberately excluded
            # from speed statistics, while its width predictions are retained.
            benchmark_runtime_ms = None if index == 1 else round(elapsed_ms, 3)
            selections = {}
            image_records = []
            for kind in measurement_types:
                rows_for_kind = [
                    {
                        **row,
                        "measurements": row["measurements_by_type"][kind],
                    }
                    for row in measurement["report"]["rows"]
                ]
                measurement_statistics = summarize_measurement_widths(
                    rows_for_kind,
                    max_deviation_um=args.max_deviation_um,
                    min_coverage=args.min_coverage,
                    center_tolerance_px=args.center_tolerance_px,
                )
                usable_statistics = [
                    item for item in measurement_statistics if item.get("usable")
                ]
                selections[kind] = {
                    "measurement_type": kind,
                    "max_deviation_um": args.max_deviation_um,
                    "min_coverage": args.min_coverage,
                    "center_tolerance_px": args.center_tolerance_px,
                    "measurement_statistics": measurement_statistics,
                    "usable_measurements": usable_statistics,
                }
                for item in measurement_statistics:
                    image_records.append(
                        {
                            "index": index,
                            "image": str(image_path),
                            "status": "ok",
                            "measurement_type": kind,
                            "measurement_index": item["measurement_index"],
                            "mean_um": item["mean_width_um_provisional"],
                            "std_um": item["std_width_um_provisional"],
                            "mean_px": item["mean_width_px_native"],
                            "std_px": item["std_width_px_native"],
                            "coverage_ratio": item["coverage_ratio"],
                            "usable": item["usable"],
                            "rejection_reason": item["rejection_reason"],
                            "usable_measurement_count": len(usable_statistics),
                            "total_measurement_count": len(measurement_statistics),
                        }
                    )
                print(
                    f"[{index:02d}/{len(images)}] {image_path.name}: "
                    f"{kind} total {len(measurement_statistics)}, "
                    f"usable {len(usable_statistics)}"
                )

            native_height = int(measurement["report"]["native_size"]["height"])
            denominator = max(native_height - 1, 1)
            report_rows = measurement["report"]["rows"]
            for row_index, row in enumerate(report_rows):
                y_fraction = (
                    Y_FRACTIONS[row_index]
                    if args.rows is None and len(report_rows) == len(Y_FRACTIONS)
                    else round(float(row["y_native"]) / denominator, 6)
                )
                for kind in measurement_types:
                    targets = list(_central_targets(row["measurements_by_type"][kind]))
                    for target_number in range(1, TARGETS_PER_ROW + 1):
                        base = {
                            "image_id": image_id,
                            "measurement_type": kind,
                            "y_fraction": y_fraction,
                            "target_id": f"{TARGET_PREFIX[kind]}{target_number}",
                            "runtime_ms": benchmark_runtime_ms,
                        }
                        if target_number > len(targets):
                            prediction_rows.append(
                                {
                                    **base,
                                    "status": "error",
                                    "message": "U-Net returned fewer than five central targets",
                                }
                            )
                            continue
                        target = targets[target_number - 1]
                        width = float(target["width_px_native"])
                        center = float(target["center_x_native"])
                        left = float(target["left_x_native"])
                        right = float(target["right_x_native"])
                        valid = (
                            all(math.isfinite(value) for value in (width, center, left, right))
                            and width > 0
                            and right > left
                        )
                        if not valid:
                            prediction_rows.append(
                                {
                                    **base,
                                    "status": "error",
                                    "message": (
                                        "Invalid U-Net edge pair: "
                                        f"left={left}, right={right}, width={width}"
                                    ),
                                }
                            )
                            continue
                        prediction_rows.append(
                            {
                                **base,
                                "width_px": width,
                                "width_um": target["width_um_provisional"],
                                "center_x_px": center,
                                "left_x_px": left,
                                "right_x_px": right,
                                "status": "ok",
                            }
                        )
            measurement["report"]["measurement_selections"] = selections
            artifact_paths = _save_artifacts(measurement, item_dir, image_path.stem)
            for record in image_records:
                record["report"] = artifact_paths["report"]
                record["overlay"] = artifact_paths["overlay"]
                records.append(record)
        except Exception as error:
            print(f"[{index:02d}/{len(images)}] {image_path.name}: ERROR - {error}")
            for kind in measurement_types:
                for y_fraction in Y_FRACTIONS:
                    for target_number in range(1, TARGETS_PER_ROW + 1):
                        prediction_rows.append(
                            {
                                "image_id": image_id,
                                "measurement_type": kind,
                                "y_fraction": y_fraction,
                                "target_id": f"{TARGET_PREFIX[kind]}{target_number}",
                                "status": "error",
                                "message": str(error),
                            }
                        )
                records.append(
                    {
                        "index": index,
                        "image": str(image_path),
                        "status": "error",
                        "measurement_type": kind,
                        "measurement_index": None,
                        "mean_um": None,
                        "std_um": None,
                        "mean_px": None,
                        "std_px": None,
                        "coverage_ratio": None,
                        "usable": False,
                        "rejection_reason": str(error),
                        "usable_measurement_count": 0,
                        "total_measurement_count": 0,
                        "report": None,
                        "overlay": None,
                        "error": str(error),
                    }
                )

    usable_values_by_type = {
        kind: [
            float(record["mean_um"])
            for record in records
            if record["measurement_type"] == kind
            and record["status"] == "ok"
            and record.get("usable")
            and record["mean_um"] is not None
        ]
        for kind in measurement_types
    }
    summaries_by_type = {
        kind: {
            "image_success_count": len(
                {
                    record["index"]
                    for record in records
                    if record["measurement_type"] == kind
                    and record["status"] == "ok"
                    and record.get("usable")
                }
            ),
            "usable_measurement_count": len(values),
            "mean_um_across_usable_measurements": (
                round(float(np.mean(values)), 4) if values else None
            ),
            "std_um_across_usable_measurements": (
                round(float(np.std(values)), 4) if values else None
            ),
        }
        for kind, values in usable_values_by_type.items()
    }
    batch_summary = {
        "image_count": len(images),
        "measurement_types": measurement_types,
        "max_deviation_um": args.max_deviation_um,
        "min_coverage": args.min_coverage,
        "summaries_by_type": summaries_by_type,
        "records": records,
    }
    json_path = output_dir / "batch_measurements_summary.json"
    csv_path = output_dir / "batch_measurements_summary.csv"
    json_path.write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "index",
            "image",
            "status",
            "measurement_type",
            "measurement_index",
            "mean_um",
            "std_um",
            "mean_px",
            "std_px",
            "coverage_ratio",
            "usable",
            "rejection_reason",
            "usable_measurement_count",
            "total_measurement_count",
            "report",
            "overlay",
            "error",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    prediction_path = save_prediction_csv(
        output_dir / f"{args.method_name}_predictions.csv",
        method=args.method_name,
        rows=prediction_rows,
    )

    print()
    print("Batch summary:", json_path)
    print("CSV:", csv_path)
    print("Benchmark prediction CSV:", prediction_path)
    for kind in measurement_types:
        item = summaries_by_type[kind]
        if item["mean_um_across_usable_measurements"] is not None:
            print(
                f"{kind}: usable average "
                f"{item['mean_um_across_usable_measurements']:.4f} um "
                f"(std {item['std_um_across_usable_measurements']:.4f} um)"
            )


def main() -> None:
    commands = {"single": main_single, "batch": main_batch}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("사용법: python unet_inference.py <single|batch> [옵션...]")
        raise SystemExit(1)
    commands[sys.argv.pop(1)]()


if __name__ == "__main__":
    main()
