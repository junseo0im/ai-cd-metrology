"""UNet 선폭 측정 핵심: 모델 정의·로드, 타일 추론, 마스크 후처리, 측정 (measure_unet_image).

Pipeline:
    capture image -> shared.load_image/native_arr -> U-Net probability
    -> minimal component filtering -> mask runs at selected rows
    -> left/right edges (inner edge of each sidewall band) -> line/gap widths

대시보드 패키지용 축약본이다. 실행은 run_example.py 또는 metrology/unet_metrology.py 로 한다.
(단일·배치 실행 명령, overlay·요약 생성 코드는 팀 저장소 linewidth/methods/unet 에 있다.)
모델은 models/best_unet_manual.pt 이며 환경변수 LINEWIDTH_UNET_CHECKPOINT 로 바꿀 수 있다.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

# Import NumPy before PyTorch for the current Anaconda OpenMP runtime.
import numpy as np
import torch
from scipy import ndimage
from torch import nn

UNET_DIR = Path(__file__).resolve().parent
LINEWIDTH_ROOT = UNET_DIR.parents[1]  # linewidth/ (holds the shared team_protocol.py)
for _path in (LINEWIDTH_ROOT, UNET_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from shared import (  # noqa: E402
    LineDetection,
    LoadedImage,
    build_width_measurements_by_type,
    compute_bands,
    default_measurement_rows,
    extract_row_profile,
    load_image,
    normalize_measurement_types,
    px_to_um,
    resolve_calibration,
)

_configured_checkpoint = os.environ.get("LINEWIDTH_UNET_CHECKPOINT")
DEFAULT_CHECKPOINT = (
    Path(_configured_checkpoint).expanduser()
    if _configured_checkpoint
    else UNET_DIR / "models" / "best_unet_manual.pt"
)


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
    measurement_types=None,
    tile_batch_size: int = DEFAULT_TILE_BATCH_SIZE,
    loaded: LoadedImage | None = None,
) -> dict:
    """Measure one image and return serializable report plus image artifacts.

    Pass ``loaded`` (from shared.load_image) when the caller has already read
    the image, so the file is not decoded twice.
    """
    total_started_at = time.perf_counter()
    kinds = normalize_measurement_types(measurement_types)
    selected_threshold = (
        float(threshold)
        if threshold is not None
        else float(checkpoint.get("threshold", 0.5))
    )
    if loaded is None:
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
        "measurement_rows": "native image height at 25%, 45%, and 65% unless --rows is supplied",
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
    }
    return {
        "loaded": loaded,
        "report_rows": report_rows,
        "report": report,
        "raw_mask": raw_mask,
        "mask": mask,
        "probability": probability,
        "calibration": calibration,
    }


# ---------------------------------------------------------------------------
# 5. 배치 측정 (옛 batch_measure.py)
# ---------------------------------------------------------------------------


