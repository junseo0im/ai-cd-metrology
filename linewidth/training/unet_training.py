"""U-Net 학습용 코드를 한 곳에 모은 파일: mask 생성부터 미세조정·보정까지 전체 파이프라인.

단계별 실행 (옛 파일 -> 이 파일의 함수):
    1) make_masks.py                     -> main_make_masks()      : 원본에서 초기 학습 mask 생성
    2) make_dataset.py                   -> main_make_dataset()    : 256x256 patch dataset 생성
    3) train_unet.py                     -> main_train()           : 베이스라인 U-Net 학습
    4) prepare_mask_review.py            -> main_prepare_mask_review()            : 사람 검수용 전체 mask 준비
    5) prepare_band_mask_review.py       -> main_prepare_band_review()            : 검수용 가로 band로 분할
    6) prepare_single_boundary_review.py -> main_prepare_single_boundary_review() : 한 경계선 추적 crop
    7) prepare_diverse_boundary_review.py-> main_prepare_diverse_boundary_review(): 이미지별 다양한 경계 crop
    8) combine_manual_reviews.py         -> main_combine_reviews() : 수정된 manifest 병합
    9) finetune_unet_manual.py           -> main_finetune()        : 수정된 crop으로 미세조정
    10) calibrate_unet_manual.py         -> main_calibrate()       : 확률 임계값 보정

직접 사람이 수정하는 GUI 도구는 별도 파일 training/mask_editor.py에 남아 있다
(위 4~7단계에서 만든 manifest를 --manifest로 열어서 사용한다).

명령행에서는 linewidth_tool 폴더에서 단계 이름을 첫 인자로 준다:
    python training/unet_training.py make-masks
    python training/unet_training.py make-dataset
    python training/unet_training.py train [--epochs ...]
    python training/unet_training.py prepare-mask-review 사진1.png 사진2.png
    python training/unet_training.py prepare-band-review
    python training/unet_training.py prepare-single-boundary-review
    python training/unet_training.py prepare-diverse-boundary-review
    python training/unet_training.py combine-reviews
    python training/unet_training.py finetune
    python training/unet_training.py calibrate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import (
    ACTIVE_UNET_CHECKPOINT,
    ANALYSIS_ROOT,
    BASELINE_MODEL_DIR,
    MANUAL_REVIEW_ROOT,
    TRAINING_SOURCE_ROOT,
    UNET_DATASET_ROOT,
    UNET_DATA_ROOT,
    UNET_METHOD_DIR,
)

if str(UNET_METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_METHOD_DIR))

from shared import detect_dark_lines, extract_row_profile, load_image


# ===========================================================================
# 1. 초기 학습 mask 생성 (옛 make_masks.py)
# ===========================================================================

MM_INPUT_FOLDERS = [
    TRAINING_SOURCE_ROOT / "lighting",
    TRAINING_SOURCE_ROOT / "labels",
]
MM_OUTPUT_BASE = UNET_DATA_ROOT
MM_IMAGE_OUT = MM_OUTPUT_BASE / "images"
MM_MASK_OUT = MM_OUTPUT_BASE / "masks"

# 화면 테두리나 잘린 패턴을 학습 정답으로 사용하지 않는다. 이미지를 실제로
# crop하지 않고 mask만 0으로 만들기 때문에 image/mask의 좌표와 크기는 유지된다.
MM_TOP_MARGIN = 8
MM_BOTTOM_MARGIN = 4
MM_SIDE_MARGIN = 8


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """가장자리 오검출과 1~2픽셀 돌출/틈을 정리한다."""
    cleaned = mask > 0

    # 1~2픽셀짜리 가로 돌출을 제거하고, 세로 방향의 작은 틈을 연결한다.
    cleaned = ndimage.binary_opening(
        cleaned,
        structure=np.ones((1, 3), dtype=bool),
    )
    cleaned = ndimage.binary_closing(
        cleaned,
        structure=np.ones((5, 1), dtype=bool),
    )
    cleaned = ndimage.median_filter(cleaned.astype(np.uint8), size=(5, 3)) > 0

    # 캡처 화면의 위쪽 표시선과 프레임 가장자리의 불완전한 패턴을 제외한다.
    cleaned[:MM_TOP_MARGIN, :] = False
    cleaned[-MM_BOTTOM_MARGIN:, :] = False
    cleaned[:, :MM_SIDE_MARGIN] = False
    cleaned[:, -MM_SIDE_MARGIN:] = False

    # 프레임 높이의 절반도 이어지지 않는 점/짧은 조각은 제거한다. 남은 세로선은
    # 행별 좌우 경계를 세로 방향 중앙값으로 다듬어 짧은 픽셀 돌출만 없앤다.
    component_labels, component_count = ndimage.label(cleaned)
    smoothed = np.zeros_like(cleaned)
    min_component_height = max(20, int(cleaned.shape[0] * 0.5))

    for component_id in range(1, component_count + 1):
        ys, xs = np.where(component_labels == component_id)
        if len(ys) == 0:
            continue

        y_min = int(ys.min())
        y_max = int(ys.max())
        if y_max - y_min + 1 < min_component_height:
            continue

        present_rows = np.unique(ys)
        left_by_row = np.array(
            [xs[ys == y].min() for y in present_rows],
            dtype=np.float64,
        )
        right_by_row = np.array(
            [xs[ys == y].max() for y in present_rows],
            dtype=np.float64,
        )

        full_rows = np.arange(y_min, y_max + 1)
        left_interp = np.interp(full_rows, present_rows, left_by_row)
        right_interp = np.interp(full_rows, present_rows, right_by_row)
        left_smooth = ndimage.median_filter(left_interp, size=15, mode="nearest")
        right_smooth = ndimage.median_filter(right_interp, size=15, mode="nearest")

        for y, left_x, right_x in zip(full_rows, left_smooth, right_smooth):
            x0 = max(MM_SIDE_MARGIN, int(round(left_x)))
            x1 = min(cleaned.shape[1] - MM_SIDE_MARGIN, int(round(right_x)) + 1)
            if x1 > x0:
                smoothed[y, x0:x1] = True

    cleaned = smoothed

    # 재구성 후에도 안전하게 가장자리 제외 조건을 다시 적용한다.
    cleaned[:MM_TOP_MARGIN, :] = False
    cleaned[-MM_BOTTOM_MARGIN:, :] = False
    cleaned[:, :MM_SIDE_MARGIN] = False
    cleaned[:, -MM_SIDE_MARGIN:] = False

    return cleaned.astype(np.uint8) * 255


def make_mask(image_path: Path) -> None:
    loaded = load_image(str(image_path))

    # 기존 계측 코드와 같은 native 좌표를 사용한다.
    image = loaded.native_arr
    height, width = image.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    # 4행 간격으로 경계선을 검출한다.
    row_step = 4

    for y in range(MM_TOP_MARGIN, height - MM_BOTTOM_MARGIN, row_step):
        profile = extract_row_profile(image, y, band=2)

        lines = detect_dark_lines(
            profile,
            smooth_sigma=0.8,
            min_prominence=12.0,
            max_width=25,
        )

        y0 = max(0, y - row_step // 2)
        y1 = min(height, y + row_step // 2 + 1)

        for line in lines:
            # Pixel centres that lie inside [left_edge, right_edge) become the
            # foreground target.  The previous floor/ceil+1 rasterisation
            # expanded every target line by about two native pixels, which
            # made U-Net widths roughly 0.32 µm larger after calibration.
            x0 = max(0, int(np.ceil(line.left_edge_x)))
            x1 = min(width, int(np.ceil(line.right_edge_x)))

            # 프레임에 걸쳐 잘린 선은 완전한 폭을 측정할 수 없으므로 제외한다.
            if x0 < MM_SIDE_MARGIN or x1 > width - MM_SIDE_MARGIN:
                continue

            mask[y0:y1, x0:x1] = 255

    mask = clean_mask(mask)

    # 서로 다른 폴더에 같은 파일명이 있을 수 있으므로 폴더명을 붙인다.
    output_name = f"{image_path.parent.name}_{image_path.stem}.png"

    image_path_out = MM_IMAGE_OUT / output_name
    mask_path_out = MM_MASK_OUT / output_name

    Image.fromarray(image).save(image_path_out)
    Image.fromarray(mask).save(mask_path_out)

    # 저장 후에도 입력과 mask가 정확히 같은 좌표 크기인지 강제로 검사한다.
    with Image.open(image_path_out) as saved_image, Image.open(mask_path_out) as saved_mask:
        if saved_image.size != saved_mask.size:
            raise RuntimeError(
                f"크기 불일치: {output_name}: "
                f"image={saved_image.size}, mask={saved_mask.size}"
            )

    white_ratio = float(np.mean(mask > 0)) * 100

    print(
        f"완료: {image_path.name} "
        f"| 크기={width}x{height} "
        f"| image/mask 동일=True "
        f"| 흰색 영역={white_ratio:.2f}%"
    )


def main_make_masks() -> None:
    MM_IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    MM_MASK_OUT.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for folder in MM_INPUT_FOLDERS:
        for path in sorted(folder.glob("*.png")):
            # 측정선·글자가 그려진 이미지는 학습에서 제외한다.
            if "표시" in path.stem:
                continue
            image_paths.append(path)

    print(f"대상 이미지: {len(image_paths)}장")

    for image_path in image_paths:
        make_mask(image_path)

    print()
    print("이미지 저장:", MM_IMAGE_OUT)
    print("Mask 저장:", MM_MASK_OUT)


# ===========================================================================
# 2. 256x256 patch dataset 생성 (옛 make_dataset.py)
# ===========================================================================

MD_INPUT_ROOT = UNET_DATA_ROOT
MD_OUTPUT_ROOT = UNET_DATASET_ROOT

MD_PATCH_SIZE = 256
MD_STRIDE = 128
MD_RANDOM_SEED = 20260918

MD_TRAIN_SOURCES = {
    "lighting_조명4.png",
    "lighting_조명5.png",
    "lighting_조명6.png",
    "labels_조명 4.png",
    "labels_조명 5.png",
    "labels_조명 6.png",
}

MD_VAL_SOURCES = {
    "lighting_조명7.png",
    "labels_조명 7.png",
}


def patch_starts(length: int) -> list[int]:
    """마지막 가장자리까지 포함하는 patch 시작 좌표를 반환한다."""
    if length < MD_PATCH_SIZE:
        raise ValueError(
            f"이미지 길이 {length}px가 patch 크기 {MD_PATCH_SIZE}px보다 작습니다. "
            "resize하지 않고 중단합니다."
        )

    starts = list(range(0, length - MD_PATCH_SIZE + 1, MD_STRIDE))
    final_start = length - MD_PATCH_SIZE
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def brightness(image: np.ndarray, factor: float) -> np.ndarray:
    result = image.astype(np.float32) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def contrast(image: np.ndarray, factor: float) -> np.ndarray:
    arr = image.astype(np.float32)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    result = (arr - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


def augmented_versions(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """선폭을 바꾸지 않는 train 전용 augmentation을 만든다."""
    noise = rng.normal(0.0, 3.0, size=image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return [
        ("orig", image, mask),
        ("hflip", np.ascontiguousarray(np.fliplr(image)), np.ascontiguousarray(np.fliplr(mask))),
        ("vflip", np.ascontiguousarray(np.flipud(image)), np.ascontiguousarray(np.flipud(mask))),
        ("bright090", brightness(image, 0.90), mask),
        ("bright110", brightness(image, 1.10), mask),
        ("contrast090", contrast(image, 0.90), mask),
        ("contrast110", contrast(image, 1.10), mask),
        ("noise", noisy, mask),
    ]


def prepare_output() -> None:
    """생성물 전용 폴더만 안전하게 초기화한다."""
    expected = UNET_DATASET_ROOT
    if MD_OUTPUT_ROOT.resolve() != expected.resolve() or MD_OUTPUT_ROOT.name != "unet_dataset":
        raise RuntimeError(f"안전하지 않은 출력 경로: {MD_OUTPUT_ROOT}")

    if MD_OUTPUT_ROOT.exists():
        shutil.rmtree(MD_OUTPUT_ROOT)

    for split in ("train", "val"):
        (MD_OUTPUT_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (MD_OUTPUT_ROOT / split / "masks").mkdir(parents=True, exist_ok=True)


def load_pair(source_name: str) -> tuple[np.ndarray, np.ndarray]:
    image_path = MD_INPUT_ROOT / "images" / source_name
    mask_path = MD_INPUT_ROOT / "masks" / source_name

    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"image-mask pair가 없습니다: {source_name}")

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"원본 크기 불일치: {source_name}: "
            f"image={image.shape[:2]}, mask={mask.shape}"
        )

    # mask를 확실한 이진값 0/255로 고정한다.
    mask = ((mask > 127).astype(np.uint8) * 255)
    return image, mask


def save_pair(split: str, name: str, image: np.ndarray, mask: np.ndarray) -> None:
    image_path = MD_OUTPUT_ROOT / split / "images" / name
    mask_path = MD_OUTPUT_ROOT / split / "masks" / name
    Image.fromarray(image, mode="RGB").save(image_path)
    Image.fromarray(mask, mode="L").save(mask_path)


def generate_split(
    split: str,
    source_names: set[str],
    rng: np.random.Generator,
) -> list[dict]:
    records: list[dict] = []

    for source_name in sorted(source_names):
        image, mask = load_pair(source_name)
        height, width = mask.shape

        for y in patch_starts(height):
            for x in patch_starts(width):
                image_patch = image[y : y + MD_PATCH_SIZE, x : x + MD_PATCH_SIZE]
                mask_patch = mask[y : y + MD_PATCH_SIZE, x : x + MD_PATCH_SIZE]

                if split == "train":
                    versions = augmented_versions(image_patch, mask_patch, rng)
                else:
                    versions = [("orig", image_patch, mask_patch)]

                source_stem = Path(source_name).stem
                for augmentation, image_aug, mask_aug in versions:
                    patch_name = (
                        f"{source_stem}__x{x:04d}_y{y:04d}__{augmentation}.png"
                    )
                    save_pair(split, patch_name, image_aug, mask_aug)
                    records.append(
                        {
                            "split": split,
                            "source": source_name,
                            "x": x,
                            "y": y,
                            "augmentation": augmentation,
                            "filename": patch_name,
                            "white_ratio": round(float(np.mean(mask_aug > 0)), 6),
                        }
                    )

    return records


def audit_dataset(records: list[dict]) -> dict:
    problems: list[str] = []
    counts: dict[str, int] = {}

    if MD_TRAIN_SOURCES & MD_VAL_SOURCES:
        problems.append("train/validation 원본이 겹칩니다.")

    for split in ("train", "val"):
        image_dir = MD_OUTPUT_ROOT / split / "images"
        mask_dir = MD_OUTPUT_ROOT / split / "masks"
        image_names = {path.name for path in image_dir.glob("*.png")}
        mask_names = {path.name for path in mask_dir.glob("*.png")}

        if image_names != mask_names:
            problems.append(f"{split}: image와 mask 파일명이 일치하지 않습니다.")

        counts[split] = len(image_names)

        for name in sorted(image_names):
            with Image.open(image_dir / name) as image, Image.open(mask_dir / name) as mask:
                if image.size != (MD_PATCH_SIZE, MD_PATCH_SIZE):
                    problems.append(f"{split}/{name}: image 크기={image.size}")
                if mask.size != (MD_PATCH_SIZE, MD_PATCH_SIZE):
                    problems.append(f"{split}/{name}: mask 크기={mask.size}")

                mask_arr = np.asarray(mask)
                values = set(int(value) for value in np.unique(mask_arr))
                if not values.issubset({0, 255}):
                    problems.append(f"{split}/{name}: mask 값={sorted(values)}")
                if not np.any(mask_arr == 255):
                    problems.append(f"{split}/{name}: 빈 mask")

    manifest_sources = {
        split: sorted({record["source"] for record in records if record["split"] == split})
        for split in ("train", "val")
    }

    if set(manifest_sources["train"]) != MD_TRAIN_SOURCES:
        problems.append("manifest의 train 원본 구성이 예상과 다릅니다.")
    if set(manifest_sources["val"]) != MD_VAL_SOURCES:
        problems.append("manifest의 validation 원본 구성이 예상과 다릅니다.")

    return {
        "status": "PASS" if not problems else "FAIL",
        "patch_size": [MD_PATCH_SIZE, MD_PATCH_SIZE],
        "stride": MD_STRIDE,
        "train_sources": manifest_sources["train"],
        "validation_sources": manifest_sources["val"],
        "train_patch_pairs": counts.get("train", 0),
        "validation_patch_pairs": counts.get("val", 0),
        "augmentation_train_only": True,
        "augmentations": [
            "orig",
            "hflip",
            "vflip",
            "bright090",
            "bright110",
            "contrast090",
            "contrast110",
            "noise_sigma3",
        ],
        "forbidden_augmentations_used": [],
        "problems": problems,
    }


def write_manifest(records: list[dict]) -> None:
    path = MD_OUTPUT_ROOT / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def make_preview(records: list[dict]) -> None:
    candidates = [
        record
        for record in records
        if record["augmentation"] == "orig"
    ]
    train_examples = [record for record in candidates if record["split"] == "train"][:2]
    val_examples = [record for record in candidates if record["split"] == "val"][:2]
    examples = train_examples + val_examples

    canvas = Image.new("RGB", (MD_PATCH_SIZE * 3, MD_PATCH_SIZE * len(examples)), "black")

    for row, record in enumerate(examples):
        split = record["split"]
        name = record["filename"]
        image = np.asarray(Image.open(MD_OUTPUT_ROOT / split / "images" / name).convert("RGB"))
        mask = np.asarray(Image.open(MD_OUTPUT_ROOT / split / "masks" / name).convert("L"))

        mask_rgb = np.repeat(mask[:, :, None], 3, axis=2)
        overlay = image.astype(np.float32)
        selected = mask > 0
        overlay[selected] = overlay[selected] * 0.45 + np.array([0, 255, 255]) * 0.55
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        y = row * MD_PATCH_SIZE
        canvas.paste(Image.fromarray(image), (0, y))
        canvas.paste(Image.fromarray(mask_rgb), (MD_PATCH_SIZE, y))
        canvas.paste(Image.fromarray(overlay), (MD_PATCH_SIZE * 2, y))

    canvas.save(MD_OUTPUT_ROOT / "dataset_preview.png")


def main_make_dataset() -> None:
    available_images = {path.name for path in (MD_INPUT_ROOT / "images").glob("*.png")}
    available_masks = {path.name for path in (MD_INPUT_ROOT / "masks").glob("*.png")}
    expected = MD_TRAIN_SOURCES | MD_VAL_SOURCES

    if expected - available_images:
        raise FileNotFoundError(f"누락 image: {sorted(expected - available_images)}")
    if expected - available_masks:
        raise FileNotFoundError(f"누락 mask: {sorted(expected - available_masks)}")

    prepare_output()
    rng = np.random.default_rng(MD_RANDOM_SEED)

    records = []
    records.extend(generate_split("train", MD_TRAIN_SOURCES, rng))
    records.extend(generate_split("val", MD_VAL_SOURCES, rng))

    write_manifest(records)
    make_preview(records)

    report = audit_dataset(records)
    (MD_OUTPUT_ROOT / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Dataset 검사:", report["status"])
    print("Train 원본:", len(report["train_sources"]), "장")
    print("Validation 원본:", len(report["validation_sources"]), "장")
    print("Train patch pair:", report["train_patch_pairs"])
    print("Validation patch pair:", report["validation_patch_pairs"])
    print("문제:", report["problems"])
    print("출력:", MD_OUTPUT_ROOT)

    if report["status"] != "PASS":
        raise RuntimeError("Dataset 자동 검사를 통과하지 못했습니다.")


# ===========================================================================
# 3. 베이스라인 U-Net 학습 (옛 train_unet.py)
# ===========================================================================
#
# torch/torch.nn 등 무거운 딥러닝 임포트는 이 섹션에서 처음 필요하므로 여기서
# 임포트한다 (make_masks/make_dataset 단계만 쓸 때는 불필요한 의존성이다).

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim import Adam  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from unet_inference import UNet, parameter_count  # noqa: E402
from training.unet_dataset import SegmentationDataset, dataset_summary  # noqa: E402

TRAIN_DEFAULT_DATASET = UNET_DATASET_ROOT
TRAIN_DEFAULT_OUTPUT = BASELINE_MODEL_DIR


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = torch.sum(probabilities * targets, dim=dims)
        denominator = torch.sum(probabilities, dim=dims) + torch.sum(targets, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return bce + (1.0 - dice.mean())


def binary_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    predictions = torch.sigmoid(logits) >= threshold
    target_binary = targets >= 0.5
    intersection = torch.sum(predictions & target_binary).item()
    total = torch.sum(predictions).item() + torch.sum(target_binary).item()
    return float(intersection), float(total)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    item_count = 0
    intersection_sum = 0.0
    dice_total_sum = 0.0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            loss_sum += float(loss.item()) * batch_size
            item_count += batch_size
            intersection, dice_total = binary_dice(logits.detach(), masks)
            intersection_sum += intersection
            dice_total_sum += dice_total

    average_loss = loss_sum / max(1, item_count)
    dice_score = (2.0 * intersection_sum + 1.0) / (dice_total_sum + 1.0)
    return average_loss, dice_score


def save_curves(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], label="validation")
    axes[0].set_title("BCE + Dice loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_dice"] for row in rows], label="train")
    axes[1].plot(epochs, [row["val_dice"] for row in rows], label="validation")
    axes[1].set_title("Binary Dice")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main_train() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net boundary segmentation model")
    parser.add_argument("--dataset", type=Path, default=TRAIN_DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=TRAIN_DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = SegmentationDataset(args.dataset, "train")
    val_dataset = SegmentationDataset(args.dataset, "val")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = UNet(base_channels=args.base_channels).to(device)
    criterion = BCEDiceLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    configuration = {
        "device": str(device),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "base_channels": args.base_channels,
        "patience": args.patience,
        "threads": args.threads,
        "seed": args.seed,
        "parameter_count": parameter_count(model),
        "dataset": dataset_summary(args.dataset),
    }
    (args.output / "training_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Device:", device)
    print("Parameters:", f"{configuration['parameter_count']:,}")
    print("Train patches:", len(train_dataset))
    print("Validation patches:", len(val_dataset))

    best_val_dice = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []
    best_path = args.output / "best_unet.pt"

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_dice = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_dice = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        elapsed = time.perf_counter() - started

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_dice": round(train_dice, 6),
            "val_loss": round(val_loss, 6),
            "val_dice": round(val_dice, 6),
            "seconds": round(elapsed, 3),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"time={elapsed:.1f}s"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "base_channels": args.base_channels,
                    "in_channels": 3,
                    "out_channels": 1,
                    "threshold": 0.5,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: {args.patience} epochs without improvement")
            break

    log_path = args.output / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_curves(history, args.output / "training_curves.png")
    summary = {
        "status": "complete",
        "epochs_completed": len(history),
        "best_val_dice": best_val_dice,
        "best_checkpoint": str(best_path),
        "training_log": str(log_path),
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Best validation Dice:", f"{best_val_dice:.6f}")
    print("Best checkpoint:", best_path)


# ===========================================================================
# 4. 사람 검수용 전체 mask 준비 (옛 prepare_mask_review.py)
# ===========================================================================

from unet_inference import load_unet, predict_probability, remove_small_components  # noqa: E402

PMR_DEFAULT_CHECKPOINT = ACTIVE_UNET_CHECKPOINT
PMR_DEFAULT_OUTPUT = MANUAL_REVIEW_ROOT / "full_mask"


def main_prepare_mask_review() -> None:
    parser = argparse.ArgumentParser(description="Prepare U-Net masks for manual correction")
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=PMR_DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=PMR_DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-area", type=int, default=30)
    parser.add_argument("--min-height-ratio", type=float, default=0.50)
    parser.add_argument("--side-margin-ratio", type=float, default=0.02)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model, checkpoint, device = load_unet(args.checkpoint)
    threshold = float(args.threshold) if args.threshold is not None else float(checkpoint.get("threshold", 0.5))

    prediction_dir = args.output / "predictions"
    corrected_dir = args.output / "corrected_masks"
    preview_dir = args.output / "corrected_previews"
    for folder in (prediction_dir, corrected_dir, preview_dir):
        folder.mkdir(parents=True, exist_ok=True)

    entries = []
    for index, image_path in enumerate(args.images, start=1):
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        loaded = load_image(str(image_path))
        image = loaded.native_arr
        probability = predict_probability(model, image, device)
        raw_mask = probability >= threshold
        if not 0.0 <= args.side_margin_ratio < 0.5:
            raise ValueError("side_margin_ratio must be between 0 and 0.5")
        review_mask = raw_mask.copy()
        side_margin_pixels = int(round(review_mask.shape[1] * args.side_margin_ratio))
        side_removed_pixels = 0
        if side_margin_pixels > 0:
            side_removed_pixels = int(
                review_mask[:, :side_margin_pixels].sum()
                + review_mask[:, -side_margin_pixels:].sum()
            )
            review_mask[:, :side_margin_pixels] = False
            review_mask[:, -side_margin_pixels:] = False
        mask, stats = remove_small_components(
            review_mask,
            min_area=args.min_area,
            min_height_ratio=args.min_height_ratio,
        )
        stats["side_margin_ratio"] = float(args.side_margin_ratio)
        stats["side_margin_pixels"] = side_margin_pixels
        stats["side_removed_pixels"] = side_removed_pixels

        stem = image_path.stem
        native_path = prediction_dir / f"{stem}_native.png"
        raw_path = prediction_dir / f"{stem}_unet_mask_raw.png"
        mask_path = prediction_dir / f"{stem}_unet_mask.png"
        probability_path = prediction_dir / f"{stem}_unet_probability.png"
        overlay_path = prediction_dir / f"{stem}_unet_overlay.png"
        corrected_path = corrected_dir / f"{stem}.png"
        corrected_preview_path = preview_dir / f"{stem}_corrected_overlay.png"

        Image.fromarray(image, mode="RGB").save(native_path)
        Image.fromarray(raw_mask.astype(np.uint8) * 255, mode="L").save(raw_path)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        Image.fromarray(np.clip(probability * 255, 0, 255).astype(np.uint8), mode="L").save(probability_path)
        overlay = image.astype(np.float32).copy()
        overlay[mask] = overlay[mask] * 0.45 + np.array([0, 255, 255]) * 0.55
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(overlay_path)

        entries.append(
            {
                "id": stem,
                "source_image": str(image_path.resolve()),
                "native_image": str(native_path.resolve()),
                "initial_mask": str(mask_path.resolve()),
                "raw_mask": str(raw_path.resolve()),
                "probability": str(probability_path.resolve()),
                "prediction_overlay": str(overlay_path.resolve()),
                "corrected_mask": str(corrected_path.resolve()),
                "corrected_preview": str(corrected_preview_path.resolve()),
                "capture_size": [int(loaded.capture_arr.shape[1]), int(loaded.capture_arr.shape[0])],
                "native_size": [int(image.shape[1]), int(image.shape[0])],
                "scale": loaded.scale.to_dict(),
                "postprocessing": stats,
            }
        )
        print(f"[{index}/{len(args.images)}] {stem}: native={image.shape[1]}x{image.shape[0]}, white={mask.mean()*100:.2f}%")

    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_dice": checkpoint.get("val_dice"),
        "threshold": threshold,
        "entries": entries,
    }
    manifest_path = args.output / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Manifest:", manifest_path)
    print("Editor: python training/mask_editor.py")


# ===========================================================================
# 5. 검수용 가로 band로 분할 (옛 prepare_band_mask_review.py)
# ===========================================================================

PBR_DEFAULT_SOURCE = MANUAL_REVIEW_ROOT / "full_mask" / "review_manifest.json"
PBR_DEFAULT_OUTPUT = MANUAL_REVIEW_ROOT / "band"
PBR_DEFAULT_FRACTIONS = (0.15, 0.325, 0.50, 0.675, 0.85)


def crop_bounds(height: int, center: int, band_height: int) -> tuple[int, int]:
    if band_height <= 0 or band_height > height:
        raise ValueError(f"band_height must be in 1..{height}, got {band_height}")
    y0 = center - band_height // 2
    y0 = min(max(0, y0), height - band_height)
    return y0, y0 + band_height


def main_prepare_band_review() -> None:
    parser = argparse.ArgumentParser(description="Prepare five editable horizontal mask bands per image")
    parser.add_argument("--source", type=Path, default=PBR_DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=PBR_DEFAULT_OUTPUT)
    parser.add_argument("--band-height", type=int, default=64)
    parser.add_argument("--fractions", type=float, nargs="+", default=PBR_DEFAULT_FRACTIONS)
    args = parser.parse_args()

    source_manifest = json.loads(args.source.read_text(encoding="utf-8"))
    image_dir = args.output / "images"
    initial_dir = args.output / "initial_masks"
    corrected_dir = args.output / "corrected_masks"
    preview_dir = args.output / "corrected_previews"
    for folder in (image_dir, initial_dir, corrected_dir, preview_dir):
        folder.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for source_entry in source_manifest.get("entries", []):
        image = np.asarray(Image.open(source_entry["native_image"]).convert("RGB"))
        initial_mask = np.asarray(Image.open(source_entry["initial_mask"]).convert("L"))
        if image.shape[:2] != initial_mask.shape:
            raise ValueError(
                f"Image/mask mismatch for {source_entry['id']}: "
                f"image={image.shape[:2]}, mask={initial_mask.shape}"
            )

        height, width = initial_mask.shape
        for band_index, fraction in enumerate(args.fractions, start=1):
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"fraction must be in 0..1, got {fraction}")
            center = round((height - 1) * fraction)
            y0, y1 = crop_bounds(height, center, args.band_height)
            band_id = f"{source_entry['id']}_band{band_index}_y{center:03d}"

            image_path = image_dir / f"{band_id}.png"
            initial_path = initial_dir / f"{band_id}.png"
            corrected_path = corrected_dir / f"{band_id}.png"
            preview_path = preview_dir / f"{band_id}_corrected_overlay.png"

            Image.fromarray(image[y0:y1], mode="RGB").save(image_path)
            Image.fromarray(initial_mask[y0:y1], mode="L").save(initial_path)

            entries.append(
                {
                    "id": band_id,
                    "parent_id": source_entry["id"],
                    "source_image": source_entry["source_image"],
                    "native_image": str(image_path.resolve()),
                    "initial_mask": str(initial_path.resolve()),
                    "corrected_mask": str(corrected_path.resolve()),
                    "corrected_preview": str(preview_path.resolve()),
                    "native_size": [width, args.band_height],
                    "crop_native": {
                        "x0": 0,
                        "x1": width,
                        "y0": y0,
                        "y1": y1,
                        "center_y": center,
                        "fraction": fraction,
                    },
                }
            )

    output_manifest = {
        "source_manifest": str(args.source.resolve()),
        "band_height": args.band_height,
        "fractions": list(args.fractions),
        "entries": entries,
    }
    manifest_path = args.output / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Source images: {len(source_manifest.get('entries', []))}")
    print(f"Bands: {len(entries)}")
    print(f"Band size: full width x {args.band_height}px")
    print(f"Manifest: {manifest_path}")


# ===========================================================================
# 6. 한 경계선을 추적하는 crop (옛 prepare_single_boundary_review.py)
# ===========================================================================

PSB_DEFAULT_SOURCE = MANUAL_REVIEW_ROOT / "band" / "review_manifest.json"
PSB_DEFAULT_OUTPUT = MANUAL_REVIEW_ROOT / "single_boundary"


def vertical_runs(mask: np.ndarray) -> list[tuple[int, int, float]]:
    minimum_pixels = max(3, round(mask.shape[0] * 0.25))
    active = np.count_nonzero(mask, axis=0) >= minimum_pixels
    changes = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(x0), int(x1), float((x0 + x1 - 1) / 2)) for x0, x1 in zip(starts, ends)]


def crop_x_bounds(width: int, center_x: float, crop_width: int) -> tuple[int, int]:
    if crop_width <= 0 or crop_width > width:
        raise ValueError(f"crop_width must be in 1..{width}, got {crop_width}")
    x0 = round(center_x) - crop_width // 2
    x0 = min(max(0, x0), width - crop_width)
    return x0, x0 + crop_width


def main_prepare_single_boundary_review() -> None:
    parser = argparse.ArgumentParser(description="Prepare five crops of one vertical boundary")
    parser.add_argument("--source", type=Path, default=PSB_DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=PSB_DEFAULT_OUTPUT)
    parser.add_argument("--parent-id", default=None, help="Original image id; default is the first image")
    parser.add_argument("--crop-width", type=int, default=64)
    args = parser.parse_args()

    source_manifest = json.loads(args.source.read_text(encoding="utf-8"))
    source_entries = source_manifest.get("entries", [])
    if not source_entries:
        raise ValueError("The band review manifest has no entries")
    parent_id = args.parent_id or source_entries[0]["parent_id"]
    bands = [entry for entry in source_entries if entry.get("parent_id") == parent_id]
    if not bands:
        raise ValueError(f"No bands found for parent_id={parent_id}")

    image_dir = args.output / "images"
    initial_dir = args.output / "initial_masks"
    corrected_dir = args.output / "corrected_masks"
    preview_dir = args.output / "corrected_previews"
    for folder in (image_dir, initial_dir, corrected_dir, preview_dir):
        folder.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    tracked_x: float | None = None
    for index, band in enumerate(bands, start=1):
        image = np.asarray(Image.open(band["native_image"]).convert("RGB"))
        corrected_source = Path(band["corrected_mask"])
        mask_source = corrected_source if corrected_source.exists() else Path(band["initial_mask"])
        mask = np.asarray(Image.open(mask_source).convert("L")) > 127
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask mismatch for {band['id']}")

        runs = vertical_runs(mask)
        if not runs:
            raise ValueError(f"No vertical boundary found in {band['id']}")
        target = (image.shape[1] - 1) / 2 if tracked_x is None else tracked_x
        chosen = min(runs, key=lambda run: abs(run[2] - target))
        tracked_x = chosen[2]
        x0, x1 = crop_x_bounds(image.shape[1], tracked_x, args.crop_width)

        crop_id = f"{parent_id}_boundary_center_part{index}_x{round(tracked_x):03d}"
        image_path = image_dir / f"{crop_id}.png"
        initial_path = initial_dir / f"{crop_id}.png"
        corrected_path = corrected_dir / f"{crop_id}.png"
        preview_path = preview_dir / f"{crop_id}_corrected_overlay.png"

        Image.fromarray(image[:, x0:x1], mode="RGB").save(image_path)
        Image.fromarray(mask[:, x0:x1].astype(np.uint8) * 255, mode="L").save(initial_path)

        entries.append(
            {
                "id": crop_id,
                "parent_id": parent_id,
                "source_band_id": band["id"],
                "source_image": band["source_image"],
                "native_image": str(image_path.resolve()),
                "initial_mask": str(initial_path.resolve()),
                "corrected_mask": str(corrected_path.resolve()),
                "corrected_preview": str(preview_path.resolve()),
                "native_size": [args.crop_width, int(image.shape[0])],
                "crop_native": {
                    "x0": x0,
                    "x1": x1,
                    "y0": band["crop_native"]["y0"],
                    "y1": band["crop_native"]["y1"],
                    "boundary_x": tracked_x,
                },
            }
        )

    manifest = {
        "source_manifest": str(args.source.resolve()),
        "parent_id": parent_id,
        "crop_width": args.crop_width,
        "entries": entries,
    }
    manifest_path = args.output / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Parent image: {parent_id}")
    print(f"Boundary crops: {len(entries)}")
    print(f"Crop size: {args.crop_width}x{entries[0]['native_size'][1]}px")
    print(f"Manifest: {manifest_path}")


# ===========================================================================
# 7. 이미지별로 다양한 경계 crop 하나씩 (옛 prepare_diverse_boundary_review.py)
# ===========================================================================

PDB_DEFAULT_SOURCE = MANUAL_REVIEW_ROOT / "band" / "review_manifest.json"
PDB_DEFAULT_OUTPUT = MANUAL_REVIEW_ROOT / "diverse_boundary"


def main_prepare_diverse_boundary_review() -> None:
    parser = argparse.ArgumentParser(description="Prepare one boundary crop from each source image")
    parser.add_argument("--source", type=Path, default=PDB_DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=PDB_DEFAULT_OUTPUT)
    parser.add_argument("--crop-width", type=int, default=64)
    parser.add_argument(
        "--boundary-offset",
        type=int,
        default=1,
        help="Offset from the boundary nearest the image center; +1 selects the next boundary to the right",
    )
    args = parser.parse_args()

    source_manifest = json.loads(args.source.read_text(encoding="utf-8"))
    source_entries = source_manifest.get("entries", [])
    if not source_entries:
        raise ValueError("The band review manifest has no entries")

    parent_ids = list(dict.fromkeys(entry["parent_id"] for entry in source_entries))
    image_dir = args.output / "images"
    initial_dir = args.output / "initial_masks"
    corrected_dir = args.output / "corrected_masks"
    preview_dir = args.output / "corrected_previews"
    for folder in (image_dir, initial_dir, corrected_dir, preview_dir):
        folder.mkdir(parents=True, exist_ok=True)

    output_entries: list[dict] = []
    for parent_id in parent_ids:
        candidates = [entry for entry in source_entries if entry["parent_id"] == parent_id]
        band = min(candidates, key=lambda entry: abs(float(entry["crop_native"]["fraction"]) - 0.5))
        image = np.asarray(Image.open(band["native_image"]).convert("RGB"))
        corrected_source = Path(band["corrected_mask"])
        mask_source = corrected_source if corrected_source.exists() else Path(band["initial_mask"])
        mask = np.asarray(Image.open(mask_source).convert("L")) > 127
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask mismatch for {band['id']}")

        runs = sorted(vertical_runs(mask), key=lambda run: run[2])
        if not runs:
            raise ValueError(f"No vertical boundary found in {band['id']}")
        center_target = (image.shape[1] - 1) / 2
        center_index = min(range(len(runs)), key=lambda index: abs(runs[index][2] - center_target))
        selected_index = center_index + args.boundary_offset
        if not 0 <= selected_index < len(runs):
            raise ValueError(
                f"boundary offset {args.boundary_offset} is outside the {len(runs)} runs in {band['id']}"
            )
        boundary_x = runs[selected_index][2]
        x0, x1 = crop_x_bounds(image.shape[1], boundary_x, args.crop_width)

        center_y = int(band["crop_native"]["center_y"])
        crop_id = f"{parent_id}_diverse_y{center_y:03d}_x{round(boundary_x):03d}"
        image_path = image_dir / f"{crop_id}.png"
        initial_path = initial_dir / f"{crop_id}.png"
        corrected_path = corrected_dir / f"{crop_id}.png"
        preview_path = preview_dir / f"{crop_id}_corrected_overlay.png"
        Image.fromarray(image[:, x0:x1], mode="RGB").save(image_path)
        Image.fromarray(mask[:, x0:x1].astype(np.uint8) * 255, mode="L").save(initial_path)

        output_entries.append(
            {
                "id": crop_id,
                "parent_id": parent_id,
                "source_band_id": band["id"],
                "source_image": band["source_image"],
                "native_image": str(image_path.resolve()),
                "initial_mask": str(initial_path.resolve()),
                "corrected_mask": str(corrected_path.resolve()),
                "corrected_preview": str(preview_path.resolve()),
                "native_size": [args.crop_width, int(image.shape[0])],
                "crop_native": {
                    "x0": x0,
                    "x1": x1,
                    "y0": int(band["crop_native"]["y0"]),
                    "y1": int(band["crop_native"]["y1"]),
                    "boundary_x": boundary_x,
                    "boundary_offset": args.boundary_offset,
                },
            }
        )

    manifest = {
        "source_manifest": str(args.source.resolve()),
        "selection": "one center-height boundary crop per different source image",
        "boundary_offset": args.boundary_offset,
        "crop_width": args.crop_width,
        "entries": output_entries,
    }
    manifest_path = args.output / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Different source images: {len(parent_ids)}")
    print(f"Boundary crops: {len(output_entries)}")
    print(f"Crop size: {args.crop_width}x{output_entries[0]['native_size'][1]}px")
    for entry in output_entries:
        print(f"- {entry['id']}")
    print(f"Manifest: {manifest_path}")


# ===========================================================================
# 8. 수정된 review manifest 병합 (옛 combine_manual_reviews.py)
# ===========================================================================

CMB_DEFAULT_MANIFESTS = (
    MANUAL_REVIEW_ROOT / "single_boundary" / "review_manifest.json",
    MANUAL_REVIEW_ROOT / "diverse_boundary" / "review_manifest.json",
)
CMB_DEFAULT_OUTPUT = MANUAL_REVIEW_ROOT / "combined" / "review_manifest.json"


def main_combine_reviews() -> None:
    parser = argparse.ArgumentParser(description="Combine corrected manual mask manifests")
    parser.add_argument("manifests", type=Path, nargs="*", default=CMB_DEFAULT_MANIFESTS)
    parser.add_argument("--output", type=Path, default=CMB_DEFAULT_OUTPUT)
    args = parser.parse_args()

    entries = []
    source_manifests = []
    seen_ids = set()
    for manifest_path in args.manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_manifests.append(str(manifest_path.resolve()))
        for entry in manifest.get("entries", []):
            if entry["id"] in seen_ids:
                raise ValueError(f"Duplicate entry id: {entry['id']}")
            corrected = Path(entry["corrected_mask"])
            if not corrected.exists():
                raise FileNotFoundError(f"Corrected mask missing: {corrected}")
            seen_ids.add(entry["id"])
            entries.append(entry)

    if not entries:
        raise ValueError("No corrected entries found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "source_manifests": source_manifests,
        "entry_count": len(entries),
        "entries": entries,
    }
    args.output.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Source manifests:", len(source_manifests))
    print("Combined corrected crops:", len(entries))
    print("Output:", args.output)


# ===========================================================================
# 9. 수정된 crop으로 미세조정 (옛 finetune_unet_manual.py)
# ===========================================================================

FT_DEFAULT_MANIFEST = MANUAL_REVIEW_ROOT / "combined" / "review_manifest.json"
FT_DEFAULT_CHECKPOINT = BASELINE_MODEL_DIR / "best_unet.pt"
FT_DEFAULT_VALIDATION = UNET_DATASET_ROOT
FT_DEFAULT_OUTPUT = ANALYSIS_ROOT / "models" / "fine_tune_runs" / "manual"


class ManualCropDataset(Dataset):
    """Load corrected crops; safe flips add samples without changing line width."""

    TRANSFORMS = ("original", "horizontal", "vertical", "both")

    def __init__(self, manifest_path: Path, augment: bool) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.entries = manifest.get("entries", [])
        if not self.entries:
            raise RuntimeError("Manual review manifest has no entries")
        missing = [entry["corrected_mask"] for entry in self.entries if not Path(entry["corrected_mask"]).exists()]
        if missing:
            raise RuntimeError(f"Corrected masks are missing: {missing}")
        transforms = self.TRANSFORMS if augment else ("original",)
        self.samples = [(entry, transform) for entry in self.entries for transform in transforms]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        entry, transform = self.samples[index]
        image = np.asarray(Image.open(entry["native_image"]).convert("RGB"), dtype=np.float32)
        mask = np.asarray(Image.open(entry["corrected_mask"]).convert("L"), dtype=np.uint8)
        if image.shape[:2] != mask.shape:
            raise RuntimeError(f"Shape mismatch for {entry['id']}")

        if transform in {"horizontal", "both"}:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
        if transform in {"vertical", "both"}:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        image = np.ascontiguousarray(image.transpose(2, 0, 1) / 255.0)
        mask = np.ascontiguousarray((mask > 127).astype(np.float32)[None])
        return torch.from_numpy(image), torch.from_numpy(mask), f"{entry['id']}__{transform}"


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    return run_epoch(model, loader, criterion, device, optimizer=None)


def run_output_epoch(model, loader, criterion, device, optimizer) -> tuple[float, float]:
    """Train only the output layer while all BatchNorm statistics stay fixed."""
    model.eval()
    loss_sum = 0.0
    item_count = 0
    intersection_sum = 0.0
    dice_total_sum = 0.0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        loss_sum += float(loss.item()) * batch_size
        item_count += batch_size
        intersection, dice_total = binary_dice(logits.detach(), masks)
        intersection_sum += intersection
        dice_total_sum += dice_total
    average_loss = loss_sum / max(1, item_count)
    dice_score = (2.0 * intersection_sum + 1.0) / (dice_total_sum + 1.0)
    return average_loss, dice_score


def mask_audit(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["entries"]:
        mask = np.asarray(Image.open(entry["corrected_mask"]).convert("L")) > 127
        widths = mask.sum(axis=1)
        nonzero = widths[widths > 0]
        rows.append(
            {
                "id": entry["id"],
                "shape": [int(mask.shape[1]), int(mask.shape[0])],
                "empty_rows": int(np.count_nonzero(widths == 0)),
                "row_width_min": int(nonzero.min()),
                "row_width_median": float(np.median(nonzero)),
                "row_width_max": int(nonzero.max()),
                "row_width_mean": float(nonzero.mean()),
            }
        )
    return rows


def main_finetune() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune U-Net on corrected 64x64 boundary crops")
    parser.add_argument("--manifest", type=Path, default=FT_DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=FT_DEFAULT_CHECKPOINT)
    parser.add_argument("--validation", type=Path, default=FT_DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=FT_DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--train-scope",
        choices=("output", "decoder"),
        default="output",
        help="Train only the final output layer or the decoder plus output layer",
    )
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-val-drop", type=float, default=0.005)
    parser.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = UNet(
        in_channels=int(checkpoint.get("in_channels", 3)),
        out_channels=int(checkpoint.get("out_channels", 1)),
        base_channels=int(checkpoint["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_dataset = ManualCropDataset(args.manifest, augment=True)
    manual_eval_dataset = ManualCropDataset(args.manifest, augment=False)
    val_dataset = SegmentationDataset(args.validation, "val")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    manual_loader = DataLoader(manual_eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = BCEDiceLoss()
    baseline_val_loss, baseline_val_dice = evaluate(model, val_loader, criterion, device)
    baseline_manual_loss, baseline_manual_dice = evaluate(model, manual_loader, criterion, device)
    val_floor = baseline_val_dice - args.max_val_drop

    print("Device:", device)
    print("Manual crops:", len(manual_eval_dataset), "augmented train samples:", len(train_dataset))
    print("Validation patches:", len(val_dataset))
    print(f"Baseline manual Dice: {baseline_manual_dice:.6f}")
    print(f"Baseline validation Dice: {baseline_val_dice:.6f}")
    print(f"Validation floor: {val_floor:.6f}")

    # Tiny manual crops are not sufficient to update the whole network safely.
    # Preserve the encoder and BatchNorm statistics.  The default calibrates
    # only the final 1x1 output layer; decoder scope can refine boundary shape.
    for parameter in model.parameters():
        parameter.requires_grad = False
    train_modules = [model.output]
    if args.train_scope == "decoder":
        train_modules = [model.up1, model.up2, model.up3, model.up4, model.output]
    for module in train_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = Adam(trainable_parameters, lr=args.learning_rate)
    print("Train scope:", args.train_scope, "trainable parameters:", sum(p.numel() for p in trainable_parameters))
    history: list[dict] = []
    best_manual_dice = -1.0
    best_path = args.output / "best_unet_manual.pt"
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_dice = run_output_epoch(model, train_loader, criterion, device, optimizer)
        manual_loss, manual_dice = evaluate(model, manual_loader, criterion, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - started
        eligible = val_dice >= val_floor
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_dice": round(train_dice, 6),
            "manual_loss": round(manual_loss, 6),
            "manual_dice": round(manual_dice, 6),
            "val_loss": round(val_loss, 6),
            "val_dice": round(val_dice, 6),
            "eligible": eligible,
            "seconds": round(elapsed, 3),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} train={train_dice:.4f} "
            f"manual={manual_dice:.4f} val={val_dice:.4f} eligible={eligible} time={elapsed:.1f}s"
        )

        if eligible and manual_dice > best_manual_dice:
            best_manual_dice = manual_dice
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "manual_dice": manual_dice,
                    "baseline_val_dice": baseline_val_dice,
                    "baseline_manual_dice": baseline_manual_dice,
                    "base_channels": int(checkpoint["base_channels"]),
                    "in_channels": int(checkpoint.get("in_channels", 3)),
                    "out_channels": int(checkpoint.get("out_channels", 1)),
                    "threshold": float(checkpoint.get("threshold", 0.5)),
                    "finetuned_from": str(args.checkpoint.resolve()),
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: {args.patience} epochs without an eligible manual improvement")
            break

    if not best_path.exists():
        raise RuntimeError("No fine-tuned epoch stayed within the allowed validation Dice drop")

    log_path = args.output / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Reuse the standard curve helper with the fields it expects.
    curve_rows = [
        {
            "epoch": row["epoch"],
            "train_loss": row["train_loss"],
            "train_dice": row["train_dice"],
            "val_loss": row["val_loss"],
            "val_dice": row["val_dice"],
        }
        for row in history
    ]
    save_curves(curve_rows, args.output / "training_curves.png")
    summary = {
        "status": "complete",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "best_checkpoint": str(best_path.resolve()),
        "epochs_completed": len(history),
        "baseline_manual_dice": baseline_manual_dice,
        "best_manual_dice": best_manual_dice,
        "baseline_validation_dice": baseline_val_dice,
        "best_checkpoint_validation_dice": max(
            row["val_dice"] for row in history if row["eligible"] and row["manual_dice"] == round(best_manual_dice, 6)
        ),
        "mask_audit": mask_audit(args.manifest),
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Best manual Dice:", f"{best_manual_dice:.6f}")
    print("Best checkpoint:", best_path)


# ===========================================================================
# 10. 확률 임계값 보정 (옛 calibrate_unet_manual.py)
# ===========================================================================

CAL_DEFAULT_MANIFEST = MANUAL_REVIEW_ROOT / "single_boundary" / "review_manifest.json"
CAL_DEFAULT_CHECKPOINT = ACTIVE_UNET_CHECKPOINT
CAL_DEFAULT_VALIDATION = UNET_DATASET_ROOT
CAL_DEFAULT_OUTPUT = ANALYSIS_ROOT / "models" / "calibration_runs" / "manual"


def collect(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    probabilities = []
    targets = []
    model.eval()
    with torch.no_grad():
        for images, masks, _ in loader:
            logits = model(images.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            targets.append((masks.numpy().reshape(-1) >= 0.5))
    return np.concatenate(probabilities), np.concatenate(targets)


def dice(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    intersection = np.count_nonzero(prediction & target)
    total = np.count_nonzero(prediction) + np.count_nonzero(target)
    return float((2 * intersection + 1) / (total + 1))


def main_calibrate() -> None:
    parser = argparse.ArgumentParser(description="Calibrate U-Net threshold from corrected masks")
    parser.add_argument("--manifest", type=Path, default=CAL_DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=CAL_DEFAULT_CHECKPOINT)
    parser.add_argument("--validation", type=Path, default=CAL_DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=CAL_DEFAULT_OUTPUT)
    parser.add_argument("--max-val-drop", type=float, default=0.005)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = UNet(
        in_channels=int(checkpoint.get("in_channels", 3)),
        out_channels=int(checkpoint.get("out_channels", 1)),
        base_channels=int(checkpoint["base_channels"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    manual_loader = DataLoader(ManualCropDataset(args.manifest, augment=False), batch_size=5, shuffle=False)
    val_loader = DataLoader(SegmentationDataset(args.validation, "val"), batch_size=5, shuffle=False)
    manual_probability, manual_target = collect(model, manual_loader, device)
    val_probability, val_target = collect(model, val_loader, device)

    baseline_threshold = 0.5
    baseline_manual = dice(manual_probability, manual_target, baseline_threshold)
    baseline_val = dice(val_probability, val_target, baseline_threshold)
    val_floor = baseline_val - args.max_val_drop

    rows = []
    for threshold in np.arange(0.20, 0.801, 0.005):
        threshold = round(float(threshold), 3)
        manual_score = dice(manual_probability, manual_target, threshold)
        val_score = dice(val_probability, val_target, threshold)
        rows.append(
            {
                "threshold": threshold,
                "manual_dice": manual_score,
                "val_dice": val_score,
                "eligible": val_score >= val_floor,
            }
        )

    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise RuntimeError("No threshold stayed within the allowed validation Dice drop")
    best = max(
        eligible,
        key=lambda row: (row["manual_dice"], row["val_dice"], -abs(row["threshold"] - 0.5)),
    )

    output_checkpoint = args.output / "best_unet_manual_calibrated.pt"
    calibrated = dict(checkpoint)
    calibrated["threshold"] = best["threshold"]
    calibrated["threshold_calibration"] = {
        "manual_manifest": str(args.manifest.resolve()),
        "baseline_threshold": baseline_threshold,
        "baseline_manual_dice": baseline_manual,
        "baseline_val_dice": baseline_val,
        "selected_threshold": best["threshold"],
        "selected_manual_dice": best["manual_dice"],
        "selected_val_dice": best["val_dice"],
        "max_val_drop": args.max_val_drop,
    }
    torch.save(calibrated, output_checkpoint)

    csv_path = args.output / "threshold_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output / "threshold_calibration.json"
    summary_path.write_text(
        json.dumps(calibrated["threshold_calibration"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Device:", device)
    print(f"Baseline threshold: {baseline_threshold:.3f}")
    print(f"Baseline manual Dice: {baseline_manual:.6f}")
    print(f"Baseline validation Dice: {baseline_val:.6f}")
    print(f"Selected threshold: {best['threshold']:.3f}")
    print(f"Selected manual Dice: {best['manual_dice']:.6f}")
    print(f"Selected validation Dice: {best['val_dice']:.6f}")
    print("Checkpoint:", output_checkpoint)


# ===========================================================================
# 명령행 진입점: 단계 이름으로 실행할 함수를 고른다.
# ===========================================================================

STAGES = {
    "make-masks": main_make_masks,
    "make-dataset": main_make_dataset,
    "train": main_train,
    "prepare-mask-review": main_prepare_mask_review,
    "prepare-band-review": main_prepare_band_review,
    "prepare-single-boundary-review": main_prepare_single_boundary_review,
    "prepare-diverse-boundary-review": main_prepare_diverse_boundary_review,
    "combine-reviews": main_combine_reviews,
    "finetune": main_finetune,
    "calibrate": main_calibrate,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        names = "\n  ".join(STAGES)
        print(f"사용법: python training/unet_training.py <stage> [옵션...]\n단계:\n  {names}")
        raise SystemExit(1)
    stage = sys.argv.pop(1)
    STAGES[stage]()


if __name__ == "__main__":
    main()
