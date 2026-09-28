"""평가 툴을 한 곳에 모은 파일: 공통 CSV 비교 지표 계산 + 전체 성능 스위트 오케스트레이션.

구성 (옛 파일 -> 이 파일의 절):
    1) benchmark_width_methods.py -> main_compare() : 방법들의 예측 CSV를 정답과 비교해 지표 산출
    2) run_performance_suite.py   -> main_suite()   : 데이터 트리에서 manifest/정답을 만들고
                                                       comparison_methods/의 unet/canny/threshold
                                                       세 방법을 실행한 뒤 main_compare()로 비교

이 파일은 어느 한 측정 알고리즘에도 종속되지 않는다. 각 팀원은 동일한 CSV
형식으로 예측을 내보내고, 이 평가기는 이를 사람 정답과 매칭해 동일한
CSV/JSON 리포트를 만든다.

실행 (linewidth_tool 폴더에서):

기본 템플릿 CSV 만들기:
    python evaluation/evaluate.py compare --make-templates benchmark_data

한 개 이상의 방법 비교:
    python evaluation/evaluate.py compare \
        --ground-truth benchmark_data/ground_truth.csv \
        --prediction unet=results/unet_predictions.csv \
        --prediction canny=results/canny_predictions.csv \
        --prediction threshold=results/threshold_predictions.csv \
        --output benchmark_results

전체 성능 스위트:
    python evaluation/evaluate.py suite prepare --data-root ... --output ...
    python evaluation/evaluate.py suite run --data-root ... --output ...

좌표와 폭은 항상 native-image 픽셀이어야 하며, 확대된 캡처 픽셀이 아니다.
--um-per-px를 주지 않으면 고정 보정값 1 px = 0.1592 um을 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np  # Import before torch in this Anaconda environment.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Local layout keeps the U-Net method in comparison_methods/unet; in the team
# repository this file sits next to unet_inference.py, so no extra path is needed.
UNET_DIR = PROJECT_ROOT / "comparison_methods" / "unet"
for _path in (UNET_DIR, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from team_protocol import TARGET_PREFIX, save_prediction_csv


# ===========================================================================
# 1. 공통 CSV 비교 지표 계산 (옛 benchmark_width_methods.py)
# ===========================================================================

CALIBRATION_UM_PER_PX = 0.1592
EXPECTED_Y_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
MEASUREMENT_TYPES = ("line", "gap", "sidewall")

GT_REQUIRED = ("image_id", "measurement_type", "y_fraction", "target_id")
GT_OPTIONAL = (
    "answer_width_px",
    "answer_width_um",
    "center_x_px",
    "left_x_px",
    "right_x_px",
    "image_path",
    "reference_image_path",
    "laptop_id",
    "evaluation_set",
    "pair_group",
    "repeat_group",
    "repeat_id",
    "brightness_group",
    "brightness_level",
    "condition",
    "annotator",
)
PRED_REQUIRED = ("image_id", "measurement_type", "y_fraction", "target_id")
PRED_OPTIONAL = (
    "method",
    "width_px",
    "width_um",
    "center_x_px",
    "left_x_px",
    "right_x_px",
    "runtime_ms",
    "status",
    "message",
)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def safe_pstdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if values else None


def safe_stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def rounded(value: float | None, digits: int = 6):
    return None if value is None else round(float(value), digits)


def normalize_y_fraction(value: str | float) -> float:
    number = float(value)
    if number > 1.0:
        number /= 100.0
    # Pixel-row rounding makes y/(height-1) slightly different from the
    # protocol fraction (for example 0.200957 instead of 0.20). Treat those
    # values as the same standardized measurement row.
    nearest = min(EXPECTED_Y_FRACTIONS, key=lambda expected: abs(number - expected))
    if abs(number - nearest) <= 0.0025:
        number = nearest
    return round(number, 6)


def normalize_type(value: str) -> str:
    kind = value.strip().lower()
    if kind not in MEASUREMENT_TYPES:
        raise ValueError(f"measurement_type must be one of {MEASUREMENT_TYPES}: {value!r}")
    return kind


def parse_float(row: dict, field: str, *, allow_empty: bool = False) -> float | None:
    text = str(row.get(field, "")).strip()
    if not text and allow_empty:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"invalid number in {field}: {text!r}") from exc


def read_csv(path: Path, required: Iterable[str], optional: Iterable[str]) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = [name for name in required if name not in fields]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            row = {name: str(raw.get(name, "")).strip() for name in (*required, *optional)}
            row["_line"] = line_number
            rows.append(row)
    return rows


def load_ground_truth(path: Path, um_per_px: float) -> list[dict]:
    rows = read_csv(path, GT_REQUIRED, GT_OPTIONAL)
    seen = set()
    for row in rows:
        try:
            row["measurement_type"] = normalize_type(row["measurement_type"])
            row["y_fraction"] = normalize_y_fraction(row["y_fraction"])
            for field in ("answer_width_px", "answer_width_um", "center_x_px", "left_x_px", "right_x_px"):
                row[field] = parse_float(row, field, allow_empty=True)
        except ValueError as exc:
            raise ValueError(f"{path}:{row['_line']}: {exc}") from exc
        if row["left_x_px"] is not None and row["right_x_px"] is not None:
            if row["right_x_px"] <= row["left_x_px"]:
                raise ValueError(f"{path}:{row['_line']}: right_x_px must exceed left_x_px")
            edge_width = row["right_x_px"] - row["left_x_px"]
            if row["center_x_px"] is None:
                row["center_x_px"] = (row["left_x_px"] + row["right_x_px"]) / 2.0
        else:
            edge_width = None
        if row["answer_width_px"] is not None:
            row["width_px"] = row["answer_width_px"]
        elif row["answer_width_um"] is not None:
            row["width_px"] = row["answer_width_um"] / um_per_px
        elif edge_width is not None:
            row["width_px"] = edge_width
        else:
            raise ValueError(
                f"{path}:{row['_line']}: enter answer_width_px, answer_width_um, or both edges"
            )
        if row["width_px"] <= 0:
            raise ValueError(f"{path}:{row['_line']}: answer width must be positive")
        key = (
            row["image_id"],
            row["measurement_type"],
            row["y_fraction"],
            row["target_id"],
        )
        if key in seen:
            raise ValueError(f"{path}:{row['_line']}: duplicate ground-truth key {key}")
        seen.add(key)
    return rows


def load_predictions(path: Path, um_per_px: float) -> list[dict]:
    rows = read_csv(path, PRED_REQUIRED, PRED_OPTIONAL)
    for row in rows:
        try:
            row["measurement_type"] = normalize_type(row["measurement_type"])
            row["y_fraction"] = normalize_y_fraction(row["y_fraction"])
            for field in ("width_px", "width_um", "center_x_px", "left_x_px", "right_x_px"):
                row[field] = parse_float(row, field, allow_empty=True)
            row["runtime_ms"] = parse_float(row, "runtime_ms", allow_empty=True)
        except ValueError as exc:
            raise ValueError(f"{path}:{row['_line']}: {exc}") from exc
        row["status"] = (row["status"] or "ok").lower()
        if row["left_x_px"] is not None and row["right_x_px"] is not None:
            if row["right_x_px"] <= row["left_x_px"]:
                raise ValueError(f"{path}:{row['_line']}: right_x_px must exceed left_x_px")
            edge_width = row["right_x_px"] - row["left_x_px"]
            if row["center_x_px"] is None:
                row["center_x_px"] = (row["left_x_px"] + row["right_x_px"]) / 2.0
        else:
            edge_width = None
        if row["width_px"] is None and row["width_um"] is not None:
            row["width_px"] = row["width_um"] / um_per_px
        if row["width_px"] is None:
            row["width_px"] = edge_width
        if row["status"] == "ok" and (row["width_px"] is None or row["width_px"] <= 0):
            raise ValueError(
                f"{path}:{row['_line']}: enter width_px, width_um, or both edges"
            )
    return rows


def load_manifest(path: Path | None) -> dict[str, dict]:
    """Load image-level metadata used to split laptop/lighting/repeat results."""
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "image_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing column image_id")
        output = {}
        for line_number, raw in enumerate(reader, start=2):
            image_id = str(raw.get("image_id", "")).strip()
            if not image_id:
                raise ValueError(f"{path}:{line_number}: empty image_id")
            if image_id in output:
                raise ValueError(f"{path}:{line_number}: duplicate image_id {image_id!r}")
            output[image_id] = {
                str(key): str(value or "").strip()
                for key, value in raw.items()
                if key and key != "image_id"
            }
    return output


def apply_manifest(rows: list[dict], manifest: dict[str, dict]) -> None:
    for row in rows:
        metadata = manifest.get(row["image_id"], {})
        for name in GT_OPTIONAL:
            if name in metadata and not row.get(name):
                row[name] = metadata[name]


def parse_prediction_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use METHOD=predictions.csv")
    method, path = value.split("=", 1)
    method = method.strip()
    if not method:
        raise argparse.ArgumentTypeError("method name cannot be empty")
    return method, Path(path.strip())


def match_one_group(gt_rows: list[dict], pred_rows: list[dict], max_distance_px: float):
    """Match exact target IDs first, then use centers when IDs are unavailable."""
    used_gt, used_pred, pairs = set(), set(), []
    pred_by_target = {
        row["target_id"]: index
        for index, row in enumerate(pred_rows)
        if row.get("target_id")
    }
    for gi, gt in enumerate(gt_rows):
        pi = pred_by_target.get(gt.get("target_id"))
        if pi is None or pi in used_pred:
            continue
        distance = None
        if gt.get("center_x_px") is not None and pred_rows[pi].get("center_x_px") is not None:
            distance = abs(gt["center_x_px"] - pred_rows[pi]["center_x_px"])
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gt, pred_rows[pi], distance))

    candidates = []
    for gi, gt in enumerate(gt_rows):
        if gi in used_gt or gt.get("center_x_px") is None:
            continue
        for pi, pred in enumerate(pred_rows):
            if pi in used_pred or pred.get("center_x_px") is None:
                continue
            distance = abs(gt["center_x_px"] - pred["center_x_px"])
            if distance <= max_distance_px:
                candidates.append((distance, gi, pi))
    candidates.sort()
    for distance, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gt_rows[gi], pred_rows[pi], distance))
    missed = [row for i, row in enumerate(gt_rows) if i not in used_gt]
    false_positive = [row for i, row in enumerate(pred_rows) if i not in used_pred]
    return pairs, missed, false_positive


def match_method(
    method: str,
    ground_truth: list[dict],
    predictions: list[dict],
    max_distance_px: float,
    um_per_px: float,
) -> list[dict]:
    gt_groups = defaultdict(list)
    pred_groups = defaultdict(list)
    for row in ground_truth:
        gt_groups[(row["image_id"], row["measurement_type"], row["y_fraction"])].append(row)
    for row in predictions:
        if row["status"] == "ok":
            pred_groups[(row["image_id"], row["measurement_type"], row["y_fraction"])].append(row)

    output = []
    all_keys = sorted(set(gt_groups) | set(pred_groups))
    for key in all_keys:
        pairs, missed, extras = match_one_group(
            sorted(
                gt_groups[key],
                key=lambda item: (
                    item.get("center_x_px") is None,
                    item.get("center_x_px") or 0.0,
                    item.get("target_id", ""),
                ),
            ),
            sorted(
                pred_groups[key],
                key=lambda item: (
                    item.get("center_x_px") is None,
                    item.get("center_x_px") or 0.0,
                    item.get("target_id", ""),
                ),
            ),
            max_distance_px,
        )
        for gt, pred, center_distance in pairs:
            error_px = pred["width_px"] - gt["width_px"]
            output.append(
                {
                    "method": method,
                    "image_id": gt["image_id"],
                    "measurement_type": gt["measurement_type"],
                    "y_fraction": gt["y_fraction"],
                    "target_id": gt["target_id"],
                    "match_status": "matched",
                    "center_distance_px": rounded(center_distance),
                    "gt_left_x_px": rounded(gt["left_x_px"]),
                    "gt_right_x_px": rounded(gt["right_x_px"]),
                    "gt_width_px": rounded(gt["width_px"]),
                    "pred_left_x_px": rounded(pred.get("left_x_px")),
                    "pred_right_x_px": rounded(pred.get("right_x_px")),
                    "pred_width_px": rounded(pred["width_px"]),
                    "error_px": rounded(error_px),
                    "abs_error_px": rounded(abs(error_px)),
                    "error_um": rounded(error_px * um_per_px),
                    "abs_error_um": rounded(abs(error_px) * um_per_px),
                    "runtime_ms": rounded(pred["runtime_ms"]),
                    **{name: gt.get(name, "") for name in GT_OPTIONAL},
                }
            )
        for gt in missed:
            output.append(
                {
                    "method": method,
                    "image_id": gt["image_id"],
                    "measurement_type": gt["measurement_type"],
                    "y_fraction": gt["y_fraction"],
                    "target_id": gt["target_id"],
                    "match_status": "missed",
                    "gt_left_x_px": rounded(gt["left_x_px"]),
                    "gt_right_x_px": rounded(gt["right_x_px"]),
                    "gt_width_px": rounded(gt["width_px"]),
                    **{name: gt.get(name, "") for name in GT_OPTIONAL},
                }
            )
        for pred in extras:
            output.append(
                {
                    "method": method,
                    "image_id": pred["image_id"],
                    "measurement_type": pred["measurement_type"],
                    "y_fraction": pred["y_fraction"],
                    "target_id": "",
                    "match_status": "false_positive",
                    "pred_left_x_px": rounded(pred.get("left_x_px")),
                    "pred_right_x_px": rounded(pred.get("right_x_px")),
                    "pred_width_px": rounded(pred["width_px"]),
                    "runtime_ms": rounded(pred["runtime_ms"]),
                }
            )
    return output


def metric_block(rows: list[dict], um_per_px: float) -> dict:
    matched = [row for row in rows if row["match_status"] == "matched"]
    missed = [row for row in rows if row["match_status"] == "missed"]
    false_positive = [row for row in rows if row["match_status"] == "false_positive"]
    errors = [float(row["error_px"]) for row in matched]
    abs_errors = [abs(value) for value in errors]
    squared = [value * value for value in errors]
    n_gt = len(matched) + len(missed)
    n_pred = len(matched) + len(false_positive)
    return {
        "n_ground_truth": n_gt,
        "n_matched": len(matched),
        "n_missed": len(missed),
        "n_false_positive": len(false_positive),
        "measurement_success_rate": rounded(len(matched) / n_gt if n_gt else None),
        "precision": rounded(len(matched) / n_pred if n_pred else None),
        "mae_px": rounded(safe_mean(abs_errors)),
        "mae_um": rounded(safe_mean(abs_errors) * um_per_px if abs_errors else None),
        "mse_px2": rounded(safe_mean(squared)),
        "mse_um2": rounded(safe_mean(squared) * (um_per_px ** 2) if squared else None),
        "rmse_px": rounded(math.sqrt(safe_mean(squared)) if squared else None),
        "rmse_um": rounded(math.sqrt(safe_mean(squared)) * um_per_px if squared else None),
        "bias_px": rounded(safe_mean(errors)),
        "bias_um": rounded(safe_mean(errors) * um_per_px if errors else None),
        "p95_abs_error_px": rounded(percentile(abs_errors, 0.95)),
        "p95_abs_error_um": rounded(
            percentile(abs_errors, 0.95) * um_per_px if abs_errors else None
        ),
    }


def runtime_metrics(predictions: list[dict]) -> dict:
    per_image = defaultdict(list)
    for row in predictions:
        if row["runtime_ms"] is not None:
            per_image[row["image_id"]].append(float(row["runtime_ms"]))
    # A method may repeat the same image runtime on every output row. Collapse it once.
    image_times = [statistics.median(values) for values in per_image.values()]
    return {
        "n_timed_images": len(image_times),
        "mean_runtime_ms": rounded(safe_mean(image_times)),
        "median_runtime_ms": rounded(statistics.median(image_times) if image_times else None),
        "p95_runtime_ms": rounded(percentile(image_times, 0.95)),
    }


def _prediction_width(row: dict) -> float | None:
    value = row.get("pred_width_px", row.get("width_px"))
    return None if value in (None, "") else float(value)


def repeatability_metrics(rows: list[dict], um_per_px: float) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row.get("match_status", "matched") != "matched" or not row.get("repeat_group"):
            continue
        width = _prediction_width(row)
        if width is None or row.get("status", "ok") != "ok":
            continue
        key = (
            row["repeat_group"],
            row["measurement_type"],
            row["y_fraction"],
            row["target_id"],
        )
        groups[key].append(width)
    sds = [safe_stdev(values) for values in groups.values() if len(values) >= 2]
    sds = [value for value in sds if value is not None]
    return {
        "n_repeat_groups": len(sds),
        "mean_repeatability_sd_px": rounded(safe_mean(sds)),
        "mean_repeatability_sd_um": rounded(safe_mean(sds) * um_per_px if sds else None),
        "p95_repeatability_sd_px": rounded(percentile(sds, 0.95)),
        "p95_repeatability_sd_um": rounded(
            percentile(sds, 0.95) * um_per_px if sds else None
        ),
    }


def brightness_metrics(rows: list[dict], um_per_px: float) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row.get("match_status", "matched") != "matched" or not row.get("brightness_group"):
            continue
        width = _prediction_width(row)
        if width is None or row.get("status", "ok") != "ok":
            continue
        key = (
            row["brightness_group"],
            row["measurement_type"],
            row["y_fraction"],
            row["target_id"],
        )
        groups[key].append(width)
    ranges = [max(values) - min(values) for values in groups.values() if len(values) >= 2]
    return {
        "n_brightness_groups": len(ranges),
        "mean_brightness_range_px": rounded(safe_mean(ranges)),
        "mean_brightness_range_um": rounded(safe_mean(ranges) * um_per_px if ranges else None),
        "p95_brightness_range_px": rounded(percentile(ranges, 0.95)),
        "p95_brightness_range_um": rounded(
            percentile(ranges, 0.95) * um_per_px if ranges else None
        ),
    }


def image_success_metrics(rows: list[dict], threshold: float) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row["match_status"] != "false_positive":
            groups[row["image_id"]].append(row)
    success = []
    for image_rows in groups.values():
        matched = sum(row["match_status"] == "matched" for row in image_rows)
        coverage = matched / len(image_rows) if image_rows else 0.0
        success.append(coverage >= threshold)
    return {
        "n_images": len(success),
        "image_success_threshold": threshold,
        "n_successful_images": sum(success),
        "image_success_rate": rounded(sum(success) / len(success) if success else None),
    }


def summarize_method(
    matched_rows: list[dict],
    predictions: list[dict],
    um_per_px: float,
    image_success_threshold: float,
) -> dict:
    summary = metric_block(matched_rows, um_per_px)
    summary.update(image_success_metrics(matched_rows, image_success_threshold))
    summary.update(runtime_metrics(predictions))
    # Repeatability and lighting robustness intentionally use predictions even
    # when those images have no manually measured ground truth.
    summary.update(repeatability_metrics(predictions, um_per_px))
    summary.update(brightness_metrics(predictions, um_per_px))
    summary["by_measurement_type"] = {
        kind: metric_block(
            [row for row in matched_rows if row["measurement_type"] == kind], um_per_px
        )
        for kind in MEASUREMENT_TYPES
        if any(row["measurement_type"] == kind for row in matched_rows)
    }
    summary["by_brightness_level"] = {}
    levels = sorted(
        {
            row.get("brightness_level", "")
            for row in matched_rows
            if row.get("brightness_level", "")
        }
    )
    for level in levels:
        selected = [row for row in matched_rows if row.get("brightness_level") == level]
        summary["by_brightness_level"][level] = metric_block(selected, um_per_px)
    summary["by_laptop"] = {}
    laptops = sorted({row.get("laptop_id", "") for row in matched_rows if row.get("laptop_id")})
    for laptop in laptops:
        selected = [row for row in matched_rows if row.get("laptop_id") == laptop]
        summary["by_laptop"][laptop] = metric_block(selected, um_per_px)
    summary["by_evaluation_set"] = {}
    sets = sorted(
        {row.get("evaluation_set", "") for row in matched_rows if row.get("evaluation_set")}
    )
    for evaluation_set in sets:
        selected = [row for row in matched_rows if row.get("evaluation_set") == evaluation_set]
        summary["by_evaluation_set"][evaluation_set] = metric_block(selected, um_per_px)
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key.startswith("_") or key in seen:
                    continue
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary(method: str, summary: dict) -> dict:
    return {
        "method": method,
        **{key: value for key, value in summary.items() if not isinstance(value, dict)},
    }


def build_image_metrics(rows: list[dict], um_per_px: float) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["match_status"] != "false_positive":
            grouped[(row["method"], row["image_id"], row["measurement_type"])].append(row)
    output = []
    for (method, image_id, kind), selected in sorted(grouped.items()):
        block = metric_block(selected, um_per_px)
        output.append(
            {
                "method": method,
                "image_id": image_id,
                "measurement_type": kind,
                "condition": selected[0].get("condition", ""),
                "brightness_level": selected[0].get("brightness_level", ""),
                "laptop_id": selected[0].get("laptop_id", ""),
                "evaluation_set": selected[0].get("evaluation_set", ""),
                **block,
            }
        )
    return output


def build_group_metrics(rows: list[dict], um_per_px: float, field: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(field, "")
        if value and row["match_status"] != "false_positive":
            grouped[(row["method"], value)].append(row)
    return [
        {"method": method, field: value, **metric_block(selected, um_per_px)}
        for (method, value), selected in sorted(grouped.items())
    ]


def make_templates(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    gt_fields = [*GT_REQUIRED, *GT_OPTIONAL]
    pred_fields = [*PRED_REQUIRED, *PRED_OPTIONAL]
    gt_example = {
        "image_id": "sample_001",
        "measurement_type": "line",
        "y_fraction": "0.20",
        "target_id": "L1",
        "answer_width_px": "55.0",
        "answer_width_um": "8.756",
        "center_x_px": "127.5",
        "image_path": "images/sample_001.png",
        "reference_image_path": "references/sample_001_i.png",
        "laptop_id": "laptop_1",
        "evaluation_set": "accuracy",
        "pair_group": "site_001",
        "repeat_group": "site_A",
        "repeat_id": "1",
        "brightness_group": "site_A",
        "brightness_level": "normal",
        "condition": "normal",
        "annotator": "reviewer_A",
    }
    pred_example = {
        "image_id": "sample_001",
        "measurement_type": "line",
        "y_fraction": "0.20",
        "target_id": "L1",
        "method": "method_name",
        "width_px": "54.3",
        "width_um": "8.64456",
        "center_x_px": "127.55",
        "left_x_px": "100.4",
        "right_x_px": "154.7",
        "runtime_ms": "120.5",
        "status": "ok",
        "message": "",
    }
    write_csv(directory / "ground_truth_template.csv", [gt_example], gt_fields)
    write_csv(directory / "prediction_template.csv", [pred_example], pred_fields)
    schema = {
        "coordinate_system": "native image pixels; never resized capture pixels",
        "calibration_um_per_px": CALIBRATION_UM_PER_PX,
        "measurement_types": list(MEASUREMENT_TYPES),
        "y_fractions": list(EXPECTED_Y_FRACTIONS),
        "ground_truth_required": list(GT_REQUIRED),
        "ground_truth_optional": list(GT_OPTIONAL),
        "prediction_required": list(PRED_REQUIRED),
        "prediction_optional": list(PRED_OPTIONAL),
        "runtime_rule": "one complete image inference, preprocessing through final widths; warm-up first",
    }
    (directory / "benchmark_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Templates written to: {directory.resolve()}")


def validate_standard(ground_truth: list[dict]) -> list[str]:
    issues = []
    actual_rows = {row["y_fraction"] for row in ground_truth}
    missing_rows = [value for value in EXPECTED_Y_FRACTIONS if value not in actual_rows]
    unexpected_rows = sorted(actual_rows - set(EXPECTED_Y_FRACTIONS))
    if missing_rows:
        issues.append(f"ground truth does not include standard rows: {missing_rows}")
    if unexpected_rows:
        issues.append(f"ground truth contains non-standard rows: {unexpected_rows}")

    counts = defaultdict(set)
    for row in ground_truth:
        counts[(row["image_id"], row["measurement_type"], row["y_fraction"])].add(
            row["target_id"]
        )
    incomplete = [key for key, targets in counts.items() if len(targets) != 5]
    if incomplete:
        preview = ", ".join(
            f"{image}/{kind}/{y_fraction:.2f}" for image, kind, y_fraction in incomplete[:10]
        )
        issues.append(
            f"expected exactly five representative targets in every image/type/Y group; "
            f"found {len(incomplete)} incomplete groups: {preview}"
        )
    image_types = {(row["image_id"], row["measurement_type"]) for row in ground_truth}
    for image_id, kind in sorted(image_types):
        present = {
            row["y_fraction"]
            for row in ground_truth
            if row["image_id"] == image_id and row["measurement_type"] == kind
        }
        missing = set(EXPECTED_Y_FRACTIONS) - present
        if missing:
            issues.append(
                f"{image_id}/{kind} is missing standard Y rows: {sorted(missing)}"
            )
    return issues


def main_compare() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple width measurement methods using one fixed protocol"
    )
    parser.add_argument("--make-templates", type=Path)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional image manifest containing laptop/repeat/brightness metadata.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        type=parse_prediction_spec,
        default=[],
        metavar="METHOD=CSV",
        help="repeat once per method",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--um-per-px", type=float, default=CALIBRATION_UM_PER_PX)
    parser.add_argument("--match-max-distance-px", type=float, default=30.0)
    parser.add_argument("--image-success-threshold", type=float, default=0.80)
    args = parser.parse_args()

    if args.make_templates:
        make_templates(args.make_templates)
        if not args.ground_truth and not args.prediction:
            return
    if not args.ground_truth or not args.prediction:
        parser.error("evaluation requires --ground-truth and at least one --prediction")
    if args.um_per_px <= 0 or args.match_max_distance_px <= 0:
        parser.error("calibration and match distance must be positive")
    if not 0 < args.image_success_threshold <= 1:
        parser.error("--image-success-threshold must be in (0, 1]")

    method_names = [method for method, _ in args.prediction]
    if len(method_names) != len(set(method_names)):
        parser.error("method names in --prediction must be unique")

    manifest = load_manifest(args.manifest)
    ground_truth = load_ground_truth(args.ground_truth, args.um_per_px)
    apply_manifest(ground_truth, manifest)
    standard_issues = validate_standard(ground_truth)
    args.output.mkdir(parents=True, exist_ok=True)

    all_matches = []
    summaries = {}
    for method, prediction_path in args.prediction:
        predictions = load_predictions(prediction_path, args.um_per_px)
        apply_manifest(predictions, manifest)
        ground_truth_image_ids = {row["image_id"] for row in ground_truth}
        accuracy_predictions = [
            row for row in predictions if row["image_id"] in ground_truth_image_ids
        ]
        matches = match_method(
            method,
            ground_truth,
            accuracy_predictions,
            args.match_max_distance_px,
            args.um_per_px,
        )
        all_matches.extend(matches)
        summaries[method] = summarize_method(
            matches, predictions, args.um_per_px, args.image_success_threshold
        )

    write_csv(args.output / "matched_measurements.csv", all_matches)
    write_csv(args.output / "image_metrics.csv", build_image_metrics(all_matches, args.um_per_px))
    write_csv(
        args.output / "laptop_metrics.csv",
        build_group_metrics(all_matches, args.um_per_px, "laptop_id"),
    )
    write_csv(
        args.output / "evaluation_set_metrics.csv",
        build_group_metrics(all_matches, args.um_per_px, "evaluation_set"),
    )
    write_csv(
        args.output / "method_summary.csv",
        [flatten_summary(method, summary) for method, summary in summaries.items()],
    )
    report = {
        "protocol": {
            "coordinate_system": "native image pixels",
            "um_per_px": args.um_per_px,
            "y_fractions": list(EXPECTED_Y_FRACTIONS),
            "representative_targets_per_image_and_type": 5,
            "match_max_distance_px": args.match_max_distance_px,
            "image_success_threshold": args.image_success_threshold,
            "repeatability_sd": "sample SD across repeated captures of the same physical target",
            "brightness_robustness": "range across brightness levels of the same physical target",
        },
        "standard_validation_issues": standard_issues,
        "methods": summaries,
    }
    (args.output / "benchmark_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Compared methods: {', '.join(method_names)}")
    for method in method_names:
        row = summaries[method]
        print(
            f"{method}: MAE={row['mae_um']} um, success={row['measurement_success_rate']}, "
            f"repeat SD={row['mean_repeatability_sd_um']} um, runtime={row['mean_runtime_ms']} ms"
        )
    if standard_issues:
        print("Protocol warnings:")
        for issue in standard_issues:
            print(f"- {issue}")
    print(f"Results: {args.output.resolve()}")


# ===========================================================================
# 2. 전체 성능 스위트 오케스트레이션 (옛 run_performance_suite.py)
# ===========================================================================
#
# annotated *_i 스크린샷을 모델 입력에서 제외하고, 전역 고유 image ID를
# 부여하고, 방법별 공통 예측 CSV를 만든 뒤 laptop/repeat/lighting 메타데이터와
# 함께 위 main_compare()를 (이 파일의 compare 명령을 별도 프로세스로) 실행한다.

SUITE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUITE_Y_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
MANIFEST_FIELDS = (
    "image_id",
    "image_path",
    "reference_image_path",
    "evaluation_set",
    "laptop_id",
    "pair_group",
    "repeat_group",
    "repeat_id",
    "brightness_group",
    "brightness_level",
    "condition",
)
SUITE_GT_FIELDS = (
    "image_id",
    "measurement_type",
    "y_fraction",
    "target_id",
    "answer_width_px",
    "answer_width_um",
    "center_x_px",
    "left_x_px",
    "right_x_px",
    "image_path",
    "reference_image_path",
    "laptop_id",
    "evaluation_set",
    "pair_group",
    "repeat_group",
    "repeat_id",
    "brightness_group",
    "brightness_level",
    "condition",
    "annotator",
)


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path.resolve()
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUITE_IMAGE_SUFFIXES
            and not path.stem.lower().endswith("_i")
        ),
        key=lambda path: str(path).lower(),
    )


def _find_child(directory: Path, compact_name: str) -> Path:
    expected = compact_name.replace(" ", "").lower()
    for child in directory.iterdir():
        if child.is_dir() and child.name.replace(" ", "").lower() == expected:
            return child
    raise FileNotFoundError(f"Could not find folder {compact_name!r} under {directory}")


def discover_dataset(data_root: Path) -> list[dict[str, str]]:
    data_root = data_root.resolve()
    originals = _find_child(data_root, "원본")
    repeat_dir = _find_child(data_root, "반복성")
    light_dir = _find_child(data_root, "조명민감도")
    rows: list[dict[str, str]] = []

    laptop_dirs = sorted(
        (path for path in originals.iterdir() if path.is_dir()),
        key=lambda path: path.name.replace(" ", "").lower(),
    )
    for laptop_number, laptop_dir in enumerate(laptop_dirs, start=1):
        laptop_id = f"laptop{laptop_number}"
        for image_path in _images(laptop_dir):
            reference = image_path.with_name(f"{image_path.stem}_i{image_path.suffix}")
            rows.append(
                {
                    "image_id": f"{laptop_id}__{image_path.stem}",
                    "image_path": str(image_path),
                    "reference_image_path": str(reference) if reference.is_file() else "",
                    "evaluation_set": "accuracy",
                    "laptop_id": laptop_id,
                    "pair_group": "",
                    "repeat_group": "",
                    "repeat_id": "",
                    "brightness_group": "",
                    "brightness_level": "",
                    "condition": laptop_id,
                }
            )

    for image_path in _images(repeat_dir):
        match = re.fullmatch(r"(?P<site>\d+)(?:_(?P<repeat>\d+))?", image_path.stem)
        site = match.group("site") if match else image_path.stem
        repeat_id = (match.group("repeat") or "0") if match else image_path.stem
        rows.append(
            {
                "image_id": f"repeatability__{image_path.stem}",
                "image_path": str(image_path),
                "reference_image_path": "",
                "evaluation_set": "repeatability",
                "laptop_id": "",
                "pair_group": "",
                "repeat_group": f"repeat_site_{site}",
                "repeat_id": repeat_id,
                "brightness_group": "",
                "brightness_level": "",
                "condition": "repeatability",
            }
        )

    light_names = {"b": "bright", "d": "dark", "n": "normal"}
    for image_path in _images(light_dir):
        match = re.fullmatch(r"(?P<site>\d+)(?P<level>[bdn])", image_path.stem.lower())
        site = match.group("site") if match else image_path.stem
        level = light_names.get(match.group("level"), "unknown") if match else "unknown"
        rows.append(
            {
                "image_id": f"lighting__{image_path.stem}",
                "image_path": str(image_path),
                "reference_image_path": "",
                "evaluation_set": "lighting",
                "laptop_id": "",
                "pair_group": "",
                "repeat_group": "",
                "repeat_id": "",
                "brightness_group": f"lighting_site_{site}",
                "brightness_level": level,
                "condition": level,
            }
        )

    ids = [row["image_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Generated image IDs are not unique")
    return rows


def _suite_write_csv(path: Path, rows: list[dict], fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare(data_root: Path, output_root: Path, force: bool = False) -> tuple[Path, Path]:
    rows = discover_dataset(data_root)
    manifest_path = output_root / "dataset_manifest.csv"
    ground_truth_path = output_root / "ground_truth_sidewall.csv"
    _suite_write_csv(manifest_path, rows, MANIFEST_FIELDS)

    if force or not ground_truth_path.exists():
        ground_truth_rows = []
        for item in rows:
            if item["evaluation_set"] != "accuracy":
                continue
            for y_fraction in SUITE_Y_FRACTIONS:
                for target_number in range(1, 6):
                    ground_truth_rows.append(
                        {
                            **item,
                            "measurement_type": "sidewall",
                            "y_fraction": f"{y_fraction:.2f}",
                            "target_id": f"S{target_number}",
                            "answer_width_px": "",
                            "answer_width_um": "",
                            "center_x_px": "",
                            "left_x_px": "",
                            "right_x_px": "",
                            "annotator": "manual",
                        }
                    )
        _suite_write_csv(ground_truth_path, ground_truth_rows, SUITE_GT_FIELDS)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["evaluation_set"]] = counts.get(row["evaluation_set"], 0) + 1
    print(f"Manifest: {manifest_path}")
    print(f"Ground-truth entry file: {ground_truth_path}")
    print("Model inputs: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print("Annotated *_i images are references only and were excluded from model inputs.")
    return manifest_path, ground_truth_path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _error_rows(image_id: str, measurement_types: list[str], message: str) -> list[dict]:
    return [
        {
            "image_id": image_id,
            "measurement_type": kind,
            "y_fraction": y_fraction,
            "target_id": f"{TARGET_PREFIX[kind]}{target}",
            "status": "error",
            "message": message,
        }
        for kind in measurement_types
        for y_fraction in SUITE_Y_FRACTIONS
        for target in range(1, 6)
    ]


def _unet_rows_from_report(
    image_id: str,
    report: dict,
    measurement_types: list[str],
    runtime_ms: float | None,
) -> list[dict]:
    """Convert one U-Net report to exactly five protocol rows per Y/type.

    Degenerate detections (zero/negative width or invalid coordinates) are kept
    as explicit error rows instead of aborting the entire prediction CSV.
    """
    from unet_inference import _central_targets

    output: list[dict] = []
    for y_fraction, report_row in zip(SUITE_Y_FRACTIONS, report["rows"], strict=True):
        for kind in measurement_types:
            targets = list(_central_targets(report_row["measurements_by_type"][kind]))
            for target_number in range(1, 6):
                base = {
                    "image_id": image_id,
                    "measurement_type": kind,
                    "y_fraction": y_fraction,
                    "target_id": f"{TARGET_PREFIX[kind]}{target_number}",
                    "runtime_ms": runtime_ms,
                }
                if target_number > len(targets):
                    output.append(
                        {
                            **base,
                            "status": "error",
                            "message": "U-Net returned fewer than five usable central targets",
                        }
                    )
                    continue

                target = targets[target_number - 1]
                width = float(target["width_px_native"])
                left = float(target["left_x_native"])
                right = float(target["right_x_native"])
                center = float(target["center_x_native"])
                valid = (
                    all(math.isfinite(value) for value in (width, left, right, center))
                    and width > 0
                    and right > left
                )
                if not valid:
                    output.append(
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

                output.append(
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
    return output


def recover_unet_predictions(
    manifest: list[dict[str, str]],
    output_root: Path,
    measurement_types: list[str],
) -> Path:
    """Rebuild the U-Net prediction CSV from already-saved JSON reports."""
    rows: list[dict] = []
    method_root = output_root / "unet"
    for index, item in enumerate(manifest):
        image_id = item["image_id"]
        report_path = method_root / "images" / image_id / f"{image_id}_measurement.json"
        if not report_path.is_file():
            rows.extend(_error_rows(image_id, measurement_types, "Saved U-Net report not found"))
            print(f"[U-Net recovery {index + 1}/{len(manifest)}] MISSING {image_id}")
            continue
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            runtime_ms = payload.get("runtime", {}).get("measurement_ms_excluding_file_save")
            rows.extend(
                _unet_rows_from_report(
                    image_id, payload, measurement_types, runtime_ms=runtime_ms
                )
            )
            print(f"[U-Net recovery {index + 1}/{len(manifest)}] {image_id}")
        except Exception as error:
            rows.extend(_error_rows(image_id, measurement_types, str(error)))
            print(f"[U-Net recovery {index + 1}/{len(manifest)}] ERROR {image_id}: {error}")
    path = method_root / "unet_predictions.csv"
    return save_prediction_csv(path, "unet", rows)


def run_unet(
    manifest: list[dict[str, str]],
    output_root: Path,
    measurement_types: list[str],
    checkpoint: Path | None,
    device_name: str,
) -> Path:
    import torch

    from unet_inference import (
        DEFAULT_CHECKPOINT,
        _save_artifacts,
        load_unet,
        measure_unet_image,
    )

    selected_device = None if device_name == "auto" else torch.device(device_name)
    model, checkpoint_data, device = load_unet(
        checkpoint or DEFAULT_CHECKPOINT, device=selected_device
    )
    rows: list[dict] = []
    method_root = output_root / "unet"
    for index, item in enumerate(manifest):
        image_id = item["image_id"]
        image_path = Path(item["image_path"])
        item_root = method_root / "images" / image_id
        try:
            started = time.perf_counter()
            measurement = measure_unet_image(
                image_path,
                model,
                checkpoint_data,
                device,
                measurement_types=measurement_types,
            )
            runtime_ms = None if index == 0 else (time.perf_counter() - started) * 1000.0
            rows.extend(
                _unet_rows_from_report(
                    image_id,
                    measurement["report"],
                    measurement_types,
                    runtime_ms=runtime_ms,
                )
            )
            _save_artifacts(measurement, item_root, image_id)
            print(f"[U-Net {index + 1}/{len(manifest)}] {image_id}")
        except Exception as error:
            rows.extend(_error_rows(image_id, measurement_types, str(error)))
            print(f"[U-Net {index + 1}/{len(manifest)}] ERROR {image_id}: {error}")
    path = method_root / "unet_predictions.csv"
    return save_prediction_csv(path, "unet", rows)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_canny(
    manifest: list[dict[str, str]], output_root: Path, measurement_types: list[str]
) -> Path:
    source = (
        PROJECT_ROOT
        / "comparison_methods"
        / "canny_method2"
        / "scripts"
        / "run_method2_team_protocol_compare.py"
    )
    module = _load_module("performance_canny_method", source)
    rows: list[dict] = []
    method_root = output_root / "canny"
    for index, item in enumerate(manifest):
        image_id = item["image_id"]
        try:
            result = module.compare_image(
                Path(item["image_path"]),
                measurement_types,
                duplicate_factor=None,
                output_root=method_root / "images" / image_id,
                runtime_is_warmup=index == 0,
            )
            for row in result.rows:
                rows.append({**row, "image_id": image_id})
            print(f"[Canny {index + 1}/{len(manifest)}] {image_id}")
        except Exception as error:
            rows.extend(_error_rows(image_id, measurement_types, str(error)))
            print(f"[Canny {index + 1}/{len(manifest)}] ERROR {image_id}: {error}")
    path = method_root / "canny_predictions.csv"
    return save_prediction_csv(path, "canny", rows)


def _middle_five(rows: list[dict]) -> list[dict]:
    def physical_order(row: dict) -> tuple[int, float]:
        match = re.search(r"(\d+)$", str(row.get("target_id", "")))
        if match:
            return (0, float(match.group(1)))
        left = float(row.get("left_edge_px", np.nan))
        return (1, left if np.isfinite(left) else float("inf"))

    ordered = sorted(
        rows,
        key=physical_order,
    )
    if len(ordered) > 5:
        start = (len(ordered) - 5) // 2
        ordered = ordered[start : start + 5]
    return ordered


def run_threshold(
    manifest: list[dict[str, str]], output_root: Path, measurement_types: list[str]
) -> Path:
    source = PROJECT_ROOT / "comparison_methods" / "threshold_method" / "unified_measurement.py"
    module = _load_module("performance_threshold_method", source)
    rows: list[dict] = []
    method_root = output_root / "threshold"
    for index, item in enumerate(manifest):
        image_id = item["image_id"]
        try:
            details, _summary, metadata = module.process_image(
                Path(item["image_path"]), measurement_types, method_root / "images" / image_id
            )
            runtime_ms = None if index == 0 else metadata.get("measurement_runtime_ms")
            records = details.to_dict("records")
            for kind in measurement_types:
                for y_fraction in SUITE_Y_FRACTIONS:
                    selected = _middle_five(
                        [
                            row
                            for row in records
                            if row["measurement_type"] == kind
                            and abs(float(row["y_ratio"]) - y_fraction) < 1e-6
                        ]
                    )
                    for number, row in enumerate(selected, start=1):
                        ok = row["status"] == "ok" and np.isfinite(float(row["width_px"]))
                        left = float(row["left_edge_px"]) if ok else None
                        right = float(row["right_edge_px"]) if ok else None
                        rows.append(
                            {
                                "image_id": image_id,
                                "measurement_type": kind,
                                "y_fraction": y_fraction,
                                "target_id": f"{TARGET_PREFIX[kind]}{number}",
                                "width_px": float(row["width_px"]) if ok else None,
                                "center_x_px": (left + right) / 2.0 if ok else None,
                                "left_x_px": left,
                                "right_x_px": right,
                                "runtime_ms": runtime_ms,
                                "status": "ok" if ok else "error",
                                "message": row.get("reason", ""),
                            }
                        )
            print(f"[Threshold {index + 1}/{len(manifest)}] {image_id}")
        except Exception as error:
            rows.extend(_error_rows(image_id, measurement_types, str(error)))
            print(f"[Threshold {index + 1}/{len(manifest)}] ERROR {image_id}: {error}")
    path = method_root / "threshold_predictions.csv"
    return save_prediction_csv(path, "threshold", rows)


def validate_ground_truth(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    missing = [
        index
        for index, row in enumerate(rows, start=2)
        if not any(
            str(row.get(field, "")).strip()
            for field in ("answer_width_px", "answer_width_um")
        )
        and not (
            str(row.get("left_x_px", "")).strip()
            and str(row.get("right_x_px", "")).strip()
        )
    ]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(
            f"Ground truth has {len(missing)} empty measurement rows (first: {preview}). "
            "Enter answer_width_px from the matching *_i reference screenshots first."
        )


def compare(
    manifest_path: Path,
    ground_truth_path: Path,
    prediction_paths: dict[str, Path],
    output_root: Path,
) -> None:
    validate_ground_truth(ground_truth_path)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "compare",
        "--manifest",
        str(manifest_path),
        "--ground-truth",
        str(ground_truth_path),
        "--output",
        str(output_root / "benchmark"),
    ]
    for method, path in prediction_paths.items():
        command.extend(("--prediction", f"{method}={path}"))
    subprocess.run(command, check=True)


def _suite_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Create manifest and GT entry CSV")
    prepare_parser.add_argument("--data-root", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run methods and compare with filled GT")
    run_parser.add_argument("--data-root", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--ground-truth", type=Path, default=None)
    run_parser.add_argument("--methods", nargs="+", choices=("unet", "canny", "threshold"), default=("unet", "canny", "threshold"))
    run_parser.add_argument("--measurement-types", nargs="+", choices=("line", "gap", "sidewall"), default=("sidewall",))
    run_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="U-Net checkpoint (default: models/best_unet_manual.pt in the U-Net folder)",
    )
    run_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    run_parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse completed prediction CSVs and rebuild U-Net CSV from saved reports",
    )
    run_parser.add_argument("--skip-compare", action="store_true")
    return parser.parse_args()


def main_suite() -> None:
    args = _suite_parse_args()
    if args.command == "prepare":
        prepare(args.data_root, args.output, args.force)
        return

    manifest_path, template_gt = prepare(args.data_root, args.output, force=False)
    filled_gt = args.output / "ground_truth_sidewall_filled.csv"
    default_gt = filled_gt if filled_gt.is_file() else template_gt
    manifest = read_manifest(manifest_path)
    predictions: dict[str, Path] = {}
    if "unet" in args.methods:
        existing = args.output / "unet" / "unet_predictions.csv"
        if args.reuse_existing and existing.is_file():
            predictions["unet"] = existing
            print(f"[U-Net] Reusing {existing}")
        elif args.reuse_existing:
            predictions["unet"] = recover_unet_predictions(
                manifest, args.output, list(args.measurement_types)
            )
        else:
            predictions["unet"] = run_unet(
                manifest, args.output, list(args.measurement_types), args.checkpoint, args.device
            )
    if "canny" in args.methods:
        existing = args.output / "canny" / "canny_predictions.csv"
        if args.reuse_existing and existing.is_file():
            predictions["canny"] = existing
            print(f"[Canny] Reusing {existing}")
        else:
            predictions["canny"] = run_canny(
                manifest, args.output, list(args.measurement_types)
            )
    if "threshold" in args.methods:
        existing = args.output / "threshold" / "threshold_predictions.csv"
        if args.reuse_existing and existing.is_file():
            predictions["threshold"] = existing
            print(f"[Threshold] Reusing {existing}")
        else:
            predictions["threshold"] = run_threshold(
                manifest, args.output, list(args.measurement_types)
            )
    if not args.skip_compare:
        compare(manifest_path, args.ground_truth or default_gt, predictions, args.output)
    else:
        print("Predictions written; benchmark comparison skipped.")


# ===========================================================================
# 명령행 진입점
# ===========================================================================
#
# 첫 인자로 "compare" 또는 "suite"를 지정한다:
#   python evaluation/evaluate.py compare ...        -> main_compare()
#   python evaluation/evaluate.py suite prepare ...  -> main_suite()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("compare", "suite"):
        print("사용법: python evaluation/evaluate.py <compare|suite> [옵션...]")
        raise SystemExit(1)
    stage = sys.argv.pop(1)
    if stage == "compare":
        main_compare()
    else:
        main_suite()


if __name__ == "__main__":
    main()
