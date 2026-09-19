#!/usr/bin/env python3
"""Unified native-pixel metrology for line, gap, and sidewall targets."""
from pathlib import Path
from datetime import datetime
from html import escape
import argparse
import json
import re
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import team_protocol


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
TYPE_BY_DIGIT = {"1": "line", "2": "gap", "3": "sidewall"}
PREFIX_BY_TYPE = {"line": "L", "gap": "G", "sidewall": "S"}
COLOR_BY_TYPE = {"line": "lime", "gap": "cyan", "sidewall": "magenta"}
Y_RATIOS = (0.20, 0.35, 0.50, 0.65, 0.80)
UM_PER_NATIVE_PIXEL = 0.1592


def parse_measurement_selection(text):
    """Accept 1, 2, 3, 13, 123, '1 3', etc. and return canonical order."""
    compact = re.sub(r"[\s,;+/_-]+", "", str(text))
    if not compact or any(char not in TYPE_BY_DIGIT for char in compact):
        raise ValueError("측정 종류는 1, 2, 3 조합으로 입력하세요. 예: 1, 13, 123")
    selected = {TYPE_BY_DIGIT[char] for char in compact}
    return [TYPE_BY_DIGIT[digit] for digit in "123" if TYPE_BY_DIGIT[digit] in selected]


def read_image(path):
    path = Path(path)
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError("8비트 PNG/JPG/TIFF/BMP 이미지를 사용하세요.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return image
    raise ValueError("지원하지 않는 이미지 채널 형식입니다.")


def _axis_equal_transitions(image, axis):
    rgb = image[:, :, :3] if image.ndim == 3 else image[:, :, None]
    h, w = rgb.shape[:2]
    y0, y1 = int(h*.05), max(int(h*.95), int(h*.05)+2)
    x0, x1 = int(w*.05), max(int(w*.95), int(w*.05)+2)
    crop = rgb[y0:y1, x0:x1]
    if axis == 1:
        equal = np.all(crop[:, 1:] == crop[:, :-1], axis=(0, 2))
        start = x0
    else:
        equal = np.all(crop[1:] == crop[:-1], axis=(1, 2))
        start = y0
    return equal, start


def detect_duplicate_factor(image, axis, max_factor=8,
                            min_duplicate_rate=.90, max_boundary_rate=.10):
    """Detect exact nearest-neighbor pixel replication without resizing."""
    equal, crop_start = _axis_equal_transitions(image, axis)
    indices = np.arange(len(equal))
    candidates = []
    for factor in range(2, max_factor+1):
        for local_offset in range(factor):
            boundary = ((indices-local_offset) % factor) == factor-1
            duplicate = ~boundary
            if not boundary.any() or not duplicate.any():
                continue
            duplicate_rate = float(equal[duplicate].mean())
            boundary_rate = float(equal[boundary].mean())
            if duplicate_rate >= min_duplicate_rate and boundary_rate <= max_boundary_rate:
                global_offset = int((crop_start+local_offset) % factor)
                candidates.append((factor, duplicate_rate-boundary_rate,
                                   global_offset, duplicate_rate, boundary_rate))
    if not candidates:
        return dict(factor=1, offset=0, duplicate_rate=0.0, boundary_rate=0.0)
    factor, _, offset, duplicate_rate, boundary_rate = max(candidates, key=lambda row: (row[0], row[1]))
    return dict(factor=factor, offset=offset, duplicate_rate=duplicate_rate,
                boundary_rate=boundary_rate)


def restore_native_pixels(image):
    """Remove repeated saved pixels by integer slicing; never interpolate/resize."""
    x_info = detect_duplicate_factor(image, axis=1)
    y_info = detect_duplicate_factor(image, axis=0)
    native = image[y_info["offset"]::y_info["factor"],
                   x_info["offset"]::x_info["factor"]].copy()
    metadata = {
        "source_width": int(image.shape[1]), "source_height": int(image.shape[0]),
        "native_width": int(native.shape[1]), "native_height": int(native.shape[0]),
        "duplicate_factor_x": int(x_info["factor"]),
        "duplicate_factor_y": int(y_info["factor"]),
        "duplicate_offset_x": int(x_info["offset"]),
        "duplicate_offset_y": int(y_info["offset"]),
        "duplicate_x_rate": float(x_info["duplicate_rate"]),
        "duplicate_y_rate": float(y_info["duplicate_rate"]),
        "native_restore_method": "integer slicing only; no interpolation or resize",
    }
    return native, metadata


def native_y_positions(native_height):
    return [int(round((native_height-1)*ratio)) for ratio in Y_RATIOS]


def detect_sidewall_centers(green, min_distance=27, edge_margin_fraction=.03):
    h, w = green.shape
    reference = np.median(green[int(h*.10):max(int(h*.90), int(h*.10)+1)], axis=0)
    smooth = cv2.GaussianBlur(reference[None, :], (0, 0), 1).ravel()
    local_min = cv2.erode(
        smooth[None, :], np.ones((1, 2*min_distance+1), np.uint8)).ravel()
    contrast = max(10., float(np.ptp(smooth))*.12)
    mask = ((smooth <= local_min+1e-8) &
            (smooth < np.percentile(smooth, 80)-contrast))
    changes = np.diff(np.r_[False, mask, False].astype(int))
    centers = np.asarray([(a+b-1)//2 for a, b in zip(
        np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))], dtype=int)
    margin = max(3, int(round(w*edge_margin_fraction)))
    centers = centers[(centers >= margin) & (centers < w-margin)]
    if len(centers) < 2:
        raise ValueError(f"측정에는 sidewall이 2개 이상 필요합니다. 검출: {len(centers)}개")
    return centers, reference


def measure_dark_band(profile, expected_center, lo, hi, fraction=.70,
                      background_fraction=.20, core_pixels=3):
    p = np.asarray(profile[lo:hi], dtype=float)
    if len(p) < 10:
        raise ValueError("sidewall ROI가 너무 좁습니다.")
    local_expected = int(expected_center-lo)
    n = max(2, int(len(p)*background_fraction))
    if not n <= local_expected < len(p)-n:
        raise ValueError("sidewall 중심이 배경 영역에 있습니다.")
    bg_left, bg_right = float(np.median(p[:n])), float(np.median(p[-n:]))
    radius = max(3, int(len(p)*.20))
    s0, s1 = max(n, local_expected-radius), min(len(p)-n, local_expected+radius+1)
    center = s0+int(np.argmin(p[s0:s1]))
    half = core_pixels//2
    bottom = float(np.median(p[max(n, center-half):min(len(p)-n, center+half+1)]))
    background = (bg_left+bg_right)/2
    threshold = background-fraction*(background-bottom)
    if min(bg_left, bg_right)-bottom < 5:
        raise ValueError("sidewall 대비가 부족합니다.")
    below = p <= threshold
    transitions = np.diff(np.r_[False, below, False].astype(int))
    starts, stops = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    containing = [(a, b) for a, b in zip(starts, stops) if a <= center < b]
    if not containing:
        raise ValueError("sidewall 중심의 경계를 찾지 못했습니다.")
    a, b = map(int, containing[0])
    if a == 0 or b >= len(p) or a < n or b > len(p)-n:
        raise ValueError("sidewall 경계가 배경 영역에 닿습니다.")
    interior = [(s, e) for s, e in zip(starts, stops)
                if s < len(p)-n and e > n]
    if len(interior) != 1:
        raise ValueError("ROI 안에 분리된 어두운 구간이 여러 개입니다.")
    left_delta, right_delta = p[a-1]-p[a], p[b]-p[b-1]
    if left_delta == 0 or right_delta == 0:
        raise ValueError("edge 보간 기울기가 없습니다.")
    left = lo+a-1+(p[a-1]-threshold)/left_delta
    right = lo+b-1+(threshold-p[b-1])/right_delta
    return dict(left_edge_px=float(left), right_edge_px=float(right),
                width_px=float(right-left), threshold=float(threshold),
                center_px=float(lo+center))


def sidewall_boundaries(centers, width):
    return np.r_[0, (centers[:-1]+centers[1:])//2, width].astype(int)


def measure_all_sidewalls(profile, centers, boundaries):
    measured = {}
    for index, center in enumerate(centers):
        lo, hi = int(boundaries[index]), int(boundaries[index+1])
        try:
            measured[index] = dict(status="ok", reason="",
                                   **measure_dark_band(profile, int(center), lo, hi))
        except ValueError as error:
            measured[index] = dict(status="invalid", reason=str(error))
    return measured


def define_targets(reference_profile, centers, boundaries, measurement_types):
    reference_walls = measure_all_sidewalls(reference_profile, centers, boundaries)
    intervals = []
    for index in range(len(centers)-1):
        left, right = reference_walls[index], reference_walls[index+1]
        if left["status"] != "ok" or right["status"] != "ok":
            continue
        x0, x1 = left["right_edge_px"], right["left_edge_px"]
        if x1 <= x0:
            continue
        a, b = int(np.ceil(x0)), int(np.floor(x1))+1
        brightness = float(np.mean(reference_profile[a:b])) if b > a else np.nan
        intervals.append(dict(index=index, center=(x0+x1)/2,
                              brightness=brightness, parity=index % 2))
    parity_brightness = {
        parity: float(np.median([item["brightness"] for item in intervals
                                if item["parity"] == parity and np.isfinite(item["brightness"])]))
        for parity in (0, 1)
    }
    line_parity = max(parity_brightness, key=parity_brightness.get)
    gap_parity = 1-line_parity
    targets = {}
    if "sidewall" in measurement_types:
        all_sidewalls = [dict(index=i, center=float(center))
                         for i, center in enumerate(centers)
                         if reference_walls[i]["status"] == "ok"]
        targets["sidewall"] = [dict(item, target_id=f"S{i}")
                               for i, item in enumerate(all_sidewalls, 1)]
    for kind, parity, prefix in (("line", line_parity, "L"),
                                 ("gap", gap_parity, "G")):
        if kind in measurement_types:
            all_intervals = [item for item in intervals if item["parity"] == parity]
            targets[kind] = [dict(item, target_id=f"{prefix}{i}")
                             for i, item in enumerate(all_intervals, 1)]
    classification = dict(line_parity=int(line_parity), gap_parity=int(gap_parity),
                          parity_0_brightness=parity_brightness[0],
                          parity_1_brightness=parity_brightness[1])
    return targets, classification


def measure_targets_at_y(profile, measured_walls, targets, y_ratio, y_px,
                         um_per_pixel=UM_PER_NATIVE_PIXEL):
    rows = []
    for kind, items in targets.items():
        for item in items:
            if kind == "sidewall":
                wall = measured_walls[item["index"]]
                if wall["status"] == "ok":
                    left, right = wall["left_edge_px"], wall["right_edge_px"]
                    status, reason = "ok", ""
                else:
                    left = right = np.nan
                    status, reason = "invalid", wall["reason"]
            else:
                left_wall = measured_walls[item["index"]]
                right_wall = measured_walls[item["index"]+1]
                if left_wall["status"] == "ok" and right_wall["status"] == "ok":
                    left = left_wall["right_edge_px"]
                    right = right_wall["left_edge_px"]
                    status, reason = ("ok", "") if right > left else ("invalid", "edge 순서 오류")
                else:
                    left = right = np.nan
                    status, reason = "invalid", "인접 sidewall edge 검출 실패"
            width_px = float(right-left) if status == "ok" else np.nan
            rows.append(dict(measurement_type=kind, target_id=item["target_id"],
                             y_ratio=float(y_ratio), y_px_native=int(y_px),
                             left_edge_px=left, right_edge_px=right,
                             width_px=width_px,
                             width_um=width_px*um_per_pixel if status == "ok" else np.nan,
                             status=status, reason=reason))
    return rows


def summarize_measurements(details):
    rows = []
    for (kind, target_id), group in details.groupby(["measurement_type", "target_id"], sort=False):
        valid = group.loc[group.status == "ok", "width_um"].dropna()
        rows.append(dict(
            measurement_type=kind, target_id=target_id,
            mean_um=float(valid.mean()) if len(valid) else np.nan,
            std_um=(float(valid.std(ddof=1)) if len(valid) > 1 else
                    (0.0 if len(valid) == 1 else np.nan)),
            valid_points=len(valid), total_points=len(group),
            status="ok" if len(valid) == len(group) else ("partial" if len(valid) else "invalid")))
    return pd.DataFrame(rows)


def render_overlay(native, details, output_path, image_name):
    rgb = cv2.cvtColor(native[:, :, :3], cv2.COLOR_BGR2RGB)
    alpha = native[:, :, 3] if native.shape[2] == 4 else None
    shown = np.dstack((rgb, alpha)) if alpha is not None else rgb
    h, w = native.shape[:2]
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(shown)
    for y_px in sorted(details.y_px_native.unique()):
        ax.axhline(y_px, color="white", alpha=.35, linestyle="--")
    for row in details.to_dict("records"):
        if row["status"] != "ok":
            continue
        color = COLOR_BY_TYPE[row["measurement_type"]]
        ax.plot([row["left_edge_px"], row["right_edge_px"]],
                [row["y_px_native"]]*2, color=color, marker="|", linewidth=2)
        ax.text((row["left_edge_px"]+row["right_edge_px"])/2,
                row["y_px_native"]+max(6, h*.012),
                f'{row["target_id"]} {row["width_um"]:.2f}',
                ha="center", fontsize=7, color="black",
                bbox=dict(facecolor="white", alpha=.86, edgecolor="none", pad=1))
    ax.set_title(f"{image_name} | native coordinates | labels in um")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def process_image(path, measurement_types, output_dir):
    path, output_dir = Path(path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = read_image(path)
    native, native_info = restore_native_pixels(source)
    cv2.imencode(".png", native)[1].tofile(output_dir/"native_image.png")
    green = native[:, :, 1].astype(float)
    y_positions = native_y_positions(native.shape[0])
    centers, vertical_reference = detect_sidewall_centers(green)
    boundaries = sidewall_boundaries(centers, native.shape[1])
    reference_profile = np.median(green[y_positions], axis=0)
    targets, classification = define_targets(
        reference_profile, centers, boundaries, measurement_types)
    records = []
    for ratio, y_px in zip(Y_RATIOS, y_positions):
        measured_walls = measure_all_sidewalls(green[y_px], centers, boundaries)
        records.extend(measure_targets_at_y(
            green[y_px], measured_walls, targets, ratio, y_px))
    details = pd.DataFrame(records)
    summary = summarize_measurements(details)
    details.to_csv(output_dir/"measurements_by_y.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir/"target_summary.csv", index=False, encoding="utf-8-sig")
    render_overlay(native, details, output_dir/"overlay.png", path.name)
    metadata = dict(image_name=path.name, measurement_types=measurement_types,
                    y_ratios=list(Y_RATIOS), y_positions_native=y_positions,
                    um_per_native_pixel=UM_PER_NATIVE_PIXEL,
                    detected_sidewalls=len(centers), targets=targets,
                    interval_classification=classification, **native_info)
    (output_dir/"metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return details, summary, metadata


def collect_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError("지원하지 않는 이미지 확장자입니다.")
        return [input_path]
    if input_path.is_dir():
        images = sorted((path for path in input_path.iterdir()
                         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                        key=lambda path: path.name.lower())
        if not images:
            raise ValueError("입력 폴더에 지원 이미지가 없습니다.")
        return images
    raise ValueError(f"입력 경로가 없습니다: {input_path}")


def make_html_report(output, predictions, timing_summary, errors):
    css = """<style>body{font-family:Arial;background:#f3f4f6;color:#111827;max-width:1500px;margin:auto;padding:24px}.card{background:white;padding:20px;border:1px solid #cbd5e1;border-radius:12px;margin:18px 0;overflow-x:auto}table{border-collapse:collapse;width:100%}th{background:#e8eef7}th,td{padding:10px 14px;border-bottom:1px solid #d1d5db;text-align:left}img{max-width:100%}</style>"""
    parts = ["<!doctype html><meta charset='utf-8'>", css,
             "<h1>Unified line / gap / sidewall measurement</h1>",
             "<div class='card'><h2>Predictions</h2>",
             predictions.to_html(index=False, border=0, na_rep="—", escape=True,
                                 float_format=lambda value: f"{value:.5f}"), "</div>",
             "<div class='card'><h2>Processing time</h2><pre>",
             escape(json.dumps(timing_summary, ensure_ascii=False, indent=2)), "</pre></div>"]
    for image_name in predictions.image_name.drop_duplicates():
        folder = predictions.loc[predictions.image_name == image_name, "result_folder"].iloc[0]
        parts.extend([f"<div class='card'><h2>{escape(image_name)}</h2>",
                      f"<img src='{escape(folder)}/overlay.png'>",
                      predictions.loc[predictions.image_name == image_name,
                                      ["measurement_type", "target_id", "mean_um", "std_um",
                                       "valid_points", "status"]].to_html(
                                           index=False, border=0, na_rep="—", escape=True,
                                           float_format=lambda value: f"{value:.5f}"), "</div>"])
    if errors:
        parts.extend(["<div class='card'><h2>Errors</h2><pre>",
                      escape(json.dumps(errors, ensure_ascii=False, indent=2)), "</pre></div>"])
    (output/"report.html").write_text("".join(parts), encoding="utf-8")


def run_batch(input_path, measurement_types, output_dir):
    images = collect_images(input_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    predictions, timings, errors = [], [], []
    for image_index, path in enumerate(images):
        folder_name = f"{image_index+1:03d}_{path.stem}"
        start = time.perf_counter()
        try:
            details, summary, metadata = process_image(
                path, measurement_types, output/folder_name)
            elapsed = time.perf_counter()-start
            for row in summary.to_dict("records"):
                predictions.append(dict(
                    image_name=path.name, **row,
                    native_width=metadata["native_width"],
                    native_height=metadata["native_height"],
                    duplicate_factor_x=metadata["duplicate_factor_x"],
                    duplicate_factor_y=metadata["duplicate_factor_y"],
                    processing_time_s=elapsed, timing_included=image_index != 0,
                    result_folder=folder_name))
            timings.append(dict(image_name=path.name, processing_time_s=elapsed,
                                timing_included=image_index != 0, status="ok"))
        except Exception as error:
            elapsed = time.perf_counter()-start
            errors.append(dict(image_name=path.name, error=str(error)))
            timings.append(dict(image_name=path.name, processing_time_s=elapsed,
                                timing_included=image_index != 0, status="error"))
            error_dir = output/folder_name
            error_dir.mkdir(parents=True, exist_ok=True)
            (error_dir/"error.txt").write_text(str(error), encoding="utf-8")
    prediction_frame = pd.DataFrame(predictions)
    if prediction_frame.empty:
        raise ValueError(f"측정에 성공한 이미지가 없습니다. 오류: {errors}")
    timing_frame = pd.DataFrame(timings)
    included = timing_frame.loc[timing_frame.timing_included, "processing_time_s"]
    timing_summary = dict(
        total_images=len(images), warmup_image=images[0].name,
        warmup_excluded=True, timed_images=len(included),
        mean_processing_time_s=float(included.mean()) if len(included) else None,
        std_processing_time_s=(float(included.std(ddof=1)) if len(included) > 1 else
                               (0.0 if len(included) == 1 else None)))
    timing_frame.to_csv(output/"processing_times.csv", index=False, encoding="utf-8-sig")
    (output/"timing_summary.json").write_text(
        json.dumps(timing_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    team_protocol.save_prediction_csv(prediction_frame, output/"predictions.csv")
    make_html_report(output, prediction_frame, timing_summary, errors)
    return prediction_frame, timing_frame, timing_summary, errors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Native-pixel line/gap/sidewall measurement for one image or a folder.")
    parser.add_argument("input", nargs="?", type=Path, help="이미지 파일 또는 이미지 폴더")
    parser.add_argument("--measure", help="1=line, 2=gap, 3=sidewall; 예: 1, 13, 123")
    parser.add_argument("--output", type=Path, help="결과 폴더")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input or Path(input("이미지 한 장 또는 폴더 경로: ").strip().strip('"'))
    selection = args.measure or input("측정 종류 선택 (1=line, 2=gap, 3=sidewall / 복수 가능): ")
    measurement_types = parse_measurement_selection(selection)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = input_path.parent if input_path.is_file() else input_path
    output = args.output or base/f"unified_measurement_results_{timestamp}"
    print(f"측정 종류: {', '.join(measurement_types)}")
    print("Resize 없이 중복 저장 픽셀을 제거하고 native 좌표에서 측정합니다.")
    predictions, timings, timing_summary, errors = run_batch(
        input_path, measurement_types, output)
    print("\n측정 결과 (um)")
    print(predictions[["image_name", "measurement_type", "target_id", "mean_um",
                       "std_um", "valid_points", "status"]].round(5).to_string(index=False))
    print(f"\n완료: {output}")
    print(f"team protocol CSV: {output/'predictions.csv'}")
    print(f"HTML report: {output/'report.html'}")
    print(f"warm-up excluded mean processing time: {timing_summary['mean_processing_time_s']} s")
    if errors:
        print(f"errors: {len(errors)} image(s); see report.html")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"오류: {error}")
        raise SystemExit(1)
