#!/usr/bin/env python3
"""Measure all red sidewall lines at ruler y=20 and y=60 in one screenshot."""
from pathlib import Path
from datetime import datetime
from html import escape
import argparse
import json

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def detect_lines(green, min_distance=20):
    """중앙 높이 구간의 중앙값으로 선 번호를 고정. 평활화는 위치 검출에만 사용."""
    h, w = green.shape
    if h < 10 or w < 10 or min_distance < 2:
        raise ValueError("이미지 또는 선 간격 설정이 너무 작습니다.")
    reference = np.nanmedian(green[int(h*.15):int(h*.85)], axis=0)
    if not np.isfinite(reference).any():
        raise ValueError("선 검출 영역이 모두 투명합니다.")
    reference = np.where(np.isfinite(reference), reference, np.nanmax(reference))
    smooth = cv2.GaussianBlur(reference[None, :], (0, 0), 1).ravel()
    local_min = cv2.erode(smooth[None, :], np.ones((1, 2*min_distance+1), np.uint8)).ravel()
    contrast = max(12., float(np.ptp(smooth))*.15)
    mask = (smooth <= local_min + 1e-8) & (smooth < np.percentile(smooth, 80)-contrast)
    transitions = np.diff(np.r_[False, mask, False].astype(int))
    centers = np.array([(a+b-1)//2 for a, b in zip(
        np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1))], dtype=int)
    # 이미지 가장자리의 1~2px 검은 테두리를 후보에서 제외.
    centers = centers[(centers >= 3) & (centers < w-3)]
    if not len(centers):
        raise ValueError("세로 선을 검출하지 못했습니다. 이미지와 MIN_LINE_DISTANCE를 확인하세요.")
    return centers, reference

def measure_row_robust(profile, expected_center, fraction=.7, background_fraction=.2,
                       core_pixels=3):
    """Measure one dark line with local backgrounds and a robust core intensity."""
    p = np.asarray(profile, dtype=float)
    if p.ndim != 1 or len(p) < 10 or not np.isfinite(p).all():
        raise ValueError("유효한 1차원 프로파일이 10픽셀 이상 필요합니다.")
    if not 0 < fraction < 1 or not 0 < background_fraction < .5:
        raise ValueError("밝기 기준 또는 배경 비율이 범위를 벗어났습니다.")
    if not 1 <= core_pixels <= 9 or core_pixels % 2 != 1:
        raise ValueError("core_pixels는 1~9 사이의 홀수여야 합니다.")
    n = max(2, int(len(p)*background_fraction))
    expected_center = int(round(expected_center))
    if not n <= expected_center < len(p)-n:
        raise ValueError("예상 선 중심이 배경 구간에 있습니다.")

    bg_l, bg_r = float(np.median(p[:n])), float(np.median(p[-n:]))
    radius = max(3, int(len(p)*.20))
    s0, s1 = max(n, expected_center-radius), min(len(p)-n, expected_center+radius+1)
    center = s0 + int(np.argmin(p[s0:s1]))
    half = core_pixels//2
    c0, c1 = max(n, center-half), min(len(p)-n, center+half+1)
    bottom = float(np.median(p[c0:c1]))
    background = (bg_l+bg_r)/2
    threshold = background-fraction*(background-bottom)
    if min(bg_l, bg_r)-bottom < 5:
        raise ValueError("선과 양쪽 배경의 대비가 부족합니다.")

    mask = p <= threshold
    changes = np.diff(np.r_[False, mask, False].astype(int))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    hit = [(i, a, b) for i, (a, b) in enumerate(zip(starts, stops)) if a <= center < b]
    if not hit:
        raise ValueError("선 중심을 포함하는 밝기 기준 이하 구간이 없습니다.")
    k, a, b = hit[0]
    a, b = int(a), int(b)
    if a < n or b > len(p)-n or a == 0 or b >= len(p):
        raise ValueError("선 경계가 배경 추정 구간에 닿습니다.")
    interior = [(s, e) for s, e in zip(starts, stops) if s < len(p)-n and e > n]
    if len(interior) != 1:
        raise ValueError("측정 영역에 분리된 어두운 구간이 여러 개입니다.")
    dl, dr = p[a-1]-p[a], p[b]-p[b-1]
    if dl == 0 or dr == 0:
        raise ValueError("경계 보간에 필요한 밝기 기울기가 없습니다.")
    left = a-1+(p[a-1]-threshold)/dl
    right = b-1+(threshold-p[b-1])/dr
    return dict(status="ok", background_left=bg_l, background_right=bg_r,
                background=background, minimum=bottom, threshold=float(threshold),
                center_px=center, first_pixel=a, last_pixel=b-1,
                included_pixels=b-a, left_px=float(left), right_px=float(right),
                width_px=float(right-left))

def measure_window(green, alpha, y, lo, hi, expected_center, um_per_pixel,
                   fraction=.7, background_fraction=.2, row_half_window=5,
                   min_valid_rows=7, core_pixels=3, max_width_um=1.5):
    """Measure each row independently and return the median width in the y window."""
    h = green.shape[0]
    if y-row_half_window < 0 or y+row_half_window >= h:
        raise ValueError("측정 행 창이 이미지 범위를 벗어납니다.")
    valid, diagnostics = [], []
    for yy in range(y-row_half_window, y+row_half_window+1):
        try:
            if np.any(alpha[yy, lo:hi] != 255):
                raise ValueError("투명 픽셀")
            row = measure_row_robust(
                green[yy, lo:hi], expected_center-lo, fraction,
                background_fraction, core_pixels)
            for key in ("left_px", "right_px", "first_pixel", "last_pixel", "center_px"):
                row[key] += lo
            row["y_row_px"] = yy
            row["width_um"] = row["width_px"]*um_per_pixel
            if max_width_um is not None and row["width_um"] >= max_width_um:
                raise ValueError(f"폭이 {max_width_um:g}µm 이상")
            valid.append(row)
            diagnostics.append(dict(row, reason=""))
        except ValueError as error:
            diagnostics.append(dict(y_row_px=yy, status="invalid", reason=str(error)))
    if len(valid) < min_valid_rows:
        raise ValueError(f"주변 행 {2*row_half_window+1}개 중 정상 행이 {len(valid)}개뿐입니다.")
    widths = np.asarray([r["width_px"] for r in valid])
    median_width = float(np.median(widths))
    representative = valid[int(np.argmin(np.abs(widths-median_width)))]
    return dict(status="ok", width_px=median_width,
                width_um=median_width*um_per_pixel,
                valid_window_rows=len(valid), total_window_rows=2*row_half_window+1,
                row_width_sd_um=(float(np.std(widths, ddof=1))*um_per_pixel
                                 if len(widths) > 1 else 0.0),
                left_px=representative["left_px"], right_px=representative["right_px"],
                representative_y_px=representative["y_row_px"],
                threshold=representative["threshold"], diagnostics=diagnostics)

def analyze_image_robust(image_path, um_per_pixel, y_positions=(322, 644, 966),
                         fraction=.7, background_fraction=.2, min_line_distance=20,
                         row_half_window=5, min_valid_rows=7, core_pixels=3,
                         max_width_um=1.5, out_dir="results/peb_image",
                         x_start=0, x_stop=None):
    image_path, out = Path(image_path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("8비트 RGB/RGBA 이미지를 사용하세요.")
    green = img[:, :, 1].astype(float)
    alpha = img[:, :, 3] if img.shape[2] == 4 else np.full(green.shape, 255, np.uint8)
    h, w = green.shape
    ys = np.asarray(y_positions, dtype=int)
    if ys.ndim != 1 or len(ys) < 2 or len(set(ys)) != len(ys) or np.any(np.diff(ys) <= 0):
        raise ValueError("서로 다른 y 위치를 두 개 이상 위에서 아래 순서로 입력하세요.")
    if np.any(ys-row_half_window < 0) or np.any(ys+row_half_window >= h):
        raise ValueError(f"이미지 높이 {h}px에서 y 위치와 ±{row_half_window}px 창을 사용할 수 없습니다.")
    x_stop = w if x_stop is None else int(x_stop)
    x_start = int(x_start)
    if not 0 <= x_start < x_stop <= w or x_stop-x_start < 10:
        raise ValueError("x 측정 범위가 이미지 밖이거나 너무 좁습니다.")
    local_centers, _ = detect_lines(
        np.where(alpha[:, x_start:x_stop] == 255, green[:, x_start:x_stop], np.nan),
        min_line_distance)
    centers = local_centers+x_start
    boundaries = np.r_[x_start, (centers[:-1]+centers[1:])//2, x_stop].astype(int)
    point_records, row_records = [], []
    for line_id, center in enumerate(centers, 1):
        lo, hi = boundaries[line_id-1:line_id+1]
        for point, y in enumerate(ys, 1):
            base = dict(line_id=line_id, point=point, y_px=int(y),
                        reference_center_px=int(center), roi_left=int(lo), roi_right=int(hi))
            try:
                result = measure_window(
                    green, alpha, int(y), int(lo), int(hi), int(center), um_per_pixel,
                    fraction, background_fraction, row_half_window, min_valid_rows,
                    core_pixels, max_width_um)
                diagnostics = result.pop("diagnostics")
                point_records.append(dict(**base, **result, reason=""))
                row_records.extend(dict(**base, **r) for r in diagnostics)
            except ValueError as error:
                point_records.append(dict(**base, status="invalid", reason=str(error),
                                          width_px=np.nan, width_um=np.nan))
    points = pd.DataFrame(point_records)
    rows = pd.DataFrame(row_records)
    summaries = []
    for line_id, group in points.groupby("line_id", sort=True):
        good = group.loc[group.status == "ok", "width_um"]
        eligible = len(good) == len(ys)
        summaries.append(dict(
            line_id=int(line_id), valid_points=len(good), eligible=eligible,
            mean_um=float(good.mean()) if eligible else np.nan,
            std_um=float(good.std(ddof=1)) if eligible else np.nan,
        ))
    lines = pd.DataFrame(summaries)
    eligible = lines.loc[lines.eligible]
    if eligible.empty:
        raise ValueError("세 위치가 모두 정상인 라인이 없습니다.")
    image_mean = float(eligible.mean_um.mean())
    image_sd = float(eligible.mean_um.std(ddof=1)) if len(eligible) > 1 else 0.0

    points.to_csv(out/"point_measurements.csv", index=False, encoding="utf-8-sig")
    rows.to_csv(out/"row_diagnostics.csv", index=False, encoding="utf-8-sig")
    lines.to_csv(out/"line_summary_um.csv", index=False, encoding="utf-8-sig")
    summary = dict(filename=image_path.name, image_mean_um=image_mean,
                   image_sd_across_lines_um=image_sd, valid_lines=len(eligible),
                   detected_lines=len(lines), status="ok")
    (out/"image_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(18, 10))
    rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    ax.imshow(np.dstack((rgb, alpha)))
    for line_id, center in enumerate(centers, 1):
        ax.text(center, h*.035, str(line_id), ha="center", color="black",
                bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
    for y in ys:
        ax.axhline(y, color="white", alpha=.35, linestyle="--")
    for r in point_records:
        if r["status"] == "ok":
            ax.plot([r["left_px"], r["right_px"]], [r["y_px"]]*2, "c|-", linewidth=2)
            ax.text(r["reference_center_px"], r["y_px"]+h*.025,
                    f'{r["width_um"]:.2f}', ha="center", fontsize=8, color="black",
                    bbox=dict(facecolor="white", alpha=.88, edgecolor="none", pad=1.5))
        else:
            ax.plot(r["reference_center_px"], r["y_px"], "mx")
    ax.set_title(f'{image_path.name} | image mean={image_mean:.3f} µm | labels in µm')
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out/"overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return points, lines, summary, out

def choose_image():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="측정할 이미지 한 장 선택",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("All files", "*.*")])
        root.destroy()
    except Exception as error:
        raise ValueError("파일 선택창을 열 수 없습니다. --input으로 파일을 지정하세요.") from error
    if not selected:
        raise ValueError("이미지를 선택하지 않았습니다.")
    return Path(selected)


def make_line_table(points, lines, ruler_values):
    good = points.loc[points.status == "ok", ["line_id", "point", "width_um"]]
    pivot = good.pivot(index="line_id", columns="point", values="width_um")
    pivot.rename(columns={i+1: f"width_y{value:g}_um"
                          for i, value in enumerate(ruler_values)}, inplace=True)
    result = lines.merge(pivot, left_on="line_id", right_index=True, how="left")
    result["status"] = np.where(result.eligible, "ok", "invalid")
    ordered = ["line_id"]+[f"width_y{value:g}_um" for value in ruler_values]
    ordered += ["mean_um", "std_um", "valid_points", "status"]
    return result.reindex(columns=ordered)


def make_report(output, source, overlay_name, line_table, summary, settings):
    css = """<style>
body{font-family:Arial,sans-serif;background:#f3f4f6;color:#111827;max-width:1500px;margin:auto;padding:24px}
.card{background:white;padding:20px 24px;border:1px solid #cbd5e1;border-radius:12px;margin:18px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%}th{background:#e8eef7}th,td{padding:11px 15px;border-bottom:1px solid #d1d5db;text-align:left}
img{max-width:100%;height:auto}h1,h2{color:#111827}
</style>"""
    shown = line_table.rename(columns={
        "line_id": "라인 번호", "width_y20_um": "y=20 폭 (µm)",
        "width_y60_um": "y=60 폭 (µm)", "mean_um": "평균 (µm)",
        "std_um": "표준편차 (µm)", "valid_points": "정상 위치 수",
        "status": "상태"})
    table = shown.to_html(index=False, border=0, na_rep="—", escape=True,
                          float_format=lambda value: f"{value:.5f}")
    html = f"""<!doctype html><meta charset='utf-8'>{css}
<h1>단일 이미지 Sidewall 측정</h1>
<div class='card'><h2>이미지 전체 결과</h2>
<p>파일: {escape(source.name)}</p>
<p>정상 라인: {summary['valid_lines']}/{summary['detected_lines']}</p>
<p>정상 라인 평균들의 평균: <b>{summary['image_mean_um']:.5f} µm</b></p>
<p>라인 간 표본 표준편차: <b>{summary['image_sd_across_lines_um']:.5f} µm</b></p></div>
<div class='card'><img src='{escape(overlay_name)}' alt='sidewall measurement overlay'></div>
<div class='card'><h2>라인별 y=20·60 측정값, 평균과 표준편차</h2>{table}</div>
<div class='card'><h2>측정 설정</h2><pre>{escape(json.dumps(settings, ensure_ascii=False, indent=2))}</pre></div>"""
    (output/"report.html").write_text(html, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="한 장의 이미지에서 왼쪽 자 y=20, y=60 위치의 모든 세로 라인을 측정합니다.")
    parser.add_argument("--input", type=Path, help="측정할 이미지")
    parser.add_argument("--output", type=Path, help="결과 폴더")
    parser.add_argument("--ruler-y-values", type=float, nargs=2, default=[20, 60])
    parser.add_argument("--ruler-zero-y-px", type=float, default=28.0)
    parser.add_argument("--ruler-pixels-per-unit", type=float, default=12.6)
    parser.add_argument("--image-left-px", type=int, default=28,
                        help="왼쪽 눈금/테두리를 제외한 실제 이미지 시작 x")
    parser.add_argument("--um-per-pixel", type=float, default=20/252,
                        help="기본값은 화면 눈금 20um=252px에서 계산")
    parser.add_argument("--edge-fraction", type=float, default=.70)
    parser.add_argument("--row-half-window", type=int, default=5)
    parser.add_argument("--min-valid-rows", type=int, default=7)
    parser.add_argument("--core-pixels", type=int, default=3)
    parser.add_argument("--max-width-um", type=float, default=1.5)
    parser.add_argument("--background-fraction", type=float, default=.20)
    parser.add_argument("--min-line-distance", type=int, default=55,
                        help="이 사진의 약 110px 라인 피치에 맞춘 오검출 방지 거리")
    return parser.parse_args()


def main():
    args = parse_args()
    source = (args.input or choose_image()).resolve()
    if not source.is_file():
        raise ValueError(f"이미지 파일이 없습니다: {source}")
    ruler_values = np.asarray(args.ruler_y_values, dtype=float)
    y_positions = np.rint(
        args.ruler_zero_y_px+ruler_values*args.ruler_pixels_per_unit).astype(int)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output.resolve() if args.output else
              source.parent/f"{source.stem}_sidewall_results_{timestamp}")
    if output.exists():
        raise ValueError(f"결과 폴더가 이미 있습니다: {output}")
    settings = dict(
        ruler_y_values=ruler_values.tolist(), file_y_pixels=y_positions.tolist(),
        ruler_zero_y_px=args.ruler_zero_y_px,
        ruler_pixels_per_unit=args.ruler_pixels_per_unit,
        image_left_px=args.image_left_px, um_per_file_pixel=args.um_per_pixel,
        edge_fraction=args.edge_fraction, row_half_window=args.row_half_window,
        rows_per_point=2*args.row_half_window+1,
        min_valid_rows=args.min_valid_rows, core_pixels=args.core_pixels,
        max_width_um=args.max_width_um, background_fraction=args.background_fraction,
        min_line_distance=args.min_line_distance,
        aggregation="median across rows -> mean and sample SD across y=20 and y=60")
    print(f"입력 이미지: {source}")
    print(f"눈금 y={ruler_values[0]:g}, {ruler_values[1]:g} -> 파일 y={y_positions[0]}, {y_positions[1]}px")
    points, lines, summary, output = analyze_image_robust(
        source, args.um_per_pixel, y_positions=y_positions,
        fraction=args.edge_fraction, background_fraction=args.background_fraction,
        min_line_distance=args.min_line_distance,
        row_half_window=args.row_half_window, min_valid_rows=args.min_valid_rows,
        core_pixels=args.core_pixels, max_width_um=args.max_width_um,
        out_dir=output, x_start=args.image_left_px)
    line_table = make_line_table(points, lines, ruler_values)
    line_table.to_csv(output/"line_results_um.csv", index=False, encoding="utf-8-sig")
    (output/"settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    make_report(output, source, "overlay.png", line_table, summary, settings)
    print("\n라인별 결과 (um)")
    print(line_table.round(5).to_string(index=False))
    print(f"\n정상 라인: {summary['valid_lines']}/{summary['detected_lines']}")
    print(f"정상 라인 평균들의 평균: {summary['image_mean_um']:.5f} um")
    print(f"완료: {output}")
    print(f"HTML 보고서: {output/'report.html'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"오류: {error}")
        raise SystemExit(1)
