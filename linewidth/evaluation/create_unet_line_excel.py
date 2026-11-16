"""Create an Excel summary of per-image, per-line U-Net widths.

The input is the common ``unet_predictions.csv`` written by the performance
suite.  For every image and target (normally S1-S5), the workbook records the
five protocol Y measurements plus their mean and population standard
deviation.  Invalid/error measurements remain blank and are counted.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from copy import copy
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


Y_FRACTIONS = (0.25, 0.45, 0.65)
TARGET_PREFIX = {"line": "L", "gap": "G", "sidewall": "S"}
REQUIRED_FIELDS = {
    "image_id",
    "measurement_type",
    "y_fraction",
    "target_id",
    "width_px",
    "status",
}
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
SECTION_FILL = PatternFill("solid", fgColor="D9EAF7")
WHITE_FONT = Font(color="FFFFFF", bold=True)
HEADER_FONT = Font(bold=True, color="1F1F1F")
THIN_GRAY = Side(style="thin", color="D9E1F2")


def _optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    number = float(text)
    return number if math.isfinite(number) else None


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def read_predictions(path: Path, measurement_type: str) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Prediction CSV is missing columns: {', '.join(sorted(missing))}")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            if str(raw.get("measurement_type", "")).strip().lower() != measurement_type:
                continue
            y_fraction = float(raw["y_fraction"])
            if y_fraction > 1:
                y_fraction /= 100.0
            rows.append(
                {
                    **raw,
                    "_line": line_number,
                    "y_fraction": round(y_fraction, 2),
                    "width_px_value": _optional_float(raw.get("width_px")),
                    "status": str(raw.get("status", "ok") or "ok").strip().lower(),
                }
            )
    if not rows:
        raise ValueError(f"No {measurement_type!r} rows found in {path}")
    return rows


def build_line_rows(
    rows: list[dict], calibration: float, measurement_type: str
) -> list[dict]:
    grouped: dict[tuple[str, str], dict[float, dict]] = defaultdict(dict)
    for row in rows:
        key = (str(row["image_id"]), str(row["target_id"]))
        y_fraction = float(row["y_fraction"])
        if y_fraction in grouped[key]:
            raise ValueError(
                f"Duplicate image/target/Y row: {key[0]}, {key[1]}, {y_fraction:.2f}"
            )
        grouped[key][y_fraction] = row

    prefix = TARGET_PREFIX.get(measurement_type, measurement_type[:1].upper())
    image_ids = sorted({str(row["image_id"]) for row in rows}, key=_natural_key)
    expected_keys = [
        (image_id, f"{prefix}{target_number}")
        for image_id in image_ids
        for target_number in range(1, 6)
    ]
    output = []
    for image_id, target_id in expected_keys:
        by_y = grouped.get((image_id, target_id), {})
        widths: list[float] = []
        y_values: dict[float, float | None] = {}
        messages = []
        for y_fraction in Y_FRACTIONS:
            source = by_y.get(y_fraction)
            value = None
            if source is not None:
                candidate = source["width_px_value"]
                if source["status"] == "ok" and candidate is not None and candidate > 0:
                    value = float(candidate)
                    widths.append(value)
                elif source.get("message"):
                    messages.append(f"Y{int(y_fraction * 100)}: {source['message']}")
            else:
                messages.append(f"Y{int(y_fraction * 100)}: missing row")
            y_values[y_fraction] = value

        valid_count = len(widths)
        mean_px = statistics.fmean(widths) if widths else None
        std_px = statistics.pstdev(widths) if widths else None
        output.append(
            {
                "image_id": image_id,
                "measurement_type": measurement_type,
                "line_id": target_id,
                "valid_count": valid_count,
                "error_count": len(Y_FRACTIONS) - valid_count,
                "mean_px": mean_px,
                "std_px": std_px,
                "mean_um": mean_px * calibration if mean_px is not None else None,
                "std_um": std_px * calibration if std_px is not None else None,
                "y_values": y_values,
                "status": "complete" if valid_count == len(Y_FRACTIONS) else (
                    "partial" if valid_count else "error"
                ),
                "message": " | ".join(messages),
            }
        )
    return output


def _style_header(ws, row: int, start_col: int, end_col: int) -> None:
    for cell in ws.iter_cols(min_col=start_col, max_col=end_col, min_row=row, max_row=row):
        item = cell[0]
        item.fill = HEADER_FILL
        item.font = WHITE_FONT
        item.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        item.border = Border(bottom=THIN_GRAY)


def _set_widths(ws, widths: dict[str, float]) -> None:
    for column, width in widths.items():
        ws.column_dimensions[column].width = width


def create_workbook(
    predictions_path: Path,
    output_path: Path,
    measurement_type: str,
    calibration: float,
) -> Path:
    raw_rows = read_predictions(predictions_path, measurement_type)
    line_rows = build_line_rows(raw_rows, calibration, measurement_type)

    wb = Workbook()
    summary = wb.active
    summary.title = "라인별 요약"
    photo = wb.create_sheet("사진 요약")
    raw = wb.create_sheet("원본 측정값")
    notes = wb.create_sheet("설명")

    summary.append(["사진별 U-Net sidewall 선폭 요약"])
    y_labels = [f"Y{round(fraction * 100)}" for fraction in Y_FRACTIONS]
    summary.append([
        f"각 라인의 {'·'.join(y_labels)} 측정값과 평균 및 모집단 표준편차"
    ])
    summary.append([])
    summary_headers = [
        "image_id", "measurement_type", "line_id", "valid_count", "error_count",
        "mean_width_px", "std_width_px", "mean_width_um", "std_width_um",
        *[f"{label}_width_px" for label in y_labels], "status", "message",
    ]
    status_col = len(summary_headers) - 1
    last_letter = get_column_letter(len(summary_headers))
    summary.append(summary_headers)
    for item in line_rows:
        summary.append([
            item["image_id"], item["measurement_type"], item["line_id"],
            item["valid_count"], item["error_count"], item["mean_px"], item["std_px"],
            item["mean_um"], item["std_um"],
            *[item["y_values"][fraction] for fraction in Y_FRACTIONS],
            item["status"], item["message"],
        ])

    summary["A1"].font = Font(size=15, bold=True, color="1F1F1F")
    summary["A2"].font = Font(size=10, italic=True, color="666666")
    _style_header(summary, 4, 1, len(summary_headers))
    summary.freeze_panes = "A5"
    summary.auto_filter.ref = f"A4:{last_letter}{summary.max_row}"
    summary.sheet_view.showGridLines = False
    summary.row_dimensions[4].height = 32
    for row in summary.iter_rows(min_row=5, min_col=4, max_col=5):
        for cell in row:
            cell.number_format = "0"
    for row in summary.iter_rows(min_row=5, min_col=6, max_col=status_col - 1):
        for cell in row:
            cell.number_format = "0.0000"
    for row in summary.iter_rows(min_row=5, min_col=status_col, max_col=status_col):
        for cell in row:
            value = str(cell.value or "")
            if value == "complete":
                cell.fill = PatternFill("solid", fgColor="E2F0D9")
            elif value == "partial":
                cell.fill = PatternFill("solid", fgColor="FFF2CC")
            else:
                cell.fill = PatternFill("solid", fgColor="FCE4D6")
    _set_widths(summary, {
        "A": 24, "B": 18, "C": 10, "D": 12, "E": 12,
        "F": 16, "G": 16, "H": 16, "I": 16,
        **{get_column_letter(10 + i): 14 for i in range(len(y_labels))},
        get_column_letter(status_col): 12, last_letter: 62,
    })
    if summary.max_row >= 5:
        table = Table(displayName="UNetLineSummary", ref=f"A4:{last_letter}{summary.max_row}")
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
            showRowStripes=True, showColumnStripes=False,
        )
        summary.add_table(table)

    photo_headers = [
        "image_id", "line_count", "complete_lines", "partial_lines", "error_lines",
        "valid_measurements", "error_measurements", "measurement_success_rate",
        "mean_width_px_all", "std_width_px_all", "mean_width_um_all", "std_width_um_all",
    ]
    photo.append(["사진 요약"])
    photo.append(["각 사진의 S1~S5 측정 완전성과 전체 유효 선폭 분포"])
    photo.append([])
    photo.append(photo_headers)
    line_by_image: dict[str, list[dict]] = defaultdict(list)
    raw_widths_by_image: dict[str, list[float]] = defaultdict(list)
    for item in line_rows:
        line_by_image[item["image_id"]].append(item)
    for row in raw_rows:
        width = row["width_px_value"]
        if row["status"] == "ok" and width is not None and width > 0:
            raw_widths_by_image[str(row["image_id"])].append(float(width))
    for image_id in sorted(line_by_image, key=_natural_key):
        items = line_by_image[image_id]
        widths = raw_widths_by_image[image_id]
        valid = len(widths)
        expected = len(items) * len(Y_FRACTIONS)
        mean_px = statistics.fmean(widths) if widths else None
        std_px = statistics.pstdev(widths) if widths else None
        photo.append([
            image_id, len(items), sum(i["status"] == "complete" for i in items),
            sum(i["status"] == "partial" for i in items),
            sum(i["status"] == "error" for i in items), valid, expected - valid,
            valid / expected if expected else None, mean_px, std_px,
            mean_px * calibration if mean_px is not None else None,
            std_px * calibration if std_px is not None else None,
        ])
    photo["A1"].font = Font(size=15, bold=True)
    photo["A2"].font = Font(size=10, italic=True, color="666666")
    _style_header(photo, 4, 1, len(photo_headers))
    photo.freeze_panes = "A5"
    photo.auto_filter.ref = f"A4:L{photo.max_row}"
    photo.sheet_view.showGridLines = False
    for cell in photo["H"][4:]:
        cell.number_format = "0.0%"
    for row in photo.iter_rows(min_row=5, min_col=9, max_col=12):
        for cell in row:
            cell.number_format = "0.0000"
    _set_widths(photo, {
        "A": 24, "B": 12, "C": 15, "D": 14, "E": 12, "F": 19,
        "G": 19, "H": 24, "I": 18, "J": 18, "K": 18, "L": 18,
    })
    if photo.max_row >= 5:
        table = Table(displayName="UNetPhotoSummary", ref=f"A4:L{photo.max_row}")
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
            showRowStripes=True, showColumnStripes=False,
        )
        photo.add_table(table)

    raw_headers = [
        "image_id", "measurement_type", "y_fraction", "target_id", "width_px",
        "width_um", "center_x_px", "left_x_px", "right_x_px", "runtime_ms",
        "status", "message",
    ]
    raw.append(raw_headers)
    raw_numeric_fields = {
        "y_fraction", "width_px", "width_um", "center_x_px", "left_x_px",
        "right_x_px", "runtime_ms",
    }
    for row in sorted(
        raw_rows,
        key=lambda item: (
            _natural_key(str(item["image_id"])), float(item["y_fraction"]),
            _natural_key(str(item["target_id"])),
        ),
    ):
        raw.append(
            [
                (
                    float(row["y_fraction"])
                    if field == "y_fraction"
                    else _optional_float(row.get(field))
                    if field in raw_numeric_fields
                    else row.get(field, "")
                )
                for field in raw_headers
            ]
        )
    _style_header(raw, 1, 1, len(raw_headers))
    raw.freeze_panes = "A2"
    raw.auto_filter.ref = f"A1:L{raw.max_row}"
    raw.sheet_view.showGridLines = False
    for row in raw.iter_rows(min_row=2, min_col=3, max_col=10):
        for cell in row:
            cell.number_format = "0.0000"
    _set_widths(raw, {
        "A": 24, "B": 18, "C": 12, "D": 10, "E": 14, "F": 14,
        "G": 14, "H": 14, "I": 14, "J": 14, "K": 11, "L": 60,
    })
    if raw.max_row >= 2:
        table = Table(displayName="UNetRawMeasurements", ref=f"A1:L{raw.max_row}")
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
            showRowStripes=True, showColumnStripes=False,
        )
        raw.add_table(table)

    notes.append(["항목", "내용"])
    notes.append(["입력 파일", predictions_path.name])
    notes.append(["측정 종류", measurement_type])
    notes.append(["Y 위치", ", ".join(f"{round(f * 100)}%" for f in Y_FRACTIONS)])
    notes.append(["Calibration", f"1 px = {calibration:g} µm"])
    notes.append(["평균", "각 사진·라인의 정상 측정값 평균"])
    notes.append(["편차", "각 사진·라인의 정상 측정값에 대한 모집단 표준편차(STDEV.P)"])
    notes.append(["오류 처리", "status가 ok가 아니거나 폭이 0 이하인 값은 평균에서 제외"])
    notes.append(["생성 시각", datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")])
    _style_header(notes, 1, 1, 2)
    notes.sheet_view.showGridLines = False
    notes.freeze_panes = "A2"
    _set_widths(notes, {"A": 20, "B": 76})
    for row in notes.iter_rows(min_row=2, min_col=1, max_col=2):
        row[0].font = HEADER_FONT
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                font = copy(cell.font)
                font.name = "Aptos"
                font.sz = font.sz or 10
                cell.font = font
                alignment = copy(cell.alignment)
                alignment.vertical = "center"
                cell.alignment = alignment

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-type", default="sidewall")
    parser.add_argument("--calibration", type=float, default=0.1592)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibration <= 0:
        raise ValueError("--calibration must be greater than zero")
    output = create_workbook(
        args.predictions.resolve(),
        args.output.resolve(),
        args.measurement_type.strip().lower(),
        args.calibration,
    )
    print(f"Excel summary: {output}")


if __name__ == "__main__":
    main()
