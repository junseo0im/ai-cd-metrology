"""Standalone protocol shared by every line-width measurement program.

Give this one file to each teammate and import it from their own algorithm.
It standardizes only the experiment interface: number keys, native Y rows,
central target IDs, and prediction CSV output.  It does not contain U-Net or
any other edge-detection algorithm.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


UM_PER_NATIVE_PX = 0.1592
Y_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
MEASUREMENT_ORDER = ("line", "gap", "sidewall")
MEASUREMENT_DEFINITIONS = {
    "sidewall": (
        "one orange/red vertical boundary itself: horizontal distance from "
        "that boundary's left edge to its right edge"
    ),
    "line": (
        "the brighter one of the two alternating intervals between adjacent "
        "sidewalls: horizontal distance between the two facing sidewall edges"
    ),
    "gap": (
        "the other alternating interval between adjacent sidewalls: horizontal "
        "distance between its two facing sidewall edges"
    ),
}
MEASUREMENT_KEY_MAP = {
    "1": "line",
    "2": "gap",
    "3": "sidewall",
    "line": "line",
    "gap": "gap",
    "sidewall": "sidewall",
}
TARGET_PREFIX = {"line": "L", "gap": "G", "sidewall": "S"}
TARGETS_PER_ROW = 5
PREDICTION_FIELDS = (
    "method",
    "image_id",
    "measurement_type",
    "y_fraction",
    "target_id",
    "width_px",
    "width_um",
    "center_x_px",
    "left_x_px",
    "right_x_px",
    "runtime_ms",
    "status",
    "message",
)


def normalize_measurement_type(value: int | str) -> str:
    key = str(value).strip().lower()
    try:
        return MEASUREMENT_KEY_MAP[key]
    except KeyError as error:
        raise ValueError("use only 1=line, 2=gap, 3=sidewall") from error


def parse_measurement_types(value) -> list[str]:
    """Parse 3, '3', '1 3', '123', or a list; preserve 1/2/3 order."""
    if isinstance(value, (list, tuple, set)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(parse_measurement_types(item))
    else:
        text = str(value).strip().lower()
        if text in {"all", "전체"}:
            tokens = list(MEASUREMENT_ORDER)
        elif re.fullmatch(r"[123]+", text):
            tokens = [normalize_measurement_type(character) for character in text]
        else:
            raw = [token for token in re.split(r"[\s,;+]+", text) if token]
            if not raw:
                raise ValueError("select at least one of 1=line, 2=gap, 3=sidewall")
            tokens = [normalize_measurement_type(token) for token in raw]
    selected = set(tokens)
    return [kind for kind in MEASUREMENT_ORDER if kind in selected]


def ask_measurement_types() -> list[str]:
    while True:
        value = input("측정 종류 [1=line, 2=gap, 3=sidewall]: ").strip()
        try:
            return parse_measurement_types(value)
        except ValueError as error:
            print(error)


def measurement_rows(native_height: int) -> list[tuple[float, int]]:
    if native_height <= 0:
        raise ValueError("native_height must be positive")
    return [
        (fraction, round((native_height - 1) * fraction))
        for fraction in Y_FRACTIONS
    ]


def select_central_targets(
    measurements: list[dict],
    measurement_type: str,
    *,
    center_field: str = "center_x_px",
) -> list[dict]:
    """Select five middle targets and assign stable left-to-right IDs."""
    kind = normalize_measurement_type(measurement_type)
    ordered = sorted(measurements, key=lambda item: float(item[center_field]))
    if len(ordered) > TARGETS_PER_ROW:
        start = (len(ordered) - TARGETS_PER_ROW) // 2
        ordered = ordered[start : start + TARGETS_PER_ROW]
    return [
        {**item, "target_id": f"{TARGET_PREFIX[kind]}{number}"}
        for number, item in enumerate(ordered, start=1)
    ]


def save_prediction_csv(
    output_path: str | Path,
    method: str,
    rows: list[dict],
    *,
    um_per_px: float = UM_PER_NATIVE_PX,
) -> Path:
    """Write the one common prediction CSV accepted by the benchmark."""
    normalized = []
    for index, source in enumerate(rows, start=1):
        row = dict(source)
        for field in ("image_id", "measurement_type", "y_fraction", "target_id"):
            if str(row.get(field, "")).strip() == "":
                raise ValueError(f"prediction row {index}: missing {field}")
        kind = normalize_measurement_type(row["measurement_type"])
        y_fraction = float(row["y_fraction"])
        if y_fraction > 1:
            y_fraction /= 100.0
        left = _optional_float(row.get("left_x_px"))
        right = _optional_float(row.get("right_x_px"))
        center = _optional_float(row.get("center_x_px"))
        width_px = _optional_float(row.get("width_px"))
        width_um = _optional_float(row.get("width_um"))
        if left is not None and right is not None:
            if right <= left:
                raise ValueError(f"prediction row {index}: right edge must exceed left edge")
            width_px = width_px if width_px is not None else right - left
            center = center if center is not None else (left + right) / 2.0
        if width_px is None and width_um is not None:
            width_px = width_um / um_per_px
        if width_um is None and width_px is not None:
            width_um = width_px * um_per_px
        status = str(row.get("status", "ok") or "ok").lower()
        if status == "ok" and (width_px is None or width_px <= 0):
            raise ValueError(f"prediction row {index}: no valid width")
        normalized.append(
            {
                "method": method,
                "image_id": str(row["image_id"]),
                "measurement_type": kind,
                "y_fraction": round(y_fraction, 6),
                "target_id": str(row["target_id"]),
                "width_px": _rounded(width_px),
                "width_um": _rounded(width_um),
                "center_x_px": _rounded(center),
                "left_x_px": _rounded(left),
                "right_x_px": _rounded(right),
                "runtime_ms": row.get("runtime_ms", ""),
                "status": status,
                "message": row.get("message", ""),
            }
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)
    return output


def _optional_float(value):
    return None if value in (None, "") else float(value)


def _rounded(value):
    return None if value is None else round(float(value), 6)


if __name__ == "__main__":
    selected = ask_measurement_types()
    print("선택:", ", ".join(selected))
    for kind in selected:
        print(f"{kind} 정의:", MEASUREMENT_DEFINITIONS[kind])
    print("native 높이 641의 Y:", measurement_rows(641))

