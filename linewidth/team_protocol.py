"""Standalone protocol shared by every line-width measurement program.

Give this one file to each teammate and import it from their own algorithm.
It standardizes only the experiment interface: measurement types, native Y rows,
central target IDs, and prediction CSV output.  It does not contain U-Net or
any other edge-detection algorithm.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


UM_PER_NATIVE_PX = 0.1592
Y_FRACTIONS = (0.25, 0.45, 0.65)
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
# Measurement types are chosen by name only (the old 1/2/3 number-key menu was
# removed).  The dashboard Metrology module always measures line and gap.
MEASUREMENT_KEY_MAP = {
    "line": "line",
    "gap": "gap",
    "sidewall": "sidewall",
}
DEFAULT_MEASUREMENT_TYPES = ("line", "gap")
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


def normalize_measurement_type(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return MEASUREMENT_KEY_MAP[key]
    except KeyError as error:
        raise ValueError("measurement type must be line, gap or sidewall") from error


def parse_measurement_types(value=None) -> list[str]:
    """Parse 'line gap', 'line,gap' or a list of names; None means line and gap."""
    if value is None:
        return list(DEFAULT_MEASUREMENT_TYPES)
    if isinstance(value, (list, tuple, set)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(parse_measurement_types(item))
    else:
        raw = [token for token in re.split(r"[\s,;+]+", str(value).strip().lower()) if token]
        if not raw:
            raise ValueError("select at least one of line, gap, sidewall")
        tokens = [normalize_measurement_type(token) for token in raw]
    selected = set(tokens)
    return [kind for kind in MEASUREMENT_ORDER if kind in selected]


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


SUMMARY_FIELDS = (
    "image_name", "measurement_type", "target_id", "mean_um", "std_um",
    "valid_points", "status", "native_width", "native_height",
    "duplicate_factor_x", "duplicate_factor_y", "processing_time_s",
    "timing_included",
)


def save_summary_csv(summary, output_path: str | Path) -> Path:
    """Save the per-target summary table (one row per target averaged over Y rows).

    Used by the threshold program's folder report; the benchmark uses
    save_prediction_csv instead.
    """
    import pandas as pd

    frame = summary.copy() if isinstance(summary, pd.DataFrame) else pd.DataFrame(summary)
    missing = [column for column in SUMMARY_FIELDS if column not in frame.columns]
    if missing:
        raise ValueError(f"summary columns missing: {missing}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
    return destination
