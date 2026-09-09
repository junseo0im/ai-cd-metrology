"""Read-only dataset audit utilities for ROI decision preparation.

This module never writes into the raw dataset. It records image metadata,
conservative path-derived labels, duplicate candidates, and lightweight geometry
diagnostics in an ignored derived-data directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})

AUDIT_COLUMNS = [
    "relative_path",
    "filename",
    "extension",
    "width_px",
    "height_px",
    "channels",
    "dtype",
    "file_size_bytes",
    "readable",
    "wafer_id",
    "die_id",
    "pattern_position",
    "status",
    "defect_label",
    "exposure_condition",
    "top_level",
    "sub_dataset",
    "metadata_notes",
    "sha256",
    "duplicate_filename_count",
    "duplicate_size_count",
    "exact_duplicate_count",
]

_CANONICAL_IMAGE_NAME = re.compile(
    r"^w(?P<wafer>\d+)_(?P<die>\d{2})_(?P<position>c|lt|rb)"
    r"(?:_(?P<defect>p|pd|pn|s|nc))?$",
    re.IGNORECASE,
)
_POSITION_NAMES = {"c": "center", "lt": "left-top", "rb": "right-bottom"}
_DEFECT_CODES = {
    "p": "particle",
    "pd": "pattern_die",
    "pn": "pattern_narrowing",
    "s": "short",
    "nc": "not_critical",
}
_FAIL_DEFECTS = {"open", "particle", "pattern_die", "pattern_narrowing", "short"}


def _as_relative_path(relative_path: str | Path) -> PurePosixPath:
    return PurePosixPath(str(relative_path).replace("\\", "/"))


def classify_sub_dataset(relative_path: str | Path) -> tuple[str, str]:
    """Return stable top-level and sub-dataset labels without guessing metadata."""

    path = _as_relative_path(relative_path)
    directories = path.parts[:-1]
    if not directories:
        return "unknown", "unknown"

    top_level = directories[0]
    if top_level == "images" and len(directories) >= 3:
        depth = 4 if directories[2] == "FAIL" and len(directories) >= 4 else 3
        return top_level, "/".join(directories[:depth])
    if top_level == "결함 총합" and len(directories) >= 3:
        return top_level, "/".join(directories[:3])
    if top_level == "웨이퍼 사진" and len(directories) >= 3:
        return top_level, "/".join(directories[:3])
    return top_level, "/".join(directories)


def _exposure_candidates(path_text: str) -> set[str]:
    candidates: set[str] = set()
    patterns = (
        r"노광[\s_-]*(60|70)(?=$|[\\/\s_-])",
        r"(?:^|[\\/\s_-])E(60|70)(?=$|[\\/\s_-])",
        r"(?:^|[\\/])(60|70)[\s_-]*wafer(?=$|[\\/\s_-])",
    )
    for pattern in patterns:
        candidates.update(re.findall(pattern, path_text, flags=re.IGNORECASE))
    return candidates


def parse_path_metadata(relative_path: str | Path) -> dict[str, str | None]:
    """Extract only metadata supported by explicit path or filename tokens."""

    path = _as_relative_path(relative_path)
    parts = path.parts
    lower_parts = [part.casefold() for part in parts]
    notes: list[str] = []

    wafer_id: str | None = None
    die_id: str | None = None
    pattern_position: str | None = None
    status: str | None = None
    defect_label: str | None = None

    canonical_match = _CANONICAL_IMAGE_NAME.fullmatch(path.stem)
    if canonical_match:
        wafer_id = f"w{canonical_match.group('wafer')}"
        die_id = canonical_match.group("die")
        position_code = canonical_match.group("position").casefold()
        pattern_position = _POSITION_NAMES[position_code]
        defect_code = canonical_match.group("defect")
        if defect_code:
            defect_label = _DEFECT_CODES[defect_code.casefold()]
            status = "WARNING" if defect_code.casefold() == "nc" else "FAIL"

    if wafer_id is None:
        for part in parts:
            match = re.match(r"^wafer\s*(\d+)(?:$|[\s_-])", part, flags=re.IGNORECASE)
            if match:
                wafer_id = f"w{match.group(1)}"
                break

    if "pass_normal" in lower_parts:
        status = "PASS"
        defect_label = "normal"
    elif "warning_nc" in lower_parts:
        status = "WARNING"
        defect_label = "not_critical"
    elif "fail" in lower_parts:
        status = "FAIL"

    folder_defects = _FAIL_DEFECTS.intersection(lower_parts)
    if len(folder_defects) == 1:
        folder_defect = next(iter(folder_defects))
        if defect_label is not None and defect_label != folder_defect:
            notes.append(f"conflicting defect labels: {defect_label}, {folder_defect}")
            defect_label = None
        else:
            defect_label = folder_defect
    elif len(folder_defects) > 1:
        notes.append("multiple defect folder labels")
        defect_label = None

    exposure_candidates = _exposure_candidates("/".join(parts))
    if len(exposure_candidates) == 1:
        exposure_condition: str | None = next(iter(exposure_candidates))
    elif len(exposure_candidates) > 1:
        exposure_condition = None
        notes.append(
            "conflicting exposure tokens: " + ", ".join(sorted(exposure_candidates))
        )
    else:
        exposure_condition = None

    return {
        "wafer_id": wafer_id,
        "die_id": die_id,
        "pattern_position": pattern_position,
        "status": status,
        "defect_label": defect_label,
        "exposure_condition": exposure_condition,
        "metadata_notes": "; ".join(notes) or None,
    }


def _decode_image_bytes(file_bytes: bytes) -> np.ndarray | None:
    encoded = np.frombuffer(file_bytes, dtype=np.uint8)
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)


def _image_shape(image: np.ndarray) -> tuple[int, int, int]:
    height_px, width_px = image.shape[:2]
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    return int(width_px), int(height_px), channels


def audit_images(raw_root: Path) -> pd.DataFrame:
    """Read every supported image recursively and return one audit row per file."""

    raw_root = raw_root.resolve()
    paths = sorted(
        (
            path
            for path in raw_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(raw_root).as_posix().casefold(),
    )

    rows: list[dict[str, Any]] = []
    for path in paths:
        relative_path = path.relative_to(raw_root).as_posix()
        metadata = parse_path_metadata(relative_path)
        top_level, sub_dataset = classify_sub_dataset(relative_path)
        file_size_bytes = path.stat().st_size
        readable = False
        width_px: int | None = None
        height_px: int | None = None
        channels: int | None = None
        dtype: str | None = None
        read_error: str | None = None

        try:
            file_bytes = path.read_bytes()
            sha256 = hashlib.sha256(file_bytes).hexdigest()
            image = _decode_image_bytes(file_bytes)
            if image is None:
                read_error = "OpenCV could not decode image"
            else:
                width_px, height_px, channels = _image_shape(image)
                dtype = str(image.dtype)
                readable = True
        except (OSError, ValueError) as error:
            sha256 = None
            read_error = f"{type(error).__name__}: {error}"

        notes = [value for value in (metadata.pop("metadata_notes"), read_error) if value]
        rows.append(
            {
                "relative_path": relative_path,
                "filename": path.name,
                "extension": path.suffix.casefold(),
                "width_px": width_px,
                "height_px": height_px,
                "channels": channels,
                "dtype": dtype,
                "file_size_bytes": file_size_bytes,
                "readable": readable,
                **metadata,
                "top_level": top_level,
                "sub_dataset": sub_dataset,
                "metadata_notes": "; ".join(notes) or None,
                "sha256": sha256,
            }
        )

    audit = pd.DataFrame(rows)
    if audit.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    audit["duplicate_filename_count"] = audit.groupby("filename")["filename"].transform("size")
    audit["duplicate_size_count"] = audit.groupby("file_size_bytes")["file_size_bytes"].transform("size")
    audit["exact_duplicate_count"] = audit.groupby("sha256", dropna=False)["sha256"].transform("size")
    return audit[AUDIT_COLUMNS]


def duplicate_groups(audit: pd.DataFrame) -> pd.DataFrame:
    """Separate filename, file-size, and exact-hash duplicate candidate groups."""

    output_columns = ["group_type", "match_value", "file_count", "relative_paths"]
    rows: list[dict[str, Any]] = []
    for group_type, column in (
        ("same_filename", "filename"),
        ("same_file_size", "file_size_bytes"),
        ("exact_sha256", "sha256"),
    ):
        for value, group in audit.dropna(subset=[column]).groupby(column, sort=True):
            if len(group) < 2:
                continue
            rows.append(
                {
                    "group_type": group_type,
                    "match_value": str(value),
                    "file_count": len(group),
                    "relative_paths": " | ".join(sorted(group["relative_path"])),
                }
            )
    return pd.DataFrame(rows, columns=output_columns)


def summary_table(audit: pd.DataFrame) -> pd.DataFrame:
    """Build normalized count tables suitable for one CSV."""

    rows: list[dict[str, Any]] = []

    def add_counts(category: str, values: Iterable[Any]) -> None:
        series = pd.Series(list(values), dtype="object").fillna("unknown").astype(str)
        for key, count in series.value_counts(dropna=False).sort_index().items():
            rows.append({"category": category, "key": key, "count": int(count)})

    rows.extend(
        [
            {"category": "overall", "key": "total", "count": len(audit)},
            {
                "category": "overall",
                "key": "readable",
                "count": int(audit["readable"].sum()) if not audit.empty else 0,
            },
            {
                "category": "overall",
                "key": "unreadable",
                "count": int((~audit["readable"]).sum()) if not audit.empty else 0,
            },
        ]
    )
    if audit.empty:
        return pd.DataFrame(rows, columns=["category", "key", "count"])

    add_counts("extension", audit["extension"])
    add_counts(
        "resolution",
        audit.apply(
            lambda row: (
                f"{int(row.width_px)}x{int(row.height_px)}" if row.readable else "unreadable"
            ),
            axis=1,
        ),
    )
    add_counts(
        "channels_dtype",
        audit.apply(
            lambda row: (
                f"{int(row.channels)}ch/{row.dtype}" if row.readable else "unreadable"
            ),
            axis=1,
        ),
    )
    for category in (
        "top_level",
        "sub_dataset",
        "status",
        "defect_label",
        "wafer_id",
        "exposure_condition",
    ):
        add_counts(category, audit[category])
    return pd.DataFrame(rows, columns=["category", "key", "count"])


def select_geometry_samples(audit: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Select deterministic, stratified samples plus obvious screenshot cases."""

    if limit <= 0 or audit.empty:
        return audit.iloc[0:0].copy()

    readable = audit[audit["readable"]].sort_values("relative_path")
    selected_indices: list[int] = []

    screenshot_mask = readable["filename"].str.contains("스크린샷|screenshot", case=False, regex=True)
    selected_indices.extend(readable[screenshot_mask].index.tolist())

    for _, group in readable.groupby("sub_dataset", sort=True):
        ordered = group.sort_values("relative_path")
        selected_indices.append(int(ordered.index[len(ordered) // 2]))

    for index in readable.index:
        if len(dict.fromkeys(selected_indices)) >= limit:
            break
        selected_indices.append(int(index))

    unique_indices = list(dict.fromkeys(selected_indices))[:limit]
    return audit.loc[unique_indices].copy()


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def geometry_diagnostics(image: np.ndarray) -> dict[str, float | str | bool | None]:
    """Calculate non-metrology orientation/alignment indicators on a full image."""

    gray = _to_grayscale(image)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    total_magnitude = float(magnitude.sum())
    if total_magnitude <= 0:
        return {
            "dominant_line_angle_deg": None,
            "vertical_deviation_deg": None,
            "orientation_confidence": 0.0,
            "gradient_center_x_norm": None,
            "gradient_center_y_norm": None,
            "border_gradient_fraction": None,
            "orientation_hint": "manual_review_needed",
        }

    jxx = float(np.square(gradient_x, dtype=np.float64).sum())
    jyy = float(np.square(gradient_y, dtype=np.float64).sum())
    jxy = float(np.multiply(gradient_x, gradient_y, dtype=np.float64).sum())
    gradient_angle = 0.5 * math.degrees(math.atan2(2.0 * jxy, jxx - jyy))
    line_angle = ((gradient_angle + 180.0) % 180.0) - 90.0
    vertical_deviation = 90.0 - abs(line_angle)
    confidence = math.hypot(jxx - jyy, 2.0 * jxy) / max(jxx + jyy, 1e-12)

    height, width = gray.shape[:2]
    x_coordinates = np.arange(width, dtype=np.float64)
    y_coordinates = np.arange(height, dtype=np.float64)
    center_x = float((magnitude.sum(axis=0) * x_coordinates).sum() / total_magnitude)
    center_y = float((magnitude.sum(axis=1) * y_coordinates).sum() / total_magnitude)
    center_x_norm = center_x / max(width - 1, 1)
    center_y_norm = center_y / max(height - 1, 1)

    border_y = max(1, round(height * 0.1))
    border_x = max(1, round(width * 0.1))
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:border_y, :] = True
    border_mask[-border_y:, :] = True
    border_mask[:, :border_x] = True
    border_mask[:, -border_x:] = True
    border_fraction = float(magnitude[border_mask].sum() / total_magnitude)

    if confidence < 0.1:
        orientation_hint = "manual_review_needed"
    elif vertical_deviation <= 15.0:
        orientation_hint = "vertical_candidate"
    elif abs(line_angle) <= 15.0:
        orientation_hint = "horizontal_candidate"
    else:
        orientation_hint = "oblique_or_mixed_candidate"

    return {
        "dominant_line_angle_deg": round(line_angle, 3),
        "vertical_deviation_deg": round(vertical_deviation, 3),
        "orientation_confidence": round(confidence, 6),
        "gradient_center_x_norm": round(center_x_norm, 6),
        "gradient_center_y_norm": round(center_y_norm, 6),
        "border_gradient_fraction": round(border_fraction, 6),
        "orientation_hint": orientation_hint,
    }


def audit_geometry_samples(
    raw_root: Path,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """Run full-image geometry diagnostics for a small representative sample."""

    rows: list[dict[str, Any]] = []
    for sample in samples.itertuples(index=False):
        path = raw_root / PurePosixPath(sample.relative_path)
        image = _decode_image_bytes(path.read_bytes())
        diagnostics = geometry_diagnostics(image) if image is not None else {}
        review_reasons = ["confirm pattern alignment, crop, and UI/ruler/text visually"]
        if "스크린샷" in sample.filename.casefold() or "screenshot" in sample.filename.casefold():
            review_reasons.append("screenshot filename")
        if sample.status == "FAIL" or sample.defect_label not in (None, "normal"):
            review_reasons.append("confirm whether defect obscures a straight measurement region")
        if diagnostics.get("orientation_hint") != "vertical_candidate":
            review_reasons.append("automated orientation indicator is not confidently vertical")

        rows.append(
            {
                "relative_path": sample.relative_path,
                "sub_dataset": sample.sub_dataset,
                "width_px": sample.width_px,
                "height_px": sample.height_px,
                "channels": sample.channels,
                "dtype": sample.dtype,
                **diagnostics,
                "ui_ruler_text_overlay": "manual_review_needed",
                "defect_occlusion": "manual_review_needed",
                "manual_review_needed": True,
                "manual_review_reasons": "; ".join(review_reasons),
            }
        )
    return pd.DataFrame(rows)


def write_audit_outputs(
    raw_root: Path,
    output_dir: Path,
    *,
    geometry_sample_limit: int = 20,
) -> dict[str, Path]:
    """Run the audit and write CSV-only derived outputs outside ``data/raw``."""

    raw_root = raw_root.resolve()
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("output_dir must not be inside the raw dataset")

    output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_images(raw_root)
    duplicates = duplicate_groups(audit)
    summary = summary_table(audit)
    samples = select_geometry_samples(audit, geometry_sample_limit)
    geometry = audit_geometry_samples(raw_root, samples)

    outputs = {
        "image_audit": output_dir / "image_audit.csv",
        "summary": output_dir / "audit_summary.csv",
        "duplicates": output_dir / "duplicate_groups.csv",
        "geometry": output_dir / "geometry_audit.csv",
    }
    audit.to_csv(outputs["image_audit"], index=False, encoding="utf-8-sig")
    summary.to_csv(outputs["summary"], index=False, encoding="utf-8-sig")
    duplicates.to_csv(outputs["duplicates"], index=False, encoding="utf-8-sig")
    geometry.to_csv(outputs["geometry"], index=False, encoding="utf-8-sig")
    return outputs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/audit"))
    parser.add_argument("--geometry-samples", type=int, default=20)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    outputs = write_audit_outputs(
        args.raw_root,
        args.output_dir,
        geometry_sample_limit=args.geometry_samples,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

