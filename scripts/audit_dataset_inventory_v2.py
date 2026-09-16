"""Build a read-only image inventory and exact-duplicate audit.

Only image bytes and filesystem metadata are read from ``data/raw``. The sole
data artifact written by this script is the requested inventory CSV under the
git-ignored ``data/derived`` tree.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
ROOT_SPECS = (
    ("CURATED_IMAGES", Path("data/raw/images/images")),
    ("NEW_DEFECT_70_2_E70_F", Path("data/raw/70 2/E70 F")),
    ("DEFECT_TOTAL", Path("data/raw/결함 총합/결함 총합")),
    ("WAFER_PHOTOS_E70", Path("data/raw/웨이퍼 사진/노광 70")),
)
KNOWN_NAME = re.compile(
    r"^(?P<wafer>w\d+)_(?P<die>\d+)_(?P<position>rb|c|lt)(?:_(?P<suffix>.+))?$",
    re.IGNORECASE,
)
WAFER_FOLDER = re.compile(r"^wafer(?P<number>\d+)(?:-E\d+)?$", re.IGNORECASE)
KNOWN_DEFECT_LABELS = frozenset({"o", "s", "pd", "pn", "p", "nc"})
CURATED_FOLDER_LABELS = {
    "open": "o",
    "particle": "p",
    "pattern_die": "pd",
    "pattern_narrowing": "pn",
    "short": "s",
}

CSV_COLUMNS = [
    "source_root",
    "source_root_path",
    "source_group",
    "relative_path",
    "source_relative_path",
    "filename",
    "extension",
    "width",
    "height",
    "channels",
    "image_mode",
    "dtype",
    "readable",
    "file_size_bytes",
    "sha256",
    "wafer_folder",
    "wafer_id",
    "wafer_id_source",
    "die_id",
    "pattern_position",
    "defect_label",
    "defect_label_source",
    "curated_status",
    "exposure_tokens_literal",
    "filename_parse_status",
    "metadata_notes",
    "duplicate_group_id",
    "duplicate_count",
    "duplicate_same_folder",
    "duplicate_cross_source",
    "duplicate_matches_curated",
    "audit_classification",
]


def _known_filename_metadata(stem: str) -> dict[str, str]:
    match = KNOWN_NAME.fullmatch(stem)
    if not match:
        return {
            "wafer_id": "unknown",
            "die_id": "unknown",
            "pattern_position": "unknown",
            "defect_label": "unknown",
            "filename_parse_status": "unknown",
            "metadata_notes": "filename does not match the known wN_die_position[_label] pattern",
        }

    suffix = match.group("suffix")
    defect_label = "unknown"
    notes = ""
    parse_status = "base_fields_only"
    if suffix is None:
        parse_status = "base_fields_parsed_no_defect_suffix"
    elif suffix.casefold() in KNOWN_DEFECT_LABELS:
        defect_label = suffix.casefold()
        parse_status = "fully_parsed"
    else:
        notes = f"unrecognized or compound defect suffix preserved as unknown: {suffix}"

    return {
        "wafer_id": match.group("wafer").casefold(),
        "die_id": match.group("die"),
        "pattern_position": match.group("position").casefold(),
        "defect_label": defect_label,
        "filename_parse_status": parse_status,
        "metadata_notes": notes,
    }


def _folder_metadata(path: Path, source_root: Path, source_name: str) -> dict[str, str]:
    relative_parts = path.relative_to(source_root).parts
    directory_parts = relative_parts[:-1]

    wafer_folder = "unknown"
    wafer_id_from_folder = "unknown"
    for part in directory_parts:
        match = WAFER_FOLDER.fullmatch(part)
        if match:
            wafer_folder = part
            wafer_id_from_folder = f"w{match.group('number')}"
            break

    exposure_tokens: list[str] = []
    for part in path.parts:
        if "노광 60" in part or "노광 70" in part or "E60" in part or "E70" in part:
            exposure_tokens.append(part)

    if source_name == "CURATED_IMAGES":
        if not directory_parts:
            source_group = "unknown"
        elif directory_parts[0] == "FAIL" and len(directory_parts) >= 2:
            source_group = f"FAIL/{directory_parts[1]}"
        else:
            source_group = directory_parts[0]
    elif directory_parts:
        source_group = "/".join(directory_parts)
    else:
        source_group = "."

    return {
        "source_group": source_group,
        "wafer_folder": wafer_folder,
        "wafer_id_from_folder": wafer_id_from_folder,
        "exposure_tokens_literal": " | ".join(dict.fromkeys(exposure_tokens)) or "unknown",
    }


def _curated_metadata(path: Path, source_root: Path, source_name: str) -> tuple[str, str]:
    if source_name != "CURATED_IMAGES":
        return "unknown", "unknown"
    parts = path.relative_to(source_root).parts
    if not parts:
        return "unknown", "unknown"
    if parts[0] == "PASS_normal":
        return "PASS", "normal"
    if parts[0] == "WARNING_nc":
        return "WARNING", "nc"
    if parts[0] == "FAIL":
        folder_label = CURATED_FOLDER_LABELS.get(parts[1], "unknown") if len(parts) >= 2 else "unknown"
        return "FAIL", folder_label
    return "unknown", "unknown"


def _decode_properties(file_bytes: bytes) -> tuple[bool, int | str, int | str, int | str, str, str, str]:
    try:
        encoded = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None
        if image is None:
            return False, "", "", "", "unknown", "unknown", "OpenCV could not decode image"
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else int(image.shape[2])
        mode = {1: "grayscale", 2: "2-channel", 3: "BGR", 4: "BGRA"}.get(channels, f"{channels}-channel")
        return True, int(width), int(height), channels, mode, str(image.dtype), ""
    except (cv2.error, ValueError) as error:
        return False, "", "", "", "unknown", "unknown", f"{type(error).__name__}: {error}"


def _iter_image_paths(project_root: Path) -> tuple[list[tuple[str, Path, Path]], list[str]]:
    records: list[tuple[str, Path, Path]] = []
    non_image_files: list[str] = []
    seen: set[str] = set()
    for source_name, relative_root in ROOT_SPECS:
        source_root = (project_root / relative_root).resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"required source root not found: {source_root}")
        for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix().casefold()):
            if not path.is_file():
                continue
            key = str(path.resolve()).casefold()
            if key in seen:
                raise ValueError(f"source roots overlap at: {path}")
            seen.add(key)
            if path.suffix.casefold() not in IMAGE_EXTENSIONS:
                non_image_files.append(path.relative_to(project_root / "data/raw").as_posix())
                continue
            records.append((source_name, source_root, path))
    return records, non_image_files


def build_inventory(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_root = (project_root / "data/raw").resolve()
    paths, non_image_files = _iter_image_paths(project_root)
    rows: list[dict[str, Any]] = []

    for source_name, source_root, path in paths:
        relative_path = path.relative_to(raw_root).as_posix()
        source_relative_path = path.relative_to(source_root).as_posix()
        filename_metadata = _known_filename_metadata(path.stem)
        folder_metadata = _folder_metadata(path, source_root, source_name)
        curated_status, folder_defect_label = _curated_metadata(path, source_root, source_name)
        try:
            file_bytes = path.read_bytes()
            sha256 = hashlib.sha256(file_bytes).hexdigest()
            readable, width, height, channels, image_mode, dtype, read_note = _decode_properties(file_bytes)
        except OSError as error:
            sha256 = ""
            readable, width, height, channels = False, "", "", ""
            image_mode, dtype = "unknown", "unknown"
            read_note = f"{type(error).__name__}: {error}"

        if filename_metadata["wafer_id"] != "unknown":
            wafer_id = filename_metadata["wafer_id"]
            wafer_id_source = "filename"
        elif folder_metadata["wafer_id_from_folder"] != "unknown":
            wafer_id = folder_metadata["wafer_id_from_folder"]
            wafer_id_source = "wafer_folder"
        else:
            wafer_id = "unknown"
            wafer_id_source = "unknown"

        if filename_metadata["defect_label"] != "unknown":
            defect_label = filename_metadata["defect_label"]
            defect_label_source = "filename"
        elif folder_defect_label != "unknown":
            defect_label = folder_defect_label
            defect_label_source = "curated_folder"
        else:
            defect_label = "unknown"
            defect_label_source = "unknown"

        notes = [note for note in (filename_metadata["metadata_notes"], read_note) if note]
        rows.append(
            {
                "source_root": source_name,
                "source_root_path": source_root.relative_to(project_root).as_posix(),
                "source_group": folder_metadata["source_group"],
                "relative_path": relative_path,
                "source_relative_path": source_relative_path,
                "filename": path.name,
                "extension": path.suffix.casefold(),
                "width": width,
                "height": height,
                "channels": channels,
                "image_mode": image_mode,
                "dtype": dtype,
                "readable": readable,
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256,
                "wafer_folder": folder_metadata["wafer_folder"],
                "wafer_id": wafer_id,
                "wafer_id_source": wafer_id_source,
                "die_id": filename_metadata["die_id"],
                "pattern_position": filename_metadata["pattern_position"],
                "defect_label": defect_label,
                "defect_label_source": defect_label_source,
                "curated_status": curated_status,
                "exposure_tokens_literal": folder_metadata["exposure_tokens_literal"],
                "filename_parse_status": filename_metadata["filename_parse_status"],
                "metadata_notes": "; ".join(notes),
            }
        )

    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["sha256"]:
            hash_groups[row["sha256"]].append(row)
    duplicate_hashes = sorted(digest for digest, group in hash_groups.items() if len(group) > 1)
    duplicate_ids = {digest: f"DUP{index:04d}" for index, digest in enumerate(duplicate_hashes, start=1)}

    curated_hashes = {
        row["sha256"] for row in rows if row["source_root"] == "CURATED_IMAGES" and row["sha256"]
    }
    for row in rows:
        group = hash_groups.get(row["sha256"], []) if row["sha256"] else []
        parents = {Path(item["relative_path"]).parent.as_posix() for item in group}
        sources = {item["source_root"] for item in group}
        has_curated = any(item["source_root"] == "CURATED_IMAGES" for item in group)
        row["duplicate_group_id"] = duplicate_ids.get(row["sha256"], "")
        row["duplicate_count"] = len(group) if group else 0
        row["duplicate_same_folder"] = len(group) > 1 and len(parents) < len(group)
        row["duplicate_cross_source"] = len(sources) > 1
        row["duplicate_matches_curated"] = has_curated and row["source_root"] != "CURATED_IMAGES"
        if row["source_root"] == "CURATED_IMAGES":
            row["audit_classification"] = "CURATED"
        elif row["sha256"] and row["sha256"] in curated_hashes:
            row["audit_classification"] = "CURATED_DUPLICATE"
        else:
            row["audit_classification"] = "NEW_UNIQUE"

    rows.sort(key=lambda row: (row["source_root"], row["relative_path"].casefold()))
    duplicate_groups = [hash_groups[digest] for digest in duplicate_hashes]
    resolution_counts = Counter(
        f"{row['width']}x{row['height']}" if row["readable"] else "unreadable" for row in rows
    )
    parsed_counts = {
        field: sum(row[field] != "unknown" for row in rows)
        for field in ("wafer_id", "die_id", "pattern_position", "defect_label")
    }
    summary = {
        "total_images": len(rows),
        "readable": sum(bool(row["readable"]) for row in rows),
        "unreadable": sum(not bool(row["readable"]) for row in rows),
        "source_root_counts": dict(sorted(Counter(row["source_root"] for row in rows).items())),
        "curated_status_counts": dict(sorted(Counter(
            row["curated_status"] for row in rows if row["source_root"] == "CURATED_IMAGES"
        ).items())),
        "classification_counts": dict(sorted(Counter(row["audit_classification"] for row in rows).items())),
        "resolution_kind_count": len(resolution_counts),
        "resolution_counts": dict(resolution_counts.most_common()),
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_file_count": sum(len(group) for group in duplicate_groups),
        "exact_duplicate_excess_copy_count": sum(len(group) - 1 for group in duplicate_groups),
        "same_folder_duplicate_group_count": sum(
            len({Path(row["relative_path"]).parent.as_posix() for row in group}) < len(group)
            for group in duplicate_groups
        ),
        "cross_source_duplicate_group_count": sum(
            len({row["source_root"] for row in group}) > 1 for group in duplicate_groups
        ),
        "curated_bridge_duplicate_group_count": sum(
            "CURATED_IMAGES" in {row["source_root"] for row in group}
            and len({row["source_root"] for row in group}) > 1
            for group in duplicate_groups
        ),
        "source_group_counts": dict(sorted(Counter(
            f"{row['source_root']}::{row['source_group']}" for row in rows
        ).items())),
        "wafer_group_counts": dict(sorted(Counter(
            f"{row['source_root']}::{row['wafer_folder']}" for row in rows
        ).items())),
        "parsed_counts": parsed_counts,
        "unknown_counts": {field: len(rows) - count for field, count in parsed_counts.items()},
        "non_image_file_count": len(non_image_files),
        "non_image_files": non_image_files,
        "unreadable_paths": [row["relative_path"] for row in rows if not row["readable"]],
    }
    return rows, summary


def write_inventory(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/dataset_audit_v2/dataset_inventory.csv"),
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    rows, summary = build_inventory(project_root)
    write_inventory(output_path, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"inventory_csv: {output_path}")


if __name__ == "__main__":
    main()
