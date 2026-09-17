"""Dataset metadata loading for the dashboard without batch metrology."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_RELATIVE_PATH = Path("data/derived/dataset_audit_v2/dataset_inventory.csv")
CURATED_ROOT_RELATIVE_PATH = Path("data/raw/images/images")

POSITION_ALIASES = {
    "c": "center",
    "center": "center",
    "lt": "left-top",
    "left-top": "left-top",
    "rb": "right-bottom",
    "right-bottom": "right-bottom",
}

_FILENAME_PATTERN = re.compile(
    r"^(?P<wafer>w\d+)_(?P<die>\d+)_(?P<position>c|lt|rb)(?:_(?P<label>[a-z]+))?$",
    re.IGNORECASE,
)


def canonical_pattern_position(value: object) -> str:
    """Return the evaluation-contract position or ``unknown``."""

    text = str(value or "").strip().lower()
    return POSITION_ALIASES.get(text, "unknown")


def _source_name(source_root: object, source_group: object) -> str:
    root = str(source_root or "unknown").strip() or "unknown"
    group = str(source_group or ".").strip() or "."
    return f"{root}::{group}"


def _normalise_inventory(frame: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "source_root",
        "source_group",
        "relative_path",
        "filename",
        "wafer_id",
        "die_id",
        "pattern_position",
        "defect_label",
        "curated_status",
        "width",
        "height",
    ):
        if column not in result:
            result[column] = ""

    result["image_id"] = result["filename"].map(lambda value: Path(value).stem)
    result["pattern_position"] = result["pattern_position"].map(
        canonical_pattern_position
    )
    result["source"] = [
        _source_name(root, group)
        for root, group in zip(
            result["source_root"], result["source_group"], strict=False
        )
    ]
    result["file_path"] = result["relative_path"].map(
        lambda value: str((project_root / "data/raw" / str(value)).resolve())
    )
    result["capture_region"] = "standard"
    result["resolution"] = [
        f"{width} × {height}" if width and height else "unknown"
        for width, height in zip(result["width"], result["height"], strict=False)
    ]
    result["status_label"] = result["curated_status"].map(
        lambda value: str(value or "unknown").upper()
    )
    result["is_curated"] = result["source_root"].eq("CURATED_IMAGES")
    return result


def _fallback_curated_inventory(project_root: Path) -> pd.DataFrame:
    """Build minimal curated metadata if the derived audit is unavailable."""

    curated_root = project_root / CURATED_ROOT_RELATIVE_PATH
    records: list[dict[str, str]] = []
    if not curated_root.is_dir():
        return pd.DataFrame()

    for path in sorted(curated_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        relative_group = path.parent.relative_to(curated_root).as_posix()
        match = _FILENAME_PATTERN.match(path.stem)
        wafer_id = match.group("wafer") if match else "unknown"
        die_id = match.group("die") if match else "unknown"
        position = canonical_pattern_position(match.group("position") if match else "")
        label = match.group("label") if match and match.group("label") else "unknown"
        top_group = relative_group.split("/")[0]
        status = (
            "PASS" if top_group == "PASS_normal" else
            "WARNING" if top_group == "WARNING_nc" else
            "FAIL" if top_group == "FAIL" else
            "UNKNOWN"
        )
        records.append(
            {
                "source_root": "CURATED_IMAGES",
                "source_group": relative_group,
                "relative_path": path.relative_to(project_root / "data/raw").as_posix(),
                "filename": path.name,
                "wafer_id": wafer_id,
                "die_id": die_id,
                "pattern_position": position,
                "defect_label": label,
                "curated_status": status,
                "width": "",
                "height": "",
            }
        )
    return pd.DataFrame.from_records(records)


def load_dataset_inventory(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load the existing inventory, falling back to curated metadata only."""

    inventory_path = project_root / INVENTORY_RELATIVE_PATH
    if inventory_path.is_file():
        frame = pd.read_csv(inventory_path, dtype=str, keep_default_na=False)
    else:
        frame = _fallback_curated_inventory(project_root)
    if frame.empty:
        return frame
    return _normalise_inventory(frame, project_root)


def curated_catalog(inventory: pd.DataFrame) -> pd.DataFrame:
    """Return the curated subset used for interactive image selection."""

    if inventory.empty or "is_curated" not in inventory:
        return inventory.copy()
    return inventory.loc[inventory["is_curated"]].reset_index(drop=True)


def filter_catalog(
    catalog: pd.DataFrame,
    *,
    source: str | None = None,
    wafer_id: str | None = None,
    die_id: str | None = None,
    pattern_position: str | None = None,
    status_label: str | None = None,
    image_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Apply exact metadata filters without running any measurement."""

    result = catalog
    filters = {
        "source": source,
        "wafer_id": wafer_id,
        "die_id": die_id,
        "pattern_position": pattern_position,
        "status_label": status_label,
    }
    for column, value in filters.items():
        if value and value != "All" and column in result:
            result = result.loc[result[column].eq(value)]
    if image_ids is not None and "image_id" in result:
        result = result.loc[result["image_id"].isin(set(image_ids))]
    return result.reset_index(drop=True)


def overview_snapshot(inventory: pd.DataFrame) -> dict[str, object]:
    """Summarize only metadata already present in the inventory."""

    curated = curated_catalog(inventory)
    known_wafers = {
        value for value in curated.get("wafer_id", pd.Series(dtype=str))
        if value and value != "unknown"
    }
    return {
        "total_image_count": int(len(inventory)),
        "curated_image_count": int(len(curated)),
        "wafer_count": len(known_wafers),
        "source_counts": inventory.get("source_root", pd.Series(dtype=str)).value_counts().to_dict(),
        "status_counts": curated.get("status_label", pd.Series(dtype=str)).value_counts().to_dict(),
        "pattern_position_counts": curated.get(
            "pattern_position", pd.Series(dtype=str)
        ).value_counts().to_dict(),
    }
