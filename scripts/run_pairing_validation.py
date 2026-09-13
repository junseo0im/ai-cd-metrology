"""Validate the existing pixel pairing rule on 24 deterministic PASS images."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from run_gradient_canny_pilot import PROJECT_ROOT, _load_roi_config, _write_png
from run_pairing_pilot import _build_diagnostic

from ai_cd_metrology.image_io import load_image
from ai_cd_metrology.metrology.gradient_canny import (
    PixelPairingCandidate,
    build_pixel_pairing_candidates,
    characterize_signed_x_gradient,
    extract_edge_diagnostics,
    pair_dark_band_doublets,
    validate_pairing_candidates,
)
from ai_cd_metrology.roi import crop_normalized_roi
from ai_cd_metrology.schemas import ImageRecord


FILENAME_PATTERN = re.compile(
    r"^(?P<wafer>w[123])_(?P<die>\d{2})_(?P<position>c|lt|rb)\.png$",
    re.IGNORECASE,
)
POSITION_NAMES = {"c": "center", "lt": "left-top", "rb": "right-bottom"}
POSITION_ORDER = ("c", "lt", "rb")
WAFER_QUOTAS = {"w1": 3, "w2": 3, "w3": 2}


@dataclass(frozen=True, slots=True)
class SelectedImage:
    filename: str
    wafer: str
    die: str
    position_code: str

    @property
    def position(self) -> str:
        return POSITION_NAMES[self.position_code]


def _evenly_spaced_indices(item_count: int, sample_count: int) -> tuple[int, ...]:
    if sample_count <= 0 or item_count < sample_count:
        raise ValueError("sample_count must be positive and no greater than item_count")
    if sample_count == 1:
        return (item_count // 2,)
    return tuple(
        round(index * (item_count - 1) / (sample_count - 1))
        for index in range(sample_count)
    )


def select_validation_images(image_dir: Path) -> tuple[SelectedImage, ...]:
    """Select 8 images per position with deterministic w1/w2/w3 quotas."""

    grouped: dict[tuple[str, str], list[SelectedImage]] = {}
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        match = FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        selected = SelectedImage(
            filename=path.name,
            wafer=match.group("wafer").lower(),
            die=match.group("die"),
            position_code=match.group("position").lower(),
        )
        grouped.setdefault((selected.wafer, selected.position_code), []).append(
            selected
        )

    selection: list[SelectedImage] = []
    for position in POSITION_ORDER:
        for wafer, quota in WAFER_QUOTAS.items():
            group = sorted(
                grouped.get((wafer, position), []),
                key=lambda item: item.filename.casefold(),
            )
            indices = _evenly_spaced_indices(len(group), quota)
            selection.extend(group[index] for index in indices)
    if len(selection) != 24:
        raise ValueError(f"validation selection must contain 24 images, got {len(selection)}")
    return tuple(selection)


def _values(
    candidates: tuple[PixelPairingCandidate, ...], attribute: str
) -> list[int]:
    return [int(getattr(candidate, attribute)) for candidate in candidates]


def _mean_std(values: list[int]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _serialize(values: list[int]) -> str:
    return ";".join(str(value) for value in values)


def _analyze_image(
    selected: SelectedImage,
    image_dir: Path,
    config_path: Path,
) -> tuple[dict[str, object], object, object, object, np.ndarray]:
    path = image_dir / selected.filename
    image = load_image(
        ImageRecord(
            image_id=path.stem,
            file_path=path,
            wafer_id=selected.wafer,
            die_id=selected.die,
            pattern_position=selected.position,
            capture_region="standard",
            defect_label="normal",
        )
    )
    roi = _load_roi_config(config_path, selected.position)
    cropped, _ = crop_normalized_roi(image, roi)
    diagnostics = extract_edge_diagnostics(cropped)
    profile = characterize_signed_x_gradient(diagnostics.x_gradient)
    pairing = pair_dark_band_doublets(profile)
    candidates = build_pixel_pairing_candidates(pairing)
    validation = validate_pairing_candidates(pairing, candidates)

    outer = _values(candidates, "outer_width_px")
    inner = _values(candidates, "inner_width_px")
    gap = _values(candidates, "gap_px")
    outer_mean, outer_std = _mean_std(outer)
    inner_mean, inner_std = _mean_std(inner)
    gap_mean, gap_std = _mean_std(gap)
    rejected_details = ";".join(
        f"x={peak.x}:{peak.polarity}:{peak.reason}"
        for peak in pairing.rejected_peaks
    )
    partial_count = sum(
        peak.reason.startswith("partial_") for peak in pairing.rejected_peaks
    )
    unexpected_count = sum(
        peak.reason.startswith("unexpected_") for peak in pairing.rejected_peaks
    )
    potential_candidate_count = len(range(0, len(pairing.doublets) - 2, 2))
    invalid_candidate_window_count = potential_candidate_count - len(candidates)

    row: dict[str, object] = {
        "filename": selected.filename,
        "wafer": selected.wafer,
        "position": selected.position,
        "width_px": int(image.shape[1]),
        "height_px": int(image.shape[0]),
        "negative_peak_count": len(profile.negative_peak_indices),
        "positive_peak_count": len(profile.positive_peak_indices),
        "accepted_doublet_count": len(pairing.doublets),
        "rejected_peak_count": len(pairing.rejected_peaks),
        "partial_peak_count": partial_count,
        "unexpected_peak_count": unexpected_count,
        "invalid_candidate_window_count": invalid_candidate_window_count,
        "outer_candidate_count": len(outer),
        "inner_candidate_count": len(inner),
        "gap_candidate_count": len(gap),
        "outer_values_px": _serialize(outer),
        "inner_values_px": _serialize(inner),
        "gap_values_px": _serialize(gap),
        "outer_mean_px": outer_mean,
        "outer_std_px": outer_std,
        "inner_mean_px": inner_mean,
        "inner_std_px": inner_std,
        "gap_mean_px": gap_mean,
        "gap_std_px": gap_std,
        "outer_greater_than_inner_count": validation.outer_greater_than_inner_count,
        "catastrophic_mispairing": validation.catastrophic_mispairing,
        "polarity_sequence_normal": len(pairing.rejected_peaks) == 0,
        "rejected_peak_details": rejected_details,
        "validation_status": validation.status,
        "validation_reason": ";".join(validation.reasons),
    }
    return row, profile, pairing, candidates, diagnostics.grayscale_roi


def _diagnostic_filenames(rows: list[dict[str, object]]) -> set[str]:
    selected: set[str] = set()
    for position in POSITION_NAMES.values():
        representative = next(
            (
                row
                for row in rows
                if row["position"] == position
                and row["validation_status"] == "PASS"
                and int(row["rejected_peak_count"]) == 0
            ),
            None,
        )
        if representative is not None:
            selected.add(str(representative["filename"]))

    rejected = sorted(
        (row for row in rows if int(row["rejected_peak_count"]) > 0),
        key=lambda row: (
            -int(row["rejected_peak_count"]),
            -int(row["invalid_candidate_window_count"]),
            -int(row["unexpected_peak_count"]),
            str(row["filename"]),
        ),
    )
    selected.update(str(row["filename"]) for row in rejected[:3])
    selected.update(
        str(row["filename"])
        for row in rows
        if row["validation_status"] == "FAIL"
        or bool(row["catastrophic_mispairing"])
        or int(row["unexpected_peak_count"]) > 0
        or int(row["invalid_candidate_window_count"]) > 0
    )
    return selected


def _pooled_position_summary(
    rows: list[dict[str, object]], position: str
) -> dict[str, object]:
    group = [row for row in rows if row["position"] == position]

    def pooled(column: str) -> list[int]:
        values: list[int] = []
        for row in group:
            text = str(row[column])
            if text:
                values.extend(int(value) for value in text.split(";"))
        return values

    outer = pooled("outer_values_px")
    inner = pooled("inner_values_px")
    gap = pooled("gap_values_px")
    return {
        "image_count": len(group),
        "pass_count": sum(row["validation_status"] == "PASS" for row in group),
        "fail_count": sum(row["validation_status"] == "FAIL" for row in group),
        "mean_rejected_peaks": float(
            np.mean([int(row["rejected_peak_count"]) for row in group])
        ),
        "mean_valid_candidates": float(
            np.mean([int(row["outer_candidate_count"]) for row in group])
        ),
        "outer_mean_std": _mean_std(outer),
        "inner_mean_std": _mean_std(inner),
        "gap_mean_std": _mean_std(gap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/images/images/PASS_normal"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/metrology.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/pairing_validation"),
    )
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    raw_root = (PROJECT_ROOT / "data" / "raw").resolve()
    try:
        output_dir.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = select_validation_images(image_dir)
    rows: list[dict[str, object]] = []
    for selected in selection:
        row, _, _, _, _ = _analyze_image(selected, image_dir, args.config)
        rows.append(row)
        print(
            f"{selected.filename}: {row['validation_status']} | "
            f"rejected={row['rejected_peak_count']} | "
            f"candidates={row['outer_candidate_count']} | "
            f"reason={row['validation_reason'] or 'none'}"
        )

    summary_path = output_dir / "pairing_validation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    diagnostic_names = _diagnostic_filenames(rows)
    selected_by_name = {item.filename: item for item in selection}
    diagnostic_paths: list[Path] = []
    for filename in sorted(diagnostic_names):
        selected = selected_by_name[filename]
        _, profile, pairing, candidates, grayscale = _analyze_image(
            selected, image_dir, args.config
        )
        diagnostic = _build_diagnostic(
            filename,
            selected.position,
            grayscale,
            profile,
            pairing,
            candidates,
        )
        path = output_dir / f"{Path(filename).stem}_pairing_validation.png"
        _write_png(path, diagnostic)
        diagnostic_paths.append(path)

    print(f"summary_csv={summary_path}")
    print("selected=" + ",".join(item.filename for item in selection))
    for position in POSITION_NAMES.values():
        print(f"position_summary {position}: {_pooled_position_summary(rows, position)}")
    print("diagnostics=" + ",".join(str(path) for path in diagnostic_paths))


if __name__ == "__main__":
    main()
