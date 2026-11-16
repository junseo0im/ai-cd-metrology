"""Extract sidewall ground truth from the annotated ``*_i.png`` references.

Most reference screenshots are displayed at native scale, so the horizontal
distance between each pair of colored vertical ticks is the displayed native
pixel width. Three screenshots were saved at a smaller display scale; their
visible numeric labels are transcribed explicitly below and identified as
``manual_transcription`` in the audit CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


Y_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
GT_FIELDS = (
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
AUDIT_FIELDS = (
    "image_id",
    "y_fraction",
    "source_method",
    "detected_count",
    "selected_count",
    "selected_widths_px",
    "reference_image_path",
    "note",
)


# Values read left-to-right from the visible labels. When six values are
# present, the common protocol's central-five rule selects the first five
# because (6 - 5) // 2 == 0.
MANUAL_OVERRIDES = {
    "laptop1__8": (
        [3, 2, 3, 2, 5, 3],
        [3, 3, 2, 3, 4, 2],
        [2, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3],
    ),
    "laptop1__14": (
        [4, 3, 3, 4],
        [3, 4, 4, 3],
        [4, 4, 5, 3],
        [4, 4, 4, 3],
        [3, 3, 3, 4],
    ),
    "laptop1__9": (
        [3, 3, 3, 3, 2, 3],
        [3, 2, 3, 4, 3, 3],
        [2, 3, 3, 3, 3, 3],
        [2, 3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3, 3],
    ),
    "laptop2__2": (
        [6, 6, 7, 6, 6, 7],
        [6, 7, 6, 7, 6, 7],
        [5, 6, 6, 7, 5, 6],
        [6, 6, 6, 7, 6, 7],
        [6, 6, 6, 7, 7, 6],
    ),
    "laptop2__4": (
        [7, 7, 7, 7, 7, 7],
        [7, 7, 6, 7, 6, 7],
        [6, 7, 6, 7, 7, 7],
        [7, 7, 6, 7, 6, 6],
        [7, 7, 6, 7, 7, 7],
    ),
    "laptop2__5": (
        [7, 6, 6, 6, 6, 5],
        [6, 6, 6, 6, 7, 6],
        [6, 6, 6, 6, 6, 6],
        [6, 6, 7, 7, 6, 5],
        [6, 6, 6, 6, 7, 7],
    ),
    "laptop2__6": (
        [12, 10, 11, 11, 12],
        [11, 10, 11, 11, 11],
        [11, 11, 11, 11, 11],
        [11, 10, 10, 10, 11],
        [11, 11, 11, 10, 11],
    ),
    "laptop2__15": (
        [7, 8, 8, 8, 8],
        [8, 8, 8, 8, 7],
        [8, 11, 8, 8, 8],
        [7, 8, 7, 6, 8],
        [7, 7, 7, 6, 7],
    ),
}


def _groups(values: np.ndarray) -> list[list[int]]:
    groups: list[list[int]] = []
    for raw in values:
        value = int(raw)
        if not groups or value > groups[-1][-1] + 1:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def _central_five(values: list) -> list:
    if len(values) <= 5:
        return values
    start = (len(values) - 5) // 2
    return values[start : start + 5]


def extract_tick_pairs(reference_path: Path) -> list[list[dict[str, float]]]:
    image = np.asarray(Image.open(reference_path).convert("RGB"))
    cyan = (
        (image[:, :, 0] < 30)
        & (image[:, :, 1] > 225)
        & (image[:, :, 2] > 225)
    )
    green = (
        (image[:, :, 0] < 30)
        & (image[:, :, 1] > 225)
        & (image[:, :, 2] < 30)
    )
    row_counts = cyan.sum(axis=1)
    guide_rows = np.where(row_counts > max(40, image.shape[1] * 0.15))[0]
    row_groups = _groups(guide_rows)
    y_positions = [round((group[0] + group[-1]) / 2) for group in row_groups]
    if len(y_positions) != 5:
        raise ValueError(
            f"expected five guide rows, detected {len(y_positions)} in {reference_path}"
        )

    use_green = int(green.sum()) > 100
    marker = green if use_green else cyan
    maximum_width = 10 if use_green else 15
    all_pairs: list[list[dict[str, float]]] = []
    for y_position in y_positions:
        low = max(0, y_position - 15)
        high = min(marker.shape[0], y_position + 16)
        band = marker[low:high].copy()
        center = y_position - low
        band[max(0, center - 1) : min(band.shape[0], center + 2)] = False
        column_counts = band.sum(axis=0)
        # Three vertical pixels are sufficient because a label may partially
        # cover one tick; adjacent anti-aliased pixels are merged below.
        x_groups = _groups(np.where(column_counts >= 3)[0])
        centers = [float(sum(group) / len(group)) for group in x_groups]
        pairs = []
        index = 0
        while index + 1 < len(centers):
            left, right = centers[index], centers[index + 1]
            width = right - left
            if 1 <= width <= maximum_width:
                native_width = float(np.floor(width + 0.5))
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "center": (left + right) / 2.0,
                        "width": native_width,
                    }
                )
                index += 2
            else:
                index += 1
        if len(pairs) < 5:
            raise ValueError(
                f"expected at least five tick pairs at y={y_position}, "
                f"detected {len(pairs)} in {reference_path}"
            )
        all_pairs.append(pairs)
    return all_pairs


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract(manifest_path: Path, output_path: Path, audit_path: Path) -> None:
    ground_truth: list[dict] = []
    audit: list[dict] = []
    accuracy_items = [
        row for row in read_manifest(manifest_path) if row["evaluation_set"] == "accuracy"
    ]
    for item in accuracy_items:
        image_id = item["image_id"]
        reference_path = Path(item["reference_image_path"])
        if image_id in MANUAL_OVERRIDES:
            row_values = MANUAL_OVERRIDES[image_id]
            row_pairs = None
            method = "manual_transcription"
        else:
            row_pairs = extract_tick_pairs(reference_path)
            row_values = [[pair["width"] for pair in pairs] for pairs in row_pairs]
            method = "tick_distance_native_scale"

        for row_index, y_fraction in enumerate(Y_FRACTIONS):
            raw_values = list(row_values[row_index])
            selected_values = _central_five(raw_values)
            selected_pairs = (
                _central_five(row_pairs[row_index]) if row_pairs is not None else None
            )
            note = ""
            if len(selected_values) < 5:
                note = "reference contains fewer than five manual measurements"
            audit.append(
                {
                    "image_id": image_id,
                    "y_fraction": f"{y_fraction:.2f}",
                    "source_method": method,
                    "detected_count": len(raw_values),
                    "selected_count": len(selected_values),
                    "selected_widths_px": "|".join(f"{float(value):g}" for value in selected_values),
                    "reference_image_path": str(reference_path),
                    "note": note,
                }
            )
            for target_number, width in enumerate(selected_values, start=1):
                pair = selected_pairs[target_number - 1] if selected_pairs is not None else None
                ground_truth.append(
                    {
                        **item,
                        "measurement_type": "sidewall",
                        "y_fraction": f"{y_fraction:.2f}",
                        "target_id": f"S{target_number}",
                        "answer_width_px": f"{float(width):g}",
                        "answer_width_um": "",
                        "center_x_px": "" if pair is None else f"{pair['center']:.3f}",
                        # Screenshot X coordinates can include a small crop, so
                        # retain the center only for tolerant target matching.
                        "left_x_px": "",
                        "right_x_px": "",
                        "annotator": method,
                    }
                )

    write_csv(output_path, ground_truth, GT_FIELDS)
    write_csv(audit_path, audit, AUDIT_FIELDS)
    incomplete = [row for row in audit if int(row["selected_count"]) < 5]
    print(f"Ground truth: {output_path} ({len(ground_truth)} measurements)")
    print(f"Audit: {audit_path} ({len(audit)} image/Y groups)")
    print(f"Incomplete image/Y groups: {len(incomplete)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract(args.manifest, args.output, args.audit)
