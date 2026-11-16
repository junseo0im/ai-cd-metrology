"""Extract line/gap ground truth from green marks drawn on ``*_y_guide.png``.

Annotation rule (see 측정_안내.txt in the ground-truth folder):
  * on each cyan measurement row, draw a short pure-green vertical mark on both
    borders of every red sidewall band, exactly where red meets the brighter
    line/gap colour;
  * save as ``<stem>_i.png``.  A screen capture of the zoomed guide is fine; a
    guide too wide for the screen may be captured in pieces (``<stem>_i.png``,
    ``<stem>_i_2.png``, ...), each showing the magenta centre line.

Screen captures change scale and add borders, so the capture is registered back
to native pixels from the guide's own drawings: the three cyan rows (known
native y) give the vertical mapping, and the magenta centre line plus the two
white side-margin lines (known native x) give the horizontal mapping.  A mark's
centre is taken as the band boundary itself.

Consecutive marks then alternate between a narrow sidewall interval and a wide
line/gap interval.  Wide intervals are labelled line (brighter) or gap (darker)
from the untouched ``<stem>_native.png`` profile, the side-margin intervals are
dropped, and the five of each type closest to the image centre become
L1-L5 / G1-G5 from left to right, the same central-target rule as the guide.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from team_protocol import TARGET_PREFIX, TARGETS_PER_ROW  # noqa: E402

GT_FIELDS = (
    "image_id", "measurement_type", "y_fraction", "target_id",
    "answer_width_px", "answer_width_um", "center_x_px", "left_x_px", "right_x_px",
    "image_path", "reference_image_path", "laptop_id", "evaluation_set", "pair_group",
    "repeat_group", "repeat_id", "brightness_group", "brightness_level", "condition",
    "annotator",
)
AUDIT_FIELDS = (
    "image_id", "y_fraction", "y_px_native", "scale_x", "scale_y", "mark_count",
    "line_count", "gap_count", "note", "reference_image_path",
)
SIDE_MARGIN_RATIO = 0.03  # same exclusion band as make_manual_y_guide.py
MARK_HALF_HEIGHT = 6      # native rows above/below the guide row searched for marks
NARROW_RATIO = 0.35       # intervals below this fraction of the wide ones are sidewalls
DUPLICATE_MARK_PX = 1.0   # marks closer than this (native px) across pieces are one mark


def color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    r, g, b = (image[:, :, i].astype(int) for i in range(3))
    return {
        "green": (g > 150) & (g - r > 80) & (g - b > 80),
        "cyan": (g > 170) & (b > 170) & (r < 110),
        "magenta": (r > 170) & (b > 170) & (g < 110),
        "white": (r > 225) & (g > 225) & (b > 225),
    }


def _line_centres(counts: np.ndarray, minimum: float) -> list[float]:
    """Centres of runs of rows/columns whose colour count exceeds ``minimum``."""
    hits = np.flatnonzero(counts >= minimum)
    if hits.size == 0:
        return []
    groups = np.split(hits, np.flatnonzero(np.diff(hits) > 1) + 1)
    return [float(np.average(group, weights=counts[group])) for group in groups]


def register(masks: dict[str, np.ndarray], native_shape, rows_native: list[int]):
    """Return (scale_x, offset_x, scale_y, offset_y) mapping capture -> native px."""
    height, width = masks["cyan"].shape
    native_h, native_w = native_shape[:2]
    if (height, width) == (native_h, native_w):
        return 1.0, 0.0, 1.0, 0.0

    # Thresholds are relative to the strongest guide line, because a capture may
    # include window chrome and the image can fill only part of it.
    cyan_counts = masks["cyan"].sum(axis=1)
    cyan_rows = _line_centres(cyan_counts, max(20, 0.5 * cyan_counts.max()))
    if len(cyan_rows) != len(rows_native):
        raise ValueError(f"found {len(cyan_rows)} cyan rows, expected {len(rows_native)}")
    scale_y, offset_y = np.polyfit(cyan_rows, rows_native, 1)

    margin = max(3, int(round(native_w * SIDE_MARGIN_RATIO)))
    magenta_counts = masks["magenta"].sum(axis=0)
    centre = _line_centres(magenta_counts, max(20, 0.5 * magenta_counts.max()))
    if len(centre) != 1:
        raise ValueError(f"found {len(centre)} magenta centre lines, expected 1")
    capture_x, native_x = [centre[0]], [native_w // 2]
    white_counts = masks["white"].sum(axis=0)
    white = _line_centres(white_counts, max(20, 0.3 * native_h / scale_y))
    tolerance = 0.3 * margin / scale_y  # stay clear of white viewer background beside the image
    for target in (margin, native_w - 1 - margin):
        expected = centre[0] + (target - native_w // 2) / scale_y
        near = [x for x in white if abs(x - expected) < tolerance]
        if near:
            capture_x.append(min(near, key=lambda x: abs(x - expected)))
            native_x.append(target)
    if len(capture_x) >= 2:
        scale_x, offset_x = np.polyfit(capture_x, native_x, 1)
    else:  # only the centre line is visible: assume square pixels
        scale_x, offset_x = scale_y, native_w // 2 - scale_y * centre[0]
    return float(scale_x), float(offset_x), float(scale_y), float(offset_y)


def mark_positions(green: np.ndarray, y_capture: float, half_height: float) -> list[float]:
    """Capture-space x centre of every green mark crossing row ``y_capture``."""
    y0 = max(0, int(round(y_capture - half_height)))
    band = green[y0: int(round(y_capture + half_height)) + 1]
    counts = band.sum(axis=0)
    return _line_centres(counts, max(2, 0.3 * band.shape[0]))


def merge_marks(marks: list[float]) -> list[float]:
    """Sort marks and collapse ones seen twice in overlapping capture pieces."""
    merged: list[list[float]] = []
    for x in sorted(marks):
        if merged and x - merged[-1][-1] < DUPLICATE_MARK_PX:
            merged[-1].append(x)
        else:
            merged.append([x])
    return [float(np.mean(group)) for group in merged]


def classify_intervals(marks: list[float], profile: np.ndarray) -> list[dict]:
    """Turn sorted marks (native x) into wide intervals labelled line/gap by brightness."""
    widths = np.diff(marks)
    if widths.size == 0:
        return []
    # About half the intervals are sidewalls, so the 75th percentile is always a
    # line/gap width (the median is not when every sidewall has the same width).
    wide_reference = float(np.percentile(widths, 75))
    bands = []
    for left, right, width in zip(marks, marks[1:], widths):
        if width < NARROW_RATIO * wide_reference:
            continue  # sidewall between its two borders
        a, b = int(np.ceil(left)), int(np.floor(right)) + 1
        inner = max(1, int((b - a) * 0.2))
        brightness = float(np.mean(profile[a + inner: max(a + inner + 1, b - inner)]))
        bands.append({"left": left, "right": right, "brightness": brightness})
    if len(bands) < 2:
        return []
    values = np.array([band["brightness"] for band in bands])
    split = (np.percentile(values, 25) + np.percentile(values, 75)) / 2.0
    for band in bands:
        band["kind"] = "line" if band["brightness"] >= split else "gap"
    return bands


def central_targets(bands: list[dict], kind: str, width: int) -> list[dict]:
    margin = max(3, int(round(width * SIDE_MARGIN_RATIO)))
    candidates = [
        band for band in bands
        if band["kind"] == kind and band["left"] > margin and band["right"] < width - 1 - margin
    ]
    centre = width / 2.0
    nearest = sorted(candidates, key=lambda band: abs((band["left"] + band["right"]) / 2 - centre))
    return sorted(nearest[:TARGETS_PER_ROW], key=lambda band: band["left"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract(folder: Path, annotator: str, suffix: str) -> tuple[Path, Path]:
    manifest = [row for row in read_csv(folder / "dataset_manifest.csv")
                if row["evaluation_set"] == "accuracy"]
    ground_truth, audit = [], []
    for item in manifest:
        laptop, stem = item["image_id"].split("__", 1)
        guide_dir = folder / "guides" / laptop
        reference = guide_dir / f"{stem}{suffix}.png"
        pieces = [reference, *sorted(guide_dir.glob(f"{stem}{suffix}_[0-9]*.png"))]
        base = {"image_id": item["image_id"], "reference_image_path": str(reference)}
        if not reference.is_file():
            audit.append({**base, "note": "annotated image missing"})
            continue
        native = np.asarray(Image.open(guide_dir / f"{stem}_native.png").convert("RGB"))
        rows = read_csv(guide_dir / f"{stem}_y_rows.csv")
        rows_native = [int(r["y_px_native"]) for r in rows]
        aligned = []  # (green mask, sx, ox, sy, oy) per capture piece
        for piece in pieces:
            masks = color_masks(np.asarray(Image.open(piece).convert("RGB")))
            try:
                aligned.append((masks["green"], *register(masks, native.shape, rows_native)))
            except ValueError as error:
                audit.append({**base, "reference_image_path": str(piece),
                              "note": f"cannot align capture: {error}"})
        if not aligned:
            continue
        sx, sy = aligned[0][1], aligned[0][3]
        gray = native.mean(axis=2)
        for row in rows:
            y = int(row["y_px_native"])
            marks = merge_marks([
                psx * x + pox
                for green, psx, pox, psy, poy in aligned
                for x in mark_positions(green, (y - poy) / psy, MARK_HALF_HEIGHT / psy)
            ])
            profile = gray[max(0, y - 2): y + 3].mean(axis=0)
            bands = classify_intervals(marks, profile)
            counts = {}
            for kind in ("line", "gap"):
                selected = central_targets(bands, kind, native.shape[1])
                counts[kind] = len(selected)
                for number, band in enumerate(selected, start=1):
                    ground_truth.append({
                        **item,
                        "measurement_type": kind,
                        "y_fraction": row["y_fraction"],
                        "target_id": f"{TARGET_PREFIX[kind]}{number}",
                        "left_x_px": f"{band['left']:.2f}",
                        "right_x_px": f"{band['right']:.2f}",
                        "reference_image_path": str(reference),
                        "annotator": annotator,
                    })
            note = "" if min(counts.values()) >= TARGETS_PER_ROW else "fewer than five targets"
            audit.append({**base, "y_fraction": row["y_fraction"], "y_px_native": y,
                          "scale_x": f"{sx:.4f}", "scale_y": f"{sy:.4f}", "mark_count": len(marks),
                          "line_count": counts["line"], "gap_count": counts["gap"], "note": note})
    name = f"ground_truth_line_gap{'_' + annotator if annotator != 'manual' else ''}_filled.csv"
    output = folder / name
    audit_path = folder / name.replace("_filled.csv", "_audit.csv")
    write_csv(output, ground_truth, GT_FIELDS)
    write_csv(audit_path, audit, AUDIT_FIELDS)
    return output, audit_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="ground-truth folder made by evaluate.py suite prepare")
    parser.add_argument("--annotator", default="manual", help="name stored in the CSV (second annotator: e.g. B)")
    parser.add_argument("--suffix", default="_i", help="annotated file suffix, e.g. _i or _iB")
    args = parser.parse_args()
    output, audit = extract(args.folder, args.annotator, args.suffix)
    print(f"ground truth: {output}")
    print(f"audit:        {audit}")


if __name__ == "__main__":
    main()
