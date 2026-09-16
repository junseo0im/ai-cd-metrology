"""Re-audit saved parity candidates using explicitly reviewed physical groups.

This is diagnostic only: source candidates, status and references are unchanged.
Dataset labels are not physical ground truth. No confidence threshold is selected.
No images are processed; the existing long-guard accepted-candidate CSV is reused.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from math import isclose, isfinite
from pathlib import Path

from run_local_band_validation import _write_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "data/derived/phase_parity_audit/phase_parity_candidate_audit.csv"
OUTPUT = PROJECT_ROOT / "data/derived/phase_parity_verified_audit/phase_parity_verified_audit.csv"

# User-supplied manual physical review, not inferred from scores or folder labels.
SHIFT_BANDS = {
    ("w1_13_rb_p.png", 3), ("w1_13_rb_p.png", 4),
    ("w1_17_c_p.png", 4), ("w1_18_c_p.png", 4),
    ("w2_38_rb.png", 3),
}
STABLE_BANDS = {
    (filename, band)
    for filename in ("w1_16_rb_p.png", "w2_38_lt.png", "w2_26_lt.png")
    for band in range(5)
} | {("w2_38_rb.png", 2), ("w2_38_rb.png", 4)}


def reviewed_rows() -> list[dict]:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 1464:
        raise AssertionError("Expected the existing 1464 accepted candidates")
    identities = {
        (r["filename"], int(r["band_index"]), int(r["candidate_sequence_index"]))
        for r in source_rows
    }
    bands = {(name, band) for name, band, _ in identities}
    if len(identities) != len(source_rows) or len(bands) != 215:
        raise AssertionError("Duplicate candidate identity or changed band population")
    image_groups = {(r["dataset_group"], r["filename"]) for r in source_rows}
    if Counter(group for group, _ in image_groups) != {"normal": 24, "narrowing": 7, "particle": 12}:
        raise AssertionError("Expected the prior exact 43-image population")
    if not SHIFT_BANDS.isdisjoint(STABLE_BANDS) or not (SHIFT_BANDS | STABLE_BANDS) <= bands:
        raise AssertionError("Conflicting or missing reviewed bands")

    rows = []
    for source in source_rows:
        identity = (source["filename"], int(source["band_index"]))
        group = (
            "VERIFIED_STABLE" if identity in STABLE_BANDS else
            "VERIFIED_PHASE_SHIFT" if identity in SHIFT_BANDS else "UNKNOWN"
        )
        inner, gap, ri, rg = [
            float(source[field]) for field in
            ("inner_px", "gap_px", "reference_inner_px", "reference_gap_px")
        ]
        if not all(isfinite(v) for v in (inner, gap, ri, rg)):
            raise AssertionError("Non-finite source measurement/reference")
        separation = abs(ri - rg)
        mean = (ri + rg) / 2
        if separation <= 0 or mean <= 0 or source["audit_available"] != "True":
            raise AssertionError("This saved audit population must have available scores")
        normal = abs(inner - ri) + abs(gap - rg)
        swapped = abs(inner - rg) + abs(gap - ri)
        values = {
            "reference_separation_px": separation,
            "reference_separation_ratio": separation / mean,
            "normal_error": normal,
            "swapped_error": swapped,
            "normalized_parity_score": (normal - swapped) / (2 * separation),
        }
        # Numerical consistency check only; this tolerance is NOT a decision gate.
        for field, value in values.items():
            if not isclose(value, float(source[field]), rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError(f"Saved parity formula mismatch: {identity}, {field}")
        rows.append({
            "filename": source["filename"],
            "band_index": identity[1],
            "candidate_sequence_index": int(source["candidate_sequence_index"]),
            "physical_review_group": group,
            "inner_px": inner, "gap_px": gap,
            "reference_inner_px": ri, "reference_gap_px": rg,
            **values,
            "dataset_group": source["dataset_group"],
            "current_status": source["current_status"],
            "reference_validation_status": source["reference_validation_status"],
            "error_units": source["error_units"],
            "index_note": source["index_note"],
        })
    return rows


def summary(rows: list[dict]) -> dict:
    scores = [r["normalized_parity_score"] for r in rows]
    ratios = [r["reference_separation_ratio"] for r in rows]
    nonzero = [abs(score) for score in scores if score != 0]
    return {
        "candidate_count": len(rows),
        "ratio_range": [min(ratios), max(ratios)],
        "score_range": [min(scores), max(scores)],
        "positive_count": sum(score > 0 for score in scores),
        "exact_zero_count": sum(score == 0 for score in scores),
        # No arbitrary near-zero cutoff is introduced.
        "minimum_nonzero_absolute_score": min(nonzero) if nonzero else None,
    }


def main() -> None:
    rows = reviewed_rows()
    groups = {
        label: [r for r in rows if r["physical_review_group"] == label]
        for label in ("VERIFIED_STABLE", "VERIFIED_PHASE_SHIFT", "UNKNOWN")
    }
    print("groups=" + json.dumps({label: summary(selected) for label, selected in groups.items()}))
    for label, selected in groups.items():
        for ratio in sorted({r["reference_separation_ratio"] for r in selected}):
            subset = [r for r in selected if r["reference_separation_ratio"] == ratio]
            print("ratio_stratum=" + json.dumps({"group": label, "ratio": ratio, **summary(subset)}))
    stable, shifted = groups["VERIFIED_STABLE"], groups["VERIFIED_PHASE_SHIFT"]
    stable_max = max(r["normalized_parity_score"] for r in stable)
    shift_min = min(r["normalized_parity_score"] for r in shifted)
    # Explicitly reviewed low-separability case, NOT a numeric confidence gate.
    other_stable = [r for r in stable if r["filename"] != "w2_26_lt.png"]
    other_max = max(r["normalized_parity_score"] for r in other_stable)
    print("separation=" + json.dumps({
        "stable_max_score": stable_max, "shift_min_score": shift_min,
        "all_reviewed_score_margin": shift_min - stable_max,
        "other_reviewed_stable_count": len(other_stable),
        "other_reviewed_stable_max_score": other_max,
        "other_reviewed_score_margin": shift_min - other_max,
        "numeric_confidence_gate_selected": False,
    }))
    for filename, indices in (("w2_26_lt.png", range(5)), ("w2_38_rb.png", (2, 3, 4))):
        for band in indices:
            subset = [r for r in rows if r["filename"] == filename and r["band_index"] == band]
            print("case_band=" + json.dumps({"filename": filename, "band": band, **summary(subset)}))
    print("exact_ties=" + json.dumps([r for r in stable if r["normalized_parity_score"] == 0]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT, rows)
    print("csv=" + str(OUTPUT))


if __name__ == "__main__":
    main()
