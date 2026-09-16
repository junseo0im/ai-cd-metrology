"""Collect empirical Inner/Gap separability evidence on the existing 43 images.

Production metrology is called unchanged. The review bins are selection aids, not
thresholds. Manual physical labels are explicit review results and are not
inferred from dataset labels, parity scores, or separation ratios.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from audit_long_pairing_ratio import analyze_pilots
from audit_pairing_start_shift import _rows
from audit_phase_parity import parity_rows
from review_phase_parity_cases import _review_png
from run_gradient_canny_pilot import PROJECT_ROOT, _write_png
from run_local_band_validation import _write_csv

OUTPUT = PROJECT_ROOT / "data/derived/phase_separability_evidence"

# Direct physical grouping review. Notes identify mixed-band cases explicitly.
REVIEWS = {
    "w2_26_lt.png": (
        "PHYSICALLY_STABLE",
        "All five bands retain semantic grouping; value tie exists in low-separation candidates.",
    ),
    "w2_25_rb.png": (
        "PHYSICALLY_STABLE",
        "All five bands retain the same repeating E1-E6 grouping.",
    ),
    "w1_26_rb_p.png": (
        "PHYSICALLY_STABLE",
        "Grouping is retained across bands despite particle-related rejection/candidate loss.",
    ),
    "w3_36_c_p.png": (
        "PHYSICALLY_STABLE",
        "Grouping is retained across bands; one weak positive parity candidate is value-based.",
    ),
    "w2_38_rb.png": (
        "PHASE_SHIFT",
        "Bands 2/4 are stable; band 3 gains a left doublet and shifts the physical E slots.",
    ),
    "w1_16_rb_p.png": (
        "PHYSICALLY_STABLE",
        "All five bands retain semantic grouping despite boundary partials and particles.",
    ),
    "w2_38_lt.png": (
        "PHYSICALLY_STABLE",
        "All five bands retain semantic grouping despite internal peak rejection.",
    ),
    "w1_13_rb_p.png": (
        "PHASE_SHIFT",
        "Bands 3/4 have a verified physical E-slot phase shift.",
    ),
    "w1_18_c_p.png": (
        "PHASE_SHIFT",
        "Band 4 has a verified physical E-slot phase shift.",
    ),
}
PLOT_NAMES = set(REVIEWS) - {"w2_26_lt.png"}


def review_bin(ratio: float) -> str:
    if ratio < 0.05:
        return "<0.05"
    if ratio < 0.10:
        return "0.05-<0.10"
    if ratio < 0.15:
        return "0.10-<0.15"
    if ratio < 0.20:
        return "0.15-<0.20"
    return ">=0.20"


def _quartiles(values: np.ndarray) -> tuple[float, float]:
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return float(q1), float(q3)


def image_row(group, path, position, diagnostic) -> tuple[dict, list[dict]]:
    candidates = diagnostic.accepted_candidates
    if not candidates:
        raise AssertionError(f"Whole-ROI accepted candidates required: {path.name}")
    inner = np.asarray([c.inner_width_px for c in candidates], dtype=float)
    gap = np.asarray([c.gap_px for c in candidates], dtype=float)
    delta = inner - gap
    reference_inner = float(np.median(inner))
    reference_gap = float(np.median(gap))
    reference_delta = float(np.median(delta))
    separation = abs(reference_inner - reference_gap)
    mean_scale = (reference_inner + reference_gap) / 2
    if mean_scale <= 0 or separation <= 0:
        raise AssertionError(f"Positive reference scale/separation required: {path.name}")
    nonzero = delta[delta != 0]
    sign_consistency = (
        float(np.mean(np.sign(nonzero) == np.sign(reference_delta)))
        if len(nonzero) and reference_delta != 0 else None
    )
    inner_q1, inner_q3 = _quartiles(inner)
    gap_q1, gap_q3 = _quartiles(gap)
    overlap = not (inner_q3 < gap_q1 or gap_q3 < inner_q1)
    scores = parity_rows(
        diagnostic, group=group, filename=path.name, position=position
    )
    score_values = [r["normalized_parity_score"] for r in scores]
    if any(value is None for value in score_values):
        raise AssertionError(f"Parity score required for this population: {path.name}")
    physical, note = REVIEWS.get(path.name, (None, None))
    ratio = separation / mean_scale
    return {
        "dataset_group": group,
        "filename": path.name,
        "position": position,
        "whole_candidate_count": len(candidates),
        "reference_inner_px": reference_inner,
        "reference_gap_px": reference_gap,
        "separation_px": separation,
        "separation_ratio": ratio,
        "review_bin": review_bin(ratio),
        "delta_median_px": reference_delta,
        "delta_min_px": float(delta.min()),
        "delta_max_px": float(delta.max()),
        "delta_sign_consistency": sign_consistency,
        "tie_count": int(np.sum(delta == 0)),
        "inner_median_px": reference_inner,
        "inner_q1_px": inner_q1,
        "inner_q3_px": inner_q3,
        "inner_iqr_px": inner_q3 - inner_q1,
        "gap_median_px": reference_gap,
        "gap_q1_px": gap_q1,
        "gap_q3_px": gap_q3,
        "gap_iqr_px": gap_q3 - gap_q1,
        "iqr_overlap": overlap,
        "physical_review_result": physical,
        "physical_review_note": note,
        "local_candidate_count": len(scores),
        "parity_score_min": min(score_values),
        "parity_score_max": max(score_values),
        "parity_positive_count": sum(value > 0 for value in score_values),
        "parity_zero_count": sum(value == 0 for value in score_values),
        "selection_note": (
            "manual grouping review selected across low/transition/high separation"
            if physical else None
        ),
        "units_note": "pixel fields in px; ratio/consistency/score dimensionless",
    }, scores


def main() -> None:
    rows, previews = [], []
    image_counts = Counter()
    for group, path, position, algorithm, image, record, diagnostic in analyze_pilots():
        image_counts[group] += 1
        row, scores = image_row(group, path, position, diagnostic)
        rows.append(row)
        if path.name in PLOT_NAMES:
            whole = algorithm.measure(image, record)[0]
            band_rows = _rows(path.name, diagnostic, whole)
            previews.append((path.name, diagnostic, band_rows, scores))
    rows.sort(key=lambda row: (row["separation_ratio"], row["filename"]))
    if dict(image_counts) != {"normal": 24, "narrowing": 7, "particle": 12}:
        raise AssertionError(f"Exact prior population required: {dict(image_counts)}")
    if len(rows) != 43 or len(previews) != 8:
        raise AssertionError("Expected 43 summaries and exactly eight review diagnostics")
    if {r["filename"] for r in rows if r["physical_review_result"]} != set(REVIEWS):
        raise AssertionError("Every selected physical review must appear exactly once")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "phase_separability_image_summary.csv"
    _write_csv(csv_path, rows)
    for filename, diagnostic, band_rows, scores in previews:
        output_path = OUTPUT / f"{Path(filename).stem}_separability_review.png"
        _write_png(
            output_path,
            _review_png(filename, diagnostic, band_rows, scores),
        )

    print("images=" + str(dict(image_counts)))
    print("bins=" + str(dict(Counter(r["review_bin"] for r in rows))))
    selected = [r for r in rows if r["physical_review_result"]]
    print("selected_bins=" + str(dict(Counter(r["review_bin"] for r in selected))))
    print("physical_labels=" + str(dict(Counter(r["physical_review_result"] for r in selected))))
    print("ties_by_bin=" + str({
        label: sum(r["tie_count"] for r in rows if r["review_bin"] == label)
        for label in ("<0.05", "0.05-<0.10", "0.10-<0.15", "0.15-<0.20", ">=0.20")
    }))
    print("iqr_overlap_by_bin=" + str({
        label: sum(r["iqr_overlap"] for r in rows if r["review_bin"] == label)
        for label in ("<0.05", "0.05-<0.10", "0.10-<0.15", "0.15-<0.20", ">=0.20")
    }))
    print("sign_consistency_range_by_bin=" + str({
        label: (
            min(r["delta_sign_consistency"] for r in rows if r["review_bin"] == label),
            max(r["delta_sign_consistency"] for r in rows if r["review_bin"] == label),
        )
        for label in ("<0.05", "0.05-<0.10", "0.10-<0.15", "0.15-<0.20", ">=0.20")
    }))
    for row in selected:
        print("review=" + str({
            key: row[key] for key in (
                "filename", "review_bin", "separation_ratio", "tie_count",
                "delta_sign_consistency", "iqr_overlap", "physical_review_result",
                "parity_score_min", "parity_score_max", "parity_positive_count",
            )
        }))
    print("csv=" + str(csv_path))
    print("diagnostic_count=" + str(len(previews)))


if __name__ == "__main__":
    main()
