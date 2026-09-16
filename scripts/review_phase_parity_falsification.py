"""Final five-image falsification review for the existing parity signal.

Uses production analysis and existing visualization helpers unchanged. The exact
five cases are selected only because their saved whole-reference separation
ratio is in [0.10, 0.15) and no prior physical review label exists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from audit_long_pairing_ratio import analyze_pilots
from audit_pairing_start_shift import _rows
from audit_phase_parity import parity_rows
from review_phase_parity_cases import _review_png
from run_gradient_canny_pilot import PROJECT_ROOT, _write_png

SUMMARY = (
    PROJECT_ROOT
    / "data/derived/phase_separability_evidence/phase_separability_image_summary.csv"
)
OUTPUT = PROJECT_ROOT / "data/derived/phase_parity_falsification"
LOW, HIGH = 0.10, 0.15


def target_names() -> dict[str, float]:
    with SUMMARY.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row["filename"]: float(row["separation_ratio"])
        for row in rows
        if LOW <= float(row["separation_ratio"]) < HIGH
        and not row["physical_review_result"]
    }
    expected = {
        "w2_25_c.png",
        "w3_24_rb_p.png",
        "w2_11_c.png",
        "w3_11_lt_p.png",
        "w3_12_rb_p.png",
    }
    if set(selected) != expected:
        raise AssertionError(f"Expected exact five unreviewed transition cases: {selected}")
    return selected


def main() -> None:
    selected = target_names()
    previews = []
    for group, path, position, algorithm, image, record, diagnostic in analyze_pilots():
        if path.name not in selected:
            continue
        whole = algorithm.measure(image, record)[0]
        bands = _rows(path.name, diagnostic, whole)
        scores = parity_rows(
            diagnostic, group=group, filename=path.name, position=position
        )
        previews.append((path.name, diagnostic, bands, scores))
        first = scores[0]
        if abs(first["reference_separation_ratio"] - selected[path.name]) > 1e-12:
            raise AssertionError(f"Saved/recomputed ratio mismatch: {path.name}")
        print(
            "image="
            + json.dumps(
                {
                    "filename": path.name,
                    "separation_ratio": selected[path.name],
                    "reference_inner_px": first["reference_inner_px"],
                    "reference_gap_px": first["reference_gap_px"],
                }
            )
        )
        for band in diagnostic.local_bands:
            band_scores = [
                row["normalized_parity_score"]
                for row in scores
                if row["band_index"] == band.band_index
            ]
            print(
                "band="
                + json.dumps(
                    {
                        "filename": path.name,
                        "band_index": band.band_index,
                        "doublets": [
                            [d.negative_peak_x, d.positive_peak_x]
                            for d in band.pairing.doublets
                        ],
                        "accepted_e1_e6": [
                            [
                                candidate.e1_x,
                                candidate.e2_x,
                                candidate.e3_x,
                                candidate.e4_x,
                                candidate.e5_x,
                                candidate.e6_x,
                            ]
                            for candidate in band.accepted_candidates
                        ],
                        "inner_gap": [
                            [candidate.inner_width_px, candidate.gap_px]
                            for candidate in band.accepted_candidates
                        ],
                        "parity_score_min": min(band_scores),
                        "parity_score_max": max(band_scores),
                        "parity_positive_count": sum(score > 0 for score in band_scores),
                        "parity_zero_count": sum(score == 0 for score in band_scores),
                    }
                )
            )

    previews.sort(key=lambda item: item[0])
    if len(previews) != 5:
        raise AssertionError(f"Expected exactly five diagnostics, got {len(previews)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, diagnostic, bands, scores in previews:
        output_path = OUTPUT / f"{Path(filename).stem}_parity_falsification.png"
        _write_png(output_path, _review_png(filename, diagnostic, bands, scores))
        print("diagnostic=" + str(output_path))


if __name__ == "__main__":
    main()
