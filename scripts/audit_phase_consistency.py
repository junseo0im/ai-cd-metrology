"""Audit image-local E1 offsets on the exact existing 43 pilot images.

Coordinates below are ROI-relative x pixels, never physical finger IDs.
No tolerance or production behavior is changed by the audit.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from audit_long_pairing_ratio import analyze_pilots
from run_gradient_canny_pilot import PROJECT_ROOT
from run_local_band_validation import _write_csv

KNOWN_SHIFT_BANDS = {
    ('w1_13_rb_p.png', 3), ('w1_13_rb_p.png', 4),
    ('w1_17_c_p.png', 4), ('w1_18_c_p.png', 4),
}
EXPECTED_IMAGES = {'normal': 24, 'narrowing': 7, 'particle': 12}


def phase_rows(diagnostic, *, group: str, filename: str, position: str) -> list[dict]:
    first_values = [
        band.accepted_candidates[0].e1_x if band.accepted_candidates else None
        for band in diagnostic.local_bands
    ]
    available = [value for value in first_values if value is not None]
    reference_e1 = float(np.median(available)) if available else None
    e1_values = [candidate.e1_x for candidate in diagnostic.accepted_candidates]
    reference_pitch = float(np.median(np.diff(e1_values))) if len(e1_values) >= 2 else None
    if reference_pitch is not None and (
        not np.isfinite(reference_pitch) or reference_pitch <= 0
    ):
        reference_pitch = None
    rows = []
    for band, first_e1 in zip(diagnostic.local_bands, first_values, strict=True):
        offset = abs(first_e1 - reference_e1) if (
            first_e1 is not None and reference_e1 is not None
        ) else None
        normalized = offset / reference_pitch if (
            offset is not None and reference_pitch is not None
        ) else None
        rows.append({
            'group': group, 'filename': filename, 'position': position,
            'band_index': band.band_index, 'first_e1_x': first_e1,
            'reference_e1_x': reference_e1, 'reference_pitch_px': reference_pitch,
            'phase_offset_px': offset, 'normalized_phase_offset': normalized,
            'current_status': band.status.value,
            'candidate_count': len(band.candidates),
            'accepted_candidate_count': len(band.accepted_candidates),
            'known_shift_band': (filename, band.band_index) in KNOWN_SHIFT_BANDS,
            'audit_available': normalized is not None,
            'unavailable_reason': (
                'no_accepted_local_candidate' if first_e1 is None else
                'reference_e1_unavailable' if reference_e1 is None else
                'positive_whole_roi_pitch_unavailable' if reference_pitch is None else None
            ),
            'coordinate_note': 'ROI-relative x px; indices are NOT physical finger IDs',
        })
    return rows


def audit() -> list[dict]:
    rows, image_counts = [], Counter()
    for group, path, position, _, _, _, diagnostic in analyze_pilots():
        image_counts[group] += 1
        rows.extend(phase_rows(diagnostic, group=group, filename=path.name, position=position))
    if dict(image_counts) != EXPECTED_IMAGES or len(rows) != 215:
        raise AssertionError(f'exact prior populations required: {dict(image_counts)}')
    if {(row['filename'], row['band_index']) for row in rows if row['known_shift_band']} != KNOWN_SHIFT_BANDS:
        raise AssertionError('all four known shift bands must be present')
    print('images:', dict(image_counts), 'bands:', len(rows))
    for label, selected in (
        ('normal', [r for r in rows if r['group'] == 'normal']),
        ('narrowing', [r for r in rows if r['group'] == 'narrowing']),
        ('particle_all', [r for r in rows if r['group'] == 'particle']),
        ('benign_w1_16_rb_p', [r for r in rows if r['filename'] == 'w1_16_rb_p.png']),
        ('long_w3_24_rb_p', [r for r in rows if r['filename'] == 'w3_24_rb_p.png']),
    ):
        values = [r['normalized_phase_offset'] for r in selected if r['audit_available']]
        print(label, 'available/unavailable/min/max:', len(values), len(selected) - len(values),
              min(values) if values else None, max(values) if values else None)
    for row in rows:
        if row['known_shift_band']:
            print('known:', {k: row[k] for k in (
                'filename', 'band_index', 'first_e1_x', 'reference_e1_x',
                'reference_pitch_px', 'phase_offset_px', 'normalized_phase_offset')})
    preserve = [r['normalized_phase_offset'] for r in rows
                if not r['known_shift_band'] and r['audit_available']]
    known = [r['normalized_phase_offset'] for r in rows if r['known_shift_band']]
    if any(value is None for value in known) or not preserve:
        print('SEPARATION UNAVAILABLE: do not implement a production guard')
    else:
        preserve_max, known_min = max(preserve), min(known)
        print('preserve_max_offset:', preserve_max)
        print('known_shift_min_offset:', known_min)
        print('separation_margin:', known_min - preserve_max)
        print('preserve_max_row:', max(
            (r for r in rows if not r['known_shift_band'] and r['audit_available']),
            key=lambda r: r['normalized_phase_offset']))
    return rows


def main() -> None:
    rows = audit()
    output = PROJECT_ROOT / 'data/derived/phase_consistency_audit'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'phase_consistency_audit.csv'
    _write_csv(path, rows)
    print('csv=' + str(path))


if __name__ == '__main__':
    main()
