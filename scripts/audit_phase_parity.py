"""Audit Inner/Gap parity errors without changing production metrology.

The exact prior 43-image population and current long-guard accepted candidates
are reused. Scores describe fit to an image-local reference, not ground truth.
No first-E1 offsets, thresholds, candidate correction or visualization.
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


def parity_rows(diagnostic, *, group: str, filename: str, position: str) -> list[dict]:
    whole = diagnostic.accepted_candidates
    reference_inner = float(np.median([c.inner_width_px for c in whole])) if whole else None
    reference_gap = float(np.median([c.gap_px for c in whole])) if whole else None
    reference_available = (
        reference_inner is not None and reference_gap is not None
        and np.isfinite(reference_inner) and np.isfinite(reference_gap)
    )
    separation = abs(reference_inner - reference_gap) if reference_available else None
    mean_scale = (reference_inner + reference_gap) / 2 if reference_available else None
    ratio = separation / mean_scale if (
        mean_scale is not None and np.isfinite(mean_scale) and mean_scale > 0
    ) else None
    rows = []
    for band in diagnostic.local_bands:
        identity = (filename, band.band_index)
        audit_group = (
            'known_preserve' if group in ('normal', 'narrowing') or filename == 'w1_16_rb_p.png' else
            'known_phase_shift' if identity in KNOWN_SHIFT_BANDS else
            'exploratory_long_pair_control' if identity == ('w3_24_rb_p.png', 1) else
            'particle_exploratory'
        )
        for candidate in band.accepted_candidates:
            inner, gap = candidate.inner_width_px, candidate.gap_px
            normal_error = (
                abs(inner - reference_inner) + abs(gap - reference_gap)
                if reference_available else None
            )
            swapped_error = (
                abs(inner - reference_gap) + abs(gap - reference_inner)
                if reference_available else None
            )
            advantage = normal_error - swapped_error if reference_available else None
            score = advantage / (2 * separation) if (
                separation is not None and separation > 0 and ratio is not None
            ) else None
            if score is not None and not -1.0000000001 <= score <= 1.0000000001:
                raise AssertionError('L1 normalized parity score must lie in [-1, 1]')
            rows.append({
                'dataset_group': group, 'filename': filename, 'position': position,
                'band_index': band.band_index,
                'candidate_sequence_index': candidate.sequence_index,
                'outer_px': candidate.outer_width_px, 'inner_px': inner, 'gap_px': gap,
                'reference_inner_px': reference_inner, 'reference_gap_px': reference_gap,
                'reference_separation_px': separation, 'reference_separation_ratio': ratio,
                'normal_error': normal_error, 'swapped_error': swapped_error,
                'swap_advantage': advantage, 'normalized_parity_score': score,
                'audit_group': audit_group, 'current_status': band.status.value,
                'raw_candidate_count': len(band.candidates),
                'accepted_candidate_count': len(band.accepted_candidates),
                'long_pairing_count': len(band.long_pairing_candidates),
                'reference_accepted_candidate_count': len(whole),
                'reference_validation_status': diagnostic.validation.status,
                'audit_available': score is not None,
                'unavailable_reason': (
                    'whole_reference_unavailable' if not reference_available else
                    'invalid_reference_mean_scale' if ratio is None else
                    'zero_reference_separation' if separation == 0 else None
                ),
                'error_units': 'px; normalized score/ratio dimensionless',
                'index_note': 'sequence index is diagnostic order, NOT physical finger ID',
            })
    return rows


def _range(rows: list[dict], field: str) -> tuple:
    values = [r[field] for r in rows if r[field] is not None]
    return (len(values), min(values) if values else None, max(values) if values else None)


def audit() -> list[dict]:
    rows, image_counts, references = [], Counter(), []
    band_count, empty_bands = 0, []
    for group, path, position, _, _, _, diagnostic in analyze_pilots():
        image_counts[group] += 1
        band_count += len(diagnostic.local_bands)
        image_rows = parity_rows(diagnostic, group=group, filename=path.name, position=position)
        rows.extend(image_rows)
        empty_bands.extend((path.name, b.band_index) for b in diagnostic.local_bands if not b.accepted_candidates)
        if image_rows:
            references.append(image_rows[0])
    if dict(image_counts) != EXPECTED_IMAGES or band_count != 215:
        raise AssertionError(f'exact prior populations required: {dict(image_counts)}, {band_count}')
    if {(r['filename'], r['band_index']) for r in rows if r['audit_group'] == 'known_phase_shift'} != KNOWN_SHIFT_BANDS:
        raise AssertionError('all four known shift bands must have accepted candidates')
    print('images:', dict(image_counts), 'bands:', band_count, 'accepted candidates:', len(rows))
    print('empty bands:', empty_bands, 'unavailable scores:', sum(r['normalized_parity_score'] is None for r in rows))
    for label, selected in (
        ('normal', [r for r in rows if r['dataset_group'] == 'normal']),
        ('narrowing', [r for r in rows if r['dataset_group'] == 'narrowing']),
        ('particle', [r for r in rows if r['dataset_group'] == 'particle']),
        ('w2_38_lt', [r for r in rows if r['filename'] == 'w2_38_lt.png']),
        ('w1_16_rb_p', [r for r in rows if r['filename'] == 'w1_16_rb_p.png']),
        ('known_shift', [r for r in rows if r['audit_group'] == 'known_phase_shift']),
    ):
        print(label, 'n/min/max:', _range(selected, 'normalized_parity_score'),
              'positive:', sum(r['normalized_parity_score'] is not None and r['normalized_parity_score'] > 0 for r in selected))
    for label in EXPECTED_IMAGES:
        selected = [r for r in references if r['dataset_group'] == label]
        print('reference', label, 'separation px n/min/p50/max:',
              len(selected), *np.quantile([r['reference_separation_px'] for r in selected], [0, .5, 1]).tolist())
        print('reference', label, 'separation ratio n/min/p50/max:',
              len(selected), *np.quantile([r['reference_separation_ratio'] for r in selected], [0, .5, 1]).tolist())
    print('reference min separation image:', min(references, key=lambda r: r['reference_separation_px']))
    for identity in sorted(KNOWN_SHIFT_BANDS | {('w3_24_rb_p.png', 1)}):
        selected = [r for r in rows if (r['filename'], r['band_index']) == identity]
        print('special', identity, 'sequence/inner/gap/normal_error/swapped_error/score:',
              [(r['candidate_sequence_index'], r['inner_px'], r['gap_px'],
                r['normal_error'], r['swapped_error'], r['normalized_parity_score']) for r in selected])
    for identity in sorted({(r['filename'], r['band_index']) for r in rows
                            if r['audit_group'] == 'known_preserve'
                            and r['normalized_parity_score'] is not None and r['normalized_parity_score'] > 0}):
        selected = [r for r in rows if (r['filename'], r['band_index']) == identity]
        print('preserve positive band', identity, 'n/min/max:', _range(selected, 'normalized_parity_score'),
              'reference:', selected[0]['reference_inner_px'], selected[0]['reference_gap_px'])
    preserve = [r for r in rows if r['audit_group'] == 'known_preserve']
    known = [r for r in rows if r['audit_group'] == 'known_phase_shift']
    if any(r['normalized_parity_score'] is None for r in preserve + known):
        print('SEPARATION UNAVAILABLE: no production guard')
    else:
        preserve_max = max(r['normalized_parity_score'] for r in preserve)
        known_min = min(r['normalized_parity_score'] for r in known)
        print('preserve_max_score:', preserve_max)
        print('known_shift_min_score:', known_min)
        print('separation_margin:', known_min - preserve_max)
        print('preserve max candidate:', max(preserve, key=lambda r: r['normalized_parity_score']))
        print('known min candidate:', min(known, key=lambda r: r['normalized_parity_score']))
    return rows


def main() -> None:
    rows = audit()
    output = PROJECT_ROOT / 'data/derived/phase_parity_audit'
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'phase_parity_candidate_audit.csv'
    _write_csv(path, rows)
    print('csv=' + str(path))


if __name__ == '__main__':
    main()
