'''Audit upward ratios on the exact previously measured pilot populations.

One CSV preserves raw local candidates, even after a guard is installed.
No unknown-position image is assigned an ROI; no absolute pixel cutoff is used.
'''

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from run_gradient_canny_pilot import PROJECT_ROOT, _load_roi_config
from run_local_band_validation import IMAGES, _write_csv
from run_pairing_reference import _read_image
from ai_cd_metrology.data_audit import parse_path_metadata
from ai_cd_metrology.metrology.gradient_canny import GradientCannyMetrology
from ai_cd_metrology.schemas import ImageRecord


FIELDS = ('outer_width_px', 'inner_width_px', 'gap_px')
KNOWN_FILENAME = 'w3_24_rb_p.png'
KNOWN_VALUES = (270, 237, 104)


def pilot_images() -> list[tuple[str, Path]]:
    root = PROJECT_ROOT
    images = [('normal', root / 'data/raw/images/images/PASS_normal' / name)
              for names in IMAGES.values() for name in names]
    if len(images) != 24:
        raise AssertionError('normal population must be the prior exact 24 images')
    populations = (
        ('narrowing', 'pattern_narrowing', 'pattern_narrowing_pilot/pattern_narrowing_band_summary.csv'),
        ('particle', 'particle', 'particle_robustness_pilot/particle_image_summary.csv'),
    )
    for group, folder, prior in populations:
        with (root / 'data/derived' / prior).open(encoding='utf-8-sig', newline='') as handle:
            names = sorted({r['filename'] for r in csv.DictReader(handle)
                            if r['measurement_executed'] == 'True'})
        images.extend((group, root / 'data/raw/images/images/FAIL' / folder / name) for name in names)
    return images


def analyze_pilots():
    for group, path in pilot_images():
        metadata = parse_path_metadata(path.relative_to(PROJECT_ROOT / 'data/raw'))
        position = metadata['pattern_position']
        if position is None:
            raise ValueError(f'previously measured image needs confirmed position: {path.name}')
        record = ImageRecord(path.stem, path, metadata['wafer_id'] or 'unknown',
                             metadata['die_id'] or 'unknown', position, 'unknown')
        algorithm = GradientCannyMetrology(
            {position: _load_roi_config(PROJECT_ROOT / 'configs/metrology.yaml', position)}, band_count=5,
        )
        image = _read_image(path)
        yield group, path, position, algorithm, image, record, algorithm.analyze(image, record)


def ratio_rows(diagnostic, *, group: str, filename: str, position: str) -> list[dict]:
    rows = []
    for band in diagnostic.local_bands:
        if not band.candidates:
            continue
        medians = tuple(float(np.median([getattr(c, field) for c in band.candidates])) for field in FIELDS)
        if min(medians) <= 0 or not np.all(np.isfinite(medians)):
            raise ValueError(f'positive finite raw medians required: {filename}, band {band.band_index}')
        for candidate in band.candidates:
            values = tuple(getattr(candidate, field) for field in FIELDS)
            ratios = tuple(value / median for value, median in zip(values, medians, strict=True))
            rows.append({
                'group': group, 'filename': filename, 'position': position,
                'band_index': band.band_index, 'candidate_index': candidate.sequence_index,
                'outer_px': values[0], 'inner_px': values[1], 'gap_px': values[2],
                'band_median_outer_px': medians[0], 'band_median_inner_px': medians[1], 'band_median_gap_px': medians[2],
                'ratio_outer': ratios[0], 'ratio_inner': ratios[1], 'ratio_gap': ratios[2],
                'max_upward_ratio': max(ratios),
                'known_long_pair': filename == KNOWN_FILENAME and band.band_index == 1 and values == KNOWN_VALUES,
            })
    return rows


def verify_guard() -> None:
    # Disable only the new helper in memory to reproduce the pre-guard chain;
    # do not modify production source, prior CSVs, or dataset files.
    import ai_cd_metrology.metrology.gradient_canny as module
    original_flag = module.flag_long_pairing_candidates
    distributions = {}
    changed_bands, whole_flags, local_flags = [], 0, 0
    for group, path, _, algorithm, image, record, after in analyze_pilots():
        try:
            module.flag_long_pairing_candidates = lambda candidates: ()
            before = algorithm.analyze(image, record)
            whole_before = algorithm.measure(image, record)[0]
        finally:
            module.flag_long_pairing_candidates = original_flag
        whole_after = algorithm.measure(image, record)[0]
        def comparable(result):
            return (result.status, result.failure_reason, result.outer_width_px,
                    result.inner_width_px, result.gap_px, result.edge_coordinates,
                    result.outer_width_um, result.inner_width_um, result.gap_um, result.calibration_id)
        assert comparable(whole_after) == comparable(whole_before), path.name
        assert after.candidates == before.candidates and after.pairing == before.pairing, path.name
        whole_flags += len(after.long_pairing_candidates)
        stats = distributions.setdefault(group, {key: Counter() for key in ('whole_before', 'whole_after', 'local_before', 'local_after')})
        stats['whole_before'][whole_before.status.value] += 1
        stats['whole_after'][whole_after.status.value] += 1
        for left, right in zip(before.local_bands, after.local_bands, strict=True):
            assert left.candidates == right.candidates and left.pairing == right.pairing
            np.testing.assert_array_equal(left.profile.values, right.profile.values)
            stats['local_before'][left.status.value] += 1
            stats['local_after'][right.status.value] += 1
            local_flags += len(right.long_pairing_candidates)
            is_known = path.name == KNOWN_FILENAME and right.band_index == 1
            if is_known:
                assert len(right.long_pairing_candidates) == 1
                flagged = right.long_pairing_candidates[0]
                assert tuple(getattr(flagged, field) for field in FIELDS) == KNOWN_VALUES
                assert flagged not in right.accepted_candidates
                assert left.status.value == 'VALID' and right.status.value == 'WARNING'
                assert len(right.candidates) == 6 and len(right.accepted_candidates) == 5
                assert right.medians == (143.0, 117.0, 107.0)
                changed_bands.append((path.name, right.band_index, left.status.value, right.status.value, left.medians, right.medians))
            else:
                assert not right.long_pairing_candidates, (path.name, right.band_index)
                assert (right.status, right.failure_reason, right.medians, right.validation) == (left.status, left.failure_reason, left.medians, left.validation)
    assert whole_flags == 0 and local_flags == 1 and len(changed_bands) == 1
    for group, counts in distributions.items():
        print(group, {key: dict(value) for key, value in counts.items()})
    print('changed bands:', changed_bands)
    print('whole/local flags:', whole_flags, local_flags)
    print('REGRESSION PASS: whole measurements unchanged; only known long local candidate excluded')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-guard', action='store_true', help='Console-only pre/post guard regression on the same pilot populations.')
    if parser.parse_args().verify_guard:
        verify_guard()
        return
    rows, image_counts = [], {}
    whole_preserve_max = 0.0
    for group, path, position, _, _, _, diagnostic in analyze_pilots():
        image_counts[group] = image_counts.get(group, 0) + 1
        rows.extend(ratio_rows(diagnostic, group=group, filename=path.name, position=position))
        if diagnostic.candidates:
            medians = tuple(float(np.median([getattr(c, f) for c in diagnostic.candidates])) for f in FIELDS)
            whole_preserve_max = max(whole_preserve_max, max(
                max(getattr(c, f) / m for f, m in zip(FIELDS, medians, strict=True))
                for c in diagnostic.candidates
            ))
    known = [r for r in rows if r['known_long_pair']]
    if len(known) != 1:
        raise AssertionError('expected exactly one known 270/237/104 band-1 candidate')
    for label, selected in (
        ('normal', [r for r in rows if r['group'] == 'normal']),
        ('narrowing', [r for r in rows if r['group'] == 'narrowing']),
        ('particle_all', [r for r in rows if r['group'] == 'particle']),
        ('particle_preserve_w1_16_rb_p', [r for r in rows if r['filename'] == 'w1_16_rb_p.png']),
        ('particle_other_than_known', [r for r in rows if r['group'] == 'particle' and not r['known_long_pair']]),
    ):
        values = [r['max_upward_ratio'] for r in selected]
        print(label, 'n/min/p50/p95/max:', len(values), *np.quantile(values, [0, .5, .95, 1]).tolist())
    preserve = max(r['max_upward_ratio'] for r in rows if not r['known_long_pair'])
    print('images:', image_counts, 'local bands:', sum(image_counts.values()) * 5)
    print('preserve_max_ratio:', preserve, 'whole_roi_preserve_max:', whole_preserve_max)
    print('known_long_pair:', known[0])
    print('separation:', known[0]['max_upward_ratio'] - preserve)
    output = PROJECT_ROOT / 'data/derived/long_pairing_guard_audit'
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'long_pairing_ratio_summary.csv', rows)
    print('output:', output / 'long_pairing_ratio_summary.csv')


if __name__ == '__main__':
    main()
