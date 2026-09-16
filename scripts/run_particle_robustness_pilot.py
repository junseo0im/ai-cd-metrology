'''Inspect particle robustness with unchanged pixel/local-band metrology.

Observation text is manual visual review, not particle localization or a
classifier. Unknown positions are not assigned an ROI or measurement FAIL.
'''

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

from run_gradient_canny_pilot import PROJECT_ROOT, _load_roi_config, _write_png
from run_local_band_validation import _write_csv
from run_pairing_reference import _read_image
from run_pattern_narrowing_pilot import _annotate_candidates
from ai_cd_metrology.data_audit import parse_path_metadata
from ai_cd_metrology.metrology.gradient_canny import GradientCannyMetrology
from ai_cd_metrology.schemas import ImageRecord


PARTIAL = {'partial_left_positive', 'partial_right_negative'}
INTERNAL = {'unexpected_positive', 'unexpected_negative'}
POSITIONS = {'c': 'center', 'lt': 'left-top', 'rb': 'right-bottom'}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
# Six manually chosen review examples, not defect classes or model thresholds.
PREVIEWS = (
    'w1_16_rb_p.png', 'w3_36_c_p.png',
    'w1_13_c_p.png', 'w3_24_rb_p.png',
    'w3_23_c_p.png', 'w1_26_rb_p.png',
)
# Free-text observations reviewed against source images and diagnostics.
# Whole impact cannot be causal without a matched particle-free image.
OBSERVATIONS = {
    'w1_13_c_p.png': (
        'Particle touches/overlaps right-side optical boundaries within ROI.',
        'Band 2: one internal positive rejection near particle; candidates 7 to 6. Other bands retain 7.',
        'Whole VALID, 7 candidates, no rejections; causal width bias not established.',
    ),
    'w1_13_rb_p.png': (
        'Small particles between edges; a left-edge particle is outside the normalized x ROI.',
        'Bands 3/4: left boundary partial and large median change; pairing start shifts. Do not attribute this to particle.',
        'Whole VALID, 7 candidates; local physical pairing phase needs review.',
    ),
    'w1_16_rb_p.png': (
        'Upper-right particle touches an optical edge inside ROI.',
        'All bands retain 7 candidates; only repeated left boundary partial, no internal rejection.',
        'Whole WARNING from boundary partial; measurement retained; causal width bias unknown.',
    ),
    'w1_17_c_p.png': (
        'Central particle touches an optical edge within ROI.',
        'Particle-area bands retain candidates. Band 4 has left partial and pairing-start/median shift away from central particle.',
        'Whole VALID, 7 candidates; local shift is not evidence of particle-induced narrowing.',
    ),
    'w1_18_c_p.png': (
        'Several particles touch edges; largest upper particle is above the configured ROI.',
        'Band 4: left partial with large pairing-start/median shift; no internal rejection. Particle causality unclear.',
        'Whole VALID, 7 candidates; local physical pairing phase needs review.',
    ),
    'w1_26_rb_p.png': (
        'Prominent round particle between boundaries is separated from edges; lower residue partly outside ROI.',
        'Particle-area bands retain 7 candidates. Band 4 internal positive rejection reduces count to 6; no clear optical correspondence to the main particle.',
        'Whole WARNING from left boundary partial; 7 candidates retained.',
    ),
    'w3_11_lt_p.png': (
        'Large bottom-right particle overlaps edges but is mostly below ROI; only its upper fringe may enter ROI.',
        'All 5 bands VALID with 7 candidates; does not validate measurement through the main particle body.',
        'Whole VALID, 7 candidates, no rejections; ROI coverage limits conclusion.',
    ),
    'w3_12_rb_p.png': (
        'Upper-right particle touches an optical edge within ROI.',
        'All 5 bands VALID, 7 candidates each; no rejection or large band-median shift.',
        'Whole VALID, 7 candidates, no rejections; measurement retained.',
    ),
    'w3_14_rb_p.png': (
        'Upper particle touches an edge at the ROI top boundary; much of particle is above ROI.',
        'All 5 bands VALID, 7 candidates each; main-particle coverage is limited.',
        'Whole VALID, 7 candidates, no rejections; no matched clean reference for bias.',
    ),
    'w3_23_c_p.png': (
        'Prominent particle lies between right-side boundaries; touching is unclear at this contrast.',
        'Band 1 contains the particle and an internal positive rejection farther right, not at particle x. Band 3 has two internal rejections elsewhere; candidates 7 to 6/5. Causality is uncertain.',
        'Whole WARNING with only boundary partial, 7 candidates; local losses are hidden by whole profile.',
    ),
    'w3_24_rb_p.png': (
        'Upper-middle particle crosses an optical boundary inside ROI.',
        'Band 1 stays VALID with 6 versus 7 candidates, no rejected peak, but raw C2 Outer/Inner/Gap=270/237/104 px crosses the occluded boundary. Median=145.5/118/105.5 px hides this long window.',
        'Whole VALID, 7 candidates, no rejections; local disturbance remains despite VALID.',
    ),
    'w3_36_c_p.png': (
        'Small particles within ROI are separated from edges; the large lower-left edge-touching particle is below the ROI.',
        'All 5 bands VALID with 7 candidates; small band-median variation despite visible particles.',
        'Whole VALID, 7 candidates, no rejections; measurement retained for the ROI, not evidence of robustness through the large excluded particle.',
    ),
}


def _rejection_counts(peaks) -> dict:
    return {
        'rejected_peak_count': len(peaks),
        'boundary_partial_count': sum(p.reason in PARTIAL for p in peaks),
        'internal_unexpected_count': sum(p.reason in INTERNAL for p in peaks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--position', action='append', default=[], metavar='FILENAME=c|lt|rb',
                        help='Human-confirmed position only for noncanonical filenames.')
    args = parser.parse_args()
    overrides = {}
    for value in args.position:
        name, separator, position = value.rpartition('=')
        if not separator or position not in POSITIONS or name in overrides:
            parser.error('each --position must be a unique FILENAME=c|lt|rb')
        overrides[name] = POSITIONS[position]
    root = PROJECT_ROOT
    folder = root / 'data/raw/images/images/FAIL/particle'
    paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if set(overrides) - {p.name for p in paths}:
        parser.error('position override filename must exist in particle folder')
    output = root / 'data/derived/particle_robustness_pilot'
    output.mkdir(parents=True, exist_ok=True)
    image_rows, band_rows, previews = [], [], {}
    for path in paths:
        metadata = parse_path_metadata(path.relative_to(root / 'data/raw'))
        position = metadata['pattern_position']
        if position is not None and path.name in overrides:
            parser.error('cannot override a canonical filename position')
        position = position or overrides.get(path.name)
        notes = OBSERVATIONS.get(path.name, (
            'Large visible particle overlaps optical bands; measured-edge overlap unknown until ROI position confirmed.',
            'Not assessed: pattern_position unknown.', 'Not assessed: no ROI selected.',
        ))
        row = {
            'filename': path.name, 'relative_path': path.relative_to(folder).as_posix(),
            'position': position, 'measurement_executed': position is not None,
            'not_run_reason': None if position is not None else 'manual pattern_position required; not measurement FAIL',
            'whole_image_status': None, 'whole_outer_px': None, 'whole_inner_px': None, 'whole_gap_px': None,
            'whole_candidate_count': None, 'whole_rejected_peak_count': None,
            'whole_boundary_partial_count': None, 'whole_internal_unexpected_count': None,
            'whole_failure_reason': None, 'local_valid_count': None, 'local_warning_count': None,
            'local_fail_count': None, 'local_boundary_partial_count': None,
            'local_internal_unexpected_count': None,
            'edge_overlap_observation': notes[0], 'local_impact_observation': notes[1],
            'whole_roi_impact_observation': notes[2],
            'observation_basis': 'manual visual review, no clean matched control; not classification',
        }
        if position is None:
            image_rows.append(row)
            print(path.name + ': UNMEASURED (position unknown)')
            continue
        record = ImageRecord(path.stem, path, metadata['wafer_id'] or 'unknown',
                             metadata['die_id'] or 'unknown', position, 'unknown', 'particle')
        algorithm = GradientCannyMetrology({position: _load_roi_config(root / 'configs/metrology.yaml', position)}, band_count=5)
        image = _read_image(path)
        diagnostic = algorithm.analyze(image, record)
        whole = algorithm.measure(image, record)[0]  # No calibration supplied.
        whole_counts = _rejection_counts(diagnostic.pairing.rejected_peaks)
        row.update({
            'whole_image_status': whole.status.value, 'whole_outer_px': whole.outer_width_px,
            'whole_inner_px': whole.inner_width_px, 'whole_gap_px': whole.gap_px,
            'whole_candidate_count': len(diagnostic.candidates), 'whole_failure_reason': whole.failure_reason,
            **{f'whole_{key}': value for key, value in whole_counts.items()},
        })
        details = []
        for band in diagnostic.local_bands:
            medians = band.medians or (None, None, None)
            peaks = band.pairing.rejected_peaks
            details.append({
                'filename': path.name, 'position': position, 'band_index': band.band_index,
                'y_start_px': band.y_bounds_px[0], 'y_end_px': band.y_bounds_px[1],
                'local_status': band.status.value, 'candidate_count': len(band.candidates),
                **_rejection_counts(peaks),
                'rejected_reasons': ';'.join(p.reason for p in peaks),
                'rejected_peak_details': ';'.join(f'{p.x + diagnostic.roi_bounds[0]}:{p.polarity}:{p.reason}' for p in peaks),
                'rejected_coordinate_note': 'original-image x; y bounds identify aggregation band, not measured edge y',
                'outer_median_px': medians[0], 'inner_median_px': medians[1], 'gap_median_px': medians[2],
                'local_failure_reason': band.failure_reason,
                'whole_image_status': whole.status.value, 'whole_outer_px': whole.outer_width_px,
                'whole_inner_px': whole.inner_width_px, 'whole_gap_px': whole.gap_px,
            })
            if peaks or len(band.candidates) != len(diagnostic.candidates):
                # Console-only raw values for spatial review; no raw-candidate CSV.
                print(path.name, 'band', band.band_index, 'raw (index, original E1/E4, O/I/G):',
                      [(c.sequence_index, c.e1_x + diagnostic.roi_bounds[0], c.e4_x + diagnostic.roi_bounds[0],
                        c.outer_width_px, c.inner_width_px, c.gap_px) for c in band.candidates])
        counts = Counter(r['local_status'] for r in details)
        row.update({
            'local_valid_count': counts['VALID'], 'local_warning_count': counts['WARNING'],
            'local_fail_count': counts['FAIL'],
            'local_boundary_partial_count': sum(r['boundary_partial_count'] for r in details),
            'local_internal_unexpected_count': sum(r['internal_unexpected_count'] for r in details),
        })
        image_rows.append(row)
        band_rows.extend(details)
        previews[path.name] = (diagnostic, details)
    _write_csv(output / 'particle_image_summary.csv', image_rows)
    if not band_rows:
        raise ValueError('no positioned particle image available')
    _write_csv(output / 'particle_band_summary.csv', band_rows)
    selected = [r['filename'] for r in image_rows if r['whole_image_status'] == 'FAIL' or r['local_fail_count']]
    selected = list(dict.fromkeys(selected + list(PREVIEWS)))[:6]
    for filename in selected:
        if filename not in previews:
            continue
        diagnostic, details = previews[filename]
        view = _annotate_candidates(diagnostic, details, filename)
        x0, y0, x1, y1 = diagnostic.roi_bounds
        height = view.shape[0] - 122  # Existing renderer: 90 px header/footer + 32 px note.
        for band in diagnostic.local_bands:
            top = 45 + round((band.y_bounds_px[0] - y0) / (y1 - y0) * height)
            bottom = 45 + round((band.y_bounds_px[1] - y0) / (y1 - y0) * height)
            for peak in band.pairing.rejected_peaks:
                x = round(peak.x / (x1 - x0) * 900)
                color = (70, 70, 255) if peak.reason in INTERNAL else (0, 180, 255)
                for y in range(top + 3, bottom - 2, 12):
                    cv2.line(view, (x, y), (x, min(y + 6, bottom - 2)), color, 2)
            # Manual review highlight, not an automatic anomaly criterion.
            if filename == 'w3_24_rb_p.png' and band.band_index == 1:
                for candidate in band.candidates:
                    if candidate.sequence_index == 2:
                        left = round(candidate.e1_x / (x1 - x0) * 900)
                        right = round(candidate.e4_x / (x1 - x0) * 900)
                        y = (top + bottom) // 2 + 20
                        cv2.arrowedLine(view, (left, y), (right, y), (0, 230, 255), 2, tipLength=0.08)
                        label = f'Manual C2 O{candidate.outer_width_px}/I{candidate.inner_width_px}/G{candidate.gap_px}'
                        cv2.putText(view, label, (left, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 3)
                        cv2.putText(view, label, (left, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 230, 255), 1)
        cv2.putText(view, 'Dashed red: internal rejection; orange: boundary partial. No particle localization.',
                    (12, view.shape[0] - 29), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1)
        _write_png(output / f'{Path(filename).stem}_particle_diagnostic.png', view)
    print('whole:', dict(Counter(r['whole_image_status'] for r in image_rows if r['measurement_executed'])))
    print('local:', dict(Counter(r['local_status'] for r in band_rows)))
    print('images:', len(paths), 'measured:', sum(r['measurement_executed'] for r in image_rows))
    print('output:', output)


if __name__ == '__main__':
    main()
