"""Review exactly two parity counterexamples with unchanged metrology."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from audit_pairing_start_shift import _overlay, _rows
from audit_phase_parity import parity_rows
from run_gradient_canny_pilot import PROJECT_ROOT, _load_roi_config, _write_png
from run_local_band_validation import _write_csv
from run_pairing_reference import _read_image
from ai_cd_metrology.metrology.gradient_canny import GradientCannyMetrology
from ai_cd_metrology.schemas import ImageRecord

IMAGES = {'w2_38_rb.png': 'right-bottom', 'w2_26_lt.png': 'left-top'}
EDGE_FIELDS = tuple(f'e{i}_x_original_px' for i in range(1, 7))


def _review_png(filename, diagnostic, band_rows, scores):
    # Reuse the existing ROI/peak/doublet/raw-window overlay unchanged.
    canvas = _overlay(filename, diagnostic, band_rows)
    focus = (2, 3, 4) if filename == 'w2_38_rb.png' else tuple(range(5))
    selected = [r for r in band_rows if r['band_index'] in focus]
    line_count = 3 + sum(3 + len(json.loads(r['raw_candidates_json'])) for r in selected)
    height = canvas.shape[0]
    canvas = cv2.copyMakeBorder(
        canvas, 0, line_count * 23 + 20, 0, 0,
        cv2.BORDER_CONSTANT, value=(24, 24, 24),
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = height + 24

    def text(label, x=12, color=(235, 235, 235), scale=0.46):
        cv2.putText(canvas, label, (x, y), font, scale, color, 1)

    first = scores[0]
    text(f"Whole reference Inner/Gap={first['reference_inner_px']:g}/{first['reference_gap_px']:g} px; separation={first['reference_separation_px']:g} px; ratio={first['reference_separation_ratio']:.6f}", scale=0.58)
    y += 23
    text('E1-E6 below: original image x px. C index is NOT a physical ID. Parity scores do NOT change acceptance.')
    y += 23
    text('Positive: swapped fit better; negative: normal fit better; zero: equal fit. No threshold or correction.')
    y += 23
    score_map = {(r['band_index'], r['candidate_sequence_index']): r for r in scores}
    for row in selected:
        raw = json.loads(row['raw_candidates_json'])
        accepted = {c['sequence_index'] for c in json.loads(row['accepted_candidates_json'])}
        values = [r['normalized_parity_score'] for r in scores if r['band_index'] == row['band_index']]
        text(f"Band {row['band_index']} {row['local_status']}; parity range {min(values):+.6f} to {max(values):+.6f}", scale=0.55)
        y += 23
        for label, x in (('C', 12), ('Long-guard state', 60), ('E1 / E2 / E3 / E4 / E5 / E6 (x px)', 245),
                         ('Outer / Inner / Gap px', 735), ('Normal err', 990), ('Swap err', 1115), ('Parity score', 1235)):
            text(label, x, (160, 210, 230), 0.42)
        y += 23
        for candidate in raw:
            index = candidate['sequence_index']
            score = score_map.get((row['band_index'], index))
            color = (220, 220, 220) if index in accepted else (80, 80, 255)
            text(f'C{index}', 12, color)
            text('ACCEPTED' if index in accepted else 'EXCLUDED', 60, color)
            text(' / '.join(str(candidate[f]) for f in EDGE_FIELDS), 245, color)
            text(' / '.join(str(candidate[f]) for f in ('outer_width_px', 'inner_width_px', 'gap_px')), 735, color)
            text('n.a.' if score is None else f"{score['normal_error']:g}", 990, color)
            text('n.a.' if score is None else f"{score['swapped_error']:g}", 1115, color)
            text('n.a.' if score is None else f"{score['normalized_parity_score']:+.6f}", 1235, color)
            y += 23
        y += 23
    return canvas


def main() -> None:
    output = PROJECT_ROOT / 'data/derived/phase_parity_case_review'
    all_rows, previews = [], []
    for filename, position in IMAGES.items():
        path = PROJECT_ROOT / 'data/raw/images/images/PASS_normal' / filename
        wafer, die, _ = path.stem.split('_')
        record = ImageRecord(path.stem, path, wafer, die, position, 'unknown', 'normal')
        algorithm = GradientCannyMetrology(
            {position: _load_roi_config(PROJECT_ROOT / 'configs/metrology.yaml', position)},
            band_count=5,
        )
        image = _read_image(path)
        diagnostic = algorithm.analyze(image, record)
        whole = algorithm.measure(image, record)[0]  # No calibration.
        bands = _rows(filename, diagnostic, whole)
        scores = parity_rows(diagnostic, group='normal', filename=filename, position=position)
        for row in bands:
            band_scores = [s for s in scores if s['band_index'] == row['band_index']]
            row.update({
                'position': position,
                'reference_inner_px': band_scores[0]['reference_inner_px'],
                'reference_gap_px': band_scores[0]['reference_gap_px'],
                'reference_separation_px': band_scores[0]['reference_separation_px'],
                'reference_separation_ratio': band_scores[0]['reference_separation_ratio'],
                'accepted_candidate_parity_json': json.dumps(band_scores, separators=(',', ':')),
                'score_note': 'image-local value fit only; no phase truth, threshold, exclusion or correction',
            })
            print(filename, 'band', row['band_index'],
                  'y', (row['y_start_px'], row['y_end_px']),
                  'peaks/doublets/raw/accepted', row['detected_peak_count'], row['doublet_count'],
                  row['raw_candidate_count'], row['candidate_count'],
                  'median', row['outer_median_px'], row['inner_median_px'], row['gap_median_px'],
                  'rejected', row['rejected_peaks_json'])
            print('C/Inner/Gap/normal error/swapped error/score',
                  [(s['candidate_sequence_index'], s['inner_px'], s['gap_px'],
                    s['normal_error'], s['swapped_error'], s['normalized_parity_score']) for s in band_scores])
            print('doublets', row['dark_band_doublets_json'])
            print('raw', row['raw_candidates_json'])
        all_rows.extend(bands)
        previews.append((filename, diagnostic, bands, scores))
    assert len(all_rows) == 10 and len(previews) == 2
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'phase_parity_case_review.csv'
    _write_csv(path, all_rows)
    print('csv=' + str(path))
    for filename, diagnostic, bands, scores in previews:
        path = output / f'{Path(filename).stem}_phase_parity_review.png'
        _write_png(path, _review_png(filename, diagnostic, bands, scores))
        print('diagnostic=' + str(path))


if __name__ == '__main__':
    main()
