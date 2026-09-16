"""Audit five known pairing-start cases; never repair or track physical fingers."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from run_gradient_canny_pilot import PROJECT_ROOT, _load_roi_config, _write_png
from run_local_band_validation import _render, _write_csv
from run_pairing_reference import _read_image
from ai_cd_metrology.metrology.gradient_canny import GradientCannyMetrology
from ai_cd_metrology.schemas import ImageRecord

IMAGES = (
    'w1_13_rb_p.png', 'w1_17_c_p.png', 'w1_18_c_p.png',
    'w3_24_rb_p.png', 'w1_16_rb_p.png',
)
POSITIONS = {'c': 'center', 'rb': 'right-bottom'}
EDGE_FIELDS = tuple(f'e{i}_x' for i in range(1, 7))


def _json(value) -> str:
    return json.dumps(value, separators=(',', ':'))


def _candidate(candidate, x0: int) -> dict:
    result = asdict(candidate)
    for field in EDGE_FIELDS:
        result[field + '_original_px'] = result.pop(field) + x0
    return result


def _rows(filename, diagnostic, whole) -> list[dict]:
    x0, y0, x1, y1 = diagnostic.roi_bounds
    rows = []
    for band in diagnostic.local_bands:
        peaks = sorted(
            [(x + x0, 'negative') for x in band.profile.negative_peak_indices]
            + [(x + x0, 'positive') for x in band.profile.positive_peak_indices]
        )
        doublets = [
            {'sequence_index': d.sequence_index, 'segment_index': d.segment_index,
             'negative_x_original_px': d.negative_peak_x + x0,
             'positive_x_original_px': d.positive_peak_x + x0,
             'dark_band_width_px': d.dark_band_width_px}
            for d in band.pairing.doublets
        ]
        rejected = [
            {'x_original_px': p.x + x0, 'polarity': p.polarity, 'reason': p.reason}
            for p in band.pairing.rejected_peaks
        ]
        medians = band.medians or (None, None, None)
        raw = [_candidate(c, x0) for c in band.candidates]
        accepted = [_candidate(c, x0) for c in band.accepted_candidates]
        row = {
            'filename': filename, 'band_index': band.band_index,
            'roi_x_start_px': x0, 'roi_y_start_px': y0,
            'roi_x_end_px': x1, 'roi_y_end_px': y1,
            'y_start_px': band.y_bounds_px[0], 'y_end_px': band.y_bounds_px[1],
            'local_status': band.status.value, 'local_failure_reason': band.failure_reason,
            'detected_peak_count': len(peaks), 'doublet_count': len(doublets),
            'raw_candidate_count': len(raw), 'candidate_count': len(accepted),
            'long_pairing_count': len(band.long_pairing_candidates),
            'rejected_peak_count': len(rejected),
            'boundary_partial_count': sum(p['reason'].startswith('partial_') for p in rejected),
            'internal_unexpected_count': sum(p['reason'].startswith('unexpected_') for p in rejected),
            'outer_median_px': medians[0], 'inner_median_px': medians[1],
            'gap_median_px': medians[2],
            'whole_image_status': whole.status.value,
            'whole_outer_px': whole.outer_width_px,
            'whole_inner_px': whole.inner_width_px, 'whole_gap_px': whole.gap_px,
            'detected_signed_peaks_json': _json([
                {'x_original_px': x, 'polarity': polarity} for x, polarity in peaks]),
            'dark_band_doublets_json': _json(doublets),
            'rejected_peaks_json': _json(rejected),
            'raw_candidates_json': _json(raw), 'accepted_candidates_json': _json(accepted),
            'long_pairing_candidates_json': _json([
                _candidate(c, x0) for c in band.long_pairing_candidates]),
            'coordinate_note': 'original image x px; half-open ROI/y bounds; y-aggregated peaks, not measured edge y',
            'index_note': 'sequence_index is diagnostic order, NOT physical finger ID; no tracking or correction',
        }
        rows.append(row)
    return rows


def _overlay(filename, diagnostic, rows) -> np.ndarray:
    gray = diagnostic.edge_diagnostics.grayscale_roi
    canvas = _render(filename, gray, diagnostic.roi_bounds, rows)
    h, w = gray.shape
    height = canvas.shape[0] - 90
    x0, y0, _, _ = diagnostic.roi_bounds
    negative, positive, rejected_color = (255, 230, 0), (180, 0, 255), (0, 150, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def x_view(x):
        return max(0, min(899, round(x / w * 900)))

    for band in diagnostic.local_bands:
        top = 45 + round((band.y_bounds_px[0] - y0) / h * height)
        bottom = 45 + round((band.y_bounds_px[1] - y0) / h * height)
        layer = canvas.copy()
        for indices, color in (
            (band.profile.negative_peak_indices, negative),
            (band.profile.positive_peak_indices, positive),
        ):
            for x in indices:
                cv2.line(layer, (x_view(x), top + 3), (x_view(x), bottom - 3), color, 1)
        canvas = cv2.addWeighted(canvas, 0.77, layer, 0.23, 0)
        for d in band.pairing.doublets:
            a, b = x_view(d.negative_peak_x), x_view(d.positive_peak_x)
            cv2.line(canvas, (a, top + 12), (b, top + 12), (100, 230, 120), 2)
        for p in band.pairing.rejected_peaks:
            x = x_view(p.x)
            for y in range(top + 3, bottom - 3, 9):
                cv2.line(canvas, (x, y), (x, min(y + 5, bottom - 3)), rejected_color, 2)
            cv2.putText(canvas, p.reason, (x + 4, bottom - 5), font, 0.32, rejected_color, 1)
        flagged = set(band.long_pairing_candidates)
        for c in band.candidates:
            center = (top + bottom) // 2 + (c.sequence_index % 2) * 9
            color = (80, 80, 255) if c in flagged else (240, 240, 240)
            xs = [x_view(getattr(c, f)) for f in EDGE_FIELDS]
            cv2.line(canvas, (xs[0], center), (xs[-1], center), color, 1)
            for index, x in enumerate(xs):
                cv2.circle(canvas, (x, center), 2, negative if index % 2 == 0 else positive, -1)
            cv2.putText(canvas, f'C{c.sequence_index}' + (' LONG' if c in flagged else ''),
                        (xs[0], center + 16), font, 0.34, color, 1)
            # Label E1-E6 of the first raw window, not a tracked physical finger.
            if c is band.candidates[0]:
                for index, x in enumerate(xs):
                    cv2.putText(canvas, f'E{index + 1}', (x - 5, center - 7 - (index % 2) * 13),
                                font, 0.30, color, 1)
        if band.candidates:
            edges = '/'.join(str(x0 + getattr(band.candidates[0], f)) for f in EDGE_FIELDS)
            cv2.putText(canvas, 'C0 original x: ' + edges, (920, bottom - 7),
                        font, 0.35, (235, 235, 235), 1)
    canvas = cv2.copyMakeBorder(canvas, 0, 52, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(canvas, 'Cyan: negative | magenta: positive | green bracket: doublet | orange dashed: rejected',
                (12, canvas.shape[0] - 32), font, 0.49, (235, 235, 235), 1)
    cv2.putText(canvas, 'C: raw E1-E6 window; red LONG excluded; counts/medians accepted. Display y is NOT detected y.',
                (12, canvas.shape[0] - 12), font, 0.47, (235, 235, 235), 1)
    return canvas


def main() -> None:
    output = PROJECT_ROOT / 'data/derived/pairing_start_shift_audit'
    rows, previews = [], []
    for filename in IMAGES:
        wafer, die, alias, _ = Path(filename).stem.split('_')
        position = POSITIONS[alias]
        path = PROJECT_ROOT / 'data/raw/images/images/FAIL/particle' / filename
        record = ImageRecord(path.stem, path, wafer, die, position, 'unknown', 'particle')
        algorithm = GradientCannyMetrology(
            {position: _load_roi_config(PROJECT_ROOT / 'configs/metrology.yaml', position)},
            band_count=5,
        )
        image = _read_image(path)
        diagnostic = algorithm.analyze(image, record)
        whole = algorithm.measure(image, record)[0]  # No calibration.
        image_rows = _rows(filename, diagnostic, whole)
        rows.extend(image_rows)
        previews.append((filename, diagnostic, image_rows))
        print(filename, 'ROI', diagnostic.roi_bounds, 'whole', whole.status.value,
              (whole.outer_width_px, whole.inner_width_px, whole.gap_px))
        for row in image_rows:
            print('band', row['band_index'], 'y', (row['y_start_px'], row['y_end_px']),
                  row['local_status'], 'peaks/doublets/raw/accepted/long',
                  [row[k] for k in ('detected_peak_count', 'doublet_count', 'raw_candidate_count',
                                   'candidate_count', 'long_pairing_count')],
                  'median', [row[k] for k in ('outer_median_px', 'inner_median_px', 'gap_median_px')],
                  'rejected', row['rejected_peaks_json'])
            print('doublets', row['dark_band_doublets_json'])
            print('raw', row['raw_candidates_json'])
    assert len(rows) == 25 and len(previews) == 5
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'pairing_start_shift_audit.csv', rows)
    for filename, diagnostic, image_rows in previews:
        path = output / f'{Path(filename).stem}_pairing_start.png'
        _write_png(path, _overlay(filename, diagnostic, image_rows))
        print('diagnostic=' + str(path))
    print('csv=' + str(output / 'pairing_start_shift_audit.csv'))


if __name__ == '__main__':
    main()
