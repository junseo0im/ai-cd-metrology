'''Inspect pattern narrowing with unchanged local-band raw pixel candidates.'''

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from run_gradient_canny_pilot import PROJECT_ROOT, _load_roi_config, _write_png
from run_local_band_validation import _render, _write_csv
from run_pairing_reference import _read_image
from ai_cd_metrology.data_audit import parse_path_metadata
from ai_cd_metrology.metrology.gradient_canny import GradientCannyMetrology
from ai_cd_metrology.schemas import ImageRecord


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
POSITION_NAMES = {'c': 'center', 'lt': 'left-top', 'rb': 'right-bottom'}
PARTIAL_REASONS = {'partial_left_positive', 'partial_right_negative'}
UNEXPECTED_REASONS = {'unexpected_positive', 'unexpected_negative'}


def _annotate_candidates(diagnostic, band_rows: list[dict], filename: str) -> np.ndarray:
    output = _render(
        filename, diagnostic.edge_diagnostics.grayscale_roi,
        diagnostic.roi_bounds, band_rows,
    )
    x0, y0, x1, y1 = diagnostic.roi_bounds
    display_height = output.shape[0] - 90
    for band in diagnostic.local_bands:
        top = 45 + round((band.y_bounds_px[0] - y0) / (y1 - y0) * display_height)
        bottom = 45 + round((band.y_bounds_px[1] - y0) / (y1 - y0) * display_height)
        selected = {
            candidate.sequence_index
            for field in ('outer_width_px', 'inner_width_px')
            for candidate in sorted(
                band.candidates, key=lambda c: (getattr(c, field), c.sequence_index)
            )[:2]
        }
        for candidate in band.candidates:
            left = round(candidate.e1_x / (x1 - x0) * 900)
            right = round(candidate.e4_x / (x1 - x0) * 900)
            centre = (left + right) // 2
            guide_y = (top + bottom) // 2
            highlighted = candidate.sequence_index in selected
            color = (255, 100, 255) if highlighted else (200, 200, 200)
            cv2.line(output, (left, guide_y), (right, guide_y), color, 2 if highlighted else 1)
            cv2.circle(output, (left, guide_y), 3, color, -1)
            cv2.circle(output, (right, guide_y), 3, color, -1)
            label = f'C{candidate.sequence_index}'
            if highlighted:
                label += f' O{candidate.outer_width_px}/I{candidate.inner_width_px}'
                inner_left = round(candidate.e2_x / (x1 - x0) * 900)
                inner_right = round(candidate.e3_x / (x1 - x0) * 900)
                cv2.line(output, (inner_left, guide_y + 9), (inner_right, guide_y + 9), (255, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            anchor = (max(0, centre - 38), guide_y - 8)
            cv2.putText(output, label, anchor, font, 0.34, (0, 0, 0), 3)
            cv2.putText(output, label, anchor, font, 0.34, color, 1)
    output = cv2.copyMakeBorder(output, 0, 32, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(
        output, 'Magenta: two smallest Outer/Inner candidates; cyan: Inner. Band-centre guides are not measured edge y.',
        (12, output.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--screenshot-position', choices=tuple(POSITION_NAMES),
        help='Only use a human-confirmed position for the noncanonical screenshot.',
    )
    args = parser.parse_args()
    root = PROJECT_ROOT
    image_dir = root / 'data/raw/images/images/FAIL/pattern_narrowing'
    paths = sorted(
        (path for path in image_dir.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name,
    )
    output_dir = root / 'data/derived/pattern_narrowing_pilot'
    output_dir.mkdir(parents=True, exist_ok=True)
    band_rows, candidate_rows = [], []
    for path in paths:
        metadata = parse_path_metadata(path.relative_to(root / 'data/raw'))
        position = metadata['pattern_position']
        metadata_note = 'position parsed from canonical filename'
        if position is None and args.screenshot_position is not None:
            if path.stem.startswith('스크린샷 '):
                position = POSITION_NAMES[args.screenshot_position]
                metadata_note = 'position explicitly supplied by human, not inferred'
        record = ImageRecord(
            path.stem, path, metadata['wafer_id'] or 'unknown',
            metadata['die_id'] or 'unknown', position or 'unknown',
            'unknown', 'pattern_narrowing',
        )
        # Existing diagnostic reader supports Windows Unicode filenames.
        image = _read_image(path)
        common = {
            'filename': path.name, 'position': position, 'band_index': None,
            'y_start_px': None, 'y_end_px': None, 'local_status': None,
            'candidate_count': None, 'boundary_partial_count': None,
            'internal_unexpected_count': None,
            'outer_median_px': None, 'inner_median_px': None, 'gap_median_px': None,
            'whole_image_status': None, 'whole_outer_px': None,
            'whole_inner_px': None, 'whole_gap_px': None,
            'measurement_executed': position is not None,
            'not_run_reason': None, 'metadata_note': metadata_note,
            'rejected_peak_count': None, 'rejected_reasons': None,
            'rejected_peak_details': None, 'local_failure_reason': None,
            'width_px': image.shape[1], 'height_px': image.shape[0],
        }
        if position is None:
            common.update({
                'metadata_note': 'pattern position unknown; cannot select existing ROI',
                'not_run_reason': 'manual pattern_position required; not measurement FAIL',
            })
            band_rows.append(common)
            if image.ndim == 2:
                view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                view = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                view = image.copy()
            view = cv2.resize(view, (1450, round(view.shape[0] * 1450 / view.shape[1])))
            view = cv2.copyMakeBorder(view, 55, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
            cv2.putText(
                view, 'UNMEASURED screenshot: pattern_position unknown; ROI not selected; ruler overlay retained',
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 255), 1,
            )
            _write_png(output_dir / f'{path.stem}_narrowing_diagnostic.png', view)
            print(f'{path.name}: UNMEASURED (position unknown)')
            continue

        algorithm = GradientCannyMetrology(
            {position: _load_roi_config(root / 'configs/metrology.yaml', position)}, band_count=5,
        )
        diagnostic = algorithm.analyze(image, record)
        whole = algorithm.measure(image, record)[0]
        image_rows = []
        for band in diagnostic.local_bands:
            values = band.medians or (None, None, None)
            rejected = band.pairing.rejected_peaks
            row = common | {
                'band_index': band.band_index,
                'y_start_px': band.y_bounds_px[0], 'y_end_px': band.y_bounds_px[1],
                'local_status': band.status.value, 'candidate_count': len(band.candidates),
                'boundary_partial_count': sum(p.reason in PARTIAL_REASONS for p in rejected),
                'internal_unexpected_count': sum(p.reason in UNEXPECTED_REASONS for p in rejected),
                'outer_median_px': values[0], 'inner_median_px': values[1], 'gap_median_px': values[2],
                'whole_image_status': whole.status.value, 'whole_outer_px': whole.outer_width_px,
                'whole_inner_px': whole.inner_width_px, 'whole_gap_px': whole.gap_px,
                'rejected_peak_count': len(rejected),
                'rejected_reasons': ';'.join(p.reason for p in rejected),
                'rejected_peak_details': ';'.join(f'{p.x}:{p.polarity}:{p.reason}' for p in rejected),
                'local_failure_reason': band.failure_reason,
            }
            image_rows.append(row)
            for candidate in band.candidates:
                # Coordinates support manual spatial review, not automatic tracking.
                candidate_rows.append({
                    'filename': path.name, 'position': position, 'band_index': band.band_index,
                    'sequence_index': candidate.sequence_index,
                    'outer_px': candidate.outer_width_px, 'inner_px': candidate.inner_width_px,
                    'gap_px': candidate.gap_px, 'local_status': band.status.value,
                    'y_start_px': band.y_bounds_px[0], 'y_end_px': band.y_bounds_px[1],
                    **{f'e{i}_x_px': getattr(candidate, f'e{i}_x') + diagnostic.roi_bounds[0] for i in range(1, 7)},
                    'coordinate_note': 'original-image x; sequence_index is band-local, not stable finger ID',
                })
        band_rows.extend(image_rows)
        path_out = output_dir / f'{path.stem}_narrowing_diagnostic.png'
        _write_png(path_out, _annotate_candidates(diagnostic, image_rows, path.name))
        print(f'{path.name}: whole={whole.status.value}; local={dict(Counter(r["local_status"] for r in image_rows))}')
    _write_csv(output_dir / 'pattern_narrowing_band_summary.csv', band_rows)
    if not candidate_rows:
        raise ValueError('no raw candidates available; candidate CSV cannot be populated')
    _write_csv(output_dir / 'pattern_narrowing_candidates.csv', candidate_rows)
    print(f'images={len(paths)}; measured={len({r["filename"] for r in band_rows if r["measurement_executed"]})}')
    print(f'output={output_dir}')


if __name__ == '__main__':
    main()
