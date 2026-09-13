'''Validate the fixed 24-image local-band baseline without changing metrology.'''

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

import cv2
import numpy as np

from run_gradient_canny_pilot import PROJECT_ROOT, _load_roi_config, _write_png
from ai_cd_metrology.image_io import load_image
from ai_cd_metrology.metrology.gradient_canny import GradientCannyMetrology
from ai_cd_metrology.schemas import ImageRecord, MeasurementStatus


# Exact prior validation list; no directory sampling or parameter sweep.
IMAGES = {
    'center': (
        'w1_11_c.png', 'w1_26_c.png', 'w1_38_c.png', 'w2_11_c.png',
        'w2_25_c.png', 'w2_38_c.png', 'w3_11_c.png', 'w3_37_c.png',
    ),
    'left-top': (
        'w1_13_lt.png', 'w1_26_lt.png', 'w1_38_lt.png', 'w2_11_lt.png',
        'w2_26_lt.png', 'w2_38_lt.png', 'w3_13_lt.png', 'w3_38_lt.png',
    ),
    'right-bottom': (
        'w1_11_rb.png', 'w1_25_rb.png', 'w1_38_rb.png', 'w2_11_rb.png',
        'w2_25_rb.png', 'w2_38_rb.png', 'w3_11_rb.png', 'w3_38_rb.png',
    ),
}
BAND_COUNT = 5
MEASUREMENTS = ('outer', 'inner', 'gap')


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(scope: str, rows: list[dict], **identity) -> dict:
    counts = Counter(row['local_status'] for row in rows)
    result = {
        'scope': scope, 'filename': None, 'position': None, 'band_index': None,
        'image_count': len({row['filename'] for row in rows}),
        'total_band_count': len(rows), 'valid_band_count': counts['VALID'],
        'warning_band_count': counts['WARNING'], 'fail_band_count': counts['FAIL'],
        'whole_image_status': None, 'whole_outer_px': None,
        'whole_inner_px': None, 'whole_gap_px': None,
        'whole_matches_prior_pairing': all(row['whole_matches_prior_pairing'] for row in rows),
        # Descriptive distributions exclude FAIL, never roll up into measure().
        'distribution_population': 'VALID/WARNING local medians; excludes FAIL',
    }
    result.update(identity)
    for field in MEASUREMENTS:
        values = [
            row[f'{field}_median_px'] for row in rows
            if row['local_status'] != 'FAIL' and row[f'{field}_median_px'] is not None
        ] if scope in ('position', 'overall') else []
        for stat in ('n', 'min', 'median', 'max'):
            result[f'{field}_{stat}_px' if stat != 'n' else f'{field}_n'] = (
                len(values) if stat == 'n' else
                float(getattr(np, stat)(values)) if values else None
            )
    return result


def _render(filename: str, grayscale: np.ndarray, bounds: tuple, rows: list[dict]) -> np.ndarray:
    # Resize only a derived visualization; the source image is never written.
    h, w = grayscale.shape
    factor = 900 / w
    height = max(500, round(h * factor))
    view = cv2.resize(grayscale, (900, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 90, 1450, 3), 24, dtype=np.uint8)
    canvas[45:45 + height, :900] = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    colors = {'VALID': (140, 230, 140), 'WARNING': (0, 210, 255), 'FAIL': (100, 100, 255)}
    cv2.putText(canvas, filename, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1)
    for row in rows:
        top = 45 + round((row['y_start_px'] - bounds[1]) / h * height)
        bottom = 45 + round((row['y_end_px'] - bounds[1]) / h * height)
        color = colors[row['local_status']]
        cv2.line(canvas, (0, min(top, 44 + height)), (899, min(top, 44 + height)), color, 2)
        cv2.line(canvas, (0, min(bottom, 44 + height)), (899, min(bottom, 44 + height)), color, 1)
        y = (top + bottom) // 2
        values = [row[f'{field}_median_px'] for field in MEASUREMENTS]
        labels = (
            f"band {row['band_index']} {row['local_status']}",
            f"original y [{row['y_start_px']},{row['y_end_px']})",
            'Outer / Inner / Gap px: ' + ' / '.join('None' if v is None else f'{v:g}' for v in values),
            f"candidates={row['candidate_count']} rejected={row['rejected_peak_count']}",
        )
        for offset, label in enumerate(labels):
            cv2.putText(canvas, label, (920, y - 27 + offset * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
    whole = rows[0]
    footer = f"Whole ROI: {whole['whole_image_status']}  Outer/Inner/Gap px = {whole['whole_outer_px']}/{whole['whole_inner_px']}/{whole['whole_gap_px']}"
    cv2.putText(canvas, footer, (12, height + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    return canvas


def main() -> None:
    root = PROJECT_ROOT
    baseline_path = root / 'data/derived/pairing_validation/pairing_validation_summary.csv'
    with baseline_path.open(encoding='utf-8-sig', newline='') as handle:
        baseline = {row['filename']: row for row in csv.DictReader(handle)}
    names = {name for group in IMAGES.values() for name in group}
    if len(names) != 24 or set(baseline) != names:
        raise ValueError('prior pairing CSV must contain the same exact 24 images')

    details, summaries, previews = [], [], {}
    for position, filenames in IMAGES.items():
        algorithm = GradientCannyMetrology(
            {position: _load_roi_config(root / 'configs/metrology.yaml', position)},
            band_count=BAND_COUNT,
        )
        for filename in filenames:
            wafer, die, _ = Path(filename).stem.split('_')
            path = root / 'data/raw/images/images/PASS_normal' / filename
            record = ImageRecord(path.stem, path, wafer, die, position, 'standard', 'normal')
            image = load_image(record)
            diagnostic = algorithm.analyze(image, record)
            whole = algorithm.measure(image, record)[0]
            if len(diagnostic.local_bands) != BAND_COUNT:
                raise AssertionError(f'local band count mismatch: {filename}')

            prior = baseline[filename]
            prior_medians = tuple(
                float(np.median([int(v) for v in prior[column].split(';') if v]))
                for column in ('outer_values_px', 'inner_values_px', 'gap_values_px')
            )
            expected_status = (
                MeasurementStatus.WARNING if int(prior['rejected_peak_count'])
                else MeasurementStatus.VALID
            )
            matches = (
                prior['validation_status'] == 'PASS'
                and (whole.outer_width_px, whole.inner_width_px, whole.gap_px) == prior_medians
                and whole.status is expected_status
                and len(diagnostic.candidates) == int(prior['outer_candidate_count'])
            )
            if not matches:
                raise AssertionError(f'whole ROI differs from prior validation: {filename}')
            image_rows = []
            for band in diagnostic.local_bands:
                values = band.medians or (None, None, None)
                rejected = band.pairing.rejected_peaks
                row = {
                    'filename': filename, 'wafer': wafer, 'position': position,
                    'band_index': band.band_index,
                    'y_start_px': band.y_bounds_px[0], 'y_end_px': band.y_bounds_px[1],
                    'local_status': band.status.value,
                    'candidate_count': len(band.candidates),
                    'rejected_peak_count': len(rejected),
                    'outer_median_px': values[0], 'inner_median_px': values[1], 'gap_median_px': values[2],
                    'whole_image_status': whole.status.value,
                    'whole_outer_px': whole.outer_width_px,
                    'whole_inner_px': whole.inner_width_px, 'whole_gap_px': whole.gap_px,
                    'local_failure_reason': band.failure_reason,
                    'partial_peak_count': sum(p.reason.startswith('partial_') for p in rejected),
                    'unexpected_peak_count': sum(p.reason.startswith('unexpected_') for p in rejected),
                    'whole_matches_prior_pairing': matches,
                }
                image_rows.append(row)
            details.extend(image_rows)
            summaries.append(_summary(
                'image', image_rows, filename=filename, position=position,
                whole_image_status=whole.status.value, whole_outer_px=whole.outer_width_px,
                whole_inner_px=whole.inner_width_px, whole_gap_px=whole.gap_px,
            ))
            previews[filename] = (diagnostic.edge_diagnostics.grayscale_roi, diagnostic.roi_bounds, image_rows)
            print(filename + ': ' + ','.join(row['local_status'] for row in image_rows))

    for position in IMAGES:
        summaries.append(_summary('position', [r for r in details if r['position'] == position], position=position))
    for index in range(BAND_COUNT):
        summaries.append(_summary('band_index', [r for r in details if r['band_index'] == index], band_index=index))
    summaries.append(_summary('overall', details))
    if len(details) != 120:
        raise AssertionError('expected 120 image x band records')

    selected = {'w1_11_c.png', 'w1_13_lt.png', 'w1_11_rb.png'}
    ranked = sorted(
        [row for row in summaries if row['scope'] == 'image' and row['warning_band_count'] + row['fail_band_count'] > 0],
        key=lambda row: (-row['fail_band_count'], -row['warning_band_count'], row['filename']),
    )
    selected.update(row['filename'] for row in ranked[:3])
    output = root / 'data/derived/local_band_validation'
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'local_band_validation.csv', details)
    _write_csv(output / 'local_band_validation_summary.csv', summaries)
    for filename in sorted(selected):
        grayscale, bounds, rows = previews[filename]
        path = output / f'{Path(filename).stem}_local_bands.png'
        _write_png(path, _render(filename, grayscale, bounds, rows))
        print(f'diagnostic={path}')
    for row in summaries:
        if row['scope'] != 'image':
            print(row)
    print(f'output={output}')


if __name__ == '__main__':
    main()
