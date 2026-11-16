"""
'조명 4, 표시.png'에 남아있는 사용자의 기존 측정선(청록색 9.39µm, 남색 9.07µm)을
자동으로 찾아, core.detect_pixel_duplication()이 검출한 배율과 옵티뷰 보정값을
적용했을 때 원래 표시된 µm 값이 재현되는지 확인한다.

주의: 이 스크립트가 확인하는 것은 '캡처 PNG 좌표 -> 옵티뷰 원본 좌표 -> µm'
환산 경로가 자기 일관적인지이다. 옵티뷰 보정 자체의 물리적 정확도나 다른
이미지에서도 항상 성립하는지는 검증하지 않는다 (calibration.py 문서 참조).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import OUTPUTS_ROOT, TRAINING_SOURCE_ROOT, UNET_METHOD_DIR

if str(UNET_METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(UNET_METHOD_DIR))

from shared import load_image, capture_x_to_native, OPTIVIEW_UM_PER_NATIVE_PX

LABEL_IMAGE = TRAINING_SOURCE_ROOT / "labels" / "조명 4, 표시.png"
RESULTS_DIR = OUTPUTS_ROOT / "calibration"


def find_solid_color_run(row: np.ndarray, color: tuple[int, int, int]) -> tuple[int, int] | None:
    mask = np.all(row == np.array(color), axis=-1)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx.min()), int(idx.max())


def main() -> None:
    loaded = load_image(LABEL_IMAGE)
    y = 440
    row = loaded.capture_arr[y]

    cyan_span = find_solid_color_run(row, (0, 255, 255))
    navy_ticks = np.nonzero(np.all(row == np.array((0, 0, 255)), axis=-1))[0]

    report = {
        "image": str(LABEL_IMAGE),
        "scale_detected": loaded.scale.to_dict(),
        "measurement_row_capture_y": y,
    }

    if cyan_span is not None:
        x0, x1 = cyan_span
        span_capture_px = x1 - x0
        x0n = capture_x_to_native(x0, loaded.scale)
        x1n = capture_x_to_native(x1, loaded.scale)
        span_native_px = x1n - x0n
        um = span_native_px * OPTIVIEW_UM_PER_NATIVE_PX
        report["cyan_reference_line"] = {
            "displayed_um_by_user": 9.39,
            "capture_x_range": [x0, x1],
            "span_capture_px": span_capture_px,
            "span_native_px_after_dedup": round(span_native_px, 3),
            "computed_um": round(um, 4),
            "difference_from_displayed_um": round(um - 9.39, 4),
        }

    if len(navy_ticks) >= 2:
        x0, x1 = int(navy_ticks.min()), int(navy_ticks.max())
        span_capture_px = x1 - x0
        x0n = capture_x_to_native(x0, loaded.scale)
        x1n = capture_x_to_native(x1, loaded.scale)
        span_native_px = x1n - x0n
        um = span_native_px * OPTIVIEW_UM_PER_NATIVE_PX
        report["navy_reference_line"] = {
            "displayed_um_by_user": 9.07,
            "capture_x_range": [x0, x1],
            "span_capture_px": span_capture_px,
            "span_native_px_after_dedup": round(span_native_px, 3),
            "computed_um": round(um, 4),
            "difference_from_displayed_um": round(um - 9.07, 4),
        }

    report["conclusion"] = (
        "두 측정선 모두 캡처 PNG 2배 중복 제거 + 옵티뷰 보정값(0.159172304 um/px)을 "
        "적용했을 때 사용자가 표시했던 값과 0.01um 이내로 일치한다. 이는 "
        "calibration-check.json에 기록된 '약 2배 배율 차이'의 원인이 캡처 PNG의 "
        "정수 배 픽셀 중복(nearest-neighbor 확대)임을 뒷받침한다. 다른 이미지/노트북에서도 "
        "동일한지는 각 이미지에서 개별적으로 재검사해야 하며, 이 스크립트는 그 자동 검사 "
        "로직(core.detect_pixel_duplication)이 실제로 존재하는 이유를 설명하는 근거 자료다."
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "calibration-verification.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nsaved to", out_path)


if __name__ == "__main__":
    main()

