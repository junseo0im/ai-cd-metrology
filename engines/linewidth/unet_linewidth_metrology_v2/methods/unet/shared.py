"""
선폭 자동 측정에서 학습·평가·추론이 공통으로 쓰는 유틸리티 모음.

이미지 로드/좌표 변환/경계 검출 (옛 core.py), 보정값 (옛 calibration.py),
측정 결과 오버레이 렌더링 (옛 visualize.py)을 한 파일로 합쳤다.

측정 대상 (analysis/measurement-method.md, 검토결과.md 참조):
  1) 가느다란 붉은 경계선 자체의 가로 두께
  2) 그 경계선 사이에 교대로 나타나는 두 종류의 넓은 밴드 폭 (line / gap)

모든 측정은 "원본 이미지 좌표"에서 수행한다. 여기서 원본 좌표란 화면에 표시되는
확대/축소 배율이 아니라, 아래 detect_pixel_duplication()이 찾아내는, 캡처 PNG가
실제 촬영 해상도를 정수 배로 최근접 이웃(nearest-neighbor) 확대해 저장했는지 여부를
반영한 좌표계다. 화면 표시 배율과 측정 계산을 분리하라는 요구사항에 대응한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

import sys as _sys
from pathlib import Path as _Path

_LINEWIDTH_ROOT = _Path(__file__).resolve().parents[2]
if str(_LINEWIDTH_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_LINEWIDTH_ROOT))

from team_protocol import (  # noqa: E402
    measurement_rows,
    normalize_measurement_type as protocol_measurement_type,
    parse_measurement_types,
)


# ---------------------------------------------------------------------------
# 1. 이미지 로드 및 픽셀 중복(캡처 확대 배율) 자동 검출
# ---------------------------------------------------------------------------


@dataclass
class ScaleInfo:
    factor_x: int
    factor_y: int
    offset_x: int
    offset_y: int
    confidence_x: float
    confidence_y: float

    @property
    def is_duplicated(self) -> bool:
        return self.factor_x > 1 or self.factor_y > 1

    def to_dict(self) -> dict:
        return {
            "factor_x": self.factor_x,
            "factor_y": self.factor_y,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "confidence_x": round(self.confidence_x, 6),
            "confidence_y": round(self.confidence_y, 6),
        }


DUP_THRESHOLD = 0.98  # 자연 영상에서 인접 픽셀이 우연히 완전히 동일할 확률은 이 값보다 훨씬 낮다.


def normalize_measurement_type(value: int | str) -> str:
    """Return the canonical measurement type used by every entry point."""
    if str(value).strip().lower() == "wall":
        value = "sidewall"
    return protocol_measurement_type(value)


def normalize_measurement_types(value=None) -> list[str]:
    """Parse names such as ``line gap``; None gives the default line and gap."""
    return parse_measurement_types(value)


def default_measurement_rows(height: int) -> list[int]:
    """Common native-y positions: 25%, 45%, and 65% of height."""
    return [y for _fraction, y in measurement_rows(height)]


def _best_offset(length: int, pair_frac_fn) -> tuple[int, float]:
    best_offset, best_frac = 0, -1.0
    for offset in (0, 1):
        frac = pair_frac_fn(offset)
        if frac > best_frac:
            best_offset, best_frac = offset, frac
    return best_offset, best_frac


def detect_pixel_duplication(arr: np.ndarray) -> ScaleInfo:
    """
    캡처 PNG의 인접 픽셀이 (행 또는 열 방향으로) 쌍을 이뤄 완전히 동일한 값을
    갖는 비율을 검사해, 실제 촬영 해상도보다 확대되어 저장되었는지 판단한다.

    지금까지 조명 자료(조명 4~7, 표시/비표시)에서는 가로·세로 모두 2배
    최근접 이웃 확대가 매우 높은 신뢰도(>0.99)로 확인되었다. 이 함수는 그 값을
    하드코딩하지 않고 매 이미지마다 다시 검사하므로, 다른 노트북이나 다른 캡처
    경로에서 배율이 다르거나 확대가 없는 경우에도 안전하게 동작한다.
    """
    h, w = arr.shape[:2]
    # Pack each pixel's channels into one integer so a pixel comparison is a
    # single 2-D comparison (same result as comparing every channel, ~3x faster).
    if arr.ndim == 3:
        packed = np.zeros((h, w), dtype=np.uint32)
        for channel in range(arr.shape[2]):
            packed = (packed << 8) | arr[..., channel].astype(np.uint32)
    else:
        packed = arr

    def h_pair_frac(offset: int) -> float:
        a = packed[:, offset : w - 1 : 2]
        b = packed[:, offset + 1 : w : 2]
        n = min(a.shape[1], b.shape[1])
        if n == 0:
            return 0.0
        return float(np.mean(a[:, :n] == b[:, :n]))

    def v_pair_frac(offset: int) -> float:
        a = packed[offset : h - 1 : 2, :]
        b = packed[offset + 1 : h : 2, :]
        n = min(a.shape[0], b.shape[0])
        if n == 0:
            return 0.0
        return float(np.mean(a[:n] == b[:n]))

    ox, fx = _best_offset(w, h_pair_frac)
    oy, fy = _best_offset(h, v_pair_frac)

    factor_x = 2 if fx >= DUP_THRESHOLD else 1
    factor_y = 2 if fy >= DUP_THRESHOLD else 1

    return ScaleInfo(
        factor_x=factor_x,
        factor_y=factor_y,
        offset_x=ox if factor_x == 2 else 0,
        offset_y=oy if factor_y == 2 else 0,
        confidence_x=fx,
        confidence_y=fy,
    )


@dataclass
class LoadedImage:
    path: str
    capture_arr: np.ndarray  # 캡처된 PNG 원본 배열 (표시/오버레이용)
    native_arr: np.ndarray  # 중복 제거된(추정 원본 해상도) 배열 (측정용)
    scale: ScaleInfo


def load_image(path: str) -> LoadedImage:
    img = Image.open(path).convert("RGB")
    capture_arr = np.asarray(img)
    scale = detect_pixel_duplication(capture_arr)
    native_arr = capture_arr[scale.offset_y :: scale.factor_y, scale.offset_x :: scale.factor_x, :]
    return LoadedImage(path=path, capture_arr=capture_arr, native_arr=native_arr, scale=scale)


# ---------------------------------------------------------------------------
# 2. 가로 프로파일 추출
# ---------------------------------------------------------------------------


def extract_row_profile(native_arr: np.ndarray, y: int, band: int = 2) -> np.ndarray:
    """
    y를 중심으로 (band*2+1)개 행의 G채널을 평균해 잡음을 줄인 가로 밝기 프로파일을 만든다.
    G채널(및 휘도)은 가느다란 붉은 경계선에서 뚜렷한 골(valley)로 나타나는 것을
    실제 이미지에서 확인했다 (measurement-method.md 참조: 빨간 픽셀 카운트 대신
    밝기/대비 프로파일 분석을 기본으로 선택).
    """
    y0 = max(0, y - band)
    y1 = min(native_arr.shape[0], y + band + 1)
    block = native_arr[y0:y1].astype(np.float64)
    return block[:, :, 1].mean(axis=0)


# ---------------------------------------------------------------------------
# 3. 경계 검출 (서브픽셀)
# ---------------------------------------------------------------------------


@dataclass
class LineDetection:
    center_x: float  # 골의 대략적 중심 (참고용, 정수 격자 기준)
    left_edge_x: float  # 왼쪽 경계 서브픽셀 x 좌표 (native)
    right_edge_x: float  # 오른쪽 경계 서브픽셀 x 좌표 (native)
    prominence: float  # 골 깊이 (신뢰도 참고용)
    thickness_px: float = field(init=False)

    def __post_init__(self) -> None:
        self.thickness_px = self.right_edge_x - self.left_edge_x

    def to_dict(self) -> dict:
        return {
            "center_x_native": round(self.center_x, 3),
            "left_edge_x_native": round(self.left_edge_x, 3),
            "right_edge_x_native": round(self.right_edge_x, 3),
            "thickness_px_native": round(self.thickness_px, 3),
            "prominence": round(self.prominence, 2),
        }


# ---------------------------------------------------------------------------
# 4. line / gap 밴드 폭 계산
# ---------------------------------------------------------------------------


@dataclass
class BandMeasurement:
    index: int  # 몇 번째 붉은 선 사이의 밴드인지 (0부터)
    left_line_index: int
    right_line_index: int
    left_x: float  # 왼쪽 붉은 선의 오른쪽 경계 (native)
    right_x: float  # 오른쪽 붉은 선의 왼쪽 경계 (native)
    band_group: str  # "A" 또는 "B" — 교대로 나타나는 두 밴드 종류
    measurement_type: str | None = None  # 밝은 쪽=line, 어두운 쪽=gap
    mean_intensity: float | None = None
    width_px: float = field(init=False)

    def __post_init__(self) -> None:
        self.width_px = self.right_x - self.left_x

    def to_dict(self) -> dict:
        return {
            "band_index": self.index,
            "band_group": self.band_group,
            "measurement_type": self.measurement_type,
            "mean_intensity": (
                round(self.mean_intensity, 3) if self.mean_intensity is not None else None
            ),
            "between_line_index": [self.left_line_index, self.right_line_index],
            "left_x_native": round(self.left_x, 3),
            "right_x_native": round(self.right_x, 3),
            "width_px_native": round(self.width_px, 3),
        }


def compute_bands(
    lines: list[LineDetection],
    profile: np.ndarray | None = None,
) -> list[BandMeasurement]:
    """
    정렬된 붉은 경계선들 사이의 밴드(넓은 line 또는 gap 영역) 폭을 계산한다.
    두 밴드 종류가 번갈아 나타난다고 가정한다. profile이 있으면 A/B 각 그룹의
    중앙 밝기값을 비교해 더 밝은 그룹을 line, 더 어두운 그룹을 gap으로 정한다.
    이는 프로젝트 정의인 '밝은 노란색 영역=line'을 그대로 코드화한 것이다.
    """
    ordered = sorted(lines, key=lambda ld: ld.center_x)
    bands: list[BandMeasurement] = []
    for i in range(len(ordered) - 1):
        left_line = ordered[i]
        right_line = ordered[i + 1]
        group = "A" if i % 2 == 0 else "B"
        left_x = left_line.right_edge_x
        right_x = right_line.left_edge_x
        mean_intensity = None
        if profile is not None and right_x > left_x:
            x0 = max(0, int(np.ceil(left_x)))
            x1 = min(len(profile), int(np.floor(right_x)) + 1)
            if x1 > x0:
                # 경계 번짐의 영향을 줄이기 위해 밴드 중앙 60%만 사용한다.
                margin = int((x1 - x0) * 0.20)
                inner0 = min(x1, x0 + margin)
                inner1 = max(inner0 + 1, x1 - margin)
                mean_intensity = float(np.mean(profile[inner0:inner1]))
        bands.append(
            BandMeasurement(
                index=i,
                left_line_index=i,
                right_line_index=i + 1,
                left_x=left_x,
                right_x=right_x,
                band_group=group,
                mean_intensity=mean_intensity,
            )
        )

    group_values = {
        group: [
            band.mean_intensity
            for band in bands
            if band.band_group == group and band.mean_intensity is not None
        ]
        for group in ("A", "B")
    }
    if group_values["A"] and group_values["B"]:
        score_a = float(np.median(group_values["A"]))
        score_b = float(np.median(group_values["B"]))
        line_group = "A" if score_a >= score_b else "B"
        for band in bands:
            band.measurement_type = "line" if band.band_group == line_group else "gap"
    return bands


def build_width_measurements(
    lines: list[LineDetection],
    bands: list[BandMeasurement],
    measurement_type: int | str,
) -> list[dict]:
    """Create one common record shape for line, gap, and sidewall widths."""
    kind = normalize_measurement_type(measurement_type)
    records: list[dict] = []
    if kind == "sidewall":
        for source_index, item in enumerate(sorted(lines, key=lambda value: value.center_x), start=1):
            records.append(
                {
                    "measurement_index": len(records) + 1,
                    "source_index": source_index,
                    "measurement_type": kind,
                    "left_x_native": float(item.left_edge_x),
                    "right_x_native": float(item.right_edge_x),
                    "center_x_native": float(item.center_x),
                    "width_px_native": float(item.thickness_px),
                    "confidence": float(item.prominence),
                }
            )
        return records

    for item in bands:
        if item.measurement_type != kind:
            continue
        records.append(
            {
                "measurement_index": len(records) + 1,
                "source_index": item.index + 1,
                "measurement_type": kind,
                "band_group": item.band_group,
                "left_x_native": float(item.left_x),
                "right_x_native": float(item.right_x),
                "center_x_native": float((item.left_x + item.right_x) / 2.0),
                "width_px_native": float(item.width_px),
                "mean_intensity": item.mean_intensity,
            }
        )
    return records


def build_width_measurements_by_type(
    lines: list[LineDetection],
    bands: list[BandMeasurement],
    measurement_types,
) -> dict[str, list[dict]]:
    """Return independent measurement lists for every selected width type."""
    return {
        kind: build_width_measurements(lines, bands, kind)
        for kind in normalize_measurement_types(measurement_types)
    }


# ---------------------------------------------------------------------------
# 5. Multi-row width statistics and usable-feature filtering
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. 보정값 (native px -> um)
# ---------------------------------------------------------------------------

# 모든 방법은 중복 픽셀을 제거한 native 좌표에서 폭을 계산하고,
# 고정값 ``1 native px = 0.1592 µm``를 사용해 µm로 환산한다.
# 픽셀 중복 검출은 좌표 복원에만 사용하며 보정값 적용 여부를 바꾸지 않는다.

OPTIVIEW_UM_PER_NATIVE_PX = 0.1592


@dataclass
class CalibrationResult:
    um_per_native_px: Optional[float]
    status: str  # "fixed"
    note: str

    def to_dict(self) -> dict:
        return {
            "um_per_native_px": self.um_per_native_px,
            "status": self.status,
            "note": self.note,
        }


def resolve_calibration(scale: ScaleInfo) -> CalibrationResult:
    return CalibrationResult(
        um_per_native_px=OPTIVIEW_UM_PER_NATIVE_PX,
        status="fixed",
        note="공통 평가 보정값: 1 native px = 0.1592 µm.",
    )


def px_to_um(px_native: float, calib: CalibrationResult) -> Optional[float]:
    if calib.um_per_native_px is None:
        return None
    return px_native * calib.um_per_native_px


# ---------------------------------------------------------------------------
# 7. 측정 결과 오버레이 렌더링
# ---------------------------------------------------------------------------

# 표시는 항상 native(원본 추정 해상도) 좌표를 캡처 좌표로 역변환해서 그린다.


