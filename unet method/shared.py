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
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from team_protocol import (
    Y_FRACTIONS,
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

# 세 측정 프로그램과 성능 비교에서 반드시 공유하는 기준이다. 측정은 캡처
# 좌표가 아니라 중복 픽셀을 제거한 native_arr 높이의 20%, 35%, 50%, 65%,
# 80%에서 수행한다.
DEFAULT_MEASUREMENT_ROW_FRACTIONS = Y_FRACTIONS
MEASUREMENT_TYPE_BY_NUMBER = {1: "line", 2: "gap", 3: "sidewall"}
MEASUREMENT_TYPE_LABELS = {
    "line": "line (bright pattern)",
    "gap": "gap (space between patterns)",
    "sidewall": "sidewall (orange boundary thickness)",
}


def normalize_measurement_type(value: int | str) -> str:
    """Return the canonical measurement type used by every entry point."""
    if str(value).strip().lower() == "wall":
        value = "sidewall"
    return protocol_measurement_type(value)


def normalize_measurement_types(value) -> list[str]:
    """Parse one or more selections such as 1, ``1 2 3``, ``1,3``, or ``123``."""
    return parse_measurement_types(value)


def default_measurement_rows(height: int) -> list[int]:
    """Common native-y positions: 20%, 35%, 50%, 65%, and 80% of height."""
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

    def h_pair_frac(offset: int) -> float:
        a = arr[:, offset : w - 1 : 2]
        b = arr[:, offset + 1 : w : 2]
        n = min(a.shape[1], b.shape[1])
        if n == 0:
            return 0.0
        return float(np.mean(np.all(a[:, :n] == b[:, :n], axis=-1)))

    def v_pair_frac(offset: int) -> float:
        a = arr[offset : h - 1 : 2, :]
        b = arr[offset + 1 : h : 2, :]
        n = min(a.shape[0], b.shape[0])
        if n == 0:
            return 0.0
        return float(np.mean(np.all(a[:n] == b[:n], axis=-1)))

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


def native_x_to_capture(x_native: float, scale: ScaleInfo) -> float:
    return x_native * scale.factor_x + scale.offset_x + (scale.factor_x - 1) / 2.0


def native_y_to_capture(y_native: float, scale: ScaleInfo) -> float:
    return y_native * scale.factor_y + scale.offset_y + (scale.factor_y - 1) / 2.0


def capture_x_to_native(x_capture: float, scale: ScaleInfo) -> float:
    return (x_capture - scale.offset_x) / scale.factor_x


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


def _parabolic_subpixel(f: np.ndarray, i: int) -> float:
    """이산 극값 위치 i 주변 3점 포물선 피팅으로 서브픽셀 위치를 보정한다."""
    if i <= 0 or i >= len(f) - 1:
        return float(i)
    fm1, f0, fp1 = f[i - 1], f[i], f[i + 1]
    denom = fm1 - 2 * f0 + fp1
    if denom == 0:
        return float(i)
    delta = 0.5 * (fm1 - fp1) / denom
    delta = max(-1.0, min(1.0, delta))
    return i + delta


def detect_dark_lines(
    profile: np.ndarray,
    smooth_sigma: float = 0.8,
    min_prominence: float = 12.0,
    max_width: int = 25,
) -> list[LineDetection]:
    """
    프로파일에서 가느다란 붉은 경계선에 해당하는 골(valley)을 찾고, 좌우 경계를
    1차 미분의 극값(가장 가파른 기울기 지점=변곡점)에서 서브픽셀로 구한다.

    Nyyssonen & Larrabee(1987)와 Steger(2013)가 제시한 '최대 기울기 지점을 경계로
    본다'는 접근을 1차원 프로파일에 적용했다. max_width로 폭을 제한해, 완만하게
    이어지는 넓은 line/gap 밴드 자체의 밝기 굴곡을 골로 오검출하지 않도록 한다.
    """
    smoothed = gaussian_filter1d(profile, sigma=smooth_sigma, mode="nearest")
    deriv = np.gradient(smoothed)

    inv = -smoothed
    valley_idx, props = find_peaks(inv, prominence=min_prominence, width=(None, max_width))
    prominences = props["prominences"]

    results: list[LineDetection] = []
    for vi, prom in zip(valley_idx, prominences):
        left_start = max(0, vi - 15)
        left_region = deriv[left_start : vi + 1]
        if len(left_region) == 0:
            continue
        li = left_start + int(np.argmin(left_region))
        left_edge = _parabolic_subpixel(deriv, li)

        right_end = min(len(deriv), vi + 16)
        right_region = deriv[vi:right_end]
        if len(right_region) == 0:
            continue
        ri = vi + int(np.argmax(right_region))
        right_edge = _parabolic_subpixel(deriv, ri)

        if right_edge <= left_edge:
            continue

        results.append(
            LineDetection(
                center_x=float(vi),
                left_edge_x=left_edge,
                right_edge_x=right_edge,
                prominence=float(prom),
            )
        )
    return results


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


def summarize_measurement_widths(
    rows: list[dict],
    max_deviation_um: float,
    min_coverage: float = 1.0,
    center_tolerance_px: float | None = None,
) -> list[dict]:
    """Match the selected physical features across rows and calculate statistics.

    max_deviation_um is compared with the population standard deviation.
    A line is usable only when it is present in enough rows and has calibrated
    micrometre values.
    """
    if max_deviation_um < 0:
        raise ValueError("max_deviation_um must be non-negative")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    if not rows:
        return []

    prepared_rows = [
        sorted(row.get("measurements", []), key=lambda item: float(item["center_x_native"]))
        for row in rows
    ]
    if not any(prepared_rows):
        return []

    anchor_index = max(range(len(prepared_rows)), key=lambda index: len(prepared_rows[index]))
    anchor_lines = prepared_rows[anchor_index]
    anchor_centers = np.array([float(line["center_x_native"]) for line in anchor_lines])
    if center_tolerance_px is None:
        pitches = np.diff(anchor_centers)
        center_tolerance_px = (
            max(3.0, float(np.median(pitches)) * 0.35) if len(pitches) else 20.0
        )
    if center_tolerance_px <= 0:
        raise ValueError("center_tolerance_px must be positive")

    def sample(line: dict, row_index: int) -> dict:
        return {
            "row_index": row_index,
            "center_x_native": float(line["center_x_native"]),
            "width_px_native": float(line["width_px_native"]),
            "width_um_provisional": (
                float(line["width_um_provisional"])
                if line.get("width_um_provisional") is not None
                else None
            ),
        }

    tracks = [
        {
            "samples": [sample(line, anchor_index)],
        }
        for line in anchor_lines
    ]

    for row_index, lines in enumerate(prepared_rows):
        if row_index == anchor_index or not lines:
            continue
        candidates = []
        for track_index, track in enumerate(tracks):
            reference = float(np.mean([item["center_x_native"] for item in track["samples"]]))
            for line_index, line in enumerate(lines):
                distance = abs(reference - float(line["center_x_native"]))
                if distance <= center_tolerance_px:
                    candidates.append((distance, track_index, line_index))

        used_tracks: set[int] = set()
        used_lines: set[int] = set()
        for _, track_index, line_index in sorted(candidates):
            if track_index in used_tracks or line_index in used_lines:
                continue
            tracks[track_index]["samples"].append(sample(lines[line_index], row_index))
            used_tracks.add(track_index)
            used_lines.add(line_index)

    expected_samples = len(rows)
    minimum_samples = max(1, int(np.ceil(expected_samples * min_coverage)))
    ordered_tracks = sorted(
        tracks,
        key=lambda track: float(np.mean([item["center_x_native"] for item in track["samples"]])),
    )

    summaries = []
    for measurement_index, track in enumerate(ordered_tracks, start=1):
        samples = sorted(track["samples"], key=lambda item: item["row_index"])
        widths_px = [item["width_px_native"] for item in samples]
        widths_um = [
            item["width_um_provisional"]
            for item in samples
            if item["width_um_provisional"] is not None
        ]
        sample_count = len(samples)
        calibrated_count = len(widths_um)
        mean_px = float(np.mean(widths_px))
        std_px = float(np.std(widths_px))
        mean_um = float(np.mean(widths_um)) if widths_um else None
        std_um = float(np.std(widths_um)) if widths_um else None

        reasons = []
        if sample_count < minimum_samples:
            reasons.append(f"coverage {sample_count}/{expected_samples} below {minimum_samples}")
        if calibrated_count != sample_count:
            reasons.append("missing micrometre calibration")
        if std_um is not None and std_um > max_deviation_um:
            reasons.append(
                f"std {std_um:.4f} um exceeds {max_deviation_um:.4f} um"
            )

        summaries.append(
            {
                "measurement_index": measurement_index,
                "center_x_native": round(
                    float(np.mean([item["center_x_native"] for item in samples])), 4
                ),
                "sample_count": sample_count,
                "expected_sample_count": expected_samples,
                "coverage_ratio": round(sample_count / expected_samples, 4),
                "mean_width_px_native": round(mean_px, 4),
                "std_width_px_native": round(std_px, 4),
                "mean_width_um_provisional": round(mean_um, 4) if mean_um is not None else None,
                "std_width_um_provisional": round(std_um, 4) if std_um is not None else None,
                "max_deviation_um": round(float(max_deviation_um), 4),
                "usable": not reasons,
                "rejection_reason": "; ".join(reasons) if reasons else None,
                "measurements": [
                    {
                        **item,
                        "center_x_native": round(item["center_x_native"], 4),
                        "width_px_native": round(item["width_px_native"], 4),
                        "width_um_provisional": (
                            round(item["width_um_provisional"], 4)
                            if item["width_um_provisional"] is not None
                            else None
                        ),
                    }
                    for item in samples
                ],
            }
        )
    return summaries


def summarize_line_widths(
    rows: list[dict],
    max_deviation_um: float,
    min_coverage: float = 1.0,
    center_tolerance_px: float | None = None,
) -> list[dict]:
    """Backward-compatible wrapper; new code should use summarize_measurement_widths."""
    normalized_rows = []
    for row in rows:
        if "measurements" in row:
            normalized_rows.append(row)
        else:
            normalized_rows.append({**row, "measurements": row.get("lines", [])})
    return summarize_measurement_widths(
        normalized_rows,
        max_deviation_um=max_deviation_um,
        min_coverage=min_coverage,
        center_tolerance_px=center_tolerance_px,
    )


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

COLOR_GUIDE = (255, 255, 255)
MEASUREMENT_COLORS = {
    "line": (255, 55, 75),       # 빨강/분홍
    "gap": (45, 105, 255),       # 파랑
    "sidewall": (0, 235, 255),   # 하늘색
}
MEASUREMENT_PREFIXES = {"line": "L", "gap": "G", "sidewall": "S"}


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:\\Windows\\Fonts\\malgun.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _text_centered(
    draw: ImageDraw.ImageDraw, cx: float, y: float, text: str, color, font: ImageFont.ImageFont
) -> None:
    try:
        tw = draw.textlength(text, font=font)
    except Exception:
        tw = len(text) * 6
    draw.text((cx - tw / 2.0, y), text, fill=color, font=font)


def _draw_dashed_horizontal(
    draw: ImageDraw.ImageDraw,
    y: float,
    width: int,
    dash: int = 12,
    gap: int = 8,
) -> None:
    x = 0
    while x < width:
        draw.line(
            [(x, y), (min(width - 1, x + dash), y)],
            fill=COLOR_GUIDE,
            width=2,
        )
        x += dash + gap


def render_measurement_overlay(
    capture_arr,
    scale: ScaleInfo,
    rows: list[dict],
    measurement_types,
    calib: Optional[CalibrationResult] = None,
    title_note: str = "",
) -> Image.Image:
    """Shared overlay used by the U-Net and batch measurement programs.

    White dashed lines show common y positions. Red=line, blue=gap, cyan=sidewall.
    Multiple selected types are vertically offset around the same measurement row.
    """
    kinds = normalize_measurement_types(measurement_types)
    source = Image.fromarray(capture_arr).convert("RGB")
    top_margin = 42
    canvas = Image.new("RGB", (source.width, source.height + top_margin), (255, 255, 255))
    canvas.paste(source, (0, top_margin))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(16)
    number_font = _load_font(15)
    value_font = _load_font(11)
    unit = "µm" if calib is not None and calib.um_per_native_px is not None else "native px"
    row_positions = ",".join(str(row.get("y_native", "?")) for row in rows)
    title = (
        f"1=LINE (red) | 2=GAP (blue) | 3=SIDEWALL (cyan) | "
        f"rows(native y)={row_positions} | labels in {unit}"
    )
    if title_note:
        title += f" | {title_note}"
    _text_centered(draw, source.width / 2.0, 10, title, (0, 0, 0), title_font)

    if rows:
        base_number_y = top_margin + max(24, round(source.height * 0.025))
        groups = rows[0].get("measurements_by_type", {})
        if not groups and len(kinds) == 1:
            groups = {kinds[0]: rows[0].get("measurements", [])}
        for kind_index, kind in enumerate(kinds):
            color = MEASUREMENT_COLORS[kind]
            number_y = base_number_y + kind_index * 20
            for item in groups.get(kind, []):
                center = native_x_to_capture(float(item["center_x_native"]), scale)
                text = f"{MEASUREMENT_PREFIXES[kind]}{item['measurement_index']}"
                try:
                    box = draw.textbbox((0, 0), text, font=number_font)
                    tw = box[2] - box[0]
                    th = box[3] - box[1]
                except Exception:
                    tw, th = len(text) * 8, 15
                x0 = center - tw / 2.0 - 4
                draw.rectangle(
                    (x0, number_y, x0 + tw + 8, number_y + th + 4),
                    fill=(255, 245, 240),
                    outline=color,
                )
                _text_centered(draw, center, number_y + 1, text, color, number_font)

    for row in rows:
        y_capture = native_y_to_capture(float(row["y_native"]), scale) + top_margin
        _draw_dashed_horizontal(draw, y_capture, source.width)
        groups = row.get("measurements_by_type", {})
        if not groups and len(kinds) == 1:
            groups = {kinds[0]: row.get("measurements", [])}
        offsets = {"line": -28, "sidewall": 0, "gap": 28} if len(kinds) > 1 else {kinds[0]: 0}
        for kind in kinds:
            color = MEASUREMENT_COLORS[kind]
            y_measure = y_capture + offsets.get(kind, 0)
            for item in groups.get(kind, []):
                left = native_x_to_capture(float(item["left_x_native"]), scale)
                right = native_x_to_capture(float(item["right_x_native"]), scale)
                center = (left + right) / 2.0
                tick = 7
                draw.line([(left, y_measure), (right, y_measure)], fill=color, width=3)
                draw.line([(left, y_measure - tick), (left, y_measure + tick)], fill=color, width=2)
                draw.line([(right, y_measure - tick), (right, y_measure + tick)], fill=color, width=2)
                value_px = float(item["width_px_native"])
                value_um = item.get("width_um_provisional")
                label = (
                    f"{value_px:.2f}px / {float(value_um):.3f}µm"
                    if value_um is not None
                    else f"{value_px:.2f}px"
                )
                label_y = y_measure - 20 if kind == "line" else y_measure + 9
                _text_centered(draw, center, label_y, label, color, value_font)
    return canvas
