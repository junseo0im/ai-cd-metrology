"""대시보드 통합용 Metrology 모듈 (UNet 선폭 측정, 공통 입출력 형식).

측정 알고리즘은 methods/unet 의 measure_unet_image 를 그대로 쓰고,
입력 요청과 출력 결과만 팀 공통 형식으로 맞춘다.

입력 (dict)
  image_id, image_path                         필수
  wafer_id, die_id, site_code, batch_id,
  analysis_run_id                              선택 (그대로 출력에 실어 보냄)
  roi          선택. {"x_min", "y_min", "x_max", "y_max"} native px.
               없으면 사진 전체. Y 25/45/65% 는 ROI 높이 기준.
  calibration  선택. {"calibration_id": str, "um_per_px": float}  (µm / native px)
               없으면 기본 0.1592 µm/px.

출력 (dict)
  image_id, status("success"|"failed"), algorithm_version, calibration_id,
  measurements[], error_code, + 입력 메타데이터, pixel_scale
  measurements 각 항목:
    y_fraction, y_px, line_id, measurement_type("line"|"gap"),
    width_px, width_um, left_x_px, right_x_px, center_x_px,
    left_sidewall_px, right_sidewall_px

좌표: 모든 *_px 는 native px (2배 중복 저장 픽셀을 걷어낸 좌표).
      원본 사진 위에 그릴 때: x_원본 = x * factor_x + offset_x + (factor_x - 1) / 2
      (factor/offset 은 출력의 pixel_scale).
측정 범위: ROI(없으면 사진) 양옆 3% 를 뺀 영역에서 검출된 모든 line/gap.
           빨간 띠 2개 사이에 온전히 들어온 구간만 측정하므로 테두리에서 잘린 구간은 없다.
판정(pn/pt/pd, PD 제외, 평균)은 이 모듈이 하지 않는다 → 통합 Runner 담당.

사용:
  from metrology.unet_metrology import MetrologyEngine
  engine = MetrologyEngine()                 # 모델은 한 번만 로드
  result = engine.analyze({"image_id": "...", "image_path": "..."})

  python metrology/unet_metrology.py requests.json --output results.json
  (requests.json: 요청 dict 하나 또는 목록)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np  # Import before torch in this Anaconda environment.

PROJECT_DIR = Path(__file__).resolve().parents[1]
UNET_DIR = PROJECT_DIR / "methods" / "unet"
for _path in (UNET_DIR, PROJECT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from team_protocol import Y_FRACTIONS  # noqa: E402

MEASUREMENT_TYPES = ("line", "gap")
SIDE_MARGIN_RATIO = 0.03
DEFAULT_CALIBRATION = {"calibration_id": "optiview-native-0.1592", "um_per_px": 0.1592}
PREFIX = {"line": "L", "gap": "G"}
META_FIELDS = ("wafer_id", "die_id", "site_code", "batch_id", "analysis_run_id")

# error_code 목록
E_BAD_REQUEST = "BAD_REQUEST"                # 필수 입력 누락·형식 오류
E_IMAGE_NOT_FOUND = "IMAGE_NOT_FOUND"        # image_path 파일 없음
E_IMAGE_READ = "IMAGE_READ_FAILED"           # 이미지 열기 실패
E_INVALID_ROI = "INVALID_ROI"                # ROI 가 사진 밖이거나 너무 작음
E_NO_SIDEWALL = "NO_SIDEWALL_DETECTED"       # 측정 줄에서 빨간 띠를 2개 이상 못 찾음
E_NO_MEASUREMENT = "NO_MEASUREMENT"          # 빨간 띠는 있으나 ROI 안에 측정 구간 없음
E_INTERNAL = "INTERNAL_ERROR"                # 그 밖의 예외


class MetrologyError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _assign_line_ids(rows: list[list[dict]], kind: str) -> None:
    """Give the same line_id to the same physical line/gap across Y rows (left to right)."""
    items = [item for row in rows for item in row if item["measurement_type"] == kind]
    if not items:
        return
    anchor_row = max(rows, key=lambda row: sum(i["measurement_type"] == kind for i in row))
    anchors = sorted(i["center_x_px"] for i in anchor_row if i["measurement_type"] == kind)
    pitch = float(np.median(np.diff(anchors))) if len(anchors) > 1 else 60.0
    tolerance = 0.35 * pitch
    clusters = [[x] for x in anchors]
    for item in items:
        x = item["center_x_px"]
        best = min(range(len(clusters)), key=lambda k: abs(np.mean(clusters[k]) - x), default=None)
        if best is not None and abs(np.mean(clusters[best]) - x) <= tolerance:
            item["_cluster"] = best
        else:
            clusters.append([x])
            item["_cluster"] = len(clusters) - 1
    order = sorted(range(len(clusters)), key=lambda k: np.mean(clusters[k]))
    rank = {cluster: number for number, cluster in enumerate(order, start=1)}
    for item in items:
        item["line_id"] = f"{PREFIX[kind]}{rank[item.pop('_cluster')]:02d}"


class MetrologyEngine:
    def __init__(self, checkpoint: Path | None = None, device: str = "auto", threshold: float = 0.50):
        import torch
        import unet_inference as ui

        self._ui = ui
        selected = None if device == "auto" else torch.device(device)
        path = Path(checkpoint) if checkpoint else ui.DEFAULT_CHECKPOINT
        self.model, self.checkpoint, self.device = ui.load_unet(path, device=selected)
        self.threshold = float(threshold)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        self.algorithm_version = (
            f"unet-sidewall-v1;ckpt={path.name}@{digest};epoch={self.checkpoint.get('epoch', '?')};"
            f"prob_threshold={self.threshold:.2f};edge=inner;y={','.join(f'{f:.2f}' for f in Y_FRACTIONS)}"
        )

    # ------------------------------------------------------------------
    def analyze(self, request: dict) -> dict:
        calibration = {**DEFAULT_CALIBRATION, **(request.get("calibration") or {})}
        result = {
            "image_id": request.get("image_id"),
            **{field: request.get(field) for field in META_FIELDS},
            "status": "failed",
            "algorithm_version": self.algorithm_version,
            "calibration_id": calibration.get("calibration_id"),
            "measurements": [],
            "error_code": None,
            "error_message": None,
            "pixel_scale": None,
        }
        try:
            result.update(self._measure(request, calibration))
            result["status"] = "success"
        except MetrologyError as error:
            result["error_code"], result["error_message"] = error.code, str(error)
        except Exception as error:  # noqa: BLE001 - one bad image must not stop the batch
            result["error_code"], result["error_message"] = E_INTERNAL, f"{type(error).__name__}: {error}"
        return result

    # ------------------------------------------------------------------
    def _measure(self, request: dict, calibration: dict) -> dict:
        ui = self._ui
        if not request.get("image_id") or not request.get("image_path"):
            raise MetrologyError(E_BAD_REQUEST, "image_id and image_path are required")
        try:
            um_per_px = float(calibration["um_per_px"])
            if um_per_px <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise MetrologyError(E_BAD_REQUEST, "calibration.um_per_px must be a positive number") from None
        path = Path(request["image_path"])
        if not path.is_file():
            raise MetrologyError(E_IMAGE_NOT_FOUND, f"image not found: {path}")
        try:
            loaded = ui.load_image(str(path))
        except Exception as error:  # noqa: BLE001
            raise MetrologyError(E_IMAGE_READ, f"cannot read image: {error}") from error

        native_h, native_w = loaded.native_arr.shape[:2]
        roi = request.get("roi") or {"x_min": 0, "y_min": 0, "x_max": native_w - 1, "y_max": native_h - 1}
        try:
            x_min, y_min = int(roi["x_min"]), int(roi["y_min"])
            x_max, y_max = int(roi["x_max"]), int(roi["y_max"])
        except (KeyError, TypeError, ValueError):
            raise MetrologyError(E_INVALID_ROI, "roi needs integer x_min, y_min, x_max, y_max") from None
        if not (0 <= x_min < x_max < native_w and 0 <= y_min < y_max < native_h) \
                or x_max - x_min < 50 or y_max - y_min < 20:
            raise MetrologyError(E_INVALID_ROI, f"roi outside image {native_w}x{native_h} or too small: {roi}")
        rows_px = [y_min + int(round((y_max - y_min) * f)) for f in Y_FRACTIONS]
        margin = SIDE_MARGIN_RATIO * (x_max - x_min)
        keep_from, keep_to = x_min + margin, x_max - margin

        measured = ui.measure_unet_image(
            path, self.model, self.checkpoint, self.device,
            y_rows=rows_px, threshold=self.threshold,
            side_margin_ratio=SIDE_MARGIN_RATIO, measurement_types=list(MEASUREMENT_TYPES),
            loaded=loaded,  # the image was already read above; do not decode it twice
        )
        report_rows = measured["report"]["rows"]
        per_row, sidewalls_seen = [], 0
        for fraction, row in zip(Y_FRACTIONS, report_rows):
            lines = sorted(row["lines"], key=lambda item: item["center_x_native"])
            sidewalls_seen = max(sidewalls_seen, len(lines))
            items = []
            for band in row["bands"]:
                kind = band.get("measurement_type")
                if kind not in MEASUREMENT_TYPES:
                    continue
                left, right = float(band["left_x_native"]), float(band["right_x_native"])
                if left < keep_from or right > keep_to or right <= left:
                    continue
                li, ri = band["between_line_index"]
                width_px = right - left
                items.append({
                    "y_fraction": float(fraction),
                    "y_px": int(row["y_native"]),
                    "line_id": None,
                    "measurement_type": kind,
                    "width_px": round(width_px, 3),
                    "width_um": round(width_px * um_per_px, 4),
                    "left_x_px": round(left, 3),
                    "right_x_px": round(right, 3),
                    "center_x_px": round((left + right) / 2.0, 3),
                    "left_sidewall_px": lines[li]["width_px_native"] if li < len(lines) else None,
                    "right_sidewall_px": lines[ri]["width_px_native"] if ri < len(lines) else None,
                })
            per_row.append(items)
        if sidewalls_seen < 2:
            raise MetrologyError(E_NO_SIDEWALL, "fewer than two sidewalls detected on every measurement row")
        if not any(per_row):
            raise MetrologyError(E_NO_MEASUREMENT, "no complete line/gap interval inside the ROI")
        for kind in MEASUREMENT_TYPES:
            _assign_line_ids(per_row, kind)
        measurements = sorted((item for row in per_row for item in row),
                              key=lambda item: (item["y_fraction"], item["center_x_px"]))
        scale = loaded.scale
        return {
            "measurements": measurements,
            "pixel_scale": {"coordinate_system": "native_px", "factor_x": scale.factor_x,
                            "factor_y": scale.factor_y, "offset_x": scale.offset_x,
                            "offset_y": scale.offset_y, "native_width": native_w, "native_height": native_h},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("requests", type=Path, help="요청 JSON (dict 하나 또는 목록)")
    parser.add_argument("--output", type=Path, required=True, help="결과 JSON 경로")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    payload = json.loads(args.requests.read_text(encoding="utf-8"))
    requests = payload if isinstance(payload, list) else [payload]
    engine = MetrologyEngine(device=args.device)
    results = []
    for number, request in enumerate(requests, start=1):
        result = engine.analyze(request)
        results.append(result)
        print(f"[{number}/{len(requests)}] {result['image_id']}: {result['status']} "
              f"{len(result['measurements'])} measurements {result['error_code'] or ''}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results if isinstance(payload, list) else results[0],
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", args.output)


if __name__ == "__main__":
    main()
