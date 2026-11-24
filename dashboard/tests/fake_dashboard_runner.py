"""TEST-ONLY stand-in for inspection.run_dashboard_analysis (no model is loaded).

Real: run_groups validation, BatchRunner, AnalysisRunner policy/validation, MetrologyAdapter
checks and the team judge. Fake: the RF-DETR and U-Net engines and overlay drawing.
Behaviour by Die code: 13 -> model PD detection, 14 -> defect engine failure, else 0 defects.
FAKE_RUNNER_MODE=crash exits before running; FAKE_RUNNER_CALLS counts launches.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import inspection.analysis_runner as analysis_runner  # noqa: E402
from inspection.analysis_runner import AnalysisRunner  # noqa: E402
from inspection.batch_runner import BatchRunner  # noqa: E402
from inspection.metrology_adapter import IDENTIFIERS, MetrologyAdapter  # noqa: E402
from inspection.run_dashboard_analysis import run_groups, validate  # noqa: E402


def _overlay(kind):
    def save(request, result, path):
        Path(path).write_bytes(b"FAKE-OVERLAY-" + kind.encode())
        return {"path": str(path), "base_image": "ORIGINAL_IMAGE", "fake": True}
    return save


analysis_runner._save_defect_overlay = _overlay("defect")
analysis_runner._save_metrology_overlay = _overlay("metrology")


class FakeDefect:
    def predict(self, request):
        die = request.get("die_id", "")[-2:]
        if die == "14":
            raise RuntimeError("FAKE defect engine failure")
        detections = ([{"class_code": "pd", "confidence": 0.9, "bbox_xyxy": [10.0, 10.0, 50.0, 500.0],
                        "source": "MODEL_POSTPROCESSED"}] if die == "13" else [])
        return {"image_id": request["image_id"], **{k: request.get(k) for k in IDENTIFIERS},
                "status": "SUCCESS", "model_version": "FAKE-DEFECT", "model_sha256": "0" * 64,
                "bbox_coordinate_system": "ORIGINAL_IMAGE", "bbox_image_width": 2000,
                "bbox_image_height": 1000, "skew_angle": 0.0, "detections": detections}


class FakeUnet:
    algorithm_version = "FAKE-UNET"

    def analyze(self, request):
        measurements = []
        for kind, ids, start, width in (("line", ["L01", "L02"], 100, 61), ("gap", ["G01"], 170, 41)):
            for index, line_id in enumerate(ids):
                left = float(start + index * 200)
                for y in (0.25, 0.45, 0.65):
                    measurements.append({"measurement_type": kind, "line_id": line_id, "y_fraction": y,
                        "y_px": int(y * 1000), "width_px": float(width), "width_um": round(width * 0.1592, 4),
                        "left_x_px": left, "right_x_px": left + width, "center_x_px": left + width / 2})
        return {"image_id": request["image_id"], **{k: request.get(k) for k in IDENTIFIERS},
                "status": "success", "algorithm_version": self.algorithm_version,
                "calibration_id": (request.get("calibration") or {}).get("calibration_id"),
                "measurements": measurements, "error_code": None,
                "pixel_scale": {"coordinate_system": "native_px", "factor_x": 1.0, "factor_y": 1.0,
                                "native_width": 2000, "native_height": 1000, "offset_x": 0.0, "offset_y": 0.0}}


def main():
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    calls = os.environ.get("FAKE_RUNNER_CALLS")
    if calls:
        with open(calls, "a", encoding="utf-8") as stream:
            stream.write("x")
    if os.environ.get("FAKE_RUNNER_MODE") == "crash":
        sys.exit(3)
    if os.environ.get("FAKE_RUNNER_DELAY"):  # Lets tests observe live progress polling.
        import time
        time.sleep(float(os.environ["FAKE_RUNNER_DELAY"]))
    payload = json.loads(Path(args["--request"]).read_text(encoding="utf-8"))
    validate(payload)
    runner = BatchRunner(AnalysisRunner(FakeDefect(), MetrologyAdapter(engine=FakeUnet())))
    print(json.dumps({"groups": run_groups(payload, args["--output-root"], runner)}))


if __name__ == "__main__":
    main()
