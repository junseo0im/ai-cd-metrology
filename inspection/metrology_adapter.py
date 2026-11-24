"""Thin adapter for the unchanged team U-Net and judgment code."""
from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys
import time

from contracts.analysis_contract import Y_FRACTIONS

ROOT = Path(__file__).resolve().parents[1]
TEAM_UNET = ROOT / "runtime_data/external_incoming/unet_linewidth_metrology"
IDENTIFIERS = ("wafer_id", "die_id", "site_code", "batch_id", "analysis_run_id")


def json_value(value):
    """Convert numeric scalars, but reject non-finite/unsupported result values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite result value")
        return number
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("result keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    raise TypeError(f"unsupported result type: {type(value).__name__}")


def checked_module(module, expected):
    if Path(module.__file__).resolve() != expected.resolve():
        raise RuntimeError(f"unexpected module location: {module.__name__}")
    return module


def measurement_quality(items):
    """Coverage of returned IDs only; not proof that all physical lines were found."""
    expected = set(Y_FRACTIONS)
    coverage = {}
    duplicates = []
    for item in items:
        key = (item["measurement_type"], item["line_id"])
        seen = coverage.setdefault(key, set())
        fraction = item["y_fraction"]
        if fraction in seen:
            duplicates.append({"measurement_type": key[0], "line_id": key[1],
                               "y_fraction": fraction})
        seen.add(fraction)
    missing = [{"measurement_type": kind, "line_id": line_id,
                "missing_y_fractions": sorted(expected - seen)}
               for (kind, line_id), seen in sorted(coverage.items()) if expected - seen]
    absent = [kind for kind in ("line", "gap")
              if not any(key[0] == kind for key in coverage)]
    partial = bool(missing or absent or duplicates)
    return {"partial_measurement": partial, "missing_positions": missing,
            "absent_measurement_types": absent, "duplicate_positions": duplicates,
            "expected_y_fractions": list(Y_FRACTIONS),
            "coverage_scope": "returned line/gap IDs; entirely undetected physical lines unknown"}


class MetrologyAdapter:
    """measure(request), reusing one lazily loaded team engine per instance."""
    def __init__(self, engine=None, judge_fn=None, device="cpu"):
        self._engine = engine
        self._judge = judge_fn
        self.device = device
        self._load_seconds = 0.0

    def _load(self):
        if str(TEAM_UNET) not in sys.path:
            sys.path.insert(0, str(TEAM_UNET))
        if self._judge is None:
            import judgment.judge as judgment_module
            checked_module(judgment_module, TEAM_UNET / "judgment/judge.py")
            self._judge = judgment_module.judge
        if self._engine is None:
            started = time.perf_counter()
            import metrology.unet_metrology as module
            checked_module(module, TEAM_UNET / "metrology/unet_metrology.py")
            self._engine = module.MetrologyEngine(device=self.device)
            self._load_seconds = time.perf_counter() - started

    def measure(self, request):
        started = time.perf_counter()
        image_id = request.get("image_id") if isinstance(request, dict) else None
        identifiers = {key: request.get(key) for key in IDENTIFIERS} if isinstance(request, dict) else {}
        result = {"image_id": image_id, **identifiers, "status": "FAILED",
                  "algorithm_version": None, "calibration_id": None, "measurements": [],
                  "error_code": None, "error_message": None}
        reused = self._engine is not None
        try:
            if not isinstance(request, dict) or not isinstance(image_id, str) or not image_id:
                raise ValueError("image_id is required")
            calibration = request.get("calibration")
            # Preserve the existing native-pixel coefficient, never infer a new one.
            if calibration is None:
                if request.get("magnification") not in (None, 20, 20.0):
                    result["error_code"] = "CALIBRATION_MAGNIFICATION_MISMATCH"
                    raise ValueError("team default calibration is verified only for supplied 20x images")
            elif (not isinstance(calibration, dict)
                  or not isinstance(calibration.get("calibration_id"), str)
                  or not calibration["calibration_id"]
                  or isinstance(calibration.get("um_per_px"), bool)
                  or not isinstance(calibration.get("um_per_px"), Real)
                  or not math.isfinite(calibration["um_per_px"])
                  or calibration["um_per_px"] <= 0):
                result["error_code"] = "INVALID_CALIBRATION"
                raise ValueError("explicit calibration needs ID and finite positive um_per_px")
            elif (calibration.get("magnification") is not None
                  and request.get("magnification") != calibration["magnification"]):
                result["error_code"] = "CALIBRATION_MAGNIFICATION_MISMATCH"
                raise ValueError("calibration magnification does not match image metadata")
            self._load()
            native_started = time.perf_counter()
            native = json_value(self._engine.analyze(request))
            measure_seconds = time.perf_counter() - native_started
            if native.get("image_id") != image_id:
                raise ValueError("metrology image_id mismatch")
            for key in IDENTIFIERS:
                if native.get(key) != request.get(key):
                    raise ValueError(f"metrology {key} mismatch")
            if native.get("status") not in ("success", "failed"):
                raise ValueError("unsupported team status")
            result.update(native)
            if native["status"] == "success":
                items = native.get("measurements")
                if not isinstance(items, list) or not items:
                    raise ValueError("successful metrology must have measurements")
                if not isinstance(native.get("algorithm_version"), str) or not native["algorithm_version"]:
                    raise ValueError("algorithm version missing")
                if not isinstance(native.get("calibration_id"), str) or not native["calibration_id"]:
                    raise ValueError("applied calibration ID missing")
                scale = native.get("pixel_scale") or {}
                if scale.get("coordinate_system") != "native_px":
                    raise ValueError("unknown measurement coordinate system")
                for field in ("factor_x", "factor_y", "native_width", "native_height"):
                    if not isinstance(scale.get(field), Real) or scale[field] <= 0:
                        raise ValueError(f"invalid pixel_scale.{field}")
                for field in ("offset_x", "offset_y"):
                    if not isinstance(scale.get(field), Real):
                        raise ValueError(f"invalid pixel_scale.{field}")
                for item in items:
                    if (item.get("measurement_type") not in ("line", "gap")
                        or not isinstance(item.get("line_id"), str) or not item["line_id"]
                        or item.get("y_fraction") not in Y_FRACTIONS):
                        raise ValueError("invalid measurement identity/Y position")
                    for field in ("width_px", "width_um"):
                        if (isinstance(item.get(field), bool)
                            or not isinstance(item.get(field), Real) or item[field] <= 0):
                            raise ValueError(f"invalid measurement {field}")
                    for field in ("y_px", "left_x_px", "right_x_px", "center_x_px"):
                        if not isinstance(item.get(field), Real):
                            raise ValueError(f"invalid measurement coordinate {field}")
                    if not (0 <= item["left_x_px"] < item["right_x_px"] < scale["native_width"]
                            and 0 <= item["y_px"] < scale["native_height"]
                            and item["left_x_px"] <= item["center_x_px"] <= item["right_x_px"]):
                        raise ValueError("measurement coordinates outside native image")
                # The team judge requires lowercase status. Do not normalize first.
                judged = json_value(self._judge(native))
                if (any(item["measurement_type"] == "line" for item in items)
                    and judged.get("judgment", {}).get("status") != "judged"):
                    raise ValueError("successful line measurements were not judged")
                result.update(judged)
                result["quality"] = measurement_quality(items)
                result["status"] = "SUCCESS"
            else:
                if native.get("measurements"):
                    raise ValueError("failed engine returned measurements")
                result["status"] = "FAILED"
                result["error_code"] = native.get("error_code") or "METROLOGY_FAILED"
            # Import original constant rather than reproducing its coefficient.
            if calibration is None:
                from metrology.unet_metrology import DEFAULT_CALIBRATION
                applied = dict(DEFAULT_CALIBRATION)
            else:
                applied = dict(calibration)
            result["calibration"] = {**applied, "unit": "um/native_px",
                                     "applied_by": "TEAM_UNET_ENGINE",
                                     "source": "request" if calibration is not None else "team_default_20x",
                                     "physical_calibration_revalidated": False}
            result["coordinate_space"] = "NATIVE_IMAGE"
            result["units"] = {"width_px": "native_px", "width_um": "um"}
            result["timing"] = {"model_load_seconds": 0.0 if reused else self._load_seconds,
                                "model_session_reused": reused,
                                "analyze_seconds": measure_seconds}
            return json_value(result)
        except Exception as error:
            return {"image_id": image_id, **identifiers, "status": "FAILED",
                    "algorithm_version": getattr(self._engine, "algorithm_version", None),
                    "calibration_id": None, "measurements": [],
                    "error_code": result.get("error_code") or "METROLOGY_ADAPTER_ERROR",
                    "error_message": f"{type(error).__name__}: {error}",
                    "timing": {"total_seconds": time.perf_counter() - started}}
