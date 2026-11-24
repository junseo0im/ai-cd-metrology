"""Single-image integration only: no Dashboard, SQLite, remote SDK or batch loop."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

from contracts.analysis_contract import DEFECT_CLASSES
from inspection.metrology_adapter import (
    IDENTIFIERS, ROOT, TEAM_UNET, MetrologyAdapter, checked_module, json_value,
)

TEAM_DEFECT = ROOT / "runtime_data/external_incoming/defect_adapter"
MODEL_CONFIG = ROOT / "runtime_data/config/defect_local_model.json"


def _load_defect():
    if str(TEAM_DEFECT) not in sys.path:
        sys.path.insert(0, str(TEAM_DEFECT))
    import defect_adapter
    checked_module(defect_adapter, TEAM_DEFECT / "defect_adapter.py")
    return defect_adapter.DefectAIAdapter()


def _save_defect_overlay(request, result, path):
    """Reuse the team's deskew and drawing helpers; never run inference twice."""
    import cv2
    import numpy as np
    import supervision as sv
    import aoi_pipeline as pipeline
    checked_module(pipeline, TEAM_DEFECT / "aoi_pipeline.py")
    image = cv2.imdecode(np.fromfile(request["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot load overlay source image")
    frame = result["bbox_coordinate_system"]
    if frame == "DESKEWED_IMAGE":
        image, angle = pipeline.auto_deskew_vertical_lines(image)
        if float(angle) != result["skew_angle"]:
            raise ValueError("deskew replay does not match inference frame")
    if list(image.shape[:2]) != [result["bbox_image_height"], result["bbox_image_width"]]:
        raise ValueError("overlay image size does not match bbox frame")
    items = result["detections"]
    detections = sv.Detections(
        xyxy=np.asarray([item["bbox_xyxy"] for item in items], dtype=np.float32).reshape(-1, 4),
        class_id=np.asarray([item["class_id"] for item in items], dtype=int),
        confidence=np.asarray([item["confidence"] for item in items], dtype=np.float32),
        data={"class_name": np.asarray([item["class_name"] for item in items], dtype=str)})
    rendered = pipeline._draw_detections(image, detections, show_corners=True, show_masks=False)
    ok, encoded = cv2.imencode(".png", rendered)
    if not ok:
        raise ValueError("defect overlay PNG encoding failed")
    encoded.tofile(str(path))
    return {"path": str(path), "base_image": frame,
            "coordinate_space": frame, "rendering": "team bbox/label/corner renderer",
            "masks": "not exposed by existing Contract Adapter; not rendered",
            "deskew_frame": "unchanged team function replay, angle verified; no second inference"}


def _save_metrology_overlay(request, result, path):
    if str(TEAM_UNET) not in sys.path:
        sys.path.insert(0, str(TEAM_UNET))
    import run_example
    checked_module(run_example, TEAM_UNET / "run_example.py")
    run_example.draw_overlay(result, Path(request["image_path"]), path)
    return {"path": str(path), "base_image": "ORIGINAL_IMAGE",
            "measurement_coordinate_space": "NATIVE_IMAGE",
            "coordinate_mapping": "team pixel_scale native-to-original",
            "display_y_offset_native_px": {"line": -4, "gap": 4}}


class AnalysisRunner:
    """Reuse this instance and its engines for subsequent single-image calls.

    run(request, output_dir=None) -> JSON-safe dict. output_dir, if supplied,
    must not exist; artifacts are separate per image. Execution SUCCESS is not
    an image acceptance verdict. Model/heuristic PD policies are explicit.
    """
    def __init__(self, defect_engine=None, metrology_engine=None):
        self._defect = defect_engine
        self._metrology = metrology_engine if metrology_engine is not None else MetrologyAdapter()

    def run(self, request, output_dir=None):
        started = time.perf_counter()
        image_id = request.get("image_id") if isinstance(request, dict) else None
        result = {"image_id": image_id if isinstance(image_id, str) else None,
                  "image_path": None, "status": "FAILED", "error_code": None,
                  "error_message": None, "defect": None,
                  "metrology": {"image_id": image_id if isinstance(image_id, str) else None,
                                "status": "SKIPPED", "measurements": [],
                                "error_code": "UPSTREAM_NOT_COMPLETED"},
                  "pd_policy": {"state": "NOT_EVALUATED", "sources": []},
                  "measurement_summary": None, "width_error": None,
                  "official_analysis_complete": False, "inspection_verdict": None,
                  "overlays": {"defect": None, "metrology": None},
                  "runner_version": "single-image-v0.1"}
        destination = None
        try:
            if not isinstance(request, dict) or not isinstance(image_id, str) or not image_id.strip():
                raise ValueError("image_id is required")
            for key in IDENTIFIERS:
                value = request.get(key)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"{key} must be a nonempty string or null")
                result[key] = value
            raw_path = request.get("image_path")
            if (not isinstance(raw_path, str) or not raw_path
                or "://" in raw_path or raw_path.startswith(("//", "\\\\"))):
                raise ValueError("image_path must be a local file path")
            path = Path(raw_path).resolve()
            result["image_path"] = str(path)
            if not path.is_file():
                result["error_code"] = "IMAGE_NOT_FOUND"
                raise ValueError("image file does not exist")
            local_request = {key: request[key] for key in
                             ("image_id", *IDENTIFIERS, "roi", "calibration", "magnification")
                             if key in request}
            local_request["image_path"] = str(path)
            # E1-equivalent team inference/postprocessing settings, not caller-tuned.
            defect_request = {**local_request, "enable_deskew": True,
                              "confidence": 0.25, "iou_threshold": 0.45,
                              "model_config_path": str(MODEL_CONFIG)}
            result["configuration"] = {
                "defect_model_config": str(MODEL_CONFIG),
                "enable_deskew": True, "confidence": 0.25, "iou_threshold": 0.45,
                "design_width_um": 10.0,
                "judgment_rules_path": str(TEAM_UNET / "judgment/rules.json"),
                "contract_path": str(ROOT / "contracts/analysis_contract.py"),
                "model_config_sha256": hashlib.sha256(MODEL_CONFIG.read_bytes()).hexdigest(),
                "judgment_rules_sha256": hashlib.sha256((TEAM_UNET / "judgment/rules.json").read_bytes()).hexdigest()}
            with path.open("rb") as stream:
                result["image_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            if output_dir is not None and Path(output_dir).exists():
                raise ValueError("output_dir must be new; existing artifacts are preserved")
            if self._defect is None:
                self._defect = _load_defect()
            defect_started = time.perf_counter()
            defect = json_value(self._defect.predict(defect_request))
            result["defect_duration_seconds"] = time.perf_counter() - defect_started
            result["defect"] = defect
            if defect.get("image_id") != image_id or defect.get("status") not in ("SUCCESS", "FAILED"):
                raise ValueError("invalid defect response identity/status")
            if defect["status"] != "SUCCESS":
                result["error_code"] = defect.get("error_code") or "DEFECT_FAILED"
                result["error_message"] = defect.get("error_message")
                result["metrology"]["error_code"] = "DEFECT_FAILED"
            else:
                if not isinstance(defect.get("model_version"), str) or not defect["model_version"]:
                    raise ValueError("defect model version missing")
                if not isinstance(defect.get("detections"), list):
                    raise ValueError("defect detections must be a list")
                frame = defect.get("bbox_coordinate_system")
                if frame not in ("ORIGINAL_IMAGE", "DESKEWED_IMAGE"):
                    raise ValueError("unknown bbox coordinate space")
                for field in ("bbox_image_width", "bbox_image_height"):
                    if not isinstance(defect.get(field), (int, float)) or defect[field] <= 0:
                        raise ValueError("invalid bbox image dimensions")
                if not isinstance(defect.get("skew_angle"), (int, float)):
                    raise ValueError("skew angle missing")
                # The actual team model mapping, NOT the Contract class tuple order.
                config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
                names = [name for index, name in enumerate(config["logit_class_names"])
                         if index != config["background_index"]]
                from_model = {name: index for index, name in enumerate(names)}
                for detection in defect["detections"]:
                    code = detection.get("class_code")
                    if code not in DEFECT_CLASSES or detection.get("source") not in ("MODEL_POSTPROCESSED", "HEURISTIC"):
                        raise ValueError("invalid defect class/source")
                    confidence = detection.get("confidence")
                    bbox = detection.get("bbox_xyxy")
                    if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                        or not 0 <= confidence <= 1 or not isinstance(bbox, list) or len(bbox) != 4
                        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox)):
                        raise ValueError("invalid confidence/bbox")
                    if not (0 <= bbox[0] < bbox[2] <= defect["bbox_image_width"]
                            and 0 <= bbox[1] < bbox[3] <= defect["bbox_image_height"]):
                        raise ValueError("bbox outside inference image")
                    detection.update(class_id=from_model[code], class_name=code)
                defect["coordinate_space"] = frame
                pd_sources = sorted({item["source"] for item in defect["detections"]
                                     if item["class_code"] == "pd"})
                result["pd_policy"]["sources"] = pd_sources
                if "MODEL_POSTPROCESSED" in pd_sources:
                    result["pd_policy"]["state"] = "MODEL_PD_EXCLUDED"
                    result["metrology"]["error_code"] = "MODEL_PD_EXCLUDED"
                    result["status"] = "SUCCESS"
                elif "HEURISTIC" in pd_sources:
                    result["pd_policy"]["state"] = "POLICY_REVIEW_REQUIRED"
                    result["metrology"]["error_code"] = "POLICY_REVIEW_REQUIRED"
                    result["status"] = "SKIPPED"
                    result["quality"] = {"policy_review_required": True}
                else:
                    result["pd_policy"]["state"] = "NO_PD"
                    metrology_started = time.perf_counter()
                    metrology = json_value(self._metrology.measure(local_request))
                    result["metrology_duration_seconds"] = time.perf_counter() - metrology_started
                    result["metrology"] = metrology
                    if metrology.get("image_id") != image_id or metrology.get("status") not in ("SUCCESS", "FAILED"):
                        raise ValueError("invalid metrology response identity/status")
                    result["status"] = metrology["status"]
                    if metrology["status"] == "FAILED":
                        result["error_code"] = metrology.get("error_code") or "METROLOGY_FAILED"
                        result["error_message"] = metrology.get("error_message")
                    else:
                        measurements = metrology.get("measurements")
                        if not isinstance(measurements, list) or not measurements:
                            raise ValueError("successful metrology has no measurements")
                        summary = {"basis": "arithmetic mean of returned calibrated measurements"}
                        for kind in ("line", "gap"):
                            values = [item["width_um"] for item in measurements
                                      if item["measurement_type"] == kind]
                            summary[kind] = {"count": len(values),
                                             "mean_um": sum(values) / len(values) if values else None}
                        result["measurement_summary"] = summary
                        quality = metrology.get("quality") or {}
                        complete = quality.get("partial_measurement") is False
                        result["official_analysis_complete"] = complete
                        if complete and summary["line"]["mean_um"] is not None:
                            signed = (summary["line"]["mean_um"] - 10.0) / 10.0 * 100.0
                            result["width_error"] = {"design_width_um": 10.0, "signed_percent": signed,
                                                     "absolute_percent": abs(signed)}
            # Strict serialization before creating any artifact directory.
            result = json_value(result)
            if output_dir is not None:
                target = Path(output_dir).resolve()
                target.mkdir(parents=True, exist_ok=False)
                destination = target
                if result["defect"] and result["defect"]["status"] == "SUCCESS":
                    result["overlays"]["defect"] = _save_defect_overlay(
                        defect_request, result["defect"], destination / "defect_overlay.png")
                if result["metrology"]["status"] == "SUCCESS":
                    result["overlays"]["metrology"] = _save_metrology_overlay(
                        local_request, result["metrology"], destination / "metrology_overlay.png")
        except Exception as error:
            result["status"] = "FAILED"
            result["official_analysis_complete"] = False
            result["width_error"] = None
            result["error_code"] = result.get("error_code") or "RUNNER_ERROR"
            result["error_message"] = f"{type(error).__name__}: {error}"
            # Malformed engine payloads must not prevent a serializable error return.
            try:
                json_value(result)
            except (ValueError, TypeError):
                result["defect"] = None
                result["metrology"] = {"image_id": result["image_id"], "status": "SKIPPED",
                                       "measurements": [], "error_code": "INVALID_ENGINE_RESULT"}
                result["measurement_summary"] = None
                result["pd_policy"] = {"state": "NOT_EVALUATED", "sources": []}
        result["duration_seconds"] = time.perf_counter() - started
        result = json_value(result)
        if destination is not None:
            (destination / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return result
