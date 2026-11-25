"""IDA integration adapter for the teammate's r33 defect pipeline (TASK18-B-21).

Based on the teammate's defect_adapter.py (ida-pipeline-incoming/aoi_pipeline, kept unchanged there).
The detection mapping, the source codes (SOURCE_CODES) and the call order are the teammate's, unchanged.
Integration additions only (marked ADDED):
  * the model call is the existing frozen local ONNX (defect_inference.py of this folder = current production
    file, same model version and SHA-256 as the SDK cache; verified identical r33 inputs and outputs);
  * contract import that also works from the project root (contracts.analysis_contract);
  * error_message and the frozen model identity (model_sha256 / model_path / inference_backend) for storage;
  * team_source (the teammate's own source code) and, for a P split out of an S (team_source "from_s"),
    the index of that S in the same output (derived_from_index), recorded by a read-only observer around
    the teammate's add_particles_from_s (its output is returned unchanged).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:  # ADDED: project-root import first, teammate-style flat import as fallback.
    from contracts.analysis_contract import DEFECT_CLASSES
except ModuleNotFoundError:
    from analysis_contract import DEFECT_CLASSES

import aoi_pipeline as _pipeline
from aoi_pipeline import run_aoi_pipeline
from defect_inference import DEFAULT_CONFIG_PATH, MODEL_ID, LocalModelError

# detections.data["source"] -> output source (teammate mapping, unchanged)
#   model     : model detection that passed the r26+ rules (incl. reclassified, e.g. PD on gap -> S)
#   rule      : S (blocked gap) / PD (broken line) added by image rules (confidence 0.0)
#   from_s    : P split out of a large S (confidence = original S)
#   heuristic : whole-image PD for a low-contrast image
#   blur      : whole-image PD for a blurry pattern (model not called)
SOURCE_CODES = {
    "blur": "HEURISTIC",
    "model": "MODEL_POSTPROCESSED",
    "rule": "RULE",
    "from_s": "RULE",
    "heuristic": "HEURISTIC",
}


# ---- ADDED: S -> P derivation record (observer only; teammate rules and outputs unchanged) -----------------
# In add_particles_from_s each new P is built from ONE S (index i): P box = foreign pixels inside that S's
# (clamped) box, P confidence = that S's confidence, and the S box is then widened (xyxy[i] = sbox); the new
# P rows are appended in S order. Later steps only drop P on PD and sort by confidence (S boxes unchanged).
# The observer matches each new P to its source S by exactly those code properties; when more than one S
# qualifies, the link is left unresolved (candidates kept) instead of guessing.
_DERIVATIONS = []


def _contains(outer, inner):
    return (outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3])


def _observe_add_particles_from_s(original):
    def wrapper(detections, lg, image_bgr):
        before_n = len(detections)
        before_names = list(detections.data.get("class_name", [])) if before_n else []
        before_boxes = detections.xyxy.copy() if before_n else np.zeros((0, 4), np.float32)
        before_conf = detections.confidence.copy() if before_n else np.zeros(0, np.float32)
        result = original(detections, lg, image_bgr)
        if len(result) > before_n:
            height, width = image_bgr.shape[:2]
            for j in range(before_n, len(result)):
                pbox, pconf = result.xyxy[j], float(result.confidence[j])
                candidates = []
                for i in range(before_n):
                    if before_names[i] != "s" or float(before_conf[i]) != pconf:
                        continue
                    # Same region as the teammate code: int() truncation, then clamped to the image.
                    x1, y1, x2, y2 = map(int, before_boxes[i])
                    clamped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
                    if _contains(clamped, pbox):
                        candidates.append(i)
                _DERIVATIONS.append({
                    "p_box": [float(v) for v in pbox], "confidence": pconf,
                    "s_boxes": [[float(v) for v in result.xyxy[i]] for i in candidates]})
        return result
    wrapper.__wrapped__ = original
    return wrapper


if not hasattr(_pipeline.rules.add_particles_from_s, "__wrapped__"):
    _pipeline.rules.add_particles_from_s = _observe_add_particles_from_s(_pipeline.rules.add_particles_from_s)


def _link_derived(items):
    """Attach derived_from_index to each from_s P using the recorded derivations (exact box values)."""
    for item in items:
        if item["team_source"] != "from_s":
            continue
        records = [r for r in _DERIVATIONS if r["p_box"] == item["bbox_xyxy"] and r["confidence"] == item["_conf32"]]
        parents = sorted({k for r in records for box in r["s_boxes"] for k, other in enumerate(items)
                          if other["class_code"] == "s" and other["bbox_xyxy"] == box})
        item["derived_from_index"] = parents[0] if len(records) == 1 and len(parents) == 1 else None
        item["derivation"] = ("S_SPLIT_P (teammate add_particles_from_s)" if item["derived_from_index"] is not None
                              else f"UNRESOLVED: candidate S indices {parents}")
    for item in items:
        item.pop("_conf32", None)
    return items


def _model_identity():
    """ADDED: frozen local model identity from the project configuration (the runtime verifies the SHA on load)."""
    config = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    return {"model_sha256": config["model_sha256"], "model_path": config["model_path"],
            "inference_backend": "onnxruntime-cpu (local frozen ONNX)"}


class DefectAIAdapter:
  def predict(self, request: dict[str, Any]) -> dict[str, Any]:
    image_id = request.get("image_id")
    image_path = request.get("image_path")

    identifiers = {
        key: request.get(key)
        for key in ("wafer_id", "die_id", "site_code", "batch_id", "analysis_run_id")
    }

    def failed(error_code: str, message: str | None = None) -> dict[str, Any]:
      return {
          "image_id": image_id,
          **identifiers,
          "status": "FAILED",
          "model_version": MODEL_ID,
          "detections": [],
          "error_code": error_code,
          "error_message": message,   # ADDED
      }

    if not isinstance(image_id, str) or not image_id:
      return failed("INVALID_IMAGE_ID")
    if not isinstance(image_path, str) or not image_path:
      return failed("INVALID_IMAGE_PATH")

    try:
      path = Path(image_path)
      if not path.is_file():
        return failed("IMAGE_NOT_FOUND", str(path))

      image_bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
      if image_bgr is None:
        return failed("IMAGE_DECODE_FAILED")

      confidence = float(request.get("confidence", 0.25))
      iou_threshold = float(request.get("iou_threshold", 0.45))
      enable_deskew = request.get("enable_deskew", True)

      _DERIVATIONS.clear()   # ADDED
      result = run_aoi_pipeline(
          image_bgr, conf_threshold=confidence, iou_threshold=iou_threshold,
          enable_deskew=enable_deskew)
      detections = result["detections"]
      names = list(detections.data.get("class_name", []))
      sources = list(detections.data.get("source", ["model"] * len(detections)))
      confidences = detections.confidence

      items = []
      for i, box in enumerate(detections.xyxy):
        class_code = str(names[i]).lower()
        if class_code not in DEFECT_CLASSES:
          return failed("UNKNOWN_DEFECT_CLASS")
        items.append({
            "class_code": class_code,
            "confidence": float(confidences[i]) if confidences is not None else 0.0,
            "bbox_xyxy": [float(value) for value in box],
            "source": SOURCE_CODES.get(str(sources[i]), "MODEL_POSTPROCESSED"),
            "team_source": str(sources[i]),   # ADDED: the teammate's own source code (model/rule/from_s/heuristic/blur)
            "_conf32": float(confidences[i]) if confidences is not None else 0.0,
        })
      _link_derived(items)   # ADDED

      skew_angle = float(result["skew_angle"])
      coordinate_system = (
          "DESKEWED_IMAGE" if enable_deskew and skew_angle != 0 else "ORIGINAL_IMAGE")
      analyzed_image = result["analyzed_image"]
      height, width = analyzed_image.shape[:2]

      return {
          "image_id": image_id,
          **identifiers,
          "status": "SUCCESS",
          "model_version": MODEL_ID,
          "detections": items,
          "error_code": None,
          "bbox_coordinate_system": coordinate_system,
          "bbox_image_width": width,
          "bbox_image_height": height,
          "skew_angle": skew_angle,
          "contrast_score": float(result["contrast_score"]),
          "postprocess_version": result.get("postprocess_version"),      # ADDED (teammate key, passed through)
          **_model_identity(),                                          # ADDED
      }
    except LocalModelError as error:   # ADDED: specific local-model error code (as in the current adapter)
      return failed(error.code, str(error))
    except Exception as error:
      return failed("DEFECT_ANALYSIS_ERROR", f"{type(error).__name__}: {error}")
