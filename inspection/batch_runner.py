"""Sequential, file-based Batch execution; registration and quality verdicts are separate.

BatchRunner.run(batch_id, requests, output_root, stop_requested=None) starts a new
UUID run. retry(previous_summary, output_root, image_ids=None) starts fresh analysis
of failed/unprocessed inputs by default. No previous analysis result is reused.
No SQLite access, aggregate width/error/Fail Rate, queue, or engine modification.
Caller image/batch/wafer/die/site IDs remain unchanged. New execution analysis_run_id
is generated; any caller analysis_run_id is retained as requested_analysis_run_id.

Counts: processed = success + failed + skipped_engine. Partial measurements are
also execution successes; skipped_metrology overlaps those counts (e.g. MODEL PD).
Only image outcomes/quality flags determine COMPLETED_WITH_ISSUES, not product
quality. Nullable, unconfirmed metadata is retained; die_code is never a DB die_id.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import os
import uuid

from inspection.analysis_runner import AnalysisRunner, MODEL_CONFIG, ROOT, TEAM_UNET
from inspection.metrology_adapter import IDENTIFIERS, json_value

FIELDS = {
    "image_id", "image_path", "batch_id", "wafer_id", "die_id", "site_code",
    "analysis_run_id", "physical_site_id", "die_code", "condition_id",
    "original_filename", "exposure_value", "exposure_unit", "peb_temperature_c",
    "dataset_role", "upload_mode", "magnification", "roi", "calibration",
    "repeat_index", "illumination_setting", "auto_exposure",
}
NUMERIC = ("exposure_value", "peb_temperature_c", "magnification")
META = ("wafer_id", "die_id", "die_code", "site_code", "physical_site_id",
        "condition_id", "original_filename", "exposure_value", "exposure_unit",
        "peb_temperature_c", "dataset_role", "upload_mode", "magnification",
        "repeat_index", "illumination_setting", "auto_exposure")


class BatchInputError(ValueError):
    """Structural request rejection before execution; no model call."""
    status = "FAILED"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _new_json(path, value):
    """Exclusive creation: never overwrite an image result or request manifest."""
    text = json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _checkpoint(path, value):
    """Atomically replace only this newly owned run's progress summary."""
    temporary = path.with_name(".summary-" + uuid.uuid4().hex + ".tmp")
    _new_json(temporary, value)
    os.replace(temporary, path)


def _context():
    # Observed model/algorithm identifiers are collected from actual image results.
    files = (MODEL_CONFIG, TEAM_UNET / "judgment/rules.json",
             ROOT / "contracts/analysis_contract.py", ROOT / "inspection/analysis_runner.py",
             ROOT / "inspection/metrology_adapter.py", Path(__file__))
    return {str(path): _sha(path) if path.is_file() else None for path in files}


def _prepared(batch_id, requests):
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise BatchInputError("batch_id must be a nonempty string")
    if not isinstance(requests, list) or not requests:
        raise BatchInputError("a nonempty request list is required")
    ids = []
    for request in requests:
        if not isinstance(request, dict):
            raise BatchInputError("each request must be a dictionary")
        image_id = request.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip():
            raise BatchInputError("each image_id must be a nonempty string")
        ids.append(image_id)
    if len(ids) != len(set(ids)):
        raise BatchInputError("duplicate image_id; no images executed")

    output = []
    for request in requests:
        error = None
        # Never persist unknown keys (e.g. credentials) or unsupported objects.
        safe = {key: deepcopy(value) for key, value in request.items() if key in FIELDS}
        try:
            if set(request) - FIELDS:
                raise ValueError("unsupported request fields")
            if safe.get("batch_id") not in (None, batch_id):
                raise ValueError("request batch_id does not match batch")
            safe["batch_id"] = batch_id
            image_path = safe.get("image_path")
            if isinstance(image_path, str) and ("://" in image_path or image_path.startswith(("//", "\\\\"))):
                safe["image_path"] = None
                raise ValueError("image_path must be local; remote URLs are not recorded")
            for key in ("wafer_id", "die_id", "site_code"):
                if key not in safe:
                    raise ValueError(f"{key} key required; use null when unconfirmed")
            for key in FIELDS - set(NUMERIC) - {"roi", "calibration", "repeat_index", "auto_exposure"}:
                value = safe.get(key)
                if value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"{key} must be a nonempty string or null")
            if safe.get("site_code") not in (None, "lt", "c", "rb"):
                raise ValueError("site_code must be lt/c/rb or null")
            for key in NUMERIC:
                value = safe.get(key)
                if value is not None and (isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError(f"{key} must be finite numeric or null")
            if safe.get("magnification") is not None and safe["magnification"] <= 0:
                raise ValueError("magnification must be positive")
            if safe.get("upload_mode") not in (None, "wafer_batch", "repeatability", "lighting", "single_image"):
                raise ValueError("unsupported upload_mode")
            if safe.get("dataset_role") not in (None, "evaluation", "training_candidate", "development"):
                raise ValueError("unsupported dataset_role")
            if safe.get("repeat_index") is not None and (
                type(safe["repeat_index"]) is not int or safe["repeat_index"] < 1):
                raise ValueError("repeat_index must be a positive integer")
            if safe.get("auto_exposure") not in (None, 0, 1):
                raise ValueError("auto_exposure must be null/0/1")
            for key, allowed in (("roi", {"x_min", "y_min", "x_max", "y_max"}),
                ("calibration", {"calibration_id", "um_per_px", "magnification"})):
                value = safe.get(key)
                if value is not None and (not isinstance(value, dict) or set(value) - allowed):
                    raise ValueError(f"invalid {key} object")
            safe = json_value(safe)
        except (ValueError, TypeError) as exception:
            error = f"INVALID_METADATA: {exception}"
            # Preserve valid metadata fields, not values that cannot be serialized.
            safe = {key: value for key, value in safe.items()
                    if value is None or isinstance(value, (str, bool)) or
                    isinstance(value, (int, float)) and math.isfinite(value)}
            safe["image_id"] = request["image_id"]
            safe["batch_id"] = batch_id
        output.append({"request": safe, "validation_error": error})
    return output


def _failure(request, code, message):
    return {"image_id": request["image_id"],
        **{key: request.get(key) for key in IDENTIFIERS}, "image_path": request.get("image_path"),
        "status": "FAILED", "error_code": code, "error_message": message,
        "defect": None, "metrology": {"image_id": request["image_id"], "status": "SKIPPED",
                                     "measurements": [], "error_code": "UPSTREAM_FAILED"},
        "measurement_summary": None, "width_error": None,
        "official_analysis_complete": False, "inspection_verdict": None}


def _outcome(result):
    if result["status"] == "FAILED":
        return "FAILED"
    pd = result.get("pd_policy", {}).get("state")
    if pd == "MODEL_PD_EXCLUDED":
        return "MODEL_PD_EXCLUDED"
    if pd == "POLICY_REVIEW_REQUIRED":
        return "POLICY_REVIEW_REQUIRED"
    metrology = result.get("metrology") or {}
    if metrology.get("quality", {}).get("partial_measurement") is True:
        return "PARTIAL_MEASUREMENT"
    if (result["status"] == "SUCCESS" and pd == "NO_PD"
        and metrology.get("status") == "SUCCESS"
        and metrology.get("quality", {}).get("partial_measurement") is False
        and metrology.get("judgment", {}).get("status") == "judged"
        and result.get("official_analysis_complete") is True):
        lines = metrology["judgment"].get("lines", [])
        if any(j.get("pitch_ok") is False or j.get("near_threshold") is True for j in lines):
            return "QUALITY_REVIEW_REQUIRED"
        return "COMPLETE"
    raise ValueError("unrecognized successful result policy/quality state")


def _validate_result(result, request, analysis_dir):
    result = json_value(result)
    if result.get("image_id") != request["image_id"] or result.get("status") not in ("SUCCESS", "FAILED", "SKIPPED"):
        raise ValueError("invalid image result identity/status")
    for key in IDENTIFIERS:
        if result.get(key) != request.get(key):
            raise ValueError(f"result {key} mismatch")
    for stage in ("defect", "metrology"):
        value = result.get(stage)
        if value is not None and value.get("image_id") != request["image_id"]:
            raise ValueError(f"{stage} image_id mismatch")
    defect = result.get("defect") or {}
    metrology = result.get("metrology") or {}
    for stage in (defect, metrology):
        if stage and stage.get("status") not in ("SUCCESS", "FAILED", "SKIPPED"):
            raise ValueError("invalid stage engine status")
        if stage.get("status") == "SUCCESS":
            for key in IDENTIFIERS:
                if stage.get(key) != request.get(key):
                    raise ValueError(f"stage {key} mismatch")
    if result["status"] in ("SUCCESS", "SKIPPED"):
        if defect.get("status") != "SUCCESS":
            raise ValueError("successful analysis requires successful defect stage")
        pd_sources = {d.get("source") for d in defect.get("detections", []) if d.get("class_code") == "pd"}
        state = result.get("pd_policy", {}).get("state")
        if state == "MODEL_PD_EXCLUDED":
            if "MODEL_POSTPROCESSED" not in pd_sources or metrology.get("status") != "SKIPPED":
                raise ValueError("MODEL PD policy does not match detection/stage")
        elif state == "POLICY_REVIEW_REQUIRED":
            if pd_sources != {"HEURISTIC"} or metrology.get("status") != "SKIPPED":
                raise ValueError("HEURISTIC PD policy does not match detection/stage")
        elif state == "NO_PD" and pd_sources:
            raise ValueError("PD detection cannot be classified NO_PD")
        if state in ("MODEL_PD_EXCLUDED", "POLICY_REVIEW_REQUIRED"):
            if result.get("width_error") is not None or result.get("official_analysis_complete") is True:
                raise ValueError("excluded/review result cannot claim official measurement")
    if metrology.get("status") == "SUCCESS" and not metrology.get("measurements"):
        raise ValueError("successful metrology has no measurements")
    if metrology.get("quality", {}).get("partial_measurement") is True:
        if result.get("width_error") is not None or result.get("official_analysis_complete") is True:
            raise ValueError("partial measurements cannot claim official error/complete")
    for overlay in (result.get("overlays") or {}).values():
        if overlay is not None:
            path = Path(overlay["path"]).resolve()
            if not path.is_relative_to(analysis_dir) or not path.is_file():
                raise ValueError("overlay path is missing or outside image output")
    _outcome(result)
    return result


def _totals(rows):
    processed = [row for row in rows if row["processing_status"] != "UNPROCESSED"]
    outcomes = Counter(row["outcome"] for row in processed)
    counts = {
        "requested_images": len(rows), "processed_images": len(processed),
        "unprocessed_images": len(rows) - len(processed),
        "success_images": sum(r["engine_status"] == "SUCCESS" for r in processed),
        "failed_images": sum(r["processing_status"] == "FAILED" for r in processed),
        "skipped_engine_images": sum(r["engine_status"] == "SKIPPED" for r in processed),
        "skipped_metrology_images": sum(r.get("metrology_status") == "SKIPPED" for r in processed),
        "complete_measurement_images": outcomes["COMPLETE"],
        "partial_measurement_images": outcomes["PARTIAL_MEASUREMENT"],
        "policy_review_images": outcomes["POLICY_REVIEW_REQUIRED"],
        "quality_review_images": outcomes["QUALITY_REVIEW_REQUIRED"],
        "model_pd_excluded_images": outcomes["MODEL_PD_EXCLUDED"],
    }
    assert counts["processed_images"] == counts["success_images"] + counts["failed_images"] + counts["skipped_engine_images"]
    assert counts["requested_images"] == counts["processed_images"] + counts["unprocessed_images"]
    return counts


class BatchRunner:
    def __init__(self, analysis_runner=None):
        self.analysis_runner = analysis_runner if analysis_runner is not None else AnalysisRunner()

    def run(self, batch_id, requests, output_root, *, stop_requested=None, retry_of_run_id=None):
        prepared = _prepared(batch_id, requests)
        if "://" in str(output_root) or str(output_root).startswith(("//", "\\\\")):
            raise BatchInputError("output_root must be a local path")
        root = Path(output_root).resolve()
        run_id = "batch-" + uuid.uuid4().hex
        run_dir = root / run_id
        # Exclusive UUID directory. Never continue inside a pre-existing run.
        run_dir.mkdir(parents=True, exist_ok=False)
        rows = []
        for index, entry in enumerate(prepared):
            request = entry["request"]
            key = hashlib.sha256(request["image_id"].encode("utf-8")).hexdigest()
            rows.append({
                "request_index": index, "image_id": request["image_id"], "image_key": key,
                "metadata": {key: request.get(key) for key in META},
                "requested_analysis_run_id": request.get("analysis_run_id"),
                "analysis_run_id": run_id + ":" + key,
                "processing_status": "UNPROCESSED", "engine_status": None,
                "outcome": "UNPROCESSED", "result_json": None, "result_sha256": None,
                "defect_status": None, "metrology_status": None,
                "reason": None, "started_at": None, "ended_at": None})
        manifest_path = run_dir / "request_manifest.json"
        summary_path = run_dir / "batch_summary.json"
        _new_json(manifest_path, {"batch_id": batch_id, "batch_run_id": run_id, "entries": prepared})
        summary = {
            "schema": "inspection-batch-v0.1", "batch_id": batch_id, "batch_run_id": run_id,
            "run_directory": str(run_dir), "started_at": _now(), "ended_at": None,
            "execution_status": "INTERRUPTED", "interruption_reason": "IN_PROGRESS_OR_UNCLEAN_STOP",
            "request_manifest_json": str(manifest_path), "request_manifest_sha256": _sha(manifest_path),
            "retry_of_run_id": retry_of_run_id, "previous_results_reused": False,
            "execution_config_sha256": _context(), "images": rows,
            "official_aggregates": "NOT_COMPUTED",
            "count_semantics": "processed=success+failed+skipped_engine; metrology skips overlap; success is execution, not defect-free/product PASS"}
        def checkpoint():
            summary["counts"] = _totals(rows)
            _checkpoint(summary_path, summary)
        checkpoint()
        try:
            for entry, row in zip(prepared, rows):
                if stop_requested is not None and stop_requested():
                    summary["interruption_reason"] = "STOP_REQUESTED"
                    break
                request = deepcopy(entry["request"])
                request["batch_id"] = batch_id
                request["analysis_run_id"] = row["analysis_run_id"]
                item_dir = run_dir / "images" / row["image_key"]
                analysis_dir = item_dir / "analysis"
                row["started_at"] = _now()
                result = None
                result_path = None
                try:
                    if entry["validation_error"]:
                        raise ValueError(entry["validation_error"])
                    image_path = request.get("image_path")
                    if not isinstance(image_path, str) or not image_path:
                        raise ValueError("MISSING_IMAGE_PATH")
                    if "://" in image_path or image_path.startswith(("//", "\\\\")):
                        raise ValueError("INVALID_LOCAL_IMAGE_PATH")
                    if not Path(image_path).is_file():
                        result = _failure(request, "IMAGE_NOT_FOUND", "image file does not exist")
                    else:
                        row["source_image_sha256"] = _sha(image_path)
                        result = self.analysis_runner.run(request, output_dir=analysis_dir)
                        result = _validate_result(result, request, analysis_dir.resolve())
                    result_path = analysis_dir / "result.json"
                    if result_path.is_file():
                        saved = json.loads(result_path.read_text(encoding="utf-8"))
                        if saved != result:
                            raise ValueError("saved result does not match returned result")
                    else:
                        analysis_dir.mkdir(parents=True, exist_ok=True)
                        _new_json(result_path, result)
                    row["outcome"] = _outcome(result)
                except (KeyboardInterrupt, SystemExit):
                    row["reason"] = "INTERRUPTED_DURING_IMAGE"
                    summary["interruption_reason"] = row["reason"]
                    checkpoint()
                    break
                except Exception as error:
                    code = ("INVALID_METADATA" if entry["validation_error"] else
                            "RESULT_WRITE_FAILED" if isinstance(error, OSError) else "IMAGE_EXECUTION_FAILED")
                    result = _failure(request, code, f"{type(error).__name__}: {error}")
                    row["outcome"] = "FAILED"
                    row["partial_artifact_directory"] = str(analysis_dir) if analysis_dir.exists() else None
                    # Keep any original/partially saved result untouched. Separate failure record.
                    result_path = item_dir / "failure.json"
                    try:
                        item_dir.mkdir(parents=True, exist_ok=True)
                        _new_json(result_path, result)
                    except Exception as save_error:
                        row["result_save_error"] = f"{type(save_error).__name__}: {save_error}"
                        result_path = None
                row["engine_status"] = result["status"]
                row["processing_status"] = "FAILED" if row["outcome"] == "FAILED" else "COMPLETED"
                row["defect_status"] = (result.get("defect") or {}).get("status")
                row["metrology_status"] = (result.get("metrology") or {}).get("status")
                row["reason"] = result.get("error_code") or (result.get("metrology") or {}).get("error_code")
                row["error_message"] = result.get("error_message")
                row["pd_policy_state"] = (result.get("pd_policy") or {}).get("state")
                row["model_config_identifiers"] = {
                    "model_version": (result.get("defect") or {}).get("model_version"),
                    "model_sha256": (result.get("defect") or {}).get("model_sha256"),
                    "algorithm_version": (result.get("metrology") or {}).get("algorithm_version"),
                    "calibration_id": (result.get("metrology") or {}).get("calibration_id"),
                    "configuration": result.get("configuration")}
                if result_path is not None:
                    row["result_json"] = str(result_path)
                    row["result_sha256"] = _sha(result_path)
                row["ended_at"] = _now()
                checkpoint()
        except (KeyboardInterrupt, SystemExit):
            summary["interruption_reason"] = "INTERRUPTED_BETWEEN_IMAGES"
        except Exception as error:
            # Summary/storage failure is a batch-level failure, not a silent success.
            summary["execution_status"] = "FAILED"
            summary["batch_error"] = f"{type(error).__name__}: {error}"
        counts = _totals(rows)
        if summary.get("batch_error") is None:
            if counts["unprocessed_images"]:
                summary["execution_status"] = "INTERRUPTED"
            elif counts["failed_images"] == counts["requested_images"]:
                summary["execution_status"] = "FAILED"
            elif any(counts[key] for key in ("failed_images", "partial_measurement_images",
                     "policy_review_images", "quality_review_images")):
                summary["execution_status"] = "COMPLETED_WITH_ISSUES"
            else:
                summary["execution_status"] = "COMPLETED"
        if summary["execution_status"] != "INTERRUPTED":
            summary["interruption_reason"] = None
        summary["ended_at"] = _now()
        summary["counts"] = counts
        summary["summary_json"] = str(summary_path)
        try:
            _checkpoint(summary_path, summary)
        except Exception as error:
            summary["execution_status"] = "FAILED"
            summary["summary_save_error"] = f"{type(error).__name__}: {error}"
        return json_value(summary)

    def retry(self, previous_summary, output_root, *, image_ids=None, stop_requested=None):
        """New run, fresh execution; defaults to failed or unprocessed image IDs."""
        path = Path(previous_summary).resolve()
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("schema") != "inspection-batch-v0.1":
            raise BatchInputError("unsupported previous run summary")
        manifest_path = Path(previous["request_manifest_json"]).resolve()
        if not manifest_path.is_relative_to(path.parent) or _sha(manifest_path) != previous["request_manifest_sha256"]:
            raise BatchInputError("previous request manifest path/hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["batch_id"] != previous["batch_id"]
            or manifest["batch_run_id"] != previous["batch_run_id"]):
            raise BatchInputError("previous manifest identity mismatch")
        available = {e["request"]["image_id"]: e["request"] for e in manifest["entries"]}
        invalid_metadata_ids = {e["request"]["image_id"] for e in manifest["entries"] if e.get("validation_error")}
        if image_ids is None:
            image_ids = [row["image_id"] for row in previous["images"]
                         if row["processing_status"] in ("FAILED", "UNPROCESSED")]
        if (not isinstance(image_ids, list) or not image_ids or
            any(not isinstance(i, str) for i in image_ids) or len(set(image_ids)) != len(image_ids)
            or any(i not in available for i in image_ids)):
            raise BatchInputError("retry requires unique IDs from previous requests")
        if any(i in invalid_metadata_ids for i in image_ids):
            raise BatchInputError("invalid metadata requires explicit corrected requests in a new run")
        selected = [deepcopy(available[i]) for i in image_ids]
        return self.run(previous["batch_id"], selected, output_root,
                        stop_requested=stop_requested, retry_of_run_id=previous["batch_run_id"])
