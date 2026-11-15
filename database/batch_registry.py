"""Shared Batch preview/approval with optional DB-1D owned-file storage.

DB-1B/C defaults keep the operational-path guard and reference files in place.
DB-1D explicitly opts into managed storage and operational access after approval.
Preview states are metadata-registration states, not AI/QC results.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import math
import json
from pathlib import Path
import sqlite3

from database.upload_storage import NewFiles
from database.filename_parser import FilenameParseError, parse_filename, validate_metadata
from database.image_registry import (
    ImageRegistrationError, _check_connection, _inspect_png, register_image,
)


DUPLICATE_ACTIONS = ("EXCLUDE", "REGISTER_SEPARATELY", "HOLD")


class BatchRegistrationError(ImageRegistrationError):
    pass


def _text_issues(values, names):
    issues = []
    for name in names:
        value = values.get(name)
        if value is None or value == "":
            issues.append({"status": "NEEDS_METADATA", "message": f"Missing {name}"})
        elif not isinstance(value, str) or not value.strip():
            issues.append({"status": "INVALID", "message": f"Invalid {name}"})
    return issues


def _common_issues(connection, batch):
    issues = _text_issues(batch, ("batch_id", "wafer_id", "upload_mode", "dataset_role", "batch_status"))
    for name, allowed in (
        ("upload_mode", ("wafer_batch", "repeatability", "lighting", "single_image")),
        ("dataset_role", ("evaluation", "training_candidate", "development")),
    ):
        if batch.get(name) not in (None, "") and batch[name] not in allowed:
            issues.append({"status": "INVALID", "message": f"Invalid {name}"})
    if batch.get("upload_mode") == "single_image" and connection.execute("PRAGMA user_version").fetchone()[0] < 2:
        issues.append({"status":"INVALID","message":"Single Image requires Schema Version 2; explicit migration required"})
    wafer = batch.get("wafer_id")
    if isinstance(wafer, str) and wafer:
        try:
            identity = validate_metadata(wafer_id=wafer, die_code="11", position_code="c")
            if wafer != identity.wafer_id:
                raise FilenameParseError("Batch wafer_id must use canonical W<digits>")
        except FilenameParseError as exc:
            issues.append({"status": "INVALID", "message": str(exc)})
    if any(item["status"] == "INVALID" for item in issues):
        return issues
    if batch.get("batch_id") and connection.execute(
            "SELECT 1 FROM upload_batches WHERE batch_id=?", (batch["batch_id"],)).fetchone():
        issues.append({"status": "INVALID", "message": "batch_id already exists; appending is not supported"})
    existing = connection.execute(
        "SELECT process_run_id FROM wafers WHERE wafer_id=?", (wafer,)).fetchone()
    process_id = batch.get("process_run_id")
    if process_id is not None and (not isinstance(process_id, str) or not process_id.strip()):
        issues.append({"status": "INVALID", "message": "Invalid process_run_id"})
        return issues
    if existing:
        if process_id is not None and process_id != existing[0]:
            issues.append({"status": "INVALID", "message": "Existing Wafer process_run_id mismatch"})
        process_id = existing[0]
    elif process_id is None:
        issues.append({"status": "NEEDS_METADATA", "message": "New Wafer requires process_run_id"})
    process = batch.get("process_run")
    if process is not None and not isinstance(process, dict):
        issues.append({"status": "INVALID", "message": "process_run must be a metadata dictionary"})
        return issues
    stored = connection.execute(
        "SELECT exposure_value,exposure_unit,recipe_version,processed_at,notes "
        "FROM process_runs WHERE process_run_id=?", (process_id,)).fetchone() if process_id else None
    if process_id and not stored and (not process or process.get("exposure_value") is None):
        issues.append({"status": "NEEDS_METADATA",
                       "message": "New process requires actual exposure_value; no default is generated"})
    if process and process.get("exposure_value") is not None:
        exposure = process["exposure_value"]
        if (isinstance(exposure, bool) or not isinstance(exposure, (int, float))
                or not math.isfinite(exposure)):
            issues.append({"status": "INVALID", "message": "exposure_value must be a finite number"})
    if stored and process:
        for index, field in enumerate(("exposure_value", "exposure_unit", "recipe_version", "processed_at", "notes")):
            if field in process and process[field] != stored[index]:
                issues.append({"status": "INVALID", "message": f"Existing process {field} mismatch"})
    condition_id = batch.get("condition_id")
    if condition_id is not None and (not isinstance(condition_id, str) or not connection.execute(
            "SELECT 1 FROM acquisition_conditions WHERE condition_id=?", (condition_id,)).fetchone()):
        issues.append({"status": "NEEDS_METADATA", "message": "Unknown common condition_id"})
    return issues


def _status(issues, duplicate=False):
    if any(issue["status"] == "INVALID" for issue in issues):
        return "INVALID"
    if issues:
        return "NEEDS_METADATA"
    return "DUPLICATE_CANDIDATE" if duplicate else "VALID"



def _condition_metadata(connection, item, batch, issues):
    condition = item.get("acquisition_condition")
    condition_id = item.get("condition_id", batch.get("condition_id"))
    if condition is not None:
        if not isinstance(condition, dict):
            issues.append({"status":"INVALID","message":"Invalid acquisition_condition"})
            return None, None
        condition_id = condition.get("condition_id")
        issues.extend(_text_issues(condition, ("condition_id",)))
        magnification = condition.get("magnification")
        if magnification is None:
            issues.append({"status":"NEEDS_METADATA","message":"촬영 배율을 입력하세요"})
        elif (isinstance(magnification,bool) or not isinstance(magnification,(int,float))
              or not math.isfinite(magnification) or magnification<=0):
            issues.append({"status":"INVALID","message":"Invalid magnification"})
        auto = condition.get("auto_exposure")
        if auto is not None and (type(auto) is not int or auto not in (0,1)):
            issues.append({"status":"INVALID","message":"Invalid auto_exposure"})
        camera = condition.get("camera_setting_json")
        if camera is not None:
            try:
                json.loads(camera)
            except (TypeError,ValueError):
                issues.append({"status":"INVALID","message":"Invalid camera_setting_json"})
        if not isinstance(condition_id,str):
            return None, condition
        stored = connection.execute(
            "SELECT magnification,illumination_setting,auto_exposure,camera_setting_json,notes "
            "FROM acquisition_conditions WHERE condition_id=?", (condition_id,)).fetchone()
        if stored:
            fields=("magnification","illumination_setting","auto_exposure","camera_setting_json","notes")
            if any(field in condition and condition[field]!=stored[index] for index,field in enumerate(fields)):
                issues.append({"status":"INVALID","message":"Acquisition condition metadata conflict"})
    elif condition_id is not None and (not isinstance(condition_id,str) or not connection.execute(
            "SELECT 1 FROM acquisition_conditions WHERE condition_id=?", (condition_id,)).fetchone()):
        issues.append({"status":"NEEDS_METADATA","message":"Unknown condition_id"})
    if batch.get("upload_mode")=="lighting":
        if condition is not None:
            illumination=condition.get("illumination_setting")
        else:
            stored=connection.execute("SELECT illumination_setting FROM acquisition_conditions WHERE condition_id=?",
                                      (condition_id,)).fetchone() if condition_id else None
            illumination=stored[0] if stored else None
        if not isinstance(illumination,str) or not illumination.strip():
            issues.append({"status":"NEEDS_METADATA","message":"조명 설정을 입력하거나 촬영 조건을 선택하세요"})
    return condition_id, deepcopy(condition)

def _build_preview(connection, inputs, batch):
    common = _common_issues(connection, batch)
    if not inputs:
        common.append({"status": "NEEDS_METADATA", "message": "At least one image is required"})
    rows = []
    ids = Counter(item.get("image_id") for item in inputs if isinstance(item.get("image_id"), str))
    for item in inputs:
        issues = list(common) + _text_issues(item, ("image_id", "image_path"))
        row = {
            "image_id": item.get("image_id"), "image_path": item.get("image_path"),
            "original_filename": None, "sha256": None, "width_px": None, "height_px": None,
            "parsed_metadata": None, "effective_metadata": None, "metadata_source": None,
            "parse_error": None, "issues": issues,
            "duplicate_image_ids": [], "duplicate_input_indices": [],
            "same_site_input_indices": [], "same_site_existing_image_ids": [],
        }
        if isinstance(item.get("image_id"), str):
            if ids[item["image_id"]] > 1:
                issues.append({"status": "INVALID", "message": "Duplicate image_id within Batch"})
            if connection.execute("SELECT 1 FROM images WHERE image_id=?", (item["image_id"],)).fetchone():
                issues.append({"status": "INVALID", "message": "image_id already exists"})
        requested = Path(item["image_path"]) if isinstance(item.get("image_path"), str) else None
        if requested is not None:
            row["original_filename"] = requested.name
            if requested.suffix.lower() != ".png":
                issues.append({"status": "INVALID", "message": "Only PNG files are supported"})
            else:
                try:
                    width, height, digest = _inspect_png(requested.resolve(strict=True))
                    row.update(width_px=width, height_px=height, sha256=digest)
                except (OSError, ImageRegistrationError) as exc:
                    issues.append({"status": "INVALID", "message": str(exc)})
                try:
                    row["parsed_metadata"] = asdict(parse_filename(requested.name))
                except FilenameParseError as exc:
                    row["parse_error"] = str(exc)
                manual = item.get("manual_metadata")
                if manual is not None:
                    row["metadata_source"] = "manual"
                    if not isinstance(manual, dict):
                        issues.append({"status": "INVALID", "message": "manual_metadata must be a dictionary"})
                    else:
                        missing = _text_issues(manual, ("wafer_id", "die_code", "position_code"))
                        issues.extend(missing)
                        if not missing:
                            try:
                                row["effective_metadata"] = asdict(validate_metadata(
                                    wafer_id=manual["wafer_id"], die_code=manual["die_code"],
                                    position_code=manual["position_code"],
                                    legacy_defect_code=manual.get("legacy_defect_code")))
                            except FilenameParseError as exc:
                                issues.append({"status": "INVALID", "message": str(exc)})
                elif row["parsed_metadata"]:
                    row["metadata_source"] = "filename"
                    row["effective_metadata"] = row["parsed_metadata"]
                else:
                    issues.append({"status": "NEEDS_METADATA", "message": "Filename needs manual metadata"})
        effective = row["effective_metadata"]
        if effective:
            row["same_site_existing_image_ids"] = [record[0] for record in connection.execute(
                "SELECT i.image_id FROM images i JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
                "JOIN dies d ON d.die_id=p.die_id "
                "WHERE d.wafer_id=? AND d.die_code=? AND p.position_code=? ORDER BY i.image_id",
                (effective["wafer_id"], effective["die_code"], effective["position_code"]))]
            if effective["wafer_id"] != batch.get("wafer_id"):
                issues.append({"status": "INVALID", "message": "Batch and image Wafer mismatch"})
        condition_id, condition = _condition_metadata(connection, item, batch, issues)
        row["condition_id"] = condition_id
        row["acquisition_condition"] = condition
        if batch.get("upload_mode")=="single_image" and len(inputs)!=1:
            issues.append({"status":"INVALID","message":"Single Image accepts exactly one PNG"})
        if batch.get("upload_mode")=="repeatability":
            repeat=item.get("repeat_index")
            if repeat is None:
                issues.append({"status":"NEEDS_METADATA","message":"반복 번호를 입력하세요"})
            elif type(repeat) is not int or repeat<1:
                issues.append({"status":"INVALID","message":"반복 번호는 양의 정수여야 합니다"})
        if row["sha256"]:
            row["duplicate_image_ids"] = [record[0] for record in connection.execute(
                "SELECT image_id FROM images WHERE lower(sha256)=? ORDER BY image_id", (row["sha256"],))]
        rows.append(row)
    conditions={}
    for row in rows:
        condition=row.get("acquisition_condition")
        if condition and isinstance(condition.get("condition_id"),str):
            key=condition["condition_id"]
            if key in conditions and condition!=conditions[key]:
                for affected in rows:
                    if affected.get("condition_id")==key:
                        affected["issues"].append({"status":"INVALID","message":"Acquisition condition ID conflict"})
            conditions[key]=condition
    if batch.get("upload_mode")=="repeatability":
        sites={tuple(row["effective_metadata"][key] for key in ("wafer_id","die_code","position_code"))
               for row in rows if row["effective_metadata"]}
        repeats=[item.get("repeat_index") for item in inputs]
        bad_repeats=len([value for value in repeats if type(value) is int]) != len(set(
            value for value in repeats if type(value) is int))
        if len(sites)>1 or bad_repeats:
            message="Repeatability는 동일 촬영 위치와 서로 다른 반복 번호가 필요합니다"
            for row in rows:
                row["issues"].append({"status":"INVALID","message":message})
    for index, row in enumerate(rows):
        row["duplicate_input_indices"] = [
            other for other, record in enumerate(rows)
            if other != index and row["sha256"] and record["sha256"] == row["sha256"]]
        effective = row["effective_metadata"]
        if effective:
            key = tuple(effective[field] for field in ("wafer_id", "die_code", "position_code"))
            row["same_site_input_indices"] = [
                other for other, record in enumerate(rows) if other != index and record["effective_metadata"]
                and tuple(record["effective_metadata"][field] for field in
                          ("wafer_id", "die_code", "position_code")) == key]
        row["status"] = _status(row["issues"], bool(
            row["duplicate_image_ids"] or row["duplicate_input_indices"]))
    return {
        "batch_metadata": deepcopy(batch), "inputs": deepcopy(inputs),
        "batch_issues": common, "images": rows,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "registration_ready": bool(rows) and not common and all(row["status"] in
                            ("VALID", "DUPLICATE_CANDIDATE") for row in rows),
        "saved_to_database": False,
    }


def preview_batch(connection, image_inputs, batch_metadata, *, allow_operational=False):
    """Read-only preview, using simple JSON-compatible dictionaries.

    Inputs: paths, or {image_path, image_id, manual_metadata?, condition_id?,
    captured_at?, repeat_index?}. Paths alone need an explicit image_id before
    registration. Batch requires batch_id/wafer_id/upload_mode/dataset_role/
    batch_status. For a new Wafer provide process_run_id; for a new process,
    also provide process_run with its actual exposure_value (no fabricated data).

    Same-site images are allowed; SHA duplicates require separate confirmation.
    Manual fields are validated without ever changing original_filename.
    Existing Die/Site IDs are reused. Only missing rows receive structural IDs
    <wafer_id>/D<die_code> and <die_id>/<position>; collisions are errors.
    PNG QA is the shared signature/IHDR/CRC check, not full image decoding.
    """
    _check_connection(connection, allow_operational=allow_operational)
    inputs = []
    for item in image_inputs:
        entry = {"image_path": item} if isinstance(item, (str, Path)) else deepcopy(item)
        if not isinstance(entry, dict):
            raise BatchRegistrationError("Each image input must be a path or dictionary")
        if isinstance(entry.get("image_path"), (str, Path)):
            entry["image_path"] = str(Path(entry["image_path"]).expanduser().absolute())
        inputs.append(entry)
    if not isinstance(batch_metadata, dict):
        raise BatchRegistrationError("Batch metadata must be a dictionary")
    return _build_preview(connection, inputs, deepcopy(batch_metadata))



def _build_registration_plan(connection, source_preview, decisions):
    current = _build_preview(connection, source_preview["inputs"], source_preview["batch_metadata"])
    if current != source_preview:
        raise BatchRegistrationError("Source Preview changed; review duplicate decisions again")
    if any(not isinstance(row["image_id"], str) or not row["image_id"].strip()
           for row in current["images"]):
        raise BatchRegistrationError("Image IDs are required before duplicate selection")
    image_ids = [row["image_id"] for row in current["images"]]
    if len(image_ids) != len(set(image_ids)):
        raise BatchRegistrationError("Image IDs must be unique before duplicate selection")
    duplicate_ids = {row["image_id"] for row in current["images"]
                     if row["duplicate_image_ids"] or row["duplicate_input_indices"]}
    if decisions is None:
        decisions = {}
    if not isinstance(decisions, dict) or set(decisions) - duplicate_ids:
        raise BatchRegistrationError("Duplicate decisions must refer only to current duplicate candidates")
    if any(action not in DUPLICATE_ACTIONS for action in decisions.values()):
        raise BatchRegistrationError("Unknown duplicate action")
    actions = {row["image_id"]: decisions.get(row["image_id"], "EXCLUDE")
               for row in current["images"] if row["image_id"] in duplicate_ids}
    excluded = [image_id for image_id, action in actions.items() if action == "EXCLUDE"]
    separate = [image_id for image_id, action in actions.items() if action == "REGISTER_SEPARATELY"]
    held = [image_id for image_id, action in actions.items() if action == "HOLD"]
    selected_inputs = [item for item in current["inputs"] if item["image_id"] not in excluded]
    # Rebuild through the SAME validator. Never delete rows from a validated
    # Preview: removing an internal duplicate can change the remaining statuses.
    final = _build_preview(connection, selected_inputs, current["batch_metadata"])
    return {"source_preview": deepcopy(source_preview), "decisions": actions,
            "final_preview": final, "excluded_image_ids": excluded,
            "separate_image_ids": separate, "held_image_ids": held,
            "uploaded_count": len(current["images"]), "selected_count": len(selected_inputs),
            "registration_ready": bool(selected_inputs) and not held and final["registration_ready"]}


def prepare_registration(connection, source_preview, decisions=None, *, allow_operational=False):
    """Read-only policy plan. EXCLUDE is the default; HOLD blocks the whole Batch.

    Image IDs in staging are draft identifiers. Excluded drafts never become DB
    records or permanent copies. All original uploaded inputs remain in the source
    Preview so stale decisions can also be detected for excluded files.
    """
    _check_connection(connection, allow_operational=allow_operational)
    return _build_registration_plan(connection, source_preview, decisions)


def register_selection(connection, plan, *, approved=False, storage_root=None, allow_operational=False):
    """Register only a revalidated selection via the existing atomic Batch path."""
    if approved is not True:
        raise BatchRegistrationError("Explicit registration approval is required")
    if plan["held_image_ids"]:
        raise BatchRegistrationError("HOLD candidates block the whole Batch")
    if not plan["selected_count"]:
        raise BatchRegistrationError("No images selected for registration")
    if not plan["registration_ready"]:
        raise BatchRegistrationError("Unresolved INVALID/NEEDS_METADATA or Batch issues")
    return register_batch(
        connection, plan["final_preview"], approved=approved,
        approved_duplicate_image_ids=plan["separate_image_ids"],
        storage_root=storage_root, allow_operational=allow_operational, _selection_plan=plan)

def _prepare_parents(connection, batch, rows):
    wafer_id = batch["wafer_id"]
    wafer = connection.execute("SELECT process_run_id FROM wafers WHERE wafer_id=?", (wafer_id,)).fetchone()
    if not wafer:
        process_id = batch["process_run_id"]
        if not connection.execute("SELECT 1 FROM process_runs WHERE process_run_id=?", (process_id,)).fetchone():
            process = batch["process_run"]
            connection.execute(
                "INSERT INTO process_runs(process_run_id,exposure_value,exposure_unit,recipe_version,processed_at,notes) "
                "VALUES (?,?,?,?,?,?)",
                (process_id, process["exposure_value"], process.get("exposure_unit"),
                 process.get("recipe_version"), process.get("processed_at"), process.get("notes")))
        connection.execute("INSERT INTO wafers(wafer_id,process_run_id) VALUES (?,?)", (wafer_id, process_id))
    connection.execute(
        "INSERT INTO upload_batches(batch_id,wafer_id,upload_mode,dataset_role,batch_status,notes) VALUES (?,?,?,?,?,?)",
        tuple(batch[name] for name in ("batch_id", "wafer_id", "upload_mode", "dataset_role", "batch_status"))
        + (batch.get("notes"),))
    sites = {}
    for row in rows:
        condition=row.get("acquisition_condition")
        if condition and not connection.execute(
                "SELECT 1 FROM acquisition_conditions WHERE condition_id=?", (condition["condition_id"],)).fetchone():
            connection.execute(
                "INSERT INTO acquisition_conditions(condition_id,magnification,illumination_setting,auto_exposure,"
                "camera_setting_json,notes) VALUES (?,?,?,?,?,?)",
                tuple(condition.get(key) for key in ("condition_id","magnification","illumination_setting",
                                                    "auto_exposure","camera_setting_json","notes")))
        metadata = row["effective_metadata"]
        die_code, position = metadata["die_code"], metadata["position_code"]
        die = connection.execute(
            "SELECT die_id FROM dies WHERE wafer_id=? AND die_code=?", (wafer_id, die_code)).fetchone()
        die_id = die[0] if die else f"{wafer_id}/D{die_code}"
        if not die:
            connection.execute("INSERT INTO dies(die_id,wafer_id,die_code,row_index,column_index) VALUES (?,?,?,?,?)",
                               (die_id, wafer_id, die_code, metadata["row_index"], metadata["column_index"]))
        site = connection.execute(
            "SELECT physical_site_id FROM pattern_sites WHERE die_id=? AND position_code=?", (die_id, position)).fetchone()
        site_id = site[0] if site else f"{die_id}/{position}"
        if not site:
            connection.execute("INSERT INTO pattern_sites(physical_site_id,die_id,position_code) VALUES (?,?,?)",
                               (site_id, die_id, position))
        sites[(die_code, position)] = site_id
    return sites


def register_batch(connection, preview, *, approved=False, approved_duplicate_image_ids=(),
                   storage_root=None, allow_operational=False, _selection_plan=None):
    """Register a reviewed preview in ONE transaction, rechecking under a write lock.

    approved must be True. Every duplicate-candidate image_id must additionally
    appear in approved_duplicate_image_ids. A stale/edited preview or source file
    is refused. Existing records are never replaced/deleted/merged. Files remain
    at their caller-provided locations unless DB-1D supplies storage_root.
    With storage_root, only new owned copies are cleaned on failure. The default
    DB-1B/C operational guard remains; DB-1D explicitly opts in after approval.
    """
    _check_connection(connection, allow_operational=allow_operational)
    if allow_operational and storage_root is None:
        raise BatchRegistrationError("Operational registration requires managed image storage")
    if approved is not True:
        raise BatchRegistrationError("Explicit registration approval is required")
    if not preview.get("registration_ready"):
        raise BatchRegistrationError("Unresolved INVALID/NEEDS_METADATA or Batch issues")
    if isinstance(approved_duplicate_image_ids, (str, bytes, bool)):
        raise BatchRegistrationError("Duplicate confirmation must list individual image IDs")
    confirmed = set(approved_duplicate_image_ids)
    storage = NewFiles(storage_root) if storage_root is not None else None
    try:
        connection.execute("BEGIN IMMEDIATE")
        if _selection_plan is not None:
            checked_plan = _build_registration_plan(
                connection, _selection_plan["source_preview"], _selection_plan["decisions"])
            if (checked_plan != _selection_plan or not checked_plan["registration_ready"]
                    or checked_plan["final_preview"] != preview
                    or set(checked_plan["separate_image_ids"]) != confirmed):
                raise BatchRegistrationError("Registration selection changed; review and approve again")
        current = _build_preview(connection, preview["inputs"], preview["batch_metadata"])
        if current != preview:
            raise BatchRegistrationError("Preview changed; review again before registering")
        if not current["registration_ready"]:
            raise BatchRegistrationError("Unresolved metadata")
        duplicates = {row["image_id"] for row in current["images"] if row["status"] == "DUPLICATE_CANDIDATE"}
        if not duplicates <= confirmed:
            raise BatchRegistrationError("Duplicate candidates require explicit per-image confirmation")
        sites = _prepare_parents(connection, current["batch_metadata"], current["images"])
        image_ids = []
        for item, row in zip(current["inputs"], current["images"]):
            metadata = row["effective_metadata"]
            parsed = validate_metadata(**{key: metadata[key] for key in
                ("wafer_id", "die_code", "position_code", "legacy_defect_code")})
            image_path = item["image_path"]
            if storage is not None:
                image_path = storage.copy(image_path, row["original_filename"], row["sha256"])
            stored = register_image(
                connection, image_path, image_id=row["image_id"],
                batch_id=current["batch_metadata"]["batch_id"],
                physical_site_id=sites[(parsed.die_code, parsed.position_code)],
                metadata_status="manual_confirmed" if row["metadata_source"] == "manual" else "filename_parsed",
                condition_id=row["condition_id"], captured_at=item.get("captured_at"),
                repeat_index=item.get("repeat_index"),
                parsed_metadata=parsed, manage_transaction=False,
                allow_operational=allow_operational)
            if stored["sha256"] != row["sha256"]:
                raise BatchRegistrationError("Image bytes changed since preview")
            image_ids.append(stored["image_id"])
        connection.commit()
        if storage is not None:
            storage.preserve()
        result = {"batch_id": current["batch_metadata"]["batch_id"],
                  "image_count": len(image_ids), "image_ids": image_ids}
        if _selection_plan is not None:
            result.update(
                uploaded_count=_selection_plan["uploaded_count"], registered_count=len(image_ids),
                duplicate_registered_count=len(set(image_ids) & set(_selection_plan["separate_image_ids"])),
                excluded_count=len(_selection_plan["excluded_image_ids"]),
                excluded_image_ids=list(_selection_plan["excluded_image_ids"]))
        return result
    except Exception as exc:
        if connection.in_transaction:
            connection.rollback()
        if storage is not None:
            storage.cleanup()
        if isinstance(exc, sqlite3.Error):
            raise BatchRegistrationError(f"Batch registration failed: {exc}") from exc
        raise
