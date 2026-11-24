"""Registered-image -> BatchRunner request candidates; never dispatches an engine.

Statuses describe input readiness only, never upload/quality/official DB status.
Approved profiles require exact Batch/Wafer/condition/image scope and matching
capture-setting evidence, or an approved new-upload default condition whose stored
condition row/notes match exactly. A 20x value alone is not sufficient. No engine fallback
calibration is inferred, and execution remains disconnected.
Use SQLiteProvider.prepare_analysis_inputs with the ID from target_preview.
"""
from pathlib import Path
import hashlib
import math
from database.image_registry import get_image
from inspection.calibration_profiles import load_profiles,load_upload_defaults,resolve_calibration

def _identity(value,name):
    if not isinstance(value,str) or not value.strip():
        raise ValueError(name+" must be an actual nonempty registered string ID")
    if value.upper().startswith(("RAW:","QA:")):
        raise ValueError(name+" is a QA/raw identifier, not a registered analysis target")
    return value

def _issue(code,message,level="BLOCKED",field=None):
    result={"code":code,"message":message,"level":level}
    if field is not None:result["field"]=field
    return result

def prepare_batch_inputs(connection,batch_id,*,storage_root,image_ids=None):
    """Read-only input plan. candidate_request is NOT permission to execute.

    ready_requests is the future dispatch list, populated only after an approved
    calibration profile and all metadata/file checks match. Identical SHA
    values are not deduplicated. Hashes are sidecar metadata, because the actual
    BatchRunner FIELDS does not accept a source_sha256/image_sha256 request key.
    """
    _identity(batch_id,"batch_id")
    if connection.execute("PRAGMA query_only").fetchone()[0]!=1:
        raise ValueError("Input preparation requires query_only=ON")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0]!=1:
        raise ValueError("Input preparation requires foreign_keys=ON")
    from database.migrate_v5 import ANALYSIS_VERSIONS, effective_process_join
    if connection.execute("PRAGMA user_version").fetchone()[0] not in ANALYSIS_VERSIONS:
        raise ValueError("Input preparation requires the validated v4/v5 dataset")
    process_join, _ = effective_process_join(connection)
    cur=connection.execute(
        "SELECT b.batch_id,b.wafer_id,b.upload_mode,b.dataset_role,b.peb_temperature_c,"
        "r.exposure_value,r.exposure_unit FROM upload_batches b "
        "JOIN wafers w ON w.wafer_id=b.wafer_id " + process_join +
        "WHERE b.batch_id=?",(batch_id,))
    found=cur.fetchone()
    if found is None:raise ValueError("Registered Batch not found")
    batch=dict(zip((d[0] for d in cur.description),found))
    # Each image's OWN Wafer and that Batch-Wafer's process (v6); v5 and older: the Batch process.
    from database.migrate_v6 import image_process_sql
    version=connection.execute("PRAGMA user_version").fetchone()[0]
    own_join,_,own_peb=image_process_sql(connection)
    own={row[0]:dict(zip(("wafer_id","exposure_value","exposure_unit","peb_temperature_c"),row[1:])) for row in connection.execute(
        "SELECT i.image_id,d.wafer_id,r.exposure_value,r.exposure_unit,"+own_peb+" FROM images i "
        "JOIN upload_batches b ON b.batch_id=i.batch_id JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
        "JOIN dies d ON d.die_id=p.die_id "+own_join+"WHERE i.batch_id=?",(batch_id,))}
    if image_ids is None:
        ids=[row[0] for row in connection.execute("SELECT image_id FROM images WHERE batch_id=? ORDER BY image_id",(batch_id,))]
    else:
        if not isinstance(image_ids,(list,tuple)):raise ValueError("image_ids must be a list/tuple of registered IDs")
        ids=list(image_ids)
        for image_id in ids:_identity(image_id,"image_id")
        if len(ids)!=len(set(ids)):raise ValueError("Repeated image_id selection; equal SHA is allowed, repeated PK is not")
    for image_id in ids:_identity(image_id,"image_id")
    root=Path(storage_root).expanduser().resolve()
    profiles=load_profiles()
    upload_defaults=load_upload_defaults(profiles=profiles)
    supported_magnifications={20}|{p["magnification"] for p in profiles+upload_defaults}
    items=[];file_hashes={}
    for image_id in ids:
        issues=[];candidate=None;actual_hash=None
        item={"image_id":image_id,"candidate_request":None,"validation_status":"BLOCKED",
              "issues":issues,"registered_sha256":None,"actual_sha256":None}
        row=get_image(connection,image_id)
        if row is None:
            issues.append(_issue("IMAGE_NOT_REGISTERED","등록된 Image ID를 확인하세요."))
        elif row["batch_id"]!=batch_id:
            issues.append(_issue("BATCH_IMAGE_MISMATCH","선택한 이미지가 해당 Batch에 속하지 않습니다."))
        else:
            item["registered_sha256"]=row["sha256"]
            site=connection.execute(
                "SELECT d.wafer_id,d.die_id,p.position_code FROM pattern_sites p "
                "JOIN dies d ON d.die_id=p.die_id WHERE p.physical_site_id=?",
                (row["physical_site_id"],)).fetchone()
            process=own.get(image_id) or {"wafer_id":None,"exposure_value":None,"exposure_unit":None,"peb_temperature_c":None}
            expected_wafer=process["wafer_id"] if version>=6 else batch["wafer_id"]
            if site is None or expected_wafer is None or tuple(site)!=(expected_wafer,row["die_id"],row["position_code"]):
                issues.append(_issue("BATCH_SITE_WAFER_MISMATCH","Batch와 촬영 위치의 Wafer/Die 관계를 확인하세요."))
            condition=None
            if row["condition_id"] is not None:
                cur=connection.execute("SELECT magnification,illumination_setting,auto_exposure,notes "
                    "FROM acquisition_conditions WHERE condition_id=?",(row["condition_id"],))
                values=cur.fetchone()
                if values is None:issues.append(_issue("ACQUISITION_CONDITION_MISSING","촬영 조건 연결을 확인하세요."))
                else:condition=dict(zip((d[0] for d in cur.description),values))
            candidate={
              "batch_id":batch_id,"image_id":image_id,"image_path":row["storage_path"],
              "wafer_id":expected_wafer,"die_id":row["die_id"],"site_code":row["position_code"],
              "physical_site_id":row["physical_site_id"],"die_code":row["die_code"],
              "condition_id":row["condition_id"],"original_filename":row["original_filename"],
              "exposure_value":process["exposure_value"],"exposure_unit":process["exposure_unit"],
              "peb_temperature_c":process["peb_temperature_c"],"dataset_role":batch["dataset_role"],
              "upload_mode":batch["upload_mode"],"magnification":condition["magnification"] if condition else None,
              "repeat_index":row["repeat_index"],
              "illumination_setting":condition["illumination_setting"] if condition else None,
              "auto_exposure":condition["auto_exposure"] if condition else None,
              "calibration":None}
            for key in ("wafer_id","die_id","site_code","physical_site_id","image_path","original_filename"):
                if not isinstance(candidate[key],str) or not candidate[key].strip():
                    issues.append(_issue("MISSING_REQUIRED_METADATA","필수 등록 Metadata를 보완하세요.",field=key))
            if candidate["site_code"] not in ("lt","c","rb"):
                issues.append(_issue("INVALID_SITE","촬영 위치 lt/c/rb를 확인하세요.",field="site_code"))
            for key in ("exposure_value","peb_temperature_c","magnification"):
                value=candidate[key]
                if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):
                    issues.append(_issue("INVALID_NUMERIC_METADATA","촬영 조건의 실제 숫자값을 확인하세요.",field=key))
            magnification=candidate["magnification"]
            if magnification is None:
                issues.append(_issue("MISSING_REQUIRED_METADATA","실제 촬영 배율이 미기록입니다.",field="magnification"))
            elif not isinstance(magnification,bool) and isinstance(magnification,(int,float)) and magnification not in supported_magnifications:
                issues.append(_issue("UNSUPPORTED_MAGNIFICATION","현재 검증 범위에 없는 배율입니다.",field="magnification"))
            matched=resolve_calibration(profiles,batch_id=batch_id,image_id=image_id,
                wafer_id=expected_wafer,condition_id=row["condition_id"],magnification=magnification,
                condition_notes=condition["notes"] if condition else None,upload_defaults=upload_defaults)
            if matched is None:
                issues.append(_issue("CALIBRATION_REVIEW_REQUIRED","해당 이미지에 적용할 Calibration 검증 근거가 연결되지 않았습니다.","REVIEW_REQUIRED"))
            else:
                candidate["calibration"]=matched["calibration"]
                item["calibration_evidence"]=matched
            if batch["upload_mode"] in ("repeatability","lighting"):
                issues.append(_issue("ANALYSIS_SCOPE_REVIEW_REQUIRED","반복/조명 이미지의 공식 분석 대상 여부는 확인이 필요합니다.","REVIEW_REQUIRED"))
            stored=row["storage_path"]
            if not isinstance(stored,str) or not stored or "://" in stored or stored.startswith(("//","\\\\")):
                issues.append(_issue("INVALID_LOCAL_PATH","등록된 로컬 원본 경로를 확인하세요.",field="image_path"))
            else:
                path=Path(stored).expanduser()
                path=(path if path.is_absolute() else root/path).resolve()
                if not path.is_relative_to(root):
                    issues.append(_issue("PATH_OUTSIDE_STORAGE","등록 이미지 저장소 밖의 경로는 사용할 수 없습니다."))
                else:
                    candidate["image_path"]=str(path)
                    try:
                        if not path.is_file():
                            issues.append(_issue("IMAGE_NOT_FOUND","등록된 원본 이미지 파일이 없습니다."))
                        else:
                            if path not in file_hashes:
                                with path.open("rb") as stream:file_hashes[path]=hashlib.file_digest(stream,"sha256").hexdigest()
                            actual_hash=file_hashes[path]
                            item["actual_sha256"]=actual_hash
                            if not isinstance(row["sha256"],str) or actual_hash.lower()!=row["sha256"].lower():
                                issues.append(_issue("SHA_MISMATCH","원본 파일과 등록 SHA-256이 일치하지 않습니다."))
                    except OSError:
                        issues.append(_issue("IMAGE_UNREADABLE","원본 파일을 읽을 수 없습니다."))
            item["candidate_request"]=candidate
        item["validation_status"]=("BLOCKED" if any(i["level"]=="BLOCKED" for i in issues) else "REVIEW_REQUIRED" if issues else "READY")
        items.append(item)
    counts={status:sum(i["validation_status"]==status for i in items) for status in ("READY","REVIEW_REQUIRED","BLOCKED")}
    return {"batch_id":batch_id,"validation_scope":"RUNNER_INPUT_ONLY","items":items,
      "counts":counts,"selected_image_count":len(items),"files_hashed":len(file_hashes),
      "status":"BLOCKED" if not items or counts["BLOCKED"] else "REVIEW_REQUIRED" if counts["REVIEW_REQUIRED"] else "READY",
      "batch_issues":[] if items else [_issue("NO_IMAGES_SELECTED","선택된 등록 이미지가 없습니다.")],
      "ready_requests":[i["candidate_request"] for i in items if i["validation_status"]=="READY"],"execution_enabled":False,
      "calibration_policy":"Exact approved profile scope and capture-setting evidence required; or approved upload-default condition; no 20x-only/engine fallback."}
