"""Explicit approved Batch condition binding; no image/analysis creation or schema change."""
from pathlib import Path
import hashlib,json
from database.init_db import DEFAULT_DB,_schema_objects
from database.migrate_v2 import data_snapshot
from database.migrate_v4 import validate_v4
from copy import deepcopy
from inspection.calibration_profiles import (validate_profile,condition_note,load_upload_defaults,
    select_upload_default,upload_default_condition)

def apply_upload_default(connection,inputs,batch_metadata,*,defaults=None):
    """Read-only: attach the approved new-upload default condition before Preview.

    The returned inputs flow through the unchanged preview/registration validators,
    which also create the shared condition row once and reject any stored-row conflict.
    Applied only on Schema v4, only for modes listed by exactly one approved default,
    and only to images with no explicit condition_id/acquisition_condition (item or
    Batch). Lighting and any explicit capture setting are never overridden. Existing
    registered images are never touched.
    """
    result=deepcopy(list(inputs))
    if not isinstance(batch_metadata,dict) or batch_metadata.get("condition_id") is not None:return result
    if connection.execute("PRAGMA user_version").fetchone()[0] not in (4,5,6):return result
    default=select_upload_default(load_upload_defaults() if defaults is None else defaults,
                                  batch_metadata.get("upload_mode"))
    if default is None:return result
    condition=upload_default_condition(default)
    for item in result:
        if isinstance(item,dict) and item.get("condition_id") is None and item.get("acquisition_condition") is None:
            item["acquisition_condition"]=dict(condition)
    return result

def data_fingerprint(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def bind_batch_condition(connection,batch_id,profile,*,approved=False,allow_operational=False,expected_data_sha256=None):
    """Own one transaction. Only exact approved image IDs' condition_id may change."""
    if approved is not True:raise ValueError("Explicit metadata binding approval required")
    profile=validate_profile(profile)
    if profile["approved"] is not True:raise ValueError("Profile is not approved")
    scope=profile["scope"]
    if scope["batch_ids"]!=[batch_id] or len(scope["condition_ids"])!=1:
        raise ValueError("Binding requires exactly one approved Batch and common condition")
    if connection.in_transaction:raise ValueError("Finish caller transaction before binding")
    main=Path(next(r[2] for r in connection.execute("PRAGMA database_list") if r[1]=="main")).resolve()
    if not allow_operational and (main==DEFAULT_DB.resolve() or (DEFAULT_DB.exists() and main.samefile(DEFAULT_DB))):
        raise PermissionError("Operational metadata binding requires explicit opt-in")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0]!=1:raise ValueError("foreign_keys must be ON")
    validate_v4(connection)
    condition_id=scope["condition_ids"][0]
    note=condition_note(profile,batch_id)
    expected_condition=(condition_id,float(profile["magnification"]),None,None,None,note)
    try:
        connection.execute("BEGIN IMMEDIATE")
        before=data_snapshot(connection);objects=_schema_objects(connection)
        if expected_data_sha256 is not None and data_fingerprint(before)!=expected_data_sha256:
            raise ValueError("DB changed after the verified backup")
        batch=connection.execute("SELECT wafer_id FROM upload_batches WHERE batch_id=?",(batch_id,)).fetchone()
        if batch is None or batch[0] not in scope["wafer_ids"]:raise ValueError("Approved Batch/Wafer mismatch")
        selected=connection.execute("SELECT image_id,condition_id FROM images WHERE batch_id=? ORDER BY image_id",(batch_id,)).fetchall()
        if {r[0] for r in selected}!=set(scope["image_ids"]):raise ValueError("Approved image membership/count changed")
        if any(r[1] not in (None,condition_id) for r in selected):raise ValueError("Existing unapproved condition; do not overwrite")
        existing=connection.execute("SELECT * FROM acquisition_conditions WHERE condition_id=?",(condition_id,)).fetchone()
        created=existing is None
        if existing is not None and tuple(existing)!=expected_condition:raise ValueError("Condition ID conflict")
        if created:
            connection.execute("INSERT INTO acquisition_conditions VALUES (?,?,?,?,?,?)",expected_condition)
        updated=sum(r[1] is None for r in selected)
        if updated:
            cur=connection.execute("UPDATE images SET condition_id=? WHERE batch_id=? AND condition_id IS NULL",(condition_id,batch_id))
            if cur.rowcount!=updated:raise ValueError("Unexpected image update count")
        after=data_snapshot(connection)
        for table in before.keys()-{"images","acquisition_conditions"}:
            if after[table]!=before[table]:raise ValueError("Unrelated table changed")
        columns=[r[1] for r in connection.execute("PRAGMA table_info(images)")]
        ix=columns.index("condition_id");bx=columns.index("batch_id")
        expected_images=[]
        for row in before["images"]:
            value=list(row)
            if row[bx]==batch_id and value[ix] is None:value[ix]=condition_id
            expected_images.append(tuple(value))
        if after["images"]!=expected_images:raise ValueError("Image fields outside condition_id changed")
        expected_conditions={r[0]:r for r in before["acquisition_conditions"]}
        expected_conditions[condition_id]=expected_condition
        if {r[0]:r for r in after["acquisition_conditions"]}!=expected_conditions:raise ValueError("Other acquisition conditions changed")
        if _schema_objects(connection)!=objects:raise ValueError("Schema objects changed")
        validate_v4(connection)
        connection.commit()
        return {"batch_id":batch_id,"condition_id":condition_id,"image_count":len(selected),
          "updated_images":updated,"condition_created":created,"profile_id":profile["profile_id"]}
    except BaseException:
        if connection.in_transaction:connection.rollback()
        raise
