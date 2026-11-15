"""Optional Batch display names in existing v1/v2 upload_batches.notes.

The reserved JSON envelope preserves the original notes verbatim (including
NULL). Removing the display name restores those notes. No schema migration,
image mutation, or automatic backfill is performed.
"""
import json

from database.image_registry import _check_connection

_FORMAT="ida.batch_display_name.v1"


def normalize_batch_name(name):
    if name is None:
        return ""
    if not isinstance(name,str):
        raise ValueError("Batch 이름은 문자열이어야 합니다.")
    if any(ord(char)<32 or ord(char)==127 for char in name):
        raise ValueError("Batch 이름은 줄바꿈 없는 한 줄로 입력하세요.")
    return name.strip()


def _envelope(notes):
    if not isinstance(notes,str):
        return None
    try:
        value=json.loads(notes)
    except (ValueError,RecursionError):
        return None
    if (not isinstance(value,dict) or set(value)!={"format","batch_name","original_notes"}
            or value["format"]!=_FORMAT or not isinstance(value["batch_name"],str)
            or not (value["original_notes"] is None or isinstance(value["original_notes"],str))):
        return None
    return value


def batch_name_from_notes(notes):
    value=_envelope(notes)
    return value["batch_name"] if value is not None else ""


def notes_with_batch_name(notes,name):
    """Change only this reserved metadata, preserving legacy text/JSON notes."""
    name=normalize_batch_name(name)
    value=_envelope(notes)
    original=value["original_notes"] if value is not None else notes
    if not name:
        return original
    return json.dumps({"format":_FORMAT,"batch_name":name,"original_notes":original},
                      ensure_ascii=False,separators=(",",":"))


def update_batch_name(connection,batch_id,name,*,allow_operational=False):
    """Explicit save only. One transaction updates notes; IDs/images are untouched."""
    _check_connection(connection,allow_operational=allow_operational)
    name=normalize_batch_name(name)
    try:
        connection.execute("BEGIN IMMEDIATE")
        row=connection.execute("SELECT notes FROM upload_batches WHERE batch_id=?",(batch_id,)).fetchone()
        if row is None:
            raise ValueError("선택한 Batch가 없습니다. 목록을 새로 확인하세요.")
        notes=notes_with_batch_name(row[0],name)
        connection.execute("UPDATE upload_batches SET notes=? WHERE batch_id=?",(notes,batch_id))
        connection.commit()
        return {"batch_id":batch_id,"batch_name":name}
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
