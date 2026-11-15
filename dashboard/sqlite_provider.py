"""Registration-only SQLite boundary; Mock provider and analysis pages stay separate."""
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path

import pandas as pd

from database.batch_registry import preview_batch, register_batch, prepare_registration, register_selection
from database.init_db import DEFAULT_DB, PROJECT_ROOT, connect_database, initialize_database
from database.upload_storage import stage_uploads, upload_signature
from database.migrate_v2 import validate_v2
from database.batch_names import batch_name_from_notes,update_batch_name


@dataclass(frozen=True)
class SQLiteProvider:
    db_path: Path
    storage_root: Path
    staging_root: Path

    def connect(self, *, readonly=True):
        path = Path(self.db_path).resolve()
        if not path.is_file():
            raise FileNotFoundError("SQLite DB가 없습니다. 명시적 초기화가 필요합니다.")
        connection=connect_database(path,readonly=True)
        try:
            version=connection.execute("PRAGMA user_version").fetchone()[0]
            if version==1:
                initialize_database(path)  # Frozen v1 path; never migrates.
            elif version==2:
                validate_v2(connection)
            else:
                raise RuntimeError("지원하지 않는 Schema Version입니다.")
            if readonly:
                return connection
        except Exception:
            connection.close()
            raise
        connection.close()
        return connect_database(path,readonly=False)

    def schema_info(self):
        with closing(self.connect()) as connection:
            version=connection.execute("PRAGMA user_version").fetchone()[0]
        return {"version":version,"single_image_supported":version>=2,
                "display_name":os.environ.get("IDA_DB_LABEL","로컬 SQLite")+" / "+Path(self.db_path).name}


    def stage(self, uploads):
        return stage_uploads(uploads, self.staging_root)

    def preview(self, inputs, metadata):
        with closing(self.connect()) as connection:
            return preview_batch(connection, inputs, metadata, allow_operational=True)

    def register(self, preview, *, approved=False, duplicate_ids=()):
        if approved is not True:
            raise ValueError("명시적 등록 승인이 필요합니다.")
        with closing(self.connect(readonly=False)) as connection:
            return register_batch(connection, preview, approved=approved,
                                  approved_duplicate_image_ids=duplicate_ids,
                                  storage_root=self.storage_root, allow_operational=True)

    def prepare_registration(self, preview, decisions):
        with closing(self.connect()) as connection:
            return prepare_registration(connection, preview, decisions, allow_operational=True)

    def register_selection(self, plan, *, approved=False):
        if approved is not True:
            raise ValueError("명시적 등록 승인이 필요합니다.")
        with closing(self.connect(readonly=False)) as connection:
            return register_selection(connection, plan, approved=approved,
                                      storage_root=self.storage_root, allow_operational=True)

    def batches(self):
        with closing(self.connect()) as connection:
            frame=pd.read_sql_query(
                "SELECT b.batch_id,b.wafer_id,b.upload_mode,b.dataset_role,b.batch_status,b.created_at,b.notes AS batch_notes,"
                "r.exposure_value,r.exposure_unit,COUNT(i.image_id) AS image_count "
                "FROM upload_batches b JOIN wafers w ON w.wafer_id=b.wafer_id "
                "JOIN process_runs r ON r.process_run_id=w.process_run_id "
                "LEFT JOIN images i ON i.batch_id=b.batch_id GROUP BY b.batch_id "
                "ORDER BY b.created_at,b.batch_id",connection)
        frame["automatic_display_name"]=[f"{wafer} · Batch {index:02}"
            for wafer,index in zip(frame["wafer_id"],frame.groupby("wafer_id").cumcount()+1)]
        frame["batch_name"]=frame["batch_notes"].map(batch_name_from_notes)
        frame["display_name"]=[automatic+(" · "+name if name else "")
            for automatic,name in zip(frame["automatic_display_name"],frame["batch_name"])]
        return frame.iloc[::-1].reset_index(drop=True)

    def rename_batch(self,batch_id,name):
        with closing(self.connect(readonly=False)) as connection:
            return update_batch_name(connection,batch_id,name,allow_operational=True)

    def image_count(self, batch_id=None, upload_mode=None):
        where,params=self._image_filter(batch_id,upload_mode)
        with closing(self.connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id "
                                      +where,params).fetchone()[0]

    @staticmethod
    def _image_filter(batch_id,upload_mode):
        clauses=[]
        params=[]
        if batch_id is not None:
            clauses.append("i.batch_id=?")
            params.append(batch_id)
        if upload_mode is not None:
            clauses.append("b.upload_mode=?")
            params.append(upload_mode)
        return ("WHERE "+" AND ".join(clauses) if clauses else ""),params

    def images(self, batch_id=None, *, upload_mode=None, limit=25, offset=0):
        if type(limit) is not int or not 1<=limit<=100 or type(offset) is not int or offset<0:
            raise ValueError("Invalid image page")
        where,params=self._image_filter(batch_id,upload_mode)
        with closing(self.connect()) as connection:
            return pd.read_sql_query(
                "SELECT i.image_id,i.batch_id,i.original_filename,i.storage_path,i.sha256,"
                "i.width_px,i.height_px,i.metadata_status,i.created_at,i.repeat_index,i.condition_id,"
                "d.wafer_id,d.die_code,p.position_code,i.legacy_defect_code,b.upload_mode,"
                "a.illumination_setting,a.magnification,a.auto_exposure "
                "FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id "
                "JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
                "JOIN dies d ON d.die_id=p.die_id "
                "LEFT JOIN acquisition_conditions a ON a.condition_id=i.condition_id "
                +where+" ORDER BY i.created_at DESC,i.image_id LIMIT ? OFFSET ?",
                connection,params=(*params,limit,offset))


    def duplicate_records(self, image_ids):
        """Resolve IDs already reported by DB-1C; do not implement a new hash scan."""
        ids = sorted(set(image_ids))
        if not ids:
            return []
        with closing(self.connect()) as connection:
            cursor = connection.execute(
                "SELECT image_id,batch_id,original_filename,sha256 FROM images "
                "WHERE image_id IN (" + ",".join("?" for _ in ids) + ") ORDER BY image_id",
                ids)
            return [dict(zip(("image_id","batch_id","original_filename","sha256"), row))
                    for row in cursor]

    def wafer_process(self, wafer_id):
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT w.process_run_id,r.exposure_value,r.exposure_unit "
                "FROM wafers w JOIN process_runs r ON r.process_run_id=w.process_run_id "
                "WHERE w.wafer_id=?", (wafer_id,)).fetchone()
            return dict(zip(("process_run_id","exposure_value","exposure_unit"), row)) if row else None


def runtime_provider():
    """Tests override all three paths; no import-time DB/file access or creation."""
    return SQLiteProvider(
        Path(os.environ.get("IDA_DB_PATH", DEFAULT_DB)),
        Path(os.environ.get("IDA_STORAGE_ROOT", PROJECT_ROOT/"runtime_data"/"raw")),
        Path(os.environ.get("IDA_STAGING_ROOT", PROJECT_ROOT/"runtime_data"/"uploads_tmp")))


def sync_uploads(state, uploads, provider):
    """Content-scoped draft. A changed upload list discards only its own old staging."""
    signature = upload_signature(uploads)
    old = state.get("sqlite_uploads")
    if old is not None and old["signature"] == signature:
        return old
    if old is not None:
        old["files"].cleanup()
    state.pop("sqlite_uploads", None)
    state.pop("sqlite_preview", None)
    if not uploads:
        return None
    bundle = provider.stage(uploads)
    bundle["manual_metadata"] = {}
    bundle["per_image_metadata"] = {}
    state["sqlite_uploads"] = bundle
    return bundle


def reviewed_inputs(bundle):
    result = deepcopy(bundle["inputs"])
    for item in result:
        item.update(deepcopy(bundle.get("per_image_metadata",{}).get(item["image_id"],{})))
        if item["image_id"] in bundle["manual_metadata"]:
            item["manual_metadata"] = deepcopy(bundle["manual_metadata"][item["image_id"]])
    return result
