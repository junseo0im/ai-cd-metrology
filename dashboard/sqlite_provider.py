"""Registration-only SQLite boundary; Mock provider and analysis pages stay separate."""
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path

import pandas as pd

from database.batch_registry import preview_batch, register_batch, prepare_registration, register_selection
from database.init_db import DEFAULT_DB, PROJECT_ROOT, connect_database, initialize_database
from database.upload_storage import stage_uploads, upload_signature
from database.migrate_v2 import validate_v2
from database.migrate_v3 import validate_v3
from database.migrate_v4 import validate_v4
from database.migrate_v5 import validate_v5, effective_process_join, ANALYSIS_VERSIONS
from database.migrate_v6 import validate_v6, image_process_sql
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
            elif version==3:
                validate_v3(connection)
            elif version==4:
                validate_v4(connection)
            elif version==5:
                validate_v5(connection)
            elif version==6:
                validate_v6(connection)
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
        return {"version":version,"single_image_supported":version>=2,"peb_supported":version>=3,
                "multi_wafer_supported":version>=6,
                "display_name":os.environ.get("IDA_DB_LABEL","로컬 SQLite")+" / "+Path(self.db_path).name}


    def prepare_analysis_inputs(self, batch_id, *, image_ids=None):
        """Read-only Runner input gate; never invokes engines or saves results."""
        from inspection.request_mapper import prepare_batch_inputs
        with closing(self.connect()) as connection:
            connection.execute("PRAGMA query_only=ON")
            return prepare_batch_inputs(connection,batch_id,storage_root=self.storage_root,image_ids=image_ids)

    def stage(self, uploads):
        return stage_uploads(uploads, self.staging_root)

    def preview(self, inputs, metadata):
        # Approved new-upload default capture condition; explicit settings win.
        from database.capture_conditions import apply_upload_default
        with closing(self.connect()) as connection:
            inputs = apply_upload_default(connection, inputs, metadata)
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
            peb="b.peb_temperature_c" if connection.execute("PRAGMA user_version").fetchone()[0]>=3 else "NULL"
            # Effective Batch process: v5 Batch record first, else inherited Wafer record.
            process_join,source=effective_process_join(connection)
            frame=pd.read_sql_query(
                "SELECT "+peb+" AS peb_temperature_c,b.batch_id,b.wafer_id,b.upload_mode,b.dataset_role,b.batch_status,b.created_at,b.notes AS batch_notes,"
                "r.exposure_value,r.exposure_unit,"+source+" AS exposure_source,w.lot_id,w.wafer_label,r.process_run_id,"
                "r.recipe_version,r.processed_at,COUNT(i.image_id) AS image_count "
                "FROM upload_batches b JOIN wafers w ON w.wafer_id=b.wafer_id "
                +process_join+
                "LEFT JOIN images i ON i.batch_id=b.batch_id GROUP BY b.batch_id "
                "ORDER BY b.created_at,b.batch_id",connection)
            members={}
            if connection.execute("PRAGMA user_version").fetchone()[0]>=6:
                for batch_id,wafer in connection.execute("SELECT batch_id,wafer_id FROM batch_wafers ORDER BY batch_id,wafer_id"):
                    members.setdefault(batch_id,[]).append(wafer)
        frame["wafer_ids"]=[members.get(b,[w]) for b,w in zip(frame["batch_id"],frame["wafer_id"])]
        frame["wafer_count"]=frame["wafer_ids"].map(len)
        frame["wafer_list"]=frame["wafer_ids"].map(", ".join)  # Display "W1, W2, W3" (wafer_label is the Wafer record).
        multi=frame["wafer_count"]>1
        # v6 multi-Wafer Batch: the process lives per Batch-Wafer (batch_wafers); never show the
        # primary Wafer's default record as if it applied to the whole Batch.
        for column in ("exposure_value","exposure_unit","peb_temperature_c","process_run_id"):
            frame[column]=frame[column].astype(object).where(~multi,None)
        frame.loc[multi,"exposure_source"]="per_wafer"
        # Display-only names (never stored, never a key). Single-Wafer: "W1 · Batch 16" (unchanged).
        # Multi-Wafer: "Batch 01" numbered among multi-Wafer Batches; the Wafers are shown in the
        # Wafer column ("W1, W2, W3"), so they are not repeated in the name.
        group=frame["wafer_ids"].map(lambda ids:ids[0] if len(ids)==1 else "")
        frame["automatic_display_name"]=[(f"{label} · " if label else "")+f"Batch {index:02}"
            for label,index in zip(group,group.groupby(group).cumcount()+1)]
        frame["batch_name"]=frame["batch_notes"].map(batch_name_from_notes)
        frame["display_name"]=[automatic+(" · "+name if name else "")
            for automatic,name in zip(frame["automatic_display_name"],frame["batch_name"])]
        return frame.iloc[::-1].reset_index(drop=True)

    def batch_record(self,batch_id):
        """Resolve the actual SQLite primary key; display number/name are never keys."""
        if not isinstance(batch_id,str) or not batch_id.strip():
            return None
        frame=self.batches()
        found=frame[frame["batch_id"]==batch_id]
        return None if found.empty else found.iloc[0].to_dict()

    def batch_processing_matrix(self,batch_id):
        """All saved image rows, without pagination. Analysis has no result repository yet."""
        with closing(self.connect()) as connection:
            frame=pd.read_sql_query(
                "SELECT image_id,original_filename,metadata_status FROM images WHERE batch_id=? "
                "ORDER BY created_at DESC,image_id",connection,params=(batch_id,))
        frame["이미지 등록"]="등록 완료"
        frame["Metrology"]="I&A에서 조회"
        frame["Defect AI"]="I&A에서 조회"
        return frame

    def rename_batch(self,batch_id,name):
        with closing(self.connect(readonly=False)) as connection:
            return update_batch_name(connection,batch_id,name,allow_operational=True)

    def image_count(self, batch_id=None, upload_mode=None, *, wafer_id=None, die_code=None, position_code=None):
        where,params=self._image_filter(batch_id,upload_mode,wafer_id,die_code,position_code)
        with closing(self.connect()) as connection:
            return connection.execute("SELECT COUNT(*) FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id "
                                      "JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
                                      "JOIN dies d ON d.die_id=p.die_id "
                                      +where,params).fetchone()[0]

    @staticmethod
    def _image_filter(batch_id,upload_mode,wafer_id=None,die_code=None,position_code=None):
        clauses=[]
        params=[]
        if batch_id is not None:
            clauses.append("i.batch_id=?")
            params.append(batch_id)
        if upload_mode is not None:
            clauses.append("b.upload_mode=?")
            params.append(upload_mode)
        # The image's own Wafer (its Die), never the Batch's primary Wafer.
        for column,value in (("d.wafer_id",wafer_id),("d.die_code",die_code),("p.position_code",position_code)):
            if value is not None:
                clauses.append(column+"=?")
                params.append(value)
        return ("WHERE "+" AND ".join(clauses) if clauses else ""),params

    def images(self, batch_id=None, *, upload_mode=None, limit=25, offset=0, image_id=None,
               wafer_id=None, die_code=None, position_code=None):
        if type(limit) is not int or not 1<=limit<=100 or type(offset) is not int or offset<0:
            raise ValueError("Invalid image page")
        where,params=self._image_filter(batch_id,upload_mode,wafer_id,die_code,position_code)
        if image_id is not None:
            where+=(" AND " if where else "WHERE ")+"i.image_id=?"
            params.append(image_id)
        with closing(self.connect()) as connection:
            # Per-image process: v6 = the image's own Batch-Wafer; v5 = the Batch process.
            process_join,source,peb=image_process_sql(connection)
            return pd.read_sql_query(
                "SELECT "+peb+" AS peb_temperature_c,"+source+" AS exposure_source,i.image_id,i.batch_id,i.original_filename,i.storage_path,i.sha256,"
                "i.width_px,i.height_px,i.metadata_status,i.created_at,i.repeat_index,i.condition_id,"
                "i.physical_site_id,i.captured_at,r.exposure_value,r.exposure_unit,"
                "d.wafer_id,d.die_code,p.position_code,i.legacy_defect_code,b.upload_mode,"
                "a.illumination_setting,a.magnification,a.auto_exposure "
                "FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id "
                "JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
                "JOIN dies d ON d.die_id=p.die_id "
                +process_join+
                "LEFT JOIN acquisition_conditions a ON a.condition_id=i.condition_id "
                +where+" ORDER BY i.created_at DESC,i.image_id LIMIT ? OFFSET ?",
                connection,params=(*params,limit,offset))


    def image_record(self,image_id,*,batch_id=None):
        if not isinstance(image_id,str) or not image_id.strip():
            return None
        frame=self.images(batch_id,image_id=image_id,limit=1)
        if frame.empty:
            return None
        record=frame.iloc[0].to_dict()
        batch=self.batch_record(record["batch_id"])
        if batch is None:
            return None
        record.update(dataset_role=batch["dataset_role"],batch_name=batch["batch_name"],
                      batch_display_name=batch["display_name"],
                      automatic_batch_display_name=batch["automatic_display_name"])
        return record

    def wafers(self):
        """Only persisted physical Wafer IDs; no Mock fallback or record creation."""
        with closing(self.connect()) as connection:
            return pd.read_sql_query(
                # Images counted by their own Wafer (Die), so a multi-Wafer Batch is split correctly.
                "SELECT w.wafer_id,COUNT(i.image_id) AS image_count FROM wafers w "
                "LEFT JOIN dies d ON d.wafer_id=w.wafer_id LEFT JOIN pattern_sites p ON p.die_id=d.die_id "
                "LEFT JOIN images i ON i.physical_site_id=p.physical_site_id GROUP BY w.wafer_id ORDER BY w.wafer_id",connection)

    def wafer_sites(self,wafer_id):
        """Existing physical identities, including Sites without registered images."""
        with closing(self.connect()) as connection:
            frame=pd.read_sql_query(
                "SELECT d.die_id,d.die_code,d.row_index,d.column_index,p.physical_site_id,p.position_code "
                "FROM dies d LEFT JOIN pattern_sites p ON p.die_id=d.die_id "
                "WHERE d.wafer_id=? ORDER BY d.row_index,d.column_index,p.position_code",
                connection,params=(wafer_id,))
        return frame.to_dict("records")

    def condition_images(self, batch_ids, wafer_id=None):
        """Read-only: images of the given Batches with their physical Die / position (NULL if unmapped).

        wafer_id keeps only that Wafer's images (a multi-Wafer Batch also holds other Wafers).
        """
        ids = sorted(set(batch_ids))
        if not ids:
            return pd.DataFrame(columns=["image_id","batch_id","original_filename","wafer_id","die_code","position_code"])
        where = "WHERE i.batch_id IN (" + ",".join("?" for _ in ids) + ") "
        params = list(ids)
        if wafer_id is not None:
            where += "AND d.wafer_id=? "
            params.append(wafer_id)
        with closing(self.connect()) as connection:
            return pd.read_sql_query(
                "SELECT i.image_id,i.batch_id,i.original_filename,d.wafer_id,d.die_code,p.position_code FROM images i "
                "LEFT JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
                "LEFT JOIN dies d ON d.die_id=p.die_id " + where +
                "ORDER BY d.die_code,p.position_code,i.image_id", connection, params=params)

    def batch_wafers(self):
        """One row per Batch × Wafer: that Wafer's images, process and PEB (s) inside the Batch.

        v6 reads batch_wafers; v5 and older have exactly one Wafer per Batch (the Batch row itself).
        image_count counts only the images of THAT Wafer, so Batch totals are never repeated per Wafer.
        """
        batches = self.batches()
        batches["batch_image_count"] = batches["image_count"]   # Whole Batch (all Wafers); never summed per Wafer.
        keep = ["batch_id","upload_mode","dataset_role","batch_status","created_at","batch_name",
                "automatic_display_name","display_name","wafer_count","wafer_list","batch_image_count"]
        with closing(self.connect()) as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] < 6:
                return batches.copy()
            rows = pd.read_sql_query(
                "SELECT bw.batch_id,bw.wafer_id,r.exposure_value,r.exposure_unit,bw.peb_time_s AS peb_temperature_c,"
                "CASE WHEN bw.process_run_id IS NULL THEN 'wafer' ELSE 'batch' END AS exposure_source,r.process_run_id,"
                "(SELECT COUNT(*) FROM images i JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
                " JOIN dies d ON d.die_id=p.die_id WHERE i.batch_id=bw.batch_id AND d.wafer_id=bw.wafer_id) AS image_count "
                "FROM batch_wafers bw JOIN wafers w ON w.wafer_id=bw.wafer_id "
                "JOIN process_runs r ON r.process_run_id=COALESCE(bw.process_run_id,w.process_run_id) "
                "ORDER BY bw.batch_id,bw.wafer_id", connection)
        frame = rows.merge(batches[keep], on="batch_id", how="inner")
        order = {batch: index for index, batch in enumerate(batches["batch_id"])}
        return frame.sort_values(["batch_id","wafer_id"], key=lambda col: col.map(order) if col.name=="batch_id" else col,
                                 kind="stable").reset_index(drop=True)

    # ---- TASK18-B-12A: 전체 Dataset (per-image Wafer and effective process; never the Batch's primary Wafer) ----
    DATASET_ANY="전체"

    def _dataset_from(self,connection):
        process_join,_,peb=image_process_sql(connection)
        base=("FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id "
              "JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id JOIN dies d ON d.die_id=p.die_id "
              +process_join)
        return base,peb

    @staticmethod
    def dataset_exposure_key(value,unit):
        value=None if value is None or value!=value else float(value)
        unit=None if unit is None or unit!=unit or str(unit).strip()=="" else str(unit)
        return json.dumps([value,unit])

    def _dataset_where(self,filters,peb):
        clauses,params=[],[]
        filters=filters or {}
        wafer=filters.get("wafer_id")
        if wafer not in (None,self.DATASET_ANY):
            clauses.append("d.wafer_id=?");params.append(wafer)
        exposure=filters.get("exposure")
        if exposure not in (None,self.DATASET_ANY):
            value,unit=json.loads(exposure)
            clauses.append("r.exposure_value IS ? AND r.exposure_unit IS ?");params+=[value,unit]
        peb_value=filters.get("peb")
        if peb_value not in (None,self.DATASET_ANY):
            clauses.append(peb+" IS ?");params.append(None if peb_value=="NULL" else float(peb_value))
        return ("WHERE "+" AND ".join(clauses)+" " if clauses else ""),params

    def dataset_filter_options(self):
        """Values that actually exist in SQLite (per image): Wafers, 노광 (value+recorded unit) and PEB (s)."""
        with closing(self.connect()) as connection:
            base,peb=self._dataset_from(connection)
            wafers=[r[0] for r in connection.execute("SELECT DISTINCT d.wafer_id "+base+"ORDER BY 1")]
            exposures=[self.dataset_exposure_key(v,u) for v,u in connection.execute(
                "SELECT DISTINCT r.exposure_value,r.exposure_unit "+base+"ORDER BY r.exposure_value IS NULL,1,2")]
            pebs=[r[0] for r in connection.execute("SELECT DISTINCT "+peb+" "+base+"ORDER BY 1 IS NULL,1")]
        return {"wafers":wafers,"exposures":list(dict.fromkeys(exposures)),
                "pebs":["NULL" if v is None else f"{float(v):g}" for v in pebs]}

    def dataset_count(self,filters=None):
        with closing(self.connect()) as connection:
            base,peb=self._dataset_from(connection)
            where,params=self._dataset_where(filters,peb)
            return connection.execute("SELECT COUNT(*) "+base+where,params).fetchone()[0]

    def dataset_ids(self,filters=None):
        """Every Image ID of the filtered result (used for 'select all results' and server re-validation)."""
        with closing(self.connect()) as connection:
            base,peb=self._dataset_from(connection)
            where,params=self._dataset_where(filters,peb)
            return [r[0] for r in connection.execute("SELECT i.image_id "+base+where+"ORDER BY i.created_at DESC,i.image_id",params)]

    def dataset_page(self,filters=None,*,limit=50,offset=0):
        if type(limit) is not int or not 1<=limit<=200 or type(offset) is not int or offset<0:
            raise ValueError("Invalid dataset page")
        with closing(self.connect()) as connection:
            base,peb=self._dataset_from(connection)
            where,params=self._dataset_where(filters,peb)
            analysis=("" if connection.execute("PRAGMA user_version").fetchone()[0] not in ANALYSIS_VERSIONS else
                ",(SELECT a.outcome FROM analysis_image_runs a JOIN analysis_batch_runs ar ON ar.batch_run_id=a.batch_run_id "
                " WHERE a.image_id=i.image_id ORDER BY julianday(ar.started_at) DESC,a.analysis_run_id DESC LIMIT 1) AS latest_outcome"
                ",(SELECT COUNT(*) FROM analysis_image_runs a WHERE a.image_id=i.image_id) AS analysis_runs")
            frame=pd.read_sql_query(
                "SELECT i.image_id,i.original_filename,d.wafer_id,i.batch_id,r.exposure_value,r.exposure_unit,"
                +peb+" AS peb_time_s,d.die_code,p.position_code,b.upload_mode,i.created_at"+analysis+" "
                +base+where+"ORDER BY i.created_at DESC,i.image_id LIMIT ? OFFSET ?",connection,params=(*params,limit,offset))
        if "latest_outcome" not in frame:
            frame["latest_outcome"]=None;frame["analysis_runs"]=0
        names=self.batches().set_index("batch_id")["display_name"].to_dict() if len(frame) else {}
        frame["batch_name"]=frame["batch_id"].map(names)
        return frame

    def latest_analysis(self, wafer_id=None):
        """Read-only: latest stored analysis run per registered image (Schema v4 only).

        "Latest" = most recent Batch run start, then analysis_run_id (same order as
        AnalysisRepository.image_history). Values are copied from the stored rows; nothing is
        recomputed, judged or adopted. Returns {} before v4 or when nothing was analysed.
        """
        import json as _json
        with closing(self.connect()) as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] not in ANALYSIS_VERSIONS:
                return {}
            # Filter by the image's own Wafer (a multi-Wafer Batch holds several Wafers).
            where, params = ("WHERE d.wafer_id=?", [wafer_id]) if wafer_id else ("", [])
            cursor = connection.execute(
                "SELECT a.image_id,a.analysis_run_id,b.batch_id,b.started_at,a.outcome,a.engine_status,d.wafer_id,"
                "a.defect_status,a.metrology_status,a.pd_policy_state,a.partial_measurement,a.line_mean_um,"
                "a.gap_mean_um,a.line_measurement_count,a.gap_measurement_count,a.calibration_id,"
                "a.calibration_json,a.configuration_json,a.is_adopted FROM analysis_image_runs a "
                "JOIN analysis_batch_runs b ON b.batch_run_id=a.batch_run_id "
                "JOIN images im ON im.image_id=a.image_id JOIN pattern_sites ps ON ps.physical_site_id=im.physical_site_id "
                "JOIN dies d ON d.die_id=ps.die_id " + where +
                " ORDER BY a.image_id,julianday(b.started_at) DESC,a.analysis_run_id DESC", params)
            names = [d[0] for d in cursor.description]
            latest = {}
            for row in cursor:
                item = dict(zip(names, row))
                latest.setdefault(item["image_id"], item)  # First row per image is the latest.
            for item in latest.values():
                item["calibration"] = _json.loads(item.pop("calibration_json")) if item.get("calibration_json") else None
                item["configuration"] = _json.loads(item.pop("configuration_json")) if item.get("configuration_json") else None
                item["line_widths_um"] = []
                item["gap_widths_um"] = []
                item["defect_classes"] = []
                item["defect_details"] = []
                item["judgments"] = []
            runs = {item["analysis_run_id"]: item for item in latest.values()}
            if runs:
                marks = ",".join("?" for _ in runs)
                for run_id, kind, width in connection.execute(
                        "SELECT analysis_run_id,measurement_type,width_um FROM metrology_measurements "
                        "WHERE analysis_run_id IN (" + marks + ") ORDER BY analysis_run_id,measurement_index", list(runs)):
                    if width is not None:
                        runs[run_id][kind + "_widths_um"].append(width)
                for run_id, code, confidence, source, payload in connection.execute(
                        "SELECT analysis_run_id,class_code,confidence,source,payload_json FROM defect_detections WHERE analysis_run_id IN (" + marks +
                        ") ORDER BY analysis_run_id,detection_index", list(runs)):
                    runs[run_id]["defect_classes"].append(code)
                    detail = {"class_code": code, "confidence": confidence, "source": source}
                    try:   # r33 (TASK18-B-22): teammate source and S -> P derivation, when stored.
                        stored = _json.loads(payload) if payload else {}
                    except ValueError:
                        stored = {}
                    for key in ("team_source", "derived_from_index"):
                        if key in stored:
                            detail[key] = stored[key]
                    runs[run_id]["defect_details"].append(detail)
                for run_id, line_id, label, low, high, near in connection.execute(
                        "SELECT analysis_run_id,line_id,label,min_um,max_um,near_threshold FROM line_judgments "
                        "WHERE analysis_run_id IN (" + marks + ") ORDER BY analysis_run_id,line_id", list(runs)):
                    runs[run_id]["judgments"].append({"line_id": line_id, "label": label, "min_um": low,
                                                      "max_um": high, "near_threshold": near})
            return latest

    def system_status(self):
        """Read-only KPIs. Registration and analysis are separate; each image counts once (latest run).

        Raises on any DB error (callers must show an error, never substitute 0).
        """
        review_outcomes = ("QUALITY_REVIEW_REQUIRED", "PARTIAL_MEASUREMENT", "POLICY_REVIEW_REQUIRED")
        with closing(self.connect()) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            batches = connection.execute("SELECT COUNT(*) FROM upload_batches").fetchone()[0]
            images = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            latest_run = None
            if version in ANALYSIS_VERSIONS:
                row = connection.execute(
                    "SELECT batch_run_id,batch_id,started_at,ended_at,execution_status,requested_images,"
                    "failed_images FROM analysis_batch_runs ORDER BY julianday(started_at) DESC,batch_run_id DESC LIMIT 1").fetchone()
                if row:
                    latest_run = dict(zip(("batch_run_id","batch_id","started_at","ended_at","execution_status",
                                           "requested_images","failed_images"), row))
        latest = self.latest_analysis()
        outcomes = {}
        for item in latest.values():
            outcomes[item["outcome"]] = outcomes.get(item["outcome"], 0) + 1
        return {"schema_version": version, "batches": batches, "images": images, "analysed_images": len(latest),
                "review_images": sum(outcomes.get(o, 0) for o in review_outcomes),
                "pd_excluded_images": outcomes.get("MODEL_PD_EXCLUDED", 0), "failed_images": outcomes.get("FAILED", 0),
                "complete_images": outcomes.get("COMPLETE", 0), "outcomes": outcomes, "latest_run": latest_run}

    def recent_analysis_runs(self, limit=5):
        """Persisted analysis runs (read-only) for the notification history; newest first."""
        with closing(self.connect()) as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] not in ANALYSIS_VERSIONS:
                return []
            cursor = connection.execute(
                "SELECT r.batch_run_id,r.batch_id,"+("(SELECT group_concat(wafer_id,', ') FROM (SELECT wafer_id FROM batch_wafers x "
                "WHERE x.batch_id=r.batch_id ORDER BY wafer_id))" if connection.execute("PRAGMA user_version").fetchone()[0]>=6
                else "b.wafer_id")+",r.started_at,r.execution_status,r.requested_images,"
                "r.failed_images FROM analysis_batch_runs r JOIN upload_batches b ON b.batch_id=r.batch_id "
                "ORDER BY julianday(r.started_at) DESC,r.batch_run_id DESC LIMIT ?", (int(limit),))
            return [dict(zip(("batch_run_id","batch_id","wafer_id","started_at","execution_status",
                              "requested_images","failed_images"), row)) for row in cursor]

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
