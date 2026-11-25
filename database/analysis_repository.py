"""v4 file-verified analysis persistence. No engines, adoption writes or ID remapping."""
from contextlib import closing
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from database.init_db import PROJECT_ROOT, connect_database
from database.image_registry import get_image
from database.migrate_v4 import refuse_operational
from database.migrate_v5 import validate_analysis_schema

# The ONLY result root accepted for operational writes (TASK18-A-4). Validation
# outputs elsewhere can therefore never be back-registered into the operational DB.
OPERATIONAL_RESULT_ROOT=PROJECT_ROOT/"runtime_data/analysis_results/dashboard_runs"

COUNTS=("requested_images","processed_images","unprocessed_images","success_images","failed_images",
        "skipped_engine_images","skipped_metrology_images","complete_measurement_images",
        "partial_measurement_images","policy_review_images","quality_review_images","model_pd_excluded_images")


class AnalysisStorageError(ValueError):
    pass


def _json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)


def _sha(path):
    with Path(path).open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(AnalysisStorageError("Non-finite JSON")))


def _num(value, positive=False, nullable=False):
    if value is None and nullable:return value
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or positive and value<=0:
        raise AnalysisStorageError("Invalid numeric result")
    return value


def _time(value, nullable=True):
    if value is None and nullable:return None
    if not isinstance(value,str) or datetime.fromisoformat(value.replace("Z","+00:00")).tzinfo is None:
        raise AnalysisStorageError("Timezone-qualified ISO8601 required")
    return value


def _is_operational(path):
    try:
        refuse_operational(path)
        return False
    except PermissionError:
        return True


def _insert(c,table,row):
    names=list(row)
    c.execute("INSERT INTO "+table+"("+",".join(names)+") VALUES ("+",".join("?" for _ in names)+")",
              [row[n] for n in names])


class AnalysisRepository:
    def __init__(self, db_path, *, result_root, image_root, allow_operational=False, read_only=False):
        # Default keeps the operational guard. Only an explicit True for the official
        # result root may open the operational DB (no other caller passes it).
        # read_only=True (TASK18-B-1) permits ONLY mode=ro connections; save_batch refused.
        self._allow_operational=allow_operational is True
        self._read_only=read_only is True
        self.db_path=self._guard(db_path,write=not self._read_only)
        self.result_root=Path(result_root).resolve(strict=True)
        self.image_root=Path(image_root).resolve(strict=True)
        if not self.result_root.is_dir() or not self.image_root.is_dir():
            raise AnalysisStorageError("Storage roots must exist")
        if ((self._allow_operational or self._read_only) and self.result_root!=OPERATIONAL_RESULT_ROOT.resolve()
                and _is_operational(self.db_path)):
            raise PermissionError("Operational analysis storage requires the official result root")

    def _guard(self,path,write=True):
        if self._allow_operational or (self._read_only and not write):
            return Path(path).expanduser().resolve()
        return refuse_operational(path)

    def _path(self,value,root):
        if not isinstance(value,str) or not value or "://" in value:
            raise AnalysisStorageError("A local file reference is required")
        p=Path(value)
        p=(p if p.is_absolute() else root/p).resolve(strict=True)
        if not p.is_relative_to(root) or not p.is_file():
            raise AnalysisStorageError("File reference outside allowed root")
        return p

    def _asset(self,value,assets,owner):
        p=self._path(value,self.result_root)
        rel=p.relative_to(self.result_root).as_posix()
        if rel in assets and assets[rel]["owner"]!=owner:
            raise AnalysisStorageError("Output path collision across records")
        sha=_sha(p)
        assets[rel]={"sha256":sha,"owner":owner}
        return rel,sha,p

    def _connection(self,readonly=False):
        if self._read_only and not readonly:
            raise PermissionError("Read-only analysis repository")
        self._guard(self.db_path,write=not readonly)
        c=connect_database(self.db_path,readonly=readonly)
        try:validate_analysis_schema(c)
        except BaseException:c.close();raise
        return c

    def _prepare(self,c,summary_file):
        assets={}
        summary_rel,summary_sha,summary_path=self._asset(str(summary_file),assets,"batch-summary")
        summary=_read(summary_path)
        if summary.get("schema")!="inspection-batch-v0.1":
            raise AnalysisStorageError("Unsupported Batch Runner schema")
        batch_id,run_id=summary["batch_id"],summary["batch_run_id"]
        if c.execute("SELECT 1 FROM upload_batches WHERE batch_id=?",(batch_id,)).fetchone() is None:
            raise AnalysisStorageError("Unregistered batch_id; QA IDs are never remapped")
        manifest_rel,manifest_sha,manifest_path=self._asset(summary["request_manifest_json"],assets,"manifest")
        if manifest_sha!=summary["request_manifest_sha256"]:
            raise AnalysisStorageError("Manifest hash mismatch")
        manifest=_read(manifest_path)
        if manifest.get("batch_id")!=batch_id or manifest.get("batch_run_id")!=run_id:
            raise AnalysisStorageError("Manifest Batch/Run mismatch")
        requests=[e["request"] for e in manifest["entries"]]
        rows=summary["images"]
        if len(rows)!=len(requests) or not rows:
            raise AnalysisStorageError("Request/result count mismatch")
        if len({r["image_id"] for r in rows})!=len(rows) or len({r["analysis_run_id"] for r in rows})!=len(rows):
            raise AnalysisStorageError("Duplicate image/run ID")
        images=[];sources=[]
        for index,(r,req) in enumerate(zip(rows,requests)):
            if r["request_index"]!=index or req["image_id"]!=r["image_id"] or req.get("batch_id")!=batch_id:
                raise AnalysisStorageError("Manifest order/identity mismatch")
            registered=get_image(c,r["image_id"])
            if registered is None or registered["batch_id"]!=batch_id:
                raise AnalysisStorageError("Image is unregistered or belongs to another Batch")
            actual_meta={"wafer_id":registered["wafer_id"],"die_id":registered["die_id"],
                         "site_code":registered["position_code"],"physical_site_id":registered["physical_site_id"],
                         "die_code":registered["die_code"]}
            for key,value in actual_meta.items():
                if r["metadata"].get(key) is not None and r["metadata"][key]!=value:
                    raise AnalysisStorageError("Registration metadata mismatch: "+key)
            source=self._path(req["image_path"],self.image_root)
            storage=self._path(registered["storage_path"],self.image_root)
            source_sha=_sha(source)
            if source_sha.lower()!=registered["sha256"].lower() or _sha(storage)!=source_sha:
                raise AnalysisStorageError("Source/registered image SHA-256 mismatch")
            if r.get("source_image_sha256") is not None and r["source_image_sha256"]!=source_sha:
                raise AnalysisStorageError("Batch source SHA-256 mismatch")
            sources.extend([(source,source_sha),(storage,source_sha)])
            raw=None;result_rel=None;result_sha=None;overlays={}
            if r["result_json"] is not None:
                result_rel,result_sha,result_path=self._asset(r["result_json"],assets,r["analysis_run_id"]+":result")
                if result_sha!=r["result_sha256"]:raise AnalysisStorageError("Result JSON hash mismatch")
                raw=_read(result_path)
                if (raw.get("image_id")!=r["image_id"] or raw.get("batch_id")!=batch_id
                    or raw.get("analysis_run_id")!=r["analysis_run_id"]):
                    raise AnalysisStorageError("Result identity/Batch/analysis_run_id mismatch")
                for key in ("wafer_id","die_id","site_code"):
                    if raw.get(key)!=actual_meta[key]:
                        raise AnalysisStorageError("Result registration identity mismatch: "+key)
                if raw.get("image_path") is not None:
                    raw_source=self._path(raw["image_path"],self.image_root)
                    if _sha(raw_source)!=source_sha:
                        raise AnalysisStorageError("Result image path/hash mismatch")
                    sources.append((raw_source,source_sha))
                if raw.get("image_sha256") is not None and raw["image_sha256"]!=source_sha:
                    raise AnalysisStorageError("Result source hash mismatch")
                if raw["status"]!=r["engine_status"]:
                    raise AnalysisStorageError("Image engine status mismatch")
                for stage in ("defect","metrology"):
                    item=raw.get(stage) or {}
                    if item and item.get("image_id")!=r["image_id"]:
                        raise AnalysisStorageError("Nested image_id mismatch")
                    if item.get("status")=="SUCCESS":
                        for key in ("batch_id","analysis_run_id","wafer_id","die_id","site_code"):
                            expected=r["analysis_run_id"] if key=="analysis_run_id" else batch_id if key=="batch_id" else actual_meta[key]
                            if item.get(key)!=expected:
                                raise AnalysisStorageError("Nested execution/registration identity mismatch")
                    if item.get("status")!=r.get(stage+"_status"):
                        raise AnalysisStorageError("Stage status mismatch")
                    overlay=(raw.get("overlays") or {}).get(stage)
                    if overlay is not None:
                        rel,sha,p=self._asset(overlay["path"],assets,r["analysis_run_id"]+":"+stage)
                        overlays[stage]=(rel,sha)
                if (raw.get("pd_policy") or {}).get("state")!=r.get("pd_policy_state"):
                    raise AnalysisStorageError("PD policy mismatch")
            elif r["processing_status"] not in ("FAILED","UNPROCESSED"):
                raise AnalysisStorageError("Completed image result JSON is missing")
            raw=raw or {}
            defect=raw.get("defect") or {}; metrology=raw.get("metrology") or {}
            quality=metrology.get("quality")
            partial=quality.get("partial_measurement") if quality is not None else None
            if partial is not None and type(partial) is not bool:
                raise AnalysisStorageError("partial_measurement must be bool/null")
            if (r["outcome"]=="PARTIAL_MEASUREMENT")!=(partial is True):
                raise AnalysisStorageError("Partial measurement state mismatch")
            if raw:
                state=(raw.get("pd_policy") or {}).get("state")
                pd_sources={d.get("source") for d in defect.get("detections",[]) if d.get("class_code")=="pd"}
                outcome=r["outcome"]
                if outcome=="MODEL_PD_EXCLUDED" and not (
                    raw["status"]=="SUCCESS" and defect.get("status")=="SUCCESS" and
                    metrology.get("status")=="SKIPPED" and state=="MODEL_PD_EXCLUDED" and "MODEL_POSTPROCESSED" in pd_sources):
                    raise AnalysisStorageError("MODEL PD state contradicts result")
                if outcome=="POLICY_REVIEW_REQUIRED" and not (
                    raw["status"]=="SKIPPED" and defect.get("status")=="SUCCESS" and
                    metrology.get("status")=="SKIPPED" and state=="POLICY_REVIEW_REQUIRED" and pd_sources and pd_sources<={"HEURISTIC","RULE"}):
                    raise AnalysisStorageError("non-model (HEURISTIC/RULE) PD state contradicts result")
                if outcome in ("COMPLETE","QUALITY_REVIEW_REQUIRED","PARTIAL_MEASUREMENT"):
                    if not (raw["status"]=="SUCCESS" and defect.get("status")=="SUCCESS" and
                            metrology.get("status")=="SUCCESS" and state=="NO_PD" and not pd_sources):
                        raise AnalysisStorageError("Measurement outcome contradicts stages/PD")
                    if outcome=="COMPLETE" and (partial is not False or raw.get("official_analysis_complete") is not True):
                        raise AnalysisStorageError("Complete measurement requires explicit complete quality")
                if outcome=="FAILED" and raw["status"]!="FAILED":
                    raise AnalysisStorageError("Failed outcome contradicts engine status")
                if raw.get("width_error"):
                    calibration=metrology.get("calibration") or {}
                    if not metrology.get("calibration_id") or calibration.get("calibration_id")!=metrology["calibration_id"]:
                        raise AnalysisStorageError("Official width error lacks applied Calibration identity")
                    _num(calibration.get("um_per_px"),positive=True)
            means=raw.get("measurement_summary") or {}
            errors=raw.get("width_error") or {}
            if r["outcome"] in ("PARTIAL_MEASUREMENT","MODEL_PD_EXCLUDED","POLICY_REVIEW_REQUIRED","FAILED","UNPROCESSED") and errors:
                raise AnalysisStorageError("Invalid official error for excluded/partial/failed result")
            image={
              "analysis_run_id":r["analysis_run_id"],"batch_run_id":run_id,"image_id":r["image_id"],"request_index":index,
              "started_at":_time(r.get("started_at")),"ended_at":_time(r.get("ended_at")),
              "processing_status":r["processing_status"],"engine_status":r.get("engine_status"),
              "defect_status":r.get("defect_status"),"metrology_status":r.get("metrology_status"),"outcome":r["outcome"],
              "pd_policy_state":r.get("pd_policy_state"),"partial_measurement":int(partial) if partial is not None else None,
              "policy_review":int(r["outcome"]=="POLICY_REVIEW_REQUIRED"),
              "reason":r.get("reason"),"error_message":r.get("error_message"),
              "line_mean_um":_num((means.get("line") or {}).get("mean_um"),positive=True,nullable=True),
              "gap_mean_um":_num((means.get("gap") or {}).get("mean_um"),positive=True,nullable=True),
              "line_measurement_count":(means.get("line") or {}).get("count"),
              "gap_measurement_count":(means.get("gap") or {}).get("count"),
              "signed_error_pct":_num(errors.get("signed_percent"),nullable=True),
              "absolute_error_pct":_num(errors.get("absolute_percent"),nullable=True),
              "calibration_id":metrology.get("calibration_id"),
              "calibration_json":_json(metrology["calibration"]) if "calibration" in metrology else None,
              "quality_json":_json(quality) if quality is not None else None,
              "pixel_scale_json":_json(metrology["pixel_scale"]) if metrology.get("pixel_scale") is not None else None,
              "metadata_json":_json({"runner":r["metadata"],"registration":actual_meta}),
              "configuration_json":_json(raw["configuration"]) if raw.get("configuration") is not None else None,
              "model_version":defect.get("model_version"),"model_sha256":defect.get("model_sha256"),
              "algorithm_version":metrology.get("algorithm_version"),
              "bbox_coordinate_space":defect.get("bbox_coordinate_system"),"bbox_width":defect.get("bbox_image_width"),
              "bbox_height":defect.get("bbox_image_height"),"skew_angle":defect.get("skew_angle"),
              "image_sha256":source_sha,"result_path":result_rel,"result_sha256":result_sha,"is_adopted":0,
              "defect_overlay_path":overlays.get("defect",(None,None))[0],
              "defect_overlay_sha256":overlays.get("defect",(None,None))[1],
              "metrology_overlay_path":overlays.get("metrology",(None,None))[0],
              "metrology_overlay_sha256":overlays.get("metrology",(None,None))[1]}
            children={"defect_detections":[],"metrology_measurements":[],"line_judgments":[]}
            for i,d in enumerate(defect.get("detections",[])):
                bbox=d["bbox_xyxy"]
                if not isinstance(bbox,list) or len(bbox)!=4:raise AnalysisStorageError("Invalid bbox")
                for value in bbox:_num(value)
                children["defect_detections"].append({"analysis_run_id":r["analysis_run_id"],"detection_index":i,
                    "class_code":d["class_code"],"class_id":d.get("class_id"),"class_name":d.get("class_name"),
                    "confidence":_num(d["confidence"]),**dict(zip(("x1","y1","x2","y2"),bbox)),
                    "source":d["source"],"coordinate_space":d.get("coordinate_space",defect.get("bbox_coordinate_system")),
                    "payload_json":_json(d)})
            for i,m in enumerate(metrology.get("measurements",[])):
                child={"analysis_run_id":r["analysis_run_id"],"measurement_index":i,
                    "measurement_type":m["measurement_type"],"line_id":m["line_id"],
                    "y_fraction":_num(m["y_fraction"]),"width_px":_num(m["width_px"],positive=True),
                    "width_um":_num(m.get("width_um"),positive=True,nullable=True),"payload_json":_json(m)}
                for key in ("y_px","left_x_px","right_x_px","center_x_px"):
                    child[key]=_num(m.get(key),nullable=True)
                children["metrology_measurements"].append(child)
            for j in (metrology.get("judgment") or {}).get("lines",[]):
                child={"analysis_run_id":r["analysis_run_id"],"line_id":j["line_id"],"label":j["label"],
                    "reason":j.get("reason"),"n_rows":j.get("n_rows"),"payload_json":_json(j)}
                for key in ("min_um","max_um","mean_um"):child[key]=_num(j.get(key),nullable=True)
                for key in ("near_threshold","pitch_ok"):
                    value=j.get(key)
                    if value is not None and type(value) is not bool:raise AnalysisStorageError("Invalid judgment quality bool")
                    child[key]=int(value) if value is not None else None
                children["line_judgments"].append(child)
            images.append((image,children))
        states=[x[0] for x in images]
        counts={
          "requested_images":len(states),"processed_images":sum(r["processing_status"]!="UNPROCESSED" for r in states),
          "unprocessed_images":sum(r["processing_status"]=="UNPROCESSED" for r in states),
          "success_images":sum(r["engine_status"]=="SUCCESS" for r in states),
          "failed_images":sum(r["processing_status"]=="FAILED" for r in states),
          "skipped_engine_images":sum(r["engine_status"]=="SKIPPED" for r in states),
          "skipped_metrology_images":sum(r["metrology_status"]=="SKIPPED" for r in states),
          "complete_measurement_images":sum(r["outcome"]=="COMPLETE" for r in states),
          "partial_measurement_images":sum(r["outcome"]=="PARTIAL_MEASUREMENT" for r in states),
          "policy_review_images":sum(r["outcome"]=="POLICY_REVIEW_REQUIRED" for r in states),
          "quality_review_images":sum(r["outcome"]=="QUALITY_REVIEW_REQUIRED" for r in states),
          "model_pd_excluded_images":sum(r["outcome"]=="MODEL_PD_EXCLUDED" for r in states)}
        if counts!=summary["counts"]:raise AnalysisStorageError("Batch count mismatch")
        fingerprint=hashlib.sha256(_json({"summary":summary_sha,"assets":assets,
            "registered_images":[(a["image_id"],a["image_sha256"]) for a,_ in images]}).encode("utf-8")).hexdigest()
        batch={"batch_run_id":run_id,"batch_id":batch_id,"parent_batch_run_id":summary.get("retry_of_run_id"),
          "started_at":_time(summary["started_at"],nullable=False),"ended_at":_time(summary.get("ended_at")),
          "execution_status":summary["execution_status"],**counts,
          "interruption_reason":summary.get("interruption_reason"),
          "execution_config_json":_json(summary["execution_config_sha256"]),
          "summary_path":summary_rel,"summary_sha256":summary_sha,
          "manifest_path":manifest_rel,"manifest_sha256":manifest_sha,"import_sha256":fingerprint}
        return batch,images,assets,sources

    def save_batch(self,summary_file):
        if self._read_only:
            raise PermissionError("Read-only analysis repository")
        self._guard(self.db_path)
        with closing(self._connection()) as c:
            try:
                c.execute("BEGIN IMMEDIATE")
                batch,images,assets,sources=self._prepare(c,summary_file)
                existing=c.execute("SELECT import_sha256 FROM analysis_batch_runs WHERE batch_run_id=?",
                                   (batch["batch_run_id"],)).fetchone()
                if existing:
                    if existing[0]!=batch["import_sha256"]:
                        raise AnalysisStorageError("Existing Run differs; overwrite/upsert forbidden")
                    c.rollback()
                    return {"batch_run_id":batch["batch_run_id"],"inserted":False}
                _insert(c,"analysis_batch_runs",batch)
                for image,children in images:
                    _insert(c,"analysis_image_runs",image)
                    for table,values in children.items():
                        for row in values:_insert(c,table,row)
                for rel,item in assets.items():
                    if _sha(self._path(rel,self.result_root))!=item["sha256"]:
                        raise AnalysisStorageError("Artifact changed during save")
                for p,sha in sources:
                    if _sha(p)!=sha:raise AnalysisStorageError("Original image changed during save")
                if c.execute("PRAGMA foreign_key_check").fetchall():
                    raise AnalysisStorageError("Foreign key violation")
                c.commit()
                return {"batch_run_id":batch["batch_run_id"],"inserted":True,
                        "images":len(images),"detections":sum(len(x["defect_detections"]) for _,x in images),
                        "measurements":sum(len(x["metrology_measurements"]) for _,x in images),
                        "judgments":sum(len(x["line_judgments"]) for _,x in images)}
            except BaseException:
                if c.in_transaction:c.rollback()
                raise

    def _rows(self,c,sql,args=()):
        cursor=c.execute(sql,args)
        return [dict(zip((d[0] for d in cursor.description),row)) for row in cursor]

    def get_batch_run(self,batch_run_id):
        with closing(self._connection(True)) as c:
            rows=self._rows(c,"SELECT * FROM analysis_batch_runs WHERE batch_run_id=?",(batch_run_id,))
            if not rows:return None
            value=rows[0]
            value["images"]=self._rows(c,"SELECT * FROM analysis_image_runs WHERE batch_run_id=? ORDER BY request_index",(batch_run_id,))
            return value

    def get_analysis_run(self,analysis_run_id):
        with closing(self._connection(True)) as c:
            rows=self._rows(c,"SELECT * FROM analysis_image_runs WHERE analysis_run_id=?",(analysis_run_id,))
            if not rows:return None
            value=rows[0]
            for name,table,order in (("detections","defect_detections","detection_index"),
                    ("measurements","metrology_measurements","measurement_index"),("judgments","line_judgments","line_id")):
                value[name]=[json.loads(r["payload_json"]) for r in self._rows(c,"SELECT payload_json FROM "+table+
                                  " WHERE analysis_run_id=? ORDER BY "+order,(analysis_run_id,))]
            if value["result_path"] is not None:
                p=self._path(value["result_path"],self.result_root)
                if _sha(p)!=value["result_sha256"]:raise AnalysisStorageError("Stored result file changed")
                value["raw_result"]=_read(p)
            else:value["raw_result"]=None
            for stage in ("defect","metrology"):
                if value[stage+"_overlay_path"] is not None:
                    p=self._path(value[stage+"_overlay_path"],self.result_root)
                    if _sha(p)!=value[stage+"_overlay_sha256"]:
                        raise AnalysisStorageError("Stored overlay file changed")
            return value

    def image_history(self,image_id):
        with closing(self._connection(True)) as c:
            return self._rows(c,"SELECT a.* FROM analysis_image_runs a JOIN analysis_batch_runs b ON b.batch_run_id=a.batch_run_id "
                "WHERE a.image_id=? ORDER BY julianday(b.started_at) DESC,a.analysis_run_id DESC",(image_id,))

    def latest_result(self,image_id):
        history=self.image_history(image_id)
        return self.get_analysis_run(history[0]["analysis_run_id"]) if history else None

    def adopted_result(self,image_id):
        with closing(self._connection(True)) as c:
            row=c.execute("SELECT analysis_run_id FROM analysis_image_runs WHERE image_id=? AND is_adopted=1",(image_id,)).fetchone()
        return self.get_analysis_run(row[0]) if row else None
