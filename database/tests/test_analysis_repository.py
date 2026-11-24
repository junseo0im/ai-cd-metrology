"""Isolated v4 Repository tests. Synthetic records are not model detections."""
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import hashlib,json,sqlite3,tempfile,unittest
from unittest.mock import patch
from database.init_db import connect_database,_schema_objects,TABLE_NAMES
from database.migrate_v3 import initialize_database_v3,validate_v3
from database.migrate_v4 import migrate_database_v4,validate_v4,ANALYSIS_TABLES
from database.analysis_repository import AnalysisRepository,AnalysisStorageError
from database.image_registry import register_image,get_image
from database.tests.test_image_registry import png_bytes

TEST_ROOT=Path(__file__).resolve().parents[2]/"runtime_data/validation/task17b/tmp"


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))


class Fixture:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.db=self.root/"test.sqlite3";self.assets=self.root/"artifacts";self.assets.mkdir()
        self.sources=self.root/"sources";self.sources.mkdir()
        initialize_database_v3(self.db)
        with closing(connect_database(self.db)) as c:
            c.execute("INSERT INTO process_runs(process_run_id,exposure_value) VALUES ('TEST_PROCESS',60)")
            c.execute("INSERT INTO wafers(wafer_id,process_run_id) VALUES ('W1','TEST_PROCESS')")
            for code in ("12","38"):
                die="W1/D"+code
                c.execute("INSERT INTO dies VALUES (?,?,?,?,?)",(die,"W1",code,int(code[0]),int(code[1])))
                c.execute("INSERT INTO pattern_sites(physical_site_id,die_id,position_code) VALUES (?,?,?)",(die+"/c",die,"c"))
            for b in ("TEST_BATCH","TEST_BATCH_2"):
                c.execute("INSERT INTO upload_batches(batch_id,wafer_id,upload_mode,dataset_role,batch_status) VALUES (?,'W1','wafer_batch','development','registered')",(b,))
            c.execute("INSERT INTO acquisition_conditions(condition_id,magnification) VALUES ('TEST_CONDITION',20)")
            c.commit()
            for image,batch,code,pixel in (("TEST_IMAGE_1","TEST_BATCH","12",30),("TEST_IMAGE_2","TEST_BATCH","38",60),("TEST_DUPLICATE","TEST_BATCH_2","12",30)):
                p=self.sources/("w1_"+code+"_c.png")
                if not p.exists():p.write_bytes(png_bytes(pixel=pixel))
                register_image(c,p,image_id=image,batch_id=batch,physical_site_id="W1/D"+code+"/c",
                               metadata_status="filename_parsed",condition_id="TEST_CONDITION")

    def upgrade(self):migrate_database_v4(self.db)
    def repo(self):return AnalysisRepository(self.db,result_root=self.assets,image_root=self.sources)

    def build(self,run_id="TEST_RUN_1",modes=None,image_ids=None,batch_id="TEST_BATCH",day=1):
        ids=image_ids or ["TEST_IMAGE_1","TEST_IMAGE_2"];modes=modes or {}
        directory=self.assets/run_id;directory.mkdir()
        summary={"schema":"inspection-batch-v0.1","batch_id":batch_id,"batch_run_id":run_id,
            "started_at":f"2026-01-{day:02d}T00:00:00.123456+00:00","ended_at":f"2026-01-{day:02d}T00:01:00.999999+00:00",
            "execution_status":"COMPLETED","retry_of_run_id":None,"execution_config_sha256":{"mock_fixture":"not a real model"},
            "interruption_reason":None,"images":[]}
        entries=[]
        with closing(connect_database(self.db,readonly=True)) as c:
            registered={i:get_image(c,i) for i in ids}
        for index,image_id in enumerate(ids):
            record=registered[image_id];mode=modes.get(image_id,"complete")
            meta={"wafer_id":"W1","die_id":record["die_id"],"die_code":record["die_code"],"site_code":"c",
                  "physical_site_id":record["physical_site_id"],"condition_id":"TEST_CONDITION",
                  "original_filename":record["original_filename"],"magnification":20,
                  "exposure_value":60,"exposure_unit":None,"peb_temperature_c":None,
                  "dataset_role":"development","upload_mode":"wafer_batch"}
            request={"image_id":image_id,"image_path":record["storage_path"],"batch_id":batch_id,**meta}
            entries.append({"request":request,"validation_error":None})
            analysis_id=run_id+":"+image_id
            common={k:request[k] for k in ("image_id","image_path","batch_id","wafer_id","die_id","site_code")}
            common["analysis_run_id"]=analysis_id
            measures=[{"measurement_type":kind,"line_id":"L01" if kind=="line" else "G01",
              "y_fraction":y,"y_px":int(y*100),"width_px":50.0,"width_um":7.96,
              "left_x_px":10.0,"right_x_px":60.0,"center_x_px":35.0}
              for y in (.25,.45,.65) for kind in ("line","gap")]
            judgment={"line_id":"L01","label":"pn","reason":"MOCK FIXTURE ONLY","min_um":7.96,"max_um":7.96,
                "mean_um":7.96,"n_rows":3,"near_threshold":False,"pitch_ok":True,
                "rows":[{"y_fraction":y,"width_um":7.96} for y in (.25,.45,.65)]}
            result={**common,"status":"SUCCESS","error_code":None,"error_message":None,
              "image_sha256":record["sha256"],"configuration":{"fixture":True},"runner_version":"MOCK",
              "official_analysis_complete":True,"inspection_verdict":None,
              "defect":{**common,"status":"SUCCESS","model_version":"MOCK","model_sha256":"a"*64,
                        "bbox_coordinate_system":"ORIGINAL_IMAGE","bbox_image_width":100,"bbox_image_height":100,
                        "skew_angle":0.0,"detections":[]},
              "metrology":{**common,"status":"SUCCESS","algorithm_version":"MOCK","calibration_id":"MOCK_CALIBRATION",
                "calibration":{"calibration_id":"MOCK_CALIBRATION","um_per_px":.1592,"source":"synthetic unit fixture"},
                "pixel_scale":{"coordinate_system":"native_px","factor_x":1,"factor_y":1,"offset_x":0,"offset_y":0},
                "quality":{"partial_measurement":False,"missing_positions":[]},
                "measurements":measures,"judgment":{"status":"judged","lines":[judgment]}},
              "pd_policy":{"state":"NO_PD","sources":[]},"measurement_summary":{},
              "width_error":{"signed_percent":-20.4,"absolute_percent":20.4},
              "overlays":{"defect":None,"metrology":None}}
            outcome="COMPLETE"
            if mode in ("model_pd","heuristic_pd","detection"):
                source="HEURISTIC" if mode=="heuristic_pd" else "MODEL_POSTPROCESSED"
                code="p" if mode=="detection" else "pd"
                result["defect"]["detections"]=[{"class_code":code,"class_id":1 if code=="p" else 2,
                    "class_name":code,"confidence":.8,"bbox_xyxy":[1.0,2.0,5.0,7.0],"source":source}]
                if mode!="detection":
                    state="MODEL_PD_EXCLUDED" if mode=="model_pd" else "POLICY_REVIEW_REQUIRED"
                    result["pd_policy"]={"state":state,"sources":[source]}
                    result["metrology"].update(status="SKIPPED",measurements=[],judgment={"status":"not_judged","lines":[]})
                    result.update(width_error=None,official_analysis_complete=False)
                    result["status"]="SUCCESS" if mode=="model_pd" else "SKIPPED";outcome=state
            if mode in ("partial","duplicate_y"):
                if mode=="partial":result["metrology"]["measurements"]=measures[1:]
                else:result["metrology"]["measurements"]=measures+[deepcopy(measures[0])]
                result["metrology"]["quality"]={"partial_measurement":True,"missing_positions":[],"fixture_only":True}
                result.update(width_error=None,official_analysis_complete=False);outcome="PARTIAL_MEASUREMENT"
            if mode=="failed":
                result.update(status="FAILED",error_code="MOCK_MEASUREMENT_FAILURE",error_message="fixture",
                              width_error=None,official_analysis_complete=False)
                result["metrology"].update(status="FAILED",measurements=[],judgment={"status":"not_judged","lines":[]})
                outcome="FAILED"
            if mode in ("partial","duplicate_y","failed","model_pd","heuristic_pd"):
                result["metrology"]["quality"]["partial_measurement"]=mode in ("partial","duplicate_y")
            for kind in ("line","gap"):
                values=[m for m in result["metrology"]["measurements"] if m["measurement_type"]==kind]
                result["measurement_summary"][kind]={"count":len(values),"mean_um":7.96 if values else None}
            itemdir=directory/image_id;itemdir.mkdir()
            for stage in ("defect","metrology"):
                if result[stage]["status"]=="SUCCESS":
                    overlay=itemdir/(stage+"_overlay.png");overlay.write_bytes(png_bytes())
                    result["overlays"][stage]={"path":str(overlay)}
            result_path=itemdir/"result.json";save(result_path,result)
            row={"request_index":index,"image_id":image_id,"analysis_run_id":analysis_id,"metadata":meta,
              "started_at":summary["started_at"],"ended_at":summary["ended_at"],
              "processing_status":"FAILED" if mode=="failed" else "COMPLETED",
              "engine_status":result["status"],"defect_status":result["defect"]["status"],
              "metrology_status":result["metrology"]["status"],"outcome":outcome,
              "pd_policy_state":result["pd_policy"]["state"],"reason":result["error_code"],
              "error_message":result["error_message"],"source_image_sha256":record["sha256"],
              "result_json":str(result_path),"result_sha256":sha(result_path)}
            if mode=="pending":
                row.update(processing_status="UNPROCESSED",engine_status=None,defect_status=None,metrology_status=None,
                           outcome="UNPROCESSED",pd_policy_state=None,result_json=None,result_sha256=None,started_at=None,ended_at=None)
            summary["images"].append(row)
        manifest=directory/"request_manifest.json";save(manifest,{"batch_id":batch_id,"batch_run_id":run_id,"entries":entries})
        summary.update(request_manifest_json=str(manifest),request_manifest_sha256=sha(manifest))
        p=directory/"batch_summary.json";save(p,summary);self.refresh(p);return p

    def refresh(self,path):
        s=read(path)
        for r in s["images"]:
            if r["result_json"] is not None:r["result_sha256"]=sha(r["result_json"])
        c={"requested_images":len(s["images"]),"processed_images":sum(r["processing_status"]!="UNPROCESSED" for r in s["images"]),
           "unprocessed_images":sum(r["processing_status"]=="UNPROCESSED" for r in s["images"]),
           "success_images":sum(r["engine_status"]=="SUCCESS" for r in s["images"]),
           "failed_images":sum(r["processing_status"]=="FAILED" for r in s["images"]),
           "skipped_engine_images":sum(r["engine_status"]=="SKIPPED" for r in s["images"]),
           "skipped_metrology_images":sum(r["metrology_status"]=="SKIPPED" for r in s["images"]),
           "complete_measurement_images":sum(r["outcome"]=="COMPLETE" for r in s["images"]),
           "partial_measurement_images":sum(r["outcome"]=="PARTIAL_MEASUREMENT" for r in s["images"]),
           "policy_review_images":sum(r["outcome"]=="POLICY_REVIEW_REQUIRED" for r in s["images"]),
           "quality_review_images":sum(r["outcome"]=="QUALITY_REVIEW_REQUIRED" for r in s["images"]),
           "model_pd_excluded_images":sum(r["outcome"]=="MODEL_PD_EXCLUDED" for r in s["images"])}
        s["counts"]=c
        s["execution_status"]="INTERRUPTED" if c["unprocessed_images"] else "FAILED" if c["failed_images"]==len(s["images"]) else "COMPLETED_WITH_ISSUES" if c["failed_images"] or c["partial_measurement_images"] or c["policy_review_images"] else "COMPLETED"
        save(path,s)
    def counts(self):
        with closing(connect_database(self.db,readonly=True)) as c:
            return {t:c.execute("SELECT COUNT(*) FROM "+t).fetchone()[0] for t in ANALYSIS_TABLES}


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        t=tempfile.TemporaryDirectory(dir=TEST_ROOT);self.addCleanup(t.cleanup)
        self.f=Fixture(Path(t.name));self.f.upgrade();self.repo=self.f.repo()

    def test_save_read_exact_values_no_auto_adoption(self):
        p=self.f.build(modes={"TEST_IMAGE_1":"detection"})
        result=self.repo.save_batch(p);self.assertTrue(result["inserted"])
        for row in read(p)["images"]:
            original=read(row["result_json"]);got=self.repo.get_analysis_run(row["analysis_run_id"])
            self.assertEqual(got["raw_result"],original)
            self.assertEqual(got["measurements"],original["metrology"]["measurements"])
            self.assertEqual(got["detections"],original["defect"]["detections"])
            self.assertEqual(got["judgments"],original["metrology"]["judgment"]["lines"])
            self.assertEqual(got["is_adopted"],0)
            self.assertIsNone(self.repo.adopted_result(row["image_id"]))
        self.assertEqual(self.repo.get_batch_run("TEST_RUN_1")["requested_images"],2)

    def test_idempotent_and_conflicting_same_run(self):
        p=self.f.build();self.repo.save_batch(p);counts=self.f.counts()
        self.assertFalse(self.repo.save_batch(p)["inserted"]);self.assertEqual(counts,self.f.counts())
        s=read(p);s["interruption_reason"]="different content";save(p,s)
        with self.assertRaises(AnalysisStorageError):self.repo.save_batch(p)
        self.assertEqual(counts,self.f.counts())

    def test_reanalysis_latest_and_history_separate(self):
        p=self.f.build();self.repo.save_batch(p)
        p2=self.f.build("TEST_RUN_2",day=2);s=read(p2);s["retry_of_run_id"]="TEST_RUN_1";save(p2,s)
        self.repo.save_batch(p2)
        self.assertEqual(len(self.repo.image_history("TEST_IMAGE_1")),2)
        self.assertEqual(self.repo.latest_result("TEST_IMAGE_1")["analysis_run_id"],"TEST_RUN_2:TEST_IMAGE_1")
        self.assertIsNone(self.repo.adopted_result("TEST_IMAGE_1"))
        self.assertIsNotNone(self.repo.get_analysis_run("TEST_RUN_1:TEST_IMAGE_1"))

    def test_pd_partial_failed_pending(self):
        modes={"TEST_IMAGE_1":"model_pd","TEST_IMAGE_2":"heuristic_pd"}
        p=self.f.build(modes=modes);self.repo.save_batch(p)
        values=self.repo.get_batch_run("TEST_RUN_1")["images"]
        self.assertEqual([r["metrology_status"] for r in values],["SKIPPED","SKIPPED"])
        self.assertEqual([r["is_adopted"] for r in values],[0,0])
        for row in values:self.assertIsNone(row["signed_error_pct"])
        p=self.f.build("TEST_RUN_2",modes={"TEST_IMAGE_1":"partial","TEST_IMAGE_2":"failed"})
        self.repo.save_batch(p)
        self.assertEqual(self.repo.get_analysis_run("TEST_RUN_2:TEST_IMAGE_1")["partial_measurement"],1)
        p=self.f.build("TEST_RUN_3",modes={"TEST_IMAGE_2":"pending"});self.repo.save_batch(p)
        pending=self.repo.get_analysis_run("TEST_RUN_3:TEST_IMAGE_2")
        self.assertEqual(pending["processing_status"],"UNPROCESSED")
        self.assertIsNone(pending["raw_result"])

    def test_duplicate_y_preserved(self):
        p=self.f.build(modes={"TEST_IMAGE_1":"duplicate_y"});self.repo.save_batch(p)
        result=self.repo.get_analysis_run("TEST_RUN_1:TEST_IMAGE_1")
        self.assertEqual(len(result["measurements"]),7)

    def test_wrong_registered_batch_and_unknown_qa_id(self):
        p=self.f.build(image_ids=["TEST_DUPLICATE"])
        with self.assertRaises(AnalysisStorageError):self.repo.save_batch(p)
        s=read(p);s["batch_id"]="QA:not-registered";save(p,s)
        with self.assertRaises(AnalysisStorageError):self.repo.save_batch(p)
        self.assertFalse(any(self.f.counts().values()))

    def test_duplicate_sha_different_registered_ids_remain_distinct(self):
        self.repo.save_batch(self.f.build(image_ids=["TEST_IMAGE_1"]))
        self.repo.save_batch(self.f.build("RUN_B2",image_ids=["TEST_DUPLICATE"],batch_id="TEST_BATCH_2"))
        self.assertEqual(len(self.repo.image_history("TEST_IMAGE_1")),1)
        self.assertEqual(len(self.repo.image_history("TEST_DUPLICATE")),1)

    def test_source_sha_mismatch_rejected(self):
        p=self.f.build()
        with closing(connect_database(self.f.db)) as c:c.execute("UPDATE images SET sha256=? WHERE image_id='TEST_IMAGE_1'",("f"*64,));c.commit()
        with self.assertRaises(AnalysisStorageError):self.repo.save_batch(p)
        self.assertFalse(any(self.f.counts().values()))

    def test_result_hash_id_and_path_escape(self):
        for case in ("hash","image_id","run_id","escape"):
            with self.subTest(case=case):
                p=self.f.build("RUN_"+case)
                s=read(p);row=s["images"][0]
                if case=="hash":Path(row["result_json"]).write_text("{}",encoding="utf-8")
                elif case=="escape":row["result_json"]=str(self.f.sources/"w1_12_c.png")
                else:
                    r=read(row["result_json"]);r["image_id" if case=="image_id" else "analysis_run_id"]="wrong";save(row["result_json"],r);row["result_sha256"]=sha(row["result_json"])
                save(p,s)
                with self.assertRaises((AnalysisStorageError,ValueError)):self.repo.save_batch(p)
        self.assertFalse(any(self.f.counts().values()))

    def test_missing_overlay_and_collision(self):
        for case in ("missing","collision"):
            p=self.f.build("RUN_"+case);s=read(p)
            raw=read(s["images"][0]["result_json"])
            overlay=Path(raw["overlays"]["defect"]["path"])
            if case=="missing":overlay.unlink()
            else:
                second=read(s["images"][1]["result_json"])
                second["overlays"]["defect"]["path"]=str(overlay)
                save(s["images"][1]["result_json"],second);self.f.refresh(p)
            with self.assertRaises((AnalysisStorageError,OSError)):self.repo.save_batch(p)
        self.assertFalse(any(self.f.counts().values()))

    def test_count_mismatch_and_nonfinite_json(self):
        p=self.f.build();s=read(p);s["counts"]["success_images"]=1;save(p,s)
        with self.assertRaises(AnalysisStorageError):self.repo.save_batch(p)
        Path(p).write_text('{"value":NaN}',encoding="utf-8")
        with self.assertRaises(AnalysisStorageError):self.repo.save_batch(p)

    def test_transaction_rollback_keeps_previous_history(self):
        self.repo.save_batch(self.f.build());before=self.f.counts()
        p=self.f.build("TEST_RUN_2")
        import database.analysis_repository as module
        original=module._insert
        def fail(c,t,row):
            if t=="metrology_measurements":raise RuntimeError("injected insert failure")
            return original(c,t,row)
        with patch.object(module,"_insert",side_effect=fail):
            with self.assertRaises(RuntimeError):self.repo.save_batch(p)
        self.assertEqual(self.f.counts(),before)
        self.assertIsNone(self.repo.get_batch_run("TEST_RUN_2"))
        self.assertTrue(self.repo.save_batch(p)["inserted"])

    def test_pk_fk_membership_and_one_adopted(self):
        self.repo.save_batch(self.f.build());self.repo.save_batch(self.f.build("TEST_RUN_2",day=2))
        with closing(connect_database(self.f.db)) as c:
            c.execute("UPDATE analysis_image_runs SET is_adopted=1 WHERE analysis_run_id='TEST_RUN_1:TEST_IMAGE_1'");c.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                c.execute("UPDATE analysis_image_runs SET is_adopted=1 WHERE analysis_run_id='TEST_RUN_2:TEST_IMAGE_1'")
            c.rollback()
            with self.assertRaises(sqlite3.IntegrityError):
                c.execute("UPDATE analysis_image_runs SET image_id='TEST_DUPLICATE' WHERE analysis_run_id='TEST_RUN_1:TEST_IMAGE_2'")
            c.rollback()
            with self.assertRaises(sqlite3.IntegrityError):
                c.execute("UPDATE images SET batch_id='TEST_BATCH_2' WHERE image_id='TEST_IMAGE_1'")
            c.rollback()
            with self.assertRaises(sqlite3.IntegrityError):
                c.execute("INSERT INTO line_judgments(analysis_run_id,line_id,label,payload_json) VALUES ('missing','L01','pn','{}')")
            c.rollback()
            self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(),[])
            self.assertEqual(c.execute("PRAGMA integrity_check").fetchall(),[("ok",)])
        self.assertEqual(self.repo.adopted_result("TEST_IMAGE_1")["analysis_run_id"],"TEST_RUN_1:TEST_IMAGE_1")

    def test_ineligible_adoption_refused(self):
        self.repo.save_batch(self.f.build(modes={"TEST_IMAGE_1":"partial","TEST_IMAGE_2":"model_pd"}))
        with closing(connect_database(self.f.db)) as c:
            for identity in ("TEST_RUN_1:TEST_IMAGE_1","TEST_RUN_1:TEST_IMAGE_2"):
                with self.assertRaises(sqlite3.IntegrityError):
                    c.execute("UPDATE analysis_image_runs SET is_adopted=1 WHERE analysis_run_id=?",(identity,))
                c.rollback()

    def test_tampered_files_rejected_on_read_and_same_run_save(self):
        p=self.f.build();self.repo.save_batch(p)
        raw=read(read(p)["images"][0]["result_json"])
        Path(raw["overlays"]["defect"]["path"]).write_bytes(b"changed")
        with self.assertRaises(AnalysisStorageError):self.repo.get_analysis_run("TEST_RUN_1:TEST_IMAGE_1")
        with self.assertRaises(AnalysisStorageError):self.repo.save_batch(p)

    def test_nested_ids_and_pd_state_contradictions_rejected(self):
        for case in ("nested","pd"):
            p=self.f.build("RUN_"+case);summary=read(p);path=summary["images"][0]["result_json"]
            raw=read(path)
            if case=="nested":raw["metrology"]["analysis_run_id"]="wrong"
            else:raw["defect"]["detections"]=[{"class_code":"pd","source":"MODEL_POSTPROCESSED","confidence":.8,"bbox_xyxy":[1,2,3,4]}]
            save(path,raw);self.f.refresh(p)
            with self.assertRaises(AnalysisStorageError):self.repo.save_batch(p)
        self.assertFalse(any(self.f.counts().values()))

    def test_null_quality_cannot_be_adopted(self):
        self.repo.save_batch(self.f.build())
        with closing(connect_database(self.f.db)) as c:
            c.execute("UPDATE analysis_image_runs SET partial_measurement=NULL WHERE analysis_run_id='TEST_RUN_1:TEST_IMAGE_1'")
            c.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                c.execute("UPDATE analysis_image_runs SET is_adopted=1 WHERE analysis_run_id='TEST_RUN_1:TEST_IMAGE_1'")
            c.rollback()

    def test_operational_repository_guard(self):
        with patch("database.migrate_v4.DEFAULT_DB",self.f.db):
            with self.assertRaises(PermissionError):self.f.repo()


if __name__=="__main__":unittest.main()
