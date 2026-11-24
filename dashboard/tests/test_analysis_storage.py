"""TASK 18-A-3: store/re-read the real TASK 18-A-2 result via the existing AnalysisRepository.

Writes go ONLY to a temporary copy of the operational DB. The real result files are
only read (hashes asserted unchanged). No model inference is executed.
"""
from contextlib import closing
from pathlib import Path
import glob,hashlib,json,os,shutil,sqlite3,tempfile,unittest
from unittest.mock import patch

from database.init_db import DEFAULT_DB
from database.analysis_repository import AnalysisRepository,AnalysisStorageError
import dashboard.components.analysis_management as am

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/"runtime_data/validation/task18a2/dashboard_runs"
RUN_DIR=RESULTS/"dcc-20261119T052558Z-9d85c0ed"
IMAGE="IMG-57eca064ca764e37be03e6cd62c7f079"
BATCH="BATCH-7896834d126b4861bb9cf6fdee44b091"
TABLES=("analysis_batch_runs","analysis_image_runs","defect_detections","metrology_measurements","line_judgments")
TEST_ROOT=ROOT/"runtime_data/validation/task18a3/tmp"
SUMMARIES=glob.glob(str(RUN_DIR/"runs/batch-*/batch_summary.json"))
AVAILABLE=DEFAULT_DB.is_file() and len(SUMMARIES)==1
try:
    import plotly  # noqa: F401
    HAS_PLOTLY=True
except ImportError:
    HAS_PLOTLY=False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_hashes():
    return {p:sha(p) for p in sorted(RUN_DIR.rglob("*")) if p.is_file()}


@unittest.skipUnless(AVAILABLE,"TASK 18-A-2 artifacts or operational DB not present")
class StorageBase(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=TEST_ROOT);self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.db=self.root/"ida_copy.sqlite3"
        self.operational_sha=sha(DEFAULT_DB);self.artifacts=artifact_hashes()
        with closing(sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro",uri=True)) as src,closing(sqlite3.connect(self.db)) as dst:
            src.backup(dst)
        self.summary=Path(SUMMARIES[0])
        row=json.loads(self.summary.read_text(encoding="utf-8"))["images"][0]
        self.raw=json.loads(Path(row["result_json"]).read_text(encoding="utf-8"))
        self.run_id=row["analysis_run_id"]
        # The operational copy may already hold approved runs (TASK18-A-4); compare deltas.
        self.base=self.counts()
    def plus(self,**delta):
        return {t:self.base[t]+delta.get(t,0) for t in TABLES}
    def tearDown(self):
        self.assertEqual(sha(DEFAULT_DB),self.operational_sha)  # Operational DB never written.
        self.assertEqual(artifact_hashes(),self.artifacts)       # Analysis files never modified.
    def repo(self):
        return AnalysisRepository(self.db,result_root=RESULTS,image_root=ROOT/"runtime_data/raw")
    def counts(self):
        with closing(sqlite3.connect(self.db)) as c:
            return {t:c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    def integrity(self):
        with closing(sqlite3.connect(self.db)) as c:
            self.assertEqual(c.execute("PRAGMA integrity_check").fetchone()[0],"ok")
            self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(),[])


class RepositoryStorageTests(StorageBase):
    def test_01_save_and_reread_matches_real_result_json(self):
        saved=self.repo().save_batch(self.summary)
        metrology=self.raw["metrology"];judgment_lines=metrology["judgment"]["lines"]
        self.assertEqual(saved,{"batch_run_id":self.raw["analysis_run_id"].split(":")[0],"inserted":True,"images":1,
            "detections":len(self.raw["defect"]["detections"]),"measurements":len(metrology["measurements"]),
            "judgments":len(judgment_lines)})
        # Reference values from the TASK 18-A-2 report, cross-checked against the JSON itself.
        self.assertEqual(len(self.raw["defect"]["detections"]),0)
        kinds=[m["measurement_type"] for m in metrology["measurements"]]
        self.assertEqual((kinds.count("line"),kinds.count("gap")),(27,24))
        batch=self.repo().get_batch_run(saved["batch_run_id"])
        self.assertEqual((batch["batch_id"],batch["execution_status"],batch["quality_review_images"],
                          batch["complete_measurement_images"]),(BATCH,"COMPLETED_WITH_ISSUES",1,0))
        run=self.repo().get_analysis_run(self.run_id)
        summary=self.raw["measurement_summary"]
        self.assertEqual((run["image_id"],run["outcome"],run["engine_status"],run["pd_policy_state"],run["is_adopted"]),
                         (IMAGE,"QUALITY_REVIEW_REQUIRED","SUCCESS","NO_PD",0))
        self.assertEqual((run["line_mean_um"],run["gap_mean_um"],run["line_measurement_count"],run["gap_measurement_count"]),
                         (summary["line"]["mean_um"],summary["gap"]["mean_um"],27,24))
        self.assertAlmostEqual(run["line_mean_um"],8.527,places=3);self.assertAlmostEqual(run["gap_mean_um"],9.806,places=3)
        self.assertEqual((run["signed_error_pct"],run["absolute_error_pct"]),
                         (self.raw["width_error"]["signed_percent"],self.raw["width_error"]["absolute_percent"]))
        self.assertEqual(run["calibration_id"],"optiview-native-0.1592")
        self.assertEqual(json.loads(run["calibration_json"])["um_per_px"],0.1592)
        self.assertEqual(run["partial_measurement"],0)
        self.assertEqual(run["detections"],[])
        self.assertEqual(run["measurements"],metrology["measurements"])
        self.assertEqual(run["judgments"],sorted(judgment_lines,key=lambda j:j["line_id"]))
        self.assertEqual(run["raw_result"],self.raw)
        self.assertEqual([j["line_id"] for j in run["judgments"] if j["near_threshold"]],["L04","L07","L08"])
        for key in ("result_path","defect_overlay_path","metrology_overlay_path"):
            self.assertTrue((RESULTS/run[key]).is_file(),key)
        self.assertEqual(sha(RESULTS/run["result_path"]),run["result_sha256"])
        self.assertIsNone(self.repo().adopted_result(IMAGE))  # Review result is never auto-adopted.
        self.assertIn(self.run_id,[r["analysis_run_id"] for r in self.repo().image_history(IMAGE)])
        self.assertEqual(self.counts(),self.plus(analysis_batch_runs=1,analysis_image_runs=1,
                                                 metrology_measurements=51,line_judgments=9))
        self.integrity()

    def test_02_duplicate_save_is_idempotent(self):
        self.repo().save_batch(self.summary);before=self.counts()
        self.assertEqual(self.repo().save_batch(self.summary)["inserted"],False)
        self.assertEqual(self.counts(),before);self.integrity()

    def test_03_failure_mid_insert_rolls_back_everything(self):
        import database.analysis_repository as module
        original=module._insert
        def failing(c,table,row):
            if table=="metrology_measurements" and row["measurement_index"]==10:raise sqlite3.OperationalError("injected")
            return original(c,table,row)
        with patch.object(module,"_insert",side_effect=failing):
            with self.assertRaises(sqlite3.OperationalError):self.repo().save_batch(self.summary)
        self.assertEqual(self.counts(),self.base);self.integrity()
        self.assertTrue(self.repo().save_batch(self.summary)["inserted"])  # Clean retry still possible.

    def test_04_operational_db_refused(self):
        with self.assertRaises(PermissionError):
            AnalysisRepository(DEFAULT_DB,result_root=RESULTS,image_root=ROOT/"runtime_data/raw")


class Done:
    def poll(self):return 0
    def wait(self,timeout=None):return 0


class SaveFunctionTests(StorageBase):
    """save_analysis_result decisions; no subprocess, no inference."""
    def provider(self,db):
        return type("P",(),{"db_path":db,"storage_root":ROOT/"runtime_data/raw"})()
    def result(self,**changes):
        with patch.dict(am._ACTIVE,{"process":Done(),"log":None,"run_dir":RUN_DIR,"image_id":IMAGE,"batch_id":BATCH}):
            result=am.wait_for_analysis()
        result.update(changes);return result
    def save(self,result,db=None):
        with patch.dict(os.environ,{"IDA_ANALYSIS_OUTPUT_ROOT":str(RESULTS)}):
            return am.save_analysis_result(result,self.provider(db or self.db))

    def test_05_saved_then_already_saved(self):
        self.assertEqual(self.save(self.result())["status"],"SAVED")
        self.assertEqual(self.save(self.result())["status"],"ALREADY_SAVED")
        self.assertEqual(self.counts(),self.plus(analysis_batch_runs=1,analysis_image_runs=1,
                                                 metrology_measurements=51,line_judgments=9))

    def test_06_failed_or_mismatched_results_never_saved(self):
        with patch.object(am,"AnalysisRepository",side_effect=AssertionError("must not be called")):
            for changes in ({"execution_status":"FAILED"},{"execution_status":"INTERRUPTED"},{"returncode":1},
                            {"requested":{"image_id":"IMG-other","batch_id":BATCH}},
                            {"requested":{"image_id":IMAGE,"batch_id":"BATCH-other"}},
                            {"groups":[{"batch_id":BATCH,"execution_status":"COMPLETED_WITH_ISSUES",
                                        "summary_json":str(self.root/"missing.json"),"images":[]}]}):
                self.assertEqual(self.save(self.result(**changes))["status"],"NOT_SAVED",changes)
        self.assertEqual(self.counts(),self.base)

    def test_07_operational_db_disabled_without_repository_call(self):
        with patch.object(am,"AnalysisRepository",side_effect=AssertionError("must not be called")):
            self.assertEqual(self.save(self.result(),DEFAULT_DB)["status"],"DISABLED")

    def test_08_storage_error_reported_separately_and_files_kept(self):
        with patch.object(AnalysisRepository,"save_batch",side_effect=AnalysisStorageError("injected")):
            saved=self.save(self.result())
        self.assertEqual(saved["status"],"FAILED");self.assertIn("결과 파일은 보존",saved["message"])
        self.assertEqual(self.counts(),self.base)


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed; Data Control page cannot load in AppTest")
class DashboardSavePathTests(StorageBase):
    """Real Data Control page; the analysis subprocess is replaced by the finished 18-A-2 run."""
    def setUp(self):
        super().setUp()
        env=patch.dict(os.environ,{"IDA_DB_PATH":str(self.db),"IDA_STORAGE_ROOT":str(ROOT/"runtime_data/raw"),
            "IDA_STAGING_ROOT":str(self.root/"staging"),"IDA_ANALYSIS_OUTPUT_ROOT":str(RESULTS)})
        def reuse_finished_run(groups,expected):
            requests=[r for g in groups for r in g["requests"]]
            self.assertEqual([(r["image_id"],r["batch_id"]) for r in requests],[(IMAGE,BATCH)])
            self.assertEqual(set(expected),{IMAGE})
            am._ACTIVE.update(process=Done(),log=None,run_dir=RUN_DIR,requested={IMAGE:BATCH})
            return RUN_DIR
        for p in (env,patch.object(am,"start_analysis",side_effect=reuse_finished_run),
                  patch.object(am,"analysis_python",return_value=Path(os.sys.executable)),
                  patch("streamlit.file_uploader",return_value=[])):
            p.start();self.addCleanup(p.stop)
        am._ACTIVE.clear()
    def click(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=60).run()
        at.switch_page("pages/data_control.py").run()
        at.session_state["data_control_selected_source"]="SQLite";at.run()
        at.radio(key="dcc_analysis_scope").set_value("Image").run()
        at.selectbox(key="dcc_analysis_target_image").select(IMAGE).run()
        self.assertEqual(at.session_state["dcc_analysis_input_plan"]["counts"],{"READY":1,"REVIEW_REQUIRED":0,"BLOCKED":0})
        self.assertEqual(at.button(key="dcc_analysis_run").label,"분석 실행")
        at.button(key="dcc_analysis_run").click().run()
        self.assertFalse(at.exception)
        return at

    def test_09_dashboard_saves_once_and_reports_duplicate(self):
        at=self.click();last=at.session_state["dcc_analysis_last_run"]
        self.assertEqual((last["execution_status"],last["outcome"],last["db_save"]["status"]),
                         ("COMPLETED_WITH_ISSUES","QUALITY_REVIEW_REQUIRED","SAVED"))
        self.assertTrue(any("DB 저장 완료" in c.value for c in at.caption))
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertEqual(self.counts()["metrology_measurements"],self.base["metrology_measurements"]+51)
        at=self.click()  # New session, same finished run: no second insert.
        self.assertEqual(at.session_state["dcc_analysis_last_run"]["db_save"]["status"],"ALREADY_SAVED")
        self.assertEqual(self.counts()["analysis_image_runs"],self.base["analysis_image_runs"]+1);self.integrity()

    def test_10_dashboard_storage_failure_keeps_analysis_status(self):
        with patch.object(AnalysisRepository,"save_batch",side_effect=AnalysisStorageError("injected")):
            at=self.click()
        last=at.session_state["dcc_analysis_last_run"]
        self.assertEqual((last["execution_status"],last["db_save"]["status"]),("COMPLETED_WITH_ISSUES","FAILED"))
        self.assertTrue(any("COMPLETED_WITH_ISSUES" in w.value for w in at.warning))
        self.assertTrue(any("DB 저장 실패" in c.value for c in at.caption))
        self.assertEqual(self.counts(),self.base)


if __name__=="__main__":unittest.main()
