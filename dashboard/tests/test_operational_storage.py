"""TASK 18-A-4: official result root + explicit operational-save opt-in.

The real operational DB is NEVER opened for writing here. A temporary Online-Backup copy
is treated as "operational" by patching the guard's DEFAULT_DB, and a copy of the real
TASK 18-A-2 run is relocated into a temporary "official root" (path strings only; all
measurements/statuses untouched). No inference is executed.
"""
from contextlib import closing
from pathlib import Path
import glob,hashlib,json,os,shutil,sqlite3,tempfile,unittest
from unittest.mock import patch

import database.analysis_repository as repo_module
from database.analysis_repository import AnalysisRepository,OPERATIONAL_RESULT_ROOT
from database.init_db import DEFAULT_DB
import dashboard.components.analysis_management as am
from dashboard.tests.test_analysis_storage import (RUN_DIR,RESULTS,IMAGE,BATCH,TABLES,TEST_ROOT,AVAILABLE,
    HAS_PLOTLY,Done,sha,artifact_hashes)

ROOT=Path(__file__).resolve().parents[2]


def _rewrite(value,old,new):
    if isinstance(value,dict):return {k:_rewrite(v,old,new) for k,v in value.items()}
    if isinstance(value,list):return [_rewrite(v,old,new) for v in value]
    return value.replace(old,new) if isinstance(value,str) else value


def _dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def relocate_run(destination_root):
    """Copy the real run under a new root; rewrite only embedded path strings."""
    target=Path(destination_root)/RUN_DIR.name
    shutil.copytree(RUN_DIR,target)
    old,new=str(RUN_DIR),str(target)
    summary_path=next(target.glob("runs/batch-*/batch_summary.json"))
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary["images"]:
        result_path=Path(row["result_json"].replace(old,new))
        _dump(result_path,_rewrite(json.loads(result_path.read_text(encoding="utf-8")),old,new))
        row["result_sha256"]=sha(result_path)
    summary=_rewrite(summary,old,new)
    _dump(summary_path,summary)
    return target


@unittest.skipUnless(AVAILABLE,"TASK 18-A-2 artifacts or operational DB not present")
class OperationalBase(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=TEST_ROOT);self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.db=self.root/"ida_operational_copy.sqlite3"
        self.official=self.root/"analysis_results/dashboard_runs";self.official.mkdir(parents=True)
        self.operational_sha=sha(DEFAULT_DB);self.artifacts=artifact_hashes()
        with closing(sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro",uri=True)) as src,closing(sqlite3.connect(self.db)) as dst:
            src.backup(dst)
        self.run=relocate_run(self.official)
        # The temporary copy plays the operational DB; the official root is the temp one.
        for p in (patch("database.migrate_v4.DEFAULT_DB",self.db),
                  patch.object(repo_module,"OPERATIONAL_RESULT_ROOT",self.official)):
            p.start();self.addCleanup(p.stop)
        os.environ.pop(am.OPERATIONAL_SAVE_SWITCH,None)
        am._ACTIVE.clear()
        self.base=self.counts()  # Operational copy may already hold approved runs.
    def plus(self,**delta):
        return {t:self.base[t]+delta.get(t,0) for t in TABLES}
    def tearDown(self):
        os.environ.pop(am.OPERATIONAL_SAVE_SWITCH,None)
        self.assertEqual(sha(DEFAULT_DB),self.operational_sha)
        self.assertEqual(artifact_hashes(),self.artifacts)
    def counts(self):
        with closing(sqlite3.connect(self.db)) as c:
            return {t:c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    def result(self,run_dir=None):
        with patch.dict(am._ACTIVE,{"process":Done(),"log":None,"run_dir":run_dir or self.run,
                                    "image_id":IMAGE,"batch_id":BATCH}):
            return am.wait_for_analysis()
    def save(self,result,*,switch=False,output_root=None):
        env={"IDA_ANALYSIS_OUTPUT_ROOT":str(output_root or self.official)}
        if switch:env[am.OPERATIONAL_SAVE_SWITCH]="1"
        provider=type("P",(),{"db_path":self.db,"storage_root":ROOT/"runtime_data/raw"})()
        with patch.dict(os.environ,env):
            return am.save_analysis_result(result,provider)


class GuardTests(OperationalBase):
    def test_01_official_root_constant_and_default_output(self):
        self.assertEqual(OPERATIONAL_RESULT_ROOT,ROOT/"runtime_data/analysis_results/dashboard_runs")
        self.assertEqual(am.DEFAULT_OUTPUT_ROOT,OPERATIONAL_RESULT_ROOT)
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop("IDA_ANALYSIS_OUTPUT_ROOT",None)
            self.assertEqual(am.analysis_output_root(),OPERATIONAL_RESULT_ROOT)

    def test_02_default_callers_still_refused_even_with_switch(self):
        with patch.dict(os.environ,{am.OPERATIONAL_SAVE_SWITCH:"1"}):
            with self.assertRaises(PermissionError):
                AnalysisRepository(self.db,result_root=self.official,image_root=ROOT/"runtime_data/raw")
            with self.assertRaises(PermissionError):  # Truthy non-True values are not an opt-in.
                AnalysisRepository(self.db,result_root=self.official,image_root=ROOT/"runtime_data/raw",allow_operational=1)

    def test_03_operational_opt_in_requires_official_root(self):
        with self.assertRaises(PermissionError):
            AnalysisRepository(self.db,result_root=RESULTS,image_root=ROOT/"runtime_data/raw",allow_operational=True)
        AnalysisRepository(self.db,result_root=self.official,image_root=ROOT/"runtime_data/raw",allow_operational=True)
        self.assertEqual(self.counts(),self.base)

    def test_04_real_operational_db_untouched_by_default_paths(self):
        with patch("database.migrate_v4.DEFAULT_DB",DEFAULT_DB):
            with self.assertRaises(PermissionError):
                AnalysisRepository(DEFAULT_DB,result_root=RESULTS,image_root=ROOT/"runtime_data/raw")


class OperationalSaveTests(OperationalBase):
    def test_05_switch_off_is_disabled(self):
        with patch.object(am,"AnalysisRepository",side_effect=AssertionError("must not be called")):
            self.assertEqual(self.save(self.result())["status"],"DISABLED")
        self.assertEqual(self.counts(),self.base)

    def test_06_switch_on_saves_verifies_and_dedupes(self):
        saved=self.save(self.result(),switch=True)
        self.assertEqual((saved["status"],saved["verified"],saved["operational"]),("SAVED",True,True))
        self.assertEqual(self.counts(),self.plus(analysis_batch_runs=1,analysis_image_runs=1,
                                                 metrology_measurements=51,line_judgments=9))
        again=self.save(self.result(),switch=True)
        self.assertEqual((again["status"],again["verified"]),("ALREADY_SAVED",True))
        self.assertEqual(self.counts()["analysis_image_runs"],self.base["analysis_image_runs"]+1)
        repository=AnalysisRepository(self.db,result_root=self.official,image_root=ROOT/"runtime_data/raw",allow_operational=True)
        stored=repository.get_analysis_run(saved["analysis_run_id"])
        original=json.loads(next(self.run.glob("runs/batch-*/images/*/analysis/result.json")).read_text(encoding="utf-8"))
        self.assertEqual(stored["raw_result"],original)
        self.assertEqual((stored["outcome"],stored["is_adopted"],stored["line_measurement_count"],stored["gap_measurement_count"]),
                         ("QUALITY_REVIEW_REQUIRED",0,27,24))
        self.assertEqual(original["measurement_summary"]["line"]["mean_um"],stored["line_mean_um"])
        self.assertIsNone(repository.adopted_result(IMAGE))
        for key in ("result_path","defect_overlay_path","metrology_overlay_path"):
            self.assertTrue(stored[key].startswith(self.run.name+"/"),stored[key])
            self.assertTrue((self.official/stored[key]).is_file())
        with closing(sqlite3.connect(self.db)) as c:
            self.assertEqual(c.execute("PRAGMA integrity_check").fetchone()[0],"ok")
            self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(),[])

    def test_07_validation_results_cannot_be_back_registered(self):
        saved=self.save(self.result(RUN_DIR),switch=True,output_root=RESULTS)
        self.assertEqual(saved["status"],"FAILED");self.assertIn("official result root",saved["message"])
        self.assertEqual(self.counts(),self.base)

    def test_08_partial_failure_rolls_back(self):
        original=repo_module._insert
        def failing(c,table,row):
            if table=="line_judgments":raise sqlite3.OperationalError("injected")
            return original(c,table,row)
        with patch.object(repo_module,"_insert",side_effect=failing):
            saved=self.save(self.result(),switch=True)
        self.assertEqual(saved["status"],"FAILED");self.assertEqual(self.counts(),self.base)
        self.assertEqual(self.save(self.result(),switch=True)["status"],"SAVED")

    def test_09_readback_mismatch_is_reported_as_failure(self):
        with patch.object(AnalysisRepository,"get_analysis_run",return_value=None):
            saved=self.save(self.result(),switch=True)
        self.assertEqual(saved["status"],"FAILED");self.assertIn("재조회 검증 실패",saved["message"])

    def test_10_run_directories_are_unique_and_never_reused(self):
        class Popen:
            def __init__(self,*a,**k):pass
            def poll(self):return 0
        groups=[{"batch_id":BATCH,"requests":[{"image_id":IMAGE,"batch_id":BATCH}]}];expected={IMAGE:"0"*64}
        with patch.dict(os.environ,{"IDA_ANALYSIS_OUTPUT_ROOT":str(self.official),"IDA_ANALYSIS_PYTHON":os.sys.executable}),\
             patch.object(am.subprocess,"Popen",Popen):
            first=am.start_analysis(groups,expected);am._ACTIVE["log"].close();am._ACTIVE.clear()
            second=am.start_analysis(groups,expected);am._ACTIVE["log"].close();am._ACTIVE.clear()
        self.assertNotEqual(first,second)
        self.assertTrue(first.parent==second.parent==self.official.resolve() or first.parent==self.official)


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed; Data Control page cannot load in AppTest")
class DashboardOperationalTests(OperationalBase):
    def setUp(self):
        super().setUp()
        def reuse_run(groups,expected):
            am._ACTIVE.update(process=Done(),log=None,run_dir=self.run,
                              requested={r["image_id"]:g["batch_id"] for g in groups for r in g["requests"]})
            return self.run
        for p in (patch.dict(os.environ,{"IDA_DB_PATH":str(self.db),"IDA_STORAGE_ROOT":str(ROOT/"runtime_data/raw"),
                      "IDA_STAGING_ROOT":str(self.root/"staging"),"IDA_ANALYSIS_OUTPUT_ROOT":str(self.official)}),
                  patch.object(am,"start_analysis",side_effect=reuse_run),
                  patch.object(am,"analysis_python",return_value=Path(os.sys.executable)),
                  patch("streamlit.file_uploader",return_value=[])):
            p.start();self.addCleanup(p.stop)
    def click(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=60).run()
        at.switch_page("pages/data_control.py").run()
        at.session_state["data_control_selected_source"]="SQLite";at.run()
        at.radio(key="dcc_analysis_scope").set_value("Image").run()
        at.selectbox(key="dcc_analysis_target_image").select(IMAGE).run()
        at.button(key="dcc_analysis_run").click().run()
        self.assertFalse(at.exception)
        return at.session_state["dcc_analysis_last_run"],at

    def test_11_dashboard_disabled_then_saved_with_switch(self):
        last,at=self.click()
        self.assertEqual((last["execution_status"],last["db_save"]["status"]),("COMPLETED_WITH_ISSUES","DISABLED"))
        self.assertTrue(any("DB 저장 비활성" in c.value for c in at.caption))
        self.assertEqual(self.counts(),self.base)
        os.environ[am.OPERATIONAL_SAVE_SWITCH]="1"
        last,at=self.click()
        self.assertEqual((last["db_save"]["status"],last["db_save"]["verified"]),("SAVED",True))
        self.assertTrue(any("DB 저장 완료 · 재조회 확인" in c.value for c in at.caption))
        last,_=self.click()
        self.assertEqual(last["db_save"]["status"],"ALREADY_SAVED")
        self.assertEqual(self.counts()["analysis_image_runs"],self.base["analysis_image_runs"]+1)


if __name__=="__main__":unittest.main()
