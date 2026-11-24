"""Scoped Calibration matching and metadata binding; isolated fixture writes only."""
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import json,os,tempfile,unittest
from unittest.mock import patch
from database.init_db import connect_database
from database.migrate_v2 import data_snapshot
from database.capture_conditions import bind_batch_condition,data_fingerprint
from database.tests.test_analysis_repository import Fixture
from inspection.calibration_profiles import load_profiles,condition_note
from dashboard.sqlite_provider import SQLiteProvider
ROOT=Path(__file__).resolve().parents[2]
TEST_ROOT=ROOT/"runtime_data/validation/task18a1/calibration_matching/tmp"

class MatchingTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=TEST_ROOT);self.addCleanup(temporary.cleanup)
        self.f=Fixture(Path(temporary.name));self.f.upgrade()
        self.p=SQLiteProvider(self.f.db,self.f.sources,self.f.root/"staging")
        with closing(connect_database(self.f.db)) as c:
            c.execute("UPDATE images SET condition_id=NULL WHERE batch_id='TEST_BATCH'");c.commit()
        self.profile={"profile_id":"fixture-approved-20x","calibration_id":"optiview-native-0.1592",
          "magnification":20,"um_per_px":.1592,"unit":"um/native_px","approved":True,
          "scope":{"batch_ids":["TEST_BATCH"],"wafer_ids":["W1"],"condition_ids":["FIXTURE-CONDITION-20X"],
            "image_ids":["TEST_IMAGE_1","TEST_IMAGE_2"]},
          "evidence":{"source":"user_confirmation","same_capture_setup":True,"description":"Synthetic unit fixture, not a physical calibration"}}
        self.path=self.f.root/"profiles.json";self.save_profile()
        patcher=patch("inspection.calibration_profiles.PROFILES_PATH",self.path);patcher.start();self.addCleanup(patcher.stop)
    def save_profile(self):
        self.path.write_text(json.dumps({"schema_version":1,"profiles":[self.profile]}),encoding="utf-8")
    def bind(self,**kwargs):
        with closing(connect_database(self.f.db)) as c:
            return bind_batch_condition(c,"TEST_BATCH",self.profile,approved=True,**kwargs)
    def sql(self,text,args=()):
        with closing(connect_database(self.f.db)) as c:c.execute(text,args);c.commit()
    def plan(self,batch="TEST_BATCH",ids=None):
        return self.p.prepare_analysis_inputs(batch,image_ids=ids)
    def test_01_shared_condition_ready_and_explicit_calibration(self):
        result=self.bind();plan=self.plan()
        self.assertEqual((result["updated_images"],result["condition_created"]),(2,True))
        self.assertEqual(plan["counts"],{"READY":2,"REVIEW_REQUIRED":0,"BLOCKED":0})
        self.assertEqual(len(plan["ready_requests"]),2);self.assertFalse(plan["execution_enabled"])
        for item in plan["items"]:
            self.assertEqual(item["candidate_request"]["calibration"],
                {"calibration_id":"optiview-native-0.1592","um_per_px":.1592,"magnification":20})
            self.assertEqual(item["calibration_evidence"]["scope_match"],"EXACT_BATCH_WAFER_CONDITION_IMAGE")
            self.assertEqual(item["issues"],[])
        with closing(connect_database(self.f.db,readonly=True)) as c:
            self.assertEqual(c.execute("SELECT COUNT(DISTINCT condition_id) FROM images WHERE batch_id='TEST_BATCH'").fetchone()[0],1)
    def test_02_binding_idempotent(self):
        self.bind();before=self.f.db.read_bytes();result=self.bind()
        self.assertEqual(result["updated_images"],0);self.assertFalse(result["condition_created"])
        self.assertEqual(self.f.db.read_bytes(),before)
    def test_03_other_batch_same_20x_never_auto_ready(self):
        self.bind();plan=self.plan("TEST_BATCH_2",["TEST_DUPLICATE"])
        self.assertEqual(plan["counts"]["READY"],0)
        self.assertIsNone(plan["items"][0]["candidate_request"]["calibration"])
        self.assertEqual(plan["items"][0]["candidate_request"]["magnification"],20)
    def test_04_unapproved_extra_image_in_same_batch_not_ready(self):
        self.bind()
        with closing(connect_database(self.f.db)) as c:
            cur=c.execute("SELECT * FROM images WHERE image_id='TEST_IMAGE_1'")
            names=[d[0] for d in cur.description];row=dict(zip(names,cur.fetchone()));row["image_id"]="EXTRA-NOT-APPROVED"
            c.execute("INSERT INTO images("+",".join(names)+") VALUES ("+",".join("?" for _ in names)+")",[row[k] for k in names]);c.commit()
        self.assertEqual(self.plan(ids=["EXTRA-NOT-APPROVED"])["counts"]["READY"],0)
    def test_05_missing_condition_is_blocked(self):
        self.assertEqual(self.plan()["counts"]["READY"],0)
        self.assertTrue(all(i["validation_status"]=="BLOCKED" for i in self.plan()["items"]))
    def test_06_magnification_mismatch_is_blocked(self):
        self.bind();self.sql("UPDATE acquisition_conditions SET magnification=40 WHERE condition_id='FIXTURE-CONDITION-20X'")
        self.assertEqual(self.plan()["counts"],{"READY":0,"REVIEW_REQUIRED":0,"BLOCKED":2})
    def test_07_unapproved_profile_requires_review(self):
        self.bind();self.profile["approved"]=False;self.save_profile()
        self.assertEqual(self.plan()["counts"]["REVIEW_REQUIRED"],2)
    def test_08_missing_settings_evidence_requires_review(self):
        self.bind();self.sql("UPDATE acquisition_conditions SET notes='{}' WHERE condition_id='FIXTURE-CONDITION-20X'")
        self.assertEqual(self.plan()["counts"]["REVIEW_REQUIRED"],2)
    def test_09_unit_and_coefficient_fail_closed(self):
        for field,value in (("unit","px/um"),("um_per_px",0),("um_per_px",float("inf"))):
            original=deepcopy(self.profile);self.profile[field]=value;self.save_profile()
            with self.assertRaises(ValueError):load_profiles(self.path)
            self.profile=original
    def test_10_approval_scope_and_stale_snapshot_rejected(self):
        before=self.f.db.read_bytes()
        with closing(connect_database(self.f.db)) as c:
            with self.assertRaises(ValueError):bind_batch_condition(c,"TEST_BATCH",self.profile,approved=False)
            with self.assertRaises(ValueError):bind_batch_condition(c,"TEST_BATCH_2",self.profile,approved=True)
            with self.assertRaises(ValueError):bind_batch_condition(c,"TEST_BATCH",self.profile,approved=True,expected_data_sha256="wrong")
        self.assertEqual(self.f.db.read_bytes(),before)
    def test_11_condition_collision_is_rejected(self):
        self.sql("INSERT INTO acquisition_conditions(condition_id,magnification) VALUES ('FIXTURE-CONDITION-20X',40)")
        before=self.f.db.read_bytes()
        with self.assertRaises(ValueError):self.bind()
        self.assertEqual(self.f.db.read_bytes(),before)
    def test_12_rollback_after_metadata_update(self):
        import database.capture_conditions as module
        original=module.validate_v4;calls=[]
        def fail(c):
            original(c);calls.append(1)
            if len(calls)==2:raise RuntimeError("injected post-update check failure")
        before=self.f.db.read_bytes()
        with patch.object(module,"validate_v4",side_effect=fail):
            with self.assertRaises(RuntimeError):self.bind()
        self.assertEqual(self.f.db.read_bytes(),before)
        self.assertEqual(self.plan()["counts"]["READY"],0)
    def test_13_existing_unknown_condition_not_overwritten(self):
        self.sql("UPDATE images SET condition_id='TEST_CONDITION' WHERE image_id='TEST_IMAGE_1'")
        before=self.f.db.read_bytes()
        with self.assertRaises(ValueError):self.bind()
        self.assertEqual(self.f.db.read_bytes(),before)
    def test_14_approved_image_membership_change_rejected(self):
        self.profile["scope"]["image_ids"]=["TEST_IMAGE_1"];self.save_profile()
        before=self.f.db.read_bytes()
        with self.assertRaises(ValueError):self.bind()
        self.assertEqual(self.f.db.read_bytes(),before)

class ReadinessUITests(unittest.TestCase):
    setUp=MatchingTests.setUp
    save_profile=MatchingTests.save_profile
    bind=MatchingTests.bind
    def test_batch_wafer_image_scope_ready_preview_no_execution(self):
        from streamlit.testing.v1 import AppTest
        self.bind()
        with patch.dict(os.environ,{"IDA_DB_PATH":str(self.f.db),"IDA_STORAGE_ROOT":str(self.f.sources),
             "IDA_STAGING_ROOT":str(self.f.root/"staging")}),patch("streamlit.file_uploader",return_value=[]):
            at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run()
            at.switch_page("pages/data_control.py").run()
            at.session_state["data_control_selected_source"]="SQLite";at.run()
            at.selectbox(key="dcc_analysis_target_batch").select("TEST_BATCH").run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["dcc_analysis_input_plan"]["counts"]["READY"],2)
            # TASK 18-B-2: a Batch with <=3 READY images is runnable (never clicked here).
            self.assertEqual(at.button(key="dcc_analysis_run").disabled,
                             __import__("dashboard.components.analysis_management",fromlist=["x"]).analysis_python() is None)
            at.radio(key="dcc_analysis_scope").set_value("Wafer").run()
            at.selectbox(key="dcc_analysis_target_wafer").select("W1").run()
            self.assertEqual(at.session_state["dcc_analysis_input_plan"]["counts"],
                {"READY":2,"REVIEW_REQUIRED":1,"BLOCKED":0})
            self.assertEqual(len(at.session_state["dcc_analysis_input_plan"]["batch_plans"]),2)
            at.radio(key="dcc_analysis_scope").set_value("Image").run()
            at.selectbox(key="dcc_analysis_target_image").select("TEST_IMAGE_1").run()
            self.assertEqual(at.session_state["dcc_analysis_input_plan"]["counts"]["READY"],1)
            # TASK 18-A-2: a single READY Image enables the run button (never clicked here).
            self.assertEqual(at.button(key="dcc_analysis_run").disabled,
                             __import__("dashboard.components.analysis_management",fromlist=["x"]).analysis_python() is None)
            at.session_state["data_control_selected_source"]="Mock";at.run()
            self.assertIsNone(at.session_state["dcc_analysis_input_plan"])
            self.assertFalse(at.exception)

if __name__=="__main__":unittest.main()
