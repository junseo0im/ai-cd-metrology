"""TASK18A1: input mapping only; all mutations below are explicit isolated fixtures."""
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import hashlib,json,os,tempfile,unittest
from unittest.mock import Mock,patch
from dashboard.sqlite_provider import SQLiteProvider
from database.init_db import connect_database
from database.tests.test_analysis_repository import Fixture
from inspection.batch_runner import _prepared
from inspection.request_mapper import prepare_batch_inputs

ROOT=Path(__file__).resolve().parents[2]
TEST_ROOT=ROOT/"runtime_data/validation/task18a1/tmp"
class MapperTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=TEST_ROOT);self.addCleanup(temporary.cleanup)
        self.f=Fixture(Path(temporary.name));self.f.upgrade()
        self.p=SQLiteProvider(self.f.db,self.f.sources,self.f.root/"staging")
    def change(self,sql,args=()):
        with closing(connect_database(self.f.db)) as c:c.execute(sql,args);c.commit()
    def plan(self,ids=None,batch="TEST_BATCH"):
        return self.p.prepare_analysis_inputs(batch,image_ids=ids)
    def codes(self,item):return {i["code"] for i in item["issues"]}
    def test_native_ids_metadata_and_runner_contract(self):
        before=self.f.db.read_bytes()
        plan=self.plan(["TEST_IMAGE_1"])
        item=plan["items"][0];r=item["candidate_request"]
        self.assertEqual((r["batch_id"],r["image_id"],r["wafer_id"],r["die_id"],r["site_code"]),
            ("TEST_BATCH","TEST_IMAGE_1","W1","W1/D12","c"))
        self.assertEqual((r["physical_site_id"],r["magnification"],r["exposure_value"]),
            ("W1/D12/c",20.0,60.0))
        self.assertIsNone(r["exposure_unit"]);self.assertIsNone(r["peb_temperature_c"])
        self.assertIsNone(r["calibration"])
        self.assertNotIn("analysis_run_id",r)
        self.assertEqual(_prepared("TEST_BATCH",[r])[0]["validation_error"],None)
        self.assertEqual(item["validation_status"],"REVIEW_REQUIRED")
        self.assertEqual(plan["ready_requests"],[]);self.assertFalse(plan["execution_enabled"])
        self.assertEqual(item["actual_sha256"],item["registered_sha256"])
        json.dumps(plan,allow_nan=False)
        self.assertEqual(before,self.f.db.read_bytes())
    def test_20x_does_not_imply_calibration_approval(self):
        item=self.plan(["TEST_IMAGE_1"])["items"][0]
        self.assertIn("CALIBRATION_REVIEW_REQUIRED",self.codes(item))
        self.assertEqual(item["validation_status"],"REVIEW_REQUIRED")
        self.assertIsNone(item["candidate_request"]["calibration"])
    def test_missing_magnification_is_blocked(self):
        self.change("UPDATE images SET condition_id=NULL WHERE image_id='TEST_IMAGE_1'")
        item=self.plan(["TEST_IMAGE_1"])["items"][0]
        self.assertEqual(item["validation_status"],"BLOCKED")
        self.assertTrue(any(i["code"]=="MISSING_REQUIRED_METADATA" and i.get("field")=="magnification" for i in item["issues"]))
        self.assertIsNone(item["candidate_request"]["magnification"])
    def test_unsupported_magnification_not_coerced(self):
        self.change("UPDATE acquisition_conditions SET magnification=40")
        item=self.plan(["TEST_IMAGE_1"])["items"][0]
        self.assertEqual(item["validation_status"],"BLOCKED")
        self.assertIn("UNSUPPORTED_MAGNIFICATION",self.codes(item))
        self.assertEqual(item["candidate_request"]["magnification"],40.0)
    def test_missing_file_is_per_image_error(self):
        self.change("UPDATE images SET storage_path=? WHERE image_id='TEST_IMAGE_1'",
            (str(self.f.sources/"does_not_exist.png"),))
        plan=self.plan()
        by_id={i["image_id"]:i for i in plan["items"]}
        self.assertIn("IMAGE_NOT_FOUND",self.codes(by_id["TEST_IMAGE_1"]))
        self.assertEqual(by_id["TEST_IMAGE_2"]["validation_status"],"REVIEW_REQUIRED")
        self.assertEqual(plan["ready_requests"],[])
    def test_sha_mismatch_and_original_preserved(self):
        p=self.f.sources/"w1_12_c.png";before=p.read_bytes()
        self.change("UPDATE images SET sha256=? WHERE image_id='TEST_IMAGE_1'",("f"*64,))
        item=self.plan(["TEST_IMAGE_1"])["items"][0]
        self.assertIn("SHA_MISMATCH",self.codes(item))
        self.assertEqual(item["validation_status"],"BLOCKED")
        self.assertEqual(p.read_bytes(),before)
    def test_wrong_batch_unknown_image_and_qa_ids(self):
        item=self.plan(["TEST_DUPLICATE"])["items"][0]
        self.assertIn("BATCH_IMAGE_MISMATCH",self.codes(item))
        self.assertIsNone(item["candidate_request"])
        self.assertIn("IMAGE_NOT_REGISTERED",self.codes(self.plan(["not-registered"])["items"][0]))
        for batch,ids in (("QA:test",None),("RAW:test",None),("TEST_BATCH",["QA:image"]),
                           ("TEST_BATCH",["TEST_IMAGE_1","TEST_IMAGE_1"]),("not-registered",None),
                           ("' OR 1=1 --",None)):
            with self.subTest(batch=batch,ids=ids):
                with self.assertRaises(ValueError):self.plan(ids,batch=batch)
    def test_equal_sha_distinct_registered_ids_and_batches(self):
        a=self.plan(["TEST_IMAGE_1"])
        b=self.plan(["TEST_DUPLICATE"],batch="TEST_BATCH_2")
        self.assertEqual(a["items"][0]["actual_sha256"],b["items"][0]["actual_sha256"])
        self.assertNotEqual(a["items"][0]["image_id"],b["items"][0]["image_id"])
        self.assertEqual((len(a["items"]),len(b["items"])),(1,1))
    def test_repeatability_lighting_require_scope_review(self):
        for mode in ("repeatability","lighting"):
            self.change("UPDATE upload_batches SET upload_mode=? WHERE batch_id='TEST_BATCH'",(mode,))
            item=self.plan(["TEST_IMAGE_1"])["items"][0]
            self.assertIn("ANALYSIS_SCOPE_REVIEW_REQUIRED",self.codes(item))
            self.assertEqual(item["candidate_request"]["upload_mode"],mode)
            self.assertEqual(item["validation_status"],"REVIEW_REQUIRED")
    def test_path_outside_storage_and_remote_url_are_blocked(self):
        for value,code in ((str(self.f.root/"outside.png"),"PATH_OUTSIDE_STORAGE"),("https://invalid.example/image.png","INVALID_LOCAL_PATH")):
            self.change("UPDATE images SET storage_path=? WHERE image_id='TEST_IMAGE_1'",(value,))
            self.assertIn(code,self.codes(self.plan(["TEST_IMAGE_1"])["items"][0]))
    def test_read_only_contract_and_selected_hash_scope(self):
        with closing(connect_database(self.f.db)) as c:
            with self.assertRaisesRegex(ValueError,"query_only"):
                prepare_batch_inputs(c,"TEST_BATCH",storage_root=self.f.sources)
        with patch("inspection.request_mapper.hashlib.file_digest",wraps=hashlib.file_digest) as hashed:
            plan=self.plan(["TEST_IMAGE_1"])
        self.assertEqual(hashed.call_count,1)
        self.assertEqual(plan["files_hashed"],1)
        self.assertEqual(plan["selected_image_count"],1)
    def test_known_peb_and_auto_exposure_zero_preserved(self):
        self.change("UPDATE upload_batches SET peb_temperature_c=95 WHERE batch_id='TEST_BATCH'")
        self.change("UPDATE acquisition_conditions SET auto_exposure=0")
        r=self.plan(["TEST_IMAGE_1"])["items"][0]["candidate_request"]
        self.assertEqual(r["peb_temperature_c"],95.0)
        self.assertEqual(r["auto_exposure"],0)
        self.assertIsNone(r["illumination_setting"])
    def test_candidate_mock_handoff_retains_all_fields_without_dispatch(self):
        plan=self.plan(["TEST_IMAGE_1"])
        requests=[deepcopy(i["candidate_request"]) for i in plan["items"]]
        mock=Mock(name="InputOnlyBatchRunner")
        mock.run("TEST_BATCH",requests)
        self.assertEqual(mock.run.call_args.args,("TEST_BATCH",requests))
        self.assertEqual(_prepared("TEST_BATCH",requests)[0]["request"],requests[0])
        self.assertEqual(plan["ready_requests"],[])

class DashboardInputConnectionTests(unittest.TestCase):
    setUp=MapperTests.setUp
    def test_dcc_selected_batch_to_provider_mapper(self):
        from streamlit.testing.v1 import AppTest
        from dashboard.sqlite_provider import runtime_provider
        env={"IDA_DB_PATH":str(self.f.db),"IDA_STORAGE_ROOT":str(self.f.sources),
             "IDA_STAGING_ROOT":str(self.f.root/"staging")}
        with patch.dict(os.environ,env),patch("streamlit.file_uploader",return_value=[]):
            with patch("inspection.batch_runner.BatchRunner.run",side_effect=AssertionError("Real BatchRunner forbidden")),\
                 patch("inspection.analysis_runner.AnalysisRunner.run",side_effect=AssertionError("Real AnalysisRunner forbidden")):
                at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run()
                at.switch_page("pages/data_control.py").run()
                at.session_state["data_control_selected_source"]="SQLite";at.run()
                at.selectbox(key="dcc_analysis_target_batch").select("TEST_BATCH").run()
                self.assertFalse(at.exception)
                selected=at.session_state["dcc_analysis_preview"]
                self.assertEqual((selected["scope"],selected["target_id"]),("Batch","TEST_BATCH"))
                plan=runtime_provider().prepare_analysis_inputs(selected["target_id"],image_ids=selected["image_ids"])
                self.assertEqual({i["image_id"] for i in plan["items"]},set(selected["image_ids"]))
                for item in plan["items"]:
                    self.assertEqual(item["candidate_request"]["batch_id"],selected["target_id"])
                self.assertEqual(plan["ready_requests"],[])
                self.assertTrue(at.button(key="dcc_analysis_run").disabled)

if __name__=="__main__":
    suite=unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(MapperTests))
    suite.addTest(DashboardInputConnectionTests("test_dcc_selected_batch_to_provider_mapper"))
    unittest.TextTestRunner(verbosity=2).run(suite)
