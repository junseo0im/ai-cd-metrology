"""TASK 18-A-1 follow-up: approved 20x default condition for NEW uploads only.

All writes go to temporary v4 fixture DBs. The operational DB is only read
(W5 readiness regression) and its bytes are asserted unchanged.
"""
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,sqlite3,tempfile,unittest,uuid
from unittest.mock import patch

from database.init_db import DEFAULT_DB,connect_database
from database.capture_conditions import apply_upload_default
from database.tests.test_analysis_repository import Fixture
from database.tests.test_image_registry import png_bytes
from database.tests.test_integrated_registration import zip_bytes
from inspection.calibration_profiles import (load_profiles,load_upload_defaults,select_upload_default,
    upload_default_condition,validate_upload_default)
from dashboard.sqlite_provider import SQLiteProvider

ROOT=Path(__file__).resolve().parents[2]
TEST_ROOT=ROOT/"runtime_data/validation/task18a1/upload_default/tmp"
DEFAULT_CONDITION="FIXTURE-UPLOAD-DEFAULT-20X"
W5_BATCH="BATCH-7896834d126b4861bb9cf6fdee44b091"


def fixture_config():
    exact={"profile_id":"fixture-approved-20x","calibration_id":"optiview-native-0.1592",
      "magnification":20,"um_per_px":.1592,"unit":"um/native_px","approved":True,
      "scope":{"batch_ids":["TEST_BATCH"],"wafer_ids":["W1"],"condition_ids":["FIXTURE-CONDITION-20X"],
               "image_ids":["TEST_IMAGE_1","TEST_IMAGE_2"]},
      "evidence":{"source":"user_confirmation","same_capture_setup":True,"description":"Synthetic fixture"}}
    default={"profile_id":"fixture-upload-default-20x","source_profile_id":"fixture-approved-20x",
      "calibration_id":"optiview-native-0.1592","magnification":20,"um_per_px":.1592,"unit":"um/native_px",
      "approved":True,"condition_id":DEFAULT_CONDITION,
      "upload_modes":["wafer_batch","single_image","repeatability"],
      "evidence":{"source":"user_confirmation","same_capture_setup":True,"description":"Synthetic fixture"}}
    return {"schema_version":1,"profiles":[exact],"upload_defaults":[default]}


class Base(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=TEST_ROOT);self.addCleanup(temporary.cleanup)
        self.f=Fixture(Path(temporary.name));self.f.upgrade()
        self.p=SQLiteProvider(self.f.db,self.f.sources,self.f.root/"staging")
        self.config=fixture_config();self.path=self.f.root/"profiles.json";self.save()
        patcher=patch("inspection.calibration_profiles.PROFILES_PATH",self.path);patcher.start();self.addCleanup(patcher.stop)
    def save(self):
        self.path.write_text(json.dumps(self.config),encoding="utf-8")
    def snapshot(self,table):
        with closing(connect_database(self.f.db,readonly=True)) as c:
            return c.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
    def stage(self,files):
        bundle=self.p.stage(files);self.addCleanup(bundle["files"].cleanup);return bundle["inputs"]
    def metadata(self,mode,**extra):
        return {"batch_id":"BATCH-"+uuid.uuid4().hex,"wafer_id":"W1","upload_mode":mode,
                "dataset_role":"development","batch_status":"registered","process_run_id":"TEST_PROCESS",**extra}
    def register(self,inputs,mode,**extra):
        preview=self.p.preview(inputs,self.metadata(mode,**extra))
        plan=self.p.prepare_registration(preview,{})
        self.assertTrue(plan["registration_ready"],[i["issues"] for i in preview["images"]])
        return preview,self.p.register_selection(plan,approved=True)
    def conditions(self,image_ids):
        with closing(connect_database(self.f.db,readonly=True)) as c:
            return {r[0]:r[1] for r in c.execute("SELECT image_id,condition_id FROM images WHERE image_id IN (%s)"
                    %",".join("?"*len(image_ids)),image_ids)}
    def png(self,name,pixel):
        return {"filename":name,"content":png_bytes(pixel=pixel)}


class UploadPathTests(Base):
    def assert_auto_ready(self,result,expected):
        self.assertEqual(result["registered_count"],expected)
        self.assertEqual(set(self.conditions(result["image_ids"]).values()),{DEFAULT_CONDITION})
        plan=self.p.prepare_analysis_inputs(result["batch_id"])
        self.assertEqual(plan["counts"],{"READY":expected,"REVIEW_REQUIRED":0,"BLOCKED":0})
        self.assertFalse(plan["execution_enabled"])
        for item in plan["items"]:
            self.assertEqual(item["candidate_request"]["calibration"],
                {"calibration_id":"optiview-native-0.1592","um_per_px":.1592,"magnification":20})
            self.assertEqual(item["calibration_evidence"]["scope_match"],"APPROVED_UPLOAD_DEFAULT_CONDITION")
        return plan

    def test_01_zip_upload_auto_matched_ready(self):
        archive=zip_bytes([(f"wafer/w1_1{i}_c.png",png_bytes(pixel=20+i)) for i in range(1,4)])
        preview,result=self.register(self.stage([{"filename":"wafer.zip","content":archive}]),"wafer_batch")
        self.assertTrue(all(r["condition_id"]==DEFAULT_CONDITION for r in preview["images"]))
        self.assert_auto_ready(result,3)
        row=[r for r in self.snapshot("acquisition_conditions") if r[0]==DEFAULT_CONDITION]
        self.assertEqual(row,[tuple(upload_default_condition(load_upload_defaults()[0]).values())])

    def test_02_single_image_mode_auto_matched_ready(self):
        _,result=self.register(self.stage([self.png("w1_21_c.png",91)]),"single_image")
        self.assert_auto_ready(result,1)

    def test_03_single_png_in_wafer_batch_auto_matched_ready(self):
        _,result=self.register(self.stage([self.png("w1_22_lt.png",92)]),"wafer_batch")
        self.assert_auto_ready(result,1)

    def test_04_multiple_png_auto_matched_and_shared_condition_reused(self):
        _,first=self.register(self.stage([self.png(f"w1_3{i}_rb.png",100+i) for i in range(1,4)]),"wafer_batch")
        self.assert_auto_ready(first,3)
        _,second=self.register(self.stage([self.png("w1_35_c.png",120),self.png("w1_36_c.png",121)]),"wafer_batch")
        self.assert_auto_ready(second,2)
        self.assertEqual(sum(r[0]==DEFAULT_CONDITION for r in self.snapshot("acquisition_conditions")),1)

    def test_05_repeatability_matched_but_scope_review_required(self):
        inputs=self.stage([self.png("w1_11_c_r1.png",130),self.png("w1_11_c_r2.png",131)])
        for index,item in enumerate(inputs,1):item["repeat_index"]=index
        _,result=self.register(inputs,"repeatability")
        self.assertEqual(set(self.conditions(result["image_ids"]).values()),{DEFAULT_CONDITION})
        plan=self.p.prepare_analysis_inputs(result["batch_id"])
        self.assertEqual(plan["counts"],{"READY":0,"REVIEW_REQUIRED":2,"BLOCKED":0})
        for item in plan["items"]:
            self.assertEqual([i["code"] for i in item["issues"]],["ANALYSIS_SCOPE_REVIEW_REQUIRED"])
            self.assertIsNotNone(item["candidate_request"]["calibration"])


class ConflictTests(Base):
    def test_06_lighting_explicit_other_magnification_not_overridden(self):
        inputs=self.stage([self.png("w1_11_c_lg.png",140)])
        explicit={"condition_id":"COND-LIGHT-40X","magnification":40.0,"illumination_setting":"lg","auto_exposure":0}
        inputs[0]["acquisition_condition"]=explicit
        _,result=self.register(inputs,"lighting")
        self.assertEqual(set(self.conditions(result["image_ids"]).values()),{"COND-LIGHT-40X"})
        self.assertFalse(any(r[0]==DEFAULT_CONDITION for r in self.snapshot("acquisition_conditions")))
        plan=self.p.prepare_analysis_inputs(result["batch_id"])
        self.assertEqual(plan["counts"]["BLOCKED"],1)
        self.assertIn("UNSUPPORTED_MAGNIFICATION",[i["code"] for i in plan["items"][0]["issues"]])

    def test_07_explicit_condition_in_wafer_batch_not_overridden(self):
        inputs=self.stage([self.png("w1_11_c.png",141)])
        inputs[0]["acquisition_condition"]={"condition_id":"COND-EXPLICIT-50X","magnification":50.0}
        _,result=self.register(inputs,"wafer_batch")
        self.assertEqual(set(self.conditions(result["image_ids"]).values()),{"COND-EXPLICIT-50X"})
        self.assertEqual(self.p.prepare_analysis_inputs(result["batch_id"])["counts"]["READY"],0)

    def test_08_explicit_existing_condition_id_item_or_batch_not_overridden(self):
        inputs=self.stage([self.png("w1_13_c.png",142)]);inputs[0]["condition_id"]="TEST_CONDITION"
        _,result=self.register(inputs,"wafer_batch")
        self.assertEqual(set(self.conditions(result["image_ids"]).values()),{"TEST_CONDITION"})
        _,result=self.register(self.stage([self.png("w1_14_c.png",143)]),"wafer_batch",condition_id="TEST_CONDITION")
        self.assertEqual(set(self.conditions(result["image_ids"]).values()),{"TEST_CONDITION"})
        plan=self.p.prepare_analysis_inputs(result["batch_id"])
        self.assertEqual(plan["counts"],{"READY":0,"REVIEW_REQUIRED":1,"BLOCKED":0})

    def test_09_conflicting_stored_default_row_blocks_registration(self):
        with closing(connect_database(self.f.db)) as c:
            c.execute("INSERT INTO acquisition_conditions(condition_id,magnification) VALUES (?,40)",(DEFAULT_CONDITION,))
            c.commit()
        before=self.f.db.read_bytes()
        preview=self.p.preview(self.stage([self.png("w1_15_c.png",144)]),self.metadata("wafer_batch"))
        self.assertFalse(preview["registration_ready"])
        self.assertIn("Acquisition condition metadata conflict",[i["message"] for i in preview["images"][0]["issues"]])
        self.assertEqual(self.f.db.read_bytes(),before)

    def test_10_tampered_default_note_or_magnification_is_not_ready(self):
        _,result=self.register(self.stage([self.png("w1_16_c.png",145)]),"wafer_batch")
        for sql,status in (("UPDATE acquisition_conditions SET notes='{}' WHERE condition_id=?","REVIEW_REQUIRED"),
                           ("UPDATE acquisition_conditions SET magnification=40 WHERE condition_id=?","BLOCKED")):
            with closing(connect_database(self.f.db)) as c:c.execute(sql,(DEFAULT_CONDITION,));c.commit()
            self.assertEqual(self.p.prepare_analysis_inputs(result["batch_id"])["counts"][status],1)

    def test_11_unapproved_or_ambiguous_default_not_applied(self):
        inputs=self.stage([self.png("w1_17_c.png",146)])
        self.config["upload_defaults"][0]["approved"]=False;self.save()
        preview=self.p.preview(inputs,self.metadata("wafer_batch"))
        self.assertIsNone(preview["images"][0]["condition_id"])
        self.config=fixture_config();second=deepcopy(self.config["upload_defaults"][0])
        second.update(profile_id="fixture-upload-default-20x-b",condition_id="FIXTURE-UPLOAD-DEFAULT-20X-B")
        self.config["upload_defaults"].append(second);self.save()
        self.assertIsNone(self.p.preview(inputs,self.metadata("wafer_batch"))["images"][0]["condition_id"])

    def test_12_pre_v4_schema_not_applied(self):
        with closing(connect_database(self.f.db,readonly=True)) as c:
            defaults=load_upload_defaults()
            item={"image_path":"x.png","image_id":"IMG-1"}
            self.assertIn("acquisition_condition",apply_upload_default(c,[item],self.metadata("wafer_batch"),defaults=defaults)[0])
        class V3:
            def execute(self,_):return SimpleNamespace(fetchone=lambda:(3,))
        self.assertNotIn("acquisition_condition",apply_upload_default(V3(),[item],self.metadata("wafer_batch"),defaults=defaults)[0])
        self.assertNotIn("acquisition_condition",item)  # Caller's input is never mutated.

    def test_13_existing_images_and_exact_profile_unchanged(self):
        with closing(connect_database(self.f.db)) as c:
            c.execute("INSERT INTO acquisition_conditions VALUES ('FIXTURE-CONDITION-20X',20,NULL,NULL,NULL,?)",
                (json.dumps({"schema_version":1,"calibration_profile_id":"fixture-approved-20x","batch_id":"TEST_BATCH",
                  "approval_source":"user_confirmation","same_capture_setup_confirmed":True}),))
            c.execute("UPDATE images SET condition_id='FIXTURE-CONDITION-20X' WHERE batch_id='TEST_BATCH'");c.commit()
        before_images=self.snapshot("images");before_conditions=self.snapshot("acquisition_conditions")
        before_plans={b:self.p.prepare_analysis_inputs(b) for b in ("TEST_BATCH","TEST_BATCH_2")}
        files={p:p.read_bytes() for p in self.f.sources.glob("*.png")}
        _,result=self.register(self.stage([self.png("w1_18_c.png",147)]),"wafer_batch")
        after_images=self.snapshot("images")
        self.assertEqual([r for r in after_images if r[0] not in result["image_ids"]],before_images)
        self.assertEqual([r for r in self.snapshot("acquisition_conditions") if r[0]!=DEFAULT_CONDITION],before_conditions)
        for batch,plan in before_plans.items():self.assertEqual(self.p.prepare_analysis_inputs(batch),plan)
        self.assertEqual(before_plans["TEST_BATCH"]["counts"]["READY"],2)
        self.assertEqual(before_plans["TEST_BATCH_2"]["counts"]["READY"],0)
        self.assertEqual({p:p.read_bytes() for p in files},files)


class ConfigTests(unittest.TestCase):
    def test_14_default_must_reuse_approved_source_calibration(self):
        config=fixture_config();profiles=config["profiles"]
        for field,value in (("um_per_px",.2),("magnification",40),("calibration_id","other"),
                            ("source_profile_id","missing"),("upload_modes",["lighting"]),
                            ("upload_modes",[]),("condition_id","FIXTURE-CONDITION-20X"),("unit","px/um")):
            default=deepcopy(config["upload_defaults"][0]);default[field]=value
            with self.assertRaises(ValueError,msg=field):validate_upload_default(default,profiles)
        unapproved=deepcopy(profiles);unapproved[0]["approved"]=False
        with self.assertRaises(ValueError):validate_upload_default(config["upload_defaults"][0],unapproved)
        self.assertIsNone(select_upload_default(config["upload_defaults"],"lighting"))

    def test_15_operational_config_reuses_w5_profile_without_new_calibration(self):
        profiles=load_profiles();defaults=load_upload_defaults()
        self.assertEqual(len(defaults),1);d=defaults[0]
        source=next(p for p in profiles if p["profile_id"]==d["source_profile_id"])
        self.assertEqual(source["profile_id"],"w5-batch02-optiview-20x-v1")
        self.assertEqual((d["calibration_id"],d["magnification"],d["um_per_px"]),("optiview-native-0.1592",20,0.1592))
        self.assertEqual(sorted(d["upload_modes"]),["repeatability","single_image","wafer_batch"])
        self.assertNotIn(d["condition_id"],source["scope"]["condition_ids"])


@unittest.skipUnless(DEFAULT_DB.is_file(),"Operational DB not present")
class OperationalRegressionTests(unittest.TestCase):
    def test_16_w5_69_ready_read_only_and_db_bytes_unchanged(self):
        digest=hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest()
        provider=SQLiteProvider(DEFAULT_DB,ROOT/"runtime_data/raw",ROOT/"runtime_data/uploads_tmp")
        plan=provider.prepare_analysis_inputs(W5_BATCH)
        self.assertEqual(plan["counts"],{"READY":69,"REVIEW_REQUIRED":0,"BLOCKED":0})
        self.assertTrue(all(i["calibration_evidence"]["scope_match"]=="EXACT_BATCH_WAFER_CONDITION_IMAGE" for i in plan["items"]))
        with closing(sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro",uri=True)) as c:
            # The default applies to NEW uploads only; W5 Batch 02 was never retro-modified.
            self.assertEqual(c.execute("SELECT COUNT(*) FROM images WHERE batch_id=? AND condition_id=?",
                (W5_BATCH,load_upload_defaults()[0]["condition_id"])).fetchone()[0],0)
        self.assertEqual(hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest(),digest)


try:
    import plotly  # noqa: F401  Data Control page dependency
    HAS_PLOTLY=True
except ImportError:
    HAS_PLOTLY=False


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed; Data Control page cannot load in AppTest")
class DashboardUploadTests(Base):
    """Actual Data Control Center registration UI on a temporary v4 DB."""
    def app(self,files,mode=None):
        from streamlit.testing.v1 import AppTest
        uploads=[SimpleNamespace(name=n,getvalue=lambda c=c:c) for n,c in files]
        env=patch.dict(os.environ,{"IDA_DB_PATH":str(self.f.db),"IDA_STORAGE_ROOT":str(self.f.sources),
            "IDA_STAGING_ROOT":str(self.f.root/"staging")})
        uploader=patch("streamlit.file_uploader",side_effect=lambda *a,**kw:[] if kw.get("disabled") else
                       (uploads[0] if uploads and not kw.get("accept_multiple_files",True) else uploads))
        env.start();uploader.start();self.addCleanup(env.stop);self.addCleanup(uploader.stop)
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=60).run()
        at.switch_page("pages/data_control.py").run()
        at.session_state["data_control_selected_source"]="SQLite";at.run()
        if mode:at.radio(key="input_mode").set_value(mode).run()
        self.assertFalse(at.exception)
        self.assertFalse(at.button(key="sqlite_register").disabled,[c.value for c in at.caption][-5:])
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        result=at.session_state["sqlite_last_result"]
        self.assertEqual(set(self.conditions(result["image_ids"]).values()),{DEFAULT_CONDITION})
        return self.p.prepare_analysis_inputs(result["batch_id"])["counts"]

    def test_17_ui_zip(self):
        archive=zip_bytes([(f"w1_2{i}_c.png",png_bytes(pixel=150+i)) for i in range(3,6)])
        self.assertEqual(self.app([("wafer.zip",archive)])["READY"],3)

    def test_18_ui_single_image(self):
        self.assertEqual(self.app([("w1_26_c.png",png_bytes(pixel=160))],"Single Image")["READY"],1)

    def test_19_ui_multiple_png(self):
        files=[(f"w1_3{i}_lt.png",png_bytes(pixel=170+i)) for i in range(1,4)]
        self.assertEqual(self.app(files)["READY"],3)


if __name__=="__main__":unittest.main()
