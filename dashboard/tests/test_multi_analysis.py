"""TASK 18-B-2: multi Image / Batch / Wafer manual runs (fake engines, temporary v4 DB).

The real BatchRunner/AnalysisRunner/MetrologyAdapter/judge and AnalysisRepository run; only
the model engines and overlay drawing are fakes (dashboard/tests/fake_dashboard_runner.py).
"""
from contextlib import closing
from pathlib import Path
import json,os,sqlite3,sys,unittest
from unittest.mock import patch

from database.analysis_repository import AnalysisRepository
from dashboard.tests.test_upload_default_matching import Base,HAS_PLOTLY
import dashboard.components.analysis_management as am
from inspection.run_dashboard_analysis import validate

FAKE=Path(__file__).with_name("fake_dashboard_runner.py")
TABLES=("analysis_batch_runs","analysis_image_runs","defect_detections","metrology_measurements","line_judgments")


class CliValidationTests(unittest.TestCase):
    def payload(self,tmp,n=2,**change):
        from database.tests.test_image_registry import png_bytes
        import hashlib,tempfile
        root=Path(tempfile.mkdtemp());groups=[];expected={}
        for i in range(n):
            p=root/f"w1_1{i}_c.png";p.write_bytes(png_bytes(pixel=10+i))
            expected[f"I{i}"]=hashlib.sha256(p.read_bytes()).hexdigest()
            groups.append({"batch_id":f"B{i%2}","requests":[{"image_id":f"I{i}","batch_id":f"B{i%2}","image_path":str(p)}]})
        payload={"groups":groups,"expected_sha256":expected,"confirmed_count":None};payload.update(change)
        return payload
    def test_01_cli_guards(self):
        validate(self.payload(None,3))                                                     # <=3: no confirmation
        with self.assertRaises(ValueError):validate(self.payload(None,4))                 # >3 unconfirmed
        with self.assertRaises(ValueError):validate(self.payload(None,4,confirmed_count=3))  # wrong count
        with self.assertRaises(ValueError):validate(self.payload(None,4,confirmed_count=True))
        validate(self.payload(None,4,confirmed_count=4))                                   # exact confirmation
        p=self.payload(None,2);p["groups"][1]["requests"][0]["image_id"]="I0"
        with self.assertRaises(ValueError):validate(p)                                     # duplicate
        p=self.payload(None,2);p["groups"][0]["requests"][0]["batch_id"]="OTHER"
        with self.assertRaises(ValueError):validate(p)                                     # batch mixing
        p=self.payload(None,2);p["expected_sha256"]["I0"]="0"*64
        with self.assertRaises(ValueError):validate(p)                                     # stale source

    def test_11_progress_never_opens_batch_summaries(self):
        """Regression: an open reader made BatchRunner's os.replace fail on Windows (stale INTERRUPTED)."""
        import tempfile,builtins
        root=Path(tempfile.mkdtemp());run=root/"runs/batch-x";(run/"images/a/analysis").mkdir(parents=True)
        (run/"images/b").mkdir(parents=True)
        (run/"batch_summary.json").write_text("{}",encoding="utf-8")
        (run/"images/a/analysis/result.json").write_text(json.dumps({"status":"SUCCESS","pd_policy":{"state":"MODEL_PD_EXCLUDED"}}),encoding="utf-8")
        (run/"images/b/failure.json").write_text("{}",encoding="utf-8")
        opened=[];real_open=builtins.open;real_read=Path.read_text
        with patch("builtins.open",side_effect=lambda f,*a,**k:(opened.append(str(f)),real_open(f,*a,**k))[1]),             patch.object(Path,"read_text",lambda self,*a,**k:(opened.append(str(self)),real_read(self,*a,**k))[1]):
            progress=am.run_progress(root)
        self.assertEqual(progress,{"done":2,"outcomes":{"MODEL_PD_EXCLUDED":1,"FAILED":1}})
        self.assertFalse(any(name.endswith("batch_summary.json") for name in opened))


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed; Data Control page cannot load in AppTest")
class MultiScopeTests(Base):
    def setUp(self):
        super().setUp()
        # Batch A: normal + PD (Die 13); Batch B: engine failure (Die 14). All READY (20x default).
        _,a=self.register(self.stage([self.png("w1_21_c.png",201),self.png("w1_13_c.png",202)]),"wafer_batch")
        _,b=self.register(self.stage([self.png("w1_14_c.png",203)]),"wafer_batch")
        self.a,self.b=a,b
        self.out=self.f.root/"results";self.out.mkdir()
        self.calls=self.f.root/"calls.txt"
        for p in (patch.dict(os.environ,{"IDA_DB_PATH":str(self.f.db),"IDA_STORAGE_ROOT":str(self.f.sources),
                      "IDA_STAGING_ROOT":str(self.f.root/"staging"),"IDA_ANALYSIS_PYTHON":sys.executable,
                      "IDA_ANALYSIS_OUTPUT_ROOT":str(self.out),"FAKE_RUNNER_CALLS":str(self.calls)}),
                  patch.object(am,"_analysis_command",lambda python,request,output:
                      [str(python),str(FAKE),"--request",str(request),"--output-root",str(output)]),
                  patch("streamlit.file_uploader",return_value=[])):
            p.start();self.addCleanup(p.stop)
        os.environ.pop("FAKE_RUNNER_MODE",None);am._ACTIVE.clear()
        self.base=self.rows()
    def rows(self):
        with closing(sqlite3.connect(self.f.db)) as c:
            return {t:c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
    def launches(self):
        return len(self.calls.read_text()) if self.calls.exists() else 0
    def app(self,scope,target=None,extras=()):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(am.ROOT/"dashboard/app.py",default_timeout=60).run()
        at.switch_page("pages/data_control.py").run()
        at.session_state["data_control_selected_source"]="SQLite";at.run()
        at.radio(key="dcc_analysis_scope").set_value(scope).run()
        if target:at.selectbox(key="dcc_analysis_target_"+scope.lower()).select(target).run()
        if extras:at.multiselect(key="dcc_analysis_extra_images").set_value(list(extras)).run()
        self.assertFalse(at.exception)
        return at
    def captions(self,at):return " | ".join(c.value for c in at.caption)
    def repo(self):
        return AnalysisRepository(self.f.db,result_root=self.out,image_root=self.f.sources)
    def assert_db_matches_files(self,last):
        """Every stored image row equals its own Runner JSON; Batch/Image never mixed."""
        repository=self.repo()
        for group in last["groups"]:
            if (last["db_save"].get("batches") or [last["db_save"]])[0] is None:continue
            for image in group["images"]:
                history=repository.image_history(image["image_id"])
                if not history:continue
                stored=repository.get_analysis_run(history[0]["analysis_run_id"])
                self.assertEqual(stored["raw_result"],json.loads(Path(image["result_json"]).read_text(encoding="utf-8")))
                self.assertEqual(repository.get_batch_run(stored["batch_run_id"])["batch_id"],group["batch_id"])
                self.assertEqual(stored["is_adopted"],0)
        with closing(sqlite3.connect(self.f.db)) as c:
            self.assertEqual(c.execute("PRAGMA integrity_check").fetchone()[0],"ok")
            self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(),[])

    def test_02_multi_image_across_two_batches(self):
        normal,pd_image=self.a["image_ids"];failing=self.b["image_ids"][0]
        at=self.app("Image",normal,[pd_image,normal])  # Duplicate of the primary is ignored.
        preview=at.session_state["dcc_analysis_preview"]
        self.assertEqual(preview["image_ids"],[normal,pd_image])
        self.assertIn("실행 대상 READY 2장 · 실행 Batch 1개 · 제외 0장",self.captions(at))
        at=self.app("Image",normal,[failing])
        self.assertIn("실행 대상 READY 2장 · 실행 Batch 2개",self.captions(at))
        at.button(key="dcc_analysis_run").click().run();self.assertFalse(at.exception)
        last=at.session_state["dcc_analysis_last_run"]
        self.assertEqual(self.launches(),1)
        self.assertEqual([g["batch_id"] for g in last["groups"]],[self.a["batch_id"],self.b["batch_id"]])
        self.assertEqual({g["batch_id"]:g["execution_status"] for g in last["groups"]}[self.b["batch_id"]],"FAILED")
        # Batch A saved, Batch B (all failed) not saved -> PARTIAL; nothing of A removed.
        self.assertEqual(last["db_save"]["status"],"PARTIAL")
        self.assertEqual({e["batch_id"]:e["status"] for e in last["db_save"]["batches"]},
                         {self.a["batch_id"]:"SAVED",self.b["batch_id"]:"NOT_SAVED"})
        self.assertEqual(self.rows()["analysis_image_runs"],self.base["analysis_image_runs"]+1)
        self.assert_db_matches_files(last)
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)  # Same images, same session.

    def test_03_batch_scope_pd_excluded_and_per_image_separation(self):
        normal,pd_image=self.a["image_ids"]
        at=self.app("Batch",self.a["batch_id"])
        at.button(key="dcc_analysis_run").click().run();self.assertFalse(at.exception)
        last=at.session_state["dcc_analysis_last_run"]
        images={i["image_id"]:i for i in last["groups"][0]["images"]}
        self.assertEqual(images[pd_image]["outcome"],"MODEL_PD_EXCLUDED")
        self.assertEqual(images[pd_image]["metrology_status"],"SKIPPED")
        self.assertIsNone(images[pd_image]["line_mean_um"])
        self.assertEqual(images[normal]["metrology_status"],"SUCCESS")
        self.assertNotEqual(images[normal]["result_json"],images[pd_image]["result_json"])
        self.assertEqual(last["db_save"]["status"],"SAVED");self.assertTrue(last["db_save"]["verified"])
        delta={t:self.rows()[t]-self.base[t] for t in TABLES}
        self.assertEqual((delta["analysis_batch_runs"],delta["analysis_image_runs"],delta["defect_detections"]),(1,2,1))
        self.assertGreater(delta["metrology_measurements"],0)
        with closing(sqlite3.connect(self.f.db)) as c:  # PD image has no measurement rows.
            self.assertEqual(c.execute("SELECT COUNT(*) FROM metrology_measurements m JOIN analysis_image_runs a "
                "USING(analysis_run_id) WHERE a.image_id=?",(pd_image,)).fetchone()[0],0)
        self.assert_db_matches_files(last)
        again=am.save_analysis_result(last,am.runtime_provider())
        self.assertEqual(again["status"],"ALREADY_SAVED");self.assertEqual(self.rows()["analysis_image_runs"],self.base["analysis_image_runs"]+2)

    def test_04_wafer_scope_spans_batches_and_excludes_review(self):
        at=self.app("Wafer","W1")
        plan=at.session_state["dcc_analysis_input_plan"]
        self.assertEqual(plan["counts"],{"READY":3,"REVIEW_REQUIRED":3,"BLOCKED":0})
        self.assertIn("실행 대상 READY 3장 · 실행 Batch 2개 · 제외 3장",self.captions(at))
        at.button(key="dcc_analysis_run").click().run();self.assertFalse(at.exception)
        last=at.session_state["dcc_analysis_last_run"]
        ran={i["image_id"] for g in last["groups"] for i in g["images"]}
        self.assertEqual(ran,set(self.a["image_ids"])|set(self.b["image_ids"]))  # REVIEW images not run.
        request=json.loads(next(self.out.glob("dcc-*/request.json")).read_text(encoding="utf-8"))
        self.assertEqual([g["batch_id"] for g in request["groups"]],[g["batch_id"] for g in last["groups"]])
        self.assert_db_matches_files(last)

    def test_05_large_run_needs_selection_bound_confirmation(self):
        _,c=self.register(self.stage([self.png("w1_23_c.png",204)]),"wafer_batch")
        at=self.app("Wafer","W1")
        self.assertEqual(at.session_state["dcc_analysis_input_plan"]["counts"]["READY"],4)
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertEqual(at.button(key="dcc_analysis_run").help,am.CONFIRM_REASON)  # TASK18-B-11: short label.
        self.assertNotIn("대량 실행입니다",self.captions(at))
        box=next(b for b in at.checkbox if b.key.startswith("dcc_analysis_confirm_"))
        self.assertEqual(box.label,"READY 4장 · 대량 분석 실행 확인")
        self.assertIn("Batch 3개",box.help);self.assertIn("현재 SQLite DB에 저장",box.help)
        box.check().run()
        self.assertFalse(at.button(key="dcc_analysis_run").disabled)
        # A different selection gets a fresh, unchecked confirmation.
        self.register(self.stage([self.png("w1_24_c.png",205)]),"wafer_batch")
        at.run()
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertFalse(next(b for b in at.checkbox if b.key.startswith("dcc_analysis_confirm_")).value)
        next(b for b in at.checkbox if b.key.startswith("dcc_analysis_confirm_")).check().run()
        seen=[]
        original=am.run_progress
        with patch.object(am,"run_progress",side_effect=lambda d:(seen.append(original(d)),seen[-1])[1]),             patch.dict(os.environ,{"FAKE_RUNNER_DELAY":"2.5"}):
            at.button(key="dcc_analysis_run").click().run()
        self.assertFalse(at.exception)
        last=at.session_state["dcc_analysis_last_run"]
        request=json.loads(next(self.out.glob("dcc-*/request.json")).read_text(encoding="utf-8"))
        self.assertEqual(request["confirmed_count"],5)
        self.assertEqual(len([i for g in last["groups"] for i in g["images"]]),5)
        self.assertEqual(len(last["groups"]),4)  # One BatchRunner run per Batch, never merged.
        self.assertEqual({e["status"] for e in last["db_save"]["batches"]},{"SAVED","NOT_SAVED"})  # B fails.
        self.assertTrue(seen)  # Progress was polled from Runner checkpoints while running.
        final=am.run_progress(Path(last["run_dir"]))
        self.assertEqual(final["done"],5);self.assertEqual(sum(final["outcomes"].values()),5)
        self.assertEqual(final["outcomes"].get("FAILED"),1);self.assertEqual(final["outcomes"].get("MODEL_PD_EXCLUDED"),1)
        table=next(d.value for d in at.dataframe if "DB 저장" in d.value.columns)
        self.assertEqual(len(table),5)
        self.assert_db_matches_files(last)

    def test_06_review_only_and_auto_mode_blocked(self):
        at=self.app("Batch","TEST_BATCH_2")
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertIn("실행 가능한 READY 이미지가 없습니다",self.captions(at))
        at=self.app("Batch",self.a["batch_id"])
        at.radio(key="dcc_analysis_execution").set_value("자동 실행").run()
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertEqual(self.launches(),0)

    def test_07_scope_switch_does_not_mix_selections(self):
        normal,pd_image=self.a["image_ids"]
        at=self.app("Image",normal,[self.b["image_ids"][0]])
        at.radio(key="dcc_analysis_scope").set_value("Batch").run()
        at.selectbox(key="dcc_analysis_target_batch").select(self.a["batch_id"]).run()
        self.assertEqual(set(at.session_state["dcc_analysis_preview"]["image_ids"]),{normal,pd_image})

    def test_08_process_crash_not_saved(self):
        os.environ["FAKE_RUNNER_MODE"]="crash"
        at=self.app("Batch",self.a["batch_id"])
        at.button(key="dcc_analysis_run").click().run();self.assertFalse(at.exception)
        last=at.session_state["dcc_analysis_last_run"]
        self.assertEqual((last["execution_status"],last["db_save"]["status"]),("FAILED","NOT_SAVED"))
        self.assertEqual(self.rows(),self.base)
        self.assertFalse(at.button(key="dcc_analysis_run").disabled)  # A failure may be retried.

    def test_09_three_or_fewer_need_no_confirmation(self):
        at=self.app("Batch",self.a["batch_id"])
        self.assertFalse(any(b.key.startswith("dcc_analysis_confirm_") for b in at.checkbox))
        self.assertFalse(at.button(key="dcc_analysis_run").disabled)
        at.button(key="dcc_analysis_run").click().run()
        self.assertEqual(json.loads(next(self.out.glob("dcc-*/request.json")).read_text(encoding="utf-8"))["confirmed_count"],None)
        self.assertEqual(at.session_state["dcc_analysis_last_run"]["db_save"]["status"],"SAVED")

    def test_10_repeatability_and_lighting_grouped_and_excluded(self):
        inputs=self.stage([self.png("w1_12_rb_r1.png",210),self.png("w1_12_rb_r2.png",211)])
        for index,item in enumerate(inputs,1):item["repeat_index"]=index
        _,rep=self.register(inputs,"repeatability")
        light=self.stage([self.png("w1_22_lt.png",212),self.png("w1_23_lt.png",213)])
        for item,setting in zip(light,("10","30")):
            item["acquisition_condition"]={"condition_id":"COND-LIGHT-"+setting,"magnification":20.0,
                                           "illumination_setting":setting,"auto_exposure":0}
        _,lit=self.register(light,"lighting")
        at=self.app("Batch",rep["batch_id"])
        plan=at.session_state["dcc_analysis_input_plan"]
        self.assertEqual(plan["counts"]["READY"],0)  # Policy unchanged: never auto READY.
        self.assertTrue(all(any(x["code"]=="ANALYSIS_SCOPE_REVIEW_REQUIRED" for x in i["issues"]) for i in plan["items"]))
        self.assertIn("반복 촬영 · W1 / Die 12 / rb · 반복 1, 2 · 2장 · REVIEW_REQUIRED 2",self.captions(at))
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        at=self.app("Batch",lit["batch_id"])
        plan=at.session_state["dcc_analysis_input_plan"]
        self.assertEqual(plan["counts"]["READY"],0)
        self.assertTrue(all(i["candidate_request"]["calibration"] is None for i in plan["items"]))  # No forced 20x default.
        self.assertIn("조명 조건 · 10 · 1장",self.captions(at));self.assertIn("조명 조건 · 30 · 1장",self.captions(at))
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertEqual(self.launches(),0)


if __name__=="__main__":unittest.main()
