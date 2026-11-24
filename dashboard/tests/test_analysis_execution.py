"""TASK 18-A-2: gated single-image execution from Data Control Center.

No model is loaded here: the subprocess command is replaced by a fake process that
writes a Runner-shaped summary. Temporary v4 fixture DB only.
"""
from contextlib import closing
from pathlib import Path
import json,os,sys,textwrap,unittest
from unittest.mock import patch

from database.init_db import connect_database
from dashboard.tests.test_upload_default_matching import Base,HAS_PLOTLY,DEFAULT_CONDITION
import dashboard.components.analysis_management as am

FAKE=textwrap.dedent('''
import json,sys,pathlib
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
payload=json.loads(pathlib.Path(args["--request"]).read_text(encoding="utf-8"))
root=pathlib.Path(args["--output-root"]);calls=pathlib.Path(sys.argv[0]).with_suffix(".calls")
calls.write_text(calls.read_text()+"x" if calls.exists() else "x")
mode=pathlib.Path(sys.argv[0]).with_suffix(".mode").read_text().strip()
if mode=="crash":sys.exit(3)
request=payload["groups"][0]["requests"][0]
run=root/"batch-fake";item=run/"images/k/analysis";item.mkdir(parents=True)
result={"image_id":request["image_id"],"status":"SUCCESS","defect":{"detections":[]},
  "metrology":{"status":"SUCCESS","judgment":{"status":"judged"}},"official_analysis_complete":True,
  "measurement_summary":{"line":{"mean_um":10.1},"gap":{"mean_um":9.9}},"overlays":{"defect":None,"metrology":None}}
(item/"result.json").write_text(json.dumps(result),encoding="utf-8")
(run/"batch_summary.json").write_text(json.dumps({"batch_id":request["batch_id"],"execution_status":"COMPLETED","images":[{
  "image_id":request["image_id"],"outcome":"COMPLETE","engine_status":"SUCCESS",
  "pd_policy_state":"NO_PD","defect_status":"SUCCESS","metrology_status":"SUCCESS",
  "result_json":str(item/"result.json")}]}),encoding="utf-8")
''')


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed; Data Control page cannot load in AppTest")
class ExecutionTests(Base):
    def setUp(self):
        super().setUp()
        _,result=self.register(self.stage([self.png("w1_21_c.png",201)]),"wafer_batch")
        self.ready=result["image_ids"][0]
        self.script=self.f.root/"fake_runner.py";self.script.write_text(FAKE,encoding="utf-8")
        self.mode("ok")
        self.out=self.f.root/"runs"
        env=patch.dict(os.environ,{"IDA_DB_PATH":str(self.f.db),"IDA_STORAGE_ROOT":str(self.f.sources),
            "IDA_STAGING_ROOT":str(self.f.root/"staging"),"IDA_ANALYSIS_PYTHON":sys.executable,
            "IDA_ANALYSIS_OUTPUT_ROOT":str(self.out)})
        command=patch.object(am,"_analysis_command",lambda python,request,output:
                             [str(python),str(self.script),"--request",str(request),"--output-root",str(output)])
        uploader=patch("streamlit.file_uploader",return_value=[])
        for p in (env,command,uploader):p.start();self.addCleanup(p.stop)
        am._ACTIVE.clear()
    def mode(self,value):self.script.with_suffix(".mode").write_text(value)
    def calls(self):
        path=self.script.with_suffix(".calls");return len(path.read_text()) if path.exists() else 0
    def app(self,scope="Image",target=None):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(am.ROOT/"dashboard/app.py",default_timeout=60).run()
        at.switch_page("pages/data_control.py").run()
        at.session_state["data_control_selected_source"]="SQLite";at.run()
        at.radio(key="dcc_analysis_scope").set_value(scope).run()
        if target:at.selectbox(key="dcc_analysis_target_"+scope.lower()).select(target).run()
        self.assertFalse(at.exception)
        return at

    def test_01_label_and_ready_image_runs_exactly_once(self):
        at=self.app("Image",self.ready);button=at.button(key="dcc_analysis_run")
        self.assertEqual(button.label,"분석 실행");self.assertFalse(button.disabled)
        db=self.f.db.read_bytes()
        button.click().run();self.assertFalse(at.exception)
        last=at.session_state["dcc_analysis_last_run"]
        self.assertEqual((last["execution_status"],last["outcome"],last["image_id"]),("COMPLETED","COMPLETE",self.ready))
        self.assertEqual(self.calls(),1)
        request=json.loads(next(self.out.glob("dcc-*/request.json")).read_text(encoding="utf-8"))
        self.assertEqual([r["image_id"] for g in request["groups"] for r in g["requests"]],[self.ready])
        self.assertEqual(request["groups"][0]["requests"][0]["calibration"]["um_per_px"],.1592)
        # Duplicate: the same image cannot be re-run by repeated clicks in this session.
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)  # A browser cannot click it again.
        at.run();self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertEqual(self.calls(),1)
        self.assertTrue(any("결함 0건은 품질 PASS가 아닙니다" in c.value for c in at.caption))
        self.assertEqual(self.f.db.read_bytes(),db)

    def test_02_batch_and_wafer_follow_ready_cap_and_auto_stays_off(self):
        # TASK18-B-2: Batch/Wafer run only their READY images (cap 3); REVIEW images excluded.
        at=self.app("Wafer")
        self.assertEqual(at.session_state["dcc_analysis_input_plan"]["counts"]["READY"],1)
        self.assertFalse(at.button(key="dcc_analysis_run").disabled)
        self.assertTrue(any("실행 대상 READY 1장" in c.value for c in at.caption))
        at.radio(key="dcc_analysis_execution").set_value("자동 실행").run()
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertEqual(self.calls(),0)

    def test_03_review_required_and_blocked_never_run(self):
        at=self.app("Image","TEST_DUPLICATE")  # No approved calibration -> REVIEW_REQUIRED.
        self.assertEqual(at.session_state["dcc_analysis_input_plan"]["counts"]["REVIEW_REQUIRED"],1)
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        with closing(connect_database(self.f.db)) as c:
            c.execute("UPDATE acquisition_conditions SET magnification=40 WHERE condition_id=?",(DEFAULT_CONDITION,));c.commit()
        at=self.app("Image",self.ready)
        self.assertEqual(at.session_state["dcc_analysis_input_plan"]["counts"]["BLOCKED"],1)
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertEqual(self.calls(),0)

    def test_04_process_failure_is_reported_without_breaking_dashboard(self):
        self.mode("crash");at=self.app("Image",self.ready)
        at.button(key="dcc_analysis_run").click().run()
        self.assertFalse(at.exception)
        last=at.session_state["dcc_analysis_last_run"]
        self.assertEqual(last["execution_status"],"FAILED");self.assertEqual(last["returncode"],3)
        self.assertTrue(at.error)
        self.assertFalse(at.button(key="dcc_analysis_run").disabled)  # Failure is not a completed duplicate.

    def test_05_missing_engine_environment_and_active_job_disable(self):
        with patch.dict(os.environ,{"IDA_ANALYSIS_PYTHON":str(self.f.root/"missing.exe")}):
            self.assertTrue(self.app("Image",self.ready).button(key="dcc_analysis_run").disabled)
        class Running:
            def poll(self):return None
        am._ACTIVE["process"]=Running()
        try:
            at=self.app("Image",self.ready)
            self.assertTrue(at.button(key="dcc_analysis_run").disabled)
            with self.assertRaises(RuntimeError):am.start_analysis({"image_id":"x","batch_id":"b"},{"actual_sha256":"0"})
        finally:am._ACTIVE.clear()
        self.assertEqual(self.calls(),0)


class EligibilityUnitTests(unittest.TestCase):
    def plan(self,counts,ready=1):
        req={"image_id":"IMG","batch_id":"B"}
        items=[{"image_id":"IMG","validation_status":"READY","actual_sha256":"a","issues":[]}] if ready else               [{"image_id":"IMG","validation_status":"REVIEW_REQUIRED" if counts["REVIEW_REQUIRED"] else "BLOCKED",
                "actual_sha256":"a","issues":[{"message":"x"}]}]
        return {"counts":counts,"batch_plans":[{"batch_id":"B","ready_requests":[req]*ready}],"items":items}
    def test_06_gate(self):
        preview={"image_count":1,"image_ids":["IMG"]}
        ok={"READY":1,"REVIEW_REQUIRED":0,"BLOCKED":0}
        with patch.object(am,"analysis_python",return_value=Path(sys.executable)):
            self.assertIsNotNone(am.run_eligibility("수동 실행","Image",preview,self.plan(ok))[0])
            self.assertIsNotNone(am.run_eligibility("수동 실행","Batch",preview,self.plan(ok))[0])  # TASK18-B-2
            for args in (("자동 실행","Image",preview,self.plan(ok)),
                         ("수동 실행","Image",{"image_count":2,"image_ids":["IMG","X"]},self.plan(ok)),
                         ("수동 실행","Image",preview,None),
                         ("수동 실행","Image",preview,self.plan({"READY":0,"REVIEW_REQUIRED":1,"BLOCKED":0},0)),
                         ("수동 실행","Image",preview,self.plan({"READY":0,"REVIEW_REQUIRED":0,"BLOCKED":1},0)),
                         ("수동 실행","Image",{"image_count":1,"image_ids":["OTHER"]},self.plan(ok))):
                self.assertIsNone(am.run_eligibility(*args)[0],args[:2])
            self.assertIsNone(am.run_eligibility("수동 실행","Image",preview,self.plan(ok),completed=["IMG"])[0])


if __name__=="__main__":unittest.main()
