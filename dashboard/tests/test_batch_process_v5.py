"""TASK 18-B-6: Batch-level exposure (schema v5) on temporary DBs + fixed-SQLite one-row Wafer Map filters.

Operational DB is never migrated or written. Analysis rows for the condition tests come from the
fake-engine runner (real BatchRunner/Repository), saved only into the temporary DB.
"""
from contextlib import closing
from pathlib import Path
import hashlib,json,os,sqlite3,subprocess,sys,unittest
from unittest.mock import patch

from database.init_db import DEFAULT_DB
from database.migrate_v5 import migrate_database_v5,validate_v5
from dashboard.tests.test_upload_default_matching import Base,HAS_PLOTLY

FAKE=Path(__file__).with_name("fake_dashboard_runner.py")


class V4BehaviourTests(Base):
    def test_01_v4_keeps_strict_wafer_process_rule(self):
        preview=self.p.preview(self.stage([self.png("w1_21_c.png",201)]),
            self.metadata("wafer_batch",process_run={"exposure_value":70.0,"exposure_unit":None}))
        self.assertFalse(preview["registration_ready"])
        self.assertIn("Existing process exposure_value mismatch",[i["message"] for i in preview["batch_issues"]])


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed")
class V5Tests(Base):
    def setUp(self):
        super().setUp()
        self.op_sha=hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest() if DEFAULT_DB.is_file() else None
        self.assertEqual(migrate_database_v5(self.f.db)["version"],5)
        self.out=self.f.root/"results";self.out.mkdir()
        env=patch.dict(os.environ,{"IDA_DB_PATH":str(self.f.db),"IDA_STORAGE_ROOT":str(self.f.sources),
            "IDA_STAGING_ROOT":str(self.f.root/"staging"),"IDA_ANALYSIS_OUTPUT_ROOT":str(self.out)})
        env.start();self.addCleanup(env.stop)
        uploader=patch("streamlit.file_uploader",return_value=[]);uploader.start();self.addCleanup(uploader.stop)
    def tearDown(self):
        if self.op_sha:self.assertEqual(hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest(),self.op_sha)
    def batch(self,files,exposure,*,name=None,peb=None,proc=None):
        extra={"process_run":{"exposure_value":exposure,"exposure_unit":None},"batch_process_run_id":proc or ("PROC-"+str(exposure))}
        if peb is not None:extra["peb_temperature_c"]=peb
        if name:extra["notes"]=json.dumps({"batch_name":name})
        _,result=self.register(self.stage(files),"wafer_batch",**extra)
        return result["batch_id"]
    def stored(self,sql,args=()):
        with closing(sqlite3.connect(self.f.db)) as c:return c.execute(sql,args).fetchall()

    def test_02_migration_inherits_and_two_exposures_on_one_wafer(self):
        with closing(sqlite3.connect(self.f.db)) as c:validate_v5(c)
        self.assertEqual(self.stored("SELECT COUNT(*) FROM upload_batches WHERE process_run_id IS NOT NULL")[0][0],0)
        a=self.batch([self.png("w1_21_c.png",201)],60.0)
        b=self.batch([self.png("w1_22_c.png",202)],70.0,name="노광60이라고 적은 이름")  # Name is never parsed.
        rows=dict(self.stored("SELECT batch_id,process_run_id FROM upload_batches WHERE batch_id IN (?,?)",(a,b)))
        self.assertEqual(rows[a],"TEST_PROCESS")                                   # Same value: Wafer record reused.
        self.assertEqual(rows[b],"PROC-70.0")                                      # Different value: Batch-only record.
        self.assertEqual(self.stored("SELECT process_run_id FROM wafers WHERE wafer_id='W1'")[0][0],"TEST_PROCESS")
        self.assertEqual(self.stored("SELECT exposure_value FROM process_runs WHERE process_run_id='TEST_PROCESS'")[0][0],60.0)
        frame=self.p.batches().set_index("batch_id")
        self.assertEqual((frame.loc[a,"exposure_value"],frame.loc[a,"exposure_source"]),(60.0,"batch"))
        self.assertEqual((frame.loc[b,"exposure_value"],frame.loc[b,"exposure_source"]),(70.0,"batch"))
        self.assertEqual((frame.loc["TEST_BATCH","exposure_value"],frame.loc["TEST_BATCH","exposure_source"]),(60.0,"wafer"))
        images=self.p.images(b,limit=10)
        self.assertEqual(set(images["exposure_value"]),{70.0})
        plan=self.p.prepare_analysis_inputs(b)
        self.assertEqual({i["candidate_request"]["exposure_value"] for i in plan["items"]},{70.0})

    def test_03_missing_exposure_is_not_defaulted_and_peb_is_per_batch(self):
        preview=self.p.preview(self.stage([self.png("w1_23_c.png",203)]),self.metadata("wafer_batch",batch_process_run_id="PROC-X"))
        self.assertFalse(preview["registration_ready"])
        self.assertTrue(any("노광 조건" in i["message"] for i in preview["batch_issues"]))
        c100=self.batch([self.png("w1_24_c.png",204)],60.0,peb=100.0)
        c110=self.batch([self.png("w1_25_c.png",205)],60.0,peb=110.0,proc="PROC-other")
        pebs=dict(self.stored("SELECT batch_id,peb_temperature_c FROM upload_batches WHERE batch_id IN (?,?)",(c100,c110)))
        self.assertEqual((pebs[c100],pebs[c110]),(100.0,110.0))

    def run_fake(self,batch_id):
        """Real BatchRunner + Repository with fake engines; saves into the temporary DB only."""
        import dashboard.components.analysis_management as am
        plan=self.p.prepare_analysis_inputs(batch_id)
        requests=plan["ready_requests"]
        run_dir=self.out/("dcc-"+batch_id[-8:]);(run_dir/"runs").mkdir(parents=True)
        payload={"groups":[{"batch_id":batch_id,"requests":requests}],
                 "expected_sha256":{i["image_id"]:i["actual_sha256"] for i in plan["items"] if i["validation_status"]=="READY"},
                 "confirmed_count":len(requests)}
        (run_dir/"request.json").write_text(json.dumps(payload),encoding="utf-8")
        subprocess.run([sys.executable,str(FAKE),"--request",str(run_dir/"request.json"),"--output-root",str(run_dir/"runs")],
                       check=True,env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[2])})
        result=am.collect_result(run_dir,0);result["requested"]={"images":{r["image_id"]:batch_id for r in requests}}
        saved=am.save_analysis_result(result,self.p)
        self.assertIn(saved["status"],("SAVED",),saved)
        return requests

    def test_04_conditions_never_mix_in_map_rates_and_graph(self):
        from dashboard.components.inspection_wafer import die_snapshot,apply_condition_scope,duplicate_exclusions
        from dashboard.components.analysis_management import collect_images
        from dashboard.die_assessment import summarize_wafer
        from dashboard.process_analysis import metadata_frame,condition_table
        from dashboard.components.process_analysis import stored_results
        a=self.batch([self.png("w1_21_c.png",201),self.png("w1_21_lt.png",206),self.png("w1_21_rb.png",207)],60.0)
        b=self.batch([self.png("w1_13_lt.png",208),self.png("w1_13_c.png",209),self.png("w1_13_rb.png",210)],70.0)  # Fake engine: Die 13 = model PD.
        self.run_fake(a);self.run_fake(b)
        images=collect_images(self.p,wafer_id="W1").to_dict("records");analysis=self.p.latest_analysis("W1")
        def assess(exposure):
            scoped=apply_condition_scope(images,exposure=exposure)
            dies=die_snapshot("W1",self.p.wafer_sites("W1"),scoped["images"],analysis,
                              exclude_from_judgment=duplicate_exclusions(scoped["images"],analysis),judge=not scoped["mixed"])
            return scoped,{d["die_code"]:d for d in dies}
        both,_=assess(None);self.assertTrue(both["mixed"])
        s60,d60=assess("60 s");s70,d70=assess("70 s")
        self.assertFalse(s60["mixed"] or s70["mixed"])
        self.assertEqual(d60["21"]["assessment"]["decision"],"PASS")          # 70 PD never leaks into 60.
        self.assertEqual(d60["13"]["assessment"]["decision"],"PENDING")
        self.assertEqual(d70["13"]["assessment"]["decision"],"FAIL")
        self.assertEqual(d70["21"]["assessment"]["decision"],"PENDING")
        self.assertEqual(summarize_wafer(list(d70.values()))["FAIL"],1)
        self.assertEqual(summarize_wafer(list(d60.values()))["FAIL"],0)
        table=condition_table(metadata_frame(collect_images(self.p)),"exposure",results=stored_results(self.p)).set_index("공정 조건")
        self.assertEqual(int(table.loc["70 s","분석 결과 이미지 수"]),3)
        self.assertEqual(int(table.loc["70 s","PD 제외 이미지 수"]),3)   # All three Die-13 images: PD.
        self.assertEqual(int(table.loc["70 s","유효 이미지 수"]),0)
        self.assertEqual(int(table.loc["60 s","유효 이미지 수"]),3)

    def test_05_wafer_map_one_row_sqlite_only_and_reset(self):
        from streamlit.testing.v1 import AppTest
        from dashboard.components.inspection_wafer import select_die
        a=self.batch([self.png("w1_21_c.png",201)],60.0);b=self.batch([self.png("w1_22_c.png",202)],70.0)
        at=AppTest.from_file(Path(__file__).resolve().parents[2]/"dashboard/app.py",default_timeout=90).run()
        at.switch_page("pages/inspection_analysis.py").run();self.assertFalse(at.exception)
        self.assertFalse(any(s.key in ("ia_wafer_source","detail_wafer_source") for s in at.selectbox))   # No Mock/SQLite UI.
        labels={s.key:s.label for s in at.selectbox if s.key and s.key.startswith("ia_wafer_")}
        self.assertEqual([labels.get(k) for k in ("ia_wafer_wafer_SQLite","ia_wafer_exposure_SQLite_W1","ia_wafer_peb_SQLite_W1",
                                                  "ia_wafer_batch_SQLite_W1")],["Wafer ID","노광","PEB","Batch"])
        exposure=at.selectbox(key="ia_wafer_exposure_SQLite_W1")
        self.assertEqual(exposure.options,["전체 (조건 혼합)","60 s","70 s"])
        self.assertTrue(any("조건을 선택하세요" in w.value for w in at.warning))      # Default = 전체 -> mixed.
        exposure.select("70 s").run()
        self.assertEqual([o for o in at.selectbox(key="ia_wafer_batch_SQLite_W1").options if o!="전체 Batch"],
                         [o for o in at.selectbox(key="ia_wafer_batch_SQLite_W1").options if b[6:14] in o])  # Only 70 Batch.
        select_die(at.session_state,"ia_wafer_map_SQLite_W1_all","22",popover=True);at.run()
        next(g for g in at.get("button_group") if g.key.startswith("ia_sqlite_position_")).set_value("c").run()
        self.assertIsNotNone(at.session_state["ia_sqlite_selected_image_id"])
        at.selectbox(key="ia_wafer_exposure_SQLite_W1").select("60 s").run()
        self.assertIsNone(at.session_state["ia_sqlite_selected_image_id"]);self.assertFalse(at.exception)

    def test_06_registration_form_does_not_preselect_wafer_exposure(self):
        from streamlit.testing.v1 import AppTest
        from types import SimpleNamespace
        from database.tests.test_image_registry import png_bytes
        content=png_bytes(pixel=211)
        with patch("streamlit.file_uploader",return_value=[SimpleNamespace(name="w1_26_c.png",getvalue=lambda:content)]):
            at=AppTest.from_file(Path(__file__).resolve().parents[2]/"dashboard/app.py",default_timeout=90).run()
            at.switch_page("pages/data_control.py").run()
            at.session_state["data_control_selected_source"]="SQLite";at.run();self.assertFalse(at.exception)
            exposure=next(s for s in at.number_input if s.label=="노광 조건")  # TASK18-B-10A.1: direct entry
            self.assertIsNone(exposure.value)                                         # No silent Wafer default.
            self.assertFalse(any("기존 Wafer 공정 기록" in c.value for c in at.caption))  # TASK18-B-10A.2: caption removed
            self.assertTrue(at.button(key="sqlite_register").disabled)
            exposure.set_value(70).run()
            self.assertFalse(at.button(key="sqlite_register").disabled)
            at.button(key="sqlite_register").click().run();self.assertFalse(at.exception)
            batch=at.session_state["sqlite_last_result"]["batch_id"]
        self.assertEqual(self.p.batch_record(batch)["exposure_value"],70.0)
        self.assertEqual(self.stored("SELECT exposure_value FROM process_runs WHERE process_run_id='TEST_PROCESS'")[0][0],60.0)


if __name__=="__main__":unittest.main()
