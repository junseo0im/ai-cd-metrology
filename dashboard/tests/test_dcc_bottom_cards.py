"""TASK 18-B-11: Data Control bottom cards — Wafer × 공정조건 analysis scope, A–E, progress across pages, KPIs.

Temporary v6 fixture DBs only (registration through the provider, analysis through the fake Runner
process). The operational DB is hashed before/after and never opened for writing.
"""
from contextlib import closing
from pathlib import Path
import json,os,sqlite3,time,unittest

from streamlit.testing.v1 import AppTest

from database.migrate_v6 import V6_IMAGE_PROCESS_SQL
import dashboard.tests.test_multi_wafer_v6 as mw  # Module import: its TestCase is not collected twice.
from dashboard.tests.test_multi_wafer_v6 import unique_png,APP
from dashboard.tests.test_upload_default_matching import HAS_PLOTLY
import dashboard.components.analysis_management as am

C=am.image_condition


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed; Data Control page cannot load in AppTest")
class BottomCardTests(mw.MultiWaferV6Tests):
    """Reuses the v6 fixture/helpers of MultiWaferV6Tests; its own tests are not re-run here."""
    def setUp(self):
        super().setUp()
        up=lambda *names:[{"filename":n,"content":unique_png()} for n in names]
        # W2 in four conditions (two share 노광 60, one PEB NULL); W3 only in one. Counts are arbitrary.
        _,self.b1=self.register_multi(up("w2_21_c.png","w2_22_c.png","w2_23_c.png","w2_24_c.png","w3_21_c.png","w3_22_c.png"),
                                      {"W2":(50,220.0),"W3":(60,None)})
        _,self.b2=self.register_multi(up("w2_25_c.png","w2_26_c.png"),{"W2":(60,220.0)})
        _,self.b3=self.register_multi(up("w2_27_c.png","w2_28_c.png"),{"W2":(60,120.0)})
        _,self.b4=self.register_multi(up("w2_31_c.png"),{"W2":(60,None)})

    def sql_ids(self,wafer,exposure,peb):
        rows=self.q(V6_IMAGE_PROCESS_SQL+"WHERE d.wafer_id=?",wafer)
        return {r[0] for r in rows if r[5]==exposure and r[7]==peb}

    def wafer_app(self,wafer="W2",condition=None):
        at=self.dcc()
        at.radio(key="dcc_analysis_scope").set_value("Wafer").run()
        at.selectbox(key="dcc_analysis_target_wafer").select(wafer).run()
        if condition:
            at.selectbox(key="dcc_analysis_condition_"+wafer).set_value(condition).run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        return at

    def captions(self,at):
        return " | ".join(c.value for c in at.caption)

    def markdown(self,at):
        return " | ".join(m.value for m in at.markdown)

    # 2/3/4/5: conditions come from SQLite per image; 노광 and PEB both separate; NULL kept apart.
    def test_01_wafer_conditions_listed_and_never_merged(self):
        preview=am.target_preview(self.p,"Wafer","W2")
        self.assertTrue(preview["condition_required"])
        found={c["id"]:c["image_count"] for c in preview["conditions"]}
        for exposure,peb in ((50.0,220.0),(60.0,220.0),(60.0,120.0),(60.0,None)):
            self.assertEqual(found[C(exposure,None,peb)],len(self.sql_ids("W2",exposure,peb)),(exposure,peb))
        self.assertEqual(sum(found.values()),preview["image_count"])
        labels=[c["label"] for c in preview["conditions"]]
        self.assertIn("노광 60 mJ/cm² · PEB 미입력",labels);self.assertIn("노광 60 mJ/cm² · PEB 120 s",labels)
        self.assertIn("노광 60 mJ/cm² · PEB 220 s",labels)
        self.assertNotIn("단위",json.dumps(labels,ensure_ascii=False))      # No unit is guessed.
        with self.assertRaises(ValueError):
            am.target_preview(self.p,"Wafer","W2",condition=C(99,None,1))   # Not a condition of this Wafer.
        one=am.target_preview(self.p,"Wafer","W3")                         # Single condition: no choice needed.
        self.assertFalse(one["condition_required"]);self.assertEqual(one["condition"],C(60.0,None,None))

    # 7/8: shown image set == analysis request set; missing condition never runs.
    def test_02_condition_scope_equals_request_set(self):
        preview=am.target_preview(self.p,"Wafer","W2")
        groups,_,reason=am.run_eligibility("수동 실행","Wafer",preview,None,confirmed=True)
        self.assertIsNone(groups);self.assertEqual(reason,am.CONDITION_REASON)
        with self.assertRaises(ValueError):
            am.prepare_selected_inputs(self.p,preview)
        for exposure,peb in ((60.0,220.0),(60.0,120.0),(60.0,None),(50.0,220.0)):
            chosen=am.target_preview(self.p,"Wafer","W2",condition=C(exposure,None,peb))
            self.assertEqual(set(chosen["image_ids"]),self.sql_ids("W2",exposure,peb))
            plan=am.prepare_selected_inputs(self.p,chosen)
            self.assertEqual({i["image_id"] for i in plan["items"]},set(chosen["image_ids"]))
            groups,expected,reason=am.run_eligibility("수동 실행","Wafer",chosen,plan,confirmed=True)
            sent={r["image_id"] for g in groups for r in g["requests"]}
            self.assertEqual(sent,set(chosen["image_ids"]));self.assertEqual(set(expected),sent)
            for g in groups:
                for r in g["requests"]:
                    self.assertEqual((r["wafer_id"],r["exposure_value"],r["peb_temperature_c"]),("W2",exposure,peb))

    # 1/6/7/10/11: UI selection, reset on Wafer change, READY count == request.json, confirmation reset.
    def test_03_ui_condition_selection_run_and_resets(self):
        at=self.wafer_app()
        self.assertIsNone(at.selectbox(key="dcc_analysis_condition_W2").value)
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertIn(am.CONDITION_REASON,self.captions(at))
        self.assertIsNone(at.session_state["dcc_analysis_input_plan"])         # Nothing prepared for the whole Wafer.
        # >3 READY condition: confirmation needed; switching the condition gives a fresh box.
        big=C(50.0,None,220.0)
        at.selectbox(key="dcc_analysis_condition_W2").set_value(big).run()
        box=next(b for b in at.checkbox if b.key.startswith("dcc_analysis_confirm_"))
        self.assertEqual(box.label,"READY 4장 · 대량 분석 실행 확인");self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        box.check().run();self.assertFalse(at.button(key="dcc_analysis_run").disabled)
        at.selectbox(key="dcc_analysis_condition_W2").set_value(C(60.0,None,220.0)).run()
        self.assertFalse(any(b.key.startswith("dcc_analysis_confirm_") for b in at.checkbox))   # 2 READY: none needed.
        at.selectbox(key="dcc_analysis_condition_W2").set_value(big).run()
        self.assertFalse(next(b for b in at.checkbox if b.key.startswith("dcc_analysis_confirm_")).value)
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        # Wafer change: invalid selection is cleared (W3 has a single condition -> preselected); back to W2 -> empty.
        at.selectbox(key="dcc_analysis_target_wafer").select("W3").run()
        self.assertEqual(at.selectbox(key="dcc_analysis_condition_W3").value,C(60.0,None,None))
        self.assertNotIn("dcc_analysis_condition_W2",at.session_state)          # Previous Wafer's choice dropped.
        at.selectbox(key="dcc_analysis_target_wafer").select("W2").run()
        self.assertIsNone(at.selectbox(key="dcc_analysis_condition_W2").value)
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        # Run one condition: request.json, Runner and DB hold exactly its images.
        chosen=C(60.0,None,220.0)
        at.selectbox(key="dcc_analysis_condition_W2").set_value(chosen).run()
        preview=at.session_state["dcc_analysis_preview"]
        self.assertEqual(set(preview["image_ids"]),self.sql_ids("W2",60.0,220.0))
        self.assertIn("실행 대상 READY 2장",self.captions(at))
        self.assertIn("READY · 실행</small><b>2</b>",self.markdown(at))
        at.button(key="dcc_analysis_run").click().run();self.assertFalse(at.exception)
        request=json.loads(next(self.out.glob("dcc-*/request.json")).read_text(encoding="utf-8"))
        sent={r["image_id"] for g in request["groups"] for r in g["requests"]}
        self.assertEqual(sent,set(preview["image_ids"]))
        last=at.session_state["dcc_analysis_last_run"]
        self.assertEqual({i["image_id"] for g in last["groups"] for i in g["images"]},sent)
        self.assertEqual({r[0] for r in self.q("SELECT image_id FROM analysis_image_runs")},sent)  # Nothing else analysed.
        # 15: E counts == stored outcomes.
        stored=[r[0] for r in self.q("SELECT outcome FROM analysis_image_runs")]
        counts=am.result_counts(last)
        self.assertEqual(counts["images"],len(stored));self.assertEqual(counts["complete"],stored.count("COMPLETE"))
        self.assertEqual(counts["failed"],stored.count("FAILED"))
        self.assertEqual(last["db_save"]["status"],"SAVED")
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)            # 14: same images, same session.
        # TASK18-B-11.1: short notice; the blocked ID set is unchanged and shown only in a collapsed list.
        self.assertEqual(at.button(key="dcc_analysis_run").help,"이미 분석된 이미지 2장 · 중복 실행 차단")
        self.assertIn("이미 분석된 이미지 2장 · 중복 실행 차단",self.captions(at))
        self.assertFalse(any(i in self.captions(at) for i in sent))                 # No ID list in C by default.
        details=next(e for e in at.expander if e.label=="중복 실행 대상 상세 보기 (2장)")
        self.assertEqual(set(details.code[0].value.split(chr(10))),sent)
        self.assertEqual(am.repeated_images(sent,at.session_state["dcc_analysis_completed_ids"]).__len__(),2)

    # 9 + Image scope: Batch keeps every Wafer of a multi-Wafer Batch; Image keeps its own Wafer/condition.
    def test_04_batch_and_image_scope_unchanged(self):
        preview=am.target_preview(self.p,"Batch",self.b1["batch_id"])
        self.assertEqual(preview["wafer_ids"],["W2","W3"]);self.assertEqual(preview["image_count"],6)
        self.assertFalse(preview["condition_required"])
        plan=am.prepare_selected_inputs(self.p,preview)
        groups,_,_=am.run_eligibility("수동 실행","Batch",preview,plan,confirmed=True)
        requests={r["image_id"]:r for g in groups for r in g["requests"]}
        self.assertEqual(set(requests),set(preview["image_ids"]))
        self.assertEqual({(r["wafer_id"],r["exposure_value"],r["peb_temperature_c"]) for r in requests.values()},
                         {("W2",50.0,220.0),("W3",60.0,None)})
        at=self.dcc()
        at.radio(key="dcc_analysis_scope").set_value("Batch").run()
        at.selectbox(key="dcc_analysis_target_batch").select(self.b1["batch_id"]).run()
        text=self.markdown(at)
        self.assertIn("W2 · 노광 50 mJ/cm² · PEB 220 s · 4장",text);self.assertIn("W3 · 노광 60 mJ/cm² · PEB 미입력 · 2장",text)
        image=self.b3["image_ids"][0]
        at.radio(key="dcc_analysis_scope").set_value("Image").run()
        at.selectbox(key="dcc_analysis_target_image").select(image).run()
        self.assertIn("W2 · 노광 60 mJ/cm² · PEB 120 s · 1장",self.markdown(at))
        self.assertEqual(at.session_state["dcc_analysis_preview"]["image_ids"],[image])

    # 12/13/14: progress on another page, run collected once, owner gets the result + duplicate guard.
    def test_05_progress_visible_on_other_page_and_collected_once(self):
        chosen=am.target_preview(self.p,"Wafer","W2",condition=C(60.0,None,220.0))
        plan=am.prepare_selected_inputs(self.p,chosen)
        groups,expected,_=am.run_eligibility("수동 실행","Wafer",chosen,plan)
        os.environ["FAKE_RUNNER_DELAY"]="25";self.addCleanup(os.environ.pop,"FAKE_RUNNER_DELAY",None)
        am.start_analysis(groups,expected)
        token="owner-token"
        with am._JOB_LOCK:
            am._ACTIVE["meta"].update(target=chosen["target_name"],owner=token)
        home=AppTest.from_file(APP,default_timeout=120)
        home.session_state["_analysis_session_token"]=token
        home.run();self.assertFalse(home.exception,[e.value for e in home.exception])
        bars=[e.proto.text for e in home.get("progress")]
        self.assertTrue(any(t.startswith("분석 중 · W2 · 노광 60 mJ/cm² · PEB 220 s · ") and "/2장" in t for t in bars),bars)
        dcc=self.wafer_app(condition=C(60.0,None,220.0))                       # Another session while running.
        self.assertTrue(dcc.button(key="dcc_analysis_run").disabled)
        self.assertIn("다른 분석이 실행 중",dcc.button(key="dcc_analysis_run").help)
        self.assertTrue(any("2장" in e.proto.text for e in dcc.get("progress")))  # D shows it on return.
        before=self.q("SELECT COUNT(*) FROM analysis_image_runs")[0][0]
        with am._JOB_LOCK:
            process=am._ACTIVE["process"]
        process.wait(timeout=60)
        with am._JOB_LOCK:
            am._ACTIVE["waiter"]="someone-waiting"
        self.assertIsNone(am.finish_pending(self.p))                           # A waiting script collects itself.
        with am._JOB_LOCK:
            am._ACTIVE.pop("waiter")
        dcc.run()                                                              # Non-owner session collects + saves.
        self.assertIsNone(am.job_snapshot())
        self.assertEqual(self.q("SELECT COUNT(*) FROM analysis_image_runs")[0][0],before+2)
        self.assertIsNone(am.finish_pending(self.p))                           # Never twice.
        self.assertIsNone(dcc.session_state["dcc_analysis_last_run"] if "dcc_analysis_last_run" in dcc.session_state else None)
        home.run()                                                             # Owner: E result, alert, guard.
        last=home.session_state["dcc_analysis_last_run"]
        self.assertEqual(last["db_save"]["status"],"SAVED")
        self.assertEqual(set(home.session_state["dcc_analysis_completed_ids"]),set(expected))
        self.assertTrue(any(e["kind"]=="분석 완료" or e["kind"]=="부분 실패" for e in home.session_state["_notifications"]))
        self.assertEqual(self.q("SELECT COUNT(*) FROM analysis_image_runs")[0][0],before+2)
        groups,_,reason=am.run_eligibility("수동 실행","Wafer",chosen,plan,completed=home.session_state["dcc_analysis_completed_ids"])
        self.assertIsNone(groups);self.assertIn("중복 실행",reason)

    # 16/17/18: KPI tiles == SQLite, mode chips sum == registered images, graph and Wafer table kept.
    def test_06_overview_kpis_match_sqlite(self):
        at=self.dcc()
        status=self.p.system_status();text=self.markdown(at)
        for label,value in (("Batch",status["batches"]),("등록 이미지",status["images"]),
                            ("분석 결과 이미지",status["analysed_images"]),("검토 필요",status["review_images"])):
            self.assertIn(f"<small>{label}</small><b>{value}</b>",text,label)
        modes={m:self.q("SELECT COUNT(*) FROM images i JOIN upload_batches b USING(batch_id) WHERE b.upload_mode=?",m)[0][0]
               for m in ("wafer_batch","repeatability","lighting","single_image")}
        self.assertEqual(sum(modes.values()),status["images"])
        self.assertIn(f"Wafer Batch {modes['wafer_batch']}",text);self.assertIn(f"합계 {status['images']} = 등록 이미지",text)
        self.assertNotIn("Single Image 지원",self.captions(at));self.assertNotIn("SQLite 연결됨",self.captions(at))
        self.assertNotIn("흐름:",self.captions(at))
        self.assertTrue(any(c.key=="sqlite_accumulation" for c in at.get("plotly_chart")) or at.get("plotly_chart"))
        table=next(d.value for d in at.dataframe if "분석 결과 이미지 수" in d.value.columns and "Wafer" in d.value.columns)
        self.assertEqual(list(table.columns),["Wafer","노광","PEB (s)","Batch 수","등록 이미지 수","분석 결과 이미지 수"])
        self.assertIn("대기 중",self.captions(at));self.assertIn("결과 없음",self.captions(at))
        overview=self.markdown(at)
        self.assertNotIn('class="pipeline"',overview);self.assertNotIn('class="image-tiles"',overview)   # TASK18-B-11.1.
        self.assertNotIn("이미지별 최신 분석 기준 · AI 잠정 결과",self.captions(at))


# The inherited MultiWaferV6Tests cases already run in their own module.
for _name in [n for n in dir(mw.MultiWaferV6Tests) if n.startswith("test_")]:
    setattr(BottomCardTests,_name,None)

if __name__=="__main__":
    unittest.main()
