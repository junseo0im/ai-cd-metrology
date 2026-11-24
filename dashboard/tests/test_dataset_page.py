"""TASK 18-B-12A: 전체 Dataset — SQLite search, selection and permanent deletion (temporary v6 DB/storage only).

Registration through the provider into temporary owned storage, analysis through the fake Runner
process (real BatchRunner / AnalysisRepository), deletion through the real service. The operational
DB is hashed before/after (inherited tearDown) and only ever used for the refusal check.
"""
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
import hashlib,json,os,shutil,sqlite3,unittest

from streamlit.testing.v1 import AppTest

import dashboard.tests.test_multi_wafer_v6 as mw  # Module import: its TestCase is not collected twice.
from dashboard.tests.test_multi_wafer_v6 import unique_png,APP
from dashboard.tests.test_upload_default_matching import HAS_PLOTLY
from database.init_db import DEFAULT_DB
from database.migrate_v6 import V6_IMAGE_PROCESS_SQL
from database import image_deletion as deletion
from database.image_deletion import DeletionRefused,delete_images,plan_deletion,recover_pending
from dashboard import dataset_service as service
import dashboard.components.analysis_management as am

E=lambda value:json.dumps([float(value),None])


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed; Dashboard pages cannot load in AppTest")
class DatasetTests(mw.MultiWaferV6Tests):
    def setUp(self):
        super().setUp()
        self.q_root=self.f.root/"quarantine";self.log_root=self.f.root/"deletion_logs"
        for p in (patch.dict(os.environ,{"IDA_DELETION_QUARANTINE":str(self.q_root),"IDA_DELETION_LOG_ROOT":str(self.log_root)}),):
            p.start();self.addCleanup(p.stop)
        up=lambda *names:[{"filename":n,"content":unique_png()} for n in names]
        # Multi-Wafer Batch (W2 4 + W3 2), W2 in another condition, W5 with PEB 120.
        _,self.b1=self.register_multi(up("w2_21_c.png","w2_22_c.png","w2_23_c.png","w2_24_c.png","w3_21_c.png","w3_22_c.png"),
                                      {"W2":(70,220.0),"W3":(60,None)})
        _,self.b2=self.register_multi(up("w2_25_c.png","w2_26_c.png"),{"W2":(60,None)})
        _,self.b3=self.register_multi(up("w5_21_c.png","w5_22_c.png","w5_23_c.png"),{"W5":(70,120.0)})
        # Analysis results for the whole multi-Wafer Batch (fake engines, real Runner/Repository).
        preview=am.target_preview(self.p,"Batch",self.b1["batch_id"]);plan=am.prepare_selected_inputs(self.p,preview)
        groups,expected,_=am.run_eligibility("수동 실행","Batch",preview,plan,confirmed=True)
        am.start_analysis(groups,expected,confirmed_count=len(expected))
        self.assertEqual(am.save_analysis_result(am.wait_for_analysis(),self.p)["status"],"SAVED")

    # ---- helpers ------------------------------------------------------------------------------------
    def ids_where(self,wafer=None,exposure=None,peb="ANY"):
        rows=self.q(V6_IMAGE_PROCESS_SQL)
        return {r[0] for r in rows if (wafer is None or r[2]==wafer) and (exposure is None or r[5]==exposure)
                and (peb=="ANY" or r[7]==peb)}
    def count(self,table,where="",*args):
        return self.q(f"SELECT COUNT(*) FROM {table} "+where,*args)[0][0]
    def by_name(self,batch):
        return {r[0]:r[1] for r in self.q("SELECT original_filename,image_id FROM images WHERE batch_id=?",batch["batch_id"])}
    def files_of(self,image_id):
        storage=self.q("SELECT storage_path FROM images WHERE image_id=?",image_id)[0][0]
        results=[self.out/r[0] for r in self.q(
            "SELECT b.summary_path FROM analysis_image_runs a JOIN analysis_batch_runs b USING(batch_run_id) WHERE a.image_id=?",image_id)]
        return Path(storage),[Path(s).parent/"images"/hashlib.sha256(image_id.encode()).hexdigest() for s in results]
    def delete(self,ids,**kw):
        plan=plan_deletion(self.f.db,ids,storage_root=self.f.sources,result_root=self.out)
        return plan,delete_images(self.f.db,ids,expected_token=plan["token"],storage_root=self.f.sources,
                                  result_root=self.out,actor="TEST",**kw)
    def state(self):
        tables=("images","analysis_image_runs","defect_detections","metrology_measurements","line_judgments",
                "upload_batches","batch_wafers","dies","pattern_sites","wafers","process_runs","analysis_batch_runs")
        files=sorted((str(p),hashlib.sha256(p.read_bytes()).hexdigest()) for root in (self.f.sources,self.out)
                     for p in Path(root).rglob("*") if p.is_file())
        return {t:self.count(t) for t in tables},files
    def page(self,admin=False):
        with patch("dashboard.auth.is_admin",return_value=admin):
            at=AppTest.from_file(APP,default_timeout=120).run()
            at.switch_page("pages/dataset.py").run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        return at

    # 1-8: real SQLite rows, filters (single/combined/PEB NULL), multi-Wafer Batch split per image.
    def test_01_filters_match_sql(self):
        p=self.p
        self.assertEqual(p.dataset_count(),self.count("images"))
        options=p.dataset_filter_options()
        self.assertIn("NULL",options["pebs"]);self.assertIn(E(70),options["exposures"])
        cases=[({"wafer_id":"W2"},self.ids_where("W2")),({"exposure":E(70)},self.ids_where(exposure=70.0)),
               ({"peb":"220"},self.ids_where(peb=220.0)),({"wafer_id":"W2","exposure":E(60)},self.ids_where("W2",60.0)),
               ({"wafer_id":"W2","exposure":E(70),"peb":"220"},self.ids_where("W2",70.0,220.0)),
               ({"peb":"NULL"},self.ids_where(peb=None)),({"wafer_id":"전체","exposure":"전체","peb":"전체"},self.ids_where())]
        for filters,expected in cases:
            self.assertTrue(expected,filters)
            self.assertEqual(set(p.dataset_ids(filters)),expected,filters);self.assertEqual(p.dataset_count(filters),len(expected))
        b1=set(self.b1["image_ids"])
        self.assertEqual(len(b1&set(p.dataset_ids({"wafer_id":"W2"}))),4)     # Own Wafer, not the Batch's primary.
        self.assertEqual(len(b1&set(p.dataset_ids({"wafer_id":"W3"}))),2)
        page=p.dataset_page({"wafer_id":"W3"},limit=10)
        self.assertEqual(set(page["image_id"]),self.ids_where("W3"))
        self.assertTrue((page["latest_outcome"].notna()).all())                # Analysed in setUp.

    # 9-13 + 15 + 26 (Mock text gone, GUEST view-only, select-all = filtered result, reset on filter change).
    def test_02_page_guest_selection_and_reset(self):
        at=self.page(admin=False)
        text=" | ".join([m.value for m in at.markdown]+[c.value for c in at.caption]+[i.value for i in at.info])
        for gone in ("참고용 예시 화면","bundled Mock catalog","이 목록은 예시입니다","DEMO",
                     "SQLite 운영 DB · 등록 완료 ≠ 분석 완료","SQLite 등록 이미지 · Wafer·노광·PEB 조건 검색"):  # 18-B-13A.1
            self.assertNotIn(gone,text)
        self.assertIn(f"<small>검색 결과</small><b>{self.count('images')}장</b>",text)
        self.assertTrue(at.button(key="ds_delete").disabled)
        self.assertIn("관리자 전용",text)
        at.selectbox(key="ds_filter_wafer").select("W2").run()
        at.selectbox(key="ds_filter_exposure").set_value(E(70)).run()
        at.button(key="ds_search").click().run()
        w2_70=self.ids_where("W2",70.0)
        self.assertEqual(at.session_state["ds_applied"],{"wafer_id":"W2","exposure":E(70),"peb":"전체"})
        at.button(key="ds_select_all").click().run()
        self.assertEqual(at.session_state["ds_selected"],w2_70)                 # Filtered result only, not the DB.
        self.assertEqual(len(at.dataframe[0].value),len(w2_70))
        self.assertTrue(at.dataframe[0].value["선택"].all())
        at.selectbox(key="ds_filter_peb").set_value("NULL").run()              # Filter change -> selection cleared.
        self.assertEqual(at.session_state["ds_selected"],set())
        at.button(key="ds_reset").click().run()
        self.assertEqual(at.session_state["ds_applied"],{"wafer_id":"전체","exposure":"전체","peb":"전체"})
        with self.assertRaises(DeletionRefused):                               # Server-side: GUEST refused.
            service.prepare(self.p,list(w2_70),{"wafer_id":"W2","exposure":E(70),"peb":"전체"})
        with patch("dashboard.sqlite_provider.runtime_provider",return_value=self.p), \
             self.assertRaises(DeletionRefused):
            service.execute(self.p,list(w2_70),{"wafer_id":"W2","exposure":E(70),"peb":"전체"},"x"*64,"영구 삭제")
        self.assertEqual(self.count("images"),len(self.ids_where()))

    # 12: select-all spans pages (more rows than one page).
    def test_03_select_all_spans_pages(self):
        extra=[{"filename":f"w6_{d}_{s}.png","content":unique_png()} for d in (11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,31,32,33)
               for s in ("lt","c","rb")]
        self.register_multi(extra,{"W6":(50,220.0)})
        with patch("dashboard.pages.dataset.PAGE_SIZE",20,create=True):
            at=self.page()
            at.selectbox(key="ds_filter_wafer").select("W6").run();at.button(key="ds_search").click().run()
            at.button(key="ds_select_all").click().run()
            at.selectbox(key="ds_page").set_value(2).run()
        self.assertEqual(len(at.session_state["ds_selected"]),57)
        self.assertEqual(at.session_state["ds_selected"],self.ids_where("W6"))

    # 14/16/19/20/23-25: ADMIN confirmation, single deletion, everything else preserved, views updated.
    def test_04_admin_confirm_and_single_delete(self):
        target=self.by_name(self.b1)["w2_21_c.png"]
        original,results=self.files_of(target)
        self.assertTrue(original.is_file() and results and all(r.is_dir() for r in results))
        before,_=self.state()
        keep={i:self.q("SELECT storage_path,sha256 FROM images WHERE image_id=?",i)[0] for i in self.ids_where() if i!=target}
        admin=patch("dashboard.auth.is_admin",return_value=True);admin.start();self.addCleanup(admin.stop)
        user=patch("dashboard.auth.current_user",return_value={"role":"ADMIN","display_name":"ADMIN"});user.start();self.addCleanup(user.stop)
        at=self.page(admin=True)
        at.selectbox(key="ds_filter_wafer").select("W2").run();at.button(key="ds_search").click().run()
        at.session_state["ds_selected"]={target};at.run()
        if True:
            at.button(key="ds_delete").click().run()
            from dashboard.pages.dataset import CONFIRM_WARNING
            self.assertIn(CONFIRM_WARNING,[e.value for e in at.error])
            panel=" | ".join([m.value for m in at.markdown if not m.value.startswith("<style>")]+[c.value for c in at.caption])
            self.assertEqual(panel.count("적용 필터"),1)                                       # Page summary only, not the panel.
            for gone in ("종속 행","유지: 선택하지 않은"):                                     # TASK18-B-12A.2.
                self.assertNotIn(gone,panel)
            for tile in ("삭제 이미지</small><b>1장","원본 파일</small><b>1개","분석 결과</small><b>1건","결과 폴더</small><b>1개"):
                self.assertIn(tile,panel)
            impact=next(d.value for d in at.dataframe if "삭제 이미지 수" in d.value.columns)
            self.assertEqual(impact.iloc[0][["삭제 이미지 수","Batch 전체 이미지 수","삭제 후 남는 이미지 수"]].tolist(),[1,6,5])
            self.assertEqual(impact.iloc[0]["Wafer별 삭제 수량"],"W2 1")
            box=next(t for t in at.text_input if t.key.startswith("ds_phrase_"))
            self.assertEqual((box.label,box.placeholder),("확인 문구","영구 삭제 입력"))
            final=at.button(key="ds_final");self.assertTrue(final.disabled);self.assertEqual(final.label,"영구 삭제")
            box.input("영구삭제").run();self.assertTrue(at.button(key="ds_final").disabled)   # Wrong phrase.
            next(t for t in at.text_input if t.key.startswith("ds_phrase_")).input("영구 삭제").run()
            at.button(key="ds_final").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("영구 삭제 완료 · 이미지 1장" in s.value for s in at.success),[e.value for e in at.error])
        after,_=self.state()
        runs=before["analysis_image_runs"]-after["analysis_image_runs"]
        self.assertEqual((before["images"]-after["images"],runs),(1,1))
        for table in ("upload_batches","batch_wafers","dies","pattern_sites","wafers","process_runs","analysis_batch_runs"):
            self.assertEqual(before[table],after[table],table)                        # Structure / run history kept.
        self.assertFalse(original.exists() or original.parent.exists())
        self.assertFalse(any(r.exists() for r in results))
        for image_id,(path,sha) in keep.items():                                       # Others byte-identical.
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),sha)
        self.assertEqual(list(self.q_root.glob("*")),[])
        log=json.loads(next(self.log_root.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual([i["image_id"] for i in log["images"]],[target]);self.assertEqual(log["actor"],"ADMIN")
        # Other views (same provider queries the pages use).
        self.assertEqual(int(self.p.batch_record(self.b1["batch_id"])["image_count"]),5)
        members=self.p.batch_wafers().set_index(["batch_id","wafer_id"])
        self.assertEqual(int(members.loc[(self.b1["batch_id"],"W2"),"image_count"]),3)
        self.assertEqual(int(members.loc[(self.b1["batch_id"],"W3"),"image_count"]),2)
        self.assertNotIn(target,self.p.latest_analysis());self.assertNotIn(target,set(am.collect_images(self.p)["image_id"]))
        self.assertNotIn(target,set(self.p.condition_images([self.b1["batch_id"]],"W2")["image_id"]))
        preview=am.target_preview(self.p,"Batch",self.b1["batch_id"])
        self.assertNotIn(target,preview["image_ids"]);self.assertEqual(preview["image_count"],5)
        self.assertEqual(self.p.system_status()["images"],self.count("images"))
        for page in ("pages/data_control.py","pages/inspection_analysis.py","pages/dataset.py"):
            view=AppTest.from_file(APP,default_timeout=120).run();view.switch_page(page).run()
            self.assertFalse(view.exception,(page,[e.value for e in view.exception]))
        view=AppTest.from_file(APP,default_timeout=120).run()
        view.session_state["selected_sqlite_batch_id"]=self.b1["batch_id"];view.session_state["batch_detail_source"]="SQLite"
        view.switch_page("pages/batch_detail.py").run();self.assertFalse(view.exception)

    # 17/19/22: several images of one Wafer of a multi-Wafer Batch; the other Wafer untouched; repeat refused.
    def test_05_multi_delete_keeps_other_wafer_and_refuses_repeat(self):
        w2=sorted(i for i in self.b1["image_ids"] if i in self.ids_where("W2"))
        plan,result=self.delete(w2)
        self.assertEqual(result["images"],4);self.assertEqual(plan["wafers"],["W2"])
        self.assertEqual(self.p.batch_record(self.b1["batch_id"])["image_count"],2)        # Batch kept with W3.
        self.assertEqual(set(self.p.dataset_ids({"wafer_id":"W3"}))&set(self.b1["image_ids"]),
                         set(self.b1["image_ids"])-set(w2))
        self.assertEqual(self.count("batch_wafers","WHERE batch_id=? AND wafer_id='W2'",self.b1["batch_id"]),1)  # Reported, kept.
        with self.assertRaises(DeletionRefused):                                            # Same request again.
            delete_images(self.f.db,w2,expected_token=plan["token"],storage_root=self.f.sources,result_root=self.out,actor="T")
        with self.assertRaises(DeletionRefused):                                            # Stale confirmation.
            other=sorted(self.b2["image_ids"])
            delete_images(self.f.db,other,expected_token=plan["token"],storage_root=self.f.sources,result_root=self.out,actor="T")
        # Whole Batch: rows/files gone, the Batch record stays (no automatic Batch deletion).
        plan,result=self.delete(self.b2["image_ids"])
        self.assertEqual(plan["emptied_batches"],[self.b2["batch_id"]])
        self.assertEqual(self.count("upload_batches","WHERE batch_id=?",self.b2["batch_id"]),1)
        self.assertEqual(self.count("images","WHERE batch_id=?",self.b2["batch_id"]),0)

    # 21: failures roll back DB and files exactly; crash journals are resolved from the DB state.
    def test_06_failure_restores_db_and_files(self):
        ids=sorted(self.b1["image_ids"])
        before=self.state()
        for stage in ("move","db"):
            with self.assertRaises(RuntimeError):
                self.delete(ids,_fail_at=stage)
            self.assertEqual(self.state(),before,stage)
            self.assertEqual(list(self.q_root.glob("*")),[])
        # Crash after moving (rows still present) -> files restored on the next call.
        plan=plan_deletion(self.f.db,ids[:1],storage_root=self.f.sources,result_root=self.out)
        folder=self.q_root/"crash";folder.mkdir(parents=True)
        moves=[]
        for n,item in enumerate(plan["files"]):
            target=folder/str(n);shutil.move(item["path"],target);moves.append({"from":item["path"],"to":str(target)})
        (folder/"manifest.json").write_text(json.dumps({"deletion_id":"crash","db_path":str(self.f.db.resolve()),
            "state":"moving","image_ids":ids[:1],"moves":moves}),encoding="utf-8")
        self.assertEqual(recover_pending(self.f.db,self.q_root)["restored"],["crash"])
        self.assertEqual(self.state(),before)

    # 18: ownership not provable -> whole request refused, nothing changed.
    def test_07_shared_or_unowned_files_refused(self):
        target=self.by_name(self.b2)["w2_25_c.png"]
        original,_=self.files_of(target)
        before=self.state()
        intruder=original.parent/"other.png";intruder.write_bytes(b"x")               # Folder not exclusively owned.
        with self.assertRaises(DeletionRefused):
            self.delete([target])
        intruder.unlink()
        backup=original.read_bytes();original.write_bytes(backup+b"tampered")         # SHA differs from registration.
        with self.assertRaises(DeletionRefused):
            self.delete([target])
        original.write_bytes(backup)
        fixture=[i for i in self.ids_where() if self.q("SELECT 1 FROM images WHERE image_id=? AND batch_id LIKE 'TEST_%'",i)]
        if fixture:                                                                   # Fixture rows: not owned copies.
            with self.assertRaises(DeletionRefused):
                self.delete(fixture[:1])
        self.assertEqual(self.state(),before)

    # 15 (server), operational guard, analysis-running guard.
    def test_08_operational_and_running_analysis_refused(self):
        op=mw.op_sha()
        with self.assertRaises(DeletionRefused):                                      # Refused before any connection.
            delete_images(DEFAULT_DB,["IMG-x"],expected_token="x",storage_root=self.f.sources,result_root=self.out,actor="T")
        if op:self.assertEqual(mw.op_sha(),op)
        from dashboard.sqlite_provider import SQLiteProvider
        operational=SQLiteProvider(DEFAULT_DB,self.f.sources,self.f.root/"staging")
        with patch("dashboard.auth.is_admin",return_value=True),patch.dict(os.environ,{service.DELETE_SWITCH:""}):
            enabled,reason=service.deletion_status(operational)
            self.assertFalse(enabled);self.assertIn("서버 설정",reason)
            self.assertTrue(service.deletion_status(self.p)[0])
            with patch("dashboard.components.analysis_management.job_snapshot",return_value={"running":True}):
                enabled,reason=service.deletion_status(self.p);self.assertFalse(enabled);self.assertIn("분석 실행 중",reason)
        with patch.dict(os.environ,{service.DELETE_SWITCH:"1"}):                       # TASK18-B-12A.1: switch ON.
            with patch("dashboard.auth.is_admin",return_value=True):
                self.assertEqual(service.deletion_status(operational),(True,""))       # Status only; nothing deleted.
            with patch("dashboard.auth.is_admin",return_value=False):
                enabled,reason=service.deletion_status(operational)
                self.assertFalse(enabled);self.assertIn("관리자 전용",reason)
                with self.assertRaises(DeletionRefused):                             # Direct call without ADMIN.
                    service.execute(operational,["IMG-x"],{},"x"*64,"영구 삭제")
        if op:self.assertEqual(mw.op_sha(),op)
        with patch("dashboard.auth.is_admin",return_value=True),patch.dict(os.environ,{service.DELETE_SWITCH:""}):
            with self.assertRaises(DeletionRefused):                                  # Selection outside the filter.
                service.prepare(self.p,list(self.b3["image_ids"]),{"wafer_id":"W2","exposure":"전체","peb":"전체"})


for _name in [n for n in dir(mw.MultiWaferV6Tests) if n.startswith("test_")]:
    setattr(DatasetTests,_name,None)

if __name__=="__main__":
    unittest.main()
