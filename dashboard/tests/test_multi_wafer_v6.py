"""TASK 18-B-10B-2: several Wafers in ONE Batch on Schema v6 (temporary DBs only).

* MultiWaferV6Tests: temporary fixture DB migrated v4 -> v5 -> v6. Registration through the real
  Dashboard UI (AppTest) or the provider, analysis through the real BatchRunner / AnalysisRunner /
  AnalysisRepository with fake model engines (fake_dashboard_runner.py), then every view.
* OperationalCopyV6Tests: an SQLite Online Backup of the operational DB, migrated to v6 in a temp
  directory; the operational DB is only opened read-only and hashed before/after.
Wafer counts / images per Wafer are deliberately varied: nothing assumes 3 Wafers or 72 images.
"""
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib,json,os,shutil,sqlite3,sys,tempfile,unittest

from streamlit.testing.v1 import AppTest

from database.init_db import DEFAULT_DB,connect_database
from database.migrate_v5 import migrate_database_v5
from database.migrate_v6 import migrate_database_v6,validate_v6,V6_IMAGE_PROCESS_SQL
from database.tests.test_image_registry import png_bytes
from database.tests.test_integrated_registration import zip_bytes
from dashboard.tests.test_upload_default_matching import Base,HAS_PLOTLY
from dashboard.sqlite_provider import SQLiteProvider
import dashboard.components.analysis_management as am

APP=str(Path(__file__).resolve().parents[1]/"app.py")
FAKE=Path(__file__).with_name("fake_dashboard_runner.py")
DIES=[f"{r}{c}" for r in range(1,4) for c in range(1,9)]
SITES=[(d,p) for d in DIES for p in ("lt","c","rb")]          # 72 = one complete set (an example only).
_COUNTER=[0]


def unique_png():
    _COUNTER[0]+=1;i=_COUNTER[0]
    return png_bytes(width=3+i%61,height=2+i//61,pixel=i%251)


def names(wafer,count,start=0):
    return [f"{wafer.lower()}_{d}_{p}.png" for d,p in SITES[start:start+count]]


def op_sha():
    return hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest() if DEFAULT_DB.is_file() else None


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed; Data Control page cannot load in AppTest")
class MultiWaferV6Tests(Base):
    def setUp(self):
        super().setUp()
        self.op=op_sha()
        migrate_database_v5(self.f.db);self.assertEqual(migrate_database_v6(self.f.db)["migrated"],True)
        self.out=self.f.root/"results";self.out.mkdir()
        self.uploads=[]
        for p in (patch.dict(os.environ,{"IDA_DB_PATH":str(self.f.db),"IDA_STORAGE_ROOT":str(self.f.sources),
                      "IDA_STAGING_ROOT":str(self.f.root/"staging"),"IDA_ANALYSIS_PYTHON":sys.executable,
                      "IDA_ANALYSIS_OUTPUT_ROOT":str(self.out)}),
                  patch.object(am,"_analysis_command",lambda python,request,output:
                      [str(python),str(FAKE),"--request",str(request),"--output-root",str(output)]),
                  patch("streamlit.file_uploader",side_effect=lambda *a,**k:[] if k.get("disabled") else self.uploads)):
            p.start();self.addCleanup(p.stop)
        os.environ.pop("FAKE_RUNNER_MODE",None);am._ACTIVE.clear()

    def tearDown(self):
        if self.op:self.assertEqual(op_sha(),self.op)                       # 23. operational DB untouched.

    # ---- helpers ---------------------------------------------------------------------------------
    def q(self,sql,*args):
        with closing(sqlite3.connect(self.f.db)) as c:
            return c.execute(sql,args).fetchall()

    def files(self,spec):
        return [{"filename":n,"content":unique_png()} for wafer,count in spec for n in names(wafer,count)]

    def multi_metadata(self,processes,**extra):
        wafers={}
        for wafer,(exposure,peb) in processes.items():
            existing=self.q("SELECT 1 FROM wafers WHERE wafer_id=?",wafer)
            config={"peb_time_s":peb,"batch_process_run_id":"PROCESS-B-"+wafer+"-"+os.urandom(4).hex()}
            if not existing:config["process_run_id"]="PROCESS-W-"+wafer+"-"+os.urandom(4).hex()
            if exposure is not None:config["process_run"]={"exposure_value":float(exposure),"exposure_unit":None}
            wafers[wafer]=config
        return {"batch_id":"BATCH-"+os.urandom(8).hex(),"wafer_id":sorted(processes)[0],"upload_mode":"wafer_batch",
                "dataset_role":"development","batch_status":"registered","wafers":wafers,**extra}

    def register_multi(self,uploads,processes,decisions=None):
        inputs=self.stage(uploads)
        preview=self.p.preview(inputs,self.multi_metadata(processes))
        plan=self.p.prepare_registration(preview,decisions or {})
        self.assertTrue(plan["registration_ready"],[(r["original_filename"],r["issues"]) for r in preview["images"] if r["issues"]][:3])
        return preview,self.p.register_selection(plan,approved=True)

    def per_wafer(self,batch_id):
        rows=self.q(V6_IMAGE_PROCESS_SQL+"WHERE i.batch_id=?",batch_id)
        out={}
        for _,_,wafer,_,_,exposure,unit,peb,_ in rows:
            entry=out.setdefault(wafer,{"n":0,"process":set()});entry["n"]+=1;entry["process"].add((exposure,unit,peb))
        return out

    def dcc(self):
        at=AppTest.from_file(APP,default_timeout=120).run()
        at.switch_page("pages/data_control.py").run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        return at

    def upload_ui(self,spec):
        self.uploads=[SimpleNamespace(name=f["filename"],getvalue=lambda c=f["content"]:c) for f in self.files(spec)]
        return self.dcc()

    def wafer_input(self,at,field,wafer):
        label=("노광 조건 · " if field=="exposure" else "PEB Time (s) · ")+wafer
        return next(n for n in at.number_input if n.label==label)

    # ---- 3/5/6/8/9/10/12: PNG, three Wafers x 72 = 216 -> ONE Batch through the Dashboard UI ---
    def test_01_png_216_three_wafers_one_batch_ui(self):
        before=self.q("SELECT COUNT(*) FROM upload_batches")[0][0]
        at=self.upload_ui([("W1",72),("W2",72),("W3",72)])
        labels=[n.label for n in at.number_input]
        self.assertEqual([l for l in labels if l.startswith("노광 조건 · ")],["노광 조건 · W1","노광 조건 · W2","노광 조건 · W3"])
        self.assertNotIn("Batch Wafer ID",[t.label for t in at.text_input])           # No single common Wafer.
        self.assertTrue(at.button(key="sqlite_register").disabled)                     # 노광 required.
        self.assertTrue(any("노광 조건 미입력" in c.value for c in at.caption))
        for wafer,exposure,peb in (("W1",50,220),("W2",60,220),("W3",70,120)):
            self.wafer_input(at,"exposure",wafer).set_value(exposure).run()
            self.wafer_input(at,"peb",wafer).set_value(peb).run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        table=next(d.value for d in at.dataframe if "REVIEW_REQUIRED" in d.value.columns and "Wafer ID" in d.value.columns)
        self.assertEqual(table[["Wafer ID","이미지","노광","PEB (s)","READY","BLOCKED"]].values.tolist(),
                         [["W1",72,"50 mJ/cm²","220",72,0],["W2",72,"60 mJ/cm²","220",72,0],["W3",72,"70 mJ/cm²","120",72,0]])
        self.assertTrue(any("전체 216장 · Wafer 3개 · READY 216" in c.value for c in at.caption))
        self.assertFalse(at.button(key="sqlite_register").disabled)
        at.button(key="sqlite_register").click().run();self.assertFalse(at.exception)
        result=at.session_state["sqlite_last_result"]
        self.assertEqual(self.q("SELECT COUNT(*) FROM upload_batches")[0][0],before+1)    # Exactly ONE Batch.
        self.assertEqual(result["image_count"],216)
        self.assertEqual(self.q("SELECT wafer_id,peb_time_s FROM batch_wafers WHERE batch_id=? ORDER BY 1",result["batch_id"]),
                         [("W1",220.0),("W2",220.0),("W3",120.0)])
        self.assertEqual({w:(v["n"],v["process"]) for w,v in self.per_wafer(result["batch_id"]).items()},
                         {"W1":(72,{(50.0,None,220.0)}),"W2":(72,{(60.0,None,220.0)}),"W3":(72,{(70.0,None,120.0)})})
        # W1 existed with exposure 60: its Wafer record is untouched, the Batch-Wafer has its own record.
        self.assertEqual(self.q("SELECT r.exposure_value FROM wafers w JOIN process_runs r USING(process_run_id) WHERE w.wafer_id='W1'"),[(60.0,)])
        with closing(connect_database(self.f.db,readonly=True)) as c:
            validate_v6(c);self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(),[])
        self.assertEqual(self.q("SELECT process_run_id,peb_temperature_c FROM upload_batches WHERE batch_id=?",result["batch_id"]),[(None,None)])

    # ---- 4/18/19: ZIP inputs (any number of ZIPs, mixed Wafers per ZIP, duplicates, name conflict)
    def test_02_zip_archives_variable_and_mixed(self):
        f=self.files([("W2",10),("W3",7),("W4",4)])
        w2,w3,w4=f[:10],f[10:17],f[17:]
        uploads=[{"filename":"W2.zip","content":zip_bytes([(x["filename"],x["content"]) for x in w2[:6]])},
                 {"filename":"lot_b_W2.zip","content":zip_bytes([(x["filename"],x["content"]) for x in w2[6:]])},  # Same Wafer in 2 ZIPs.
                 {"filename":"mixed.zip","content":zip_bytes([(x["filename"],x["content"]) for x in w3+w4])},     # Several Wafers, 1 ZIP.
                 {"filename":"extra.png","content":unique_png()}]                                                # PNG+ZIP mixed.
        uploads[-1]["filename"]="w4_38_rb.png"
        _,result=self.register_multi(uploads,{"W2":(50,220.0),"W3":(50,220.0),"W4":(80,None)})   # Same and different processes.
        counts={w:v["n"] for w,v in self.per_wafer(result["batch_id"]).items()}
        self.assertEqual(counts,{"W2":10,"W3":7,"W4":5})
        self.assertEqual(result["image_count"],22)
        # Conflict: a ZIP named for W5 that contains W6 images -> blocked, never re-labelled.
        bad=[{"filename":"W5.zip","content":zip_bytes([(n,unique_png()) for n in names("W6",2)])}]
        preview=self.p.preview(self.stage(bad),self.multi_metadata({"W6":(60,None)}))
        self.assertFalse(preview["registration_ready"])
        self.assertTrue(all(any("ZIP 이름 Wafer(W5) ≠ 이미지 Wafer(W6)" in i["message"] for i in r["issues"]) for r in preview["images"]))
        # Duplicate bytes across ZIPs (SHA) vs. a re-shot of the same site (different bytes).
        same=unique_png();other=unique_png()
        dup=[{"filename":"a.zip","content":zip_bytes([("w7_11_c.png",same)])},
             {"filename":"b.zip","content":zip_bytes([("w7_11_c.png",same)])},
             {"filename":"c.zip","content":zip_bytes([("w7_12_c.png",other)])},
             {"filename":"d.zip","content":zip_bytes([("w7_12_c.png",unique_png())])}]
        preview=self.p.preview(self.stage(dup),self.multi_metadata({"W7":(60,None)}))
        rows={(r["original_filename"],bool(r["duplicate_input_indices"]),bool(r["same_site_input_indices"])) for r in preview["images"]}
        self.assertEqual(rows,{("w7_11_c.png",True,True),("w7_12_c.png",False,True)})
        summary=preview["wafer_summary"][0]
        self.assertEqual((summary["wafer_id"],summary["image_count"],summary["duplicate_count"],summary["site_count"]),("W7",4,2,2))

    # ---- 7/8 + variable Wafer counts (1, 2 with different counts, 6) ----------------------------
    def test_03_variable_wafer_and_image_counts(self):
        for spec in ([("W2",3)],[("W2",5),("W9",2)],[("W2",1),("W3",4),("W5",2),("W8",3),("W10",1),("W11",6)]):
            processes={w:(40+i*5,None if i%2 else 110.0) for i,(w,_) in enumerate(spec)}
            _,result=self.register_multi(self.files(spec),processes)
            got=self.per_wafer(result["batch_id"])
            self.assertEqual({w:v["n"] for w,v in got.items()},dict(spec))
            self.assertEqual({w:v["process"] for w,v in got.items()},
                             {w:{(float(e),None,p)} for w,(e,p) in processes.items()})
            members=self.q("SELECT COUNT(*) FROM batch_wafers WHERE batch_id=?",result["batch_id"])[0][0]
            self.assertEqual(members,len(spec))
            single=len(spec)==1   # A single-Wafer v6 Batch mirrors its process in upload_batches.
            mirrored=self.q("SELECT peb_temperature_c IS NOT NULL OR process_run_id IS NOT NULL FROM upload_batches WHERE batch_id=?",result["batch_id"])[0][0]
            self.assertEqual(bool(mirrored),single)
        with closing(connect_database(self.f.db,readonly=True)) as c:
            validate_v6(c)

    # ---- unidentified Wafer: never assigned automatically; manual fix joins its own Wafer row -----
    def test_04_missing_wafer_needs_user_and_is_not_guessed(self):
        uploads=self.files([("W2",2)])+[{"filename":"scan_0001.png","content":unique_png()}]
        self.uploads=[SimpleNamespace(name=f["filename"],getvalue=lambda c=f["content"]:c) for f in uploads]
        at=self.dcc()
        self.assertEqual([n.label for n in at.number_input if n.label.startswith("노광 조건 · ")],["노광 조건 · W2"])
        self.assertTrue(any("Wafer 미확인 1장" in c.value for c in at.caption))
        self.wafer_input(at,"exposure","W2").set_value(55).run()
        self.assertTrue(at.button(key="sqlite_register").disabled)                     # Unknown image blocks.
        unknown=next(r["image_id"] for r in at.session_state["sqlite_preview"]["images"] if r["original_filename"]=="scan_0001.png")
        at.session_state["sqlite_uploads"]["manual_metadata"][unknown]={"wafer_id":"W6","die_code":"21","position_code":"lt","legacy_defect_code":None}
        at.run()
        self.assertEqual([n.label for n in at.number_input if n.label.startswith("노광 조건 · ")],["노광 조건 · W2","노광 조건 · W6"])
        self.assertTrue(at.button(key="sqlite_register").disabled)                     # W6 still needs 노광.
        self.wafer_input(at,"exposure","W6").set_value(65).run()
        self.assertEqual(self.wafer_input(at,"exposure","W2").value,55)                 # Other Wafer unchanged.
        self.assertFalse(at.button(key="sqlite_register").disabled)
        at.button(key="sqlite_register").click().run()
        got=self.per_wafer(at.session_state["sqlite_last_result"]["batch_id"])
        self.assertEqual({w:(v["n"],v["process"]) for w,v in got.items()},{"W2":(2,{(55.0,None,None)}),"W6":(1,{(65.0,None,None)})})

    # ---- common value: explicit targets, empty cells only -------------------------------------
    def test_05_common_value_fills_only_empty_cells(self):
        at=self.upload_ui([("W2",2),("W3",2),("W4",1)])
        self.wafer_input(at,"exposure","W2").set_value(55).run()
        scope=at.session_state["sqlite_uploads"]["inputs"][0]["image_id"]
        at.number_input(key="sqlite_wafer_common_exposure_"+scope).set_value(70).run()
        at.number_input(key="sqlite_wafer_common_peb_"+scope).set_value(200).run()
        at.multiselect(key="sqlite_wafer_common_targets_"+scope).set_value(["W2","W3"]).run()
        at.button(key="sqlite_wafer_common_apply_"+scope).click().run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        values={w:(self.wafer_input(at,"exposure",w).value,self.wafer_input(at,"peb",w).value) for w in ("W2","W3","W4")}
        self.assertEqual(values,{"W2":(55,200),"W3":(70,200),"W4":(None,None)})     # W2 kept; W4 not targeted.
        self.assertTrue(any("기존 값 유지(덮어쓰지 않음): W2" in c.value for c in at.caption))

    # ---- 20: forced failure mid-registration rolls back everything --------------------------------
    def test_06_failure_rolls_back_whole_batch(self):
        import database.batch_registry as registry
        before={t:self.q(f"SELECT COUNT(*) FROM {t}")[0][0] for t in ("upload_batches","batch_wafers","images","process_runs","wafers","dies","pattern_sites")}
        inputs=self.stage(self.files([("W2",3),("W3",3)]))
        preview=self.p.preview(inputs,self.multi_metadata({"W2":(50,None),"W3":(60,None)}))
        plan=self.p.prepare_registration(preview,{})
        real=registry.register_image;calls=[0]
        def failing(*a,**k):
            calls[0]+=1
            if calls[0]==5:raise registry.ImageRegistrationError("injected failure")
            return real(*a,**k)
        with patch.object(registry,"register_image",side_effect=failing):
            with self.assertRaises(Exception):
                self.p.register_selection(plan,approved=True)
        after={t:self.q(f"SELECT COUNT(*) FROM {t}")[0][0] for t in before}
        self.assertEqual(after,before)
        self.assertEqual(list(self.f.sources.rglob("w2_*")),[])                        # Owned copies removed.

    # ---- 13/14/15: analysis requests per image, whole-Batch run, fake Runner, saved and queried ----
    def test_07_batch_analysis_requests_fake_run_and_views(self):
        uploads=[{"filename":n,"content":unique_png()} for n in ("w2_13_c.png","w2_21_lt.png","w3_14_c.png","w3_22_rb.png","w5_23_c.png")]
        _,result=self.register_multi(uploads,{"W2":(50,220.0),"W3":(60,None),"W5":(70,120.0)})
        batch=result["batch_id"]
        plan=self.p.prepare_analysis_inputs(batch)
        self.assertEqual(plan["counts"],{"READY":5,"REVIEW_REQUIRED":0,"BLOCKED":0})       # Whole Batch, all Wafers.
        own={i["image_id"]:i["candidate_request"] for i in plan["items"]}
        wafer_of={r[0]:r[2] for r in self.q(V6_IMAGE_PROCESS_SQL+"WHERE i.batch_id=?",batch)}
        expected={"W2":(50.0,220.0),"W3":(60.0,None),"W5":(70.0,120.0)}
        for image_id,request in own.items():
            self.assertEqual(request["wafer_id"],wafer_of[image_id])                        # Never the primary Wafer.
            self.assertEqual((request["exposure_value"],request["peb_temperature_c"]),expected[request["wafer_id"]])
            self.assertEqual(request["calibration"]["calibration_id"],"optiview-native-0.1592")
        preview=am.target_preview(self.p,"Batch",batch);selected=am.prepare_selected_inputs(self.p,preview)
        groups,_,reason=am.run_eligibility("수동 실행","Batch",preview,selected)
        self.assertIsNone(groups);self.assertIn("대량 실행",reason)                          # >3 needs confirmation.
        groups,expected_sha,reason=am.run_eligibility("수동 실행","Batch",preview,selected,confirmed=True)
        self.assertEqual(reason,"");self.assertEqual([g["batch_id"] for g in groups],[batch])  # ONE Runner group.
        am.start_analysis(groups,expected_sha,confirmed_count=len(expected_sha))
        run=am.wait_for_analysis()
        saved=am.save_analysis_result(run,self.p)
        self.assertEqual(saved["status"],"SAVED",saved)
        outcomes={r[0]:r[1] for r in self.q("SELECT image_id,outcome FROM analysis_image_runs")}
        by_name={r[0]:r[1] for r in self.q("SELECT original_filename,image_id FROM images WHERE batch_id=?",batch)}
        self.assertEqual(outcomes[by_name["w2_13_c.png"]],"MODEL_PD_EXCLUDED")             # PD policy kept.
        self.assertEqual(outcomes[by_name["w3_14_c.png"]],"FAILED")                        # Partial failure saved.
        groups,_,reason=am.run_eligibility("수동 실행","Batch",preview,selected,confirmed=True,completed=set(expected_sha))
        self.assertIsNone(groups);self.assertIn("중복 실행",reason)
        latest=self.p.latest_analysis()
        self.assertEqual({latest[i]["wafer_id"] for i in by_name.values()},{"W2","W3","W5"})
        self.assertEqual({i:latest[i]["wafer_id"] for i in by_name.values()},{i:wafer_of[i] for i in by_name.values()})
        self.assertEqual(sorted(self.p.latest_analysis("W3")),sorted(i for i in by_name.values() if wafer_of[i]=="W3"))
        from dashboard.components.sqlite_input import analysed_per_batch_wafer,condition_accumulation
        analysed=analysed_per_batch_wafer(self.p)
        self.assertEqual({w:analysed[(batch,w)] for w in ("W2","W3","W5")},{"W2":2,"W3":2,"W5":1})
        table=condition_accumulation(self.p.batch_wafers(),analysed)
        mine=table[table["Wafer"].isin(["W2","W3","W5"])].values.tolist()
        self.assertEqual(mine,[["W2","50 mJ/cm²","220",1,2,2],["W3","60 mJ/cm²","—",1,2,2],["W5","70 mJ/cm²","120",1,1,1]])
        # Batch list: one row, Wafer list and Batch total.
        row=self.p.batch_record(batch)
        self.assertEqual((row["wafer_list"],row["wafer_count"],int(row["image_count"])),("W2, W3, W5",3,5))
        # Display name: no Wafer prefix for a multi-Wafer Batch (Wafers are in their own column);
        # single-Wafer names unchanged. Display-only: the stored notes / Batch ID are untouched.
        self.assertEqual((row["automatic_display_name"],row["display_name"]),("Batch 01","Batch 01"))
        self.assertTrue(all(n.startswith("W1 · Batch ") for n in self.p.batches().query("wafer_count==1")["display_name"]))
        self.assertIsNone(row["exposure_value"])                                            # No Batch-level 노광.
        # Process graph / Wafer Map inputs: per-image Wafer and process.
        images=am.collect_images(self.p)
        mine=images[images["batch_id"]==batch]
        self.assertEqual({(r.wafer_id,r.exposure_value,None if r.peb_temperature_c!=r.peb_temperature_c else r.peb_temperature_c)
                          for r in mine.itertuples()},{("W2",50.0,220.0),("W3",60.0,None),("W5",70.0,120.0)})
        self.assertEqual(set(self.p.images(wafer_id="W5",limit=100)["batch_id"]),{batch})
        self.assertEqual(int(self.p.wafers().set_index("wafer_id").loc["W5","image_count"]),1)
        # Batch detail: Wafer composition + Batch total.
        at=AppTest.from_file(APP,default_timeout=120)
        at.session_state["batch_detail_source"]="SQLite";at.session_state["selected_sqlite_batch_id"]=batch
        at.run();at.switch_page("pages/batch_detail.py").run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        composition=next(d.value for d in at.dataframe if "READY" in d.value.columns)
        self.assertEqual(composition[["Wafer ID","노광","PEB (s)","이미지 수","분석 결과 이미지 수"]].values.tolist(),
                         [["W2","50 mJ/cm²","220",2,2],["W3","60 mJ/cm²","—",2,2],["W5","70 mJ/cm²","120",1,1],["Batch 합계","Wafer별","Wafer별",5,5]])
        self.assertIn(">W2, W3, W5<"," ".join(m.value for m in at.main.markdown))
        self.assertIn(">Batch 01<"," ".join(m.value for m in at.main.markdown))
        # Wafer detail for W3 only (other Wafers of the same Batch are not counted).
        at=AppTest.from_file(APP,default_timeout=120)
        at.session_state["selected_sqlite_wafer"]={"wafer_id":"W3","exposure":"60 mJ/cm²","peb":"—"}
        at.run();at.switch_page("pages/wafer_detail.py").run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        values=" ".join(m.value for m in at.main.markdown)
        for text in (">W3<",">1개<",">2장<"):self.assertIn(text,values)
        self.assertTrue(any(t.value=="2 / Batch 전체 5" for t in at.main.markdown) or "2 / Batch 전체 5" in values)
        # I&A Wafer Map for W5: only W5 images and its own condition.
        at=AppTest.from_file(APP,default_timeout=120)
        at.session_state["_wafer_selected_SQLite"]="W5"
        at.run();at.switch_page("pages/inspection_analysis.py").run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        self.assertTrue(any("선택 범위: W5 / 노광 70 mJ/cm² / 120 s / 전체 Batch 1개" in c.value for c in at.caption))

    # ---- 11: single-Wafer Batches keep working on v6 (old metadata shape and v6 shape) ------------
    def test_08_single_wafer_paths_on_v6(self):
        _,old=self.register(self.stage([self.png("w1_21_c.png",201)]),"wafer_batch",process_run={"exposure_value":60.0})
        self.assertEqual(self.q("SELECT wafer_id,process_run_id,peb_time_s FROM batch_wafers WHERE batch_id=?",old["batch_id"]),
                         [("W1","TEST_PROCESS",None)])
        for mode in ("repeatability",):
            preview=self.p.preview(self.stage([self.png("w1_22_c.png",202)]),self.multi_metadata({"W1":(60,None)},upload_mode=mode))
            self.assertTrue(any("Wafer Batch 모드에서만" in i["message"] for i in preview["batch_issues"]))
        with closing(connect_database(self.f.db,readonly=True)) as c:
            validate_v6(c)


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed")
class V5StaysSingleWaferTests(Base):
    """v5 DB: the multi-Wafer shape is refused and the UI keeps the single common Wafer input."""
    def setUp(self):
        super().setUp();migrate_database_v5(self.f.db)
        for p in (patch.dict(os.environ,{"IDA_DB_PATH":str(self.f.db),"IDA_STORAGE_ROOT":str(self.f.sources),
                      "IDA_STAGING_ROOT":str(self.f.root/"staging")}),
                  patch("streamlit.file_uploader",side_effect=lambda *a,**k:[] if k.get("disabled") else
                        [SimpleNamespace(name="w1_21_c.png",getvalue=lambda:png_bytes(pixel=77))])):
            p.start();self.addCleanup(p.stop)

    def test_09_v5_refuses_multi_and_ui_unchanged(self):
        metadata={"batch_id":"BATCH-V5","wafer_id":"W1","upload_mode":"wafer_batch","dataset_role":"development",
                  "batch_status":"registered","wafers":{"W1":{"process_run":{"exposure_value":60.0}}}}
        preview=self.p.preview(self.stage([self.png("w1_23_c.png",203)]),metadata)
        self.assertFalse(preview["registration_ready"])
        self.assertTrue(any("Schema v6" in i["message"] for i in preview["batch_issues"]))
        at=AppTest.from_file(APP,default_timeout=120).run();at.switch_page("pages/data_control.py").run()
        self.assertFalse(at.exception)
        self.assertIn("Batch Wafer ID",[t.label for t in at.text_input])
        self.assertIn("노광 조건",[n.label for n in at.number_input])


@unittest.skipUnless(DEFAULT_DB.is_file(),"operational DB needed for a backup copy")
class OperationalCopyV6Tests(unittest.TestCase):
    """1/2/21/22: v6 on a copy keeps every v5 row and every Dashboard query result."""
    @classmethod
    def setUpClass(cls):
        cls.op=op_sha()
        cls.tmp=Path(tempfile.mkdtemp(prefix="v6_copy_"))
        cls.v5=cls.tmp/"v5.sqlite3";cls.v6=cls.tmp/"v6.sqlite3"
        with closing(sqlite3.connect(DEFAULT_DB.resolve().as_uri()+"?mode=ro",uri=True)) as src,closing(sqlite3.connect(cls.v5)) as dst:
            src.backup(dst)
        shutil.copyfile(cls.v5,cls.v6)
        cls.result=migrate_database_v6(cls.v6)
        cls.second=migrate_database_v6(cls.v6)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp,ignore_errors=True)
        assert op_sha()==cls.op,"operational DB changed"

    def provider(self,path):
        return SQLiteProvider(path,self.tmp/"raw",self.tmp/"staging")

    def test_10_counts_preserved_and_idempotent(self):
        self.assertEqual((self.result["migrated"],self.second),(True,{"version":6,"migrated":False}))
        tables=("upload_batches","images","analysis_batch_runs","analysis_image_runs","defect_detections","metrology_measurements","wafers","process_runs")
        counts=[{t:c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
                for c in (sqlite3.connect(self.v5),sqlite3.connect(self.v6))]
        self.assertEqual(counts[0],counts[1])
        self.assertEqual(self.result["batch_wafers"],counts[0]["upload_batches"])

    def test_11_dashboard_queries_identical_on_v5_and_v6(self):
        from dashboard.components.sqlite_input import analysed_per_batch_wafer,condition_accumulation
        old,new=self.provider(self.v5),self.provider(self.v6)
        cols=["batch_id","wafer_id","display_name","exposure_value","exposure_unit","peb_temperature_c","image_count","wafer_list"]
        a,b=old.batches()[cols],new.batches()[cols]
        self.assertTrue(a.astype(str).equals(b.astype(str)))
        self.assertTrue(condition_accumulation(old.batch_wafers(),analysed_per_batch_wafer(old)).equals(
                        condition_accumulation(new.batch_wafers(),analysed_per_batch_wafer(new))))
        img=["image_id","batch_id","wafer_id","die_code","position_code","exposure_value","peb_temperature_c","exposure_source"]
        self.assertTrue(old.images(limit=100)[img].astype(str).equals(new.images(limit=100)[img].astype(str)))
        self.assertTrue(old.wafers().equals(new.wafers()))
        self.assertEqual(old.latest_analysis().keys(),new.latest_analysis().keys())
        self.assertEqual(old.system_status()["analysed_images"],new.system_status()["analysed_images"])


if __name__=="__main__":
    unittest.main()
