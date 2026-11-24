"""TASK 18-B-8: final lab-use UI. Temporary DBs only; the operational DB bytes are asserted unchanged."""
from pathlib import Path
import hashlib,os,unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from database.init_db import DEFAULT_DB
from dashboard.tests.test_upload_default_matching import Base

APP=str(Path(__file__).resolve().parents[1]/"app.py")


class FinalUIBase(Base):
    def setUp(self):
        super().setUp()
        self.op_sha=hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest() if DEFAULT_DB.is_file() else None
        self.out=self.f.root/"results";self.out.mkdir()
        self.env={"IDA_DB_PATH":str(self.f.db),"IDA_STORAGE_ROOT":str(self.f.sources),
                  "IDA_STAGING_ROOT":str(self.f.root/"staging"),"IDA_ANALYSIS_OUTPUT_ROOT":str(self.out)}
        env=patch.dict(os.environ,self.env);env.start();self.addCleanup(env.stop)
    def tearDown(self):
        if self.op_sha:self.assertEqual(hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest(),self.op_sha)
    def app(self,page=None):
        at=AppTest.from_file(APP,default_timeout=60).run()
        if page:at.switch_page(page).run()
        self.assertFalse(at.exception,[e.value for e in at.exception])
        return at


class MainHomeTests(FinalUIBase):
    def test_01_home_layout_texts(self):
        at=self.app()
        text=" ".join(m.value for m in at.markdown)
        for gone in ("데이터에서 인사이트","A / DATA WORKSPACE","B / ANALYSIS WORKSPACE"):
            self.assertNotIn(gone,text)
        self.assertIn("DATA WORKSPACE",text);self.assertIn("ANALYSIS WORKSPACE",text)
        self.assertIn("데이터 입력부터 Metadata, Calibration, Batch 관리까지.",text)
        self.assertIn("Wafer Map, 선폭·결함 분석 및 공정조건별 결과 확인.",text)
        labels=[b.label for b in at.button]
        self.assertIn("Data Control Center 열기",labels);self.assertIn("Inspection & Analysis 열기",labels)
        self.assertNotIn("실험 환경 및 조건 열기",labels)
        self.assertFalse(any("UI 예시" in c.value or "Mock" in c.value for c in at.main.caption))

    def test_02_kpis_come_from_sqlite(self):
        self.register(self.stage([self.png("w1_21_c.png",201),self.png("w1_22_c.png",202)]),"wafer_batch")
        status=self.p.system_status()
        at=self.app()
        metrics={m.label:m.value for m in at.metric}
        self.assertEqual(metrics["등록 Batch"],str(status["batches"]))
        self.assertEqual(metrics["등록 이미지"],str(status["images"]))
        self.assertEqual(metrics["분석 완료 이미지"],str(status["analysed_images"]))
        self.assertEqual(metrics["최근 분석 실행"],"실행 이력 없음")

    def test_03_db_failure_shows_error_not_zero(self):
        with patch.dict(os.environ,{"IDA_DB_PATH":str(self.f.root/"missing.sqlite3")}):
            at=self.app()
        self.assertTrue(any("SQLite 상태를 조회할 수 없습니다" in e.value for e in at.error))
        self.assertEqual(len(at.metric),0)


class MainHomeRefinementTests(FinalUIBase):
    """TASK 18-B-9: Main Home only. Temporary DBs; read-only queries."""
    def sql(self,query):
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(self.f.db)) as c:return c.execute(query).fetchall()

    def test_13_header_caption_only_removed_on_home(self):
        at=self.app()
        self.assertFalse(any("등록 완료 ≠ 분석 완료" in c.value for c in at.main.caption))
        self.assertFalse(any("SQLite Schema v" in c.value for c in at.main.caption))
        self.assertTrue(any("IDA Inspection Platform" in m.value and "Main Home" in m.value for m in at.markdown))
        self.assertTrue(any(m.value=="**이번 세션 알림**" for m in at.markdown))              # Bell popover kept.
        at.switch_page("pages/inspection_analysis.py").run()                                  # Other pages unchanged.
        self.assertTrue(any("AI 잠정 결과" in c.value for c in at.main.caption))  # (DCC caption removed in 18-B-10A.)

    def test_14_one_line_titles_and_buttons(self):
        at=self.app()
        titles=[m.value for m in at.markdown if 'class="ws-title"' in m.value]
        self.assertEqual(len(titles),2)
        for area,feature in (("DATA WORKSPACE","Data Control Center"),("ANALYSIS WORKSPACE","Inspection &amp; Analysis")):
            self.assertTrue(any(area in t and '<span class="ws-sep">|</span>' in t and feature in t for t in titles))
        self.assertFalse(any(h.value in ("Data Control Center","Inspection & Analysis") for h in at.subheader))
        next(b for b in at.button if b.label=="Data Control Center 열기").click().run()
        self.assertEqual(at.session_state["active_page_key"],"data_control")
        at=self.app()
        next(b for b in at.button if b.label=="Inspection & Analysis 열기").click().run()
        self.assertEqual(at.session_state["active_page_key"],"inspection_analysis")

    def test_15_five_kpis_one_row_quick_access_compact_no_recent_work(self):
        """TASK 18-B-9.1: selective rollback (KPIs, Quick Access, recent work)."""
        status=self.p.system_status()
        at=self.app()
        metrics={m.label:m.value for m in at.metric}
        self.assertEqual(list(metrics),["등록 Batch","등록 이미지","분석 완료 이미지","검토 필요 이미지","최근 분석 실행"])
        self.assertEqual((metrics["등록 Batch"],metrics["등록 이미지"],metrics["분석 완료 이미지"],metrics["검토 필요 이미지"]),
                         tuple(str(status[k]) for k in ("batches","images","analysed_images","review_images")))
        self.assertEqual([h.value for h in at.main.subheader],["시스템 상태"])                 # No "최근 작업".
        self.assertFalse(any(m.value in ("**최근 등록**","**최근 분석**") for m in at.main.markdown))
        self.assertTrue(any(c.value=="QUICK ACCESS" for c in at.main.caption))
        from dashboard.navigation import PAGE_SPECS
        links=[l.proto.page for l in at.main.get("page_link")]
        self.assertEqual(links,[PAGE_SPECS[k][2] for k in ("dataset","model_performance","settings")])
        source=(Path(APP).parent/"pages/main_home.py").read_text(encoding="utf-8")
        self.assertIn('key="home_center"',source);self.assertNotIn("home_quick",source)


class DataControlTests(FinalUIBase):
    def test_04_title_no_source_switch_and_batch_columns(self):
        _,result=self.register(self.stage([self.png("w1_21_c.png",201)]),"wafer_batch",peb_temperature_c=95.0)
        at=self.app("pages/data_control.py")
        self.assertEqual(at.title[0].value,"Data Control Center")
        self.assertTrue(any("등록·Batch 조회·분석 대상 미리보기·축적 현황을 확인합니다." in c.value for c in at.caption))
        self.assertFalse(any("A / DATA CONTROL" in m.value for m in at.main.markdown))  # Sidebar nav label stays.
        self.assertFalse(any(r.key=="data_control_source" for r in at.radio))       # No Mock/SQLite switch.
        self.assertFalse(any(b.label in ("확대","복원") for b in at.button))          # Icon-only zoom.
        self.assertEqual(at.button(key="dcc_expand_batches").label,":material/open_in_full:")
        headers=[m.value for m in at.markdown if m.value in ("**Batch**","**Wafer**","**노광**","**PEB**","**이미지**","**분석**","**상태**")]
        self.assertEqual(headers,["**Batch**","**Wafer**","**노광**","**PEB**","**이미지**","**분석**","**상태**"])
        self.assertTrue(any(m.value=="95" for m in at.markdown))
        self.assertFalse(any(c.value.startswith("Wafer 요약") for c in at.caption))   # TASK18-B-10A: replaced by the table.
        for letter,group in (("A","실행 설정"),("B","분석 대상"),("C","실행"),("D","진행 상태"),("E","완료 결과")):  # TASK18-B-11
            self.assertTrue(any(f'{letter}<span>|</span>{group}</div>' in m.value for m in at.markdown),group)

    def test_05_accumulation_table_matches_sqlite(self):
        self.register(self.stage([self.png("w1_21_c.png",201),self.png("w1_22_c.png",202)]),"wafer_batch",peb_temperature_c=95.0)
        at=self.app("pages/data_control.py")
        table=next(d.value for d in at.dataframe if "분석 결과 이미지 수" in d.value.columns and "Wafer" in d.value.columns)
        self.assertEqual(list(table.columns),["Wafer","노광","PEB (s)","Batch 수","등록 이미지 수","분석 결과 이미지 수"])
        batches=self.p.batches()
        self.assertEqual(int(table["등록 이미지 수"].sum()),int(batches["image_count"].sum()))
        self.assertEqual(int(table["Batch 수"].sum()),len(batches))
        self.assertEqual(int(table["분석 결과 이미지 수"].sum()),len(self.p.latest_analysis()))
        self.assertIn("95",table["PEB (s)"].tolist())


class DataControlTopCardsTests(FinalUIBase):
    """TASK 18-B-10A: DCC top two cards (temporary DB; uploads via the patched uploader only)."""
    def test_20_header_titles_and_no_idle_notice(self):
        at=self.app("pages/data_control.py")
        self.assertFalse(any("등록 완료 ≠ 분석 완료" in c.value for c in at.main.caption))     # DCC header notice removed.
        self.assertTrue(any(c.value=="|  업로드 → Metadata 확인 → 승인 후 SQLite 등록" for c in at.caption))
        self.assertTrue(any(c.value=="|  Batch를 선택하면 상세 페이지로 이동합니다." for c in at.caption))
        self.assertFalse(any("이미지를 선택하세요" in c.value for c in at.caption))
        self.assertTrue(at.button(key="sqlite_register").disabled)                              # Still blocked.
        self.assertTrue(any(p.proto.popover.label=="Upload / Drop" for p in at.get("popover")))
        at.switch_page("pages/inspection_analysis.py").run()                                     # Other page unchanged.
        self.assertTrue(any("AI 잠정 결과" in c.value for c in at.main.caption))

    def test_21_batch_pager_wafer_table_and_image_lookup_order(self):
        for i in range(4):
            self.register(self.stage([self.png(f"w1_2{i+1}_c.png",220+i)]),"wafer_batch",peb_temperature_c=95.0)
        at=self.app("pages/data_control.py")
        rows=lambda:[b.key.removeprefix("sqlite_batch_") for b in at.button if b.key and b.key.startswith("sqlite_batch_")]
        expected=self.p.batches().sort_values("created_at",ascending=False,kind="stable")["batch_id"].tolist()
        self.assertEqual(rows(),expected)                     # TASK18-B-10A.1: all rows in the scroll area, newest first.
        self.assertFalse(any(b.key in ("batch_table_next","batch_table_prev") for b in at.button if b.key))
        from dashboard.components.sqlite_input import condition_accumulation,analysed_per_batch_wafer
        table=next(d.value for d in at.dataframe if "Wafer ID" in d.value.columns)
        reference=condition_accumulation(self.p.batch_wafers(),analysed_per_batch_wafer(self.p)).rename(columns={"Wafer":"Wafer ID"})
        self.assertEqual(table.astype(str).values.tolist(),reference.astype(str).values.tolist())
        self.assertEqual(list(table.columns),["Wafer ID","노광","PEB (s)","Batch 수","등록 이미지 수","분석 결과 이미지 수"])
        self.assertIn("95",table["PEB (s)"].tolist())
        source=(Path(APP).parent/"components/sqlite_input.py").read_text(encoding="utf-8")
        body=source[source.index("def render_batches"):]
        table_at=body.index("visible_rows=3");wafer_at=body.index('key="dcc_wafer_conditions"')
        lookup_at=body.index('st.expander("전체 등록 이미지 조회")',wafer_at)
        self.assertLess(table_at,wafer_at);self.assertLess(wafer_at,lookup_at)


class ReviewTabTests(unittest.TestCase):
    """TASK 18-B-10A: filename review tab layout (temporary DB, synthetic uploads)."""
    from dashboard.tests.test_sqlite_input import SQLiteInputTests as _F
    setUp=_F.setUp;uploaded=_F.uploaded;app=_F.app;confirm_process=_F.confirm_process

    def test_22_review_tooltip_counts_and_moved_expanders(self):
        self.uploads=[self.uploaded("unknown.png",80),self.uploaded("w1_12_c.png",40)]
        at=self.app();self.confirm_process(at)
        preview=at.session_state["sqlite_preview"]
        radio=at.radio(key="sqlite_review_"+preview["batch_metadata"]["batch_id"])
        self.assertIn("공통 Metadata에서 보완",radio.proto.help)
        from dashboard.components.sqlite_input import review_help
        from types import SimpleNamespace
        from dashboard.sqlite_provider import runtime_provider
        version=runtime_provider().schema_info()["version"]
        self.assertIn("이 Batch에만 저장" if version>=5 else "기존 Wafer의 공정 조건과 일치",radio.proto.help)
        v5=review_help(SimpleNamespace(schema_info=lambda:{"version":5}));v4=review_help(SimpleNamespace(schema_info=lambda:{"version":4}))
        self.assertIn("이 Batch에만 저장",v5);self.assertNotIn("일치해야",v5);self.assertIn("일치해야",v4)
        self.assertFalse(any("공통 공정/촬영 조건 누락은" in c.value for c in at.caption))
        self.assertFalse(any("기존 Wafer의 공정 조건과도 일치" in c.value for c in at.caption))
        counts=next(c for c in at.caption if c.value.startswith("전체 2장 · 검증 완료"))
        self.assertIn("수정 필요 1장",counts.value)
        self.assertTrue(any(c.label=="전체 이미지 보기" for c in at.checkbox))
        labels=[e.label for e in at.expander]
        self.assertIn("전체 파일명 및 입력 출처",labels)
        source=Path(__file__).resolve().parents[1].joinpath("components/sqlite_input.py").read_text(encoding="utf-8")
        body=source[source.index("def render_review"):source.index("def render_overview")]
        self.assertLess(body.index('"선택 이미지 미리보기"'),body.index('"검증 기술 상세"'))
        self.assertLess(body.index('"검증 기술 상세"'),body.index('"전체 파일명 및 입력 출처"'))
        before=dict(at.session_state["sqlite_uploads"]["common_metadata"])
        at.checkbox(key="sqlite_review_all_"+preview["batch_metadata"]["batch_id"]).check().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["sqlite_uploads"]["common_metadata"],before)            # Draft kept.


class ProcessInputTests(unittest.TestCase):
    """TASK 18-B-10A.1: direct exposure entry, no unit widget, PEB time (s). Temporary DB only."""
    from dashboard.tests.test_sqlite_input import SQLiteInputTests as _F
    setUp=_F.setUp;uploaded=_F.uploaded;app=_F.app

    def test_23_widgets_and_blocking(self):
        self.uploads=[self.uploaded("w1_12_c.png",40)]
        at=self.app()
        self.assertFalse(any(s.label=="노광 조건" for s in at.selectbox))
        exposure=next(n for n in at.number_input if n.label=="노광 조건")
        self.assertIsNone(exposure.value)                                                     # Not entered != 0.
        self.assertFalse(any("노광 단위" in t.label for t in at.text_input))
        self.assertTrue(any(n.label=="PEB Time (s, 선택)" for n in at.number_input))
        self.assertFalse(any("PEB Temperature" in n.label for n in at.number_input))
        texts=" ".join(c.value for c in at.caption)
        self.assertNotIn("필수: Wafer ID · 신규 공정의 실제 노광 조건",texts)
        self.assertNotIn("공통 Metadata에서 이 Batch의 실제 노광 조건을 선택하세요",texts)
        self.assertTrue(at.button(key="sqlite_register").disabled)                            # Still blocked.
        self.assertIn("노광 조건 미입력",texts)                                                 # Real input error shown.
        exposure.set_value(60).run()
        self.assertFalse(at.button(key="sqlite_register").disabled)



class ProcessRoundTripTests(FinalUIBase):
    """TASK 18-B-10A.1: UI registration on a temporary v5 DB, then re-read from SQLite."""
    def setUp(self):
        super().setUp()
        from database.migrate_v5 import migrate_database_v5
        self.assertEqual(migrate_database_v5(self.f.db)["version"],5)
        self.uploads=[]
        uploader=patch("streamlit.file_uploader",side_effect=lambda *a,**kw:[] if kw.get("disabled") else self.uploads)
        uploader.start();self.addCleanup(uploader.stop)

    def test_24_exposure_and_peb_round_trip(self):
        from types import SimpleNamespace
        from database.tests.test_image_registry import png_bytes
        from contextlib import closing
        import sqlite3
        cases=[("w2_12_c.png",50.0,220.0),("w3_12_c.png",60.0,None),("w4_12_c.png",70.0,120.0),
               ("w5_12_c.png",80.0,None),("w6_12_c.png",62.5,220.0),
               ("w1_12_c.png",75.0,220.0)]        # Existing W1 (fixture exposure 60): Batch-only record in v5.
        with closing(sqlite3.connect(self.f.db)) as c:
            w1_before=c.execute("SELECT r.exposure_value,r.exposure_unit FROM wafers w JOIN process_runs r USING(process_run_id) "
                                "WHERE w.wafer_id='W1'").fetchone()
        for index,(name,exposure,peb) in enumerate(cases):
            content=png_bytes(pixel=150+index)
            self.uploads=[SimpleNamespace(name=name,getvalue=lambda content=content:content)]
            at=self.app("pages/data_control.py")
            next(n for n in at.number_input if n.label=="노광 조건").set_value(exposure).run()
            if peb is not None:
                next(n for n in at.number_input if n.label=="PEB Time (s, 선택)").set_value(peb).run()
            self.assertFalse(at.button(key="sqlite_register").disabled,[c.value for c in at.caption][-5:])
            at.button(key="sqlite_register").click().run();self.assertFalse(at.exception)
            record=self.p.batch_record(at.session_state["sqlite_last_result"]["batch_id"])
            self.assertEqual(record["exposure_value"],exposure,name)
            self.assertIsNone(record["exposure_unit"])                                            # No unit invented.
            stored=record["peb_temperature_c"]
            self.assertTrue((stored is None or stored!=stored) if peb is None else stored==peb,name)
        self.assertEqual(record["exposure_source"],"batch")
        with closing(sqlite3.connect(self.f.db)) as c:
            self.assertEqual(c.execute("SELECT r.exposure_value,r.exposure_unit FROM wafers w JOIN process_runs r USING(process_run_id) "
                                       "WHERE w.wafer_id='W1'").fetchone(),w1_before)                # Wafer record untouched.
        from dashboard.components.inspection_wafer import peb_label
        self.assertEqual((peb_label(220.0),peb_label(120.0),peb_label(None)),("220 s","120 s","PEB 미입력"))


class InspectionTabsTests(FinalUIBase):
    def test_06_four_tabs_title_and_no_demo_selector(self):
        self.register(self.stage([self.png("w1_21_c.png",201)]),"wafer_batch")
        at=self.app("pages/inspection_analysis.py")
        self.assertEqual(at.title[0].value,"Inspection & Analysis")
        # TASK18-B-13A: both top notices removed (subtitle and header line).
        self.assertFalse(any("공정 조건은 선택한 데이터 소스의 Metadata입니다." in c.value for c in at.caption))
        self.assertFalse(any("Wafer Map·선택 이미지·공정 조건" in c.value for c in at.caption))
        heads=" | ".join(m.value for m in at.markdown)
        for title in ("조건별 평균 Line Width","설계 대비 오차율","이미지별 평균 분포 (Line / Gap)"):
            self.assertEqual(heads.count('<span class="ia-graph-title">'+title+'<span class="ia-graph-help"'),2,title)  # 노광·PEB.
        # TASK18-B-13A.1: Fail trend first among the four detail expanders (each axis tab); mJ/cm² display unit.
        labels=[e.label for e in at.expander if e.label in ("Wafer별 Die/Row Fail 경향","조건별 요약","조건별·Wafer별 상세 결과","대상 이미지 Metadata")]
        self.assertEqual(labels,["Wafer별 Die/Row Fail 경향","조건별 요약","조건별·Wafer별 상세 결과","대상 이미지 Metadata"]*2)
        self.assertTrue(all(not e.proto.expanded for e in at.expander if e.label in labels))
        self.assertIn("X: 노광 조건 (mJ/cm²)",heads);self.assertIn("X: PEB 조건 (s)",heads)
        top=[t.label for t in at.tabs if t.label in ("Wafer / Image","선폭 / 오차율","Repeatability","Lighting")]
        self.assertEqual(top,["Wafer / Image","선폭 / 오차율","Repeatability","Lighting"])
        self.assertFalse(any(s.key=="inspection_batch" for s in at.selectbox))
        self.assertFalse(any("B / INSPECTION" in m.value for m in at.main.markdown))

    def test_07_repeatability_and_lighting_tabs_use_sqlite_groups(self):
        inputs=self.stage([self.png("w1_11_c_r1.png",130),self.png("w1_11_c_r2.png",131)])
        for index,item in enumerate(inputs,1):item["repeat_index"]=index
        self.register(inputs,"repeatability")
        light=self.stage([self.png("w1_11_c_lg.png",140)])
        light[0]["acquisition_condition"]={"condition_id":"COND-LIGHT-40X","magnification":40.0,"illumination_setting":"lg","auto_exposure":0}
        self.register(light,"lighting")
        at=self.app("pages/inspection_analysis.py")
        repeat=next(d.value for d in at.dataframe if "반복 번호" in d.value.columns)
        self.assertEqual(repeat.iloc[0][["Wafer","Die","Site","반복 번호","이미지 수"]].tolist(),["W1","11","c","1, 2",2])
        self.assertEqual(repeat.iloc[0]["입력 상태"],"REVIEW_REQUIRED 2")              # Policy kept, not relaxed.
        lighting=next(d.value for d in at.dataframe if "조명 설정" in d.value.columns)
        row=lighting.iloc[0]
        self.assertEqual((row["조명 설정"],row["배율"],row["Auto Exposure"],row["입력 상태"]),("lg","40.0","OFF","BLOCKED"))
        self.assertNotEqual(row["제한 사유"],"-")                                       # Restriction reason shown.
        self.assertTrue(any("자동 실행 대상에서 제외" in c.value for c in at.caption))


class RegistrationNotificationTests(unittest.TestCase):
    """Toast + bell for upload / duplicate / registration / failure on a temporary DB."""
    from dashboard.tests.test_sqlite_input import SQLiteInputTests as _F
    setUp=_F.setUp;uploaded=_F.uploaded;app=_F.app;confirm_process=_F.confirm_process

    def kinds(self,at):
        return [e["kind"] for e in at.session_state["_notifications"]] if "_notifications" in at.session_state else []

    def test_08_upload_duplicate_and_registration_events_once(self):
        self.uploads=[self.uploaded("w1_12_c.png",40),self.uploaded("w1_13_c.png",40)]  # Same bytes: duplicate.
        at=self.app();self.confirm_process(at)
        self.assertIn("업로드 완료",self.kinds(at));self.assertIn("중복 감지",self.kinds(at))
        self.assertNotIn("SQLite 등록 완료",self.kinds(at))                        # Never before a commit.
        before=len(self.kinds(at));at.run();at.run()
        self.assertEqual(len(self.kinds(at)),before)                                 # No repeat per rerun.
        self.uploads=[self.uploaded("w1_14_c.png",60)]
        at=self.app();self.confirm_process(at)
        at.button(key="sqlite_register").click().run();self.assertFalse(at.exception)
        self.assertIn("SQLite 등록 완료",self.kinds(at))
        event=next(e for e in at.session_state["_notifications"] if e["kind"]=="SQLite 등록 완료")
        self.assertEqual((event["count"],event["level"]),(1,"success"))
        self.assertTrue(any("SQLite 등록 완료" in t.value for t in at.toast))
        self.assertTrue(at.button(key="notifications_read_all"))

    def test_09_registration_failure_is_not_reported_as_success(self):
        import sqlite3
        from dashboard.sqlite_provider import SQLiteProvider
        self.uploads=[self.uploaded("w1_15_c.png",70)]
        at=self.app();self.confirm_process(at)
        with patch.object(SQLiteProvider,"register_selection",side_effect=sqlite3.OperationalError("disk I/O")):
            at.button(key="sqlite_register").click().run()
        self.assertIn("SQLite 저장 실패",self.kinds(at));self.assertNotIn("SQLite 등록 완료",self.kinds(at))


class AnalysisNotificationTests(FinalUIBase):
    """Analysis events via the fake runner (real BatchRunner/Repository) on a temporary DB."""
    def setUp(self):
        super().setUp()
        import sys
        import dashboard.components.analysis_management as am
        from dashboard.tests.test_multi_analysis import FAKE
        self.am=am
        _,self.a=self.register(self.stage([self.png("w1_21_c.png",201),self.png("w1_13_c.png",202)]),"wafer_batch")
        _,self.b=self.register(self.stage([self.png("w1_14_c.png",203)]),"wafer_batch")
        for p in (patch.dict(os.environ,{"IDA_ANALYSIS_PYTHON":sys.executable,"FAKE_RUNNER_CALLS":str(self.f.root/"calls.txt")}),
                  patch.object(am,"_analysis_command",lambda python,request,output:
                      [str(python),str(FAKE),"--request",str(request),"--output-root",str(output)]),
                  patch("streamlit.file_uploader",return_value=[])):
            p.start();self.addCleanup(p.stop)
        os.environ.pop("FAKE_RUNNER_MODE",None);am._ACTIVE.clear()
    def run_scope(self,scope,target,extras=()):
        at=self.app("pages/data_control.py")
        at.radio(key="dcc_analysis_scope").set_value(scope).run()
        at.selectbox(key="dcc_analysis_target_"+scope.lower()).select(target).run()
        if extras:at.multiselect(key="dcc_analysis_extra_images").set_value(list(extras)).run()
        at.button(key="dcc_analysis_run").click().run();self.assertFalse(at.exception)
        return at
    def events(self,at):
        return [(e["kind"],e["status"],e["count"]) for e in at.session_state["_notifications"]]

    def test_10_start_and_complete_after_save(self):
        at=self.run_scope("Batch",self.a["batch_id"])
        last=at.session_state["dcc_analysis_last_run"]
        self.assertEqual(last["db_save"]["status"],"SAVED")
        kinds=[k for k,_,_ in self.events(at)]
        self.assertIn("분석 시작",kinds);self.assertIn("분석 완료",kinds);self.assertNotIn("SQLite 저장 실패",kinds)
        self.assertTrue(any("분석 완료" in t.value for t in at.toast))
        count=len(kinds);at.run();at.run()
        self.assertEqual(len(at.session_state["_notifications"]),count)
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(self.f.db)) as c:
            stored=c.execute("SELECT execution_status FROM analysis_batch_runs ORDER BY julianday(started_at) DESC,"
                             "batch_run_id DESC LIMIT 1").fetchone()[0]
        at.switch_page("pages/main_home.py").run();self.assertFalse(at.exception)
        self.assertEqual({m.label:m.value for m in at.metric}["최근 분석 실행"],stored)  # Main Home KPI = stored run.

    def test_11_partial_failure_event(self):
        normal=self.a["image_ids"][0];failing=self.b["image_ids"][0]
        at=self.run_scope("Image",normal,[failing])
        kinds=[k for k,_,_ in self.events(at)]
        self.assertIn("부분 실패",kinds);self.assertNotIn("SQLite 저장 실패",kinds)   # Failed Batch NOT_SAVED by design.

    def test_12_save_failure_event(self):
        with patch.object(self.am,"save_analysis_result",return_value={"status":"FAILED","message":"x"}):
            at=self.run_scope("Batch",self.a["batch_id"])
        self.assertIn("SQLite 저장 실패",[k for k,_,_ in self.events(at)])


if __name__=="__main__":
    unittest.main()
