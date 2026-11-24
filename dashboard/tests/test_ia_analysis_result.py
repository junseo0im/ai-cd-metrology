"""TASK 18-B-1: I&A shows the stored analysis of one image (read-only).

UI tests use a temporary Online-Backup copy of the operational DB (which holds the real
stored run) and read the official result root. Expected values come from the DB/JSON,
never from constants. The operational DB and result files are asserted unchanged.
"""
from contextlib import closing
from pathlib import Path
import hashlib,json,os,sqlite3,tempfile,unittest
from unittest.mock import patch

from database.init_db import DEFAULT_DB
from database.analysis_repository import AnalysisRepository,OPERATIONAL_RESULT_ROOT
import dashboard.components.ia_analysis_result as ia

ROOT=Path(__file__).resolve().parents[2]
IMAGE="IMG-57eca064ca764e37be03e6cd62c7f079"   # w5_15_c.png, stored run exists
NO_RESULT_DIE="11"                              # W5 Die 11 c: only Repeatability images, never analysed
PD_IMAGE="IMG-289cd4e2259f429ca5ce96da1d2afa87"  # W4 Die 18 rb: stored MODEL_PD_EXCLUDED run (TASK18-B-1.6)
TEST_ROOT=ROOT/"runtime_data/validation/task18b1/tmp"
try:
    import plotly  # noqa: F401
    HAS_PLOTLY=True
except ImportError:
    HAS_PLOTLY=False


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def result_files():return {p:sha(p) for p in OPERATIONAL_RESULT_ROOT.rglob("*") if p.is_file()}


def stored_rows():
    repository=AnalysisRepository(DEFAULT_DB,result_root=OPERATIONAL_RESULT_ROOT,
                                  image_root=ROOT/"runtime_data/raw",read_only=True)
    return repository,repository.image_history(IMAGE)


AVAILABLE=DEFAULT_DB.is_file() and OPERATIONAL_RESULT_ROOT.is_dir()


@unittest.skipUnless(AVAILABLE,"operational DB / official result root not present")
class ReadOnlyRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.db_sha=sha(DEFAULT_DB);self.files=result_files()
    def tearDown(self):
        self.assertEqual(sha(DEFAULT_DB),self.db_sha);self.assertEqual(result_files(),self.files)

    def test_01_operational_read_only_allows_reads_only(self):
        repository,history=stored_rows()
        # The 18-A-4 run is kept; later runs (e.g. the user's 69-image run) are newer history.
        self.assertIn("batch-80212943768b4e43b7fee5a4f797b032",[h["batch_run_id"] for h in history])
        started=[repository.get_batch_run(h["batch_run_id"])["started_at"] for h in history]
        self.assertEqual(started,sorted(started,reverse=True))  # Newest first.
        detail=repository.get_analysis_run(history[0]["analysis_run_id"])
        raw_disk=json.loads((OPERATIONAL_RESULT_ROOT/history[0]["result_path"]).read_text(encoding="utf-8"))
        self.assertEqual(detail["raw_result"],raw_disk)
        self.assertEqual((detail["outcome"],detail["is_adopted"]),("QUALITY_REVIEW_REQUIRED",0))
        with self.assertRaises(PermissionError):repository.save_batch(OPERATIONAL_RESULT_ROOT/"x.json")
        with self.assertRaises(PermissionError):repository._connection(readonly=False)
        with self.assertRaises(PermissionError):  # Official root is required for operational reads too.
            AnalysisRepository(DEFAULT_DB,result_root=ROOT/"runtime_data/validation",image_root=ROOT/"runtime_data/raw",read_only=True)
        with self.assertRaises(PermissionError):  # Default callers remain fully refused.
            AnalysisRepository(DEFAULT_DB,result_root=OPERATIONAL_RESULT_ROOT,image_root=ROOT/"runtime_data/raw")

    def test_02_view_values_come_from_db_and_json(self):
        from dashboard.sqlite_provider import SQLiteProvider
        provider=SQLiteProvider(DEFAULT_DB,ROOT/"runtime_data/raw",ROOT/"runtime_data/uploads_tmp")
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop("IDA_ANALYSIS_OUTPUT_ROOT",None)
            view=ia.load_image_analysis(provider,provider.image_record(IMAGE))
        _,history=stored_rows()
        self.assertEqual(view["state"],"OK");self.assertIsNone(view["detail_error"])
        self.assertEqual(view["row"],history[0])
        raw=view["detail"]["raw_result"]
        self.assertEqual(view["detail"]["measurements"],raw["metrology"]["measurements"])
        self.assertEqual(len(view["detail"]["detections"]),len(raw["defect"]["detections"]))
        for stage in ("defect","metrology"):
            self.assertEqual(sha(view["overlays"][stage]["path"]),history[0][stage+"_overlay_sha256"])
            self.assertIn(history[0]["batch_run_id"],view["overlays"][stage]["path"])


def result_table(at):
    """TASK18-B-14B.2: the '항목 | 결과' table as {항목: 결과} (empty when absent)."""
    table=next((d.value for d in at.dataframe if list(d.value.columns)==["항목","결과"]),None)
    return {} if table is None else dict(zip(table["항목"],table["결과"]))


@unittest.skipUnless(AVAILABLE and HAS_PLOTLY,"operational DB copy or plotly not available")
class IAPageTests(unittest.TestCase):
    def setUp(self):
        from database.migrate_v4 import refuse_operational  # noqa: F401
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=TEST_ROOT);self.addCleanup(temporary.cleanup)
        self.db=Path(temporary.name)/"ida_copy.sqlite3"
        with closing(sqlite3.connect(f"file:{DEFAULT_DB.as_posix()}?mode=ro",uri=True)) as s,closing(sqlite3.connect(self.db)) as d:
            s.backup(d)
        self.db_sha=sha(DEFAULT_DB);self.files=result_files();self.copy_sha=sha(self.db)
        env=patch.dict(os.environ,{"IDA_DB_PATH":str(self.db),"IDA_STORAGE_ROOT":str(ROOT/"runtime_data/raw"),
            "IDA_STAGING_ROOT":str(Path(temporary.name)/"staging"),"IDA_ANALYSIS_OUTPUT_ROOT":str(OPERATIONAL_RESULT_ROOT)})
        env.start();self.addCleanup(env.stop)
        _,self.history=stored_rows()
        self.row=self.history[0]
    def tearDown(self):
        self.assertEqual(sha(DEFAULT_DB),self.db_sha);self.assertEqual(result_files(),self.files)

    def app(self,die="15",position="c",wafer="W5"):
        from streamlit.testing.v1 import AppTest
        from dashboard.components.inspection_wafer import select_die
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=60).run()
        at.switch_page("pages/inspection_analysis.py").run()
        at.run()  # TASK18-B-6: SQLite is the fixed Wafer Map source (no selector).
        at.selectbox(key="ia_wafer_wafer_SQLite").select(wafer).run()
        select_die(at.session_state,"ia_wafer_map_SQLite_"+wafer+"_all",die,popover=True);at.run()
        next(b for b in at.get("button_group") if b.key.startswith("ia_sqlite_position_")).set_value(position).run()
        self.assertFalse(at.exception)
        return at
    def captions(self,at):return [c.value for c in at.caption]
    def tiles(self,at):return result_table(at)

    def test_03_stored_result_shown_with_states_and_values(self):
        at=self.app()
        self.assertEqual(at.session_state["ia_sqlite_selected_image_id"],IMAGE)
        # TASK18-B-15: '선택 Die 분석 요약' (Die lt/c/rb) replaces the single-image table; no 공식 채택 row.
        # TASK18-B-16: '선택 이미지 분석 요약 (위치)' + '선택 Die 분석 요약 (lt/c/rb)' with identical rows.
        heads=[s.value for s in at.subheader]
        self.assertIn("선택 Die 분석 요약 (lt/c/rb)",heads);self.assertIn("선택 이미지 분석 요약 (c)",heads)
        tables=[d.value for d in at.dataframe if list(d.value.columns)==["항목","결과"]]
        self.assertEqual(len(tables),2);self.assertEqual(tables[0]["항목"].tolist(),tables[1]["항목"].tolist())
        text=" | ".join(self.captions(at));tiles=self.tiles(at)
        self.assertEqual(list(tiles),["분석 상태","Line Width 평균 (측정 수)","Gap 평균 (측정 수)","선폭 오차율 (Signed / Abs)","결함 종류","결함 개수"])
        r=self.row   # Image table = this image only: stored counts of the latest run.
        self.assertIn(f"({r['line_measurement_count']}개)",tiles["Line Width 평균 (측정 수)"])
        self.assertNotIn("공식 채택",tiles)
        self.assertIn("PN ",tiles["Line Width 평균 (측정 수)"])
        self.assertNotIn("결함 0건은 PASS가 아닙니다",text)                              # Bottom notice removed.
        self.assertTrue(any(e.label.startswith("Line 판정 (PN/PT) · c") for e in at.expander))
        judgments=next(d.value for d in at.dataframe if "near_threshold" in d.value.columns)
        detail=AnalysisRepository(self.db,result_root=OPERATIONAL_RESULT_ROOT,image_root=ROOT/"runtime_data/raw").get_analysis_run(self.row["analysis_run_id"])
        self.assertEqual(judgments["Line"].tolist(),[j["line_id"] for j in detail["judgments"]])
        self.assertEqual(judgments.loc[judgments["near_threshold"]==True,"Line"].tolist(),
                         [j["line_id"] for j in detail["judgments"] if j["near_threshold"]])
        measurements=next(d.value for d in at.dataframe if "폭 (µm)" in d.value.columns)
        self.assertEqual(measurements["폭 (µm)"].tolist(),[m["width_um"] for m in detail["measurements"]])
        self.assertEqual(sha(self.db),self.copy_sha)  # Page reads never write.

    def test_04_overlay_toggles_show_stored_files_of_this_image(self):
        import streamlit
        real=streamlit.image;seen=[]
        at=self.app()
        with patch("streamlit.image",side_effect=lambda v,*a,**k:(seen.append(str(v)),real(v,*a,**k))[1]):
            at.toggle(key="ia_metrology_overlay").set_value(True)
            at.toggle(key="ia_defect_overlay").set_value(True).run()
        self.assertFalse(at.exception)
        names=[Path(v).name for v in seen]
        # TASK18-B-15: 선폭 Overlay | 결함 Overlay on top, then the original of the same image.
        self.assertEqual(names,["metrology_overlay.png","defect_overlay.png","w5_15_c.png"])
        for value in seen[:2]:self.assertIn(self.row["batch_run_id"],value)
        self.assertTrue(any(c.startswith("결함 검출 ") for c in self.captions(at)))
        seen.clear()
        with patch("streamlit.image",side_effect=lambda v,*a,**k:(seen.append(str(v)),real(v,*a,**k))[1]):
            at.toggle(key="ia_defect_overlay").set_value(False).run()
        self.assertEqual([Path(v).name for v in seen],["metrology_overlay.png","w5_15_c.png"])
        seen.clear()
        with patch("streamlit.image",side_effect=lambda v,*a,**k:(seen.append(str(v)),real(v,*a,**k))[1]):
            at.toggle(key="ia_metrology_overlay").set_value(False).run()
        self.assertEqual([Path(v).name for v in seen],["w5_15_c.png"])

    def test_05_image_without_result_keeps_placeholder(self):
        at=self.app(die=NO_RESULT_DIE)
        selected=at.session_state["ia_sqlite_selected_image_id"]
        if AnalysisRepository(self.db,result_root=OPERATIONAL_RESULT_ROOT,image_root=ROOT/"runtime_data/raw").image_history(selected):
            self.skipTest("chosen image now has a stored result")
        self.assertNotEqual(at.session_state["ia_sqlite_selected_image_id"],IMAGE)
        self.assertIn(ia.NO_RESULT_CAPTION,self.captions(at))
        self.assertTrue(any("저장된 분석 결과가 없습니다" in i.value for i in at.info))
        self.assertFalse(any(set(d.value.columns)=={"항목","평균 (µm)","표준편차 (µm)"} for d in at.dataframe))

    def test_06_missing_overlay_warns_but_rest_is_shown(self):
        original=ia._sha
        with patch.object(ia,"_sha",side_effect=lambda p:"0"*64 if Path(p).name=="metrology_overlay.png" else original(p)):
            at=self.app()
            at.toggle(key="ia_metrology_overlay").set_value(True).run()
        self.assertFalse(at.exception)
        self.assertTrue(any("선폭 Overlay" in w.value and "표시할 수 없습니다" in w.value for w in at.warning))
        self.assertIn("선택 Die 분석 요약 (lt/c/rb)",[s.value for s in at.subheader])

    def test_07_multiple_runs_default_latest_and_selectable(self):
        with closing(sqlite3.connect(self.db)) as c:  # Synthetic newer run, temp copy only.
            c.execute("PRAGMA foreign_keys=ON")
            batch=c.execute("SELECT * FROM analysis_batch_runs").fetchone();names=[d[0] for d in c.execute("SELECT * FROM analysis_batch_runs").description]
            b=dict(zip(names,batch));b.update(batch_run_id="batch-synthetic",started_at="2099-01-01T00:00:00+00:00",
                ended_at="2099-01-01T00:01:00+00:00",summary_path="synthetic/summary.json",import_sha256="f"*64)
            c.execute(f"INSERT INTO analysis_batch_runs({','.join(b)}) VALUES ({','.join('?'*len(b))})",list(b.values()))
            image=c.execute("SELECT * FROM analysis_image_runs").fetchone();names=[d[0] for d in c.execute("SELECT * FROM analysis_image_runs").description]
            r=dict(zip(names,image));r.update(analysis_run_id="batch-synthetic:x",batch_run_id="batch-synthetic",result_path=None,
                result_sha256=None,defect_overlay_path=None,defect_overlay_sha256=None,metrology_overlay_path=None,metrology_overlay_sha256=None)
            c.execute(f"INSERT INTO analysis_image_runs({','.join(r)}) VALUES ({','.join('?'*len(r))})",list(r.values()));c.commit()
        at=self.app()
        selector=at.selectbox(key="ia_analysis_run_"+IMAGE)
        self.assertEqual(selector.value,"batch-synthetic:x")  # Default: latest run, labelled 미채택.
        self.assertTrue(any("결과 파일 검증 실패" in e.value for e in at.error))  # Synthetic run has no files.
        selector.select(self.row["analysis_run_id"]).run()
        self.assertFalse(at.exception);self.assertFalse(at.error)
        self.assertTrue(any("measurement" not in str(d.value.columns) and "near_threshold" in d.value.columns for d in at.dataframe))

    def test_08_db_query_failure_is_contained(self):
        with patch.object(AnalysisRepository,"image_history",side_effect=sqlite3.OperationalError("injected")):
            at=self.app()
        self.assertFalse(at.exception)
        self.assertTrue(any("분석 결과 조회 실패" in e.value for e in at.error))
        self.assertEqual(at.session_state["ia_sqlite_selected_image_id"],IMAGE)  # Original still shown.
        self.assertTrue(any(s.value=="공정 조건별 분석" for s in at.subheader))  # Other cards intact.

    def pd_row(self):
        repository=AnalysisRepository(self.db,result_root=OPERATIONAL_RESULT_ROOT,image_root=ROOT/"runtime_data/raw")
        history=repository.image_history(PD_IMAGE)
        if not history:self.skipTest("W4 PD run not stored in this DB")
        return history[0],repository.get_analysis_run(history[0]["analysis_run_id"])

    def test_09_pd_excluded_result_wording_and_defects(self):
        row,detail=self.pd_row()
        self.assertEqual((row["outcome"],row["metrology_status"],row["is_adopted"]),("MODEL_PD_EXCLUDED","SKIPPED",0))
        import streamlit
        real=streamlit.image;seen=[]
        at=self.app(die="18",position="rb",wafer="W4")
        self.assertEqual(at.session_state["ia_sqlite_selected_image_id"],PD_IMAGE)
        with patch("streamlit.image",side_effect=lambda v,*a,**k:(seen.append(str(v)),real(v,*a,**k))[1]):
            at.toggle(key="ia_metrology_overlay").set_value(True)
            at.toggle(key="ia_defect_overlay").set_value(True).run()
        self.assertFalse(at.exception)
        text=" | ".join(self.captions(at));tiles=self.tiles(at)
        self.assertEqual((tiles["분석 상태"],tiles["Line Width"],tiles["Gap"],tiles["공식 채택"]),
                         ("PD 검출 · 선폭 계측 제외","PD 검출로 선폭 계측 제외","PD 검출로 선폭 계측 제외","미채택"))
        self.assertIn("Metrology Overlay 없음 · PD 검출로 선폭 계측 제외",text)
        self.assertIn(f"결함 검출 상세 ({len(detail['detections'])}건)",[e.label for e in at.expander])
        self.assertNotIn("MODEL_PD_EXCLUDED · 검토 필요",text+str(tiles))
        self.assertNotIn("Calibration: ",text);self.assertNotIn("Calibration 등록됨",text)  # TASK18-B-14B.1: line removed.
        self.assertFalse(at.warning)
        self.assertFalse(any(list(d.value.columns)==["항목","평균 (µm)","측정 수"] for d in at.dataframe))
        self.assertFalse(any("near_threshold" in d.value.columns for d in at.dataframe))
        table=next(d.value for d in at.dataframe if "confidence" in d.value.columns)
        self.assertEqual(table["class"].tolist(),[d["class_code"] for d in detail["detections"]])
        self.assertEqual(table["confidence"].tolist(),[d["confidence"] for d in detail["detections"]])
        self.assertEqual([Path(v).name for v in seen],["defect_overlay.png"])  # In place; Metrology absent (PD).
        self.assertIn(row["batch_run_id"],seen[0])
        self.assertEqual(sha(self.db),self.copy_sha)

    def test_10_pd_without_registered_calibration_is_not_claimed(self):
        self.pd_row()
        with patch.object(ia,"_registered_calibration",return_value=None):
            at=self.app(die="18",position="rb",wafer="W4")
        text=" | ".join(self.captions(at))
        self.assertNotIn("Calibration: ",text);self.assertNotIn("Calibration 등록됨",text)   # TASK18-B-14B.1: line removed.


if __name__=="__main__":unittest.main()
