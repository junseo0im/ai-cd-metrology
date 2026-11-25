"""TASK 18-B-5: Wafer Map exposure/PEB/Batch filters, condition separation and process charts.

The schema stores exposure per Wafer, so a real DB cannot hold two exposures on one Wafer;
mixed-exposure separation is tested on synthetic records with the same pure functions the UI uses.
UI tests run on a temporary Online-Backup copy of the operational DB (read-only checks).
"""
from contextlib import closing
from pathlib import Path
import hashlib,os,sqlite3,statistics,tempfile,unittest
from unittest.mock import patch

from database.init_db import DEFAULT_DB
from database.analysis_repository import OPERATIONAL_RESULT_ROOT

ROOT=Path(__file__).resolve().parents[2]
TEST_ROOT=ROOT/"runtime_data/validation/task18b5/tmp"
try:
    import plotly  # noqa: F401
    HAS_PLOTLY=True
except ImportError:
    HAS_PLOTLY=False


def record(image_id,die,pos,exposure=60.0,peb=None,batch="B1",sha=None):
    return {"image_id":image_id,"wafer_id":"WX","die_code":die,"position_code":pos,"exposure_value":exposure,
            "exposure_unit":None,"peb_temperature_c":peb,"batch_id":batch,"upload_mode":"wafer_batch","sha256":sha or image_id}


def result(classes=(),started="2026-01-01T00:00:00+00:00"):
    return {"outcome":"COMPLETE","defect_status":"SUCCESS","metrology_status":"SUCCESS","pd_policy_state":"NO_PD",
            "partial_measurement":0,"line_widths_um":[10.0],"line_mean_um":10.0,"gap_mean_um":10.0,"defect_details":[{"class_code":c,"source":"MODEL_POSTPROCESSED",
            "confidence":.5} for c in classes],"defect_classes":list(classes),"judgments":[],"started_at":started,
            "configuration":{"design_width_um":10.0},"is_adopted":0}


@unittest.skipUnless(HAS_PLOTLY,"plotly not installed")
class PureFilterTests(unittest.TestCase):
    def setUp(self):
        from dashboard.components import inspection_wafer as iw
        self.iw=iw
        # Die 11: clean at exposure 60; a Short only in the exposure-70 / PEB-110 capture.
        self.images=[record("a","11","lt"),record("b","11","c"),record("c","11","rb"),
                     record("d","11","lt",exposure=70.0,batch="B2"),record("e","11","c",peb=110.0,batch="B3")]
        self.analysis={"a":result(),"b":result(),"c":result(),"d":result(["s"]),"e":result(["o"])}
    def die11(self,images,judge=True):
        dies=self.iw.die_snapshot("WX",[],images,self.analysis,judge=judge)
        return next(d for d in dies if d["die_code"]=="11")
    def test_01_exposure_and_peb_never_mix(self):
        scope=self.iw.apply_condition_scope
        self.assertTrue(scope(self.images)["mixed"])
        only60=scope(self.images,exposure="60 s",peb="PEB 미입력")
        self.assertFalse(only60["mixed"]);self.assertEqual({r["image_id"] for r in only60["images"]},{"a","b","c"})
        die=self.die11(only60["images"])
        self.assertEqual((die["assessment"]["decision"],die["assessment"]["critical_defects"]),("PASS",[]))  # 70/PEB defects absent.
        only70=scope(self.images,exposure="70 s")
        die70=self.die11(only70["images"])
        self.assertEqual((die70["assessment"]["decision"],die70["assessment"]["expected_fail"]),("PENDING",True))
        peb110=scope(self.images,peb="110 s")
        self.assertEqual([r["image_id"] for r in peb110["images"]],["e"])
        blocked=self.die11(self.images,judge=False)                     # Mixed scope: no judgment at all.
        self.assertNotIn("assessment",blocked);self.assertIn("조건 선택 필요",blocked["judgment_blocked"])
        self.assertIn("조건 선택 필요",self.iw.die_hover(self.iw.apply_judgment_view([blocked],"AI 잠정 위험도")[0]))
    def test_02_same_condition_batches_and_duplicates(self):
        images=[record("a","11","lt",sha="X"),record("a2","11","lt",batch="B2",sha="X"),record("b","11","c"),record("c","11","rb")]
        excluded=self.iw.duplicate_exclusions(images,{"a2":result(),"b":result(),"c":result()})
        self.assertEqual(excluded,{"a"})                                       # Analysed copy kept.
        self.assertEqual(self.iw.duplicate_exclusions(images,{}),{"a2"})       # None analysed: earliest ID kept.
        dies=self.iw.die_snapshot("WX",[],images,{"a2":result(),"b":result(),"c":result()},exclude_from_judgment=excluded)
        die=next(d for d in dies if d["die_code"]=="11")
        self.assertEqual(die["positions"]["lt"]["image_ids"],["a","a2"])    # Both stay listed (nothing deleted).
        self.assertEqual(die["assessment"]["decision"],"PASS")              # Judged once, from the analysed copy.
    def test_03_empty_and_mixed_messages(self):
        from streamlit.testing.v1 import AppTest
        def page(images,mixed):
            import streamlit as st  # noqa: F401
            from dashboard.components.inspection_wafer import render_scope_summary
            render_scope_summary({"source":"SQLite","scope_label":"WX / 노광 60","images":images,"mixed":mixed,
                                  "dies":[],"duplicates_excluded":0})
        empty=AppTest.from_function(page,kwargs={"images":[],"mixed":False}).run()
        self.assertTrue(any("해당 조건 데이터 없음" in i.value for i in empty.info))
        mixed=AppTest.from_function(page,kwargs={"images":[{"x":1}],"mixed":True}).run()
        self.assertTrue(any("조건을 선택하세요" in w.value for w in mixed.warning))


@unittest.skipUnless(DEFAULT_DB.is_file() and OPERATIONAL_RESULT_ROOT.is_dir() and HAS_PLOTLY,"operational copy/plotly unavailable")
class LiveDataFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=TEST_ROOT);cls.db=Path(cls.tmp.name)/"copy.sqlite3"
        with closing(sqlite3.connect(f"file:{DEFAULT_DB.as_posix()}?mode=ro",uri=True)) as s,closing(sqlite3.connect(cls.db)) as d:
            s.backup(d)
        cls.copy_sha=hashlib.sha256(cls.db.read_bytes()).hexdigest();cls.op_sha=hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest()
        with closing(sqlite3.connect(cls.db)) as c:
            cls.analysed_w1=[r[0] for r in c.execute("""SELECT u.batch_id FROM upload_batches u JOIN images i USING(batch_id)
                JOIN analysis_image_runs a USING(image_id) WHERE u.wafer_id='W1' GROUP BY u.batch_id""")]
        if len(cls.analysed_w1)!=1:raise unittest.SkipTest("expects exactly one analysed W1 Batch")
        cls.b15=cls.analysed_w1[0]
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):
        env=patch.dict(os.environ,{"IDA_DB_PATH":str(self.db),"IDA_STORAGE_ROOT":str(ROOT/"runtime_data/raw"),
            "IDA_STAGING_ROOT":str(Path(self.tmp.name)/"staging"),"IDA_ANALYSIS_OUTPUT_ROOT":str(OPERATIONAL_RESULT_ROOT)})
        env.start();self.addCleanup(env.stop)
    def tearDown(self):
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(),self.copy_sha)
        self.assertEqual(hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest(),self.op_sha)
    def ia(self,wafer):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=90).run()
        at.switch_page("pages/inspection_analysis.py").run()
        at.run()  # TASK18-B-6: SQLite is the fixed Wafer Map source (no selector).
        at.selectbox(key="ia_wafer_wafer_SQLite").select(wafer).run();self.assertFalse(at.exception)
        return at
    def expected(self,wafer,batch=None):
        from dashboard.sqlite_provider import runtime_provider
        from dashboard.components.inspection_wafer import die_snapshot,duplicate_exclusions
        from dashboard.components.analysis_management import collect_images
        from dashboard.die_assessment import summarize_wafer
        p=runtime_provider();images=[r for r in collect_images(p,wafer_id=wafer).to_dict("records") if batch is None or r["batch_id"]==batch]
        analysis=p.latest_analysis(wafer)
        return summarize_wafer(die_snapshot(wafer,p.wafer_sites(wafer),images,analysis,exclude_from_judgment=duplicate_exclusions(images,analysis)))
    def summary_caption(self,at):return next(c.value for c in at.caption if c.value.startswith("AI 잠정 · Die PASS"))

    def test_04_w1_filter_options_batch15_and_selection_reset(self):
        from dashboard.die_assessment import rate_text
        from dashboard.components.inspection_wafer import select_die
        at=self.ia("W1")
        self.assertEqual(at.selectbox(key="ia_wafer_exposure_SQLite_W1").options,["60 s"])
        self.assertEqual(at.selectbox(key="ia_wafer_peb_SQLite_W1").options,["PEB 미입력"])
        batch=at.selectbox(key="ia_wafer_batch_SQLite_W1")
        self.assertEqual(len(batch.options)-1,15)                                    # 전체 + 15 W1 Batches.
        all_s=self.expected("W1")
        self.assertIn(f"Die PASS {all_s['PASS']} / FAIL {all_s['FAIL']} / PENDING {all_s['PENDING']}",self.summary_caption(at))
        batch.select(self.b15).run();self.assertFalse(at.exception)
        s=self.expected("W1",self.b15)
        caption=self.summary_caption(at)
        self.assertIn(f"Die PASS {s['PASS']} / FAIL {s['FAIL']} / PENDING {s['PENDING']}",caption)
        self.assertIn(f"Die Fail Rate {rate_text(s['die_fail_rate'])}",caption)
        self.assertIn(f"Row Fail Rate {rate_text(s['row_fail_rate'])}",caption)
        self.assertTrue(any(c.value.startswith("선택 범위: W1 / 노광 60") and self.b15[6:14] in c.value for c in at.caption))
        self.assertNotEqual((s["PASS"],s["FAIL"],s["PENDING"]),(all_s["PASS"],all_s["FAIL"],all_s["PENDING"]))  # Old tests no longer mixed in.
        scope="ia_wafer_map_SQLite_W1_"+self.b15
        select_die(at.session_state,scope,"16",popover=True);at.run()
        self.assertIsNotNone(at.session_state["ia_sqlite_selected_image_id"])
        at.selectbox(key="ia_wafer_batch_SQLite_W1").select(None).run()                  # Filter change -> reset.
        self.assertIsNone(at.session_state["ia_sqlite_selected_image_id"])
        self.assertFalse(at.exception)

    def test_05_w5_unchanged_and_w4_still_viewable(self):
        at=self.ia("W5")
        s=self.expected("W5")
        self.assertIn(f"Die PASS {s['PASS']} / FAIL {s['FAIL']} / PENDING {s['PENDING']}",self.summary_caption(at))
        self.assertFalse(any("조건을 선택하세요" in w.value for w in at.warning))   # W5 Batch02+Repeatability: one condition.
        at.selectbox(key="ia_wafer_wafer_SQLite").select("W4").run();self.assertFalse(at.exception)
        self.assertTrue(any(c.value.startswith("선택 범위: W4") for c in at.caption))

    def test_06_line_chart_bar_chart_and_sql_agreement(self):
        from dashboard.components.process_analysis import width_line_chart,error_bar_chart,_sorted_conditions,stored_results
        from dashboard.components.analysis_management import collect_images
        from dashboard.process_analysis import metadata_frame,condition_table
        from dashboard.sqlite_provider import runtime_provider
        p=runtime_provider();frame=metadata_frame(collect_images(p))
        summary=condition_table(frame,"exposure",results=stored_results(p))
        valid=_sorted_conditions(summary.dropna(subset=["평균 선폭 (µm)"]),frame,"exposure")
        line=width_line_chart(valid);bar=error_bar_chart(valid,"Absolute Error (%)")
        self.assertEqual((line.data[0].type,line.data[0].mode),("scatter","lines+markers+text"))
        self.assertEqual(bar.data[0].type,"bar")
        self.assertEqual(list(line.data[0].x),sorted(valid["공정 조건"],key=lambda v:float(v.split(" ")[0])))
        with closing(sqlite3.connect(self.db)) as c:
            sql={}
            for ev,run in c.execute("""WITH l AS (SELECT a.*,ROW_NUMBER() OVER (PARTITION BY a.image_id ORDER BY julianday(b.started_at) DESC,
                a.analysis_run_id DESC) rn FROM analysis_image_runs a JOIN analysis_batch_runs b USING(batch_run_id))
                SELECT r.exposure_value,l.analysis_run_id FROM l JOIN images i USING(image_id) JOIN upload_batches u ON u.batch_id=i.batch_id
                JOIN wafers w ON w.wafer_id=u.wafer_id JOIN process_runs r ON r.process_run_id=w.process_run_id
                WHERE l.rn=1 AND l.metrology_status='SUCCESS' AND l.partial_measurement=0 AND l.pd_policy_state='NO_PD'"""):
                sql.setdefault(ev,[]).append(statistics.mean([w for (w,) in c.execute(
                    "SELECT width_um FROM metrology_measurements WHERE analysis_run_id=? AND measurement_type='line'",(run,))]))
        for label,y,n,sd in zip(line.data[0].x,line.data[0].y,[d[0] for d in line.data[0].customdata],[d[1] for d in line.data[0].customdata]):
            values=sql[float(label.split(" ")[0])]
            self.assertEqual(n,len(values));self.assertAlmostEqual(y,statistics.mean(values),places=9)
            self.assertAlmostEqual(sd,statistics.stdev(values),places=9)
        for label,error in zip(bar.data[0].x,bar.data[0].y):
            self.assertAlmostEqual(error,abs(statistics.mean(sql[float(label.split(" ")[0])])-10)/10*100,places=9)


if __name__=="__main__":unittest.main()
