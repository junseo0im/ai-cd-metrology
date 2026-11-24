"""TASK 18-B-3: stored W5 results in Wafer Map / I&A / process graph (read-only, no inference).

UI tests run on a temporary Online-Backup copy of the operational DB (which holds the real
W5 results); expectations are derived from that DB, never hard-coded values.
"""
from contextlib import closing
from pathlib import Path
import hashlib,os,sqlite3,statistics,tempfile,unittest
from unittest.mock import patch

import pandas as pd
from database.init_db import DEFAULT_DB
from database.analysis_repository import OPERATIONAL_RESULT_ROOT
from dashboard.process_analysis import image_mean_um,condition_table,metadata_frame

ROOT=Path(__file__).resolve().parents[2]
W5_BATCH="BATCH-7896834d126b4861bb9cf6fdee44b091"
TEST_ROOT=ROOT/"runtime_data/validation/task18b3/tmp"
try:
    import plotly  # noqa: F401
    HAS_PLOTLY=True
except ImportError:
    HAS_PLOTLY=False


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AggregationUnitTests(unittest.TestCase):
    def result(self,**c):
        r={"status":"SUCCESS","pd_detected":False,"partial_measurement":False,
           "calibration":{"calibration_id":"cal","valid":True,"magnification":20},"calibration_applied":True,
           "unit":"um","valid_line_widths_um":[8.0,9.0],"valid_gap_widths_um":[10.0,12.0],"defect_classes":[]}
        r.update(c);return r
    def test_01_partial_and_pd_excluded_gap_and_detections_counted(self):
        self.assertEqual(image_mean_um(self.result(partial_measurement=True),magnification=20)["status"],"PARTIAL_EXCLUDED")
        frame=metadata_frame(pd.DataFrame([{"image_id":k,"wafer_id":"W","batch_id":"B","exposure_value":70,
            "exposure_unit":None,"peb_temperature_c":None,"magnification":20} for k in "ABCD"]))
        results={"A":self.result(),"B":self.result(valid_line_widths_um=[10.0],valid_gap_widths_um=[8.0],defect_classes=["p"]),
                 "C":self.result(partial_measurement=True,valid_line_widths_um=[1.0]),
                 "D":self.result(pd_detected=True,defect_classes=["pd","pd","p"])}
        row=condition_table(frame,"exposure",results=results).iloc[0]
        self.assertEqual((row["유효 이미지 수"],row["PD 제외 이미지 수"],row["부분 계측 제외 이미지 수"],row["분석 결과 이미지 수"]),(2,1,1,4))
        self.assertEqual(row["평균 선폭 (µm)"],statistics.mean([8.5,10.0]))   # Partial C never enters.
        self.assertEqual(row["평균 Gap (µm)"],statistics.mean([11.0,8.0]))
        self.assertEqual((row["pd 검출 건수"],row["p 검출 건수"],row["p 포함 이미지 수"]),(2,2,2))
        self.assertTrue(pd.isna(condition_table(frame,"exposure").iloc[0]["p 검출 건수"]))  # No results -> unknown.


@unittest.skipUnless(DEFAULT_DB.is_file() and OPERATIONAL_RESULT_ROOT.is_dir() and HAS_PLOTLY,"operational copy/plotly unavailable")
class W5DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=TEST_ROOT);cls.db=Path(cls.tmp.name)/"copy.sqlite3"
        with closing(sqlite3.connect(f"file:{DEFAULT_DB.as_posix()}?mode=ro",uri=True)) as s,closing(sqlite3.connect(cls.db)) as d:
            s.backup(d)
        cls.copy_sha=sha(cls.db);cls.op_sha=sha(DEFAULT_DB)
        with closing(sqlite3.connect(cls.db)) as c:
            cls.latest={r[0]:r for r in c.execute("""WITH l AS (SELECT a.image_id,a.outcome,a.analysis_run_id,
                ROW_NUMBER() OVER (PARTITION BY a.image_id ORDER BY julianday(b.started_at) DESC,a.analysis_run_id DESC) rn
                FROM analysis_image_runs a JOIN analysis_batch_runs b USING(batch_run_id)) SELECT image_id,outcome,analysis_run_id FROM l WHERE rn=1""")}
            cls.sites={r[0]:(r[1],r[2]) for r in c.execute("SELECT i.image_id,d.die_code,p.position_code FROM images i "
                "JOIN pattern_sites p USING(physical_site_id) JOIN dies d ON d.die_id=p.die_id WHERE i.batch_id=?",(W5_BATCH,))}
        if not all(i in cls.latest for i in cls.sites):
            raise unittest.SkipTest("W5 Batch 02 is not fully analysed in this DB")
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):
        env=patch.dict(os.environ,{"IDA_DB_PATH":str(self.db),"IDA_STORAGE_ROOT":str(ROOT/"runtime_data/raw"),
            "IDA_STAGING_ROOT":str(Path(self.tmp.name)/"staging"),"IDA_ANALYSIS_OUTPUT_ROOT":str(OPERATIONAL_RESULT_ROOT)})
        env.start();self.addCleanup(env.stop)
    def tearDown(self):
        self.assertEqual(sha(self.db),self.copy_sha);self.assertEqual(sha(DEFAULT_DB),self.op_sha)
    def site_of(self,outcome):
        return next((iid,self.sites[iid]) for iid in sorted(self.sites) if self.latest[iid][1]==outcome)
    def ia(self,die,position,wafer="W5"):
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
    def captions(self,at):return " | ".join(c.value for c in at.caption)

    def test_02_wafer_map_mapping_and_status_texts(self):
        from dashboard.sqlite_provider import runtime_provider
        from dashboard.components.inspection_wafer import die_snapshot
        from dashboard.components.analysis_management import collect_images
        provider=runtime_provider();images=collect_images(provider,wafer_id="W5").to_dict("records")
        dies=die_snapshot("W5",provider.wafer_sites("W5"),images,provider.latest_analysis("W5"))
        self.assertEqual(len(dies),24);self.assertEqual({d["criticality"] for d in dies},{"NOT_EVALUATED"})
        placed=[(d["die_code"],pos,i) for d in dies for pos,s in d["positions"].items() for i in s["image_ids"]]
        self.assertEqual(len(placed),len({p[2] for p in placed}))                       # No image twice.
        for die,pos,image_id in placed:
            record=next(r for r in images if r["image_id"]==image_id)
            self.assertEqual((record["die_code"],record["position_code"]),(die,pos))    # Never another Die.
        analysed={i for _,_,i in placed if i in self.latest}
        self.assertEqual(sum(d["analysis_counts"]["analysed"] for d in dies),len(analysed))
        self.assertTrue(set(self.sites)<=analysed)
        pd_id,(die,pos)=self.site_of("MODEL_PD_EXCLUDED")
        self.assertIn("PD 검출 · 계측 제외",next(d for d in dies if d["die_code"]==die)["positions"][pos]["metrology"])
        partial_id,(die,pos)=self.site_of("PARTIAL_MEASUREMENT")
        self.assertIn("부분 계측",next(d for d in dies if d["die_code"]==die)["positions"][pos]["metrology"])

    def test_03_die_click_image_overlay_and_values(self):
        import streamlit
        image_id,(die,pos)=self.site_of("QUALITY_REVIEW_REQUIRED")
        real=streamlit.image;seen=[]
        at=self.ia(die,pos)
        if at.session_state["ia_sqlite_selected_image_id"]!=image_id:  # Multi-image site: choose the image.
            at.selectbox(key=next(s.key for s in at.selectbox if s.key and s.key.startswith("ia_sqlite_image_"))).select(image_id).run()
        self.assertEqual(at.session_state["ia_sqlite_selected_image_id"],image_id)
        with patch("streamlit.image",side_effect=lambda v,*a,**k:(seen.append(str(v)),real(v,*a,**k))[1]):
            at.toggle(key="ia_metrology_overlay").set_value(True)
            at.toggle(key="ia_defect_overlay").set_value(True).run()
        # TASK18-B-14B: original + the two stored composites of THIS image (nothing redrawn).
        self.assertEqual([Path(v).name for v in seen][1:],["metrology_overlay.png","defect_overlay.png"])
        self.assertTrue(all(self.latest[image_id][2].split(":")[0] in v for v in seen[1:]))
        with closing(sqlite3.connect(self.db)) as c:
            line,gap=c.execute("SELECT line_mean_um,gap_mean_um FROM analysis_image_runs WHERE analysis_run_id=?",(self.latest[image_id][2],)).fetchone()
        from dashboard.tests.test_ia_analysis_result import result_table
        tiles=result_table(at)                                                           # TASK18-B-14B.2 table
        self.assertTrue(tiles["Line Width"].startswith(f"{line:.3f} µm (") and tiles["Gap"].startswith(f"{gap:.3f} µm ("))
        self.assertEqual(tiles["분석 상태"],"검토 필요 · 품질")

    def test_04_pd_partial_and_policy_states(self):
        for outcome,expected in (("MODEL_PD_EXCLUDED","PD 검출 · 선폭 계측 제외"),
                                 ("PARTIAL_MEASUREMENT","부분 계측 · 검토 필요 · 평균은 참고값"),
                                 ("POLICY_REVIEW_REQUIRED","PD 의심(규칙 기반) · 정책 검토 필요")):
            image_id,(die,pos)=self.site_of(outcome)
            at=self.ia(die,pos)
            self.assertEqual(at.session_state["ia_sqlite_selected_image_id"],image_id)
            from dashboard.tests.test_ia_analysis_result import result_table
            tiles=result_table(at)
            self.assertTrue(tiles["분석 상태"].startswith(expected),outcome)
            self.assertEqual(tiles["공식 채택"],"미채택")

    def test_05_multi_image_site_lists_each_image(self):
        multi=[s for s in set(self.sites.values()) if list(self.sites.values()).count(s)>1]
        if not multi:self.skipTest("no multi-image site")
        die,pos=sorted(multi)[0]
        ids=sorted(i for i,s in self.sites.items() if s==(die,pos))
        at=self.ia(die,pos)
        box=next(s for s in at.selectbox if s.key and s.key.startswith("ia_sqlite_image_"))
        self.assertTrue(set(ids)<=set(box.options) or all(any(i in o for o in box.options) for i in ids))

    def test_06_process_graph_matches_sql_and_w4_still_viewable(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=60).run()
        at.switch_page("pages/inspection_analysis.py").run()
        at.run()  # TASK18-B-6: SQLite is the fixed Wafer Map source (no selector).;self.assertFalse(at.exception)
        self.assertTrue(any("AI 잠정 분석 결과(공식 채택 아님)" in c.value for c in at.caption))
        table=next(d.value for d in at.dataframe if "평균 Gap (µm)" in getattr(d.value,"columns",[]))
        with closing(sqlite3.connect(self.db)) as c:
            means=[]
            for (run,) in c.execute("""WITH l AS (SELECT a.*,ROW_NUMBER() OVER (PARTITION BY a.image_id ORDER BY julianday(b.started_at) DESC,
                a.analysis_run_id DESC) rn FROM analysis_image_runs a JOIN analysis_batch_runs b USING(batch_run_id))
                SELECT l.analysis_run_id FROM l JOIN images i USING(image_id) JOIN upload_batches u ON u.batch_id=i.batch_id
                JOIN wafers w ON w.wafer_id=u.wafer_id JOIN process_runs r ON r.process_run_id=w.process_run_id
                WHERE l.rn=1 AND r.exposure_value=70 AND l.metrology_status='SUCCESS' AND l.partial_measurement=0 AND l.pd_policy_state='NO_PD'"""):
                means.append(statistics.mean([w for (w,) in c.execute(
                    "SELECT width_um FROM metrology_measurements WHERE analysis_run_id=? AND measurement_type='line'",(run,))]))
        row=next(r for r in table.to_dict("records") if str(r["공정 조건"]).startswith("70"))
        self.assertEqual(int(row["유효 이미지 수"]),len(means))
        self.assertAlmostEqual(row["평균 선폭 (µm)"],statistics.mean(means),places=9)
        self.assertAlmostEqual(row["이미지 간 표준편차 (µm)"],statistics.stdev(means),places=9)
        w4=self.ia("18","rb",wafer="W4")
        self.assertIn("PD 검출 · 선폭 계측 제외",self.captions(w4))


if __name__=="__main__":unittest.main()
