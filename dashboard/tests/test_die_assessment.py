"""TASK 18-B-4: AI provisional Die assessment v0.1 (pure rules + read-only W5 integration)."""
from contextlib import closing
from pathlib import Path
import hashlib,os,sqlite3,tempfile,unittest
from unittest.mock import patch

from database.init_db import DEFAULT_DB
from database.analysis_repository import OPERATIONAL_RESULT_ROOT
from dashboard.die_assessment import (assess_image,assess_die,summarize_wafer,cd_level,rate,rate_text,NOT_EVALUATED)

ROOT=Path(__file__).resolve().parents[2]
TEST_ROOT=ROOT/"runtime_data/validation/task18b4/tmp"
try:
    import plotly  # noqa: F401
    HAS_PLOTLY=True
except ImportError:
    HAS_PLOTLY=False


def item(**c):
    base={"outcome":"COMPLETE","defect_status":"SUCCESS","metrology_status":"SUCCESS","pd_policy_state":"NO_PD",
          "partial_measurement":0,"line_widths_um":[10.0,10.2],"defect_details":[],"defect_classes":[],
          "judgments":[{"label":"normal"}],"configuration":{"design_width_um":10.0},"is_adopted":0}
    base.update(c);return base


def det(code,source="MODEL_POSTPROCESSED",confidence=0.5):
    return {"class_code":code,"source":source,"confidence":confidence}


class RuleTests(unittest.TestCase):
    def test_01_cd_boundaries_inclusive(self):
        for width,level in ((10.5,"NORMAL"),(9.5,"NORMAL"),(9.49,"MINOR"),(9.0,"MINOR"),(8.99,"WARNING"),
                            (8.0,"WARNING"),(12.0,"WARNING"),(7.99,"CRITICAL"),(12.01,"CRITICAL")):
            self.assertEqual(assess_image(item(line_widths_um=[width],judgments=[]))["cd_level"],level,width)
        self.assertEqual(cd_level(5.000000000000001),"NORMAL")  # Float noise at the boundary.

    def test_02_defect_classes(self):
        for code in ("o","s","pd"):
            a=assess_image(item(defect_details=[det(code)]))
            self.assertEqual((a["defect_level"],a["risk"],a["critical_defects"]),("UNUSABLE","UNUSABLE",[code]))
        for code in ("p","sc"):
            a=assess_image(item(defect_details=[det(code,confidence=0.99)]))
            self.assertEqual(a["defect_level"],"WARNING");self.assertTrue(any("P/SC" in r for r in a["review"]))
            self.assertNotIn(a["risk"],("CRITICAL","UNUSABLE"))  # No count/confidence escalation.
        many=assess_image(item(defect_details=[det("p")]*20))
        self.assertEqual(many["defect_level"],"WARNING")
        clean=assess_image(item())
        self.assertIsNone(clean["defect_level"]);self.assertEqual(clean["risk"],"NORMAL")  # CD decides; not a PASS.

    def test_03_states_pd_policy_partial_design(self):
        pd=assess_image(item(outcome="MODEL_PD_EXCLUDED",metrology_status="SKIPPED",line_widths_um=[],
                             pd_policy_state="MODEL_PD_EXCLUDED",defect_details=[det("pd")]))
        self.assertEqual((pd["cd_state"],pd["cd_level"],pd["risk"],pd["complete"]),("PD_EXCLUDED",None,"UNUSABLE",True))
        policy=assess_image(item(outcome="POLICY_REVIEW_REQUIRED",metrology_status="SKIPPED",line_widths_um=[],
                                 defect_details=[det("pd","HEURISTIC",0.6)]))
        self.assertEqual((policy["cd_state"],policy["critical_defects"],policy["complete"]),("POLICY_HOLD",[],False))
        self.assertTrue(policy["heuristic_pd"])
        partial=assess_image(item(outcome="PARTIAL_MEASUREMENT",partial_measurement=1,line_widths_um=[7.0]))
        self.assertEqual((partial["cd_state"],partial["cd_level"],partial["complete"]),("PROVISIONAL_PARTIAL","CRITICAL",False))
        nodesign=assess_image(item(configuration={}))
        self.assertEqual((nodesign["cd_state"],nodesign["cd_level"]),("NO_DESIGN_WIDTH",None))  # 10 µm never forced.
        self.assertEqual(assess_image(None)["risk"],NOT_EVALUATED)
        boundary=assess_image(item(line_widths_um=[8.0],judgments=[{"label":"pn"}]))
        self.assertEqual(boundary["cd_level"],"WARNING");self.assertTrue(boundary["conflicts"])  # PN at 8.0 µm shown.

    def test_04_die_completeness_binary_and_colour_are_separate(self):
        modes={k:"wafer_batch" for k in ("a","b","c","d","r")};modes["r"]="repeatability"
        ok={"a":item(line_widths_um=[7.0],judgments=[{"label":"pn"}]),"b":item(),"c":item(),"d":item()}
        die=assess_die({"lt":["a"],"c":["b"],"rb":["c","d"]},ok,modes)
        self.assertEqual((die["display_risk"],die["decision"]),("CRITICAL","PASS"))  # Colour vs O/S/PD verdict.
        self.assertEqual(len(die["positions"]["rb"]["image_ids"]),2)                 # Two images, one position.
        fail=assess_die({"lt":["a"],"c":["b"],"rb":["c"]},{**ok,"c":item(defect_details=[det("s")])},modes)
        self.assertEqual((fail["decision"],fail["display_risk"]),("FAIL","UNUSABLE"))
        pending=assess_die({"lt":[],"c":["b"],"rb":["c"]},{**ok,"c":item(defect_details=[det("o")])},modes)
        self.assertEqual((pending["decision"],pending["display_risk"],pending["provisional_risk"]),("PENDING",NOT_EVALUATED,"UNUSABLE"))
        self.assertTrue(pending["expected_fail"])
        only_repeat=assess_die({"lt":["a"],"c":["r"],"rb":["c"]},ok,modes)
        self.assertEqual(only_repeat["missing_positions"],["c"])  # Repeatability never judged.
        unanalysed=assess_die({"lt":["a"],"c":["b"],"rb":["d"]},{"a":ok["a"],"b":ok["b"]},modes)
        self.assertEqual(unanalysed["decision"],"PENDING")

    def test_05_rows_and_rates(self):
        def die(row,decision):return {"row":row,"assessment":{"decision":decision,"display_risk":NOT_EVALUATED}}
        dies=[die(1,"PASS")]*5+[die(1,"FAIL")]*3+[die(2,"PASS")]*4+[die(2,"FAIL")]*4+[die(3,"PASS")]*7+[die(3,"PENDING")]
        s=summarize_wafer(dies)
        self.assertEqual([r["verdict"] for r in s["rows"]],["PASS","FAIL","PENDING"])  # >=5 PASS rule; PENDING blocks.
        self.assertEqual((s["PASS"],s["FAIL"],s["PENDING"],s["evaluated"]),(16,7,1,23))
        self.assertAlmostEqual(s["die_fail_rate"],7/23*100);self.assertEqual(s["row_fail_rate"],50.0)
        self.assertIsNone(rate(0,0));self.assertEqual(rate_text(None),"계산 불가 (평가 완료 없음)")
        self.assertEqual(rate_text(rate(0,3)),"0.0%")


@unittest.skipUnless(DEFAULT_DB.is_file() and OPERATIONAL_RESULT_ROOT.is_dir() and HAS_PLOTLY,"operational copy/plotly unavailable")
class W5AssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=TEST_ROOT);cls.db=Path(cls.tmp.name)/"copy.sqlite3"
        with closing(sqlite3.connect(f"file:{DEFAULT_DB.as_posix()}?mode=ro",uri=True)) as s,closing(sqlite3.connect(cls.db)) as d:
            s.backup(d)
        cls.copy_sha=hashlib.sha256(cls.db.read_bytes()).hexdigest()
        cls.op_sha=hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest()
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def setUp(self):
        env=patch.dict(os.environ,{"IDA_DB_PATH":str(self.db),"IDA_STORAGE_ROOT":str(ROOT/"runtime_data/raw"),
            "IDA_STAGING_ROOT":str(Path(self.tmp.name)/"staging"),"IDA_ANALYSIS_OUTPUT_ROOT":str(OPERATIONAL_RESULT_ROOT)})
        env.start();self.addCleanup(env.stop)
    def tearDown(self):
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(),self.copy_sha)
        self.assertEqual(hashlib.sha256(DEFAULT_DB.read_bytes()).hexdigest(),self.op_sha)
    def dies(self,wafer="W5"):
        from dashboard.sqlite_provider import runtime_provider
        from dashboard.components.inspection_wafer import die_snapshot,apply_judgment_view
        from dashboard.components.analysis_management import collect_images
        p=runtime_provider()
        dies=die_snapshot(wafer,p.wafer_sites(wafer),collect_images(p,wafer_id=wafer).to_dict("records"),p.latest_analysis(wafer))
        return apply_judgment_view(dies,"AI 잠정 위험도"),p

    def test_06_w5_structure_pending_dies_and_official_view(self):
        dies,_=self.dies()
        self.assertEqual(len(dies),24);self.assertEqual(sorted({d["row"] for d in dies}),[1,2,3])
        by={d["die_code"]:d["assessment"] for d in dies}
        for code,missing in (("11",["lt","c"]),("12",["c"]),("33",["c"]),("34",["c"])):
            self.assertEqual(by[code]["decision"],"PENDING",code)
            self.assertTrue(set(missing)<=set(by[code]["missing_positions"]),code)
            # TASK18-B-15: stored PD detection / PD-suspect hold makes the Die UNUSABLE ahead of missing positions.
            self.assertEqual(by[code]["display_risk"],"UNUSABLE" if by[code]["pd_unusable"]["unusable"] else NOT_EVALUATED)
        self.assertEqual({d["official"]["display_risk"] for d in dies},{NOT_EVALUATED})  # Nothing adopted.
        self.assertEqual({d["official"]["decision"] for d in dies},{"PENDING"})
        with closing(sqlite3.connect(self.db)) as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM analysis_image_runs WHERE is_adopted<>0").fetchone()[0],0)

    def test_07_colour_equals_detail_and_rates_consistent(self):
        from dashboard.components.inspection_wafer import build_inspection_wafer,die_hover,die_hover_lines
        dies,_=self.dies()
        fig=build_inspection_wafer(dies=dies)
        for trace in fig.data:
            for row in trace.customdata:
                die=next(d for d in dies if d["die_code"]==row[0])
                self.assertEqual(trace.name,die["assessment"]["display_risk"])          # Map colour == detail.
                self.assertEqual(row[3],die_hover(die))
                lines=die_hover_lines(die)   # TASK18-B-15: 'Wafer · Die' / reason / positions, max 3 lines.
                self.assertEqual(lines[0],f"{die['wafer_id']} · Die {die['die_code']}");self.assertLessEqual(len(lines),3)
                self.assertNotIn("점수",row[4]);self.assertNotIn("이미지 참조",row[4])
        s=summarize_wafer(dies)
        self.assertEqual(s["PASS"]+s["FAIL"]+s["PENDING"],24)
        self.assertEqual(s["die_fail_rate"],rate(s["FAIL"],s["PASS"]))
        for r in s["rows"]:
            members=[d for d in dies if d["row"]==r["row"]]
            self.assertEqual(r["PENDING"],sum(d["assessment"]["decision"]=="PENDING" for d in members))
            if r["PENDING"]:self.assertEqual(r["verdict"],"PENDING")
            else:self.assertEqual(r["verdict"],"PASS" if r["PASS"]>=5 else "FAIL")
        # A Die with model O/S/PD is never PASS; PD-excluded image keeps its defect evidence.
        for d in dies:
            if d["assessment"]["critical_defects"]:self.assertNotEqual(d["assessment"]["decision"],"PASS")

    def test_08_pages_render_views_and_fail_rate_table(self):
        from streamlit.testing.v1 import AppTest
        from dashboard.components.inspection_wafer import select_die
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=90).run()
        at.switch_page("pages/inspection_analysis.py").run()
        at.run()  # TASK18-B-6: SQLite is the fixed Wafer Map source (no selector).
        at.selectbox(key="ia_wafer_wafer_SQLite").select("W5").run();self.assertFalse(at.exception)
        # TASK18-B-15: I&A shows the AI provisional risk only (no AI / official switch).
        self.assertFalse([r for r in at.radio if r.key and r.key.endswith("_judgment_view")])
        dies,_=self.dies();s=summarize_wafer(dies)
        table=next(d.value for d in at.dataframe if "Die Fail Rate (AI 잠정)" in getattr(d.value,"columns",[]))
        w5=table[table["Wafer ID"]=="W5"].iloc[0]
        self.assertEqual((w5["Die PASS"],w5["Die FAIL"],w5["Die PENDING"]),(s["PASS"],s["FAIL"],s["PENDING"]))
        self.assertEqual(w5["Die Fail Rate (AI 잠정)"],rate_text(s["die_fail_rate"]))
        self.assertEqual(w5["Row Fail Rate (AI 잠정)"],rate_text(s["row_fail_rate"]))
        select_die(at.session_state,"ia_wafer_map_SQLite_W5_all","16",popover=True);at.run()
        self.assertTrue(any("공식 판정 NOT_EVALUATED" in c.value for c in at.caption))   # Popover still shows it.
        # W4 still renders and its stored PD result remains viewable.
        at.selectbox(key="ia_wafer_wafer_SQLite").select("W4").run()
        select_die(at.session_state,"ia_wafer_map_SQLite_W4_all","18",popover=True);at.run()
        next(b for b in at.get("button_group") if b.key.startswith("ia_sqlite_position_")).set_value("rb").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("PD 검출 · 선폭 계측 제외" in c.value for c in at.caption))


if __name__=="__main__":unittest.main()
