"""TASK 5 regression checks. AppTest is not a browser/visual verification."""
import json
from pathlib import Path
import re
import unittest

from streamlit.testing.v1 import AppTest
from dashboard.navigation import page_data_source
from dashboard.components.inspection_wafer import build_inspection_wafer
import dashboard.tests.test_sqlite_input as fixtures

ROOT=Path(__file__).resolve().parents[2]


class Task5SourceTests(unittest.TestCase):
    setUp=fixtures.SQLiteInputTests.setUp
    app=fixtures.SQLiteInputTests.app

    def test_source_notice_changes_without_claiming_sqlite_analysis(self):
        at=self.app()
        self.assertTrue(any("SQLite 등록·조회" in row.value for row in at.caption))
        self.assertTrue(any("SQLITE" in row.value for row in at.sidebar.caption))
        self.assertFalse(any("MOCK DATA ONLY" in row.value for row in at.sidebar.caption))
        self.assertTrue(any("계측" in row.value and "미연결" in row.value for row in at.caption))
        at.switch_page("pages/inspection_analysis.py").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("MOCK DATA ONLY" in row.value for row in at.sidebar.caption))
        at.switch_page("pages/data_control.py").run()
        self.assertEqual(at.radio(key="data_control_source").value,"SQLite")
        self.assertTrue(any("SQLITE" in row.value for row in at.sidebar.caption))
        at.radio(key="data_control_source").set_value("Mock").run()
        self.assertTrue(any("MOCK DATA ONLY" in row.value for row in at.sidebar.caption))
        self.assertFalse(any("SQLite 등록·조회" in row.value for row in at.caption))

    def test_source_context_does_not_leak_into_other_pages(self):
        state={"data_control_selected_source":"SQLite","batch_detail_source":"SQLite"}
        self.assertEqual(page_data_source("data_control",state),"SQLite")
        self.assertEqual(page_data_source("batch_detail",state),"SQLite")
        for active in ("main_home","wafer_map","image_analysis","repeatability","lighting","calibration"):
            self.assertEqual(page_data_source(active,state),"Mock")


class Task5MapTests(unittest.TestCase):
    def test_standalone_uses_same_72_regions_manual_selection_reset_and_image_binding(self):
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run()
        at.switch_page("pages/wafer_map.py").run()
        self.assertFalse(at.exception)
        spec=json.loads(at.get("plotly_chart")[0].proto.spec)
        self.assertEqual(sum(len(trace["customdata"]) for trace in spec["data"]),72)
        self.assertEqual({r[0] for trace in spec["data"] for r in trace["customdata"]},
                         {f"{row}{col}" for row in range(1,4) for col in range(1,9)})
        self.assertEqual({r[1] for trace in spec["data"] for r in trace["customdata"]},{"lt","c","rb"})
        self.assertEqual({trace["name"] for trace in spec["data"]},
                         {trace.name for trace in build_inspection_wafer("BATCH-DEMO-060").data})
        at.selectbox(key="map_image").select("IMG-DEMO-1-17").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["selected_die_id"],"31")
        self.assertEqual(at.session_state["selected_position"],"c")
        self.assertEqual(at.session_state["selected_image_id"],"IMG-DEMO-1-17")
        at.button(key="map_reset").click().run()
        self.assertFalse(at.exception)
        self.assertIsNone(at.session_state["selected_die_id"])
        self.assertIsNone(at.session_state["selected_position"])
        self.assertEqual(at.selectbox(key="map_image").value,"IMG-DEMO-1-01")
        self.assertTrue(any("IMG-DEMO-1-01" in row.value and "원본 영역 예시" in row.value for row in at.caption))
        self.assertTrue(any("공용 예시" in row.value for row in at.caption))


class Task5StylesTests(unittest.TestCase):
    def test_disabled_button_color_pairs_have_readable_contrast_and_scoped_rules(self):
        css=(ROOT/"dashboard/components/styles.css").read_text(encoding="utf-8")
        selector='.st-key-card_data_input [data-testid="stBaseButton-primary"]:disabled'
        block=css.split(selector+" {",1)[1].split("}",1)[0]
        pairs={}
        for property_name,light,dark in re.findall(r"(background|color):light-dark\((#[0-9a-f]{6}),(#[0-9a-f]{6})\)",block):
            pairs[property_name]=(light,dark)
        def luminance(hex_color):
            rgb=[int(hex_color[i:i+2],16)/255 for i in (1,3,5)]
            rgb=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in rgb]
            return sum(a*b for a,b in zip(rgb,(.2126,.7152,.0722)))
        for index in (0,1):
            a,b=sorted([luminance(pairs["background"][index]),luminance(pairs["color"][index])])
            self.assertGreaterEqual((b+.05)/(a+.05),4.5)
        for selector in ('[data-testid="stTextInputRootElement"]',
                         '[data-testid="stSelectbox"] [role="group"]',
                         '[data-testid="stNumberInputContainer"]'):
            self.assertIn(".st-key-card_data_input "+selector,css)
        self.assertNotIn("data-baseweb",(ROOT/"dashboard/pages/data_control.py").read_text(encoding="utf-8"))


if __name__=="__main__":
    unittest.main()
