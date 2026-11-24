"""Phase 1.2 changed surfaces only; all existing regression tests remain."""
from pathlib import Path
import sys
import unittest
import tempfile
import os
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from dashboard import data_provider as data
from dashboard.navigation import PAGE_SPECS
from dashboard.components.inspection_wafer import (
    PatternCriticality,CRITICALITY_STYLES,POSITION_ORDER,
    build_inspection_wafer,set_selection,
)


class Phase12Tests(unittest.TestCase):
    def setUp(self):
        (ROOT/"dashboard/.tmp").mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=ROOT/"dashboard/.tmp",prefix="phase12_records_")
        self.addCleanup(temporary.cleanup)
        env=patch.dict(os.environ,{"IDA_RECORDS_DIR":temporary.name,"IDA_RECORDS_EDIT_ENABLED":"0"})
        env.start();self.addCleanup(env.stop)

    def app(self):
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=20)
        at.session_state["_wafer_source"]="Mock"  # TASK18-B-6: operational Wafer Map is SQLite-only; this suite tests the preserved Mock UI explicitly.
        at.run()
        self.assertEqual(len(at.exception),0)
        return at

    def click(self,at,label):
        at.switch_page(PAGE_SPECS[at.session_state["active_page_key"]][0])
        next(b for b in at.button if b.label==label).click().run()
        self.assertEqual(len(at.exception),0)

    def test_home_card_round_trip_and_matching_buttons(self):
        at=self.app()
        for label in ("Data Control Center 열기","Inspection & Analysis 열기"):
            self.assertEqual(next(b for b in at.button if b.label==label).proto.type,"primary")
        self.assertFalse(any("테마: 우상단" in c.value for c in at.sidebar.caption))
        self.assertFalse(any("MOCK DATA ONLY" in c.value for c in at.sidebar.caption))
        self.assertTrue(any("SQLITE" in c.value for c in at.sidebar.caption))  # TASK18-B-8: Main Home KPIs are SQLite.
        self.assertFalse(any(b.label=="실험 환경 및 조건 열기" for b in at.button))
        at.switch_page(PAGE_SPECS["experiment_environment"][0]).run()
        at.session_state["active_page_key"]="experiment_environment"
        self.assertEqual(len(at.exception),0)
        self.assertEqual(at.title[0].value,"실험 환경 및 조건")
        self.assertEqual([t.label for t in at.tabs],["장비 사양","Wafer / Die / Pattern 구조","공정 조건",
                         "Mask 설계 / 제작값","오차율 계산 기준(허용값)","Calibration","계측 방법"])
        self.assertTrue(any("항목" in df.value and "X축 Calibration 값" in df.value["항목"].tolist() for df in at.dataframe))
        self.click(at,"Main Home으로 돌아가기")
        self.assertEqual(at.session_state["active_page_key"],"main_home")

    def test_die_fixture_has_24_dies_and_72_independent_regions(self):
        original=data.catalog()
        for batch_id in data.batches("Wafer Batch")["batch_id"]:
            regions=data.inspection_wafer_regions(batch_id)
            self.assertEqual(len(regions),72)
            self.assertEqual({r["die_id"] for r in regions},
                             {f"{row}{col}" for row in range(1,4) for col in range(1,9)})
            self.assertEqual({r["row"] for r in regions},{1,2,3})
            self.assertEqual({r["col"] for r in regions},set(range(1,9)))
            for r in regions:
                self.assertEqual(r["die_id"],f'{r["row"]}{r["col"]}')
                suffix=f'{(r["row"]-1)*8+r["col"]:02}'
                self.assertTrue(r["image_id"].endswith("-"+suffix))
            self.assertEqual({r["criticality"] for r in regions},{s.value for s in PatternCriticality})
            self.assertEqual({r["image_id"] for r in regions},set(data.images(batch_id)["image_id"]))
            for die_id in {r["die_id"] for r in regions}:
                parts=[r for r in regions if r["die_id"]==die_id]
                self.assertEqual([r["position"] for r in parts],list(POSITION_ORDER))
                self.assertEqual(len({r["image_id"] for r in parts}),1)
                self.assertTrue(all(r["image_binding"]=="shared_die_mock" for r in parts))
        self.assertEqual(original,data.catalog())

    def test_selection_toggle_change_reset_and_overlay_preference(self):
        regions=data.inspection_wafer_regions("BATCH-DEMO-060")
        state={"ia_overlay":False}
        set_selection(state,regions,"22","lt")
        self.assertEqual((state["selected_die_id"],state["selected_position"]),("22","lt"))
        self.assertEqual(state["selected_image_id"],"IMG-DEMO-1-10")
        set_selection(state,regions,"24","rb")
        self.assertEqual((state["selected_die_id"],state["selected_position"]),("24","rb"))
        self.assertEqual(state["selected_image_id"],"IMG-DEMO-1-12")
        set_selection(state,regions,"24","rb")
        self.assertIsNone(state["selected_die_id"])
        self.assertIsNone(state["selected_position"])
        self.assertEqual(state["selected_image_id"],"IMG-DEMO-1-01")
        set_selection(state,regions,"38","c")
        set_selection(state,regions)
        self.assertIsNone(state["selected_die_id"])
        self.assertEqual(state["_pending_map_image"],"IMG-DEMO-1-01")
        self.assertFalse(state["ia_overlay"])

    def test_plot_regions_circle_and_no_criticality_dimming(self):
        default=build_inspection_wafer("BATCH-DEMO-060")
        selected=build_inspection_wafer("BATCH-DEMO-060","22","lt")
        self.assertEqual(sum(len(t.x) for t in default.data),24)
        self.assertEqual(len(default.layout.annotations),24)
        self.assertEqual(default.layout.shapes[0].type,"circle")
        self.assertEqual({r[0] for t in default.data for r in t.customdata},
                         {f"{row}{col}" for row in range(1,4) for col in range(1,9)})
        self.assertEqual(len({x for t in default.data for x in t.x}),8)
        self.assertEqual(len({round(base+height/2,8) for t in default.data
                             for base,height in zip(t.base,t.y)}),3)
        for rect in default.layout.shapes[1:]:
            for x in (rect.x0,rect.x1):
                for y in (rect.y0,rect.y1):
                    self.assertLessEqual(x*x+(y-3.5)**2,4.65**2)
        self.assertEqual(len(selected.layout.shapes),len(default.layout.shapes)+1)
        self.assertEqual([t.marker.color for t in selected.data],[t.marker.color for t in default.data])
        for trace in selected.data:
            self.assertEqual(trace.selected.marker.opacity,1)
            self.assertEqual(trace.unselected.marker.opacity,1)
        self.assertEqual(set(CRITICALITY_STYLES),set(PatternCriticality))

    def test_manual_image_selector_updates_die_and_center_region(self):
        at=self.app().switch_page("pages/inspection_analysis.py").run()
        at.selectbox(key="ia_image").select("IMG-DEMO-1-17").run()
        self.assertEqual(len(at.exception),0)
        self.assertEqual(at.session_state["selected_die_id"],"31")
        self.assertEqual(at.session_state["selected_position"],"c")
        self.assertEqual(at.session_state["selected_image_id"],"IMG-DEMO-1-17")

    def test_previous_layout_selection_is_reset_without_changing_overlay(self):
        at=self.app().switch_page("pages/inspection_analysis.py").run()
        at.toggle(key="ia_overlay").set_value(False).run()
        at.session_state["selected_die_id"]="62"
        at.session_state["selected_position"]="c"
        at.session_state["_inspection_map_layout"]="3cols_8rows"
        at.run()
        self.assertEqual(len(at.exception),0)
        self.assertIsNone(at.session_state["selected_die_id"])
        self.assertIsNone(at.session_state["selected_position"])
        self.assertEqual(at.session_state["selected_image_id"],"IMG-DEMO-1-01")
        self.assertFalse(at.toggle(key="ia_overlay").value)


if __name__=="__main__":
    unittest.main()
