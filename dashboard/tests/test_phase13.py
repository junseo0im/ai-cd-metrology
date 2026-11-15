"""Phase 1.3: changed cards, placeholders and provider counts only."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from dashboard import data_provider as data
from dashboard.navigation import PAGE_SPECS


class Phase13Tests(unittest.TestCase):
    def app(self,page):
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=20).run()
        at.switch_page(PAGE_SPECS[page][0]).run()
        self.assertEqual(len(at.exception),0)
        return at

    def click(self,at,label):
        at.switch_page(PAGE_SPECS[at.session_state["active_page_key"]][0])
        next(b for b in at.button if b.label==label).click().run()
        self.assertEqual(len(at.exception),0)

    def test_accumulation_recounts_updated_mock_records_without_writing(self):
        original=data.catalog()
        base=data.accumulation_summary()
        self.assertEqual(base["batch_count"],5)
        self.assertEqual(base["image_count"],78)
        self.assertEqual(base["mode_image_counts"],{"Wafer Batch":72,"Repeatability":3,"Lighting":3})
        self.assertEqual(base["recent_image_count"],6)
        changed=data.catalog()
        extra=dict(changed["images"][0],image_id="MOCK-ADDED")
        changed["images"].append(extra)
        updated=data.accumulation_summary(changed)
        self.assertEqual(updated["image_count"],79)
        self.assertEqual(updated["mode_image_counts"]["Wafer Batch"],73)
        self.assertEqual(updated["history"][-1]["image_count"],79)
        self.assertEqual(data.catalog(),original)

    def test_comparison_is_explicitly_unconnected_not_calculated(self):
        table=data.design_comparison_table()
        self.assertEqual(table["항목"].tolist(),["Line","Gap","Edge"])
        self.assertEqual(set(table["측정 (µm)"]),{"미연결"})
        self.assertEqual(set(table["설계 (µm)"]),{"미확정"})
        self.assertEqual(set(table["오차 (%)"]),{"미계산"})

    def test_inspection_tabs_and_simulation_return_preserve_selection_overlay(self):
        at=self.app("inspection_analysis")
        self.assertEqual([t.label for t in at.tabs],["공정 조건별 그래프","MLflow"])
        self.assertEqual(len(at.dataframe),2)
        at.selectbox(key="ia_image").select("IMG-DEMO-1-17").run()
        at.toggle(key="ia_overlay").set_value(False).run()
        self.click(at,"Simulation")
        self.assertEqual(at.title[0].value,"Simulation")
        self.click(at,"Inspection & Analysis로 돌아가기")
        self.assertEqual(at.title[0].value,"Inspection & Analysis")
        self.assertEqual(at.session_state["selected_image_id"],"IMG-DEMO-1-17")
        self.assertEqual(at.session_state["selected_die_id"],"31")
        self.assertFalse(at.toggle(key="ia_overlay").value)

    def test_data_control_progress_is_mock_and_pipeline_tiles_remain(self):
        at=self.app("data_control")
        self.assertTrue(any(s.value=="등록·처리 상태 요약" for s in at.subheader))
        self.assertEqual(len(at.get("progress")),1)
        self.assertTrue(any(df.value.equals(data.processing_stage_table()) for df in at.dataframe))
        self.assertTrue(any("class=\"pipeline\"" in m.value for m in at.markdown))
        self.assertTrue(any("class=\"image-tiles\"" in m.value for m in at.markdown))
        self.assertEqual(len(data.processing_stage_table()),4)
        progress=data.processing_progress_mock()
        self.assertGreaterEqual(progress["overall_percent"],0)
        self.assertLessEqual(progress["overall_percent"],100)


if __name__=="__main__":
    unittest.main()
