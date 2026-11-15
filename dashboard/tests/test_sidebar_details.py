"""Sidebar/detail extension: shared content and existing route reuse."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from dashboard import data_provider as data
from dashboard.navigation import PAGE_SPECS,COMMON

EXPERIMENT_TABS=["장비 사양","Wafer / Die / Pattern 구조","공정 조건",
                 "Mask 설계 / 제작값","오차율 계산 기준(허용값)","Calibration"]


class SidebarDetailTests(unittest.TestCase):
    def app(self,page="main_home"):
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run()
        if page!="main_home":
            at.switch_page(PAGE_SPECS[page][0]).run()
        self.assertEqual(len(at.exception),0)
        return at

    def test_common_menu_order_and_existing_experiment_route(self):
        keys=["experiment_environment","critical_criteria","ai_usage","model_selection"]
        start=COMMON.index("model_performance")+1
        self.assertEqual(COMMON[start:start+4],keys)
        self.assertEqual(COMMON[start+4],"settings")
        self.assertEqual(PAGE_SPECS["experiment_environment"][0],"pages/experiment_environment.py")
        at=self.app()
        for page in ["main_home","data_control","inspection_analysis",*keys]:
            at.switch_page(PAGE_SPECS[page][0]).run()
            self.assertEqual(len(at.exception),0)
            labels=[link.proto.label for link in at.sidebar.get("page_link")]
            for key in keys:
                self.assertEqual(labels.count(PAGE_SPECS[key][1]),1)
            self.assertLess(labels.index("모델 성능"),labels.index("실험 환경 및 조건"))
            self.assertLess(labels.index("모델 선정 기준"),labels.index("설정"))

    def test_calibration_page_and_tab_reuse_one_renderer_and_provider(self):
        from dashboard.components.calibration import render_calibration
        reference=data.calibration_records().copy()
        reference["test_only_note"]="shared source"
        with patch("dashboard.components.calibration.render_calibration",wraps=render_calibration) as renderer:
            with patch.object(data,"calibration_records",return_value=reference):
                at=self.app("calibration")
                self.assertTrue(any(df.value.equals(reference) for df in at.dataframe))
                at.switch_page(PAGE_SPECS["experiment_environment"][0]).run()
                self.assertEqual(len(at.exception),0)
                self.assertEqual([t.label for t in at.tabs],EXPERIMENT_TABS)
                self.assertTrue(any(df.value.equals(reference) for df in at.dataframe))
        self.assertEqual(renderer.call_count,2)

    def test_mlflow_tab_and_model_page_use_same_renderer(self):
        from dashboard.components.mlflow import render_mlflow
        with patch("dashboard.components.mlflow.render_mlflow",wraps=render_mlflow) as renderer:
            at=self.app("inspection_analysis")
            ia_body=next(m.value for m in at.markdown if "MLflow · 미연결" in m.value)
            at.switch_page(PAGE_SPECS["model_selection"][0]).run()
            self.assertEqual(len(at.exception),0)
            model_body=next(m.value for m in at.markdown if "MLflow · 미연결" in m.value)
            self.assertEqual(ia_body,model_body)
            self.assertEqual([e.label for e in at.expander],
                             ["모델 후보","선정 지표","성능 비교","선정 이유","최종 모델 및 버전"])
        self.assertEqual(renderer.call_count,2)

    def test_critical_and_ai_tabs_are_placeholders(self):
        for key,labels in [
            ("critical_criteria",["측정 Critical 기준","결함 Critical 기준"]),
            ("ai_usage",["Before & After AI","AI Tool"])
        ]:
            at=self.app(key)
            self.assertEqual([t.label for t in at.tabs],labels)
            self.assertTrue(any("Main Home으로 돌아가기"==b.label for b in at.button))
            links=[l.proto.label for l in at.get("page_link")]
            self.assertIn("Data Control Center",links)
            self.assertIn("Inspection & Analysis",links)

    def test_existing_repeatability_lighting_routes_have_correct_mode(self):
        for key,mode,batch in [
            ("repeatability","Repeatability","BATCH-DEMO-REP"),
            ("lighting","Lighting","BATCH-DEMO-LGT")
        ]:
            at=self.app(key)
            self.assertEqual(at.title[0].value,mode)
            self.assertEqual(at.session_state["selected_batch_id"],batch)
            self.assertTrue(next(b for b in at.button if b.label==mode).disabled)
            labels=[l.proto.label for l in at.sidebar.get("page_link")]
            self.assertIn("반복촬영 분석",labels)
            self.assertIn("밝기 분석",labels)


if __name__=="__main__":
    unittest.main()
