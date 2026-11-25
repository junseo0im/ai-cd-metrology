"""No real datasets/APIs/training. Run with dashboard's isolated environment."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from dashboard import data_provider as data
from dashboard.navigation import PAGE_SPECS
from streamlit.testing.v1 import AppTest


class MockDataTests(unittest.TestCase):
    def test_ids_and_assets_are_complete(self):
        catalog = data.catalog()
        batch_ids = {r["batch_id"] for r in catalog["batches"]}
        image_ids = [r["image_id"] for r in catalog["images"]]
        self.assertEqual(len(image_ids),len(set(image_ids)))
        calibration_ids = {r["calibration_id"] for r in catalog["calibrations"]}
        for row in catalog["images"]:
            self.assertTrue(all(k in row for k in data.IDENTITY_FIELDS))
            self.assertIn(row["batch_id"],batch_ids)
            self.assertTrue(row["calibration_id"] is None or row["calibration_id"] in calibration_ids)
            self.assertTrue(data.asset(row["asset"]).is_file())
            self.assertTrue(data.asset(row["overlay_asset"]).is_file())

    def test_provider_is_read_only_and_assets_cannot_escape(self):
        original = data.catalog()
        copy = data.catalog()
        copy["images"][0]["image_id"] = "changed"
        self.assertEqual(original,data.catalog())
        with self.assertRaises(ValueError):
            data.asset("../../README.md")

    def test_comparison_sites_and_labels(self):
        for mode,expected in [("Repeatability",["R1","R2","R3"]),("Lighting",["lg","lm","lb"])]:
            frame = data.images(mode=mode)
            self.assertEqual(frame["label"].tolist(),expected)
            self.assertEqual(frame["physical_site_id"].nunique(),1)
            self.assertEqual(frame["image_id"].nunique(),3)


class PageTests(unittest.TestCase):
    def app(self):
        at = AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=20)
        at.session_state["data_control_selected_source"]="Mock"  # TASK18-B-8: DCC defaults to SQLite; Mock UI tested explicitly.
        at.session_state["_wafer_source"]="Mock"  # TASK18-B-6: operational Wafer Map is SQLite-only; this suite tests the preserved Mock UI explicitly.
        at.run()
        self.assertEqual(len(at.exception),0,list(at.exception))
        return at

    def assert_page_ok(self,at):
        self.assertEqual(len(at.exception),0,list(at.exception))
        if at.session_state["active_page_key"] in {"experiment_environment","critical_criteria"}:
            self.assertTrue(any("LOCAL RECORDS" in c.value for c in at.sidebar.caption))
        elif at.session_state["active_page_key"]=="model_performance":
            self.assertTrue(any("MLFLOW" in c.value for c in at.sidebar.caption))
        elif at.session_state["active_page_key"] in ("main_home","dataset"):  # TASK18-B-9 / 18-B-13A.1: no header caption.
            self.assertTrue(any("SQLITE" in c.value for c in at.sidebar.caption))
        else:
            # TASK18-B-18: no description line under the platform title; the source stays in the sidebar.
            self.assertTrue(any(k in c.value for c in at.sidebar.caption for k in ("SQLITE", "참고용 예시 화면")))
        self.assertFalse(any(c.value.startswith(("참고용 예시 화면 · 운영 SQLite", "SQLite 운영 DB ·", "로컬 JSON 기록 ·",
                                                 "MLflow 실험 기록 조회 ·")) for c in at.caption))

    def button(self,at,label):
        return next(b for b in at.button if b.label == label)

    def click(self,at,label):
        # AppTest.run() does not adopt the page hash produced by st.switch_page.
        # Align with the page visible in the returned tree, as the real client does.
        active = at.session_state["active_page_key"]
        at.switch_page(PAGE_SPECS[active][0])
        self.button(at,label).click().run()

    def test_all_registered_pages_render_with_shared_shell(self):
        at = self.app()
        for key,(path,title,url,icon) in PAGE_SPECS.items():
            with self.subTest(page=key):
                at.switch_page(path).run()
                if key=="image_detail":
                    self.assertFalse(at.exception)
                    self.assertTrue(any("SQLITE" in row.value for row in at.sidebar.caption))
                    self.assertTrue(any("Image를 찾을 수 없습니다" in row.value for row in at.warning))
                else:
                    self.assert_page_ok(at)
                if key == "main_home":
                    self.assertFalse(any("데이터에서 인사이트" in m.value for m in at.markdown))
                    self.assertTrue(any("DATA WORKSPACE" in m.value for m in at.markdown))
                    self.assertTrue(any("ANALYSIS WORKSPACE" in m.value for m in at.markdown))
                else:
                    self.assertTrue(any(e.value == title for e in at.title),title)

    def test_home_card_and_batch_detail_navigation(self):
        at = self.app()
        self.click(at,"Data Control Center 열기")
        self.assert_page_ok(at)
        self.assertEqual(at.title[0].value,"Data Control Center")
        selected = data.batches().iloc[1]["batch_id"]
        self.assertFalse(any(b.label == selected+" →" for b in at.button))
        self.assertTrue(any(b.label==selected for b in at.button))
        self.click(at,selected)
        self.assert_page_ok(at)
        self.assertEqual(at.title[0].value,"Batch 상세")
        self.assertEqual(at.session_state["selected_batch_id"],selected)
        self.assertEqual(at.selectbox(key="detail_batch").value,selected)

    def test_mode_switches_change_comparison_layout(self):
        at = self.app()
        self.click(at,"Inspection & Analysis 열기")
        self.assert_page_ok(at)
        for mode in ["Repeatability","Lighting","Wafer Batch"]:
            self.click(at,mode)
            self.assert_page_ok(at)
            expected = "Inspection & Analysis" if mode=="Wafer Batch" else mode
            self.assertEqual(at.title[0].value,expected)

    def test_metadata_is_preview_only(self):
        at = self.app().switch_page("pages/data_control.py").run()
        before = data.catalog()
        self.button(at,"Mock Metadata 미리보기").click().run()
        self.assert_page_ok(at)
        self.assertFalse(at.session_state["mock_metadata_preview"]["saved_to_database"])
        self.assertEqual(before,data.catalog())

    def test_selected_image_and_pending_map_selection(self):
        at = self.app().switch_page("pages/inspection_analysis.py").run()
        options = data.images("BATCH-DEMO-060")["image_id"].tolist()
        at.selectbox(key="ia_image").select(options[1]).run()
        self.assertEqual(at.session_state["selected_image_id"],options[1])
        at.session_state["_pending_map_image"] = options[2]
        at.run()
        self.assert_page_ok(at)
        self.assertEqual(at.selectbox(key="ia_image").value,options[2])
        self.assertEqual(at.session_state["selected_image_id"],options[2])

    def test_batch_link_replaces_stale_detail_selection(self):
        at = self.app().switch_page("pages/batch_detail.py").run()
        ids = data.batches()["batch_id"].tolist()
        at.selectbox(key="detail_batch").select(ids[1]).run()
        self.assertEqual(at.session_state["selected_batch_id"],ids[1])
        at.query_params["batch_id"] = ids[0]
        at.run()
        self.assert_page_ok(at)
        self.assertEqual(at.selectbox(key="detail_batch").value,ids[0])
        self.assertEqual(at.session_state["selected_batch_id"],ids[0])
        self.assertNotIn("batch_id",at.query_params)

    def test_batch_summary_is_a_table_for_the_selected_batch(self):
        at = self.app().switch_page("pages/inspection_analysis.py").run()
        self.assertTrue(any("Batch 결과 표" in s.value for s in at.subheader))
        table=next(df.value for df in at.dataframe if set(df.value.columns)=={"항목","평균 (µm)","표준편차 (µm)"})
        self.assertEqual(table["항목"].tolist(),["Line Width","Gap","Edge"])
        self.assertTrue(all("예시" in value for value in table["평균 (µm)"]))
        self.assertEqual(set(table["표준편차 (µm)"]),{"데이터 없음"})


class Phase11ProviderTests(unittest.TestCase):
    def test_wafer_grid_has_24_unique_patterns_inside_circle(self):
        from dashboard.components.charts import build_wafer_figure
        for batch_id in data.batches("Wafer Batch")["batch_id"]:
            frame = data.images(batch_id)
            self.assertEqual(len(frame),24)
            self.assertEqual(set(frame["site_row"]),{1,2,3})
            self.assertEqual(set(frame["site_col"]),set(range(1,9)))
            self.assertEqual(frame["pattern_id"].tolist(),[f"P{i:02}" for i in range(1,25)])
            fig = build_wafer_figure(batch_id)
            self.assertEqual(fig.layout.shapes[0].type,"circle")
            self.assertEqual(sum(len(trace.x) for trace in fig.data),24)
            for trace in fig.data:
                for x,y in zip(trace.x,trace.y):
                    self.assertLessEqual((x-3.5)**2+y*y,4.5**2)

    def test_process_aggregation_recomputes_mean_spread_and_count(self):
        import pandas as pd
        original = data.process_observations()
        summary = data.process_trend_summary(original)
        self.assertEqual(len(summary),8)
        self.assertEqual(set(summary["sample_count"]),{4,6})
        self.assertTrue((summary["std"] >= 0).all())
        new = original.iloc[[0]].copy()
        new["is_new"] = True
        new["outer_um_example"] = 45
        updated = data.process_trend_summary(pd.concat([original,new],ignore_index=True))
        selected = updated[(updated["exposure"]==50)&(updated["series"]=="현재 평균 (예시)")].iloc[0]
        self.assertEqual(selected["sample_count"],7)
        self.assertNotEqual(selected["mean"],summary[(summary["exposure"]==50)&
                            (summary["series"]=="현재 평균 (예시)")].iloc[0]["mean"])

    def test_native_theme_and_css_share_light_dark_policy(self):
        import tomllib
        settings = tomllib.loads((ROOT/"dashboard/.streamlit/config.toml").read_text(encoding="utf-8"))
        self.assertEqual(settings["theme"]["base"],"light")
        self.assertIn("dark",settings["theme"])
        self.assertNotEqual(settings["theme"]["dark"]["backgroundColor"],"#000000")
        css = (ROOT/"dashboard/components/styles.css").read_text()
        self.assertIn("light-dark(",css)
        self.assertNotIn("display:none",css.split(".stAppHeader")[1].split("}")[0])

    def test_batch_links_and_status_families(self):
        from dashboard.components.tables import batch_link
        self.assertIn("batch_id=BATCH-DEMO-060",batch_link("BATCH-DEMO-060"))
        self.assertEqual({r["state_key"] for r in data.catalog()["tile_states"]},
                         {"registered","processing","completed","review","error"})


if __name__ == "__main__":
    unittest.main()
