"""I&A Image tab: stored results of Single Image registrations (temporary DB, fake engines).

Registration, the DCC manual run, BatchRunner/AnalysisRunner policy and AnalysisRepository are
real; only the model engines and overlay drawing are fakes (fake_dashboard_runner.py:
Die 13 -> model PD, Die 14 -> defect engine failure, else 0 defects). The Image tab only reads.
"""
from contextlib import closing
import os
import sqlite3
import sys
import unittest
from unittest.mock import patch

import dashboard.components.analysis_management as am
import dashboard.components.ia_analysis_result as ia
import dashboard.components.ia_single_images as single
from dashboard.components.inspection_wafer import select_die
from dashboard.tests.test_multi_analysis import FAKE, TABLES
from dashboard.tests.test_upload_default_matching import Base, HAS_PLOTLY


def result_table(at):
    """The Image tab's '항목 | 결과' table: the first one after the Single list (AppTest hides its key)."""
    frames = [getattr(d.value, "data", d.value) for d in at.dataframe]  # Styled table -> its frame.
    start = next(i for i, d in enumerate(at.dataframe) if d.key == single.LIST_KEY)
    frame = next(f for f in frames[start + 1:] if list(f.columns) == ["항목", "결과"])
    return dict(zip(frame["항목"], frame["결과"]))


@unittest.skipUnless(HAS_PLOTLY, "plotly not installed; I&A / Data Control pages cannot load in AppTest")
class ImageTabTests(Base):
    def setUp(self):
        super().setUp()
        # One PNG per Single Image registration (existing rule): normal, PD (Die 13), failure (Die 14), never analysed.
        self.normal = self.register(self.stage([self.png("w1_21_c.png", 201)]), "single_image")[1]["image_ids"][0]
        self.pd = self.register(self.stage([self.png("w1_13_c.png", 202)]), "single_image")[1]["image_ids"][0]
        self.failed = self.register(self.stage([self.png("w1_14_c.png", 203)]), "single_image")[1]["image_ids"][0]
        self.unanalysed = self.register(self.stage([self.png("w1_22_c.png", 204)]), "single_image")[1]["image_ids"][0]
        self.wafer_image = self.register(self.stage([self.png("w1_31_c.png", 205)]), "wafer_batch")[1]["image_ids"][0]
        self.out = self.f.root / "results"; self.out.mkdir()
        for p in (patch.dict(os.environ, {"IDA_DB_PATH": str(self.f.db), "IDA_STORAGE_ROOT": str(self.f.sources),
                                          "IDA_STAGING_ROOT": str(self.f.root / "staging"),
                                          "IDA_ANALYSIS_PYTHON": sys.executable,
                                          "IDA_ANALYSIS_OUTPUT_ROOT": str(self.out),
                                          # Team judge rules from the repository copy (runtime_data is Git-excluded).
                                          "IDA_TEAM_UNET_DIR": str(am.ROOT / "engines/linewidth/unet_linewidth_metrology_v2")}),
                  patch.object(am, "_analysis_command", lambda python, request, output:
                               [str(python), str(FAKE), "--request", str(request), "--output-root", str(output)]),
                  patch("streamlit.file_uploader", return_value=[])):
            p.start(); self.addCleanup(p.stop)
        os.environ.pop("FAKE_RUNNER_MODE", None); am._ACTIVE.clear()

    def rows(self):
        with closing(sqlite3.connect(self.f.db)) as c:
            return {t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("images", *TABLES)}

    def analyse(self):
        """Existing DCC manual run, Image scope, three Single Images in one run (one Batch each)."""
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(am.ROOT / "dashboard/app.py", default_timeout=60).run()
        at.switch_page("pages/data_control.py").run()
        at.session_state["data_control_selected_source"] = "SQLite"; at.run()
        at.radio(key="dcc_analysis_scope").set_value("Image").run()
        at.selectbox(key="dcc_analysis_target_image").select(self.normal).run()
        at.multiselect(key="dcc_analysis_extra_images").set_value([self.pd, self.failed]).run()
        at.button(key="dcc_analysis_run").click().run()
        self.assertFalse(at.exception)

    def ia(self, index=None, draw=True):
        """draw=False: skip st.image (the fake engines write placeholder bytes, not PNG overlays)."""
        from contextlib import ExitStack
        from streamlit.testing.v1 import AppTest
        with ExitStack() as stack:
            stack.enter_context(patch.object(single, "_selected_index",
                                             (lambda count: index) if index is not None else single._selected_index))
            if not draw:
                stack.enter_context(patch("streamlit.image", return_value=None))
            at = AppTest.from_file(am.ROOT / "dashboard/app.py", default_timeout=60).run()
            at.switch_page("pages/inspection_analysis.py").run()
            at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def index_of(self, filename):
        with closing(sqlite3.connect(self.f.db)) as c:
            names = [r[0] for r in c.execute(
                "SELECT i.original_filename FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id "
                "WHERE b.upload_mode='single_image' ORDER BY i.created_at DESC,i.image_id")]
        return names.index(filename)

    def listed(self, at):
        return next(d.value for d in at.dataframe if d.key == single.LIST_KEY)

    def captions(self, at):
        return " | ".join(c.value for c in at.caption)

    def test_01_tabs_and_single_list_newest_first_without_wafer_batch_images(self):
        at = self.ia()
        self.assertEqual([t.label for t in at.tabs][:3], ["Wafer", "Image", "선폭 / 오차율 / 결함"])
        self.assertFalse(any(t.label in ("Wafer / Image", "Repeatability", "Lighting") for t in at.tabs))
        frame = self.listed(at)
        with closing(sqlite3.connect(self.f.db)) as c:  # Expected order = provider order: created_at DESC, image_id.
            expected = [r[0] for r in c.execute(
                "SELECT i.original_filename FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id "
                "WHERE b.upload_mode='single_image' ORDER BY i.created_at DESC,i.image_id")]
        self.assertEqual(frame["파일명"].tolist(), expected)
        self.assertEqual(sorted(expected), ["w1_13_c.png", "w1_14_c.png", "w1_21_c.png", "w1_22_c.png"])
        self.assertNotIn("w1_31_c.png", frame["파일명"].tolist())                     # Wafer Batch image not listed.
        self.assertTrue(all(v.startswith("W1 / ") and v.endswith(" / c") for v in frame["Wafer / Die / 위치"]))
        self.assertTrue(all(s == "미분석 · 저장된 분석 결과 없음" for s in frame["분석 상태"]))
        self.assertIn("4장", self.captions(at))

    def test_02_states_after_one_run_and_per_image_results(self):
        before = self.rows()
        self.analyse()
        after_run = self.rows()
        self.assertEqual(after_run["analysis_image_runs"], before["analysis_image_runs"] + 2)  # Failure not saved.
        with closing(sqlite3.connect(self.f.db)) as c:
            outcome = c.execute("SELECT outcome FROM analysis_image_runs WHERE image_id=?", (self.normal,)).fetchone()[0]
        # Existing wording for the stored outcome (fake engines give COMPLETE or QUALITY_REVIEW_REQUIRED).
        self.assertIn(outcome, ("COMPLETE", "QUALITY_REVIEW_REQUIRED"))
        normal_text = ia.STATUS_LABELS[outcome][0]
        at = self.ia()
        status = dict(zip(self.listed(at)["파일명"], self.listed(at)["분석 상태"]))
        self.assertEqual(status["w1_21_c.png"], normal_text)
        self.assertEqual(status["w1_13_c.png"], "사용 불가 · PD 검출")
        self.assertEqual(status["w1_22_c.png"], "미분석 · 저장된 분석 결과 없음")
        # Normal image: stored values, 0 defects shown as a completed result.
        at = self.ia(self.index_of("w1_21_c.png"), draw=False)
        table = result_table(at)
        self.assertEqual(table["분석 상태"], normal_text)
        self.assertEqual(table["결함 개수"], "0건")
        self.assertIn("µm", table["Line Width 평균 (측정 수)"])
        self.assertIn("결함 검출 0건 · Overlay에 표시할 결함 없음", self.captions(at))
        self.assertIn(self.normal, self.captions(at))
        # PD image: metrology excluded with the existing reason; defect listed.
        at = self.ia(self.index_of("w1_13_c.png"))
        table = result_table(at)
        self.assertEqual(table["분석 상태"], "사용 불가 · PD 검출")
        self.assertTrue(any(i.value == "선폭 Overlay 미생성" for i in at.info))
        self.assertIn("PD 검출로 선폭 계측 제외", self.captions(at))
        self.assertTrue(any(e.label.startswith("결함 검출 상세") for e in at.expander))
        # Never analysed: placeholder text, no error.
        at = self.ia(self.index_of("w1_22_c.png"))
        self.assertEqual(result_table(at)["분석 상태"], "미분석 · 저장된 분석 결과 없음")
        self.assertTrue(any("이 이미지의 저장된 분석 결과가 없습니다" in i.value for i in at.info))
        self.assertEqual(self.rows(), after_run)                                       # Image tab is read-only.

    def test_03_single_images_only_in_image_tab_not_on_wafer_map(self):
        from dashboard.die_assessment import JUDGED_MODES
        self.assertEqual(JUDGED_MODES, ("wafer_batch",))                          # Single: no Die judgment.
        self.register(self.stage([self.png("w2_11_c.png", 206)]), "single_image", wafer_id="W2")  # Single-only Wafer.
        self.analyse()
        from streamlit.testing.v1 import AppTest
        index = self.index_of("w1_21_c.png")
        with patch.object(single, "_selected_index", lambda count: index):   # Image tab: w1_21_c.png
            at = AppTest.from_file(am.ROOT / "dashboard/app.py", default_timeout=60).run()
            at.switch_page("pages/inspection_analysis.py").run(); at.run()
            self.assertEqual(at.selectbox(key="ia_wafer_wafer_SQLite").options, ["W1"])  # W2 holds only a Single.
            select_die(at.session_state, "ia_wafer_map_SQLite_W1_all", "21", popover=True); at.run()
            self.assertFalse(at.exception, [e.value for e in at.exception])
            self.assertIsNone(at.session_state["ia_sqlite_selected_image_id"])   # Die 21 = Single only -> not on the Map.
            self.assertIn(self.normal, self.captions(at))                        # Still shown in the Image tab.
            select_die(at.session_state, "ia_wafer_map_SQLite_W1_all", "31", popover=True); at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        self.assertEqual(at.session_state["ia_sqlite_selected_image_id"], self.wafer_image)  # Wafer Batch kept.
        self.assertNotIn("w1_31_c.png", self.listed(at)["파일명"].tolist())

    def test_05_process_graphs_exclude_single_images(self):
        from dashboard.components.process_analysis import source_metadata
        ids = set(source_metadata("SQLite", self.p)["image_id"])
        self.assertIn(self.wafer_image, ids)                                       # Wafer Batch image kept.
        self.assertFalse(ids & {self.normal, self.pd, self.failed, self.unanalysed})  # No Single Image.

    def test_04_empty_states(self):
        with closing(sqlite3.connect(self.f.db)) as c:
            modes = c.execute("SELECT upload_mode,COUNT(*) FROM upload_batches GROUP BY upload_mode").fetchall()
        self.assertIn(("single_image", 4), modes)
        with patch.object(single, "single_images", side_effect=sqlite3.OperationalError("injected")):
            at = self.ia()
        self.assertTrue(any("Single Image 조회 실패" in e.value for e in at.error))
        with patch.object(single, "single_images", return_value=single.pd.DataFrame()):
            at = self.ia()
        self.assertTrue(any("등록된 Single Image가 없습니다" in i.value for i in at.info))


if __name__ == "__main__":
    unittest.main()
