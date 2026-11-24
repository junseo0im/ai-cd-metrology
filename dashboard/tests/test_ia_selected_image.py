"""TASK 9 I&A selected-image integration. All writes stay in isolated fixture DBs."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import unittest
from streamlit.testing.v1 import AppTest
from dashboard import data_provider as mock
from dashboard.sqlite_provider import runtime_provider
from dashboard.components.inspection_wafer import select_die
import dashboard.tests.test_wafer_browser as fixtures

ROOT=Path(__file__).resolve().parents[2]


class IASelectedImageTests(unittest.TestCase):
    setUp=fixtures.WaferProviderTests.setUp
    seed=fixtures.WaferProviderTests.seed

    def app(self):
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run()
        at.switch_page("pages/inspection_analysis.py").run()
        at.run()  # TASK18-B-6: SQLite is the fixed Wafer Map source (no selector).
        self.assertFalse(at.exception)
        return at

    def choose_die(self,at,die,wafer="W1"):
        select_die(at.session_state,"ia_wafer_map_SQLite_"+wafer+"_all",die,popover=True)
        at.run()
        self.assertFalse(at.exception)

    def position(self,at):
        return next(row for row in at.get("button_group") if row.key.startswith("ia_sqlite_position_"))

    def row(self,at):
        iid=at.session_state["ia_sqlite_selected_image_id"]
        return runtime_provider().image_record(iid) if iid else None

    def test_sqlite_requires_die_selection_and_never_shows_mock_original(self):
        self.seed("A",["w1_11_lt.png"])
        import streamlit
        real=streamlit.image;seen=[]
        at=self.app()
        with patch("streamlit.image",side_effect=lambda value,*a,**kw:(seen.append(value),real(value,*a,**kw))[1]):
            at.run()
        self.assertFalse(at.exception)
        # TASK18-B-14B: the first Die with registered images opens immediately; only its real original is drawn.
        row=self.row(at)
        self.assertEqual((row["wafer_id"],row["die_code"],row["position_code"]),("W1","11","lt"))
        self.assertEqual([Path(str(v)).name for v in seen],["w1_11_lt.png"])
        self.assertTrue(all("runtime_data" not in str(v) or ".tmp" in str(v) for v in seen))   # Never a Mock asset.
        self.assertEqual(at.toggle(key="ia_metrology_overlay").value,True)
        self.assertEqual(at.toggle(key="ia_defect_overlay").value,True)   # TASK18-B-15: both overlays by default.

    def test_die_click_keeps_popup_and_opens_native_lt_then_position_transitions(self):
        self.seed("A",["w1_11_lt.png","w1_11_c.png","w1_11_rb.png"])
        at=self.app();self.choose_die(at,"11")
        self.assertTrue(at.session_state["ia_wafer_map_SQLite_W1_all_popover"])
        self.assertEqual(at.session_state["active_page_key"],"inspection_analysis")
        self.assertEqual(self.position(at).value,"lt")
        for position in ("lt","c","rb"):
            self.position(at).set_value(position).run()
            self.assertFalse(at.exception)
            row=self.row(at)
            self.assertEqual((row["wafer_id"],row["die_code"],row["position_code"]),("W1","11",position))
            self.assertTrue(any(row["original_filename"] in text.value and row["image_id"] in text.value for text in at.caption))
            self.assertEqual(at.session_state["ia_sqlite_selected_location"]["physical_site_id"],row["physical_site_id"])

    def test_new_die_and_repeated_die_click_reset_lt_and_clear_old_empty_image(self):
        self.seed("A",["w1_11_lt.png","w1_11_c.png","w1_12_c.png"])
        at=self.app();self.choose_die(at,"11")
        self.position(at).set_value("c").run()
        self.choose_die(at,"12")
        # TASK18-B-14B: Die 12 has no lt image, so it opens on c (its own image), never Die 11's.
        self.assertEqual(self.position(at).value,"c")
        self.assertEqual(self.row(at)["die_code"],"12")
        self.position(at).set_value("lt").run()
        self.assertIsNone(self.row(at))
        self.assertTrue(any(row.value=="등록 이미지 없음" for row in at.info))
        self.position(at).set_value("c").run()
        self.assertEqual(self.row(at)["die_code"],"12")
        self.choose_die(at,"11")
        self.assertEqual(self.position(at).value,"lt")
        self.position(at).set_value("c").run()
        self.choose_die(at,"11")
        self.assertEqual(self.position(at).value,"lt")
        self.assertEqual(self.row(at)["position_code"],"lt")

    def test_multiple_images_use_real_ids_batches_and_recorded_acquisition_fields(self):
        self.seed("A",["w1_11_lt.png"])
        other=self.seed("B",["w1_11_lt.png"],mode="repeatability")
        p=runtime_provider();p.rename_batch("B","반복촬영")
        at=self.app();self.choose_die(at,"11")
        selector=next(row for row in at.selectbox if row.key.startswith("ia_sqlite_image_"))
        self.assertEqual(len(selector.options),2)
        self.assertTrue(any("반복촬영" in label and "반복 1" in label and "노광 60" in label for label in selector.options))
        selector.select(other["image_ids"][0]).run()
        self.assertEqual(at.session_state["ia_sqlite_selected_image_id"],other["image_ids"][0])
        self.assertEqual(self.row(at)["batch_id"],"B")
        self.assertFalse(at.exception)

    def test_wafer_change_clears_selection_and_mock_batch_filter_does_not_scope_sqlite(self):
        self.seed("A",["w1_11_lt.png"])
        other=self.seed("B",["w5_11_lt.png"],wafer="W5")
        at=self.app();self.choose_die(at,"11")
        self.assertFalse(any(s.key=="inspection_batch" for s in at.selectbox))  # TASK18-B-8: no demo Batch selector.
        self.assertEqual(self.row(at)["wafer_id"],"W1")
        at.selectbox(key="ia_wafer_wafer_SQLite").select("W5").run()
        self.assertEqual(self.row(at)["wafer_id"],"W5")              # Never the previous Wafer's image.
        self.choose_die(at,"11",wafer="W5")
        self.assertEqual(at.session_state["ia_sqlite_selected_image_id"],other["image_ids"][0])
        self.assertEqual(self.row(at)["wafer_id"],"W5")

    def test_original_path_and_all_overlay_combinations_leave_source_and_db_unchanged(self):
        self.seed("A",["w1_11_lt.png"])
        at=self.app();self.choose_die(at,"11")
        record=self.row(at);path=Path(record["storage_path"])
        before=hashlib.sha256(path.read_bytes()).hexdigest();db_before=self.db_path.read_bytes()
        import streamlit
        real=streamlit.image;seen=[]
        with patch("streamlit.image",side_effect=lambda value,*a,**kw:(seen.append(value),real(value,*a,**kw))[1]):
            for m,d in ((False,False),(True,False),(False,True),(True,True)):
                at.toggle(key="ia_metrology_overlay").set_value(m)
                at.toggle(key="ia_defect_overlay").set_value(d).run()
                self.assertFalse(at.exception)
                self.assertTrue(any("분석 결과 없음" in row.value and "원본만" in row.value for row in at.caption))
        self.assertTrue(seen)
        self.assertTrue(all(Path(value)==path for value in seen))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
        self.assertEqual(self.db_path.read_bytes(),db_before)

    def test_missing_original_warns_without_showing_another_file(self):
        self.seed("A",["w1_11_lt.png"])
        at=self.app();self.choose_die(at,"11")
        path=Path(self.row(at)["storage_path"])
        exists=Path.is_file
        with patch.object(Path,"is_file",lambda value:False if value==path else exists(value)):
            at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("원본 파일을 찾거나" in row.value for row in at.warning))

    def test_single_image_needs_no_extra_selection_and_real_detail_keeps_batch(self):
        result=self.seed("A",["w1_11_lt.png"])
        at=self.app();self.choose_die(at,"11")
        self.assertFalse(any(row.key.startswith("ia_sqlite_image_") for row in at.selectbox))
        at.button(key="ia_sqlite_detail").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["active_page_key"],"image_detail")
        self.assertIn(result["image_ids"][0],[row.value for row in at.code])
        at.switch_page("pages/image_detail.py")
        next(row for row in at.button if row.label=="← Batch 상세로 돌아가기").click().run()
        self.assertEqual(at.session_state["selected_sqlite_batch_id"],"A")
        self.assertEqual(at.title[0].value,"Batch 상세")

    def test_sqlite_empty_results_and_mock_image_overlay_preference_are_preserved(self):
        self.seed("A",["w1_11_lt.png"])
        at=self.app();self.choose_die(at,"11")
        self.assertTrue(any("저장된 분석 결과가 없습니다" in row.value for row in at.info))
        self.assertFalse(any(row.value.equals(mock.design_comparison_table()) for row in at.dataframe))
        at.session_state["_wafer_source"]="Mock";at.run()  # Mock only via explicit session (no UI).
        at.selectbox(key="ia_image").select("IMG-DEMO-1-17").run()
        at.toggle(key="ia_overlay").set_value(False).run()
        at.session_state["_wafer_source"]="SQLite";at.run()
        at.session_state["_wafer_source"]="Mock";at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.selectbox(key="ia_image").value,"IMG-DEMO-1-17")
        self.assertFalse(at.toggle(key="ia_overlay").value)
        self.assertIsNone(at.session_state["ia_sqlite_selected_image_id"])


if __name__=="__main__":
    unittest.main()
