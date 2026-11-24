"""TASK 11-A target selection; all DB writes are explicit isolated fixture setup."""
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from dashboard import data_provider as mock
from dashboard.sqlite_provider import runtime_provider
from dashboard.components.analysis_management import collect_images,target_preview,render_analysis_management
import dashboard.tests.test_wafer_browser as fixtures
import dashboard.tests.test_data_control_unified as unified

ROOT=Path(__file__).resolve().parents[2]


class AnalysisManagementTests(unittest.TestCase):
    setUp=fixtures.WaferProviderTests.setUp
    seed=fixtures.WaferProviderTests.seed
    cards=unified.UnifiedDataControlTests.cards

    def app(self):
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run()
        at.switch_page("pages/data_control.py").run()
        at.session_state["data_control_selected_source"]="SQLite";at.run()
        self.assertFalse(at.exception)
        return at

    def test_scope_queries_include_all_modes_batches_and_distinct_site_images(self):
        a=self.seed("A",["w1_11_lt.png","w1_12_c.png"])
        b=self.seed("B",["w1_11_lt.png"],mode="repeatability")
        c=self.seed("C",["w1_11_lt.png"],mode="lighting")
        d=self.seed("D",["w1_11_lt.png"],mode="single_image")
        other=self.seed("E",["w5_11_c.png"],wafer="W5")
        p=runtime_provider();before=self.db_path.read_bytes()
        preview=target_preview(p,"Wafer","W1")
        self.assertEqual(preview["image_count"],5)
        self.assertEqual(set(preview["batch_ids"]),{"A","B","C","D"})
        self.assertEqual(set(preview["images"]["upload_mode"]),{"wafer_batch","repeatability","lighting","single_image"})
        self.assertEqual(len(preview["image_ids"]),len(set(preview["image_ids"])))
        self.assertFalse(set(preview["image_ids"])&set(other["image_ids"]))
        self.assertEqual(target_preview(p,"Batch","B")["image_ids"],b["image_ids"])
        self.assertEqual(target_preview(p,"Image",c["image_ids"][0])["image_count"],1)
        self.assertEqual(self.db_path.read_bytes(),before)
        with self.assertRaises(ValueError):target_preview(p,"Batch","not-a-real-id")
        with self.assertRaises(ValueError):target_preview(p,"Wafer","not-a-real-id")
        with self.assertRaises(ValueError):target_preview(p,"Image","not-a-real-id")

    def test_more_than_100_rows_and_equal_hashes_are_not_truncated_or_deduplicated(self):
        result=self.seed("A",["w1_11_c.png"]);p=runtime_provider()
        with closing(p.connect(readonly=False)) as connection,connection:
            cursor=connection.execute("SELECT * FROM images WHERE image_id=?",(result["image_ids"][0],))
            names=[r[0] for r in cursor.description];original=dict(zip(names,cursor.fetchone()))
            for index in range(120):
                record=dict(original,image_id="TARGET-EXTRA-"+str(index))
                connection.execute("INSERT INTO images("+",".join(names)+") VALUES ("+",".join("?" for _ in names)+")",
                                   [record[key] for key in names])
        before=self.db_path.read_bytes()
        for scope,key in [("Batch","A"),("Wafer","W1")]:
            preview=target_preview(p,scope,key)
            self.assertEqual(preview["image_count"],121)
            self.assertEqual(len(set(preview["image_ids"])),121)
            self.assertEqual(preview["images"]["sha256"].nunique(),1)
        self.assertEqual(len(collect_images(p)),121)
        self.assertEqual(self.db_path.read_bytes(),before)

    def test_exact_card_order_heights_and_removed_progress_in_both_modes(self):
        at=self.app()
        for source in ("SQLite","Mock"):
            at.session_state["data_control_selected_source"]=source;at.run()
            cards=self.cards(at)
            self.assertEqual(list(cards),["card_data_input","card_data_batches","card_data_analysis","card_data_pipeline"])
            self.assertEqual([item[1] for item in cards.values()],[480,480,620,620])
            self.assertFalse(any(row.value=="등록·처리 상태 요약" for row in at.subheader))
            self.assertEqual(at.radio(key="dcc_analysis_execution").value,"수동 실행")
            self.assertEqual(at.radio(key="dcc_analysis_scope").value,"Batch")
            self.assertTrue(at.button(key="dcc_analysis_run").disabled)
            self.assertFalse(at.get("progress"))

    def test_batch_wafer_image_switches_refresh_exact_preview_and_collapsed_details(self):
        a=self.seed("A",["w1_11_c.png","w1_11_lt.png"])
        b=self.seed("B",["w1_11_c.png"],mode="repeatability")
        self.seed("C",["w5_11_c.png"],wafer="W5")
        at=self.app()
        at.selectbox(key="dcc_analysis_target_batch").select("A").run()
        self.assertEqual(set(at.session_state["dcc_analysis_preview"]["image_ids"]),set(a["image_ids"]))
        at.radio(key="dcc_analysis_scope").set_value("Wafer").run()
        at.selectbox(key="dcc_analysis_target_wafer").select("W1").run()
        self.assertEqual(at.session_state["dcc_analysis_preview"]["image_count"],3)
        details=next(row for row in at.expander if row.label=="대상 이미지 전체 보기 (3장)")
        self.assertFalse(details.proto.expanded)
        at.radio(key="dcc_analysis_scope").set_value("Image").run()
        at.selectbox(key="dcc_analysis_target_image").select(b["image_ids"][0]).run()
        preview=at.session_state["dcc_analysis_preview"]
        self.assertEqual(preview["image_ids"],b["image_ids"])
        self.assertEqual(preview["images"].iloc[0]["repeat_index"],1)
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertFalse(at.exception)

    def test_renamed_batch_keeps_id_and_automatic_mode_has_no_writes_or_jobs(self):
        a=self.seed("A",["w1_11_c.png"]);self.seed("B",["w1_12_c.png"])
        p=runtime_provider();at=self.app()
        at.selectbox(key="dcc_analysis_target_batch").select("A").run()
        p.rename_batch("A","변경된 이름")
        before=self.db_path.read_bytes()
        at.radio(key="dcc_analysis_execution").set_value("자동 실행").run()
        self.assertEqual(at.selectbox(key="dcc_analysis_target_batch").value,"A")
        self.assertEqual(at.session_state["dcc_analysis_preview"]["target_name"],p.batch_record("A")["display_name"])
        self.assertTrue(any("예약하지" in row.value for row in at.caption))
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.assertEqual(self.db_path.read_bytes(),before)

    def test_mock_clears_preview_and_never_asks_sqlite_for_targets(self):
        self.seed("A",["w1_11_c.png"]);at=self.app()
        self.assertIsNotNone(at.session_state["dcc_analysis_preview"])
        with patch("dashboard.components.analysis_management.runtime_provider",side_effect=AssertionError("Mock must not query SQLite")):
            at.session_state["data_control_selected_source"]="Mock";at.run()
        self.assertFalse(at.exception)
        self.assertIsNone(at.session_state["dcc_analysis_preview"])
        self.assertTrue(any("SQLite 데이터 소스에서" in row.value for row in at.info))
        self.assertTrue(at.button(key="dcc_analysis_run").disabled)

    def test_empty_database_and_registered_wafer_without_images(self):
        p=runtime_provider();at=self.app()
        for scope in ("Batch","Wafer","Image"):
            at.radio(key="dcc_analysis_scope").set_value(scope).run()
            self.assertIsNone(at.session_state["dcc_analysis_preview"])
            self.assertTrue(any("등록된 분석 대상 없음" in row.value for row in at.info))
            self.assertTrue(at.button(key="dcc_analysis_run").disabled)
        self.seed("A",["w1_11_c.png"])
        # Only isolated fixture rows are modified to represent an empty saved Batch.
        with closing(p.connect(readonly=False)) as connection,connection:
            connection.execute("DELETE FROM images")
        before=self.db_path.read_bytes()
        self.assertEqual(target_preview(p,"Batch","A")["image_count"],0)
        self.assertEqual(target_preview(p,"Wafer","W1")["image_count"],0)
        self.assertEqual(self.db_path.read_bytes(),before)

    def test_expand_restore_keeps_target_scope_mode_and_existing_input_state(self):
        result=self.seed("A",["w1_11_c.png"]);at=self.app()
        at.radio(key="dcc_analysis_scope").set_value("Image").run()
        at.selectbox(key="dcc_analysis_target_image").select(result["image_ids"][0]).run()
        at.radio(key="dcc_analysis_execution").set_value("자동 실행").run()
        at.radio(key="dataset_role").set_value("DEVELOPMENT").run()
        for card in ("analysis","batches","pipeline","input"):
            at.button(key="dcc_expand_"+card).click().run()
            self.assertFalse(at.exception)
            self.assertEqual(len(self.cards(at)),4)
            self.assertEqual(at.radio(key="dcc_analysis_execution").value,"자동 실행")
            self.assertEqual(at.radio(key="dcc_analysis_scope").value,"Image")
            self.assertEqual(at.selectbox(key="dcc_analysis_target_image").value,result["image_ids"][0])
            self.assertEqual(at.radio(key="dataset_role").value,"DEVELOPMENT")
            at.button(key="dcc_expand_"+card).click().run()
        at.session_state["dcc_expanded_card"]="progress";at.run()
        self.assertIsNone(at.session_state["dcc_expanded_card"])
        self.assertNotIn("card_data_progress",self.cards(at))


if __name__=="__main__":
    unittest.main()
