"""Focused preview-only Data Control input tests; no dataset/DB access."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from dashboard import data_provider as data


class DataInputTabsTests(unittest.TestCase):
    def app(self):
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run()
        at.switch_page("pages/data_control.py").run()
        self.assertEqual(len(at.exception),0)
        return at

    def click(self,at,label):
        next(b for b in at.button if b.label==label).click().run()
        self.assertEqual(len(at.exception),0)

    def test_exposure_options_preview_and_existing_card_layout(self):
        at=self.app()
        source=(ROOT/"dashboard/pages/data_control.py").read_text(encoding="utf-8")
        for key in ("card_data_input","card_data_pipeline","card_data_batches","card_data_progress"):
            self.assertIn('key="'+key+'"',source)
        self.assertIn('height=480,key="card_data_input"',source)
        for exposure in (50,60,70,80):
            at.selectbox(key="input_exposure").select(exposure)
            self.click(at,"Mock Metadata 미리보기")
            self.assertEqual(at.session_state["mock_metadata_preview"]["exposure_example"],exposure)

    def test_upload_is_notice_only(self):
        at=self.app()
        before=data.catalog()
        self.click(at,"이미지 업로드")
        self.assertTrue(any("미연결" in i.value and "ZIP" in i.value for i in at.info))
        self.assertNotIn("mock_metadata_preview",at.session_state)
        self.assertEqual(data.catalog(),before)

    def test_mock_selection_loads_metadata_without_automatic_correction(self):
        at=self.app()
        items=data.filename_review_items()
        self.assertEqual(len(data.REVIEW_DIE_CODES),24)
        self.assertEqual(set(data.REVIEW_DIE_CODES),
                         {f"{row}{col}" for row in range(1,4) for col in range(1,9)})
        for item in items:
            key=item["mock_image_id"]
            at.radio(key="filename_review_image").set_value(key).run()
            self.assertEqual(len(at.exception),0)
            metadata=item["metadata"]
            self.assertEqual(at.text_input(key="review_wafer_"+key).value,metadata["wafer_id"] or "")
            self.assertEqual(at.selectbox(key="review_die_"+key).value,metadata["die_code"])
            self.assertEqual(at.selectbox(key="review_position_"+key).value,metadata["position_code"])
            self.assertEqual(at.text_input(key="review_defect_"+key).value,metadata["legacy_defect_code"] or "")

    def test_edits_are_per_image_session_previews_only(self):
        at=self.app()
        before=data.filename_review_items()
        catalog=data.catalog()
        key=before[0]["mock_image_id"]
        at.text_input(key="review_wafer_"+key).set_value("W9")
        at.selectbox(key="review_die_"+key).select("38")
        at.selectbox(key="review_position_"+key).select("rb")
        at.text_input(key="review_defect_"+key).set_value("custom")
        self.click(at,"수정 내용 확인")
        preview=at.session_state["filename_review_previews"][key]
        self.assertEqual(preview["wafer_id"],"W9")
        self.assertEqual(preview["die_code"],"38")
        self.assertEqual(preview["position_code"],"rb")
        self.assertEqual(preview["legacy_defect_code"],"custom")
        self.assertEqual(preview["original_filename"],before[0]["filename"])
        self.assertEqual(preview["review_status"],"preview_only")
        self.assertFalse(preview["saved_to_database"])
        self.assertFalse(preview["file_renamed"])
        at.radio(key="filename_review_image").set_value(before[1]["mock_image_id"]).run()
        at.radio(key="filename_review_image").set_value(key).run()
        self.assertEqual(at.text_input(key="review_wafer_"+key).value,"W9")
        self.assertEqual(at.selectbox(key="review_die_"+key).value,"38")
        self.assertEqual(at.selectbox(key="review_position_"+key).value,"rb")
        self.assertEqual(data.filename_review_items(),before)
        self.assertEqual(data.catalog(),catalog)


if __name__=="__main__":
    unittest.main()
