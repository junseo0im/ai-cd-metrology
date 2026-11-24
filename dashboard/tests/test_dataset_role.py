"""Dataset Role is a preview-only batch field, independent of upload mode."""
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from dashboard import data_provider as data


class DatasetRoleTests(unittest.TestCase):
    def app(self):
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30)
        at.session_state["data_control_selected_source"]="Mock"  # TASK18-B-8: Mock DCC only when explicitly selected.
        at.run()
        at.switch_page("pages/data_control.py").run()
        self.assertEqual(len(at.exception),0)
        return at

    def submit(self,at):
        next(b for b in at.button if b.label=="Mock Metadata 미리보기").click().run()
        self.assertEqual(len(at.exception),0)
        return at.session_state["mock_metadata_preview"]

    def test_preview_contract_defaults_codes_and_validation(self):
        metadata=dict(batch_id="TEST-BATCH",wafer_id="TEST-WAFER",physical_site_id="TEST-SITE",
                      mode="Wafer Batch",exposure_example=60)
        self.assertEqual(data.DATASET_ROLES,("EVALUATION","TRAINING_CANDIDATE","DEVELOPMENT"))
        self.assertEqual(data.upload_batch_preview(**metadata)["dataset_role"],"EVALUATION")
        before=data.catalog()
        for role in data.DATASET_ROLES:
            preview=data.upload_batch_preview(**metadata,dataset_role=role)
            self.assertEqual(preview["dataset_role"],role)
            self.assertEqual(preview["mode"],"Wafer Batch")
            self.assertFalse(preview["saved_to_database"])
            self.assertNotIn("split",preview)
        with self.assertRaises(ValueError):
            data.upload_batch_preview(**metadata,dataset_role="UNKNOWN")
        self.assertEqual(data.catalog(),before)

    def test_all_role_mode_combinations_are_independent_and_reach_preview(self):
        at=self.app()
        self.assertEqual(at.radio(key="dataset_role").value,"EVALUATION")
        previous_role="EVALUATION"
        before=data.catalog()
        for mode in ["Wafer Batch","Repeatability","Lighting"]:
            at.radio(key="input_mode").set_value(mode).run()
            self.assertEqual(at.radio(key="dataset_role").value,previous_role)
            for role in data.DATASET_ROLES:
                at.radio(key="dataset_role").set_value(role).run()
                self.assertEqual(at.radio(key="input_mode").value,mode)
                self.assertEqual(at.radio(key="dataset_role").value,role)
                preview=self.submit(at)
                self.assertEqual(preview["mode"],mode)
                self.assertEqual(preview["dataset_role"],role)
                self.assertFalse(preview["saved_to_database"])
                previous_role=role
        self.assertEqual(data.catalog(),before)

    def test_registration_preview_keeps_ids_unresolved_and_exposure(self):
        at=self.app()
        self.assertEqual([t.label for t in at.tabs],["이미지 등록","파일명 확인·수정"])
        self.assertEqual(len(at.tabs[0].text_input),0)
        at.selectbox(key="input_exposure").select(70)
        self.submit(at)
        at.radio(key="dataset_role").set_value("TRAINING_CANDIDATE").run()
        at.radio(key="input_mode").set_value("Lighting").run()
        preview=self.submit(at)
        for field in ("batch_id","wafer_id","physical_site_id"):
            self.assertIsNone(preview[field])
        self.assertEqual(preview["exposure_example"],70)
        self.assertEqual(preview["mode"],"Lighting")
        self.assertEqual(preview["dataset_role"],"TRAINING_CANDIDATE")
        self.assertEqual(preview["source"],"mock_input_preview")


if __name__=="__main__":
    unittest.main()
