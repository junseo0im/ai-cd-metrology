"""Task B: actual UX state transitions using temporary SQLite/PNG fixtures only."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import dashboard.tests.test_sqlite_input as fixtures
import dashboard.tests.test_data_control_unified as unified
from dashboard import data_provider as mock
from dashboard.sqlite_provider import runtime_provider
from database.tests.test_image_registry import png_bytes


class DataControlUXTests(unittest.TestCase):
    setUp=fixtures.SQLiteInputTests.setUp
    app=fixtures.SQLiteInputTests.app
    uploaded=fixtures.SQLiteInputTests.uploaded
    confirm_process=fixtures.SQLiteInputTests.confirm_process
    seed_registered_image=fixtures.SQLiteInputTests.seed_registered_image
    duplicate_choices=fixtures.SQLiteInputTests.duplicate_choices
    cards=unified.UnifiedDataControlTests.cards

    def input_card(self,at):
        return self.cards(at)["card_data_input"][2]

    def toggle(self,at,key):
        at.button(key="dcc_expand_"+key).click().run()
        self.assertFalse(at.exception)

    def test_completed_result_retires_on_mode_change_without_deleting_saved_rows(self):
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        saved=deepcopy(at.session_state["sqlite_last_result"])
        self.assertTrue(any("SQLite 등록 완료" in row.value for row in self.input_card(at).success))
        old_id=saved["image_ids"][0]
        for mode in ("Repeatability","Lighting","Single Image","Wafer Batch"):
            at.radio(key="input_mode").set_value(mode).run()
            self.assertFalse(at.exception)
            self.assertFalse(any("SQLite 등록 완료" in row.value for row in self.input_card(at).success))
            self.assertEqual(runtime_provider().image_count(),1)
            self.assertEqual(runtime_provider().batches()["batch_id"].tolist(),[saved["batch_id"]])
            if mode!="Single Image":
                self.assertNotIn(old_id,[row["image_id"] for row in at.session_state["sqlite_uploads"]["inputs"]])
        at.button(key="sqlite_new_batch").click().run()
        self.assertFalse(at.exception)
        self.assertFalse(any("SQLite 등록 완료" in row.value for row in self.input_card(at).success))
        self.assertEqual(runtime_provider().image_count(),1)
        for path in runtime_provider().images()["storage_path"]:
            self.assertTrue(Path(path).is_file())

    def test_expand_restore_preserves_manual_metadata_duplicate_choice_and_file_selection(self):
        seed=self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png"),self.uploaded("unknown.png",80)]
        at=self.app()
        self.confirm_process(at)
        missing=next(row for row in at.session_state["sqlite_preview"]["images"]
                     if row["original_filename"]=="unknown.png")
        key=missing["image_id"]
        at.selectbox(key="sqlite_edit_die_"+key).select("38")
        at.selectbox(key="sqlite_edit_position_"+key).select("lt")
        next(row for row in at.button if row.label=="수정 후 Preview 재검증").click().run()
        self.assertFalse(at.exception)
        at.radio(key="dataset_role").set_value("DEVELOPMENT").run()
        self.duplicate_choices(at)[0].select("REGISTER_SEPARATELY").run()
        manual=deepcopy(at.session_state["sqlite_uploads"]["manual_metadata"])
        scope=at.session_state["sqlite_confirmation_scope"]
        chosen=at.selectbox(key="sqlite_file_view_NoneNone").value
        self.uploads=[]  # Exercise the draft retained across a client/UI rerender.
        with patch("streamlit.image") as image:
            at.button(key="sqlite_open_file_NoneNone").click().run()
            self.assertTrue(image.called)
            for card in ("input","batches","pipeline","analysis"):
                self.toggle(at,card)
                self.assertEqual(at.button(key="dcc_expand_"+card).label,":material/close_fullscreen:")
                self.assertEqual(self.cards(at)["card_data_"+card][1],0)  # content height, no fixed inner scroll
                self.assertEqual(len(self.cards(at)),4)  # Other widgets stay mounted.
                self.assertEqual(at.radio(key="dataset_role").value,"DEVELOPMENT")
                self.assertEqual(at.session_state["sqlite_uploads"]["manual_metadata"],manual)
                self.assertEqual(at.session_state["sqlite_confirmation_scope"],scope)
                self.assertEqual(self.duplicate_choices(at)[0].value,"REGISTER_SEPARATELY")
                self.assertEqual(at.selectbox(key="sqlite_file_view_NoneNone").value,chosen)
                self.assertFalse(at.button(key="sqlite_register").disabled)
                self.toggle(at,card)
                self.assertEqual(self.cards(at)["card_data_input"][1],480)
                self.assertEqual(self.cards(at)["card_data_batches"][1],480)
        self.assertEqual(runtime_provider().image_count(),1)
        self.assertEqual(runtime_provider().batches()["batch_id"].tolist(),[seed["batch_id"]])

    def test_mock_expansion_keeps_preview_and_common_default_layout(self):
        at=self.app()
        at.session_state["data_control_selected_source"]="Mock";at.run()
        at.radio(key="input_mode").set_value("Lighting").run()
        at.radio(key="dataset_role").set_value("TRAINING_CANDIDATE").run()
        at.selectbox(key="input_exposure").select(70)
        next(row for row in at.button if row.label=="Mock Metadata 미리보기").click().run()
        expected=deepcopy(at.session_state["mock_metadata_preview"])
        catalog=mock.catalog()
        self.toggle(at,"input")
        self.toggle(at,"input")
        self.assertEqual(at.radio(key="input_mode").value,"Lighting")
        self.assertEqual(at.radio(key="dataset_role").value,"TRAINING_CANDIDATE")
        self.assertEqual(at.selectbox(key="input_exposure").value,70)
        self.assertEqual(at.session_state["mock_metadata_preview"],expected)
        self.assertEqual(mock.catalog(),catalog)
        self.assertTrue(runtime_provider().batches().empty)

    def test_missing_metadata_uses_visible_labels_remedies_and_technical_details(self):
        self.uploads=[self.uploaded("unknown.png",80)]
        at=self.app()
        card=self.input_card(at)
        for row in list(card.text_input)+list(card.selectbox)+list(card.number_input)+list(card.radio):
            self.assertTrue(row.label.strip())
            self.assertEqual(row.proto.label_visibility.value,0)
        wafer=next(row for row in card.text_input if row.label=="Batch Wafer ID")
        self.assertIn("필수",wafer.proto.placeholder)
        self.assertTrue(wafer.proto.help)
        self.assertTrue(at.button(key="sqlite_register").disabled)
        self.assertTrue(any("실제 노광" in row.value or "파일명 확인" in row.value for row in card.caption))
        self.assertFalse(any("New process requires" in row.value for row in card.caption))
        technical=next(row for row in at.expander if row.label=="검증 기술 상세")
        self.assertFalse(technical.proto.expanded)
        self.assertTrue(any("requires" in str(row.value) or "Filename needs" in str(row.value) for row in technical.json))
        at.radio(key="input_mode").set_value("Lighting").run()
        magnification=next(row for row in at.number_input if row.label=="촬영 배율")
        self.assertIsNone(magnification.value)
        self.assertIn("필수",magnification.proto.placeholder)
        self.assertTrue(any(row.proto.placeholder for row in at.text_input if row.key.startswith("sqlite_light_")))
        at.radio(key="input_mode").set_value("Wafer Batch").run()
        self.assertFalse(any(row.key.startswith("sqlite_light_") for row in at.text_input))
        self.assertFalse(at.exception)
        self.assertTrue(runtime_provider().batches().empty)

    def test_repeatability_lighting_lookup_and_batch_condition_fields(self):
        self.seed_registered_image()
        provider=runtime_provider()
        for mode,pixel in (("repeatability",81),("lighting",91)):
            bundle=provider.stage([{"filename":"w1_12_c_"+mode+".png","content":png_bytes(pixel=pixel)}])
            try:
                item=bundle["inputs"][0]
                if mode=="repeatability":
                    item["repeat_index"]=1
                else:
                    item["acquisition_condition"]={"condition_id":"UX-LIGHT","magnification":20,
                        "illumination_setting":"lm","auto_exposure":0}
                metadata={"batch_id":"UX-"+mode,"wafer_id":"W1","process_run_id":"SEED-PROCESS",
                          "upload_mode":mode,"dataset_role":"evaluation","batch_status":"registered",
                          "process_run":{"exposure_value":60}}
                provider.register(provider.preview(bundle["inputs"],metadata),approved=True)
            finally:
                bundle["files"].cleanup()
        at=self.app()
        batch_card=self.cards(at)["card_data_batches"][2]
        # TASK18-B-8: 유형·Role·등록일은 Batch 이름 tooltip으로 이동 (정보 유지).
        helps=[b.proto.help for b in batch_card.button if b.key and b.key.startswith("sqlite_batch_")]
        self.assertTrue(any("유형 반복촬영" in h for h in helps))
        self.assertTrue(any("유형 밝기" in h for h in helps))
        self.assertTrue(any("Role EVAL" in h for h in helps))
        cells=batch_card.markdown
        self.assertTrue(any(row.value=="60" and "노광 60" in row.proto.help for row in cells))
        self.assertTrue(any("등록 2026-" in h for h in helps))
        for label,code in (("Lighting","lighting"),("Repeatability","repeatability")):
            at.selectbox(key="sqlite_image_filter_None").select(label).run()
            self.assertFalse(at.exception)
            frame=next(row.value for row in at.dataframe if "position_code" in row.value)
            self.assertEqual(set(frame.columns),{"original_filename","wafer_id","die_code","position_code",
                                               "upload_mode","created_at","metadata_status"})
            self.assertEqual(set(frame["upload_mode"]),{code})
            selected=at.selectbox(key="sqlite_file_view_None"+code).value
            detail=next(json.loads(row.value) for row in at.json if json.loads(row.value).get("image_id")==selected)
            self.assertEqual(detail["wafer_id"],"W1")
            self.assertEqual(detail["die_code"],"12")
            self.assertEqual(detail["position_code"],"c")
            self.assertEqual(detail["upload_mode"],code)
            if code=="lighting":
                self.assertEqual(detail["illumination_setting"],"lm")
                self.assertEqual(detail["magnification"],20)
            else:
                self.assertEqual(detail["repeat_index"],1)

    def test_lighting_draft_values_survive_expand_restore_and_are_cleared_for_other_mode(self):
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        at.radio(key="input_mode").set_value("Lighting").run()
        self.confirm_process(at)
        first=at.session_state["sqlite_uploads"]["inputs"][0]["image_id"]
        next(row for row in at.number_input if row.label=="촬영 배율").set_value(20.0)
        at.text_input(key="sqlite_light_"+first).set_value("lg")
        at.run()
        self.toggle(at,"input")
        self.toggle(at,"input")
        self.assertEqual(at.text_input(key="sqlite_light_"+first).value,"lg")
        self.assertEqual(next(row for row in at.number_input if row.label=="촬영 배율").value,20.0)
        self.assertFalse(at.button(key="sqlite_register").disabled)
        at.radio(key="input_mode").set_value("Repeatability").run()
        self.assertFalse(at.exception)
        self.assertFalse(any(row.key.startswith("sqlite_light_") for row in at.text_input))
        repeats=[row for row in at.number_input if row.key.startswith("sqlite_repeat_")]
        self.assertEqual(len(repeats),1)
        self.assertEqual(repeats[0].value,1)
        self.assertTrue(any(row.key.startswith("sqlite_peb_") for row in at.number_input))
        self.assertTrue(runtime_provider().batches().empty)


if __name__=="__main__":
    unittest.main()
