"""Registration UI/provider tests with explicit TEMP paths; never the runtime DB."""
from pathlib import Path
from copy import deepcopy
from contextlib import closing
import sqlite3
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from dashboard import data_provider as mock
from dashboard.sqlite_provider import SQLiteProvider, runtime_provider, sync_uploads, reviewed_inputs
from database.batch_registry import BatchRegistrationError
from database.init_db import initialize_database
from database.tests.test_image_registry import png_bytes

TEST_ROOT=Path(__file__).resolve().parent/".tmp"


class SQLiteInputTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix="db1d_ui_",dir=TEST_ROOT)
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.db_path=self.root/"test.sqlite3"
        initialize_database(self.db_path)
        environment=patch.dict(os.environ, {
            "IDA_DB_PATH":str(self.db_path), "IDA_STORAGE_ROOT":str(self.root/"raw"),
            "IDA_STAGING_ROOT":str(self.root/"staging")})
        environment.start()
        self.addCleanup(environment.stop)
        self.uploads=[]
        # AppTest has no file-upload action; exercise the actual upload consumer
        # using synthetic UploadedFile-compatible values at that boundary.
        uploader=patch("streamlit.file_uploader",side_effect=lambda *a,**kw:
                       [] if kw.get("disabled") else self.uploads)
        uploader.start()
        self.addCleanup(uploader.stop)

    def uploaded(self,name,pixel=40):
        content=png_bytes(pixel=pixel)
        return SimpleNamespace(name=name,getvalue=lambda:content)

    def app(self):
        at=AppTest.from_file(ROOT/"dashboard/app.py",default_timeout=30).run()
        at.switch_page("pages/data_control.py").run()
        self.assertFalse(at.exception)
        at.radio(key="data_control_source").set_value("SQLite").run()
        self.assertFalse(at.exception)
        return at

    def checkbox(self,at,prefix):
        return next(item for item in at.checkbox if item.label.startswith(prefix))

    def button(self,at,label):
        if label=="승인 후 Batch 등록":
            return at.button(key="sqlite_register")
        return next(item for item in at.button if item.label==label)

    def approve(self,at):
        self.assertFalse(any("Preview와 원본" in item.label for item in at.checkbox))
        at.run()
        self.assertFalse(at.exception)

    def confirm_process(self,at):
        next(item for item in at.selectbox if item.label=="노광 조건").select(60).run()
        self.assertFalse(at.exception)

    def test_empty_sqlite_view_has_no_mock_batches_or_auto_insert(self):
        at=self.app()
        self.assertTrue(any("등록된 SQLite Batch 없음" in item.value for item in at.info))
        self.assertTrue(runtime_provider().batches().empty)
        self.assertFalse(any(frame.value.equals(mock.batches()) for frame in at.dataframe))
        self.assertFalse((self.root/"raw").exists())

    def test_upload_manual_review_approval_store_and_query_three_images(self):
        self.uploads=[self.uploaded("w1_12_c.png"),self.uploaded("unknown.png",80),
                      self.uploaded("w1_12_rb_pn.png",120)]
        at=self.app()
        self.confirm_process(at)
        preview=at.session_state["sqlite_preview"]
        self.assertEqual(preview["status_counts"],{"VALID":2,"NEEDS_METADATA":1})
        self.assertTrue(self.button(at,"승인 후 Batch 등록").disabled)
        self.assertTrue(runtime_provider().batches().empty)
        missing=next(item for item in preview["images"] if item["original_filename"]=="unknown.png")
        key=missing["image_id"]
        at.radio(key="sqlite_review_"+preview["batch_metadata"]["batch_id"]).set_value(key).run()
        at.text_input(key="sqlite_edit_wafer_"+key).set_value("W1")
        at.selectbox(key="sqlite_edit_die_"+key).select("38")
        at.selectbox(key="sqlite_edit_position_"+key).select("lt")
        self.button(at,"수정 후 Preview 재검증").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["sqlite_preview"]["status_counts"],{"VALID":3})
        self.assertTrue(runtime_provider().batches().empty)
        self.approve(at)
        self.assertFalse(self.button(at,"승인 후 Batch 등록").disabled)
        self.button(at,"승인 후 Batch 등록").click().run()
        self.assertFalse(at.exception)
        result=at.session_state["sqlite_last_result"]
        self.assertEqual(result["image_count"],3)
        provider=runtime_provider()
        images=provider.images(result["batch_id"])
        self.assertEqual(len(images),3)
        self.assertEqual(provider.batches()["image_count"].tolist(),[3])
        self.assertTrue(all(Path(path).is_file() for path in images["storage_path"]))
        self.assertIn("unknown.png",images["original_filename"].tolist())
        row=images[images["original_filename"]=="unknown.png"].iloc[0]
        self.assertEqual((row["die_code"],row["position_code"]),("38","lt"))
        self.assertEqual(len(list((self.root/"raw").rglob("*.png"))),3)
        at.switch_page("pages/main_home.py").run()
        at.switch_page("pages/data_control.py").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("SQLite 등록 완료" in item.value for item in at.success))
        at.radio(key="data_control_source").set_value("Mock").run()
        self.assertFalse(at.exception)
        self.assertTrue(any(frame.value.equals(mock.batches()) for frame in at.dataframe)
                        or any(item.label.startswith("BATCH-DEMO") for item in at.button))
        self.assertEqual(provider.batches()["image_count"].tolist(),[3])

    def test_upload_or_new_batch_cannot_reuse_old_manual_edits(self):
        provider=runtime_provider()
        state={}
        first=sync_uploads(state,[{"filename":"unknown.png","content":png_bytes()}],provider)
        first["manual_metadata"][first["inputs"][0]["image_id"]]={
            "wafer_id":"W1","die_code":"12","position_code":"c"}
        old_path=Path(first["inputs"][0]["image_path"])
        state["sqlite_preview"]={"old":True}
        second=sync_uploads(state,[{"filename":"unknown.png","content":png_bytes(pixel=80)}],provider)
        self.assertNotEqual(first["inputs"][0]["image_id"],second["inputs"][0]["image_id"])
        self.assertEqual(second["manual_metadata"],{})
        self.assertNotIn("sqlite_preview",state)
        self.assertNotIn("manual_metadata",reviewed_inputs(second)[0])
        self.assertFalse(old_path.exists())
        second["files"].cleanup()

    def test_metadata_changes_require_no_cached_approval_and_do_not_auto_register(self):
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        self.assertFalse(self.button(at,"승인 후 Batch 등록").disabled)
        at.radio(key="dataset_role").set_value("DEVELOPMENT").run()
        self.assertFalse(at.exception)
        self.assertFalse(any("Preview와 원본" in item.label for item in at.checkbox))
        self.assertFalse(self.button(at,"승인 후 Batch 등록").disabled)
        self.assertEqual(at.session_state["sqlite_preview"]["batch_metadata"]["dataset_role"],"development")
        self.assertTrue(runtime_provider().batches().empty)

    def test_duplicate_candidates_require_explicit_per_image_policy_selection(self):
        self.uploads=[self.uploaded("w1_12_c.png"),self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        self.assertEqual(at.session_state["sqlite_preview"]["status_counts"],{"DUPLICATE_CANDIDATE":2})
        self.assertTrue(all(item.value=="EXCLUDE" for item in self.duplicate_choices(at)))
        self.assertTrue(self.button(at,"승인 후 Batch 등록").disabled)
        keys=[item.key for item in self.duplicate_choices(at)]
        at.selectbox(key=keys[0]).select("REGISTER_SEPARATELY").run()
        self.assertEqual(at.session_state["sqlite_registration_plan"]["selected_count"],1)
        self.assertFalse(self.button(at,"승인 후 Batch 등록").disabled)
        at.selectbox(key=keys[1]).select("REGISTER_SEPARATELY").run()
        self.assertEqual(at.session_state["sqlite_registration_plan"]["selected_count"],2)
        self.assertFalse(self.button(at,"승인 후 Batch 등록").disabled)
        self.assertFalse(any("Preview와 원본" in item.label for item in at.checkbox))
        self.assertTrue(runtime_provider().batches().empty)

    def test_explicit_provider_paths_and_mock_contract_stay_separate(self):
        provider=runtime_provider()
        self.assertEqual(provider.db_path,self.db_path)
        self.assertEqual(provider.storage_root,self.root/"raw")
        self.assertEqual(provider.staging_root,self.root/"staging")
        before=mock.catalog()
        bundle=provider.stage([{"filename":"w1_12_c.png","content":png_bytes()}])
        try:
            metadata={"batch_id":"TEST","wafer_id":"W1","upload_mode":"lighting",
                      "dataset_role":"development","batch_status":"registered",
                      "process_run_id":"TEST-P","process_run":{"exposure_value":1}}
            preview=provider.preview(bundle["inputs"],metadata)
            with self.assertRaises(ValueError):
                provider.register(preview)
            self.assertTrue(provider.batches().empty)
            self.assertEqual(mock.catalog(),before)
        finally:
            bundle["files"].cleanup()


    def seed_registered_image(self, pixel=40):
        """Explicit fixture registration in this test's isolated DB only."""
        provider=runtime_provider()
        bundle=provider.stage([{"filename":"w1_12_c.png","content":png_bytes(pixel=pixel)}])
        try:
            metadata={"batch_id":"SEED-"+str(len(provider.batches())), "wafer_id":"W1",
                      "upload_mode":"wafer_batch", "dataset_role":"evaluation",
                      "batch_status":"registered", "process_run_id":"SEED-PROCESS",
                      "process_run":{"exposure_value":60.0}}
            preview=provider.preview(bundle["inputs"],metadata)
            duplicates=[row["image_id"] for row in preview["images"]
                        if row["status"]=="DUPLICATE_CANDIDATE"]
            return provider.register(preview,approved=True,duplicate_ids=duplicates)
        finally:
            bundle["files"].cleanup()

    def duplicate_choices(self, at):
        return [item for item in at.selectbox if item.label.startswith("중복 처리 방법")]

    def acknowledge_duplicate(self, at):
        for key in [item.key for item in self.duplicate_choices(at)]:
            at.selectbox(key=key).select("REGISTER_SEPARATELY").run()
        self.approve(at)
        self.assertFalse(at.exception)

    def test_existing_batch_three_images_then_same_bytes_new_batch_warns_and_registers_only_after_ack(self):
        self.uploads=[self.uploaded("w1_12_c.png"),self.uploaded("w1_12_rb.png",80),
                      self.uploaded("w1_38_lt.png",120)]
        at=self.app()
        self.confirm_process(at)
        self.assertFalse(any("중복 이미지 발견" in item.value for item in at.warning))
        self.assertEqual(self.duplicate_choices(at),[])
        original_preview=deepcopy(at.session_state["sqlite_preview"])
        self.approve(at)
        self.button(at,"승인 후 Batch 등록").click().run()
        first=deepcopy(at.session_state["sqlite_last_result"])
        self.assertEqual(first["image_count"],3)
        self.uploads=[self.uploaded("w1_12_c.png")]
        self.button(at,"새 업로드 준비 (입력 초기화)").click().run()
        self.confirm_process(at)
        self.assertFalse(at.exception)
        preview=at.session_state["sqlite_preview"]
        self.assertEqual(preview["status_counts"],{"DUPLICATE_CANDIDATE":1})
        self.assertTrue(any("중복 이미지 발견 — 1장" in item.value for item in at.warning))
        original_id=next(row["image_id"] for row in original_preview["images"]
                         if row["original_filename"]=="w1_12_c.png")
        details="\n".join(str(item.value) for item in at.json)
        self.assertIn(original_id,details)
        self.assertIn(first["batch_id"],details)
        self.assertIn("w1_12_c.png",details)
        self.assertEqual(len(self.duplicate_choices(at)),1)
        self.assertEqual(self.duplicate_choices(at)[0].value,"EXCLUDE")
        self.approve(at)
        self.assertTrue(self.button(at,"승인 후 Batch 등록").disabled)
        # Removing/resetting UI confirmations cannot bypass the unchanged backend.
        with self.assertRaisesRegex(BatchRegistrationError,"per-image"):
            runtime_provider().register(preview,approved=True,duplicate_ids=[])
        self.assertEqual(runtime_provider().batches()["image_count"].tolist(),[3])
        self.duplicate_choices(at)[0].select("REGISTER_SEPARATELY").run()
        self.approve(at)
        self.assertFalse(self.button(at,"승인 후 Batch 등록").disabled)
        self.button(at,"승인 후 Batch 등록").click().run()
        self.assertFalse(at.exception)
        second=at.session_state["sqlite_last_result"]
        self.assertNotEqual(first["batch_id"],second["batch_id"])
        self.assertEqual(second["image_count"],1)
        self.assertEqual(sorted(runtime_provider().batches()["image_count"].tolist()),[1,3])

    def test_all_existing_duplicate_image_and_batch_ids_are_visible(self):
        first=self.seed_registered_image()
        second=self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        self.assertTrue(any("중복 이미지 발견 — 1장" in item.value for item in at.warning))
        details="\n".join(str(item.value) for item in at.json)
        for result in (first,second):
            self.assertIn(result["batch_id"],details)
            self.assertIn(result["image_ids"][0],details)
        self.assertEqual(len(self.duplicate_choices(at)),1)
        self.assertEqual(len(runtime_provider().batches()),2)

    def test_same_filename_different_bytes_is_not_a_sha_duplicate(self):
        self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png",80)]
        at=self.app()
        self.confirm_process(at)
        self.assertEqual(at.session_state["sqlite_preview"]["status_counts"],{"VALID":1})
        self.assertFalse(any("중복 이미지 발견" in item.value for item in at.warning))
        self.assertEqual(self.duplicate_choices(at),[])
        self.assertEqual(len(runtime_provider().batches()),1)

    def test_duplicate_metadata_changes_and_restoration_do_not_reuse_confirmations(self):
        self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        self.acknowledge_duplicate(at)
        old_keys=list(at.session_state["sqlite_confirmation_keys"])
        self.assertFalse(self.button(at,"승인 후 Batch 등록").disabled)
        at.radio(key="dataset_role").set_value("DEVELOPMENT").run()
        self.assertEqual(self.duplicate_choices(at)[0].value,"EXCLUDE")
        self.assertFalse(any("Preview와 원본" in item.label for item in at.checkbox))
        self.assertTrue(self.button(at,"승인 후 Batch 등록").disabled)
        self.assertTrue(all(key not in at.session_state for key in old_keys))
        at.radio(key="dataset_role").set_value("EVALUATION").run()
        self.assertEqual(self.duplicate_choices(at)[0].value,"EXCLUDE")
        self.assertFalse(any("Preview와 원본" in item.label for item in at.checkbox))
        self.assertTrue(self.button(at,"승인 후 Batch 등록").disabled)

    def test_upload_changes_and_new_batch_reset_duplicate_confirmation(self):
        self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        self.acknowledge_duplicate(at)
        self.uploads=[self.uploaded("w1_12_c.png",80)]
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(self.duplicate_choices(at),[])
        self.assertFalse(any("Preview와 원본" in item.label for item in at.checkbox))
        self.uploads=[self.uploaded("w1_12_c.png")]
        at.run()
        self.confirm_process(at)
        self.assertEqual(self.duplicate_choices(at)[0].value,"EXCLUDE")
        self.acknowledge_duplicate(at)
        old_keys=list(at.session_state["sqlite_confirmation_keys"])
        self.button(at,"새 업로드 준비 (입력 초기화)").click().run()
        self.confirm_process(at)
        self.assertFalse(at.exception)
        self.assertEqual(self.duplicate_choices(at)[0].value,"EXCLUDE")
        self.assertFalse(any("Preview와 원본" in item.label for item in at.checkbox))
        self.assertTrue(all(key not in at.session_state for key in old_keys))
        self.assertEqual(len(runtime_provider().batches()),1)

    def test_staged_sha_change_resets_confirmation_and_backend_rejects_stale_preview(self):
        self.seed_registered_image(40)
        self.seed_registered_image(80)
        self.uploads=[self.uploaded("w1_12_c.png",40)]
        at=self.app()
        self.confirm_process(at)
        self.acknowledge_duplicate(at)
        old=deepcopy(at.session_state["sqlite_preview"])
        source=Path(at.session_state["sqlite_uploads"]["inputs"][0]["image_path"])
        source.write_bytes(png_bytes(pixel=80))
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(self.duplicate_choices(at)[0].value,"EXCLUDE")
        self.assertFalse(any("Preview와 원본" in item.label for item in at.checkbox))
        self.assertNotEqual(old["images"][0]["sha256"],at.session_state["sqlite_preview"]["images"][0]["sha256"])
        with self.assertRaisesRegex(BatchRegistrationError,"Preview changed"):
            runtime_provider().register(old,approved=True,duplicate_ids=[old["images"][0]["image_id"]])
        self.assertEqual(len(runtime_provider().batches()),2)

    def test_changed_existing_candidate_set_resets_confirmations(self):
        self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        self.acknowledge_duplicate(at)
        old=deepcopy(at.session_state["sqlite_preview"])
        self.seed_registered_image()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state["sqlite_preview"]["images"][0]["duplicate_image_ids"]),2)
        self.assertEqual(self.duplicate_choices(at)[0].value,"EXCLUDE")
        self.assertFalse(any("Preview와 원본" in item.label for item in at.checkbox))
        with self.assertRaisesRegex(BatchRegistrationError,"Preview changed"):
            runtime_provider().register(old,approved=True,duplicate_ids=[old["images"][0]["image_id"]])

    def test_duplicate_warning_remains_distinct_from_missing_filename_metadata(self):
        self.seed_registered_image()
        self.uploads=[self.uploaded("unknown.png")]
        at=self.app()
        key=at.session_state["sqlite_uploads"]["inputs"][0]["image_id"]
        at.text_input(key="sqlite_wafer_"+key).set_value("W1").run()
        self.confirm_process(at)
        row=at.session_state["sqlite_preview"]["images"][0]
        self.assertEqual(row["status"],"NEEDS_METADATA")
        self.assertTrue(row["duplicate_image_ids"])
        self.assertTrue(any("중복 이미지 발견 — 1장" in item.value for item in at.warning))
        self.assertTrue(self.button(at,"승인 후 Batch 등록").disabled)
        self.assertEqual(len(runtime_provider().batches()),1)

    def test_duplicate_reference_lookup_failure_blocks_ui_registration(self):
        self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        self.acknowledge_duplicate(at)
        with patch.object(SQLiteProvider,"duplicate_records",side_effect=sqlite3.OperationalError("test lookup failure")):
            at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("입력·검증 실패" in item.value for item in at.error))
        self.assertTrue(at.button(key="sqlite_register").disabled)
        at.run()
        self.assertEqual(self.duplicate_choices(at)[0].value,"EXCLUDE")
        self.assertFalse(any("Preview와 원본" in item.label for item in at.checkbox))
        self.assertEqual(len(runtime_provider().batches()),1)


    def mixed_policy_app(self):
        self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png",40),self.uploaded("w1_12_rb.png",80),
                      self.uploaded("w1_38_lt.png",120)]
        at=self.app()
        self.confirm_process(at)
        self.assertEqual(at.session_state["sqlite_preview"]["status_counts"],
                         {"DUPLICATE_CANDIDATE":1,"VALID":2})
        return at

    def assert_policy_counts(self,result,uploaded,registered,separate,excluded):
        self.assertEqual((result["uploaded_count"],result["registered_count"],
                          result["duplicate_registered_count"],result["excluded_count"]),
                         (uploaded,registered,separate,excluded))
        self.assertEqual(result["image_count"],registered)
        self.assertEqual(len(result["image_ids"]),registered)

    def test_policy_default_exclude_mixed_batch_registers_only_two_and_reports_counts(self):
        at=self.mixed_policy_app()
        source=deepcopy(at.session_state["sqlite_preview"])
        excluded_id=source["images"][0]["image_id"]
        plan=at.session_state["sqlite_registration_plan"]
        self.assertEqual(self.duplicate_choices(at)[0].value,"EXCLUDE")
        self.assertEqual(plan["final_preview"]["status_counts"],{"VALID":2})
        self.assertEqual(plan["selected_count"],2)
        self.approve(at)
        self.assertFalse(self.button(at,"승인 후 Batch 등록").disabled)
        self.button(at,"승인 후 Batch 등록").click().run()
        self.assertFalse(at.exception)
        result=at.session_state["sqlite_last_result"]
        self.assert_policy_counts(result,3,2,0,1)
        self.assertNotIn(excluded_id,result["image_ids"])
        with closing(runtime_provider().connect()) as connection:
            self.assertIsNone(connection.execute("SELECT image_id FROM images WHERE image_id=?",
                                                (excluded_id,)).fetchone())
        images=runtime_provider().images(result["batch_id"])
        self.assertEqual(set(images["original_filename"]),{"w1_12_rb.png","w1_38_lt.png"})
        self.assertEqual(len(list((self.root/"raw").rglob("*.png"))),3)  # seed + two selected
        self.assertEqual(sorted(runtime_provider().batches()["image_count"].tolist()),[1,2])
        self.assertTrue(any("신규 등록 2장" in item.value and "중복 제외 1장" in item.value for item in at.markdown))

    def test_policy_register_separately_mixed_batch_keeps_all_three(self):
        at=self.mixed_policy_app()
        self.duplicate_choices(at)[0].select("REGISTER_SEPARATELY").run()
        self.assertTrue(any("데이터 집계 시 중복에 주의" in item.value for item in at.caption))
        self.assertEqual(at.session_state["sqlite_registration_plan"]["selected_count"],3)
        self.approve(at)
        self.button(at,"승인 후 Batch 등록").click().run()
        self.assertFalse(at.exception)
        result=at.session_state["sqlite_last_result"]
        self.assert_policy_counts(result,3,3,1,0)
        self.assertEqual(len(runtime_provider().images(result["batch_id"])),3)
        self.assertEqual(len(list((self.root/"raw").rglob("*.png"))),4)

    def test_policy_hold_blocks_mixed_batch_and_names_held_image(self):
        at=self.mixed_policy_app()
        self.duplicate_choices(at)[0].select("HOLD").run()
        self.approve(at)
        self.assertTrue(self.button(at,"승인 후 Batch 등록").disabled)
        self.assertTrue(any("HOLD" in item.value and "w1_12_c.png" in item.value
                            for item in list(at.warning)+list(at.caption)))
        plan=at.session_state["sqlite_registration_plan"]
        with self.assertRaisesRegex(BatchRegistrationError,"HOLD"):
            runtime_provider().register_selection(plan,approved=True)
        self.assertEqual(runtime_provider().batches()["image_count"].tolist(),[1])
        self.assertEqual(len(list((self.root/"raw").rglob("*.png"))),1)

    def test_policy_all_excluded_duplicates_create_no_empty_batch(self):
        self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        self.assertTrue(any("등록할 새 이미지가 없습니다" in item.value for item in at.info))
        self.assertEqual(at.session_state["sqlite_registration_plan"]["selected_count"],0)
        self.assertTrue(self.button(at,"승인 후 Batch 등록").disabled)
        with self.assertRaisesRegex(BatchRegistrationError,"No images selected"):
            runtime_provider().register_selection(at.session_state["sqlite_registration_plan"],approved=True)
        self.assertEqual(len(runtime_provider().batches()),1)
        self.assertEqual(len(list((self.root/"raw").rglob("*.png"))),1)

    def test_policy_multiple_duplicate_actions_can_mix_independently(self):
        self.seed_registered_image()
        self.uploads=[self.uploaded("w1_12_c.png"),self.uploaded("w1_12_rb.png"),
                      self.uploaded("w1_38_lt.png",120)]
        at=self.app()
        self.confirm_process(at)
        choices=self.duplicate_choices(at)
        self.assertEqual(len(choices),2)
        choices[1].select("REGISTER_SEPARATELY").run()
        plan=at.session_state["sqlite_registration_plan"]
        self.assertEqual(len(plan["excluded_image_ids"]),1)
        self.assertEqual(len(plan["separate_image_ids"]),1)
        self.assertEqual(plan["selected_count"],2)
        self.approve(at)
        self.button(at,"승인 후 Batch 등록").click().run()
        self.assertFalse(at.exception)
        self.assert_policy_counts(at.session_state["sqlite_last_result"],3,2,1,1)

    def test_policy_choice_changes_update_final_set_without_auto_registration(self):
        at=self.mixed_policy_app()
        self.assertEqual(at.session_state["sqlite_registration_plan"]["selected_count"],2)
        self.duplicate_choices(at)[0].select("REGISTER_SEPARATELY").run()
        self.assertEqual(at.session_state["sqlite_registration_plan"]["selected_count"],3)
        self.assertFalse(self.button(at,"승인 후 Batch 등록").disabled)
        self.duplicate_choices(at)[0].select("EXCLUDE").run()
        self.assertEqual(at.session_state["sqlite_registration_plan"]["selected_count"],2)
        self.assertFalse(self.button(at,"승인 후 Batch 등록").disabled)
        self.assertEqual(runtime_provider().batches()["image_count"].tolist(),[1])

    def test_policy_valid_decision_survives_rerender_and_unrelated_navigation(self):
        at=self.mixed_policy_app()
        at.radio(key="dataset_role").set_value("DEVELOPMENT").run()
        self.duplicate_choices(at)[0].select("REGISTER_SEPARATELY").run()
        source_scope=at.session_state["sqlite_confirmation_scope"]
        actions=deepcopy(at.session_state["sqlite_duplicate_decisions"])
        at.run()
        self.assertEqual(self.duplicate_choices(at)[0].value,"REGISTER_SEPARATELY")
        at.switch_page("pages/main_home.py").run()
        # Simulate the real uploader widget losing its value while not rendered.
        self.uploads=[]
        at.switch_page("pages/data_control.py").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.radio(key="dataset_role").value,"DEVELOPMENT")
        self.assertEqual(at.session_state["sqlite_confirmation_scope"],source_scope)
        self.assertEqual(at.session_state["sqlite_duplicate_decisions"],actions)
        self.assertEqual(self.duplicate_choices(at)[0].value,"REGISTER_SEPARATELY")
        self.assertEqual(at.session_state["sqlite_registration_plan"]["selected_count"],3)

    def test_policy_explicit_upload_clear_discards_draft_and_decisions(self):
        at=self.mixed_policy_app()
        self.duplicate_choices(at)[0].select("HOLD").run()
        self.uploads=[]
        at.session_state["sqlite_upload_selection_changed"]=True  # uploader on_change
        at.run()
        self.assertFalse(at.exception)
        self.assertNotIn("sqlite_uploads",at.session_state)
        self.assertNotIn("sqlite_duplicate_decisions",at.session_state)
        self.assertNotIn("sqlite_registration_plan",at.session_state)
        self.assertEqual(runtime_provider().batches()["image_count"].tolist(),[1])

    def test_policy_final_insert_failure_rolls_back_and_cleans_new_files(self):
        from database.image_registry import get_image
        at=self.mixed_policy_app()
        target=at.session_state["sqlite_registration_plan"]["final_preview"]["images"][-1]["image_id"]
        original_files={str(path):path.read_bytes() for path in (self.root/"raw").rglob("*.png")}
        self.approve(at)

        def failure(connection,image_id):
            if image_id==target:
                raise RuntimeError("test selected-image readback failure")
            return get_image(connection,image_id)

        with patch("database.image_registry.get_image",side_effect=failure):
            self.button(at,"승인 후 Batch 등록").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("등록 실패" in item.value for item in at.error))
        self.assertEqual(runtime_provider().batches()["image_count"].tolist(),[1])
        self.assertEqual({str(path):path.read_bytes() for path in (self.root/"raw").rglob("*.png")},original_files)


    def test_unified_button_always_present_and_one_click_registration(self):
        at=self.app()
        self.assertTrue(at.button(key="sqlite_register").disabled)
        self.assertTrue(at.button(key="sqlite_register").label.strip())
        self.uploads=[self.uploaded("w1_12_c.png")]
        at.run()
        self.assertTrue(at.button(key="sqlite_register").disabled)  # No fabricated exposure.
        self.assertFalse(any("Preview와 원본" in item.label or "실제 공정 Metadata임을 확인" in item.label
                             for item in at.checkbox))
        self.confirm_process(at)
        self.assertFalse(at.button(key="sqlite_register").disabled)
        self.assertEqual(at.button(key="sqlite_register").label,"이미지 1장 DB 등록")
        self.assertTrue(runtime_provider().batches().empty)
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["sqlite_last_result"]["registered_count"],1)

    def test_single_v1_is_explicitly_blocked_and_v2_registers_real_single_mode(self):
        from database.migrate_v2 import migrate_database_v2
        at=self.app()
        at.radio(key="input_mode").set_value("Single Image").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.button(key="sqlite_register").label,"이미지 0장 DB 등록")
        self.assertTrue(at.button(key="sqlite_register").disabled)
        self.assertTrue(any("Schema v1에서는 Single" in item.value for item in at.caption))
        self.assertTrue(runtime_provider().batches().empty)
        migrate_database_v2(self.db_path)
        self.uploads=[self.uploaded("w1_12_c.png")]
        at.run()
        self.confirm_process(at)
        self.assertFalse(at.button(key="sqlite_register").disabled)
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(runtime_provider().batches()["upload_mode"].tolist(),["single_image"])
        self.assertEqual(runtime_provider().image_count(upload_mode="single_image"),1)

    def test_repeatability_ui_checks_sequence_and_same_site(self):
        self.uploads=[self.uploaded("w1_11_c_r1.png"),self.uploaded("w1_11_c_r2.png",80)]
        at=self.app()
        at.radio(key="input_mode").set_value("Repeatability").run()
        self.confirm_process(at)
        numbers=[item for item in at.number_input if item.label.startswith("반복 번호")]
        self.assertEqual([item.value for item in numbers],[1,2])
        numbers[1].set_value(1).run()
        self.assertTrue(at.button(key="sqlite_register").disabled)
        next(item for item in at.number_input if item.key==numbers[1].key).set_value(2).run()
        self.assertFalse(at.button(key="sqlite_register").disabled)
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        result=at.session_state["sqlite_last_result"]
        self.assertEqual(set(runtime_provider().images(result["batch_id"])["repeat_index"]),{1,2})

    def test_lighting_ui_requires_actual_magnification_and_illumination(self):
        self.uploads=[self.uploaded("w1_11_c_lg.png"),self.uploaded("w1_11_c_lm.png",80)]
        at=self.app()
        at.radio(key="input_mode").set_value("Lighting").run()
        self.confirm_process(at)
        self.assertTrue(at.button(key="sqlite_register").disabled)
        next(item for item in at.number_input if item.label=="촬영 배율").set_value(20.0).run()
        inputs=at.session_state["sqlite_uploads"]["inputs"]
        for item,illumination in zip(inputs,["lg","lm"]):
            at.text_input(key="sqlite_light_"+item["image_id"]).set_value(illumination)
        at.run()
        self.assertFalse(at.button(key="sqlite_register").disabled)
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        rows=runtime_provider().images(at.session_state["sqlite_last_result"]["batch_id"])
        self.assertEqual(set(rows["illumination_setting"]),{"lg","lm"})
        self.assertTrue(rows["auto_exposure"].isna().all())

    def test_wafer_zip_plus_png_more_than_three_images_registers_once(self):
        from database.tests.test_integrated_registration import zip_bytes
        archive=zip_bytes([(f"wafer/w1_1{index}_c.png",png_bytes(pixel=20+index)) for index in range(1,6)])
        self.uploads=[SimpleNamespace(name="wafer.zip",getvalue=lambda:archive),self.uploaded("w1_16_c.png",200)]
        at=self.app()
        self.confirm_process(at)
        self.assertEqual(at.session_state["sqlite_preview"]["status_counts"],{"VALID":6})
        self.assertFalse(at.button(key="sqlite_register").disabled)
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["sqlite_last_result"]["image_count"],6)
        self.assertEqual(len(list((self.root/"raw").rglob("*.png"))),6)

    def test_broken_zip_keeps_visible_disabled_button_and_no_insert(self):
        self.uploads=[SimpleNamespace(name="broken.zip",getvalue=lambda:b"not ZIP")]
        at=self.app()
        self.assertFalse(at.exception)
        self.assertTrue(at.button(key="sqlite_register").disabled)
        self.assertTrue(at.button(key="sqlite_register").label.strip())
        self.assertTrue(at.error)
        self.assertTrue(runtime_provider().batches().empty)
        self.assertFalse((self.root/"raw").exists())

    def test_review_defaults_to_errors_and_uses_common_wafer_without_overwriting_parsed_values(self):
        self.uploads=[self.uploaded("w1_12_c.png"),self.uploaded("unknown.png",80)]
        at=self.app()
        self.confirm_process(at)
        preview=at.session_state["sqlite_preview"]
        unknown=next(row["image_id"] for row in preview["images"] if row["original_filename"]=="unknown.png")
        known=next(row["image_id"] for row in preview["images"] if row["original_filename"]=="w1_12_c.png")
        selector=at.radio(key="sqlite_review_"+preview["batch_metadata"]["batch_id"])
        self.assertEqual(selector.value,unknown)
        self.assertEqual(len(selector.options),1)
        self.assertEqual(at.text_input(key="sqlite_edit_wafer_"+unknown).value,"W1")
        at.checkbox(key="sqlite_review_all_"+preview["batch_metadata"]["batch_id"]).set_value(True).run()
        at.radio(key="sqlite_review_"+preview["batch_metadata"]["batch_id"]).set_value(known).run()
        self.assertEqual(at.text_input(key="sqlite_edit_wafer_"+known).value,"W1")
        scope=at.session_state["sqlite_uploads"]["inputs"][0]["image_id"]
        at.text_input(key="sqlite_wafer_"+scope).set_value("W2").run()
        self.assertEqual(at.text_input(key="sqlite_edit_wafer_"+known).value,"W1")
        self.assertTrue(at.button(key="sqlite_register").disabled)

    def test_real_image_filters_pagination_short_batch_selection_and_existing_detail_route(self):
        from database.migrate_v2 import migrate_database_v2
        migrate_database_v2(self.db_path)
        provider=runtime_provider()
        for mode,count,start in (("wafer_batch",27,0),("repeatability",2,80),
                                 ("lighting",1,90),("single_image",1,100)):
            uploads=[{"filename":f"w1_11_c_t{index}.png","content":png_bytes(pixel=start+index)}
                     for index in range(count)]
            bundle=provider.stage(uploads)
            try:
                if mode=="repeatability":
                    for index,item in enumerate(bundle["inputs"],1):
                        item["repeat_index"]=index
                if mode=="lighting":
                    bundle["inputs"][0]["acquisition_condition"]={
                        "condition_id":"TEST-LIGHT","magnification":20,"illumination_setting":"lg"}
                metadata={"batch_id":"LOOKUP-"+mode,"wafer_id":"W1","upload_mode":mode,
                          "dataset_role":"evaluation","batch_status":"registered",
                          "process_run_id":"LOOKUP-P","process_run":{"exposure_value":60}}
                preview=provider.preview(bundle["inputs"],metadata)
                self.assertTrue(preview["registration_ready"])
                provider.register(preview,approved=True)
            finally:
                bundle["files"].cleanup()
        self.assertEqual(provider.image_count(),31)
        self.assertEqual(len(provider.images()),25)
        at=self.app()
        filters=next(item for item in at.selectbox if item.label=="등록 유형 필터")
        for label,code,count in (("Repeatability","repeatability",2),("Lighting","lighting",1),
                                 ("Single Image","single_image",1),("Wafer Batch","wafer_batch",27)):
            at.selectbox(key=filters.key).select(label).run()
            self.assertFalse(at.exception)
            self.assertEqual(provider.image_count(upload_mode=code),count)
            frames=[item.value for item in at.dataframe if "position_code" in item.value.columns]
            self.assertTrue(frames)
            expected=provider.images(upload_mode=code,limit=5)
            self.assertEqual(set(frames[0]["original_filename"]),set(expected["original_filename"]))
            self.assertNotIn("sha256",frames[0].columns)
            self.assertNotIn("image_id",frames[0].columns)
            self.assertLessEqual(len(frames[0]),25)
        at.selectbox(key=filters.key).select("전체").run()
        next(item for item in at.selectbox if item.label=="표시 개수").select(25).run()
        next(item for item in at.selectbox if item.label=="이미지 페이지").select(2).run()
        frames=[item.value for item in at.dataframe if "position_code" in item.value.columns]
        self.assertEqual(len(frames[0]),6)
        button=next(item for item in at.button if item.key and item.key.startswith("sqlite_batch_"))
        selected=button.key.removeprefix("sqlite_batch_")
        button.click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["selected_sqlite_batch_id"],selected)
        self.assertEqual(at.session_state["batch_detail_source"],"SQLite")
        self.assertEqual(at.title[0].value,"Batch 상세")
        self.assertTrue(any("SQLite 등록 Metadata" in item.value for item in at.caption))


if __name__=="__main__":
    unittest.main()
