"""DB-1C uses only disposable SQLite DBs and synthetic PNGs."""
from pathlib import Path
from copy import deepcopy
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from database.batch_registry import (
    BatchRegistrationError, preview_batch, register_batch, prepare_registration, register_selection,
)
from database.image_registry import ImageRegistrationError, get_image, register_image
from database.init_db import TABLE_NAMES, connect_database, initialize_database
from database.tests.test_image_registry import png_bytes

TEST_ROOT = Path(__file__).resolve().parent / ".tmp"


class BatchRegistryTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="db1c_", dir=TEST_ROOT)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "test.sqlite3"
        initialize_database(self.path)
        self.db = connect_database(self.path)
        self.addCleanup(self.db.close)
        # Explicit synthetic fixture values, never inserted into the operational DB.
        self.batch = {"batch_id": "B1", "wafer_id": "W1", "upload_mode": "wafer_batch",
                      "dataset_role": "evaluation", "batch_status": "metadata_reviewed",
                      "process_run_id": "TEST-PROCESS", "process_run": {"exposure_value": 1.0}}

    def image(self, name="w1_12_c.png", image_id="I1", pixel=40, **fields):
        path = self.directory / name
        path.write_bytes(png_bytes(pixel=pixel))
        return {"image_path": str(path), "image_id": image_id, **fields}

    def inputs(self):
        return [self.image(), self.image("w1_12_rb_pn.png", "I2", 80),
                self.image("w1_38_lt.png", "I3", 120)]

    def counts(self):
        return {table: self.db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
                for table in TABLE_NAMES}

    def test_valid_preview_is_read_only_and_json_compatible(self):
        import json
        inputs = self.inputs()
        before = self.path.read_bytes()
        preview = preview_batch(self.db, inputs, self.batch)
        self.assertEqual(preview["status_counts"], {"VALID": 3})
        self.assertTrue(preview["registration_ready"])
        self.assertFalse(preview["saved_to_database"])
        json.dumps(preview, allow_nan=False)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(all(value == 0 for value in self.counts().values()))
        self.assertFalse(self.db.in_transaction)
        self.assertEqual(preview["images"][0]["metadata_source"], "filename")

    def test_bad_filename_needs_manual_metadata_and_preserves_original(self):
        item = self.image("unknown.png")
        before = Path(item["image_path"]).read_bytes()
        preview = preview_batch(self.db, [item], self.batch)
        self.assertEqual(preview["images"][0]["status"], "NEEDS_METADATA")
        with self.assertRaises(BatchRegistrationError):
            register_batch(self.db, preview, approved=True)
        item["manual_metadata"] = {"wafer_id": "W1", "die_code": "12"}
        self.assertEqual(preview_batch(self.db, [item], self.batch)["images"][0]["status"], "NEEDS_METADATA")
        item["manual_metadata"].update(position_code="rb", legacy_defect_code="future")
        preview = preview_batch(self.db, [item], self.batch)
        row = preview["images"][0]
        self.assertEqual(row["status"], "VALID")
        self.assertEqual(row["metadata_source"], "manual")
        self.assertIsNone(row["parsed_metadata"])
        self.assertEqual(row["original_filename"], "unknown.png")
        register_batch(self.db, preview, approved=True)
        stored = get_image(self.db, "I1")
        self.assertEqual(stored["original_filename"], "unknown.png")
        self.assertEqual((stored["wafer_id"], stored["die_code"], stored["position_code"]), ("W1", "12", "rb"))
        self.assertEqual(stored["legacy_defect_code"], "future")
        self.assertEqual(stored["metadata_status"], "manual_confirmed")
        self.assertEqual(Path(item["image_path"]).read_bytes(), before)
        self.assertFalse((self.directory / "W1_12_rb_future.png").exists())

    def test_manual_invalid_die_position_and_wafer_mismatch(self):
        for fields in ({"die_code": "39"}, {"die_code": "12_c"}, {"position_code": "c_p"},
                       {"position_code": "left"}, {"wafer_id": "W2"}):
            item = self.image("unknown.png")
            item["manual_metadata"] = {"wafer_id": "W1", "die_code": "12", "position_code": "c", **fields}
            preview = preview_batch(self.db, [item], self.batch)
            self.assertEqual(preview["images"][0]["status"], "INVALID")
            with self.assertRaises(BatchRegistrationError):
                register_batch(self.db, preview, approved=True)
        item = self.image("w2_12_c.png")
        self.assertEqual(preview_batch(self.db, [item], self.batch)["images"][0]["status"], "INVALID")
        self.assertTrue(all(value == 0 for value in self.counts().values()))

    def test_missing_process_data_is_not_invented(self):
        item = self.image()
        batch = {key: value for key, value in self.batch.items() if key not in ("process_run_id", "process_run")}
        preview = preview_batch(self.db, [item], batch)
        self.assertEqual(preview["images"][0]["status"], "NEEDS_METADATA")
        batch["process_run_id"] = "UNKNOWN"
        preview = preview_batch(self.db, [item], batch)
        self.assertEqual(preview["images"][0]["status"], "NEEDS_METADATA")
        self.assertIn("actual exposure_value", str(preview["batch_issues"]))
        batch["process_run"] = {"exposure_value": float("nan")}
        self.assertEqual(preview_batch(self.db, [item], batch)["images"][0]["status"], "INVALID")
        self.assertTrue(all(value == 0 for value in self.counts().values()))

    def test_internal_duplicates_need_each_explicit_confirmation(self):
        inputs = [self.image(), self.image("w1_12_rb.png", "I2")]
        originals = {row["image_path"]: Path(row["image_path"]).read_bytes() for row in inputs}
        preview = preview_batch(self.db, inputs, self.batch)
        self.assertEqual(preview["status_counts"], {"DUPLICATE_CANDIDATE": 2})
        self.assertEqual(preview["images"][0]["duplicate_input_indices"], [1])
        for confirmed in ((), ["I1"]):
            with self.assertRaisesRegex(BatchRegistrationError, "per-image"):
                register_batch(self.db, preview, approved=True, approved_duplicate_image_ids=confirmed)
            self.assertTrue(all(value == 0 for value in self.counts().values()))
        result = register_batch(self.db, preview, approved=True, approved_duplicate_image_ids=["I1", "I2"])
        self.assertEqual(result["image_count"], 2)
        for path, content in originals.items():
            self.assertEqual(Path(path).read_bytes(), content)

    def test_existing_db_duplicate_is_reported_and_preserved(self):
        item = self.image()
        register_batch(self.db, preview_batch(self.db, [item], self.batch), approved=True)
        self.db.execute("UPDATE images SET sha256=upper(sha256)")
        self.db.commit()
        original = get_image(self.db, "I1")
        second = self.image("w1_12_c_p.png", "I2")
        batch = {**self.batch, "batch_id": "B2"}
        preview = preview_batch(self.db, [second], batch)
        row = preview["images"][0]
        self.assertEqual(row["status"], "DUPLICATE_CANDIDATE")
        self.assertEqual(row["duplicate_image_ids"], ["I1"])
        self.assertEqual(row["same_site_existing_image_ids"], ["I1"])
        with self.assertRaises(BatchRegistrationError):
            register_batch(self.db, preview, approved=True)
        register_batch(self.db, preview, approved=True, approved_duplicate_image_ids=["I2"])
        self.assertEqual(get_image(self.db, "I1"), original)
        self.assertEqual(self.counts()["images"], 2)

    def test_no_approval_means_no_insert_or_transaction(self):
        preview = preview_batch(self.db, self.inputs(), self.batch)
        before = self.path.read_bytes()
        statements = []
        self.db.set_trace_callback(statements.append)
        for approved in (False, None, 1, "yes"):
            with self.assertRaisesRegex(BatchRegistrationError, "approval"):
                register_batch(self.db, preview, approved=approved)
        self.assertFalse(any(sql.startswith(("INSERT", "BEGIN")) for sql in statements))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(self.db.in_transaction)

    def test_whole_batch_success_and_same_site_multiple_images(self):
        inputs = self.inputs()
        inputs.append(self.image("w1_12_c_repeat.png", "I4", 160, repeat_index=2))
        preview = preview_batch(self.db, inputs, self.batch)
        self.assertEqual(preview["images"][0]["same_site_input_indices"], [3])
        originals = {row["image_path"]: Path(row["image_path"]).read_bytes() for row in inputs}
        result = register_batch(self.db, preview, approved=True)
        self.assertEqual(result, {"batch_id": "B1", "image_count": 4, "image_ids": ["I1", "I2", "I3", "I4"]})
        self.assertFalse(self.db.in_transaction)
        self.assertEqual(self.counts(), {"process_runs": 1, "wafers": 1, "dies": 2,
                         "pattern_sites": 3, "upload_batches": 1, "acquisition_conditions": 0, "images": 4})
        self.assertEqual(self.db.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(get_image(self.db, "I4")["physical_site_id"], get_image(self.db, "I1")["physical_site_id"])
        self.assertEqual(get_image(self.db, "I4")["repeat_index"], 2)
        self.assertTrue(all(get_image(self.db, key)["condition_id"] is None for key in result["image_ids"]))
        for path, content in originals.items():
            self.assertEqual(Path(path).read_bytes(), content)

    def test_second_image_insert_failure_rolls_back_all_parents_and_images(self):
        # Test-only trigger: forces a real SQLite INSERT failure after I1 exists.
        self.db.execute("""CREATE TRIGGER test_fail_second BEFORE INSERT ON images
            WHEN NEW.image_id='I2' AND EXISTS(SELECT 1 FROM images WHERE image_id='I1')
            BEGIN SELECT RAISE(ABORT, 'test second INSERT failure'); END""")
        self.db.commit()
        before = self.counts()
        inputs = self.inputs()
        preview = preview_batch(self.db, inputs, self.batch)
        with self.assertRaisesRegex(ImageRegistrationError, "test second INSERT failure"):
            register_batch(self.db, preview, approved=True)
        self.assertEqual(self.counts(), before)
        self.assertFalse(self.db.in_transaction)
        self.assertEqual(self.db.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_file_change_during_registration_rolls_back_all_records(self):
        item = self.image()
        preview = preview_batch(self.db, [item], self.batch)
        original_register = register_image

        def changed_register(*args, **kwargs):
            Path(item["image_path"]).write_bytes(png_bytes(pixel=200))
            return original_register(*args, **kwargs)

        with patch("database.batch_registry.register_image", side_effect=changed_register):
            with self.assertRaisesRegex(BatchRegistrationError, "bytes changed"):
                register_batch(self.db, preview, approved=True)
        self.assertTrue(all(value == 0 for value in self.counts().values()))
        self.assertFalse(self.db.in_transaction)

    def test_existing_parent_ids_are_reused_without_changing_real_metadata(self):
        self.db.execute("INSERT INTO process_runs(process_run_id,exposure_value) VALUES ('EXISTING',2)")
        self.db.execute("INSERT INTO wafers(wafer_id,process_run_id) VALUES ('W1','EXISTING')")
        self.db.execute("INSERT INTO dies VALUES ('CUSTOM-DIE','W1','12',1,2)")
        self.db.execute("INSERT INTO pattern_sites(physical_site_id,die_id,position_code) VALUES ('CUSTOM-SITE','CUSTOM-DIE','c')")
        self.db.commit()
        batch = {key: value for key, value in self.batch.items() if key not in ("process_run_id", "process_run")}
        preview = preview_batch(self.db, [self.image()], batch)
        self.assertTrue(preview["registration_ready"])
        register_batch(self.db, preview, approved=True)
        self.assertEqual(get_image(self.db, "I1")["physical_site_id"], "CUSTOM-SITE")
        self.assertEqual(self.db.execute("SELECT exposure_value FROM process_runs").fetchone()[0], 2)
        self.assertEqual(self.counts()["process_runs"], 1)

    def test_sources_changed_since_preview_are_refused(self):
        item = self.image()
        preview = preview_batch(self.db, [item], self.batch)
        Path(item["image_path"]).write_bytes(png_bytes(pixel=200))
        with self.assertRaisesRegex(BatchRegistrationError, "Preview changed"):
            register_batch(self.db, preview, approved=True)
        self.assertTrue(all(value == 0 for value in self.counts().values()))

    def test_new_db_duplicate_after_preview_requires_a_new_review(self):
        pending = self.image()
        preview = preview_batch(self.db, [pending], self.batch)
        existing = self.image("w1_12_c_old.png", "OLD")
        register_batch(self.db, preview_batch(self.db, [existing], {**self.batch, "batch_id": "OLD-BATCH"}), approved=True)
        before = self.counts()
        with self.assertRaisesRegex(BatchRegistrationError, "Preview changed"):
            register_batch(self.db, preview, approved=True, approved_duplicate_image_ids=["I1"])
        self.assertEqual(self.counts(), before)

    def test_bad_files_missing_ids_and_unknown_condition(self):
        for name, content in (("broken.png", b"not PNG"), ("image.jpg", png_bytes())):
            item = self.image(name)
            Path(item["image_path"]).write_bytes(content)
            self.assertEqual(preview_batch(self.db, [item], self.batch)["images"][0]["status"], "INVALID")
        item = self.image()
        del item["image_id"]
        self.assertEqual(preview_batch(self.db, [item], self.batch)["images"][0]["status"], "NEEDS_METADATA")
        item["image_id"] = "I1"
        item["condition_id"] = "UNKNOWN"
        self.assertEqual(preview_batch(self.db, [item], self.batch)["images"][0]["status"], "NEEDS_METADATA")
        item["image_path"] = str(self.directory / "missing.png")
        self.assertEqual(preview_batch(self.db, [item], self.batch)["images"][0]["status"], "INVALID")

    def test_preview_accepts_a_readonly_connection_and_empty_input_is_not_ready(self):
        item = self.image()
        readonly = connect_database(self.path, readonly=True)
        try:
            self.assertTrue(preview_batch(readonly, [item], self.batch)["registration_ready"])
            self.assertFalse(preview_batch(readonly, [], self.batch)["registration_ready"])
        finally:
            readonly.close()

    def test_shared_registry_caller_transaction_option_requires_active_transaction(self):
        with self.assertRaisesRegex(ImageRegistrationError, "active transaction"):
            register_image(self.db, self.image()["image_path"], image_id="I1", batch_id="B1",
                           physical_site_id="site", metadata_status="test", manage_transaction=False)
        self.assertTrue(all(value == 0 for value in self.counts().values()))

    def test_operational_guard_and_caller_transaction_are_preserved(self):
        item = self.image()
        with patch("database.image_registry.DEFAULT_DB", self.path):
            with self.assertRaisesRegex(ImageRegistrationError, "operational DB"):
                preview_batch(self.db, [item], self.batch)
        self.db.execute("INSERT INTO process_runs(process_run_id,exposure_value) VALUES ('CALLER',1)")
        with self.assertRaisesRegex(ImageRegistrationError, "Commit fixture"):
            preview_batch(self.db, [item], self.batch)
        self.assertTrue(self.db.in_transaction)
        self.db.rollback()


    def duplicate_selection_fixture(self):
        existing=self.image("w1_12_c.png","EXISTING",40)
        register_batch(self.db,preview_batch(self.db,[existing],self.batch),approved=True)
        inputs=self.inputs()
        preview=preview_batch(self.db,inputs,{**self.batch,"batch_id":"B2"})
        return inputs,preview

    def test_selection_default_exclude_registers_only_two_new_images_and_files(self):
        inputs,preview=self.duplicate_selection_fixture()
        original=deepcopy(preview)
        plan=prepare_registration(self.db,preview)
        self.assertEqual(preview,original)
        self.assertEqual(plan["decisions"],{"I1":"EXCLUDE"})
        self.assertEqual(plan["final_preview"]["status_counts"],{"VALID":2})
        self.assertEqual([row["image_id"] for row in plan["final_preview"]["images"]],["I2","I3"])
        result=register_selection(self.db,plan,approved=True,storage_root=self.directory/"raw")
        self.assertEqual((result["uploaded_count"],result["registered_count"],
                          result["duplicate_registered_count"],result["excluded_count"]),(3,2,0,1))
        self.assertEqual(result["image_ids"],["I2","I3"])
        self.assertIsNone(get_image(self.db,"I1"))
        self.assertIsNotNone(get_image(self.db,"EXISTING"))
        self.assertEqual(len(list((self.directory/"raw").rglob("*.png"))),2)
        self.assertEqual(self.counts()["upload_batches"],2)

    def test_selection_register_separately_reuses_duplicate_approval_and_keeps_three(self):
        inputs,preview=self.duplicate_selection_fixture()
        plan=prepare_registration(self.db,preview,{"I1":"REGISTER_SEPARATELY"})
        self.assertEqual(plan["separate_image_ids"],["I1"])
        self.assertEqual(plan["final_preview"]["images"][0]["status"],"DUPLICATE_CANDIDATE")
        with self.assertRaisesRegex(BatchRegistrationError,"per-image"):
            register_batch(self.db,plan["final_preview"],approved=True)
        result=register_selection(self.db,plan,approved=True,storage_root=self.directory/"raw")
        self.assertEqual((result["uploaded_count"],result["registered_count"],
                          result["duplicate_registered_count"],result["excluded_count"]),(3,3,1,0))
        self.assertEqual(self.counts()["images"],4)
        self.assertNotEqual(get_image(self.db,"I1")["storage_path"],get_image(self.db,"EXISTING")["storage_path"])

    def test_selection_hold_blocks_whole_batch_without_files(self):
        inputs,preview=self.duplicate_selection_fixture()
        before=self.counts()
        plan=prepare_registration(self.db,preview,{"I1":"HOLD"})
        self.assertFalse(plan["registration_ready"])
        with self.assertRaisesRegex(BatchRegistrationError,"HOLD"):
            register_selection(self.db,plan,approved=True,storage_root=self.directory/"raw")
        self.assertEqual(self.counts(),before)
        self.assertFalse((self.directory/"raw").exists())

    def test_selection_all_duplicates_excluded_does_not_create_empty_batch(self):
        inputs,preview=self.duplicate_selection_fixture()
        one=preview_batch(self.db,[inputs[0]],{**self.batch,"batch_id":"B2"})
        plan=prepare_registration(self.db,one)
        self.assertEqual(plan["selected_count"],0)
        self.assertFalse(plan["registration_ready"])
        before=self.counts()
        with self.assertRaisesRegex(BatchRegistrationError,"No images selected"):
            register_selection(self.db,plan,approved=True,storage_root=self.directory/"raw")
        self.assertEqual(self.counts(),before)
        self.assertFalse((self.directory/"raw").exists())

    def test_selection_mixed_duplicate_actions_are_independent(self):
        inputs,preview=self.duplicate_selection_fixture()
        Path(inputs[1]["image_path"]).write_bytes(Path(inputs[0]["image_path"]).read_bytes())
        preview=preview_batch(self.db,inputs,{**self.batch,"batch_id":"B2"})
        plan=prepare_registration(self.db,preview,{"I1":"EXCLUDE","I2":"REGISTER_SEPARATELY"})
        result=register_selection(self.db,plan,approved=True,storage_root=self.directory/"raw")
        self.assertEqual(result["image_ids"],["I2","I3"])
        self.assertEqual((result["registered_count"],result["duplicate_registered_count"],result["excluded_count"]),(2,1,1))

    def test_selection_final_preview_rebuilds_internal_duplicate_relationships(self):
        inputs=[self.image("w1_12_c.png","I1",40),self.image("w1_12_rb.png","I2",40)]
        preview=preview_batch(self.db,inputs,self.batch)
        self.assertEqual(preview["status_counts"],{"DUPLICATE_CANDIDATE":2})
        plan=prepare_registration(self.db,preview,{"I1":"REGISTER_SEPARATELY","I2":"EXCLUDE"})
        self.assertEqual(plan["final_preview"]["status_counts"],{"VALID":1})
        self.assertEqual(plan["final_preview"]["images"][0]["duplicate_input_indices"],[])
        result=register_selection(self.db,plan,approved=True)
        self.assertEqual(result["image_ids"],["I1"])

    def test_selection_rechecks_excluded_file_and_tampered_final_set(self):
        inputs,preview=self.duplicate_selection_fixture()
        plan=prepare_registration(self.db,preview)
        changed=deepcopy(plan)
        changed["final_preview"]["inputs"].reverse()
        with self.assertRaisesRegex(BatchRegistrationError,"selection changed"):
            register_selection(self.db,changed,approved=True,storage_root=self.directory/"raw")
        Path(inputs[0]["image_path"]).write_bytes(png_bytes(pixel=200))
        with self.assertRaisesRegex(BatchRegistrationError,"Source Preview changed"):
            register_selection(self.db,plan,approved=True,storage_root=self.directory/"raw")
        self.assertEqual(self.counts()["upload_batches"],1)
        self.assertFalse((self.directory/"raw").exists())

    def test_selection_failure_rolls_back_selected_images_and_new_files(self):
        inputs,preview=self.duplicate_selection_fixture()
        self.db.execute("""CREATE TRIGGER test_selection_failure BEFORE INSERT ON images
            WHEN NEW.image_id='I3' AND EXISTS(SELECT 1 FROM images WHERE image_id='I2')
            BEGIN SELECT RAISE(ABORT,'selection INSERT failure'); END""")
        self.db.commit()
        preview=preview_batch(self.db,inputs,{**self.batch,"batch_id":"B2"})
        plan=prepare_registration(self.db,preview)
        before=self.counts()
        with self.assertRaisesRegex(ImageRegistrationError,"selection INSERT failure"):
            register_selection(self.db,plan,approved=True,storage_root=self.directory/"raw")
        self.assertEqual(self.counts(),before)
        self.assertEqual(list((self.directory/"raw").rglob("*.png")),[])
        self.assertIsNotNone(get_image(self.db,"EXISTING"))
        self.assertFalse(self.db.in_transaction)

    def test_selection_rejects_non_duplicate_exclusions_and_unknown_actions(self):
        inputs,preview=self.duplicate_selection_fixture()
        for decisions in ({"I2":"EXCLUDE"},{"I1":"DELETE"}):
            with self.assertRaises(BatchRegistrationError):
                prepare_registration(self.db,preview,decisions)
        invalid_inputs=deepcopy(inputs)
        invalid_inputs[1]["image_id"]=invalid_inputs[0]["image_id"]
        invalid_preview=preview_batch(self.db,invalid_inputs,{**self.batch,"batch_id":"B2"})
        with self.assertRaisesRegex(BatchRegistrationError,"unique"):
            prepare_registration(self.db,invalid_preview)
        invalid_inputs=deepcopy(inputs)
        del invalid_inputs[0]["image_id"]
        invalid_preview=preview_batch(self.db,invalid_inputs,{**self.batch,"batch_id":"B2"})
        with self.assertRaisesRegex(BatchRegistrationError,"Image IDs"):
            prepare_registration(self.db,invalid_preview)
        plan=prepare_registration(self.db,preview)
        before=self.counts()
        with self.assertRaisesRegex(BatchRegistrationError,"approval"):
            register_selection(self.db,plan)
        self.assertEqual(self.counts(),before)


if __name__ == "__main__":
    unittest.main()
