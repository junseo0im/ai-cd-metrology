"""User-named Batch UI tests use isolated SQLite and synthetic PNGs only."""
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import hashlib
import unittest

import dashboard.tests.test_sqlite_input as fixtures
from dashboard.sqlite_provider import runtime_provider
from database.batch_names import batch_name_from_notes


class BatchNameUITests(unittest.TestCase):
    setUp=fixtures.SQLiteInputTests.setUp
    app=fixtures.SQLiteInputTests.app
    uploaded=fixtures.SQLiteInputTests.uploaded
    confirm_process=fixtures.SQLiteInputTests.confirm_process

    def save_name(self,at,value):
        # Like existing navigation tests, retain the route entered by a button:
        # AppTest does not synchronize its next rerun URL after st.switch_page.
        at.switch_page("pages/batch_detail.py")
        identity=at.session_state["selected_sqlite_batch_id"]
        at.text_input(key="sqlite_saved_batch_name_"+identity).set_value(value)
        next(row for row in at.button if row.label=="Batch 이름 저장").click().run()
        self.assertFalse(at.exception)

    def test_button_labels_reset_only_draft_and_blank_name_uses_automatic_number(self):
        at=self.app()
        self.assertEqual(at.button(key="sqlite_register").label,"이미지 0장 DB 등록")
        self.assertEqual(at.button(key="sqlite_new_batch").label,"새 업로드 준비 (입력 초기화)")
        at.text_input(key="sqlite_batch_name").set_value("저장하지 않을 초안").run()
        at.button(key="sqlite_new_batch").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.text_input(key="sqlite_batch_name").value,"")
        self.assertTrue(runtime_provider().batches().empty)
        self.uploads=[self.uploaded("w1_12_c.png")]
        at.run()
        self.confirm_process(at)
        self.assertEqual(at.button(key="sqlite_register").label,"이미지 1장 DB 등록")
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        provider=runtime_provider()
        row=provider.batches().iloc[0]
        self.assertEqual(row["batch_name"],"")
        self.assertEqual(row["display_name"],"W1 · Batch 01")
        with closing(provider.connect()) as db:
            self.assertIsNone(db.execute("SELECT notes FROM upload_batches").fetchone()[0])
        saved=provider.images().to_dict("records")
        self.uploads=[]
        at.button(key="sqlite_new_batch").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(provider.image_count(),1)
        self.assertEqual(len(provider.batches()),1)
        self.assertEqual(provider.images().to_dict("records"),saved)
        self.assertEqual(at.text_input(key="sqlite_batch_name").value,"")

    def test_named_registration_detail_rename_clear_persist_without_changing_image_identity(self):
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        name="W5 중앙 반복촬영 3회"
        at.text_input(key="sqlite_batch_name").set_value(name).run()
        at.button(key="dcc_expand_input").click().run()
        at.button(key="dcc_expand_input").click().run()
        self.assertEqual(at.text_input(key="sqlite_batch_name").value,name)
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        provider=runtime_provider()
        row=provider.batches().iloc[0]
        identity=row["batch_id"]
        self.assertEqual(row["batch_name"],name)
        self.assertEqual(row["display_name"],"W1 · Batch 01 · "+name)
        self.assertEqual(at.session_state["sqlite_last_result"]["batch_id"],identity)
        saved=provider.images(identity).to_dict("records")
        digests={item["storage_path"]:hashlib.sha256(Path(item["storage_path"]).read_bytes()).hexdigest() for item in saved}
        with closing(provider.connect()) as db:
            batch_before=db.execute("SELECT batch_id,wafer_id,upload_mode,dataset_role,created_at,batch_status FROM upload_batches WHERE batch_id=?",(identity,)).fetchone()
            self.assertEqual(batch_name_from_notes(db.execute("SELECT notes FROM upload_batches WHERE batch_id=?",(identity,)).fetchone()[0]),name)
        at.button(key="sqlite_batch_"+identity).click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any(name in header.value for header in at.subheader))
        self.save_name(at,"수정한 Batch 이름")
        self.assertTrue(any("수정한 Batch 이름" in header.value for header in at.subheader))
        self.assertEqual(runtime_provider().batches().iloc[0]["batch_name"],"수정한 Batch 이름")
        self.save_name(at,"")
        self.assertEqual(runtime_provider().batches().iloc[0]["display_name"],"W1 · Batch 01")
        self.assertTrue(any(header.value=="W1 · Batch 01" for header in at.subheader))
        self.assertEqual(provider.images(identity).to_dict("records"),saved)
        with closing(provider.connect()) as db:
            self.assertEqual(db.execute("SELECT batch_id,wafer_id,upload_mode,dataset_role,created_at,batch_status FROM upload_batches WHERE batch_id=?",(identity,)).fetchone(),batch_before)
            self.assertIsNone(db.execute("SELECT notes FROM upload_batches WHERE batch_id=?",(identity,)).fetchone()[0])
        for path,digest in digests.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),digest)
        next(row for row in at.button if row.label=="← Data Home").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.button(key="sqlite_batch_"+identity).label,"W1 · Batch 01")

    def test_long_name_is_shortened_in_table_and_mock_does_not_persist_names(self):
        self.uploads=[self.uploaded("w1_12_c.png")]
        at=self.app()
        self.confirm_process(at)
        name="긴 Batch 이름 "*30
        at.text_input(key="sqlite_batch_name").set_value(name).run()
        at.button(key="sqlite_register").click().run()
        self.assertFalse(at.exception)
        provider=runtime_provider()
        row=provider.batches().iloc[0]
        button=at.button(key="sqlite_batch_"+row["batch_id"])
        self.assertTrue(button.label.startswith("W1 · Batch 01 · "))
        self.assertLess(len(button.label),50)
        self.assertIn(name.strip(),button.proto.help)
        self.assertEqual(row["batch_name"],name.strip())
        before=deepcopy(provider.batches().to_dict("records"))
        at.radio(key="data_control_source").set_value("Mock").run()
        self.assertFalse(at.exception)
        self.assertFalse(any(widget.key=="sqlite_batch_name" for widget in at.text_input))
        self.assertEqual(provider.batches().to_dict("records"),before)


if __name__=="__main__":
    unittest.main()
