"""Batch names: disposable v1/v2 DBs, original notes and image rows preserved."""
from contextlib import closing
from pathlib import Path
import hashlib
import sqlite3
import tempfile
import unittest

from database.batch_names import batch_name_from_notes,notes_with_batch_name,update_batch_name
from database.batch_registry import preview_batch,register_batch
from database.init_db import initialize_database,connect_database,TABLE_NAMES
from database.migrate_v2 import initialize_database_v2
from database.tests.test_image_registry import png_bytes

TEST_ROOT=Path(__file__).resolve().parent/".tmp"


class BatchNameTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix="batch_names_",dir=TEST_ROOT)
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def seed(self,version=1,notes=None):
        path=self.root/f"v{version}.sqlite3"
        (initialize_database if version==1 else initialize_database_v2)(path)
        db=connect_database(path)
        self.addCleanup(db.close)
        source=self.root/f"v{version}_w1_12_c.png"
        source.write_bytes(png_bytes(pixel=70))
        metadata={"batch_id":"B1","wafer_id":"W1","upload_mode":"wafer_batch",
                  "dataset_role":"evaluation","batch_status":"registered","process_run_id":"P1",
                  "process_run":{"exposure_value":60},"notes":notes}
        inputs=[{"image_id":"I1","image_path":str(source),
                 "manual_metadata":{"wafer_id":"W1","die_code":"12","position_code":"c"}}]
        register_batch(db,preview_batch(db,inputs,metadata),approved=True)
        return path,db,source

    def snapshot_except_batch_notes(self,db):
        result={}
        for table in TABLE_NAMES:
            cursor=db.execute("SELECT * FROM "+table)
            columns=[row[0] for row in cursor.description]
            result[table]=[tuple(value for column,value in zip(columns,row)
                                 if table!="upload_batches" or column!="notes") for row in cursor]
        return result

    def test_codec_preserves_legacy_text_json_null_and_restores_on_clear(self):
        for original in (None,"","기존 메모 그대로","{not-json",'{ "source": "existing", "notes": 1 }'):
            with self.subTest(original=original):
                encoded=notes_with_batch_name(original,"  W5 중앙 반복촬영 3회  ")
                self.assertEqual(batch_name_from_notes(encoded),"W5 중앙 반복촬영 3회")
                renamed=notes_with_batch_name(encoded,"수정 이름")
                self.assertEqual(batch_name_from_notes(renamed),"수정 이름")
                self.assertEqual(notes_with_batch_name(renamed,"  "),original)
                self.assertEqual(batch_name_from_notes(original),"")
        self.assertEqual(batch_name_from_notes('{"format":"ida.batch_display_name.v1"}'),"")

    def test_registration_rename_clear_persist_across_connections_in_v1_and_v2(self):
        for version in (1,2):
            with self.subTest(version=version):
                original="원래 Batch 메모"
                path,db,source=self.seed(version,notes_with_batch_name(original,"첫 이름"))
                saved=self.snapshot_except_batch_notes(db)
                digest=hashlib.sha256(source.read_bytes()).hexdigest()
                self.assertEqual(batch_name_from_notes(db.execute("SELECT notes FROM upload_batches").fetchone()[0]),"첫 이름")
                update_batch_name(db,"B1","다음 이름")
                with closing(connect_database(path,readonly=True)) as reread:
                    self.assertEqual(batch_name_from_notes(reread.execute("SELECT notes FROM upload_batches").fetchone()[0]),"다음 이름")
                self.assertEqual(self.snapshot_except_batch_notes(db),saved)
                update_batch_name(db,"B1","")
                self.assertEqual(db.execute("SELECT notes FROM upload_batches").fetchone()[0],original)
                self.assertEqual(self.snapshot_except_batch_notes(db),saved)
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0],version)
                self.assertEqual(db.execute("PRAGMA integrity_check").fetchall(),[("ok",)])
                self.assertEqual(db.execute("PRAGMA foreign_key_check").fetchall(),[])

    def test_update_failure_rolls_back_without_changing_saved_notes(self):
        path,db,source=self.seed(notes="original")
        db.execute("CREATE TRIGGER reject_name BEFORE UPDATE OF notes ON upload_batches "
                   "BEGIN SELECT RAISE(ABORT,'test save failure'); END")
        before=self.snapshot_except_batch_notes(db)
        with self.assertRaises(sqlite3.IntegrityError):
            update_batch_name(db,"B1","not saved")
        self.assertFalse(db.in_transaction)
        self.assertEqual(db.execute("SELECT notes FROM upload_batches").fetchone()[0],"original")
        self.assertEqual(self.snapshot_except_batch_notes(db),before)

    def test_unknown_batch_and_invalid_name_never_update_rows(self):
        path,db,source=self.seed(notes="original")
        for identity,name in (("missing","new"),("B1","line"+chr(10)+"break"),("B1",42)):
            with self.subTest(identity=identity,name=name),self.assertRaises(ValueError):
                update_batch_name(db,identity,name)
            self.assertFalse(db.in_transaction)
            self.assertEqual(db.execute("SELECT notes FROM upload_batches").fetchone()[0],"original")


if __name__=="__main__":
    unittest.main()
