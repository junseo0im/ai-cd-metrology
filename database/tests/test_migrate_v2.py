"""Migration fixtures/copies only; never migrate the operational DB."""
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from database.init_db import initialize_database,connect_database,_schema_objects
from database.migrate_v2 import (
    migrate_database_v2,initialize_database_v2,validate_v2,data_snapshot,_rebuild_batches,
)

ROOT=Path(__file__).resolve().parent/".tmp"


class MigrationV2Tests(unittest.TestCase):
    def setUp(self):
        ROOT.mkdir(exist_ok=True)
        temp=tempfile.TemporaryDirectory(prefix="migration_",dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()
        assert self.root.is_relative_to(ROOT.resolve())
        self.source=self.root/"source.sqlite3"
        initialize_database(self.source)
        with closing(connect_database(self.source)) as c:
            c.execute("INSERT INTO process_runs(process_run_id,exposure_value) VALUES ('P',60)")
            c.execute("INSERT INTO wafers(wafer_id,process_run_id) VALUES ('W1','P')")
            c.execute("INSERT INTO dies VALUES ('D','W1','11',1,1)")
            c.execute("INSERT INTO pattern_sites(physical_site_id,die_id,position_code) VALUES ('S','D','c')")
            c.execute("INSERT INTO acquisition_conditions(condition_id,magnification) VALUES ('C',20)")
            for batch in ("B1","B2"):
                c.execute("INSERT INTO upload_batches(batch_id,wafer_id,upload_mode,batch_status) "
                          "VALUES (?,'W1','wafer_batch','registered')",(batch,))
            for ident,batch in (("I1","B1"),("I2","B1"),("I3","B2")):
                c.execute("INSERT INTO images(image_id,batch_id,physical_site_id,condition_id,original_filename,"
                          "storage_path,sha256,width_px,height_px,metadata_status) VALUES (?,?,"
                          "'S','C','test.png','test-only/path',?,3,2,'fixture')",(ident,batch,"a"*64))
            c.commit()
        self.copy=self.root/"copy.sqlite3"
        with closing(connect_database(self.source,readonly=True)) as src, closing(sqlite3.connect(self.copy)) as dst:
            src.backup(dst)

    def snapshot(self,path):
        with closing(connect_database(path,readonly=True)) as c:
            return data_snapshot(c)

    def test_populated_copy_preserves_all_records_keys_hashes_indexes_and_triggers(self):
        original_hash=hashlib.sha256(self.source.read_bytes()).hexdigest()
        before=self.snapshot(self.copy)
        with closing(connect_database(self.copy,readonly=True)) as c:
            objects=_schema_objects(c)
        result=migrate_database_v2(self.copy)
        self.assertTrue(result["migrated"])
        self.assertEqual(self.snapshot(self.copy),before)
        self.assertEqual(self.snapshot(Path(result["backup"])),before)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(),original_hash)
        with closing(connect_database(self.copy)) as c:
            validate_v2(c)
            for key,value in objects.items():
                if key!=("table","upload_batches"):
                    self.assertEqual(_schema_objects(c)[key],value)
            self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(),[])
            self.assertEqual(c.execute("SELECT DISTINCT upload_mode FROM upload_batches").fetchall(),[("wafer_batch",)])
            c.execute("INSERT INTO upload_batches(batch_id,wafer_id,upload_mode,batch_status) "
                      "VALUES ('SINGLE','W1','single_image','registered')")
            with self.assertRaises(sqlite3.IntegrityError):
                c.execute("INSERT INTO upload_batches(batch_id,wafer_id,upload_mode,batch_status) "
                          "VALUES ('BAD','W1','unknown','registered')")
            with self.assertRaises(sqlite3.IntegrityError):
                c.execute("INSERT INTO dies VALUES ('D2','W1','11',1,1)")
            with self.assertRaises(sqlite3.IntegrityError):
                c.execute("INSERT INTO pattern_sites(physical_site_id,die_id,position_code) VALUES ('X','D','bad')")
            c.execute("INSERT INTO wafers(wafer_id,process_run_id) VALUES ('W2','P')")
            c.execute("INSERT INTO upload_batches(batch_id,wafer_id,upload_mode,batch_status) "
                      "VALUES ('OTHER','W2','wafer_batch','registered')")
            with self.assertRaisesRegex(sqlite3.IntegrityError,"same wafer"):
                c.execute("UPDATE images SET batch_id='OTHER' WHERE image_id='I1'")

    def test_failure_rolls_back_and_reuses_valid_backup_on_retry(self):
        before=self.snapshot(self.copy)
        before_hash=hashlib.sha256(self.copy.read_bytes()).hexdigest()
        def fail(connection,expected):
            _rebuild_batches(connection,expected)
            raise RuntimeError("migration test failure")
        with patch("database.migrate_v2._rebuild_batches",side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,"test failure"):
                migrate_database_v2(self.copy)
        self.assertEqual(self.snapshot(self.copy),before)
        self.assertEqual(hashlib.sha256(self.copy.read_bytes()).hexdigest(),before_hash)
        initialize_database(self.copy)  # Still a valid v1 DB.
        self.assertTrue(migrate_database_v2(self.copy)["migrated"])
        self.assertEqual(self.snapshot(self.copy),before)

    def test_rerun_is_readonly_noop_and_v1_init_remains_separate(self):
        migrate_database_v2(self.copy)
        before=self.copy.read_bytes()
        self.assertFalse(migrate_database_v2(self.copy)["migrated"])
        self.assertEqual(self.copy.read_bytes(),before)
        with self.assertRaises(RuntimeError):
            initialize_database(self.copy)
        with self.assertRaises(RuntimeError):
            initialize_database_v2(self.source)
        new=self.root/"new.sqlite3"
        initialize_database_v2(new)
        with closing(connect_database(new)) as c:
            validate_v2(c)

    def test_unknown_version_and_operational_path_are_refused(self):
        with closing(connect_database(self.copy)) as c:
            c.execute("PRAGMA user_version=9")
        with self.assertRaises(RuntimeError):
            migrate_database_v2(self.copy)
        with patch("database.migrate_v2.DEFAULT_DB",self.source):
            with self.assertRaises(PermissionError):
                migrate_database_v2(self.source)
        initialize_database(self.source)
