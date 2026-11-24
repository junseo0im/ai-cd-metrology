"""PEB extension tests use populated isolated DBs; never the operational path."""
from contextlib import closing
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch
from database.init_db import connect_database,_schema_objects
from database.migrate_v2 import migrate_database_v2,validate_v2,data_snapshot
from database.migrate_v3 import migrate_database_v3,validate_v3,initialize_database_v3
import database.tests.test_migrate_v2 as fixtures


class MigrationV3Tests(unittest.TestCase):
    def setUp(self):
        fixtures.MigrationV2Tests.setUp(self)
        migrate_database_v2(self.copy)

    def test_only_nullable_peb_column_added_all_existing_fields_objects_preserved(self):
        with closing(connect_database(self.copy,readonly=True)) as c:
            before=data_snapshot(c);objects=_schema_objects(c)
        result=migrate_database_v3(self.copy)
        with closing(connect_database(self.copy)) as c:
            validate_v3(c)
            after=data_snapshot(c)
            self.assertEqual([r[:-1] for r in after["upload_batches"]],before["upload_batches"])
            self.assertTrue(all(r[-1] is None for r in after["upload_batches"]))
            self.assertEqual({k:v for k,v in after.items() if k!="upload_batches"},
                             {k:v for k,v in before.items() if k!="upload_batches"})
            for key,value in objects.items():
                if key!=("table","upload_batches"):self.assertEqual(_schema_objects(c)[key],value)
            c.execute("UPDATE upload_batches SET peb_temperature_c=110 WHERE batch_id='B1'")
            c.commit()
            for invalid in ("not-numeric",float("inf")):
                with self.assertRaises(sqlite3.IntegrityError):
                    c.execute("UPDATE upload_batches SET peb_temperature_c=? WHERE batch_id='B1'",(invalid,))
        with closing(connect_database(result["backup"],readonly=True)) as c:
            validate_v2(c);self.assertEqual(data_snapshot(c),before)
        snapshot=self.copy.read_bytes()
        self.assertFalse(migrate_database_v3(self.copy)["migrated"])
        self.assertEqual(self.copy.read_bytes(),snapshot)

    def test_failure_rolls_back_and_valid_backup_can_be_reused(self):
        before=self.copy.read_bytes()
        with patch("database.migrate_v3.validate_v3",side_effect=RuntimeError("QA rollback")):
            with self.assertRaises(RuntimeError):migrate_database_v3(self.copy)
        self.assertEqual(self.copy.read_bytes(),before)
        with closing(connect_database(self.copy,readonly=True)) as c:validate_v2(c)
        self.assertTrue(migrate_database_v3(self.copy)["migrated"])

    def test_operational_guard_and_no_silent_migration(self):
        with patch("database.migrate_v3.DEFAULT_DB",self.copy):
            with self.assertRaises(PermissionError):migrate_database_v3(self.copy)
        with self.assertRaises(RuntimeError):initialize_database_v3(self.copy)
        fresh=self.root/"new_v3.sqlite3";initialize_database_v3(fresh)
        with closing(connect_database(fresh,readonly=True)) as c:validate_v3(c)


if __name__=="__main__":
    unittest.main()
