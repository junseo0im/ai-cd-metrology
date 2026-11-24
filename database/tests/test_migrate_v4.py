"""v3->v4 additive migration tests, isolated Fixtures only."""
from contextlib import closing
from pathlib import Path
import os,sqlite3,tempfile,unittest
from unittest.mock import patch
from database.init_db import connect_database,_schema_objects
from database.migrate_v2 import data_snapshot
from database.migrate_v3 import validate_v3
from database.migrate_v4 import migrate_database_v4,validate_v4,ANALYSIS_TABLES
from database.tests.test_analysis_repository import Fixture,TEST_ROOT


class MigrationV4Tests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        t=tempfile.TemporaryDirectory(dir=TEST_ROOT);self.addCleanup(t.cleanup)
        self.f=Fixture(Path(t.name))

    def test_migration_preserves_seven_tables_indexes_triggers(self):
        with closing(connect_database(self.f.db,readonly=True)) as c:
            data=data_snapshot(c);objects=_schema_objects(c)
        self.assertTrue(migrate_database_v4(self.f.db)["migrated"])
        with closing(connect_database(self.f.db,readonly=True)) as c:
            validate_v4(c);self.assertEqual(data_snapshot(c),data)
            after=_schema_objects(c)
            self.assertTrue(all(after[k]==v for k,v in objects.items()))
            self.assertEqual(len([k for k in after if k[0]=="table"]),12)
        before=self.f.db.read_bytes()
        self.assertFalse(migrate_database_v4(self.f.db)["migrated"])
        self.assertEqual(before,self.f.db.read_bytes())

    def test_failure_rolls_back_then_retries(self):
        import database.migrate_v4 as module
        original=module._apply
        def fail(c):original(c);raise RuntimeError("injected migration failure")
        with patch.object(module,"_apply",side_effect=fail):
            with self.assertRaises(RuntimeError):migrate_database_v4(self.f.db)
        with closing(connect_database(self.f.db,readonly=True)) as c:
            validate_v3(c)
            self.assertEqual(c.execute("SELECT count(*) FROM sqlite_master WHERE name='analysis_batch_runs'").fetchone()[0],0)
        self.assertTrue(migrate_database_v4(self.f.db)["migrated"])

    def test_operational_path_and_hardlink_refused(self):
        alias=self.f.root/"alias.sqlite3";os.link(self.f.db,alias)
        before=self.f.db.read_bytes()
        with patch("database.migrate_v4.DEFAULT_DB",self.f.db):
            for p in (self.f.db,alias):
                with self.assertRaises(PermissionError):migrate_database_v4(p)
        self.assertEqual(before,self.f.db.read_bytes())

    def test_v4_tampering_rejected(self):
        migrate_database_v4(self.f.db)
        with closing(connect_database(self.f.db)) as c:
            c.execute("DROP INDEX idx_analysis_one_adopted");c.commit()
        with self.assertRaises(RuntimeError):migrate_database_v4(self.f.db)

    def test_existing_provider_and_registration_on_v3_v4(self):
        from dashboard.sqlite_provider import SQLiteProvider
        from database.image_registry import register_image
        from database.tests.test_image_registry import png_bytes
        provider=SQLiteProvider(self.f.db,self.f.sources,self.f.root/"staging")
        self.assertEqual(provider.schema_info()["version"],3)
        self.assertEqual(len(provider.images()),3)
        migrate_database_v4(self.f.db)
        self.assertEqual(provider.schema_info()["version"],4)
        self.assertEqual(len(provider.images()),3)
        p=self.f.sources/"w1_12_c_extra.png";p.write_bytes(png_bytes(pixel=99))
        with closing(connect_database(self.f.db)) as c:
            register_image(c,p,image_id="TEST_NEW",batch_id="TEST_BATCH",physical_site_id="W1/D12/c",
                           metadata_status="filename_parsed",condition_id="TEST_CONDITION")
        self.assertEqual(provider.image_count(),4)
        self.assertEqual(provider.image_record("TEST_NEW")["die_code"],"12")


if __name__=="__main__":unittest.main()
