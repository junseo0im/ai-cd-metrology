"""DB-1A tests: synthetic records in disposable databases only."""
import hashlib
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "database"))
from init_db import (
    DatabaseCompatibilityError, SCHEMA_VERSION, TABLE_NAMES,
    connect_database, initialize_database,
)

TEST_ROOT = ROOT / "runtime_data" / "test_tmp"
W1 = "TEST-W1"
W2 = "TEST-W2"
B1 = "TEST-BATCH-1"
S1 = W1 + "/D11/c"


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="db1a_", dir=TEST_ROOT)
        self.directory = Path(temporary.name).resolve()
        if not self.directory.is_relative_to(TEST_ROOT.resolve()):
            raise RuntimeError("Temporary DB must stay inside runtime_data/test_tmp")
        self.addCleanup(temporary.cleanup)
        self.path = self.directory / "test.sqlite3"
        initialize_database(self.path)
        self.db = connect_database(self.path)
        self.addCleanup(self.db.close)

    def digest(self, path=None):
        return hashlib.sha256((path or self.path).read_bytes()).hexdigest()

    def wafer(self, wafer_id, *, grid=True):
        self.db.execute(
            "INSERT INTO wafers(wafer_id,process_run_id) VALUES (?,?)",
            (wafer_id, "TEST-PROCESS"),
        )
        if grid:
            for row in range(1, 4):
                for col in range(1, 9):
                    code = f"{row}{col}"
                    die_id = f"{wafer_id}/D{code}"
                    self.db.execute("INSERT INTO dies VALUES (?,?,?,?,?)",
                                    (die_id, wafer_id, code, row, col))
                    for position in ("lt", "c", "rb"):
                        self.db.execute(
                            "INSERT INTO pattern_sites(physical_site_id,die_id,position_code) VALUES (?,?,?)",
                            (die_id + "/" + position, die_id, position),
                        )

    def batch(self, batch_id=B1, wafer_id=W1, mode="wafer_batch", role="evaluation"):
        self.db.execute(
            "INSERT INTO upload_batches(batch_id,wafer_id,upload_mode,dataset_role,batch_status) "
            "VALUES (?,?,?,?,?)", (batch_id, wafer_id, mode, role, "test_pending"),
        )

    def populate(self):
        # Test-only exposure and filenames; never inserted into runtime ida.sqlite3.
        self.db.execute("INSERT INTO process_runs(process_run_id,exposure_value) VALUES (?,?)",
                        ("TEST-PROCESS", 1.0))
        self.wafer(W1)
        self.batch()
        self.db.commit()

    def image(self, image_id="TEST-IMAGE", **changes):
        record = dict(
            image_id=image_id, batch_id=B1, physical_site_id=S1, condition_id=None,
            original_filename="test_without_defect_code.png",
            storage_path="test-only/not-a-real-image.png", sha256="a"*64,
            width_px=100, height_px=80, repeat_index=None,
            legacy_defect_code=None, metadata_status="test_pending",
        )
        record.update(changes)
        columns = ",".join(record)
        self.db.execute(f"INSERT INTO images({columns}) VALUES ({','.join('?' for _ in record)})",
                        tuple(record.values()))

    def test_01_empty_seven_tables_fk_and_version(self):
        names = {r[0] for r in self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT GLOB 'sqlite_*'"
        )}
        self.assertEqual(names, set(TABLE_NAMES))
        for name in TABLE_NAMES:
            self.assertEqual(self.db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0], 0)
        self.assertEqual(self.db.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        self.assertEqual(self.db.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)

    def test_02_full_grid_and_image_hierarchy(self):
        self.populate()
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM wafers").fetchone()[0], 1)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM dies").fetchone()[0], 24)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM pattern_sites").fetchone()[0], 72)
        self.assertEqual([r[0] for r in self.db.execute("SELECT die_code FROM dies ORDER BY die_code")],
                         [f"{row}{col}" for row in range(1,4) for col in range(1,9)])
        self.image()
        result = self.db.execute(
            "SELECT w.wafer_id,d.die_code,p.position_code,i.image_id FROM images i "
            "JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
            "JOIN dies d ON d.die_id=p.die_id JOIN wafers w ON w.wafer_id=d.wafer_id"
        ).fetchone()
        self.assertEqual(result, (W1, "11", "c", "TEST-IMAGE"))
        self.assertEqual(self.db.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_03_required_primary_keys_and_foreign_keys(self):
        self.populate()
        for sql, values in [
            ("INSERT INTO process_runs(process_run_id,exposure_value) VALUES (?,?)", (None, 1)),
            ("INSERT INTO process_runs(process_run_id,exposure_value) VALUES (?,?)", ("TEST-PROCESS", 2)),
            ("INSERT INTO wafers(wafer_id,process_run_id) VALUES (?,?)", ("ORPHAN", "MISSING")),
            ("INSERT INTO pattern_sites(physical_site_id,die_id,position_code) VALUES (?,?,?)",
             ("ORPHAN-SITE", "MISSING-DIE", "c")),
        ]:
            with self.subTest(sql=sql, values=values), self.assertRaises(sqlite3.IntegrityError):
                self.db.execute(sql, values)
        for changes in [
            {"batch_id":"MISSING-BATCH"}, {"physical_site_id":"MISSING-SITE"},
            {"condition_id":"MISSING-CONDITION"}, {"image_id":None},
        ]:
            with self.subTest(changes=changes), self.assertRaises(sqlite3.IntegrityError):
                self.image(**changes)

    def test_04_die_coordinates_code_and_unique_indexes(self):
        self.populate()
        for row, col, code in [(0,1,"01"),(4,1,"41"),(1,0,"10"),(1,9,"19"),
                               (1.5,1,"11"),(1,2,"11")]:
            with self.subTest(row=row,col=col,code=code), self.assertRaises(sqlite3.IntegrityError):
                self.db.execute("INSERT INTO dies VALUES (?,?,?,?,?)",
                                ("INVALID-DIE", W1, code, row, col))
        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE"):
            self.db.execute("INSERT INTO dies VALUES (?,?,?,?,?)", ("DUPLICATE",W1,"11",1,1))
        unique_columns = {
            tuple(r[0] for r in self.db.execute(
                "SELECT name FROM pragma_index_info(?) ORDER BY seqno", (idx[1],)))
            for idx in self.db.execute("PRAGMA index_list(dies)").fetchall() if idx[2]
        }
        self.assertIn(("wafer_id","die_code"), unique_columns)
        self.assertIn(("wafer_id","row_index","column_index"), unique_columns)
        self.wafer(W2)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM dies WHERE die_code='11'").fetchone()[0],2)

    def test_05_site_positions_and_duplicate_rejection(self):
        self.populate()
        for position in ("center","LT","unknown"):
            with self.subTest(position=position), self.assertRaises(sqlite3.IntegrityError):
                self.db.execute("INSERT INTO pattern_sites(physical_site_id,die_id,position_code) VALUES (?,?,?)",
                                ("BAD-SITE", W1+"/D11", position))
        with self.assertRaisesRegex(sqlite3.IntegrityError, "UNIQUE"):
            self.db.execute("INSERT INTO pattern_sites(physical_site_id,die_id,position_code) VALUES (?,?,?)",
                            ("DUPLICATE-SITE", W1+"/D11", "c"))

    def test_06_cross_wafer_image_insert_and_update_rejected(self):
        self.populate()
        self.wafer(W2)
        self.batch("TEST-BATCH-2", W2)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "same wafer"):
            self.image(physical_site_id=W2+"/D11/c")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "same wafer"):
            self.image(batch_id="TEST-BATCH-2")
        self.image()
        for column, value in [("batch_id","TEST-BATCH-2"),("physical_site_id",W2+"/D11/c")]:
            with self.subTest(column=column), self.assertRaisesRegex(sqlite3.IntegrityError,"same wafer"):
                self.db.execute(f"UPDATE images SET {column}=? WHERE image_id=?", (value,"TEST-IMAGE"))

    def test_07_parent_reassignment_cannot_bypass_wafer_guard(self):
        self.populate()
        self.wafer(W2, grid=False)
        self.db.execute("INSERT INTO dies VALUES (?,?,?,?,?)", (W2+"/D12",W2,"12",1,2))
        self.image()
        for sql, values in [
            ("UPDATE upload_batches SET wafer_id=? WHERE batch_id=?", (W2,B1)),
            ("UPDATE dies SET wafer_id=? WHERE die_id=?", (W2,W1+"/D11")),
            ("UPDATE pattern_sites SET die_id=? WHERE physical_site_id=?", (W2+"/D12",S1)),
        ]:
            with self.subTest(sql=sql), self.assertRaisesRegex(sqlite3.IntegrityError,"same wafer"):
                self.db.execute(sql, values)

    def test_08_many_images_per_site_and_duplicate_hash_preserved(self):
        self.populate()
        for index in range(80):
            self.image(f"TEST-IMAGE-{index}", repeat_index=index)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM images").fetchone()[0],80)
        self.assertEqual(self.db.execute("SELECT COUNT(DISTINCT sha256) FROM images").fetchone()[0],1)
        self.assertEqual(self.db.execute("SELECT legacy_defect_code,metadata_status FROM images LIMIT 1").fetchone(),
                         (None,"test_pending"))

    def test_09_batch_modes_and_dataset_roles(self):
        self.populate()
        for mode in ("wafer_batch","repeatability","lighting"):
            for role in ("evaluation","training_candidate","development"):
                self.batch(mode+"-"+role,mode=mode,role=role)
        for mode, role in [("Wafer Batch","evaluation"),("wafer_batch","EVALUATION"),
                           ("unknown","development"),("lighting","test")]:
            with self.subTest(mode=mode,role=role), self.assertRaises(sqlite3.IntegrityError):
                self.batch("INVALID-BATCH",mode=mode,role=role)
        self.db.execute(
            "INSERT INTO upload_batches(batch_id,wafer_id,upload_mode,batch_status) VALUES (?,?,?,?)",
            ("DEFAULT-ROLE",W1,"wafer_batch","test_pending"))
        self.assertEqual(self.db.execute(
            "SELECT dataset_role FROM upload_batches WHERE batch_id='DEFAULT-ROLE'").fetchone()[0],"evaluation")

    def test_10_conditions_keep_unknown_off_on_distinct_and_allow_other_magnification(self):
        for i, auto in enumerate((None,0,1)):
            self.db.execute(
                "INSERT INTO acquisition_conditions(condition_id,magnification,auto_exposure,camera_setting_json) "
                "VALUES (?,?,?,?)", (f"COND-{i}",60,auto,'{"test_only":true}'))
        self.assertEqual([r[0] for r in self.db.execute(
            "SELECT auto_exposure FROM acquisition_conditions ORDER BY condition_id")],[None,0,1])
        for auto in (2,-1,1.5,"ON"):
            with self.subTest(auto=auto), self.assertRaises(sqlite3.IntegrityError):
                self.db.execute(
                    "INSERT INTO acquisition_conditions(condition_id,magnification,auto_exposure) VALUES (?,?,?)",
                    ("BAD-COND",20,auto))
        for magnification, camera_json in [(0,None),(-1,None),(20,"not json")]:
            with self.subTest(magnification=magnification,camera_json=camera_json), self.assertRaises(sqlite3.IntegrityError):
                self.db.execute(
                    "INSERT INTO acquisition_conditions(condition_id,magnification,camera_setting_json) VALUES (?,?,?)",
                    ("BAD-COND",magnification,camera_json))

    def test_11_image_dimensions_hash_and_repeat_type(self):
        self.populate()
        for changes in [{"width_px":0},{"height_px":-1},{"width_px":1.5},
                        {"sha256":"short"},{"sha256":"z"*64},{"repeat_index":"not-an-index"}]:
            with self.subTest(changes=changes), self.assertRaises(sqlite3.IntegrityError):
                self.image(**changes)
        self.image(sha256="A"*64)
        self.assertEqual(self.db.execute("SELECT sha256 FROM images").fetchone()[0],"A"*64)

    def test_12_timestamps_use_canonical_utc_iso8601(self):
        self.populate()
        self.image(captured_at="2026-01-01T12:30:00Z")
        stamp=self.db.execute("SELECT created_at FROM images").fetchone()[0]
        self.assertRegex(stamp,r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        for invalid in ("not-a-date","2026-02-30T12:00:00Z","2026-01-01 12:30:00",
                        "2026-01-01T12:30:00+09:00"):
            with self.subTest(invalid=invalid), self.assertRaises(sqlite3.IntegrityError):
                self.db.execute("UPDATE images SET captured_at=?", (invalid,))
        self.assertIsNone(self.db.execute("SELECT processed_at FROM process_runs").fetchone()[0])

    def test_13_reinitialization_preserves_file_and_data(self):
        self.populate()
        self.image()
        self.db.commit()
        before=self.digest()
        self.assertEqual(initialize_database(self.path), self.path)
        self.assertEqual(self.digest(), before)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM images").fetchone()[0],1)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM dies").fetchone()[0],24)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM pattern_sites").fetchone()[0],72)

    def test_14_unknown_version_refused_without_modification(self):
        self.db.execute("PRAGMA user_version=999")
        self.db.commit()
        before=self.digest()
        with self.assertRaisesRegex(DatabaseCompatibilityError,"Unsupported schema version"):
            initialize_database(self.path)
        self.assertEqual(self.digest(),before)

    def test_15_same_version_changed_structure_refused_without_modification(self):
        self.db.execute("ALTER TABLE images ADD COLUMN unexpected TEXT")
        self.db.commit()
        before=self.digest()
        with self.assertRaisesRegex(DatabaseCompatibilityError,"Schema mismatch"):
            initialize_database(self.path)
        self.assertEqual(self.digest(),before)

    def test_16_unversioned_existing_database_is_not_overwritten(self):
        other=self.directory/"unversioned.sqlite3"
        connection=sqlite3.connect(other)
        try:
            connection.execute("CREATE TABLE preserved(value TEXT)")
            connection.execute("INSERT INTO preserved VALUES ('test-only')")
            connection.commit()
        finally:
            connection.close()
        before=self.digest(other)
        with self.assertRaises(DatabaseCompatibilityError):
            initialize_database(other)
        self.assertEqual(self.digest(other),before)

    def test_17_existing_fk_corruption_is_reported_not_repaired(self):
        self.db.close()
        connection=sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("INSERT INTO wafers(wafer_id,process_run_id) VALUES ('ORPHAN','MISSING')")
            connection.commit()
        finally:
            connection.close()
        before=self.digest()
        with self.assertRaisesRegex(DatabaseCompatibilityError,"Foreign key violations"):
            initialize_database(self.path)
        self.assertEqual(self.digest(),before)


if __name__=="__main__":
    unittest.main()
