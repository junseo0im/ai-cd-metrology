"""TASK 18-B-10B-1: prototype Schema v6 on TEMPORARY copies of the operational DB.

The operational DB is opened read-only once per class for an SQLite Online Backup and is hashed
before/after; nothing is written to it. All writes go to per-test copies in a temp directory.
Images inserted here are DB rows only (fake paths/hashes); no model, Runner or file is used.
"""
from contextlib import closing
from pathlib import Path
import hashlib,shutil,sqlite3,tempfile,unittest

from database.init_db import DEFAULT_DB, DatabaseCompatibilityError, connect_database
from database.migrate_v2 import data_snapshot
from database.migrate_v4 import ANALYSIS_TABLES
from database.migrate_v6 import (migrate_database_v6, validate_v6, image_process_rows,
                                 V5_IMAGE_PROCESS_SQL, V6_IMAGE_PROCESS_SQL)

MULTI="BATCH-TEST-MULTI-W123"
PLAN={"W1":("TEST-PROC-W1-50",50.0,220.0),"W2":("TEST-PROC-W2-60",60.0,220.0),"W3":("TEST-PROC-W3-70",70.0,120.0)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipUnless(DEFAULT_DB.is_file(),"operational DB needed for a backup copy")
class SchemaV6CopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op_sha=sha(DEFAULT_DB)
        cls.tmp=Path(tempfile.mkdtemp(prefix="v6_proto_"))
        cls.source=cls.tmp/"backup_v5.sqlite3"
        with closing(sqlite3.connect(DEFAULT_DB.resolve().as_uri()+"?mode=ro",uri=True)) as src, \
             closing(sqlite3.connect(cls.source)) as dst:
            src.backup(dst)                                   # SQLite Online Backup API, read-only source.

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp,ignore_errors=True)
        assert sha(DEFAULT_DB)==cls.op_sha,"operational DB changed"

    def setUp(self):
        self.db=self.tmp/(self._testMethodName+".sqlite3")
        shutil.copyfile(self.source,self.db)

    def snapshot(self):
        with closing(connect_database(self.db,readonly=True)) as c:
            return (data_snapshot(c),{t:c.execute(f'SELECT * FROM "{t}" ORDER BY 1').fetchall() for t in ANALYSIS_TABLES},
                    image_process_rows(c,V5_IMAGE_PROCESS_SQL))

    # --- 1, 8, 9, 10: preservation, single-Wafer compatibility, FK, idempotency ------------------
    def test_01_migration_preserves_everything_and_reruns_safely(self):
        data,analysis,process=self.snapshot()
        result=migrate_database_v6(self.db)
        self.assertEqual(result,{"version":6,"migrated":True,"batch_wafers":len(data["upload_batches"])})
        with closing(connect_database(self.db,readonly=True)) as c:
            validate_v6(c)
            self.assertEqual(data_snapshot(c),data)                                  # IDs, paths, SHA unchanged.
            for table,rows in analysis.items():
                self.assertEqual(c.execute(f'SELECT * FROM "{table}" ORDER BY 1').fetchall(),rows)
            self.assertEqual(image_process_rows(c,V6_IMAGE_PROCESS_SQL),process)     # Same 노광/PEB per image.
            self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(),[])
            members=c.execute("SELECT batch_id,wafer_id,process_run_id,peb_time_s FROM batch_wafers ORDER BY 1").fetchall()
            legacy=c.execute("SELECT batch_id,wafer_id,process_run_id,peb_temperature_c FROM upload_batches ORDER BY 1").fetchall()
            self.assertEqual(members,legacy)                                          # 1 row per existing Batch.
        before=sha(self.db)
        self.assertEqual(migrate_database_v6(self.db),{"version":6,"migrated":False})
        self.assertEqual(migrate_database_v6(self.db),{"version":6,"migrated":False})
        self.assertEqual(sha(self.db),before)                                         # Reruns write nothing.

    def test_02_failure_rolls_back_completely(self):
        before=sha(self.db)
        with self.assertRaises(DatabaseCompatibilityError):
            migrate_database_v6(self.db,_fail_after_apply=True)
        self.assertEqual(sha(self.db),before)
        with closing(sqlite3.connect(self.db)) as c:
            self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0],5)
            self.assertIsNone(c.execute("SELECT 1 FROM sqlite_master WHERE name='batch_wafers'").fetchone())

    def test_03_operational_path_is_refused_without_touching_it(self):
        with self.assertRaises(PermissionError):
            migrate_database_v6(DEFAULT_DB)
        self.assertEqual(sha(DEFAULT_DB),self.op_sha)

    # --- 2-7: one Batch with three Wafers and different processes --------------------------------
    def sites(self,c,wafer):
        rows=c.execute("SELECT p.physical_site_id FROM pattern_sites p JOIN dies d ON d.die_id=p.die_id "
                       "WHERE d.wafer_id=? ORDER BY d.die_code,p.position_code",(wafer,)).fetchall()
        self.assertEqual(len(rows),72,wafer)
        return [r[0] for r in rows]

    def register_multi(self,c,wafers=PLAN):
        """Prototype of the stage-2 write order, all in the caller's transaction."""
        c.execute("INSERT INTO upload_batches(batch_id,wafer_id,upload_mode,dataset_role,batch_status) "
                  "VALUES (?,?,?,?,?)",(MULTI,sorted(wafers)[0],"wafer_batch","evaluation","registered"))
        for wafer,(process_id,exposure,peb) in sorted(wafers.items()):
            c.execute("INSERT INTO process_runs(process_run_id,exposure_value,exposure_unit) VALUES (?,?,NULL)",
                      (process_id,exposure))
            c.execute("INSERT INTO batch_wafers(batch_id,wafer_id,process_run_id,peb_time_s) VALUES (?,?,?,?)",
                      (MULTI,wafer,process_id,peb))
        n=0
        for wafer in sorted(wafers):
            for site in self.sites(c,wafer):
                n+=1
                c.execute("INSERT INTO images(image_id,batch_id,physical_site_id,original_filename,storage_path,sha256,"
                          "width_px,height_px,metadata_status) VALUES (?,?,?,?,?,?,?,?,?)",
                          (f"IMG-TEST-{n:03}",MULTI,site,f"{wafer.lower()}_{n:03}.png",f"fake/{n:03}.png",
                           f"{n:064x}",10,10,"filename_parsed"))
        return n

    def test_04_three_wafers_in_one_batch_keep_their_own_process(self):
        migrate_database_v6(self.db)
        with closing(connect_database(self.db)) as c:
            with c:
                total=self.register_multi(c)
            validate_v6(c)
            self.assertEqual(c.execute("PRAGMA foreign_key_check").fetchall(),[])
            self.assertEqual(total,216)
            # 6. Batch view: every image of the Batch.
            self.assertEqual(c.execute("SELECT COUNT(*) FROM images WHERE batch_id=?",(MULTI,)).fetchone()[0],216)
            # 3/4/7. Wafer view: per-image Wafer/Die/site and that Wafer's own 노광·PEB.
            rows=c.execute(V6_IMAGE_PROCESS_SQL+"WHERE i.batch_id=? ORDER BY i.image_id",(MULTI,)).fetchall()
            groups={}
            for _,_,wafer,die,position,exposure,unit,peb,source in rows:
                groups.setdefault((wafer,exposure,peb,unit,source),set()).add((die,position))
            self.assertEqual({k:len(v) for k,v in groups.items()},
                             {("W1",50.0,220.0,None,"batch"):72,("W2",60.0,220.0,None,"batch"):72,
                              ("W3",70.0,120.0,None,"batch"):72})
            self.assertEqual(sum(len(v) for v in groups.values()),216)              # No double counting.
            # 5. Same Wafer, other Batches/conditions stay separate (W1 already has 50/220 and 60/— Batches).
            w1=c.execute(V6_IMAGE_PROCESS_SQL+"WHERE d.wafer_id='W1'").fetchall()
            by_condition={}
            for image_id,batch_id,_,_,_,exposure,_,peb,_ in w1:
                by_condition.setdefault((exposure,peb),set()).add(batch_id)
            self.assertIn(MULTI,by_condition[(50.0,220.0)])
            self.assertNotIn(MULTI,by_condition.get((60.0,None),set()))
            # 8. Legacy single-Wafer Batches: unchanged effective process.
            legacy=c.execute(V6_IMAGE_PROCESS_SQL+"WHERE i.batch_id<>? ORDER BY i.image_id",(MULTI,)).fetchall()
            v5_view=c.execute(V5_IMAGE_PROCESS_SQL+"WHERE i.batch_id<>? ORDER BY i.image_id",(MULTI,)).fetchall()
            self.assertEqual(legacy,v5_view)
            # Multi-Wafer Batch keeps no Batch-level process.
            self.assertEqual(c.execute("SELECT process_run_id,peb_temperature_c FROM upload_batches WHERE batch_id=?",
                                       (MULTI,)).fetchone(),(None,None))

    def test_05_integrity_triggers(self):
        migrate_database_v6(self.db)
        with closing(connect_database(self.db)) as c:
            with c:
                self.register_multi(c,{"W1":PLAN["W1"],"W2":PLAN["W2"]})
            w3_site=self.sites(c,"W3")[0]
            with self.assertRaisesRegex(sqlite3.IntegrityError,"member Wafer"):     # W3 is not in this Batch.
                with c:
                    c.execute("INSERT INTO images(image_id,batch_id,physical_site_id,original_filename,storage_path,sha256,"
                              "width_px,height_px,metadata_status) VALUES ('IMG-TEST-X',?,?,'x.png','x',?,1,1,'filename_parsed')",
                              (MULTI,w3_site,"f"*64))
            with self.assertRaisesRegex(sqlite3.IntegrityError,"registered images"):
                with c:
                    c.execute("DELETE FROM batch_wafers WHERE batch_id=? AND wafer_id='W2'",(MULTI,))
            with self.assertRaisesRegex(sqlite3.IntegrityError,"immutable"):
                with c:
                    c.execute("UPDATE batch_wafers SET wafer_id='W3' WHERE batch_id=? AND wafer_id='W2'",(MULTI,))
            with self.assertRaisesRegex(sqlite3.IntegrityError,"member Wafer"):     # Moving an image to another Wafer.
                with c:
                    c.execute("UPDATE images SET physical_site_id=? WHERE image_id='IMG-TEST-001'",(w3_site,))
            validate_v6(c)

    def test_06_failed_registration_leaves_no_partial_rows(self):
        migrate_database_v6(self.db)
        before=sha(self.db)
        with closing(connect_database(self.db)) as c:
            with self.assertRaises(sqlite3.IntegrityError):
                with c:
                    self.register_multi(c)
                    c.execute("INSERT INTO images(image_id,batch_id,physical_site_id,original_filename,storage_path,sha256,"
                              "width_px,height_px,metadata_status) VALUES ('IMG-TEST-001',?,?,'dup.png','d',?,1,1,'x')",
                              (MULTI,self.sites(c,"W1")[0],"e"*64))                 # Duplicate PK -> whole Batch rolls back.
            self.assertIsNone(c.execute("SELECT 1 FROM upload_batches WHERE batch_id=?",(MULTI,)).fetchone())
            self.assertIsNone(c.execute("SELECT 1 FROM process_runs WHERE process_run_id LIKE 'TEST-PROC-%'").fetchone())
        self.assertEqual(sha(self.db),before)

    def test_07_v6_readers_accept_v6_and_get_image_uses_image_wafer(self):
        """TASK 18-B-10B-2: the 10B-1 conflict points are resolved (v6 accepted, image's own Wafer)."""
        from database.migrate_v5 import validate_analysis_schema
        from database.image_registry import get_image
        migrate_database_v6(self.db)
        with closing(connect_database(self.db)) as c:
            self.assertEqual(validate_analysis_schema(c),6)
            with c:
                self.register_multi(c)
            w2=c.execute("SELECT i.image_id FROM images i JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
                         "JOIN dies d ON d.die_id=p.die_id WHERE i.batch_id=? AND d.wafer_id='W2' LIMIT 1",(MULTI,)).fetchone()[0]
            self.assertEqual(get_image(c,w2)["wafer_id"],"W2")   # Never the Batch primary Wafer (W1).


if __name__=="__main__":
    unittest.main()
