"""DB-1B: create 1-3 tiny PNG fixtures per case; never use the operational DB."""
import hashlib
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from database.filename_parser import FilenameParseError, parse_filename
from database.image_registry import ImageRegistrationError, PngHeaderError, get_image, register_image
from database.init_db import connect_database, initialize_database

TEST_ROOT = ROOT / "runtime_data" / "test_tmp"


def png_bytes(width=3, height=2, pixel=40):
    """Small real RGB PNG generated entirely with the standard library."""
    def chunk(kind, data):
        return (struct.pack(">I",len(data)) + kind + data
                + struct.pack(">I",zlib.crc32(kind+data) & 0xffffffff))
    header=struct.pack(">IIBBBBB",width,height,8,2,0,0,0)
    pixels=(b"\0"+bytes([pixel,pixel,pixel])*width)*height
    return (b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",header)
            +chunk(b"IDAT",zlib.compress(pixels))+chunk(b"IEND",b""))


class FilenameParserTests(unittest.TestCase):
    def test_01_valid_names_optional_and_unknown_defect_codes(self):
        rb=parse_filename("w1_12_rb_pn.png")
        self.assertEqual((rb.wafer_id,rb.die_code,rb.row_index,rb.column_index,
                          rb.position_code,rb.legacy_defect_code),("W1","12",1,2,"rb","pn"))
        center=parse_filename("w1_12_c.png")
        self.assertIsNone(center.legacy_defect_code)
        corner=parse_filename("w1_38_lt.png")
        self.assertEqual((corner.row_index,corner.column_index,corner.position_code),(3,8,"lt"))
        self.assertEqual(parse_filename("w1_12_rb_future_label-9.png").legacy_defect_code,"future_label-9")
        self.assertEqual(parse_filename("w01_12_rb_PN.PNG").wafer_id,"W01")
        self.assertEqual(parse_filename("w1_12_rb_PN.png").legacy_defect_code,"PN")

    def test_02_invalid_names_die_position_and_extension_rejected(self):
        for name in ("w1_39_c.png","w1_09_lt.png","w1_41_rb.png","w1_12_left.png",
                     "w1_12_c.jpg","w1_1_c.png","w1_12_c_.png",
                     "w1_12_c_pn.png.bak","folder/w1_12_c.png",r"folder\w1_12_c.png","",None):
            with self.subTest(name=name), self.assertRaises(FilenameParseError):
                parse_filename(name)


class ImageRegistryTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix="db1b_",dir=TEST_ROOT)
        self.directory=Path(temporary.name).resolve()
        if not self.directory.is_relative_to(TEST_ROOT.resolve()):
            raise RuntimeError("Test files must remain inside runtime_data/test_tmp")
        self.addCleanup(temporary.cleanup)
        self.db_path=self.directory/"test.sqlite3"
        initialize_database(self.db_path)
        self.db=connect_database(self.db_path)
        self.addCleanup(self.db.close)
        self.db.execute("INSERT INTO process_runs(process_run_id,exposure_value) VALUES ('TEST-PROCESS',1)")
        self.add_wafer("W1","B1")
        self.db.execute(
            "INSERT INTO acquisition_conditions(condition_id,magnification) VALUES ('C1',20)")
        self.db.commit()

    def add_wafer(self,wafer,batch):
        self.db.execute("INSERT INTO wafers(wafer_id,process_run_id) VALUES (?,?)",(wafer,"TEST-PROCESS"))
        for code in ("12","38"):
            die=f"{wafer}/D{code}"
            self.db.execute("INSERT INTO dies VALUES (?,?,?,?,?)",(die,wafer,code,int(code[0]),int(code[1])))
            for position in ("lt","c","rb"):
                self.db.execute(
                    "INSERT INTO pattern_sites(physical_site_id,die_id,position_code) VALUES (?,?,?)",
                    (die+"/"+position,die,position))
        self.db.execute(
            "INSERT INTO upload_batches(batch_id,wafer_id,upload_mode,dataset_role,batch_status) "
            "VALUES (?,?,'wafer_batch','evaluation','test_pending')",(batch,wafer))

    def png(self,filename="w1_12_rb_pn.png",*,width=3,height=2,pixel=40):
        path=self.directory/filename
        path.write_bytes(png_bytes(width,height,pixel))
        return path

    def register(self,path,image_id="I1",**changes):
        fields=dict(image_id=image_id,batch_id="B1",physical_site_id="W1/D12/rb",
                    metadata_status="test_pending")
        fields.update(changes)
        return register_image(self.db,path,**fields)

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def test_03_three_images_registered_queried_and_originals_unchanged(self):
        examples=[
            ("w1_12_rb_pn.png","W1/D12/rb",3,2,40,"C1","pn"),
            ("w1_12_c.png","W1/D12/c",4,3,80,None,None),
            ("w1_38_lt.png","W1/D38/lt",5,4,120,"C1",None),
        ]
        self.assertIsNone(get_image(self.db,"MISSING"))
        for index,(filename,site,width,height,pixel,condition,legacy) in enumerate(examples):
            path=self.png(filename,width=width,height=height,pixel=pixel)
            before=path.read_bytes()
            result=self.register(path,f"I{index}",physical_site_id=site,condition_id=condition)
            stored=get_image(self.db,f"I{index}")
            self.assertEqual(stored["image_id"],f"I{index}")
            self.assertEqual(stored["wafer_id"],"W1")
            self.assertEqual(stored["die_code"],filename.split("_")[1])
            self.assertEqual(stored["position_code"],filename.split("_")[2].split(".")[0])
            self.assertEqual(stored["physical_site_id"],site)
            self.assertEqual(stored["condition_id"],condition)
            self.assertEqual(stored["original_filename"],filename)
            self.assertEqual(stored["storage_path"],str(path.resolve()))
            self.assertEqual((stored["width_px"],stored["height_px"]),(width,height))
            self.assertEqual(stored["legacy_defect_code"],legacy)
            self.assertEqual(stored["metadata_status"],"test_pending")
            self.assertEqual(stored["sha256"],hashlib.sha256(before).hexdigest())
            self.assertFalse(result["duplicate_sha256"])
            self.assertEqual(result["duplicate_image_ids"],())
            self.assertEqual(path.read_bytes(),before)
        self.assertEqual(self.count(),3)
        self.assertEqual(self.db.execute("PRAGMA foreign_key_check").fetchall(),[])

    def test_04_cross_wafer_batch_site_rejected(self):
        self.add_wafer("W2","B2")
        self.db.commit()
        path=self.png()
        for changes in ({"physical_site_id":"W2/D12/rb"},{"batch_id":"B2"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ImageRegistrationError,"same wafer"):
                self.register(path,**changes)
            self.assertFalse(self.db.in_transaction)
        self.assertEqual(self.count(),0)

    def test_05_filename_identity_must_match_existing_site(self):
        for filename,site in [("w1_12_c.png","W1/D12/rb"),("w1_38_lt.png","W1/D12/lt"),
                              ("w2_12_rb.png","W1/D12/rb")]:
            path=self.png(filename)
            with self.subTest(filename=filename), self.assertRaisesRegex(ImageRegistrationError,"Filename"):
                self.register(path,physical_site_id=site)
            self.assertFalse(self.db.in_transaction)
        self.assertEqual(self.count(),0)

    def test_06_missing_parents_do_not_create_records(self):
        path=self.png()
        tables=("wafers","dies","pattern_sites","upload_batches","acquisition_conditions")
        before={t:self.db.execute("SELECT COUNT(*) FROM "+t).fetchone()[0] for t in tables}
        for changes in ({"batch_id":"MISSING"},{"physical_site_id":"MISSING"},{"condition_id":"MISSING"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ImageRegistrationError,"Unknown"):
                self.register(path,**changes)
        self.assertEqual(self.count(),0)
        self.assertEqual(before,{t:self.db.execute("SELECT COUNT(*) FROM "+t).fetchone()[0] for t in tables})

    def test_07_same_site_multiple_images_and_repeat_indices(self):
        path=self.png()
        self.register(path,"REPEAT-1",repeat_index=1)
        result=self.register(path,"REPEAT-2",repeat_index=2)
        self.assertEqual(self.count(),2)
        self.assertTrue(result["duplicate_sha256"])
        self.assertEqual(result["duplicate_image_ids"],("REPEAT-1",))
        self.assertEqual([r[0] for r in self.db.execute("SELECT repeat_index FROM images ORDER BY image_id")],[1,2])

    def test_08_duplicate_hash_flag_preserves_both_records_and_files(self):
        first=self.png("w1_12_rb.png")
        second=self.png("w1_12_rb_future.png")
        originals={path:(path.read_bytes(),path.stat().st_mtime_ns) for path in (first,second)}
        self.register(first,"ORIGINAL")
        # Hex case does not change checksum identity, and is valid in DB-1A.
        self.db.execute("UPDATE images SET sha256=upper(sha256) WHERE image_id='ORIGINAL'")
        self.db.commit()
        result=self.register(second,"DUPLICATE")
        self.assertTrue(result["duplicate_sha256"])
        self.assertEqual(result["duplicate_image_ids"],("ORIGINAL",))
        self.assertEqual(self.count(),2)
        self.assertIsNone(get_image(self.db,"ORIGINAL")["legacy_defect_code"])
        self.assertEqual(get_image(self.db,"DUPLICATE")["legacy_defect_code"],"future")
        for path,original in originals.items():
            self.assertEqual((path.read_bytes(),path.stat().st_mtime_ns),original)

    def test_09_image_id_collision_rolls_back_without_overwriting(self):
        path=self.png()
        self.register(path,"SAME-ID")
        before=get_image(self.db,"SAME-ID")
        with self.assertRaisesRegex(ImageRegistrationError,"UNIQUE"):
            self.register(path,"SAME-ID")
        self.assertFalse(self.db.in_transaction)
        self.assertEqual(self.count(),1)
        self.assertEqual(get_image(self.db,"SAME-ID"),before)

    def test_10_failure_after_insert_rolls_back_and_connection_can_retry(self):
        path=self.png()
        before=path.read_bytes()
        with patch("database.image_registry.get_image",side_effect=RuntimeError("test readback failure")):
            with self.assertRaisesRegex(RuntimeError,"test readback failure"):
                self.register(path)
        self.assertFalse(self.db.in_transaction)
        self.assertEqual(self.count(),0)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(self.register(path)["image_id"],"I1")

    def test_11_invalid_png_headers_are_rejected_unchanged(self):
        corrupt=bytearray(png_bytes())
        corrupt[29]^=1
        cases=(b"not a PNG",bytes(corrupt),png_bytes(width=0))
        for index,content in enumerate(cases):
            path=self.directory/f"w1_12_rb_bad{index}.png"
            path.write_bytes(content)
            with self.subTest(index=index), self.assertRaises(PngHeaderError):
                self.register(path)
            self.assertEqual(path.read_bytes(),content)
            self.assertFalse(self.db.in_transaction)
        self.assertEqual(self.count(),0)

    def test_12_connection_requirements_do_not_commit_caller_work(self):
        path=self.png()
        self.db.execute("PRAGMA foreign_keys=OFF")
        with self.assertRaisesRegex(ImageRegistrationError,"foreign_keys"):
            self.register(path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("UPDATE process_runs SET notes='caller work'")
        with self.assertRaisesRegex(ImageRegistrationError,"Commit fixture"):
            self.register(path)
        self.assertTrue(self.db.in_transaction)
        self.db.rollback()
        self.assertEqual(self.count(),0)

    def test_13_operational_db_guard_is_tested_using_a_mocked_temp_path(self):
        path=self.png()
        before=hashlib.sha256(self.db_path.read_bytes()).hexdigest()
        with patch("database.image_registry.DEFAULT_DB",self.db_path):
            with self.assertRaisesRegex(ImageRegistrationError,"operational DB"):
                self.register(path)
        self.assertEqual(self.count(),0)
        self.assertEqual(hashlib.sha256(self.db_path.read_bytes()).hexdigest(),before)

    def test_14_database_constraint_error_rolls_back(self):
        path=self.png()
        for changes in ({"metadata_status":""},{"captured_at":"not-an-ISO-date"},{"repeat_index":"text"}):
            with self.subTest(changes=changes), self.assertRaises(ImageRegistrationError):
                self.register(path,**changes)
            self.assertFalse(self.db.in_transaction)
            self.assertEqual(self.count(),0)


if __name__=="__main__":
    unittest.main()
