"""Integrated upload/type tests. All files and DBs are disposable fixtures."""
from contextlib import closing
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from database import upload_storage as store
from database.batch_registry import preview_batch,prepare_registration,register_selection
from database.init_db import initialize_database,connect_database
from database.migrate_v2 import initialize_database_v2
from database.tests.test_image_registry import png_bytes

ROOT=Path(__file__).resolve().parent/".tmp"


def zip_bytes(entries):
    output=io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as archive:
            for name,data in entries:
                archive.writestr(name,data)
    return output.getvalue()


class IntegratedRegistrationTests(unittest.TestCase):
    def setUp(self):
        ROOT.mkdir(exist_ok=True)
        tmp=tempfile.TemporaryDirectory(prefix="integrated_",dir=ROOT)
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name).resolve()
        assert self.root.is_relative_to(ROOT.resolve())
        self.path=self.root/"test.sqlite3"
        initialize_database(self.path)
        self.db=connect_database(self.path)
        self.addCleanup(self.db.close)
        self.metadata={"batch_id":"B","wafer_id":"W1","upload_mode":"wafer_batch",
                       "dataset_role":"evaluation","batch_status":"registered",
                       "process_run_id":"P","process_run":{"exposure_value":60}}

    def stage(self,uploads):
        bundle=store.stage_uploads(uploads,self.root/"staging")
        self.addCleanup(bundle["files"].cleanup)
        return bundle

    def two(self):
        return self.stage([{"filename":"w1_11_c_r1.png","content":png_bytes(pixel=40)},
                           {"filename":"w1_11_c_r2.png","content":png_bytes(pixel=80)}])["inputs"]

    def test_zip_more_than_three_and_basename_collisions_preserve_bytes(self):
        entries=[("a/w1_11_c.png",png_bytes(pixel=40)),("b/w1_11_c.png",png_bytes(pixel=80)),
                 ("w1_12_c.png",png_bytes(pixel=100)),("w1_13_c.png",png_bytes(pixel=120)),
                 ("w1_14_c.png",png_bytes(pixel=140))]
        archive=zip_bytes(entries)
        bundle=self.stage([{"filename":"wafer.zip","content":archive}])
        self.assertEqual(len(bundle["inputs"]),5)
        self.assertEqual(len({item["image_path"] for item in bundle["inputs"]}),5)
        for item,(_,content) in zip(bundle["inputs"],entries):
            self.assertEqual(Path(item["image_path"]).read_bytes(),content)
            self.assertIn("archive_member",item)
        preview=preview_batch(self.db,bundle["inputs"],self.metadata)
        self.assertEqual(preview["status_counts"],{"VALID":5})
        plan=prepare_registration(self.db,preview)
        result=register_selection(self.db,plan,approved=True,storage_root=self.root/"raw")
        self.assertEqual(result["image_count"],5)
        self.assertEqual(len(list((self.root/"raw").rglob("*.png"))),5)

    def test_zip_paths_links_duplicate_paths_non_png_and_broken_files_are_rejected(self):
        link=zipfile.ZipInfo("link.png")
        link.create_system=3
        link.external_attr=(stat.S_IFLNK|0o777)<<16
        cases=[
            zip_bytes([("../escape.png",png_bytes())]),zip_bytes([("/escape.png",png_bytes())]),
            zip_bytes([("a/b.png",png_bytes())]).replace(b"a/b.png",bytes((97,92,98,46,112,110,103))),
            zip_bytes([(link,png_bytes())]),zip_bytes([("note.txt",b"x")]),
            zip_bytes([("same.png",png_bytes()),("same.png",png_bytes())]),
            zip_bytes([("broken.png",png_bytes()[:-12])]),b"not zip",zip_bytes([])]
        for index,content in enumerate(cases):
            root=self.root/f"bad_{index}"
            with self.subTest(index=index),self.assertRaises(ValueError):
                store.stage_uploads([{"filename":"bad.zip","content":content}],root)
            self.assertFalse(root.exists())

    def test_upload_limits_cover_count_bytes_ratio_archive_entries_and_png_decode(self):
        png=png_bytes()
        archive=zip_bytes([("w1_11_c.png",png),("w1_12_c.png",png_bytes(pixel=80))])
        for key,value in (("MAX_IMAGES",1),("MAX_FILE_BYTES",1),("MAX_TOTAL_BYTES",1),
                          ("MAX_ARCHIVE_BYTES",1),("MAX_ARCHIVE_ENTRIES",1),
                          ("MAX_COMPRESSION_RATIO",0.1),("MAX_DECODED_PNG_BYTES",1)):
            root=self.root/key
            with self.subTest(limit=key),patch.object(store,key,value),self.assertRaises(ValueError):
                store.stage_uploads([{"filename":"wafer.zip","content":archive}],root)
            self.assertFalse(root.exists())

    def test_single_image_v1_blocked_v2_supported_without_reclassifying_existing_batches(self):
        inputs=self.two()
        one=inputs[:1]
        preview=preview_batch(self.db,one,{**self.metadata,"upload_mode":"single_image"})
        self.assertEqual(preview["images"][0]["status"],"INVALID")
        self.assertIn("Schema Version 2",str(preview["batch_issues"]))
        path=self.root/"v2.sqlite3"
        initialize_database_v2(path)
        with closing(connect_database(path)) as v2:
            preview=preview_batch(v2,one,{**self.metadata,"upload_mode":"single_image"})
            self.assertTrue(preview["registration_ready"])
            result=register_selection(v2,prepare_registration(v2,preview),approved=True,storage_root=self.root/"raw")
            self.assertEqual(result["image_count"],1)
            self.assertEqual(v2.execute("SELECT upload_mode FROM upload_batches").fetchone()[0],"single_image")
            bad=preview_batch(v2,inputs,{**self.metadata,"batch_id":"TWO","upload_mode":"single_image"})
            self.assertTrue(all(row["status"]=="INVALID" for row in bad["images"]))

    def test_repeatability_numbers_and_same_site_are_validated(self):
        inputs=self.two()
        batch={**self.metadata,"upload_mode":"repeatability"}
        preview=preview_batch(self.db,inputs,batch)
        self.assertTrue(all(row["status"]=="NEEDS_METADATA" for row in preview["images"]))
        for index,item in enumerate(inputs,1):
            item["repeat_index"]=index
        preview=preview_batch(self.db,inputs,batch)
        self.assertEqual(preview["status_counts"],{"VALID":2})
        self.assertFalse(any(row["duplicate_image_ids"] or row["duplicate_input_indices"] for row in preview["images"]))
        inputs[1]["repeat_index"]=1
        self.assertEqual(preview_batch(self.db,inputs,batch)["status_counts"],{"INVALID":2})
        inputs[1]["repeat_index"]=2
        result=register_selection(self.db,prepare_registration(self.db,preview),approved=True)
        self.assertEqual(result["image_count"],2)
        self.assertEqual(self.db.execute("SELECT repeat_index FROM images ORDER BY repeat_index").fetchall(),[(1,),(2,)])
        inputs[1]["manual_metadata"]={"wafer_id":"W1","die_code":"12","position_code":"c"}
        bad=preview_batch(self.db,inputs,{**batch,"batch_id":"OTHER"})
        self.assertEqual(bad["status_counts"],{"INVALID":2})

    def test_lighting_requires_real_metadata_and_preserves_unknown_off_on(self):
        inputs=self.two()
        batch={**self.metadata,"upload_mode":"lighting"}
        self.assertEqual(preview_batch(self.db,inputs,batch)["status_counts"],{"NEEDS_METADATA":2})
        for index,item in enumerate(inputs):
            item["acquisition_condition"]={"condition_id":f"C{index}","magnification":20,
                "illumination_setting":["lg","lm"][index],"auto_exposure":[None,0][index]}
        preview=preview_batch(self.db,inputs,batch)
        self.assertEqual(preview["status_counts"],{"VALID":2})
        register_selection(self.db,prepare_registration(self.db,preview),approved=True)
        self.assertEqual(self.db.execute("SELECT illumination_setting,auto_exposure FROM acquisition_conditions "
                                        "ORDER BY condition_id").fetchall(),[("lg",None),("lm",0)])
        self.assertEqual(self.db.execute("SELECT COUNT(DISTINCT condition_id) FROM images").fetchone()[0],2)

    def test_lighting_condition_conflict_and_missing_magnification_do_not_insert(self):
        inputs=self.two()
        for index,item in enumerate(inputs):
            item["acquisition_condition"]={"condition_id":"SAME","magnification":20,
                "illumination_setting":["lg","lm"][index]}
        batch={**self.metadata,"upload_mode":"lighting"}
        self.assertEqual(preview_batch(self.db,inputs,batch)["status_counts"],{"INVALID":2})
        inputs[0]["acquisition_condition"]["magnification"]=None
        inputs[1]["acquisition_condition"]["condition_id"]="OTHER"
        preview=preview_batch(self.db,inputs,batch)
        self.assertEqual(preview["images"][0]["status"],"NEEDS_METADATA")
        self.assertFalse(preview["registration_ready"])
        self.assertEqual(self.db.execute("SELECT count(*) FROM acquisition_conditions").fetchone()[0],0)
