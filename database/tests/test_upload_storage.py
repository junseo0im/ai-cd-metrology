"""DB-1D storage tests: synthetic PNGs, disposable DB and storage only."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from database.batch_registry import BatchRegistrationError, preview_batch, register_batch
from database.image_registry import ImageRegistrationError, get_image
from database.init_db import TABLE_NAMES, connect_database, initialize_database
from database.upload_storage import NewFiles, file_sha256, safe_filename, stage_uploads
from database.tests.test_image_registry import png_bytes

TEST_ROOT=Path(__file__).resolve().parent/".tmp"


class UploadStorageTests(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(exist_ok=True)
        temporary=tempfile.TemporaryDirectory(prefix="db1d_",dir=TEST_ROOT)
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.path=self.root/"test.sqlite3"
        initialize_database(self.path)
        self.db=connect_database(self.path)
        self.addCleanup(self.db.close)
        self.storage=self.root/"raw"
        self.batch={"batch_id":"TEST-BATCH","wafer_id":"W1","upload_mode":"wafer_batch",
                    "dataset_role":"evaluation","batch_status":"registered",
                    "process_run_id":"TEST-PROCESS","process_run":{"exposure_value":1.0}}

    def stage(self, uploads=None):
        uploads=uploads or [{"filename":"w1_12_c.png","content":png_bytes(pixel=40)},
                            {"filename":"w1_12_rb_pn.png","content":png_bytes(pixel=80)},
                            {"filename":"unknown.png","content":png_bytes(pixel=120)}]
        bundle=stage_uploads(uploads,self.root/"staging")
        self.addCleanup(bundle["files"].cleanup)
        if uploads[-1]["filename"]=="unknown.png":
            bundle["inputs"][-1]["manual_metadata"]={"wafer_id":"W1","die_code":"38","position_code":"lt"}
        return bundle

    def counts(self):
        return {name:self.db.execute("SELECT COUNT(*) FROM "+name).fetchone()[0] for name in TABLE_NAMES}

    def saved_files(self):
        return list(self.storage.rglob("*")) if self.storage.exists() else []

    def test_three_uploads_staged_and_preview_without_db_insert(self):
        bundle=self.stage()
        preview=preview_batch(self.db,bundle["inputs"],self.batch)
        self.assertEqual(preview["status_counts"],{"VALID":3})
        self.assertFalse(any(self.counts().values()))
        self.assertFalse(self.storage.exists())
        self.assertEqual(len(bundle["inputs"]),3)
        for item in bundle["inputs"]:
            self.assertEqual(file_sha256(item["image_path"]),
                             next(r["sha256"] for r in preview["images"] if r["image_id"]==item["image_id"]))

    def test_approval_required_before_persistent_files_or_insert(self):
        bundle=self.stage()
        preview=preview_batch(self.db,bundle["inputs"],self.batch)
        with self.assertRaises(BatchRegistrationError):
            register_batch(self.db,preview,storage_root=self.storage)
        self.assertFalse(self.storage.exists())
        self.assertFalse(any(self.counts().values()))

    def test_success_preserves_native_bytes_and_persistent_readback(self):
        bundle=self.stage()
        preview=preview_batch(self.db,bundle["inputs"],self.batch)
        content={item["image_id"]:Path(item["image_path"]).read_bytes() for item in bundle["inputs"]}
        result=register_batch(self.db,preview,approved=True,storage_root=self.storage)
        self.assertEqual(result["image_count"],3)
        bundle["files"].cleanup()
        reader=connect_database(self.path,readonly=True)
        try:
            for row in preview["images"]:
                stored=get_image(reader,row["image_id"])
                path=Path(stored["storage_path"])
                self.assertTrue(path.is_relative_to(self.storage.resolve()))
                self.assertEqual(path.name,row["original_filename"])
                self.assertEqual(path.read_bytes(),content[row["image_id"]])
                self.assertEqual(file_sha256(path),stored["sha256"])
                self.assertEqual(stored["sha256"],row["sha256"])
                self.assertEqual((stored["width_px"],stored["height_px"]),(3,2))
        finally:
            reader.close()
        self.assertEqual(len(list(self.storage.rglob("*.png"))),3)

    def test_duplicate_filenames_get_separate_paths_and_require_confirmation(self):
        uploads=[{"filename":"w1_12_c.png","content":png_bytes()}]*2
        bundle=self.stage(uploads)
        self.assertNotEqual(bundle["inputs"][0]["image_path"],bundle["inputs"][1]["image_path"])
        preview=preview_batch(self.db,bundle["inputs"],self.batch)
        self.assertEqual(preview["status_counts"],{"DUPLICATE_CANDIDATE":2})
        with self.assertRaisesRegex(BatchRegistrationError,"per-image"):
            register_batch(self.db,preview,approved=True,storage_root=self.storage)
        self.assertFalse(self.saved_files())
        result=register_batch(self.db,preview,approved=True,storage_root=self.storage,
                              approved_duplicate_image_ids=[item["image_id"] for item in bundle["inputs"]])
        paths=[get_image(self.db,key)["storage_path"] for key in result["image_ids"]]
        self.assertEqual(len(set(paths)),2)
        self.assertTrue(all(Path(path).name=="w1_12_c.png" for path in paths))

    def test_file_write_failure_rolls_back_and_cleans_partial_new_files_only(self):
        bundle=self.stage()
        preview=preview_batch(self.db,bundle["inputs"],self.batch)
        self.storage.mkdir()
        existing=self.storage/"existing.png"
        existing.write_bytes(b"pre-existing bytes")
        def failure(source,target):
            target.write(b"partial copy")
            raise OSError("test disk write failure")
        with patch("database.upload_storage.shutil.copyfileobj",side_effect=failure):
            with self.assertRaisesRegex(OSError,"disk write"):
                register_batch(self.db,preview,approved=True,storage_root=self.storage)
        self.assertFalse(any(self.counts().values()))
        self.assertEqual(existing.read_bytes(),b"pre-existing bytes")
        self.assertEqual(self.saved_files(),[existing])
        self.assertFalse(self.db.in_transaction)

    def test_second_insert_failure_rolls_back_db_and_all_new_files(self):
        bundle=self.stage()
        inputs=bundle["inputs"]
        self.db.execute("""CREATE TRIGGER test_storage_failure BEFORE INSERT ON images
            WHEN NEW.original_filename='w1_12_rb_pn.png'
            AND EXISTS(SELECT 1 FROM images WHERE original_filename='w1_12_c.png')
            BEGIN SELECT RAISE(ABORT,'test INSERT failure'); END""")
        self.db.commit()
        preview=preview_batch(self.db,inputs,self.batch)
        self.storage.mkdir()
        existing=self.storage/"keep.txt"
        existing.write_text("keep")
        with self.assertRaisesRegex(ImageRegistrationError,"test INSERT failure"):
            register_batch(self.db,preview,approved=True,storage_root=self.storage)
        self.assertFalse(any(self.counts().values()))
        self.assertEqual(self.saved_files(),[existing])
        self.assertEqual(existing.read_text(),"keep")
        self.assertTrue(all(Path(item["image_path"]).exists() for item in inputs))
        self.assertFalse(self.db.in_transaction)

    def test_unsafe_original_names_never_escape_staging(self):
        for name in ("../escape.png",r"..\escape.png",r"C:\escape.png","x/y.png",
                     "/escape.png","CON.png","bad:stream.png","w1_12_c.png ","file.jpg"):
            with self.subTest(name=name),self.assertRaises(ValueError):
                safe_filename(name)
            with self.assertRaises(ValueError):
                stage_uploads([{"filename":name,"content":png_bytes()}],self.root/"staging")
        self.assertFalse((self.root/"escape.png").exists())

    def test_existing_storage_collision_is_never_adopted_or_deleted(self):
        self.storage.mkdir()
        collision=self.storage/"collision"
        collision.mkdir()
        keep=collision/"keep.png"
        keep.write_bytes(b"existing")
        class FixedUUID:
            hex="collision"
        files=NewFiles(self.storage)
        with patch("database.upload_storage.uuid.uuid4",return_value=FixedUUID()):
            with self.assertRaises(FileExistsError):
                files.write("w1_12_c.png",lambda target:target.write(png_bytes()))
        files.cleanup()
        self.assertEqual(keep.read_bytes(),b"existing")

    def test_content_changed_during_copy_is_rejected_and_cleaned(self):
        bundle=self.stage()
        preview=preview_batch(self.db,bundle["inputs"],self.batch)
        original=NewFiles.copy
        def changed(files,source,name,digest):
            Path(source).write_bytes(png_bytes(pixel=200))
            return original(files,source,name,digest)
        with patch("database.batch_registry.NewFiles.copy",new=changed):
            with self.assertRaisesRegex(ValueError,"hash differs"):
                register_batch(self.db,preview,approved=True,storage_root=self.storage)
        self.assertFalse(any(self.counts().values()))
        self.assertFalse(self.saved_files())

    def test_stale_metadata_preview_is_refused_before_copy(self):
        bundle=self.stage()
        preview=preview_batch(self.db,bundle["inputs"],self.batch)
        preview["inputs"][0]["manual_metadata"]={"wafer_id":"W1","die_code":"38","position_code":"rb"}
        with self.assertRaisesRegex(BatchRegistrationError,"Preview changed"):
            register_batch(self.db,preview,approved=True,storage_root=self.storage)
        self.assertFalse(self.saved_files())
        self.assertFalse(any(self.counts().values()))

    def test_explicit_operational_optin_still_requires_managed_storage(self):
        bundle=self.stage()
        preview=preview_batch(self.db,bundle["inputs"],self.batch)
        with self.assertRaisesRegex(BatchRegistrationError,"managed image storage"):
            register_batch(self.db,preview,approved=True,allow_operational=True)


if __name__=="__main__":
    unittest.main()
