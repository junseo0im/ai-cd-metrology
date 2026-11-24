"""Permanent deletion of registered images (TASK 18-B-12A).

Scope of ONE image (ownership verified before anything is touched):
* DB rows: images row + its analysis_image_runs and their defect_detections / metrology_measurements /
  line_judgments. Physical Die/Site, Wafer, process, Batch, Batch-Wafer and analysis_batch_runs (run
  history) are kept — an emptied Batch is reported, never removed automatically.
* Files: the registered original's own folder (`<storage_root>/<registration>/<uuid>/<file>`; exactly one
  file, not referenced by any other image) and, per analysis run, the image's own result folder
  `<run>/images/<sha256(image_id)>/` inside the result root. Anything shared, outside those roots or not
  matching the registered SHA-256 refuses the WHOLE deletion.

SQLite and the file system cannot share one transaction, so:
  1. BEGIN IMMEDIATE (no concurrent writer) and re-plan inside the lock; the plan token must match the
     one the user confirmed (stale confirmation -> refused, already-deleted IDs -> refused).
  2. Move every owned folder into a quarantine folder, journaling each move (manifest.json).
  3. Delete the DB rows, verify row counts and PRAGMA foreign_key_check, COMMIT.
  4. Remove the quarantine folder and write a deletion log.
Any error before COMMIT rolls the DB back and moves the files back. A crash leaves a journal that
`recover_pending` resolves from the DB state (rows present -> restore files; rows gone -> purge).
"""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import uuid

from database.init_db import PROJECT_ROOT, connect_database
from database.migrate_v4 import refuse_operational

DEFAULT_QUARANTINE = PROJECT_ROOT / "runtime_data" / "deletion_quarantine"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "runtime_data" / "deletion_logs"
MAX_IMAGES = 5000
_LOCK = threading.Lock()
DEPENDENT_TABLES = ("line_judgments", "metrology_measurements", "defect_detections")


class DeletionRefused(ValueError):
    """User-facing reason; nothing was changed."""


def _norm(path):
    return os.path.normcase(os.path.abspath(str(path)))


def _inside(path, root):
    root = _norm(root).rstrip("\\/")
    return _norm(path).startswith(root + os.sep)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _marks(values):
    return ",".join("?" for _ in values)


def build_plan(connection, image_ids, *, storage_root, result_root):
    """Read-only deletion plan with a token binding IDs, files and dependent rows."""
    ids = sorted({str(i) for i in image_ids or () if isinstance(i, str) and i.strip()})
    if not ids:
        raise DeletionRefused("삭제할 이미지를 선택하세요.")
    if len(ids) > MAX_IMAGES:
        raise DeletionRefused(f"한 번에 최대 {MAX_IMAGES}장까지 삭제할 수 있습니다.")
    rows = connection.execute(
        "SELECT i.image_id,i.batch_id,i.storage_path,i.sha256,i.original_filename,d.wafer_id FROM images i "
        "JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id JOIN dies d ON d.die_id=p.die_id "
        f"WHERE i.image_id IN ({_marks(ids)}) ORDER BY i.image_id", ids).fetchall()
    found = {r[0] for r in rows}
    if found != set(ids):
        raise DeletionRefused(f"이미 삭제되었거나 존재하지 않는 이미지 {len(set(ids) - found)}장이 포함되어 있습니다. 다시 검색하세요.")
    others = connection.execute(f"SELECT storage_path FROM images WHERE image_id NOT IN ({_marks(ids)})", ids).fetchall()
    other_dirs = {_norm(Path(p).parent) for (p,) in others}
    other_files = {_norm(p) for (p,) in others}
    files, seen = [], set()
    for image_id, batch_id, storage_path, sha, name, wafer in rows:
        path = Path(storage_path)
        folder = path.parent
        if not _inside(path, storage_root) or not path.is_file():
            raise DeletionRefused(f"원본 파일 위치를 확인할 수 없어 삭제하지 않았습니다: {name}")
        if _norm(path) in other_files or _norm(folder) in other_dirs or _norm(folder) in seen:
            raise DeletionRefused(f"다른 이미지와 공유하는 원본 파일·폴더가 있어 삭제하지 않았습니다: {name}")
        if [entry.name for entry in folder.iterdir()] != [path.name]:
            raise DeletionRefused(f"원본 폴더에 다른 파일이 있어 소유권을 확인할 수 없습니다: {name}")
        if _sha256(path).lower() != str(sha).lower():
            raise DeletionRefused(f"원본 SHA-256이 등록 정보와 달라 삭제하지 않았습니다: {name}")
        seen.add(_norm(folder))
        files.append({"kind": "original", "image_id": image_id, "path": str(folder)})
    runs = connection.execute(
        "SELECT a.analysis_run_id,a.image_id,a.result_path,a.defect_overlay_path,a.metrology_overlay_path,b.summary_path "
        "FROM analysis_image_runs a JOIN analysis_batch_runs b ON b.batch_run_id=a.batch_run_id "
        f"WHERE a.image_id IN ({_marks(ids)}) ORDER BY a.analysis_run_id", ids).fetchall()
    run_ids = [r[0] for r in runs]
    missing_result_dirs = 0
    for run_id, image_id, *paths, summary in runs:
        if not summary:
            if any(paths):
                raise DeletionRefused("분석 결과 폴더 위치를 확인할 수 없어 삭제하지 않았습니다.")
            continue
        folder = (Path(result_root) / summary).parent / "images" / hashlib.sha256(image_id.encode("utf-8")).hexdigest()
        if not _inside(folder, result_root):
            raise DeletionRefused("분석 결과 폴더가 결과 저장소 밖에 있어 삭제하지 않았습니다.")
        for value in paths:
            if value and not _inside(Path(result_root) / value, folder):
                raise DeletionRefused("분석 결과 파일이 이미지 전용 폴더 밖에 있어 삭제하지 않았습니다.")
        if folder.is_dir() and _norm(folder) not in seen:
            seen.add(_norm(folder))
            files.append({"kind": "result", "image_id": image_id, "path": str(folder)})
        elif not folder.is_dir():
            missing_result_dirs += 1
    dependents = {"analysis_image_runs": len(run_ids)}
    for table in DEPENDENT_TABLES:
        dependents[table] = connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE analysis_run_id IN ({_marks(run_ids)})", run_ids).fetchone()[0] if run_ids else 0
    batches = {}
    for _, batch_id, _, _, _, wafer in rows:
        entry = batches.setdefault(batch_id, {"batch_id": batch_id, "deleting": 0, "wafers": {}})
        entry["deleting"] += 1
        entry["wafers"][wafer] = entry["wafers"].get(wafer, 0) + 1
    for batch_id, entry in batches.items():
        entry["total"] = connection.execute("SELECT COUNT(*) FROM images WHERE batch_id=?", (batch_id,)).fetchone()[0]
        entry["remaining"] = entry["total"] - entry["deleting"]
    plan = {"image_ids": ids, "files": files, "run_ids": run_ids, "dependents": dependents,
            "wafers": sorted({r[5] for r in rows}), "batches": sorted(batches.values(), key=lambda b: b["batch_id"]),
            "original_files": sum(f["kind"] == "original" for f in files),
            "result_folders": sum(f["kind"] == "result" for f in files), "missing_result_folders": missing_result_dirs,
            "emptied_batches": sorted(b for b, e in batches.items() if e["remaining"] == 0),
            "images": [{"image_id": r[0], "batch_id": r[1], "wafer_id": r[5], "sha256": r[3], "original_filename": r[4]}
                       for r in rows]}
    payload = json.dumps([ids, [f["path"] for f in files], run_ids, dependents], sort_keys=True)
    plan["token"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return plan


def plan_deletion(db_path, image_ids, *, storage_root, result_root):
    with closing(connect_database(db_path, readonly=True)) as connection:
        return build_plan(connection, image_ids, storage_root=storage_root, result_root=result_root)


def _write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(temp, path)


def _restore(manifest):
    for move in reversed(manifest["moves"]):
        source, target = Path(move["from"]), Path(move["to"])
        if target.exists() and not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(source))


def recover_pending(db_path, quarantine_root=None):
    """Resolve journals left by a crash: DB rows present -> files restored; rows gone -> quarantine purged."""
    root = Path(quarantine_root or os.environ.get("IDA_DELETION_QUARANTINE") or DEFAULT_QUARANTINE)
    report = {"restored": [], "purged": []}
    if not root.is_dir():
        return report
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _norm(manifest.get("db_path", "")) != _norm(Path(db_path).resolve()):
            continue
        with closing(connect_database(db_path, readonly=True)) as connection:
            ids = manifest["image_ids"]
            present = connection.execute(f"SELECT COUNT(*) FROM images WHERE image_id IN ({_marks(ids)})", ids).fetchone()[0]
        if present:
            _restore(manifest)
            report["restored"].append(manifest["deletion_id"])
        else:
            report["purged"].append(manifest["deletion_id"])
        shutil.rmtree(manifest_path.parent, ignore_errors=True)
    return report


def delete_images(db_path, image_ids, *, expected_token, storage_root, result_root, actor,
                  quarantine_root=None, log_root=None, allow_operational=False, _fail_at=None):
    """Delete exactly the confirmed images. Raises DeletionRefused (nothing changed) or restores on error."""
    if not allow_operational:
        try:
            refuse_operational(db_path)
        except PermissionError:
            raise DeletionRefused("운영 DB 영구 삭제는 서버 설정으로 비활성 상태입니다.") from None
    quarantine_root = Path(quarantine_root or os.environ.get("IDA_DELETION_QUARANTINE") or DEFAULT_QUARANTINE)
    log_root = Path(log_root or os.environ.get("IDA_DELETION_LOG_ROOT") or DEFAULT_LOG_ROOT)
    if not _LOCK.acquire(blocking=False):
        raise DeletionRefused("다른 삭제 작업이 진행 중입니다. 완료 후 다시 시도하세요.")
    try:
        recover_pending(db_path, quarantine_root)
        connection = connect_database(db_path)
        manifest = None
        try:
            connection.execute("BEGIN IMMEDIATE")
            plan = build_plan(connection, image_ids, storage_root=storage_root, result_root=result_root)
            if plan["token"] != expected_token:
                raise DeletionRefused("확인 이후 삭제 대상이 바뀌었습니다. 다시 확인하세요.")
            deletion_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
            folder = quarantine_root / deletion_id
            folder.mkdir(parents=True, exist_ok=False)
            manifest = {"deletion_id": deletion_id, "db_path": str(Path(db_path).resolve()), "state": "moving",
                        "image_ids": plan["image_ids"], "moves": []}
            journal = folder / "manifest.json"
            _write(journal, manifest)
            for index, item in enumerate(plan["files"]):
                target = folder / f"{index:05d}_{item['kind']}"
                shutil.move(item["path"], str(target))
                manifest["moves"].append({"from": item["path"], "to": str(target), "kind": item["kind"]})
                _write(journal, manifest)
                if _fail_at == "move" and index == len(plan["files"]) // 2:
                    raise RuntimeError("injected failure while moving files")
            ids, runs = plan["image_ids"], plan["run_ids"]
            deleted = {}
            if runs:
                for table in DEPENDENT_TABLES:
                    deleted[table] = connection.execute(
                        f"DELETE FROM {table} WHERE analysis_run_id IN ({_marks(runs)})", runs).rowcount
            deleted["analysis_image_runs"] = connection.execute(
                f"DELETE FROM analysis_image_runs WHERE image_id IN ({_marks(ids)})", ids).rowcount
            deleted["images"] = connection.execute(f"DELETE FROM images WHERE image_id IN ({_marks(ids)})", ids).rowcount
            expected = {**{t: plan["dependents"][t] for t in DEPENDENT_TABLES if runs},
                        "analysis_image_runs": plan["dependents"]["analysis_image_runs"], "images": len(ids)}
            if deleted != expected:
                raise RuntimeError(f"deleted row counts differ from the plan: {deleted} != {expected}")
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise RuntimeError("foreign key check failed after deletion")
            if _fail_at == "db":
                raise RuntimeError("injected failure before commit")
            manifest["state"] = "committing"
            _write(journal, manifest)
            connection.commit()
        except BaseException:
            connection.rollback()
            if manifest is not None:
                _restore(manifest)
                shutil.rmtree(quarantine_root / manifest["deletion_id"], ignore_errors=True)
            raise
        finally:
            connection.close()
        log_root.mkdir(parents=True, exist_ok=True)
        log = {"deletion_id": deletion_id, "deleted_at": datetime.now(timezone.utc).isoformat(), "actor": actor,
               "db_path": manifest["db_path"], "images": plan["images"], "deleted_rows": deleted,
               "original_folders": plan["original_files"], "result_folders": plan["result_folders"],
               "emptied_batches": plan["emptied_batches"]}
        _write(log_root / f"{deletion_id}.json", log)
        shutil.rmtree(folder, ignore_errors=True)
        return {"deletion_id": deletion_id, "images": len(plan["image_ids"]), "deleted_rows": deleted,
                "original_folders": plan["original_files"], "result_folders": plan["result_folders"],
                "emptied_batches": plan["emptied_batches"], "quarantine_left": folder.exists()}
    finally:
        _LOCK.release()
