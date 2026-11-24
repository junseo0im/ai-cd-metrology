"""Schema v6 (TASK 18-B-10B): several Wafers in ONE upload Batch.

Design: database/SCHEMA_V6_DESIGN.md. The operational DB is refused unless
allow_operational=True (explicit user approval only; never called by the Dashboard).

v5 assumption removed here: "Batch 1 = Wafer 1" (upload_batches.wafer_id plus triggers
images_same_wafer_insert/update that force every image site onto that single Wafer).

v6 additions (additive; no existing row, ID, path, hash or analysis row is changed):
  * batch_wafers(batch_id, wafer_id, process_run_id, peb_time_s): the Wafers of a Batch, each
    with its OWN process reference and PEB time (seconds). process_run_id NULL = inherit the
    Wafer's process record (same meaning as v5 upload_batches.process_run_id).
    Backfill: exactly one row per existing Batch, copied from upload_batches.
  * Image-membership triggers: an image's site Wafer (images -> pattern_sites -> dies) must be a
    member Wafer of its Batch. They replace the two v5 single-Wafer triggers.
  * Membership is immutable while images exist (no DELETE / key UPDATE).
upload_batches.wafer_id stays (NOT NULL, unchanged) as the Batch's primary (first) Wafer and is
never used to identify an image's Wafer. A single-Wafer Batch mirrors its process/PEB in
upload_batches; a multi-Wafer Batch keeps them NULL (batch_wafers is the only source).

Readers use image_process_sql() so the same query works on v5 (Batch process) and v6 (the
image's own Batch-Wafer process).
"""
from contextlib import closing
import sqlite3

from database.init_db import DatabaseCompatibilityError, connect_database, _schema_objects
from database.migrate_v2 import data_snapshot
from database.migrate_v4 import ANALYSIS_TABLES, refuse_operational
from database.migrate_v5 import BATCH_PROCESS_SQL, validate_v5


def reference_v5_connection():
    """In-memory v5 schema built by the same steps as migrate_v5.reference_v5()."""
    from database.migrate_v2 import V2_SCHEMA
    from database.migrate_v3 import PEB_SQL
    from database.migrate_v4 import _apply
    c = sqlite3.connect(":memory:")
    c.executescript(V2_SCHEMA.read_text(encoding="utf-8"))
    c.execute(PEB_SQL)
    _apply(c)
    c.execute(BATCH_PROCESS_SQL)
    return c

V5_SINGLE_WAFER_TRIGGERS = ("images_same_wafer_insert", "images_same_wafer_update")

V6_STATEMENTS = (
    "CREATE TABLE batch_wafers ("
    " batch_id TEXT NOT NULL REFERENCES upload_batches(batch_id) ON UPDATE RESTRICT ON DELETE RESTRICT,"
    " wafer_id TEXT NOT NULL REFERENCES wafers(wafer_id) ON UPDATE RESTRICT ON DELETE RESTRICT,"
    " process_run_id TEXT REFERENCES process_runs(process_run_id) ON UPDATE RESTRICT ON DELETE RESTRICT,"
    " peb_time_s REAL CHECK(peb_time_s IS NULL OR (typeof(peb_time_s) IN ('real','integer')"
    " AND peb_time_s BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308)),"
    " PRIMARY KEY (batch_id, wafer_id))",
    "CREATE INDEX idx_batch_wafers_wafer ON batch_wafers(wafer_id)",
    "DROP TRIGGER images_same_wafer_insert",
    "DROP TRIGGER images_same_wafer_update",
    "CREATE TRIGGER images_batch_wafer_insert BEFORE INSERT ON images WHEN NOT EXISTS ("
    " SELECT 1 FROM pattern_sites p JOIN dies d ON d.die_id = p.die_id"
    " JOIN batch_wafers bw ON bw.batch_id = NEW.batch_id AND bw.wafer_id = d.wafer_id"
    " WHERE p.physical_site_id = NEW.physical_site_id)"
    " BEGIN SELECT RAISE(ABORT, 'Image Wafer must be a member Wafer of its Batch'); END",
    "CREATE TRIGGER images_batch_wafer_update BEFORE UPDATE OF batch_id, physical_site_id ON images WHEN NOT EXISTS ("
    " SELECT 1 FROM pattern_sites p JOIN dies d ON d.die_id = p.die_id"
    " JOIN batch_wafers bw ON bw.batch_id = NEW.batch_id AND bw.wafer_id = d.wafer_id"
    " WHERE p.physical_site_id = NEW.physical_site_id)"
    " BEGIN SELECT RAISE(ABORT, 'Image Wafer must be a member Wafer of its Batch'); END",
    "CREATE TRIGGER batch_wafers_keep_images BEFORE DELETE ON batch_wafers WHEN EXISTS ("
    " SELECT 1 FROM images i JOIN pattern_sites p ON p.physical_site_id = i.physical_site_id"
    " JOIN dies d ON d.die_id = p.die_id WHERE i.batch_id = OLD.batch_id AND d.wafer_id = OLD.wafer_id)"
    " BEGIN SELECT RAISE(ABORT, 'Batch Wafer membership has registered images'); END",
    "CREATE TRIGGER batch_wafers_immutable_keys BEFORE UPDATE OF batch_id, wafer_id ON batch_wafers"
    " BEGIN SELECT RAISE(ABORT, 'Batch Wafer membership keys are immutable'); END",
)
BACKFILL_SQL = ("INSERT INTO batch_wafers(batch_id,wafer_id,process_run_id,peb_time_s) "
                "SELECT batch_id,wafer_id,process_run_id,peb_temperature_c FROM upload_batches ORDER BY batch_id")

# Per-image effective process in v6: the image's OWN Wafer inside its Batch.
V6_IMAGE_PROCESS_SQL = (
    "SELECT i.image_id,i.batch_id,d.wafer_id,d.die_code,p.position_code,r.exposure_value,r.exposure_unit,"
    "bw.peb_time_s AS peb_time_s,CASE WHEN bw.process_run_id IS NULL THEN 'wafer' ELSE 'batch' END AS exposure_source "
    "FROM images i JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
    "JOIN dies d ON d.die_id=p.die_id "
    "JOIN batch_wafers bw ON bw.batch_id=i.batch_id AND bw.wafer_id=d.wafer_id "
    "JOIN wafers w ON w.wafer_id=d.wafer_id "
    "JOIN process_runs r ON r.process_run_id=COALESCE(bw.process_run_id,w.process_run_id) ")
# The same values through the v5 path (Batch-level process, Batch Wafer) for comparison.
V5_IMAGE_PROCESS_SQL = (
    "SELECT i.image_id,i.batch_id,d.wafer_id,d.die_code,p.position_code,r.exposure_value,r.exposure_unit,"
    "b.peb_temperature_c AS peb_time_s,CASE WHEN b.process_run_id IS NULL THEN 'wafer' ELSE 'batch' END AS exposure_source "
    "FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id "
    "JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id JOIN dies d ON d.die_id=p.die_id "
    "JOIN wafers w ON w.wafer_id=b.wafer_id "
    "JOIN process_runs r ON r.process_run_id=COALESCE(b.process_run_id,w.process_run_id) ")


def has_batch_wafers(connection):
    return connection.execute("PRAGMA user_version").fetchone()[0] >= 6


def image_process_sql(connection, *, image="i", die="d", batch="b"):
    """(join, exposure_source, peb) for the effective process of each IMAGE.

    The caller has already joined images AS {image}, dies AS {die} (via pattern_sites) and
    upload_batches AS {batch}; the returned join adds aliases w (Wafer) and r (process_runs).
    v6: the image's own Wafer inside its Batch. v5 and older: the Batch-level process.
    """
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version >= 6:
        join = (f"JOIN batch_wafers bw ON bw.batch_id={image}.batch_id AND bw.wafer_id={die}.wafer_id "
                f"JOIN wafers w ON w.wafer_id={die}.wafer_id "
                "JOIN process_runs r ON r.process_run_id=COALESCE(bw.process_run_id,w.process_run_id) ")
        return join, "CASE WHEN bw.process_run_id IS NULL THEN 'wafer' ELSE 'batch' END", "bw.peb_time_s"
    from database.migrate_v5 import effective_process_join
    join, source = effective_process_join(connection, batch_alias=batch)
    peb = f"{batch}.peb_temperature_c" if version >= 3 else "NULL"
    return f"JOIN wafers w ON w.wafer_id={batch}.wafer_id " + join, source, peb


def apply_v6(connection):
    for statement in V6_STATEMENTS:
        connection.execute(statement)


def reference_v6():
    with closing(reference_v5_connection()) as c:
        apply_v6(c)
        return _schema_objects(c)


def image_process_rows(connection, sql):
    return connection.execute(sql + "ORDER BY i.image_id").fetchall()


def validate_v6(connection):
    if connection.execute("PRAGMA user_version").fetchone()[0] != 6:
        raise DatabaseCompatibilityError("Schema v6 required; no automatic migration")
    if _schema_objects(connection) != reference_v6():
        raise DatabaseCompatibilityError("Schema v6 mismatch; DB preserved")
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise DatabaseCompatibilityError("SQLite integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise DatabaseCompatibilityError("Foreign key violation")
    checks = {
        "image Wafer outside its Batch":
            "SELECT 1 FROM images i JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
            "JOIN dies d ON d.die_id=p.die_id LEFT JOIN batch_wafers bw ON bw.batch_id=i.batch_id AND bw.wafer_id=d.wafer_id "
            "WHERE bw.batch_id IS NULL LIMIT 1",
        "Batch primary Wafer is not a member":
            "SELECT 1 FROM upload_batches b LEFT JOIN batch_wafers bw ON bw.batch_id=b.batch_id AND bw.wafer_id=b.wafer_id "
            "WHERE bw.batch_id IS NULL LIMIT 1",
        "single-Wafer Batch process differs from its membership":
            "SELECT 1 FROM upload_batches b JOIN batch_wafers bw ON bw.batch_id=b.batch_id "
            "WHERE (SELECT COUNT(*) FROM batch_wafers x WHERE x.batch_id=b.batch_id)=1 "
            "AND (bw.process_run_id IS NOT b.process_run_id OR bw.peb_time_s IS NOT b.peb_temperature_c) LIMIT 1",
        "multi-Wafer Batch keeps a Batch-level process":
            "SELECT 1 FROM upload_batches b WHERE (SELECT COUNT(*) FROM batch_wafers x WHERE x.batch_id=b.batch_id)>1 "
            "AND (b.process_run_id IS NOT NULL OR b.peb_temperature_c IS NOT NULL) LIMIT 1",
        "analysis image outside its run Batch":
            "SELECT 1 FROM analysis_image_runs a JOIN images i ON i.image_id=a.image_id "
            "JOIN analysis_batch_runs b ON b.batch_run_id=a.batch_run_id WHERE i.batch_id<>b.batch_id LIMIT 1",
    }
    for message, sql in checks.items():
        if connection.execute(sql).fetchone():
            raise DatabaseCompatibilityError("v6 membership violation: " + message)


def migrate_database_v6(path, *, allow_operational=False, _fail_after_apply=False):
    """v5 -> v6 in ONE transaction. Rerun on v6 only validates. Any failure rolls everything back."""
    path = refuse_operational(path) if allow_operational is not True else path
    with closing(connect_database(path)) as c:
        try:
            c.execute("BEGIN IMMEDIATE")
            version = c.execute("PRAGMA user_version").fetchone()[0]
            if version == 6:
                validate_v6(c)
                c.rollback()
                return {"version": 6, "migrated": False}
            validate_v5(c)
            before = data_snapshot(c)
            analysis_before = {t: c.execute(f'SELECT * FROM "{t}" ORDER BY 1').fetchall() for t in ANALYSIS_TABLES}
            process_before = image_process_rows(c, V5_IMAGE_PROCESS_SQL)
            objects = _schema_objects(c)
            apply_v6(c)
            c.execute(BACKFILL_SQL)
            if _fail_after_apply:
                raise DatabaseCompatibilityError("injected failure (rollback test)")
            c.execute("PRAGMA user_version=6")
            validate_v6(c)
            if data_snapshot(c) != before:
                raise DatabaseCompatibilityError("Existing registration data changed")
            for table, rows in analysis_before.items():
                if c.execute(f'SELECT * FROM "{table}" ORDER BY 1').fetchall() != rows:
                    raise DatabaseCompatibilityError("Analysis data changed: " + table)
            if image_process_rows(c, V6_IMAGE_PROCESS_SQL) != process_before:
                raise DatabaseCompatibilityError("Per-image effective process changed")
            current = _schema_objects(c)
            removed = {("trigger", name) for name in V5_SINGLE_WAFER_TRIGGERS}
            if any(current.get(k) != v for k, v in objects.items() if k not in removed):
                raise DatabaseCompatibilityError("Unrelated schema object changed")
            c.commit()
        except BaseException:
            if c.in_transaction:
                c.rollback()
            raise
    return {"version": 6, "migrated": True, "batch_wafers": len(before["upload_batches"])}
