"""Schema v5: Batch-level process (exposure) reference. Additive, one nullable column.

upload_batches.process_run_id (NULL = inherit the Wafer's process record, i.e. every existing
Batch). Effective Batch process = COALESCE(batch.process_run_id, wafer.process_run_id).
Existing rows are never reclassified. The operational DB is refused unless the caller passes
allow_operational=True (requires explicit user approval; not used by the Dashboard).
"""
from contextlib import closing
import sqlite3

from database.init_db import DatabaseCompatibilityError, connect_database, _schema_objects
from database.migrate_v2 import data_snapshot
from database.migrate_v4 import ANALYSIS_TABLES, reference_v4, refuse_operational, validate_v4

BATCH_PROCESS_SQL = ("ALTER TABLE upload_batches ADD COLUMN process_run_id TEXT "
                     "REFERENCES process_runs(process_run_id) ON UPDATE RESTRICT ON DELETE RESTRICT")
ANALYSIS_VERSIONS = (4, 5, 6)  # Versions that hold the v4 analysis tables (v6: + batch_wafers).


def reference_v5():
    with closing(sqlite3.connect(":memory:")) as c:
        from database.migrate_v2 import V2_SCHEMA
        from database.migrate_v3 import PEB_SQL
        from database.migrate_v4 import V4_SCHEMA, _apply
        c.executescript(V2_SCHEMA.read_text(encoding="utf-8"))
        c.execute(PEB_SQL)
        _apply(c)
        c.execute(BATCH_PROCESS_SQL)
        return _schema_objects(c)


def validate_v5(connection):
    if connection.execute("PRAGMA user_version").fetchone()[0] != 5:
        raise DatabaseCompatibilityError("Schema v5 required; no automatic migration")
    if _schema_objects(connection) != reference_v5():
        raise DatabaseCompatibilityError("Schema v5 mismatch; DB preserved")
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise DatabaseCompatibilityError("SQLite integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise DatabaseCompatibilityError("Foreign key violation")
    for sql in (
        "SELECT 1 FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id JOIN dies d ON d.die_id=p.die_id WHERE b.wafer_id<>d.wafer_id LIMIT 1",
        "SELECT 1 FROM analysis_image_runs a JOIN images i ON i.image_id=a.image_id JOIN analysis_batch_runs b ON b.batch_run_id=a.batch_run_id WHERE i.batch_id<>b.batch_id LIMIT 1"):
        if connection.execute(sql).fetchone():
            raise DatabaseCompatibilityError("Batch/Wafer membership violation")


def validate_analysis_schema(connection):
    """Validate a DB holding the analysis tables (v4, v5 or v6); anything else is refused."""
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == 4:
        validate_v4(connection)
    elif version == 5:
        validate_v5(connection)
    elif version == 6:
        from database.migrate_v6 import validate_v6  # Lazy: migrate_v6 imports this module.
        validate_v6(connection)
    else:
        raise DatabaseCompatibilityError("Schema v4, v5 or v6 required; no automatic migration")
    return version


def has_batch_process(connection):
    return connection.execute("PRAGMA user_version").fetchone()[0] >= 5


def effective_process_join(connection, batch_alias="b", wafer_alias="w", process_alias="r"):
    """SQL join/select for the effective Batch process (v5: Batch first, else Wafer)."""
    if has_batch_process(connection):
        join = (f"JOIN process_runs {process_alias} ON {process_alias}.process_run_id="
                f"COALESCE({batch_alias}.process_run_id,{wafer_alias}.process_run_id) ")
        source = f"CASE WHEN {batch_alias}.process_run_id IS NULL THEN 'wafer' ELSE 'batch' END"
    else:
        join = f"JOIN process_runs {process_alias} ON {process_alias}.process_run_id={wafer_alias}.process_run_id "
        source = "'wafer'"
    return join, source


def migrate_database_v5(path, *, allow_operational=False):
    """v4 -> v5 in one transaction; every pre-existing value is verified unchanged."""
    path = refuse_operational(path) if allow_operational is not True else path
    with closing(connect_database(path)) as c:
        try:
            c.execute("BEGIN IMMEDIATE")
            version = c.execute("PRAGMA user_version").fetchone()[0]
            if version == 5:
                validate_v5(c)
                c.rollback()
                return {"version": 5, "migrated": False}
            validate_v4(c)
            before = data_snapshot(c)
            analysis_before = {t: c.execute(f'SELECT * FROM "{t}" ORDER BY 1').fetchall() for t in ANALYSIS_TABLES}
            objects = _schema_objects(c)
            batch_columns = [r[1] for r in c.execute("PRAGMA table_info(upload_batches)")]
            c.execute(BATCH_PROCESS_SQL)
            c.execute("PRAGMA user_version=5")
            validate_v5(c)
            after = data_snapshot(c)
            for table, rows in before.items():
                if table == "upload_batches":
                    projected = [row[:len(batch_columns)] for row in after[table]]
                    if projected != rows or any(row[len(batch_columns)] is not None for row in after[table]):
                        raise DatabaseCompatibilityError("Existing Batch rows changed or were reclassified")
                elif after[table] != rows:
                    raise DatabaseCompatibilityError("Existing data changed: " + table)
            for table, rows in analysis_before.items():
                if c.execute(f'SELECT * FROM "{table}" ORDER BY 1').fetchall() != rows:
                    raise DatabaseCompatibilityError("Analysis data changed: " + table)
            current = _schema_objects(c)
            # Only the upload_batches DDL (one added column) may differ; every index/trigger is unchanged.
            if any(current.get(k) != v for k, v in objects.items() if k != ("table", "upload_batches")):
                raise DatabaseCompatibilityError("Unrelated schema object changed")
            c.commit()
        except BaseException:
            if c.in_transaction:
                c.rollback()
            raise
    return {"version": 5, "migrated": True, "batches_inheriting_wafer_process": len(before["upload_batches"]),
            "analysis_tables": list(ANALYSIS_TABLES)}
