"""Add v4 analysis tables to isolated v3 DBs; operational DB is always refused."""
from contextlib import closing
from pathlib import Path
import sqlite3
from database.init_db import DEFAULT_DB, TABLE_NAMES, DatabaseCompatibilityError, connect_database, _schema_objects
from database.migrate_v2 import V2_SCHEMA, data_snapshot
from database.migrate_v3 import PEB_SQL, validate_v3

V4_SCHEMA = Path(__file__).with_name("schema_v4.sql")
ANALYSIS_TABLES = ("analysis_batch_runs","analysis_image_runs","defect_detections",
                   "metrology_measurements","line_judgments")


def refuse_operational(path):
    path=Path(path).expanduser().resolve()
    operational=DEFAULT_DB.resolve()
    if path==operational or (path.exists() and operational.exists() and path.samefile(operational)):
        raise PermissionError("Operational DB writes/migration are forbidden in TASK17B")
    return path


def reference_v4():
    with closing(sqlite3.connect(":memory:")) as c:
        c.executescript(V2_SCHEMA.read_text(encoding="utf-8"))
        c.execute(PEB_SQL)
        c.executescript(V4_SCHEMA.read_text(encoding="utf-8"))
        return _schema_objects(c)


def validate_v4(connection):
    if connection.execute("PRAGMA user_version").fetchone()[0]!=4:
        raise DatabaseCompatibilityError("Schema v4 required; no automatic migration")
    if _schema_objects(connection)!=reference_v4():
        raise DatabaseCompatibilityError("Schema v4 mismatch; DB preserved")
    if connection.execute("PRAGMA quick_check").fetchall()!=[("ok",)]:
        raise DatabaseCompatibilityError("SQLite integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise DatabaseCompatibilityError("Foreign key violation")
    for sql in (
        "SELECT 1 FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id JOIN dies d ON d.die_id=p.die_id WHERE b.wafer_id<>d.wafer_id LIMIT 1",
        "SELECT 1 FROM analysis_image_runs a JOIN images i ON i.image_id=a.image_id JOIN analysis_batch_runs b ON b.batch_run_id=a.batch_run_id WHERE i.batch_id<>b.batch_id LIMIT 1"):
        if connection.execute(sql).fetchone():
            raise DatabaseCompatibilityError("Batch/Wafer membership violation")


def _apply(c):
    statement=""
    for line in V4_SCHEMA.read_text(encoding="utf-8").splitlines(True):
        statement+=line
        if sqlite3.complete_statement(statement):
            c.execute(statement)
            statement=""
    if statement.strip():
        raise DatabaseCompatibilityError("Incomplete v4 SQL")


def migrate_database_v4(path):
    path=refuse_operational(path)
    with closing(connect_database(path)) as c:
        try:
            c.execute("BEGIN IMMEDIATE")
            version=c.execute("PRAGMA user_version").fetchone()[0]
            if version==4:
                validate_v4(c)
                c.rollback()
                return {"version":4,"migrated":False}
            validate_v3(c)  # Strictly v3, including all existing objects/data integrity.
            before=data_snapshot(c)
            objects=_schema_objects(c)
            _apply(c)       # executescript is deliberately avoided inside this transaction.
            c.execute("PRAGMA user_version=4")
            validate_v4(c)
            if data_snapshot(c)!=before:
                raise DatabaseCompatibilityError("Original seven-table data changed")
            current=_schema_objects(c)
            if any(current.get(k)!=v for k,v in objects.items()):
                raise DatabaseCompatibilityError("Original schema/index/trigger changed")
            c.commit()
        except BaseException:
            if c.in_transaction:c.rollback()
            raise
    return {"version":4,"migrated":True,"original_counts":{k:len(v) for k,v in before.items()}}
