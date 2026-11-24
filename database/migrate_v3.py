"""Explicit v2 -> v3: one nullable Batch PEB temperature, no result tables."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import sqlite3

from database.init_db import DEFAULT_DB, TABLE_NAMES, DatabaseCompatibilityError, connect_database, _schema_objects
from database.migrate_v2 import V2_SCHEMA, validate_v2, data_snapshot, initialize_database_v2

PEB_SQL = """ALTER TABLE upload_batches ADD COLUMN peb_temperature_c REAL
CHECK(peb_temperature_c IS NULL OR
(typeof(peb_temperature_c) IN ('real','integer') AND
peb_temperature_c BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308))"""


def reference_v3():
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(V2_SCHEMA.read_text(encoding="utf-8"))
        connection.execute(PEB_SQL)
        return _schema_objects(connection)


def validate_v3(connection):
    if connection.execute("PRAGMA user_version").fetchone()[0] != 3:
        raise DatabaseCompatibilityError("Schema v3 required; PEB migration is explicit")
    if _schema_objects(connection) != reference_v3():
        raise DatabaseCompatibilityError("Schema v3 mismatch; existing DB is preserved")
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise DatabaseCompatibilityError("SQLite integrity check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise DatabaseCompatibilityError("Foreign key violations")
    if connection.execute(
        "SELECT 1 FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id "
        "JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
        "JOIN dies d ON d.die_id=p.die_id WHERE b.wafer_id<>d.wafer_id LIMIT 1"
    ).fetchone():
        raise DatabaseCompatibilityError("Cross-wafer image reference")


def _backup(path, backup, snapshot):
    if path == backup:
        raise ValueError("Backup must differ from the operational DB")
    if backup.exists():
        with closing(connect_database(backup, readonly=True)) as existing:
            validate_v2(existing)
            if data_snapshot(existing) != snapshot:
                raise DatabaseCompatibilityError("Existing backup differs; choose a new path")
        return
    backup.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    with closing(connect_database(path, readonly=True)) as source, closing(sqlite3.connect(backup)) as target:
        source.backup(target)
    with closing(connect_database(backup, readonly=True)) as saved:
        validate_v2(saved)
        if data_snapshot(saved) != snapshot:
            raise DatabaseCompatibilityError("Backup verification failed; source was not changed")


def migrate_database_v3(path, *, backup_path=None, allow_operational=False):
    path = Path(path).expanduser().resolve()
    if path == DEFAULT_DB.resolve() and not allow_operational:
        raise PermissionError("Operational PEB migration requires explicit approval")
    with closing(connect_database(path, readonly=True)) as source:
        version = source.execute("PRAGMA user_version").fetchone()[0]
        if version == 3:
            validate_v3(source)
            return {"version": 3, "migrated": False, "backup": None}
        validate_v2(source)
    backup = Path(backup_path).expanduser().resolve() if backup_path else path.with_name(path.name + ".v2.backup.sqlite3")
    with closing(connect_database(path)) as connection:
        try:
            connection.execute("BEGIN IMMEDIATE")
            validate_v2(connection)
            before = data_snapshot(connection)
            fields = {table: [row[1] for row in connection.execute("PRAGMA table_info("+table+")")]
                      for table in TABLE_NAMES}
            _backup(path, backup, before)
            connection.execute(PEB_SQL)
            connection.execute("PRAGMA user_version=3")
            validate_v3(connection)
            after = {table: connection.execute(
                "SELECT "+",".join('"'+column+'"' for column in fields[table])+
                ' FROM "'+table+'" ORDER BY 1').fetchall() for table in TABLE_NAMES}
            if after != before or connection.execute(
                    "SELECT 1 FROM upload_batches WHERE peb_temperature_c IS NOT NULL").fetchone():
                raise DatabaseCompatibilityError("Existing metadata changed; migration is rolled back")
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
    return {"version": 3, "migrated": True, "backup": str(backup),
            "counts": {table: len(rows) for table, rows in before.items()}}


def initialize_database_v3(path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        with closing(connect_database(path, readonly=True)) as connection:
            validate_v3(connection)
        return path
    initialize_database_v2(path)
    migrate_database_v3(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--allow-operational", action="store_true")
    args = parser.parse_args()
    print(migrate_database_v3(args.db, backup_path=args.backup, allow_operational=args.allow_operational))
