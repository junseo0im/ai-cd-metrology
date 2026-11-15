"""Explicit v1 -> v2 migration. Never called automatically by the dashboard."""
import argparse
from contextlib import closing
import os
from pathlib import Path
import sqlite3

from database.init_db import (
    DEFAULT_DB, TABLE_NAMES, DatabaseCompatibilityError, connect_database,
    initialize_database, _schema_objects,
)

V2_SCHEMA = Path(__file__).with_name("schema_v2.sql")


def reference_v2():
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(V2_SCHEMA.read_text(encoding="utf-8"))
        return _schema_objects(connection)


def validate_v2(connection):
    if connection.execute("PRAGMA user_version").fetchone()[0] != 2:
        raise DatabaseCompatibilityError("Schema v2 required; migration is explicit")
    if _schema_objects(connection) != reference_v2():
        raise DatabaseCompatibilityError("Schema v2 mismatch; DB is preserved")
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


def data_snapshot(connection):
    """All seven tables, including PK/FK/hash/paths; no image files are opened."""
    return {name: connection.execute('SELECT * FROM "'+name+'" ORDER BY 1').fetchall()
            for name in TABLE_NAMES}


def _backup_locked(path, backup, snapshot):
    if backup.resolve() == path.resolve():
        raise ValueError("Backup path must differ from the source DB")
    if backup.exists():
        initialize_database(backup)  # Must be a compatible v1 backup.
        with closing(connect_database(backup, readonly=True)) as existing:
            if data_snapshot(existing) != snapshot:
                raise DatabaseCompatibilityError("Existing backup differs; choose a new backup path")
        return
    backup.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    with closing(connect_database(path, readonly=True)) as source, closing(sqlite3.connect(backup)) as target:
        source.backup(target)


def _rebuild_batches(connection, expected):
    # Rebuild only the table whose CHECK changes; preserve external image FKs.
    for (kind,name), (_,sql) in expected.items():
        if kind == "trigger":
            connection.execute('DROP TRIGGER "'+name+'"')
    create = expected[("table","upload_batches")][1]
    connection.execute(create.replace('CREATE TABLE "upload_batches"', "CREATE TABLE upload_batches_v2", 1))
    columns = [row[1] for row in connection.execute("PRAGMA table_info(upload_batches)")]
    fields = ",".join('"'+name+'"' for name in columns)
    connection.execute(f"INSERT INTO upload_batches_v2({fields}) SELECT {fields} FROM upload_batches")
    connection.execute("DROP TABLE upload_batches")
    connection.execute("ALTER TABLE upload_batches_v2 RENAME TO upload_batches")
    for (kind,_), (table,sql) in expected.items():
        if kind == "trigger" or (kind == "index" and table == "upload_batches"):
            connection.execute(sql)


def migrate_database_v2(path, *, backup_path=None, allow_operational=False):
    path = Path(path).expanduser().resolve()
    if path == DEFAULT_DB.resolve() and not allow_operational:
        raise PermissionError("Operational migration requires separate approval and --allow-operational")
    with closing(connect_database(path, readonly=True)) as existing:
        version = existing.execute("PRAGMA user_version").fetchone()[0]
        if version == 2:
            validate_v2(existing)
            return {"version":2,"migrated":False,"backup":None}
        if version != 1:
            raise DatabaseCompatibilityError("Only v1 -> v2 migration is supported")
    initialize_database(path)  # Frozen v1 validator, read-only on existing DB.
    backup = Path(backup_path) if backup_path else path.with_name(path.name+".v1.backup.sqlite3")
    backup = backup.expanduser().resolve()
    expected = reference_v2()
    with closing(connect_database(path)) as connection:
        try:
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("BEGIN IMMEDIATE")
            # A second process must not migrate between the initial validation
            # and this write lock.
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise DatabaseCompatibilityError("Source version changed; retry explicitly")
            from database.init_db import _reference_schema, _validate, SCHEMA_FILE
            _validate(connection, _reference_schema(SCHEMA_FILE.read_text(encoding="utf-8")))
            before = data_snapshot(connection)
            _backup_locked(path, backup, before)
            _rebuild_batches(connection, expected)
            connection.execute("PRAGMA user_version=2")
            validate_v2(connection)
            if data_snapshot(connection) != before:
                raise DatabaseCompatibilityError("Migration data preservation check failed")
            connection.commit()
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.execute("PRAGMA foreign_keys=ON")
    return {"version":2,"migrated":True,"backup":str(backup)}


def initialize_database_v2(path):
    """Create a NEW v2 DB. Existing v1 DBs are never silently migrated."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        with closing(connect_database(path, readonly=True)) as connection:
            validate_v2(connection)
        return path
    if path == DEFAULT_DB.resolve():
        raise PermissionError("Use the explicit approved operational migration procedure")
    initialize_database(path)
    migrate_database_v2(path)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",type=Path,required=True)
    parser.add_argument("--backup",type=Path)
    parser.add_argument("--create",action="store_true",help="Create a new v2 DB only")
    parser.add_argument("--allow-operational",action="store_true")
    args=parser.parse_args()
    if args.create:
        print(initialize_database_v2(args.db))
    else:
        print(migrate_database_v2(args.db,backup_path=args.backup,allow_operational=args.allow_operational))


if __name__=="__main__":
    main()
