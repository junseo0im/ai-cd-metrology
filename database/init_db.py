"""Create DB-1A once, or validate an existing DB without changing it."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sqlite3

SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_FILE = Path(__file__).with_name("schema.sql")
DEFAULT_DB = PROJECT_ROOT / "runtime_data" / "database" / "ida.sqlite3"
TABLE_NAMES = (
    "process_runs", "wafers", "dies", "pattern_sites",
    "upload_batches", "acquisition_conditions", "images",
)


class DatabaseCompatibilityError(RuntimeError):
    """Existing schema/version/data does not match DB-1A; no repair is attempted."""


def connect_database(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    """Open an EXISTING DB with FK enforcement on this connection.

    SQLite FK enforcement is per connection. Other clients must also enable it.
    mode=rw/ro deliberately prevents accidental creation at a wrong path.
    """
    target = Path(path).expanduser().resolve()
    mode = "ro" if readonly else "rw"
    connection = sqlite3.connect(target.as_uri() + f"?mode={mode}", uri=True)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise RuntimeError("Could not enable SQLite foreign keys")
        return connection
    except Exception:
        connection.close()
        raise


def _schema_objects(connection: sqlite3.Connection) -> dict:
    rows = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE name NOT GLOB 'sqlite_*' AND type IN ('table','index','trigger','view') "
        "ORDER BY type, name"
    ).fetchall()
    return {(kind, name): (table, re.sub(r"\s+", " ", sql.strip()) if sql else None)
            for kind, name, table, sql in rows}


def _reference_schema(sql: str) -> dict:
    reference = sqlite3.connect(":memory:")
    try:
        reference.execute("PRAGMA foreign_keys = ON")
        reference.executescript(sql)
        if reference.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise RuntimeError("schema.sql and SCHEMA_VERSION disagree")
        objects = _schema_objects(reference)
        if {name for kind, name in objects if kind == "table"} != set(TABLE_NAMES):
            raise RuntimeError("schema.sql must define exactly the seven DB-1A tables")
        return objects
    finally:
        reference.close()


def _validate(connection: sqlite3.Connection, expected: dict) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        raise DatabaseCompatibilityError(
            f"Unsupported schema version {version}; expected {SCHEMA_VERSION}. "
            "Existing DB is preserved; automatic migration is not supported."
        )
    actual = _schema_objects(connection)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(k for k in set(actual) & set(expected) if actual[k] != expected[k])
        raise DatabaseCompatibilityError(
            f"Schema mismatch: missing={missing}, extra={extra}, changed={changed}. "
            "Existing DB is preserved."
        )
    if connection.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise DatabaseCompatibilityError("SQLite integrity check failed; existing DB is preserved")
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise DatabaseCompatibilityError(f"Foreign key violations: {violations!r}")
    mismatch = connection.execute(
        "SELECT i.image_id FROM images i "
        "JOIN upload_batches b ON b.batch_id=i.batch_id "
        "JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
        "JOIN dies d ON d.die_id=p.die_id WHERE b.wafer_id<>d.wafer_id LIMIT 1"
    ).fetchone()
    if mismatch:
        raise DatabaseCompatibilityError(f"Cross-wafer image reference: {mismatch[0]}")


def initialize_database(path: str | Path = DEFAULT_DB) -> Path:
    """Create a new empty DB, or validate an existing one in read-only mode.

    No destructive SQL or example data is applied. Existing files are never
    recreated. If new-file initialization fails, the file is left for review.
    """
    target = Path(path).expanduser().resolve()
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    expected = _reference_schema(sql)  # Validate SQL before touching the target.
    if target.exists():
        with_existing = connect_database(target, readonly=True)
        try:
            _validate(with_existing, expected)
        finally:
            with_existing.close()
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a DB created by another process.
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    connection = connect_database(target)
    try:
        connection.executescript("BEGIN IMMEDIATE;\n" + sql + "\nCOMMIT;")
        _validate(connection, expected)
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite DB path")
    args = parser.parse_args(argv)
    existed = args.db.expanduser().resolve().exists()
    try:
        target = initialize_database(args.db)
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    action = "Validated existing DB (read-only)" if existed else "Created empty DB"
    print(f"{action}: {target}")
    print(f"Schema version: {SCHEMA_VERSION}; tables: {len(TABLE_NAMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
