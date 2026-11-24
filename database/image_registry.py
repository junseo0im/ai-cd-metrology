"""DB-1B prototype: explicit existing references, PNG-header inspection, no uploads."""
import hashlib
from pathlib import Path
import sqlite3
import struct
import zlib

from database.filename_parser import ParsedFilename, parse_filename, validate_metadata
from database.init_db import DEFAULT_DB, SCHEMA_VERSION


class ImageRegistrationError(RuntimeError):
    pass


class PngHeaderError(ImageRegistrationError):
    pass


def _inspect_png(path: Path) -> tuple[int, int, str]:
    """Read IHDR geometry/CRC and hash original bytes; never decode or alter pixels."""
    try:
        with path.open("rb") as source:
            header = source.read(33)
            if (len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n"
                    or header[8:12] != struct.pack(">I", 13) or header[12:16] != b"IHDR"):
                raise PngHeaderError("Invalid or truncated PNG signature/IHDR header")
            if zlib.crc32(header[12:29]) & 0xffffffff != struct.unpack(">I", header[29:33])[0]:
                raise PngHeaderError("Invalid PNG IHDR checksum")
            width, height = struct.unpack(">II", header[16:24])
            if width <= 0 or height <= 0:
                raise PngHeaderError("PNG width and height must be positive")
            digest = hashlib.sha256(header)
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
            return width, height, digest.hexdigest()
    except OSError as exc:
        raise ImageRegistrationError(f"Cannot read PNG: {path}") from exc


def _check_connection(connection: sqlite3.Connection, *, allow_transaction: bool = False,
                      allow_operational: bool = False) -> None:
    """DB-1B/C default guard; DB-1D requires explicit operational opt-in."""
    main_path = next(row[2] for row in connection.execute("PRAGMA database_list") if row[1] == "main")
    if not allow_operational and main_path and Path(main_path).resolve() == DEFAULT_DB.resolve():
        raise ImageRegistrationError("DB-1B registration against the operational DB is forbidden")
    if connection.in_transaction and not allow_transaction:
        raise ImageRegistrationError("Commit fixture/caller work before registering an image")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise ImageRegistrationError("foreign_keys must be ON; use connect_database")
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version not in (SCHEMA_VERSION, 2, 3, 4, 5, 6):
        raise ImageRegistrationError("An initialized schema version 1, 2, 3, 4, 5 or 6 is required")
    if version in (4, 5, 6):
        from database.migrate_v5 import validate_analysis_schema
        validate_analysis_schema(connection)


def get_image(connection: sqlite3.Connection, image_id: str) -> dict | None:
    """Read stored image fields plus Wafer/Die/Position identity; no inferred labels.

    wafer_id is the image's OWN Wafer (its site's Die). In v1-v5 it always equals the Batch Wafer
    (trigger-enforced); in v6 a Batch may hold several Wafers, so the Batch Wafer is never used.
    """
    cursor = connection.execute(
        "SELECT i.*, d.wafer_id, d.die_id, d.die_code, d.row_index, d.column_index, p.position_code "
        "FROM images i JOIN upload_batches b ON b.batch_id=i.batch_id "
        "JOIN pattern_sites p ON p.physical_site_id=i.physical_site_id "
        "JOIN dies d ON d.die_id=p.die_id WHERE i.image_id=?", (image_id,),
    )
    row = cursor.fetchone()
    return dict(zip((column[0] for column in cursor.description), row)) if row is not None else None


def register_image(
    connection: sqlite3.Connection, image_path: str | Path, *,
    image_id: str, batch_id: str, physical_site_id: str, metadata_status: str,
    condition_id: str | None = None, captured_at: str | None = None,
    repeat_index: int | None = None,
    parsed_metadata: ParsedFilename | None = None, manage_transaction: bool = True,
    allow_operational: bool = False,
) -> dict:
    """Register one temporary PNG, owning one transaction on an idle connection.

    Parent rows must already exist and be committed. Filename identity must agree
    with the selected site and Batch. No ID/parent/condition is auto-created.
    DB-1C may supply validated manual identity and manage_transaction=False
    inside its own active transaction. That caller owns commit/rollback.
    Same-hash rows remain separate: duplicate_sha256/duplicate_image_ids are
    return-only diagnostics. Files are read in place; no copy, mutation or removal.
    """
    _check_connection(connection, allow_transaction=not manage_transaction,
                      allow_operational=allow_operational)
    if not manage_transaction and not connection.in_transaction:
        raise ImageRegistrationError("Caller-managed registration requires an active transaction")
    requested = Path(image_path).expanduser()
    if requested.suffix.lower() != ".png":
        raise ImageRegistrationError("Only PNG files are supported")
    if parsed_metadata is None:
        parsed = parse_filename(requested.name)
    else:
        parsed = validate_metadata(
            wafer_id=parsed_metadata.wafer_id, die_code=parsed_metadata.die_code,
            position_code=parsed_metadata.position_code,
            legacy_defect_code=parsed_metadata.legacy_defect_code)
        if parsed != parsed_metadata:
            raise ImageRegistrationError("Manual metadata row/column do not match the Die code")
    try:
        path = requested.resolve(strict=True)
    except OSError as exc:
        raise ImageRegistrationError(f"PNG path is unavailable: {requested}") from exc
    width, height, sha256 = _inspect_png(path)
    try:
        if manage_transaction:
            connection.execute("BEGIN IMMEDIATE")
        batch = connection.execute(
            "SELECT wafer_id FROM upload_batches WHERE batch_id=?", (batch_id,),
        ).fetchone()
        if batch is None:
            raise ImageRegistrationError(f"Unknown batch_id: {batch_id}")
        site = connection.execute(
            "SELECT d.wafer_id,d.die_code,d.row_index,d.column_index,p.position_code "
            "FROM pattern_sites p JOIN dies d ON d.die_id=p.die_id WHERE p.physical_site_id=?",
            (physical_site_id,),
        ).fetchone()
        if site is None:
            raise ImageRegistrationError(f"Unknown physical_site_id: {physical_site_id}")
        if connection.execute("PRAGMA user_version").fetchone()[0] >= 6:
            if connection.execute("SELECT 1 FROM batch_wafers WHERE batch_id=? AND wafer_id=?",
                                  (batch_id, site[0])).fetchone() is None:
                raise ImageRegistrationError("Image Wafer must be a member Wafer of its Batch")
        elif batch[0] != site[0]:
            raise ImageRegistrationError("Batch and site must belong to the same wafer")
        expected = (parsed.wafer_id, parsed.die_code, parsed.row_index,
                    parsed.column_index, parsed.position_code)
        if tuple(site) != expected:
            raise ImageRegistrationError("Filename Wafer/Die/Position does not match the selected site")
        if condition_id is not None and connection.execute(
            "SELECT 1 FROM acquisition_conditions WHERE condition_id=?", (condition_id,),
        ).fetchone() is None:
            raise ImageRegistrationError(f"Unknown condition_id: {condition_id}")
        duplicates = tuple(row[0] for row in connection.execute(
            "SELECT image_id FROM images WHERE lower(sha256)=? ORDER BY image_id", (sha256,),
        ))
        connection.execute(
            "INSERT INTO images(image_id,batch_id,physical_site_id,condition_id,original_filename,"
            "storage_path,sha256,width_px,height_px,captured_at,repeat_index,legacy_defect_code,metadata_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (image_id,batch_id,physical_site_id,condition_id,requested.name,str(path),sha256,
             width,height,captured_at,repeat_index,parsed.legacy_defect_code,metadata_status),
        )
        stored = get_image(connection, image_id)
        if stored is None:
            raise ImageRegistrationError("Registration readback did not find the inserted image")
        result = {**stored, "duplicate_sha256":bool(duplicates), "duplicate_image_ids":duplicates}
        if manage_transaction:
            connection.commit()
        return result
    except sqlite3.Error as exc:
        if manage_transaction and connection.in_transaction:
            connection.rollback()
        raise ImageRegistrationError(f"Image registration failed: {exc}") from exc
    except Exception:
        if manage_transaction and connection.in_transaction:
            connection.rollback()
        raise
