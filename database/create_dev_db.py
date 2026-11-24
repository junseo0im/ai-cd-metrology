"""Create an EMPTY development SQLite DB at schema v6 for local dashboard runs.

Reuses the existing initializer and migrations in order: v3 initialize -> v4 -> v5 -> v6.
The output path is required, an existing file is never overwritten, and the operational
DB path (runtime_data/database/ida.sqlite3) is refused.

    python -B -m database.create_dev_db --db <path\\to\\dev.sqlite3>
"""
import argparse
from pathlib import Path
import sys

from database.migrate_v3 import initialize_database_v3
from database.migrate_v4 import migrate_database_v4, refuse_operational
from database.migrate_v5 import migrate_database_v5
from database.migrate_v6 import migrate_database_v6

SIDECARS = ("-journal", "-wal", "-shm")


def create_dev_db(path):
    path = Path(path).expanduser().resolve()
    refuse_operational(path)
    if path.exists() or any(Path(str(path) + suffix).exists() for suffix in SIDECARS):
        raise FileExistsError(f"이미 존재하는 DB(또는 보조 파일)는 덮어쓰지 않습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        initialize_database_v3(path)
        migrate_database_v4(path)
        migrate_database_v5(path)
        result = migrate_database_v6(path)
    except BaseException:
        # Only the file this call just created is removed; nothing pre-existing is touched.
        for candidate in (path, *(Path(str(path) + suffix) for suffix in SIDECARS)):
            candidate.unlink(missing_ok=True)
        raise
    return {"db": str(path), "version": 6, "migrated": result.get("migrated", True)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="빈 v6 개발용 SQLite DB 생성 (기존 파일은 덮어쓰지 않음)")
    parser.add_argument("--db", type=Path, required=True, help="새 DB 파일 경로 (존재하지 않아야 함)")
    args = parser.parse_args(argv)
    try:
        print(create_dev_db(args.db))
    except (FileExistsError, PermissionError) as exc:
        parser.exit(1, str(exc) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
