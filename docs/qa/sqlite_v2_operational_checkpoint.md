# SQLite v2 operational checkpoint

Date: 2026-11-15 (KST)
Branch: feature/streamlit-dashboard

## Applied migration

The separately authorized operational migration used the existing database.migrate_v2.
Streamlit/SQLite tool process checks, port 8502 listener checks, and an exclusive
read-open check passed before migration. A fresh SQLite Backup API snapshot was
created and verified before applying the migration.

- Before: Schema v1, 14 Batches, 34 images.
- After: Schema v2, 14 Batches, 34 images.
- All seven tables exactly match the pre-migration backup, including IDs, notes,
  user Batch names, storage paths and SHA-256 fields.
- integrity_check: ok; foreign_key_check: zero violations.
- All 34 original files exist and match their recorded SHA-256.
- One existing user-named Batch was preserved.
- No synthetic image registration or name save was performed against the operational DB.

The fresh backup is local-only:
runtime_data/database/ida.pre_v2_20261115T085630Z.sqlite3

## Minimum runtime verification

Actual installed Google Chrome was exercised with Playwright in headless mode at
http://127.0.0.1:8502. Scope: Data Control Center and one existing Batch detail.

| Check | Result |
|---|---|
| SQLite Data Control renders | PASS |
| Wafer Batch / Repeatability / Lighting / Single Image selectable | PASS |
| Schema v2 / Single Image support displayed without the v1 restriction | PASS |
| Registration and reset button labels; empty upload remains disabled | PASS |
| Optional Batch name input | PASS |
| Existing 34 images query | PASS |
| Registered original image display | PASS |
| All 14 Batch rows; no Batch pagination | PASS |
| Existing Batch detail and name edit controls (no save) | PASS |
| Return from Batch detail | PASS |

Actual Single Image registration and Batch name persistence/edit/clear were
verified only in isolated temporary QA DBs. Eight focused regression tests passed.
The existing 168-test regression result was reused; this checkpoint did not rerun
all eighteen browser pages or implement any new UI/analysis feature.

## Local evidence and backup scope

Detailed records, the browser verification script and screenshots are local-only:
runtime_data/migration_apply/20261115T085630Z/

The earlier Claude safety snapshot preserved code, Git history, the operational
DB and original images. Its overall status is PARTIAL because derived images
were recorded by hash rather than fully copied. It is not described as a full
dataset backup. The fresh pre-v2 DB backup above is the restoration source for
this migration; older TASK A snapshots predate later registrations.

## Sharing and remaining limits

Only source, tests, schema/migration, configuration, Markdown and bundled Mock SVG
assets are shared. Operational DBs, research images, generated screenshots, model
weights, datasets and local credentials are excluded from Git.
Historical browser audit screenshots remain local; the earlier audit is a dated
record and is not relabelled as a current PASS.
Cold direct-URL startup behaviour and all other pages were outside this minimum
runtime check. Actual Calibration/metrology/YOLO/MLflow remain disconnected.
No R02 source restoration, main merge, or algorithm changes were performed.
