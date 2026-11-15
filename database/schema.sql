-- IDA DB-1A, schema version 1. Apply only to a new DB via init_db.py.
-- No experiment data or grid is inserted by this schema.
PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

CREATE TABLE process_runs (
    process_run_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(process_run_id)) > 0),
    exposure_value REAL NOT NULL CHECK(typeof(exposure_value) IN ('real', 'integer')),
    exposure_unit TEXT,
    recipe_version TEXT,
    processed_at TEXT CHECK(processed_at IS NULL OR (length(processed_at) = 20
        AND processed_at GLOB '????-??-??T??:??:??Z'
        AND strftime('%Y-%m-%dT%H:%M:%SZ', processed_at, '+0 seconds') IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%SZ', processed_at, '+0 seconds') = processed_at)),
    notes TEXT
);

CREATE TABLE wafers (
    wafer_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(wafer_id)) > 0),
    process_run_id TEXT NOT NULL REFERENCES process_runs(process_run_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    lot_id TEXT,
    wafer_label TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        CHECK(length(created_at) = 20
        AND created_at GLOB '????-??-??T??:??:??Z'
        AND strftime('%Y-%m-%dT%H:%M:%SZ', created_at, '+0 seconds') IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%SZ', created_at, '+0 seconds') = created_at),
    notes TEXT
);

CREATE TABLE dies (
    die_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(die_id)) > 0),
    wafer_id TEXT NOT NULL REFERENCES wafers(wafer_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    die_code TEXT NOT NULL CHECK(die_code = CAST(row_index AS TEXT) || CAST(column_index AS TEXT)),
    row_index INTEGER NOT NULL CHECK(typeof(row_index) = 'integer' AND row_index BETWEEN 1 AND 3),
    column_index INTEGER NOT NULL CHECK(typeof(column_index) = 'integer' AND column_index BETWEEN 1 AND 8),
    UNIQUE(wafer_id, die_code),
    UNIQUE(wafer_id, row_index, column_index)
);

CREATE TABLE pattern_sites (
    physical_site_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(physical_site_id)) > 0),
    die_id TEXT NOT NULL REFERENCES dies(die_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    position_code TEXT NOT NULL CHECK(position_code IN ('lt', 'c', 'rb')),
    notes TEXT,
    UNIQUE(die_id, position_code)
);

CREATE TABLE upload_batches (
    batch_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(batch_id)) > 0),
    wafer_id TEXT NOT NULL REFERENCES wafers(wafer_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    upload_mode TEXT NOT NULL CHECK(upload_mode IN ('wafer_batch', 'repeatability', 'lighting')),
    dataset_role TEXT NOT NULL DEFAULT 'evaluation'
        CHECK(dataset_role IN ('evaluation', 'training_candidate', 'development')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        CHECK(length(created_at) = 20
        AND created_at GLOB '????-??-??T??:??:??Z'
        AND strftime('%Y-%m-%dT%H:%M:%SZ', created_at, '+0 seconds') IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%SZ', created_at, '+0 seconds') = created_at),
    batch_status TEXT NOT NULL CHECK(length(trim(batch_status)) > 0),
    notes TEXT
);

CREATE TABLE acquisition_conditions (
    condition_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(condition_id)) > 0),
    magnification REAL NOT NULL CHECK(typeof(magnification) IN ('real', 'integer') AND magnification > 0),
    illumination_setting TEXT,
    auto_exposure INTEGER CHECK(auto_exposure IS NULL OR
        (typeof(auto_exposure) = 'integer' AND auto_exposure IN (0, 1))),
    camera_setting_json TEXT CHECK(camera_setting_json IS NULL OR json_valid(camera_setting_json)),
    notes TEXT
);

CREATE TABLE images (
    image_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(image_id)) > 0),
    batch_id TEXT NOT NULL REFERENCES upload_batches(batch_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    physical_site_id TEXT NOT NULL REFERENCES pattern_sites(physical_site_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    condition_id TEXT REFERENCES acquisition_conditions(condition_id) ON UPDATE RESTRICT ON DELETE RESTRICT,
    original_filename TEXT NOT NULL CHECK(length(trim(original_filename)) > 0),
    storage_path TEXT NOT NULL CHECK(length(trim(storage_path)) > 0),
    sha256 TEXT NOT NULL CHECK(length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-fA-F]*'),
    width_px INTEGER NOT NULL CHECK(typeof(width_px) = 'integer' AND width_px > 0),
    height_px INTEGER NOT NULL CHECK(typeof(height_px) = 'integer' AND height_px > 0),
    captured_at TEXT CHECK(captured_at IS NULL OR (length(captured_at) = 20
        AND captured_at GLOB '????-??-??T??:??:??Z'
        AND strftime('%Y-%m-%dT%H:%M:%SZ', captured_at, '+0 seconds') IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%SZ', captured_at, '+0 seconds') = captured_at)),
    repeat_index INTEGER CHECK(repeat_index IS NULL OR typeof(repeat_index) = 'integer'),
    legacy_defect_code TEXT,
    metadata_status TEXT NOT NULL CHECK(length(trim(metadata_status)) > 0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        CHECK(length(created_at) = 20
        AND created_at GLOB '????-??-??T??:??:??Z'
        AND strftime('%Y-%m-%dT%H:%M:%SZ', created_at, '+0 seconds') IS NOT NULL
        AND strftime('%Y-%m-%dT%H:%M:%SZ', created_at, '+0 seconds') = created_at)
);

CREATE INDEX idx_wafers_process_run ON wafers(process_run_id);
CREATE INDEX idx_batches_wafer ON upload_batches(wafer_id);
CREATE INDEX idx_images_batch ON images(batch_id);
CREATE INDEX idx_images_site ON images(physical_site_id);
CREATE INDEX idx_images_condition ON images(condition_id);
-- Intentionally non-unique: duplicates are retained, never merged or removed.
CREATE INDEX idx_images_sha256 ON images(sha256);

-- Cross-table wafer consistency without duplicating wafer_id on images.
CREATE TRIGGER images_same_wafer_insert
BEFORE INSERT ON images
WHEN EXISTS (
    SELECT 1 FROM upload_batches b
    JOIN pattern_sites p ON p.physical_site_id = NEW.physical_site_id
    JOIN dies d ON d.die_id = p.die_id
    WHERE b.batch_id = NEW.batch_id AND b.wafer_id <> d.wafer_id
)
BEGIN
    SELECT RAISE(ABORT, 'Batch and site must belong to the same wafer');
END;

CREATE TRIGGER images_same_wafer_update
BEFORE UPDATE OF batch_id, physical_site_id ON images
WHEN EXISTS (
    SELECT 1 FROM upload_batches b
    JOIN pattern_sites p ON p.physical_site_id = NEW.physical_site_id
    JOIN dies d ON d.die_id = p.die_id
    WHERE b.batch_id = NEW.batch_id AND b.wafer_id <> d.wafer_id
)
BEGIN
    SELECT RAISE(ABORT, 'Batch and site must belong to the same wafer');
END;

CREATE TRIGGER batches_preserve_image_wafer
BEFORE UPDATE OF wafer_id ON upload_batches
WHEN EXISTS (
    SELECT 1 FROM images i
    JOIN pattern_sites p ON p.physical_site_id = i.physical_site_id
    JOIN dies d ON d.die_id = p.die_id
    WHERE i.batch_id = OLD.batch_id AND d.wafer_id <> NEW.wafer_id
)
BEGIN
    SELECT RAISE(ABORT, 'Batch and site must belong to the same wafer');
END;

CREATE TRIGGER dies_preserve_image_wafer
BEFORE UPDATE OF wafer_id ON dies
WHEN EXISTS (
    SELECT 1 FROM pattern_sites p
    JOIN images i ON i.physical_site_id = p.physical_site_id
    JOIN upload_batches b ON b.batch_id = i.batch_id
    WHERE p.die_id = OLD.die_id AND b.wafer_id <> NEW.wafer_id
)
BEGIN
    SELECT RAISE(ABORT, 'Batch and site must belong to the same wafer');
END;

CREATE TRIGGER sites_preserve_image_wafer
BEFORE UPDATE OF die_id ON pattern_sites
WHEN EXISTS (
    SELECT 1 FROM images i
    JOIN upload_batches b ON b.batch_id = i.batch_id
    JOIN dies d ON d.die_id = NEW.die_id
    WHERE i.physical_site_id = OLD.physical_site_id AND b.wafer_id <> d.wafer_id
)
BEGIN
    SELECT RAISE(ABORT, 'Batch and site must belong to the same wafer');
END;
