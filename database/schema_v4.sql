-- Additive v4 analysis schema. Apply to a validated v3 DB inside a transaction.
CREATE TABLE analysis_batch_runs (
 batch_run_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(batch_run_id))>0),
 batch_id TEXT NOT NULL REFERENCES upload_batches(batch_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
 parent_batch_run_id TEXT REFERENCES analysis_batch_runs(batch_run_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
 started_at TEXT NOT NULL CHECK(julianday(started_at) IS NOT NULL),
 ended_at TEXT CHECK(ended_at IS NULL OR julianday(ended_at) IS NOT NULL),
 execution_status TEXT NOT NULL CHECK(execution_status IN ('COMPLETED','COMPLETED_WITH_ISSUES','FAILED','INTERRUPTED')),
 requested_images INTEGER NOT NULL CHECK(requested_images>0),
 processed_images INTEGER NOT NULL CHECK(processed_images>=0),
 unprocessed_images INTEGER NOT NULL CHECK(unprocessed_images>=0),
 success_images INTEGER NOT NULL CHECK(success_images>=0),
 failed_images INTEGER NOT NULL CHECK(failed_images>=0),
 skipped_engine_images INTEGER NOT NULL CHECK(skipped_engine_images>=0),
 skipped_metrology_images INTEGER NOT NULL CHECK(skipped_metrology_images>=0),
 complete_measurement_images INTEGER NOT NULL CHECK(complete_measurement_images>=0),
 partial_measurement_images INTEGER NOT NULL CHECK(partial_measurement_images>=0),
 policy_review_images INTEGER NOT NULL CHECK(policy_review_images>=0),
 quality_review_images INTEGER NOT NULL CHECK(quality_review_images>=0),
 model_pd_excluded_images INTEGER NOT NULL CHECK(model_pd_excluded_images>=0),
 interruption_reason TEXT,
 execution_config_json TEXT NOT NULL CHECK(json_valid(execution_config_json)),
 summary_path TEXT NOT NULL UNIQUE,
 summary_sha256 TEXT NOT NULL CHECK(length(summary_sha256)=64),
 manifest_path TEXT NOT NULL,
 manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256)=64),
 import_sha256 TEXT NOT NULL CHECK(length(import_sha256)=64),
 CHECK(requested_images=processed_images+unprocessed_images),
 CHECK(processed_images=success_images+failed_images+skipped_engine_images),
 CHECK(parent_batch_run_id IS NULL OR parent_batch_run_id<>batch_run_id),
 CHECK(ended_at IS NULL OR julianday(ended_at)>=julianday(started_at))
);
CREATE TABLE analysis_image_runs (
 analysis_run_id TEXT PRIMARY KEY NOT NULL CHECK(length(trim(analysis_run_id))>0),
 batch_run_id TEXT NOT NULL REFERENCES analysis_batch_runs(batch_run_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
 image_id TEXT NOT NULL REFERENCES images(image_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
 request_index INTEGER NOT NULL CHECK(request_index>=0),
 started_at TEXT CHECK(started_at IS NULL OR julianday(started_at) IS NOT NULL),
 ended_at TEXT CHECK(ended_at IS NULL OR julianday(ended_at) IS NOT NULL),
 processing_status TEXT NOT NULL CHECK(processing_status IN ('UNPROCESSED','COMPLETED','FAILED')),
 engine_status TEXT CHECK(engine_status IS NULL OR engine_status IN ('SUCCESS','FAILED','SKIPPED')),
 defect_status TEXT CHECK(defect_status IS NULL OR defect_status IN ('SUCCESS','FAILED','SKIPPED')),
 metrology_status TEXT CHECK(metrology_status IS NULL OR metrology_status IN ('SUCCESS','FAILED','SKIPPED')),
 outcome TEXT NOT NULL CHECK(outcome IN ('UNPROCESSED','COMPLETE','FAILED','MODEL_PD_EXCLUDED','POLICY_REVIEW_REQUIRED','PARTIAL_MEASUREMENT','QUALITY_REVIEW_REQUIRED')),
 pd_policy_state TEXT,
 partial_measurement INTEGER CHECK(partial_measurement IS NULL OR partial_measurement IN (0,1)),
 policy_review INTEGER NOT NULL CHECK(policy_review IN (0,1)),
 reason TEXT, error_message TEXT,
 line_mean_um REAL, gap_mean_um REAL,
 line_measurement_count INTEGER, gap_measurement_count INTEGER,
 signed_error_pct REAL, absolute_error_pct REAL,
 calibration_id TEXT, calibration_json TEXT CHECK(calibration_json IS NULL OR json_valid(calibration_json)),
 quality_json TEXT CHECK(quality_json IS NULL OR json_valid(quality_json)),
 pixel_scale_json TEXT CHECK(pixel_scale_json IS NULL OR json_valid(pixel_scale_json)),
 metadata_json TEXT NOT NULL CHECK(json_valid(metadata_json)),
 configuration_json TEXT CHECK(configuration_json IS NULL OR json_valid(configuration_json)),
 model_version TEXT, model_sha256 TEXT, algorithm_version TEXT,
 bbox_coordinate_space TEXT, bbox_width INTEGER, bbox_height INTEGER, skew_angle REAL,
 image_sha256 TEXT NOT NULL CHECK(length(image_sha256)=64),
 result_path TEXT UNIQUE, result_sha256 TEXT,
 defect_overlay_path TEXT, defect_overlay_sha256 TEXT,
 metrology_overlay_path TEXT, metrology_overlay_sha256 TEXT,
 is_adopted INTEGER NOT NULL DEFAULT 0 CHECK(is_adopted IN (0,1)),
 UNIQUE(batch_run_id,image_id), UNIQUE(batch_run_id,request_index),
 CHECK((result_path IS NULL)=(result_sha256 IS NULL)),
 CHECK(result_sha256 IS NULL OR length(result_sha256)=64),
 CHECK((defect_overlay_path IS NULL)=(defect_overlay_sha256 IS NULL)),
 CHECK((metrology_overlay_path IS NULL)=(metrology_overlay_sha256 IS NULL)),
 CHECK(is_adopted=0 OR COALESCE((outcome='COMPLETE' AND engine_status='SUCCESS' AND metrology_status='SUCCESS'
       AND pd_policy_state='NO_PD' AND partial_measurement=0 AND policy_review=0),0)),
 CHECK(outcome NOT IN ('PARTIAL_MEASUREMENT','MODEL_PD_EXCLUDED','POLICY_REVIEW_REQUIRED','FAILED','UNPROCESSED')
       OR (signed_error_pct IS NULL AND absolute_error_pct IS NULL))
);
CREATE TABLE defect_detections (
 analysis_run_id TEXT NOT NULL REFERENCES analysis_image_runs(analysis_run_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
 detection_index INTEGER NOT NULL CHECK(detection_index>=0),
 class_code TEXT NOT NULL, class_id INTEGER, class_name TEXT,
 confidence REAL NOT NULL CHECK(confidence>=0 AND confidence<=1),
 x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
 source TEXT NOT NULL, coordinate_space TEXT,
 payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
 PRIMARY KEY(analysis_run_id,detection_index), CHECK(x2>x1 AND y2>y1)
);
CREATE TABLE metrology_measurements (
 analysis_run_id TEXT NOT NULL REFERENCES analysis_image_runs(analysis_run_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
 measurement_index INTEGER NOT NULL CHECK(measurement_index>=0),
 measurement_type TEXT NOT NULL CHECK(measurement_type IN ('line','gap')),
 line_id TEXT NOT NULL,
 y_fraction REAL NOT NULL CHECK(y_fraction>0 AND y_fraction<1),
 y_px REAL, width_px REAL NOT NULL CHECK(width_px>0),
 width_um REAL CHECK(width_um IS NULL OR width_um>0),
 left_x_px REAL, right_x_px REAL, center_x_px REAL,
 payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
 PRIMARY KEY(analysis_run_id,measurement_index)
);
CREATE TABLE line_judgments (
 analysis_run_id TEXT NOT NULL REFERENCES analysis_image_runs(analysis_run_id) ON DELETE RESTRICT ON UPDATE RESTRICT,
 line_id TEXT NOT NULL, label TEXT NOT NULL, reason TEXT,
 min_um REAL, max_um REAL, mean_um REAL, n_rows INTEGER,
 near_threshold INTEGER CHECK(near_threshold IS NULL OR near_threshold IN (0,1)),
 pitch_ok INTEGER CHECK(pitch_ok IS NULL OR pitch_ok IN (0,1)),
 payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
 PRIMARY KEY(analysis_run_id,line_id)
);
CREATE INDEX idx_analysis_batch_history ON analysis_batch_runs(batch_id,started_at);
CREATE INDEX idx_analysis_image_history ON analysis_image_runs(image_id,started_at);
CREATE UNIQUE INDEX idx_analysis_one_adopted ON analysis_image_runs(image_id) WHERE is_adopted=1;
CREATE INDEX idx_analysis_measurement_identity ON metrology_measurements(analysis_run_id,measurement_type,line_id,y_fraction);
CREATE TRIGGER analysis_batch_retry_membership BEFORE INSERT ON analysis_batch_runs
WHEN NEW.parent_batch_run_id IS NOT NULL AND NOT EXISTS (
 SELECT 1 FROM analysis_batch_runs p WHERE p.batch_run_id=NEW.parent_batch_run_id AND p.batch_id=NEW.batch_id)
BEGIN SELECT RAISE(ABORT,'Retry parent must belong to the same registered Batch'); END;
CREATE TRIGGER analysis_batch_identity_immutable BEFORE UPDATE OF batch_run_id,batch_id,parent_batch_run_id ON analysis_batch_runs
BEGIN SELECT RAISE(ABORT,'Analysis run identity is immutable'); END;
CREATE TRIGGER analysis_image_membership_insert BEFORE INSERT ON analysis_image_runs
WHEN NOT EXISTS (SELECT 1 FROM images i JOIN analysis_batch_runs r ON r.batch_id=i.batch_id
                 WHERE i.image_id=NEW.image_id AND r.batch_run_id=NEW.batch_run_id)
BEGIN SELECT RAISE(ABORT,'Analysis Image and Batch membership mismatch'); END;
CREATE TRIGGER analysis_image_membership_update BEFORE UPDATE OF image_id,batch_run_id ON analysis_image_runs
WHEN NOT EXISTS (SELECT 1 FROM images i JOIN analysis_batch_runs r ON r.batch_id=i.batch_id
                 WHERE i.image_id=NEW.image_id AND r.batch_run_id=NEW.batch_run_id)
BEGIN SELECT RAISE(ABORT,'Analysis Image and Batch membership mismatch'); END;
CREATE TRIGGER analysis_preserve_registration_batch BEFORE UPDATE OF batch_id ON images
WHEN EXISTS (SELECT 1 FROM analysis_image_runs a JOIN analysis_batch_runs b ON b.batch_run_id=a.batch_run_id
             WHERE a.image_id=OLD.image_id AND b.batch_id<>NEW.batch_id)
BEGIN SELECT RAISE(ABORT,'Registered Image has immutable analysis Batch history'); END;
