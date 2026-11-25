"""Isolated fixtures only; synthetic detections are NOT actual model evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from inspection.analysis_runner import AnalysisRunner
from inspection.metrology_adapter import MetrologyAdapter, json_value


CALIBRATION = {"calibration_id": "unit-test-only", "um_per_px": 0.1592}


def request(path="unused.png"):
    return {"image_id": "TEST_IMAGE", "image_path": str(path), "wafer_id": "TEST_W",
            "die_id": "TEST_D", "site_code": "c", "batch_id": "TEST_B",
            "analysis_run_id": "TEST_RUN", "calibration": deepcopy(CALIBRATION)}


def native(req):
    measurements = []
    for fraction in (0.25, 0.45, 0.65):
        for kind, code, left in (("line", "L01", 10), ("gap", "G01", 65)):
            width = 20.0
            measurements.append({"measurement_type": kind, "line_id": code,
                "y_fraction": fraction, "y_px": int(fraction * 100), "width_px": width,
                "width_um": round(width * CALIBRATION["um_per_px"], 4),
                "left_x_px": left, "right_x_px": left + width,
                "center_x_px": left + width / 2, "left_sidewall_px": 2, "right_sidewall_px": 2})
    return {"image_id": req["image_id"], **{key: req.get(key) for key in
            ("wafer_id", "die_id", "site_code", "batch_id", "analysis_run_id")},
            "status": "success", "algorithm_version": "unit-test-only",
            "calibration_id": CALIBRATION["calibration_id"], "measurements": measurements,
            "pixel_scale": {"coordinate_system": "native_px", "factor_x": 1, "factor_y": 1,
                            "offset_x": 0, "offset_y": 0, "native_width": 100, "native_height": 100},
            "error_code": None, "error_message": None}


class NativeEngine:
    algorithm_version = "unit-test-only"
    def __init__(self, change=None):
        self.change, self.calls, self.requests = change, 0, []
    def analyze(self, req):
        self.calls += 1
        self.requests.append(deepcopy(req))
        result = native(req)
        if self.change:
            self.change(result)
        return result


class Defect:
    def __init__(self, detections=None, change=None):
        self.items, self.change, self.calls, self.requests = detections or [], change, 0, []
    def predict(self, req):
        self.calls += 1
        self.requests.append(deepcopy(req))
        result = {"image_id": req["image_id"], "status": "SUCCESS", "model_version": "test-only",
                  "detections": deepcopy(self.items), "error_code": None,
                  "bbox_coordinate_system": "ORIGINAL_IMAGE", "bbox_units": "px",
                  "bbox_image_width": 100, "bbox_image_height": 100, "skew_angle": 0}
        if self.change:
            self.change(result)
        return result


def detection(code, source="MODEL_POSTPROCESSED"):
    return {"class_code": code, "confidence": 0.8, "bbox_xyxy": [1, 2, 10, 20], "source": source}


class AdapterTests(unittest.TestCase):
    def adapter(self, change=None):
        self.engine = NativeEngine(change)
        return MetrologyAdapter(engine=self.engine)

    def test_normalize_after_original_judgment(self):
        out = self.adapter().measure(request())
        self.assertEqual(out["status"], "SUCCESS")
        self.assertEqual(out["judgment"]["status"], "judged")
        self.assertEqual(out["judgment"]["lines"][0]["label"], "pn")

    def test_values_preserved_without_double_conversion(self):
        req = request()
        expected = native(req)
        out = self.adapter().measure(req)
        self.assertEqual(out["measurements"], expected["measurements"])
        self.assertEqual(out["calibration"]["applied_by"], "TEAM_UNET_ENGINE")

    def test_reuses_engine(self):
        adapter = self.adapter()
        adapter.measure(request())
        second = adapter.measure(request())
        self.assertEqual(self.engine.calls, 2)
        self.assertTrue(second["timing"]["model_session_reused"])

    def test_roi_calibration_forwarded(self):
        req = request()
        req["roi"] = {"x_min": 0, "y_min": 0, "x_max": 99, "y_max": 99}
        self.adapter().measure(req)
        self.assertEqual(self.engine.requests[0], req)

    def test_partial_missing_y(self):
        out = self.adapter(lambda r: r["measurements"].pop(0)).measure(request())
        self.assertEqual(out["status"], "SUCCESS")
        self.assertTrue(out["quality"]["partial_measurement"])
        self.assertEqual(out["quality"]["missing_positions"][0]["missing_y_fractions"], [0.25])

    def test_missing_type(self):
        out = self.adapter(lambda r: r.update(measurements=[
            m for m in r["measurements"] if m["measurement_type"] == "line"])).measure(request())
        self.assertEqual(out["quality"]["absent_measurement_types"], ["gap"])

    def test_only_gap_is_partial_not_failed(self):
        out = self.adapter(lambda r: r.update(measurements=[
            m for m in r["measurements"] if m["measurement_type"] == "gap"])).measure(request())
        self.assertEqual(out["status"], "SUCCESS")
        self.assertEqual(out["judgment"]["status"], "not_judged")
        self.assertTrue(out["quality"]["partial_measurement"])

    def test_duplicate_y_is_quality_issue(self):
        out = self.adapter(lambda r: r["measurements"].append(
            deepcopy(r["measurements"][0]))).measure(request())
        self.assertTrue(out["quality"]["partial_measurement"])
        self.assertEqual(len(out["quality"]["duplicate_positions"]), 1)

    def test_failure_preserved(self):
        out = self.adapter(lambda r: r.update(status="failed", measurements=[],
            error_code="NO_SIDEWALL_DETECTED", error_message="test failure")).measure(request())
        self.assertEqual((out["status"], out["error_code"]), ("FAILED", "NO_SIDEWALL_DETECTED"))
        self.assertEqual(out["measurements"], [])

    def test_bad_engine_status(self):
        self.assertEqual(self.adapter(lambda r: r.update(status="PARTIAL_SUCCESS")).measure(request())["status"], "FAILED")

    def test_wrong_identity(self):
        self.assertEqual(self.adapter(lambda r: r.update(image_id="wrong")).measure(request())["status"], "FAILED")

    def test_success_empty_is_failure(self):
        self.assertEqual(self.adapter(lambda r: r.update(measurements=[])).measure(request())["status"], "FAILED")

    def test_nonfinite_rejected(self):
        out = self.adapter(lambda r: r["measurements"][0].update(width_um=float("nan"))).measure(request())
        self.assertEqual(out["status"], "FAILED")
        json.dumps(out, allow_nan=False)

    def test_zero_width_rejected(self):
        self.assertEqual(self.adapter(lambda r: r["measurements"][0].update(width_px=0)).measure(request())["status"], "FAILED")

    def test_unknown_y_rejected(self):
        self.assertEqual(self.adapter(lambda r: r["measurements"][0].update(y_fraction=0.8)).measure(request())["status"], "FAILED")

    def test_calibration_invalid_or_mismatched(self):
        for coefficient in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(coefficient=coefficient):
                req = request()
                req["calibration"]["um_per_px"] = coefficient
                self.assertEqual(self.adapter().measure(req)["error_code"], "INVALID_CALIBRATION")
        req = request()
        req.pop("calibration")
        req["magnification"] = 40
        self.assertEqual(self.adapter().measure(req)["error_code"], "CALIBRATION_MAGNIFICATION_MISMATCH")

    def test_unknown_scalar_cannot_serialize(self):
        for value in (object(), float("inf"), {1: "bad key"}):
            with self.assertRaises((TypeError, ValueError)):
                json_value(value)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "test.png"
        self.file.write_bytes(b"unit-test-only; no model/image decoding in these fixtures")
        self.req = request(self.file)
        self.native = NativeEngine()
        self.adapter = MetrologyAdapter(engine=self.native)

    def run_case(self, defect=None):
        self.defect = defect or Defect()
        return AnalysisRunner(self.defect, self.adapter).run(self.req)

    def test_empty_detections_still_measures_not_acceptance_pass(self):
        out = self.run_case()
        self.assertEqual(out["status"], "SUCCESS")
        self.assertEqual(self.native.calls, 1)
        self.assertIsNone(out["inspection_verdict"])
        self.assertTrue(out["official_analysis_complete"])
        self.assertIsNotNone(out["width_error"])

    def test_non_pd_defects_do_not_skip(self):
        out = self.run_case(Defect([detection("o"), detection("s")]))
        self.assertEqual(self.native.calls, 1)
        self.assertEqual(out["pd_policy"]["state"], "NO_PD")

    def test_model_pd_skips_without_invoking_metrology(self):
        out = self.run_case(Defect([detection("pd")]))
        self.assertEqual(self.native.calls, 0)
        self.assertEqual(out["metrology"]["status"], "SKIPPED")
        self.assertEqual(out["pd_policy"]["state"], "MODEL_PD_EXCLUDED")
        self.assertIsNone(out["width_error"])

    def test_heuristic_pd_requires_review(self):
        out = self.run_case(Defect([detection("pd", "HEURISTIC")]))
        self.assertEqual(out["status"], "SKIPPED")
        self.assertEqual(out["pd_policy"]["state"], "POLICY_REVIEW_REQUIRED")
        self.assertFalse(out["official_analysis_complete"])
        self.assertIsNone(out["width_error"])
        self.assertEqual(self.native.calls, 0)

    def test_model_pd_takes_priority_when_both_present(self):
        out = self.run_case(Defect([detection("pd", "HEURISTIC"), detection("pd")]))
        self.assertEqual(out["pd_policy"]["state"], "MODEL_PD_EXCLUDED")

    # ---- r33 RULE source (TASK18-B-22) ----
    def test_rule_derived_p_is_valid_and_metrology_runs(self):
        p = {**detection("p", "RULE"), "team_source": "from_s", "derived_from_index": 0}
        out = self.run_case(Defect([detection("s"), p]))
        self.assertEqual(out["status"], "SUCCESS")
        self.assertEqual(out["pd_policy"]["state"], "NO_PD")
        self.assertEqual(self.native.calls, 1)
        self.assertEqual(out["defect"]["detections"][1]["team_source"], "from_s")
        self.assertEqual(out["defect"]["detections"][1]["derived_from_index"], 0)

    def test_rule_pd_only_requires_review_without_metrology(self):
        out = self.run_case(Defect([detection("pd", "RULE")]))
        self.assertEqual(out["status"], "SKIPPED")
        self.assertEqual(out["pd_policy"]["state"], "POLICY_REVIEW_REQUIRED")
        self.assertEqual(self.native.calls, 0)

    def test_rule_and_heuristic_pd_require_review(self):
        out = self.run_case(Defect([detection("pd", "RULE"), detection("pd", "HEURISTIC")]))
        self.assertEqual(out["pd_policy"]["state"], "POLICY_REVIEW_REQUIRED")
        self.assertEqual(self.native.calls, 0)

    def test_model_pd_takes_priority_over_rule_pd(self):
        out = self.run_case(Defect([detection("pd", "RULE"), detection("pd")]))
        self.assertEqual(out["pd_policy"]["state"], "MODEL_PD_EXCLUDED")
        self.assertEqual(self.native.calls, 0)

    def test_defect_failure_does_not_imply_no_pd(self):
        out = self.run_case(Defect(change=lambda r: r.update(status="FAILED", error_code="MODEL_MISSING")))
        self.assertEqual(out["status"], "FAILED")
        self.assertEqual(out["error_code"], "MODEL_MISSING")
        self.assertEqual(out["pd_policy"]["state"], "NOT_EVALUATED")
        self.assertEqual(self.native.calls, 0)

    def test_unet_failure_preserved(self):
        self.adapter = MetrologyAdapter(engine=NativeEngine(lambda r:
            r.update(status="failed", error_code="NO_MEASUREMENT", measurements=[])))
        out = self.run_case()
        self.assertEqual(out["status"], "FAILED")
        self.assertEqual(out["error_code"], "NO_MEASUREMENT")
        self.assertIsNone(out["width_error"])

    def test_partial_success_is_separate_from_engine_status(self):
        self.adapter = MetrologyAdapter(engine=NativeEngine(lambda r: r["measurements"].pop(0)))
        out = self.run_case()
        self.assertEqual(out["metrology"]["status"], "SUCCESS")
        self.assertFalse(out["official_analysis_complete"])
        self.assertIsNone(out["width_error"])

    def test_settings_and_model_class_ids(self):
        out = self.run_case(Defect([detection("s"), detection("p")]))
        settings = self.defect.requests[0]
        self.assertIs(settings["enable_deskew"], True)
        self.assertEqual((settings["confidence"], settings["iou_threshold"]), (0.25, 0.45))
        self.assertEqual([d["class_id"] for d in out["defect"]["detections"]], [3, 1])
        self.assertEqual(out["metrology"]["image_id"], self.req["image_id"])

    def test_missing_path_and_id(self):
        runner = AnalysisRunner(Defect(), self.adapter)
        for req in ({}, {"image_id": "test"}, {"image_path": str(self.file)}):
            self.assertEqual(runner.run(req)["status"], "FAILED")

    def test_missing_file(self):
        self.req["image_path"] = str(self.file.with_name("absent.png"))
        self.assertEqual(self.run_case()["error_code"], "IMAGE_NOT_FOUND")

    def test_network_image_path_rejected(self):
        for path in ("https://example.invalid/a.png", "//server/a.png"):
            self.req["image_path"] = path
            self.assertEqual(self.run_case()["status"], "FAILED")

    def test_invalid_defect_geometry(self):
        for box in ([0, 0, 0, 20], [-1, 0, 20, 30], [0, 0, 101, 30]):
            item = detection("p")
            item["bbox_xyxy"] = box
            out = self.run_case(Defect([item]))
            self.assertEqual(out["status"], "FAILED")
        self.assertEqual(self.native.calls, 0)

    def test_nonfinite_and_nonserializable_payload(self):
        for value in (float("nan"), object()):
            item = detection("p")
            item["confidence"] = value
            out = self.run_case(Defect([item]))
            self.assertEqual(out["status"], "FAILED")
            json.dumps(out, allow_nan=False)

    def test_invalid_class_source_and_identity(self):
        for change in (lambda r: r.update(image_id="different"),
                       lambda r: r.update(detections=[detection("pn")]),
                       lambda r: r.update(detections=[detection("pd", "UNKNOWN")])):
            self.assertEqual(self.run_case(Defect(change=change))["status"], "FAILED")

    def test_existing_output_is_not_overwritten(self):
        target = Path(self.tmp.name) / "existing"
        target.mkdir()
        sentinel = target / "result.json"
        sentinel.write_text("preserve", encoding="utf-8")
        out = AnalysisRunner(Defect(), self.adapter).run(self.req, target)
        self.assertEqual(out["status"], "FAILED")
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_new_output_json_and_separate_overlay_metadata(self):
        target = Path(self.tmp.name) / "new"
        with patch("inspection.analysis_runner._save_defect_overlay", return_value={
            "base_image": "DESKEWED_IMAGE", "path": "test-only"}), patch(
            "inspection.analysis_runner._save_metrology_overlay", return_value={
            "base_image": "ORIGINAL_IMAGE", "path": "test-only"}):
            out = AnalysisRunner(Defect(), self.adapter).run(self.req, target)
        self.assertEqual(json.loads((target / "result.json").read_text()), out)
        self.assertNotEqual(out["overlays"]["defect"]["base_image"],
                            out["overlays"]["metrology"]["base_image"])

    def test_same_runner_reuses_both_adapters(self):
        defect = Defect()
        runner = AnalysisRunner(defect, self.adapter)
        runner.run(self.req)
        runner.run(self.req)
        self.assertEqual((defect.calls, self.native.calls), (2, 2))


if __name__ == "__main__":
    unittest.main()
