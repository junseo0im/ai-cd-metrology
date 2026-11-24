"""Batch-only fixtures: no real model inference and no operational DB access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from inspection.batch_runner import BatchRunner, BatchInputError
import inspection.batch_runner as batch


def response(request, kind="complete"):
    identifiers = {key: request.get(key) for key in ("wafer_id","die_id","site_code","batch_id","analysis_run_id")}
    base = {"image_id":request["image_id"], **identifiers}
    measurements = [{"measurement_type":t, "line_id":"L01" if t=="line" else "G01",
       "y_fraction":y,"width_um":9.0,"width_px":9.0/0.1592}
       for y in (0.25,0.45,0.65) for t in ("line","gap")]
    result = {**base,"image_path":request["image_path"],"status":"SUCCESS","error_code":None,
       "error_message":None,"defect":{**base,"status":"SUCCESS","detections":[],
                                    "model_version":"UNIT_FIXTURE","model_sha256":"fixture"},
       "metrology":{**base,"status":"SUCCESS","measurements":measurements,
          "algorithm_version":"UNIT_FIXTURE","calibration_id":"UNIT_FIXTURE",
          "quality":{"partial_measurement":False,"missing_positions":[]},
          "judgment":{"status":"judged","lines":[{"line_id":"L01","label":"normal",
                                                "pitch_ok":True,"near_threshold":False}]}},
       "pd_policy":{"state":"NO_PD","sources":[]},"measurement_summary":{"fixture_only":True},
       "width_error":{"fixture_only":True},"official_analysis_complete":True,"inspection_verdict":None,
       "overlays":{"defect":None,"metrology":None}}
    if kind in ("model_pd","heuristic_pd"):
        source = "MODEL_POSTPROCESSED" if kind=="model_pd" else "HEURISTIC"
        state = "MODEL_PD_EXCLUDED" if kind=="model_pd" else "POLICY_REVIEW_REQUIRED"
        result["defect"]["detections"]=[{"class_code":"pd","source":source,"confidence":0.8,
                                       "bbox_xyxy":[0,0,1,1]}]
        result.update(pd_policy={"state":state,"sources":[source]},
                      width_error=None,official_analysis_complete=False,measurement_summary=None)
        result["metrology"].update(status="SKIPPED",measurements=[],error_code=state)
        if kind=="heuristic_pd":
            result["status"]="SKIPPED"
    elif kind=="partial":
        result["metrology"]["measurements"].pop(0)
        result["metrology"]["quality"]={"partial_measurement":True,
              "missing_positions":[{"measurement_type":"line","line_id":"L01","missing_y_fractions":[0.25]}]}
        result.update(width_error=None,official_analysis_complete=False)
    elif kind=="quality":
        result["metrology"]["judgment"]["lines"][0]["pitch_ok"]=False
    elif kind in ("unet_failed","defect_failed","unreadable"):
        result.update(status="FAILED",width_error=None,measurement_summary=None,official_analysis_complete=False,
                      error_code="IMAGE_DECODE_FAILED" if kind=="unreadable" else "NO_MEASUREMENT",
                      error_message="unit fixture failure")
        result["metrology"].update(status="FAILED",measurements=[])
        if kind in ("defect_failed","unreadable"):
            result["defect"].update(status="FAILED",detections=[])
            result["metrology"]["status"]="SKIPPED"
    return result


class FakeSingleRunner:
    def __init__(self, actions=None):
        self.actions=actions or {}
        self.calls=[]
    def run(self, request, output_dir):
        self.calls.append(deepcopy(request))
        action=self.actions.get(request["image_id"],"complete")
        if action=="interrupt":
            raise KeyboardInterrupt()
        if action=="raise":
            raise RuntimeError("unit engine exception")
        target=Path(output_dir)
        target.mkdir(parents=True,exist_ok=False)
        if action=="save_failure":
            (target/"result.json").write_text("partial file preserved",encoding="utf-8")
            raise OSError("unit JSON save failure")
        if Path(request["image_path"]).read_bytes()==b"corrupt":
            action="unreadable"
        result=response(request,action)
        if action=="wrong_id":
            result["image_id"]="different"
        if action=="nonfinite":
            result["extra"]=float("nan")
            return result
        if action=="escape_overlay":
            result["overlays"]["defect"]={"path":request["image_path"]}
        (target/"result.json").write_text(json.dumps(result),encoding="utf-8")
        return result


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.outputs=self.root/"runs"
        self.single=FakeSingleRunner()
        self.runner=BatchRunner(self.single)
    def req(self, image_id="I1", filename=None, **metadata):
        path=self.root/(filename or image_id.replace("/","_").replace(":","_")+".png")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b"unit image fixture")
        return {"image_id":image_id,"image_path":str(path),"wafer_id":"W1","die_id":"W1/D13",
                "site_code":"c",**metadata}
    def run_batch(self, requests):
        return self.runner.run("TEST_BATCH",requests,self.outputs)
    def saved(self, result):
        return json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    def tree_hash(self, path):
        return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in path.rglob("*") if p.is_file()}

    def test_success_three_sequential_same_single_instance(self):
        requests=[self.req("I"+str(n)) for n in range(3)]
        out=self.run_batch(requests)
        self.assertEqual(out["execution_status"],"COMPLETED")
        self.assertEqual([r["image_id"] for r in self.single.calls],["I0","I1","I2"])
        self.assertIs(self.runner.analysis_runner,self.single)
        self.assertEqual(out["counts"]["complete_measurement_images"],3)
        self.assertEqual(self.saved(out),out)

    def test_missing_file_isolated_next_processed(self):
        good=self.req("good"); absent=self.req("absent"); Path(absent["image_path"]).unlink()
        out=self.run_batch([absent,good])
        self.assertEqual(out["execution_status"],"COMPLETED_WITH_ISSUES")
        self.assertEqual(out["counts"]["failed_images"],1)
        self.assertEqual(len(self.single.calls),1)
        failed=json.loads(Path(out["images"][0]["result_json"]).read_text())
        self.assertEqual(failed["error_code"],"IMAGE_NOT_FOUND")
        self.assertIsNone(failed["measurement_summary"])
        self.assertIsNone(failed["defect"])

    def test_unreadable_image(self):
        req=self.req(); Path(req["image_path"]).write_bytes(b"corrupt")
        out=self.run_batch([req,self.req("next")])
        self.assertEqual(out["images"][0]["reason"],"IMAGE_DECODE_FAILED")
        self.assertEqual(out["images"][1]["outcome"],"COMPLETE")

    def test_duplicate_id_rejected_before_execution(self):
        req=self.req()
        with self.assertRaises(BatchInputError) as error:
            self.run_batch([req,req])
        self.assertEqual(error.exception.status,"FAILED")
        self.assertEqual(self.single.calls,[])
        self.assertFalse(self.outputs.exists())

    def test_invalid_batch_shape(self):
        for args in (("",[self.req()]),("B",[]),("B",{}),("B",[None]),("B",[{}])):
            with self.assertRaises(BatchInputError):
                self.runner.run(args[0],args[1],self.outputs)

    def test_missing_required_metadata(self):
        req=self.req(); req.pop("wafer_id")
        out=self.run_batch([req,self.req("good")])
        self.assertEqual(out["images"][0]["reason"],"INVALID_METADATA")
        self.assertEqual(out["counts"]["processed_images"],2)
        self.assertEqual(len(self.single.calls),1)

    def test_unknown_metadata_values_never_persisted(self):
        req=self.req(private_value="do-not-record-this")
        out=self.run_batch([req])
        manifest=Path(out["request_manifest_json"]).read_text(encoding="utf-8")
        self.assertNotIn("do-not-record-this",manifest)
        self.assertEqual(out["execution_status"],"FAILED")

    def test_null_metadata_not_imputed(self):
        req=self.req(wafer_id=None,die_id=None,site_code=None)
        out=self.run_batch([req])
        meta=out["images"][0]["metadata"]
        for key in ("wafer_id","die_id","site_code","exposure_value","peb_temperature_c","dataset_role"):
            self.assertIsNone(meta[key])

    def test_real_metadata_names_and_db_id_preserved(self):
        req=self.req(exposure_value=60,exposure_unit=None,peb_temperature_c=110,
            dataset_role="evaluation",upload_mode="lighting",die_code="13")
        original=deepcopy(req)
        out=self.run_batch([req])
        self.assertEqual(req,original)
        meta=out["images"][0]["metadata"]
        self.assertEqual((meta["die_id"],meta["die_code"]),("W1/D13","13"))
        self.assertEqual((meta["exposure_value"],meta["peb_temperature_c"]),(60,110))
        self.assertEqual(meta["upload_mode"],"lighting")
        self.assertEqual(out["official_aggregates"],"NOT_COMPUTED")

    def test_input_batch_mismatch_rejected_per_image(self):
        out=self.run_batch([self.req(batch_id="OTHER"),self.req("good")])
        self.assertEqual(out["images"][0]["reason"],"INVALID_METADATA")
        self.assertEqual(len(self.single.calls),1)

    def test_invalid_numeric_metadata(self):
        for value in (float("nan"),float("inf"),True,"60"):
            out=self.run_batch([self.req(exposure_value=value)])
            self.assertEqual(out["counts"]["failed_images"],1)
            json.dumps(out,allow_nan=False)

    def test_invalid_site_role_mode(self):
        for field,value in (("site_code","bad"),("dataset_role","TRAIN"),("upload_mode","other")):
            out=self.run_batch([self.req(**{field:value})])
            self.assertEqual(out["images"][0]["reason"],"INVALID_METADATA")

    def test_same_filename_and_same_hash_distinct_image_ids(self):
        reqs=[self.req("a","folder_a/same.png"),self.req("b","folder_b/same.png")]
        out=self.run_batch(reqs)
        paths=[r["result_json"] for r in out["images"]]
        self.assertEqual(len(set(paths)),2)
        self.assertEqual(out["counts"]["success_images"],2)
        self.assertEqual([json.loads(Path(p).read_text())["image_id"] for p in paths],["a","b"])

    def test_hostile_identifiers_cannot_escape_output(self):
        reqs=[self.req("../outside","a.png"),self.req("C:/nested","b.png")]
        out=self.runner.run("../batch",reqs,self.outputs)
        for row in out["images"]:
            self.assertTrue(Path(row["result_json"]).resolve().is_relative_to(Path(out["run_directory"])))
            self.assertRegex(row["image_key"],r"^[a-f0-9]{64}$")

    def test_model_pd_expected_skip_not_batch_failure(self):
        self.single.actions["I1"]="model_pd"
        out=self.run_batch([self.req()])
        self.assertEqual(out["execution_status"],"COMPLETED")
        self.assertEqual(out["counts"]["skipped_metrology_images"],1)
        self.assertEqual(out["counts"]["model_pd_excluded_images"],1)
        self.assertEqual(out["counts"]["failed_images"],0)

    def test_heuristic_pd_review_is_not_failure_or_complete_measurement(self):
        self.single.actions["I1"]="heuristic_pd"
        out=self.run_batch([self.req()])
        self.assertEqual(out["execution_status"],"COMPLETED_WITH_ISSUES")
        self.assertEqual(out["counts"]["policy_review_images"],1)
        self.assertEqual(out["counts"]["skipped_engine_images"],1)
        self.assertEqual(out["counts"]["failed_images"],0)

    def test_partial_values_and_judgment_preserved_without_aggregate(self):
        self.single.actions["I1"]="partial"
        out=self.run_batch([self.req()])
        raw=json.loads(Path(out["images"][0]["result_json"]).read_text())
        self.assertEqual(len(raw["metrology"]["measurements"]),5)
        self.assertEqual(raw["metrology"]["judgment"]["lines"][0]["label"],"normal")
        self.assertIsNone(raw["width_error"])
        self.assertEqual(out["counts"]["partial_measurement_images"],1)
        self.assertEqual(out["official_aggregates"],"NOT_COMPUTED")
        self.assertNotIn("measurements",json.dumps(out))

    def test_unet_and_defect_failure_are_isolated(self):
        self.single.actions.update(a="unet_failed",b="defect_failed",c="raise")
        out=self.run_batch([self.req(i) for i in ("a","b","c","good")])
        self.assertEqual(out["counts"]["failed_images"],3)
        self.assertEqual(out["images"][-1]["outcome"],"COMPLETE")

    def test_counts_partition_and_overlap_defined(self):
        modes=["complete","model_pd","heuristic_pd","partial","defect_failed","unet_failed","quality"]
        reqs=[self.req(str(i)) for i in range(len(modes))]
        self.single.actions={str(i):mode for i,mode in enumerate(modes)}
        out=self.run_batch(reqs); c=out["counts"]
        self.assertEqual((c["requested_images"],c["processed_images"],c["unprocessed_images"]),(7,7,0))
        self.assertEqual((c["success_images"],c["failed_images"],c["skipped_engine_images"]),(4,2,1))
        self.assertEqual((c["skipped_metrology_images"],c["partial_measurement_images"],c["policy_review_images"]),(3,1,1))
        self.assertEqual(c["quality_review_images"],1)

    def test_all_fail_batch_failed(self):
        self.single.actions.update(a="raise",b="raise")
        self.assertEqual(self.run_batch([self.req("a"),self.req("b")])["execution_status"],"FAILED")

    def test_json_save_failure_preserves_partial_file_continues(self):
        self.single.actions["a"]="save_failure"
        out=self.run_batch([self.req("a"),self.req("good")])
        row=out["images"][0]
        self.assertEqual(row["reason"],"RESULT_WRITE_FAILED")
        partial=Path(row["partial_artifact_directory"])/"result.json"
        self.assertEqual(partial.read_text(),"partial file preserved")
        self.assertEqual(Path(row["result_json"]).name,"failure.json")
        self.assertEqual(out["images"][1]["outcome"],"COMPLETE")

    def test_failure_record_write_error_remains_visible_in_summary(self):
        req=self.req(); Path(req["image_path"]).unlink()
        real=batch._new_json
        def fail_results(path,value):
            if Path(path).name in ("result.json","failure.json"):
                raise OSError("unit disk error")
            return real(path,value)
        with patch.object(batch,"_new_json",side_effect=fail_results):
            out=self.run_batch([req,self.req("good")])
        self.assertIsNone(out["images"][0]["result_json"])
        self.assertIn("unit disk error",out["images"][0]["result_save_error"])
        self.assertEqual(out["images"][1]["outcome"],"COMPLETE")

    def test_wrong_result_id_and_escaped_overlay_rejected(self):
        for action in ("wrong_id","escape_overlay","nonfinite"):
            self.single.actions["I1"]=action
            out=self.run_batch([self.req()])
            self.assertEqual(out["counts"]["failed_images"],1)
            json.dumps(out,allow_nan=False)

    def test_same_batch_rerun_gets_new_id_and_preserves_history(self):
        req=self.req()
        first=self.run_batch([req]); before=self.tree_hash(Path(first["run_directory"]))
        second=self.run_batch([req])
        self.assertNotEqual(first["batch_run_id"],second["batch_run_id"])
        self.assertEqual(before,self.tree_hash(Path(first["run_directory"])))
        self.assertFalse(second["previous_results_reused"])
        self.assertNotEqual(first["images"][0]["analysis_run_id"],second["images"][0]["analysis_run_id"])

    def test_input_analysis_id_retained_as_correlation_only(self):
        out=self.run_batch([self.req(analysis_run_id="caller-correlation")])
        row=out["images"][0]
        self.assertEqual(row["requested_analysis_run_id"],"caller-correlation")
        self.assertTrue(row["analysis_run_id"].startswith(out["batch_run_id"]))
        self.assertEqual(self.single.calls[0]["analysis_run_id"],row["analysis_run_id"])

    def test_stop_and_retry_unprocessed_in_new_run(self):
        reqs=[self.req(str(i)) for i in range(3)]
        out=self.runner.run("B",reqs,self.outputs,stop_requested=lambda:len(self.single.calls)>=1)
        self.assertEqual(out["execution_status"],"INTERRUPTED")
        self.assertEqual((out["counts"]["processed_images"],out["counts"]["unprocessed_images"]),(1,2))
        before=self.tree_hash(Path(out["run_directory"]))
        retried=self.runner.retry(out["summary_json"],self.outputs)
        self.assertEqual([r["image_id"] for r in retried["images"]],["1","2"])
        self.assertEqual(retried["retry_of_run_id"],out["batch_run_id"])
        self.assertEqual(retried["execution_status"],"COMPLETED")
        self.assertEqual(before,self.tree_hash(Path(out["run_directory"])))

    def test_keyboard_interrupt_during_image_keeps_current_unprocessed(self):
        self.single.actions["b"]="interrupt"
        out=self.run_batch([self.req(i) for i in ("a","b","c")])
        self.assertEqual(out["execution_status"],"INTERRUPTED")
        self.assertEqual(out["counts"]["unprocessed_images"],2)
        self.assertEqual(out["images"][1]["reason"],"INTERRUPTED_DURING_IMAGE")
        self.single.actions.pop("b")
        retried=self.runner.retry(out["summary_json"],self.outputs)
        self.assertEqual(retried["counts"]["success_images"],2)

    def test_failed_only_retry_does_fresh_execution(self):
        self.single.actions["bad"]="raise"
        out=self.run_batch([self.req("bad"),self.req("good")])
        self.single.actions["bad"]="complete"
        retry=self.runner.retry(out["summary_json"],self.outputs)
        self.assertEqual([r["image_id"] for r in retry["images"]],["bad"])
        self.assertEqual(len(self.single.calls),3)

    def test_explicit_retry_ids_and_invalid_selection(self):
        out=self.run_batch([self.req("a"),self.req("b")])
        retry=self.runner.retry(out["summary_json"],self.outputs,image_ids=["b"])
        self.assertEqual(retry["counts"]["requested_images"],1)
        for ids in ([],["not-in-run"],["a","a"]):
            with self.assertRaises(BatchInputError):
                self.runner.retry(out["summary_json"],self.outputs,image_ids=ids)

    def test_retry_manifest_tamper_is_rejected(self):
        self.single.actions["I1"]="raise"
        out=self.run_batch([self.req()])
        manifest=Path(out["request_manifest_json"])
        manifest.write_text(manifest.read_text()+" ",encoding="utf-8")
        with self.assertRaises(BatchInputError):
            self.runner.retry(out["summary_json"],self.outputs)

    def test_invalid_metadata_retry_requires_explicit_correction(self):
        req=self.req(); req.pop("site_code")
        out=self.run_batch([req])
        with self.assertRaises(BatchInputError):
            self.runner.retry(out["summary_json"],self.outputs)

    def test_existing_run_directory_collision_preserved(self):
        old=self.outputs/"batch-fixed"; old.mkdir(parents=True)
        marker=old/"sentinel"; marker.write_text("preserve",encoding="utf-8")
        with patch.object(batch.uuid,"uuid4",return_value=SimpleNamespace(hex="fixed")):
            with self.assertRaises(FileExistsError):
                self.run_batch([self.req()])
        self.assertEqual(marker.read_text(),"preserve")
        self.assertEqual(self.single.calls,[])

    def test_unwritable_initial_summary_stops_before_model(self):
        with patch.object(batch,"_checkpoint",side_effect=OSError("unwritable output")):
            with self.assertRaises(OSError):
                self.run_batch([self.req()])
        self.assertEqual(self.single.calls,[])

    def test_originals_unchanged(self):
        req=self.req()
        before=hashlib.sha256(Path(req["image_path"]).read_bytes()).hexdigest()
        self.run_batch([req])
        self.assertEqual(before,hashlib.sha256(Path(req["image_path"]).read_bytes()).hexdigest())


if __name__=="__main__":
    unittest.main()
