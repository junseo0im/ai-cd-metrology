"""TASK18-B-19: default model-comparison tables per experiment (synthetic MLflow runs/search rows)."""
import unittest

from dashboard.components import mlflow_tables as mt


def run(rid, name, metrics=None, params=None, tags=None, start=0):
    tags = dict(tags or {}, **{"mlflow.runName": name})
    return {"info": {"run_id": rid, "experiment_id": "1", "status": "FINISHED", "start_time": start},
            "data": {"metrics": [{"key": k, "value": v} for k, v in (metrics or {}).items()],
                     "params": [{"key": k, "value": v} for k, v in (params or {}).items()],
                     "tags": [{"key": k, "value": v} for k, v in tags.items()]}}


def frame_rows(table):
    return table["frame"].to_dict("records")


class WidthTests(unittest.TestCase):
    def test_success_rate_denominator_from_recorded_sets_and_best_cells(self):
        params = {"method": "threshold", "stability_set": "27 images: repeat x2 folders, lighting x1 folder",
                  "y_fractions": "0.25,0.45,0.65", "targets_per_row": "5", "measurement_types": "line,gap",
                  "test_set": "laptop2 15 images (450 GT)", "edge_parameter": "x"}
        runs = [run("a" * 32, "t", {"stability/success_rate": 734 / 810, "test/mae_um": .2}, params, {"dataset_version": "v1"}),
                run("b" * 32, "u", {"stability/success_rate": 1.0, "test/mae_um": 0.0}, dict(params, method="unet"),
                    {"dataset_version": "v1"})]
        table = mt.width_table(runs)
        rows = frame_rows(table)
        self.assertEqual(rows[0]["계측 성공률 ↑"], "734/810 측정 행 (90.6%)")
        self.assertEqual(rows[1]["MAE ↓ (µm)"], "0.0000")                              # Real 0, not 미기록.
        self.assertEqual(rows[0]["처리시간 중앙값 ↓ (ms/장)"], mt.MISSING)
        self.assertIn((1, "MAE ↓ (µm)"), table["best"])
        self.assertIn((1, "계측 성공률 ↑"), table["best"])


class ClassificationTests(unittest.TestCase):
    def test_four_conditions_always_listed_and_effect_only_on_same_key(self):
        runs = [run("1" * 32, "mobilenet_real_only_v1", {"val/best_macro_f1": .48625, "val/macro_f1": .48},
                    {"model": "mobilenet_v3_small"}),
                run("2" * 32, "mobilenet_oversampling_v1", {"val/best_macro_f1": .48865}, {"model": "mobilenet_v3_small"}),
                run("3" * 32, "resnet18_oversampling_v1", {"val/macro_f1": .47}, {"model": "resnet18"})]
        rows = frame_rows(mt.classification_table(runs))
        self.assertEqual([(r["모델"], r["조건"]) for r in rows],
                         [("MobileNet V3 Small", "원본만"), ("MobileNet V3 Small", "오버샘플링"),
                          ("ResNet18", "원본만"), ("ResNet18", "오버샘플링")])
        self.assertEqual(rows[1]["오버샘플링 효과 (%p)"], "+0.24%p (검증 최고)")
        self.assertEqual(rows[2]["검증 최고 Macro F1 ↑"], "Run 없음")                  # Missing run kept visible.
        self.assertEqual(rows[3]["검증 최고 Macro F1 ↑"], mt.MISSING)                  # val/macro_f1 never substituted.
        self.assertTrue(rows[3]["오버샘플링 효과 (%p)"].startswith("비교 보류"))
        self.assertEqual(rows[0]["Test Macro F1 ↑"], mt.MISSING)


class DetectionTests(unittest.TestCase):
    def setUp(self):
        base = {"conf_threshold": "0.25", "dataset_total_images": "228"}
        tags = {"test_split_id": "roboflow_228_v1_test23", "model_name": "YOLOv11m-Seg"}
        self.runs = [
            run("a" * 32, "YOLOv11m-Seg_unified228", {"test/recall": .13, "test/tp": 16, "test/fn": 106,
                "test/avg_latency_ms": 238.2}, base, tags, start=1),
            run("b" * 32, "YOLOv11m-Seg_unified228", {"test/recall": .56, "test/tp": 23, "test/fn": 18,
                "test/avg_latency_ms": 245.6}, base, tags, start=2),
            run("c" * 32, "RF-DETR-Small-Seg_unified228", {"test/recall": .41, "test/tp": 50, "test/fn": 72,
                "test/avg_latency_ms": 499.6, "test/roboflow_map50": .871}, base, dict(tags, model_name="RF-DETR-Small-Seg")),
            run("d" * 32, "baseline_real_only_v1", {"val/map50_box": .246, "val/recall_box": .278}, {"model_name": "yolov8n.pt"}),
            run("e" * 32, "augmented_v1", {"val/map50_box": .242, "val/recall_box": .260}, {"model_name": "yolov8n.pt"}),
            run("f" * 32, "local_connection_test", {"connection_success": 1}),
        ]

    def test_same_name_runs_kept_and_ranked_only_within_the_same_evaluation(self):
        table = mt.detection_table(self.runs)
        rows = frame_rows(table)
        self.assertEqual(sorted(r["Run"] for r in rows), ["aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd", "eeeeeeee"])
        sets = {r["Run"]: r["평가셋"] for r in rows}
        self.assertNotEqual(sets["aaaaaaaa"], sets["bbbbbbbb"])                         # GT 122 vs GT 41.
        self.assertEqual(sets["aaaaaaaa"], sets["cccccccc"])
        best = {(rows[i]["Run"], c) for i, c in table["best"]}
        self.assertIn(("cccccccc", "Recall ↑"), best)                                  # .41 > .13 in GT 122.
        self.assertNotIn(("bbbbbbbb", "Recall ↑"), best)                               # Alone in GT 41: not ranked.
        self.assertFalse(any(run in ("dddddddd", "eeeeeeee") for run, _ in best))      # Val set not recorded.
        self.assertIn("Box/Mask 정의 미확인", next(r["비고"] for r in rows if r["Run"] == "cccccccc"))
        self.assertEqual(next(r["Box mAP@50 ↑"] for r in rows if r["Run"] == "cccccccc"), mt.MISSING)
        self.assertEqual([r["Run"] for r in rows if r["모델"] == "YOLOv8n"], ["dddddddd", "eeeeeeee"])
        self.assertEqual([c["info"]["run_id"][:8] for c in table["extra"]["connection"]], ["ffffffff"])

    def test_generic_table_lists_every_recorded_metric_and_missing(self):
        rows = frame_rows(mt.build("another-experiment", self.runs[:2] + self.runs[-1:]))
        self.assertEqual(rows[2]["connection_success"], "1")
        self.assertEqual(rows[2]["test/recall"], mt.MISSING)


if __name__ == "__main__":
    unittest.main()
