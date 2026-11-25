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
        self.assertEqual(mt.MISSING, "-")
        self.assertNotIn("Test Macro F1 ↑", rows[0])                                  # test/macro_f1 never logged: column removed.
        runs.append(run("4" * 32, "resnet18_real_only_v1", {"test/macro_f1": .5}, {"model": "resnet18"}))
        self.assertIn("Test Macro F1 ↑", frame_rows(mt.classification_table(runs))[0])  # Logged -> column shown.


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
        for column in ("Box mAP@50 ↑", "Box mAP@50–95 ↑", "Mask mAP@50 ↑", "비고"):    # mAP columns/notes removed.
            self.assertNotIn(column, rows[0])
        self.assertEqual([rows[0][c] for c in ("AP@50 O ↑", "AP@50 S ↑", "AP@50 PD ↑")], ["-"] * 3)
        self.assertEqual([r["Run"] for r in rows if r["모델"] == "YOLOv8n"], ["dddddddd", "eeeeeeee"])
        self.assertEqual([c["info"]["run_id"][:8] for c in table["extra"]["connection"]], ["ffffffff"])

    def test_per_class_ap50_columns_from_recorded_keys(self):
        runs = [run("g" * 32, "RF-DETR-Seg-Small-v2-RAW-box-o-s-pd",
                    {"test/local_box_ap50_o": .8808, "test/local_box_ap50_s": .9413, "test/local_box_ap50_pd": .9984}),
                run("h" * 32, "yolov8s_1536", {"test/o_mAP50": .907, "test/s_mAP50": .959}, {"model_name": "yolov8s"})]
        rows = {r["Run"]: r for r in frame_rows(mt.detection_table(runs))}
        self.assertEqual([rows["gggggggg"][c] for c in ("AP@50 O ↑", "AP@50 S ↑", "AP@50 PD ↑")],
                         ["0.8808", "0.9413", "0.9984"])
        self.assertEqual(rows["hhhhhhhh"]["AP@50 PD ↑"], "-")                         # Not logged -> "-", never 0.

    def test_roboflow_hosted_runs_per_class_keys_names_order_and_dataset_version(self):
        tags = {"test_split_id": "unverified", "evaluation_source": "team_test_evaluation_table"}
        runs = [run("r" * 32, "RF-DETR-Seg-Small-Roboflow-v2-t1",
                    {"test/ap50_o": .858, "test/ap50_s": .926, "test/ap50_pd": .997,
                     "test/recall_o": .864, "test/recall_s": .909, "test/recall_pd": .984},
                    tags=dict(tags, model_name="rfdetr-seg-small", dataset_version="Photo Defects 305 - Revised Labels (v2)")),
                run("t" * 32, "YOLO26s-Roboflow-dataa-v2-v1", {"test/ap50_o": .8379, "test/recall_o": .773},
                    tags=dict(tags, model_name="yolo26s", dataset_version="YOLO26 comparison - 305 images")),
                run("u" * 32, "YOLO11s-Roboflow-dataa-mpts7-v1", {"test/ap50_o": .5643, "test/recall_o": .727},
                    tags=dict(tags, model_name="yolo11s", dataset_version="YOLO11 comparison"))]
        table = mt.detection_table(runs)
        rows = frame_rows(table)
        self.assertEqual([r["모델"] for r in rows], ["YOLO11s", "YOLO26s", "RF-DETR Small"])   # Poster order.
        self.assertEqual([r["AP@50 O ↑"] for r in rows], ["0.5643", "0.8379", "0.8580"])
        self.assertEqual([r["Recall O ↑"] for r in rows], ["0.7270", "0.7730", "0.8640"])
        self.assertEqual(rows[2]["Recall PD ↑"], "0.9840")
        self.assertEqual(rows[1]["AP@50 S ↑"], "-")
        self.assertEqual(rows[0]["평가셋"], "YOLO11 comparison")
        self.assertEqual(table["best"], [])                                            # Different sets: never ranked.
        self.assertNotIn("추론시간 ↓ (ms)", rows[0])                                     # No latency logged: hidden.
        runs[0]["data"]["metrics"].append({"key": "test/avg_latency_ms", "value": 28.32})
        runs[0]["data"]["params"] += [{"key": "trainer", "value": "roboflow_hosted"},
                                      {"key": "model_input", "value": "384x384"}]
        rows = {r["Run"]: r for r in frame_rows(mt.detection_table(runs))}
        self.assertEqual(rows["rrrrrrrr"]["추론시간 ↓ (ms)"], "28.3")                   # Logged: column back.
        self.assertEqual(rows["tttttttt"]["추론시간 ↓ (ms)"], "-")
        self.assertEqual(rows["rrrrrrrr"]["실험 조건"], "Roboflow 학습 · 384x384")

    def test_generic_table_lists_every_recorded_metric_and_missing(self):
        rows = frame_rows(mt.build("another-experiment", self.runs[:2] + self.runs[-1:]))
        self.assertEqual(rows[2]["connection_success"], "1")
        self.assertEqual(rows[2]["test/recall"], mt.MISSING)


if __name__ == "__main__":
    unittest.main()
