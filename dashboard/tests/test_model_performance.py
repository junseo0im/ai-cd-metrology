"""AppTest uses offline responses only; live-server verification is separate."""
from contextlib import contextmanager
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from dashboard import mlflow_provider as provider

ROOT = Path(__file__).resolve().parents[2]


def make_run(rid, exp="9", metrics=None, name="same", start=2000):
    return {"info": {"run_id": rid, "experiment_id": exp, "status": "FINISHED", "start_time": start},
            "data": {"metrics": [{"key": k, "value": v} for k, v in (metrics or {}).items()],
                     "params": [{"key": "model", "value": "recorded-model"}],
                     "tags": [{"key": "mlflow.runName", "value": name}]}}


class ModelPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.experiments = [{"experiment_id": "9", "name": "IDA_Width_Measurement"},
                            {"experiment_id": "42", "name": "another-experiment"}]
        self.runs = [make_run("aaaaaaaa1111", metrics={"test/mae_um": .2, "stability/success_rate": 1}),
                     make_run("bbbbbbbb2222", metrics={"custom/score": 0}, start=1000)]
        self.fetch_patch = patch("dashboard.components.mlflow.fetch", side_effect=self.fetch)
        self.fetch_patch.start()
        self.addCleanup(self.fetch_patch.stop)
        self.settings_patch = patch("dashboard.mlflow_provider.connection_from_settings",
                                    return_value=provider.Connection())
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        # Roboflow tab: deterministic regardless of a locally running Gradio app (empty = not configured).
        self.gradio_env = patch.dict("os.environ", {"IDA_ROBOFLOW_GRADIO_URL": ""})
        self.gradio_env.start()
        self.addCleanup(self.gradio_env.stop)
        # TASK18-B-19: the former browser now lives in the collapsed 'MLflow 상세 조회' (lazy). AppTest cannot
        # click a tracked expander, so tests open it by default unless self.details_open is False.
        import streamlit as st
        native = st.expander
        self.details_open = True
        def expander(label, *args, **kwargs):
            if label == "MLflow 상세 조회":
                kwargs["expanded"] = self.details_open
            return native(label, *args, **kwargs)
        self.expander_patch = patch("dashboard.components.mlflow.st.expander", side_effect=expander)
        self.expander_patch.start()
        self.addCleanup(self.expander_patch.stop)

    def fetch(self, connection, operation, identifier="", limit=50, token="", path="", revision=0):
        self.calls.append((operation, identifier, token, path))
        if operation == "experiments":
            return self.experiments
        if operation in ("runs", "all_runs"):
            if identifier == "42":
                return {"runs": []}
            return {"runs": self.runs + [make_run("other-exp", exp="42")]}
        if operation == "run":
            return next(run for run in self.runs if provider.run_id(run) == identifier)
        if operation == "artifacts":
            return {"files": [{"path": "results", "is_dir": True}]} if not path else {
                "files": [{"path": path + "/report.csv", "is_dir": False, "file_size": 123}]}
        raise AssertionError(operation)

    def app(self):
        at = AppTest.from_file(ROOT / "dashboard/app.py", default_timeout=30).run()
        at.switch_page("pages/model_performance.py").run()
        self.assertFalse(at.exception)
        return at

    @contextmanager
    def expanded_detail(self):
        # Run detail is now a tab inside 'MLflow 상세 조회', which tests open by default (see setUp).
        yield

    def test_tabs_real_data_controls_and_disconnected_roboflow(self):
        before = set(sys.modules)
        at = self.app()
        self.assertEqual([tab.label for tab in at.tabs if tab.label in ("MLflow", "Roboflow")], ["MLflow", "Roboflow"])
        self.assertIn("Metric 비교", [tab.label for tab in at.tabs])            # Former browser, inside 'MLflow 상세 조회'.
        self.assertEqual(at.session_state["active_page_key"], "model_performance")
        self.assertEqual(at.selectbox(key="mf_experiment").options,
                         ["IDA_Width_Measurement", "another-experiment"])
        self.assertTrue(any("연결 준비 중" in row.value for row in at.caption))
        self.assertIn("별도 Gradio 실험 앱으로 실행하기 (선택)", [e.label for e in at.expander])
        self.assertTrue(any("RF-DETR 결함 검출 시연" == s.value for s in at.subheader))
        self.assertTrue(any("Tracking Server" in row.value for row in at.caption))
        self.assertTrue(any("GUEST" in str(row.proto) for row in at.get("popover")))
        self.assertFalse(any("roboflow_app" in name for name in set(sys.modules) - before))
        self.assertFalse(any("Gradio" in button.label for button in at.button))

    def test_roboflow_gradio_embedded_by_url_only_when_running(self):
        """Configured + reachable -> iframe of the local Gradio app; not running -> notice with the run command."""
        import os
        before = set(sys.modules)
        with patch.dict(os.environ, {"IDA_ROBOFLOW_GRADIO_URL": "http://127.0.0.1:65530"}):
            at = self.app()
            frames = at.get("iframe")
            self.assertFalse(frames)                                   # Nothing listens on that port.
            self.assertTrue(any("실행 중이 아닙니다" in row.value for row in at.caption))
            self.assertTrue(any("roboflow_app" in row.value and "app.py" in row.value for row in at.markdown))
            with patch("urllib.request.urlopen") as opened:
                opened.return_value.__enter__.return_value.status = 200
                at = self.app()
            frames = at.get("iframe")
            self.assertEqual(len(frames), 1)
            self.assertIn("http://127.0.0.1:65530", str(frames[0].proto))
        self.assertFalse(any("roboflow_app" in name for name in set(sys.modules) - before))   # Never imported.

    def test_sparse_metric_chart_table_and_duplicate_name_labels(self):
        at = self.app()
        at.selectbox(key="mf_metric_9_1_50_0").select("test/mae_um").run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("plotly_chart")), 1)
        self.assertTrue(any("미기록 Run 1개" in row.value for row in at.caption))
        at.multiselect(key="mf_table_metrics_9_1_50_0").set_value(["custom/score"]).run()
        table = next(row.value for row in at.dataframe if "Run" in row.value and "Parameter 요약" in row.value)
        self.assertEqual(table["Run"].tolist(), ["same · aaaaaaaa", "same · bbbbbbbb"])
        self.assertNotIn("Run ID", table)
        self.assertNotIn("시작 (UTC)", table)
        self.assertNotIn("종료 (UTC)", table)
        self.assertEqual(table["Run"].nunique(), 2)
        self.assertTrue(table["test/mae_um"].isna().iloc[1])
        at.selectbox(key="mf_metric_9_1_50_0").select("stability/success_rate").run()
        self.assertTrue(any("측정 행 기준" in row.value for row in at.caption))
        at.selectbox(key="mf_metric_9_1_50_0").select("custom/score").run()
        self.assertTrue(any("기록값 1개" in row.value for row in at.caption))

    def test_run_details_artifact_directory_browsing_and_no_download(self):
        self.details_open = False
        at = self.app()
        self.assertFalse(any(call[0] in {"run", "artifacts"} for call in self.calls))
        self.assertFalse(next(item for item in at.expander if item.label == "MLflow 상세 조회").proto.expanded)
        with self.expanded_detail():
            self.details_open = True
            at.session_state["mf_details_9"] = True
            at.run()
            self.assertFalse(at.exception)
            self.assertTrue(all(label in [tab.label for tab in at.tabs] for label in ["Parameters", "Metrics", "Tags"]))
            self.assertTrue(any(row.value == "aaaaaaaa1111" for row in at.code))
            at.selectbox(key="mf_artifact_folder_aaaaaaaa1111_").select("results").run()
            at.button(key="mf_artifact_open").click().run()
            self.assertFalse(at.exception)
            table = next(row.value for row in at.dataframe if "파일명" in row.value)
            self.assertEqual(table.iloc[0]["파일명"], "report.csv")
            self.assertEqual(table.iloc[0]["크기 (bytes)"], 123)
            self.assertEqual(len(at.get("download_button")), 0)
            at.button(key="mf_artifact_parent").click().run()
            self.assertFalse(at.exception)
            self.assertTrue(any("경로: /" == row.value for row in at.text))

    def test_experiment_change_empty_runs_and_empty_experiments(self):
        at = self.app()
        at.selectbox(key="mf_experiment").select("42").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("조회된 Run이 없습니다" in row.value for row in at.info))
        self.assertEqual(len(at.get("plotly_chart")), 0)
        self.experiments = []
        at.button(key="mf_refresh").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Experiment가 없습니다" in row.value for row in at.info))

    def test_no_metrics_is_not_fake_zero(self):
        self.runs = [make_run("aaaaaaaa1111")]
        at = self.app()
        self.assertEqual(len(at.get("plotly_chart")), 0)
        self.assertTrue(any("기록된 Metric이 없습니다" in row.value for row in at.info))

    def test_server_failure_is_contained_and_roboflow_navigation_remain(self):
        with patch("dashboard.components.mlflow.fetch", side_effect=provider.MLflowError("MLflow 연결 실패")):
            at = self.app()
        self.assertFalse(at.exception)
        self.assertTrue(any("MLflow 연결 실패" in row.value for row in at.warning))
        self.assertEqual([tab.label for tab in at.tabs], ["MLflow", "Roboflow"])
        self.assertTrue(any("연결 준비 중" in row.value for row in at.caption))
        self.assertTrue(any('모델 성능' in row.value for row in at.sidebar.markdown))

    def test_pagination_refresh_and_run_identity(self):
        original = self.fetch
        def paged(connection, operation, identifier="", limit=50, token="", path="", revision=0):
            if operation == "runs":
                return {"runs": self.runs[:1] if not token else self.runs[1:],
                        **({"next_page_token": "next"} if not token else {})}
            return original(connection, operation, identifier, limit, token, path, revision)
        with patch("dashboard.components.mlflow.fetch", side_effect=paged):
            at = self.app()
            at.button(key="mf_runs_next").click().run()
            self.assertFalse(at.exception)
            at.session_state["mf_detail_9_2_50_0"] = True
            at.run()
            self.assertEqual(at.selectbox(key="mf_run_9_2_50_0").value, "bbbbbbbb2222")
            at.button(key="mf_runs_previous").click().run()
            at.session_state["mf_detail_9_1_50_0"] = True
            at.run()
            self.assertEqual(at.selectbox(key="mf_run_9_1_50_0").value, "aaaaaaaa1111")
            at.button(key="mf_refresh").click().run()
            self.assertEqual(at.session_state["_mf_revision"], 1)
            self.assertTrue(at.button(key="mf_runs_previous").disabled)


    def test_primary_metric_is_always_in_table_and_defaults_to_test(self):
        at = self.app()
        self.assertEqual(at.selectbox(key="mf_metric_9_1_50_0").value, "test/mae_um")
        for key in ["stability/success_rate", "custom/score", "test/mae_um"]:
            at.selectbox(key="mf_metric_9_1_50_0").select(key).run()
            self.assertFalse(at.exception)
            table = next(item.value for item in at.dataframe if "Parameter 요약" in item.value)
            self.assertEqual(table.columns[1], key)
            self.assertNotIn(key, at.multiselect(key="mf_table_metrics_9_1_50_0").options)

    def test_run_change_resets_artifact_scope_and_closed_detail_avoids_requests(self):
        self.details_open = False
        at = self.app()
        self.assertFalse(any(call[0] in {"run", "artifacts"} for call in self.calls))
        with self.expanded_detail():
            self.details_open = True
            at.session_state["mf_details_9"] = True
            at.run()
            at.selectbox(key="mf_artifact_folder_aaaaaaaa1111_").select("results").run()
            at.button(key="mf_artifact_open").click().run()
            at.selectbox(key="mf_run_9_1_50_0").select("bbbbbbbb2222").run()
            self.assertFalse(at.exception)
            self.assertEqual([row.value for row in at.code], ["bbbbbbbb2222"])
            self.assertEqual(at.session_state["_mf_artifact_path"], "")
        self.calls.clear()
        self.details_open = False
        at.session_state["mf_details_9"] = False
        at.run()
        self.assertFalse(any(call[0] in {"run", "artifacts"} for call in self.calls))

    def test_experiment_switch_replaces_metrics_runs_and_detail(self):
        other = make_run("cccccccc3333", exp="42", metrics={"train/accuracy": .999, "val/macro_f1": .8})
        original = self.fetch
        def alternate(connection, operation, identifier="", limit=50, token="", path="", revision=0):
            if operation in ("runs", "all_runs") and identifier == "42":
                return {"runs": [other]}
            if operation == "run" and identifier == "cccccccc3333":
                return other
            return original(connection, operation, identifier, limit, token, path, revision)
        with patch("dashboard.components.mlflow.fetch", side_effect=alternate):
            at = self.app()
            at.selectbox(key="mf_experiment").select("42").run()
            self.assertFalse(at.exception)
            metric = at.selectbox(key="mf_metric_42_1_50_0")
            self.assertEqual(metric.value, "val/macro_f1")
            self.assertNotIn("test/mae_um", metric.options)
            self.assertEqual([row.value for row in at.code], ["cccccccc3333"])   # Detail follows the experiment.
            table = next(item.value for item in at.dataframe if "Run 이름" in item.value)   # Generic default table.
            self.assertEqual(list(table["Run"]), ["cccccccc"])


    def test_metric_help_updates_and_unknown_definitions_stay_unregistered(self):
        at = self.app()
        self.assertTrue(any(row.value == "기준 측정값 대비 평균 절대오차(µm)" for row in at.caption))
        at.selectbox(key="mf_metric_9_1_50_0").select("custom/score").run()
        self.assertTrue(any(row.value == "상세 정의 미등록" for row in at.caption))
        self.assertEqual(at.selectbox(key="mf_metric_9_1_50_0").value, "custom/score")
        at.selectbox(key="mf_metric_9_1_50_0").select("stability/success_rate").run()
        self.assertTrue(any(row.value == "측정 행 기준 성공 비율(해당 실험에 한함)" for row in at.caption))
        self.assertFalse(at.exception)

    def test_rendered_missing_text_zero_and_link_are_distinct(self):
        import io
        import json
        import pyarrow.ipc as ipc
        self.runs = [make_run("a" * 32, metrics={"custom/score": 0}),
                     make_run("b" * 32)]
        at = self.app()
        element = next(row for row in at.dataframe if "Parameter 요약" in row.value)
        raw = element.value
        self.assertEqual(raw["custom/score"].iloc[0], 0)
        self.assertTrue(raw["custom/score"].isna().iloc[1])
        rendered = ipc.open_stream(io.BytesIO(element.proto.arrow_data.styler.display_values)).read_all().to_pandas()
        self.assertEqual(float(rendered["custom/score"].iloc[0]), 0)
        self.assertEqual(rendered["custom/score"].iloc[1], "-")                        # Not logged -> "-".
        self.assertTrue(raw["MLflow 열기"].isna().iloc[0])
        self.assertEqual(raw["MLflow 열기"].iloc[1],
                         provider.DEFAULT_TRACKING_URI + "/#/experiments/9/runs/" + "b" * 32)
        columns = json.loads(element.proto.columns)
        self.assertEqual(columns["MLflow 열기"]["type_config"]["type"], "link")
        self.assertTrue(any("미기록 Run 1개" in row.value for row in at.caption))

    def test_query_failure_never_becomes_missing_data(self):
        with patch("dashboard.components.mlflow.fetch", side_effect=provider.MLflowError("연결 오류")):
            at = self.app()
        self.assertTrue(any("조회 실패" in row.value for row in at.warning))
        self.assertEqual(len(at.dataframe), 0)
        self.assertFalse(any("미기록 Run" in row.value for row in at.caption))


class MetricPresentationTests(unittest.TestCase):
    def test_evaluation_default_priorities_use_available_keys_only(self):
        from dashboard.components.mlflow import _default_metric
        keys = ["train/accuracy", "val/accuracy", "test/accuracy", "test/macro_f1"]
        self.assertEqual(_default_metric(keys, "IDA_Defect_Classification"), "test/macro_f1")
        self.assertEqual(_default_metric(["train/accuracy", "val/recall"], "Classification"), "val/recall")
        self.assertEqual(_default_metric(["test/precision", "test/map50"], "Detection"), "test/map50")
        self.assertEqual(_default_metric(["test/runtime_median_ms", "test/mae_um"], "Width Measurement"), "test/mae_um")
        self.assertEqual(_default_metric(["contest/accuracy", "val/recall"], "unknown"), "val/recall")
        self.assertEqual(_default_metric(["train/accuracy", "eval/score"], "unknown"), "eval/score")
        self.assertEqual(_default_metric(["train/accuracy"], "unknown"), "train/accuracy")
        self.assertIsNone(_default_metric([], "unknown"))

    def test_chart_zero_baseline_hover_and_unmodified_values(self):
        from dashboard.components.mlflow import _comparison_figure
        name = "a very long complete original Run name " * 4
        runs = [make_run("aaaaaaaa1111", metrics={"val/accuracy": .99123}, name=name),
                make_run("bbbbbbbb2222", metrics={"val/accuracy": .99234}, name=name),
                make_run("cccccccc3333")]
        figure = _comparison_figure(runs, "val/accuracy")
        self.assertEqual(figure.layout.height, 320)
        self.assertEqual(figure.layout.yaxis.rangemode, "tozero")
        self.assertEqual(list(figure.data[0].y), [.99123, .99234])
        self.assertNotIn(name, figure.data[0].x)
        self.assertEqual(figure.data[0].customdata[0][0], name)
        self.assertEqual(figure.data[0].customdata[0][1], "aaaaaaaa1111")
        self.assertIn("%{y", figure.data[0].hovertemplate)
        self.assertIn("%{y", figure.data[0].texttemplate)
        self.assertEqual(len(set(figure.data[0].x)), 2)
        self.assertIsNone(_comparison_figure(runs, "missing"))

    def test_negative_values_are_not_clipped_or_normalized_and_unknown_guidance(self):
        from dashboard.components.mlflow import _comparison_figure, _comparison_guidance
        figure = _comparison_figure([make_run("a", metrics={"custom/error": -.02})], "custom/error")
        self.assertEqual(list(figure.data[0].y), [-.02])
        self.assertEqual(figure.layout.yaxis.rangemode, "tozero")
        self.assertIn("지표 정의", _comparison_guidance("unknown", "custom/error"))
        self.assertIn("낮을수록", _comparison_guidance("IDA_Width_Measurement", "test/mae_um"))


    def test_definitions_are_scoped_to_verified_experiments(self):
        from dashboard.components.mlflow import _metric_description
        self.assertEqual(_metric_description("IDA_Defect_Classification", "train/accuracy"),
                         "학습 데이터에서 기록된 정확도")
        self.assertIn("최고 Macro F1", _metric_description("IDA_Defect_Classification", "val/best_macro_f1"))
        self.assertEqual(_metric_description("other", "stability/success_rate"), "상세 정의 미등록")
        self.assertEqual(_metric_description("IDA_Defect_Classification", "new_metric"), "상세 정의 미등록")

    def test_run_urls_allow_only_verified_server_and_actual_safe_identifiers(self):
        from dashboard.components.mlflow import _mlflow_run_url
        from types import SimpleNamespace
        run = make_run("a" * 32, exp="37")
        config = provider.Connection(username="qa", password="not-in-url", token="private-test-token")
        expected = provider.DEFAULT_TRACKING_URI + "/#/experiments/37/runs/" + "a" * 32
        self.assertEqual(_mlflow_run_url(config, run), expected)
        self.assertNotIn("not-in-url", expected)
        self.assertNotIn("private-test-token", expected)
        for uri in ("https://example.com/mlflow", provider.DEFAULT_TRACKING_URI + "?token=private",
                    "javascript:alert(1)", "https://user:pass@dagshub.com/example.mlflow"):
            self.assertIsNone(_mlflow_run_url(SimpleNamespace(tracking_uri=uri), run))
        for bad_run in (make_run("../escape", exp="37"), make_run("a" * 32, exp="37?token=secret"),
                        make_run("a" * 32, exp="../2")):
            self.assertIsNone(_mlflow_run_url(config, bad_run))
        self.assertIsNone(_mlflow_run_url(None, run))

    def test_invalid_recorded_value_is_not_absent_and_graph_data_unchanged(self):
        from dashboard.components.mlflow import _comparison_display, _metric_counts, _comparison_figure
        runs = [make_run("a" * 32, metrics={"score": 0}),
                make_run("b" * 32),
                make_run("c" * 32, metrics={"score": float("nan")})]
        self.assertEqual(_metric_counts(runs, "score"), (1, 1, 1))
        displayed = _comparison_display(runs, ["score"], provider.Connection())
        values = displayed._translate(False, False)["body"]
        texts = [[cell.get("display_value") for cell in row] for row in values]
        self.assertIn("-", texts[1])
        self.assertIn("유효하지 않은 값", texts[2])
        self.assertEqual(list(_comparison_figure(runs, "score").data[0].y), [0])


if __name__ == "__main__":
    unittest.main()
