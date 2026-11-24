"""Offline transport/normalization tests. No live MLflow writes or model imports."""
import io
import json
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import Mock, patch
from dashboard.mlflow_provider import (
    Connection, MLflowError, ReadonlyClient, connection_from_settings,
    metric_value, recorded_time, run_labels, metric_note,
)


def response(value):
    result = Mock()
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.read.return_value = json.dumps(value).encode()
    return result


class ReadonlyMLflowTests(unittest.TestCase):
    def client(self, **kwargs):
        return ReadonlyClient(Connection(**kwargs))

    def test_config_environment_precedes_secrets_and_default_is_existing_server(self):
        config = connection_from_settings({"mlflow": {"tracking_uri": "https://server.example/mlflow",
                                                     "token": "secret-test-token"}}, {})
        self.assertEqual(config.tracking_uri, "https://server.example/mlflow")
        self.assertNotIn("secret-test-token", repr(config))
        config = connection_from_settings({"mlflow": {"tracking_uri": "https://other.example"}},
                                          {"MLFLOW_TRACKING_URI": "https://env.example"})
        self.assertEqual(config.tracking_uri, "https://env.example")
        self.assertIn("mnitocea/IDA-Defect-AI-MLflow.mlflow", connection_from_settings({}, {}).tracking_uri)

    def test_unsafe_uri_and_partial_credentials_fail_without_echoing_values(self):
        for uri in ("https://user:secret@example.com", "https://example.com/?token=secret",
                    "file:///tmp/server", "http://public.example"):
            with self.assertRaises(MLflowError) as error:
                Connection(uri)
            self.assertNotIn(uri, str(error.exception))
        with self.assertRaises(MLflowError):
            Connection(username="only-user")
        self.assertEqual(Connection("http://127.0.0.1:5000").tracking_uri, "http://127.0.0.1:5000")

    @patch("dashboard.mlflow_provider.build_opener")
    def test_experiment_pagination_and_run_query_are_read_only(self, build):
        build.return_value.open.side_effect = [
            response({"experiments": [{"experiment_id": "9", "name": "New"}], "next_page_token": "page-two"}),
            response({"experiments": [{"experiment_id": "3", "name": "Existing"}]}),
            response({"runs": []}),
        ]
        client = self.client()
        self.assertEqual(len(client.experiments()), 2)
        client.runs("9", 25, "run-next")
        requests = [call.args[0] for call in build.return_value.open.call_args_list]
        self.assertTrue(all(request.method == "POST" for request in requests))
        self.assertEqual(json.loads(requests[1].data)["page_token"], "page-two")
        self.assertEqual(json.loads(requests[2].data)["experiment_ids"], ["9"])
        self.assertEqual(json.loads(requests[2].data)["page_token"], "run-next")
        self.assertEqual(json.loads(requests[2].data)["order_by"], ["attributes.start_time DESC"])
        self.assertEqual(json.loads(requests[0].data)["view_type"], "ACTIVE_ONLY")

    @patch("dashboard.mlflow_provider.build_opener")
    def test_run_and_artifact_get_no_download_and_auth_server_side(self, build):
        build.return_value.open.side_effect = [response({"run": {"info": {"run_id": "r"}}}),
                                               response({"files": []})]
        client = self.client(token="test-secret-token")
        self.assertEqual(client.run("r")["info"]["run_id"], "r")
        client.artifacts("r", "results", "more")
        for call in build.return_value.open.call_args_list:
            request = call.args[0]
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-secret-token")
            self.assertNotIn("test-secret-token", request.full_url)
            self.assertNotIn("download", request.full_url)
        self.assertIn("page_token=more", build.return_value.open.call_args.args[0].full_url)
        with self.assertRaises(MLflowError):
            client._request("runs/create", {})

    @patch("dashboard.mlflow_provider.build_opener")
    def test_basic_auth_is_header_only(self, build):
        build.return_value.open.return_value = response({"files": []})
        self.client(username="qa", password="never-log-this").artifacts("r")
        request = build.return_value.open.call_args.args[0]
        self.assertTrue(request.get_header("Authorization").startswith("Basic "))
        self.assertNotIn("never-log-this", request.full_url)

    @patch("dashboard.mlflow_provider.build_opener")
    def test_connection_and_http_errors_never_echo_credentials(self, build):
        for error in (
            URLError("test-secret-token"),
            TimeoutError("test-secret-token"),
            HTTPError("https://secret/", 401, "test-secret-token", {}, io.BytesIO(b"test-secret-token")),
            HTTPError("https://secret/", 500, "test-secret-token", {}, io.BytesIO()),
        ):
            build.return_value.open.side_effect = error
            with self.assertRaises(MLflowError) as caught:
                self.client(token="test-secret-token").experiments()
            self.assertNotIn("test-secret-token", str(caught.exception))

    @patch("dashboard.mlflow_provider.build_opener")
    def test_bad_response_and_repeated_page_token_fail_safely(self, build):
        invalid = response({})
        invalid.read.return_value = b"not-json"
        build.return_value.open.return_value = invalid
        with self.assertRaises(MLflowError):
            self.client().experiments()
        build.return_value.open.return_value = response({"experiments": [], "next_page_token": "repeat"})
        with self.assertRaises(MLflowError):
            self.client().experiments()
        build.return_value.open.return_value = response({"error_code": "BAD", "message": "secret"})
        with self.assertRaises(MLflowError):
            self.client().experiments()

    def test_secrets_redacted_from_remote_tags_and_pagination_preserved(self):
        config = Connection(token="test-secret-token", password="another-secret")
        result = config.clean({"next_page_token": "continuation", "token": "hidden",
                               "data": {"tags": [{"key": "api_key", "value": "test-secret-token"}]},
                               "message": "Bearer abc123xyz https://u:p@host/path?access_token=abcdef"})
        self.assertEqual(result["next_page_token"], "continuation")
        self.assertEqual(result["token"], "[비공개]")
        self.assertNotIn("test-secret-token", str(result))
        self.assertNotIn("abc123xyz", str(result))
        self.assertNotIn("u:p", str(result))
        self.assertNotIn("abcdef", str(result))

    def test_sparse_metrics_zero_and_duplicate_names(self):
        a = {"info": {"run_id": "aaaaaaaa1"}, "data": {"metrics": [{"key": "loss", "value": 0}],
                                                       "tags": [{"key": "mlflow.runName", "value": "same"}]}}
        b = {"info": {"run_id": "bbbbbbbb2"}, "data": {"metrics": []}}
        self.assertEqual(metric_value(a, "loss"), 0)
        self.assertIsNone(metric_value(b, "loss"))
        self.assertIsNone(metric_value(a, "other"))
        a["data"]["metrics"][0]["value"] = float("nan")
        self.assertIsNone(metric_value(a, "loss"))
        b["data"]["tags"] = [{"key": "mlflow.runName", "value": "same"}]
        self.assertNotEqual(run_labels([a, b])["aaaaaaaa1"], run_labels([a, b])["bbbbbbbb2"])
        self.assertEqual(recorded_time(None), "미기록")
        self.assertIn("1970-01-01", recorded_time(0))
        self.assertIn("측정 행", metric_note("IDA_Width_Measurement", "stability/success_rate"))
        self.assertIn("미확인", metric_note("different", "stability/success_rate"))


if __name__ == "__main__":
    unittest.main()
