"""Read-only MLflow REST access. No SDK, artifact downloads or model execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import base64
import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener

DEFAULT_TRACKING_URI = "https://dagshub.com/mnitocea/IDA-Defect-AI-MLflow.mlflow"
_SENSITIVE = re.compile(r"password|secret|token|api.?key|authorization|credential", re.I)


class MLflowError(Exception):
    """Only safe, user-facing messages leave the transport boundary."""


@dataclass(frozen=True)
class Connection:
    tracking_uri: str = DEFAULT_TRACKING_URI
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    token: str = field(default="", repr=False)

    def __post_init__(self):
        uri = self.tracking_uri.strip().rstrip("/")
        parts = urlsplit(uri)
        local_http = parts.scheme == "http" and parts.hostname in {"localhost", "127.0.0.1", "::1"}
        if (not parts.hostname or not (parts.scheme == "https" or local_http)
                or parts.username or parts.password or parts.query or parts.fragment):
            raise MLflowError("Tracking URI 설정을 확인하세요. 인증정보·query 없는 HTTPS 주소를 사용하세요.")
        if bool(self.username) != bool(self.password) and not self.token:
            raise MLflowError("MLflow 사용자명과 비밀번호를 모두 설정하거나 Tracking Token을 설정하세요.")
        object.__setattr__(self, "tracking_uri", uri)

    def clean(self, value, key=""):
        if key not in {"next_page_token", "page_token"} and _SENSITIVE.search(str(key)):
            return "[비공개]"
        if isinstance(value, dict):
            return {str(k): ("[비공개]" if k == "value" and _SENSITIVE.search(str(value.get("key", "")))
                             else self.clean(v, k)) for k, v in value.items()}
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if not isinstance(value, str):
            return value
        for secret in (self.password, self.token):
            if secret:
                value = value.replace(secret, "[비공개]")
        value = re.sub(r"(?i)(https?://)[^/\s@]+@", r"\1[비공개]@", value)
        value = re.sub(r"(?i)(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+", r"\1 [비공개]", value)
        return re.sub(r"(?i)([?&](?:token|access_token|api_key|password|signature)=)[^&\s]+",
                      r"\1[비공개]", value)


def connection_from_settings(secrets=None, environ=None):
    """Environment wins; optional Streamlit [mlflow] settings remain server-side."""
    env = os.environ if environ is None else environ
    settings = (secrets or {}).get("mlflow", {})
    def get(env_key, secret_key, default=""):
        value = env.get(env_key, settings.get(secret_key, default))
        return str(value or "")
    return Connection(
        get("MLFLOW_TRACKING_URI", "tracking_uri", DEFAULT_TRACKING_URI),
        get("MLFLOW_TRACKING_USERNAME", "username"),
        get("MLFLOW_TRACKING_PASSWORD", "password"),
        get("MLFLOW_TRACKING_TOKEN", "token"),
    )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward server credentials to a redirect destination.


class ReadonlyClient:
    _ENDPOINTS = {
        "experiments/search": "POST",
        "runs/search": "POST",  # Search POSTs do not create/update records.
        "runs/get": "GET",
        "artifacts/list": "GET",
    }

    def __init__(self, connection):
        self.connection = connection

    def _request(self, endpoint, payload):
        if endpoint not in self._ENDPOINTS:
            raise MLflowError("허용되지 않은 MLflow 작업입니다.")
        method = self._ENDPOINTS[endpoint]
        url = self.connection.tracking_uri + "/api/2.0/mlflow/" + endpoint
        headers = {"Accept": "application/json"}
        if self.connection.token:
            headers["Authorization"] = "Bearer " + self.connection.token
        elif self.connection.username:
            encoded = base64.b64encode(
                (self.connection.username + ":" + self.connection.password).encode()).decode()
            headers["Authorization"] = "Basic " + encoded
        body = None
        if method == "POST":
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        elif payload:
            url += "?" + urlencode(payload)
        try:
            request = Request(url, data=body, headers=headers, method=method)
            with build_opener(_NoRedirect()).open(request, timeout=8) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise MLflowError("MLflow 응답이 너무 큽니다. 조회 Run 수를 줄이세요.")
            result = json.loads(raw)
        except HTTPError as error:
            reason = ("인증 또는 읽기 권한을 확인하세요." if error.code in (401, 403)
                      else "서버 주소·상태를 확인하세요.")
            raise MLflowError(f"MLflow 조회 실패 (HTTP {error.code}). {reason}") from None
        except (URLError, TimeoutError, OSError):
            raise MLflowError("MLflow 연결 실패: 네트워크·서버 주소 또는 응답 시간을 확인하세요.") from None
        except (ValueError, UnicodeError):
            raise MLflowError("MLflow 서버에서 올바른 JSON 응답을 받지 못했습니다.") from None
        if not isinstance(result, dict) or result.get("error_code"):
            raise MLflowError("MLflow 조회 응답을 확인할 수 없습니다. 서버 설정과 읽기 권한을 확인하세요.")
        # Redact before anything is cached or passed to the browser.
        return self.connection.clean(result)

    def experiments(self):
        result, token, seen = [], "", set()
        for _ in range(100):
            payload = {"max_results": 1000, "view_type": "ACTIVE_ONLY"}
            if token:
                payload["page_token"] = token
            page = self._request("experiments/search", payload)
            result.extend(page.get("experiments", []))
            token = page.get("next_page_token", "")
            if not token:
                unique = {str(e["experiment_id"]): e for e in result if "experiment_id" in e}
                return sorted(unique.values(), key=lambda e: (str(e.get("name", "")), str(e["experiment_id"])))
            if token in seen:
                break
            seen.add(token)
        raise MLflowError("실험 목록을 끝까지 조회하지 못했습니다. 서버 페이지 응답을 확인하세요.")

    def runs(self, experiment_id, limit=50, page_token=""):
        if limit not in (25, 50, 100, 200):
            raise MLflowError("지원되지 않는 Run 조회 개수입니다.")
        payload = {"experiment_ids": [str(experiment_id)], "max_results": limit,
                   "order_by": ["attributes.start_time DESC"]}
        if page_token:
            payload["page_token"] = page_token
        return self._request("runs/search", payload)

    def run(self, run_id):
        return self._request("runs/get", {"run_id": run_id}).get("run", {})

    def artifacts(self, run_id, path="", page_token=""):
        payload = {"run_id": run_id}
        if path:
            payload["path"] = path
        if page_token:
            payload["page_token"] = page_token
        return self._request("artifacts/list", payload)


def pairs(run, section):
    return {item["key"]: item.get("value") for item in run.get("data", {}).get(section, [])
            if isinstance(item, dict) and "key" in item}


def metric_value(run, key):
    value = pairs(run, "metrics").get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return value if math.isfinite(value) else None


def run_id(run):
    return str(run.get("info", {}).get("run_id") or run.get("info", {}).get("run_uuid") or "")


def run_name(run):
    return str(pairs(run, "tags").get("mlflow.runName") or run.get("info", {}).get("run_name") or "이름 미기록")


def run_labels(runs):
    initial = {run_id(r): f"{run_name(r)} · {run_id(r)[:8]}" for r in runs}
    return {rid: label if list(initial.values()).count(label) == 1 else label + " · " + rid
            for rid, label in initial.items()}


def recorded_time(value):
    if value is None or isinstance(value, bool):
        return "미기록"
    try:
        return datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc).isoformat(timespec="seconds")
    except (ValueError, OverflowError, OSError, TypeError):
        return "미기록"


def metric_note(experiment_name, key):
    if experiment_name != "IDA_Width_Measurement":
        return "단위·집계 정의 미확인 — 서버에 기록된 Metric Key와 값을 그대로 표시합니다."
    notes = {
        "test/mae_um": "단위 µm · 기록된 Test MAE",
        "test/runtime_median_ms": "단위 ms/장 · 기록된 처리시간 중앙값",
        "stability/repeat_target_sd_um": "단위 µm · 반복 측정 대상별 표준편차",
        "stability/lighting_image_mean_range_um": "단위 µm · 조명 조건의 이미지 평균 측정값 범위",
        "stability/success_rate": "단위 0–1 비율 · 측정 행 기준 성공률이며 이미지 검출 성공률이 아닙니다.",
    }
    return notes.get(key, "단위·집계 정의 미확인 — 기록된 Metric Key를 그대로 표시합니다.")
