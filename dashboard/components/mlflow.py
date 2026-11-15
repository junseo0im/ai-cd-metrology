"""Shared MLflow display boundary. No API client, metrics or model execution."""
from dashboard.components.layout import placeholder


def render_mlflow():
    placeholder("MLflow · 미연결",
                "최근 Run · 모델 버전 · 모델 비교 · Precision / Recall / mAP 표시 예정입니다. 실제 API 호출이나 성능지표는 없습니다.")
