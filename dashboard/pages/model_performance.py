"""Model-performance records and the teammate's RF-DETR defect inspection.

The inspection UI runs in this Streamlit process (dashboard/defect_inspection);
the teammate's Gradio app is still shown by URL when it happens to be running.
Neither writes its experiment results to the operational DB.
"""
import os
from pathlib import Path
import tomllib
import urllib.request
import streamlit as st
from dashboard.components.layout import page_title
from dashboard.components.mlflow import render_mlflow_browser

SETTINGS = Path(__file__).resolve().parents[1] / "roboflow_gradio.toml"
RUN_COMMAND = ('$env:MODEL_CACHE_DIR="runtime_data\\envs\\roboflow_cache"\n'
               "runtime_data\\envs\\roboflow_gradio\\Scripts\\python.exe dashboard\\database\\roboflow_app\\app.py")


def gradio_url():
    """IDA_ROBOFLOW_GRADIO_URL (empty = off) first, then roboflow_gradio.toml; None when not configured."""
    if "IDA_ROBOFLOW_GRADIO_URL" in os.environ:
        return os.environ["IDA_ROBOFLOW_GRADIO_URL"].strip() or None
    try:
        with SETTINGS.open("rb") as handle:
            return (tomllib.load(handle).get("url") or "").strip() or None
    except (OSError, ValueError):
        return None


def gradio_running(url):
    try:
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


page_title("모델 성능", "실제 MLflow 실험 기록 조회 · Roboflow 실험 앱", "COMMON / MODEL")
mlflow_tab, roboflow_tab = st.tabs(["MLflow", "Roboflow"])
with mlflow_tab:
    render_mlflow_browser()

with roboflow_tab:
    url = gradio_url()
    if url and gradio_running(url):
        st.caption("팀원 RF-DETR 실험 앱 (Gradio · 로컬 " + url + ") · 실험 결과는 운영 DB에 저장되지 않습니다.")
        st.components.v1.iframe(url, height=1350, scrolling=True)
    else:
        with st.container(border=True):
            st.subheader("RF-DETR 결함 검출 시연")
            try:  # Teammate UI (dashboard/defect_inspection); imported here so MLflow never breaks with it.
                from dashboard.defect_inspection.ui import render_inspection
            except ImportError as exc:
                st.error("결함 검사 UI 의존성이 이 환경에 설치되지 않았습니다 (" + (exc.name or str(exc)) + ") · "
                         "dashboard/requirements-inspection.txt")
            else:
                render_inspection()
        with st.expander("별도 Gradio 실험 앱으로 실행하기 (선택)"):
            st.caption("연결 준비 중 · " + ("Gradio 실험 앱이 실행 중이 아닙니다 (" + url + ")" if url
                                        else "Gradio 주소가 설정되지 않았습니다"))
            # Markdown code block, not st.code: the page's st.code elements stay the MLflow Run IDs.
            st.markdown("```powershell" + chr(10) + RUN_COMMAND + chr(10) + "```")
            st.caption("주소 설정: dashboard/roboflow_gradio.toml 의 url (또는 환경변수 IDA_ROBOFLOW_GRADIO_URL). "
                       "API 키는 roboflow_app/.env 또는 환경변수 ROBOFLOW_API_KEY에서 읽습니다.")
