import streamlit as st
from dashboard import data_provider as data
from dashboard.navigation import navigate,PAGE_SPECS

# Display-only labels: remove/change here when A/B prefixes are no longer needed.
WORKSPACE_TITLES = {
    "data": "A / DATA WORKSPACE",
    "analysis": "B / ANALYSIS WORKSPACE",
}

intro,environment = st.columns([1.4,1],gap="large",vertical_alignment="center")
with intro:
    st.markdown('<div class="hero"><div class="eyebrow">IDA / INSPECTION PLATFORM</div>'
                '<h1>데이터에서 인사이트까지,<br>하나의 검사 워크스페이스.</h1>'
                '<p>입력 데이터와 검사 결과를 한곳에서 연결합니다.<br>'
                '지금은 화면 구조와 이동을 확인하는 UI Phase 1입니다. 모든 값은 예시입니다.</p></div>',
                unsafe_allow_html=True)
with environment,st.container(border=True,key="card_home_environment"):
    st.subheader("실험 환경 및 조건")
    st.write("실험 장비, 공정 조건, 촬영 환경 및 진행 과정을 확인합니다.")
    if st.button("실험 환경 및 조건 열기",width="stretch"):
        navigate("experiment_environment")
st.divider()
left,right = st.columns(2,gap="large")
with left,st.container(border=True):
    st.markdown("#### "+WORKSPACE_TITLES["data"])
    st.subheader("Data Control Center")
    st.write("데이터 입력부터 Metadata, Calibration, Batch 관리까지.")
    st.caption("예시 입력 화면 · Processing Matrix · 처리 흐름")
    if st.button("Data Control Center 열기",type="primary",width="stretch"):
        navigate("data_control")
with right,st.container(border=True):
    st.markdown("#### "+WORKSPACE_TITLES["analysis"])
    st.subheader("Inspection & Analysis")
    st.write("Wafer Batch, 반복 측정, 조명 비교를 한곳에서.")
    st.caption("Mock Pattern Map · 이미지/Overlay · 결과 리포트")
    if st.button("Inspection & Analysis 열기",type="primary",width="stretch"):
        navigate("inspection_analysis")

st.write("")
st.subheader("시스템 상태 · 예시")
stats = st.columns(4)
stats[0].metric("Batch (예시)",len(data.batches()))
stats[1].metric("Image (예시)",len(data.images()))
stats[2].metric("데이터 소스","Mock only")
stats[3].metric("실제 처리","미연결")
with st.container(border=True):
    st.caption("QUICK ACCESS")
    for col,key in zip(st.columns(3),["dataset","model_performance","settings"]):
        with col:
            st.page_link(PAGE_SPECS[key][0],label=PAGE_SPECS[key][1],icon=PAGE_SPECS[key][3])
st.caption("Database / AI / MLflow / Roboflow 연결 및 모델 실행은 포함하지 않습니다.")
