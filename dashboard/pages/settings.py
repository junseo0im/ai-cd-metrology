import streamlit as st
from dashboard.components.tables import data_table
from dashboard import data_provider as data
from dashboard.components.layout import page_title,placeholder
from dashboard.components.auth_ui import account_panel

page_title("설정","현재 계정과 기록표 수정 권한을 확인합니다.","COMMON / SETTINGS")
with st.container(border=True):
    account_panel()

with st.container(border=True):
    st.subheader("사용자 및 권한")
    st.table([
        {"기능":"대시보드 조회","ADMIN":"가능","GUEST":"가능"},
        {"기능":"실험 환경 및 조건 조회","ADMIN":"가능","GUEST":"가능"},
        {"기능":"실험 환경 및 조건 편집·저장","ADMIN":"가능","GUEST":"불가"},
        {"기능":"Critical 기준 조회","ADMIN":"가능","GUEST":"가능"},
        {"기능":"Critical 기준 편집·저장","ADMIN":"가능","GUEST":"불가"},
    ])
    st.caption("ADMIN: 단일 계정 · GUEST: 로그인 없이 조회 · MEMBER: 향후 확장 예정")
    st.caption("기존 이미지 등록 등 다른 기능의 권한 정책은 변경하지 않습니다.")

with st.container(border=True):
    st.subheader("시스템 연결 상태")
    st.caption("아래는 기존 연결 계획용 Placeholder이며 실시간 연결 검사 결과가 아닙니다.")
    data_table(data.catalog()["connections"],hide_index=True,width="stretch")
    placeholder("연결 설정 · Placeholder","계측·YOLO / MLflow / Roboflow API 연결은 이번 단계에서 구현하지 않습니다.")
