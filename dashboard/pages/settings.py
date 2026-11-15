import streamlit as st
from dashboard.components.tables import data_table
from dashboard import data_provider as data
from dashboard.components.layout import page_title,placeholder

page_title("설정","플랫폼 설정을 위한 화면 골격입니다.","COMMON / SETTINGS")
with st.container(border=True):
    st.subheader("현재 모드")
    st.toggle("MOCK DATA 표시",value=True,disabled=True)
    st.caption("Phase 1에서는 Mock 표시를 끌 수 없습니다.")
    data_table(data.catalog()["connections"],hide_index=True,width="stretch")
    placeholder("연결 설정 · Placeholder","접속 정보·인증 토큰·DB 설정을 입력하거나 저장하지 않습니다. 실제 API 연결은 다음 단계 범위입니다.")
