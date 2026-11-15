import streamlit as st
from dashboard.components.layout import page_title,placeholder
from dashboard.navigation import navigate

page_title("Simulation","향후 공정 조건 입력과 결과 분석을 연결할 Placeholder입니다.","B / SIMULATION")
st.info("UI 전용 · 실제 Simulation 계산, 모델 실행 및 API 연결은 없습니다.")
left,right=st.columns(2)
with left:
    placeholder("공정 조건 입력","입력 항목과 허용 범위는 미확정입니다. 향후 조건 입력 화면을 연결합니다.")
with right:
    placeholder("결과 분석","계산 모델과 결과 형식은 미확정입니다. 현재 결과를 생성하지 않습니다.")
if st.button("Inspection & Analysis로 돌아가기",width="content"):
    navigate("inspection_analysis")
