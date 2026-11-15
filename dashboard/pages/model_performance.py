import streamlit as st
from dashboard.components.layout import page_title,placeholder

page_title("모델 성능","평가 결과와 model run을 연결할 예정입니다.","COMMON / MODEL")
with st.container(border=True):
    placeholder("모델 성능 · Placeholder","평가 데이터가 연결되지 않았습니다. 실제 Precision / Recall / mAP / F1 수치는 표시하지 않습니다.")
    st.caption("MLflow · 미연결 / Roboflow · 미연결 / 실제 모델 평가 · 미실행")
