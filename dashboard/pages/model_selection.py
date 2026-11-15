import streamlit as st
from dashboard.components.layout import page_title,placeholder,detail_navigation
from dashboard.components.mlflow import render_mlflow

page_title("모델 선정 기준","학습 Run과 선정 근거를 함께 확인할 Placeholder입니다.","MODEL / PLACEHOLDER")
left,right=st.columns(2,gap="large")
with left,st.container(border=True,key="model_selection_mlflow"):
    render_mlflow()
with right,st.container(border=True,key="model_selection_criteria"):
    st.subheader("모델 선정 기준")
    for title in ["모델 후보","선정 지표","성능 비교","선정 이유","최종 모델 및 버전"]:
        with st.expander(title):
            placeholder(title+" · 미확정","추후 확정된 정보를 등록할 공간입니다. 실제 선정 결과는 입력하지 않았습니다.")
detail_navigation()
