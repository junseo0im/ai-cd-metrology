import streamlit as st
from dashboard.components.layout import page_title,placeholder,detail_navigation

page_title("AI 활용","향후 AI 적용 전후의 작업 흐름과 활용 도구를 정리할 공간입니다.","AI / PLACEHOLDER")
before_after,tools=st.tabs(["Before & After AI","AI Tool"])
with before_after:
    placeholder("AI 적용 전후 작업 흐름",
                "기존 작업과 AI 적용 이후의 절차·역할·개선 효과를 기록할 공간입니다. 실제 결과나 개선 수치는 아직 등록하지 않았습니다.")
with tools:
    placeholder("활용 도구 및 역할",
                "사용 도구, 적용 목적, 담당 역할과 사용 범위를 기록할 공간입니다. 현재 활용 결과는 미확정입니다.")
detail_navigation()
