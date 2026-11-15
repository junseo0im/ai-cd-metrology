import streamlit as st
from dashboard.components.layout import page_title,placeholder,detail_navigation
from dashboard.components.calibration import render_calibration

page_title("실험 환경 및 조건",
           "실험 장비, 공정 조건, 촬영 환경 및 진행 과정을 확인합니다.", "EXPERIMENT / PLACEHOLDER")
st.info("내용 미확정 · 실제 실험값, 장비 사양 및 결과는 아직 등록하지 않았습니다.")
sections = [
    ("장비 사양","확정된 장비 사양·촬영 환경·실험 기록·표·이미지를 등록할 공간입니다."),
    ("Wafer / Die / Pattern 구조","Wafer·Die·Pattern 계층과 촬영 위치의 관계를 기록할 공간입니다. UI Mock 배치는 실제 좌표 정보가 아닙니다."),
    ("공정 조건","확정된 공정 조건과 실험 과정의 기록·표·참고 이미지를 등록할 공간입니다."),
    ("Mask 설계 / 제작값","Mask 설계치와 제작 확인값을 기록할 공간입니다. 현재 기준값은 미연결입니다."),
    ("오차율 계산 기준(허용값)","오차율 정의와 허용값을 기록할 공간입니다. 현재 기준값·판정은 미확정이며 계산하지 않습니다."),
    ("Calibration",None),
]
for tab,(title,description) in zip(st.tabs([title for title,_ in sections]),sections):
    with tab:
        if title=="Calibration":
            render_calibration()
        else:
            placeholder(title+" · Placeholder",description)
detail_navigation()
