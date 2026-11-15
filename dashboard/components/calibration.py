"""One read-only Calibration body shared by its page and experiment tab."""
import streamlit as st
from dashboard import data_provider as data
from dashboard.components.layout import placeholder


def render_calibration():
    with st.container(border=True):
        placeholder("Calibration 상세 · Placeholder","Ruler 등록·scale 검증·승인 기능은 미구현입니다. 현재 calibration metadata는 예시이며 µm/px는 비어 있습니다.")
        st.dataframe(data.calibration_records(),hide_index=True,width="stretch")
        st.caption("이 페이지에서 scale 저장·변경·환산을 수행하지 않습니다.")
