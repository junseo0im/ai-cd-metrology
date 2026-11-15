import streamlit as st
from dashboard import data_provider as data
from dashboard.components.layout import page_title

page_title("전체 Dataset","bundled Mock catalog만 표시합니다. 실제 data/ 폴더는 탐색하지 않습니다.","COMMON / DATASET")
st.info("이 목록은 예시입니다. 실제 데이터셋 inventory·split·annotation과 연결되어 있지 않습니다.")
mode = st.selectbox("모드 필터",["전체","Wafer Batch","Repeatability","Lighting"])
frame = data.images(mode=None if mode=="전체" else mode)
st.caption(str(len(frame))+" records (예시)")
st.dataframe(frame[[*data.IDENTITY_FIELDS,"mode","status"]],hide_index=True,width="stretch")
