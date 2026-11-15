import streamlit as st
from dashboard import data_provider as data
from dashboard.components.layout import page_title,select_batch,select_image,placeholder
from dashboard.components.images import image_panel

page_title("이미지 분석","선택한 예시 이미지와 Overlay 및 identity를 확인합니다.","B / IMAGE")
batch_id = select_batch("image_batch")
image_id = select_image(batch_id,"detail_image")
left,right = st.columns(2)
with left,st.container(border=True):
    st.subheader("원본 영역 예시")
    image_panel(image_id)
with right,st.container(border=True):
    st.subheader("Overlay 예시")
    image_panel(image_id,overlay=True)
with st.container(border=True):
    placeholder("분석 실행 · Placeholder","실제 metrology / YOLO 추론 / calibration / 전처리 / defect 판정 기능은 연결하지 않았습니다.")
    st.caption("현재 이미지 ID 연결 예시")
    row = data.image(image_id)
    st.json({key:row[key] for key in data.IDENTITY_FIELDS})
