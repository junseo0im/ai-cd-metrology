import streamlit as st
from dashboard.components.layout import page_title,select_batch,select_image
from dashboard.components.inspection_wafer import inspection_wafer,reset_selection
from dashboard.components.images import image_panel

page_title("Wafer Map","Pattern 위치와 예시 이미지의 연결을 확인합니다. 실제 wafer geometry는 아닙니다.","B / MAP")
batch_id = select_batch("map_batch","Wafer Batch")
left,right = st.columns([1.3,1])
with left,st.container(border=True):
    inspection_wafer(batch_id,image_key="map_image")
    st.button("전체 보기 / 선택 초기화",key="map_reset",on_click=reset_selection,args=(batch_id,))
    st.caption("Mock 위치 표현 · 각 Die의 lt/c/rb는 기존 Die 공용 예시 이미지에 연결됩니다.")
with right,st.container(border=True):
    image_id = select_image(batch_id,"map_image")
    image_panel(image_id)
    st.info("실제 wafer map 생성·수율 계산·공정 좌표 변환은 미구현입니다.")
