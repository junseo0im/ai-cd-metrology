"""Shared presentation. All example values come from data_provider."""
from html import escape
import streamlit as st
from dashboard import data_provider as data
from dashboard.navigation import navigate,PAGE_SPECS,page_data_source
from dashboard.components.theme import apply_theme

def header(page_title):
    apply_theme()
    source=page_data_source(st.session_state.get("active_page_key"),st.session_state)
    batch_id = st.session_state.get("selected_batch_id")
    selected = data.batch(batch_id) if batch_id else None
    with st.container(key="platform_header"):
        brand,search,bell,user = st.columns([5.6,3.1,.55,1.25],gap="small")
        with brand:
            st.markdown('<div class="brand-row"><span class="ida-mark">IDA</span>'
                        '<span class="ida-brand">IDA Inspection Platform</span>'
                        '<span class="ida-separator">|</span><span class="ida-page">'+
                        escape(page_title)+'</span></div>',unsafe_allow_html=True)
        search.text_input("Dataset / Batch 검색 (미연결)",placeholder="Dataset / Batch 검색 · 미연결",
                          disabled=True,label_visibility="collapsed",key="header_search")
        bell.button("🔔",disabled=True,help="알림 Mock · 실제 알림 기능 미연결",width="stretch")
        user.button("DEMO USER",disabled=True,help="사용자 메뉴 Mock · 인증 기능 미연결",width="stretch")
        context = ("Batch "+selected["batch_id"]+" · Wafer "+selected["wafer_id"]) if selected else "Batch / Wafer 선택 없음"
        if source=="SQLite":
            st.caption("SQLite 등록·조회 모드 · Calibration·계측·YOLO 미연결 · 등록 완료 ≠ 분석 완료")
        else:
            st.caption(data.catalog()["notice"]+" · "+context)


def page_title(title,description,kicker="WORKSPACE"):
    with st.container(key="page_heading"):
        st.markdown('<div class="eyebrow">'+escape(kicker)+'</div>',unsafe_allow_html=True)
        st.title(title)
        st.caption(description)


def card_title(title,description=""):
    st.subheader(title)
    if description:
        st.caption(description)


def placeholder(title,description):
    st.markdown('<div class="placeholder"><b>'+escape(title)+'</b><p>'+escape(description)+'</p></div>',
                unsafe_allow_html=True)


def detail_navigation():
    """Shared return links for information pages, using the existing routes."""
    st.divider()
    home,data_control,inspection = st.columns(3)
    if home.button("Main Home으로 돌아가기",width="stretch"):
        navigate("main_home")
    data_control.page_link(PAGE_SPECS["data_control"][0],label="Data Control Center",icon=PAGE_SPECS["data_control"][3])
    inspection.page_link(PAGE_SPECS["inspection_analysis"][0],label="Inspection & Analysis",icon=PAGE_SPECS["inspection_analysis"][3])


def select_batch(key,mode=None):
    frame = data.batches(mode)
    options = frame["batch_id"].tolist()
    saved = st.session_state.get("selected_batch_id")
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state[key] = options[0]
    index = None if key in st.session_state else options.index(saved) if saved in options else 0
    def remember_batch():
        st.session_state["selected_batch_id"] = st.session_state[key]
    selected = st.selectbox("Batch 선택 (예시)",options,index=index,key=key,on_change=remember_batch)
    st.session_state["selected_batch_id"] = selected
    return selected


def select_image(batch_id,key):
    frame = data.images(batch_id)
    options = frame["image_id"].tolist()
    pending = st.session_state.pop("_pending_map_image",None)
    if pending in options:
        st.session_state[key] = pending
    elif key in st.session_state and st.session_state[key] not in options:
        st.session_state[key] = options[0]
    saved = st.session_state.get("selected_image_id")
    index = None if key in st.session_state else options.index(saved) if saved in options else 0
    selected = st.selectbox("Image ID (예시)",options,index=index,key=key)
    st.session_state["selected_image_id"] = selected
    return selected


def analysis_modes(current):
    st.caption("분석 모드 · 수동 전환 / 모든 화면은 예시")
    routes = [("Wafer Batch","inspection_analysis"),("Repeatability","repeatability"),("Lighting","lighting")]
    for column,(label,route) in zip(st.columns(3),routes):
        if column.button(label,type="primary" if current==label else "secondary",
                         disabled=current==label,key="mode_"+current+"_"+route,width="stretch"):
            navigate(route)



def metric_summary(batch_id):
    cols = st.columns(3)
    for col,(name,value) in zip(cols,data.measurement_summary(batch_id).items()):
        col.metric(name+" 평균 (예시)",f"{value:.2f}",help="미리 정의한 UI 예시값. 계측/µm 환산을 실행한 값이 아닙니다.")
    st.caption("단위 µm · 정적 Mock 값 / 실제 calibration·계측 없음")
