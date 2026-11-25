"""Shared presentation. All example values come from data_provider."""
from html import escape
import streamlit as st
from dashboard import data_provider as data
from dashboard.navigation import navigate,PAGE_SPECS
from dashboard.components.theme import apply_theme
from dashboard.components.auth_ui import user_menu

def header(page_title):
    apply_theme()
    with st.container(key="platform_header"):
        brand,search,bell,user = st.columns([5.6,3.1,.55,1.25],gap="small")
        with brand:
            st.markdown('<div class="brand-row"><span class="ida-mark">IDA</span>'
                        '<span class="ida-brand">IDA Inspection Platform</span>'
                        '<span class="ida-separator">|</span><span class="ida-page">'+
                        escape(page_title)+'</span></div>',unsafe_allow_html=True)
        with bell:
            from dashboard.notifications import render_bell
            render_bell()
        with user:
            user_menu()
        # TASK18-B-18: no per-page description line under the platform title (status lives in each page/sidebar).
        try:  # TASK18-B-11: running analysis is visible on every page (Data Control shows it in D).
            from dashboard.components.analysis_management import render_global_progress
            render_global_progress(st.session_state.get("active_page_key"))
        except Exception as exc:  # Never break a page because of the status strip.
            st.caption("분석 진행 상태를 확인할 수 없습니다: "+type(exc).__name__)
    from dashboard.notifications import flush_toasts
    flush_toasts()


def page_title(title,description,kicker="WORKSPACE"):
    with st.container(key="page_heading"):
        st.markdown('<div class="eyebrow">'+escape(kicker)+'</div>',unsafe_allow_html=True)
        st.title(title)
        if description:
            st.caption(description)


def inline_title(title,description):
    """One-line page heading: `Title | description` (no kicker)."""
    with st.container(key="page_heading_inline",horizontal=True,vertical_alignment="bottom",gap="small"):
        st.title(title,anchor=False,width="content")
        if description:
            st.caption("|  "+description,width="content")


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
