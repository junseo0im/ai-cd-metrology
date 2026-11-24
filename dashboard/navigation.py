"""Routes only; no processing, network or dataset dependencies."""
import streamlit as st

PAGE_SPECS = {
    "main_home":("pages/main_home.py","Main Home","home",":material/home:"),
    "data_control":("pages/data_control.py","Data Control Center","data-home",":material/database:"),
    "inspection_analysis":("pages/inspection_analysis.py","Inspection & Analysis","inspection-home",":material/analytics:"),
    "dataset":("pages/dataset.py","전체 Dataset","datasets",":material/photo_library:"),
    "model_performance":("pages/model_performance.py","모델 성능","models",":material/monitoring:"),
    "settings":("pages/settings.py","설정","platform-settings",":material/settings:"),
    "calibration":("pages/calibration.py","Calibration","data-calibration",":material/straighten:"),
    "batch_detail":("pages/batch_detail.py","Batch 상세","data-batch-detail",":material/grid_view:"),
    "wafer_detail":("pages/wafer_detail.py","Wafer 상세","data-wafer-detail",":material/blur_circular:"),
    "wafer_map":("pages/wafer_map.py","Wafer Map","wafer-map",":material/grid_on:"),
    "image_detail":("pages/image_detail.py","Image 상세","data-image-detail",":material/image:"),
    "image_analysis":("pages/image_analysis.py","이미지 분석","image-analysis",":material/image_search:"),
    "results_report":("pages/results_report.py","결과 리포트","reports",":material/description:"),
    "repeatability":("pages/repeatability.py","Repeatability","inspection-repeatability",":material/repeat:"),
    "lighting":("pages/lighting.py","Lighting","inspection-lighting",":material/light_mode:"),
    "experiment_environment":("pages/experiment_environment.py","실험 환경 및 조건","experiment-environment",":material/science:"),
    "simulation":("pages/simulation.py","Simulation","simulation-preview",":material/science:"),
    "critical_criteria":("pages/critical_criteria.py","Critical 기준","critical-criteria",":material/rule:"),
    "ai_usage":("pages/ai_usage.py","AI 활용","ai-usage",":material/psychology:"),
    "model_selection":("pages/model_selection.py","모델 선정 기준","model-selection-criteria",":material/fact_check:"),
}
A_PAGES = {"data_control","calibration","batch_detail","wafer_detail","image_detail"}
B_PAGES = {"inspection_analysis","wafer_map","image_analysis","results_report","repeatability","lighting","simulation"}
COMMON = ["main_home","data_control","inspection_analysis","dataset","model_performance",
          "experiment_environment","critical_criteria","ai_usage","model_selection","settings"]


def make_pages():
    return {key:st.Page(path,title=title,url_path=url,icon=icon,default=key=="main_home")
            for key,(path,title,url,icon) in PAGE_SPECS.items()}


def navigate(key,**context):
    st.session_state.update(context)
    st.switch_page(PAGE_SPECS[key][0])

def apply_batch_link(active):
    """Consume a valid table link before header/widgets; do not keep a stale URL override."""
    if active != "batch_detail" or "batch_id" not in st.query_params:
        return
    from dashboard import data_provider as data
    requested = st.query_params["batch_id"]
    del st.query_params["batch_id"]
    if data.batch(requested) is None:
        st.session_state["invalid_batch_link"] = requested
        return
    st.session_state.pop("invalid_batch_link",None)
    st.session_state["batch_detail_source"] = "Mock"
    st.session_state["selected_batch_id"] = requested
    st.session_state["detail_batch"] = requested


def prepare_page_context(active):
    """Set the visible page's default/context Batch before the shared Header."""
    from dashboard import data_provider as data
    mode = "Repeatability" if active == "repeatability" else "Lighting" if active == "lighting" else None
    if mode:
        st.session_state["selected_batch_id"] = data.batches(mode).iloc[0]["batch_id"]
    elif active in {"inspection_analysis","wafer_map","batch_detail","image_analysis"}:
        options = data.batches("Wafer Batch" if active in {"inspection_analysis","wafer_map"} else None)["batch_id"].tolist()
        if st.session_state.get("selected_batch_id") not in options:
            st.session_state["selected_batch_id"] = options[0]


def page_data_source(active,state):
    """Registration source is local to its pages, never inherited by Mock analysis."""
    if active in {"experiment_environment","critical_criteria"}:
        return "Records"
    if active in {"inspection_analysis","wafer_map"}:
        return state.get("_wafer_source","SQLite")  # Operational Wafer Map: SQLite (no selector).
    if active in {"image_detail","wafer_detail","main_home","dataset","results_report"}:  # dataset: SQLite (TASK18-B-12A); results_report (I&A 리포트)
        return "SQLite"
    if active=="data_control":
        # Operational default: SQLite. Mock only when explicitly selected (tests / internal use).
        return state.get("data_control_selected_source","SQLite")
    if active=="batch_detail":
        return state.get("batch_detail_source","Mock")
    return "Mock"
