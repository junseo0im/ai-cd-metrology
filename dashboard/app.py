"""Single entrypoint: python -m streamlit run dashboard/app.py"""
from pathlib import Path
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from dashboard.components.layout import header
from dashboard.components.sidebar import render
from dashboard.navigation import make_pages,PAGE_SPECS,apply_batch_link,prepare_page_context

st.set_page_config(page_title="IDA Inspection Platform",page_icon="🔬",
                   layout="wide",initial_sidebar_state="expanded")
pages = make_pages()
selected = st.navigation(list(pages.values()),position="hidden")
active = next(key for key,page in pages.items() if page is selected)
st.session_state["active_page_key"] = active
apply_batch_link(active)
prepare_page_context(active)
header(PAGE_SPECS[active][1])
render(pages,active)
from dashboard.components.waepi import render_waepi
render_waepi()  # TASK18-B-17: floating '웨피' (before the page body: pages that st.stop() keep it).
selected.run()
from dashboard.components.waepi import sync as waepi_sync
if st.session_state.get("_waepi_slot") is not None:
    waepi_sync()  # TASK18-B-18: state changed by this page run is shown at once (no extra rerun).
from dashboard.notifications import flush_toasts
flush_toasts()  # Events raised by the page itself in this run.
