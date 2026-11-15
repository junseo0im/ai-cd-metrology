"""Native Streamlit theme detection plus shared CSS; no per-page theme overrides."""
from pathlib import Path
import streamlit as st


def apply_theme():
    # Native Streamlit sets color-scheme on .stApp. CSS light-dark() reacts in
    # the browser immediately, including before st.context.theme updates.
    css = (Path(__file__).with_name("styles.css")).read_text(encoding="utf-8")
    st.markdown("<style>"+css+"</style>",unsafe_allow_html=True)


def current_theme():
    return st.context.theme.type or "light"
