"""Existing Header and Settings account controls only."""
import streamlit as st
from dashboard import auth
from dashboard.navigation import navigate


def user_menu():
    user = auth.current_user()
    with st.popover(user["display_name"], width="stretch"):
        st.write("현재 상태: " + user["role"])
        if user["role"] == "ADMIN":
            st.write(user["display_name"])
        elif st.button("관리자 로그인", key="header_login", width="stretch"):
            navigate("settings")
        if st.button("설정 / 사용자 관리", key="header_settings", width="stretch"):
            navigate("settings")
        if user["role"] == "ADMIN":
            st.button("로그아웃", key="header_logout", on_click=auth.logout, width="stretch")


def _submit_login():
    username = st.session_state.pop("admin_login_username", "")
    password = st.session_state.pop("admin_login_password", "")
    try:
        auth.login(username, password)
    except auth.AuthenticationError as exc:
        st.session_state["_ida_login_error"] = str(exc)
    else:
        st.session_state.pop("_ida_login_error", None)


def account_panel():
    user = auth.current_user()
    st.subheader("내 계정")
    st.table([{"항목": "로그인 상태", "내용": "관리자 로그인됨" if user["role"] == "ADMIN" else "로그인하지 않은 방문자"},
              {"항목": "표시 이름", "내용": user["display_name"]},
              {"항목": "사용자 역할", "내용": user["role"]}])
    if user["role"] == "ADMIN":
        st.button("로그아웃", key="settings_logout", on_click=auth.logout)
    elif auth.configured_admin() is None:
        st.info("관리자 계정 미설정 — GUEST 조회 모드입니다. 서버에서 최초 설정이 필요합니다.")
    else:
        with st.form("admin_login", clear_on_submit=True):
            st.text_input("관리자 사용자명", key="admin_login_username", max_chars=128)
            st.text_input("비밀번호", type="password", key="admin_login_password", max_chars=1024)
            st.form_submit_button("로그인", key="admin_login_submit", type="primary", on_click=_submit_login)
        error = st.session_state.pop("_ida_login_error", None)
        if error:
            st.error(error)
