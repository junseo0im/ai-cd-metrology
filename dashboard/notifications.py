"""Session notifications: Toast once per event + bell history.

Session events live only in this browser session (no new DB table). The bell also lists persisted
analysis runs read from SQLite, labelled separately, so session events never pose as permanent records.
Callers must only notify after the outcome is confirmed (e.g. after a successful DB commit).
"""
from datetime import datetime
import sqlite3
import streamlit as st

EVENTS = "_notifications"
PENDING = "_pending_toasts"
KEYS = "_notification_keys"
ICONS = {"success": "✅", "warning": "⚠️", "error": "❌", "info": "ℹ️"}
KINDS = ("업로드 완료", "중복 감지", "SQLite 등록 완료", "분석 시작", "분석 완료", "부분 실패", "SQLite 저장 실패")


def notify(kind, target, status, *, count=None, level="info", key=None, detail=""):
    """Record one event. The same `key` is recorded (and toasted) only once per session."""
    state = st.session_state
    keys = state.setdefault(KEYS, set())
    dedupe = key or (kind, target, status, count)
    if dedupe in keys:
        return False
    keys.add(dedupe)
    event = {"time": datetime.now().strftime("%H:%M:%S"), "kind": kind, "target": target, "status": status,
             "count": count, "level": level, "detail": detail, "read": False}
    state.setdefault(EVENTS, []).insert(0, event)
    state.setdefault(PENDING, []).append(event)
    return True


def flush_toasts():
    for event in st.session_state.pop(PENDING, []):
        count = f" · {event['count']}장" if event["count"] is not None else ""
        st.toast(f"{event['kind']} · {event['target']} · {event['status']}{count}", icon=ICONS.get(event["level"]))


def events():
    return list(st.session_state.get(EVENTS, []))


def unread():
    return sum(not event["read"] for event in st.session_state.get(EVENTS, []))


def mark_all_read():
    for event in st.session_state.get(EVENTS, []):
        event["read"] = True


def _persisted_runs(limit=5):
    from dashboard.sqlite_provider import runtime_provider
    try:
        return runtime_provider().recent_analysis_runs(limit), None
    except (sqlite3.Error, OSError, RuntimeError, ValueError) as exc:
        return None, str(exc)


def render_bell():
    count = unread()
    with st.popover("🔔" + (f" {count}" if count else ""), help="알림", width="stretch"):
        st.markdown("**이번 세션 알림**")
        rows = events()
        if rows:
            for e in rows[:20]:
                amount = f" · {e['count']}장" if e["count"] is not None else ""
                mark = "" if e["read"] else "● "
                st.markdown(f"{mark}`{e['time']}` {ICONS.get(e['level'], '')} **{e['kind']}** · {e['target']} · "
                            f"{e['status']}{amount}")
            if count and st.button("모두 읽음", key="notifications_read_all"):
                mark_all_read()
                st.rerun()
        else:
            st.caption("이번 세션의 알림이 없습니다.")
        st.caption("세션 알림은 이 브라우저 세션에만 유지되며 새로고침·재접속 시 사라집니다.")
        st.markdown("**최근 분석 실행 (SQLite 저장 기록)**")
        runs, error = _persisted_runs()
        if error:
            st.error("분석 실행 기록을 조회할 수 없습니다. " + error)
        elif runs:
            for r in runs:
                st.markdown(f"`{r['started_at'][:16].replace('T', ' ')} UTC` **{r['execution_status']}** · "
                            f"{r['wafer_id']} · {r['requested_images']}장 · 실패 {r['failed_images']}장",
                            help="Batch ID: " + r["batch_id"] + " · Run: " + r["batch_run_id"])
        else:
            st.caption("저장된 분석 실행 기록이 없습니다.")
