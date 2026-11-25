"""'웨피' character with the real analysis progress (TASK18-B-17, 1차).

* Rendering: st.components.v2 (Streamlit 1.64) mounts HTML/CSS/JS in the page DOM (shadow-root
  styles). The character is position:fixed at the bottom right; dragging, the saved position
  (localStorage, same browser) and click pass-through are handled entirely in the browser.
* Only opaque character pixels take the pointer; the bubble and transparent areas never block clicks.
* Status = the session-visible job state, read-only: job_snapshot() (processed count, existing
  definition), the save-in-progress mark around save_analysis_result, and this session's own
  finished run (dcc_analysis_last_run["db_save"]). Nothing here starts, stops or saves anything.
* Updates: a 2 s st.fragment (never a full app rerun). The session blocked in its own run gets the
  same text pushed from the existing progress callback.
"""
from base64 import b64encode
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import time

import streamlit as st

ASSET = Path(__file__).resolve().parents[1] / "assets" / "characters" / "waepi.png"
DONE_SECONDS = 120   # A finished run's save outcome stays visible this long in its own session.


@lru_cache(maxsize=1)
def _image_uri():
    """Transparent PNG, cropped to its opaque area and downscaled for the page (original untouched)."""
    try:
        from PIL import Image
        with Image.open(ASSET) as source:
            image = source.convert("RGBA")
            box = image.getchannel("A").getbbox()
            if box:
                image = image.crop(box)
            image.thumbnail((160, 200))
            buffer = BytesIO(); image.save(buffer, format="PNG", optimize=True)
            data = buffer.getvalue()
    except (OSError, ValueError):
        return None
    return "data:image/png;base64," + b64encode(data).decode("ascii")


CSS = """
.waepi-root { position: fixed; right: 20px; bottom: 18px; z-index: 990; pointer-events: none;
  display: flex; flex-direction: column; align-items: flex-end; gap: 4px; user-select: none; }
.waepi-root.flip { flex-direction: column-reverse; }
.waepi-bubble { pointer-events: none; max-width: 240px; padding: 6px 10px; border-radius: 12px;
  font: 600 12.5px/1.35 "Source Sans Pro", "Malgun Gothic", sans-serif;
  background: var(--st-background-color, #fff); color: var(--st-text-color, #172b46);
  border: 1px solid rgba(120, 140, 170, .55); box-shadow: 0 2px 8px rgba(20, 40, 70, .18);
  white-space: normal; word-break: keep-all; overflow-wrap: anywhere; }
@media (max-width: 700px) { .waepi-bubble { max-width: 170px; font-size: 11.5px; } }
.waepi-bubble[data-tone="running"] { border-color: #568ee9; }
.waepi-bubble[data-tone="saving"] { border-color: #e0883e; }
.waepi-bubble[data-tone="done"] { border-color: #35a86b; }
.waepi-bubble[data-tone="error"] { border-color: #e56666; color: #c23b3b; }
.waepi-bar { height: 4px; margin-top: 4px; border-radius: 2px; background: rgba(120, 140, 170, .25); overflow: hidden; }
.waepi-bar > i { display: block; height: 100%; background: #568ee9; }
.waepi-img { width: 95px; height: auto; pointer-events: none; display: block;
  filter: drop-shadow(0 2px 3px rgba(20, 40, 70, .25)); }
.waepi-root { transition: opacity .15s ease; }
.waepi-root.peek { opacity: .25; }   /* Pointer over its area (not on the character): content underneath stays visible. */
.waepi-root.dragging .waepi-img { filter: drop-shadow(0 4px 8px rgba(20, 40, 70, .35)); }
"""

JS = r"""
const GREETING = "안녕하세요 저는 '웨피'입니다. 작업을 시작해주세요";
export default function (component) {
  const { data, parentElement } = component;
  const KEY = "ida_waepi_position";
  let root = parentElement.querySelector(".waepi-root");
  if (!root) {
    root = document.createElement("div");
    root.className = "waepi-root";
    root.innerHTML = '<div class="waepi-bubble" role="status" aria-live="polite"><span></span>' +
                     '<div class="waepi-bar" hidden><i></i></div></div>' +
                     '<img class="waepi-img" alt="웨피" draggable="false">';
    parentElement.appendChild(root);
  }
  const bubble = root.querySelector(".waepi-bubble");
  const img = root.querySelector(".waepi-img");
  const bar = root.querySelector(".waepi-bar");
  // ---- status (every update) ----
  bubble.dataset.tone = (data && data.tone) || "idle";
  bubble.querySelector("span").textContent = (data && data.text) || GREETING;
  const pct = data && typeof data.pct === "number" ? data.pct : null;
  bar.hidden = pct === null;
  if (pct !== null) bar.querySelector("i").style.width = Math.max(0, Math.min(100, pct)) + "%";
  const source = parentElement.querySelector(".waepi-src");
  if (source && img.getAttribute("src") !== source.getAttribute("src")) img.setAttribute("src", source.getAttribute("src"));

  // ---- position: restore + clamp (same browser) ----
  const read = () => { try { return JSON.parse(window.localStorage.getItem(KEY)) || null; } catch (e) { return null; } };
  const write = (p) => { try { window.localStorage.setItem(KEY, JSON.stringify(p)); } catch (e) {} };
  const place = (p) => {
    const w = root.offsetWidth || 90, h = root.offsetHeight || 120;
    const right = Math.max(0, Math.min(window.innerWidth - w, p.right));
    const bottom = Math.max(0, Math.min(window.innerHeight - h, p.bottom));
    root.style.right = right + "px"; root.style.bottom = bottom + "px";
    root.classList.toggle("flip", window.innerHeight - bottom - h < 40 && bottom + h > window.innerHeight * 0.6);
    return { right, bottom };
  };
  if (parentElement.__waepi) {            // Already wired: only the text/progress changed.
    return parentElement.__waepi.cleanup;
  }
  let pos = place(read() || { right: 20, bottom: 18 });
  img.addEventListener("load", () => { pos = place(pos); }, { once: true });

  // ---- hit test on opaque pixels only (transparent areas pass clicks through) ----
  const canvas = document.createElement("canvas");
  let alpha = null;
  const loadAlpha = () => {
    if (!img.naturalWidth) return;
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d"); ctx.drawImage(img, 0, 0);
    try { alpha = ctx.getImageData(0, 0, canvas.width, canvas.height).data; } catch (e) { alpha = null; }
  };
  if (img.complete) loadAlpha(); else img.addEventListener("load", loadAlpha, { once: true });
  const onCharacter = (x, y) => {
    const r = img.getBoundingClientRect();
    if (x < r.left || x > r.right || y < r.top || y > r.bottom || !alpha) return false;
    const px = Math.floor((x - r.left) / r.width * canvas.width);
    const py = Math.floor((y - r.top) / r.height * canvas.height);
    return alpha[(py * canvas.width + px) * 4 + 3] > 40;
  };
  let drag = null, hover = false;
  const down = (e) => {
    if (e.button !== 0 || !onCharacter(e.clientX, e.clientY)) return;
    e.preventDefault(); e.stopPropagation();
    drag = { x: e.clientX, y: e.clientY, right: pos.right, bottom: pos.bottom };
    root.classList.add("dragging"); document.documentElement.style.cursor = "grabbing";
  };
  const move = (e) => {
    if (drag) {
      e.preventDefault();
      pos = place({ right: drag.right - (e.clientX - drag.x), bottom: drag.bottom - (e.clientY - drag.y) });
      return;
    }
    const over = onCharacter(e.clientX, e.clientY);
    if (over !== hover) { hover = over; document.documentElement.style.cursor = over ? "grab" : ""; }
    const box = root.getBoundingClientRect();
    const inside = e.clientX >= box.left && e.clientX <= box.right && e.clientY >= box.top && e.clientY <= box.bottom;
    root.classList.toggle("peek", inside && !over);
  };
  const up = (e) => {
    if (!drag) return;
    e.preventDefault(); e.stopPropagation();
    drag = null; root.classList.remove("dragging");
    document.documentElement.style.cursor = hover ? "grab" : "";
    write(pos);
  };
  const swallowClick = (e) => { if (root.classList.contains("dragging") || onCharacter(e.clientX, e.clientY)) { e.preventDefault(); e.stopPropagation(); } };
  const resize = () => { pos = place(pos); };
  document.addEventListener("pointerdown", down, true);
  window.addEventListener("pointermove", move, true);
  window.addEventListener("pointerup", up, true);
  document.addEventListener("click", swallowClick, true);
  window.addEventListener("resize", resize);
  const cleanup = () => {
    document.removeEventListener("pointerdown", down, true);
    window.removeEventListener("pointermove", move, true);
    window.removeEventListener("pointerup", up, true);
    document.removeEventListener("click", swallowClick, true);
    window.removeEventListener("resize", resize);
    if (hover || drag) document.documentElement.style.cursor = "";
    delete parentElement.__waepi;
  };
  parentElement.__waepi = { cleanup };
  return cleanup;
}
"""

def _register():
    """(Re)register with the current Streamlit runtime; the image is in the HTML (sent once), not the data."""
    return st.components.v2.component(
        "ida_waepi", html=f'<img class="waepi-src" hidden alt="" src="{_image_uri() or ""}">',
        css=CSS + ".waepi-src{display:none}", js=JS, isolate_styles=True)


_COMPONENT = _register()


# ---- status (read-only) --------------------------------------------------------------------------
def progress_text(done, total, failed):
    """Same numbers as the existing progress bar: done/total, round(100*done/total), failed separately."""
    pct = round(100 * done / total) if total else 0
    return f"{done} / {total}장 처리 · {pct}%" + (f" · 실패 {failed}장" if failed else ""), pct


def _finished_state(last):
    """This session's own finished run: the stored save outcome, never inferred from 100 %."""
    from dashboard.components.analysis_management import SAVE_LABELS
    saved = last.get("db_save")
    status = last.get("execution_status")
    if not saved:
        return ("분석 실패" + (" · 저장하지 않음" if status == "FAILED" else ""), "error") if status == "FAILED" else None
    state = saved.get("status")
    if state in ("SAVED", "ALREADY_SAVED"):
        return ("저장 완료" if state == "SAVED" else "저장 완료 · 이미 저장된 실행"), "done"
    return SAVE_LABELS.get(state, str(state)), "error"


def session_state_view():
    """{'text','tone','pct'} for this session; only states this session can already see on existing screens."""
    from dashboard.components import analysis_management as am
    saving = am.saving_in_progress()
    if saving:
        return {"text": "결과 저장 중", "tone": "saving", "pct": None}
    snapshot = am.job_snapshot()
    if snapshot is not None:
        if snapshot["running"]:
            text, pct = progress_text(snapshot["done"], snapshot["total"], snapshot["failed"])
            return {"text": text, "tone": "running", "pct": pct}
        text, pct = progress_text(snapshot["done"], snapshot["total"], snapshot["failed"])
        return {"text": text + " · 분석 종료, 결과 확인 중", "tone": "running", "pct": pct}
    last = st.session_state.get("dcc_analysis_last_run")
    if last:
        key = str(last.get("run_dir") or last.get("image_ids"))
        seen = st.session_state.setdefault("_waepi_seen", {})
        first = seen.setdefault(key, time.time())
        state = _finished_state(last)
        if state and time.time() - first < DONE_SECONDS:
            return {"text": state[0], "tone": state[1], "pct": None}
    flow = _registration_view()
    if flow:
        return flow
    return {"text": GREETING, "tone": "idle", "pct": None}


GREETING = "안녕하세요 저는 '웨피'입니다. 작업을 시작해주세요"
HOLD_SECONDS = 12  # A registration success/failure line stays at least this long (no sleep, timestamp only).


def event(kind):
    """Mark a real UI event of this session: 'registered' | 'register_failed' | 'analysis_started'."""
    st.session_state["_waepi_event"] = {"kind": kind, "time": time.time()}


def _registration_view():
    """Upload/registration states before an analysis is started (session-local, read-only)."""
    happened = st.session_state.get("_waepi_event") or {}
    kind, since = happened.get("kind"), time.time() - happened.get("time", 0)
    bundle = st.session_state.get("sqlite_uploads") or {}
    pending = bool(bundle.get("inputs")) and "registered" not in bundle
    if kind == "register_failed" and since < HOLD_SECONDS:
        return {"text": "DB 등록 실패 · 오류 내용을 확인해 주세요", "tone": "error", "pct": None}
    if pending:                                   # Files selected/dropped, not yet registered.
        return {"text": "이미지 확인 중입니다", "tone": "running", "pct": None}
    if kind == "registered":
        if since < HOLD_SECONDS:
            return {"text": "이미지 DB 등록 완료", "tone": "done", "pct": None}
        return {"text": "분석 실행 전", "tone": "idle", "pct": None}
    return None


def sync():
    """After a full page run: show a changed state at once (the 2 s fragment would otherwise catch up)."""
    try:
        view = session_state_view()
        if view != st.session_state.get("_waepi_last_view"):
            push(view)
    except Exception:
        pass


def _mount(slot, view, key):
    global _COMPONENT
    from streamlit.errors import StreamlitAPIException
    try:
        with slot.container():
            _COMPONENT(data=view, key=key, width="content", height="content")
        st.session_state["_waepi_last_view"] = view
    except StreamlitAPIException as exc:
        if "not registered" not in str(exc):
            raise
        _COMPONENT = _register()   # A new runtime in the same process (e.g. tests) lost the registration.
        with slot.container():
            _COMPONENT(data=view, key=key, width="content", height="content")


@st.fragment(run_every=2)
def _waepi_fragment():
    slot = st.empty()
    st.session_state["_waepi_slot"] = slot          # Lets the session's own blocking run push updates.
    st.session_state["_waepi_push"] = 0
    try:
        _mount(slot, session_state_view(), "ida_waepi")
    except Exception:   # Display only: the character must never break a page.
        st.session_state["_waepi_slot"] = None
        slot.empty()


def enabled():
    """Server diagnostic switch IDA_WAEPI_DISABLED=1 (used for the ON/OFF performance comparison)."""
    import os
    return os.environ.get("IDA_WAEPI_DISABLED") != "1" and _image_uri() is not None


def render_waepi():
    """Mounted once per page run from app.py (before the page body, so st.stop() pages keep it)."""
    if not enabled():
        st.session_state["_waepi_slot"] = None
        return
    with st.container(key="waepi_host"):
        _waepi_fragment()


def push(view):
    """Update from the session's own blocking progress callback (same numbers as its progress bar)."""
    slot = st.session_state.get("_waepi_slot")
    if slot is None:
        return
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is None or getattr(ctx, "current_fragment_id", None):
            return   # Fragment runs refresh the character themselves; only the blocking full run pushes.
    except Exception:
        return
    try:
        count = st.session_state.get("_waepi_push", 0) + 1
        st.session_state["_waepi_push"] = count
        _mount(slot, view, f"ida_waepi_push_{count}")
    except Exception:   # Display only: never break the analysis wait.
        pass
