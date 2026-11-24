"""Shared tables and clickable Batch ID cells; no duplicate controls below the table."""
from urllib.parse import urlencode
import streamlit as st
from dashboard.navigation import PAGE_SPECS,navigate


def data_table(frame,**kwargs):
    settings = dict(hide_index=True,width="stretch",row_height=31)
    settings.update(kwargs)
    return st.dataframe(frame,**settings)


def batch_link(batch_id):
    route = PAGE_SPECS["batch_detail"][2]
    return "./"+route+"?"+urlencode({"batch_id":batch_id})


BATCH_HEADER_PX=38
BATCH_ROW_PX=42


def batch_table(frame, *, source="Mock", analysed=None, visible_rows=None):
    """All Batches in one bounded scroll area; one widget per cell, no stacked captions.

    `analysed`: {batch_id: images with a stored (latest) analysis result} from SQLite; None for Mock.
    """
    is_mock=source=="Mock"
    analysed=analysed or {}
    if "created_at" in frame:
        frame=frame.sort_values("created_at",ascending=False,kind="stable")
    if is_mock:
        headings=["Batch","Wafer","유형","이미지","Role","노광","등록 일시","상태"]
        widths=[1.8,.65,1,.6,.85,.9,1.35,.9]
    else:
        headings=["Batch","Wafer","노광","PEB","이미지","분석","상태"]
        widths=[2.3,.6,.7,.65,.6,.75,.9]
    modes={"wafer_batch":"Wafer","repeatability":"반복촬영","lighting":"밝기","single_image":"단일"}
    roles={"evaluation":"EVAL","training_candidate":"TRAIN","development":"DEV"}
    expanded=st.session_state.get("dcc_expanded_card")=="batches"
    # TASK18-B-10A.1: every Batch is rendered (newest first) inside a scroll area about `visible_rows` tall;
    # wheel and scrollbar drag scroll it. The expanded card keeps its taller full-list view.
    height=420 if expanded else (BATCH_HEADER_PX+visible_rows*BATCH_ROW_PX if visible_rows else 250)
    with st.container(height=height,border=False,key="batch_table"):
        for col,heading in zip(st.columns(widths,gap="small"),headings):
            col.markdown("**"+heading+"**")
        for item in frame.to_dict("records"):
            cols=st.columns(widths,gap="small",vertical_alignment="center")
            identity=item["batch_id"]
            full_label=identity if is_mock else item["display_name"]
            name=item.get("batch_name","") if not is_mock else ""
            short_name=name if len(name)<=24 else name[:20]+"…"
            label=item["automatic_display_name"]+" · "+short_name if name else full_label
            key=("batch_cell_" if is_mock else "sqlite_batch_")+identity
            help_text="Batch ID: "+identity+" · "+full_label
            if not is_mock:
                mode=item["upload_mode"]
                help_text+=(f" · 유형 {modes.get(mode,mode)} · Role {roles.get(item['dataset_role'],item['dataset_role'])}"
                            f" · 등록 {item['created_at'][:16].replace('T',' ')}")
            if cols[0].button(label,key=key,type="tertiary",help=help_text):
                if is_mock:
                    navigate("batch_detail",selected_batch_id=identity,detail_batch=identity,batch_detail_source="Mock")
                else:
                    navigate("batch_detail",selected_sqlite_batch_id=identity,batch_detail_source="SQLite")
            multi=not is_mock and int(item.get("wafer_count",1) or 1)>1
            # v6: a Batch may hold several Wafers -> "W1, W2, W3" (the primary Wafer alone would mislead).
            cols[1].write(item.get("wafer_list") or item["wafer_id"] if not is_mock else item["wafer_id"])
            if is_mock:
                cols[2].write(item.get("mode","—"))
                cols[3].write(item.get("image_count","—"))
                cols[4].write("예시")
                cols[5].write(str(item["exposure"]))
                cols[6].write("—")
                cols[7].write(item["status"])
            elif multi:
                cols[2].markdown("Wafer별",help="Wafer마다 노광이 다를 수 있습니다 · Batch 상세의 Wafer 구성에서 확인")
                cols[3].markdown("Wafer별",help="PEB Time (s)는 Wafer별로 저장됩니다 · Batch 상세에서 확인")
                total=int(item["image_count"])
                done=int(analysed.get(identity,0))
                cols[4].write(str(total))
                cols[5].markdown(f"{done}/{total}",help="Batch 전체(모든 Wafer) 기준 · 최신 분석 결과 이미지 수 / 등록 이미지 수")
                cols[6].write("등록 완료" if item["batch_status"]=="registered" else item["batch_status"])
            else:
                from dashboard.units import exposure_display,stored_unit_text
                exposure=item["exposure_value"]
                origin="Batch 기록" if item.get("exposure_source")=="batch" else "Wafer 기록 상속"
                shown=exposure_display(exposure,item["exposure_unit"])
                cols[2].markdown(shown,help="노광 미확인" if exposure is None else
                                 f"노광 {shown} · 단위: {stored_unit_text(item['exposure_unit'])} · {origin}")
                peb=item.get("peb_temperature_c")
                cols[3].markdown("—" if peb is None or peb!=peb else f"{peb:g}",help="PEB Time (s) · 미입력은 —")
                total=int(item["image_count"])
                done=int(analysed.get(identity,0))
                cols[4].write(str(total))
                cols[5].markdown(f"{done}/{total}",help="최신 분석 결과가 저장된 이미지 수 / 등록 이미지 수 · AI 잠정 결과")
                cols[6].write("등록 완료" if item["batch_status"]=="registered" else item["batch_status"])


def image_detail_link(image_id,batch_id):
    """An existing Navigation page URL with exact IDs, never a filename/name key."""
    return "./"+PAGE_SPECS["image_detail"][2]+"?"+urlencode({"image_id":image_id,"batch_id":batch_id})


def image_button(image_id,batch_id,*,key,filename=""):
    label=image_id if len(image_id)<=22 else image_id[:12]+"…"+image_id[-6:]
    if st.button(label,key=key,type="tertiary",width="stretch",
                 help="Image ID: "+image_id+" · "+filename):
        open_image_detail(image_id,batch_id)


def open_wafer_detail(wafer_id,exposure,peb):
    """Same-tab Navigation to Wafer detail with the exact Wafer × 노광 × PEB condition (table labels)."""
    st.session_state.pop("wafer_detail_condition_"+wafer_id,None)  # The clicked row wins over an old choice.
    navigate("wafer_detail",selected_sqlite_wafer={"wafer_id":wafer_id,"exposure":exposure,"peb":peb},
             wafer_table_to_clear="dcc_wafer_conditions")


def open_image_detail(image_id,batch_id):
    """Shared same-tab Navigation for Matrix buttons and image table cells."""
    navigate("image_detail",selected_sqlite_image_id=image_id,
             selected_sqlite_image_batch_id=batch_id,selected_sqlite_batch_id=batch_id,
             batch_detail_source="SQLite")
