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


def batch_table(frame, *, source="Mock"):
    """All Batches in one bounded scroll area; one widget per cell, no stacked captions."""
    is_mock=source=="Mock"
    if "created_at" in frame:
        frame=frame.sort_values("created_at",ascending=False,kind="stable")
    headings=["Batch","Wafer","유형","이미지","Role","노광","등록 일시","상태"]
    widths=[1.8,.65,1,.6,.85,.9,1.35,.9]
    modes={"wafer_batch":"Wafer","repeatability":"반복촬영","lighting":"밝기","single_image":"단일"}
    roles={"evaluation":"EVAL","training_candidate":"TRAIN","development":"DEV"}
    height=380 if st.session_state.get("dcc_expanded_card")=="batches" else 160
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
            if cols[0].button(label,key=key,type="tertiary",help="Batch ID: "+identity+" · "+full_label):
                if is_mock:
                    navigate("batch_detail",selected_batch_id=identity,detail_batch=identity,batch_detail_source="Mock")
                else:
                    navigate("batch_detail",selected_sqlite_batch_id=identity,batch_detail_source="SQLite")
            cols[1].write(item["wafer_id"])
            if is_mock:
                cols[2].write(item.get("mode","—"))
                cols[3].write(item.get("image_count","—"))
                cols[4].write("예시")
                cols[5].write(str(item["exposure"]))
                cols[6].write("—")
                cols[7].write(item["status"])
            else:
                mode=item["upload_mode"]
                role=item["dataset_role"]
                cols[2].markdown(modes.get(mode,mode),help=mode)
                cols[3].write(str(item["image_count"]))
                cols[4].markdown(roles.get(role,role),help=role.upper())
                exposure=item["exposure_value"]
                unit=item["exposure_unit"] or "단위 미상"
                cols[5].markdown("미확인" if exposure is None else f"{exposure:g}",
                                 help="노광 미확인" if exposure is None else f"노광 {exposure:g} · {unit}")
                stamp=item["created_at"]
                cols[6].markdown(stamp[5:16].replace("T"," "),help=stamp)
                cols[7].write("등록 완료" if item["batch_status"]=="registered" else item["batch_status"])
