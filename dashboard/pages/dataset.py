"""전체 Dataset — registered SQLite images: Wafer/노광/PEB search, selection and ADMIN-only deletion (TASK18-B-12A)."""
import json
import math
import sqlite3
import pandas as pd
import streamlit as st
from dashboard.units import exposure_display
from dashboard.components.layout import page_title
from dashboard.components.data_overview import kpi_tiles,chips
from dashboard.components.sqlite_input import MODE_CODES
from dashboard.sqlite_provider import runtime_provider
from dashboard import dataset_service as service
from database.image_deletion import DeletionRefused

ANY="전체"
PAGE_SIZE=50
MODES={code:label for label,code in MODE_CODES.items()}
OUTCOMES={"COMPLETE":"완료","QUALITY_REVIEW_REQUIRED":"품질 검토","PARTIAL_MEASUREMENT":"부분 계측",
          "POLICY_REVIEW_REQUIRED":"PD 정책 검토","MODEL_PD_EXCLUDED":"PD 계측 제외","FAILED":"실패"}
FILTER_KEYS={"wafer_id":"ds_filter_wafer","exposure":"ds_filter_exposure","peb":"ds_filter_peb"}
CONFIRM_WARNING="선택한 원본 이미지와 관련 분석 데이터가 영구 삭제됩니다. 삭제 후 복구할 수 없습니다."
state=st.session_state

page_title("전체 Dataset","","COMMON / DATASET")  # TASK18-B-13A.1: subtitle removed.


def exposure_label(key):
    if key==ANY:
        return ANY
    value,unit=json.loads(key)
    return exposure_display(value,unit,missing="미기록")  # TASK18-B-13A.2: shared unit rule.


def peb_label(key):
    return ANY if key==ANY else "미입력" if key=="NULL" else key+" s"


def filter_text(filters):
    return (f"Wafer {filters.get('wafer_id',ANY)} · 노광 {exposure_label(filters.get('exposure',ANY))} · "
            f"PEB {peb_label(filters.get('peb',ANY))}")


def clear_selection():
    state["ds_selected"]=set()
    state["ds_plan"]=None
    state["ds_rev"]=state.get("ds_rev",0)+1


def search():
    state["ds_applied"]={name:state.get(key,ANY) for name,key in FILTER_KEYS.items()}
    state["ds_page"]=1
    clear_selection()


def reset_filters():
    for key in FILTER_KEYS.values():
        state[key]=ANY
    search()


def select_all(ids):
    state["ds_selected"]=set(ids)
    state["ds_plan"]=None
    state["ds_rev"]=state.get("ds_rev",0)+1


try:
    provider=runtime_provider()
    options=provider.dataset_filter_options()
except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
    st.error("SQLite Dataset을 조회할 수 없습니다. "+str(exc))
    st.stop()

state.setdefault("ds_applied",{name:ANY for name in FILTER_KEYS})
state.setdefault("ds_selected",set())
state.setdefault("ds_page",1)
pebs=sorted((p for p in options["pebs"] if p!="NULL"),key=float)+(["NULL"] if "NULL" in options["pebs"] else [])
choices={"wafer_id":[ANY,*options["wafers"]],"exposure":[ANY,*options["exposures"]],"peb":[ANY,*pebs]}
for name,key in FILTER_KEYS.items():  # Values that no longer exist fall back to 전체.
    if state.get(key) not in choices[name]:
        state[key]=ANY
applied={name:(value if value in choices[name] else ANY) for name,value in state["ds_applied"].items()}
state["ds_applied"]=applied

with st.container(border=True,key="card_dataset_filters"):
    wafer_col,exposure_col,peb_col,search_col,reset_col=st.columns([1.2,1.2,1.2,.8,.8],vertical_alignment="bottom")
    wafer_col.selectbox("Wafer ID",choices["wafer_id"],key=FILTER_KEYS["wafer_id"],on_change=clear_selection)
    exposure_col.selectbox("노광조건",choices["exposure"],key=FILTER_KEYS["exposure"],format_func=exposure_label,
                           on_change=clear_selection,help="이미지별 실제 공정(Batch-Wafer 기록, 없으면 Wafer 기록)의 노광값")
    peb_col.selectbox("PEB Time (s)",choices["peb"],key=FILTER_KEYS["peb"],format_func=peb_label,
                      on_change=clear_selection,help="미입력 = PEB 기록 없음(NULL)")
    search_col.button("검색",type="primary",width="stretch",key="ds_search",on_click=search)
    reset_col.button("필터 초기화",width="stretch",key="ds_reset",on_click=reset_filters)
    pending={name:state.get(key,ANY) for name,key in FILTER_KEYS.items()}
    if pending!=applied:
        st.caption("필터를 바꿨습니다 · '검색'을 눌러야 결과에 적용됩니다.")

all_ids=provider.dataset_ids(applied)
state["ds_selected"]=set(state["ds_selected"])&set(all_ids)   # Deleted / out-of-filter IDs never stay selected.
total=len(all_ids)
selected=state["ds_selected"]
kpi_tiles([("검색 결과",f"{total}장","accent"),("선택",f"{len(selected)}장","ready" if selected else "neutral"),
           ("전체 등록 이미지",f"{provider.dataset_count()}장")])
chips("적용 필터",[filter_text(applied)])

pages=max(1,math.ceil(total/PAGE_SIZE))
state["ds_page"]=min(max(1,int(state.get("ds_page",1))),pages)
with st.container(horizontal=True,vertical_alignment="center",key="ds_selection_bar"):
    st.button(f"검색 결과 전체 선택 ({total}장)",key="ds_select_all",disabled=not total,on_click=select_all,args=(all_ids,),
              help="현재 검색 조건의 결과 전체(모든 페이지)를 선택합니다. 전체 DB가 아닙니다.")
    st.button("선택 해제",key="ds_clear",disabled=not selected,on_click=clear_selection)
    st.space("stretch")
    st.selectbox("페이지",list(range(1,pages+1)),key="ds_page",format_func=lambda n:f"{n} / {pages}",width=140)

frame=provider.dataset_page(applied,limit=PAGE_SIZE,offset=(state["ds_page"]-1)*PAGE_SIZE) if total else pd.DataFrame()
if frame.empty:
    st.info("검색 조건에 맞는 등록 이미지가 없습니다.")
else:
    view=pd.DataFrame({
        "선택":frame["image_id"].isin(selected),"Image ID":frame["image_id"],"파일명":frame["original_filename"],
        "Wafer":frame["wafer_id"],"Batch":frame["batch_name"].fillna(frame["batch_id"]),
        "노광":[exposure_label(provider.dataset_exposure_key(v,u)) for v,u in zip(frame["exposure_value"],frame["exposure_unit"])],
        "PEB (s)":["미입력" if v is None or v!=v else f"{v:g}" for v in frame["peb_time_s"]],
        "Die":frame["die_code"],"위치":frame["position_code"],"입력 모드":frame["upload_mode"].map(MODES).fillna(frame["upload_mode"]),
        "분석 상태":[OUTCOMES.get(o,o) if isinstance(o,str) else "미분석" for o in frame["latest_outcome"]],
        "등록일":frame["created_at"].str[:16].str.replace("T"," ")})
    edited=st.data_editor(view,hide_index=True,width="stretch",height=min(560,(len(view)+1)*35+3),
        key=f"ds_editor_{state.get('ds_rev',0)}_{state['ds_page']}",disabled=[c for c in view.columns if c!="선택"],
        column_config={"선택":st.column_config.CheckboxColumn("선택",width="small"),
                       "Image ID":st.column_config.TextColumn("Image ID",width="medium")})
    for image_id,checked in zip(edited["Image ID"],edited["선택"]):
        (selected.add if checked else selected.discard)(image_id)
    st.caption(f"{(state['ds_page']-1)*PAGE_SIZE+1}–{(state['ds_page']-1)*PAGE_SIZE+len(view)} / {total}장 · "
               "체크한 선택은 페이지를 넘겨도 유지되며, 필터를 바꾸거나 다시 검색하면 초기화됩니다.")

# ---- Permanent deletion (ADMIN + server checks; GUEST: search/view only) --------------------------
with st.container(border=True,key="card_dataset_delete"):
    enabled,reason=service.deletion_status(provider)
    st.markdown("**영구 삭제**")
    if not enabled:
        st.caption(reason)
    if st.button(f"선택 이미지 영구 삭제 ({len(selected)}장)",key="ds_delete",disabled=not enabled or not selected):
        try:
            state["ds_plan"]={"ids":sorted(selected),"filters":dict(applied),
                              "plan":service.prepare(provider,sorted(selected),applied)}
        except DeletionRefused as exc:
            state["ds_plan"]=None
            st.error(str(exc))
    pending=state.get("ds_plan")
    if enabled and pending and set(pending["ids"])==selected and pending["filters"]==applied:
        plan=pending["plan"]
        # TASK18-B-12A.2: summary → warning → Batch impact → phrase → buttons (plan values unchanged).
        kpi_tiles([("삭제 이미지",f"{len(plan['image_ids'])}장","blocked"),("원본 파일",f"{plan['original_files']}개"),
                   ("분석 결과",f"{plan['dependents']['analysis_image_runs']}건"),
                   ("결과 폴더",f"{plan['result_folders']}개")],compact=True)
        st.error(CONFIRM_WARNING)
        impact=pd.DataFrame([{"Batch":b["batch_id"],"삭제 이미지 수":b["deleting"],"Batch 전체 이미지 수":b["total"],
                              "삭제 후 남는 이미지 수":b["remaining"],
                              "Wafer별 삭제 수량":", ".join(f"{w} {n}" for w,n in sorted(b["wafers"].items()))} for b in plan["batches"]])
        st.dataframe(impact,hide_index=True,width="stretch",height=min(220,(len(impact)+1)*35+3))
        if plan["emptied_batches"]:  # Real consequence, kept: such Batches stay as empty Batches.
            st.caption(f"이미지가 모두 삭제되는 Batch {len(plan['emptied_batches'])}개는 빈 Batch로 남습니다.")
        with st.container(key="ds_confirm"):
            phrase=st.text_input("확인 문구",key=f"ds_phrase_{plan['token'][:12]}",placeholder="영구 삭제 입력")
            col_cancel,col_ok=st.columns(2,vertical_alignment="center")
        if col_cancel.button("취소",key="ds_cancel",width="stretch"):
            state["ds_plan"]=None;st.rerun()
        if col_ok.button("영구 삭제",key="ds_final",type="primary",width="stretch",disabled=phrase!=service.CONFIRM_PHRASE):
            state["ds_plan"]=None   # One request per confirmation: a re-run/double click finds no plan.
            try:
                result=service.execute(provider,pending["ids"],pending["filters"],plan["token"],phrase)
            except DeletionRefused as exc:
                state["ds_message"]=("error",str(exc))
            except Exception as exc:
                state["ds_message"]=("error","삭제 실패 · DB와 파일은 삭제 전 상태로 복구되었습니다: "+type(exc).__name__+": "+str(exc))
            else:
                state["ds_message"]=("success",f"영구 삭제 완료 · 이미지 {result['images']}장 · 원본 {result['original_folders']}개 · "
                                     f"분석 결과 {result['deleted_rows'].get('analysis_image_runs',0)}건 · 기록 {result['deletion_id']}")
                from dashboard.notifications import notify
                notify("영구 삭제",filter_text(pending["filters"]),"완료",count=result["images"],level="warning",
                       key=("dataset_delete",result["deletion_id"]))
            clear_selection()
            st.rerun()
    message=state.pop("ds_message",None)
    if message:
        (st.success if message[0]=="success" else st.error)(message[1])
