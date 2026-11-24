"""Shared record-table UI for two pages only; no analysis settings are applied."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import pandas as pd
import streamlit as st
from dashboard.record_defaults import definition,STATUSES,NOTICE
from dashboard.auth import clear_record_drafts
from dashboard.record_store import RecordStore,RecordError,editing_enabled,default_rows,tab_token

COLUMNS={"item":"항목","value":"기록값","unit":"단위","status":"상태","note":"비고"}


def frame(rows,*,readonly=False):
    records=[]
    for row in rows:
        record={"id":row["id"],**{label:row[key] for key,label in COLUMNS.items()}}
        if readonly and not record["기록값"]:
            record["기록값"]="미입력"
        records.append(record)
    return pd.DataFrame(records).set_index("id")


def edited_rows(table):
    return [{"id":str(identity),**{key:"" if pd.isna(row[label]) else str(row[label])
                                 for key,label in COLUMNS.items()}}
            for identity,row in table.iterrows()]


def local_time(stamp):
    return datetime.fromisoformat(stamp).astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


def render_records(document_id):
    spec=definition(document_id);store=RecordStore()
    st.info(NOTICE)
    enabled=editing_enabled()
    if not enabled:
        clear_record_drafts()
        st.info("GUEST 조회 전용 — 기록 수정은 관리자 로그인이 필요합니다.")
    st.caption("기록 편집 권한: "+("ADMIN" if enabled else "GUEST · 조회 전용")+
               " · 기록은 분석 설정으로 자동 적용되지 않으며 자동 분석 결과가 아닙니다.")
    try:
        document=store.load(document_id)
    except (RecordError,OSError) as exc:
        st.error("기록 파일을 읽거나 초기화하지 못했습니다. 기존 파일은 자동 복구·덮어쓰기하지 않습니다. "+str(exc))
        return
    st.caption(f"마지막 수정: {local_time(document['updated_at'])} · 기록 버전 {document['record_version']}")
    st.caption("기록 파일: "+store.path(document_id).as_posix())
    flash=st.session_state.pop("record_flash_"+document_id,None)
    if flash:
        st.success(flash)
    tabs=st.tabs([title for key,title,rows in spec["tabs"]],key="record_tabs_"+document_id)
    for tab,(tab_id,title,definitions) in zip(tabs,spec["tabs"]):
        with tab:
            saved=document["tabs"][tab_id]
            prefix="record_"+document_id+"_"+tab_id
            draft_key=prefix+"_draft";epoch_key=prefix+"_epoch"
            draft=st.session_state.get(draft_key)
            heading,action=st.columns([4,1],vertical_alignment="center")
            heading.subheader(title)
            st.caption(f"탭 수정: {local_time(saved['updated_at'])} · 버전 {saved['revision']}")
            if draft is None:
                if action.button("수정",key=prefix+"_edit",disabled=not enabled,width="stretch"):
                    st.session_state[draft_key]={"rows":deepcopy(saved["rows"]),"token":tab_token(saved)}
                    st.session_state[epoch_key]=st.session_state.get(epoch_key,0)+1
                    st.rerun()
                st.dataframe(frame(saved["rows"],readonly=True),hide_index=True,width="stretch",
                    height=min(560,(len(saved["rows"])+1)*36+3),row_height=36,
                    column_config={"항목":st.column_config.TextColumn(width="medium"),
                        "기록값":st.column_config.TextColumn(width="large"),
                        "단위":st.column_config.TextColumn(width="small"),
                        "상태":st.column_config.TextColumn(width="small"),
                        "비고":st.column_config.TextColumn(width="large")})
                continue
            if draft["token"]!=tab_token(saved):
                st.warning("저장 파일의 이 탭이 변경됐습니다. 취소 후 최신 내용을 확인하세요. 이전 초안은 자동 덮어쓰지 않습니다.")
            if draft.get("restored"):
                st.info("기본값을 편집 초안에 불러왔습니다. 저장해야 파일에 반영됩니다.")
            edited=st.data_editor(frame(draft["rows"]),hide_index=True,width="stretch",
                height=min(560,(len(draft["rows"])+1)*36+3),row_height=36,num_rows="fixed",
                disabled=["항목"] if enabled else True,
                column_config={"항목":st.column_config.TextColumn(width="medium",disabled=True),
                    "기록값":st.column_config.TextColumn(width="large",help="숫자는 단위를 제외하고 입력하세요. 빈 값은 허용됩니다."),
                    "단위":st.column_config.TextColumn(width="small"),
                    "상태":st.column_config.SelectboxColumn(options=list(STATUSES),required=True,width="small"),
                    "비고":st.column_config.TextColumn(width="large")},
                key=prefix+"_editor_"+str(st.session_state.get(epoch_key,0)))
            save,cancel,restore=st.columns(3)
            if save.button("저장",key=prefix+"_save",type="primary",disabled=not enabled,width="stretch"):
                try:
                    store.save_tab(document_id,tab_id,edited_rows(edited),expected_token=draft["token"])
                except (RecordError,OSError) as exc:
                    st.error("저장하지 않았습니다. 입력값·편집 허용 상태를 확인하세요. "+str(exc))
                else:
                    st.session_state.pop(draft_key,None)
                    st.session_state["record_flash_"+document_id]=title+" 기록을 저장했습니다."
                    st.rerun()
            if cancel.button("취소",key=prefix+"_cancel",width="stretch"):
                st.session_state.pop(draft_key,None)
                st.session_state["record_flash_"+document_id]=title+" 수정 내용을 폐기했습니다. 저장값은 유지됩니다."
                st.rerun()
            if restore.button("기본값 복원",key=prefix+"_restore",disabled=not enabled,width="stretch"):
                draft["rows"]=default_rows(document_id,tab_id);draft["restored"]=True
                st.session_state[epoch_key]=st.session_state.get(epoch_key,0)+1
                st.rerun()
