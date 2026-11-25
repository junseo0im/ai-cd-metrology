"""Data Control registration/query UI; algorithms remain disconnected."""
import hashlib
import json
import math
import os
import re
from pathlib import Path
import sqlite3
import uuid

import streamlit as st
from dashboard.notifications import notify

from dashboard import data_provider as mock
from dashboard.sqlite_provider import runtime_provider,sync_uploads,reviewed_inputs
from dashboard.components.tables import batch_table,open_image_detail,open_wafer_detail
from dashboard.components.data_overview import accumulation_overview,processing_progress
from database.batch_registry import DUPLICATE_ACTIONS
from database.batch_names import notes_with_batch_name
from database.filename_parser import FilenameParseError,parse_filename,validate_metadata
from database.upload_storage import MAX_IMAGES

MODE_CODES={"Wafer Batch":"wafer_batch","Repeatability":"repeatability",
            "Lighting":"lighting","Single Image":"single_image"}
ACTION_LABELS={"EXCLUDE":"등록 제외","REGISTER_SEPARATELY":"중복 허용 후 별도 등록","HOLD":"등록 보류"}


def _provider():
    return runtime_provider()


def _short_id(value):
    return value if len(value)<=24 else value[:14]+"…"+value[-6:]


def _short_name(value):
    return value if len(value)<=40 else value[:24]+"…"+value[-12:]


def issue_guidance(message):
    """Present remedies without changing any backend validation rule or result."""
    text=str(message)
    if "ZIP 이름 Wafer" in text:
        return text  # ZIP name/content conflict: shown as-is (never auto-resolved).
    per_wafer=re.match(r"(W[0-9]+): ",text)
    if per_wafer and ("exposure_value" in text or "process_run" in text):
        return (f"{per_wafer.group(1)} 노광 조건 미입력 · Wafer별 공정조건에 실제 노광 수치를 입력해야 등록할 수 있습니다.")
    if "is not one of this Batch's Wafers" in text:
        return "Batch Wafer 목록에 없는 Wafer의 이미지입니다. 파일명·이미지별 Wafer ID를 확인하세요."
    remedies=(
        (("Wafer mismatch","wafer_id mismatch","same wafer"),"공통 Wafer와 파일명/개별 Wafer가 다릅니다. 촬영 대상을 확인하고 Metadata를 수정하세요."),
        (("exposure_value","process_run"),"노광 조건 미입력 · 공통 Metadata에 실제 노광 수치를 입력해야 등록할 수 있습니다."),
        (("Filename needs","manual_metadata"),"파일명 확인·수정 탭에서 Wafer / Die / 촬영 위치를 입력하고 Preview를 재검증하세요."),
        (("wafer_id","canonical W"),"Wafer ID를 W1 형식으로 입력하세요. Batch와 이미지의 Wafer가 같아야 합니다."),
        (("die","Die"),"Die ID를 11~18 / 21~28 / 31~38 중에서 선택하세요."),
        (("position",),"촬영 위치를 lt / c / rb 중에서 선택하세요."),
        (("Repeatability",),"같은 촬영 위치의 이미지인지 확인하고 서로 다른 양의 반복 번호를 지정하세요."),
        (("magnification",),"반복 / 조명 Metadata에서 실제 촬영 배율을 입력하세요."),
        (("illumination","condition_id","Acquisition condition"),"반복 / 조명 Metadata에서 실제 조명 설정과 촬영 조건을 확인하세요."),
        (("Single Image",),"Single Image는 PNG 1장과 Schema v2가 필요합니다. 현재 Schema와 파일 수를 확인하세요."),
        (("PNG","ZIP","image_path","Cannot read"),"PNG/ZIP 형식·손상·용량과 파일 선택을 확인하세요. 지원하지 않는 파일을 제거한 뒤 다시 선택하세요."),
    )
    for tokens,remedy in remedies:
        if any(token in text for token in tokens):
            return remedy
    if re.search("[가-힣]",text):
        return text
    return "파일과 필수 Metadata를 확인하세요. 기술 오류 상세에서 원인을 확인할 수 있습니다."


def _technical_message(message):
    if os.environ.get("IDA_SHOW_DB_PATH")=="1":
        return str(message)
    return re.sub(r"[A-Za-z]:[\\/][^\r\n]*", "[로컬 경로 숨김]", str(message))


def _show_error(title,exc):
    st.error(title+" "+issue_guidance(str(exc)))
    with st.expander("기술 오류 상세"):
        st.code(type(exc).__name__+": "+_technical_message(exc))


def _preview_rows(preview):
    rows=[]
    for item in preview["images"]:
        identity=item["effective_metadata"] or {}
        rows.append({"파일명":_short_name(item["original_filename"] or ""),
                     "Wafer":identity.get("wafer_id"),"Die":identity.get("die_code"),
                     "위치":identity.get("position_code"),"상태":item["status"],
                     "입력 출처":item["metadata_source"] or "확인 필요"})
    return rows


def _clear_registration_approval():
    old=st.session_state.pop("sqlite_final_confirmation_key",None)
    if old:
        st.session_state.pop(old,None)


def _clear_confirmations():
    _clear_registration_approval()
    for key in st.session_state.pop("sqlite_confirmation_keys",[]):
        st.session_state.pop(key,None)
    for key in ("sqlite_confirmation_scope","sqlite_duplicate_decisions","sqlite_registration_plan"):
        st.session_state.pop(key,None)


def _confirmation_scope(preview):
    token=hashlib.sha256(json.dumps(preview,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if st.session_state.get("sqlite_confirmation_scope")!=token:
        _clear_confirmations()
    st.session_state["sqlite_confirmation_scope"]=token
    st.session_state.setdefault("sqlite_duplicate_decisions",{})
    st.session_state["sqlite_confirmation_keys"]=[
        "sqlite_duplicate_action_"+token+row["image_id"] for row in preview["images"]
        if row["duplicate_image_ids"] or row["duplicate_input_indices"]]
    return token


def _upload_selection_changed():
    st.session_state["sqlite_upload_selection_changed"]=True


def _save_duplicate_action(image_id,key):
    st.session_state["sqlite_duplicate_decisions"][image_id]=st.session_state[key]


def _render_duplicate_warning(preview,provider,token):
    rows=[row for row in preview["images"] if row["duplicate_image_ids"] or row["duplicate_input_indices"]]
    if not rows:
        return {}
    records=provider.duplicate_records(key for row in rows for key in row["duplicate_image_ids"])
    by_id={row["image_id"]:row for row in records}
    st.warning(f"중복 이미지 발견 — {len(rows)}장 · 기본: 등록 제외")
    choices=st.session_state["sqlite_duplicate_decisions"]
    for row in rows:
        key="sqlite_duplicate_action_"+token+row["image_id"]
        action=st.selectbox("중복 처리 방법 · "+_short_name(row["original_filename"]),DUPLICATE_ACTIONS,
            index=DUPLICATE_ACTIONS.index(choices.get(row["image_id"],"EXCLUDE")),
            format_func=lambda value:ACTION_LABELS[value],key=key,
            on_change=_save_duplicate_action,args=(row["image_id"],key))
        choices[row["image_id"]]=action
        if action=="REGISTER_SEPARATELY":
            st.caption("이번 등록에도 새 Image ID로 기록합니다. 데이터 집계 시 중복에 주의하세요.")
        with st.expander("기존 등록 정보 보기 · "+_short_name(row["original_filename"])):
            st.json({"uploaded_filename":row["original_filename"],"sha256":row["sha256"],
                     "existing_records":[by_id.get(key,{"image_id":key,"batch_id":None})
                                         for key in row["duplicate_image_ids"]],
                     "same_upload_candidates":[preview["images"][index]["original_filename"]
                                               for index in row["duplicate_input_indices"]]})
    return {row["image_id"]:choices[row["image_id"]] for row in rows}


def _schema_notice(provider):
    info=provider.schema_info()
    st.caption(f"SQLite · {info['display_name']} · Schema v{info['version']} · "
               +("Single Image 지원" if info["single_image_supported"] else "Single Image는 v2 적용 후 사용 가능"))
    if os.environ.get("IDA_SHOW_DB_PATH")=="1":
        with st.expander("관리자 연결 정보"):
            st.code(str(provider.db_path.resolve()))
    return info


def input_mode_changed():
    if st.session_state.get("data_control_source")=="SQLite":
        _synchronize_mode(st.session_state.get("input_mode","Wafer Batch"))


def _synchronize_mode(mode):
    bundle=st.session_state.get("sqlite_uploads")
    if not bundle or bundle.get("input_mode")==mode:
        return
    if "registered" in bundle:
        _reset_draft()  # Keep the saved DB rows; retire only the completed UI draft.
        return
    _clear_confirmations()
    st.session_state.pop("sqlite_preview",None)
    bundle["per_image_metadata"]={}
    common=bundle.get("common_metadata",{})
    common.pop("magnification",None)
    common.pop("auto_exposure",None)
    bundle["input_mode"]=mode
    for key in list(st.session_state):
        if key.startswith(("sqlite_repeat_","sqlite_light_","sqlite_magnification_","sqlite_auto_")):
            st.session_state.pop(key,None)


def _registration_body(provider,mode,dataset_role,upload_slot=None,batch_name=""):
    _synchronize_mode(mode)
    info=provider.schema_info()
    if mode=="Single Image" and not info["single_image_supported"]:
        return None,None,"Schema v1에서는 Single Image를 저장할 수 없습니다. 별도 승인 후 v2 Migration이 필요합니다."
    generation=st.session_state.get("sqlite_uploader_generation","initial")
    with (upload_slot or st).popover("Upload / Drop",width="stretch",
                                     help="클릭해 파일을 선택하거나, 열린 업로드 영역에 파일을 끌어다 놓으세요. "
                                          +("PNG / ZIP" if mode=="Wafer Batch" else "PNG")+"만 지원합니다."):
        uploaded=st.file_uploader("PNG / ZIP 선택" if mode=="Wafer Batch" else "PNG 이미지 선택",
            type=["png","zip"] if mode=="Wafer Batch" else ["png"],
            accept_multiple_files=mode!="Single Image",key="sqlite_files_"+generation,
            on_change=_upload_selection_changed)
    uploaded=uploaded if isinstance(uploaded,list) else ([uploaded] if uploaded else [])
    if mode=="Single Image" and len(uploaded)>1:
        return None,None,"Single Image는 PNG 한 장만 선택하세요."
    payloads=[{"filename":item.name,"content":item.getvalue()} for item in uploaded]
    if mode!="Wafer Batch" and any(item["filename"].lower().endswith(".zip") for item in payloads):
        return None,None,"ZIP은 Wafer Batch에서 사용하세요."
    changed=st.session_state.pop("sqlite_upload_selection_changed",False)
    retained=st.session_state.get("sqlite_uploads")
    if not payloads and retained and not changed:
        bundle=retained
        st.caption("이전 임시 업로드를 유지합니다. 새 파일 선택 또는 새 등록 시작으로 변경할 수 있습니다.")
    else:
        bundle=sync_uploads(st.session_state,payloads,provider)
        if bundle and "registered" not in bundle:
            notify("업로드 완료",f"{len(payloads)}개 파일","임시 업로드 · DB 미등록",count=len(bundle["inputs"]),
                   level="info",key=("upload",bundle["inputs"][0]["image_id"]))
    if not bundle:
        _clear_confirmations()
        return None,None,""  # No files yet: the register button stays disabled without an extra notice.
    if "registered" in bundle:
        _clear_confirmations()
        result=bundle["registered"]
        st.success("SQLite 등록 완료 · "+bundle.get("input_mode",mode))
        st.write(f"업로드 {result.get('uploaded_count',len(bundle['inputs']))}장 · "
                 f"신규 등록 {result.get('registered_count',result['image_count'])}장 · "
                 f"중복 허용 후 등록 {result.get('duplicate_registered_count','미기록')}장 · "
                 f"중복 제외 {result.get('excluded_count','미기록')}장")
        saved=provider.batches()
        row=saved[saved["batch_id"]==result["batch_id"]]
        if not row.empty:
            st.text(row.iloc[0]["display_name"])
        with st.expander("등록 결과 상세"):
            st.json(result)
        return None,bundle,"등록 완료 · 새 업로드 준비 (입력 초기화)를 사용하세요."
    bundle.setdefault("batch_id","BATCH-"+uuid.uuid4().hex)
    bundle.setdefault("process_run_id","PROCESS-"+uuid.uuid4().hex)
    bundle.setdefault("batch_process_run_id","PROCESS-"+uuid.uuid4().hex)  # Used only for a Batch-only exposure (v5).
    batch_process=info.get("version",0)>=5
    if bundle.get("input_mode")!=mode:
        bundle["per_image_metadata"]={}
        bundle["input_mode"]=mode
    common=bundle.setdefault("common_metadata",{})
    bundle["batch_name"]=batch_name
    inferred=set()
    for item in bundle["inputs"]:
        try:
            inferred.add(parse_filename(Path(item["image_path"]).name).wafer_id)
        except FilenameParseError:
            pass
    scope=bundle["inputs"][0]["image_id"]
    # TASK18-B-10B: Schema v6 + Wafer Batch -> any number of Wafers in ONE Batch, each with its own
    # 노광/PEB. Other modes and v5 DBs keep the single-Wafer common Metadata exactly as before.
    multi=mode=="Wafer Batch" and info.get("multi_wafer_supported",False)
    status_slots={}
    if multi:
        metadata,wafer,status_slots=_multi_wafer_metadata(provider,bundle,scope,info,dataset_role,batch_name)
        common.update(mode=mode,dataset_role=dataset_role)
    else:
        metadata,wafer=_single_wafer_metadata(provider,bundle,common,scope,inferred,info,batch_process,
                                              mode,dataset_role,batch_name)
    with st.expander("반복 / 조명 Metadata",expanded=True) if mode in ("Repeatability","Lighting") else st.container():
        if mode=="Repeatability":
            st.caption("반복 번호 초기값은 업로드 순서입니다. 실제 순서를 확인·수정하세요.")
            for index,item in enumerate(bundle["inputs"],1):
                number=st.number_input("반복 번호 · "+_short_name(Path(item["image_path"]).name),
                    min_value=1,value=bundle["per_image_metadata"].get(item["image_id"],{}).get("repeat_index",index),
                    step=1,key="sqlite_repeat_"+item["image_id"],help="필수 · 같은 위치의 서로 다른 양의 반복 번호")
                bundle["per_image_metadata"][item["image_id"]]={"repeat_index":number}
        elif mode=="Lighting":
            magnification=st.number_input("촬영 배율",min_value=0.1,value=common.get("magnification"),
                                          step=1.0,key="sqlite_magnification_"+scope,placeholder="필수 · 실제 배율",
                                          help="필수 · 실제 촬영 배율을 입력하세요. 임의 기본값은 없습니다.")
            auto=st.selectbox("Auto Exposure",[None,0,1],index=[None,0,1].index(common.get("auto_exposure")),
                             format_func=lambda value:{None:"알 수 없음",0:"OFF",1:"ON"}[value],
                             key="sqlite_auto_"+scope,help="선택 · 알 수 없으면 미상으로 유지하세요.")
            common.update(magnification=magnification,auto_exposure=auto)
            condition_ids=bundle.setdefault("condition_ids",{})
            for item in bundle["inputs"]:
                previous=bundle["per_image_metadata"].get(item["image_id"],{}).get("acquisition_condition",{})
                illumination=st.text_input("조명 설정 · "+_short_name(Path(item["image_path"]).name),
                    value=previous.get("illumination_setting",""),key="sqlite_light_"+item["image_id"],placeholder="필수 · 실제 조명 설정",help="필수 · 이미지별 실제 조명 조건")
                condition_ids.setdefault(item["image_id"],"COND-"+uuid.uuid4().hex)
                bundle["per_image_metadata"][item["image_id"]]={"acquisition_condition":{
                    "condition_id":condition_ids[item["image_id"]],"magnification":magnification,
                    "illumination_setting":illumination,"auto_exposure":auto}}
    preview=provider.preview(reviewed_inputs(bundle),metadata)
    st.session_state["sqlite_preview"]=preview
    token=_confirmation_scope(preview)
    decisions=_render_duplicate_warning(preview,provider,token)
    plan=provider.prepare_registration(preview,decisions)
    st.session_state["sqlite_registration_plan"]=plan
    duplicate_count=sum(bool(row["duplicate_image_ids"] or row["duplicate_input_indices"]) for row in preview["images"])
    if duplicate_count:
        notify("중복 감지",wafer or "Wafer 미입력","등록 전 처리 선택 필요",count=duplicate_count,
               level="warning",key=("duplicate",token))
    error_names=[_short_name(row["original_filename"]) for row in preview["images"]
                 if row["status"] in ("NEEDS_METADATA","INVALID")]
    errors=len(error_names)
    valid=sum(row["status"]=="VALID" for row in preview["images"])
    st.caption(f"전체 업로드 {len(preview['images'])} · 정상 검증 {valid} · 중복 {duplicate_count} · "
               f"오류·수정 필요 {errors} · 최종 등록 예정 {plan['selected_count']} · "
               f"제외 {len(plan['excluded_image_ids'])} · 중복 허용 등록 {len(plan['separate_image_ids'])}")
    if error_names:
        st.caption("오류·수정 필요 파일: "+", ".join(error_names[:10])+(f" 외 {len(error_names)-10}개" if len(error_names)>10 else ""))
    with st.expander("검증 목록"):
        st.dataframe(_preview_rows(preview),hide_index=True,width="stretch")
    if multi:
        _render_multi_preview(preview,plan,status_slots,batch_name)
    elif mode=="Wafer Batch":
        sites=[(row["effective_metadata"]["die_code"],row["effective_metadata"]["position_code"])
               for row in plan["final_preview"]["images"] if row["effective_metadata"]]
        st.caption(f"기본 72개 촬영 위치 중 {len(set(sites))}개 · 누락 {72-len(set(sites))}개 · "
                   f"동일 위치 추가 촬영 {len(sites)-len(set(sites))}개 (72장 고정 제한 아님)")
    if plan["held_image_ids"]:
        names=[_short_name(row["original_filename"]) for row in preview["images"] if row["image_id"] in plan["held_image_ids"]]
        return plan,bundle,"HOLD 보류: "+", ".join(names)
    if not plan["selected_count"]:
        st.info("등록할 새 이미지가 없습니다")
        return plan,bundle,"등록 대상 0장입니다."
    if not plan["registration_ready"]:
        issues=preview["batch_issues"]+[issue for row in preview["images"] for issue in row["issues"]]
        return plan,bundle,issue_guidance(issues[0]["message"]) if issues else "필수 Metadata 또는 파일명·촬영 조건을 수정하세요."
    return plan,bundle,""


# --- TASK18-B-10B: several Wafers in ONE Batch (Schema v6, Wafer Batch mode) --------------------
READINESS={"VALID":"READY","DUPLICATE_CANDIDATE":"REVIEW_REQUIRED","NEEDS_METADATA":"REVIEW_REQUIRED","INVALID":"BLOCKED"}


def _natural_wafer(wafer):
    digits=wafer[1:] if isinstance(wafer,str) else ""
    return (0,int(digits),wafer) if digits.isdigit() else (1,0,str(wafer))


def detected_wafers(bundle):
    """{Wafer: image count} from each image's own identity (manual edit first, else filename) + unknowns.

    Dynamic: any number of Wafers and any number of images per Wafer; nothing is assumed (no fixed
    Wafer IDs, no 72 images). A ZIP name is never used. Unidentified images are counted, never assigned.
    """
    counts={}
    unknown=0
    for item in bundle["inputs"]:
        manual=bundle.get("manual_metadata",{}).get(item["image_id"])
        wafer=None
        if isinstance(manual,dict):
            wafer=manual.get("wafer_id")
            if wafer:
                try:
                    wafer=validate_metadata(wafer_id=wafer,die_code="11",position_code="c").wafer_id
                except FilenameParseError:
                    pass  # Kept as typed; the registry reports the invalid ID.
        else:
            try:
                wafer=parse_filename(Path(item["image_path"]).name).wafer_id
            except FilenameParseError:
                wafer=None
        if wafer:
            counts[wafer]=counts.get(wafer,0)+1
        else:
            unknown+=1
    return dict(sorted(counts.items(),key=lambda pair:_natural_wafer(pair[0]))),unknown


def _apply_common_values(scope):
    """Fill ONLY empty 노광/PEB cells of the selected Wafers; entered values are never overwritten."""
    bundle=st.session_state.get("sqlite_uploads")
    if not bundle:
        return
    meta=bundle.setdefault("wafer_meta",{})
    revision=bundle.get("wafer_revision",0)
    for wafer,entry in meta.items():   # Keep what the user typed in the current widgets first.
        for field in ("exposure","peb"):
            key=f"sqlite_wafer_{field}_{scope}_{wafer}_{revision}"
            if key in st.session_state:
                entry[field]=st.session_state[key]
    values={"exposure":st.session_state.get("sqlite_wafer_common_exposure_"+scope),
            "peb":st.session_state.get("sqlite_wafer_common_peb_"+scope)}
    filled={"exposure":[],"peb":[]}
    kept=[]
    for wafer in st.session_state.get("sqlite_wafer_common_targets_"+scope) or []:
        entry=meta.get(wafer)
        if entry is None:
            continue
        for field,value in values.items():
            if value is None:
                continue
            if entry[field] is None:
                entry[field]=float(value)
                filled[field].append(wafer)
            elif entry[field]!=float(value):
                kept.append(wafer)
    bundle["wafer_revision"]=revision+1   # New widget keys pick up the stored values without state conflicts.
    note=[]
    if filled["exposure"]:
        note.append("노광 적용: "+", ".join(filled["exposure"]))
    if filled["peb"]:
        note.append("PEB 적용: "+", ".join(filled["peb"]))
    if kept:
        note.append("기존 값 유지(덮어쓰지 않음): "+", ".join(dict.fromkeys(kept)))
    bundle["wafer_apply_note"]=" · ".join(note) or "적용할 빈 칸이 없습니다."


def _multi_wafer_metadata(provider,bundle,scope,info,dataset_role,batch_name):
    """One input row per DETECTED Wafer (dynamic count): 노광 (required number) and PEB Time (s, optional)."""
    wafers,unknown=detected_wafers(bundle)
    meta=bundle.setdefault("wafer_meta",{})
    for wafer in wafers:
        meta.setdefault(wafer,{"exposure":None,"peb":None,"process_run_id":"PROCESS-"+uuid.uuid4().hex,
                               "batch_process_run_id":"PROCESS-"+uuid.uuid4().hex})
    revision=bundle.get("wafer_revision",0)
    missing=unknown or any(meta[w]["exposure"] is None for w in wafers)
    slots={}
    with st.expander("Wafer별 공정조건",expanded=bool(missing)):
        st.caption(f"감지된 Wafer {len(wafers)}개 · 이미지 {sum(wafers.values())}장"
                   +(f" · Wafer 미확인 {unknown}장 (업로드 이미지 확인 탭에서 Wafer / Die / 위치 입력)" if unknown else "")
                   +" · Wafer는 이미지 파일명(또는 이미지별 수정값)으로만 구분합니다.")
        widths=[.8,.7,1.3,1.3,1.4]
        for column,heading in zip(st.columns(widths,gap="small"),
                                  ["Wafer ID","이미지","노광 조건","PEB Time (s)","검증 상태"]):
            column.markdown("**"+heading+"**")
        for wafer,count in wafers.items():
            entry=meta[wafer]
            cols=st.columns(widths,gap="small",vertical_alignment="center")
            cols[0].write(wafer)
            cols[1].write(f"{count}장")
            entry["exposure"]=cols[2].number_input("노광 조건 · "+wafer,value=entry["exposure"],step=1.0,format="%g",
                key=f"sqlite_wafer_exposure_{scope}_{wafer}_{revision}",placeholder="필수 · 실제 노광 시간 (s)",
                label_visibility="collapsed",help="이 Wafer의 실제 노광 · 단위는 입력·추정하지 않습니다.")
            entry["peb"]=cols[3].number_input("PEB Time (s) · "+wafer,value=entry["peb"],step=1.0,format="%g",
                key=f"sqlite_wafer_peb_{scope}_{wafer}_{revision}",placeholder="미입력은 NULL",
                disabled=not info.get("peb_supported",False),label_visibility="collapsed",
                help="이 Wafer의 PEB 시간(초) · 입력한 숫자를 그대로 저장합니다.")
            slots[wafer]=cols[4].empty()
        if wafers:
            with st.popover("공통값 적용 (빈 칸에만)",width="content"):
                st.number_input("공통 노광 조건",value=None,step=1.0,format="%g",key="sqlite_wafer_common_exposure_"+scope,
                                placeholder="선택")
                st.number_input("공통 PEB Time (s)",value=None,step=1.0,format="%g",key="sqlite_wafer_common_peb_"+scope,
                                placeholder="선택",disabled=not info.get("peb_supported",False))
                st.multiselect("적용 대상 Wafer",list(wafers),default=list(wafers),key="sqlite_wafer_common_targets_"+scope)
                st.caption("선택한 Wafer 중 비어 있는 칸에만 채웁니다. 이미 입력된 Wafer별 값은 바꾸지 않습니다.")
                st.button("빈 칸에 적용",key="sqlite_wafer_common_apply_"+scope,on_click=_apply_common_values,args=(scope,))
        if bundle.get("wafer_apply_note"):
            st.caption(bundle["wafer_apply_note"])
    configs={}
    for wafer in wafers:
        entry=meta[wafer]
        existing=provider.wafer_process(wafer)
        exposure=entry["exposure"]
        # No unit widget: the Wafer record's unit is reused only for exactly its own value.
        unit=(existing["exposure_unit"] if existing and exposure is not None and existing["exposure_value"]==float(exposure)
              else None)
        config={"peb_time_s":None if entry["peb"] is None else float(entry["peb"]),
                "batch_process_run_id":entry["batch_process_run_id"]}
        if not existing:
            config["process_run_id"]=entry["process_run_id"]
        if exposure is not None:
            config["process_run"]={"exposure_value":float(exposure),"exposure_unit":unit}
        configs[wafer]=config
    bundle["wafer_label"]=", ".join(wafers)
    metadata={"batch_id":bundle["batch_id"],"wafer_id":next(iter(wafers),""),"upload_mode":"wafer_batch",
              "dataset_role":dataset_role.lower(),"batch_status":"registered",
              "notes":notes_with_batch_name(None,batch_name),"wafers":configs}
    return metadata,bundle["wafer_label"],slots


def _render_multi_preview(preview,plan,slots,batch_name):
    """Batch totals and the per-Wafer validation table (counts from the registry Preview, not recomputed)."""
    rows=preview["images"]
    states=[READINESS.get(row["status"],"BLOCKED") for row in rows]
    duplicates=sum(bool(row["duplicate_image_ids"] or row["duplicate_input_indices"]) for row in rows)
    listed=[item for item in preview.get("wafer_summary",[]) if item["wafer_id"]]
    st.caption(f"Batch {batch_name or '(이름 미입력 · 자동 표시명)'} · 전체 {len(rows)}장 · Wafer {len(listed)}개 · "
               f"READY {states.count('READY')} · REVIEW_REQUIRED {states.count('REVIEW_REQUIRED')} · "
               f"BLOCKED {states.count('BLOCKED')} · 중복 {duplicates} · 최종 등록 예정 {plan['selected_count']}")
    by_wafer={}
    for row in rows:
        wafer=row["effective_metadata"]["wafer_id"] if row["effective_metadata"] else None
        by_wafer.setdefault(wafer,[]).append(row)
    config=preview["batch_metadata"].get("wafers") or {}
    table=[]
    for item in preview.get("wafer_summary",[]):
        wafer=item["wafer_id"]
        members=by_wafer.get(wafer,[])
        counts={state:sum(READINESS.get(r["status"])==state for r in members) for state in ("READY","REVIEW_REQUIRED","BLOCKED")}
        reasons=list(dict.fromkeys(issue_guidance(issue["message"]) for r in members for issue in r["issues"]))
        process=(config.get(wafer) or {}).get("process_run") or {}
        peb=(config.get(wafer) or {}).get("peb_time_s")
        table.append({"Wafer ID":wafer or "미확인","이미지":item["image_count"],
                      "노광":"미입력" if process.get("exposure_value") is None else exposure_text(process["exposure_value"],process.get("exposure_unit")),
                      "PEB (s)":"—" if peb is None else f"{peb:g}",
                      "Die·위치":f"{item['site_count']}개 위치"+(f" · 추가 촬영 {item['same_site_extra']}" if item["same_site_extra"] else ""),
                      "READY":counts["READY"],"REVIEW_REQUIRED":counts["REVIEW_REQUIRED"],"BLOCKED":counts["BLOCKED"],
                      "중복 후보":item["duplicate_count"],"사유":" / ".join(reasons[:2]) or "—"})
        if wafer in slots:
            slots[wafer].caption(f"READY {counts['READY']} · 검토 {counts['REVIEW_REQUIRED']} · 차단 {counts['BLOCKED']}")
    with st.expander("Wafer별 검증",expanded=any(t["BLOCKED"] or t["REVIEW_REQUIRED"] for t in table)):
        st.dataframe(table,hide_index=True,width="stretch",key="sqlite_wafer_preview_table",
                     column_config={"사유":st.column_config.TextColumn("사유",width="large")})
        st.caption("Die·위치는 등록 대상 위치 수입니다. 72개(24 Die × lt/c/rb)는 완전 촬영 세트의 예시이며 필수 수량이 아닙니다.")

def _single_wafer_metadata(provider,bundle,common,scope,inferred,info,batch_process,mode,dataset_role,batch_name):
    """Pre-v6 / non-Wafer-Batch common Metadata: ONE Batch Wafer, one 노광/PEB (unchanged behaviour)."""
    initial_wafer=common.get("wafer",next(iter(inferred)) if len(inferred)==1 else "")
    initial_process=provider.wafer_process(initial_wafer)
    missing_common=not initial_wafer or (common.get("exposure") is None and (initial_process is None or batch_process))
    with st.expander("공통 Metadata",expanded=missing_common):
        wafer=st.text_input("Batch Wafer ID",value=initial_wafer,placeholder="필수 · 예: W1",
                            help="필수 · 촬영 대상 Wafer입니다. 파일명과 충돌하면 자동 덮어쓰지 않습니다.",
                            key="sqlite_wafer_"+scope)
        existing=provider.wafer_process(wafer)
        known_exposure=existing["exposure_value"] if existing else None
        # Unknown process values are never replaced by the Mock default. With Batch-level process
        # records (v5) the Wafer's existing exposure is shown as a reference, never pre-selected.
        default=common.get("exposure") if batch_process else common.get("exposure",known_exposure)
        # TASK18-B-10A.2: the Wafer-record reference caption is no longer shown (lookup and policy unchanged).
        # TASK18-B-10A.1: direct numeric entry (no preset list, no range). Empty = not entered (registration blocked).
        exposure=st.number_input("노광 조건",value=None if default is None else float(default),step=1.0,format="%g",
            key="sqlite_exposure_value_"+scope,placeholder="필수 · 실제 노광 시간 (s)",
            help="신규 공정은 필수 · 실제 사용한 값만 입력하세요 (소수 가능).")
        # No unit widget: a unit is never invented. The Wafer record's own unit is reused only when the
        # entered value equals that record (so the same record is reused exactly as before); otherwise none.
        unit=(existing["exposure_unit"] if existing and exposure is not None and existing["exposure_value"]==float(exposure)
              else None)
        peb=st.number_input("PEB Time (s, 선택)",value=common.get("peb_temperature_c"),
            step=1.0,format="%g",key="sqlite_peb_"+scope,disabled=not info.get("peb_supported",False),
            placeholder="미입력은 NULL",help="Batch 공통 공정 Metadata · PEB 시간(초). 입력한 숫자를 그대로 저장합니다.")
        if not info.get("peb_supported",False):
            st.caption("PEB 저장은 명시적 Schema v3 적용 후 사용할 수 있습니다.")
    common.update(wafer=wafer,exposure=None if exposure is None else float(exposure),unit=unit,mode=mode,dataset_role=dataset_role,
                  peb_temperature_c=peb)
    metadata={"batch_id":bundle["batch_id"],"wafer_id":wafer,"upload_mode":MODE_CODES[mode],
              "dataset_role":dataset_role.lower(),"batch_status":"registered",
              "process_run_id":existing["process_run_id"] if existing else bundle["process_run_id"],
              "notes":notes_with_batch_name(None,batch_name),"peb_temperature_c":peb}
    if exposure is not None:
        metadata["process_run"]={"exposure_value":float(exposure),"exposure_unit":unit or None}
    if batch_process:
        metadata["batch_process_run_id"]=bundle["batch_process_run_id"]
    return metadata,wafer


def _reset_draft():
    _clear_confirmations()
    old=st.session_state.pop("sqlite_uploads",None)
    if old:
        old["files"].cleanup()
    st.session_state.pop("sqlite_preview",None)
    st.session_state.pop("sqlite_batch_name",None)
    st.session_state["sqlite_uploader_generation"]=uuid.uuid4().hex


def render_registration(mode,dataset_role,upload_slot=None):
    provider=_provider()
    _synchronize_mode(mode)
    pending=st.session_state.get("sqlite_uploads",{})
    if "registered" not in pending:
        batch_name=st.text_input("Batch 이름 (선택)",value=pending.get("batch_name",""),
            placeholder="예: W5 중앙 반복촬영 3회",key="sqlite_batch_name",
            help="선택 · 등록 시 DB에 저장합니다. 자동 Batch ID와 번호는 그대로 유지됩니다.")
    else:
        batch_name=pending.get("batch_name","")
    # Reserved before variable-length details: the button never disappears.
    button_slot=st.container()
    plan=bundle=None
    reason=""
    try:
        plan,bundle,reason=_registration_body(provider,mode,dataset_role,upload_slot,batch_name)
    except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
        _clear_confirmations()
        st.session_state.pop("sqlite_preview",None)
        _show_error("입력·검증 실패.",exc)
        reason="오류를 해결한 뒤 다시 검증하세요."
    ready=bool(plan and plan["registration_ready"] and not reason)
    count=plan["selected_count"] if plan else 0
    label=f"이미지 {count}장 DB 등록"
    with button_slot:
        register_col,reset_col=st.columns([3,2.5],vertical_alignment="center")
        clicked=register_col.button(label,type="primary",disabled=not ready,key="sqlite_register",width="stretch")
        reset_col.button("새 업로드 준비 (입력 초기화)",
                         key="sqlite_new_batch",on_click=_reset_draft,width="stretch")
        if reason:
            st.caption(reason)
    if clicked:
        try:
            with st.spinner(f"SQLite 등록 중 · {count}장"):
                result=provider.register_selection(plan,approved=True)  # Click is explicit approval.
            bundle["registered"]=result
            st.session_state["sqlite_last_result"]=result
            from dashboard.components.waepi import event as waepi_event
            waepi_event("registered")   # Only after register_selection returned (committed).
            # Notified only after the registration call returned (committed).
            notify("SQLite 등록 완료",result.get("wafer_id") or bundle.get("wafer_label") or bundle["common_metadata"].get("wafer") or result["batch_id"],
                   "등록 완료",count=result.get("registered_count",result.get("image_count")),level="success",
                   key=("registered",result["batch_id"]))
            bundle["files"].cleanup()
            st.rerun()
        except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
            from dashboard.components.waepi import event as waepi_event
            waepi_event("register_failed")
            notify("SQLite 저장 실패","이미지 등록","실패 · 저장되지 않음",count=count,level="error",
                   key=("register_failed",uuid.uuid4().hex))
            _show_error("등록 실패.",exc)


def _repairable(row):
    if row["status"]=="NEEDS_METADATA":
        return True
    if row["status"]!="INVALID":
        return False
    return any(any(word in issue["message"] for word in ("Wafer","die","position","metadata","Metadata"))
               for issue in row["issues"])


def review_help(provider):
    """Tooltip text (moved from captions). Matches the actual registration rule of the DB schema."""
    base=("공통 공정/촬영 조건(노광·PEB 시간·촬영 배율·조명) 누락은 이미지 등록 탭의 공통 Metadata에서 보완하세요. "
          "이 탭에서는 이미지별 Wafer / Die / 촬영 위치만 수정합니다.")
    try:
        version=provider.schema_info().get("version",0)
    except (ValueError,OSError,RuntimeError,sqlite3.Error):
        version=0
    if version>=5:
        rule=("노광은 이 Batch의 실제 조건을 직접 입력하세요. 기존 Wafer 기록과 다른 노광은 이 Batch에만 저장되며 "
              "Wafer 기록은 바뀌지 않습니다.")
    else:
        rule="노광은 실제 조건을 입력하세요. 이 DB Schema에서는 기존 Wafer의 공정 조건과 일치해야 합니다."
    return base+" "+rule


def render_review():
    bundle=st.session_state.get("sqlite_uploads")
    preview=st.session_state.get("sqlite_preview")
    if not bundle or not preview:
        st.info("이미지 등록 탭에서 파일과 공통 Metadata를 준비하세요.")
        return
    if "registered" in bundle:
        st.info("등록 완료 · 아래 조회 목록에서 확인하세요.")
        return
    needs=[row for row in preview["images"] if _repairable(row)]
    counts=(f"전체 {len(preview['images'])}장 · 검증 완료 "
            f"{sum(row['status'] in ('VALID','DUPLICATE_CANDIDATE') for row in preview['images'])}장 · "
            f"수정 필요 {len(needs)}장")
    with st.container(horizontal=True,vertical_alignment="center",gap="medium",key="sqlite_review_counts"):
        show_all=st.checkbox("전체 이미지 보기",key="sqlite_review_all_"+bundle["batch_id"],width="content")
        st.caption(counts,width="content")
    rows=preview["images"] if show_all else needs
    if not rows:
        st.info("파일명 수정이 필요한 이미지가 없습니다. 중복 처리는 이미지 등록 탭에서 선택하세요.")
        return
    items={row["image_id"]:row for row in rows}
    left,right=st.columns([1,1.35])
    with left:
        selected=st.radio("업로드 이미지 확인",list(items),
            format_func=lambda key:_short_name(items[key]["original_filename"])+" · "+items[key]["status"]+" · "+key[-6:],
            key="sqlite_review_"+bundle["batch_id"],help=review_help(_provider()))
        row=items[selected]
        if row["parse_error"]:
            st.caption("파일명에서 촬영 위치를 확인할 수 없습니다. 아래 Metadata를 입력하세요.")
        # The exposure remedy now lives in the "업로드 이미지 확인" tooltip; other remedies stay visible.
        exposure_hint=issue_guidance("exposure_value")
        for guidance in dict.fromkeys(issue_guidance(issue["message"]) for issue in row["issues"]):
            if guidance!=exposure_hint:
                st.caption(guidance)
    values=bundle["manual_metadata"].get(selected) or row["effective_metadata"] or {}
    with right:
        with st.form("sqlite_edit_"+selected,border=False):
            wafer=st.text_input("Wafer ID",value=values.get("wafer_id") or bundle["common_metadata"].get("wafer",""),
                                key="sqlite_edit_wafer_"+selected,placeholder="필수 · 예: W1",help="필수 · 공통 Wafer와 일치해야 합니다.")
            dies=[None,*mock.REVIEW_DIE_CODES]
            positions=[None,"lt","c","rb"]
            die=st.selectbox("Die ID",dies,index=dies.index(values.get("die_code")),
                            format_func=lambda value:value or "확인 필요",key="sqlite_edit_die_"+selected,help="필수 · 11~18 / 21~28 / 31~38")
            position=st.selectbox("촬영 위치",positions,index=positions.index(values.get("position_code")),
                            format_func=lambda value:value or "확인 필요",key="sqlite_edit_position_"+selected,help="필수 · lt / c / rb")
            legacy=st.text_input("결함 코드 (선택)",value=values.get("legacy_defect_code") or "",
                                 key="sqlite_edit_legacy_"+selected,placeholder="선택 · 미상은 공란")
            submitted=st.form_submit_button("수정 후 Preview 재검증",width="stretch")
        if submitted:
            bundle["manual_metadata"][selected]={"wafer_id":wafer or None,"die_code":die,
                "position_code":position,"legacy_defect_code":legacy or None}
            st.session_state.pop("sqlite_preview",None)
            st.rerun()
    if st.button("선택 이미지 미리보기",key="sqlite_review_image_"+selected):
        st.image(row["image_path"],width="stretch")
    # Validation evidence, collapsed under the preview (moved from the left column; content unchanged).
    if row["issues"]:
        with st.expander("검증 기술 상세"):
            st.json([{"status":issue["status"],"message":_technical_message(issue["message"])}
                     for issue in row["issues"]])
    with st.expander("전체 파일명 및 입력 출처"):
        st.json({"filename":row["original_filename"],"source":row["metadata_source"],
                 "parsed":row["parsed_metadata"],
                 "archive_member":next((item.get("archive_member") for item in bundle["inputs"]
                                        if item["image_id"]==selected),None)})


def exposure_text(value,unit=None):
    """Condition label shared by the Wafer table and Wafer detail (same grouping, same counts).

    TASK18-B-18: value + display unit (exposure is a time in s; legacy mJ text is shown as s).
    """
    from dashboard.units import exposure_display
    return exposure_display(value,unit,missing="미확인")


def exposure_texts(frame):
    units=frame["exposure_unit"] if "exposure_unit" in frame else [None]*len(frame)
    return [exposure_text(v,u) for v,u in zip(frame["exposure_value"],units)]


def peb_text(value):
    return "—" if value is None or value!=value else f"{value:g}"


def condition_batches(batches,wafer_id,exposure,peb):
    """Batch × Wafer rows of exactly one Wafer × effective 노광 × PEB (labels from `condition_accumulation`).

    Pass provider.batch_wafers(): a multi-Wafer Batch contributes only its row for this Wafer.
    """
    if batches.empty:
        return batches
    import pandas as pd
    keep=((batches["wafer_id"]==wafer_id)&(pd.Series(exposure_texts(batches),index=batches.index)==exposure)
          &(batches["peb_temperature_c"].map(peb_text)==peb))
    return batches[keep]


def condition_accumulation(batches,analysed):
    """Per Wafer × effective 노광 × PEB: Batch/image/analysed counts (SQLite values only).

    `batches` = provider.batch_wafers() (one row per Batch × Wafer, image_count of THAT Wafer only) and
    `analysed` = analysed_per_batch_wafer(); so a multi-Wafer Batch is never counted twice per Wafer.
    """
    import pandas as pd
    frame=batches.assign(analysed=[int(analysed.get((b,w),0)) for b,w in zip(batches["batch_id"],batches["wafer_id"])],
        exposure=exposure_texts(batches),
        peb=batches["peb_temperature_c"].map(peb_text))
    if frame.empty:
        return pd.DataFrame(columns=["Wafer","노광","PEB (s)","Batch 수","등록 이미지 수","분석 결과 이미지 수"])
    table=frame.groupby(["wafer_id","exposure","peb"],sort=True).agg(
        batches=("batch_id","count"),images=("image_count","sum"),done=("analysed","sum")).reset_index()
    return table.rename(columns={"wafer_id":"Wafer","exposure":"노광","peb":"PEB (s)","batches":"Batch 수",
                                 "images":"등록 이미지 수","done":"분석 결과 이미지 수"})


def render_overview():
    try:
        provider=_provider()
        if os.environ.get("IDA_SHOW_DB_PATH")=="1":  # TASK18-B-11: Schema/DB caption removed; admin info kept.
            with st.expander("관리자 연결 정보"):
                st.code(str(provider.db_path.resolve()))
        batches=provider.batches()
        status=provider.system_status()
        analysed=analysed_per_batch(provider)
        members=provider.batch_wafers()
        mode_counts=batches.groupby("upload_mode")["image_count"].sum().to_dict()
        history=[]
        running=0
        done=0
        for row in batches.iloc[::-1].to_dict("records"):
            running+=row["image_count"]
            done+=analysed.get(row["batch_id"],0)
            history.append({"label":row["created_at"][:16].replace("T"," "),"image_count":running,"analysed_count":done})
        summary={"batch_count":status["batches"],"image_count":status["images"],
                 "analysed_count":status["analysed_images"],"review_count":status["review_images"],
                 "recent_image_count":int(batches.iloc[0]["image_count"]) if len(batches) else 0,
                 "mode_image_counts":{label:int(mode_counts.get(code,0)) for label,code in MODE_CODES.items()},
                 "history":history}
        accumulation_overview(summary,source="SQLite")
        conditions=condition_accumulation(members,analysed_per_batch_wafer(provider))
        st.dataframe(conditions,hide_index=True,width="stretch",
                     height=min(36*(len(batches)+1),260))
        st.caption("등록일 순 누적 · 분석 결과는 이미지별 최신 실행 1건 기준 (AI 잠정 · 공식 채택 아님) · "
                   "노광은 Batch 기록 우선, 없으면 Wafer 기록")
        # TASK18-B-11.1: step-flow and status tiles removed (display only; the KPIs above show the same counts).
    except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
        _show_error("등록 현황 조회 실패.",exc)


def render_progress():
    try:
        processing_progress(image_count=_provider().image_count())
    except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
        _show_error("등록 상태 조회 실패.",exc)


def render_registered_images(provider,batch_id=None,*,detailed=False):
    is_mock=provider is None
    source="Mock" if is_mock else "SQLite"
    labels={"전체":None,**MODE_CODES}
    filters,controls=st.columns([2,1])
    with filters:
        selected=st.selectbox("등록 유형 필터",list(labels),key="sqlite_image_filter_"+str(batch_id))
    mode=labels[selected]
    if is_mock:
        all_images=mock.images(batch_id,mode=selected if mode else None)
        count=len(all_images)
    else:
        count=provider.image_count(batch_id,mode)
    with controls:
        page_size=st.selectbox("표시 개수",[5,10,25],key="sqlite_image_size_"+str(batch_id))
    pages=max(1,math.ceil(count/page_size))
    caption_col,page_col=st.columns([2,1])
    caption_col.caption(f"등록 이미지 {count}장 · "+source)
    with page_col:
        page=st.selectbox("이미지 페이지",list(range(1,pages+1)),
                          key="sqlite_image_page_"+str(batch_id)+str(mode)+str(page_size))
    if is_mock:
        images=all_images.iloc[(page-1)*page_size:page*page_size]
        if not images.empty:
            images=images.assign(original_filename=images["asset"],position_code="미제공",
                                 die_code=images["die_id"],upload_mode=images["mode"],
                                 created_at=images["batch_id"].map(mock.batches().set_index("batch_id")["created_at"]),
                                 metadata_status=images["metadata_state_example"])
    else:
        images=provider.images(batch_id,upload_mode=mode,limit=page_size,offset=(page-1)*page_size)
    if images.empty:
        st.info("해당 유형의 등록 이미지가 없습니다.")
        return
    columns=["original_filename","wafer_id","die_code","position_code",
             "upload_mode","created_at","metadata_status"]
    if detailed and not is_mock:
        columns=["image_id","original_filename","wafer_id","die_code","position_code",
                 "physical_site_id","upload_mode","repeat_index","exposure_value","exposure_unit","peb_temperature_c",
                 "illumination_setting","magnification","auto_exposure","captured_at","created_at","metadata_status"]
    display=images[columns].copy()
    if not detailed:
        display["original_filename"]=display["original_filename"].map(_short_name)
    selectable=detailed and not is_mock
    table_key=f"sqlite_image_table_{batch_id}_{mode}_{page_size}_{page}"
    presented=display.style.format(na_rep="미기록",precision=3) if detailed and not is_mock else display
    event=st.dataframe(presented,hide_index=True,width="stretch",height=(len(display)+1)*35+3,
                 key=table_key if selectable else None,
                 on_select="rerun" if selectable else "ignore",selection_mode="single-cell" if selectable else "multi-row",
                 column_config={"original_filename":st.column_config.TextColumn("원본 파일명",width="medium"),
                     "wafer_id":"Wafer ID","die_code":"Die ID","position_code":"촬영 위치",
                     "upload_mode":"등록 유형","created_at":"등록일","metadata_status":"Metadata 상태",
                     "image_id":st.column_config.TextColumn("Image ID · 상세",help="Image ID 셀을 클릭하면 현재 탭에서 상세페이지를 엽니다.")
                        if selectable else "Image ID","physical_site_id":"Physical Site ID","repeat_index":"반복 번호",
                     "exposure_value":"노광","exposure_unit":"노광 단위","peb_temperature_c":"PEB (s)","illumination_setting":"조명",
                     "magnification":"배율","auto_exposure":"Auto Exposure","captured_at":"촬영 일시"})
    if selectable:
        cells=event["selection"]["cells"]
        if cells:
            index,column=cells[-1]
            if column=="image_id" and isinstance(index,int) and 0<=index<len(images):
                st.session_state["sqlite_image_table_to_clear"]=table_key
                open_image_detail(str(images.iloc[index]["image_id"]),batch_id)
    records={row["image_id"]:row for row in images.to_dict("records")}
    with st.expander("이미지 상세 정보"):
        chosen=st.selectbox("파일 보기",list(records),
                           format_func=lambda key:_short_name(records[key]["original_filename"])+" · "+key[-6:],
                           key="sqlite_file_view_"+str(batch_id)+str(mode))
        row=records[chosen]
        fields=("image_id","batch_id","original_filename","physical_site_id","status") if is_mock else (
            "image_id","batch_id","original_filename","wafer_id","die_code","position_code","metadata_status",
            "width_px","height_px","created_at","upload_mode","sha256","repeat_index",
            "illumination_setting","magnification","auto_exposure","peb_temperature_c")
        if detailed and not is_mock:
            fields=(*fields,"physical_site_id","exposure_value","exposure_unit","captured_at")
            st.session_state["selected_sqlite_image_id"]=chosen
            st.session_state["selected_sqlite_image_batch_id"]=batch_id
            st.text("Image ID: "+chosen)
            st.text("원본 파일명: "+row["original_filename"])
            st.caption(f"원본 크기 {row['width_px']} × {row['height_px']} px · 원본 비율 유지")
        st.json({key:row[key] for key in fields})
        opened_key="sqlite_opened_image_"+str(batch_id)+str(mode)
        if st.button("선택 파일 열기",key="sqlite_open_file_"+str(batch_id)+str(mode)):
            st.session_state[opened_key]=chosen
        if st.session_state.get(opened_key)==chosen:
            if is_mock:
                st.image(str(mock.asset(row["asset"])),width="stretch")
                st.caption("Mock 예시 이미지 · 실제 등록 영상 아님")
            else:
                render_original_preview(provider,row,verify=detailed)


def render_original_preview(provider,row,*,verify=True):
    """Shared read-only preview for Batch and Image detail; never rewrites originals."""
    try:
        path=Path(row["storage_path"]).resolve()
        if not path.is_file() or not path.is_relative_to(provider.storage_root.resolve()):
            st.warning("로컬 저장소에서 원본 파일을 찾거나 표시할 수 없습니다.")
        elif verify and path.name!=row["original_filename"]:
            st.warning("저장 파일명과 원본 파일명이 일치하지 않아 미리보기를 중단했습니다.")
        elif verify and hashlib.sha256(path.read_bytes()).hexdigest().lower()!=row["sha256"].lower():
            st.warning("원본 SHA-256이 등록 정보와 일치하지 않아 미리보기를 중단했습니다.")
        else:
            st.image(str(path),width="stretch")
    except (OSError,ValueError):
        st.warning("이 이미지의 원본을 읽을 수 없습니다. 다른 이미지를 선택하세요.")


def analysed_per_batch_wafer(provider):
    """{(batch_id, wafer_id): images whose latest stored analysis belongs to that Batch × Wafer}."""
    counts={}
    for item in provider.latest_analysis().values():
        key=(item["batch_id"],item.get("wafer_id"))
        counts[key]=counts.get(key,0)+1
    return counts


def analysed_per_batch(provider):
    """Images whose latest stored analysis belongs to each upload Batch (read-only)."""
    counts={}
    for item in provider.latest_analysis().values():
        counts[item["batch_id"]]=counts.get(item["batch_id"],0)+1
    return counts


def wafer_summary(batches,analysed):
    rows=[]
    for wafer,group in batches.groupby("wafer_id",sort=True):
        rows.append({"Wafer":wafer,"Batch 수":len(group),"등록 이미지 수":int(group["image_count"].sum()),
                     "분석 결과 이미지 수":int(sum(analysed.get(b,0) for b in group["batch_id"]))})
    return rows


def render_batches(*,source="SQLite"):
    try:
        provider=_provider() if source=="SQLite" else None
        batches=provider.batches() if provider else mock.batches()
        if provider is None:
            with st.expander("전체 등록 이미지 조회"):
                render_registered_images(provider)
            batch_table(batches,source=source)
            return
        if batches.empty:
            st.info("등록된 SQLite Batch가 없습니다.")
        else:
            analysed=analysed_per_batch(provider)
            # 1) Batch table: a 3-row-high scroll area holding every Batch (newest first).
            batch_table(batches,source=source,analysed=analysed,visible_rows=3)
            # 2) Wafer x effective exposure x PEB summary (same SQLite query as the accumulation table).
            table=condition_accumulation(provider.batch_wafers(),analysed_per_batch_wafer(provider)).rename(columns={"Wafer":"Wafer ID"})
            # TASK18-B-10A.2: a Wafer ID cell opens Wafer detail for exactly that row's 노광 × PEB condition
            # (same single-cell pattern as the Image ID cells of the image table).
            event=st.dataframe(table,hide_index=True,width="stretch",height=min(36*(len(table)+1)+2,146),
                         key="dcc_wafer_conditions",on_select="rerun",selection_mode="single-cell",
                         column_config={"Wafer ID":st.column_config.TextColumn("Wafer ID",
                             help="Wafer ID 셀을 클릭하면 이 행의 노광·PEB 조건으로 Wafer 상세를 엽니다.")})
            cells=event["selection"]["cells"] if event else []
            if cells:
                index,column=cells[-1]
                if column=="Wafer ID" and isinstance(index,int) and 0<=index<len(table):
                    row=table.iloc[index]
                    open_wafer_detail(str(row["Wafer ID"]),str(row["노광"]),str(row["PEB (s)"]))
        # 3) Full registered-image lookup at the bottom (unchanged feature and filters).
        with st.expander("전체 등록 이미지 조회"):
            render_registered_images(provider)
    except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
        _show_error("등록 목록 조회 실패.",exc)
