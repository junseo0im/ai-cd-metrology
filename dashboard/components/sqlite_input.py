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

from dashboard import data_provider as mock
from dashboard.sqlite_provider import runtime_provider,sync_uploads,reviewed_inputs
from dashboard.components.tables import batch_table
from dashboard.components.data_overview import accumulation_overview,processing_progress
from dashboard.components.pipeline import processing_pipeline,image_status_tiles
from database.batch_registry import DUPLICATE_ACTIONS
from database.batch_names import notes_with_batch_name
from database.filename_parser import FilenameParseError,parse_filename
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
    remedies=(
        (("Wafer mismatch","wafer_id mismatch","same wafer"),"공통 Wafer와 파일명/개별 Wafer가 다릅니다. 촬영 대상을 확인하고 Metadata를 수정하세요."),
        (("exposure_value","process_run"),"공통 Metadata에서 실제 노광 조건을 선택하세요. 기존 Wafer의 공정 조건과도 일치해야 합니다."),
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
    with (upload_slot or st).popover("파일 선택",width="stretch"):
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
    if not bundle:
        _clear_confirmations()
        return None,None,"이미지를 선택하세요. 필수 Metadata 입력 후 등록할 수 있습니다."
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
    initial_wafer=common.get("wafer",next(iter(inferred)) if len(inferred)==1 else "")
    initial_process=provider.wafer_process(initial_wafer)
    missing_common=not initial_wafer or (common.get("exposure") is None and initial_process is None)
    with st.expander("공통 Metadata",expanded=missing_common):
        st.caption("필수: Wafer ID · 신규 공정의 실제 노광 조건 / 선택: 노광 단위. Batch는 등록 작업 단위입니다.")
        wafer=st.text_input("Batch Wafer ID",value=initial_wafer,placeholder="필수 · 예: W1",
                            help="필수 · 촬영 대상 Wafer입니다. 파일명과 충돌하면 자동 덮어쓰지 않습니다.",
                            key="sqlite_wafer_"+scope)
        existing=provider.wafer_process(wafer)
        known_exposure=existing["exposure_value"] if existing else None
        # Unknown process values are never replaced by the Mock default.
        default=common.get("exposure",known_exposure)
        exposure_options=list(dict.fromkeys([None,50,60,70,80]+([default] if default is not None else [])))
        exposure=st.selectbox("노광 조건",exposure_options,index=exposure_options.index(default),
            format_func=lambda value:"선택 필요 · 실제 노광 조건" if value is None else str(value),help="신규 공정은 필수 · 실제 사용한 값만 선택하세요.",key="sqlite_exposure_"+scope)
        unit=st.text_input("노광 단위 (모르면 공란)",
            value=common.get("unit",(existing["exposure_unit"] or "") if existing else ""),key="sqlite_unit_"+scope,placeholder="선택 · 단위를 모르면 비워 두세요")
    common.update(wafer=wafer,exposure=exposure,unit=unit,mode=mode,dataset_role=dataset_role)
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
    metadata={"batch_id":bundle["batch_id"],"wafer_id":wafer,"upload_mode":MODE_CODES[mode],
              "dataset_role":dataset_role.lower(),"batch_status":"registered",
              "process_run_id":existing["process_run_id"] if existing else bundle["process_run_id"],
              "notes":notes_with_batch_name(None,batch_name)}
    if exposure is not None:
        metadata["process_run"]={"exposure_value":float(exposure),"exposure_unit":unit or None}
    preview=provider.preview(reviewed_inputs(bundle),metadata)
    st.session_state["sqlite_preview"]=preview
    token=_confirmation_scope(preview)
    decisions=_render_duplicate_warning(preview,provider,token)
    plan=provider.prepare_registration(preview,decisions)
    st.session_state["sqlite_registration_plan"]=plan
    duplicate_count=sum(bool(row["duplicate_image_ids"] or row["duplicate_input_indices"]) for row in preview["images"])
    errors=sum(row["status"] in ("NEEDS_METADATA","INVALID") for row in preview["images"])
    valid=sum(row["status"]=="VALID" for row in preview["images"])
    st.caption(f"전체 업로드 {len(preview['images'])} · 정상 검증 {valid} · 중복 {duplicate_count} · "
               f"오류·수정 필요 {errors} · 최종 등록 예정 {plan['selected_count']} · "
               f"제외 {len(plan['excluded_image_ids'])} · 중복 허용 등록 {len(plan['separate_image_ids'])}")
    with st.expander("검증 목록"):
        st.dataframe(_preview_rows(preview),hide_index=True,width="stretch")
    if mode=="Wafer Batch":
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
            result=provider.register_selection(plan,approved=True)  # Click is explicit approval.
            bundle["registered"]=result
            st.session_state["sqlite_last_result"]=result
            bundle["files"].cleanup()
            st.rerun()
        except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
            _show_error("등록 실패.",exc)


def _repairable(row):
    if row["status"]=="NEEDS_METADATA":
        return True
    if row["status"]!="INVALID":
        return False
    return any(any(word in issue["message"] for word in ("Wafer","die","position","metadata","Metadata"))
               for issue in row["issues"])


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
    st.caption(f"전체 {len(preview['images'])}장 · 검증 완료 "
               f"{sum(row['status'] in ('VALID','DUPLICATE_CANDIDATE') for row in preview['images'])}장 · "
               f"수정 필요 {len(needs)}장")
    show_all=st.checkbox("전체 이미지 보기",key="sqlite_review_all_"+bundle["batch_id"])
    rows=preview["images"] if show_all else needs
    if not rows:
        st.info("파일명 수정이 필요한 이미지가 없습니다. 중복 처리는 이미지 등록 탭에서 선택하세요.")
        return
    items={row["image_id"]:row for row in rows}
    left,right=st.columns([1,1.35])
    with left:
        selected=st.radio("업로드 이미지 확인",list(items),
            format_func=lambda key:_short_name(items[key]["original_filename"])+" · "+items[key]["status"]+" · "+key[-6:],
            key="sqlite_review_"+bundle["batch_id"])
        row=items[selected]
        st.caption("공통 공정/촬영 조건 누락은 이미지 등록 탭에서 보완하세요.")
        if row["parse_error"]:
            st.caption("파일명에서 촬영 위치를 확인할 수 없습니다. 아래 Metadata를 입력하세요.")
        for guidance in dict.fromkeys(issue_guidance(issue["message"]) for issue in row["issues"]):
            st.caption(guidance)
        if row["issues"]:
            with st.expander("검증 기술 상세"):
                st.json([{"status":issue["status"],"message":_technical_message(issue["message"])}
                         for issue in row["issues"]])
        with st.expander("전체 파일명 및 입력 출처"):
            st.json({"filename":row["original_filename"],"source":row["metadata_source"],
                     "parsed":row["parsed_metadata"],
                     "archive_member":next((item.get("archive_member") for item in bundle["inputs"]
                                            if item["image_id"]==selected),None)})
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


def render_overview():
    try:
        provider=_provider()
        _schema_notice(provider)
        batches=provider.batches()
        mode_counts=batches.groupby("upload_mode")["image_count"].sum().to_dict()
        history=[]
        running=0
        for row in batches.iloc[::-1].to_dict("records"):
            running+=row["image_count"]
            history.append({"label":row["created_at"],"image_count":running})
        summary={"batch_count":len(batches),"image_count":provider.image_count(),
                 "recent_image_count":int(batches.iloc[0]["image_count"]) if len(batches) else 0,
                 "mode_image_counts":{label:int(mode_counts.get(code,0)) for label,code in MODE_CODES.items()},
                 "history":history}
        accumulation_overview(summary,source="SQLite")
        processing_pipeline([
            {"stage":"업로드","detail":"SQLite 등록"},
            {"stage":"Metadata","detail":"등록 정보"},
            {"stage":"Calibration","detail":"미연결"},
            {"stage":"계측·AI","detail":"미연결"},
            {"stage":"DB","detail":"저장"},
        ])
        image_status_tiles([
            {"label":"이미지 등록","state":str(summary["image_count"])+"장","state_key":"registered"},
            {"label":"Calibration","state":"미연결","state_key":"registered"},
            {"label":"계측·AI","state":"미연결","state_key":"registered"},
        ],notice="실제 등록 Metadata · Calibration/계측·AI 결과 미연결")
    except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
        _show_error("등록 현황 조회 실패.",exc)


def render_progress():
    try:
        processing_progress(image_count=_provider().image_count())
    except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
        _show_error("등록 상태 조회 실패.",exc)


def render_registered_images(provider,batch_id=None):
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
    display=images[["original_filename","wafer_id","die_code","position_code",
                    "upload_mode","created_at","metadata_status"]].copy()
    display["original_filename"]=display["original_filename"].map(_short_name)
    st.dataframe(display,hide_index=True,width="stretch",height=(len(display)+1)*35+3,
                 column_config={"original_filename":st.column_config.TextColumn("원본 파일명",width="medium"),
                     "wafer_id":"Wafer ID","die_code":"Die ID","position_code":"촬영 위치",
                     "upload_mode":"등록 유형","created_at":"등록일","metadata_status":"Metadata 상태"})
    records={row["image_id"]:row for row in images.to_dict("records")}
    with st.expander("이미지 상세 정보"):
        chosen=st.selectbox("파일 보기",list(records),
                           format_func=lambda key:_short_name(records[key]["original_filename"])+" · "+key[-6:],
                           key="sqlite_file_view_"+str(batch_id)+str(mode))
        row=records[chosen]
        fields=("image_id","batch_id","original_filename","physical_site_id","status") if is_mock else (
            "image_id","batch_id","original_filename","wafer_id","die_code","position_code","metadata_status",
            "width_px","height_px","created_at","upload_mode","sha256","repeat_index",
            "illumination_setting","magnification","auto_exposure")
        st.json({key:row[key] for key in fields})
        opened_key="sqlite_opened_image_"+str(batch_id)+str(mode)
        if st.button("선택 파일 열기",key="sqlite_open_file_"+str(batch_id)+str(mode)):
            st.session_state[opened_key]=chosen
        if st.session_state.get(opened_key)==chosen:
            if is_mock:
                st.image(str(mock.asset(row["asset"])),width="stretch")
                st.caption("Mock 예시 이미지 · 실제 등록 영상 아님")
            else:
                path=Path(row["storage_path"]).resolve()
                if path.is_file() and path.is_relative_to(provider.storage_root.resolve()):
                    st.image(str(path),width="stretch")
                else:
                    st.info("로컬 저장소에서 표시할 수 없는 파일입니다.")


def render_batches(*,source="SQLite"):
    try:
        provider=_provider() if source=="SQLite" else None
        with st.expander("전체 등록 이미지 조회"):
            render_registered_images(provider)
        batches=provider.batches() if provider else mock.batches()
        if batches.empty:
            st.info("등록된 SQLite Batch 없음 · Mock Batch를 표시하지 않습니다.")
            return
        batch_table(batches,source=source)
    except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
        _show_error("등록 목록 조회 실패.",exc)
