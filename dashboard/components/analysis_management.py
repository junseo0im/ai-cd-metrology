"""Analysis-target preview plus gated manual runs of the unchanged BatchRunner.

The Runner executes in the separate model environment (subprocess), one BatchRunner run
per registered Batch. Only mapper-READY images run; more than CONFIRM_THRESHOLD need an
explicit, selection-bound confirmation. Results go to
the official result root; saves use AnalysisRepository (operational DB only with the
explicit switch). Nothing is adopted or fed to SPC.
"""
from datetime import datetime,timezone
from html import escape
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import uuid
import pandas as pd
import time
import streamlit as st
from dashboard.notifications import notify
from dashboard.sqlite_provider import runtime_provider
from database.analysis_repository import AnalysisRepository,AnalysisStorageError,OPERATIONAL_RESULT_ROOT
from database.migrate_v4 import refuse_operational
from dashboard.components.tables import data_table
from dashboard.components.data_overview import kpi_tiles,chips

SCOPES=("Batch","Wafer","Image")
ENGINE_NOTICE="실행 가능한 SQLite 분석 대상이 없습니다 — 범위와 대상을 선택하세요."
# Runs above this size need an explicit confirmation of the exact selection (not a cap).
CONFIRM_THRESHOLD=3
# Short policy tags (TASK18-B-11); the same rules are enforced in run_eligibility/run_selection.
RUN_TAGS=("READY만 실행","검토·차단 제외","공식 채택 별도")
CONFIRM_REASON="대량 실행 확인 필요"
CONDITION_REASON="공정조건을 선택하세요. 같은 Wafer의 다른 노광·PEB 조건은 합산하지 않습니다."
PER_IMAGE_TIMEOUT_SECONDS=120
ROOT=Path(__file__).resolve().parents[2]
# Model stack (torch/onnxruntime/cv2/supervision) used by the earlier verified runs.
DEFAULT_ANALYSIS_PYTHON=ROOT/"runtime_data/validation/task16b/.venv_defect/Scripts/python.exe"
# Official per-run result folders (TASK18-A-4); earlier validation folders stay untouched.
DEFAULT_OUTPUT_ROOT=OPERATIONAL_RESULT_ROOT
# Explicit switch: operational DB saves stay disabled unless this is exactly "1".
OPERATIONAL_SAVE_SWITCH="IDA_ANALYSIS_OPERATIONAL_SAVE"
RUN_TIMEOUT_SECONDS=1800
_JOB_LOCK=threading.Lock()
_ACTIVE={}  # Process-wide: at most one analysis subprocess at a time.


def analysis_python():
    configured=os.environ.get("IDA_ANALYSIS_PYTHON")
    path=Path(configured) if configured else DEFAULT_ANALYSIS_PYTHON
    return path if path.is_file() else None


def analysis_output_root():
    return Path(os.environ.get("IDA_ANALYSIS_OUTPUT_ROOT",DEFAULT_OUTPUT_ROOT))


def _analysis_command(python,request_path,output_root):
    return [str(python),"-m","inspection.run_dashboard_analysis",
            "--request",str(request_path),"--output-root",str(output_root)]


def _job_running():
    with _JOB_LOCK:
        process=_ACTIVE.get("process")
        return process is not None and process.poll() is None


def run_selection(plan):
    """READY requests grouped per Batch (deduplicated) plus excluded items with reasons."""
    groups=[];expected={};seen=set()
    for batch_plan in plan["batch_plans"]:
        requests=[]
        for request in batch_plan["ready_requests"]:
            if request["image_id"] in seen or request.get("batch_id")!=batch_plan["batch_id"]:
                continue
            seen.add(request["image_id"]);requests.append(request)
        if requests:
            groups.append({"batch_id":batch_plan["batch_id"],"requests":requests})
    for item in plan["items"]:
        if item["image_id"] in seen and item["validation_status"]=="READY":
            expected[item["image_id"]]=item["actual_sha256"]
    excluded=[{"image_id":i["image_id"],"status":i["validation_status"],
               "reason":" / ".join(x["message"] for x in i["issues"]) or "-"}
              for i in plan["items"] if i["validation_status"]!="READY"]
    return groups,expected,excluded


def _save_target_text():
    """What happens to results, for the large-run confirmation text."""
    try:
        refuse_operational(runtime_provider().db_path)
        return "결과는 현재 SQLite DB에 저장"
    except PermissionError:
        return ("결과는 운영 DB에 자동 저장" if os.environ.get(OPERATIONAL_SAVE_SWITCH)=="1"
                else "운영 DB 저장 비활성 · 결과 파일만 보존")


def selection_token(expected):
    """Stable token of the exact READY selection; a different selection needs a new confirmation."""
    import hashlib
    return hashlib.sha256("|".join(sorted(expected)).encode("utf-8")).hexdigest()[:16]


def repeated_images(expected,completed):
    """READY images of this selection already analysed in this session (the duplicate-run block set)."""
    done=set(completed or ())
    return [image_id for image_id in expected if image_id in done]


def duplicate_details(plan,completed):
    """Collapsed ID list of the blocked duplicates; height-limited, copyable."""
    _,expected,_=run_selection(plan)
    repeated=repeated_images(expected,completed)
    if not repeated:
        return
    with st.expander(f"중복 실행 대상 상세 보기 ({len(repeated)}장)",expanded=False):
        with st.container(height=min(220,40+22*len(repeated)),border=False,key="dcc_duplicate_ids"):
            st.code(chr(10).join(repeated),language=None)


def run_eligibility(mode,scope,preview,plan,*,completed=(),confirmed=False):
    """Return (groups, expected_sha256, reason); groups is None when blocked."""
    if mode!="수동 실행":
        return None,None,"자동 실행은 지원하지 않습니다. 수동 실행을 선택하세요."
    if preview and preview.get("condition_required"):
        return None,None,CONDITION_REASON
    if not preview or not preview.get("image_count"):
        return None,None,"분석할 등록 이미지를 선택하세요."
    if not plan:
        return None,None,"입력 검증(Schema v4) 결과가 없어 실행할 수 없습니다."
    if {i["image_id"] for i in plan["items"]}!=set(preview["image_ids"]):
        return None,None,"입력 검증 결과와 선택 대상이 일치하지 않습니다. 다시 선택하세요."
    groups,expected,excluded=run_selection(plan)
    ready=sum(len(g["requests"]) for g in groups)
    if ready==0:
        return None,None,f"실행 가능한 READY 이미지가 없습니다 (제외 {len(excluded)}장)."
    if ready>CONFIRM_THRESHOLD and confirmed is not True:
        return None,None,CONFIRM_REASON
    if set(expected)!={r["image_id"] for g in groups for r in g["requests"]}:
        return None,None,"READY 이미지의 원본 SHA-256 확인이 누락되었습니다."
    if analysis_python() is None:
        return None,None,"분석 엔진 실행 환경을 찾을 수 없습니다."
    if _job_running():
        return None,None,"다른 분석이 실행 중입니다. 완료 후 다시 시도하세요."
    repeated=repeated_images(expected,completed)
    if repeated:  # Block unchanged; the full ID list is shown on demand (duplicate_details).
        return None,None,f"이미 분석된 이미지 {len(repeated)}장 · 중복 실행 차단"
    return groups,expected,""


def start_analysis(groups,expected,confirmed_count=None,meta=None):
    """Start the model-environment subprocess; refuses while another run is active.

    `meta` (target label, owner session token) is display-only and lets any page show the progress.
    """
    python=analysis_python()
    if python is None:
        raise RuntimeError("분석 엔진 실행 환경을 찾을 수 없습니다.")
    with _JOB_LOCK:
        if _ACTIVE.get("process") is not None and _ACTIVE["process"].poll() is None:
            raise RuntimeError("다른 분석이 실행 중입니다.")
        _ACTIVE.clear()
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir=analysis_output_root()/("dcc-"+stamp+"-"+uuid.uuid4().hex[:8])
        run_dir.mkdir(parents=True,exist_ok=False)
        request_path=run_dir/"request.json"
        request_path.write_text(json.dumps({"groups":groups,"expected_sha256":expected,
            "confirmed_count":confirmed_count},ensure_ascii=False,indent=2),encoding="utf-8")
        log=(run_dir/"process.log").open("w",encoding="utf-8")
        environment={**os.environ,"PYTHONPATH":str(ROOT),"PYTHONUTF8":"1"}
        try:
            process=subprocess.Popen(_analysis_command(python,request_path,run_dir/"runs"),cwd=str(ROOT),
                env=environment,stdout=log,stderr=subprocess.STDOUT)
        except BaseException:
            log.close()
            raise
        requested={r["image_id"]:g["batch_id"] for g in groups for r in g["requests"]}
        _ACTIVE.update(process=process,log=log,run_dir=run_dir,requested=requested,
                       meta={**(meta or {}),"total":len(requested),"started_at":datetime.now().strftime("%H:%M:%S")})
        return run_dir


# ---- Shared job state (TASK18-B-11) ---------------------------------------------------------------
# The Runner is a separate OS process tracked process-wide in _ACTIVE, so it does not depend on the page
# that started it. A page change only interrupts the Streamlit script that was WAITING; the functions
# below let any page show progress and let exactly one session collect + save the finished run.
_LAST={}  # Last collected run (process-wide): {"result":..., "owner": session token}.


def session_token():
    return st.session_state.setdefault("_analysis_session_token",uuid.uuid4().hex)


def job_snapshot():
    """Read-only view of the active run (None when idle). Progress comes from per-image artifacts only."""
    with _JOB_LOCK:
        process=_ACTIVE.get("process");run_dir=_ACTIVE.get("run_dir")
        meta=dict(_ACTIVE.get("meta") or {});requested=dict(_ACTIVE.get("requested") or {})
    if process is None:
        return None
    total=meta.get("total") or len(requested)
    progress=run_progress(run_dir) if run_dir else {"done":0,"outcomes":{}}
    done=min(progress["done"],total) if total else progress["done"]
    return {"running":process.poll() is None,"target":meta.get("target") or "선택 대상","total":total,"done":done,
            "remaining":max(0,total-done),"failed":progress["outcomes"].get("FAILED",0),
            "outcomes":progress["outcomes"],"started_at":meta.get("started_at"),"owner":meta.get("owner")}


def finish_pending(provider):
    """Collect + save a FINISHED run once (claimed under the lock); None if running, idle or already claimed.

    A session that is still blocking in wait_for_analysis holds the `waiter` mark and collects itself.
    """
    with _JOB_LOCK:
        process=_ACTIVE.get("process")
        if process is None or process.poll() is None or _ACTIVE.get("waiter") or _ACTIVE.get("claimed"):
            return None
        _ACTIVE["claimed"]=True
        meta=dict(_ACTIVE.get("meta") or {})
    try:
        result=wait_for_analysis()
        if result is not None:
            result["db_save"]=save_analysis_result(result,provider)
    except Exception as exc:  # Never leave a claimed job behind.
        result={"execution_status":"FAILED","error":str(exc),"run_dir":"-","image_ids":[]}
    finally:
        with _JOB_LOCK:
            if _ACTIVE.get("process") is process:
                _ACTIVE.clear()
    if result is not None:
        result["target"]=meta.get("target")
        _LAST.update(result=result,owner=meta.get("owner"))
    return result


def merge_finished_into_session():
    """Give the starting session its finished run: E result, duplicate guard and completion alert (once)."""
    result=_LAST.get("result")
    if not result or _LAST.get("owner")!=session_token():
        return None
    merged=st.session_state.setdefault("_analysis_merged_runs",[])
    key=result.get("run_dir") or str(result.get("image_ids"))
    if key in merged:
        return None
    merged.append(key)
    st.session_state["dcc_analysis_last_run"]=result
    if result.get("execution_status") in ("COMPLETED","COMPLETED_WITH_ISSUES"):
        completed=st.session_state.setdefault("dcc_analysis_completed_ids",[])
        completed.extend(i for i in result.get("image_ids") or [] if i not in completed)
    notify_result(result,None)
    return result


def _progress_lines(snapshot):
    total=snapshot["total"] or 0
    st.markdown('<div class="dcc-status-line"><b>'+escape(snapshot["target"])+'</b> · '+
                ("실행 중" if snapshot["running"] else "종료 · 결과 확인·저장 중")+
                (" · 시작 "+escape(snapshot["started_at"]) if snapshot.get("started_at") else "")+'</div>',
                unsafe_allow_html=True)
    st.progress(min(1.0,snapshot["done"]/total) if total else 0.0,
                text=f"{snapshot['done']}/{total}장 · {round(100*snapshot['done']/total) if total else 0}%")
    kpi_tiles([("전체",f"{total}장"),("완료",f"{snapshot['done']}장","ready"),
               ("남은",f"{snapshot['remaining']}장"),("실패",f"{snapshot['failed']}장","blocked")],compact=True)


def _job_progress_body(*,global_bar=False):
    snapshot=job_snapshot()
    if snapshot is None:  # Collected elsewhere meanwhile: one app rerun removes this fragment.
        merge_finished_into_session()
        st.rerun(scope="app")
    if not snapshot["running"]:
        if finish_pending(runtime_provider()) is not None:
            merge_finished_into_session()
            st.rerun(scope="app")  # Refresh bell, D/E and the run button once the run is over.
        st.caption("분석 종료 · 결과 확인·저장 중")  # Another session/the waiting run is collecting it.
        return
    if global_bar:
        total=snapshot["total"] or 0
        st.progress(min(1.0,snapshot["done"]/total) if total else 0.0,
                    text=f"분석 중 · {snapshot['target']} · {snapshot['done']}/{total}장 · 남은 {snapshot['remaining']}장"
                         f" · 실패 {snapshot['failed']}장 · Data Control Center에서 상세 확인")
    else:
        _progress_lines(snapshot)


@st.fragment(run_every=2)
def _job_progress_fragment():
    _job_progress_body()


@st.fragment(run_every=3)
def _global_progress_fragment():
    _job_progress_body(global_bar=True)


def render_global_progress(active_page):
    """Header strip on every page except Data Control (which shows D). Renders nothing when idle."""
    finished=finish_pending(runtime_provider()) if job_snapshot() is not None else None
    merge_finished_into_session()
    snapshot=job_snapshot()
    if finished is not None or snapshot is None or active_page=="data_control":
        return
    with st.container(key="global_analysis_status"):
        _global_progress_fragment()


def _image_outcome(row):
    """Per-image view from the Runner's own row and result JSON; nothing inferred."""
    value={"image_id":row.get("image_id"),"outcome":row.get("outcome"),"engine_status":row.get("engine_status"),
           "pd_policy_state":row.get("pd_policy_state"),"defect_status":row.get("defect_status"),
           "metrology_status":row.get("metrology_status"),"reason":row.get("reason"),
           "error":row.get("error_message"),"result_json":row.get("result_json")}
    if row.get("result_json") and Path(row["result_json"]).is_file():
        result=json.loads(Path(row["result_json"]).read_text(encoding="utf-8"))
        defect=result.get("defect") or {};metrology=result.get("metrology") or {}
        summary_values=result.get("measurement_summary") or {}
        value.update(detections=len(defect.get("detections") or []),
            line_mean_um=(summary_values.get("line") or {}).get("mean_um"),
            gap_mean_um=(summary_values.get("gap") or {}).get("mean_um"),
            judgment=(metrology.get("judgment") or {}).get("status"),
            official_analysis_complete=result.get("official_analysis_complete"),
            overlays={k:(v or {}).get("path") for k,v in (result.get("overlays") or {}).items()})
    return value


def _overall(statuses):
    if not statuses:
        return "FAILED"
    if all(s=="COMPLETED" for s in statuses):
        return "COMPLETED"
    if all(s=="FAILED" for s in statuses):
        return "FAILED"
    if "INTERRUPTED" in statuses:
        return "INTERRUPTED"
    return "COMPLETED_WITH_ISSUES"


def collect_result(run_dir,returncode):
    """Read every Batch summary the Runner wrote; a single image keeps the flat shape."""
    summaries=[]
    for path in Path(run_dir).glob("runs/batch-*/batch_summary.json"):
        summaries.append((path,json.loads(path.read_text(encoding="utf-8"))))
    summaries.sort(key=lambda item:item[1].get("started_at") or "")
    outcome={"run_dir":str(run_dir),"returncode":returncode,"log":str(Path(run_dir)/"process.log"),"groups":[]}
    for path,summary in summaries:
        outcome["groups"].append({"batch_id":summary.get("batch_id"),"batch_run_id":summary.get("batch_run_id"),
            "summary_json":str(path),"execution_status":summary.get("execution_status"),
            "error":summary.get("batch_error"),"images":[_image_outcome(r) for r in summary.get("images",[])]})
    if not summaries:
        outcome.update(execution_status="FAILED",error="Runner summary was not created")
        return outcome
    outcome["execution_status"]=_overall([g["execution_status"] for g in outcome["groups"]])
    images=[i for g in outcome["groups"] for i in g["images"]]
    outcome["image_ids"]=[i["image_id"] for i in images]
    if len(outcome["groups"])==1 and len(images)==1:  # Unchanged single-image result shape.
        group=outcome["groups"][0]
        outcome.update(summary_json=group["summary_json"],**{k:v for k,v in images[0].items() if k!="error"},
                       error=images[0].get("error") or group.get("error"))
    return outcome


def run_progress(run_dir):
    """Approximate live progress from per-image artifacts only.

    batch_summary.json is NEVER opened here: BatchRunner replaces it atomically after each
    image, and on Windows an open reader makes that os.replace fail (stale checkpoint).
    result.json/failure.json are written once and never replaced. Final counts always
    come from the summaries after the process has exited.
    """
    done=0;outcomes={}
    for image_dir in Path(run_dir).glob("runs/batch-*/images/*"):
        if (image_dir/"failure.json").is_file():
            state="FAILED"
        elif (image_dir/"analysis"/"result.json").is_file():
            try:
                result=json.loads((image_dir/"analysis"/"result.json").read_text(encoding="utf-8"))
            except (OSError,ValueError):
                continue  # Still being written; counted on the next poll.
            pd_state=(result.get("pd_policy") or {}).get("state")
            state=("FAILED" if result.get("status")=="FAILED" else
                   pd_state if pd_state in ("MODEL_PD_EXCLUDED","POLICY_REVIEW_REQUIRED") else "MEASURED")
        else:
            continue
        done+=1;outcomes[state]=outcomes.get(state,0)+1
    return {"done":done,"outcomes":outcomes}


def wait_for_analysis(timeout=RUN_TIMEOUT_SECONDS,on_progress=None):
    with _JOB_LOCK:
        process=_ACTIVE.get("process");run_dir=_ACTIVE.get("run_dir");log=_ACTIVE.get("log")
        requested=dict(_ACTIVE.get("requested") or {})
        if not requested and _ACTIVE.get("image_id"):
            requested={_ACTIVE["image_id"]:_ACTIVE.get("batch_id")}
    if process is None:
        return None
    try:
        import time
        deadline=time.monotonic()+timeout
        while True:
            try:
                returncode=process.wait(timeout=1 if on_progress else max(0.0,deadline-time.monotonic()))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic()>=deadline:
                    process.kill();returncode=process.wait();break
                if on_progress:
                    on_progress(run_progress(run_dir))
    finally:
        if log is not None:
            log.close()
    with _JOB_LOCK:
        if _ACTIVE.get("process") is process:
            _ACTIVE.clear()
    result=collect_result(run_dir,returncode)
    result["requested"]={"images":requested}
    if len(requested)==1:
        image_id,batch_id=next(iter(requested.items()))
        result["requested"].update(image_id=image_id,batch_id=batch_id)
        result["image_id"]=result.get("image_id") or image_id
    result.setdefault("image_ids",list(requested))
    if returncode!=0 and result.get("execution_status")!="FAILED":
        result["execution_status"]="FAILED"
        result["error"]=result.get("error") or "Analysis process exited with code "+str(returncode)
    return result


def _expected_images(requested):
    if "image_id" in requested:  # Explicit single identity wins (also used by tests/overrides).
        return {requested.get("image_id"):requested.get("batch_id")}
    return dict(requested.get("images") or {})


def _save_group(repository,group,expected,operational):
    """Save one Batch summary (one transaction in the repository) and re-read every image."""
    try:
        saved=repository.save_batch(Path(group["summary_json"]))
    except Exception as exc:  # Analysis artifacts stay untouched; storage failure is separate.
        return {"status":"FAILED","batch_id":group["batch_id"],
                "message":"DB 저장 실패 (분석 결과 파일은 보존됨): "+type(exc).__name__+": "+str(exc)}
    status="SAVED" if saved["inserted"] else "ALREADY_SAVED"
    summary=json.loads(Path(group["summary_json"]).read_text(encoding="utf-8"))
    try:  # Re-read through the same repository and compare with the Runner's own JSON.
        for row in summary["images"]:
            stored=repository.get_analysis_run(row["analysis_run_id"])
            original=json.loads(Path(row["result_json"]).read_text(encoding="utf-8")) if row.get("result_json") else None
            if (stored is None or stored["raw_result"]!=original or stored["image_id"]!=row["image_id"]
                    or expected.get(row["image_id"])!=group["batch_id"]
                    or stored["outcome"]!=row["outcome"] or stored["is_adopted"]!=0):
                raise AnalysisStorageError("stored record does not match the analysis result")
    except Exception as exc:
        return {"status":"FAILED","batch_id":group["batch_id"],"batch_run_id":saved["batch_run_id"],
                "message":"저장 후 재조회 검증 실패 ("+status+"): "+type(exc).__name__+": "+str(exc)}
    entry={"status":status,"verified":True,"operational":operational,"batch_id":group["batch_id"],**saved}
    if len(summary["images"])==1:
        entry["analysis_run_id"]=summary["images"][0]["analysis_run_id"]
    return entry


def save_analysis_result(result,provider):
    """Persist verified Runner results via the existing AnalysisRepository, one save per Batch.

    Only normally finished Batch runs whose images exactly match the request are passed on.
    The repository re-verifies registration, hashes and status contradictions, writes
    is_adopted=0, is idempotent per batch_run_id and needs the explicit operational switch.
    A failure in one Batch never deletes another Batch's already-saved rows.
    """
    expected=_expected_images(result.get("requested") or {})
    if result.get("returncode")!=0 or result.get("execution_status") not in ("COMPLETED","COMPLETED_WITH_ISSUES"):
        return {"status":"NOT_SAVED","message":"분석이 정상 종료되지 않아 DB에 저장하지 않았습니다."}
    groups=result.get("groups") or []
    try:
        found={}
        for group in groups:
            summary=json.loads(Path(group["summary_json"]).read_text(encoding="utf-8"))
            if summary.get("batch_id")!=group["batch_id"]:
                raise ValueError("summary batch mismatch")
            for row in summary.get("images") or []:
                if row.get("image_id") in found:
                    raise ValueError("image appears in two Batch runs")
                found[row.get("image_id")]=summary["batch_id"]
        if not groups or found!=expected or (len(expected)==1 and result.get("image_id") not in expected):
            return {"status":"NOT_SAVED","message":"분석 결과의 Image/Batch ID가 실행 요청과 달라 저장하지 않았습니다."}
    except (KeyError,TypeError,OSError,ValueError) as exc:
        return {"status":"NOT_SAVED","message":"분석 결과 파일을 확인할 수 없어 저장하지 않았습니다: "+str(exc)}
    try:
        refuse_operational(provider.db_path)
        operational=False
    except PermissionError:
        operational=True
        if os.environ.get(OPERATIONAL_SAVE_SWITCH)!="1":
            return {"status":"DISABLED","message":"운영 DB 결과 저장은 승인 전까지 비활성입니다. 결과 파일만 보존합니다."}
    try:
        # The repository additionally requires the official result root for operational use.
        repository=AnalysisRepository(provider.db_path,result_root=analysis_output_root(),
                                      image_root=provider.storage_root,allow_operational=operational)
    except Exception as exc:
        return {"status":"FAILED","message":"DB 저장 실패 (분석 결과 파일은 보존됨): "+type(exc).__name__+": "+str(exc)}
    entries=[]
    for group in groups:
        if group["execution_status"] not in ("COMPLETED","COMPLETED_WITH_ISSUES"):
            entries.append({"status":"NOT_SAVED","batch_id":group["batch_id"],
                            "message":"이 Batch 실행이 정상 종료되지 않아 저장하지 않았습니다."})
        else:
            entries.append(_save_group(repository,group,expected,operational))
    if len(entries)==1:
        return entries[0]
    statuses={e["status"] for e in entries}
    if "FAILED" in statuses:
        status="FAILED"
    elif statuses=={"ALREADY_SAVED"}:
        status="ALREADY_SAVED"
    elif statuses<={"SAVED","ALREADY_SAVED"}:
        status="SAVED"
    elif statuses=={"NOT_SAVED"}:
        status="NOT_SAVED"
    else:
        status="PARTIAL"
    saved_runs=[e["batch_run_id"] for e in entries if e["status"] in ("SAVED","ALREADY_SAVED")]
    return {"status":status,"operational":operational,"batches":entries,
            "verified":all(e.get("verified") for e in entries if e["status"] in ("SAVED","ALREADY_SAVED")),
            "message":f"Batch {len(entries)}개 중 저장·확인 {len(saved_runs)}개"}


# ---- Save-in-progress mark (TASK18-B-17, display only) ---------------------------------------------
# Wraps the unchanged save above so the '웨피' can show '결과 저장 중' only while a save really runs.
# The save logic, its inputs and its result are not touched.
_save_analysis_result_impl=save_analysis_result
_SAVING={"count":0}


def saving_in_progress():
    with _JOB_LOCK:
        return _SAVING["count"]>0


def save_analysis_result(result,provider):  # noqa: F811 (observation wrapper)
    with _JOB_LOCK:
        _SAVING["count"]+=1
    try:
        try:
            from dashboard.components.waepi import push
            push({"text":"결과 저장 중","tone":"saving","pct":None})
        except Exception:
            pass
        return _save_analysis_result_impl(result,provider)
    finally:
        with _JOB_LOCK:
            _SAVING["count"]-=1


SAVE_LABELS={"SAVED":"DB 저장 완료 · 재조회 확인","ALREADY_SAVED":"이미 저장된 실행 · 중복 저장하지 않음",
             "NOT_SAVED":"DB 저장 안 함","DISABLED":"DB 저장 비활성","FAILED":"DB 저장 실패",
             "PARTIAL":"일부 Batch만 저장"}


REVIEW_OUTCOMES=("QUALITY_REVIEW_REQUIRED","PARTIAL_MEASUREMENT","POLICY_REVIEW_REQUIRED")


def result_counts(result):
    """Counts straight from the Runner's per-image outcomes (the rows that were saved and re-verified)."""
    images=[i for g in result.get("groups") or [] for i in g["images"]]
    outcomes=[i.get("outcome") for i in images]
    return {"images":len(images),"complete":outcomes.count("COMPLETE"),
            "review":sum(o in REVIEW_OUTCOMES for o in outcomes),
            "pd_excluded":outcomes.count("MODEL_PD_EXCLUDED"),"failed":outcomes.count("FAILED")}


def _render_run_result(result):
    status=result.get("execution_status")
    groups=result.get("groups") or []
    images=[i for g in groups for i in g["images"]]
    counts=result_counts(result)
    line=(f"{status} · {result.get('target') or _target_label(groups)} · {counts['images']}장"
          +(f" · Batch {len(groups)}개" if len(groups)>1 else ""))
    if status=="COMPLETED":
        st.success(line)
    elif status=="COMPLETED_WITH_ISSUES":
        st.warning(line)
    else:
        st.error(line+(" · "+str(result["error"]) if result.get("error") else ""))
    if images:
        kpi_tiles([("완료",counts["complete"],"ready"),("검토 필요",counts["review"],"review"),
                   ("PD 계측 제외",counts["pd_excluded"]),("실패",counts["failed"],"blocked")],compact=True)
    saved=result.get("db_save")
    if saved:
        detail=saved.get("batch_run_id") or saved.get("message") or ""
        st.caption("저장 상태 · "+SAVE_LABELS.get(saved["status"],saved["status"])+(" · "+detail if detail else ""))
    with st.expander(f"이미지별 결과 ({len(images)}장)",expanded=False):
        if images:
            by_batch={e.get("batch_id"):e["status"] for e in (saved or {}).get("batches") or []}
            def save_state(batch_id):
                state=by_batch.get(batch_id) or ((saved or {}).get("status") if not (saved or {}).get("batches") else None)
                return SAVE_LABELS.get(state,state or "-")
            data_table(pd.DataFrame([{"Image ID":i["image_id"],"Batch":g["batch_id"],"결과":i["outcome"],
                "PD":i.get("pd_policy_state") or "-","결함":i.get("detections","-"),"계측":i.get("metrology_status") or "-",
                "Line (µm)":i.get("line_mean_um"),"Gap (µm)":i.get("gap_mean_um"),"DB 저장":save_state(g["batch_id"])}
                for g in groups for i in g["images"]]),height=min(200,(len(images)+1)*31+3))
        st.caption("결함 0건은 품질 PASS가 아닙니다. 자동 공식 채택 없음(is_adopted=0) · 저장된 결과는 Inspection & Analysis에서 조회 · "
                   "결과 폴더: "+str(result.get("run_dir")))


def collect_images(provider,*,batch_id=None,wafer_id=None,image_id=None):
    """Reuse the paged provider without a 72-image or 100-image ceiling."""
    chunks=[];offset=0
    while True:
        frame=provider.images(batch_id,wafer_id=wafer_id,image_id=image_id,limit=100,offset=offset)
        chunks.append(frame)
        if len(frame)<100:
            break
        offset+=100
    return pd.concat(chunks,ignore_index=True)


def _number(value):
    return None if value is None or value!=value else float(value)


def image_condition(exposure,unit,peb):
    """Process-condition identity of one image: (노광 값, 노광 단위, PEB s). NULL stays NULL (no unit guess)."""
    unit=None if unit is None or unit!=unit or str(unit).strip()=="" else str(unit)
    return json.dumps([_number(exposure),unit,_number(peb)])


def condition_label(condition):
    exposure,unit,peb=json.loads(condition)
    from dashboard.units import exposure_display
    text="노광 "+exposure_display(exposure,unit,missing="미확인")
    return text+" · PEB "+("미입력" if peb is None else f"{peb:g} s")


def _conditions_of(images):
    return [image_condition(e,u,p) for e,u,p in
            zip(images["exposure_value"],images["exposure_unit"],images["peb_temperature_c"])]


def wafer_conditions(images):
    """Distinct per-image (노광, 단위, PEB) of the given images with counts; never merged."""
    counts={}
    for condition in _conditions_of(images):
        counts[condition]=counts.get(condition,0)+1
    def order(condition):
        exposure,unit,peb=json.loads(condition)
        return (exposure is None,exposure or 0,unit or "",peb is None,peb or 0)
    return [{"id":c,"label":condition_label(c),"image_count":counts[c]} for c in sorted(counts,key=order)]


def target_preview(provider,scope,target_id,extra_image_ids=(),condition=None):
    """Identity-based read only; equal hashes/Sites remain distinct Image IDs.

    Wafer scope is one Wafer × one process condition (노광·단위·PEB from each image's own Batch-Wafer).
    With several conditions and none chosen the preview is marked `condition_required` and can never run.
    """
    if scope not in SCOPES or not isinstance(target_id,str) or not target_id:
        raise ValueError("분석 범위와 실제 대상 ID를 선택하세요.")
    conditions=None;required=False
    if scope=="Batch":
        record=provider.batch_record(target_id)
        if record is None:
            raise ValueError("현재 SQLite에 해당 Batch가 없습니다.")
        name=record["display_name"]
        images=collect_images(provider,batch_id=target_id)
    elif scope=="Wafer":
        if target_id not in provider.wafers()["wafer_id"].tolist():
            raise ValueError("현재 SQLite에 해당 Wafer가 없습니다.")
        name=target_id
        images=collect_images(provider,wafer_id=target_id)
        conditions=wafer_conditions(images)
        if condition is None and len(conditions)==1:
            condition=conditions[0]["id"]  # Only one condition exists: nothing to choose.
        if condition is None:
            required=bool(conditions)
        elif condition not in {c["id"] for c in conditions}:
            raise ValueError("선택한 공정조건이 이 Wafer에 없습니다. 다시 선택하세요.")
        else:
            images=images[[c==condition for c in _conditions_of(images)]]
            name=target_id+" · "+condition_label(condition)
    else:
        ids=list(dict.fromkeys([target_id,*extra_image_ids]))  # Duplicate selections removed.
        frames=[collect_images(provider,image_id=image_id) for image_id in ids]
        if any(frame.empty for frame in frames):
            raise ValueError("현재 SQLite에 해당 Image가 없습니다.")
        images=pd.concat(frames,ignore_index=True)
        name=images.iloc[0]["original_filename"]+(f" 외 {len(ids)-1}장" if len(ids)>1 else "")
    images=images.drop_duplicates("image_id",ignore_index=True)
    return {"source":"SQLite","scope":scope,"target_id":target_id,"target_name":name,
            "image_count":len(images),"image_ids":images["image_id"].tolist(),
            "wafer_ids":sorted(images["wafer_id"].unique().tolist()),
            "batch_ids":sorted(images["batch_id"].unique().tolist()),"images":images,
            "condition":condition if scope=="Wafer" else None,"conditions":conditions,
            "condition_required":required}


def prepare_selected_inputs(provider,preview):
    """Read-only plans for the selected Batch/Wafer/Image, retaining Batch boundaries."""
    if preview.get("source")!="SQLite":
        raise ValueError("SQLite 등록 대상만 입력 검증할 수 있습니다.")
    if preview.get("condition_required"):
        raise ValueError("Wafer 공정조건을 선택해야 입력 검증을 할 수 있습니다.")
    plans=[]
    for batch_id,frame in preview["images"].groupby("batch_id",sort=False):
        plans.append(provider.prepare_analysis_inputs(batch_id,image_ids=frame["image_id"].tolist()))
    items=[item for plan in plans for item in plan["items"]]
    counts={key:sum(i["validation_status"]==key for i in items) for key in ("READY","REVIEW_REQUIRED","BLOCKED")}
    return {"batch_plans":plans,"items":items,"counts":counts,"execution_enabled":False}

def mode_groups(preview,plan):
    """Repeatability groups (same Wafer/Die/Site) and Lighting groups (illumination); display only."""
    images=preview["images"];status={i["image_id"]:i["validation_status"] for i in plan["items"]}
    lines=[]
    def counts(frame):
        found={}
        for image_id in frame["image_id"]:
            found[status.get(image_id,"-")]=found.get(status.get(image_id,"-"),0)+1
        return " / ".join(f"{k} {v}" for k,v in sorted(found.items()))
    repeat=images[images["upload_mode"]=="repeatability"]
    for (wafer,die,site),frame in repeat.groupby(["wafer_id","die_code","position_code"],sort=True):
        indices=", ".join(str(int(v)) for v in sorted(frame["repeat_index"].dropna()))
        lines.append(f"반복 촬영 · {wafer} / Die {die} / {site} · 반복 {indices or '미기록'} · {len(frame)}장 · {counts(frame)}")
    light=images[images["upload_mode"]=="lighting"]
    for setting,frame in light.groupby(light["illumination_setting"].fillna("미기록").astype(str),sort=True):
        lines.append(f"조명 조건 · {setting} · {len(frame)}장 · {counts(frame)}")
    return lines


def _render_input_readiness(provider,preview):
    plan=prepare_selected_inputs(provider,preview)
    st.session_state["dcc_analysis_input_plan"]=plan
    counts=plan["counts"]
    groups,_,excluded=run_selection(plan)
    ready=sum(len(g["requests"]) for g in groups)
    kpi_tiles([("대상",f"{preview['image_count']}장"),("READY · 실행",ready,"ready","실제 분석 요청으로 전달되는 이미지 수"),
               ("REVIEW_REQUIRED",counts["REVIEW_REQUIRED"],"review"),("BLOCKED",counts["BLOCKED"],"blocked")])
    st.caption(f"실행 대상 READY {ready}장 · 실행 Batch {len(groups)}개 · 제외 {len(excluded)}장"
               +(" (사유: 입력 준비 근거)" if excluded else ""))
    groups_text=mode_groups(preview,plan)
    if groups_text:
        with st.expander("반복 / 조명 그룹",expanded=False):
            for line in groups_text:
                st.caption(line)
            st.caption("반복·조명 이미지는 기존 정책상 분석 범위 검토가 필요해 자동 실행 대상에서 제외됩니다.")
    with st.expander("입력 준비 근거",expanded=False):
        if plan["items"]:
            rows=[]
            for item in plan["items"]:
                request=item.get("candidate_request") or {}
                calibration=request.get("calibration") or {}
                rows.append({"Image ID":item["image_id"],"입력 상태":item["validation_status"],
                    "촬영조건":request.get("condition_id") or "미등록",
                    "Calibration":calibration.get("calibration_id") or "미확인",
                    "사유":" / ".join(i["message"] for i in item["issues"]) or "승인 촬영조건·Calibration 일치 / 경로·SHA 검증"})
            data_table(pd.DataFrame(rows),height=min(240,(len(rows)+1)*31+3))
        else:
            st.caption("입력 검증 대상 없음")

def _valid_selection(key,options):
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state.pop(key,None)


def _image_options(provider):
    rows=collect_images(provider).to_dict("records")
    return {r["image_id"]:r["original_filename"]+" · "+r["image_id"]+" · "+
            r["wafer_id"]+" / Die "+r["die_code"]+" / "+r["position_code"] for r in rows}


def _selection(provider,scope):
    if scope=="Batch":
        rows=provider.batches().to_dict("records")
        options={r["batch_id"]:r["display_name"]+" · "+r["batch_id"]+
                 " · "+str(r.get("wafer_list") or r["wafer_id"])+" · "+r["upload_mode"] for r in rows}
        label="분석 Batch"
    elif scope=="Wafer":
        options={key:key for key in provider.wafers()["wafer_id"].tolist()}
        label="분석 Wafer"
    else:
        options=_image_options(provider)
        label="분석 Image"
    key="dcc_analysis_target_"+scope.lower()
    _valid_selection(key,options)
    return st.selectbox(label,list(options),format_func=lambda identity:options[identity],key=key,
                        index=0 if options else None,disabled=not options,
                        placeholder="등록 대상 없음")


CONDITION_KEY="dcc_analysis_condition"


def condition_widget_key(wafer_id):
    return CONDITION_KEY+"_"+wafer_id


def _condition_selection(conditions,wafer_id):
    """Explicit 노광·PEB condition of the chosen Wafer; reset whenever the Wafer changes.

    One widget per Wafer (a shared key let the browser keep showing the previous Wafer's condition).
    """
    previous=st.session_state.get(CONDITION_KEY+"_wafer")
    key=condition_widget_key(wafer_id)
    if previous!=wafer_id:
        if previous:
            st.session_state.pop(condition_widget_key(previous),None)
        st.session_state.pop(key,None)
        st.session_state[CONDITION_KEY+"_wafer"]=wafer_id
    options={c["id"]:f"{c['label']} · {c['image_count']}장" for c in conditions}
    _valid_selection(key,options)
    return st.selectbox("공정조건 (노광 · PEB)",list(options),format_func=lambda c:options[c],key=key,
                        index=0 if len(options)==1 else None,disabled=not options,
                        placeholder=f"조건 {len(options)}개 중 선택 필요",
                        help="선택한 Wafer × 노광 × PEB 조건의 이미지만 분석 대상이 됩니다. 노광 단위 미기록은 사용자 확인 단위 mJ/cm²로 표시합니다 (DB 값 변경 없음).")


def _extra_images(provider,primary):
    """Optional additional Images (Image scope only); never includes the primary."""
    options={k:v for k,v in _image_options(provider).items() if k!=primary}
    key="dcc_analysis_extra_images"
    if key in st.session_state:
        st.session_state[key]=[v for v in st.session_state[key] if v in options]
    return st.multiselect("추가 Image (선택)",list(options),format_func=lambda identity:options[identity],
                          key=key,placeholder="1장만 분석하면 비워 두세요")


def _target_context(preview):
    """Batch·Wafer·조건 of the selected images (per image; a multi-Wafer Batch lists each Wafer)."""
    images=preview["images"]
    if images.empty:
        return []
    pairs={}
    for wafer,condition in zip(images["wafer_id"],_conditions_of(images)):
        pairs[(wafer,condition)]=pairs.get((wafer,condition),0)+1
    return [f"{wafer} · {condition_label(condition)} · {count}장" for (wafer,condition),count in sorted(pairs.items())]


def _target_label(groups):
    batches=[g["batch_id"] for g in groups or []]
    if not batches:
        return "선택 대상"
    return batches[0][:14]+("…" if len(batches[0])>14 else "")+(f" 외 {len(batches)-1}개 Batch" if len(batches)>1 else "")


def notify_result(result,groups):
    """Toast/bell only after the run finished and the DB save outcome is known."""
    if not result:
        return
    status=result.get("execution_status")
    ids=result.get("image_ids") or [result.get("image_id")]
    target=result.get("target") or _target_label(groups or result.get("groups"))
    key=result.get("run_dir") or str(ids)
    images=[i for g in result.get("groups") or [] for i in g["images"]]
    failed=sum(i.get("outcome")=="FAILED" for i in images)
    if status=="COMPLETED":
        notify("분석 완료",target,"완료",count=len(ids),level="success",key=("analysis_done",key))
    elif status=="COMPLETED_WITH_ISSUES" and failed:
        notify("부분 실패",target,f"실패 {failed}장 포함",count=len(ids),level="warning",key=("analysis_done",key))
    elif status=="COMPLETED_WITH_ISSUES":
        notify("분석 완료",target,"완료 · 검토 항목 있음",count=len(ids),level="success",key=("analysis_done",key))
    else:
        notify("분석 실패",target,str(status or "FAILED"),count=len(ids),level="error",key=("analysis_done",key))
    saved=result.get("db_save") or {}
    # PARTIAL alone is not a save failure (fully failed Batches are intentionally NOT_SAVED).
    if saved.get("status")=="FAILED" or any(e.get("status")=="FAILED" for e in saved.get("batches") or []):
        notify("SQLite 저장 실패",target,SAVE_LABELS.get(saved["status"],saved["status"]),count=len(ids),
               level="error",key=("save_failed",key))


def _group_title(box,letter,title):
    box.markdown('<div class="dcc-group-title">'+letter+'<span>|</span>'+escape(title)+'</div>',unsafe_allow_html=True)


def _field_label(text):
    st.markdown('<div class="dcc-field-label">'+escape(text)+'</div>',unsafe_allow_html=True)


def render_analysis_management(source):
    # Never leave SQLite rows visible when switching to Mock or a new empty scope.
    st.session_state["dcc_analysis_preview"]=None
    st.session_state["dcc_analysis_input_plan"]=None
    # A 실행 설정 · B 분석 대상 (left) / C 실행 · D 진행 상태 · E 완료 결과 (right). Logic order unchanged.
    left,right=st.columns(2,gap="medium")
    box_a=left.container(border=True,key="dcc_group_a");box_b=left.container(border=True,key="dcc_group_b")
    box_c=right.container(border=True,key="dcc_group_c");box_d=right.container(border=True,key="dcc_group_d")
    box_e=right.container(border=True,key="dcc_group_e")
    for box,letter,title in ((box_a,"A","실행 설정"),(box_b,"B","분석 대상"),(box_c,"C","실행"),
                             (box_d,"D","진행 상태"),(box_e,"E","완료 결과")):
        _group_title(box,letter,title)
    with box_a:
        mode_col,scope_col=st.columns([1,1.25],gap="small")
        with mode_col.container(key="dcc_setting_mode"):
            _field_label("실행 방식")
            mode=st.radio("실행 방식",["수동 실행","자동 실행"],horizontal=True,key="dcc_analysis_execution",
                          label_visibility="collapsed")
        with scope_col.container(key="dcc_setting_scope"):
            _field_label("분석 대상 범위")
            scope=st.radio("분석 대상 범위",SCOPES,horizontal=True,key="dcc_analysis_scope",disabled=source!="SQLite",
                           label_visibility="collapsed")
        if mode=="자동 실행":
            st.caption("자동 실행 미지원 · 선택해도 분석을 시작하거나 예약하지 않습니다.")
    if source!="SQLite":
        box_b.info("SQLite 데이터 소스에서 실제 등록된 분석 대상을 확인할 수 있습니다. Mock 분석 대상·결과는 생성하지 않습니다.")
    else:
        with box_b:
            try:
                provider=runtime_provider()
                schema=provider.schema_info()
                selected=_selection(provider,scope)
                if selected is None:
                    st.info("등록된 분석 대상 없음 · 대상 이미지 0장")
                else:
                    extras=_extra_images(provider,selected) if scope=="Image" else ()
                    condition=None
                    if scope=="Wafer":
                        whole=target_preview(provider,"Wafer",selected)
                        condition=_condition_selection(whole["conditions"],selected)
                    preview=target_preview(provider,scope,selected,extras,condition=condition)
                    st.session_state["dcc_analysis_preview"]=preview
                    st.markdown('<div class="dcc-target-name">'+escape(scope+" · "+str(preview["target_name"]))+'</div>',
                                unsafe_allow_html=True)
                    if preview["condition_required"]:
                        kpi_tiles([("Wafer 전체",f"{preview['image_count']}장"),
                                   ("공정조건",f"{len(preview['conditions'])}개","review","조건을 선택해야 실행 대상이 정해집니다.")])
                        st.caption("공정조건 선택 필요 · 조건을 고르면 해당 조건 이미지의 READY 수가 표시됩니다.")
                    else:
                        context=_target_context(preview)
                        if context:
                            chips("Wafer · 조건",context,note=f"관련 Batch {len(preview['batch_ids'])}개")
                        if schema["version"] in (4,5,6):
                            _render_input_readiness(provider,preview)
                        if not preview["image_count"]:
                            st.info("선택 범위에 등록 이미지 없음")
                    with st.expander("대상 이미지 전체 보기 ("+str(preview["image_count"])+"장)",expanded=False):
                        if preview["condition_required"]:
                            st.caption("공정조건을 선택하면 대상 이미지가 표시됩니다.")
                        elif preview["image_count"]:
                            columns={
                                "image_id":"Image ID","original_filename":"파일명","wafer_id":"Wafer ID",
                                "die_code":"Die 번호","position_code":"Site","batch_id":"Batch ID",
                                "upload_mode":"등록 유형","metadata_status":"Metadata 상태",
                                "exposure_value":"노광","exposure_unit":"노광 단위","peb_temperature_c":"PEB (s)",
                                "magnification":"배율","illumination_setting":"조명","auto_exposure":"Auto Exposure",
                                "repeat_index":"반복 번호","captured_at":"촬영 일시"}
                            display=preview["images"][list(columns)].rename(columns=columns)
                            display=display.map(lambda value:"정보 없음" if pd.isna(value) else str(value))
                            data_table(display,height=min(360,(len(display)+1)*31+3))
                        else:
                            st.caption("표시할 이미지 정보가 없습니다.")
            except (ValueError,OSError,RuntimeError,sqlite3.Error) as exc:
                st.session_state["dcc_analysis_preview"]=None
                st.session_state["dcc_analysis_input_plan"]=None
                st.error("SQLite 대상 조회 실패. 선택 항목과 연결을 다시 확인하세요. "+str(exc))
    # A run that finished while this session was elsewhere (or collected by another session).
    if source=="SQLite":
        if job_snapshot() is not None:
            finish_pending(runtime_provider())
        merge_finished_into_session()
    completed=st.session_state.setdefault("dcc_analysis_completed_ids",[])
    groups=expected=None;reason=ENGINE_NOTICE
    plan=st.session_state.get("dcc_analysis_input_plan")
    preview=st.session_state.get("dcc_analysis_preview")
    confirmed=False;large=0
    if source=="SQLite" and plan and mode=="수동 실행":
        pending_groups,pending_expected,_=run_selection(plan)
        large=sum(len(g["requests"]) for g in pending_groups)
        if large>CONFIRM_THRESHOLD:
            # Key = exact READY selection: another target or condition gets a fresh, unchecked box.
            confirmed=box_c.checkbox(f"READY {large}장 · 대량 분석 실행 확인",
                key="dcc_analysis_confirm_"+selection_token(pending_expected),
                help=(f"Batch {len(pending_groups)}개 · {_save_target_text()} · "
                      "선택 대상·공정조건이 바뀌면 확인이 초기화됩니다. 확인 없이 대량 실행되지 않습니다."))
    if source=="SQLite":
        try:
            groups,expected,reason=run_eligibility(mode,scope,preview,plan,completed=completed,confirmed=confirmed)
        except (KeyError,TypeError,ValueError) as exc:
            reason="실행 조건 확인 실패: "+str(exc)
    with box_c:
        chips("실행 정책",RUN_TAGS if source=="SQLite" else ("SQLite 전용",),
              note=f"{CONFIRM_THRESHOLD}장 초과 시 확인")
    snapshot=job_snapshot() if source=="SQLite" else None
    clicked=box_c.button("분석 실행",disabled=groups is None,key="dcc_analysis_run",width="stretch",
                      help=reason or "선택 범위의 READY 이미지를 분석합니다.")
    if groups is None and source=="SQLite" and preview and reason and reason!=CONFIRM_REASON:
        box_c.caption(reason)
        if plan and reason.endswith("중복 실행 차단"):
            with box_c:
                duplicate_details(plan,completed)
    if clicked and groups is not None:
        image_ids=[r["image_id"] for g in groups for r in g["requests"]]
        target=(preview or {}).get("target_name") or _target_label(groups)
        token=session_token()
        try:
            if len(image_ids)>CONFIRM_THRESHOLD:
                start_analysis(groups,expected,confirmed_count=len(image_ids))
            else:
                start_analysis(groups,expected)
            with _JOB_LOCK:  # Display meta for every page; this script collects it (released on page change).
                _ACTIVE["meta"]={"total":len(image_ids),**(_ACTIVE.get("meta") or {}),
                                 "target":str(target),"owner":token,"scope":scope}
                _ACTIVE["waiter"]=token
            notify("분석 시작",str(target),"실행 중",count=len(image_ids),level="info",
                   key=("analysis_start",selection_token(expected),time.time()))
            with box_d:
                st.markdown('<div class="dcc-status-line"><b>'+escape(str(target))+'</b> · 실행 중</div>',
                            unsafe_allow_html=True)
                bar=st.progress(0.0,text=f"0/{len(image_ids)}장 · 0%")
            def show(progress):
                done=min(progress["done"],len(image_ids));o=progress["outcomes"]
                bar.progress(min(1.0,done/len(image_ids)),text=(f"{done}/{len(image_ids)}장 · {round(100*done/len(image_ids))}%"
                    f" · 남은 {len(image_ids)-done}장 · 실패 {o.get('FAILED',0)}장"))
                try:  # TASK18-B-17: same numbers to the '웨피' of this (blocked) session.
                    from dashboard.components.waepi import push,progress_text
                    text,pct=progress_text(done,len(image_ids),o.get("FAILED",0))
                    push({"text":text,"tone":"running","pct":pct})
                except Exception:
                    pass
            try:
                with box_d,st.spinner(f"분석 실행 중 · {len(image_ids)}장 (RF-DETR → PD 정책 → U-Net → PN/PT)…"):
                    result=wait_for_analysis(max(RUN_TIMEOUT_SECONDS,PER_IMAGE_TIMEOUT_SECONDS*len(image_ids)),on_progress=show)
            except BaseException:
                # Page change/rerun stops only this waiting script, not the Runner process: hand it back so
                # any page can show the progress and collect + save it once.
                with _JOB_LOCK:
                    if _ACTIVE.get("waiter")==token:
                        _ACTIVE.pop("waiter",None)
                raise
            if result is not None:
                result["db_save"]=save_analysis_result(result,runtime_provider())
                result["target"]=str(target)
                _LAST.update(result=result,owner=token)
                st.session_state.setdefault("_analysis_merged_runs",[]).append(result.get("run_dir"))
            st.session_state["dcc_analysis_last_run"]=result
            notify_result(result,groups)
            if result and result.get("execution_status") in ("COMPLETED","COMPLETED_WITH_ISSUES"):
                completed.extend(image_ids)
        except Exception as exc:  # The Dashboard must survive any execution failure.
            st.session_state["dcc_analysis_last_run"]={"execution_status":"FAILED","error":str(exc),
                "run_dir":"-","image_id":image_ids[0],"image_ids":image_ids,"target":str(target)}
            notify_result(st.session_state["dcc_analysis_last_run"],groups)
        st.rerun()  # Redraw the button with the post-run state (duplicate guard visible).
    if source=="SQLite" and snapshot is not None:
        with box_d:
            _job_progress_fragment()  # Polls every 2 s; collects once when the Runner has exited.
    else:
        box_d.caption("대기 중")
    last=st.session_state.get("dcc_analysis_last_run")
    if not last or source!="SQLite":
        box_e.caption("결과 없음")
    else:
        # Always this session's latest run (widgets reset after a page change; the result must stay visible).
        shown=set(last.get("image_ids") or [last.get("image_id")])
        with box_e:
            if not shown & set((preview or {}).get("image_ids") or []):
                st.caption("이번 세션 최근 실행 · 현재 선택 대상과 다름")
            _render_run_result(last)
