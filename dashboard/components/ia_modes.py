"""I&A Repeatability / Lighting tabs: real SQLite groups, input readiness and stored-result status.

Read-only. No statistics beyond counts are derived: the official repeat-image selection rule and the
Lighting analysis scope are not approved, so policy exclusions are shown, never relaxed.
"""
import sqlite3
import pandas as pd
import streamlit as st
from dashboard.components.tables import data_table

POLICY = ("반복·조명 이미지는 기존 정책상 분석 범위 검토가 필요해 자동 실행 대상에서 제외됩니다 · "
          "REVIEW_REQUIRED/BLOCKED 이미지는 실행하지 않습니다 · 공식 채택·SPC 반영 없음")


def mode_images(provider, upload_mode):
    chunks, offset = [], 0
    while True:
        frame = provider.images(upload_mode=upload_mode, limit=100, offset=offset)
        chunks.append(frame)
        if len(frame) < 100:
            break
        offset += 100
    return pd.concat(chunks, ignore_index=True)


def readiness(provider, images):
    """{image_id: input item} from the read-only Runner input gate (query_only)."""
    items = {}
    for batch_id, frame in images.groupby("batch_id", sort=False):
        plan = provider.prepare_analysis_inputs(batch_id, image_ids=frame["image_id"].tolist())
        items.update({item["image_id"]: item for item in plan["items"]})
    return items


def _counts(values):
    found = {}
    for value in values:
        found[value] = found.get(value, 0) + 1
    return " / ".join(f"{k} {v}" for k, v in sorted(found.items())) or "-"


def _reasons(item):
    return " / ".join(i["message"] for i in (item or {}).get("issues") or []) or "-"


def _text(value, empty="미기록"):
    return empty if value is None or (isinstance(value, float) and value != value) else str(value)


def repeatability_rows(images, items, latest):
    rows = []
    for (wafer, die, site), frame in images.groupby(["wafer_id", "die_code", "position_code"], sort=True):
        ids = frame["image_id"].tolist()
        indices = ", ".join(str(int(v)) for v in sorted(frame["repeat_index"].dropna()))
        reasons = sorted({_reasons(items.get(i)) for i in ids} - {"-"})
        rows.append({"Wafer": wafer, "Die": die, "Site": site, "반복 번호": indices or "미기록", "이미지 수": len(ids),
                     "입력 상태": _counts(items.get(i, {}).get("validation_status", "-") for i in ids),
                     "저장된 분석 결과": sum(i in latest for i in ids),
                     "제한 사유": " / ".join(reasons) or "-"})
    return rows


def lighting_rows(images, items, latest):
    rows = []
    for record in images.sort_values(["illumination_setting", "image_id"], na_position="last").to_dict("records"):
        item = items.get(record["image_id"]) or {}
        calibration = ((item.get("candidate_request") or {}).get("calibration") or {}).get("calibration_id")
        auto = record.get("auto_exposure")
        rows.append({"조명 설정": _text(record.get("illumination_setting")), "배율": _text(record.get("magnification")),
                     "Auto Exposure": {0: "OFF", 1: "ON"}.get(auto, "알 수 없음") if auto == auto else "알 수 없음",
                     "노광": _text(record.get("exposure_value"), "미확인"), "Wafer / Die / Site":
                     f"{record['wafer_id']} / {record['die_code']} / {record['position_code']}",
                     "Calibration": calibration or "미확인", "입력 상태": item.get("validation_status", "-"),
                     "저장된 분석 결과": "있음" if record["image_id"] in latest else "없음",
                     "제한 사유": _reasons(item), "Image ID": record["image_id"]})
    return rows


def render_mode_tab(provider, mode):
    """mode: 'repeatability' | 'lighting'."""
    title = "Repeatability" if mode == "repeatability" else "Lighting"
    if provider is None:
        st.info("SQLite 데이터를 조회할 수 없습니다 · 등록 Wafer 없음 또는 DB 조회 실패")
        return
    try:
        images = mode_images(provider, mode)
        if images.empty:
            st.info(f"등록된 {title} 이미지가 없습니다.")
            st.caption(POLICY)
            return
        items = readiness(provider, images)
        latest = provider.latest_analysis()
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        st.error(f"{title} 데이터 조회 실패. " + str(exc))
        return
    statuses = [items.get(i, {}).get("validation_status", "-") for i in images["image_id"]]
    cols = st.columns(4)
    cols[0].metric("등록 이미지", len(images))
    cols[1].metric("READY", statuses.count("READY"))
    cols[2].metric("REVIEW_REQUIRED / BLOCKED", statuses.count("REVIEW_REQUIRED") + statuses.count("BLOCKED"))
    cols[3].metric("저장된 분석 결과", sum(i in latest for i in images["image_id"]))
    if mode == "repeatability":
        rows = repeatability_rows(images, items, latest)
        st.markdown(f"**반복 촬영 그룹 (Wafer / Die / Site) · {len(rows)}개**")
        data_table(pd.DataFrame(rows), height=min(320, (len(rows) + 1) * 31 + 3))
        st.caption("반복성 통계(σ 등)는 공식 반복 이미지 선정 기준이 확정되기 전에는 산출하지 않습니다.")
    else:
        rows = lighting_rows(images, items, latest)
        settings = images["illumination_setting"].fillna("미기록").astype(str).nunique()
        st.markdown(f"**조명 조건별 이미지 · 조건 {settings}개 · {len(rows)}장**")
        data_table(pd.DataFrame(rows), height=min(360, (len(rows) + 1) * 31 + 3))
        st.caption("Calibration은 입력 준비 단계의 매칭 결과이며, 미확인 이미지는 µm 환산·계측을 하지 않습니다.")
    st.caption(POLICY)
