"""Read-only process cohorts and calculation boundary; no engines or Calibration lookup.

Future Metrology input (not connected): SUCCESS, pd_detected (verified bool),
calibration {calibration_id, valid, magnification}, calibration_applied=True,
unit='um', valid_line_widths_um. Pixel results require upstream verified
Calibration conversion. This module never reads Calibration record JSON or
converts pixels; already-calibrated µm values are never scaled again.
"""
import math
from statistics import mean, stdev
import pandas as pd

DESIGN_LINE_UM = 10.0
ALL = "__all__"
MISSING = "__missing__"
REASONS = {
    "NO_RESULT":"계측 결과 없음", "MEASUREMENT_FAILED":"계측 실패",
    "DEFECT_NOT_ANALYZED":"PD 여부 미분석", "PD_EXCLUDED":"PD 검출 이미지 제외",
    "CALIBRATION_MISSING":"Calibration 누락 또는 무효", "MAGNIFICATION_MISSING":"촬영 배율 미기록",
    "MAGNIFICATION_MISMATCH":"Calibration 배율 불일치",
    "NEEDS_CALIBRATED_UM":"Calibration 적용 µm 결과 필요 · px 자동 변환 미구현",
    "NO_VALID_WIDTHS":"유효 선폭 없음", "READY":"유효 µm 결과",
    "PARTIAL_EXCLUDED":"부분 계측 이미지 제외",
}


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def image_mean_um(result=None, *, magnification=None):
    def state(code,value=None):
        return {"status":code,"mean_um":value,"reason":REASONS[code]}
    if result is None:
        return state("NO_RESULT")
    if not isinstance(result,dict):
        return state("MEASUREMENT_FAILED")
    if result.get("status")!="SUCCESS":
        return state("MEASUREMENT_FAILED")
    if not isinstance(result.get("pd_detected"),bool):
        return state("DEFECT_NOT_ANALYZED")
    if result["pd_detected"]:
        return state("PD_EXCLUDED")
    if result.get("partial_measurement") is True:  # Partial results never enter averages.
        return state("PARTIAL_EXCLUDED")
    calibration=result.get("calibration")
    if not isinstance(calibration,dict) or calibration.get("valid") is not True or not isinstance(calibration.get("calibration_id"),str) or not calibration["calibration_id"].strip():
        return state("CALIBRATION_MISSING")
    if not finite(magnification) or magnification<=0:
        return state("MAGNIFICATION_MISSING")
    if not finite(calibration.get("magnification")) or calibration["magnification"]<=0:
        return state("CALIBRATION_MISSING")
    if calibration["magnification"]!=magnification:
        return state("MAGNIFICATION_MISMATCH")
    if result.get("calibration_applied") is not True or result.get("unit") not in ("um","µm","μm"):
        return state("NEEDS_CALIBRATED_UM")
    widths=result.get("valid_line_widths_um")
    if not isinstance(widths,(list,tuple)) or not widths or not all(finite(v) and v>0 for v in widths):
        return state("NO_VALID_WIDTHS")
    return state("READY",mean(widths))


def summarize_um(values, *, design_um=DESIGN_LINE_UM):
    """One entry per eligible image; sample std is between image means."""
    if not finite(design_um) or design_um<=0:
        raise ValueError("설계 선폭은 유효한 양수 µm 값이어야 합니다.")
    if any(not finite(v) or v<=0 for v in values):
        raise ValueError("Calibration이 확인된 유효 이미지별 평균 µm 값만 전달하세요.")
    n=len(values)
    average=mean(values) if n else None
    signed=(average-design_um)/design_um*100 if average is not None else None
    return {"valid_image_count":n,"mean_um":average,"std_um":stdev(values) if n>=2 else None,
            "std_state":"데이터 없음" if n==0 else "표준편차 산출 불가" if n==1 else "표본 표준편차",
            "signed_error_pct":signed,"absolute_error_pct":abs(signed) if signed is not None else None}


# User-confirmed exposure unit (TASK18-B-13A.1). Display only: rows whose unit is NOT recorded are labelled with
# it; an explicitly recorded unit is always shown as stored, so different recorded units never share a label.
from dashboard.units import EXPOSURE_DISPLAY_UNIT,exposure_display  # noqa: F401  (shared rule, TASK18-B-13A.2)


def exposure_label(value,unit):
    return exposure_display(value,unit,missing="노광 미입력")


def metadata_frame(images):
    result=images.copy()
    if "peb_temperature_c" not in result:
        result["peb_temperature_c"]=None
    result["exposure_condition"]=[exposure_label(v,u) for v,u in
                                  zip(result["exposure_value"],result["exposure_unit"])]
    result["peb_condition"]=result["peb_temperature_c"].map(
        lambda v:"PEB 미입력" if pd.isna(v) else f"{v:g} s")
    return result


def condition_table(images, axis, *, by_wafer=False, results=None):
    """Metadata counts are real; missing analysis stays NULL rather than zero."""
    column="exposure_condition" if axis=="exposure" else "peb_condition"
    fields=[column]+(["wafer_id"] if by_wafer else [])
    rows=[]
    for key,group in images.groupby(fields,dropna=False,sort=False):
        keys=key if isinstance(key,tuple) else (key,)
        evaluated=[image_mean_um((results or {}).get(r["image_id"]),magnification=r.get("magnification"))
                   for r in group.to_dict("records")]
        eligible=[r["mean_um"] for r in evaluated if r["status"]=="READY"]
        summary=summarize_um(eligible)
        # Gap: same eligibility as Line (READY images only); no separate design target exists.
        gaps=[]
        for record,state in zip(group.to_dict("records"),evaluated):
            widths=((results or {}).get(record["image_id"]) or {}).get("valid_gap_widths_um")
            if state["status"]=="READY" and isinstance(widths,(list,tuple)) and widths and all(finite(v) and v>0 for v in widths):
                gaps.append(mean(widths))
        row={"공정 조건":keys[0],"등록 이미지 수":len(group),"포함 Wafer 수":group["wafer_id"].nunique(),
             "평균 선폭 (µm)":summary["mean_um"],"이미지 간 표준편차 (µm)":summary["std_um"],
             "Signed Error (%)":summary["signed_error_pct"],"Absolute Error (%)":summary["absolute_error_pct"],
             "유효 이미지 수":summary["valid_image_count"] if results is not None else None,
             "PD 제외 이미지 수":sum(r["status"]=="PD_EXCLUDED" for r in evaluated) if results is not None else None,
             "부분 계측 제외 이미지 수":sum(r["status"]=="PARTIAL_EXCLUDED" for r in evaluated) if results is not None else None,
             "분석 결과 이미지 수":sum(r["status"]!="NO_RESULT" for r in evaluated) if results is not None else None,
             "평균 Gap (µm)":mean(gaps) if gaps else None,
             "Gap 이미지 간 표준편차 (µm)":stdev(gaps) if len(gaps)>=2 else None,
             "표준편차 상태":summary["std_state"],"분석 상태":"계측 결과 없음" if not eligible else "유효 µm 결과"}
        if by_wafer:
            row["Wafer ID"]=keys[1]
        # Stored detections (defect_classes) when results are given; otherwise unknown (None).
        for code in ("s","o","sc","p","pd"):
            classes=[((results or {}).get(r["image_id"]) or {}).get("defect_classes") for r in group.to_dict("records")]
            known=[c for c in classes if isinstance(c,list)]
            row[code+" 검출 건수"]=sum(c.count(code) for c in known) if results is not None and known else None
            row[code+" 포함 이미지 수"]=sum(code in c for c in known) if results is not None and known else None
        rows.append(row)
    return pd.DataFrame(rows)


# ---- Condition defect share (TASK18-B-18) ----------------------------------------------------------
def count_defects(details):
    """Detected defects of one run; an S and the P derived from it count once (via the S)."""
    total = 0
    for d in details:
        if d.get("team_source") == "from_s":
            continue   # Part of its source S (counted with that S), whether or not the link was resolved.
        total += d.get("source") == "MODEL_POSTPROCESSED"
    return total


def derived_pairs(details):
    """(S index, P index) pairs from the stored derivation link, plus derived P without a resolved S."""
    pairs, unresolved = [], []
    for index, d in enumerate(details):
        if d.get("team_source") == "from_s":
            parent = d.get("derived_from_index")
            ok = isinstance(parent, int) and 0 <= parent < len(details) and details[parent].get("class_code") == "s"
            (pairs.append((parent, index)) if ok else unresolved.append(index))
    return pairs, unresolved
def defect_ratio_table(images, axis, analysis):
    """Per condition: detected defects / all detected defects in the current filter x 100.

    * Defects = stored MODEL_POSTPROCESSED detections of each image's latest run (same 'detected defect'
      rule as the Die/image summaries; rule-based PD suspicion is not a detection).
    * r33 (TASK18-B-22): a P split out of an S (team_source 'from_s') and that S are ONE defect: the pair is
      counted through its S (by the S's own source rule) and the derived P adds nothing. The pair is taken
      from the stored derivation link (derived_from_index), never from box overlap; an independently
      detected P is counted as before.
    * One row per image's latest run -> no join duplication; the same capture (SHA-256) registered in
      several Batches is counted once (the copy with the newest stored analysis, as on the Wafer Map).
    * 결함 분석 완료 = stored result with defect_status SUCCESS (0-defect images included);
      images without a result / without a successful defect stage are counted separately.
    """
    from dashboard.components.inspection_wafer import duplicate_exclusions
    column = "exposure_condition" if axis == "exposure" else "peb_condition"
    records = images.to_dict("records")
    excluded = duplicate_exclusions(records, analysis or {})
    rows = []
    for condition, group in images.groupby(column, dropna=False, sort=False):
        defects = analysed = missing = merged = 0
        for record in group.to_dict("records"):
            if record["image_id"] in excluded:
                continue
            item = (analysis or {}).get(record["image_id"])
            if not item or item.get("defect_status") != "SUCCESS":
                missing += 1
                continue
            analysed += 1
            defects += count_defects(item.get("defect_details") or [])
            merged += sum(d.get("team_source") == "from_s" for d in item.get("defect_details") or [])
        rows.append({"공정 조건": condition, "결함 건수": defects, "결함 분석 완료 이미지 수": analysed,
                     "결함 분석 미실행·결과 없음 이미지 수": missing, "S 파생 P (S와 1건 합산)": merged})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    total = int(table.loc[table["결함 분석 완료 이미지 수"] > 0, "결함 건수"].sum())
    table["결함 개수 비율 (%)"] = [None if n == 0 or total == 0 else d / total * 100
                                 for d, n in zip(table["결함 건수"], table["결함 분석 완료 이미지 수"])]
    table["이미지 100장당 결함 건수"] = [None if n == 0 else d / n * 100
                                    for d, n in zip(table["결함 건수"], table["결함 분석 완료 이미지 수"])]
    table.attrs["total_defects"] = total
    table.attrs["duplicates_excluded"] = len(excluded)
    return table
