"""AI provisional Die assessment, policy v0.1 (TASK 18-B-4). Pure functions; no DB, no engines.

Input per image = SQLiteProvider.latest_analysis() row (stored values, never recomputed by the
engines). Output is an AI provisional judgment, NOT an official quality verdict: official
results are only those with is_adopted=1 (none exist), handled by the caller.

Image  : risk = max(CD level, Defect level); NOT_EVALUATED is not part of the max.
  CD   : worst individual LINE measurement error vs. the run's own design_width_um
         (<=5 NORMAL, <=10 MINOR, <=20 WARNING, >20 CRITICAL). Gap has no design target -> not graded.
         MODEL_PD_EXCLUDED -> no CD (defect decides); POLICY_REVIEW_REQUIRED -> CD on hold;
         PARTIAL_MEASUREMENT -> provisional CD only (evaluation incomplete).
  Defect: model o/s/pd -> UNUSABLE candidate; p/sc -> WARNING candidate + review;
         none -> no risk signal (NOT a quality PASS). No confidence/area thresholds exist.
Die    : colour only when lt/c/rb all have complete CD+Defect evaluation, else NOT_EVALUATED
         (provisional max kept separately). PASS/FAIL/PENDING (B rule) needs complete defect
         evaluation at all three positions: any model O/S/PD -> FAIL, else PASS (O/S/PD basis).
Row    : 8 Dies decided -> PASS if PASS Dies >= 5 else FAIL; any PENDING -> PENDING.
Rates  : FAIL/(PASS+FAIL)*100; PENDING excluded; zero denominator -> None ("계산 불가").
"""
from collections import Counter
import json
import math

POLICY_VERSION = "AI 잠정 판정 정책 v0.1"
LEVELS = ("NORMAL", "MINOR", "WARNING", "CRITICAL", "UNUSABLE")
NOT_EVALUATED = "NOT_EVALUATED"
CRITICAL_DEFECTS = ("o", "s", "pd")
REVIEW_DEFECTS = ("p", "sc")
DEFECT_NAMES = {"o": "Open", "s": "Short", "pd": "Pattern Damage", "p": "Particle", "sc": "Scratch"}
JUDGED_MODES = ("wafer_batch", "single_image")  # Repeatability/Lighting are not judged here.
POSITIONS = ("lt", "c", "rb")


def worst(levels):
    found = [level for level in levels if level in LEVELS]
    return max(found, key=LEVELS.index) if found else None


def cd_level(error_pct):
    """Inclusive v0.1 boundaries; rounded to absorb float noise at 5/10/20 %."""
    error = round(error_pct, 9)
    if error <= 5:
        return "NORMAL"
    if error <= 10:
        return "MINOR"
    if error <= 20:
        return "WARNING"
    return "CRITICAL"


def _design_width(item):
    config = item.get("configuration")
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except ValueError:
            config = None
    value = (config or {}).get("design_width_um")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return float(value)
    return None


def assess_image(item):
    """Assessment of one stored result (or None = not analysed)."""
    result = {"analysed": item is not None, "cd_level": None, "cd_state": "NOT_ANALYSED", "cd_worst_error_pct": None,
              "cd_mean_error_pct": None, "worst_line_um": None, "defect_level": None, "defects": [],
              "critical_defects": [], "heuristic_pd": False, "defect_complete": False, "cd_complete": False,
              "pn": 0, "pt": 0, "review": [], "conflicts": [], "risk": NOT_EVALUATED, "complete": False,
              "outcome": None, "is_adopted": None}
    if item is None:
        return result
    outcome = item.get("outcome")
    result.update(outcome=outcome, is_adopted=item.get("is_adopted"))
    # Defect evaluation: model-postprocessed classes only; heuristic PD is a review signal.
    details = item.get("defect_details") or [{"class_code": c, "confidence": None, "source": "MODEL_POSTPROCESSED"}
                                             for c in item.get("defect_classes") or []]
    model = [d for d in details if d.get("source") == "MODEL_POSTPROCESSED"]
    result["defects"] = details
    result["critical_defects"] = sorted({d["class_code"] for d in model if d["class_code"] in CRITICAL_DEFECTS})
    review_defects = sorted({d["class_code"] for d in model if d["class_code"] in REVIEW_DEFECTS})
    # Non-model PD (HEURISTIC, or r33 RULE PD: same POLICY_REVIEW policy, TASK18-B-22).
    result["heuristic_pd"] = any(d.get("source") in ("HEURISTIC", "RULE") and d["class_code"] == "pd" for d in details)
    defect_ok = item.get("defect_status") == "SUCCESS"
    if result["critical_defects"]:
        result["defect_level"] = "UNUSABLE"
    elif result["heuristic_pd"]:
        result["defect_level"] = "UNUSABLE"  # Provisional signal only; evaluation stays incomplete.
    elif review_defects:
        result["defect_level"] = "WARNING"
    if review_defects:
        result["review"].append("P/SC 검출 · 검토 필요 (" + ", ".join(review_defects) + ")")
    result["defect_complete"] = defect_ok and outcome != "POLICY_REVIEW_REQUIRED"
    # CD evaluation.
    design = _design_width(item)
    if outcome == "MODEL_PD_EXCLUDED":
        result.update(cd_state="PD_EXCLUDED", cd_complete=True)  # No CD; the defect decides.
    elif outcome == "POLICY_REVIEW_REQUIRED":
        result.update(cd_state="POLICY_HOLD")
        result["review"].append("규칙 기반 PD 의심 · 정책 검토 필요 · CD 판정 보류")
    elif item.get("metrology_status") == "SUCCESS" and item.get("line_widths_um"):
        if design is None:
            result.update(cd_state="NO_DESIGN_WIDTH")
            result["review"].append("설계 선폭 Metadata 없음 · CD 미판정")
        else:
            widths = item["line_widths_um"]
            errors = [abs(w - design) / design * 100 for w in widths]
            worst_index = max(range(len(errors)), key=errors.__getitem__)
            result.update(cd_worst_error_pct=errors[worst_index], worst_line_um=widths[worst_index],
                          cd_mean_error_pct=abs(sum(widths) / len(widths) - design) / design * 100,
                          cd_level=cd_level(errors[worst_index]))
            if item.get("partial_measurement"):
                result.update(cd_state="PROVISIONAL_PARTIAL")
                result["review"].append("부분 계측 · CD 잠정값 (평가 미완료)")
            else:
                result.update(cd_state="EVALUATED", cd_complete=True)
    else:
        result.update(cd_state="NOT_MEASURED")
    if outcome == "QUALITY_REVIEW_REQUIRED":
        result["review"].append("기존 품질 검토 필요 (pitch/근접 임계)")
    # Stored PN/PT labels (judge rule unchanged) vs. the CD grade.
    labels = [j.get("label") for j in item.get("judgments") or []]
    result["pn"], result["pt"] = labels.count("pn"), labels.count("pt")
    if (result["pn"] or result["pt"]) and result["cd_level"] not in (None, "CRITICAL"):
        result["conflicts"].append(f"PN/PT {result['pn'] + result['pt']}건이지만 CD {result['cd_level']}: "
                                   "PN ≤8 µm(=오차 20%)는 v0.1 경계상 WARNING일 수 있음")
    if result["cd_level"] == "CRITICAL" and not (result["pn"] or result["pt"]):
        result["conflicts"].append("CD CRITICAL이지만 저장된 PN/PT 없음")
    risk = worst([result["cd_level"], result["defect_level"]])
    result["risk"] = risk or NOT_EVALUATED
    result["complete"] = result["defect_complete"] and result["cd_complete"]
    return result


def assess_die(positions, analysis, modes):
    """positions: {pos: [image_id,...]}; modes: {image_id: upload_mode}; analysis: {image_id: item}."""
    per_position = {}
    for position in POSITIONS:
        ids = [i for i in positions.get(position, []) if modes.get(i) in JUDGED_MODES]
        images = {i: assess_image(analysis.get(i)) for i in ids}
        per_position[position] = {
            "image_ids": ids, "images": images, "present": bool(ids),
            "defect_complete": bool(ids) and all(a["defect_complete"] for a in images.values()),
            "complete": bool(ids) and all(a["complete"] for a in images.values()),
            "risk": worst(a["risk"] for a in images.values()) or NOT_EVALUATED,
            "critical_defects": sorted({c for a in images.values() for c in a["critical_defects"]}),
            "heuristic_pd": any(a["heuristic_pd"] for a in images.values()),
            "review": [r for a in images.values() for r in a["review"]],
            "conflicts": [c for a in images.values() for c in a["conflicts"]]}
    missing = [p for p in POSITIONS if not per_position[p]["present"]]
    provisional = worst(p["risk"] for p in per_position.values()) or NOT_EVALUATED
    color_complete = not missing and all(p["complete"] for p in per_position.values())
    defect_complete = not missing and all(p["defect_complete"] for p in per_position.values())
    critical = sorted({c for p in per_position.values() for c in p["critical_defects"]})
    if defect_complete:
        decision = "FAIL" if critical else "PASS"
    else:
        decision = "PENDING"
    expected_fail = decision == "PENDING" and (bool(critical) or any(p["heuristic_pd"] for p in per_position.values()))
    basis = None
    if provisional != NOT_EVALUATED:
        position = next(p for p in POSITIONS if per_position[p]["risk"] == provisional)
        basis = {"position": position, "critical_defects": per_position[position]["critical_defects"],
                 "cd_worst_error_pct": max((a["cd_worst_error_pct"] for a in per_position[position]["images"].values()
                                            if a["cd_worst_error_pct"] is not None), default=None)}
    reasons = []
    if missing:
        reasons.append("판정 대상 이미지 없음: " + ", ".join(missing))
    incomplete = [p for p in POSITIONS if per_position[p]["present"] and not per_position[p]["complete"]]
    if incomplete:
        reasons.append("평가 미완료 위치: " + ", ".join(incomplete))
    return {"policy": POLICY_VERSION, "positions": per_position, "missing_positions": missing,
            "provisional_risk": provisional, "color_complete": color_complete,
            "display_risk": provisional if color_complete else NOT_EVALUATED,
            "defect_complete": defect_complete, "decision": decision, "expected_fail": expected_fail,
            "critical_defects": critical, "basis": basis, "reasons": reasons,
            "review": sorted({r for p in per_position.values() for r in p["review"]}),
            "conflicts": sorted({c for p in per_position.values() for c in p["conflicts"]})}


# ---- Wafer Map AI provisional risk colour, 기준 v0.1 (TASK 18-B-14B) ---------------------------------
# Display only: computed from the stored per-image assessment above; nothing is written back.
RISK_POLICY = "AI 잠정 위험도 기준 v0.1"
SCORE_LEVELS = ("NORMAL", "MINOR", "WARNING", "CRITICAL")
CD_BOUNDS = (10, 20, 30)        # |Line - design| / design x 100: <=10 -> 0, <=20 -> 1, <=30 -> 2, >30 -> 3
PNPT_BOUNDS = (0, 2, 5)         # stored pn/pt Line labels per Die: 0 -> 0, 1-2 -> 1, 3-5 -> 2, >=6 -> 3


def cd_score(error_pct):
    error = round(error_pct, 9)   # Absorb float noise exactly at 10/20/30 %.
    return next((score for score, bound in enumerate(CD_BOUNDS) if error <= bound), 3)


def pnpt_score(count):
    return next((score for score, bound in enumerate(PNPT_BOUNDS) if count <= bound), 3)


def map_risk(assessment):
    """S = min(3, max(S_CD, S_PNPT) + I_OSPD) for a Die whose lt/c/rb evaluation is complete.

    * S_CD   : worst valid Line error over every measured image of the Die (existing aggregation).
    * S_PNPT : stored metrology pn/pt Line labels (not Defect-AI objects), per position the largest
               count among its images (re-shots of one Site are not added), summed over lt/c/rb.
               Images without metrology (PD-excluded) cannot contribute -> marked partial.
    * I_OSPD : +1 when any image has a model-postprocessed O/S/PD, never more (no per-image stacking).
    Incomplete Dies and Dies without any valid measurement stay NOT_EVALUATED (never NORMAL by default).
    P/SC and heuristic PD keep their existing review roles and are not part of the score.
    """
    result = {"policy": RISK_POLICY, "display_risk": NOT_EVALUATED, "score": None, "s_cd": None, "s_pnpt": None,
              "ospd": 0, "ospd_images": 0, "cd_worst_error_pct": None, "pnpt_count": None, "pnpt_partial": False,
              "state": "INCOMPLETE"}
    if not assessment or not assessment.get("color_complete"):
        return result
    errors, per_position, partial = [], [], False
    for position in POSITIONS:
        images = assessment["positions"][position]["images"].values()
        measured = [a for a in images if a["cd_state"] == "EVALUATED"]
        errors += [a["cd_worst_error_pct"] for a in measured if a["cd_worst_error_pct"] is not None]
        if measured:
            per_position.append(max(a["pn"] + a["pt"] for a in measured))
        partial = partial or len(measured) < len(images)
        result["ospd_images"] += sum(bool(a["critical_defects"]) for a in images)
    result["ospd"] = 1 if result["ospd_images"] else 0
    result["pnpt_partial"] = partial
    if errors:
        result["cd_worst_error_pct"] = max(errors)
        result["s_cd"] = cd_score(result["cd_worst_error_pct"])
    if per_position:
        result["pnpt_count"] = sum(per_position)
        result["s_pnpt"] = pnpt_score(result["pnpt_count"])
    bases = [s for s in (result["s_cd"], result["s_pnpt"]) if s is not None]
    if not bases:
        result["state"] = "NO_VALID_MEASUREMENT"
        return result
    result["score"] = min(3, max(bases) + result["ospd"])
    result["display_risk"] = SCORE_LEVELS[result["score"]]
    result["state"] = "EVALUATED"
    return result


def rate(fail, passed):
    denominator = fail + passed
    return None if denominator == 0 else fail / denominator * 100


def summarize_wafer(dies, key="assessment"):
    """Row (3 x 8) and Wafer AI provisional aggregation from assessed Dies."""
    rows = []
    for row in sorted({d["row"] for d in dies}):
        members = [d for d in dies if d["row"] == row]
        decisions = Counter(d[key]["decision"] for d in members)
        if decisions["PENDING"] or len(members) != 8:
            verdict = "PENDING"
        else:
            verdict = "PASS" if decisions["PASS"] >= 5 else "FAIL"
        rows.append({"row": row, "dies": len(members), "PASS": decisions["PASS"], "FAIL": decisions["FAIL"],
                     "PENDING": decisions["PENDING"], "verdict": verdict,
                     "die_fail_rate": rate(decisions["FAIL"], decisions["PASS"])})
    decisions = Counter(d[key]["decision"] for d in dies)
    row_verdicts = Counter(r["verdict"] for r in rows)
    colors = Counter(d[key]["display_risk"] for d in dies)
    return {"dies": len(dies), "PASS": decisions["PASS"], "FAIL": decisions["FAIL"], "PENDING": decisions["PENDING"],
            "evaluated": decisions["PASS"] + decisions["FAIL"],
            "die_fail_rate": rate(decisions["FAIL"], decisions["PASS"]),
            "rows": rows, "row_PASS": row_verdicts["PASS"], "row_FAIL": row_verdicts["FAIL"],
            "row_PENDING": row_verdicts["PENDING"],
            "row_fail_rate": rate(row_verdicts["FAIL"], row_verdicts["PASS"]),
            "colors": dict(colors)}


def rate_text(value):
    return "계산 불가 (평가 완료 없음)" if value is None else f"{value:.1f}%"


# ---- PD 사용 불가 표시 (TASK 18-B-15) -------------------------------------------------------------------
# Display only, read from stored per-image results (the score formula and PN/PT bounds above are unchanged):
#   PD 검출       : model-postprocessed 'pd' detection or stored outcome MODEL_PD_EXCLUDED
#   PD 의심 보류  : stored outcome POLICY_REVIEW_REQUIRED (rule-based PD suspicion, line metrology on hold)
# Any of them at lt/c/rb makes the Die UNUSABLE, ahead of other positions' NOT_EVALUATED. Held images are
# not counted as PD detections.
PD_DETECTED_OUTCOME = "MODEL_PD_EXCLUDED"
PD_HOLD_OUTCOME = "POLICY_REVIEW_REQUIRED"


def model_pd_count(image):
    return sum(d.get("source") == "MODEL_POSTPROCESSED" and d.get("class_code") == "pd" for d in image["defects"])


def pd_unusable(assessment):
    result = {"unusable": False, "detected": [], "held": [], "pd_count": 0}
    if not assessment:
        return result
    for position in POSITIONS:
        images = list(assessment["positions"][position]["images"].values())
        count = sum(model_pd_count(a) for a in images)
        if count or any(a["outcome"] == PD_DETECTED_OUTCOME for a in images):
            result["detected"].append(position)
            result["pd_count"] += count
        if any(a["outcome"] == PD_HOLD_OUTCOME for a in images):
            result["held"].append(position)
    result["unusable"] = bool(result["detected"] or result["held"])
    return result
