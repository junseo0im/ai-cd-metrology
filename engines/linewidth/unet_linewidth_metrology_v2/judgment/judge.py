"""선폭 판정 (pn / pt) — 측정 코드와 분리된 판정 모듈.

측정 결과(metrology.unet_metrology 의 출력 dict)를 받아 line 마다 pn / pt / normal 을 붙인다.
기준값은 같은 폴더의 rules.json 에 있으며, 회의에서 바뀌면 그 파일만 고친다.

현재 규칙 (rules.json, 2026-11-v1)
  - 경계: 빨간 sidewall 띠 안쪽 끝 (측정값 그대로)
  - 대상: line (밝은 띠) 만. gap 은 판정하지 않는다.
  - pn: 그 line 의 Y 3곳 중 가장 좁은 값 <= 8 µm
  - pt: 그 line 의 Y 3곳 중 가장 넓은 값 >= 12 µm
  - 사진 단위·Die/Row 판정과 pd 는 하지 않는다 (통합 Runner·결함 파트 담당).

판정에 영향을 주지 않는 참고 표시
  - near_threshold: 기준값 ± 0.2 µm 안 (사람끼리 측정 차이 수준이라 재확인 권장)
  - pitch_ok: 이웃 line 과의 중심 간격이 20 ± 2 µm 인가.
    False 면 빨간 띠를 하나 놓쳐 구간이 합쳐졌을 가능성이 있다.
    (line 과 gap 이 뒤바뀐 경우는 간격이 그대로라 이 점검으로 잡히지 않는다.)

사용:
  from judgment.judge import judge, load_rules
  judged = judge(result)                  # result = MetrologyEngine().analyze(...)
  judged["judgment"]["lines"]             # line 별 판정
"""
from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

RULES_PATH = Path(__file__).with_name("rules.json")
_OPS = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b, ">": lambda a, b: a > b}
_STATS = {"min": min, "max": max, "mean": lambda values: sum(values) / len(values)}


def load_rules(path: str | Path | None = None) -> dict:
    rules = json.loads(Path(path or RULES_PATH).read_text(encoding="utf-8"))
    if rules.get("edge_basis") not in ("inner", "center"):
        raise ValueError("edge_basis must be 'inner' or 'center'")
    for rule in rules["rules"]:
        if rule["op"] not in _OPS or rule.get("statistic", "min") not in _STATS:
            raise ValueError(f"unsupported rule: {rule}")
    return rules


def _um_per_px(measurements: list[dict]) -> float | None:
    for m in measurements:
        if m.get("width_px"):
            return m["width_um"] / m["width_px"]
    return None


def _width_um(m: dict, edge_basis: str, um_per_px: float) -> float:
    """inner: 측정값 그대로. center: 양옆 빨간 띠 폭의 절반씩 더한 값."""
    if edge_basis == "inner":
        return m["width_um"]
    left, right = m.get("left_sidewall_px"), m.get("right_sidewall_px")
    if left is None or right is None:
        return m["width_um"]
    return m["width_um"] + (left + right) / 2.0 * um_per_px


def _pitch_flags(rows: dict, um_per_px: float, pitch_um: float, tolerance_um: float) -> dict:
    """(line_id, y_fraction) -> 이웃 line 과의 중심 간격이 정상인지 (이웃이 없으면 None)."""
    flags = {}
    for y, items in rows.items():
        ordered = sorted(items, key=lambda m: m["center_x_px"])
        for index, m in enumerate(ordered):
            gaps = []
            for neighbour in (index - 1, index + 1):
                if 0 <= neighbour < len(ordered):
                    gaps.append(abs(ordered[neighbour]["center_x_px"] - m["center_x_px"]) * um_per_px)
            flags[(m["line_id"], y)] = (any(abs(g - pitch_um) <= tolerance_um for g in gaps)
                                        if gaps else None)
    return flags


def judge(result: dict, rules: dict | None = None) -> dict:
    """측정 결과 dict 에 'judgment' 를 붙여 새 dict 로 돌려준다 (원본은 바꾸지 않음)."""
    rules = rules or load_rules()
    judged = deepcopy(result)
    base = {"rule_version": rules["rule_version"], "edge_basis": rules["edge_basis"],
            "measurement_type": rules["measurement_type"]}
    if result.get("status") != "success":
        judged["judgment"] = {**base, "status": "not_judged",
                              "reason": f"measurement {result.get('status')}: {result.get('error_code')}",
                              "lines": []}
        return judged

    kind = rules["measurement_type"]
    targets = [m for m in result["measurements"] if m["measurement_type"] == kind]
    um_per_px = _um_per_px(result["measurements"])
    if not targets or um_per_px is None:
        judged["judgment"] = {**base, "status": "not_judged", "reason": f"no {kind} measurement",
                              "lines": []}
        return judged

    by_row, by_line = defaultdict(list), defaultdict(list)
    for m in targets:
        by_row[m["y_fraction"]].append(m)
        by_line[m["line_id"]].append(m)
    pitch = _pitch_flags(by_row, um_per_px, rules["pitch_um"], rules["pitch_tolerance_um"])
    margin = rules["near_threshold_um"]

    lines = []
    for line_id in sorted(by_line, key=lambda i: min(m["center_x_px"] for m in by_line[i])):
        items = sorted(by_line[line_id], key=lambda m: m["y_fraction"])
        widths = [round(_width_um(m, rules["edge_basis"], um_per_px), 4) for m in items]
        label, reason, near = "normal", "", False
        for rule in rules["rules"]:
            value = _STATS[rule.get("statistic", "min")](widths)
            if abs(value - rule["um"]) <= margin:
                near = True
            if label == "normal" and _OPS[rule["op"]](value, rule["um"]):
                label = rule["label"]
                reason = f"{rule.get('statistic', 'min')} {value:.3f} µm {rule['op']} {rule['um']} µm"
        pitch_values = [pitch.get((line_id, m["y_fraction"])) for m in items]
        known = [v for v in pitch_values if v is not None]
        lines.append({
            "line_id": line_id,
            "label": label,
            "reason": reason,
            "min_um": min(widths), "max_um": max(widths),
            "mean_um": round(sum(widths) / len(widths), 4),
            "n_rows": len(widths),
            "near_threshold": near,
            "pitch_ok": all(known) if known else None,
            "rows": [{"y_fraction": m["y_fraction"], "width_um": w} for m, w in zip(items, widths)],
        })
    judged["judgment"] = {**base, "status": "judged", "lines": lines}
    return judged
