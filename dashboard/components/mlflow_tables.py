"""Default model-comparison tables per MLflow experiment (TASK18-B-19). Pure functions over runs/search rows.

Every number comes from the MLflow run record as stored. 'missing' (key not logged) is kept apart from a
real 0. A best value is highlighted only among rows that share the same evaluation set AND the same metric
key; rows without a recorded evaluation-set identity are never ranked.
"""
import re

import pandas as pd

from dashboard import mlflow_provider as provider

MISSING = "미기록"
WIDTH, CLASSIFICATION, DETECTION = "IDA_Width_Measurement", "IDA_Defect_Classification", "IDA_Defect_YOLO"


def _m(run, key):
    return provider.metric_value(run, key)


def _fmt(value, digits=4):
    return MISSING if value is None else f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits > 4 else f"{value:.{digits}f}"


def _best(rows, column, *, lower=False, group=None):
    """(row, column) of the best numeric value per group; groups with < 2 numeric rows are not ranked."""
    cells = []
    groups = {}
    for index, row in enumerate(rows):
        key = group(row) if group else "all"
        if key is None or row.get("_v_" + column) is None:
            continue
        groups.setdefault(key, []).append(index)
    for members in groups.values():
        if len(members) < 2:
            continue
        values = [rows[i]["_v_" + column] for i in members]
        target = min(values) if lower else max(values)
        cells += [(i, column) for i in members if rows[i]["_v_" + column] == target]
    return cells


def _table(rows, columns, best, notes, help_text, *, extra=None):
    frame = pd.DataFrame([{c: row.get(c, MISSING) for c in columns} for row in rows], columns=columns)
    return {"frame": frame, "best": best, "notes": notes, "help": help_text, "extra": extra or {}}


def is_connection_run(run):
    """Connection / smoke checks: shown only in the MLflow detail section."""
    tags = provider.pairs(run, "tags")
    name = provider.run_name(run).lower()
    metrics = set(provider.pairs(run, "metrics"))
    return (tags.get("purpose") == "connection_test" or metrics <= {"connection_success"} and bool(metrics)
            or "smoke" in name or "connection" in name)


# ---- 선폭 계측 ---------------------------------------------------------------------------------------
METHOD_NAMES = {"threshold": "Threshold", "canny": "Canny", "unet": "U-Net"}


def width_table(runs):
    rows = []
    for run in runs:
        if is_connection_run(run):
            continue
        params = provider.pairs(run, "params")
        rate = _m(run, "stability/success_rate")
        # Denominator from the recorded stability set: images x Y rows x targets x measurement types.
        denominator = None
        try:
            images = int(re.match(r"\s*(\d+)\s*images", params.get("stability_set", "")).group(1))
            denominator = (images * len(params["y_fractions"].split(",")) * int(params["targets_per_row"])
                           * len(params["measurement_types"].split(",")))
        except (AttributeError, KeyError, ValueError, TypeError):
            pass
        if rate is None:
            success = MISSING
        elif denominator and abs(rate * denominator - round(rate * denominator)) < 1e-6:
            success = f"{round(rate * denominator)}/{denominator} 측정 행 ({rate * 100:.1f}%)"
        else:
            success = f"{rate * 100:.1f}% (분모 미확인)"
        test_set = params.get("test_set", "")
        rows.append({
            "방법": METHOD_NAMES.get(params.get("method", ""), params.get("method") or provider.run_name(run)),
            "평가 조건": " · ".join(x for x in (("Test " + test_set.replace(" images", "장")) if test_set else "",
                                              params.get("edge_parameter", "")) if x) or MISSING,
            "계측 성공률 ↑": success, "_v_계측 성공률 ↑": rate,
            "MAE ↓ (µm)": _fmt(_m(run, "test/mae_um")), "_v_MAE ↓ (µm)": _m(run, "test/mae_um"),
            "반복 촬영 표준편차 ↓ (µm)": _fmt(_m(run, "stability/repeat_target_sd_um")),
            "_v_반복 촬영 표준편차 ↓ (µm)": _m(run, "stability/repeat_target_sd_um"),
            "조명 변화에 따른 범위 ↓ (µm)": _fmt(_m(run, "stability/lighting_image_mean_range_um")),
            "_v_조명 변화에 따른 범위 ↓ (µm)": _m(run, "stability/lighting_image_mean_range_um"),
            "처리시간 중앙값 ↓ (ms/장)": MISSING if _m(run, "test/runtime_median_ms") is None else f"{_m(run, 'test/runtime_median_ms'):.1f}",
            "_v_처리시간 중앙값 ↓ (ms/장)": _m(run, "test/runtime_median_ms"),
            "Run": provider.run_id(run)[:8],
            "_set": (params.get("test_set"), params.get("stability_set"), provider.pairs(run, "tags").get("dataset_version")),
        })
    rows.sort(key=lambda r: list(METHOD_NAMES.values()).index(r["방법"]) if r["방법"] in METHOD_NAMES.values() else 9)
    same = lambda r: r["_set"] if all(r["_set"]) else None
    best = (_best(rows, "계측 성공률 ↑", group=same) + _best(rows, "MAE ↓ (µm)", lower=True, group=same)
            + _best(rows, "반복 촬영 표준편차 ↓ (µm)", lower=True, group=same)
            + _best(rows, "조명 변화에 따른 범위 ↓ (µm)", lower=True, group=same)
            + _best(rows, "처리시간 중앙값 ↓ (ms/장)", lower=True, group=same))
    columns = ["방법", "평가 조건", "계측 성공률 ↑", "MAE ↓ (µm)", "반복 촬영 표준편차 ↓ (µm)",
               "조명 변화에 따른 범위 ↓ (µm)", "처리시간 중앙값 ↓ (ms/장)", "Run"]
    help_text = {
        "계측 성공률 ↑": "안정성 세트 측정 행 기준 (이미지 × Y 3높이 × 대상 5 × Line·Gap). 이미지 검출 성공률이 아님",
        "MAE ↓ (µm)": "Test 세트 수동 GT 대비 평균 절대오차",
        "반복 촬영 표준편차 ↓ (µm)": "반복 촬영 대상별 표준편차",
        "조명 변화에 따른 범위 ↓ (µm)": "조명 조건별 이미지 평균값의 범위",
        "처리시간 중앙값 ↓ (ms/장)": "같은 PC·CPU, 첫 이미지 제외 (기록 tag)",
    }
    notes = ["굵게: 같은 Test·안정성 세트 안에서 최고값 · 기록 상태 logged_unverified"] if best else []
    return _table(rows, columns, best, notes, help_text)


# ---- 결함 분류 ---------------------------------------------------------------------------------------
MODEL_NAMES = {"mobilenet_v3_small": "MobileNet V3 Small", "resnet18": "ResNet18"}
CONDITIONS = (("real_only", "원본만"), ("oversampl", "오버샘플링"))


def classification_table(runs):
    cells = {}
    for run in runs:
        if is_connection_run(run):
            continue
        params = provider.pairs(run, "params")
        model = MODEL_NAMES.get(params.get("model", ""), params.get("model") or "모델 미기록")
        name = provider.run_name(run).lower()
        condition = next((label for token, label in CONDITIONS if token in name), "기타")
        cells.setdefault((model, condition), []).append(run)
    rows = []
    models = list(MODEL_NAMES.values()) + sorted({m for m, _ in cells} - set(MODEL_NAMES.values()))
    for model in models:
        for _, condition in CONDITIONS + (("", "기타"),):
            found = cells.get((model, condition), [])
            if not found and condition == "기타":
                continue
            for run in found or [None]:
                best = _m(run, "val/best_macro_f1") if run else None
                test = _m(run, "test/macro_f1") if run else None
                rows.append({"모델": model, "조건": condition,
                             "검증 최고 Macro F1 ↑": _fmt(best) if run else "Run 없음", "_v_검증 최고 Macro F1 ↑": best,
                             "Test Macro F1 ↑": _fmt(test) if run else "Run 없음", "_v_Test Macro F1 ↑": test,
                             "Run": provider.run_id(run)[:8] if run else "-", "_model": model})
    # Oversampling effect only when BOTH conditions of a model carry the same metric key.
    for row in rows:
        if row["조건"] != "오버샘플링":
            row["오버샘플링 효과 (%p)"] = ""
            continue
        base = next((r for r in rows if r["_model"] == row["_model"] and r["조건"] == "원본만"), None)
        parts = []
        for column, label in (("검증 최고 Macro F1 ↑", "검증 최고"), ("Test Macro F1 ↑", "Test")):
            a, b = row["_v_" + column], (base or {}).get("_v_" + column)
            if a is not None and b is not None:
                parts.append(f"{(a - b) * 100:+.2f}%p ({label})")
        row["오버샘플링 효과 (%p)"] = " · ".join(parts) or "비교 보류 (양쪽 기록 없음)"
    best = _best(rows, "검증 최고 Macro F1 ↑", group=lambda r: "val") + _best(rows, "Test Macro F1 ↑", group=lambda r: "test")
    columns = ["모델", "조건", "검증 최고 Macro F1 ↑", "Test Macro F1 ↑", "오버샘플링 효과 (%p)", "Run"]
    help_text = {"검증 최고 Macro F1 ↑": "val/best_macro_f1 (학습 중 검증 Macro F1 최고값)",
                 "Test Macro F1 ↑": "test/macro_f1",
                 "오버샘플링 효과 (%p)": "같은 모델·같은 지표 Key가 양쪽 모두 기록된 경우에만 계산"}
    notes = ["val/macro_f1(마지막 기록값)은 Run마다 기록 시점이 달라 검증 최고값과 같은 열에서 비교하지 않음 · MLflow 상세 조회에서 확인",
             "4개 Run 모두 과거 기록 import (metrics_verified=false)"] if rows else []
    return _table(rows, columns, best, notes, help_text)


# ---- 결함 검출 ---------------------------------------------------------------------------------------
def _family(run):
    tags, params = provider.pairs(run, "tags"), provider.pairs(run, "params")
    text = " ".join([provider.run_name(run), tags.get("model_name", ""), params.get("model_name", ""),
                     params.get("roboflow_model_id", ""), tags.get("model_id", "")]).lower()
    for token, label in (("rf-detr", "RF-DETR Small"), ("rfdetr", "RF-DETR Small"), ("yolov11m", "YOLOv11m-Seg"),
                         ("yolov8s", "YOLOv8s"), ("yolov8n", "YOLOv8n")):
        if token in text:
            return label
    return tags.get("model_name") or provider.run_name(run)


CONDITION_RULES = (("postprocess", "후처리 파이프라인"), ("real_only", "원본만"), ("oversampl", "오버샘플링"),
                   ("augment", "증강"), ("raw-box", "원본 모델 · Box 평가"), ("baseline", "원본 (Baseline)"),
                   ("stratified", "stratified v2"), ("seg", "Seg 모델"))


def _condition(run):
    """First matching rule (most specific first) + a non-default image size."""
    name = (provider.run_name(run) + " " + provider.pairs(run, "tags").get("model_name", "")).lower()
    params = provider.pairs(run, "params")
    label = next((text for token, text in CONDITION_RULES if token in name), "기록 조건 없음")
    if params.get("imgsz") and params.get("imgsz") != "640":
        label += " · " + params["imgsz"] + "px"
    return label


def _eval_set(run):
    """(label, identity) — identity None when the evaluation set is not recorded (never ranked).

    The same test_split_id was scored against different ground-truth totals (tp+fn) in some runs, so the
    GT count is part of the identity whenever tp/fn are logged.
    """
    label, identity = _eval_set_base(run)
    tp, fn = _m(run, "test/tp"), _m(run, "test/fn")
    if tp is not None and fn is not None:
        label += f" · GT {int(tp + fn)}건"
        identity = identity + (int(tp + fn),) if identity else None
    return label, identity


def _eval_set_base(run):
    tags, params, metrics = provider.pairs(run, "tags"), provider.pairs(run, "params"), provider.pairs(run, "metrics")
    split = tags.get("test_split_id")
    if tags.get("evaluation_type", "").startswith("raw_model_local_box"):
        return f"Test {params.get('test_images', '?')}장 · {tags.get('included_classes', '').replace(',', '·')}만", \
               ("local_box", tags.get("dataset_fingerprint_sha256"))
    if split == "roboflow_test_dataset":
        return f"Roboflow Test {params.get('dataset_total_images', '?')}장", (split, params.get("dataset_total_images"))
    if split and split.startswith("roboflow_228"):
        return "unified-228 Test " + split.rsplit("test", 1)[-1] + "장", (split,)
    if split and split.startswith("sha256-"):
        return f"stratified v2 Test {params.get('test_images', '?')}장", (split,)
    if any(k.startswith("val/") for k in metrics) and not any(k.startswith("test/") for k in metrics):
        return "검증셋 (과거 기록)", None
    return "평가셋 미기록", None


def detection_table(runs):
    rows, connection, classes = [], [], []
    for run in runs:
        if is_connection_run(run):
            connection.append(run)
            continue
        label, identity = _eval_set(run)
        box50 = next((v for v in (_m(run, "test/mAP50"), _m(run, "test/local_box_map50_o_s_pd"), _m(run, "val/map50_box"))
                      if v is not None), None)
        box5095 = next((v for v in (_m(run, "test/mAP50-95"), _m(run, "val/map50_95_box")) if v is not None), None)
        recall = next((v for v in (_m(run, "test/recall"), _m(run, "val/recall_box")) if v is not None), None)
        latency = _m(run, "test/avg_latency_ms")
        mask50 = next((v for v in (_m(run, "test/mask_mAP50"), _m(run, "val/map50_mask")) if v is not None), None)
        notes = []
        if _m(run, "test/roboflow_map50") is not None:
            notes.append(f"Roboflow mAP@50 {_m(run, 'test/roboflow_map50'):.3f} (Box/Mask 정의 미확인)")
        if _m(run, "test/local_box_map50_o_s_pd") is not None:
            notes.append("Box AP50: o·s·pd 3종 평균")
        rows.append({"모델": _family(run), "실험 조건": _condition(run), "평가셋": label,
                     "Box mAP@50 ↑": _fmt(box50), "_v_Box mAP@50 ↑": box50,
                     "Box mAP@50–95 ↑": _fmt(box5095), "_v_Box mAP@50–95 ↑": box5095,
                     "Recall ↑": _fmt(recall), "_v_Recall ↑": recall,
                     "추론시간 ↓ (ms)": MISSING if latency is None else f"{latency:.1f}", "_v_추론시간 ↓ (ms)": latency,
                     "Mask mAP@50 ↑": _fmt(mask50), "비고": " · ".join(notes), "Run": provider.run_id(run)[:8],
                     "_id": identity, "_start": run.get("info", {}).get("start_time") or 0})
        for code in ("o", "s", "pd", "p", "sc"):
            ap = next((v for v in (_m(run, f"test/{code}_mAP50"), _m(run, f"test/local_box_ap50_{code}")) if v is not None), None)
            rec = _m(run, f"test/{code}_recall")
            if ap is not None or rec is not None:
                classes.append({"모델": _family(run), "실험 조건": _condition(run), "Run": provider.run_id(run)[:8],
                                "클래스": code, "AP@50": _fmt(ap), "Recall": _fmt(rec)})
    order = {"RF-DETR Small": 0, "YOLOv11m-Seg": 1, "YOLOv8s": 2, "YOLOv8n": 3}
    rows.sort(key=lambda r: (order.get(r["모델"], 9), r["모델"], r["평가셋"], r["실험 조건"], -r["_start"]))
    same = lambda r: r["_id"]
    best = (_best(rows, "Box mAP@50 ↑", group=same) + _best(rows, "Box mAP@50–95 ↑", group=same)
            + _best(rows, "Recall ↑", group=same) + _best(rows, "추론시간 ↓ (ms)", lower=True, group=same))
    columns = ["모델", "실험 조건", "평가셋", "Box mAP@50 ↑", "Box mAP@50–95 ↑", "Recall ↑", "추론시간 ↓ (ms)", "비고", "Run"]
    if any(r["Mask mAP@50 ↑"] != MISSING for r in rows):
        columns.insert(7, "Mask mAP@50 ↑")
    help_text = {"평가셋": "기록된 test_split_id·평가 이미지 수. 다른 평가셋끼리는 순위를 매기지 않음",
                 "Box mAP@50 ↑": "Box 기준 (test/mAP50 · test/local_box_map50 · val/map50_box)",
                 "Recall ↑": "test/recall (IoU 0.5 매칭) 또는 val/recall_box",
                 "추론시간 ↓ (ms)": "test/avg_latency_ms (이미지당 평균)",
                 "Run": "같은 이름이라도 Run ID가 다르면 별도 행"}
    notes = ["굵게: 같은 평가셋 안에서 최고값 · 평가셋이 다르면 비교하지 않음"]
    if connection:
        notes.append(f"연결·스모크 확인 Run {len(connection)}개는 MLflow 상세 조회에 표시")
    return _table(rows, columns, best, notes, help_text,
                  extra={"classes": pd.DataFrame(classes), "connection": connection})


# ---- 기타 실험 ----------------------------------------------------------------------------------------
def generic_table(runs):
    keys = sorted({k for run in runs for k in provider.pairs(run, "metrics")})
    rows = [{"Run 이름": provider.run_name(run), "Run": provider.run_id(run)[:8],
             **{k: _fmt(_m(run, k), 6) if k in provider.pairs(run, "metrics") else MISSING for k in keys}} for run in runs]
    return _table(rows, ["Run 이름", "Run"] + keys, [], [], {})


BUILDERS = {WIDTH: width_table, CLASSIFICATION: classification_table, DETECTION: detection_table}


def build(experiment_name, runs):
    return BUILDERS.get(experiment_name, generic_table)(runs)
