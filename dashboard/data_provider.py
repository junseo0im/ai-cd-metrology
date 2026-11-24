"""Read-only mock repository. No DB, training, API, raw-data or upload access."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd

MOCK_ROOT = Path(__file__).resolve().parent / "mock_data"
IDENTITY_FIELDS = ("batch_id","wafer_id","image_id","physical_site_id",
                   "calibration_id","measurement_run_id","model_version")

# Stable codes for the future upload_batches.dataset_role field.
# Upload mode and dataset split assignment are independent of this role.
DATASET_ROLE_LABELS = {
    "EVALUATION":"독립 평가용",
    "TRAINING_CANDIDATE":"향후 학습 편입 후보",
    "DEVELOPMENT":"알고리즘 개발용",
}
DATASET_ROLES = tuple(DATASET_ROLE_LABELS)
REVIEW_DIE_CODES = tuple(f"{row}{column}" for row in range(1,4) for column in range(1,9))


def upload_batch_preview(*,batch_id=None,wafer_id=None,physical_site_id=None,mode,exposure_example,
                         dataset_role="EVALUATION",peb_temperature_c=None):
    """Mock batch metadata contract; dataset_role can map to upload_batches.

    Returns a session preview only. No upload, persistence, or split is created.
    Unresolved IDs remain None; this preview is not a DB insertion payload.
    """
    if dataset_role not in DATASET_ROLES:
        raise ValueError("Unknown dataset_role")
    return {"batch_id":batch_id,"wafer_id":wafer_id,"physical_site_id":physical_site_id,
            "mode":mode,"exposure_example":exposure_example,"dataset_role":dataset_role,"peb_temperature_c":peb_temperature_c,
            "source":"mock_input_preview","saved_to_database":False}


def filename_review_items():
    """Hand-authored UI examples; no image files or filename parser are opened."""
    return json.loads((MOCK_ROOT/"filename_review.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _catalog():
    return json.loads((MOCK_ROOT/"platform.json").read_text(encoding="utf-8"))


def catalog():
    return deepcopy(_catalog())


def inspection_wafer_regions(batch_id):
    """Separate UI-only 8x3 Die fixture; existing catalog/images stay unchanged.

    All three positions share the Die's existing illustration, explicitly not
    three position-specific acquisition files or parsed physical coordinates.
    """
    layout = json.loads((MOCK_ROOT/"inspection_wafer.json").read_text(encoding="utf-8"))
    valid_images = set(images(batch_id,mode="Wafer Batch")["image_id"])
    result = []
    for die in layout["dies"]:
        image_id = die["image_ids"][batch_id]
        if image_id not in valid_images:
            raise ValueError("Mock Die links must reference an existing Batch image")
        for region in die["regions"]:
            result.append({"die_id":die["die_id"],"row":die["row"],"col":die["col"],
                           "position":region["position"],"criticality":region["criticality"],
                           "image_id":image_id,"image_binding":"shared_die_mock"})
    return result


def batches(mode=None):
    rows = catalog()["batches"]
    return pd.DataFrame([r for r in rows if mode is None or r["mode"] == mode])


def images(batch_id=None, mode=None):
    rows = catalog()["images"]
    return pd.DataFrame([r for r in rows if (batch_id is None or r["batch_id"] == batch_id)
                         and (mode is None or r["mode"] == mode)])


def batch(batch_id):
    return next((r for r in catalog()["batches"] if r["batch_id"] == batch_id),None)


def image(image_id):
    return next((r for r in catalog()["images"] if r["image_id"] == image_id),None)


def asset(filename):
    root = (MOCK_ROOT/"images").resolve()
    path = (root/filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Only bundled mock image assets are permitted")
    return path


def exposure_comparison():
    return pd.DataFrame(catalog()["exposure_comparison"])


def calibration_records():
    return pd.DataFrame(catalog()["calibrations"])


def processing_matrix(batch_id):
    frame = images(batch_id)
    return frame[["image_id","metadata_state_example","calibration_state_example",
                  "metrology_state_example","db_state_example"]].rename(
        columns={"image_id":"Image ID (예시)","metadata_state_example":"Metadata (예시)",
                 "calibration_state_example":"Calibration (예시)",
                 "metrology_state_example":"계측·AI (예시)","db_state_example":"DB"})


def status_counts(batch_id):
    return images(batch_id)["status"].value_counts().rename_axis("status").reset_index(name="count")


def defect_counts(batch_id):
    return images(batch_id)["defect_example"].value_counts().rename_axis("defect").reset_index(name="count")


def measurement_summary(batch_id):
    frame = images(batch_id)
    return {name:round(float(frame[column].mean()),2) for name,column in [
        ("Outer","outer_um_example"),("Inner","inner_um_example"),("Gap","gap_um_example")]}


def identity_frame(batch_id=None):
    return images(batch_id)[list(IDENTITY_FIELDS)]

def process_observations():
    return pd.DataFrame(catalog()["exposure_observations"])


def process_trend_summary(observations=None):
    """Recompute mean/std/n from supplied observations; only bundled mock data by default."""
    frame = process_observations() if observations is None else observations.copy()
    phases = [("이전 평균 (예시)",frame[~frame["is_new"]]),("현재 평균 (예시)",frame)]
    results = []
    for phase,subset in phases:
        result = subset.groupby("exposure")["outer_um_example"].agg(
            mean="mean",std="std",sample_count="count").reset_index()
        result["std"] = result["std"].fillna(0.)
        result["series"] = phase
        results.append(result)
    return pd.concat(results,ignore_index=True)


def batch_result_tables(batch_id):
    frame = images(batch_id)
    averages = pd.DataFrame([
        {"측정 항목 (예시)":name,"평균 (µm 예시)":round(float(frame[column].mean()),3)}
        for name,column in [("Line","line_um_example"),("Gap","gap_um_example"),("Edge","edge_um_example")]
    ])
    defects = defect_counts(batch_id).rename(columns={"defect":"결함 Class (예시)","count":"검출 개수 (예시)"})
    statuses = status_counts(batch_id).rename(columns={"status":"Measurement Status (예시)","count":"이미지 수 (예시)"})
    return averages,defects,statuses

def batch_summary_table(batch_id):
    averages,defects,_ = batch_result_tables(batch_id)
    rows = [{"항목":"평균 "+r["측정 항목 (예시)"],"예시 값":f'{r["평균 (µm 예시)"]:.3f} µm'}
            for r in averages.to_dict("records")]
    rows += [{"항목":r["결함 Class (예시)"],"예시 값":str(r["검출 개수 (예시)"])+" 개 (예시)"}
             for r in defects.to_dict("records")]
    rows.append({"항목":"Measurement Status","예시 값":batch(batch_id)["status"]})
    return pd.DataFrame(rows)


def accumulation_summary(catalog_data=None):
    """Counts update from catalog records; historical points remain labeled UI examples."""
    records=catalog() if catalog_data is None else catalog_data
    fixture=json.loads((MOCK_ROOT/"control_overview.json").read_text(encoding="utf-8"))
    image_ids={r["image_id"] for r in records["images"]}
    counts={mode:sum(r["mode"]==mode for r in records["images"])
            for mode in ("Wafer Batch","Repeatability","Lighting")}
    return {"batch_count":len(records["batches"]),"image_count":len(records["images"]),
            "mode_image_counts":counts,
            "recent_image_count":len(image_ids.intersection(fixture["recent_image_ids"])),
            "history":fixture["history"]+[{"label":"현재 Mock","image_count":len(records["images"])}]}


def processing_progress_mock():
    """Fixed UI fixture; deliberately no rate calculation or actual processing status."""
    return json.loads((MOCK_ROOT/"control_overview.json").read_text(encoding="utf-8"))["progress"]


def processing_stage_table():
    return pd.DataFrame(processing_progress_mock()["stages"]).rename(
        columns={"stage":"처리 단계","state":"상태 · 예시"})


def design_comparison_table():
    """Display contract only: calibration, design values and errors are unknown."""
    return pd.DataFrame([{"항목":name,"측정 (µm)":"미연결","설계 (µm)":"미확정","오차 (%)":"미계산"}
                         for name in ("Line","Gap","Edge")])
