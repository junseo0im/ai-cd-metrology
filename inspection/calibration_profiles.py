"""Explicit approved native-pixel calibration profiles. No engine/default fallback."""
from copy import deepcopy
from pathlib import Path
import json
import math
PROFILES_PATH=Path(__file__).resolve().parents[1]/"runtime_data/config/calibration_profiles.json"

def validate_profile(profile):
    if not isinstance(profile,dict):raise ValueError("Calibration profile must be an object")
    for key in ("profile_id","calibration_id"):
        if not isinstance(profile.get(key),str) or not profile[key].strip():raise ValueError("Calibration profile ID is required")
    for key in ("magnification","um_per_px"):
        value=profile.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError("Calibration profile needs positive finite "+key)
    if profile.get("unit")!="um/native_px":raise ValueError("Calibration unit must be um/native_px")
    if type(profile.get("approved")) is not bool:raise ValueError("Calibration approval must be explicit")
    scope=profile.get("scope",{})
    for key in ("batch_ids","wafer_ids","condition_ids","image_ids"):
        values=scope.get(key)
        if not isinstance(values,list) or not values or any(not isinstance(v,str) or not v.strip() for v in values) or len(values)!=len(set(values)):
            raise ValueError("Calibration scope requires explicit unique "+key)
    evidence=profile.get("evidence",{})
    if profile["approved"] and not (evidence.get("source")=="user_confirmation" and evidence.get("same_capture_setup") is True):
        raise ValueError("Approved profile requires capture-setting evidence")
    return profile

def load_profiles(path=None):
    path=Path(path) if path is not None else PROFILES_PATH
    if not path.exists():return []
    config=json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version")!=1 or not isinstance(config.get("profiles"),list):
        raise ValueError("Unsupported calibration profile configuration")
    result=[validate_profile(p) for p in config["profiles"]]
    if len({p["profile_id"] for p in result})!=len(result):raise ValueError("Duplicate calibration profile ID")
    return result

UPLOAD_MODES=("wafer_batch","repeatability","lighting","single_image")
UPLOAD_DEFAULT_SCOPE="upload_default"

def validate_upload_default(default,profiles):
    """A new-upload default only REUSES an approved exact profile's calibration.

    It never defines a new coefficient: calibration_id/magnification/um_per_px/unit
    must equal the approved source profile. Its own shared condition_id must not
    collide with any exact profile condition (no retroactive scope widening).
    """
    if not isinstance(default,dict):raise ValueError("Upload default must be an object")
    for key in ("profile_id","source_profile_id","calibration_id","condition_id"):
        if not isinstance(default.get(key),str) or not default[key].strip():raise ValueError("Upload default needs "+key)
    for key in ("magnification","um_per_px"):
        value=default.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError("Upload default needs positive finite "+key)
    if default.get("unit")!="um/native_px":raise ValueError("Calibration unit must be um/native_px")
    if type(default.get("approved")) is not bool:raise ValueError("Upload default approval must be explicit")
    modes=default.get("upload_modes")
    if (not isinstance(modes,list) or not modes or len(modes)!=len(set(modes))
            or any(mode not in UPLOAD_MODES for mode in modes)):
        raise ValueError("Upload default requires explicit unique upload_modes")
    if "lighting" in modes:raise ValueError("Lighting uploads carry explicit capture settings; no default")
    evidence=default.get("evidence",{})
    if default["approved"] and not (evidence.get("source")=="user_confirmation" and evidence.get("same_capture_setup") is True):
        raise ValueError("Approved upload default requires capture-setting evidence")
    source=next((p for p in profiles if p["profile_id"]==default["source_profile_id"]),None)
    if source is None or source["approved"] is not True:raise ValueError("Upload default source profile must be approved")
    if any(default[k]!=source[k] for k in ("calibration_id","magnification","um_per_px","unit")):
        raise ValueError("Upload default must reuse the approved source calibration unchanged")
    if any(default["condition_id"] in p["scope"]["condition_ids"] for p in profiles):
        raise ValueError("Upload default condition_id must differ from exact-scope conditions")
    if default["profile_id"] in {p["profile_id"] for p in profiles}:raise ValueError("Duplicate calibration profile ID")
    return default

def load_upload_defaults(path=None,profiles=None):
    path=Path(path) if path is not None else PROFILES_PATH
    if not path.exists():return []
    config=json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version")!=1:raise ValueError("Unsupported calibration profile configuration")
    defaults=config.get("upload_defaults",[])
    if not isinstance(defaults,list):raise ValueError("upload_defaults must be a list")
    profiles=load_profiles(path) if profiles is None else profiles
    result=[validate_upload_default(d,profiles) for d in defaults]
    for key in ("profile_id","condition_id"):
        if len({d[key] for d in result})!=len(result):raise ValueError("Duplicate upload default "+key)
    return result

def upload_default_note(default):
    return json.dumps({"schema_version":1,"calibration_profile_id":default["profile_id"],
      "scope":UPLOAD_DEFAULT_SCOPE,"approval_source":"user_confirmation","same_capture_setup_confirmed":True},
      ensure_ascii=False,sort_keys=True,separators=(",",":"))

def upload_default_condition(default):
    """Exact acquisition_conditions row values; unspecified settings stay NULL."""
    return {"condition_id":default["condition_id"],"magnification":float(default["magnification"]),
      "illumination_setting":None,"auto_exposure":None,"camera_setting_json":None,
      "notes":upload_default_note(default)}

def select_upload_default(defaults,upload_mode):
    """Exactly one approved default per mode, otherwise none (fail closed)."""
    matches=[d for d in defaults if d["approved"] is True and upload_mode in d["upload_modes"]]
    return matches[0] if len(matches)==1 else None

def condition_note(profile,batch_id):
    return json.dumps({"schema_version":1,"calibration_profile_id":profile["profile_id"],
      "batch_id":batch_id,"approval_source":"user_confirmation","same_capture_setup_confirmed":True},
      ensure_ascii=False,sort_keys=True,separators=(",",":"))

def resolve_calibration(profiles,*,batch_id,image_id,wafer_id,condition_id,magnification,condition_notes,
                        upload_defaults=()):
    matches=[];default_matches=[]
    for d in upload_defaults:
        if d["approved"] is not True or condition_id!=d["condition_id"]:continue
        if isinstance(magnification,bool) or magnification!=d["magnification"]:continue
        try:note=json.loads(condition_notes)
        except (TypeError,ValueError):continue
        if note!=json.loads(upload_default_note(d)):continue
        default_matches.append(d)
    for p in profiles:
        if not p["approved"]:continue
        scope=p["scope"]
        if not all(value in scope[key] for key,value in (
            ("batch_ids",batch_id),("wafer_ids",wafer_id),("image_ids",image_id),("condition_ids",condition_id))):continue
        if isinstance(magnification,bool) or magnification!=p["magnification"]:continue
        try:note=json.loads(condition_notes)
        except (TypeError,ValueError):continue
        if not isinstance(note,dict):continue
        if not (note.get("batch_id")==batch_id and note.get("calibration_profile_id")==p["profile_id"]
          and note.get("approval_source")=="user_confirmation" and note.get("same_capture_setup_confirmed") is True):continue
        matches.append(p)
    if len(matches)+len(default_matches)>1:raise ValueError("Ambiguous approved calibration profiles")
    if default_matches:
        d=default_matches[0]
        return {"profile_id":d["profile_id"],"unit":d["unit"],"source_profile_id":d["source_profile_id"],
          "calibration":{"calibration_id":d["calibration_id"],"um_per_px":d["um_per_px"],"magnification":d["magnification"]},
          "evidence":deepcopy(d["evidence"]),"scope_match":"APPROVED_UPLOAD_DEFAULT_CONDITION"}
    if not matches:return None
    p=matches[0]
    return {"profile_id":p["profile_id"],"unit":p["unit"],
      "calibration":{"calibration_id":p["calibration_id"],"um_per_px":p["um_per_px"],"magnification":p["magnification"]},
      "evidence":deepcopy(p["evidence"]),"scope_match":"EXACT_BATCH_WAFER_CONDITION_IMAGE"}
