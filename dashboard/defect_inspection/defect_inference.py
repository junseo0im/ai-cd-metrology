"""로컬 ONNX 추론 연결 (Roboflow API 미사용).

분석 실행과 같은 엔진 파일(engines/defect/defect_adapter_r33/defect_inference.py)을 그대로 불러와
같은 weights.onnx·설정(runtime_data/config/defect_local_model.json)·SHA-256 검사로 추론합니다.
엔진 폴더는 IDA_TEAM_DEFECT_DIR이 있으면 그 값을, 없으면 저장소의 engines 사본을 씁니다.
"""
import importlib.util
import os
from pathlib import Path

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "engines" / "defect" / "defect_adapter_r33"


def _load_engine():
    folder = Path(os.environ.get("IDA_TEAM_DEFECT_DIR") or _DEFAULT_DIR)
    path = folder / "defect_inference.py"
    if not path.is_file():
        raise ImportError(f"로컬 결함 추론 엔진 파일이 없습니다: {path}", name="defect_inference")
    spec = importlib.util.spec_from_file_location("ida_local_defect_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_engine = _load_engine()
MODEL_ID = _engine.MODEL_ID
LocalModelError = _engine.LocalModelError
DEFAULT_CONFIG_PATH = _engine.DEFAULT_CONFIG_PATH


def infer_defects(image_bgr, confidence=0.25, iou_threshold=0.45, model_id=MODEL_ID):
    """기존 호출 형태 유지. model_id는 기록용이며 서비스로 조회하지 않습니다."""
    return _engine.infer_defects(image_bgr, confidence, iou_threshold)
