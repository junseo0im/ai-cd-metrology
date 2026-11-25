"""결함 검사 탭의 모델 추론: Roboflow API 대신 로컬 RF-DETR ONNX를 사용합니다.

분석 실행 엔진(engines/defect/defect_adapter_r33/defect_inference.py)을 그대로 불러와 쓰므로
대시보드 분석과 이 탭이 같은 weights.onnx·같은 전처리·같은 SHA-256 검사를 공유합니다.
API 키·인터넷 연결은 필요 없습니다. 가중치 배치는 engines/README.md 참고.
"""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE_DIR = ROOT / "engines" / "defect" / "defect_adapter_r33"


def _engine_dir():
  # 분석 실행과 같은 환경변수를 따르고, 없으면 저장소 안의 엔진 폴더를 씁니다.
  value = os.environ.get("IDA_TEAM_DEFECT_DIR")
  return Path(value) if value else DEFAULT_ENGINE_DIR


def _load_engine():
  path = _engine_dir() / "defect_inference.py"
  if not path.is_file():
    raise ImportError(f"로컬 결함 추론 엔진 파일이 없습니다: {path}", name="defect_inference")
  spec = importlib.util.spec_from_file_location("ida_local_defect_inference", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


_engine = _load_engine()

MODEL_ID = _engine.MODEL_ID
DEFAULT_CONFIG_PATH = _engine.DEFAULT_CONFIG_PATH
LocalModelError = _engine.LocalModelError
infer_defects = _engine.infer_defects
infer_defects_with_metadata = _engine.infer_defects_with_metadata
