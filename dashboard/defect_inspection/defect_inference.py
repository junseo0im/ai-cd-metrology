import os
from functools import lru_cache
import numpy as np
import supervision as sv
from inference import get_model

MODEL_ID = "rfdetr-v8tqh/-wdlu9-2-rfdetr-seg-small-t1"

@lru_cache(maxsize=2)
def _load_model(model_id: str, api_key: str):
    return get_model(model_id=model_id, api_key=api_key)

def infer_defects(
    image_bgr: np.ndarray,
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    model_id: str = MODEL_ID,
) -> sv.Detections:
    if (
        not isinstance(image_bgr, np.ndarray)
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.dtype != np.uint8
        or image_bgr.size == 0
    ):
        raise ValueError("image_bgr은 비어 있지 않은 H×W×3 BGR uint8 이미지여야 합니다.")

    if not 0 <= confidence <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("confidence와 iou_threshold는 0~1 사이여야 합니다.")

    # 운영체제 환경 변수에서 API 키를 안전하게 불러옵니다.
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY 환경 변수가 설정되지 않았습니다.")

    model = _load_model(model_id, api_key)
    result = model.infer(
        image_bgr,
        confidence=confidence,
        iou_threshold=iou_threshold,
    )[0]

    detections = sv.Detections.from_inference(result)
    if "class_name" not in detections.data:
        detections.data["class_name"] = np.array([], dtype=str)

    return detections