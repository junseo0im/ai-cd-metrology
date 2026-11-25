"""Frozen local RF-DETR ONNX inference. No Roboflow SDK, keys, downloads or disk cache."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
import onnxruntime as ort
import supervision as sv

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "runtime_data/config/defect_local_model.json"
# Historical model revision identifier only; never resolved through a service.
MODEL_ID = "rfdetr-v8tqh/-wdlu9-2-rfdetr-seg-small-t1"


class LocalModelError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _local_path(value, base=ROOT):
    text = str(value)
    if "://" in text or text.startswith(("\\\\", "//")):
        raise LocalModelError("LOCAL_MODEL_PATH_INVALID", "Only local filesystem paths are allowed")
    path = Path(text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalRFDETR:
    def __init__(self, config_path):
        started = time.perf_counter()
        self.config_path = config_path
        try:
            self.config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise LocalModelError("LOCAL_MODEL_CONFIG_INVALID", f"Cannot read local model configuration: {config_path}") from error
        config = self.config
        required = ["model_path", "model_sha256", "model_version", "input_shape",
                    "logit_class_names", "background_index", "mean", "std"]
        if any(key not in config for key in required):
            raise LocalModelError("LOCAL_MODEL_CONFIG_INVALID", "Missing required local model configuration fields")
        self.model_path = _local_path(config["model_path"])
        if not self.model_path.is_file():
            raise LocalModelError("LOCAL_MODEL_NOT_FOUND", f"Configured ONNX file does not exist: {self.model_path}")
        if self.model_path.suffix.lower() != ".onnx":
            raise LocalModelError("LOCAL_MODEL_FORMAT_UNSUPPORTED", "This revision requires its verified ONNX artifact")
        if _sha256(self.model_path) != config["model_sha256"]:
            raise LocalModelError("LOCAL_MODEL_HASH_MISMATCH", "Configured ONNX SHA-256 does not match the local file")
        if config["input_shape"] != [1, 3, 384, 384]:
            raise LocalModelError("LOCAL_MODEL_CONFIG_INVALID", "Expected the verified revision's 1x3x384x384 input")
        if config["logit_class_names"] != ["background_class83422", "o", "p", "pd", "s", "sc"] or config["background_index"] != 0:
            raise LocalModelError("LOCAL_MODEL_CLASS_SCHEMA_INVALID", "Class order/background differs from the verified export")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.enable_mem_pattern = True
        try:
            self.session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                                providers=["CPUExecutionProvider"])
        except Exception as error:
            raise LocalModelError("LOCAL_MODEL_LOAD_FAILED", f"ONNX Runtime could not load the configured file: {type(error).__name__}") from error
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "input" or inputs[0].shape != [1, 3, 384, 384] or inputs[0].type != "tensor(float)":
            raise LocalModelError("LOCAL_MODEL_SIGNATURE_INVALID", "ONNX input signature does not match its manifest")
        if [item.name for item in outputs] != ["dets", "labels", "masks"] or outputs[1].shape[-1] != 6:
            raise LocalModelError("LOCAL_MODEL_SIGNATURE_INVALID", "ONNX output signature does not match its manifest")
        self.mean = np.asarray(config["mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(config["std"], dtype=np.float32).reshape(1, 1, 3)
        if not np.allclose(self.mean.reshape(-1), [0.485, 0.456, 0.406]) or not np.allclose(self.std.reshape(-1), [0.229, 0.224, 0.225]):
            raise LocalModelError("LOCAL_MODEL_CONFIG_INVALID", "Normalization differs from the verified export")
        self.initial_load_seconds = time.perf_counter() - started

    def infer(self, image_bgr, confidence):
        height, width = image_bgr.shape[:2]
        started = time.perf_counter()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb /= 255.0
        rgb = (rgb - self.mean) / self.std
        # This exact export used Stretch to 384x384, not letterboxing.
        resized = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_LINEAR)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)
        preprocess_seconds = time.perf_counter() - started
        started = time.perf_counter()
        boxes, logits, masks = self.session.run(["dets", "labels", "masks"], {"input": tensor})
        inference_seconds = time.perf_counter() - started
        started = time.perf_counter()
        if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[-1] != 4 or logits.shape[:2] != boxes.shape[:2] or logits.shape[-1] != 6:
            raise LocalModelError("LOCAL_MODEL_OUTPUT_INVALID", "Unexpected RF-DETR output dimensions")
        if masks.ndim != 4 or masks.shape[:2] != boxes.shape[:2] or not all(np.isfinite(x).all() for x in [boxes, logits, masks]):
            raise LocalModelError("LOCAL_MODEL_OUTPUT_INVALID", "Invalid masks or nonfinite model outputs")
        logits = logits.astype(np.float32)
        z = np.exp(-np.abs(logits))
        scores = np.where(logits >= 0, 1 / (1 + z), z / (1 + z))[0].reshape(-1)
        # Preserve original SDK policy: global top-300 first, strict confidence > threshold,
        # then remove the background logit. No objectness multiplication or raw NMS.
        if scores.size > 300:
            top = np.argpartition(-scores, 300)[:300]
            order = top[np.argsort(-scores[top])]
        else:
            order = np.argsort(-scores)
        order = order[scores[order] > confidence]
        query_ids, logit_ids = order // 6, order % 6
        keep = logit_ids != 0
        order, query_ids, class_ids = order[keep], query_ids[keep], logit_ids[keep] - 1
        chosen = boxes[0, query_ids].astype(np.float32)
        xyxy = np.concatenate([chosen[:, :2] - 0.5 * chosen[:, 2:],
                               chosen[:, :2] + 0.5 * chosen[:, 2:]], axis=1)
        xyxy *= np.asarray([width, height, width, height], dtype=np.float32)
        np.clip(xyxy, [0, 0, 0, 0], [width, height, width, height], out=xyxy)
        if len(xyxy) and (np.any(xyxy[:, 2] <= xyxy[:, 0]) or np.any(xyxy[:, 3] <= xyxy[:, 1])):
            raise LocalModelError("LOCAL_MODEL_OUTPUT_INVALID", "Model produced a degenerate clipped bbox")
        names = np.asarray(["o", "p", "pd", "s", "sc"], dtype=str)[class_ids]
        # Match the original accurate segmentation decode: resize logits, threshold at 0,
        # largest external contour by vertex count, then rasterize for the existing renderer.
        decoded = []
        for query in query_ids:
            full = cv2.resize(masks[0, query].astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            contours = cv2.findContours((full > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            raster = np.zeros((height, width), dtype=np.uint8)
            if contours:
                contour = max(contours, key=len)
                if len(contour) >= 3:
                    cv2.fillPoly(raster, [contour], 1)
            decoded.append(raster.astype(bool))
        mask_array = np.stack(decoded) if decoded else np.zeros((0, height, width), dtype=bool)
        result = sv.Detections(xyxy=xyxy, confidence=scores[order].astype(np.float32),
                               class_id=class_ids.astype(int), mask=mask_array,
                               data={"class_name": names})
        metadata = {
            "model_version": self.config["model_version"], "model_path": str(self.model_path),
            "model_sha256": self.config["model_sha256"], "backend": "onnxruntime_cpu",
            "input_shape": [1, 3, 384, 384], "bbox_coordinate_system": "INPUT_IMAGE",
            "preprocessing_seconds": preprocess_seconds, "inference_seconds": inference_seconds,
            "decode_seconds": time.perf_counter() - started,
            "model_initial_load_seconds": self.initial_load_seconds,
        }
        return result, metadata


@lru_cache(maxsize=2)
def _runtime(path_string, config_mtime_ns):
    return LocalRFDETR(Path(path_string))


def infer_defects_with_metadata(image_bgr, confidence=0.25, iou_threshold=0.45, config_path=None):
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8 or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty HxWx3 BGR uint8 array")
    if not np.isfinite(confidence) or not np.isfinite(iou_threshold) or not 0 <= confidence <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("confidence and iou_threshold must be finite values in [0,1]")
    path = _local_path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise LocalModelError("LOCAL_MODEL_CONFIG_NOT_FOUND", f"Local model configuration does not exist: {path}")
    started = time.perf_counter()
    hits = _runtime.cache_info().hits
    model = _runtime(str(path), path.stat().st_mtime_ns)
    load_seconds = time.perf_counter() - started
    reused = _runtime.cache_info().hits > hits
    result, info = model.infer(image_bgr, confidence)
    info.update(model_load_seconds=load_seconds, model_session_reused=reused)
    return result, info


def infer_defects(image_bgr, confidence=0.25, iou_threshold=0.45, config_path=None):
    return infer_defects_with_metadata(image_bgr, confidence, iou_threshold, config_path)[0]
