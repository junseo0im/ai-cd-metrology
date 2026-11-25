"""AOI 결함 검사 파이프라인 (대시보드 연결용).

모델 추론은 defect_inference.infer_defects 를 그대로 쓰고,
후처리는 defect_rules.py (r26 line/gap 규칙)로 교체했습니다.
run_aoi_pipeline() 의 입력 인자와 반환 dict 키는 기존과 같습니다.
"""
import cv2
import numpy as np
import supervision as sv

import defect_rules as rules
from defect_inference import MODEL_ID, infer_defects

CLASS_IDS = rules.CLASS_IDS  # {"o": 0, "p": 1, "pd": 2, "s": 3, "sc": 4}
POSTPROCESS_VERSION = rules.RULES_VERSION


def evaluate_pattern_contrast(image_bgr):
  """패턴 대비 점수 (기존과 동일). 너무 흐린 이미지 판단에 사용."""
  h, w = image_bgr.shape[:2]
  roi = image_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.05):int(w * 0.95)]
  if roi.size == 0:
    return 0.0
  redness = roi[:, :, 2].astype(np.float32) - roi[:, :, 1].astype(np.float32)
  col_profile = np.mean(redness, axis=0)
  score = np.percentile(col_profile, 85) - np.percentile(col_profile, 15)
  return round(float(score), 1)


def _add_blurry_pd(image_bgr, detections, contrast_score, pd_min_contrast):
  """패턴 대비가 너무 낮은(흐린) 이미지는 이미지 전체를 PD 1건으로 추가 (기존과 동일).
  추가된 PD는 항상 0번째, source='heuristic'."""
  h, w = image_bgr.shape[:2]
  if "pd" in list(detections.data.get("class_name", [])):
    return detections, False
  pad = min(12, max(0, min(h, w) // 4))
  pd_box = np.array([[pad, pad, max(pad + 1, w - pad), max(pad + 1, h - pad)]], dtype=np.float32)
  pd_box[:, [0, 2]] = np.clip(pd_box[:, [0, 2]], 0, w)
  pd_box[:, [1, 3]] = np.clip(pd_box[:, [1, 3]], 0, h)
  pd_conf = round(min(0.99, max(0.60, 1.0 - contrast_score / max(1.0, pd_min_contrast * 1.5))), 2)
  pd_det = sv.Detections(
      xyxy=pd_box,
      confidence=np.array([pd_conf], dtype=np.float32),
      class_id=np.array([CLASS_IDS["pd"]], dtype=int),
      mask=np.zeros((1, h, w), dtype=bool) if detections.mask is not None or len(detections) == 0 else None,
      data={"class_name": np.array(["pd"]), "source": np.array(["heuristic"]),
            "position": np.array(["-"])},
  )
  return rules._concat_detections(pd_det, detections), True


def run_aoi_pipeline(
    image_bgr,
    conf_threshold=0.25,
    iou_threshold=0.45,
    pd_min_contrast=45,
    show_corners=True,
    show_masks=True,
    filter_line_only=True,
    filter_uturn_open=True,
    enable_deskew=True,
    # ↓ r26 후처리 세부 옵션 (기본값 = Gradio 앱 기본 설정)
    find_shorts=True,
    split_particle=True,
    protect_conf=True,
    scratch_rule=True,
    nms_mode=rules.NMS_PER_CLASS,
):
  """이미지 1장 결함 검사.

  filter_line_only  : r26 line/gap 규칙 전체 on/off
                      (P·O·PD는 line 위만, S는 line끼리 연결된 것만, gap 막은 P→S 등)
  filter_uturn_open : 호환용 인자. r26에서는 line 끝 U자 open 제거가 line/gap 규칙에
                      포함되어 filter_line_only 를 따른다.
  pd_min_contrast   : 패턴 대비가 이 값보다 낮고 line/gap 줄무늬도 찾지 못하면
                      흐린 이미지로 보고 전체 PD 1건 추가 (0이면 사용 안 함)
  bbox 좌표 기준: enable_deskew=True 이고 회전했다면 '수평 보정 후 크롭된 이미지'(analyzed_image)
  """
  if enable_deskew:
    analyzed_image, skew_angle = rules.deskew_image(image_bgr)
  else:
    analyzed_image, skew_angle = image_bgr.copy(), 0.0

  # 패턴이 흐리면(line/gap 구분 불가) 모델·규칙 없이 이미지 전체를 PD 1건으로 판정
  blurry, blur_score = rules.is_blurry_pattern(analyzed_image)
  if blurry:
    annotated_bgr, detections, info = rules._draw_blur_pd(
        analyzed_image, skew_angle, blur_score, show_corners)
    return {
        "annotated_image": annotated_bgr,
        "status": "FAIL",
        "total_defects": len(detections),
        "contrast_score": evaluate_pattern_contrast(analyzed_image),
        "skew_angle": float(round(skew_angle, 2)),
        "detections": detections,
        "model_id": MODEL_ID,
        "raw_detections": rules._empty_detections(),
        "heuristic_pd_added": True,
        "analyzed_image": analyzed_image,
        "postprocess_version": POSTPROCESS_VERSION,
        "rule_log": info.get("log", []),
        "line_gap_stripes": "-",
    }

  raw_detections = infer_defects(analyzed_image, confidence=conf_threshold,
                                 iou_threshold=iou_threshold)

  contrast_score = evaluate_pattern_contrast(analyzed_image)
  low_contrast = pd_min_contrast > 0 and contrast_score < pd_min_contrast

  annotated_bgr, detections, info = rules.run_rules(
      analyzed_image,
      raw_detections,
      iou_threshold=iou_threshold,
      show_corners=show_corners,
      show_masks=show_masks,
      nms_mode=nms_mode,
      line_rule=filter_line_only,
      scratch_rule=scratch_rule,
      find_shorts=find_shorts,
      split_particle=split_particle,
      protect_conf=protect_conf,
      angle=skew_angle,
  )

  # 흐린 이미지 판정: 대비 점수가 낮고, line/gap 줄무늬도 찾지 못했을 때만.
  #   (노란 패턴은 붉은기 대비가 원래 낮아서 대비 점수만 보면 선명한 사진도 흐림으로 오판함)
  stripes_found = info.get("stripes", "-") != "-"
  is_blurry_pd = low_contrast and not (filter_line_only and stripes_found)
  heuristic_pd_added = False
  if is_blurry_pd:
    detections, heuristic_pd_added = _add_blurry_pd(
        analyzed_image, detections, contrast_score, pd_min_contrast)
    if heuristic_pd_added:
      # 전체 PD 박스까지 포함해 다시 마킹만 함 (규칙은 모두 끄고 그리기만)
      log = info.get("log", [])
      annotated_bgr, detections, _ = rules._apply_rules_and_draw(
          analyzed_image, detections, skew_angle, show_corners, show_masks,
          False, False, False, False, False, False)
      info["log"] = log + [f"흐린 이미지 (패턴 대비 {contrast_score} < {pd_min_contrast}) → 전체 PD 추가"]

  total_defects = len(detections)
  return {
      "annotated_image": annotated_bgr,
      "status": "PASS" if total_defects == 0 else "FAIL",
      "total_defects": total_defects,
      "contrast_score": contrast_score,
      "skew_angle": float(round(skew_angle, 2)),
      "detections": detections,
      "model_id": MODEL_ID,
      "raw_detections": raw_detections,
      "heuristic_pd_added": heuristic_pd_added,
      "analyzed_image": analyzed_image,
      # ↓ 추가 정보 (기존 코드에서는 없던 키, 사용하지 않아도 됨)
      "postprocess_version": POSTPROCESS_VERSION,
      "rule_log": info.get("log", []),
      "line_gap_stripes": info.get("stripes", "-"),
  }
