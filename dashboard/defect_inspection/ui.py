"""Roboflow 탭의 결함 검사 화면 (Streamlit 전용, Gradio 미사용).

모델 추론은 각자 PC의 로컬 weights.onnx로 수행하며(API 키 불필요), 결과를 운영 DB에 저장하지 않습니다.
설정은 본문 expander 안에 두어 대시보드 공통 sidebar를 건드리지 않습니다.
"""
import io
import os
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
CONF_LABEL = {"rule": "규칙 검출", "from_s": "S 안 이물",
              "blur": "흐린 패턴", "heuristic": "흐린 패턴"}


@st.cache_resource(show_spinner="검사 모듈을 불러오는 중...")
def _engine():
  """inference·supervision 등 무거운 의존성은 탭을 열 때 한 번만 불러온다."""
  from . import aoi_pipeline, defect_rules
  return aoi_pipeline, defect_rules


def _decode(file_bytes):
  return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)


def _settings(rules, key):
  with st.expander("검사 설정", expanded=False):
    c1, c2, c3 = st.columns(3)
    s = {
        "conf": c1.slider("신뢰도", 0.05, 0.95, 0.25, 0.05, key=f"{key}_conf"),
        "iou": c2.slider("IoU", 0.10, 0.90, 0.45, 0.05, key=f"{key}_iou"),
        "nms_mode": c3.selectbox("겹친 박스 정리", rules.NMS_CHOICES, key=f"{key}_nms"),
    }
    c4, c5 = st.columns(2)
    s["line_rule"] = c4.checkbox("line/gap 규칙 적용", True, key=f"{key}_line")
    s["find_shorts"] = c4.checkbox("모델이 놓친 S·PD 찾기", True, key=f"{key}_short")
    s["split_particle"] = c4.checkbox("큰 이물은 P + S 함께 표시", True, key=f"{key}_split")
    s["protect_conf"] = c5.checkbox("신뢰도 50% 이상 O·PD·S 보호", True, key=f"{key}_prot")
    s["scratch_rule"] = c5.checkbox("세로 직선 SC → PD", True, key=f"{key}_sc")
    s["deskew"] = c5.checkbox("자동 수평 보정", True, key=f"{key}_deskew")
    c6, c7 = st.columns(2)
    s["masks"] = c6.checkbox("마스크 표시", True, key=f"{key}_mask")
    s["corners"] = c7.checkbox("박스 모서리 표시", True, key=f"{key}_corner")
  return s


def _inspect(pipeline, image_bgr, s):
  return pipeline.run_aoi_pipeline(
      image_bgr,
      conf_threshold=s["conf"], iou_threshold=s["iou"],
      show_corners=s["corners"], show_masks=s["masks"],
      filter_line_only=s["line_rule"], enable_deskew=s["deskew"],
      find_shorts=s["find_shorts"], split_particle=s["split_particle"],
      protect_conf=s["protect_conf"], scratch_rule=s["scratch_rule"],
      nms_mode=s["nms_mode"])


def _table(rules, detections):
  label = {code: name for code, name, _ in rules.CLASS_INFO}
  names = list(detections.data.get("class_name", []))
  sources = list(detections.data.get("source", ["model"] * len(detections)))
  positions = list(detections.data.get("position", ["-"] * len(detections)))
  rows = []
  for i, box in enumerate(detections.xyxy):
    conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
    area = int(detections.mask[i].sum()) if detections.mask is not None else 0
    x1, y1, x2, y2 = (int(v) for v in box)
    rows.append({
        "No.": i + 1,
        "결함": f"{str(names[i]).upper()} ({label.get(str(names[i]), '')})",
        "신뢰도": CONF_LABEL.get(str(sources[i]), f"{conf * 100:.0f}%"),
        "위치": f"({x1}, {y1}) ~ ({x2}, {y2})",
        "line/gap": str(positions[i]),
        "마스크 면적(px)": f"{area:,}",
    })
  return pd.DataFrame(rows)


def _single(pipeline, rules, s, key):
  up = st.file_uploader("검사할 이미지", type=IMAGE_TYPES, key=f"{key}_one")
  if up is None:
    return
  bgr = _decode(up.getvalue())
  if bgr is None:
    st.error("이미지를 읽을 수 없습니다.")
    return
  with st.spinner("검사 중..."):
    result = _inspect(pipeline, bgr, s)
  left, right = st.columns(2)
  left.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="원본", use_container_width=True)
  right.image(cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB),
              caption="검사 및 마킹 결과", use_container_width=True)
  total = result["total_defects"]
  if total:
    st.error("FAIL (불량) — 총 " + str(total) + "건 검출")
  else:
    st.success("PASS (정상) — 검출된 결함 없음")
  angle = result["skew_angle"]
  st.caption("수평 보정: " + ("보정 없음" if not angle else f"{abs(angle):.2f}° 회전")
             + " · 후처리 " + result["postprocess_version"])
  frame = _table(rules, result["detections"])
  if len(frame):
    st.dataframe(frame, hide_index=True, use_container_width=True)
  if result.get("rule_log"):
    with st.expander("판정 근거 (결함별)"):
      for line in result["rule_log"]:
        st.markdown("- " + line)
  ok, encoded = cv2.imencode(".png", result["annotated_image"])
  if ok:
    st.download_button("결과 이미지 내려받기", encoded.tobytes(),
                       "result_" + up.name.rsplit(".", 1)[0] + ".png", "image/png",
                       key=f"{key}_dl")


def _batch(pipeline, rules, s, key):
  ups = st.file_uploader("검사할 이미지 여러 장", type=IMAGE_TYPES,
                         accept_multiple_files=True, key=f"{key}_many")
  if not ups or not st.button("일괄 검사 시작", key=f"{key}_run", type="primary"):
    return
  rows, archive = [], io.BytesIO()
  bar = st.progress(0.0)
  with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
    for i, item in enumerate(ups):
      bgr = _decode(item.getvalue())
      if bgr is None:
        rows.append({"파일명": item.name, "판정": "읽기 실패", "결함 수": 0, "상세": ""})
      else:
        result = _inspect(pipeline, bgr, s)
        ok, encoded = cv2.imencode(".png", result["annotated_image"])
        if ok:
          zf.writestr("result_" + item.name.rsplit(".", 1)[0] + ".png", encoded.tobytes())
        frame = _table(rules, result["detections"])
        detail = ", ".join(f"{r['결함'].split(' ')[0]}({r['신뢰도']})" for _, r in frame.iterrows())
        rows.append({"파일명": item.name, "판정": "FAIL" if len(frame) else "PASS",
                     "결함 수": len(frame), "상세": detail})
      bar.progress((i + 1) / len(ups), text=f"{i + 1}/{len(ups)}  {item.name}")
  summary = pd.DataFrame(rows)
  fails = int((summary["판정"] == "FAIL").sum())
  st.markdown(f"**총 {len(summary)}장 · FAIL {fails}장 · PASS {len(summary) - fails}장**")
  st.dataframe(summary, hide_index=True, use_container_width=True)
  d1, d2 = st.columns(2)
  d1.download_button("결과 이미지 ZIP", archive.getvalue(), "inspection_results.zip",
                     "application/zip", key=f"{key}_zip")
  d2.download_button("결과 CSV", summary.to_csv(index=False).encode("utf-8-sig"),
                     "inspection_summary.csv", "text/csv", key=f"{key}_csv")


def available():
  """검사에 필요한 패키지와 로컬 ONNX 엔진이 준비됐는지. (missing 패키지, 엔진 오류 메시지 또는 None)"""
  missing = []
  for module in ("cv2", "supervision", "onnxruntime"):
    try:
      __import__(module)
    except ImportError:
      missing.append({"cv2": "opencv-python"}.get(module, module))
  if missing:
    return missing, None
  try:
    from . import defect_inference
  except ImportError as exc:
    return missing, str(exc)
  if not defect_inference.DEFAULT_CONFIG_PATH.is_file():
    return missing, ("로컬 모델 설정이 없습니다: runtime_data\\config\\defect_local_model.json "
                     "(engines/README.md 준비 1단계)")
  return missing, None


def render_inspection(key="defect_insp"):
  """Roboflow 탭 본문. 준비되지 않은 환경에서는 안내만 표시한다."""
  missing, engine_error = available()
  if missing:
    st.info("결함 검사 실행에 필요한 패키지가 없습니다: " + ", ".join(missing))
    st.markdown("```powershell\n.\\dashboard\\.venv\\Scripts\\python.exe -m pip install "
                + " ".join(missing) + "\n```")
    return
  if engine_error:
    st.info(engine_error)
    return
  pipeline, rules = _engine()
  st.caption("RF-DETR (로컬 ONNX · weights.onnx) + 도메인 규칙 후처리 · API 호출 없음 · "
             "결과는 운영 DB에 저장되지 않습니다.")
  s = _settings(rules, key)
  single_tab, batch_tab = st.tabs(["단일 이미지", "여러 이미지 일괄 검사"])
  with single_tab:
    _single(pipeline, rules, s, key)
  with batch_tab:
    _batch(pipeline, rules, s, key)
