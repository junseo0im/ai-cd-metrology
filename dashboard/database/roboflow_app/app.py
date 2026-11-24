import gradio as gr
from inference import get_model
import supervision as sv
import numpy as np
import cv2
import pandas as pd
import os
import zipfile
import tempfile

# 1. 단일 통합 모델 ID. API 키는 환경변수 ROBOFLOW_API_KEY 또는 같은 폴더의 .env(Git 제외)에서 읽는다.
MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "rf-detr-zpad9/1")  # 5종 통합 Instance Segmentation 모델 (228장 v1)


def _api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not key and os.path.isfile(env_file):
        with open(env_file, encoding="utf-8") as handle:
            for line in handle:
                name, _, value = line.strip().partition("=")
                if name == "ROBOFLOW_API_KEY" and value:
                    key = value
    if not key:
        raise RuntimeError("ROBOFLOW_API_KEY가 설정되지 않았습니다 (환경변수 또는 roboflow_app/.env).")
    return key


_MODEL = None


def get_loaded_model():
    """Load the model on first inference (not at import)."""
    global _MODEL
    if _MODEL is None:
        print(f"통합 모델({MODEL_ID})을 불러오는 중입니다...")
        _MODEL = get_model(model_id=MODEL_ID, api_key=_api_key())
        print("단일 통합 모델 로드 완료!")
    return _MODEL

# 2. 시인성 높은 클래스별 색상 팔레트
CUSTOM_COLORS = sv.ColorPalette.from_hex([
    "#FF3B30",  # 선명한 빨강
    "#00E5FF",  # 형광 시안
    "#FFCC00",  # 밝은 노랑
    "#AF52DE",  # 보라
    "#34C759",  # 초록
])

# 3-0. 기울어진 세로 패턴을 수직(90도)으로 반듯하게 자동 회전 보정하는 함수
def auto_deskew_vertical_lines(image_bgr, max_angle=25.0):
    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]
    redness = roi[:, :, 2].astype(np.float32) - roi[:, :, 1].astype(np.float32)

    rh, rw = redness.shape
    scale = min(1.0, 300.0 / max(rh, rw))
    small_red = cv2.resize(redness, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small_red.shape
    center_small = (sw / 2.0, sh / 2.0)

    def get_vertical_sharpness(angle):
        M = cv2.getRotationMatrix2D(center_small, angle, 1.0)
        rot = cv2.warpAffine(small_red, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        margin_x = int(sw * 0.2)
        col_means = np.mean(rot[:, margin_x:sw - margin_x], axis=0)
        return float(np.var(col_means))

    coarse_angles = np.arange(-max_angle, max_angle + 1.0, 1.0)
    best_angle = max(coarse_angles, key=get_vertical_sharpness)

    fine_angles = np.arange(best_angle - 1.2, best_angle + 1.25, 0.1)
    best_angle = float(max(fine_angles, key=get_vertical_sharpness))

    if abs(best_angle) < 0.2:
        return image_bgr, 0.0

    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated_bgr = cv2.warpAffine(
        image_bgr,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated_bgr, round(best_angle, 2)

# 3-1. 패턴 선명도(경계 대비) 측정으로 흐린/미형성 패턴(pd) 판별 함수
def evaluate_pattern_contrast(image_bgr):
    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.05):int(w * 0.95)]
    redness = roi[:, :, 2].astype(np.float32) - roi[:, :, 1].astype(np.float32)
    col_profile = np.mean(redness, axis=0)
    contrast_score = float(np.percentile(col_profile, 85) - np.percentile(col_profile, 15))
    return round(contrast_score, 1)

# 3-2. 세로 패턴에서 '순수 밝기'만으로 Line 구간 + 평균 Line 폭을 반환하는 함수
def detect_line_intervals(image_bgr):
    h, w = image_bgr.shape[:2]
    roi = image_bgr[int(h * 0.25):int(h * 0.75), :]

    redness = roi[:, :, 2].astype(np.float32) - roi[:, :, 1].astype(np.float32)
    col_red_profile = np.mean(redness, axis=0)

    kernel = np.ones(7) / 7.0
    smooth_red = np.convolve(col_red_profile, kernel, mode='same')
    thresh = np.percentile(smooth_red, 65)

    peaks = []
    min_dist = max(15, w // 45)
    for x in range(2, w - 2):
        if smooth_red[x] > thresh and smooth_red[x] >= smooth_red[x - 1] and smooth_red[x] >= smooth_red[x + 1]:
            if not peaks or (x - peaks[-1]) > min_dist:
                peaks.append(x)
            elif smooth_red[x] > smooth_red[peaks[-1]]:
                peaks[-1] = x

    if len(peaks) < 3:
        return [], 0.0

    avg_lane_width = float(np.median(np.diff(peaks)))

    edges = [0] + peaks + [w]
    lanes = []
    brightness_profile = np.mean(roi[:, :, 1].astype(np.float32), axis=0)

    for i in range(len(edges) - 1):
        x1, x2 = edges[i], edges[i + 1]
        width = x2 - x1
        if width < 4:
            continue
        pad = min(4, width // 4)
        inner_bright = np.mean(brightness_profile[x1 + pad:x2 - pad]) if (x2 - x1) > 2 * pad else np.mean(brightness_profile[x1:x2])
        lanes.append({"x1": x1, "x2": x2, "width": width, "brightness": inner_bright})

    if len(lanes) < 4:
        return [], avg_lane_width

    even_lanes = [lanes[i]["brightness"] for i in range(2, len(lanes) - 2, 2)]
    odd_lanes = [lanes[i]["brightness"] for i in range(1, len(lanes) - 2, 2)]

    if even_lanes and odd_lanes:
        line_parity = 1 if np.mean(odd_lanes) >= np.mean(even_lanes) else 0
    else:
        line_parity = 1

    internal_line_widths = [
        lanes[i]["width"] for i in range(1, len(lanes) - 1) if i % 2 == line_parity
    ]
    avg_line_width = float(np.median(internal_line_widths)) if internal_line_widths else avg_lane_width

    line_intervals = [(lanes[i]["x1"] - 5, lanes[i]["x2"] + 5) for i in range(len(lanes)) if i % 2 == line_parity]
    return line_intervals, avg_line_width

# 4. 기울기 보정 + 단일 통합 모델 추론 + 현장 규칙 필터 + pd 판정 함수
def process_single_bgr(image_bgr, conf_threshold, iou_threshold, pd_min_contrast, show_corners, show_masks, filter_line_only, filter_uturn_open, enable_deskew):
    if enable_deskew:
        image_bgr, skew_angle = auto_deskew_vertical_lines(image_bgr)
    else:
        skew_angle = 0.0

    results = get_loaded_model().infer(image_bgr, confidence=conf_threshold, iou_threshold=iou_threshold)[0]
    detections = sv.Detections.from_inference(results)

    if len(detections) == 0:
        detections = sv.Detections.empty()
        detections.data["class_name"] = np.array([], dtype=str)

    h, w = image_bgr.shape[:2]
    contrast_score = evaluate_pattern_contrast(image_bgr)
    is_blurry_pd = (pd_min_contrast > 0) and (contrast_score < pd_min_contrast)

    if len(detections) > 0:
        line_intervals, avg_line_width = detect_line_intervals(image_bgr) if not is_blurry_pd else ([], 0.0)
        keep_indices = []
        new_class_names = list(detections.data["class_name"])

        for idx, (cname, box) in enumerate(zip(new_class_names, detections.xyxy)):
            cname_lower = str(cname).lower()

            # 0) Scratch(sc)는 크기/위치 필터 없이 그대로 유지
            if cname_lower == "sc" or "scratch" in cname_lower:
                new_class_names[idx] = "sc"
                keep_indices.append(idx)
                continue

            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            box_w = box[2] - box[0]

            is_half_line_width = (avg_line_width <= 0) or (box_w >= avg_line_width * 0.48)

            bx1, by1, bx2, by2 = map(int, box)
            box_roi = image_bgr[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
            red_fill_ratio = 0.0
            if box_roi.size > 0:
                b_redness = box_roi[:, :, 2].astype(np.float32) - box_roi[:, :, 1].astype(np.float32)
                red_fill_ratio = float(np.mean(b_redness > 105))

            is_real_short = (avg_line_width > 0) and (box_w >= avg_line_width * 1.6) and (red_fill_ratio >= 0.18)

            matched_line = None
            for (x1, x2) in line_intervals:
                if x1 <= cx <= x2:
                    matched_line = (x1, x2)
                    break

            # 1) Particle(p) 검사
            if "particle" in cname_lower or cname_lower == "p":
                if is_real_short:
                    new_class_names[idx] = "s"
                    keep_indices.append(idx)
                elif (not filter_line_only or not line_intervals or matched_line is not None) and is_half_line_width:
                    keep_indices.append(idx)

            # 2) Short(s) 검사
            elif "short" in cname_lower or cname_lower == "s":
                if is_real_short:
                    keep_indices.append(idx)
                elif matched_line is not None and is_half_line_width and box_w < avg_line_width * 1.2:
                    new_class_names[idx] = "p"
                    keep_indices.append(idx)

            # 3) Open(o) 검사: 상·하단 18% 끝단 U자 굴곡부 오검출은 p로 바꾸지 않고 깔끔하게 제외
            elif filter_uturn_open and ("open" in cname_lower or cname_lower == "o"):
                if (h * 0.18) < cy < (h * 0.82):
                    keep_indices.append(idx)

            else:
                keep_indices.append(idx)

        detections.data["class_name"] = np.array(new_class_names, dtype=str)
        detections = detections[keep_indices]
        if len(detections) > 0:
            detections = detections.with_nms(threshold=iou_threshold, class_agnostic=True)

    # 4) 패턴 대비가 기준치 미만이면 전체 이미지에 'pd' 결함 박스 추가
    if is_blurry_pd:
        already_has_pd = any("pd" in str(c).lower() or "pattern" in str(c).lower() for c in detections.data.get("class_name", []))
        if not already_has_pd:
            pad = 12
            pd_xyxy = np.array([[pad, pad, w - pad, h - pad]], dtype=np.float32)
            pd_conf = round(min(0.99, max(0.60, 1.0 - (contrast_score / max(1.0, pd_min_contrast * 1.5)))), 2)
            pd_conf_arr = np.array([pd_conf], dtype=np.float32)
            pd_cls_id = np.array([4], dtype=int)
            pd_cls_name = np.array(["pd"], dtype=str)

            if len(detections) > 0:
                existing_mask = detections.mask
                new_mask = None
                if existing_mask is not None:
                    full_mask = np.zeros((1, h, w), dtype=bool)
                    full_mask[0, pad:h-pad, pad:w-pad] = True
                    new_mask = np.vstack([full_mask, existing_mask])

                detections = sv.Detections(
                    xyxy=np.vstack([pd_xyxy, detections.xyxy]),
                    mask=new_mask,
                    confidence=np.concatenate([pd_conf_arr, detections.confidence]),
                    class_id=np.concatenate([pd_cls_id, detections.class_id]),
                    data={"class_name": np.concatenate([pd_cls_name, detections.data["class_name"]])}
                )
            else:
                detections = sv.Detections(
                    xyxy=pd_xyxy,
                    confidence=pd_conf_arr,
                    class_id=pd_cls_id,
                    data={"class_name": pd_cls_name}
                )

    scale_factor = max(w, h) / 1000.0
    dynamic_text_scale = max(0.55, round(0.65 * scale_factor, 2))
    dynamic_thickness = max(2, int(round(2.2 * scale_factor)))
    dynamic_padding = max(5, int(round(6 * scale_factor)))

    mask_annotator = sv.MaskAnnotator(color=CUSTOM_COLORS, opacity=0.35)
    polygon_annotator = sv.PolygonAnnotator(color=CUSTOM_COLORS, thickness=dynamic_thickness)
    box_annotator = sv.BoxAnnotator(color=CUSTOM_COLORS, thickness=dynamic_thickness)
    corner_annotator = sv.BoxCornerAnnotator(
        color=CUSTOM_COLORS,
        thickness=dynamic_thickness + 2,
        corner_length=max(16, int(20 * scale_factor))
    )

    annotated_bgr = image_bgr.copy()
    if show_masks and getattr(detections, "mask", None) is not None:
        annotated_bgr = mask_annotator.annotate(scene=annotated_bgr, detections=detections)
        annotated_bgr = polygon_annotator.annotate(scene=annotated_bgr, detections=detections)

    annotated_bgr = box_annotator.annotate(scene=annotated_bgr, detections=detections)
    if show_corners:
        annotated_bgr = corner_annotator.annotate(scene=annotated_bgr, detections=detections)

    palette_hex = ["#FF3B30", "#00E5FF", "#FFCC00", "#AF52DE", "#34C759"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thick = max(1, int(round(1.8 * scale_factor)))

    if len(detections) > 0:
        for i, (class_name, confidence, box, cls_id) in enumerate(
            zip(detections.data["class_name"], detections.confidence, detections.xyxy, detections.class_id)
        ):
            label_text = f"#{i+1} {class_name} {confidence*100:.0f}%"
            (tw, th), baseline = cv2.getTextSize(label_text, font, dynamic_text_scale, text_thick)
            box_w_lbl = tw + dynamic_padding * 2
            box_h_lbl = th + baseline + dynamic_padding

            x1, y1, x2, y2 = map(int, box)
            lbl_x1 = max(2, min(x1, w - box_w_lbl - 2))
            if y1 - box_h_lbl >= 4:
                lbl_y2 = y1
                lbl_y1 = y1 - box_h_lbl
            else:
                lbl_y1 = min(y2 + 2, h - box_h_lbl - 2)
                lbl_y2 = lbl_y1 + box_h_lbl

            hex_c = palette_hex[int(cls_id) % len(palette_hex)].lstrip("#")
            bgr_c = tuple(int(hex_c[j:j+2], 16) for j in (4, 2, 0))

            cv2.rectangle(annotated_bgr, (lbl_x1, lbl_y1), (lbl_x1 + box_w_lbl, lbl_y2), bgr_c, -1)
            cv2.putText(
                annotated_bgr,
                label_text,
                (lbl_x1 + dynamic_padding, lbl_y2 - baseline - dynamic_padding // 2 + 2),
                font,
                dynamic_text_scale,
                (0, 0, 0),
                text_thick,
                cv2.LINE_AA
            )

    return annotated_bgr, detections, contrast_score, skew_angle

# [탭 1] 1장 정밀 검사 함수
def detect_single(input_image, conf_threshold, iou_threshold, pd_min_contrast, show_corners, show_masks, filter_line_only, filter_uturn_open, enable_deskew):
    empty_df = pd.DataFrame(columns=["No.", "결함 유형 (Class)", "신뢰도 (Conf)", "위치 좌표 (x1, y1, x2, y2)"])
    if input_image is None:
        status_html = """
        <div style="padding:16px; border-radius:10px; background:#1e293b; border:1px solid #475569; text-align:center; color:#e2e8f0; font-weight:500;">
            이미지를 업로드하면 자동 보정 및 검사가 시작됩니다.
        </div>
        """
        return None, status_html, empty_df

    image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    annotated_bgr, detections, contrast_score, skew_angle = process_single_bgr(
        image_bgr, conf_threshold, iou_threshold, pd_min_contrast, show_corners, show_masks, filter_line_only, filter_uturn_open, enable_deskew
    )
    output_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    angle_info = f"기울기 보정: {skew_angle:+.2f}°" if enable_deskew else "기울기 보정: OFF"
    total = len(detections)
    if total == 0:
        status_html = f"""
        <div style="padding:18px; border-radius:12px; background:linear-gradient(135deg, #064e3b, #065f46); border:1px solid #10b981; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:13px; color:#a7f3d0; font-weight:700;">INSPECTION RESULT · 선명도: {contrast_score} · {angle_info}</div>
                <div style="font-size:22px; color:#ffffff; font-weight:800; margin-top:2px;">PASS — 정상 (검출된 결함 0건)</div>
            </div>
            <div style="background:#10b981; color:#022c22; padding:6px 14px; border-radius:999px; font-weight:800; font-size:14px;">OK</div>
        </div>
        """
        return output_rgb, status_html, empty_df

    class_names = detections.data["class_name"]
    counts = {name: list(class_names).count(name) for name in set(class_names)}
    badges = " ".join([
        f"<span style='background:rgba(255,255,255,0.2); padding:4px 10px; border-radius:6px; font-size:13px; color:#ffffff; margin-right:6px;'>{k}: <b>{v}건</b></span>"
        for k, v in counts.items()
    ])

    status_html = f"""
    <div style="padding:18px; border-radius:12px; background:linear-gradient(135deg, #7f1d1d, #991b1b); border:1px solid #ef4444; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-size:13px; color:#fecaca; font-weight:700;">INSPECTION RESULT · 선명도: {contrast_score} (기준: {pd_min_contrast}) · {angle_info}</div>
            <div style="font-size:22px; color:#ffffff; font-weight:800; margin-top:2px;">FAIL — 총 {total}건 결함 검출</div>
            <div style="margin-top:8px; color:#ffffff;">{badges}</div>
        </div>
        <div style="background:#ef4444; color:#ffffff; padding:6px 14px; border-radius:999px; font-weight:800; font-size:14px;">DEFECT</div>
    </div>
    """

    rows = []
    for i, (cname, conf, box) in enumerate(zip(class_names, detections.confidence, detections.xyxy)):
        x1, y1, x2, y2 = map(int, box)
        rows.append([i + 1, cname, f"{conf*100:.1f}%", f"({x1}, {y1}) ~ ({x2}, {y2})"])

    df = pd.DataFrame(rows, columns=["No.", "결함 유형 (Class)", "신뢰도 (Conf)", "위치 좌표 (x1, y1, x2, y2)"])
    return output_rgb, status_html, df

# [탭 2] 다중 이미지 일괄 검사 함수
def detect_batch(files, conf_threshold, iou_threshold, pd_min_contrast, show_corners, show_masks, filter_line_only, filter_uturn_open, enable_deskew, progress=gr.Progress()):
    empty_df = pd.DataFrame(columns=["파일명", "판정", "보정 각도(°)", "패턴 선명도", "총 결함 수", "결함 상세 내역"])
    if not files:
        return None, empty_df, None, "<div style='color:#e2e8f0; padding:14px; text-align:center;'>검사할 여러 장의 사진을 한꺼번에 올려주세요.</div>"

    temp_dir = tempfile.mkdtemp()
    out_img_dir = os.path.join(temp_dir, "marked_images")
    os.makedirs(out_img_dir, exist_ok=True)

    gallery_items = []
    summary_rows = []
    pass_count = 0
    fail_count = 0

    total_files = len(files)
    for idx, file_path in enumerate(files):
        progress((idx + 1) / total_files, desc=f"검사 진행 중... ({idx + 1}/{total_files}장)")
        filename = os.path.basename(file_path)

        img_array = np.fromfile(file_path, np.uint8)
        image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue

        annotated_bgr, detections, contrast_score, skew_angle = process_single_bgr(
            image_bgr, conf_threshold, iou_threshold, pd_min_contrast, show_corners, show_masks, filter_line_only, filter_uturn_open, enable_deskew
        )
        total_defects = len(detections)

        save_path = os.path.join(out_img_dir, f"result_{filename}")
        ext = os.path.splitext(filename)[1] or ".jpg"
        _, encoded_img = cv2.imencode(ext, annotated_bgr)
        encoded_img.tofile(save_path)

        output_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        status_label = "PASS" if total_defects == 0 else f"FAIL ({total_defects}건)"
        gallery_items.append((output_rgb, f"{filename} [{status_label}]"))

        if total_defects == 0:
            pass_count += 1
            detail_str = "-"
        else:
            fail_count += 1
            cnames = detections.data["class_name"]
            counts = {name: list(cnames).count(name) for name in set(cnames)}
            detail_str = ", ".join([f"{k}({v})" for k, v in counts.items()])

        summary_rows.append([filename, "PASS" if total_defects == 0 else "FAIL", skew_angle, contrast_score, total_defects, detail_str])

    df = pd.DataFrame(summary_rows, columns=["파일명", "판정", "보정 각도(°)", "패턴 선명도", "총 결함 수", "결함 상세 내역"])

    csv_path = os.path.join(temp_dir, "defect_inspection_report.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    zip_path = os.path.join(temp_dir, "batch_inspection_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="defect_inspection_report.csv")
        for root_dir, _, fnames in os.walk(out_img_dir):
            for f in fnames:
                full_p = os.path.join(root_dir, f)
                zf.write(full_p, arcname=os.path.join("marked_images", f))

    batch_banner = f"""
    <div style="padding:18px; border-radius:12px; background:#1e293b; border:1px solid #38bdf8; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-size:13px; color:#38bdf8; font-weight:700;">BATCH INSPECTION COMPLETE</div>
            <div style="font-size:20px; color:#ffffff; font-weight:800; margin-top:4px;">총 {len(summary_rows)}장 일괄 검사 완료</div>
        </div>
        <div style="display:flex; gap:12px;">
            <div style="background:#065f46; border:1px solid #10b981; padding:8px 16px; border-radius:8px; text-align:center;">
                <div style="font-size:11px; color:#a7f3d0;">PASS (정상)</div>
                <div style="font-size:18px; font-weight:800; color:#ffffff;">{pass_count}장</div>
            </div>
            <div style="background:#7f1d1d; border:1px solid #ef4444; padding:8px 16px; border-radius:8px; text-align:center;">
                <div style="font-size:11px; color:#fecaca;">FAIL (불량)</div>
                <div style="font-size:18px; font-weight:800; color:#ffffff;">{fail_count}장</div>
            </div>
        </div>
    </div>
    """
    return gallery_items, df, zip_path, batch_banner

# 5. 고대비 다크 테마 CSS
custom_css = """
:root, .dark {
    --body-background-fill: #0b1120 !important;
    --body-text-color: #f8fafc !important;
    --block-background-fill: #111827 !important;
    --block-border-color: #334155 !important;
    --block-label-text-color: #e2e8f0 !important;
    --block-title-text-color: #f8fafc !important;
    --input-background-fill: #1e293b !important;
    --input-text-color: #ffffff !important;
    --table-even-background-fill: #1e293b !important;
    --table-odd-background-fill: #0f172a !important;
    --table-text-color: #f8fafc !important;
}
body, .gradio-container { background-color: #0b1120 !important; color: #f8fafc !important; }
.header-box {
    background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-left: 5px solid #38bdf8;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.panel-card { background: #111827 !important; border: 1px solid #334155 !important; border-radius: 12px !important; padding: 14px !important; }
.img-title-bar {
    background: #0284c7;
    color: #ffffff !important;
    font-size: 13px;
    font-weight: 700;
    padding: 7px 12px;
    border-radius: 8px 8px 0 0;
    margin-bottom: -4px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.clean-image-box {
    border-radius: 0 0 8px 8px !important;
    overflow: hidden !important;
}
.clean-image-box [data-testid="block-label"] {
    display: none !important;
}
button[role="tab"] { color: #cbd5e1 !important; font-weight: 600 !important; font-size: 15px !important; }
button[role="tab"][aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }
label, span, p, th, td, .progress-text, .eta-bar { color: #f1f5f9 !important; }
table tr { background-color: #1e293b !important; border-bottom: 1px solid #334155 !important; }
table tr:nth-child(even) { background-color: #0f172a !important; }
table td, table th { color: #f8fafc !important; }
.progress-bar-wrap, .wrap { background-color: rgba(15, 23, 42, 0.88) !important; }
footer { display: none !important; }
"""

force_dark_js = "() => { document.body.classList.add('dark'); }"

theme = gr.themes.Soft(primary_hue="sky", neutral_hue="slate").set(
    body_background_fill="#0b1120",
    body_text_color="#f8fafc",
    block_background_fill="#111827",
    block_border_color="#334155",
    block_label_text_color="#e2e8f0",
    block_title_text_color="#f8fafc",
    input_background_fill="#1e293b",
    table_even_background_fill="#1e293b",
    table_odd_background_fill="#0f172a",
)

with gr.Blocks(title="AI Optical Defect Inspection") as demo:
    gr.HTML(f"""
    <div class="header-box">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="background:#0284c7; color:#ffffff; font-size:11px; font-weight:700; padding:3px 8px; border-radius:4px; letter-spacing:0.5px;">UNIFIED SINGLE MODEL AOI</span>
                <h1 style="margin:6px 0 4px 0; font-size:24px; font-weight:800; color:#ffffff;">🔬 반도체·패턴 광학 결함 자동 검출 대시보드</h1>
                <p style="margin:0; font-size:13px; color:#cbd5e1;">단일 통합 모델(228장 · 5종 결함: p, s, pd, o, sc) · 폴리곤 분할 시각화 · 현장 규칙 필터</p>
            </div>
            <div style="text-align:right; color:#cbd5e1; font-size:13px;">
                <div>Model: <b style="color:#38bdf8;">{MODEL_ID} (Unified)</b></div>
                <div>Engine: <b style="color:#34d399;">RF-DETR-Seg + Auto-Align</b></div>
            </div>
        </div>
    </div>
    """)

    with gr.Tabs():
        # [탭 1] 1장 정밀 검사
        with gr.TabItem("🔍 1장 정밀 검사"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes="panel-card"):
                    gr.HTML('<div class="img-title-bar">📥 원본 이미지 입력 (드래그 & 드롭)</div>')
                    input_img = gr.Image(type="numpy", show_label=False, elem_classes="clean-image-box", height=420)
                    conf_slider = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="🎛️ 검출 민감도 (Confidence - 낮출수록 많이 잡음)")
                    iou_slider = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="🧩 중복 박스 제거 강도 (IoU)")
                    pd_slider = gr.Slider(0, 120, value=45, step=5, label="🌫️ 흐린/미형성 패턴(pd) 판정 기준 (선명도 점수가 이 값 미만이면 pd 검출)")
                    deskew_chk = gr.Checkbox(value=True, label="📐 자동 수직 정렬 (기울어진 패턴 사진을 반듯하게 자동 회전 보정)")
                    line_filter_chk = gr.Checkbox(value=True, label="🔬 Particle 필터: Line 위 & 가로 폭 절반 이상 결함만 검출")
                    uturn_filter_chk = gr.Checkbox(value=True, label="🛡️ U자 끝단 보호: 깨끗한 굴곡부 오검출 제외")
                    mask_chk = gr.Checkbox(value=True, label="🎨 폴리곤 마스크 표시 (Instance Segmentation 영역 강조)")
                    corner_chk = gr.Checkbox(value=True, label="타겟 모서리 강조")
                    run_btn = gr.Button("🚀 결함 정밀 스캔 실행", variant="primary", size="lg")

                with gr.Column(scale=6, elem_classes="panel-card"):
                    gr.HTML('<div class="img-title-bar">🎯 기울기 보정 및 결함 마킹 결과 (우측 상단 확대 아이콘으로 전체 화면 보기)</div>')
                    output_img = gr.Image(type="numpy", show_label=False, elem_classes="clean-image-box", height=420)
                    status_banner = gr.HTML("""
                    <div style="padding:16px; border-radius:10px; background:#1e293b; border:1px solid #475569; text-align:center; color:#e2e8f0; font-weight:500;">
                        이미지를 업로드하면 자동 보정 및 검사가 시작됩니다.
                    </div>
                    """)
                    result_table = gr.Dataframe(
                        headers=["No.", "결함 유형 (Class)", "신뢰도 (Conf)", "위치 좌표 (x1, y1, x2, y2)"],
                        datatype=["number", "str", "str", "str"],
                        label="📋 검출 결함 상세 리포트 (번호 #1, #2... 와 대조 확인)",
                        interactive=False
                    )

            inputs_1 = [input_img, conf_slider, iou_slider, pd_slider, corner_chk, mask_chk, line_filter_chk, uturn_filter_chk, deskew_chk]
            outputs_1 = [output_img, status_banner, result_table]
            run_btn.click(fn=detect_single, inputs=inputs_1, outputs=outputs_1)
            input_img.change(fn=detect_single, inputs=inputs_1, outputs=outputs_1)
            conf_slider.release(fn=detect_single, inputs=inputs_1, outputs=outputs_1)
            iou_slider.release(fn=detect_single, inputs=inputs_1, outputs=outputs_1)
            pd_slider.release(fn=detect_single, inputs=inputs_1, outputs=outputs_1)
            corner_chk.change(fn=detect_single, inputs=inputs_1, outputs=outputs_1)
            mask_chk.change(fn=detect_single, inputs=inputs_1, outputs=outputs_1)
            line_filter_chk.change(fn=detect_single, inputs=inputs_1, outputs=outputs_1)
            uturn_filter_chk.change(fn=detect_single, inputs=inputs_1, outputs=outputs_1)
            deskew_chk.change(fn=detect_single, inputs=inputs_1, outputs=outputs_1)

        # [탭 2] 다중 이미지 일괄 검사
        with gr.TabItem("📦 다중 이미지 일괄 검사 (일괄 처리)"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, elem_classes="panel-card"):
                    batch_files = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="📂 여러 장의 사진 한꺼번에 드래그 & 드롭 (Ctrl+A로 전체 선택 가능)"
                    )
                    batch_conf = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="🎛️ 검출 민감도 (Confidence)")
                    batch_iou = gr.Slider(0.1, 0.9, value=0.45, step=0.05, label="🧩 중복 박스 제거 강도 (IoU)")
                    batch_pd = gr.Slider(0, 120, value=45, step=5, label="🌫️ 흐린/미형성 패턴(pd) 판정 기준 (선명도 미만 시 pd)")
                    batch_deskew = gr.Checkbox(value=True, label="📐 자동 수직 정렬 (기울어진 패턴 사진을 반듯하게 자동 회전 보정)")
                    batch_line_filter = gr.Checkbox(value=True, label="🔬 Particle 필터: Line 위 & 가로 폭 절반 이상 결함만 검출")
                    batch_uturn_filter = gr.Checkbox(value=True, label="🛡️ U자 끝단 보호: 깨끗한 굴곡부 오검출 제외")
                    batch_mask = gr.Checkbox(value=True, label="🎨 폴리곤 마스크 표시 (Instance Segmentation 영역 강조)")
                    batch_corner = gr.Checkbox(value=True, label="타겟 모서리 강조")
                    batch_btn = gr.Button("⚡ 전체 이미지 일괄 스캔 시작", variant="primary", size="lg")
                    zip_download = gr.File(label="💾 전체 결과 다운로드 (보정·마킹 이미지 + 엑셀 CSV 리포트 ZIP)")

                with gr.Column(scale=7, elem_classes="panel-card"):
                    batch_status = gr.HTML("""
                    <div style="padding:16px; border-radius:10px; background:#1e293b; border:1px solid #475569; text-align:center; color:#e2e8f0; font-weight:500;">
                        왼쪽 상자에 여러 장의 사진을 한꺼번에 드래그한 뒤 [전체 이미지 일괄 스캔 시작]을 누르세요.
                    </div>
                    """)
                    batch_gallery = gr.Gallery(
                        label="🖼️ 일괄 검사 결과 미리보기 (클릭하면 크게 확대됩니다)",
                        columns=4,
                        height=380,
                        object_fit="contain"
                    )
                    batch_table = gr.Dataframe(
                        headers=["파일명", "판정", "보정 각도(°)", "패턴 선명도", "총 결함 수", "결함 상세 내역"],
                        datatype=["str", "str", "number", "number", "number", "str"],
                        label="📊 파일별 전체 요약 리포트",
                        interactive=False
                    )

            batch_btn.click(
                fn=detect_batch,
                inputs=[batch_files, batch_conf, batch_iou, batch_pd, batch_corner, batch_mask, batch_line_filter, batch_uturn_filter, batch_deskew],
                outputs=[batch_gallery, batch_table, zip_download, batch_status]
            )

if __name__ == "__main__":
    # Local only: no public share link, no browser pop-up. Port/host configurable.
    demo.launch(server_name=os.environ.get("ROBOFLOW_APP_HOST", "127.0.0.1"),
                server_port=int(os.environ.get("ROBOFLOW_APP_PORT", "8080")),
                inbrowser=False, share=False, theme=theme, css=custom_css, js=force_dark_js)