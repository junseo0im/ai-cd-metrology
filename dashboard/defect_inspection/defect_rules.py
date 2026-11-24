"""결함 검출 후처리 규칙 (r26) — 모델 추론 없이 후처리만 담당.

defect_inspection_app.py(r26, Gradio 앱)의 후처리 코드를 그대로 옮긴 모듈입니다.
Gradio·모델 로드 없이 import 할 수 있어서 대시보드(aoi_pipeline.py)에서 사용합니다.

  deskew_image(image_bgr)            -> (보정·크롭된 이미지, 회전 각도)
  run_rules(image_bgr, raw_dets, …)  -> (마킹 이미지, 후처리된 detections, info)
"""
import cv2
import numpy as np
import supervision as sv

RULES_VERSION = "v2026.11.24-r33"

# 클래스 정의: (코드, 표시 이름, 색상)
#   패턴 자체가 노랑·연두·주황·빨강(테두리)이라, 결함 색은 그와 겹치지 않는 색만 사용.
CLASS_INFO = [
    ("o", "Open", "#1d4ed8"),            # 진한 파랑
    ("p", "Particle", "#d946ef"),        # 자홍(마젠타)
    ("pd", "Pattern defect", "#6d28d9"), # 보라
    ("s", "Short", "#06b6d4"),           # 하늘(시안)
    ("sc", "Scratch", "#111827"),        # 검정
]
CLASS_IDS = {code: i for i, (code, _, _) in enumerate(CLASS_INFO)}
UNKNOWN_HEX = "#475569"  # 정의되지 않은 클래스용 회색

CLASS_PALETTE = sv.ColorPalette.from_hex(
    [hex_ for _, _, hex_ in CLASS_INFO] + [UNKNOWN_HEX]
)
UNKNOWN_ID = len(CLASS_INFO)


def _hex_to_bgr(hex_str):
  hex_str = hex_str.lstrip("#")
  r, g, b = (int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
  return (b, g, r)


def _text_bgr_on(bg_bgr):
  """라벨 배경색이 밝으면 검은 글씨, 어두우면 흰 글씨."""
  b, g, r = bg_bgr
  return (0, 0, 0) if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else (255, 255, 255)


def _class_bgr(class_id):
  if 0 <= class_id < len(CLASS_INFO):
    return _hex_to_bgr(CLASS_INFO[class_id][2])
  return _hex_to_bgr(UNKNOWN_HEX)


def _normalize_class_name(name):
  name = str(name).lower()
  if name == "sc" or "scratch" in name:
    return "sc"
  if name == "pd" or "pattern" in name:
    return "pd"
  if name == "p" or "particle" in name:
    return "p"
  if name == "s" or "short" in name:
    return "s"
  if name == "o" or "open" in name:
    return "o"
  return name


def _empty_detections():
  detections = sv.Detections.empty()
  detections.data["class_name"] = np.array([], dtype=str)
  detections.data["position"] = np.array([], dtype=str)
  detections.data["source"] = np.array([], dtype=str)
  return detections


# 2. 모델 추론 + 규칙 보정 + 시각화
#    (박스를 새로 만들지는 않고, 모델 결과를 line/gap·스크래치 규칙으로 걸러내거나 S로 전환만 함)
NMS_PER_CLASS = "같은 클래스끼리만 정리 (P·S 동시 표시)"
NMS_AGNOSTIC = "클래스 무관, 1개만 남김"
NMS_OFF = "정리 안 함 (모델 원본)"
NMS_CHOICES = [NMS_PER_CLASS, NMS_AGNOSTIC, NMS_OFF]


def _overlaps(a, b):
  return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


# ---------------------------------------------------------------------------
# line / gap 판별
#   패턴은 line, gap이 번갈아 나오는 줄무늬이고, 경계(붉은 테두리)가 가장 어둡다.
#   어느 쪽이 먼저 나오는지는 사진마다 다르지만 line이 gap보다 항상 살짝 밝다(하얗다).
#   ※ 흑백 밝기는 사진에 따라 차이가 거의 없거나 반대로 나오기도 해서(노랑 vs 연두),
#     '하얀 정도' = B·G·R 중 가장 작은 값으로 비교한다. 지금까지 받은 사진 4종에서
#     line이 gap보다 이 값이 항상 25~30 높았다. 경계(붉은 테두리) 찾기는 흑백 밝기 사용.
#   1) 줄무늬 방향(세로/가로)을 자동 판단
#   2) 결함 박스를 뺀 나머지 영역에서 줄무늬를 가로지르는 밝기 프로파일(중앙값) 계산
#   3) 어두운 골짜기 = 경계 → 경계 사이를 하나의 줄무늬로 분할
#   4) 각 줄무늬 안쪽의 '하얀 정도'를 양옆 줄무늬와 비교해 더 하얗면 line, 아니면 gap
# ---------------------------------------------------------------------------
LINE, GAP, EDGE = 1, 0, -1
LEVEL_MIN_SPREAD = 8.0  # 줄무늬 흰 정도 차이가 이보다 작으면 초록 채널로 line/gap 판정
WIDE_VALLEY_RATIO = 0.22  # 경계 골짜기 폭 / 줄무늬 간격 이 이보다 크면 '어둡게 망가진 line' 


def _smooth_1d(arr, sigma=1.2):
  radius = max(1, int(3 * sigma))
  xs = np.arange(-radius, radius + 1)
  kernel = np.exp(-(xs**2) / (2 * sigma**2))
  kernel /= kernel.sum()
  padded = np.pad(arr, radius, mode="edge")
  return np.convolve(padded, kernel, mode="valid")


def _edge_spacing(prof):
  """밝기 프로파일의 자기상관으로 경계 간격(줄무늬 하나의 폭)을 추정."""
  x = prof - prof.mean()
  ac = np.correlate(x, x, "full")[len(x) - 1 :]
  if ac[0] <= 0:
    return None
  ac = ac / ac[0]
  for lag in range(4, len(x) // 2):
    if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.1:
      return lag
  return None


def analyze_line_gap(image_bgr, exclude_boxes=None):
  """줄무늬를 line/gap으로 분류. 판별이 어려우면 None 반환."""
  gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
  whiteness = image_bgr.min(axis=2).astype(np.float32)  # line/gap 구분용
  grad_x = np.abs(np.diff(gray, axis=1)).mean()
  grad_y = np.abs(np.diff(gray, axis=0)).mean()
  vertical = grad_x >= grad_y  # 세로 줄무늬면 x 방향 변화가 큼
  g = gray if vertical else gray.T
  w_img = whiteness if vertical else whiteness.T
  n_rows, n_cols = g.shape

  # 결함이 있는 행은 빼고 프로파일 계산 (결함이 밝기를 왜곡하지 않도록)
  keep = np.ones(n_rows, dtype=bool)
  if exclude_boxes is not None:
    for x1, y1, x2, y2 in exclude_boxes:
      r1, r2 = (y1, y2) if vertical else (x1, x2)
      keep[max(0, int(r1)) : max(0, int(np.ceil(r2)))] = False
  if keep.sum() < 0.2 * n_rows:
    keep[:] = True
  prof = _smooth_1d(np.median(g[keep], axis=0))  # 경계 찾기용
  prof_w = _smooth_1d(np.median(w_img[keep], axis=0))  # line/gap 판정용
  # 노란 line 사진은 파랑 채널이 line·gap 모두 0 근처라 흰 정도로 구분이 안 됨 → 초록 채널로 대신 판정
  g_ch = image_bgr[:, :, 1].astype(np.float32)
  g_ch = g_ch if vertical else g_ch.T
  prof_gc = _smooth_1d(np.median(g_ch[keep], axis=0))

  # 경계(어두운 골짜기) 찾기
  #  - 경계는 '바로 주변보다 뾰족하게 어둡고' + '줄무늬(line·gap) 자체보다 확실히 어두운' 곳.
  #  - 비교 범위는 줄무늬 폭에 맞춰 자동 조절(해상도와 무관).
  #  ※ 예전에는 ±30px 범위와 비교해서, gap이 line보다 많이 어두운 사진에서는
  #    gap 한가운데가 '양옆 line보다 어두운 골짜기'로 잘못 잡혔다.
  spacing = _edge_spacing(prof) or 40
  r = max(3, spacing // 8)          # 극소점 판정 범위
  win = max(2 * r, spacing // 4)    # 주변 밝기 비교 범위 (경계 폭 정도)
  spread = np.percentile(prof, 95) - np.percentile(prof, 5)
  min_prom = max(5.0, 0.1 * spread)
  p_mid, p_low = np.percentile(prof, 50), np.percentile(prof, 2)
  dark_limit = p_mid - 0.3 * (p_mid - p_low)  # 줄무늬보다 확실히 어두워야 경계
  minima = []
  for x in range(n_cols):
    lo, hi = max(0, x - r), min(n_cols, x + r + 1)
    if prof[x] > prof[lo:hi].min() or prof[x] > dark_limit:
      continue
    wl, wh = max(0, x - win), min(n_cols, x + win + 1)
    prom = min(prof[wl : x + 1].max(), prof[x:wh].max()) - prof[x]
    if prom >= min_prom and (not minima or x - minima[-1] > r):
      minima.append(x)
  if len(minima) < 3:
    return None

  def build(use_wide):
    # 넓은 어두운 골짜기 = 경계(붉은 테두리)가 아니라 PD 등으로 어둡게 망가진 line.
    #   정상 테두리 폭은 줄무늬 간격의 0.12~0.17, PD로 어두워진 line은 0.25~0.36 이었음.
    #   간격의 WIDE_VALLEY_RATIO 를 넘으면 골짜기 양끝을 경계로 하고 그 사이를 LINE으로 고정.
    spacing_med = float(np.median(np.diff(minima)))
    level_ref = float(np.percentile(prof, 60))
    bounds, forced_line = [0], set()
    for x in minima:
      mid = (prof[x] + level_ref) / 2.0
      l, rr = x, x
      while l > 0 and prof[l] < mid:
        l -= 1
      while rr < n_cols - 1 and prof[rr] < mid:
        rr += 1
      wide = use_wide and rr - l > WIDE_VALLEY_RATIO * spacing_med and l > bounds[-1]
      if wide:
        # 망가진 line은 위아래 대부분(50% 이상)에서 어둡다. 일부만 어두우면(큰 결함 덩어리 등) 제외
        band_rows = g[keep][:, l : rr + 1].mean(axis=1)
        wide = float((band_rows < mid).mean()) >= 0.5
      if wide:
        bounds += [l, rr]
        forced_line.add((l, rr))
      elif x > bounds[-1]:
        bounds.append(x)
    bounds.append(n_cols - 1)

    # 경계 사이 = 줄무늬. 이미지 양끝의 잘린 줄무늬도 충분히 넓으면 포함
    widths = np.diff(minima)
    med_w = float(np.median(widths))
    stripes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
      if (a, b) in forced_line:
        stripes.append([a, b, None, None])
        continue
      if b - a < max(3, 0.4 * med_w):
        continue
      # 경계 쪽만 잘라내고 안쪽 밝기를 사용 (이미지 끝에서 잘린 줄무늬는 바깥쪽을 자르지 않음)
      cut = max(2, int(0.25 * med_w))
      lo = a + (cut if a > 0 else 0)
      hi = b - (cut if b < n_cols - 1 else 0)
      if hi <= lo:
        continue
      level = float(np.median(prof_w[lo : hi + 1]))
      stripes.append([a, b, level, float(np.median(prof_gc[lo : hi + 1]))])
    if len(stripes) < 3:
      return None
    # 흰 정도 차이가 거의 없으면(노란 line) 초록 채널 밝기로 line/gap 비교
    lv_w = [st[2] for st in stripes if st[2] is not None]
    if lv_w and max(lv_w) - min(lv_w) < LEVEL_MIN_SPREAD:
      for st in stripes:
        if st[2] is not None:
          st[2] = st[3]
    stripes = [st[:3] for st in stripes]

    # 1) 양옆과 비교해 더 밝으면 line (어둡게 망가진 line은 LINE으로 고정)
    forced_idx = [i for i, st in enumerate(stripes) if st[2] is None]
    valid_lv = [st[2] for st in stripes if st[2] is not None]
    fill = float(np.median(valid_lv)) if valid_lv else 0.0
    levels = np.array([fill if st[2] is None else st[2] for st in stripes])
    local = []
    for i in range(len(stripes)):
      if i in forced_idx:
        local.append(LINE)
        continue
      neigh = [levels[j] for j in (i - 1, i + 1)
               if 0 <= j < len(stripes) and j not in forced_idx]
      if not neigh:  # 양옆이 모두 망가진 line이면 이 줄무늬는 gap
        local.append(GAP)
        continue
      local.append(LINE if levels[i] > np.mean(neigh) else GAP)
    for i in forced_idx:  # 망가진 line 바로 옆은 gap
      for j in (i - 1, i + 1):
        if 0 <= j < len(stripes) and j not in forced_idx:
          local[j] = GAP

    # 2) line/gap은 반드시 번갈아 나오므로, 짝수/홀수 번째 중 어느 쪽이 line인지
    #    전체 투표로 정하고, 대부분(70% 이상) 일치하면 엄격히 번갈아 배정한다.
    #    (결함 등으로 한두 줄 밝기가 틀어져도 바로잡힘)
    even_line = np.mean([
        (k == LINE) == (i % 2 == 0) for i, k in enumerate(local)
    ])
    parity = [
        LINE if ((i % 2 == 0) == (even_line >= 0.5)) else GAP
        for i in range(len(stripes))
    ]
    agree = max(even_line, 1 - even_line)
    kinds = parity if agree >= 0.7 else local
    for i in forced_idx:
      kinds[i] = LINE

    non_forced = [i for i in range(len(stripes)) if i not in forced_idx]
    score = (float(np.mean([(local[i] == LINE) == (i % 2 == 0) for i in non_forced]))
             if non_forced else 0.0)
    return stripes, kinds, max(score, 1 - score), len(forced_idx), med_w

  # 두 가지로 해석해 보고, line/gap이 더 깔끔하게 번갈아 나오는 쪽을 택한다.
  #  A: 어두운 골짜기는 모두 경계(붉은 테두리)  ← 보통 사진
  #  B: 넓고 위아래로 계속 어두운 골짜기는 PD 등으로 망가진 line  ← PD 사진
  #  B는 A가 잘 안 맞을 때(번갈아 일치율 < 0.85)만, 그리고 A보다 확실히 나을 때만 채택.
  res_a = build(False)
  res_b = build(True)
  if res_a is None and res_b is None:
    return None
  use = res_a
  if res_b is not None and res_b[3] > 0 and (
      res_a is None or (res_a[2] < 0.85 and res_b[2] >= res_a[2] + 0.1)):
    use = res_b
  stripes, kinds, _, _, med_w = use

  # 열(column)마다 라벨 지정. 경계 근처는 EDGE로 두어 판정에서 제외
  edge_hw = max(2, int(round(0.12 * med_w)))
  col_label = np.full(n_cols, EDGE, dtype=int)
  col_stripe = np.full(n_cols, -1, dtype=int)
  for idx, ((a, b, _), kind) in enumerate(zip(stripes, kinds)):
    lo = a + (edge_hw if a > 0 else 0)
    hi = b - (edge_hw if b < n_cols - 1 else 0)
    if hi > lo:
      col_label[lo : hi + 1] = kind
      col_stripe[lo : hi + 1] = idx

  return {
      "vertical": vertical,
      "stripes": [(a, b, k) for (a, b, _), k in zip(stripes, kinds)],
      "col_label": col_label,
      "col_stripe": col_stripe,
      "n_cross": n_cols,
  }


def _cross_coords(det_i, detections, vertical):
  """결함이 차지하는 픽셀의 '줄무늬를 가로지르는 방향' 좌표 (마스크 우선)."""
  if detections.mask is not None and detections.mask[det_i].any():
    ys, xs = np.nonzero(detections.mask[det_i])
    return xs if vertical else ys
  x1, y1, x2, y2 = detections.xyxy[det_i]
  a, b = (x1, x2) if vertical else (y1, y2)
  return np.arange(int(a), int(np.ceil(b)) + 1)


# line 위에서만 나올 수 있는 결함
#   P(particle): line 위 이물만 결함으로 인정
#   O(open)    : line이 끊어진 것이므로 gap에서는 절대 나올 수 없음
#   PD(pattern defect): line 패턴 자체의 결함이므로 gap에서는 나올 수 없음
LINE_ONLY_CLASSES = ("p", "o", "pd")

# P(이물) 인정 기준
#  - 위치: 이물 면적의 P_MIN_ON_LINE(1/3) 이상이 line(테두리 포함) 위에 걸쳐 있어야 함
#          (gap 쪽에 있으면서 line 테두리에 살짝 닿기만 한 것은 제외)
#  - 크기: 이물의 가장 긴 쪽 길이 ≥ line 폭 × P_MIN_SIZE_RATIO
#  - 색  : line보다 확실히 진해야 함. (line 밝기 − 이물에서 진한 30% 픽셀의 밝기) ≥ P_MIN_DARKNESS
#          받은 사진에서 실제 이물은 61~81, 연한 얼룩은 0~39 였음
P_MIN_SIZE_RATIO = 0.5
P_MIN_DARKNESS = 50
P_MIN_ON_LINE = 1.0 / 3.0  # 이물 면적 중 line(테두리 포함) 위에 있어야 하는 최소 비율

# S(short) 인정 기준: 아래 둘 중 하나여야 함 (연한 얼룩은 S 아님)
#   (a) 진한 물질이 gap을 막음: (line 밝기 − S 영역에서 가장 진한 10% 밝기) ≥ S_MIN_DARKNESS
#       받은 사진에서 진한 S는 101~143, S가 아닌 연한 얼룩은 50~74 였음
#   (b) line과 같은 물질이 gap을 가로질러 채움(패턴이 붙어 버린 short):
#       gap 안쪽 폭의 S_BRIDGE_FRAC 이상이 line 색인 행이 연속으로 있음
S_MIN_DARKNESS = 90
S_BRIDGE_FRAC = 0.8

# SC(스크래치)는 패턴 방향과 나란한 세로(가로) 직선으로 나오지 않는다.
#   줄무늬 방향과 SC_ALIGN_DEG 이내로 나란하고, 한 line 폭의 SC_MAX_WIDTH_RATIO 배 안에 들어가는
#   가늘고 긴 SC는 스크래치가 아니라 망가진 line(PD)으로 본다.
SC_ALIGN_DEG = 8.0
SC_MAX_WIDTH_RATIO = 1.3
RULE_LOG = []  # 마지막 검사에서 결함별 판정 근거 (화면·터미널 표시용)

# 신뢰도 보호: 모델이 이 값 이상으로 확신한 검출은 line/gap 규칙이 지우거나 바꾸지 않는다.
#   O·PD는 모델 단독 성능이 이미 높고(test AP50 O 0.88, PD 0.998), 규칙이 고쳐야 했던 사례는
#   모두 신뢰도 50% 미만이었다. 규칙 때문에 사라진 정답은 대부분 63~87% 였다.
#   None 이면 그 클래스는 보호 없이 규칙을 그대로 적용.
PROTECT_CONF = {"o": 0.5, "pd": 0.5, "p": None, "s": 0.5, "sc": None}


def _along_coords(det_i, detections, vertical):
  """결함이 차지하는 픽셀의 '줄무늬를 따라가는 방향' 좌표 (마스크 우선)."""
  if detections.mask is not None and detections.mask[det_i].any():
    ys, xs = np.nonzero(detections.mask[det_i])
    return ys if vertical else xs
  x1, y1, x2, y2 = detections.xyxy[det_i]
  a, b = (y1, y2) if vertical else (x1, x2)
  return np.arange(int(a), int(np.ceil(b)) + 1)


def is_line_break(image_bgr, lg, stripe_idx, along):
  """O가 진짜 '끊어진 line'인지 확인.
  O 범위 안에서 line이 사라진 구간(끊김)을 찾고, 그 끊김의 양쪽에서 line이 다시 이어지면 open.
  - 이어지는 line은 짧아도 된다(줄무늬 폭의 0.3 이상): 사진 끝이나 finger 끝 근처 open도 인정.
  - 한쪽에 line이 전혀 없으면(사진 밖까지 gap뿐) line이 원래 끝나는 자리(U자 끝) → open 아님.
  line 여부는 line 기준색에 '충분히 가까운'지로 엄격하게 판단한다
  (U자 끝 너머의 칙칙한 영역을 line으로 오인하지 않도록)."""
  stripes = lg["stripes"]
  img = image_bgr if lg["vertical"] else image_bgr.transpose(1, 0, 2)
  n_along = img.shape[0]
  a, b, _ = stripes[stripe_idx]
  width = b - a
  cut = max(2, int(0.25 * width))
  if b - a - 2 * cut < 2:
    return True  # 판단 불가 → 그대로 둠

  def band_colors(sa, sb):
    c = max(2, int(0.25 * (sb - sa)))
    return np.median(img[:, sa + c : sb - c + 1].astype(np.float32), axis=1)

  row_col = band_colors(a, b)  # 줄무늬를 따라 각 위치의 색
  gaps = [
      j for j in (stripe_idx - 1, stripe_idx + 1)
      if 0 <= j < len(stripes) and stripes[j][2] == GAP
      and stripes[j][1] - stripes[j][0] > 8
  ]
  if not gaps:
    return True
  line_ref = np.median(row_col, axis=0)
  gap_ref = np.median(
      np.concatenate([band_colors(stripes[j][0], stripes[j][1]) for j in gaps]),
      axis=0,
  )
  d_lg = float(np.linalg.norm(line_ref - gap_ref))
  d_line = np.linalg.norm(row_col - line_ref, axis=1)
  d_gap = np.linalg.norm(row_col - gap_ref, axis=1)
  is_line = (d_line < d_gap) & (d_line < 0.35 * d_lg)

  need = max(10, int(0.3 * width))    # 다시 이어지는 line의 최소 길이
  search = max(10, int(0.5 * width))  # 끊김 끝에서 이 거리 안에서 line이 시작돼야 함
  pad = int(0.25 * width)
  o1, o2 = int(along.min()), int(along.max())
  lo, hi = max(0, o1 - pad), min(n_along - 1, o2 + pad)

  # O 범위 안에서 가장 긴 '비 line' 구간 = 끊김
  best, cur_s, br = 0, None, None
  for t in range(lo, hi + 1):
    if not is_line[t]:
      if cur_s is None:
        cur_s = t
      if t - cur_s + 1 > best:
        best, br = t - cur_s + 1, (cur_s, t)
    else:
      cur_s = None
  if br is None:
    return True  # 끊긴 곳이 안 보이면 지우지 않음

  def resumes(seq):
    """끊김 끝에서 바깥쪽으로 진행하며 line이 다시 이어지는지."""
    run, start = 0, None
    for k, v in enumerate(seq):
      if v:
        if run == 0:
          start = k
        run += 1
        if run >= need and start <= search:
          return True
      else:
        run = 0
      if run == 0 and k > search:
        return False
    # 사진 끝에 닿음: line이 끝까지 이어지면 인정
    return run >= 5 and start is not None and start <= search

  before = is_line[: br[0]][::-1]
  after = is_line[br[1] + 1 :]
  return resumes(before) and resumes(after)


def _line_cover(coords, lg):
  """결함이 각 line 줄무늬 폭을 얼마나 덮는지 → (최대 비율, 그 줄무늬 번호)."""
  cols = set(int(c) for c in np.unique(coords))
  best, idx = 0.0, None
  for j, (a, b, k) in enumerate(lg["stripes"]):
    if k != LINE:
      continue
    c = max(1, int(0.12 * (b - a)))
    lo, hi = a + c, b - c
    if hi <= lo:
      continue
    cov = sum(1 for x in range(lo, hi + 1) if x in cols) / (hi - lo + 1)
    if cov > best:
      best, idx = cov, j
  return best, idx


def _bridges_gap(coords, lg):
  """결함이 어떤 gap의 양쪽 line에 모두 닿아 있는지(= line과 line을 연결) 확인."""
  if coords.size == 0:
    return False
  cmin, cmax = coords.min(), coords.max()
  for a, b, kind in lg["stripes"]:
    if kind != GAP:
      continue
    tol = 0.2 * (b - a)
    if cmin <= a + tol and cmax >= b - tol:
      return True
  return False


# ---------------------------------------------------------------------------
# gap이 실제로 막혔는지 이미지로 확인
#   gap 안쪽에서 행(줄무늬 방향)마다 'gap 고유의 하얀 정도'를 가진 픽셀 비율을 계산.
#   어떤 구간에서 이 비율이 거의 0이면(이물·찌꺼기·line 물질로 gap 폭 전체가 덮임)
#   그 자리는 gap이 막힌 것 = 양쪽 line이 연결된 것(short).
# ---------------------------------------------------------------------------
CLOSED_FRAC = 0.15  # 이 비율 미만이면 그 행은 'gap 막힘'
OPEN_FRAC = 0.7     # 이 비율 초과면 그 행은 '정상 gap'


def compute_gap_openness(image_bgr, lg):
  """gap 줄무늬별로 행마다 정상 gap 픽셀 비율을 계산해 {(a, b): 배열} 로 반환."""
  img = image_bgr if lg["vertical"] else image_bgr.transpose(1, 0, 2)
  wh = img.min(axis=2).astype(np.float32)

  def interior(a, b, f):
    c = max(2, int(f * (b - a)))
    return a + c, b - c

  def pixels(kind):
    parts = [wh[:, lo : hi + 1].ravel() for a, b, k in lg["stripes"] if k == kind
             for lo, hi in [interior(a, b, 0.25)] if hi > lo]
    return np.concatenate(parts) if parts else None

  lp, gp = pixels(LINE), pixels(GAP)
  if lp is None or gp is None:
    return {}
  lw, gw = float(np.median(lp)), float(np.median(gp))
  d = abs(lw - gw)
  if d < 5:  # line/gap 대비가 너무 작으면 판단하지 않음
    return {}
  res = {}
  for a, b, k in lg["stripes"]:
    if k != GAP:
      continue
    lo, hi = interior(a, b, 0.15)
    if hi - lo < 2:
      continue
    band = wh[:, lo : hi + 1]
    gaplike = (band > gw - 0.8 * d) & (band < gw + 0.4 * d)
    res[(a, b)] = gaplike.mean(axis=1)
    linelike = np.abs(band - lw) < 0.35 * d
    lg.setdefault("gap_linefrac", {})[(a, b)] = linelike.mean(axis=1)
  return res


def _closed_runs(frac, min_len):
  runs, y, n = [], 0, len(frac)
  closed = frac < CLOSED_FRAC
  while y < n:
    if not closed[y]:
      y += 1
      continue
    y0 = y
    while y < n and closed[y]:
      y += 1
    if y - y0 >= min_len:
      runs.append((y0, y - 1))
  return runs


def _longest_true_run(seg):
  run = best = 0
  for v in seg:
    run = run + 1 if v else 0
    best = max(best, run)
  return best


def gap_closed_between(lg, gap_key, a0, a1):
  """gap 줄무늬 gap_key에서 줄무늬 방향 a0~a1 구간(약간 여유) 안에 막힌 곳이 있는지."""
  frac = lg.get("gap_open", {}).get(gap_key)
  if frac is None:
    return False
  w = gap_key[1] - gap_key[0]
  pad = int(0.25 * w)
  seg = frac[max(0, int(a0) - pad) : int(a1) + pad + 1]
  return bool(_closed_runs(seg, max(3, int(0.1 * w))))


def line_bridge_between(lg, gap_key, a0, a1):
  """gap_key gap 안에서 a0~a1 구간(약간 여유)에 line 물질이 gap 폭을 가로질러 채운 곳이 있는지."""
  frac = lg.get("gap_linefrac", {}).get(gap_key)
  if frac is None:
    return False
  w = gap_key[1] - gap_key[0]
  pad = int(0.25 * w)
  seg = frac[max(0, int(a0) - pad) : int(a1) + pad + 1] >= S_BRIDGE_FRAC
  return _longest_true_run(seg) >= max(3, int(0.1 * w))


def s_material_ok(lg, coords_cross, along, dk):
  """S 인정: 진한 물질(dk ≥ 기준) 또는 line 물질로 gap이 연결됨. (통과, 설명)"""
  if dk is not None and dk >= S_MIN_DARKNESS:
    return True, f"진한 물질, 진한 정도 {dk:.0f}"
  for a, b, k in lg["stripes"]:
    if k == GAP and coords_cross.min() < b and coords_cross.max() > a:
      if line_bridge_between(lg, (a, b), along.min(), along.max()):
        return True, "line 물질이 gap을 채워 연결"
  return False, (f"연한 얼룩 (진한 정도 {dk:.0f} < {S_MIN_DARKNESS}, line 물질 연결도 없음)"
                 if dk is not None else "line 물질 연결 없음")


def find_closed_gaps(lg):
  """이미지에서 gap이 막힌 곳(= short)을 모두 찾음. [(a, b, s0, s1), ...]
  gap 줄무늬 끝(사진 가장자리)과 구분하려고, 막힌 곳 위아래로 정상 gap이 이어져야 인정."""
  out = []
  for (a, b), frac in lg.get("gap_open", {}).items():
    if a == 0 or b >= lg["n_cross"] - 1:
      continue  # 한쪽 경계가 사진 밖인 gap은 제외
    w = b - a
    runs = _closed_runs(frac, max(3, int(0.1 * w)))
    merged = []
    for r0, r1 in runs:  # 가까운 막힘 구간은 하나로
      if merged and r0 - merged[-1][1] <= w:
        merged[-1] = (merged[-1][0], r1)
      else:
        merged.append((r0, r1))
    openr = frac > OPEN_FRAC
    for r0, r1 in merged:
      if r1 - r0 + 1 < max(5, int(0.15 * w)):
        continue  # 너무 얇은 막힘(가는 선 등)은 제외
      if (_longest_true_run(openr[:r0]) >= w
          and _longest_true_run(openr[r1 + 1 :]) >= w):
        out.append((a, b, r0, r1))
  return out


def _covered_by(box, others, frac=0.5):
  px1, py1, px2, py2 = box
  area = max(1.0, (px2 - px1) * (py2 - py1))
  for sx1, sy1, sx2, sy2 in others:
    iw = max(0.0, min(px2, sx2) - max(px1, sx1))
    ih = max(0.0, min(py2, sy2) - max(py1, sy1))
    if iw * ih / area > frac:
      return True
  return False


def _touches_any(box, others, vertical, max_gap):
  """box가 others 중 하나와 줄무늬를 가로지르는 방향으로 max_gap 이내로 붙어 있고,
  줄무늬 방향으로는 겹치는지."""
  x1, y1, x2, y2 = box
  for ox1, oy1, ox2, oy2 in others:
    if vertical:
      cross_gap = max(ox1, x1) - min(ox2, x2)
      along_overlap = min(oy2, y2) - max(oy1, y1)
    else:
      cross_gap = max(oy1, y1) - min(oy2, y2)
      along_overlap = min(ox2, x2) - max(ox1, x1)
    if cross_gap <= max_gap and along_overlap > 0:
      return True
  return False


def _det_pixels(det_i, detections):
  """결함 픽셀 좌표 (ys, xs) — 마스크 우선, 없으면 박스."""
  if detections.mask is not None and detections.mask[det_i].any():
    return np.nonzero(detections.mask[det_i])
  x1, y1, x2, y2 = map(int, detections.xyxy[det_i])
  ys, xs = np.mgrid[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
  return ys.ravel(), xs.ravel()


def _line_gray(gray, lg):
  vals = []
  for a, b, k in lg["stripes"]:
    if k != LINE:
      continue
    c = max(2, int(0.25 * (b - a)))
    if b - c <= a + c:
      continue
    band = gray[:, a + c : b - c + 1] if lg["vertical"] else gray[a + c : b - c + 1, :]
    vals.append(band.ravel())
  return float(np.median(np.concatenate(vals))) if vals else None


def region_darkness(ys, xs, gray, line_gray, pct=10):
  if gray is None or line_gray is None or len(xs) == 0:
    return None
  ys = np.clip(ys, 0, gray.shape[0] - 1)
  xs = np.clip(xs, 0, gray.shape[1] - 1)
  return line_gray - float(np.percentile(gray[ys, xs], pct))


def _gap_pixels(det_i, detections, lg):
  """S 판정용: 결함 픽셀 중 gap 위에 있는 것만 (gap을 막은 물질의 진한 정도를 보기 위해).
  line 위 이물(P)이 S 영역 안에 같이 있어도 그 진한 색으로 S가 인정되지 않도록."""
  ys, xs = _det_pixels(det_i, detections)
  if lg is None or len(xs) == 0:
    return ys, xs
  cross = xs if lg["vertical"] else ys
  lab = lg["col_label"][np.clip(cross, 0, len(lg["col_label"]) - 1)]
  keep = lab == GAP
  return (ys[keep], xs[keep]) if keep.sum() >= 20 else (ys, xs)


def check_particle(det_i, detections, lg, gray, line_gray):
  """P 크기·색 기준 확인. (통과 여부, 설명)"""
  ys, xs = _det_pixels(det_i, detections)
  if len(xs) == 0:
    return True, ""
  size = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
  w_line = float(np.median([b - a for a, b, k in lg["stripes"] if k == LINE]))
  need = P_MIN_SIZE_RATIO * w_line
  if size < need:
    return False, f"너무 작음 (크기 {size}px < line 폭 {w_line:.0f}px의 절반 {need:.0f}px)"
  if gray is not None and line_gray is not None:
    ys = np.clip(ys, 0, gray.shape[0] - 1)
    xs = np.clip(xs, 0, gray.shape[1] - 1)
    darkness = line_gray - float(np.percentile(gray[ys, xs], 30))
    if darkness < P_MIN_DARKNESS:
      return False, f"색이 연함 (진한 정도 {darkness:.0f} < {P_MIN_DARKNESS})"
    return True, f"크기 {size}px, 진한 정도 {darkness:.0f}"
  return True, f"크기 {size}px"


def apply_line_gap_rule(detections, lg, image_bgr=None, protect=True):
  """S는 gap 양쪽 line을 실제로 연결하고 있을 때만 인정
  (한쪽 line에만 붙은 돌기·찌꺼기는 S 아님 → 제거).
  P, O, PD는 line 위에 있을 때만 인정.
  O는 추가로 line 양쪽이 모두 이어져 있을 때만 open (line 끝의 U자 모양 등은 제외).
  gap 위 O가 옆 line의 O와 붙어 있으면 같은 open 결함의 조각으로 보고 제거.
  gap 위의 P/O/PD가 gap 양쪽 line까지 닿아 있으면(line과 line이 붙음) short(S)로 보고,
  이미 그 자리에 S가 있으면 제거, 없으면 S로 전환. 그 외 gap 위 P/O/PD는 제거."""
  n = len(detections)
  positions = np.array(["-"] * n, dtype=object)
  if lg is None or n == 0:
    detections.data["position"] = positions.astype(str)
    return detections

  vertical = lg["vertical"]
  col_label, col_stripe = lg["col_label"], lg["col_stripe"]
  names = list(detections.data["class_name"])
  confs = detections.confidence
  log = RULE_LOG
  log.clear()
  gray = (cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
          if image_bgr is not None else None)
  line_gray = _line_gray(gray, lg) if gray is not None else None

  def note(i, text):
    x1, _, x2, _ = map(int, detections.xyxy[i])
    log.append(f"{names[i].upper()} {confs[i]*100:.0f}% (x {x1}~{x2}): {text}")

  def ext(coords):
    return f"결함 범위 {int(coords.min())}~{int(coords.max())}"
  s_boxes = []  # line끼리 연결해서 실제로 남는 S만 (아래 루프에서 채움)
  line_o_boxes = []  # line 위에서 open으로 인정된 O
  to_convert = []  # gap 위에서 line을 연결해 S로 바꿀 후보
  pending_s = []  # 마스크는 안 닿지만 이미지상 gap이 막힌 S (open 조각인지 확인 필요)

  keep, new_names = [], list(names)
  # S를 먼저 처리해서 s_boxes를 채운 뒤 P/O를 처리
  order = [i for i in range(n) if names[i] == "s"] + [
      i for i in range(n) if names[i] != "s"
  ]
  for i in order:
    thr = PROTECT_CONF.get(names[i]) if protect else None
    if thr is not None and float(confs[i]) >= thr:
      positions[i] = "신뢰도 보호"
      keep.append(i)
      if names[i] == "o":
        line_o_boxes.append(detections.xyxy[i])
      elif names[i] == "s":
        s_boxes.append(detections.xyxy[i])
      note(i, f"유지 — 신뢰도 {confs[i]*100:.0f}% ≥ {thr*100:.0f}% → 규칙 적용 안 함 (모델 결과 그대로)")
      continue
    if names[i] == "s":
      coords = np.clip(_cross_coords(i, detections, vertical), 0, len(col_label) - 1)
      along = _along_coords(i, detections, vertical)
      closed_img = any(
          gap_closed_between(lg, (a, b), along.min(), along.max())
          for a, b, k in lg["stripes"]
          if k == GAP and coords.min() < b and coords.max() > a
      )
      dk = region_darkness(*_gap_pixels(i, detections, lg), gray, line_gray)
      ok_mat, why_mat = s_material_ok(lg, coords, along, dk)
      if not ok_mat:
        note(i, f"제거 — {why_mat} → S 아님")
        continue
      if _bridges_gap(coords, lg):
        positions[i] = "gap 연결"
        keep.append(i)
        s_boxes.append(detections.xyxy[i])
        note(i, f"유지 — gap 양쪽 line 연결 ({ext(coords)}, {why_mat})")
      elif closed_img:
        pending_s.append((i, coords))  # 옆 line의 O를 모두 확인한 뒤 결정
      else:
        note(i, f"제거 — line끼리 연결 안 함 ({ext(coords)})")
      continue  # line과 line을 연결하지 않는 S → 제거
    if names[i] not in LINE_ONLY_CLASSES:
      keep.append(i)
      continue
    coords = np.clip(_cross_coords(i, detections, vertical), 0, len(col_label) - 1)
    labels = col_label[coords]
    n_line, n_gap = int((labels == LINE).sum()), int((labels == GAP).sum())
    share = f"line {n_line}px / gap {n_gap}px"
    if names[i] == "p":
      # P: 면적의 1/3 이상이 line(테두리 포함) 위에 걸쳐 있으면 'line 위 이물' 후보
      n_edge = int((labels == EDGE).sum())
      on_line_frac = (n_line + n_edge) / max(1, len(coords))
      share += f", line 위 {on_line_frac*100:.0f}%"
      # gap 쪽이 더 많고 그 자리 gap이 막혀 양쪽 line이 이어졌으면 P가 아니라 S 후보
      bridged_gap = False
      if n_gap > n_line:
        g_ids = col_stripe[coords][labels == GAP]
        ga_, gb_, _ = lg["stripes"][int(np.bincount(g_ids).argmax())]
        tol_ = 0.2 * (gb_ - ga_)
        along_ = _along_coords(i, detections, vertical)
        bridged_gap = (coords.min() <= ga_ + tol_ and coords.max() >= gb_ - tol_) or \
            gap_closed_between(lg, (ga_, gb_), along_.min(), along_.max())
        if bridged_gap:
          share += ", gap 막힘 → S 후보"
      if (n_gap == 0 or on_line_frac >= P_MIN_ON_LINE) and not bridged_gap:
        if n_line < n_gap and _covered_by(detections.xyxy[i], s_boxes):
          note(i, f"제거 — gap 위 이물이 S 영역 안에 있음 → S의 일부 ({share})")
          continue
        ok, why = check_particle(i, detections, lg, gray, line_gray)
        where = "line 위" if n_line > 0 else "line 테두리"
        if not ok:
          note(i, f"제거 — {where}지만 {why}")
          continue
        positions[i] = where
        keep.append(i)
        note(i, f"유지 — {where} 이물 ({why}; {share})")
        continue
      # line에 닿지 않은 gap 위 이물 → 아래 gap 규칙(line 연결 시 S, 아니면 제거)
    elif (n_line + n_gap == 0 or n_line >= n_gap
          or (names[i] == "o" and _line_cover(coords, lg)[0] >= 0.5)):
      # O는 찌꺼기가 양옆 gap으로 퍼져도 line 폭의 절반 이상을 덮으면 line 위 open
      if names[i] == "o" and image_bgr is not None and n_line > 0:
        cov, cov_idx = _line_cover(coords, lg)
        if cov >= 0.5:
          stripe_idx = cov_idx
        else:
          line_ids = col_stripe[coords][labels == LINE]
          stripe_idx = int(np.bincount(line_ids).argmax())
        along = _along_coords(i, detections, vertical)
        if not is_line_break(image_bgr, lg, stripe_idx, along):
          note(i, f"제거 — line 위지만 line이 원래 끝나는 자리 ({share})")
          continue  # line이 원래 끝나는 자리 → open 아님
      positions[i] = "line"
      keep.append(i)
      if names[i] == "o":
        line_o_boxes.append(detections.xyxy[i])
      note(i, f"유지 — line 위 ({share})")
      continue

    # gap 위의 P/O/PD
    gap_ids = col_stripe[coords][labels == GAP]
    gap_idx = int(np.bincount(gap_ids).argmax())
    ga, gb, _ = lg["stripes"][gap_idx]
    tol = 0.2 * (gb - ga)
    bridges = coords.min() <= ga + tol and coords.max() >= gb - tol
    if not bridges:
      # 마스크가 양쪽 line까지 안 닿아도, 이미지상 그 자리 gap이 막혀 있으면 연결로 봄
      along = _along_coords(i, detections, vertical)
      if gap_closed_between(lg, (ga, gb), along.min(), along.max()):
        bridges = True
        share += ", 이미지상 gap 막힘"
    if not bridges:
      note(i, f"제거 — gap 위, line 연결 안 함 ({share}, gap {ga}~{gb}, {ext(coords)})")
      continue  # gap 위에만 있는 P → 결함 아님

    to_convert.append((i, share, ga, gb))  # line 위 O를 모두 확인한 뒤 결정

  # 이미지상으로만 gap 막힘이 확인된 S: 옆 line의 O와 붙어 있으면 open 조각으로 보고 제거
  med_gap = float(np.median([b - a for a, b, k in lg["stripes"] if k == GAP] or [40]))
  for i, coords in pending_s:
    if _touches_any(detections.xyxy[i], line_o_boxes, vertical, 0.25 * med_gap):
      note(i, f"제거 — 옆 line의 O와 붙어 있음 → 같은 open 결함의 일부 ({ext(coords)})")
      continue
    positions[i] = "gap 연결"
    keep.append(i)
    s_boxes.append(detections.xyxy[i])
    note(i, f"유지 — 이미지상 gap 막힘으로 line 연결 확인 ({ext(coords)})")

  # gap 위에서 line을 연결한 P/O/PD → S 전환.
  # 단, gap 위 O가 바로 옆 line의 O(open)와 붙어 있으면 그 open에서 떨어져 나온
  # 조각으로 보고 S로 바꾸지 않고 제거한다 (같은 open 결함의 일부).
  for i, share, ga, gb in to_convert:
    if names[i] == "o" and _touches_any(detections.xyxy[i], line_o_boxes,
                                        vertical, 0.25 * (gb - ga)):
      note(i, f"제거 — 옆 line의 O와 붙어 있음 → 같은 open 결함의 일부 ({share})")
      continue
    px1, py1, px2, py2 = detections.xyxy[i]
    p_area = max(1.0, (px2 - px1) * (py2 - py1))
    covered = False
    for sx1, sy1, sx2, sy2 in s_boxes:
      iw = max(0.0, min(px2, sx2) - max(px1, sx1))
      ih = max(0.0, min(py2, sy2) - max(py1, sy1))
      if iw * ih / p_area > 0.5:
        covered = True
        break
    if covered:
      note(i, f"제거 — gap 위, 같은 자리에 S 있음 ({share})")
      continue  # 이미 S로 잡혀 있음
    dk = region_darkness(*_gap_pixels(i, detections, lg), gray, line_gray)
    ok_mat, why_mat = s_material_ok(
        lg, np.clip(_cross_coords(i, detections, vertical), 0, len(col_label) - 1),
        _along_coords(i, detections, vertical), dk)
    if not ok_mat:
      note(i, f"제거 — gap 위 {why_mat} → S 아님")
      continue
    new_names[i] = "s"
    positions[i] = "gap→S"
    keep.append(i)
    note(i, f"S로 전환 — gap 위에서 line 연결 ({share}, gap {ga}~{gb})")

  detections.data["class_name"] = np.asarray(new_names, dtype=str)
  detections.class_id = np.array(
      [CLASS_IDS.get(nm, UNKNOWN_ID) for nm in new_names], dtype=int
  )
  detections.data["position"] = positions.astype(str)
  return (
      detections[np.array(sorted(keep), dtype=int)] if keep else _empty_detections()
  )


def fix_line_shaped_shorts(detections, lg, image_shape=None):
  """줄무늬를 따라 길게 뻗은 가는 S는 short가 아니라 망가진 line(PD).
  short는 gap을 가로질러 line끼리 잇는 것이라 줄무늬 방향으로 길쭉할 수 없다.
  (모든 line이 망가진 사진은 line/gap 판별이 어긋날 수 있어 모양만으로 판단,
   단 대부분이 gap 위면 gap을 길게 메운 것일 수 있으므로 S로 둔다)"""
  n = len(detections)
  if n == 0:
    return detections
  names = list(detections.data["class_name"])
  if "s" not in names:
    return detections
  if lg is not None:
    vertical = lg["vertical"]
  else:
    return detections
  extent = None
  if image_shape is not None:
    extent = image_shape[0] if vertical else image_shape[1]
  new_names = list(names)
  for i in range(n):
    if names[i] != "s":
      continue
    ys, xs = _det_pixels(i, detections)
    if len(xs) < 20:
      continue
    along = (ys if vertical else xs).astype(np.float64)
    cross = (xs if vertical else ys).astype(np.float64)
    length = float(np.percentile(along, 98) - np.percentile(along, 2))
    width = max(1.0, float(np.percentile(cross, 95) - np.percentile(cross, 5)))
    cov = np.cov(np.vstack([cross - cross.mean(), along - along.mean()]))
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    angle = np.degrees(np.arctan2(abs(major[0]), abs(major[1])))
    long_enough = length >= 6.0 * width and (extent is None or length >= 0.4 * extent)
    if not (angle <= SC_ALIGN_DEG and long_enough):
      continue
    coords = np.clip(cross.astype(int), 0, len(lg["col_label"]) - 1)
    labels = lg["col_label"][coords]
    if (labels == GAP).mean() >= 0.6:
      continue
    new_names[i] = "pd"
    x1, _, x2, _ = map(int, detections.xyxy[i])
    RULE_LOG.append(f"S {detections.confidence[i]*100:.0f}% (x {x1}~{x2}): PD로 변경 — 줄무늬를 따라 길게 뻗음 (길이 {length:.0f}px, 폭 {width:.0f}px) → short 아님, 망가진 line")
  detections.data["class_name"] = np.asarray(new_names, dtype=str)
  detections.class_id = np.array([CLASS_IDS.get(nm, UNKNOWN_ID) for nm in new_names], dtype=int)
  return detections


def fix_straight_scratches(detections, lg, image_bgr=None):
  """줄무늬와 나란한 가늘고 긴 SC는 스크래치가 아니라 망가진 line(PD).
  line/gap 판별이 실패해도(lg 없음, line 0개) 세로 직선 SC는 PD로 바꾼다."""
  n = len(detections)
  if n == 0:
    return detections
  names = list(detections.data["class_name"])
  if "sc" not in names:
    return detections
  if lg is not None:
    vertical = lg["vertical"]
  elif image_bgr is not None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    vertical = np.abs(np.diff(gray, axis=1)).mean() >= np.abs(np.diff(gray, axis=0)).mean()
  else:
    return detections
  line_ws = [b - a for a, b, k in lg["stripes"] if k == LINE] if lg is not None else []
  all_ws = [b - a for a, b, k in lg["stripes"]] if lg is not None else []
  has_lines = len(line_ws) > 0
  w_line = float(np.median(line_ws or all_ws)) if (line_ws or all_ws) else None
  pd_boxes = [detections.xyxy[i] for i in range(n) if names[i] == "pd"]
  positions = list(detections.data.get("position", ["-"] * n))
  keep, new_names = [], list(names)
  for i in range(n):
    if names[i] != "sc":
      keep.append(i)
      continue
    ys, xs = _det_pixels(i, detections)
    if len(xs) < 20:
      keep.append(i)
      continue
    along = (ys if vertical else xs).astype(np.float64)
    cross = (xs if vertical else ys).astype(np.float64)
    cov = np.cov(np.vstack([cross - cross.mean(), along - along.mean()]))
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    angle = np.degrees(np.arctan2(abs(major[0]), abs(major[1])))  # 줄무늬 방향과의 각도
    elong = np.sqrt(max(evals.max(), 1e-9) / max(evals.min(), 1e-9))
    width = max(1.0, float(np.percentile(cross, 95) - np.percentile(cross, 5)))
    length = float(np.percentile(along, 98) - np.percentile(along, 2))
    extent = detections.mask.shape[1] if (detections.mask is not None and vertical) else (
        detections.mask.shape[2] if detections.mask is not None else None)
    # 줄무늬 방향으로 쭉 뻗은 것은 폭과 상관없이 스크래치가 아님 (SC는 세로 직선으로 나오지 않음)
    long_run = length >= 6.0 * width or (extent is not None and length >= 0.5 * extent)
    narrow = w_line is None or width <= SC_MAX_WIDTH_RATIO * w_line or long_run
    if not (angle <= SC_ALIGN_DEG and elong >= 3.0 and narrow):
      keep.append(i)  # 대각선 등 진짜 스크래치 모양
      continue
    if has_lines:
      coords = np.clip(cross.astype(int), 0, len(lg["col_label"]) - 1)
      cov_line, _ = _line_cover(coords, lg)
      on_line = cov_line >= 0.5
    else:
      on_line = True  # line 위치를 모르면 망가진 line(PD)으로 본다
    if on_line and _covered_by(detections.xyxy[i], pd_boxes, 0.3):
      RULE_LOG.append(f"SC {detections.confidence[i]*100:.0f}%: 제거 — 줄무늬와 나란한 직선(각도 {angle:.0f}°) → 스크래치 아님, 같은 자리에 PD 있음")
      continue
    if on_line:
      new_names[i] = "pd"
      positions[i] = "SC→PD (세로 직선)"
      keep.append(i)
      RULE_LOG.append(f"SC {detections.confidence[i]*100:.0f}%: PD로 변경 — 줄무늬와 나란한 직선(각도 {angle:.0f}°)이 line 위에 있음 → 망가진 line")
    else:
      RULE_LOG.append(f"SC {detections.confidence[i]*100:.0f}%: 제거 — 줄무늬와 나란한 직선(각도 {angle:.0f}°) → 스크래치 아님")
  detections.data["class_name"] = np.asarray(new_names, dtype=str)
  detections.class_id = np.array([CLASS_IDS.get(nm, UNKNOWN_ID) for nm in new_names], dtype=int)
  detections.data["position"] = np.asarray(positions, dtype=str)
  return detections[np.array(keep, dtype=int)] if keep else _empty_detections()


def apply_scratch_rule(detections, overlap_thresh=0.5):
  """웨이퍼 스크래치(SC)는 SC로만 본다.
  S가 SC와 같은 자리에 겹쳐 있으면(스크래치가 line을 가로질러 short처럼 보이는 경우) S를 제거.
  마스크가 있으면 'S 마스크 중 SC 마스크(살짝 두껍게)와 겹치는 비율'로 판단하고,
  없으면 'S 박스 중 SC 박스와 겹치는 비율'로 판단한다.
  (대각선 스크래치는 박스가 넓어도 마스크 기준이라 주변의 진짜 S를 지우지 않음)"""
  n = len(detections)
  if n == 0:
    return detections
  names = list(detections.data["class_name"])
  sc_idx = [i for i in range(n) if names[i] == "sc"]
  if not sc_idx:
    return detections

  use_mask = detections.mask is not None
  if use_mask:
    sc_union = np.zeros(detections.mask.shape[1:], dtype=np.uint8)
    for i in sc_idx:
      sc_union |= detections.mask[i].astype(np.uint8)
    h, w = sc_union.shape
    k = max(3, int(round(max(h, w) * 0.005)) | 1)  # 마스크 경계 오차 보정
    sc_union = cv2.dilate(sc_union, np.ones((k, k), np.uint8)).astype(bool)

  keep = []
  for i in range(n):
    if names[i] != "s":
      keep.append(i)
      continue
    if use_mask and detections.mask[i].any():
      s_mask = detections.mask[i]
      frac = (s_mask & sc_union).sum() / s_mask.sum()
    else:
      sx1, sy1, sx2, sy2 = detections.xyxy[i]
      s_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
      frac = 0.0
      for j in sc_idx:
        cx1, cy1, cx2, cy2 = detections.xyxy[j]
        iw = max(0.0, min(sx2, cx2) - max(sx1, cx1))
        ih = max(0.0, min(sy2, cy2) - max(sy1, cy1))
        frac = max(frac, iw * ih / s_area)
    if frac < overlap_thresh:
      keep.append(i)
  return detections[np.array(keep, dtype=int)] if keep else _empty_detections()


def _concat_detections(det, new):
  if len(det) == 0:
    return new
  keys = set(det.data) | set(new.data)
  data = {}
  for k in keys:
    a = np.asarray(det.data.get(k, [""] * len(det)), dtype=str)
    b = np.asarray(new.data.get(k, [""] * len(new)), dtype=str)
    data[k] = np.concatenate([a, b])
  mask = None
  if det.mask is not None and new.mask is not None:
    mask = np.concatenate([det.mask, new.mask])
  return sv.Detections(
      xyxy=np.concatenate([det.xyxy, new.xyxy]).astype(np.float32),
      confidence=np.concatenate([det.confidence, new.confidence]).astype(np.float32),
      class_id=np.concatenate([det.class_id, new.class_id]).astype(int),
      mask=mask,
      data=data,
  )


def merge_overlapping_shorts(detections, tol=2.0):
  """같은 자리에 겹쳐 잡힌 S(또는 PD) 여러 개를 하나로 합친다 (박스·마스크 합집합, 신뢰도는 최고값).
  PD는 같은 line에 여러 박스가 잡혀도 line 전체를 PD 1건으로 본다."""
  for cls in ("s", "pd"):
    detections = _merge_same_class(detections, cls, tol)
  return detections


def _merge_same_class(detections, cls, tol=2.0):
  n = len(detections)
  if n < 2:
    return detections
  names = list(detections.data["class_name"])
  s_idx = [i for i in range(n) if names[i] == cls]
  if len(s_idx) < 2:
    return detections
  parent = {i: i for i in s_idx}

  def find(i):
    while parent[i] != i:
      parent[i] = parent[parent[i]]
      i = parent[i]
    return i

  box = detections.xyxy
  for a_ in range(len(s_idx)):
    for b_ in range(a_ + 1, len(s_idx)):
      i, j = s_idx[a_], s_idx[b_]
      if (min(box[i][2], box[j][2]) - max(box[i][0], box[j][0]) > -tol and
          min(box[i][3], box[j][3]) - max(box[i][1], box[j][1]) > -tol):
        parent[find(i)] = find(j)
  groups = {}
  for i in s_idx:
    groups.setdefault(find(i), []).append(i)
  if all(len(g) == 1 for g in groups.values()):
    return detections
  xyxy = detections.xyxy.copy()
  conf = detections.confidence.copy()
  mask = detections.mask.copy() if detections.mask is not None else None
  drop = set()
  for g in groups.values():
    if len(g) == 1:
      continue
    best = max(g, key=lambda k: float(conf[k]))
    xyxy[best] = [xyxy[g, 0].min(), xyxy[g, 1].min(), xyxy[g, 2].max(), xyxy[g, 3].max()]
    if mask is not None:
      mask[best] = mask[g].any(axis=0)
    drop.update(k for k in g if k != best)
    x1, _, x2, _ = map(int, xyxy[best])
    RULE_LOG.append(
        f"{cls.upper()} {conf[best]*100:.0f}% (x {x1}~{x2}): 같은 자리 {cls.upper()} {len(g)}개를 하나로 합침 "
        f"({', '.join(f'{conf[k]*100:.0f}%' for k in sorted(g, key=lambda k: -conf[k]))})")
  detections.xyxy = xyxy
  if mask is not None:
    detections.mask = mask
  keep = np.array([i for i in range(n) if i not in drop], dtype=int)
  return detections[keep]


def drop_duplicate_particles(detections, frac=0.6):
  """같은 자리에 겹친 P는 신뢰도 높은 것 하나만 남긴다."""
  n = len(detections)
  names = list(detections.data.get("class_name", []))
  p_idx = sorted([i for i in range(n) if names[i] == "p"], key=lambda i: -detections.confidence[i])
  if len(p_idx) < 2:
    return detections
  drop = set()
  kept = []
  for i in p_idx:
    if any(_covered_by(detections.xyxy[i], [detections.xyxy[j]], frac)
           or _covered_by(detections.xyxy[j], [detections.xyxy[i]], frac) for j in kept):
      drop.add(i)
      continue
    kept.append(i)
  if not drop:
    return detections
  keep = np.array([i for i in range(n) if i not in drop], dtype=int)
  return detections[keep]


def drop_particles_on_pd(detections, frac=0.5):
  """PD(망가진 line)로 판정된 line 위의 P는 따로 표시하지 않는다 (line 전체가 PD)."""
  n = len(detections)
  if n < 2:
    return detections
  names = list(detections.data["class_name"])
  pd_boxes = [detections.xyxy[i] for i in range(n) if names[i] == "pd"]
  if not pd_boxes:
    return detections
  keep = []
  for i in range(n):
    if names[i] == "p" and _covered_by(detections.xyxy[i], pd_boxes, frac):
      x1, _, x2, _ = map(int, detections.xyxy[i])
      RULE_LOG.append(f"P {detections.confidence[i]*100:.0f}% (x {x1}~{x2}): 제거 — PD로 판정된 line 위 → line 전체를 PD로 봄")
      continue
    keep.append(i)
  return detections[np.array(keep, dtype=int)] if keep else _empty_detections()


def add_rule_shorts(detections, lg, image_shape, image_bgr=None):
  """모델이 놓친 short: 이미지에서 gap이 막힌 곳 중 아직 아무 결함도 없는 자리를 S로 추가."""
  cands = find_closed_gaps(lg)
  if not cands:
    return detections
  h, w = image_shape[:2]
  vertical = lg["vertical"]
  boxes = [tuple(b) for b in detections.xyxy]
  names = list(detections.data.get("class_name", []))
  positions = list(detections.data.get("position", []))
  line_o = [detections.xyxy[i] for i in range(len(detections))
            if names[i] == "o" and positions[i] == "line"]
  new_boxes = []
  for a, b, s0, s1 in cands:
    box = (a, s0, b, s1) if vertical else (s0, a, s1, b)
    txt = f"x {box[0]}~{box[2]}, y {box[1]}~{box[3]}"
    if any(_overlaps(box, ob) for ob in boxes):
      RULE_LOG.append(f"규칙 S 후보 ({txt}): 생략 — 이미 다른 결함으로 잡힌 자리")
      continue
    if _touches_any(box, line_o, vertical, 0.25 * (b - a)):
      RULE_LOG.append(f"규칙 S 후보 ({txt}): 생략 — 옆 line의 O와 붙어 있음 (open 조각)")
      continue
    if image_bgr is not None:
      gray_r = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
      x1, y1, x2, y2 = map(int, box)
      ys, xs = np.mgrid[y1 : y2 + 1, x1 : x2 + 1]
      dk = region_darkness(ys.ravel(), xs.ravel(), gray_r, _line_gray(gray_r, lg))
      ok_mat, why_mat = s_material_ok(lg, np.arange(a, b + 1), np.arange(s0, s1 + 1), dk)
      if not ok_mat:
        RULE_LOG.append(f"규칙 S 후보 ({txt}): 생략 — {why_mat}")
        continue
    RULE_LOG.append(f"S (규칙) ({txt}): 추가 — 모델은 못 잡았지만 이미지상 gap이 막혀 line 연결")
    new_boxes.append(box)
  if not new_boxes:
    return detections
  n = len(new_boxes)
  mask = None
  if detections.mask is not None or len(detections) == 0:
    mask = np.zeros((n, h, w), dtype=bool)
    for k, (x1, y1, x2, y2) in enumerate(new_boxes):
      mask[k, int(y1) : int(y2) + 1, int(x1) : int(x2) + 1] = True
  new = sv.Detections(
      xyxy=np.array(new_boxes, dtype=np.float32),
      confidence=np.zeros(n, dtype=np.float32),
      class_id=np.full(n, CLASS_IDS["s"], dtype=int),
      mask=mask,
      data={
          "class_name": np.array(["s"] * n),
          "position": np.array(["규칙: gap 막힘"] * n),
          "source": np.array(["rule"] * n),
      },
  )
  return _concat_detections(detections, new)


# 모델이 놓친 PD(망가진 line) 찾기
#   정상 line = 밝은 속 + 양쪽 얇은 빨간 테두리 → 어두운 띠(테두리)마다 극소점이 1개.
#   망가진 line = 속까지 어두워져 양쪽 테두리가 한 띠로 합쳐짐 → 한 띠 안에 극소점 2개 이상.
#   위아래 대부분(75% 이상)이 어두워야 함 (부분 이물·S 덩어리 제외).
#   망가진 line이 2개 이상이면 그 간격(line 주기)에 맞는 자리의 어두운 띠도 PD로 본다.
PD_BAND_DEPTH = 35.0     # 배경(초록 채널 75백분위)보다 이만큼 어두우면 어두운 띠
PD_BAND_ROWS = 0.75      # 띠가 어두운 행 비율
PD_BAND_BUMP = 6.0       # 띠 안 두 극소점 사이 볼록한 정도


def find_damaged_lines(image_bgr, vertical=True):
  """망가진 line 후보 [(c0, c1, r0, r1, 이유)] — c: 줄무늬를 가로지르는 좌표, r: 줄무늬 방향 좌표."""
  g = image_bgr[:, :, 1].astype(np.float32)
  if not vertical:
    g = g.T
  n_rows, n_cols = g.shape
  s = _smooth_1d(np.median(g, axis=0), 2.0)
  bg = float(np.percentile(s, 75))
  thr = bg - PD_BAND_DEPTH
  dark = s < thr
  bands = []
  x = 0
  while x < n_cols:
    if not dark[x]:
      x += 1
      continue
    y = x
    while y < n_cols and dark[y]:
      y += 1
    seg = s[x:y]
    mins = [i for i in range(1, len(seg) - 1) if seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]]
    double = False
    for i in range(len(mins)):
      for j in range(i + 1, len(mins)):
        a, b = mins[i], mins[j]
        if b - a >= 6 and seg[a : b + 1].max() - max(seg[a], seg[b]) >= PD_BAND_BUMP:
          double = True
    row_dark = np.median(g[:, x:y], axis=1) < thr
    bands.append({"c0": x, "c1": y - 1, "double": double, "rows": float(row_dark.mean()),
                  "row_dark": row_dark})
    x = y

  def row_span(rd):
    idx = np.flatnonzero(rd)
    return (int(idx.min()), int(idx.max())) if len(idx) else (0, n_rows - 1)

  seeds = [b for b in bands if b["double"] and b["rows"] >= PD_BAND_ROWS]
  normal = [b["c1"] - b["c0"] for b in bands if not b["double"] and b["rows"] >= PD_BAND_ROWS]
  if normal and len(normal) >= len(seeds):
    # 정상 테두리가 더 많은 사진: 테두리보다 확실히 넓은 띠만 (얇은 선이 테두리 옆에 붙은 경우 제외)
    w_edge = float(np.median(normal))
    seeds = [b for b in seeds if (b["c1"] - b["c0"]) >= 1.6 * w_edge]
  out = [(b["c0"], b["c1"], *row_span(b["row_dark"]), "테두리 두 개가 합쳐진 어두운 띠 (속이 어두운 line)")
         for b in seeds]
  if len(seeds) >= 2:
    centers = np.array(sorted((b["c0"] + b["c1"]) / 2.0 for b in seeds))
    diffs = np.diff(centers)
    period = float(np.median(diffs[diffs <= 1.5 * diffs.min()]))  # 이웃한 망가진 line 간격
    w_med = float(np.median([b["c1"] - b["c0"] for b in seeds]))
    if period > 0:
      for b in bands:
        if b in seeds or b["rows"] < PD_BAND_ROWS or (b["c1"] - b["c0"]) < 0.5 * w_med:
          continue
        c = (b["c0"] + b["c1"]) / 2.0
        near = centers[np.argmin(np.abs(centers - c))]
        k = abs(c - near) / period
        if round(k) >= 1 and abs(k - round(k)) <= 0.15:
          out.append((b["c0"], b["c1"], *row_span(b["row_dark"]),
                      "망가진 line들과 같은 간격 자리의 어두운 띠"))
  return sorted(out)


def flip_lines_by_defects(lg, detections, image_bgr=None, min_conf=0.5):
  """line/gap 색이 비슷하거나(노란 패턴) 망가진 line 때문에 판별이 뒤집힌 경우 바로잡기.
  '반드시 line인 곳' = 신뢰도 높은 PD·O 위치 + 이미지에서 찾은 망가진 line(어두운 띠).
  줄무늬는 line/gap이 일정한 주기로 반복되므로, 이 위치들이 지금 판별의 gap 쪽 주기에
  더 잘 맞으면 line/gap을 서로 바꾼다."""
  if lg is None:
    return lg
  stripes = lg["stripes"]
  centers = np.array([(a + b) / 2.0 for a, b, _ in stripes])
  kinds = np.array([k for *_, k in stripes])
  if len(centers) < 5 or (kinds == LINE).sum() < 2 or (kinds == GAP).sum() < 2:
    return lg
  period = float(np.median(centers[2:] - centers[:-2]))
  if period <= 0:
    return lg

  def phase(vals):
    ang = 2 * np.pi * (np.asarray(vals) % period) / period
    return float(np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi) * period / (2 * np.pi))

  def cdist(x, ph):
    d = abs((x % period) - ph)
    return min(d, period - d)

  ph_line, ph_gap = phase(centers[kinds == LINE]), phase(centers[kinds == GAP])
  evidence = []
  names = list(detections.data["class_name"]) if len(detections) else []
  for i in range(len(names)):
    if names[i] in ("pd", "o") and detections.confidence[i] >= min_conf:
      x1, y1, x2, y2 = detections.xyxy[i]
      evidence.append((x1 + x2) / 2.0 if lg["vertical"] else (y1 + y2) / 2.0)
  if image_bgr is not None:
    for c0, c1, *_ in find_damaged_lines(image_bgr, lg["vertical"]):
      c = (c0 + c1) / 2.0
      if all(abs(c - e) > 0.25 * period for e in evidence):
        evidence.append(c)
  if len(evidence) < 2:
    return lg
  votes_gap = sum(1 for c in evidence if cdist(c, ph_gap) < cdist(c, ph_line))
  if votes_gap >= 2 and votes_gap > 0.7 * len(evidence):
    swap = {LINE: GAP, GAP: LINE}
    lg["stripes"] = [(a, b, swap.get(k, k)) for a, b, k in stripes]
    cl = lg["col_label"].copy()
    cl[lg["col_label"] == LINE] = GAP
    cl[lg["col_label"] == GAP] = LINE
    lg["col_label"] = cl
    RULE_LOG.append(f"line/gap 뒤바꿈 — 망가진 line·신뢰도 높은 PD/O {len(evidence)}곳 중 {votes_gap}곳이 "
                    f"gap 쪽 주기에 맞음 → line 색 판별이 뒤집힌 것으로 보고 바로잡음")
  return lg


# line 한쪽 끝(맨 위 또는 맨 아래)만 진하고 나머지가 정상 → 그 부분만 P
# 양쪽 끝이 다 진하거나 line 가운데까지 이상 → PD
END_DARK_DEPTH = 35.0     # 진한 행: 밝은 배경(초록 75백분위)보다 이만큼 어두움
END_ABNORMAL_DEPTH = 15.0 # 이상한 행: 이만큼 어두움 (정상 line은 배경과 거의 같은 밝기)
END_REST_MAX = 0.3        # 진한 끝부분을 뺀 나머지 중 이상한 행 비율 상한
END_REST_LEVEL = 45.0     # 나머지 부분 밝기가 배경보다 이 이상 어두우면 line 전체가 망가진 것(PD)


def _g_plane(image_bgr, vertical):
  g = image_bgr[:, :, 1].astype(np.float32)
  return g if vertical else g.T


def line_end_profile(image_bgr, vertical, c0, c1):
  """c0~c1(줄무늬 가로 방향) line 속의 행별 밝기로 끝부분 진함 판정.
  반환: (판정, (진한 구간 시작, 끝), 설명)
    'p'       한쪽 끝만 진하고 나머지 정상 → 그 부분만 P
    'pd'      위·아래 끝이 모두 진함 → PD
    'pd_rest' 한쪽 끝이 진하고 나머지도 이상 → (모델 PD면 PD 유지, 규칙으로 새로 추가하지는 않음)
    None      끝부분이 진하지 않음"""
  g = _g_plane(image_bgr, vertical)
  n_rows, n_cols = g.shape
  bg = float(np.percentile(_smooth_1d(np.median(g, axis=0), 2.0), 75))
  w = c1 - c0
  lo, hi = int(c0 + 0.25 * w), int(c1 - 0.25 * w)
  if hi <= lo:
    lo, hi = int(c0), int(c1) + 1
  lo, hi = max(0, lo), min(n_cols, max(hi, lo + 1))
  rowv = np.median(g[:, lo:hi], axis=1)
  k = max(3, n_rows // 100) | 1
  rowv = np.convolve(rowv, np.ones(k) / k, mode="same")
  # 기준 = 이 line 자신의 정상 밝기(행 밝기 75백분위). 노란 line은 배경보다 원래 조금 어두울 수 있음
  ref = min(bg, float(np.percentile(rowv, 75)))
  dark = rowv < ref - END_DARK_DEPTH
  abnormal = rowv < ref - END_ABNORMAL_DEPTH
  edge_n = max(5, int(0.05 * n_rows))

  def run_from(start, step):
    if dark[start::step][:edge_n].mean() < 0.6:
      return 0
    idx, miss, length = start, 0, 0
    while 0 <= idx < n_rows:
      if dark[idx]:
        miss = 0
      else:
        miss += 1
        if miss > max(5, n_rows // 50):
          break
      length += 1
      idx += step
    return max(0, length - miss)

  top = run_from(0, 1)
  bot = run_from(n_rows - 1, -1)
  if top and bot:
    return "pd", (0, n_rows - 1), f"line 위·아래 끝이 모두 진함 (위 {top}px, 아래 {bot}px)"
  if not top and not bot:
    level = float(np.median(rowv))
    if float(abnormal.mean()) < 0.1 and level >= bg - END_REST_LEVEL:
      return "normal", None, f"line 전체가 정상 (이상 행 {abnormal.mean()*100:.0f}%, 밝기 {level:.0f} / 배경 {bg:.0f})"
    return None, None, "끝부분 진하지 않음"
  seg = (0, top - 1) if top else (n_rows - bot, n_rows - 1)
  rest = np.ones(n_rows, dtype=bool)
  rest[seg[0] : seg[1] + 1] = False
  bad = float(abnormal[rest].mean()) if rest.any() else 1.0
  where = "위" if top else "아래"
  rest_level = float(np.median(rowv[rest])) if rest.any() else 0.0
  rest_ok = rest_level >= bg - END_REST_LEVEL  # 나머지가 정상 line 밝기 (망가진 line은 훨씬 어두움)
  if bad <= END_REST_MAX and rest_ok and (seg[1] - seg[0] + 1) <= 0.5 * n_rows:
    return "p", seg, f"line {where}쪽 끝만 진함 ({seg[1]-seg[0]+1}px), 나머지 정상 (이상 {bad*100:.0f}%)"
  return "pd_rest", (0, n_rows - 1), (f"line {where}쪽 끝이 진하고 나머지도 이상 "
                                 f"(이상 행 {bad*100:.0f}%, 나머지 밝기 {rest_level:.0f} / 배경 {bg:.0f})")


def refine_pd_by_line_ends(detections, lg, image_bgr):
  """PD로 잡힌 line이 한쪽 끝만 진하고 나머지는 정상이면 → 진한 끝부분만 P로 바꾼다."""
  n = len(detections)
  if n == 0:
    return detections
  names = list(detections.data["class_name"])
  if "pd" not in names:
    return detections
  vertical = lg["vertical"] if lg is not None else True
  h, w = image_bgr.shape[:2]
  xyxy = detections.xyxy.copy()
  new_names = list(names)
  mask = detections.mask.copy() if detections.mask is not None else None
  for i in range(n):
    if names[i] != "pd":
      continue
    x1, y1, x2, y2 = xyxy[i]
    c0, c1 = (x1, x2) if vertical else (y1, y2)
    # 박스가 line에서 조금 비껴 잡힐 수 있으므로 가장 많이 겹치는 line 줄무늬 기준으로 본다
    pc0, pc1 = c0, c1
    if lg is not None:
      best, best_ov = None, 0.0
      for a, b, k in lg["stripes"]:
        ov = min(b, c1) - max(a, c0)
        if k == LINE and ov > best_ov and (b - a) <= 2.5 * (c1 - c0):
          best, best_ov = (a, b), ov
      if best is not None and best_ov >= 0.3 * (c1 - c0):
        pc0, pc1 = best
    verdict, seg, why = line_end_profile(image_bgr, vertical, pc0, pc1)
    if verdict == "normal" and detections.confidence[i] < 0.5 and detections.data.get("source", ["model"] * n)[i] == "model":
      new_names[i] = "__drop__"
      RULE_LOG.append(f"PD {detections.confidence[i]*100:.0f}% (x {int(x1)}~{int(x2)}): 제거 — {why}")
      continue
    if verdict != "p":
      continue
    new_names[i] = "p"
    xyxy[i] = [c0, seg[0], c1, seg[1]] if vertical else [seg[0], c0, seg[1], c1]
    if mask is not None:
      band = np.zeros((h, w), dtype=bool)
      if vertical:
        band[seg[0] : seg[1] + 1, :] = True
      else:
        band[:, seg[0] : seg[1] + 1] = True
      m = mask[i] & band
      if not m.any():
        m = np.zeros((h, w), dtype=bool)
        bx1, by1, bx2, by2 = map(int, xyxy[i])
        m[by1 : by2 + 1, bx1 : bx2 + 1] = True
      mask[i] = m
    RULE_LOG.append(f"PD {detections.confidence[i]*100:.0f}% (x {int(x1)}~{int(x2)}): P로 변경 — {why}")
  detections.xyxy = xyxy.astype(np.float32)
  if mask is not None:
    detections.mask = mask
  keep = np.array([i for i in range(n) if new_names[i] != "__drop__"], dtype=int)
  new_names = [nm if nm != "__drop__" else "pd" for nm in new_names]
  detections.data["class_name"] = np.asarray(new_names, dtype=str)
  detections.class_id = np.array([CLASS_IDS.get(nm, UNKNOWN_ID) for nm in new_names], dtype=int)
  if "position" in detections.data:
    pos = list(detections.data["position"])
    for i in range(n):
      if names[i] == "pd" and new_names[i] == "p":
        pos[i] = "line 끝 진함"
    detections.data["position"] = np.asarray(pos, dtype=str)
  if len(keep) < n:
    return detections[keep] if len(keep) else _empty_detections()
  return detections


def _add_rule_particles(detections, items, image_shape):
  h, w = image_shape[:2]
  n = len(items)
  for box, why in items:
    txt = f"x {box[0]}~{box[2]}"
    RULE_LOG.append(f"P (규칙) ({txt}): 추가 — {why}")
  mask = None
  if detections.mask is not None or len(detections) == 0:
    mask = np.zeros((n, h, w), dtype=bool)
    for k, ((x1, y1, x2, y2), _) in enumerate(items):
      mask[k, int(y1) : int(y2) + 1, int(x1) : int(x2) + 1] = True
  new = sv.Detections(
      xyxy=np.array([b for b, _ in items], dtype=np.float32),
      confidence=np.zeros(n, dtype=np.float32),
      class_id=np.full(n, CLASS_IDS["p"], dtype=int),
      mask=mask,
      data={"class_name": np.array(["p"] * n), "position": np.array(["규칙: line 끝 진함"] * n),
            "source": np.array(["rule"] * n)},
  )
  return _concat_detections(detections, new)


def find_end_dark_lines(image_bgr, vertical=True):
  """위쪽/아래쪽 끝부분에서만 속이 어두운 line 후보 (전체가 어두운 line은 find_damaged_lines가 담당).
  반환 [(c0, c1)]"""
  g = _g_plane(image_bgr, vertical)
  n_rows = g.shape[0]
  win = max(10, int(0.12 * n_rows))
  bg = float(np.percentile(_smooth_1d(np.median(g, axis=0), 2.0), 75))
  out = []
  for part in (g[:win], g[n_rows - win :]):
    s = _smooth_1d(np.median(part, axis=0), 2.0)
    dark = s < bg - PD_BAND_DEPTH
    x = 0
    while x < len(s):
      if not dark[x]:
        x += 1
        continue
      y = x
      while y < len(s) and dark[y]:
        y += 1
      seg = s[x:y]
      mins = [i for i in range(1, len(seg) - 1) if seg[i] <= seg[i - 1] and seg[i] <= seg[i + 1]]
      double = any(b - a >= 6 and seg[a : b + 1].max() - max(seg[a], seg[b]) >= PD_BAND_BUMP
                   for ii, a in enumerate(mins) for b in mins[ii + 1 :])
      if double:
        out.append((x, y - 1))
      x = y
  # 위·아래에서 같은 line이 나오면 하나로
  out.sort()
  merged = []
  for c0, c1 in out:
    if merged and c0 <= merged[-1][1]:
      merged[-1] = (merged[-1][0], max(merged[-1][1], c1))
    else:
      merged.append((c0, c1))
  return merged


def add_rule_pds(detections, lg, image_bgr):
  """모델이 놓친 PD: 속까지 어두워진(망가진) line을 이미지에서 찾아 PD로 추가."""
  h, w = image_bgr.shape[:2]
  vertical = lg["vertical"] if lg is not None else (
      np.abs(np.diff(image_bgr[:, :, 1].astype(np.float32), axis=1)).mean()
      >= np.abs(np.diff(image_bgr[:, :, 1].astype(np.float32), axis=0)).mean())
  cands = find_damaged_lines(image_bgr, vertical)
  names = list(detections.data.get("class_name", []))
  pd_boxes = [detections.xyxy[i] for i in range(len(detections)) if names[i] == "pd"]
  all_boxes = list(detections.xyxy)
  # 끝부분만 진한 line: 한쪽 끝 → P, 양쪽 끝 → PD
  n_rows = image_bgr.shape[0] if vertical else image_bgr.shape[1]
  end_p = []
  full = [(c0, c1) for c0, c1, *_ in cands]
  for c0, c1 in find_end_dark_lines(image_bgr, vertical):
    if any(c0 <= f1 and c1 >= f0 for f0, f1 in full):
      continue
    verdict, seg, why = line_end_profile(image_bgr, vertical, c0, c1)
    if verdict == "pd":  # 위·아래 끝이 모두 진함
      box = (c0, 0, c1, n_rows - 1) if vertical else (0, c0, n_rows - 1, c1)
      if not _covered_by(box, all_boxes, 0.3):
        cands.append((c0, c1, 0, n_rows - 1, why))
    elif verdict == "p":
      box = (c0, seg[0], c1, seg[1]) if vertical else (seg[0], c0, seg[1], c1)
      if _covered_by(box, all_boxes, 0.3):
        continue
      end_p.append((box, why))
  if end_p:
    detections = _add_rule_particles(detections, end_p, image_bgr.shape)
    names = list(detections.data.get("class_name", []))
  if not cands:
    return detections
  new_boxes = []
  for c0, c1, r0, r1, why in cands:
    box = (c0, r0, c1, r1) if vertical else (r0, c0, r1, c1)
    txt = f"x {box[0]}~{box[2]}" if vertical else f"y {box[1]}~{box[3]}"
    if _covered_by(box, pd_boxes, 0.3):
      RULE_LOG.append(f"규칙 PD 후보 ({txt}): 생략 — 이미 PD로 잡힌 line")
      continue
    RULE_LOG.append(f"PD (규칙) ({txt}): 추가 — {why}")
    new_boxes.append(box)
  if not new_boxes:
    return detections
  n = len(new_boxes)
  mask = None
  if detections.mask is not None or len(detections) == 0:
    mask = np.zeros((n, h, w), dtype=bool)
    for k, (x1, y1, x2, y2) in enumerate(new_boxes):
      mask[k, int(y1) : int(y2) + 1, int(x1) : int(x2) + 1] = True
  new = sv.Detections(
      xyxy=np.array(new_boxes, dtype=np.float32),
      confidence=np.zeros(n, dtype=np.float32),
      class_id=np.full(n, CLASS_IDS["pd"], dtype=int),
      mask=mask,
      data={
          "class_name": np.array(["pd"] * n),
          "position": np.array(["규칙: 망가진 line"] * n),
          "source": np.array(["rule"] * n),
      },
  )
  return _concat_detections(detections, new)


def add_particles_from_s(detections, lg, image_bgr):
  """이물이 패턴 위를 덮어 line을 연결한 경우: 이물 자체는 P(작은 박스),
  이물 때문에 연결된 line들까지 포함한 영역은 S(큰 박스)로 함께 표시.
  - P는 S 영역 안에서 line·gap 어느 색과도 다른 '이물' 픽셀이 line 위에 충분히 있을 때만 추가
    (line 위 이물 ≥ 2 × line폭², 그리고 gap 위 이물의 절반 이상 = 패턴 위를 고르게 덮음).
    gap 안에만 있는 찌꺼기(short 잔여물)는 P를 만들지 않는다.
  - 이미 그 S와 겹치는 P가 있으면 추가하지 않는다.
  - P 박스 = 이물 픽셀의 범위, S 박스 = 연결된 line 바깥쪽 끝까지 넓힘."""
  n = len(detections)
  if n == 0 or lg is None:
    return detections
  names = list(detections.data["class_name"])
  if "s" not in names:
    return detections
  vertical = lg["vertical"]
  img = (image_bgr if vertical else image_bgr.transpose(1, 0, 2)).astype(np.float32)
  col_label = lg["col_label"]

  def interior(a, b, f):
    c = max(2, int(f * (b - a)))
    return a + c, b - c

  def ref(kind):
    parts = [img[:, lo : hi + 1].reshape(-1, 3) for a, b, k in lg["stripes"]
             if k == kind for lo, hi in [interior(a, b, 0.25)] if hi > lo]
    return np.median(np.concatenate(parts), axis=0) if parts else None

  lref, gref = ref(LINE), ref(GAP)
  if lref is None or gref is None:
    return detections
  dist = float(np.linalg.norm(lref - gref))
  if dist < 10:
    return detections
  line_ws = [b - a for a, b, k in lg["stripes"] if k == LINE]
  w_line = float(np.median(line_ws))
  p_boxes = [detections.xyxy[i] for i in range(n) if names[i] == "p"]

  xyxy = detections.xyxy.copy()
  new_boxes, logs = [], []
  for i in range(n):
    if names[i] != "s":
      continue
    box = xyxy[i]
    if any(_overlaps(box, pb) for pb in p_boxes):
      continue  # 이미 P가 함께 검출됨
    # 줄무늬 기준 좌표(가로지르는 방향 c, 따라가는 방향 t)로 변환
    x1, y1, x2, y2 = map(int, box)
    c1, t1, c2, t2 = (x1, y1, x2, y2) if vertical else (y1, x1, y2, x2)
    c1, c2 = max(0, c1), min(img.shape[1], c2)
    t1, t2 = max(0, t1), min(img.shape[0], t2)
    if c2 - c1 < 3 or t2 - t1 < 3:
      continue
    reg = img[t1:t2, c1:c2]
    region = np.ones(reg.shape[:2], dtype=bool)
    if detections.mask is not None and detections.mask[i].any():
      m = detections.mask[i] if vertical else detections.mask[i].T
      region = m[t1:t2, c1:c2]
    foreign = ((np.linalg.norm(reg - lref, axis=2) > 0.6 * dist)
               & (np.linalg.norm(reg - gref, axis=2) > 0.6 * dist) & region)
    foreign = cv2.morphologyEx(foreign.astype(np.uint8), cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8)).astype(bool)
    lab = col_label[c1:c2]
    on_line = int(foreign[:, lab == LINE].sum())
    on_gap = int(foreign[:, lab == GAP].sum())
    on_edge = int(foreign[:, lab == EDGE].sum())
    total_f = max(1, on_line + on_gap + on_edge)
    old_ok = on_line >= 2 * w_line**2 and on_line >= 0.5 * on_gap
    # P 규칙과 같은 기준: 이물 면적 1/3 이상이 line(테두리 포함) 위 + 충분히 진함 + 크기 line 폭 절반 이상
    if not old_ok and foreign.any():
      fts, fcs = np.nonzero(foreign)
      size = max(fts.max() - fts.min(), fcs.max() - fcs.min())
      gray0 = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
      gray_i = gray0 if vertical else gray0.T
      dk_f = region_darkness(t1 + fts, c1 + fcs, gray_i, _line_gray(gray0, lg))
      old_ok = ((on_line + on_edge) / total_f >= P_MIN_ON_LINE and on_line >= 0.5 * w_line**2
                and size >= P_MIN_SIZE_RATIO * w_line and dk_f is not None and dk_f >= P_MIN_DARKNESS)
    if not old_ok:
      continue
    ts, cs = np.nonzero(foreign)
    pc1, pc2 = c1 + cs.min(), c1 + cs.max()
    pt1, pt2 = t1 + ts.min(), t1 + ts.max()
    # S: 이물이 막은 gap의 양옆 line(= 실제로 short된 line)과
    #    이물이 걸친 line까지 포함하도록 넓힘
    sc1, sc2 = c1, c2
    st = lg["stripes"]
    for j, (a, b, k) in enumerate(st):
      if not (a < c2 and b > c1):
        continue
      if k == LINE:
        sc1, sc2 = min(sc1, a), max(sc2, b)
        continue
      tol = 0.2 * (b - a)
      closed = (gap_closed_between(lg, (a, b), t1, t2)
                or (c1 <= a + tol and c2 >= b - tol))
      if not closed:
        continue
      for jj in (j - 1, j + 1):
        if 0 <= jj < len(st) and st[jj][2] == LINE:
          sc1, sc2 = min(sc1, st[jj][0]), max(sc2, st[jj][1])
    pbox = (pc1, pt1, pc2, pt2) if vertical else (pt1, pc1, pt2, pc2)
    sbox = (sc1, t1, sc2, t2) if vertical else (t1, sc1, t2, sc2)
    xyxy[i] = sbox
    new_boxes.append((pbox, float(detections.confidence[i])))
    logs.append(
        f"S {detections.confidence[i]*100:.0f}% 안에서 이물 확인 → P 추가 "
        f"(line 위 이물 {on_line}px / gap 위 {on_gap}px), S는 연결된 line까지 넓힘")
  if not new_boxes:
    return detections
  detections.xyxy = xyxy.astype(np.float32)
  RULE_LOG.extend(logs)
  k = len(new_boxes)
  h, w = image_bgr.shape[:2]
  mask = None
  if detections.mask is not None:
    mask = np.zeros((k, h, w), dtype=bool)
    for j, ((bx1, by1, bx2, by2), _) in enumerate(new_boxes):
      sub = np.zeros((h, w), dtype=bool)
      sub[int(by1) : int(by2) + 1, int(bx1) : int(bx2) + 1] = True
      # 이물 픽셀만 마스크로 (원래 S 마스크 ∩ 박스)
      src = detections.mask[[ii for ii in range(n) if names[ii] == "s"]].any(axis=0)
      mask[j] = sub & src if (sub & src).any() else sub
  new = sv.Detections(
      xyxy=np.array([b for b, _ in new_boxes], dtype=np.float32),
      confidence=np.array([c for _, c in new_boxes], dtype=np.float32),
      class_id=np.full(k, CLASS_IDS["p"], dtype=int),
      mask=mask,
      data={
          "class_name": np.array(["p"] * k),
          "position": np.array(["S 안 이물"] * k),
          "source": np.array(["from_s"] * k),
      },
  )
  return _concat_detections(detections, new)


def draw_line_overlay(image_bgr, lg):
  """line 영역을 옅은 초록으로 칠해서 판별 결과를 눈으로 확인할 수 있게 함."""
  if lg is None:
    return image_bgr
  overlay = image_bgr.copy()
  h, w = image_bgr.shape[:2]
  for a, b, kind in lg["stripes"]:
    if kind != LINE:
      continue
    if lg["vertical"]:
      cv2.rectangle(overlay, (int(a), 0), (int(b), h - 1), (80, 200, 60), -1)
    else:
      cv2.rectangle(overlay, (0, int(a)), (w - 1, int(b)), (80, 200, 60), -1)
  return cv2.addWeighted(overlay, 0.25, image_bgr, 0.75, 0)


# ---------------------------------------------------------------------------
# 자동 수평 보정
#   줄무늬가 정확히 세로(또는 가로)일 때 열(행) 평균 밝기의 대비가 가장 커진다.
#   축소 이미지를 여러 각도로 돌려 보며 그 대비가 최대인 각도를 찾고(거친→세밀),
#   원본을 그 각도로 회전한 뒤 회전으로 생긴 빈 모서리가 없도록 안쪽만 잘라낸다.
# ---------------------------------------------------------------------------
MAX_SKEW_DEG = 15.0
MIN_SKEW_DEG = 0.15  # 이보다 작으면 회전하지 않음


def _stripe_contrast(small, angle):
  h, w = small.shape
  M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
  rot = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_LINEAR)
  valid = cv2.warpAffine(np.ones_like(small), M, (w, h), flags=cv2.INTER_NEAREST)
  scores = []
  for axis in (0, 1):  # 0: 세로 줄무늬(열 평균), 1: 가로 줄무늬(행 평균)
    cnt = valid.sum(axis=axis)
    ok = cnt > 0.5 * valid.shape[axis]
    if ok.sum() < 10:
      scores.append(0.0)
      continue
    mean = rot.sum(axis=axis)[ok] / cnt[ok]
    scores.append(float(mean.var()))
  return max(scores)


def estimate_skew(image_bgr, max_angle=MAX_SKEW_DEG):
  gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
  h, w = gray.shape
  s = min(1.0, 600.0 / max(h, w))
  small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))),
                     interpolation=cv2.INTER_AREA).astype(np.float32)
  base = _stripe_contrast(small, 0.0)
  best, best_score = 0.0, base
  for step, span in ((1.0, max_angle), (0.1, 1.0), (0.02, 0.1)):
    center = best
    for a in np.arange(center - span, center + span + 1e-9, step):
      if abs(a) > max_angle:
        continue
      sc = _stripe_contrast(small, float(a))
      if sc > best_score:
        best, best_score = float(a), sc
  # 줄무늬가 거의 없는 사진 등에서 의미 없는 회전 방지
  if base > 0 and best_score < base * 1.02:
    return 0.0
  return round(best, 2)


def _max_inner_rect(w, h, angle_deg):
  """w×h 이미지를 angle만큼 돌렸을 때 빈 영역 없이 잘라낼 수 있는 최대 직사각형."""
  a = np.radians(abs(angle_deg))
  sin_a, cos_a = np.sin(a), np.cos(a)
  long_side, short_side = (w, h) if w >= h else (h, w)
  if short_side <= 2.0 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
    x = 0.5 * short_side
    wr, hr = (x / sin_a, x / cos_a) if w >= h else (x / cos_a, x / sin_a)
  else:
    cos_2a = cos_a * cos_a - sin_a * sin_a
    wr = (w * cos_a - h * sin_a) / cos_2a
    hr = (h * cos_a - w * sin_a) / cos_2a
  return int(wr), int(hr)


def deskew_image(image_bgr):
  """기울기를 찾아 바로 세운 이미지와 회전 각도(도)를 반환."""
  angle = estimate_skew(image_bgr)
  if abs(angle) < MIN_SKEW_DEG:
    return image_bgr, 0.0
  h, w = image_bgr.shape[:2]
  M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
  rot = cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REPLICATE)
  cw, ch = _max_inner_rect(w, h, angle)
  cw, ch = max(1, cw - 2), max(1, ch - 2)
  x0, y0 = (w - cw) // 2, (h - ch) // 2
  return rot[y0 : y0 + ch, x0 : x0 + cw].copy(), angle


# 흐린 패턴 → 이미지 전체를 PD 1건으로.
#   줄무늬 방향으로 평균 낸 밝기(gray) 프로파일의 진폭(5~95 백분위)으로 판단.
#   선명한 사진은 line·gap·빨간 테두리가 또렷해 진폭이 크다 (실측 55~131).
#   흐린 사진은 모두 같은 주황/노랑으로 뭉개져 진폭이 작다 (실측 17~34).
BLUR_GRAY_RANGE = 44.0


def pattern_gray_range(image_bgr):
  """줄무늬 프로파일 밝기 진폭. 세로/가로 줄무늬 중 큰 쪽. 작을수록 흐림."""
  gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
  out = 0.0
  for axis in (0, 1):
    prof = np.median(gray, axis=axis)
    if prof.size >= 8:
      p5, p95 = np.percentile(prof, [5, 95])
      out = max(out, float(p95 - p5))
  return out


def is_blurry_pattern(image_bgr):
  score = pattern_gray_range(image_bgr)
  return score < BLUR_GRAY_RANGE, score


def blur_pd_detections(image_bgr, score):
  """이미지 전체를 덮는 PD 1건 (source='blur', 마스크 없음)."""
  h, w = image_bgr.shape[:2]
  conf = round(min(0.99, max(0.60, 1.0 - score / (BLUR_GRAY_RANGE * 1.5))), 2)
  return sv.Detections(
      xyxy=np.array([[1, 1, w - 2, h - 2]], dtype=np.float32),
      confidence=np.array([conf], dtype=np.float32),
      class_id=np.array([CLASS_IDS["pd"]], dtype=int),
      data={"class_name": np.array(["pd"]), "source": np.array(["blur"]),
            "position": np.array(["흐린 패턴"])},
  )


def _angle_text(angle):
  if abs(angle) < MIN_SKEW_DEG:
    return "보정 없음"
  # OpenCV에서 양수 각도 = 반시계 방향 회전
  return f"{'반시계' if angle > 0 else '시계'} 방향 {abs(angle):.2f}° 회전"


def _draw_blur_pd(image_bgr, angle, blur_score, show_corners=True):
  """흐린 패턴: 모델·규칙 없이 전체 PD 1건만 표시."""
  RULE_LOG.clear()
  msg = (f"흐린 패턴 (줄무늬 밝기 진폭 {blur_score:.0f} < {BLUR_GRAY_RANGE:.0f})"
         " → 이미지 전체를 PD로 판정")
  RULE_LOG.append(f"PD (전체): {msg}")
  dets = blur_pd_detections(image_bgr, blur_score)
  annotated, dets, info = _apply_rules_and_draw(
      image_bgr, dets, angle, show_corners, False, False, False, False, False, False, False)
  info["rule_msg"] = msg
  info["log"] = [f"PD (전체): {msg}"]
  return annotated, dets, info


def _prepare_model_detections(detections):
  """모델 결과의 클래스 이름을 o/p/pd/s/sc로 통일하고 표시용 데이터를 채움."""
  # 클래스 이름 확보 (없으면 class_id로부터 복원)
  raw_names = detections.data.get("class_name")
  if raw_names is None or len(raw_names) != len(detections):
    rev_ids = {i: code for code, i in CLASS_IDS.items()}
    raw_names = (
        [rev_ids.get(int(cid), str(cid)) for cid in detections.class_id]
        if detections.class_id is not None
        else ["unknown"] * len(detections)
    )

  # 이름을 o/p/pd/s/sc로 통일하고, 색상이 클래스별로 고정되도록 class_id를 재지정
  norm_names = [_normalize_class_name(n) for n in raw_names]
  detections.data["class_name"] = np.asarray(norm_names, dtype=str)
  detections.class_id = np.array(
      [CLASS_IDS.get(n, UNKNOWN_ID) for n in norm_names], dtype=int
  )
  if detections.confidence is None:
    detections.confidence = np.ones(len(detections), dtype=np.float32)
  detections.data["source"] = np.array(["model"] * len(detections))
  return detections


# 같은 결함에 여러 박스가 겹쳐 나오는 경우 정리 (process_single_bgr에서 사용).
  # RF-DETR는 NMS를 쓰지 않는 구조라 같은 클래스 박스가 겹쳐 나올 수 있다.
  #  - NMS_PER_CLASS : 같은 클래스끼리만 정리 → 이물(P)과 그로 인한 단락(S)처럼
  #                    한 자리에 서로 다른 결함이 함께 있으면 둘 다 남는다.
  #  - NMS_AGNOSTIC  : 클래스와 관계없이 겹치는 것 중 가장 신뢰도 높은 1개만 남김
  #  - NMS_OFF       : 정리하지 않음 (모델 원본 그대로)
def _apply_rules_and_draw(image_bgr, detections, angle, show_corners, show_masks,
                          line_rule, show_line_overlay, scratch_rule, find_shorts,
                          split_particle=True, protect_conf=True):
  # line/gap 판별 후 P·O·S 위치 규칙 적용
  n_model = len(detections)
  RULE_LOG.clear()
  lg = None
  if line_rule or show_line_overlay:
    lg = analyze_line_gap(image_bgr, exclude_boxes=detections.xyxy)
    if line_rule and lg is not None:
      lg = flip_lines_by_defects(lg, detections, image_bgr)
  if not line_rule:
    rule_msg = "line/gap 규칙 꺼짐 (체크박스 해제됨)"
  elif lg is None:
    rule_msg = "⚠ line/gap 판별 실패 (줄무늬를 찾지 못함) → line/gap 규칙 미적용"
  else:
    n_line = sum(1 for *_, k in lg["stripes"] if k == LINE)
    rule_msg = (
        f"line/gap 판별 성공 ({'세로' if lg['vertical'] else '가로'} 줄무늬 "
        f"{len(lg['stripes'])}개 중 line {n_line}개)"
    )
  if lg is not None:
    lg["gap_open"] = compute_gap_openness(image_bgr, lg)
  if line_rule:
    detections = fix_line_shaped_shorts(detections, lg, image_bgr.shape)
    detections = apply_line_gap_rule(detections, lg, image_bgr, protect_conf)
    if find_shorts and lg is not None:
      detections = add_rule_shorts(detections, lg, image_bgr.shape, image_bgr)
    if find_shorts:
      detections = add_rule_pds(detections, lg, image_bgr)
  elif "position" not in detections.data:
    detections.data["position"] = np.array(["-"] * len(detections), dtype=str)
  # 줄무늬와 나란한 세로 직선 SC → PD, 그 다음 SC와 같은 자리의 S 제거
  if scratch_rule:
    detections = fix_straight_scratches(detections, lg, image_bgr)
    detections = apply_scratch_rule(detections)
  if line_rule:
    detections = refine_pd_by_line_ends(detections, lg, image_bgr)
    detections = merge_overlapping_shorts(detections)
    detections = drop_duplicate_particles(detections)
  if line_rule and split_particle:
    detections = add_particles_from_s(detections, lg, image_bgr)
  if line_rule:
    detections = drop_particles_on_pd(detections)

  rule_msg += f" · 모델 검출 {n_model}건 → 규칙 적용 후 {len(detections)}건"
  print(f"[규칙] {rule_msg}")
  for line_ in RULE_LOG:
    print(f"   - {line_}")
  stripe_txt = (
      " ".join(f"{a}~{b}:{'L' if k == LINE else 'G'}" for a, b, k in lg["stripes"])
      if lg is not None else "-"
  )
  info = {"angle": angle, "rule_msg": rule_msg, "log": list(RULE_LOG),
          "stripes": stripe_txt}

  annotated_bgr = image_bgr.copy()
  if show_line_overlay:
    annotated_bgr = draw_line_overlay(annotated_bgr, lg)

  if len(detections) == 0:
    return annotated_bgr, _empty_detections(), info

  # 신뢰도 높은 순으로 번호(#1, #2 ...)가 매겨지도록 정렬
  order = np.argsort(-detections.confidence)
  detections = detections[order]

  h, w = image_bgr.shape[:2]
  scale_factor = max(w, h) / 1000.0
  dynamic_thickness = max(2, int(round(2.0 * scale_factor)))

  # 세그멘테이션 마스크 (모델이 낸 마스크를 그대로 사용)
  if show_masks and detections.mask is not None:
    mask_annotator = sv.MaskAnnotator(color=CLASS_PALETTE, opacity=0.35)
    annotated_bgr = mask_annotator.annotate(
        scene=annotated_bgr, detections=detections
    )

  # 화면 표시용 박스: P(이물)와 S가 거의 같은 크기면 S 테두리를 바깥으로 살짝 띄워 그려
  # 둘 다 보이게 한다 (표의 좌표는 실제 값 그대로).
  draw_xyxy = detections.xyxy.copy()
  names_d = list(detections.data["class_name"])
  pad = max(6, int(round(0.012 * max(w, h))))
  for si in [k for k in range(len(detections)) if names_d[k] == "s"]:
    for pi in [k for k in range(len(detections)) if names_d[k] == "p"]:
      a, b = draw_xyxy[si], detections.xyxy[pi]
      iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
      ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
      union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
               - iw * ih)
      if union > 0 and iw * ih / union > 0.7:
        draw_xyxy[si] = [max(0, a[0] - pad), max(0, a[1] - pad),
                         min(w - 1, a[2] + pad), min(h - 1, a[3] + pad)]
  draw_det = sv.Detections(xyxy=draw_xyxy, class_id=detections.class_id,
                           confidence=detections.confidence)

  # 바운딩 박스 (큰 박스부터 그려서 안쪽 작은 박스가 가려지지 않게)
  areas = (draw_xyxy[:, 2] - draw_xyxy[:, 0]) * (draw_xyxy[:, 3] - draw_xyxy[:, 1])
  draw_order = np.argsort(-areas)
  for box, cid in zip(
      draw_xyxy[draw_order], detections.class_id[draw_order]
  ):
    bx1, by1, bx2, by2 = map(int, box)
    cv2.rectangle(
        annotated_bgr,
        (bx1, by1),
        (bx2, by2),
        _class_bgr(int(cid)),
        dynamic_thickness,
    )

  # 모서리 강조
  if show_corners:
    corner_annotator = sv.BoxCornerAnnotator(
        color=CLASS_PALETTE,
        thickness=dynamic_thickness + 1,
        corner_length=max(14, int(14 * scale_factor)),
    )
    annotated_bgr = corner_annotator.annotate(
        scene=annotated_bgr, detections=draw_det
    )

  # 라벨 (박스 위쪽 → 아래쪽 → 안쪽 순으로 시도하고, 다른 라벨과 겹치면 비켜서 배치)
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.7 * max(1.0, scale_factor)
  font_thickness = max(2, int(round(2 * scale_factor)))
  padding = 6
  placed_labels = []

  sources = detections.data.get("source", ["model"] * len(detections))
  for i, (cname, conf, box, cid) in enumerate(
      zip(
          detections.data["class_name"],
          detections.confidence,
          draw_xyxy,
          detections.class_id,
      )
  ):
    x1, y1, x2, y2 = map(int, box)
    if sources[i] in ("rule", "from_s", "blur"):
      # 규칙으로 찾은 결함은 신뢰도가 없으므로 번호와 클래스만 표시
      label_text = f"#{i+1} {str(cname).upper()}"
    else:
      label_text = f"#{i+1} {str(cname).upper()} {conf*100:.0f}%"
    (tw, th), baseline = cv2.getTextSize(
        label_text, font, font_scale, font_thickness
    )
    lbl_h = th + baseline + padding * 2
    lbl_w = tw + padding * 2
    lbl_x1 = max(2, min(x1, w - lbl_w - 2))

    def _clamp_y(y):
      return int(min(max(0, y), max(0, h - lbl_h - 1)))

    if str(cname) == "s":
      candidates = [y2 + 4, y1 - lbl_h - 4, y2 - lbl_h - 4]
    else:
      candidates = [y1 - lbl_h - 4, y2 + 4, y1 + 4]
    # 후보 위치에서도 겹치면 한 줄씩 위/아래로 밀어본다
    for step in range(1, 6):
      candidates += [y1 - lbl_h - 4 - step * (lbl_h + 2),
                     y2 + 4 + step * (lbl_h + 2)]
    lbl_y1 = _clamp_y(candidates[0])
    for cy in candidates:
      if cy < 0 or cy + lbl_h > h:
        continue
      rect = (lbl_x1, cy, lbl_x1 + lbl_w, cy + lbl_h)
      if not any(_overlaps(rect, r) for r in placed_labels):
        lbl_y1 = int(cy)
        break
    lbl_y2 = lbl_y1 + lbl_h
    placed_labels.append((lbl_x1, lbl_y1, lbl_x1 + lbl_w, lbl_y2))

    cv2.rectangle(
        annotated_bgr,
        (lbl_x1, lbl_y1),
        (lbl_x1 + lbl_w, lbl_y2),
        _class_bgr(int(cid)),
        -1,
    )
    cv2.putText(
        annotated_bgr,
        label_text,
        (lbl_x1 + padding, lbl_y2 - baseline - padding),
        font,
        font_scale,
        _text_bgr_on(_class_bgr(int(cid))),
        font_thickness,
        cv2.LINE_AA,
    )

  return annotated_bgr, detections, info


def _mask_area(detections, i):
  if detections.mask is None:
    return "-"
  return f"{int(detections.mask[i].sum()):,}"


def run_rules(
    image_bgr,
    raw_detections,
    iou_threshold=0.45,
    show_corners=True,
    show_masks=True,
    nms_mode=NMS_PER_CLASS,
    line_rule=True,
    show_line_overlay=False,
    scratch_rule=True,
    find_shorts=True,
    split_particle=True,
    protect_conf=True,
    angle=0.0,
):
  """모델 원본 detections(sv.Detections)에 r26 후처리를 적용하고 마킹한다.

  image_bgr 는 모델에 넣은 이미지(수평 보정 후)와 같아야 한다.
  반환: (annotated_bgr, detections, info)
    detections.data["class_name"] : o/p/pd/s/sc
    detections.data["source"]     : model / rule(규칙으로 찾은 S) / from_s(S 안의 이물 P)
    detections.data["position"]   : line/gap 판정 결과
    info = {"angle", "rule_msg", "log"(결함별 판정 근거), "stripes"}
  """
  if raw_detections is None or len(raw_detections) == 0:
    detections = _empty_detections()
  else:
    detections = _prepare_model_detections(raw_detections)
  if nms_mode != NMS_OFF and len(detections) > 1:
    detections = detections.with_nms(
        threshold=iou_threshold, class_agnostic=(nms_mode == NMS_AGNOSTIC))
  if len(detections) == 0 and not (line_rule and find_shorts):
    base = image_bgr.copy()
    if show_line_overlay:
      base = draw_line_overlay(base, analyze_line_gap(image_bgr))
    return base, _empty_detections(), {"angle": angle, "rule_msg": "모델 검출 결과 없음",
                                       "log": [], "stripes": "-"}
  return _apply_rules_and_draw(
      image_bgr, detections, angle, show_corners, show_masks, line_rule,
      show_line_overlay, scratch_rule, find_shorts, split_particle, protect_conf)
