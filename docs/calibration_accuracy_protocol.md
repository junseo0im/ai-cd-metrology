# Calibration and CD Accuracy Validation Protocol

## 1. 목적과 적용 범위

이 문서는 frozen Method 2를 변경하지 않고 20× 광학 이미지의 공식 x-axis calibration과 Manual OptiView reference 기반 CD accuracy validation을 수행하기 위한 사전 고정 프로토콜이다. 같은 evaluation/report contract를 Method 1(Intensity/Threshold), Method 2(Gradient/Edge), Method 3(U-Net)에 공통 적용한다.

이 단계에서는 calibration 값, manual 측정값, 자동 측정값을 생성하지 않는다. Dashboard, defect algorithm, wafer map, process root-cause AI, What-if Simulator도 구현하지 않는다.

## 2. 20× calibration ruler 촬영 요구사항

Calibration ruler는 현재 CD 이미지와 같은 pixel scale을 입증할 수 있도록 다음 조건을 기록하고 확인한다.

- objective/magnification: 20×
- microscope, camera, OptiView acquisition/export 경로
- camera sensor mode, binning, digital zoom, capture resolution 및 export resolution
- ruler 식별자와 traceable known interval, 표시 단위
- 촬영 일시와 operator
- 원본 파일 경로 및 원본 width/height
- acquisition 이후 crop/resize 여부

Ruler의 눈금 방향은 CD measurement axis인 image x-axis와 대응해야 한다. Known interval의 두 endpoint가 x좌표로 명확하지 않거나 ruler가 기울어져 x축 투영 거리를 확정할 수 없으면 scale을 산출하지 않는다. 원본 ruler 이미지는 수정하지 않는다.

## 3. Known interval과 µm-per-pixel 계산 원칙

Known physical distance와 두 endpoint가 명확한 interval만 사용한다. 단위는 먼저 µm로 정규화하고 원본 이미지 좌표를 사용한다.

```text
pixel_distance_x = abs(x2 - x1)
scale_x_um_per_px = known_distance_um / pixel_distance_x
```

각 interval에 대해 ruler ID, known distance와 단위, x1, x2, pixel distance, 계산 scale을 함께 보존한다. 같은 ruler에서 독립적으로 읽을 수 있는 interval이 여러 개면 interval별 scale을 먼저 계산하고 일관성을 검토한다. 최종값은 유효 interval들의 median을 기본 대표값으로 사용할 수 있지만, interval 간 불일치가 보이면 임의 평균으로 숨기지 않고 원인과 acquisition metadata를 먼저 확인한다. 수용 오차는 ruler 사양과 실험 책임자의 기준이 정해지기 전에는 임의로 설정하지 않는다.

### Crop과 resize

- Crop은 pixel spacing을 바꾸지 않으므로 좌표 원점과 image extent만 변한다.
- Resize, resampling, 비율 변경은 pixel spacing을 바꾸므로 원본 scale을 그대로 적용할 수 없다.
- 동일/상이한 image resolution만으로 scale 동일성을 판정하지 않는다.
- Ruler와 wafer 이미지에 resize 여부를 입증할 metadata가 없으면 official calibration 적용 상태를 보류한다.

## 4. Manual OptiView reference 측정 프로토콜

1. `manual_reference_manifest.csv`를 automatic Method 2 결과나 status를 열람하지 않은 상태에서 먼저 고정한다.
2. Operator는 가능하면 automatic CD 값, error 및 method status를 보지 않은 상태에서 측정한다.
3. Manifest가 가리키는 원본 이미지를 OptiView에서 열고 display zoom만 사용한다. 파일 자체를 crop/resize/resample하지 않는다.
4. 각 image의 Method 2 measurement ROI 내부에서 Y10, Y50, Y90 세 scanline을 사용한다. 이는 ROI y-range의 10%, 50%, 90% 위치이며 전체 image 기준 좌표가 아니다.
5. 각 scanline에서 ROI horizontal center에 가장 가까운 완전한 physical E1~E6 triplet을 manual target으로 선택한다. Automatic candidate, result, status 또는 error를 보고 target을 바꾸지 않는다.
6. 선택한 triplet에서 동일한 physical structure의 Outer, Inner, Gap endpoint를 지정한다.
7. OptiView가 표시하는 µm 값을 `manual_measurement_template.csv`의 sample-level 행에 입력한다. 빈 값은 0으로 대체하지 않는다.
8. Complete E1~E6 triplet이 없거나 manual measurement가 불가능하면 값을 추정하지 않고 `measurement_available`과 `unavailable_reason`에 상태를 기록한다.
9. `operator`와 ISO 8601 형식의 `measurement_date`를 기록한다.
10. 입력 후 manifest identity와 원본 파일을 대조하고, automatic result와 결합하기 전에 manual table을 보존한다.

### ROI-relative manual sampling

| sample_id | roi_y_fraction | 의미 |
|---|---:|---|
| `Y10` | 0.10 | ROI y-range 상단 쪽 sample |
| `Y50` | 0.50 | ROI y-range 중앙 sample |
| `Y90` | 0.90 | ROI y-range 하단 쪽 sample |

이 세 위치는 기존 five-band local diagnostic의 첫 번째, 중앙, 마지막 band center와 개념적으로 대응한다. `manual_reference_manifest.csv`의 9 images 각각에 세 sample을 만들므로 최대 27 sample rows와 81 manual dimension values를 기록한다.

### Image-level manual reference

Image-level accuracy 비교에는 sample-level 값을 직접 하나씩 matching하지 않고 세 Y sample의 dimension별 median을 사용한다.

```text
manual_outer_um = median(outer_Y10, outer_Y50, outer_Y90)
manual_inner_um = median(inner_Y10, inner_Y50, inner_Y90)
manual_gap_um   = median(gap_Y10, gap_Y50, gap_Y90)
```

Image-level manual reference는 dimension별로 최소 2개의 valid Y sample이 있을 때만 계산한다.

- 3/3 valid: 세 값의 median을 사용한다.
- 2/3 valid: 두 valid 값의 median을 사용한다.
- 1/3 valid: 해당 dimension의 image-level manual reference는 unavailable이다.
- 0/3 valid: 해당 dimension의 image-level manual reference는 unavailable이다.

Unavailable sample의 raw row와 `unavailable_reason`은 보존한다. 값을 0으로 대체하거나 추정하지 않으며 unavailable dimension은 accuracy comparison에서 제외한다. 이 2/3 minimum rule은 measurement 전에 고정하는 reference protocol이며 측정 결과를 본 뒤 조정하지 않는다.

### Manual ↔ automatic spatial sampling limitation

Frozen Method 2 image-level automatic result는 whole ROI를 y-direction median으로 1D profile화한 뒤 여러 x-direction accepted candidate의 Outer, Inner, Gap median을 사용한다. Manual image-level reference는 ROI의 Y10/Y50/Y90 각 scanline에서 horizontal center에 가장 가까운 complete physical E1~E6 triplet을 측정하고 dimension별 median을 사용한다.

두 방식은 Outer/Inner/Gap의 physical endpoint definition은 같지만 spatial sampling은 완전히 동일하지 않다. 따라서 이 validation은 **image-level pilot accuracy comparison**이다. Pixel-to-pixel correspondence, candidate-to-candidate correspondence 또는 동일 spatial sample의 직접 비교를 의미하지 않는다. 현재 sample-level automatic matching metric은 추가하지 않는다.

### 공통 physical definition

Signed edge 표기 E1~E6과 무관하게 세 method와 manual reference는 같은 physical endpoints를 사용해야 한다.

```text
Outer Width = E4 - E1
Inner Width = E3 - E2
Gap         = E5 - E4
```

Method별 edge representation은 달라도 되지만 Outer/Inner/Gap의 물리적 시작점과 끝점은 바꾸지 않는다. Reference와 자동 방법 사이에 endpoint 의미가 다르면 accuracy 비교 대상에서 제외하고 정의 불일치로 기록한다.

## 5. Measurement validity와 accuracy의 분리

`measurement_status`와 `failure_reason`은 알고리즘이 측정값을 생성할 수 있었는지와 그 신뢰성에 관한 정보다. Accuracy는 independent manual reference와 자동 µm 결과의 수치 차이다. 따라서 다음 원칙을 적용한다.

- VALID가 accuracy를 보장하지 않는다.
- WARNING은 accuracy 계산에서 자동 제외하지 않는다. 포함 여부와 이유를 report policy에 명시한다.
- FAIL 또는 값이 없는 행을 0으로 치환하지 않는다.
- Accuracy는 해당 dimension의 automatic µm와 manual µm가 모두 존재하는 pair에만 계산한다.
- Image-level accuracy는 automatic image-level result와 세 Y sample에서 계산한 manual image-level median을 비교한다.
- Y10/Y50/Y90 sample 값은 reference traceability와 local variation 확인용으로 보존하며 sample-level automatic matching metric은 만들지 않는다.
- `sample_count`는 Outer/Inner/Gap별로 실제 계산에 포함된 pair 수다.
- Validity 분포와 accuracy summary는 별도 표로 보고한다.

## 6. Accuracy metric

Outer, Inner, Gap 각각에 대해 같은 식을 적용한다.

```text
error_um = auto_um - manual_um
absolute_error_um = abs(auto_um - manual_um)
relative_error_percent = abs(auto_um - manual_um) / manual_um * 100
```

Manual 값이 0이거나 없으면 relative error를 계산하지 않는다. Summary는 dimension별로 다음만 산출한다.

- `sample_count`
- `mean_absolute_error_um`
- `mean_relative_error_percent`

## 7. Untouched validation policy

- Reference manifest는 automatic result 조회 전에 identity metadata만으로 deterministic하게 선택하고 고정한다.
- Manual 측정의 Y sample과 center-nearest complete E1~E6 target은 automatic 결과를 보지 않은 상태에서 고정된 규칙으로 선택한다.
- Calibration과 manual reference를 고정한 뒤 frozen Method 2를 한 번 실행한다.
- Validation 결과를 본 뒤 ROI, Sobel, peak threshold, pairing, long-pair guard, phase logic 또는 calibration 값을 맞추지 않는다.
- Error가 큰 image를 사후 제외하거나 Inner/Gap을 자동 교환하지 않는다.
- 실패와 정의 불일치는 그대로 보존하고 별도로 설명한다.
- Method를 변경하면 새 `method_version`으로 분리하며 기존 untouched 결과를 덮어쓰지 않는다.

## 8. Method 1/2/3 공통 CD evaluation contract

이 contract는 production `MeasurementResult`를 대체하거나 변경하지 않는 evaluation/report layer의 long-form record다. 한 행은 한 image와 한 method/version의 대표 CD 결과를 뜻한다.

| Group | Field | Requirement / meaning |
|---|---|---|
| Identity | `image_id` | 원본 image의 안정적인 식별자 |
| Identity | `wafer_id` | wafer 식별자 |
| Identity | `die_id` | wafer 내부 die 식별자 |
| Identity | `pattern_position` | die 내부 pattern 위치. Canonical stored value는 `center`, `left-top`, `right-bottom` |
| Identity | `source` | image/reference provenance 또는 dataset source |
| Method | `method_id` | `INTENSITY_THRESHOLD`, `GRADIENT_CANNY`, `UNET` 중 하나 |
| Method | `method_version` | 재현 가능한 method implementation/version 식별자 |
| Measurement | `outer_px` | automatic Outer Width in px; unavailable이면 null |
| Measurement | `inner_px` | automatic Inner Width in px; unavailable이면 null |
| Measurement | `gap_px` | automatic Gap in px; unavailable이면 null |
| Measurement | `outer_um` | calibrated automatic Outer Width; unavailable이면 null |
| Measurement | `inner_um` | calibrated automatic Inner Width; unavailable이면 null |
| Measurement | `gap_um` | calibrated automatic Gap; unavailable이면 null |
| Reference | `manual_outer_um` | Y10/Y50/Y90의 independent manual Outer image-level median; unavailable이면 null |
| Reference | `manual_inner_um` | Y10/Y50/Y90의 independent manual Inner image-level median; unavailable이면 null |
| Reference | `manual_gap_um` | Y10/Y50/Y90의 independent manual Gap image-level median; unavailable이면 null |
| Accuracy | `outer_error_um` | `outer_um - manual_outer_um` |
| Accuracy | `inner_error_um` | `inner_um - manual_inner_um` |
| Accuracy | `gap_error_um` | `gap_um - manual_gap_um` |
| Accuracy | `outer_absolute_error_um` | absolute Outer error |
| Accuracy | `inner_absolute_error_um` | absolute Inner error |
| Accuracy | `gap_absolute_error_um` | absolute Gap error |
| Accuracy | `outer_relative_error_percent` | absolute Outer error / manual Outer × 100 |
| Accuracy | `inner_relative_error_percent` | absolute Inner error / manual Inner × 100 |
| Accuracy | `gap_relative_error_percent` | absolute Gap error / manual Gap × 100 |
| Reliability | `measurement_status` | method의 VALID/WARNING/FAIL validity |
| Reliability | `failure_reason` | null 또는 method가 제공한 실패/경고 사유 |
| Traceability | `calibration_id` | 적용한 calibration record 식별자; px-only이면 null |
| Traceability | `runtime_ms` | 해당 method 실행 시간 |

Accuracy field는 automatic/manual pair가 없을 때 null로 둔다. Method 비교 시 동일 `image_id`, 동일 physical definition, 동일 calibration provenance를 사용한다. 각 method의 내부 diagnostic을 이 공통 contract에 억지로 넣지 않으며 method-specific companion table로 분리한다. Manual reference의 `source`는 image-level manifest에서 관리하고 sample-level table은 `image_id`로 join한다. 현재 manifest의 `CURATED_IMAGES::PASS_normal` 표기는 dataset audit의 `source_root::source_group` convention을 따른다.

### Future operational extension fields

다음 field는 향후 운영 연결을 위한 optional extension이며 현재 값을 생성하지 않는다.

- `review_required`
- `critical_flag`
- `inspection_timestamp`
- `algorithm_version`
- `acquisition_id`

`method_version`은 비교 대상 method implementation의 재현성 식별자이고, future `algorithm_version`은 배포 pipeline/build 식별자가 필요할 때만 사용한다.

## 9. Identity hierarchy와 aggregation direction

현재 schema를 교체하지 않고 data containment와 processing direction을 구분한다.

### 용어 구분

- `pattern_position`: die 내부 physical pattern site다. Canonical stored value는 `center`, `left-top`, `right-bottom`이다. Filename parsing이나 UI에서는 `c`, `lt`, `rb` abbreviation을 사용할 수 있지만 evaluation, join, grouping에는 canonical value를 사용한다.
- `capture_region`: acquisition 내 촬영 영역 metadata다. `ImageRecord.capture_region`은 required `str` field이므로 모든 record에 존재한다. 의미 있는 영역 구분이 있으면 `top`, `middle`, `bottom` 등을 기록하고, 별도 구분이 없으면 현재 repository convention인 `standard`를 명시한다.
- `manual_sample`: 한 image의 measurement ROI 내부 reference scanline이다. 현재 값은 `Y10`, `Y50`, `Y90`이다.

이 세 개념은 서로 다른 identity/granularity를 가지며 하나의 field로 합치지 않는다.

### Data containment / identity hierarchy

```text
Wafer
  → Die
    → Pattern
      → Acquisition
        → Image
          → Measurement Sample
```

- Wafer는 physical wafer다.
- Die는 wafer 내부 die이며 `wafer_id + die_id`로 연결한다.
- Pattern은 die 내부 site이며 현재 범위에서 `wafer_id + die_id + pattern_position`으로 연결한다.
- Acquisition은 동일 physical pattern의 촬영 event다. 재촬영 구분에는 optional `acquisition_id`를 사용할 수 있다.
- Image는 실제 image file이며 `image_id`로 식별한다.
- Measurement Sample은 image 내부 Y10/Y50/Y90 reference sample이며 `image_id + sample_id`로 식별한다.
- `source`는 dataset/provenance를 보존하고 동일 ID 충돌을 방지하는 데 사용한다.

### Processing / aggregation direction

```text
Image
  → CD / Defect Result
    → Pattern-level Result
      → Die-level Aggregation
        → Wafer-level Aggregation
          → Wafer Map
```

Containment hierarchy는 physical/data ownership을 나타내고 processing direction은 분석 결과가 상위 단위로 집계되는 흐름을 나타낸다. Wafer map이나 yield aggregation은 이 identity key를 소비하는 별도 downstream layer로 두며 지금 구현하지 않는다.

## 10. Future defect integration boundary

Defect Detection은 image/region 단위 observation을 만들고, Classification은 그 observation에 label/confidence를 부여하는 별도 pipeline으로 둔다. 두 단계는 `image_id`와 필요 시 pattern identity로 CD evaluation record에 join한다.

Defect 결과로 calibration을 변경하거나 Outer/Inner/Gap 정의를 바꾸지 않는다. CD validity/accuracy와 defect presence/classification을 별도 field/table로 유지한다. 향후 `review_required`와 `critical_flag`는 결합된 운영 정책이 소비할 수 있지만, 현재 defect PASS/FAIL rule이나 classifier는 정의하지 않는다.

## 11. Future What-if / Sensitivity Simulator boundary

초기 simulator는 causal predictor나 process root-cause AI가 아니라 명시적인 입력 변화에 대한 What-if/sensitivity module로 취급한다. 입력은 versioned structured CD/wafer data와 사용자 지정 scenario parameter이고, 출력은 measured fact와 분리된 scenario/result record여야 한다.

Simulator 결과는 원본 measurement, manual reference, calibration record를 덮어쓰지 않으며 automatic metrology의 threshold나 validity에 feedback하지 않는다. Causal claim은 별도의 실험 설계와 검증이 있기 전까지 하지 않는다.

## 12. Ruler 도착 후 실행 순서

1. 20× acquisition/export metadata와 resize 여부를 기록한다.
2. 명확한 known interval endpoint를 선택하고 interval별 x-axis µm/px를 계산한다.
3. Interval consistency와 현재 wafer image 적용 가능성을 검토한다.
4. 근거가 충분할 때만 official calibration ID와 scale을 확정한다.
5. 고정된 manifest의 9장을 Y10/Y50/Y90 blind manual protocol로 측정하고 sample-level raw 값을 보존한다.
6. Frozen Method 2 결과를 common contract로 변환해 dimension별 validity와 accuracy를 분리 보고한다.
7. 같은 contract로 Method 1과 Method 3를 추후 비교한다.
