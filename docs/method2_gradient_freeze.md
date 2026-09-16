# Method 2 Gradient/Edge Metrology — Freeze & Closure

이 문서는 Method 2의 알고리즘 개발 및 diagnostic 단계를 닫고, calibration 전 기준선을 고정한다. 검토 기준은 `phase-a-skeleton` 브랜치의 production 코드와 `3575e26` (`fix: guard against long edge pairings`)까지의 Git 이력이다. 이 문서는 새로운 알고리즘이나 threshold를 제안하지 않는다.

## 1. Scope

전체 프로젝트는 20× optical microscopy 이미지의 IDA(interdigital electrode) 패턴에서 Outer Width, Inner Width, Gap을 자동 계측하고, pixel-to-µm calibration, manual/reference 비교, 신뢰도 및 오차 평가, 여러 측정 방법 비교까지 수행하는 것을 목표로 한다.

Method 2의 범위는 **Gradient / Edge 기반 automatic CD metrology**다. Particle 및 pattern-narrowing 데이터는 defect classifier 개발이 아니라 sensitivity, robustness 및 failure mode 확인에만 사용했다.

이 freeze에서 고정하는 것은 현재 pixel-level baseline, production safety guard, diagnostic 경계, 검증 evidence와 알려진 한계다. 공식 calibration과 µm accuracy 검증은 아직 완료되지 않았다.

## 2. Physical Measurement Definition

Signed Sobel-x profile에서 하나의 optical dark band는 인접한 `negative → positive` peak doublet으로 정의한다.

- Dark band A: E1 = outer edge, E2 = inner edge
- Dark band B: E3 = inner edge, E4 = outer edge
- 다음 dark band: E5, E6

Pixel measurement 정의는 다음과 같다.

```text
Outer Width = E4 - E1
Inner Width = E3 - E2
Gap         = E5 - E4
```

이 정의는 OptiView manual reference와 physical pairing validation에서 확인했다. Inner와 Gap endpoint는 reference gradient peak와 약 0–2 px 수준으로 대응했다. Outer의 시작 endpoint에는 수 px 차이가 있었지만 offset correction은 도입하지 않았다.

`PixelPairingCandidate`는 `sequence_index`, E1–E6의 ROI-relative x 좌표, 세 pixel distance를 보존한다. `sequence_index`는 검출 순서일 뿐 physical finger ID가 아니다.

## 3. Frozen Production Pipeline

현재 `GradientCannyMetrology`의 고정 pipeline은 다음과 같다.

1. `ImageRecord.pattern_position`에 맞는 position-specific normalized ROI를 선택한다.
2. ROI를 grayscale uint8로 변환한다.
3. Gaussian blur를 적용한다: kernel `5×5`, sigma `0.0`.
4. signed Sobel-x를 계산한다: kernel size `3`.
5. y 방향 median으로 1D x-gradient profile을 만든다.
6. profile 양·음 extrema의 35% 상대 threshold로 signed peak를 찾고, 연속 threshold run마다 strongest peak 하나를 남긴다.
7. x 순서상 인접한 `negative → positive`만 dark-band doublet으로 채택한다. Partial 또는 unexpected peak는 강제 연결하지 않고 reason과 함께 보존한다.
8. 끊기지 않은 세 doublet에서 E1–E6 candidate를 구성하고 위 physical 정의로 Outer/Inner/Gap을 계산한다.
9. long-pair guard를 적용하고 flagged candidate를 accepted set에서 제외한다.
10. accepted candidate를 검증한다.
11. accepted Outer/Inner/Gap 각각의 median을 image-level pixel 결과로 사용한다.
12. median triplet에 가장 가까운 실제 accepted candidate 하나를 representative candidate로 선택해 E1–E6를 `edge_coordinates`에 기록한다.
13. optional `CalibrationRecord`가 있으면 x-axis scale로 세 결과를 µm로 변환한다.

현재 ROI config는 다음과 같다.

| Position | x range | y range |
|---|---:|---:|
| center (`c`) | 0.08–0.92 | 0.15–0.85 |
| left-top (`lt`) | 0.08–0.92 | 0.35–0.90 |
| right-bottom (`rb`) | 0.08–0.92 | 0.10–0.65 |

`_roi_for`는 축약 position alias `c → center`, `lt → left-top`, `rb → right-bottom`도 지원한다.

이 값들은 현재 baseline에 사용되지만 `configs/metrology.yaml`에는 여전히 contact-sheet 기반 **PILOT candidate**로 표시돼 있다. 공식 calibration 또는 전체 acquisition geometry 검증을 거친 최종 ROI라는 뜻은 아니다.

Candidate validation의 현재 조건은 다음과 같다.

- accepted candidate가 최소 3개여야 한다.
- detected peak가 없어서는 안 된다.
- rejected peak가 전체 detected peak의 절반 이상이면 실패한다.
- Outer/Inner/Gap은 모두 양수여야 한다.
- candidate 과반에서 `Outer > Inner`가 성립해야 한다.

`VALID`는 validation failure, rejected peak, long-pair flag가 없는 결과다. Validation을 통과했지만 rejected peak 또는 long-pair flag가 남으면 `WARNING`이다. Validation failure 또는 catastrophic mis-pairing이면 `FAIL`이며 image-level pixel/µm result는 저장하지 않는다.

Class 이름은 `GradientCannyMetrology`지만 physical pairing과 측정의 primary signal은 signed Sobel-x다. Canny는 visualization diagnostic일 뿐 candidate 생성이나 Width/Gap 계산에 사용하지 않는다. Production diagnostic 기본 Canny 값은 50/150이고, 과거 characterization용 20/60 상수도 별도로 남아 있다. `configs/metrology.yaml`의 Canny 및 edge-pairing 항목은 현재 `null`이며 production measurement를 구성하지 않는다.

`MeasurementResult`는 image/method/measurement ID, Outer/Inner/Gap의 px와 optional µm, optional calibration ID, representative E1–E6 좌표, status, failure reason, runtime을 보존한다. Reported median과 representative candidate의 실제 세 거리는 candidate 수가 짝수일 때 정확히 같지 않을 수 있다. E1–E6의 x는 original-image 좌표로 변환되지만 y는 ROI 중앙 placeholder이며 실제로 검출한 edge y 좌표가 아니다.

## 4. Local Diagnostics

기본 `band_count`는 5다. Whole ROI에 대해 grayscale, blur, Sobel-x를 한 번만 수행한 뒤, x-gradient를 y 방향의 연속된 half-open local band 5개로 나눈다. 각 band는 whole ROI와 동일한 profile, doublet, candidate, long-pair guard, validation 및 status helper를 재사용한다.

Local-band의 용도는 다음으로 제한한다.

- y-local CD variation 관찰
- raw/accepted/long-pair candidate 진단
- narrowing 또는 국소 disturbance 관찰

Local y bounds는 original-image 좌표이고 candidate x는 ROI-relative 좌표다. Local medians는 accepted candidate에서 계산하며 FAIL band에서도 남을 수 있으므로 반드시 status와 함께 해석해야 한다.

Local-band status나 median은 image-level `measure()` 결과를 roll-up, 수정 또는 대체하지 않는다. Image-level 결과는 whole ROI chain만으로 결정된다.

## 5. Long-Pair Safety Guard

확인된 failure chain은 다음과 같다.

```text
missing/occluded edge
→ non-adjacent edge pairing
→ abnormally long candidate
→ false VALID 가능
```

대표 사례 `w3_24_rb_p`의 local candidate는 Outer/Inner/Gap = `270/237/104 px`였다. 43-image audit에서 정상 보존 대상의 최대 upward ratio는 약 `1.2353`, 확인된 long pair는 약 `2.0085`였다. 현재 production 상수 `LONG_PAIRING_RELATIVE_CEILING`은 `1.6`이다. 이는 defect/QC threshold가 아니라 두 영역을 분리하기 위해 채택한 pilot safety ceiling이다.

Guard semantics는 다음과 같다.

- Raw candidate set의 Outer/Inner/Gap median을 각각 한 번 계산한다.
- candidate의 세 값 중 하나라도 해당 median 대비 비율이 **1.6을 초과**하면 flag한다.
- 감소 방향 criterion, 반복 filtering, 보정 또는 window 재연결은 하지 않는다.
- Flagged candidate는 원래 sequence index와 E1–E6를 유지한 채 diagnostic에 남는다.
- Accepted set에서만 제외한 후 기존 validation과 median 계산을 수행한다.
- Accepted candidate가 충분하면 결과는 `WARNING`; 3개 미만이면 기존 insufficient-candidate `FAIL`을 사용한다.

Guard 적용 후 known long candidate는 제외됐고, 남은 candidate로 측정을 유지하면서 local status가 WARNING으로 바뀌었다. Normal, narrowing 및 benign-particle regression은 보존됐다.

## 6. Validation Evidence

핵심 evidence는 다음과 같다.

- Normal physical pairing: 24/24 image validation PASS.
- 위 24장에서 valid candidate 167개, `Outer > Inner` 167/167, catastrophic mis-pairing 0.
- Normal local-band: 24 images × 5 bands = 120 bands. VALID 78, WARNING 42, FAIL 0.
- Pattern narrowing: local/raw candidate 수준에서 국소 CD variation을 확인했다. 이를 defect classifier 또는 production defect rule로 확장하지 않았다.
- Particle: 작은 particle에서 measurement가 유지되는 사례가 있었고, edge occlusion에서 long-pair false-valid failure를 발견했다. 현재 long-pair guard가 해당 known failure를 차단한다.
- Phase investigation: 실제 phase shift, first-E1 failure, parity signal, low-separability 한계를 확인한 뒤 추가 heuristic 없이 종료했다.
- Current automated regression: `python -m pytest -q` 결과 **64 passed**.

Validation 결과는 현재 표본과 pixel-level baseline에 대한 evidence다. 공식 µm accuracy, production repeatability 또는 전체 데이터 일반화 성능을 의미하지 않는다.

## 7. Phase Investigation — Adopted / Rejected / Limitation

### Diagnostic으로 채택

Boundary 또는 internal edge loss 뒤에는 candidate physical phase가 이동해 Inner/Gap semantic meaning이 바뀔 수 있다. Image-local Inner/Gap parity score는 검토된 사례에서 다음 경향을 보였다.

- Physically stable candidate: negative 또는 weak/zero score
- Verified phase-shift candidate: clear positive score

따라서 parity score는 phase shift를 살펴보는 diagnostic 및 measurement-confidence evidence로만 남긴다. Production `GradientCannyMetrology`에는 parity score 계산, guard 또는 correction이 없다.

### 폐기

- First-E1 position anchoring: 앞 candidate가 사라지면 first E1이 크게 이동해도 remaining grouping은 정상일 수 있어 폐기했다.
- Global absolute Width/Gap threshold: wafer/die 및 image별 pixel variation을 phase truth로 사용할 수 없어 폐기했다.
- 다른 wafer/die의 absolute pixel 값과 비교해 phase를 결정하는 방식도 폐기했다.

### 검증했지만 production 비채택

- Automatic parity guard
- Inner/Gap automatic swap
- Automatic phase correction 또는 candidate rejection

### 확인된 한계

`w2_26_lt`는 whole reference Inner/Gap이 약 `118/113 px`로 가깝다. 일부 local candidate에서 `Inner = Gap = 116 px` tie가 발생했다. 이 경우 normal/swapped value hypothesis를 구분할 정보 자체가 부족하다. 따라서 image-local parity만으로 physical phase를 항상 판별할 수 없으며 production threshold를 정당화할 수 없다.

## 8. Calibration Status

Calibration은 완료되지 않았다.

현재 존재하는 구조는 다음과 같다.

- `CalibrationRecord`: calibration ID, objective, x/y µm-per-pixel scale과 finite positive validation
- `pixels_to_micrometers()` 및 세 측정값 공통 conversion helper
- calibration이 없으면 px 결과는 유지하고 모든 `*_um` 및 `calibration_id`는 `None`
- calibration이 있으면 현재 x-axis scale로 Outer/Inner/Gap을 변환하며 pixel result와 status는 바꾸지 않음

OptiView reference에서 얻은 약 `0.079923 µm/px`는 manual annotation과 정수 pixel span의 일관성을 확인하기 위한 implied scale이다. 독립 calibration ruler로 확정한 official production calibration이 아니며, 이 값을 config에 등록하거나 accuracy ground truth로 사용하지 않는다. `configs/metrology.yaml`의 calibration 값은 모두 `null`이다.

남은 핵심은 calibration ruler 기반 µm/px 확정과 독립 manual/reference 대비 µm accuracy 및 error 검증이다.

## 9. Dataset Status

최근 read-only inventory 결과는 다음과 같다.

- 전체 1,377 images, readable 1,377, unreadable 0
- Curated 223: PASS 178, WARNING 12, FAIL 33
- Non-curated 1,154 records
- Non-curated끼리의 exact duplicate를 접으면 SHA-256 content 695개
- `DEFECT_TOTAL ↔ WAFER_PHOTOS_E70`: exact duplicate 432쌍
- `DEFECT_TOTAL ↔ NEW_DEFECT_70_2_E70_F`: exact duplicate 27쌍
- Curated와 non-curated 사이 exact duplicate: 0

새 데이터는 현재 Method 2 tuning에 사용하지 않는다. Calibration 이후 frozen Method 2의 small untouched validation 후보로 보존한다. Source folder가 다르다는 사실만으로 statistical independence가 보장되지는 않으므로 provenance를 먼저 확인해야 한다.

## 10. Known Limitations

- 현재 ROI는 position-specific normalized pilot baseline이며 final acquisition geometry policy가 아니다.
- Rotation correction과 pattern-detected ROI가 없다. 패턴이 대체로 vertical이라는 가정에 의존한다.
- y-median 1D profile은 edge의 실제 y 위치와 local curvature를 보존하지 않는다.
- Integer-pixel peak 위치를 사용하며 subpixel fitting이 없다.
- Candidate sequence index는 physical finger ID가 아니며 finger tracking이 없다.
- Missing edge 후 semantic phase shift를 production에서 자동 판별하거나 복구하지 않는다.
- Long-pair guard는 큰 upward outlier 한 종류만 방어하며 일반 defect/QC framework가 아니다.
- Local-band 결과는 diagnostic이며 image-level 결과를 보정하지 않는다.
- Canny, parity 및 defect observation은 final Width/Gap 계산 신호가 아니다.
- Official calibration과 µm accuracy/error가 없어 현재 결과의 physical accuracy는 확정되지 않았다.
- Validation 범위 밖의 신규 데이터에 대한 일반화 성능은 아직 확인하지 않았다.

## 11. Frozen / Deferred Items

### Frozen production baseline

- Position-specific normalized ROI와 x measurement axis
- Grayscale, Gaussian 5×5, signed Sobel-x 3×3
- y-median profile과 0.35 relative signed-peak threshold
- Adjacent negative→positive dark-band doublet
- E1–E6 physical pairing 및 Outer/Inner/Gap 정의
- Accepted-candidate median image result
- Existing VALID/WARNING/FAIL mapping
- Long-pair guard ceiling 1.6과 현재 exclusion semantics
- `MeasurementResult` 및 optional calibration hook
- Five-band local diagnostic 구조와 non-roll-up semantics

### Rejected / not adopted

- First-E1 phase anchoring
- Global absolute Width/Gap phase threshold
- Cross-wafer/die absolute pixel comparison for phase
- Automatic parity guard, Inner/Gap swap, phase correction

### Deferred

- External physical phase anchor
- Automatic finger tracking
- Edge reconstruction/interpolation
- Advanced QC framework
- Subpixel fitting
- Final ROI/rotation strategy
- Defect inspection/classification

## 12. Next Steps

고정된 다음 순서는 다음과 같다.

1. Official calibration ruler validation
2. x-axis µm/px 확정과 필요한 calibration metadata 기록
3. Manual/reference 대비 µm accuracy 및 error 검증
4. Frozen Method 2로 신규 unique data의 소규모 untouched validation
5. Method 2 최종 결과와 limitation 정리
6. Method 1 및 Method 3과 공통 comparison framework에 삽입
7. Defect inspection은 별도 단계에서 진행

Untouched validation 결과를 보고 현재 Method 2 threshold를 다시 tuning하지 않는다. 문제가 발견되면 기존 baseline의 limitation 또는 별도 차기 버전 요구사항으로 기록한다.
