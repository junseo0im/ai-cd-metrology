# Defect Split Policy v1

## Split 원칙

파일 단위 무작위 분할은 exact duplicate와 동일 wafer/die가 서로 다른 split에 들어갈 수 있어 사용하지 않는다. 동일 SHA-256을 먼저 묶고, 유효한 `wafer_id + die_id`가 같은 hash group을 connected component로 합친다. Pattern position이 달라도 같은 wafer/die이면 같은 component다.

Wafer 또는 die metadata가 `unknown`/blank인 이미지는 하나의 giant group으로 합치지 않는다. Exact hash만 유지하고 `provenance_risk=UNKNOWN` 또는 `PARTIAL`로 표시한다. Near-duplicate 검색은 추가하지 않았다.

명시적 EXCLUDE는 train/val/test에 넣지 않는다. TBD 또는 `HOLD_*`는 hold로 둔다. 같은 component에 EXCLUDE/HOLD와 일반 후보가 섞이면 non-excluded 후보도 hold하여 provenance group을 쪼개지 않는다.

## 목표와 실제 결과

목표는 split-eligible unique SHA-256 기준 70/15/15다. Group integrity가 정확한 비율보다 우선한다.

| Split | File rows | Unique contents | Assigned unique % |
| --- | ---: | ---: | ---: |
| TRAIN | 898 | 596 | 69.30% |
| VAL | 193 | 132 | 15.35% |
| TEST | 198 | 132 | 15.35% |
| HOLD | 85 | 55 | - |
| EXCLUDE | 5 | 5 | - |

## 분포 확인

### TRAIN

- Source dataset: DEFECT_TOTAL 467, WAFER_PHOTOS_E70 279, CURATED_IMAGES 129, NEW_DEFECT_70_2_E70_F 23
- Source label hint: unlabeled 349, nc 227, normal 101, particle 96, pattern_narrowing 47, pattern_die 25
- Provisional category hint: UNCERTAIN 582, PHYSICAL_PATTERN_DEFECT 112, SURFACE_OR_BACKGROUND_ARTIFACT 103, NORMAL 98, ACQUISITION_QUALITY_ISSUE 3

### VAL

- Source dataset: DEFECT_TOTAL 93, WAFER_PHOTOS_E70 60, CURATED_IMAGES 39, NEW_DEFECT_70_2_E70_F 1
- Source label hint: unlabeled 87, nc 36, normal 31, particle 17, pattern_narrowing 12, pattern_die 3
- Provisional category hint: UNCERTAIN 125, NORMAL 30, PHYSICAL_PATTERN_DEFECT 20, SURFACE_OR_BACKGROUND_ARTIFACT 18

### TEST

- Source dataset: DEFECT_TOTAL 93, WAFER_PHOTOS_E70 63, CURATED_IMAGES 39, NEW_DEFECT_70_2_E70_F 3
- Source label hint: unlabeled 79, nc 48, normal 34, particle 17, pattern_die 5, pattern_narrowing 5
- Provisional category hint: UNCERTAIN 128, NORMAL 34, SURFACE_OR_BACKGROUND_ARTIFACT 19, PHYSICAL_PATTERN_DEFECT 17

## Integrity validation

- Same hash across active splits: 0
- Same valid wafer/die across active splits: 0
- Same split group across active splits: 0
- Explicit HOLD/EXCLUDE violations: 0
- Total split groups: 181
- Provenance risks: KNOWN 120, UNKNOWN 61

## Test set 사용 원칙

Test 후보는 향후 model tuning에 사용하지 않는다. 이 결과는 annotation 전 `SPLIT DRAFT v1`이다. Annotation 완료 후 class/instance 분포와 severe class absence만 확인하고, 학습 시작 전에 split을 한 번 최종 확정한다.

## 아직 draft인 이유

Polygon annotation과 annotation QA가 끝나지 않았고 전체 canonical class/instance 분포가 아직 없다. 따라서 현재 상태는 학습-ready dataset이 아니라 leakage-safe split draft다.

다음 순서: split draft → polygon annotation → annotation QA → class별 image/instance count 확인 → split 최종 확인 → train-only augmentation 결정 → YOLOv8-seg baseline.
