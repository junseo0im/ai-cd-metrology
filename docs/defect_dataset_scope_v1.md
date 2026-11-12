# Defect Dataset Scope v1

## TRAINING INCLUDE

- human adjudication에서 `training_eligibility=KEEP`으로 승인된 이미지.
- canonical classes: `normal`, `open`, `short`, `pattern_narrowing`, `pattern_thickening`, `deformation`, `particle`, `scratch`.
- 정상 이미지는 polygon 없이 포함한다.
- 세부 class가 아직 TBD인 KEEP 이미지는 annotation을 시작하기 전 HOLD 상태로 둘 수 있다.

## TRAINING EXCLUDE

- reference 및 CD measurement evaluation asset.
- 측정 비교용 이미지와 사용자가 명시적으로 제외한 이미지.
- physical defect와 acquisition/process-quality issue를 구분할 수 없는 visual index 556의 v1 training copy.
- annotation guideline의 eligibility를 충족하지 못한 screenshot/acquisition-only 이미지.

## UNCERTAIN / HOLD

- visual index 100의 tiny scratch.
- defect 존재는 확인됐지만 세부 class가 미정인 visual index 312, 330, 341, 437, 815.
- 경계가 반복 가능하지 않은 tiny/border/overlap instance.
- HOLD는 삭제가 아니며 원본과 provenance를 그대로 유지한다.

## SOURCE LABEL ONLY

- `pattern_die`, `nc`, `unlabeled`.
- compound filename label.
- 이 값은 source provenance이며 YOLO canonical class로 직접 사용하지 않는다.

HIGH 32개 pre-annotation eligibility는 **KEEP 26 / EXCLUDE 5 / TBD 1**이며 history로 보존한다. 전체 [split draft v1](defect_split_policy_v1.md)은 이미 존재한다 (고유 콘텐츠 TRAIN 596 / VAL 132 / TEST 132 / HOLD 55 / EXCLUDE 5).

현재 Pilot 판정은 [최신 completed review](../data/metadata/defect_annotation_pilot_review_v1_completed_draft.csv)를 우선한다 (DEFECT 29 / HOLD 5 / NORMAL 2). 위 HOLD 사례와 과거 source/history를 소급 수정하지 않는다. 현재 annotation 운영 기준은 [Guideline v2](yolov8_seg_annotation_guideline_v2.md)이며 Pilot image-level HOLD 내부의 confirmed polygon도 보존한다. Full annotation/training은 아직 시작하지 않았다.
