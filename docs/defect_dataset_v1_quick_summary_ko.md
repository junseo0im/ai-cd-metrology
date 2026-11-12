# Defect Dataset v1 빠른 요약

## 현재 상태: Pilot 완료 + Annotation Guideline v2 FREEZE

- 현재 운영 기준은 [YOLOv8-seg Annotation Guideline v2](yolov8_seg_annotation_guideline_v2.md)다.
- CVAT Pilot 36장 완료: export 34 label files / 221 polygons, label-free NORMAL은 P010/P025다.
- [최신 completed review](../data/metadata/defect_annotation_pilot_review_v1_completed_draft.csv): DEFECT 29 / HOLD 5 / NORMAL 2, 36장 모두 REVIEWED다. 파일명의 `_draft`는 checkpoint 원본 이름을 보존했다.
- 과거 HIGH review와 inventory/queue/pilot selection은 pre-annotation history다. 현재 Pilot 판정은 completed review를 우선한다. P009는 particle 6개의 DEFECT, P014/P015는 confirmed particle을 보존하는 HOLD, P016은 short + particle, P025는 NORMAL이다.
- Leakage-safe split draft는 이미 존재한다. 고유 콘텐츠 기준 TRAIN 596 / VAL 132 / TEST 132 / HOLD 55 / EXCLUDE 5이며 [split policy](defect_split_policy_v1.md)를 유지한다.
- 현재 [9] guideline v2 freeze까지 완료했다. [10] TRAIN 전체 annotation 이후 작업과 augmentation/training은 아직 시작하지 않았다.

## 1. 데이터 확인 결과

- `data/raw` 이미지 1,379장을 모두 조사했다.
- SHA-256 기준 고유 이미지는 920개다.
- exact duplicate group은 459개다.
- 해상도는 352종이다.
- 고유 이미지 전체를 contact sheet로 확인했다.
- HIGH 32장은 사용자가 원본을 다시 보고 직접 판정했다.
- 원본 이미지는 수정하거나 이동하지 않았다.

## 2. 사람이 판단한 기준

- 패턴이 끊기면 `open`이다.
- 인접 패턴이 실제로 붙으면 `short`다.
- 폭이 좁아지면 `pattern_narrowing`이다.
- 폭이 두꺼워지면 `pattern_thickening`이다.
- 구조 자체가 비정상적으로 변하면 `deformation`이다.
- 이물질은 `particle`이다.
- wafer 표면의 선형 흔적은 `scratch`다.
- 정상 반복 pattern이고 검출할 이상이 없으면 `normal`이다.

이번 프로젝트에서는 particle과 scratch도 검출 대상으로 포함한다. 한 이미지에 여러 defect가 있으면 각각 따로 판단한다. Screenshot은 기본적으로 제외하고, reference·평가용·명시적 EXCLUDE 이미지는 학습하지 않는다. 단, 실제 결함과 KEEP가 확인된 visual index 23은 human-approved 예외다. 촬영 문제와 실제 pattern defect를 구분할 수 없으면 억지로 class를 지정하지 않는다.

## 3. HIGH 32장 pre-annotation review 결론 (history)

- 기존 normal label 안에도 scratch와 particle이 있었다.
- compound filename label은 실제로 여러 defect가 함께 있는 경우가 있었다.
- particle과 scratch를 detection 대상으로 확정했다.
- `pattern_thickening` 사례 3개를 확인해 별도 candidate class로 채택했다.
- 일부 deformation source label은 실제 physical pattern defect였다.
- reference, 길이 측정 비교용, 제외 판정 및 불명확 acquisition 사례는 v1 학습에서 제외했다.
- filename이나 folder label만으로 training class를 결정하면 안 된다.
- 최종 eligibility는 KEEP 26, EXCLUDE 5, TBD 1이다.

## 4. YOLOv8-seg annotation v1 기준 (history; 현재 운영은 v2 참조)

- 결함 하나에는 polygon 하나를 만든다.
- 여러 결함이면 결함별로 여러 polygon을 만들고 각각 class를 지정한다.
- normal image에는 polygon을 만들지 않는다.
- 결함 주변의 정상 pattern 전체를 mask하지 않는다.
- compound filename을 하나의 class로 만들지 않는다.
- tiny scratch는 원본에서 경계를 반복해서 그릴 수 있을 때만 annotation한다.
- segmentation polygon이 primary annotation이며 별도 수작업 BBox는 만들지 않는다.

## 5. 앞으로 학습 순서

[9] Annotation Guideline v2 FREEZE (현재) → [10] TRAIN 전체 annotation → [11] VAL annotation → [12] TEST annotation → [13] 최종 dataset QA → [14] YOLOv8-seg baseline. 기존 split draft와 original CVAT GT를 보존하며 이번 단계에서는 후속 annotation/augmentation/training을 시작하지 않는다.

## 6. 아직 보류

- visual index 100 tiny scratch의 최종 포함 여부
- visual index 556처럼 physical/acquisition 구분이 불명확한 사례의 장기 처리
- Pilot 외 일부 KEEP 이미지의 instance별 세부 class (Pilot 현재 상태는 completed review 우선)
- annotation QA 후 split draft의 최종 확인 (split 자체는 이미 존재)
- IGNORE candidate의 training-time 처리 방식
- augmentation 종류와 강도
- YOLO model size와 hyperparameter
- VLM 사용 여부
