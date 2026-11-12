# Defect Annotation Workflow v1

## 현재 checkpoint: CVAT Pilot 완료 / Guideline v2 FREEZE

현재 full TRAIN/VAL/TEST 운영 기준은 [Annotation Guideline v2](yolov8_seg_annotation_guideline_v2.md)다. 기존 v1 내용은 준비 단계의 history를 보존하며 충돌 시 v2와 [최신 completed review log](../data/metadata/defect_annotation_pilot_review_v1_completed_draft.csv)를 우선한다. 초기 review/queue는 pre-annotation 상태이며 소급 수정하지 않는다.

CVAT Online Pilot 36장 완료: 34 label files, 221 polygons, DEFECT 29 / HOLD 5 / NORMAL 2 (P010/P025). 원본 export는 `data/derived/cvat_pilot_v1/export/defect_pilot_v1_yolo_seg.zip`에 보존한다. P009는 현재 particle DEFECT, P014/P015는 confirmed particle을 보존하는 HOLD, P016은 short + particle, P025는 NORMAL이다. Image HOLD와 confirmed instance 보존을 구분한다. IGNORE candidate의 training-time 처리는 미정이다.

현재 [9] guideline v2 FREEZE까지 완료했다. 다음 [10] TRAIN 전체 annotation → [11] VAL → [12] TEST → [13] 최종 dataset QA → [14] YOLOv8-seg baseline은 이번 작업에서 시작하지 않는다.

## 1. 도구와 보안

Primary annotation tool은 **CVAT**다. Polygon instance annotation, 다중 object/class, Ultralytics YOLO segmentation export를 지원한다. 팀 보안 및 데이터 정책을 확인하기 전에는 raw image를 외부 cloud service에 업로드하지 않는다. 초기 준비 단계에서는 Local CVAT를 기본 후보로 두었으나 완료된 Pilot은 CVAT Online을 사용했다. 이번 반영에서는 설치, 추가 upload 또는 새 task를 만들지 않는다.

## 2. YOLO object class mapping

| ID | Object class |
| ---: | --- |
| 0 | `open` |
| 1 | `short` |
| 2 | `pattern_narrowing` |
| 3 | `pattern_thickening` |
| 4 | `deformation` |
| 5 | `particle` |
| 6 | `scratch` |

`normal`, `MULTI`, `pattern_die`, `nc`, `unlabeled`, `UNCERTAIN`은 YOLO object class가 아니다. `MULTI`는 image-level 상태다.

## 3. Normal 처리

Normal image는 defect instance가 0개인 negative image다. Polygon을 만들지 않는다. `source_label=normal`만으로 확정하지 않고 원본 확인 후 `CONFIRM_NORMAL`로 승인한다.

## 4. Queue status

아래 status는 annotation 전 queue의 의미다. 완료된 Pilot의 현재 판정은 위 completed review를 확인하며 과거 queue와 NORMAL/class 판단이 달라질 수 있다.

- `READY_POLYGON`: human adjudication으로 단일 canonical defect가 명확하다.
- `READY_MULTI_POLYGON`: 여러 defect class 또는 여러 instance가 확인됐다.
- `CONFIRM_NORMAL`: human-confirmed normal이며 polygon 없는 negative image다.
- `CLASS_REVIEW_REQUIRED`: polygon 전에 canonical class 또는 normal 여부를 확인해야 한다.
- `HOLD`: 현재 annotation하지 않는다.
- `EXCLUDE`: annotation 및 training 대상이 아니다.

## 5. 작업 순서

전체 순서는 TRAIN → VAL → TEST다. TEST는 마지막에 작업하고 tuning에 사용하지 않는다. TRAIN 내부는 human-reviewed READY/normal → clear source hint(open/short/narrowing/particle) → normal source candidate → pattern_die/nc/unlabeled/deformation_unknown 등 class review 순서다. HOLD와 EXCLUDE는 작업하지 않는다.

## 6. Polygon 기본 규칙

- 결함 실제 영역만 polygon으로 표시하고 정상 pattern 전체를 mask하지 않는다.
- Object 1개당 polygon 1개를 만든다. 별도 수작업 BBox는 만들지 않는다.
- Border defect는 이미지 안에서 보이는 영역까지만 닫는다.
- Tiny scratch는 native 원본에서 안정적인 경계가 있을 때만 그린다.
- Ambiguous instance는 새 class를 만들지 않고 HOLD한다.

## 7. Multi-defect와 여러 particle

여러 defect는 visible defect별 canonical class와 polygon으로 분리한다. 같은 class의 particle이 여러 개면 분리 가능한 object마다 polygon 하나를 만든다. Compound class와 cluster class를 만들지 않는다.

## 8. Scratch

배경 noise와 구분되는 선형 흔적의 실제 영역만 그린다. 경계 재현성이 낮은 tiny/border scratch는 HOLD한다.

## 9. Thickening, short, deformation

인접 feature가 실제 연결되면 `short`다. 연결되지 않고 폭만 증가하면 `pattern_thickening`이다. Open/short/narrowing/thickening으로 설명되지 않는 실제 pattern geometry 변형만 `deformation`으로 지정한다. Acquisition-only failure는 defect class로 만들지 않는다.

## 10. Pilot에서 Full annotation으로

TRAIN Pilot 36장과 Pilot QA는 완료했고 현재 guideline v2를 freeze했다. 다음 승인 단계에서 TRAIN 전체를 진행하고, 이어서 VAL, 마지막으로 TEST를 annotation한다. Exact duplicate는 SHA-256 unique content 대표 한 장만 사람이 annotation한다. Label 재사용은 이번 단계에서 수행하지 않는다.

## 11. Annotation 후 QA

Annotation 완료 후 object class ID, polygon closure, 정상 pattern 과다 mask, multi-instance 누락, ambiguous/HOLD 처리, duplicate label 재사용 가능성, class별 image/instance count를 점검한다. 이 QA 후 split의 severe class absence만 확인하고 학습 전 split을 최종 확정한다.
