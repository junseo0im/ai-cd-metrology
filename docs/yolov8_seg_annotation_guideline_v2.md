# YOLOv8-seg Annotation Guideline v2

## 1. 운영 상태와 적용 범위

**FREEZE — 현재 full TRAIN / VAL / TEST annotation 운영 기준.** 이번 반영은 guideline 확정이며 full annotation, augmentation 또는 baseline training을 시작한 것이 아니다.

이번 승인 요청의 확정 정책을 기준으로 한다. Checkpoint ZIP의 `docs_draft/`는 분석 과정의 provenance이며 이 문서를 대체하지 않는다. [v1 guideline](yolov8_seg_annotation_guideline_v1.md)은 삭제하거나 덮어쓰지 않고 과거 기준으로 보존한다.

Segmentation polygon이 primary annotation이다. 별도 수작업 BBox는 만들지 않는다. 원본 image와 original CVAT polygon GT를 수정·삭제·덮어쓰지 않는다. 분석용 자동 candidate와 사람이 확정한 판정은 별도 자료로 유지한다.

## 2. YOLO object classes

| ID | Class |
| ---: | --- |
| 0 | `open` |
| 1 | `short` |
| 2 | `pattern_narrowing` |
| 3 | `pattern_thickening` |
| 4 | `deformation` |
| 5 | `particle` |
| 6 | `scratch` |

`normal`, `HOLD`, `IGNORE`, morphology subtype은 YOLO class가 아니다. `MULTI`, compound filename, `pattern_die`, `nc`, `unlabeled`도 object class로 직접 사용하지 않는다. Filename/folder/source label은 provenance이며 실제 annotation ground truth를 대신하지 않는다.

Normal은 defect instance가 없는 사람이 확인한 negative image이며 polygon을 만들지 않는다. 한 이미지에 여러 class/instance가 있으면 실제로 분리되는 결함마다 별도 polygon을 만든다.

## 3. 공통 polygon / eligibility 원칙

- 결함의 localized visible region만 표시한다. 주변 정상 반복 pattern 전체를 mask하지 않는다.
- 서로 분리되는 object는 같은 class여도 별도 instance로 기록한다. Compound/cluster class를 만들지 않는다.
- Border object는 이미지 안에서 보이는 영역까지만 polygon을 닫는다.
- Class 또는 boundary가 불명확하면 새 class나 임의 boundary를 만들지 않고 HOLD metadata로 남긴다.
- Reference, CD evaluation-only 및 명시적 EXCLUDE asset은 annotation/training 대상이 아니다.
- Screenshot은 기본 제외하되 human-approved visual index 23의 기존 예외를 유지한다. Acquisition/process-quality issue 자체를 physical defect class로 만들지 않는다.

## 4. Particle target selection과 boundary

Fixed minimum pixel cutoff는 없다. 식별 가능성과 **boundary reproducibility**가 primary 기준이다. Native image에서 시각적으로 구분되는 이물질이고 visible boundary를 재현 가능하게 그릴 수 있으면 작더라도 target으로 유지한다. Pattern과 가까이 있거나 edge에 붙었다는 이유만으로 IGNORE하지 않는다.

Particle polygon은 reproducible visible outer boundary를 따른다. Blur, halo, shadow, illumination variation을 particle boundary에 포함하지 않는다. 희미하거나 경계가 안정적으로 반복되지 않는 candidate는 재검토하고 HOLD 또는 IGNORE candidate로 기록한다. Medium/large low-contrast object도 수치만으로 자동 제외하지 않는다.

`d640 < 7.30 px AND Lab ΔE < 10.20`은 **pilot review triage signal**일 뿐 final cutoff가 아니다. `d640`은 nominal 640 px input으로 투영한 equivalent diameter이며 native pixel cutoff가 아니다. Pattern-relative size와 자동 size/contrast 결과도 annotation 포함/제외를 단독 결정하지 않는다. 자동 필터나 label 삭제 규칙은 이 문서로 승인되지 않는다.

## 5. Hollow / loop-like particle

YOLO class는 계속 `particle`이다. Baseline segmentation은 visible **outer silhouette 한 instance**로 유지한다. 내부가 비어 있거나 loop/fiber-like인 특성은 morphology metadata로 기록하며 별도 YOLO class나 내부 hole polygon을 만들지 않는다.

Edge attachment는 context이며 hollow 여부나 morphology subtype과 별도로 기록한다. `P024 #3`은 `edge_attached`, `internally_filled`, `NOT_HOLLOW`로 사람이 확인했다. 이 판단만으로 기존 자동 morphology 후보를 human-confirmed subtype으로 승격하지 않는다.

## 6. Secondary morphology metadata

| Metadata value | 육안 판단 의미 |
| --- | --- |
| `compact` | 비교적 응집된 blob-like 외형 |
| `elongated` | 한 방향으로 뚜렷하게 늘어난 외형 |
| `irregular` | 비대칭, 오목/다엽 또는 복잡한 불규칙 외곽 |
| `hollow_loop_like` | 비어 있거나 loop/fiber-like 내부 특성이 확인된 particle |
| morphology `HOLD` | subtype을 안정적으로 판단하기 어려움 |

이는 secondary metadata이며 YOLO class split이 아니다. Detection-target KEEP/HOLD/IGNORE와 morphology 판단은 별개다. Morphology HOLD만으로 confirmed particle polygon을 삭제하지 않는다. Aspect ratio/circularity/solidity 기반 자동 후보는 review 보조 자료이고 frozen subtype threshold가 아니다. 크기, contrast 또는 pattern-relative scale을 morphology subtype과 혼동하지 않는다.

사람 판정은 자동 candidate table을 덮어쓰지 않는 별도 [morphology review metadata](../data/metadata/particle_morphology_human_review_v1.csv)와 [hollow review metadata](../data/metadata/particle_hollow_human_review_v1.csv)에 기록한다. `P008 #7`의 수동 subtype은 `hollow_loop_like`다. 자동 grouping/clustering 수치의 상충은 원본 분석 provenance에 남기며 v2 운영 규칙으로 채택하지 않는다.

`instance_index`는 **해당 이미지 원본 YOLO label file 전체 polygon의 1-based row index**다. Particle-only index로 재번호화하지 않는다. `file_hash + label_file + instance_index`로 original export와 판정을 추적한다.

## 7. Open / short / width / deformation

우선순위는 **topology change > width change > waviness**다.

1. 실제 연결이 끊기면 `open`.
2. 원래 분리돼야 할 구조가 실제로 비정상 연결되면 `short`.
3. Connectivity가 유지되면서 local width가 감소하면 `pattern_narrowing`.
4. Connectivity가 유지되면서 local width가 증가하면 `pattern_thickening`.
5. 위 class로 설명되지 않는 실제 pattern geometry 변형은 `deformation`.

단 한 번의 local notch/bump도 명확하고 재현 가능하면 narrowing/thickening에 포함한다. 단순 waviness/edge variation만으로 label하지 않는다. 밝기나 source label만으로 topology/width 변화를 단정하지 않는다. 서로 다른 localized defect가 있으면 각각 class와 polygon을 기록한다.

## 8. Scratch

Background noise와 구분되는 선형 표면 흔적의 실제 visible region만 annotation한다. Particle과 함께 있으면 각각 별도 polygon을 만든다. Tiny/border scratch는 native image에서 boundary가 안정적으로 반복 가능할 때만 annotation하며 고정 pixel cutoff는 사용하지 않는다. Identity 또는 boundary가 불명확하면 HOLD한다.

## 9. KEEP / HOLD / IGNORE와 training 경계

- `KEEP`: 재현 가능한 detection target.
- `HOLD`: class 또는 boundary에 추가 판단이 필요한 상태.
- `IGNORE candidate`: 재검토 후에도 target annotation reliability가 낮은 candidate.

HOLD/IGNORE는 metadata이며 object class가 아니다. Confirmed polygon과 original GT는 계속 보존한다. Image-level HOLD 안의 confirmed instances도 보존하며 미확정 anomaly를 임의로 class에 넣지 않는다.

**IGNORE candidate의 실제 training-time 처리 방식은 아직 결정하지 않았다.** Annotation 누락으로 background 학습을 유도하거나 polygon을 삭제하는 정책을 암묵적으로 적용하지 않는다. Training 준비 단계에서 별도 처리 원칙을 정해야 하며 이번에는 filtering, training-label 생성 또는 학습을 실행하지 않는다.

## 10. Pilot evidence와 현재 판정 우선순위

현재 Pilot 운영 상태는 [completed review log](../data/metadata/defect_annotation_pilot_review_v1_completed_draft.csv)를 우선한다. `_completed_draft` 파일명은 package 원본 이름을 보존한 것이며 기록된 36개 row는 모두 REVIEWED다. 초기 review log, HIGH human review, inventory/queue/pilot selection은 pre-annotation history로 보존하고 소급 수정하지 않는다.

| 항목 | 검증 결과 |
| --- | --- |
| Pilot images / mapping match | 36 / 36 of 36 |
| Label files | 34 |
| Label-free NORMAL | P010, P025 |
| Polygon instances | 221 |
| Completed image review | DEFECT 29, HOLD 5, NORMAL 2 |
| Class instances | open 4, short 21, narrowing 4, thickening 2, deformation 1, particle 152, scratch 37 |
| Mechanical QA | class ID, coordinate range, <3 points, degenerate/self-intersection invalid, exact duplicate: 모두 0 |

과거/현재 차이: P009는 현재 particle 6개의 DEFECT, P014/P015는 confirmed particle을 보존하는 image-level HOLD, P016은 short + particle, P025는 NORMAL이다. 이 판정으로 과거 source/history를 덮어쓰지 않는다.

Original archive: [defect_pilot_v1_yolo_seg.zip](../data/derived/cvat_pilot_v1/export/defect_pilot_v1_yolo_seg.zip). 자동 intermediate CSV와 review PNG는 `data/derived/cvat_pilot_v1/analysis/`에 둔다. Human target 판정은 [manual decisions](../data/metadata/particle_pilot_manual_decisions_v3.csv)로 분리한다. Transfer ZIP과 `docs_draft/`는 분석 provenance이며 canonical 운영 기준은 이 문서다.

## 11. 다음 단계

현재 **[9] Annotation Guideline v2 FREEZE**까지 완료했다. 이후 순서는 [10] TRAIN 전체 annotation → [11] VAL annotation → [12] TEST annotation → [13] 최종 dataset QA → [14] YOLOv8-seg baseline이다. TEST는 tuning에 사용하지 않는다. 기존 [split draft policy](defect_split_policy_v1.md)를 재생성하지 않으며 학습 전 최종 QA에서 class/instance 분포와 severe class absence를 확인한다.

이번 반영으로 [10] 이후 작업, augmentation 또는 YOLO baseline training을 시작하지 않는다. Method 2 production source/config/dashboard/CD algorithm에는 변경이 없다.
