# YOLOv8-seg Annotation Guideline v1

## Primary rule

Segmentation polygon을 primary annotation으로 사용한다. 별도 수작업 BBox는 만들지 않는다.

1. defect의 localized region만 polygon으로 표시한다.
2. 정상 반복 pattern 전체를 mask하지 않는다.
3. normal image에는 polygon을 만들지 않는다.
4. 한 이미지에 여러 defect가 있으면 defect별로 polygon을 나눈다.
5. class가 다르면 각 class의 별도 instance로 기록한다.
6. particle이 여러 개면 실제로 분리되는 object별 instance로 기록한다.
7. compound filename label을 하나의 YOLO class로 만들지 않는다.
8. border defect는 이미지 안에서 보이는 영역까지만 polygon을 닫는다.
9. tiny scratch는 native 원본에서 경계가 안정적으로 반복 가능할 때만 annotation한다. 고정 pixel threshold는 사용하지 않는다.
10. reference, evaluation-only, 명시적 EXCLUDE 이미지는 annotation하지 않는다. Screenshot은 기본 제외하되 human adjudication에서 KEEP된 visual index 23은 승인된 v1 예외다.
11. acquisition/process-quality issue 자체는 YOLO physical defect class로 넣지 않는다.
12. 애매한 instance에 새 class를 만들지 않는다. UNCERTAIN 또는 annotation hold로 남긴다.

## Class separation

- 실제 연결은 `short`, 연결되지 않고 폭만 증가하면 `pattern_thickening`이다.
- `pattern_narrowing`은 폭 감소, `deformation`은 나머지 physical geometry 변형이다.
- particle과 scratch가 함께 있으면 각각 별도 polygon을 만든다.
- Manifest의 `canonical_class=MULTI`는 image-level provenance이며 YOLO class가 아니다.

## Annotation eligibility

- `READY_FOR_POLYGON`, `READY_FOR_OBJECT_REVIEW`: 원본 확인 후 annotation 가능.
- `NO_POLYGON_NORMAL`: 학습 가능한 normal image이며 polygon 없음.
- `HOLD_*`: class 또는 polygon boundary 결정 전 annotation 보류.
- `EXCLUDED*`: v1 annotation 및 training 제외.
