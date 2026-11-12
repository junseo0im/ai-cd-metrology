# CVAT Pilot Annotation Checklist v1

## 이미지 한 장을 열었을 때

1. 원본을 전체적으로 확인한다.
2. 실제 결함이 있는지 판단한다.
3. 7개 class 중 class를 선택한다.
4. Defect 실제 영역만 polygon으로 그린다. 정상 pattern 전체를 mask하지 않는다.
5. 여러 defect면 각각 별도 polygon을 만든다.
6. 같은 particle이 여러 개면 object별 polygon을 만든다.
7. Normal이면 polygon을 만들지 않고 review log에서 `NORMAL`로 확인한다.
8. 애매하면 억지로 annotation하지 않고 `HOLD`로 기록한다.
9. 저장 후 다음 이미지로 이동한다.

추가 class와 수작업 BBox를 만들지 않는다. Border object는 이미지에서 보이는 영역까지만 그린다. Tiny scratch는 경계가 안정적일 때만 그린다.

## Class quick reference

- `open`: 실제 pattern 단절
- `short`: 인접 pattern 실제 연결
- `pattern_narrowing`: 주변 대비 폭 감소
- `pattern_thickening`: 주변 대비 폭 증가. 실제 연결되면 `short`
- `deformation`: 위 네 class로 설명되지 않는 실제 pattern geometry 변형
- `particle`: 분리 가능한 이물질 또는 오염 object
- `scratch`: background noise와 구분되는 wafer 표면의 선형 흔적
- `normal`: defect object 없음. YOLO class가 아님

## Review log 입력

- `review_status` 초기값은 `NOT_REVIEWED`다.
- `final_image_decision` 허용값은 `DEFECT`, `NORMAL`, `HOLD`다.
- `NORMAL`은 사람이 원본을 확인해 defect가 없다고 판정한 경우에만 기록한다. Polygon이 없다는 사실만으로 자동 `NORMAL` 처리하지 않는다.
