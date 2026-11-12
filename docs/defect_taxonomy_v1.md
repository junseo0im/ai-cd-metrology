# Defect Taxonomy v1

## Freeze status

HIGH 32개 원본 human adjudication을 반영한 v1 operational taxonomy다. 기존 source label은 provenance로 보존하며 자동으로 canonical class가 되지 않는다.

| Class | 정의 | 포함 | 제외 | 구분 기준 |
| --- | --- | --- | --- | --- |
| `normal` | 정상 반복 pattern이며 검출할 이상이 없는 이미지 | 안정적인 반복 line과 허용 가능한 배경 | 검출 대상 scratch, particle, pattern 이상 | normal image는 polygon 없음 |
| `open` | 의도된 pattern이 국소적으로 끊어진 결함 | 명확한 line break 또는 open region | crop, border truncation, blur | material이 연결되지 않은 실제 단절 |
| `short` | 인접 feature가 의도치 않게 실제 연결된 결함 | bridge 또는 neck connection | 단순 blur로 붙어 보이는 현상 | 연결되면 thickening보다 short 우선 |
| `pattern_narrowing` | 주변 반복 pattern보다 intended width가 명확히 감소한 결함 | 국소 또는 연속 narrowing과 transition | 전체 scale 차이, defocus | 폭 감소 방향의 geometry 이상 |
| `pattern_thickening` | 주변 반복 pattern보다 intended width가 명확히 증가한 결함 | 국소 또는 연속 thickening, 인접 feature에 거의 닿는 상태 | 실제 연결, blur/defocus, 전체 image scale 차이 | 연결되면 short, 연결되지 않고 폭만 증가하면 thickening |
| `deformation` | open/short/narrowing/thickening으로 충분히 설명되지 않는 실제 pattern geometry 변형 | 휘어짐, 비정상 형상, physical pattern failure | acquisition-only failure | 구조 변화가 원본에서 확인돼야 함 |
| `particle` | 검출 대상으로 승인된 국소 이물질/오염 물체 | pattern 위·사이·edge에 붙은 분리 가능한 물체 | optical noise, 불명확한 얼룩 | 분리 가능한 물체별 instance, 내부 모양으로 subtype을 만들지 않음 |
| `scratch` | wafer 표면에서 배경 noise와 구분되는 선형 흔적 | native 원본에서 반복 가능한 polygon 경계가 보이는 mark | optical streak, 불안정한 극소 흔적 | 고정 pixel threshold 없이 boundary 재현성으로 결정 |

## v1 decisions

- `pattern_thickening`은 visual index 23, 493, 532를 일관되게 설명하므로 별도 candidate class로 채택한다.
- particle과 scratch는 이번 프로젝트의 검출 대상이다.
- 복수 defect는 `MULTI`라는 object class로 합치지 않는다. Manifest의 `MULTI`는 image-level 표시이며 실제 annotation은 visible defect별 canonical class/instance로 나눈다.
- `pattern_die`, `nc`, `unlabeled`는 source/review label로만 보존한다. 사례별로 위 canonical class, quality issue 또는 UNCERTAIN으로 재판정한다.
- visual index 100은 tiny-scratch borderline으로 TBD다. visual index 556은 UNCERTAIN으로 보존하고 v1 training에서 제외한다.
