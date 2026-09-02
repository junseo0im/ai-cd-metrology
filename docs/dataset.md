# 데이터셋

## 개요

데이터셋은 광학 현미경으로 촬영한 반도체 IDA(Interdigital Array) 패턴 이미지로 구성되어
있습니다. CD 측정과 패턴 결함 검출 방법을 개발하기 위한 초기 데이터입니다.

## 구성

| 구분 | 수량 |
|---|---:|
| 전체 이미지 | 223 |
| 표준 촬영 이미지 | 216 |
| 결함 확대 이미지 | 7 |

표준 촬영 이미지는 다음 구조로 구성되어 있습니다.

```text
3 wafers × 24 dies × 3 positions = 216 images
```

- Wafer: `w1`, `w2`, `w3`
- Die: wafer별 3 rows × 8 columns
- Position: `center`, `left-top`, `right-bottom`

## 라벨

표준 촬영 이미지의 기존 라벨 분포입니다.

| 판정 | 결함 유형 | 수량 |
|---|---|---:|
| PASS | normal | 178 |
| WARNING | not critical | 12 |
| FAIL | particle | 12 |
| FAIL | pattern die | 6 |
| FAIL | pattern narrowing | 7 |
| FAIL | short | 1 |

결함 확대 이미지는 open 1장, particle 3장, pattern die 1장, pattern narrowing 1장,
short 1장으로 구성되어 있습니다.

## 파일명

표준 촬영 이미지의 파일명은 wafer, die 위치, 촬영 위치와 결함 정보를 포함합니다.

```text
w{wafer}_{die_row}{die_col}_{position}[_{defect}].png
```

예시:

```text
w1_11_c.png
w3_11_lt_p.png
w1_12_rb_pn.png
```

| 구분 | 코드 |
|---|---|
| 촬영 위치 | `c`: center, `lt`: left-top, `rb`: right-bottom |
| 결함 | `p`: particle, `pd`: pattern die, `pn`: pattern narrowing |
| 기타 | `s`: short, `nc`: not critical |

## 이미지 정보

- Format: PNG
- Standard images: RGB
- Defect close-ups: RGBA
- Standard image size: approximately 2295–2310 × 1246–1261 px

## 참고사항

- 기존 라벨은 폴더와 파일명에 기록된 정보를 기준으로 정리한 값입니다.
- 물리 단위 CD 계산에 필요한 pixel calibration 정보는 아직 확보되지 않았습니다.
- open과 short 이미지 수가 적어 결함 유형별 데이터 불균형이 큽니다.
- 결함 확대 이미지는 촬영 위치가 없어 표준 촬영 이미지와 구분해 사용합니다.
- 학습 및 평가 데이터를 나눌 때 같은 die의 이미지가 서로 다른 split에 들어가지 않도록
  구성할 예정입니다.
