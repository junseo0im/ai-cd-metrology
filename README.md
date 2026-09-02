# AI-Based CD Metrology and Defect Detection System

> AI 기반 반도체 선폭 계측 및 결함 검출 시스템

광학 현미경 이미지에서 반도체 패턴의 CD(Critical Dimension)를 측정하고 결함을 검출하는
AI 기반 검사 시스템입니다.

## 프로젝트 소개

반도체 패턴의 선폭과 형상은 공정 품질을 판단하는 주요 지표입니다. 이 프로젝트는 현미경
이미지 분석을 통해 반복적인 선폭 측정을 자동화하고, 패턴의 이상을 탐지하는 것을 목표로
합니다.

주요 개발 항목은 다음과 같습니다.

- 반도체 패턴 이미지 전처리 및 영역 추출
- CD 측정과 측정 결과 분석
- particle, short, open 등 패턴 결함 검출
- wafer와 die 단위 결과 시각화

## 데이터셋

광학 현미경으로 촬영한 반도체 IDA(Interdigital Array) 패턴 이미지를 사용합니다.

| 구분 | 수량 | 설명 |
|---|---:|---|
| 전체 이미지 | 223 | PNG |
| 표준 촬영 이미지 | 216 | 3 wafers × 24 dies × 3 positions |
| 결함 확대 이미지 | 7 | 결함 부위를 별도로 촬영한 참고 이미지 |

표준 촬영 이미지는 `center`, `left-top`, `right-bottom` 세 위치로 구성되어 있습니다. 기존
폴더와 파일명에 PASS, WARNING, FAIL 및 결함 유형이 구분되어 있으나, 라벨 기준은 추가
검토가 필요합니다.

자세한 구성은 [데이터셋 설명](docs/dataset.md)에서 확인할 수 있습니다.

## 개발 계획

1. 데이터셋 구조와 라벨 정리
2. CD 측정 방법 개발
3. 결함 검출 모델 개발
4. wafer map과 검사 결과 시각화

세부 계획은 [로드맵](docs/roadmap.md)에 정리되어 있습니다.
CD 측정 방법과 평가 기준은 [CD 측정 문서](docs/cd_measurement.md)에 정리되어 있습니다.
