# Release Manifest

## Scope and provenance

6개 완료 실험의 ORIGINAL 소스에서 만든 공개용 복사본입니다. RECONSTRUCTED 코드는 없습니다.
아래 source 경로는 원래 프로젝트 root 기준이며 개인 PC 절대경로가 아닙니다.
공개본은 주석/ROOT 탐색/helper module 파일명 경로만 조정했습니다. 학습 함수·설정·metric·best 선택 로직은 원본과 동일합니다.
원본과 공개본의 AST를 비교하여 허용한 경로/import 수정 외 차이가 없음을 확인했습니다.

| Experiment | Public/source script | Status | Config check | Syntax/import resolution |
|---|---|---|---|---|
| YOLOv8n Detection Real-only Baseline | `scripts/train_yolov8_detection_baseline_v1.py` | ORIGINAL | PASS | PASS |
| YOLOv8n Detection Rare-class Oversampling v1 | `scripts/train_yolov8_oversampling_v1.py` | ORIGINAL | PASS | PASS |
| MobileNetV3-Small Real-only Baseline | `scripts/train_mobilenet_baseline_v1.py` | ORIGINAL | PASS | PASS |
| MobileNetV3-Small Rare-class Oversampling v1 | `scripts/train_mobilenet_oversampling_v1.py` | ORIGINAL | PASS | PASS |
| ResNet18 Real-only Baseline | `scripts/train_resnet18_baseline_v1.py` | ORIGINAL | PASS | PASS |
| ResNet18 Rare-class Oversampling v1 | `scripts/train_resnet18_oversampling_v1.py` | ORIGINAL | PASS | PASS |

MobileNet baseline은 원래 실행명 `train_mobilenet_real_only_v1.py`를 보존한 byte-identical 원본 alias `scripts/train_mobilenet_baseline_v1.py`에서 복사했습니다.
원본 archive는 `runs/mobilenet/mobilenet_v3_small_real_only_v1/train_mobilenet_real_only_v1.py`입니다.
MobileNet oversampling과 ResNet18 2개는 원래 script와 run archive의 byte SHA256 및 saved summary의 script hash가 일치합니다.
YOLO 2개는 기존 source와 code-recovery/config 기록이 일치합니다. 실행 당시 script hash/archive는 없으므로 historical byte identity까지 증명한 것은 아닙니다.

## Exact path-only modifications

- 모든 public script: 요청한 release-copy provenance 주석 추가.
- YOLO 2개: font 경로의 고정 OS-directory fallback을 WINDIR/SystemRoot 환경변수 참조로 변경. 학습 파라미터와 무관한 경로 처리입니다.
- `train_mobilenet_baseline_v1.py`: pyproject 탐색을 script 상위 root 상대경로로 변경. 별도 pyproject 없이 공개 디렉터리 root를 찾습니다.
- `train_mobilenet_oversampling_v1.py`, `train_resnet18_baseline_v1.py`: helper import 경로 `train_mobilenet_real_only_v1`를 공개 파일명 `train_mobilenet_baseline_v1`로 연결. import alias는 유지했습니다.
- 나머지 코드, 모델, preprocessing, factor, RNG/seed, optimizer, threshold, aggregation, metric 및 checkpoint 정책 변경 없음.

## Source result files

| Experiment | Source summary | Public summary |
|---|---|---|
| YOLOv8n Detection Real-only Baseline | `data/derived/yolov8_baseline_v1/baseline_summary.json` | `results/yolo/baseline_summary.json` |
| YOLOv8n Detection Rare-class Oversampling v1 | `data/derived/yolov8_oversampling_v1/oversampling_summary.json` | `results/yolo/oversampling_summary.json` |
| MobileNetV3-Small Real-only Baseline | `data/derived/mobilenet_baseline_v1/mobilenet_baseline_summary.json` | `results/mobilenet/baseline/summary.json` |
| MobileNetV3-Small Rare-class Oversampling v1 | `data/derived/mobilenet_oversampling_v1/mobilenet_oversampling_summary.json` | `results/mobilenet/oversampling/summary.json` |
| ResNet18 Real-only Baseline | `data/derived/resnet18_baseline_v1/resnet18_baseline_summary.json` | `results/resnet18/baseline/summary.json` |
| ResNet18 Rare-class Oversampling v1 | `data/derived/resnet18_oversampling_v1/resnet18_oversampling_summary.json` | `results/resnet18/oversampling/summary.json` |

Classifier 4개는 각 source summary와 같은 디렉터리의 `classification_report.csv`, `confusion_matrix.csv`를 results 하위에 byte-identical하게 복사했습니다.
각 `confusion_matrix.png`도 figures의 해당 experiment 이름으로 byte-identical하게 복사했습니다.
JSON은 개인 경로가 포함되어 있어 원본을 덮어쓰지 않고 metric/필요 설정을 추출했습니다. Path field, raw hash inventory와 runtime log는 제외했습니다.
Metric 수치는 원본 JSON과 재귀적으로 일치하는지 검증했습니다. 공개 JSON의 `script_sha256`는 과거 실행본의 provenance이지 아래 공개본의 hash가 아닙니다.

## Source/public code SHA256

### train_yolov8_detection_baseline_v1.py

- Source: `0edf828c1f37ebcf930ed2d402a567d8b598b208456c443bd6e9dba2ed2fcce6`
- Public: `3a87c6583802e2ed92d8227c2b3868d2bec8244d563863f265d87d5e32a366c7`

### train_yolov8_oversampling_v1.py

- Source: `3cf7658ca230c4644945cc1a136400caa21a0b3f306f43cf477c90a9ffe55ba7`
- Public: `c5d8e6fc2c92733172b6b0ae381d4422685e6eb0ab598cb79e638efd83201a40`

### train_mobilenet_baseline_v1.py

- Source: `aee736f8cece7bf1da4625b085d1fe896a47ce48d8e13b8d537de81756f10a74`
- Public: `3a3707eb1abd5b7ceaad8a3101aa6d04e09e10752a332c54d2f6c0b27e3a8d92`

### train_mobilenet_oversampling_v1.py

- Source: `2220fdf1ac5f434e25f0baabea543aaf14831242f1d9da62fdbfdc7c7314e4de`
- Public: `8d1571fb40678f316d4051d93665d9c3baa0c9297db0b90c2630f45eb03c6cb3`

### train_resnet18_baseline_v1.py

- Source: `573ab606e9b7cfb82e564183593980cd46ee39909f9fe0d76a7049485b94fbc3`
- Public: `ae4e16b3a91e7bfeb84539c1cc07e3cc662d7fdd4c479d48f4b1033701354467`

### train_resnet18_oversampling_v1.py

- Source: `f0b50e92de3b5074520740dfc11f5377e10b6fbf425bcab8a29751cfcc3e4c4f`
- Public: `95199fb8e7eda785da225cce6ce49e211abf4462ef39f9b28a0df9592fd08906`

## Static checks

- Scripts 6/6: `ast.parse` + in-memory `compile(..., "exec")` PASS. 코드 객체를 실행하지 않았고 pyc/cache를 만들지 않았습니다.
- Local helper import 6/6 및 사용 package/stdlib resolution PASS. 패키지 경로/설치 metadata로 확인했으며 ML module runtime import는 수행하지 않았습니다.
- Symbol-table 검사에서 명백한 미정의 global name 0.
- 경로/import 수정만 정규화한 원본/public AST 동일: 6/6.
- Saved classifier config/summary와 original archive SHA256 동일: 4/4. YOLO config/source consistency는 기존 recovery evidence와 확인했습니다.
- Training, inference, model loading 및 Test 평가를 실행하지 않았습니다.

## Reproduction prerequisites and limitations

이 폴더만으로 원본 데이터가 필요한 학습을 바로 실행할 수 있다고 주장하지 않습니다.
Fixed split/crop manifests와 해당 audit metadata, private data, pretrained weights, 일부 선행 실험 full artifacts가 별도로 필요합니다.
YOLO의 external dataset QA helper `scripts/convert_coco_detection_fixed_v1.py --verify-only`는 의존성으로 남겨 두었으며 최소 6개 training code release에는 포함하지 않았습니다.
후속 classifier/oversampling 코드의 기존 baseline SHA256·overwrite guard는 그대로 유지했습니다.
따라서 공개 수정본과 historical full artifacts를 섞으면 hash 검증이 실패할 수 있으며, 이를 우회하거나 과거 source hash를 위조하지 않았습니다.
Public summary는 경로를 제외한 보고서이므로 original PREPARED/full run-state 대신 사용할 수 없습니다.
End-to-end rerun은 이번 범위 밖이므로 검증하지 않았습니다.

## Exclusions

원본 이미지/crop/dataset, pretrained/checkpoint(best/last 포함), run/cache/log, learning curve, contact sheet를 포함하지 않았습니다.
원본 프로젝트 파일은 수정하지 않고 새 `git_release_v1/`만 생성했습니다.
Git init/stage/commit/push 및 GitHub repository 생성을 수행하지 않았습니다.

