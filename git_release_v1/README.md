# Optical Microscopy Defect Baselines

## 1. Project Overview

20× optical microscopy 이미지의 결함을 분석하는 연구 프로젝트입니다.
현재 taxonomy는 `p`(particle), `o`(open), `s`(short), `sc`(scratch)입니다.
YOLOv8n은 whole-image object detection을, MobileNetV3-Small과 ResNet18은 GT bbox crop classification을 수행합니다.
이 release는 완료된 real-only / rare-class oversampling 실험 6개의 코드와 핵심 결과만 모은 기록입니다.

## 2. Dataset

- 629 source images, 5,953 annotations: p 5,444 / o 3 / s 7 / sc 499.
- Fixed detection split: Train 505 / Val 62 / Test 62 images.
- 동일 source split을 따르는 GT crop: Train 4,758 / Val 597 / Test 598.
- Unique Train crop: p 4,352 / o 2 / s 5 / sc 399. Class imbalance가 매우 큽니다.
- Source image, crop, dataset, checkpoint는 이 release에 포함하지 않습니다.
- Detection class ID order는 `[o, p, s, sc]`, classification은 `[p, o, s, sc]`입니다.

## 3. Experiments

| Model | Training | Main Test Metric |
|---|---|---|
| YOLOv8n | Real-only | mAP50 0.1581 |
| YOLOv8n | Oversampling | mAP50 0.2728 |
| MobileNetV3-Small | Real-only | Macro-F1 0.4601 |
| MobileNetV3-Small | Oversampling | Macro-F1 0.4533 |
| ResNet18 | Real-only | Macro-F1 0.4504 |
| ResNet18 | Oversampling | Macro-F1 0.4230 |

## 4. Detailed Results

### YOLO Detection

| Experiment | Precision | Recall | mAP50 | mAP50-95 | sc AP50 |
|---|---:|---:|---:|---:|---:|
| Real-only | 0.7742 | 0.1526 | 0.1581 | 0.0566 | 0.1992 |
| Oversampling | 0.8092 | 0.1861 | 0.2728 | 0.1159 | 0.3179 |

이 고정 Test에서 rare-class Train image 반복 노출 후 detection mAP와 scratch AP가 증가했습니다.
Test open/short GT는 각각 1개뿐이므로 rare-class 수치로 일반화 성능을 판단하지 않습니다.

### Crop Classification

| Model | Training | Accuracy | Macro F1 | sc Precision | sc Recall | sc F1 |
|---|---|---:|---:|---:|---:|---:|
| MobileNetV3-Small | Real-only | 0.9749 | 0.4601 | 0.9744 | 0.7600 | 0.8539 |
| MobileNetV3-Small | Oversampling | 0.9716 | 0.4533 | 0.9730 | 0.7200 | 0.8276 |
| ResNet18 | Real-only | 0.9649 | 0.4504 | 0.8200 | 0.8200 | 0.8200 |
| ResNet18 | Oversampling | 0.9415 | 0.4230 | 0.6232 | 0.8600 | 0.7227 |

전체 precision/recall/F1 및 confusion matrix는 `results/`에 저장되어 있습니다.
Confusion matrix의 행은 GT, 열은 prediction이며 순서는 `p, o, s, sc`입니다.

## 5. Key Findings

1. 이 고정 evaluation set에서 YOLO oversampling의 detection mAP가 real-only보다 높았습니다.
2. MobileNet과 ResNet18의 단순 반복 oversampling은 Test Macro-F1을 개선하지 못했습니다.
3. 비교한 classifier 4개 중 MobileNet real-only의 Test Accuracy와 Macro-F1이 가장 높았습니다.
4. ResNet18은 대응하는 MobileNet 실험보다 scratch Recall이 높고 Precision은 낮았습니다.
5. Unique Train o=2/s=5, Test o=1/s=1이므로 o/s 일반화 성능은 판단할 수 없습니다. 반복은 새 morphology를 추가하지 않습니다.

## 6. Important Metric Note

**YOLO와 MobileNet/ResNet의 metric 크기를 직접 비교하지 않습니다.**
YOLO는 전체 이미지에서 위치와 class를 함께 예측하는 detection이며 mAP를 사용합니다.
Classifier는 정답 bbox가 이미 주어진 crop의 class만 예측하며 Accuracy/F1을 사용합니다.
GT box를 사용하는 classification은 end-to-end detection 평가가 아닙니다.

## 7. Repository Structure

```text
scripts/  # 6개 공개용 학습 코드 및 공통 helper 기능
results/yolo/  # baseline / oversampling summary JSON
results/mobilenet/{baseline,oversampling}/  # report CSV, confusion CSV, summary JSON
results/resnet18/{baseline,oversampling}/   # report CSV, confusion CSV, summary JSON
figures/  # classifier confusion matrix 4개
docs/experiment_summary.md  # 실험 조건과 결과
docs/release_manifest.md    # 원본 코드·결과·검증 provenance
```

## 8. Reproducibility

- 공통 seed 42. 기록된 실행환경은 Python 3.11.9, CPU입니다.
- YOLO: official `yolov8n.pt`, imgsz 640, batch 8, max 100 epochs, patience 20, workers 0.
  Ultralytics optimizer/lr 기본 정책을 유지했고 실제 resolved optimizer는 AdamW/lr0=0.00125입니다.
- Classifier: official ImageNet weights, 전체 network fine-tuning, RGB / direct bilinear antialiased 224×224 resize / Tensor / ImageNet normalization.
  AdamW lr=1e-3, weight_decay=1e-4, unweighted CrossEntropyLoss, batch 32, max 50 epochs, patience 10, workers 0, deterministic=True.
- MobileNet weight: `MobileNet_V3_Small_Weights.IMAGENET1K_V1`.
  ResNet: `ResNet18_Weights.DEFAULT`가 `IMAGENET1K_V1`로 resolve되었습니다.
- Classifier best 기준은 Val macro-F1 최대이며 strict improvement / earliest tie입니다.
  고정 4개 class를 모두 포함하고 zero_division=0입니다. Val open support는 0입니다.
- YOLO best 선택은 설치된 Ultralytics detection fitness 정책입니다. Classifier F1 정책과 혼동하지 않습니다.
- 추가 augmentation은 OFF입니다. YOLO 기본 augmentation도 명시적으로 OFF이며 적용 값은 JSON에 보존했습니다.
- Oversampling: o×10 / s×5 / sc×2 / other×1.
  YOLO는 image 단위 최대 factor를 사용하여 505→764 samples, classifier는 crop 단위 4,758→5,195 indices를 사용했습니다.
- 공개 코드의 ROOT는 각 script의 상위 repository root입니다.
  MobileNet helper import는 공개 파일명 `train_mobilenet_baseline_v1`로 연결했습니다.
  학습 조건과 알고리즘은 변경하지 않았습니다.

### 실행 재현의 전제

이 release는 데이터가 포함된 standalone training bundle이나 resume snapshot이 아닙니다.
원래 구조의 fixed manifests, private datasets/audit metadata, official pretrained weights 및 선행 실험 artifact가 별도로 필요합니다.
YOLO 코드는 외부 QA helper `scripts/convert_coco_detection_fixed_v1.py --verify-only`에 의존하며 이 최소 release에는 포함하지 않습니다.
Classifier 후속 실험은 선행 baseline의 full summary/config/checkpoint/prediction과 원본 SHA256를 검증합니다.
**경로·주석을 조정한 공개본을 과거 full run-state에 섞으면 SHA256 검증에 실패할 수 있습니다.**
이 guard는 약화하지 않았으며 공개 summary JSON은 경로를 제외한 결과 보고용이지 실행-state 대체물이 아닙니다.
공개본의 end-to-end 재실행은 검증하지 않았습니다. 이번 작업에서는 syntax/import resolution만 정적으로 검사했고 학습·inference·Test 재평가는 수행하지 않았습니다.

CSV와 PNG는 원본과 byte-identical합니다. JSON은 원본 metric과 필요한 설정만 추출하고 개인 경로·대용량 파일 inventory·runtime log를 제외했습니다.
원본 결과와 checkpoint는 변경하지 않았습니다.

