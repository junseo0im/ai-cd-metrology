# Experiment Summary

완료된 6개 실험의 saved result만 정리했습니다. 새 학습·inference·Test 재평가는 수행하지 않았습니다.
모든 metric은 해당 summary JSON에서 그대로 가져왔으며 표시는 반올림했습니다.

## 1. YOLOv8n Detection Real-only Baseline

- 목적: 추가 불균형 대응이 없는 real-only 비교 기준 확보.
- Dataset: fixed detection source images Train 505 / Val 62 / Test 62. Class order [o,p,s,sc].
- 변경 변수: 없음. Real-only, 추가 augmentation/oversampling/synthetic/class weighting 없음.
- 주요 조건: official yolov8n.pt, imgsz640, batch8, max100 epochs, patience20, seed42, workers0, deterministic=True, CPU. 기본 augmentation 명시적 OFF. Ultralytics auto optimizer 정책은 동일하며 실제 AdamW/lr0=0.00125로 resolve. Best는 Ultralytics detection fitness.
- Actual epochs / best epoch: 28 / 8.
- Test Precision / Recall / mAP50 / mAP50-95: 0.774205 / 0.152604 / 0.158140 / 0.056632.
- Test p/o/s/sc AP50: 0.433377 / 0.000000 / 0.000000 / 0.199181.
- 결론: 이 고정 split의 real-only detection 비교 기준입니다.
- 코드: `scripts/train_yolov8_detection_baseline_v1.py`. 공개 결과: `results/yolo/baseline_summary.json`.
- 원본 결과: `data/derived/yolov8_baseline_v1/baseline_summary.json`. 원본 run: `runs/baseline/yolov8n_real_only_v1/`.

## 2. YOLOv8n Detection Rare-class Oversampling v1

- 목적: 동일 모델에서 Train 반복 노출만 추가한 효과 확인.
- Dataset: fixed detection source images Train 505 / Val 62 / Test 62. Class order [o,p,s,sc].
- 변경 변수: Train oversampling만 적용. o×10/s×5/sc×2/other×1; image당 최대 factor로 505→764 effective samples. 이미지 복제는 원본과 byte-identical.
- 주요 조건: official yolov8n.pt, imgsz640, batch8, max100 epochs, patience20, seed42, workers0, deterministic=True, CPU. 기본 augmentation 명시적 OFF. Ultralytics auto optimizer 정책은 동일하며 실제 AdamW/lr0=0.00125로 resolve. Best는 Ultralytics detection fitness.
- Actual epochs / best epoch: 29 / 9.
- Test Precision / Recall / mAP50 / mAP50-95: 0.809238 / 0.186093 / 0.272809 / 0.115865.
- Test p/o/s/sc AP50: 0.441693 / 0.000000 / 0.331667 / 0.317874.
- 결론: 이 Test에서 mAP와 sc AP는 증가했습니다. Test short GT 1개이므로 short AP 변화는 일반화 근거가 아닙니다.
- 코드: `scripts/train_yolov8_oversampling_v1.py`. 공개 결과: `results/yolo/oversampling_summary.json`.
- 원본 결과: `data/derived/yolov8_oversampling_v1/oversampling_summary.json`. 원본 run: `runs/oversampling/yolov8n_oversample_v1/`.

## 3. MobileNetV3-Small Real-only Baseline

- 목적: 추가 불균형 대응이 없는 real-only 비교 기준 확보.
- Dataset: GT bbox crops unique Train 4,758 / Val 597 / Test 598. Class order [p,o,s,sc].
- 변경 변수: 없음. Real-only, 추가 augmentation/oversampling/synthetic/class weighting 없음.
- 주요 조건: ImageNet pretrained, full fine-tuning, deterministic RGB/direct bilinear224×224/Tensor/ImageNet normalization. AdamW lr1e-3/wd1e-4, unweighted CE, batch32, max50, patience10, seed42, workers0, CPU. Val macro-F1 최대(strict improvement/earliest tie), 고정4 class/zero_division=0. Augmentation 없음.
- Actual epochs / best epoch: 13 / 3.
- Pretrained weight: MobileNet_V3_Small_Weights.IMAGENET1K_V1; best Val macro-F1 0.486250.
- Test Accuracy / Macro Precision / Macro Recall / Macro-F1 / Weighted-F1: 0.974916 / 0.487329 / 0.439542 / 0.460089 / 0.972048.
- Test p/o/s/sc F1: 0.986425 / 0.000000 / 0.000000 / 0.853933.
- Test sc Precision / Recall / F1: 0.974359 / 0.760000 / 0.853933.
- 결론: 이번 classifier 4개 중 Test Accuracy와 Macro-F1이 가장 높았습니다.
- 코드: `scripts/train_mobilenet_baseline_v1.py`. 공개 결과: `results/mobilenet/baseline/summary.json`.
- 원본 결과: `data/derived/mobilenet_baseline_v1/mobilenet_baseline_summary.json`. 원본 run: `runs/mobilenet/mobilenet_v3_small_real_only_v1/`.

## 4. MobileNetV3-Small Rare-class Oversampling v1

- 목적: 동일 모델에서 Train 반복 노출만 추가한 효과 확인.
- Dataset: GT bbox crops unique Train 4,758 / Val 597 / Test 598. Class order [p,o,s,sc].
- 변경 변수: Train oversampling만 적용. o×10/s×5/sc×2/other×1; crop당 반복으로 4,758→5,195 effective indices; p4,352/o20/s25/sc798; 새 morphology 없음.
- 주요 조건: ImageNet pretrained, full fine-tuning, deterministic RGB/direct bilinear224×224/Tensor/ImageNet normalization. AdamW lr1e-3/wd1e-4, unweighted CE, batch32, max50, patience10, seed42, workers0, CPU. Val macro-F1 최대(strict improvement/earliest tie), 고정4 class/zero_division=0. Augmentation 없음.
- Actual epochs / best epoch: 24 / 14.
- Pretrained weight: MobileNet_V3_Small_Weights.IMAGENET1K_V1; best Val macro-F1 0.488654.
- Test Accuracy / Macro Precision / Macro Recall / Macro-F1 / Weighted-F1: 0.971572 / 0.486547 / 0.429542 / 0.453280 / 0.969031.
- Test p/o/s/sc F1: 0.985533 / 0.000000 / 0.000000 / 0.827586.
- Test sc Precision / Recall / F1: 0.972973 / 0.720000 / 0.827586.
- 결론: 반복 노출 후 Test Macro-F1은 real-only보다 낮았습니다.
- 코드: `scripts/train_mobilenet_oversampling_v1.py`. 공개 결과: `results/mobilenet/oversampling/summary.json`.
- 원본 결과: `data/derived/mobilenet_oversampling_v1/mobilenet_oversampling_summary.json`. 원본 run: `runs/mobilenet/mobilenet_v3_small_oversample_v1/`.

## 5. ResNet18 Real-only Baseline

- 목적: 추가 불균형 대응이 없는 real-only 비교 기준 확보.
- Dataset: GT bbox crops unique Train 4,758 / Val 597 / Test 598. Class order [p,o,s,sc].
- 변경 변수: 없음. Real-only, 추가 augmentation/oversampling/synthetic/class weighting 없음.
- 주요 조건: ImageNet pretrained, full fine-tuning, deterministic RGB/direct bilinear224×224/Tensor/ImageNet normalization. AdamW lr1e-3/wd1e-4, unweighted CE, batch32, max50, patience10, seed42, workers0, CPU. Val macro-F1 최대(strict improvement/earliest tie), 고정4 class/zero_division=0. Augmentation 없음.
- Actual epochs / best epoch: 32 / 22.
- Pretrained weight: ResNet18_Weights.IMAGENET1K_V1; best Val macro-F1 0.549245.
- Test Accuracy / Macro Precision / Macro Recall / Macro-F1 / Weighted-F1: 0.964883 / 0.450421 / 0.450421 / 0.450421 / 0.964883.
- Test p/o/s/sc F1: 0.981685 / 0.000000 / 0.000000 / 0.820000.
- Test sc Precision / Recall / F1: 0.820000 / 0.820000 / 0.820000.
- 결론: MobileNet real-only보다 sc Recall은 높지만 Precision과 전체 Macro-F1은 낮았습니다.
- 코드: `scripts/train_resnet18_baseline_v1.py`. 공개 결과: `results/resnet18/baseline/summary.json`.
- 원본 결과: `data/derived/resnet18_baseline_v1/resnet18_baseline_summary.json`. 원본 run: `runs/resnet/resnet18_real_only_v1/`.

## 6. ResNet18 Rare-class Oversampling v1

- 목적: 동일 모델에서 Train 반복 노출만 추가한 효과 확인.
- Dataset: GT bbox crops unique Train 4,758 / Val 597 / Test 598. Class order [p,o,s,sc].
- 변경 변수: Train oversampling만 적용. o×10/s×5/sc×2/other×1; crop당 반복으로 4,758→5,195 effective indices; p4,352/o20/s25/sc798; 새 morphology 없음.
- 주요 조건: ImageNet pretrained, full fine-tuning, deterministic RGB/direct bilinear224×224/Tensor/ImageNet normalization. AdamW lr1e-3/wd1e-4, unweighted CE, batch32, max50, patience10, seed42, workers0, CPU. Val macro-F1 최대(strict improvement/earliest tie), 고정4 class/zero_division=0. Augmentation 없음.
- Actual epochs / best epoch: 16 / 6.
- Pretrained weight: ResNet18_Weights.IMAGENET1K_V1; best Val macro-F1 0.701710.
- Test Accuracy / Macro Precision / Macro Recall / Macro-F1 / Weighted-F1: 0.941472 / 0.402476 / 0.453095 / 0.422984 / 0.945388.
- Test p/o/s/sc F1: 0.969245 / 0.000000 / 0.000000 / 0.722689.
- Test sc Precision / Recall / F1: 0.623188 / 0.860000 / 0.722689.
- 결론: sc Recall 증가와 함께 sc false positive가 늘었고 전체 Macro-F1은 감소했습니다.
- 코드: `scripts/train_resnet18_oversampling_v1.py`. 공개 결과: `results/resnet18/oversampling/summary.json`.
- 원본 결과: `data/derived/resnet18_oversampling_v1/resnet18_oversampling_summary.json`. 원본 run: `runs/resnet/resnet18_oversample_v1/`.

## Evaluation limits

- Val open GT=0; Test open/short GT는 각각 1개입니다. o/s 결과는 개별 관측치이며 일반화 성능으로 표현하지 않습니다.
- Oversampling은 기존 real example을 반복할 뿐 unique sample/morphology를 늘리지 않습니다.
- Detection의 mAP와 GT-bbox classification의 Accuracy/F1은 다른 task의 지표이므로 직접 순위를 매기지 않습니다.
- YOLO의 image-level oversampling과 classifier의 crop-level oversampling은 단위와 effective sample 수가 다릅니다.
- Source split과 Val/Test는 각 모델의 기존 실험 그대로입니다.
- 실험별 상세 precision/recall 및 기존 비교 delta는 JSON/CSV에 보존되어 있습니다.
- Requirements는 현재 확인된 설치 버전을 고정했으며 패키지 설치나 재실행으로 재검증하지 않았습니다.

