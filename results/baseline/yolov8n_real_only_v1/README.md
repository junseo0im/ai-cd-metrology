# YOLOv8n Real-only Baseline

## 목적

현재 실제 GT 데이터만 사용했을 때의 기본 detection 성능을 확인하고,
향후 Oversampling / Augmentation / Synthetic 실험의 비교 기준을 만들기 위한 baseline.

## Dataset

- Train: 505 images
- Validation: 62 images
- Test: 62 images
- Classes: `p`, `o`, `s`, `sc`

Train annotations:

- p: 4352
- o: 2
- s: 5
- sc: 399

## Training

- Model: YOLOv8n pretrained
- Input size: 640
- Seed: 42
- Requested epochs: 100
- Actual epochs: 28
- Best epoch: 8
- Batch: 8

사용하지 않은 방법:

- Additional augmentation
- Oversampling
- Synthetic data
- Class weighting
- Copy-paste

즉, real-only fixed baseline.

## Results

| Split | Precision | Recall | mAP50 | mAP50-95 |
| ---------- | --------- | ------ | ------ | -------- |
| Validation | 0.6825 | 0.3005 | 0.2789 | 0.1069 |
| Test | 0.7742 | 0.1526 | 0.1581 | 0.0566 |

### Test class results

| Class | AP50 | AP50-95 |
| ----- | ------ | ------- |
| p | 0.4334 | 0.1518 |
| o | 0.0000 | 0.0000 |
| s | 0.0000 | 0.0000 |
| sc | 0.1992 | 0.0747 |

주의:

- Validation에는 `o` GT가 없음.
- Test의 `o`, `s` GT는 각각 1개이므로 해당 class 수치는 참고용이다.

## 결론

- Particle은 일부 학습됨.
- Scratch는 검출 가능하지만 성능이 낮음.
- Open / Short는 현재 real-only 데이터로는 학습이 부족함.
- 전체적으로 Precision 대비 Recall이 낮아 실제 결함 누락이 많음.
- 이후 불균형 대응 실험의 비교 baseline으로 사용한다.

## 주요 파일

- `best.pt` : baseline checkpoint
- `baseline_results.md` : 최종 결과 요약
- `baseline_summary.json` : 구조화된 결과
- `results.csv` : epoch별 결과
- `results.png` : 학습 curve
- `BoxPR_curve.png`, `BoxF1_curve.png` : detection 성능 curve
- `confusion_matrix*.png` : class별 오류 확인
- `training_config.yaml`, `args.yaml` : 학습 설정
- `train_yolov8_detection_baseline_v1.py` : 실제 학습 코드

