# YOLOv8n real-only fixed baseline

Model: yolov8n.pt. Epochs: 28/100. Best epoch: 8.
imgsz=640, batch=8, device=cpu, workers=0, seed=42, deterministic=True, patience=20.
Pretrained=True. Installed Ultralytics optimizer/lr/weight-decay defaults unchanged.
Train/Val/Test: 505/62/62. Source image/label bytes and split assignments unchanged.
Explicitly disabled augmentations: hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective, flipud, fliplr, mosaic, mixup, copy_paste, erasing, bgr, cutmix, multi_scale
auto_augment=None; augment=False; no custom augmentation/oversampling/synthetic/class weighting.
Albumentations is not installed. Standard input resizing/letterboxing is not data augmentation.

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0.682549 | 0.300493 | 0.278915 | 0.106948 |
| test | 0.774205 | 0.152604 | 0.158140 | 0.056632 |

| Test class | GT instances | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| p | 546 | 0.501967 | 0.450416 | 0.433377 | 0.151798 |
| o | 1 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| s | 1 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| sc | 50 | 0.594855 | 0.160000 | 0.199181 | 0.074731 |

Val open: no GT, not evaluable. Test open/short: one instance each; numeric results only.
Best checkpoint: C:\Users\cindi\Desktop\ai-cd-metrology\runs\baseline\yolov8n_real_only_v1\weights\best.pt
Training time: 49.33 minutes.
Training errors: none. NaN: none. Loader exclusions: 0.
Warnings: none
Execution: .venv/Scripts/python.exe -B -u scripts/train_yolov8_detection_baseline_v1.py
