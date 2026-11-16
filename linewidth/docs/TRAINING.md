# U-Net 학습 자료

## 반드시 보존

- `linewidth/methods/unet/models/best_unet_manual.pt`: 현재 추론 모델
  (`calibrate` 단계로 새 모델을 만들면 이 파일을 교체한다. 그 모델의 학습 기록은
  `analysis/models/active/`에 있다.)
- `analysis/training/source`: 최초 원본 이미지
- `analysis/training/unet_data/images`: 학습 원본
- `analysis/training/unet_data/masks`: 정답 mask
- `analysis/training/manual_reviews`: 사람이 수정한 mask와 manifest

## 없어도 되는 생성물

`analysis/training/unet_dataset`의 256×256 patch는 `training/unet_training.py`의
`make-dataset` 단계로 다시 생성할 수 있다. 평소에는 삭제해도 되며, 재학습
직전에 `linewidth_tool` 폴더에서 아래 순서로 만든다.

```powershell
python .\training\unet_training.py make-dataset
python .\training\unet_training.py train
```

`training/unet_training.py`에는 mask 생성부터 미세조정·보정까지 U-Net 학습의
전체 단계가 모여 있다. 각 단계는 `python .\training\unet_training.py <단계>`로
실행한다 (`make-masks`, `make-dataset`, `train`, `prepare-mask-review`,
`prepare-band-review`, `prepare-single-boundary-review`,
`prepare-diverse-boundary-review`, `combine-reviews`, `finetune`, `calibrate`).

추론만 할 때는 학습 사진·patch가 필요하지 않고 `.pt` 모델만 필요하다. 다만
나중에 재학습할 가능성이 있으므로 원본과 정답 mask는 보존한다.
