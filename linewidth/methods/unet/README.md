# UNet 방법 (대시보드 채택)

UNet 이 빨간 sidewall 띠를 분할하고, 이웃한 두 띠의 안쪽 끝 사이 거리로 line·gap 폭을 잰다.
공통 기준은 `../../docs/COMMON_PROTOCOL.md`, 기준값은 `../../team_protocol.py`.

| 파일 | 역할 |
|---|---|
| `unet_inference.py` | 모델 정의·로드, 타일 추론, 마스크 후처리, 측정 (`single`, `batch`) |
| `shared.py` | 이미지 로드·중복 픽셀 제거, 경계 계산, 보정값, overlay |
| `models/best_unet_manual.pt` | 학습된 모델 (epoch 11, 확률 임계값 0.50) |

## 실행 (`linewidth` 폴더에서)

```powershell
pip install -r requirements.txt
python methods/unet/unet_inference.py single "사진.png"
python methods/unet/unet_inference.py batch "사진폴더" --device cpu --expected-count 0
```

- 항상 line·gap 을 잰다 (측정 종류 선택 메뉴 없음).
- `batch` 결과: `unet_predictions.csv`(성능 평가용 중앙 5개 형식) + 사진별 측정 JSON·overlay·mask.
  측정 JSON 의 `rows[].measurements` 에는 양옆 3% 를 뺀 모든 line·gap 이 들어 있다.
- 대시보드 연동은 `../../metrology/unet_metrology.py` 를 쓴다 (공통 입출력 형식).
- 다른 모델을 쓰려면 `--checkpoint` 또는 환경변수 `LINEWIDTH_UNET_CHECKPOINT`.
