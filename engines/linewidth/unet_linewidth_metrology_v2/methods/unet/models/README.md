# U-Net 가중치 (Git 제외: `*.pt`)

이 폴더에 `best_unet_manual.pt`를 둡니다 (SHA-256 `58a815efaf910c26db5ba8d875eafdec42e7a764d409a981b94a4abc44fd4c49`, 1,991,379 bytes).

같은 파일이 원격 브랜치 `unet-method`에 이미 있습니다. 프로젝트 루트에서:

```powershell
git fetch origin
git restore --source origin/unet-method --worktree -- "unet method/models/best_unet_manual.pt"
Move-Item "unet method\models\best_unet_manual.pt" engines\linewidth\unet_linewidth_metrology_v2\methods\unet\models\
Remove-Item -Recurse "unet method" -ErrorAction SilentlyContinue
```

다른 경로의 가중치를 쓰려면 환경변수 `LINEWIDTH_UNET_CHECKPOINT`로 지정할 수 있습니다 (팀원 코드 기본 동작).
