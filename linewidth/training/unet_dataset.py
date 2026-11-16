"""PyTorch Dataset for the generated U-Net image/mask patches."""

from __future__ import annotations

from pathlib import Path

# In this Anaconda environment NumPy must be imported before PyTorch so that
# both packages use the same OpenMP runtime.
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Load paired RGB image patches and binary masks with identical names."""

    def __init__(self, dataset_root: str | Path, split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be train or val, got {split!r}")

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_dir = self.dataset_root / split / "images"
        self.mask_dir = self.dataset_root / split / "masks"

        image_names = {path.name for path in self.image_dir.glob("*.png")}
        mask_names = {path.name for path in self.mask_dir.glob("*.png")}
        if image_names != mask_names:
            missing_masks = sorted(image_names - mask_names)
            missing_images = sorted(mask_names - image_names)
            raise RuntimeError(
                f"image-mask filenames do not match: "
                f"missing_masks={missing_masks[:5]}, missing_images={missing_images[:5]}"
            )

        self.names = sorted(image_names)
        if not self.names:
            raise RuntimeError(f"No PNG patches found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        name = self.names[index]

        image = np.asarray(
            Image.open(self.image_dir / name).convert("RGB"),
            dtype=np.float32,
        )
        mask = np.asarray(
            Image.open(self.mask_dir / name).convert("L"),
            dtype=np.uint8,
        )

        if image.shape[:2] != mask.shape:
            raise RuntimeError(
                f"shape mismatch for {name}: image={image.shape}, mask={mask.shape}"
            )

        # HWC uint8 0..255 -> CHW float32 0..1
        image = np.ascontiguousarray(image.transpose(2, 0, 1) / 255.0)
        # HW uint8 0/255 -> 1HW float32 0/1
        mask = np.ascontiguousarray((mask > 127).astype(np.float32)[None, :, :])

        return torch.from_numpy(image), torch.from_numpy(mask), name


def dataset_summary(dataset_root: str | Path) -> dict:
    train = SegmentationDataset(dataset_root, "train")
    val = SegmentationDataset(dataset_root, "val")
    train_image, train_mask, _ = train[0]
    val_image, val_mask, _ = val[0]
    return {
        "train_count": len(train),
        "val_count": len(val),
        "train_image_shape": tuple(train_image.shape),
        "train_mask_shape": tuple(train_mask.shape),
        "val_image_shape": tuple(val_image.shape),
        "val_mask_shape": tuple(val_mask.shape),
        "train_image_range": (float(train_image.min()), float(train_image.max())),
        "train_mask_values": sorted(float(x) for x in torch.unique(train_mask)),
    }


