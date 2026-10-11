"""Segmentation Dataset Implementations.

Provides dataset classes for segmentation tasks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.segmentation.registries.datasets import register_dataset


class RandomSegmentationDataset(Dataset):
    """Random image and mask dataset for testing.

    Generates random images with random binary masks for each class.

    Args:
        num_samples: Number of samples to generate
        image_size: Size of the generated images (square)
        num_classes: Number of segmentation classes
        in_channels: Number of input image channels
        seed: Random seed for reproducibility
        transforms: Optional albumentations transform (Compose)
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: int = 128,
        num_classes: int = 6,
        in_channels: int = 3,
        seed: int | None = None,
        transforms=None,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.transforms = transforms

        rng = np.random.default_rng(seed)

        # Generate images [num_samples, H, W, C] as uint8 (0-255)
        self.images = rng.integers(0, 256, (num_samples, image_size, image_size, in_channels), dtype=np.uint8)

        # Generate masks [num_samples, H, W, num_classes] as float32 (binary)
        # Create random ellipses/shapes for each class
        self.masks = np.zeros((num_samples, image_size, image_size, num_classes), dtype=np.float32)

        for i in range(num_samples):
            for c in range(num_classes):
                # Random probability of having this class in the image
                if rng.random() > 0.5:
                    # Create random ellipse
                    cx = rng.integers(image_size // 4, 3 * image_size // 4)
                    cy = rng.integers(image_size // 4, 3 * image_size // 4)
                    rx = rng.integers(image_size // 8, image_size // 3)
                    ry = rng.integers(image_size // 8, image_size // 3)

                    y, x = np.ogrid[:image_size, :image_size]
                    mask = ((x - cx) ** 2 / (rx**2 + 1e-8) + (y - cy) ** 2 / (ry**2 + 1e-8)) <= 1
                    self.masks[i, :, :, c] = mask.astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Get a sample.

        Returns:
            Dict with keys:
                - image: Tensor[C, H, W]
                - masks: Tensor[num_classes, H, W]
                - sample_id: str
        """
        image = self.images[idx].copy()  # [H, W, C]
        mask = self.masks[idx].copy()  # [H, W, num_classes]

        if self.transforms:
            # Albumentations applies the same geometric transform to a
            # multi-channel [H, W, num_classes] mask passed via the `mask` key.
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensor
        # Image: [H, W, C] -> [C, H, W]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        elif isinstance(image, torch.Tensor):
            # ToTensorV2 returns [C, H, W] for image, no need to permute
            image = image.float()
        else:
            image = torch.tensor(image).float()

        # Mask: [H, W, num_classes] -> [num_classes, H, W]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.transpose(2, 0, 1).copy()).float()
        elif isinstance(mask, torch.Tensor):
            # ToTensorV2 returns [H, W, C] for multi-channel mask, need to permute
            if mask.ndim == 3:
                mask = mask.permute(2, 0, 1).float()
            else:
                mask = mask.float()
        else:
            mask = torch.tensor(mask).float()

        return {
            "image": image,
            "masks": mask,
            "sample_id": f"sample_{idx:05d}",
        }


@register_dataset("random_segmentation", aliases=["random_mask", "dummy_segmentation"])
def build_random_segmentation_dataset(
    num_samples: int = 100,
    image_size: int = 128,
    num_classes: int = 6,
    in_channels: int = 3,
    seed: int | None = None,
    transforms=None,
    **kwargs,
) -> RandomSegmentationDataset:
    """Build a random segmentation dataset.

    Args:
        num_samples: Number of samples
        image_size: Size of generated images
        num_classes: Number of segmentation classes
        in_channels: Number of input channels
        seed: Random seed
        transforms: Optional transform (Compose)

    Returns:
        RandomSegmentationDataset instance
    """
    return RandomSegmentationDataset(
        num_samples=num_samples,
        image_size=image_size,
        num_classes=num_classes,
        in_channels=in_channels,
        seed=seed,
        transforms=transforms,
    )


class FoldSegmentationDataset(Dataset):
    """Fold-aware segmentation dataset for cross-validation.

    Splits data into K folds and returns train/val split based on fold number.

    Args:
        num_samples: Total number of samples (before split)
        image_size: Image size
        num_classes: Number of segmentation classes
        in_channels: Number of image channels
        num_folds: Number of folds for cross-validation
        fold: Current fold number (0 to num_folds-1)
        is_train: If True, return training split; if False, return validation split
        transforms: Optional albumentations transform (Compose)
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int = 500,
        image_size: int = 128,
        num_classes: int = 6,
        in_channels: int = 3,
        num_folds: int = 5,
        fold: int = 0,
        is_train: bool = True,
        transforms=None,
        seed: int = 42,
    ):
        self.image_size = image_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.transforms = transforms

        all_indices = np.arange(num_samples)

        # Shuffle indices with seed for reproducibility
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)

        fold_size = num_samples // num_folds
        fold_indices = []
        for i in range(num_folds):
            start = i * fold_size
            end = start + fold_size if i < num_folds - 1 else num_samples
            fold_indices.append(all_indices[start:end])

        val_indices = fold_indices[fold]

        # Get training fold indices (all other folds)
        train_indices = np.concatenate([fold_indices[i] for i in range(num_folds) if i != fold])

        self.indices = train_indices if is_train else val_indices

        # Pre-generate all data (using separate RNG for data generation)
        data_rng = np.random.default_rng(seed + 2000)

        self.all_images = data_rng.integers(0, 256, (num_samples, image_size, image_size, in_channels), dtype=np.uint8)

        self.all_masks = np.zeros((num_samples, image_size, image_size, num_classes), dtype=np.float32)
        for i in range(num_samples):
            for c in range(num_classes):
                if data_rng.random() > 0.5:
                    cx = data_rng.integers(image_size // 4, 3 * image_size // 4)
                    cy = data_rng.integers(image_size // 4, 3 * image_size // 4)
                    rx = data_rng.integers(image_size // 8, image_size // 3)
                    ry = data_rng.integers(image_size // 8, image_size // 3)
                    y, x = np.ogrid[:image_size, :image_size]
                    mask = ((x - cx) ** 2 / (rx**2 + 1e-8) + (y - cy) ** 2 / (ry**2 + 1e-8)) <= 1
                    self.all_masks[i, :, :, c] = mask.astype(np.float32)

        split_type = "train" if is_train else "valid"
        print(f"[FoldSegmentationDataset] fold={fold}, {split_type}: {len(self.indices)} samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample_idx = self.indices[idx]

        image = self.all_images[sample_idx].copy()
        mask = self.all_masks[sample_idx].copy()

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensor
        # Image: [H, W, C] -> [C, H, W]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        elif isinstance(image, torch.Tensor):
            image = image.float()

        # Mask: [H, W, num_classes] -> [num_classes, H, W]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.transpose(2, 0, 1).copy()).float()
        elif isinstance(mask, torch.Tensor):
            if mask.ndim == 3:
                mask = mask.permute(2, 0, 1).float()
            else:
                mask = mask.float()

        return {
            "image": image,
            "masks": mask,
            "sample_id": f"sample_{sample_idx:05d}",
        }


@register_dataset("fold_segmentation", aliases=["fold_seg", "cv_segmentation"])
def build_fold_segmentation_dataset(
    num_samples: int = 500,
    image_size: int = 128,
    num_classes: int = 6,
    in_channels: int = 3,
    num_folds: int = 5,
    fold: int = 0,
    is_train: bool = True,
    transforms=None,
    seed: int = 42,
    **kwargs,
) -> FoldSegmentationDataset:
    """Build a fold-aware segmentation dataset.

    Args:
        num_samples: Total number of samples
        image_size: Image size
        num_classes: Number of classes
        in_channels: Number of channels
        num_folds: Number of folds
        fold: Current fold number
        is_train: Training or validation split
        transforms: Optional transform (Compose)
        seed: Random seed

    Returns:
        FoldSegmentationDataset instance
    """
    return FoldSegmentationDataset(
        num_samples=num_samples,
        image_size=image_size,
        num_classes=num_classes,
        in_channels=in_channels,
        num_folds=num_folds,
        fold=fold,
        is_train=is_train,
        transforms=transforms,
        seed=seed,
    )


class ImageMaskDataset(Dataset):
    """Image + label-map mask dataset for real segmentation data.

    Reads RGB images from ``images_dir`` and matching single-channel label-map
    PNGs from ``masks_dir`` (each mask pixel = class id in [0, num_classes-1]).
    Image and mask files are paired by identical filename stems.

    Args:
        images_dir: Directory containing input images.
        masks_dir: Directory containing label-map masks (pixel = class id).
        num_classes: Number of segmentation classes (incl. background).
        transforms: Optional albumentations Compose (applied to image+mask).
        fold: Cross-validation fold index (used for the train/val split).
        is_train: If True return the training split, else the validation split.
        num_folds: Number of folds for the deterministic split.
        seed: Seed for the deterministic fold assignment.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        num_classes: int,
        transforms=None,
        fold: int = 0,
        is_train: bool = True,
        num_folds: int = 5,
        seed: int = 42,
    ):
        self.num_classes = num_classes
        self.transforms = transforms
        self.paths = self._collect(images_dir, masks_dir, fold, is_train, num_folds, seed)
        split = "train" if is_train else "valid"
        print(f"[ImageMaskDataset] fold={fold}, {split}: {len(self.paths)} samples")

    @staticmethod
    def _collect(images_dir, masks_dir, fold, is_train, num_folds, seed):
        """Pair image/mask files by stem and return the requested fold split."""
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)
        image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

        pairs = []
        for img_path in image_files:
            mask_path = masks_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                # Fall back to any extension with the same stem.
                candidates = list(masks_dir.glob(f"{img_path.stem}.*"))
                if not candidates:
                    raise FileNotFoundError(f"No mask found for image {img_path}")
                mask_path = candidates[0]
            pairs.append((img_path, mask_path))

        if not pairs:
            raise RuntimeError(f"No images found in {images_dir}")

        # Deterministic fold assignment so train/val are disjoint and stable.
        # Assign each pair a fold id via a shuffled permutation, then select the
        # validation fold (== `fold`) or its complement (training).
        order = np.arange(len(pairs))
        np.random.default_rng(seed).shuffle(order)
        fold_of = np.empty(len(pairs), dtype=int)
        for rank, pair_idx in enumerate(order):
            fold_of[pair_idx] = rank % num_folds
        selected = [pairs[i] for i in range(len(pairs)) if (fold_of[i] != fold) == is_train]
        return selected

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Return {"image": [C,H,W] float, "masks": [num_classes,H,W] float, "sample_id": str}."""
        img_path, mask_path = self.paths[idx]
        image = np.array(Image.open(img_path).convert("RGB"))  # [H, W, 3] uint8
        label_map = np.array(Image.open(mask_path))  # [H, W] each pixel = class id
        if label_map.ndim == 3:  # guard: mask saved as RGB by mistake
            label_map = label_map[..., 0]
        # One-hot encode to [H, W, num_classes] so albumentations treats it as a
        # multi-channel mask and applies geometric transforms per channel.
        mask = np.eye(self.num_classes, dtype=np.float32)[label_map]

        if self.transforms is not None:
            out = self.transforms(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]

        # Image -> [C, H, W]
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        elif isinstance(image, torch.Tensor):
            image = image.float()  # ToTensorV2 already gives [C, H, W]

        # Mask -> [num_classes, H, W]
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask.transpose(2, 0, 1).copy()).float()
        elif isinstance(mask, torch.Tensor):
            if mask.ndim == 3:  # ToTensorV2 returns [H, W, C] for multi-channel masks
                mask = mask.permute(2, 0, 1).float()
            else:
                mask = mask.float()

        return {"image": image, "masks": mask, "sample_id": Path(img_path).stem}


@register_dataset("image_mask")
def build_image_mask_dataset(
    images_dir: str,
    masks_dir: str,
    num_classes: int,
    transforms=None,
    fold: int = 0,
    is_train: bool = True,
    num_folds: int = 5,
    seed: int = 42,
    **kwargs,
) -> ImageMaskDataset:
    """Build an ImageMaskDataset from a directory of images + label-map masks."""
    return ImageMaskDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        num_classes=num_classes,
        transforms=transforms,
        fold=fold,
        is_train=is_train,
        num_folds=num_folds,
        seed=seed,
    )
