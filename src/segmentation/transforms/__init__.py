"""Transform building utilities for segmentation.

Transforms are built directly from config (not via registry pattern).

Usage:
    from src.segmentation.transforms import build_transforms

    train_transform = build_transforms(
        preprocessing_config=config.preprocessing,
        augmentation_config=config.augmentation,
        is_train=True,
    )
    valid_transform = build_transforms(
        preprocessing_config=config.preprocessing,
        augmentation_config=None,
        is_train=False,
    )
"""

from .build_transforms import (
    build_augmentation,
    build_normalization,
    build_transforms,
)
from .normalization import NormalizeImageNet, NormalizeMinMax, NormalizePercentile

__all__ = [
    "build_transforms",
    "build_augmentation",
    "build_normalization",
    "NormalizeImageNet",
    "NormalizeMinMax",
    "NormalizePercentile",
]
