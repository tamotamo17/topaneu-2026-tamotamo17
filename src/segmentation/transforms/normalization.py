"""Normalization transforms for image preprocessing."""

from typing import Any

import albumentations as alb
import numpy as np


class NormalizeMinMax(alb.ImageOnlyTransform):
    """Min-max normalization to [min_value, max_value] range."""

    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 1.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.min_value = min_value
        self.max_value = max_value

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        img_min = img.min()
        img_max = img.max()

        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = img - img_min

        img = img * (self.max_value - self.min_value) + self.min_value
        return img.astype(np.float32)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("min_value", "max_value")


class NormalizeImageNet(alb.ImageOnlyTransform):
    """ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])."""

    def __init__(
        self,
        max_pixel_value: float = 255.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.max_pixel_value = max_pixel_value
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        img = img.astype(np.float32) / self.max_pixel_value

        # Handle grayscale images
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        img = (img - self.mean) / self.std
        return img.astype(np.float32)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("max_pixel_value",)


class NormalizePercentile(alb.ImageOnlyTransform):
    """Percentile-based normalization.

    Clips values to [lower_percentile, upper_percentile] and normalizes to [0, 1].
    """

    def __init__(
        self,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        eps: float = 1e-8,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.eps = eps

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        lower = np.percentile(img, self.lower_percentile)
        upper = np.percentile(img, self.upper_percentile)

        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower + self.eps)
        return img.astype(np.float32)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("lower_percentile", "upper_percentile", "eps")
