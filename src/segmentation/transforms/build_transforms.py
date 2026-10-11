"""Transform building utilities for segmentation.

Supports nested Compose/OneOf/SomeOf structures in YAML configuration.
Includes mask support for segmentation tasks.
"""

import warnings
from typing import Any, Callable

import albumentations as alb
import numpy as np
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf

from .normalization import NormalizeImageNet, NormalizeMinMax, NormalizePercentile


class PadToSizeMinValue(alb.DualTransform):
    """Center-pad to at least (min_height, min_width).

    The image is padded with its **own per-sample minimum value** (so the padded
    border matches the darkest background — important for CT where air is very
    negative and 0 would be soft tissue), and the mask is padded with 0.
    Works for any number of image channels (uses ``np.pad``, not cv2, so 2.5D
    multi-slice stacks are supported). Does not crop; pair with a CenterCrop if
    inputs may exceed the target size.
    """

    def __init__(self, min_height: int, min_width: int, p: float = 1.0):
        super().__init__(p=p)
        self.min_height = min_height
        self.min_width = min_width

    def _pads(self, h: int, w: int) -> tuple[int, int, int, int]:
        ph = max(0, self.min_height - h)
        pw = max(0, self.min_width - w)
        top = ph // 2
        left = pw // 2
        return top, ph - top, left, pw - left

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        top, bottom, left, right = self._pads(img.shape[0], img.shape[1])
        pad_width = [(top, bottom), (left, right)] + [(0, 0)] * (img.ndim - 2)
        return np.pad(img, pad_width, mode="constant", constant_values=float(img.min()))

    def apply_to_mask(self, mask: np.ndarray, **params: Any) -> np.ndarray:
        top, bottom, left, right = self._pads(mask.shape[0], mask.shape[1])
        pad_width = [(top, bottom), (left, right)] + [(0, 0)] * (mask.ndim - 2)
        return np.pad(mask, pad_width, mode="constant", constant_values=0)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("min_height", "min_width")

# Supported augmentation name to albumentations class mapping
AUGMENTATION_REGISTRY: dict[str, type[alb.BasicTransform]] = {
    # Blur
    "MotionBlur": alb.MotionBlur,
    "GaussianBlur": alb.GaussianBlur,
    "MedianBlur": alb.MedianBlur,
    # Color/Intensity
    "RandomGamma": alb.RandomGamma,
    "RandomBrightnessContrast": alb.RandomBrightnessContrast,
    "CLAHE": alb.CLAHE,
    "Sharpen": alb.Sharpen,
    "Emboss": alb.Emboss,
    "ColorJitter": alb.ColorJitter,
    "HueSaturationValue": alb.HueSaturationValue,
    "RGBShift": alb.RGBShift,
    # Flip/Rotate
    "HorizontalFlip": alb.HorizontalFlip,
    "VerticalFlip": alb.VerticalFlip,
    "Rotate": alb.Rotate,
    "ShiftScaleRotate": alb.ShiftScaleRotate,
    "Affine": alb.Affine,
    "RandomRotate90": alb.RandomRotate90,
    # Geometric distortion
    "ElasticTransform": alb.ElasticTransform,
    "GridDistortion": alb.GridDistortion,
    "OpticalDistortion": alb.OpticalDistortion,
    # Noise/Dropout
    "GaussNoise": alb.GaussNoise,
    "CoarseDropout": alb.CoarseDropout,
    "Cutout": alb.CoarseDropout,  # Alias
    # Resize/Crop/Pad
    "Resize": alb.Resize,
    "RandomCrop": alb.RandomCrop,
    "CenterCrop": alb.CenterCrop,
    "RandomResizedCrop": alb.RandomResizedCrop,
    "PadIfNeeded": alb.PadIfNeeded,
    "Pad": alb.Pad,
    # Normalize (for explicit use)
    "Normalize": alb.Normalize,
}

# Container type augmentations (support nested structures)
CONTAINER_TYPES = {"OneOf", "SomeOf", "Compose"}


def _to_dict(config: DictConfig | dict | Any) -> dict | Any:
    """Convert OmegaConf DictConfig to plain dict recursively."""
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return config


def build_transform_recursive(
    transform_config: dict | DictConfig,
) -> alb.BasicTransform | None:
    """Recursively build a transform from config.

    Supports nested Compose/OneOf/SomeOf structures.

    Args:
        transform_config: Transform configuration dict.
            Can be either:
            - Single transform: {"Resize": {"height": 512, "width": 512}}
            - Container: {"OneOf": {"p": 0.5, "transforms": [...]}}
            - Compose: {"Compose": {"transforms": [...]}}

    Returns:
        Built transform or None if unknown.
    """
    if not isinstance(transform_config, (dict, DictConfig)):
        return None

    config = _to_dict(transform_config)

    for transform_name, params in config.items():
        params = params or {}

        # OneOf
        if transform_name == "OneOf":
            return _build_oneof(params)

        # SomeOf
        if transform_name == "SomeOf":
            return _build_someof(params)

        # Compose
        if transform_name == "Compose":
            return _build_compose(params)

        # Regular augmentation
        if transform_name in AUGMENTATION_REGISTRY:
            aug_class = AUGMENTATION_REGISTRY[transform_name]
            return aug_class(**params)

        warnings.warn(f"Unknown transform '{transform_name}', skipping.", UserWarning, stacklevel=2)
        return None

    return None


def _build_transforms_list(transforms_config: list) -> list[alb.BasicTransform]:
    """Build a list of transforms from config list."""
    transforms = []
    for t_config in transforms_config:
        t = build_transform_recursive(t_config)
        if t is not None:
            transforms.append(t)
    return transforms


def _build_oneof(params: dict) -> alb.OneOf | None:
    """Build OneOf from params."""
    p = params.get("p", 0.5)
    transforms_config = params.get("transforms", [])

    if not transforms_config:
        print("Warning: OneOf requires 'transforms' key, skipping.")
        return None

    transforms = _build_transforms_list(transforms_config)
    if transforms:
        return alb.OneOf(transforms, p=p)
    return None


def _build_someof(params: dict) -> alb.SomeOf | None:
    """Build SomeOf from params."""
    p = params.get("p", 0.5)
    n = params.get("n", 1)
    transforms_config = params.get("transforms", [])

    if not transforms_config:
        print("Warning: SomeOf requires 'transforms' key, skipping.")
        return None

    transforms = _build_transforms_list(transforms_config)
    if transforms:
        return alb.SomeOf(transforms, n=n, p=p)
    return None


def _build_compose(params: dict) -> alb.Compose | None:
    """Build Compose from params."""
    p = params.get("p", 1.0)
    transforms_config = params.get("transforms", [])

    if not transforms_config:
        print("Warning: Compose requires 'transforms' key, skipping.")
        return None

    transforms = _build_transforms_list(transforms_config)
    if transforms:
        return alb.Compose(transforms, p=p)
    return None


def build_single_augmentation(aug_name: str, aug_config: DictConfig | dict) -> alb.BasicTransform | None:
    """Build a single augmentation.

    Args:
        aug_name: Augmentation name
        aug_config: Augmentation config

    Returns:
        Built augmentation or None if unknown.
    """
    config = _to_dict(aug_config) or {}

    if aug_name in AUGMENTATION_REGISTRY:
        aug_class = AUGMENTATION_REGISTRY[aug_name]
        return aug_class(**config)
    return None


def build_normalization(normalization_config: DictConfig | dict) -> Callable:
    """Build normalization transform.

    Args:
        normalization_config: Normalization config with 'type' key.
            type: "minmax" | "imagenet" | "percentile"

    Returns:
        Normalization transform.
    """
    config = _to_dict(normalization_config)
    norm_type = config.get("type", "imagenet")

    if norm_type == "imagenet":
        imagenet_config = config.get("imagenet", {})
        return NormalizeImageNet(
            max_pixel_value=imagenet_config.get("max_pixel_value", 255.0),
            p=1.0,
        )

    elif norm_type == "minmax":
        minmax_config = config.get("minmax", {})
        return NormalizeMinMax(
            min_value=minmax_config.get("min_value", 0.0),
            max_value=minmax_config.get("max_value", 1.0),
            p=1.0,
        )

    elif norm_type == "percentile":
        percentile_config = config.get("percentile", {})
        return NormalizePercentile(
            lower_percentile=percentile_config.get("lower_percentile", 1.0),
            upper_percentile=percentile_config.get("upper_percentile", 99.0),
            eps=percentile_config.get("eps", 1e-8),
            p=1.0,
        )

    else:
        raise ValueError(f"Invalid normalization type: {norm_type}")


def build_augmentation(
    augmentation_config: DictConfig | dict,
) -> list[alb.BasicTransform]:
    """Build augmentation transforms.

    Args:
        augmentation_config: Augmentation config.

    Returns:
        List of augmentation transforms.

    Example YAML (simple):
        augmentation:
          HorizontalFlip:
            p: 0.5
          RandomGamma:
            gamma_limit: [70, 130]
            p: 0.5

    Example YAML (nested):
        augmentation:
          OneOf:
            p: 1.0
            transforms:
              - Resize: {height: 256, width: 256}
              - RandomResizedCrop: {height: 224, width: 224}
          HorizontalFlip:
            p: 0.5
    """
    config = _to_dict(augmentation_config)
    augmentations = []

    for aug_key, aug_config in config.items():
        # Skip special keys
        if aug_key in ("mixup", "cutmix"):
            continue

        # Container types (OneOf, SomeOf, Compose)
        if aug_key in CONTAINER_TYPES:
            transform = build_transform_recursive({aug_key: aug_config})
            if transform is not None:
                augmentations.append(transform)

        # Regular augmentation
        else:
            aug = build_single_augmentation(aug_key, aug_config or {})
            if aug is not None:
                augmentations.append(aug)
            else:
                warnings.warn(f"Unknown augmentation '{aug_key}', skipping.", UserWarning, stacklevel=2)

    return augmentations


def build_transforms(
    preprocessing_config: DictConfig | dict,
    augmentation_config: DictConfig | dict | None = None,
    is_train: bool = True,
) -> alb.Compose:
    """Build transform pipeline for segmentation.

    Supports both image and mask transforms.

    Args:
        preprocessing_config: Preprocessing config with 'image_size' and 'normalization'.
        augmentation_config: Augmentation config (optional).
        is_train: Whether training mode.

    Returns:
        Composed transform pipeline.

    Example YAML:
        preprocessing:
          image_size:
            x: 224
            y: 224
          normalization:
            type: imagenet
            imagenet:
              max_pixel_value: 255.0

        augmentation:
          HorizontalFlip:
            p: 0.5
          RandomBrightnessContrast:
            brightness_limit: 0.2
            contrast_limit: 0.2
            p: 0.5

    Note:
        The returned Compose should be called with:
            transformed = transform(image=image, mask=mask)
        where mask is [H, W, num_classes] for multi-class segmentation.
    """
    config = _to_dict(preprocessing_config)
    transforms = []

    image_size = config.get("image_size", {})
    if image_size:
        height = image_size.get("y", image_size.get("height", 224))
        width = image_size.get("x", image_size.get("width", 224))
        # mode: "resize" (default, stretch to HxW) | "pad" (center-pad with the
        # per-sample min value, preserving aspect ratio and voxel spacing).
        mode = image_size.get("mode", "resize")
        if mode == "pad":
            transforms.append(PadToSizeMinValue(min_height=height, min_width=width))
            # Guarantee exact HxW even if an input exceeds the target size.
            transforms.append(alb.CenterCrop(height=height, width=width))
        elif mode == "resize":
            transforms.append(alb.Resize(height=height, width=width))
        else:
            raise ValueError(f"Unknown image_size.mode: {mode!r} (use 'resize' or 'pad')")

    normalization_config = config.get("normalization", {"type": "imagenet"})
    aug_transforms = build_augmentation(augmentation_config) if (is_train and augmentation_config is not None) else []

    # Order of augmentation vs normalization. Default keeps the historical order
    # (augment on raw intensities, then normalize). With ``normalize_before_aug:
    # true`` normalization runs FIRST, so intensity augmentations (brightness /
    # contrast / gamma / noise) actually take effect — otherwise the per-sample
    # percentile normalization applied afterwards largely cancels them out. Use
    # this when following nnU-Net-style "moreDA" intensity augmentation.
    if config.get("normalize_before_aug", False):
        transforms.append(build_normalization(normalization_config))
        transforms.extend(aug_transforms)
    else:
        transforms.extend(aug_transforms)
        transforms.append(build_normalization(normalization_config))

    # ToTensor - note: for segmentation, mask should NOT be normalized
    # ToTensorV2 handles both image and mask properly
    transforms.append(ToTensorV2())

    return alb.Compose(transforms, p=1.0)
