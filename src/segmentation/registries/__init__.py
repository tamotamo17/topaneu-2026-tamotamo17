"""Segmentation Task Registries.

Usage:
    from src.segmentation.registries import models, losses, datasets

    model = models.build_model("model_v1", arch="Unet", backbone="resnet50", num_classes=6)
    loss = losses.build_loss("bce_dice")
    dataset = datasets.build_dataset("random_segmentation", num_samples=1000)

Transforms are built directly from config (not via registry):
    from src.segmentation.transforms import build_transforms

    train_transform = build_transforms(
        preprocessing_config=config.preprocessing,
        augmentation_config=config.augmentation,
        is_train=True,
    )
"""

from src.segmentation.registries import (
    datasets,
    lightning_modules,
    losses,
    models,
    optimizers,
    samplers,
)

__all__ = [
    "models",
    "losses",
    "datasets",
    "optimizers",
    "samplers",
    "lightning_modules",
]
