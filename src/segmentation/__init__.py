"""Segmentation Task Module.

This module provides components for semantic segmentation tasks:
- Models: U-Net variants and other segmentation architectures
- Losses: Dice, BCE, and combined losses
- Datasets: Segmentation dataset implementations
- Transforms: Albumentations-based image+mask transforms
- Lightning Module: Training and evaluation logic

Usage:
    from src.segmentation.registries import models, losses, datasets
    from src.segmentation.transforms import build_transforms

    model = models.build_model("model_v1", arch="Unet", backbone="resnet50", num_classes=6)
    loss = losses.build_loss("bce_dice")
    dataset = datasets.build_dataset("random_segmentation", num_samples=100)
    transform = build_transforms(preprocessing_config, augmentation_config, is_train=True)
"""
