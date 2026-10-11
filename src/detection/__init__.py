"""Detection Task Module.

This module provides detection model wrappers and datasets following
the BaseDetectionWrapper pattern.

Usage:
    from src.detection.registries import models, datasets

    # Build a detection model
    model = models.build_model("rfdetr_base", num_classes=6, pretrained=False)

    # Build a dataset
    dataset = datasets.build_dataset("random_detection", num_samples=100)

    # Run training pipeline
    model.run_training_pipeline(config, dataset_dir, output_dir, common_params)

Design:
    - train_detection.py is library-agnostic
    - All training logic lives in the wrapper's run_training_pipeline()
    - Wrappers handle YAML config → library format conversion

Note:
    Detection delegates training/optimization/loss to the underlying library
    (via BaseDetectionWrapper), so unlike classification/segmentation it only
    provides `models` and `datasets` registries -- there are intentionally no
    losses/optimizers/samplers/lightning_modules registries.
"""
