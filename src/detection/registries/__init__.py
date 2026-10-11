"""Detection Task Registries.

This module provides registry access for detection components.

Unlike classification/segmentation, detection delegates training, loss and
optimization to the underlying library through BaseDetectionWrapper. Therefore
only `models` and `datasets` registries exist here -- there are intentionally
no losses/optimizers/samplers/lightning_modules registries.
"""

from src.detection.registries import datasets, models

__all__ = ["datasets", "models"]
