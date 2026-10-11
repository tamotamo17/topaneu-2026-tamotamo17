"""Base Detection Wrapper for all detection models.

This module defines the BaseDetectionWrapper abstract class that all detection
model wrappers must inherit from. Design:

- Wrappers encapsulate external library-specific training loops
- train_detection.py remains library-agnostic
- All config conversion, training, and evaluation logic lives in the wrapper

Design Principle:
    scripts/train_detection.py calls:
        model.run_training_pipeline(config, dataset_dir, output_dir, common_params)

    The wrapper handles everything:
    - extract_train_kwargs(): YAML config → library format
    - train(): Call library's training API
    - predict(): Library output → unified Detections format
    - run_training_pipeline(): Training + evaluation flow
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass
class Detections:
    """Unified detection output format.

    All wrappers must convert their library-specific outputs to this format.

    Attributes:
        xyxy: Bounding boxes in [x1, y1, x2, y2] format, shape [N, 4]
        confidence: Confidence scores, shape [N]
        class_id: Class IDs (1-indexed for evaluation), shape [N]
    """

    xyxy: np.ndarray | torch.Tensor  # [N, 4]
    confidence: np.ndarray | torch.Tensor  # [N]
    class_id: np.ndarray | torch.Tensor  # [N] (1-indexed)

    def __len__(self) -> int:
        """Return number of detections."""
        return len(self.confidence)

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to the project-standard detection output format.

        Keys follow the unified detection interface used across the repository:
        boxes / scores / labels.
        """
        return {
            "boxes": np.asarray(self.xyxy),  # [N, 4] in xyxy format
            "scores": np.asarray(self.confidence),  # [N]
            "labels": np.asarray(self.class_id),  # [N] (1-indexed)
        }


class BaseDetectionWrapper(ABC):
    """Abstract base class for detection model wrappers.

    Each wrapper encapsulates a detection library (RF-DETR, YOLO, etc.)
    and provides a unified interface for training and inference.

    Subclasses must implement:
        - train(): Execute library's training
        - predict(): Convert library output to Detections
        - load_checkpoint(): Load model weights
        - extract_train_kwargs(): Convert YAML config to library kwargs

    Optionally override:
        - run_training_pipeline(): Customize training + evaluation flow
        - find_best_checkpoint(): Custom checkpoint search logic

    Class variables (override as needed):
        eval_nms_threshold: NMS IoU threshold for evaluation (None = disabled)
        eval_iou_threshold: mAP IoU threshold
        best_checkpoint_name: Name of best checkpoint file
    """

    # Class variables (override in subclasses)
    eval_nms_threshold: float | None = 0.1  # NMS threshold for evaluation
    eval_iou_threshold: float = 0.5  # mAP IoU threshold
    best_checkpoint_name: str = "checkpoint_best.pth"

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        resolution: int = 640,
        checkpoint_path: str | None = None,
    ):
        """Initialize base wrapper.

        Args:
            num_classes: Number of detection classes
            pretrained: Whether to use pretrained weights
            resolution: Input image resolution
            checkpoint_path: Path to custom checkpoint (optional)
        """
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.resolution = resolution
        self.checkpoint_path = checkpoint_path

    @abstractmethod
    def train(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path,
        epochs: int,
        batch_size: int,
        lr: float,
        num_workers: int,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute training using the library's native API.

        This method should call the underlying library's training function
        (e.g., rfdetr.train(), ultralytics.YOLO.train(), etc.)

        Args:
            dataset_dir: Path to COCO-format dataset directory
            output_dir: Output directory for checkpoints and logs
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            num_workers: Number of data loading workers
            **kwargs: Library-specific training parameters

        Returns:
            Training results dictionary (e.g., {"history": ..., "metrics": ...})
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> Detections:
        """Run inference and return detections in unified format.

        Convert library-specific output to Detections format:
        - xyxy: [x1, y1, x2, y2] coordinates
        - confidence: Detection confidence scores
        - class_id: 1-indexed class IDs (will be converted to 0-indexed for metrics)

        Args:
            image: Input image (PIL, numpy, or torch tensor)
            threshold: Confidence threshold
            **kwargs: Library-specific inference parameters

        Returns:
            Detections object with xyxy, confidence, class_id
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        raise NotImplementedError

    @abstractmethod
    def extract_train_kwargs(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract library-specific training kwargs from YAML config.

        Different libraries have different config formats (RF-DETR uses
        lr_encoder/use_ema/multi_scale, YOLO uses mosaic/mixup, etc.).
        This method converts the project's YAML structure to the format
        expected by the library's training API.

        Typically reads from config["rfdetr_training"] or config["yolo_training"].

        Args:
            config: Full YAML config dict (OmegaConf.to_container())

        Returns:
            Library-specific training kwargs dict
        """
        raise NotImplementedError

    def find_best_checkpoint(self, output_dir: str | Path) -> Path | None:
        """Find the best checkpoint in output directory.

        Default implementation looks for self.best_checkpoint_name.
        Override if the library uses a different structure.

        Args:
            output_dir: Training output directory

        Returns:
            Path to best checkpoint, or None if not found
        """
        output_path = Path(output_dir)
        best_ckpt = output_path / self.best_checkpoint_name

        if best_ckpt.exists():
            return best_ckpt

        # Alternative: look for latest checkpoint
        ckpt_pattern = "*.pth"
        ckpts = list(output_path.glob(ckpt_pattern))
        if ckpts:
            return max(ckpts, key=lambda p: p.stat().st_mtime)

        return None

    def run_training_pipeline(
        self,
        config: dict[str, Any],
        dataset_dir: str | Path,
        output_dir: str | Path,
        common_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute full training + evaluation pipeline.

        This is the main entry point called by train_detection.py.
        Default implementation:
        1. Extract library-specific kwargs from config
        2. Merge with common_params
        3. Call self.train()
        4. Optionally run evaluation

        Override this method to customize the training flow
        (e.g., D-FINE needs subprocess evaluation due to namespace conflicts).

        Args:
            config: Full YAML config dict
            dataset_dir: Path to dataset
            output_dir: Output directory
            common_params: Common training params (epochs, batch_size, lr, num_workers)

        Returns:
            Results dictionary with training history and evaluation metrics
        """
        # Extract library-specific kwargs
        library_kwargs = self.extract_train_kwargs(config)

        # Merge common params with library-specific params
        all_kwargs = {**common_params, **library_kwargs}

        # Execute training
        print(f"Starting training with {self.__class__.__name__}...")
        print(f"Dataset: {dataset_dir}")
        print(f"Output: {output_dir}")
        print(f"Config: {all_kwargs}")

        train_results = self.train(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            **all_kwargs,
        )

        # TODO: Add evaluation logic here if needed
        # best_ckpt = self.find_best_checkpoint(output_dir)
        # if best_ckpt:
        #     self.load_checkpoint(best_ckpt)
        #     eval_results = self.evaluate(...)
        #     train_results["evaluation"] = eval_results

        print(f"Training completed: {output_dir}")
        return train_results
