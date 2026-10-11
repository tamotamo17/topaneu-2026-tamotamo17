"""RF-DETR Wrapper for Detection.

This module provides wrappers for RF-DETR models of different sizes,
following the BaseDetectionWrapper pattern.

Supported model sizes (default input resolution):
- nano: Smallest, fastest (384)
- small: Good balance (512)
- medium: More capacity (576)
- base: Recommended baseline (560)
- large: High accuracy (560)

Resolutions must be divisible by the backbone block size
(patch_size x num_windows): 32 for nano/small/medium, 56 for base/large.

Design:
- Lazy imports RF-DETR library (only when needed)
- Uses RF-DETR's native training API
- Converts outputs to unified Detections format
- Handles all RF-DETR-specific config in extract_train_kwargs()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.detection.models.base import BaseDetectionWrapper, Detections
from src.detection.registries.models import register_model


class RFDETRWrapper(BaseDetectionWrapper):
    """Wrapper for RF-DETR detection models.

    RF-DETR provides real-time detection with transformer architecture.
    This wrapper uses RF-DETR's Python API directly for training.

    Args:
        model_size: Model size ('nano', 'small', 'medium', 'base', 'large')
        num_classes: Number of detection classes
        pretrained: Whether to use pretrained weights
        resolution: Input image resolution
        checkpoint_path: Path to custom checkpoint (optional)
    """

    # Class variables (eval thresholds inherit from BaseDetectionWrapper).
    # RF-DETR finalizes training by writing checkpoint_best_total.pth, so override
    # the generic default to match the actual file find_best_checkpoint() looks for.
    best_checkpoint_name = "checkpoint_best_total.pth"

    # Model size configurations. Defaults match RF-DETR's validated resolutions.
    # block_size = patch_size x num_windows; input resolution must be divisible
    # by it or the DINOv2 backbone forward pass asserts.
    MODEL_SIZES = {
        "nano": {"default_resolution": 384, "block_size": 32},
        "small": {"default_resolution": 512, "block_size": 32},
        "medium": {"default_resolution": 576, "block_size": 32},
        "base": {"default_resolution": 560, "block_size": 56},
        "large": {"default_resolution": 560, "block_size": 56},
    }

    def __init__(
        self,
        model_size: str = "base",
        num_classes: int = 6,
        pretrained: bool = True,
        resolution: int | None = None,
        checkpoint_path: str | None = None,
    ):
        """Initialize RF-DETR wrapper.

        Args:
            model_size: Model size ('nano', 'small', 'medium', 'base', 'large')
            num_classes: Number of detection classes
            pretrained: Whether to use pretrained weights
            resolution: Input resolution (uses default if None)
            checkpoint_path: Path to custom checkpoint
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"Invalid model_size '{model_size}'. Choose from: {list(self.MODEL_SIZES.keys())}")

        self.model_size = model_size

        # Use default resolution if not provided
        if resolution is None:
            resolution = self.MODEL_SIZES[model_size]["default_resolution"]

        # Resolution must be divisible by the backbone block size, otherwise the
        # DINOv2 forward pass asserts at training time.
        block_size = self.MODEL_SIZES[model_size]["block_size"]
        if resolution % block_size != 0:
            lower = (resolution // block_size) * block_size
            upper = lower + block_size
            raise ValueError(
                f"resolution={resolution} is invalid for RF-DETR '{model_size}': "
                f"it must be divisible by {block_size} (patch_size x num_windows). "
                f"Nearest valid values: {lower} or {upper}."
            )

        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            resolution=resolution,
            checkpoint_path=checkpoint_path,
        )

        # Model is created lazily on first use
        self._model = None

    def _ensure_rfdetr_imported(self):
        """Lazy import RF-DETR library."""
        try:
            import rfdetr  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "RF-DETR library not found. Install with: pip install rfdetr\nOr add to pyproject.toml dependencies."
            ) from e

    def _create_model(self):
        """Create RF-DETR model instance (lazy initialization)."""
        if self._model is not None:
            return self._model

        self._ensure_rfdetr_imported()

        # Import appropriate model class based on size
        if self.model_size == "nano":
            from rfdetr import RFDETRNano

            model_class = RFDETRNano
        elif self.model_size == "small":
            from rfdetr import RFDETRSmall

            model_class = RFDETRSmall
        elif self.model_size == "medium":
            from rfdetr import RFDETRMedium

            model_class = RFDETRMedium
        elif self.model_size == "base":
            from rfdetr import RFDETRBase

            model_class = RFDETRBase
        elif self.model_size == "large":
            from rfdetr import RFDETRLarge

            model_class = RFDETRLarge
        else:
            raise ValueError(f"Unknown model size: {self.model_size}")

        # Create model instance
        # Pass resolution so the config-specified (or per-size default) input
        # resolution is honored instead of RF-DETR's library default.
        self._model = model_class(
            num_classes=self.num_classes,
            pretrained=self.pretrained,
            resolution=self.resolution,
        )

        # Load custom checkpoint if provided
        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path)

        return self._model

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
        """Execute training using RF-DETR's native API.

        Args:
            dataset_dir: Path to COCO-format dataset directory
            output_dir: Output directory for checkpoints
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            num_workers: Number of data loading workers
            **kwargs: RF-DETR-specific training parameters

        Returns:
            Training results dictionary
        """
        model = self._create_model()

        # Train using RF-DETR API
        print(f"Training RF-DETR {self.model_size} model...")
        print(f"  Dataset: {dataset_dir}")
        print(f"  Output: {output_dir}")
        print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")

        try:
            if dataset_dir is None:
                raise ValueError(
                    "dataset_dir is None. Dummy datasets are now materialized to "
                    "disk in train_detection.py's prepare_dataset(); a valid "
                    "Roboflow-format COCO directory is required for training."
                )

            # RF-DETR expects a Roboflow-format COCO directory (train/valid/test
            # each with _annotations.coco.json). The image resolution is fixed at
            # model-creation time, so it is not passed here.
            history = model.train(
                dataset_dir=str(dataset_dir),
                output_dir=str(output_dir),
                dataset_file="roboflow",
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                num_workers=num_workers,
                **kwargs,
            )

            return {"history": history, "status": "success"}

        except FileNotFoundError as e:
            # RF-DETR finalizes training by copying checkpoint_best_regular.pth ->
            # checkpoint_best_total.pth. That "best" file is only written when the
            # validation mAP beats the running best (init 0.0, strict >). When mAP
            # never exceeds 0.0 -- e.g. the tiny random dummy dataset or extremely
            # short training -- the best file is absent and the copy raises. The
            # per-epoch checkpoint.pth is still saved, so the run did complete.
            latest_ckpt = Path(output_dir) / "checkpoint.pth"
            if "checkpoint_best" in str(e) and latest_ckpt.exists():
                warnings.warn(
                    "Training completed, but no best checkpoint was saved because "
                    "validation mAP never exceeded 0.0 (expected for the random "
                    f"dummy dataset). Using per-epoch checkpoint: {latest_ckpt}"
                )
                return {
                    "status": "success",
                    "checkpoint": str(latest_ckpt),
                    "note": "no_best_checkpoint",
                }
            # Any other missing file is a genuine failure: re-raise so the caller's
            # exit code reflects it (training must not silently report success).
            raise

    def predict(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> Detections:
        """Run inference and return detections.

        Args:
            image: Input image (PIL Image, numpy array, or file path)
            threshold: Confidence threshold
            **kwargs: Additional inference parameters

        Returns:
            Detections object (xyxy, confidence, class_id)
        """
        model = self._create_model()

        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Run inference using RF-DETR's predict method
        results = model.predict(
            source=image,
            conf=threshold,
            **kwargs,
        )

        # RF-DETR predict() returns a list of Results objects
        if not results or len(results) == 0:
            # No detections
            return Detections(
                xyxy=np.array([]).reshape(0, 4),
                confidence=np.array([]),
                class_id=np.array([]),
            )

        # Get first result (single image)
        result = results[0]

        # Extract detection information
        # RF-DETR uses ultralytics-style Results object
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.array([]).reshape(0, 4)
            confidence = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.array([])
            class_id = boxes.cls.cpu().numpy() + 1 if hasattr(boxes, "cls") else np.array([])  # 1-indexed
        else:
            # No boxes detected
            xyxy = np.array([]).reshape(0, 4)
            confidence = np.array([])
            class_id = np.array([])

        return Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Loading checkpoint: {ckpt_path}")

        # Load checkpoint weights. weights_only=False: RF-DETR checkpoints store
        # an argparse.Namespace (train args) alongside the weights, which the
        # torch>=2.6 default (weights_only=True) refuses to unpickle.
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Get model if not already created
        model = self._create_model()

        # RF-DETR wraps the underlying nn.Module in a `.model` attribute.
        if not hasattr(model, "model"):
            raise AttributeError(
                "Expected RF-DETR object to expose a '.model' attribute holding the "
                f"nn.Module, but got {type(model).__name__}. The RF-DETR API may have "
                "changed; update load_checkpoint() accordingly."
            )

        # Load state dict (checkpoints may store weights under a 'model' key)
        weights = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict
        model.model.load_state_dict(weights)

        print("Checkpoint loaded successfully")

    @staticmethod
    def _known_rfdetr_train_keys() -> set[str] | None:
        """Best-effort set of valid RF-DETR ``train()`` keyword names.

        Used only to warn on likely typos -- never to filter. Sourced from
        RF-DETR's own ``TrainConfig`` model and argument parser so it tracks the
        installed library version. Returns ``None`` if neither can be introspected
        (API drift), in which case validation is skipped rather than risk emitting
        false-positive warnings.
        """
        keys: set[str] = set()
        try:
            from rfdetr.config import TrainConfig

            keys |= set(TrainConfig.model_fields)
        except Exception:
            pass
        try:
            from rfdetr.main import get_args_parser

            keys |= {a.dest for a in get_args_parser()._actions if a.dest and a.dest != "help"}
        except Exception:
            pass
        return keys or None

    def extract_train_kwargs(self, config: dict[str, Any]) -> dict[str, Any]:
        """Forward the ``rfdetr_training`` block to RF-DETR's ``train()`` API.

        Following the principle "YAML = the complete experiment record",
        keys are passed through verbatim instead of being enumerated here with
        hardcoded defaults. Omitting a key lets RF-DETR apply its own library
        default. The benefits:

        - **Single source of truth**: defaults live only in RF-DETR / the YAML,
          never duplicated in this wrapper (so they cannot silently drift).
        - **Zero-code extensibility**: new RF-DETR training parameters work by
          adding them to the YAML -- no edit to this wrapper required.

        Unknown keys are *warned* about (likely typos) but still forwarded, in the
        same spirit as the signature-based kwargs filtering in
        ``src/common/registry_base.py``.

        Args:
            config: Full YAML config dict

        Returns:
            RF-DETR training kwargs
        """
        kwargs: dict[str, Any] = dict(config.get("rfdetr_training", {}))

        # Reproducibility: re-seed inside RF-DETR right before it builds and trains
        # the model so a fixed top-level seed yields the same weights. Only bridge
        # the top-level seed when the user did not set one in rfdetr_training.
        kwargs.setdefault("seed", config.get("seed", 42))

        # A YAML `key: null` means "use RF-DETR's default", so drop None sentinels.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Typo guard: warn (never filter) on keys RF-DETR's train API does not know.
        known = self._known_rfdetr_train_keys()
        if known is not None:
            unknown = sorted(set(kwargs) - known)
            if unknown:
                warnings.warn(
                    f"Unknown key(s) in `rfdetr_training`: {unknown}. "
                    "They are forwarded to RF-DETR as-is -- check for typos. "
                    "(Validated against the installed RF-DETR's TrainConfig / arg parser.)",
                    stacklevel=2,
                )

        return kwargs


# Registry factory functions
@register_model("rfdetr_nano")
def build_rfdetr_nano(
    num_classes: int = 6,
    pretrained: bool = True,
    resolution: int | None = None,
    checkpoint_path: str | None = None,
    **kwargs: Any,
) -> RFDETRWrapper:
    """Build RF-DETR nano model."""
    return RFDETRWrapper(
        model_size="nano",
        num_classes=num_classes,
        pretrained=pretrained,
        resolution=resolution,
        checkpoint_path=checkpoint_path,
    )


@register_model("rfdetr_small")
def build_rfdetr_small(
    num_classes: int = 6,
    pretrained: bool = True,
    resolution: int | None = None,
    checkpoint_path: str | None = None,
    **kwargs: Any,
) -> RFDETRWrapper:
    """Build RF-DETR small model."""
    return RFDETRWrapper(
        model_size="small",
        num_classes=num_classes,
        pretrained=pretrained,
        resolution=resolution,
        checkpoint_path=checkpoint_path,
    )


@register_model("rfdetr_medium")
def build_rfdetr_medium(
    num_classes: int = 6,
    pretrained: bool = True,
    resolution: int | None = None,
    checkpoint_path: str | None = None,
    **kwargs: Any,
) -> RFDETRWrapper:
    """Build RF-DETR medium model."""
    return RFDETRWrapper(
        model_size="medium",
        num_classes=num_classes,
        pretrained=pretrained,
        resolution=resolution,
        checkpoint_path=checkpoint_path,
    )


@register_model("rfdetr_base", aliases=["rfdetr"])
def build_rfdetr_base(
    num_classes: int = 6,
    pretrained: bool = True,
    resolution: int | None = None,
    checkpoint_path: str | None = None,
    **kwargs: Any,
) -> RFDETRWrapper:
    """Build RF-DETR base model (recommended baseline)."""
    return RFDETRWrapper(
        model_size="base",
        num_classes=num_classes,
        pretrained=pretrained,
        resolution=resolution,
        checkpoint_path=checkpoint_path,
    )


@register_model("rfdetr_large")
def build_rfdetr_large(
    num_classes: int = 6,
    pretrained: bool = True,
    resolution: int | None = None,
    checkpoint_path: str | None = None,
    **kwargs: Any,
) -> RFDETRWrapper:
    """Build RF-DETR large model (high accuracy, default 560 resolution)."""
    return RFDETRWrapper(
        model_size="large",
        num_classes=num_classes,
        pretrained=pretrained,
        resolution=resolution,
        checkpoint_path=checkpoint_path,
    )
