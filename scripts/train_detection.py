#!/usr/bin/env python3
"""Detection Training Script.

Entry point for training detection models (Step1 vessel-ROI detection).

- The script is library-agnostic (it does not know model details)
- All training logic lives in the model wrapper (src/detection/models/)
- The wrapper handles config conversion and training execution

Usage:
    PYTHONPATH=. uv run python scripts/train_detection.py \
        --config configs/detection/step1_vessel_roi.yaml \
        --fold 0 --gpu 0

Using a real dataset (no Dataset class needed):
    RF-DETR reads a Roboflow-format COCO directory directly. Provide:
        <coco_root>/{train,valid,test}/  each with _annotations.coco.json + images
        (all three splits are required by RF-DETR; category_id is 1-indexed).
    Then set in the YAML:
        dataset: {type: coco_detection, coco_root: ./data/my_coco}
        model:   {type: rfdetr_base, num_classes: <your N>}
    or pass a prebuilt directory via --dataset-dir <dir> --skip-dataset-build.
    See docs/dataset_detection.md.

Design:
    1. Load YAML config
    2. Prepare dataset directory (if needed)
    3. Build model wrapper from registry
    4. Call model.run_training_pipeline()

The wrapper handles everything else:
    - Config conversion (YAML → library format)
    - Training execution
    - Evaluation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf


def prepare_dataset(dataset_config: dict, seed: int = 42) -> Path | None:
    """Prepare dataset if needed.

    For dummy datasets, generates a reproducible Roboflow-format COCO dataset on
    disk (under the project) so RF-DETR can actually train on it.
    For COCO datasets, validates the directory exists.

    Args:
        dataset_config: Dataset configuration from YAML
        seed: Global seed used to make the generated dummy data reproducible

    Returns:
        Path to dataset directory
    """
    dataset_type = dataset_config.get("type", "")

    # Dummy datasets: materialize a COCO dataset on disk for RF-DETR.
    if dataset_type in ["random_detection", "random", "dummy"]:
        from src.detection.dataloaders.datasets import create_dummy_coco_dataset

        num_samples = dataset_config.get("num_samples", 100)
        num_classes = dataset_config.get("num_classes", 6)
        image_size = dataset_config.get("image_size", 640)
        max_boxes = dataset_config.get("max_boxes_per_image", 10)
        # Allow override; default to a project-local, reproducible cache path.
        out_root = dataset_config.get(
            "output_dir",
            f"./data/detection_dummy/seed{seed}_n{num_samples}_c{num_classes}_s{image_size}",
        )
        print(f"Generating dummy COCO dataset at: {out_root}")
        return create_dummy_coco_dataset(
            root=out_root,
            num_samples=num_samples,
            num_classes=num_classes,
            image_size=image_size,
            max_boxes_per_image=max_boxes,
            seed=seed,
        )

    # COCO datasets need validation
    if dataset_type in ["coco_detection", "coco"]:
        coco_root = dataset_config.get("coco_root")
        if not coco_root:
            raise ValueError(
                "COCO dataset requires 'coco_root' parameter in config.\n"
                "Example:\n"
                "  dataset:\n"
                "    type: coco_detection\n"
                "    coco_root: /path/to/coco_dataset\n"
            )

        coco_path = Path(coco_root)
        if not coco_path.exists():
            raise FileNotFoundError(f"COCO dataset directory not found: {coco_path}")

        return coco_path

    # Unknown dataset type
    raise ValueError(f"Unknown dataset type: '{dataset_type}'\nSupported types: random_detection, coco_detection")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for cross-validation (default: 0)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID (default: auto / leave CUDA_VISIBLE_DEVICES untouched)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override dataset directory (for pre-built datasets)",
    )
    parser.add_argument(
        "--skip-dataset-build",
        action="store_true",
        help="Skip dataset preparation (use existing dataset)",
    )

    args = parser.parse_args()

    # Set GPU device only when explicitly requested
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Loading config: {args.config}")
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Fix all RNGs before building the model so model weights are reproducible
    from src.common.seed import set_seed

    seed = config_dict.get("seed", 42)
    deterministic = config_dict.get("deterministic", False)
    set_seed(seed, deterministic=deterministic)
    print(f"Seed: {seed} (deterministic={deterministic})")

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        print(f"Using provided dataset directory: {dataset_dir}")
    elif args.skip_dataset_build:
        if "coco_root" in config_dict.get("dataset", {}):
            dataset_dir = Path(config_dict["dataset"]["coco_root"])
        else:
            raise ValueError("--skip-dataset-build requires either --dataset-dir or dataset.coco_root in config")
        print(f"Using existing dataset: {dataset_dir}")
    else:
        print("Preparing dataset...")
        dataset_dir = prepare_dataset(config_dict.get("dataset", {}), seed=seed)
        print(f"Dataset ready: {dataset_dir}")

    print(f"Building model: {config_dict['model']['type']}")
    from src.detection.registries.models import build_model

    model = build_model(**config_dict["model"])
    print(f"Model created: {model.__class__.__name__}")

    output_root = Path(config_dict["output"]["root_path"])
    exp_name = config_dict["output"]["exp_name"]
    output_dir = output_root / exp_name / f"fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    import shutil

    config_save_path = output_dir / "config.yaml"
    shutil.copy(args.config, config_save_path)
    print(f"Config saved to: {config_save_path}")

    print(f"Output directory: {output_dir}")

    training_config = config_dict.get("training", {})
    common_params = {
        "epochs": training_config.get("epochs", 10),
        "batch_size": training_config.get("batch_size", 16),
        "lr": training_config.get("lr", 1e-4),
        "num_workers": training_config.get("num_workers", 4),
    }

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"Model: {config_dict['model']['type']}")
    print(f"Dataset: {dataset_dir}")
    print(f"Output: {output_dir}")
    print(f"Fold: {args.fold}")
    print(f"GPU: {args.gpu}")
    print(f"Epochs: {common_params['epochs']}")
    print(f"Batch size: {common_params['batch_size']}")
    print(f"Learning rate: {common_params['lr']}")
    print("=" * 80 + "\n")

    # Run training pipeline (wrapper handles everything)
    try:
        results = model.run_training_pipeline(
            config=config_dict,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            common_params=common_params,
        )

        print("\n" + "=" * 80)
        print("Training Completed")
        print("=" * 80)
        print(f"Output directory: {output_dir}")
        print(f"Results: {results.get('status', 'unknown')}")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        print(f"\nError during training: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
