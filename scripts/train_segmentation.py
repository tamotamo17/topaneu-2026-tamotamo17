#!/usr/bin/env python
"""Segmentation Training Script.

Dedicated training script for segmentation tasks.

Usage:
    python scripts/train_segmentation.py --config configs/segmentation/step2_vessel_2_5d_fold0.yaml

    # With debug mode
    python scripts/train_segmentation.py --config configs/segmentation/step2_vessel_2_5d_fold0.yaml --debug

    # Specific fold for cross-validation
    python scripts/train_segmentation.py --config configs/segmentation/step2_vessel_2_5d_fold0.yaml --fold 0

    # Specify GPU
    python scripts/train_segmentation.py --config configs/segmentation/step2_vessel_2_5d_fold0.yaml --gpu 0

Using a real dataset (no change to this script needed):
    1. Implement a Dataset in src/segmentation/dataloaders/datasets.py and
       register it with @register_dataset("<name>").
    2. Point `dataset.type` in the YAML at "<name>"; its other keys become the
       factory's kwargs. `transforms` / `fold` / `is_train` are injected here
       automatically (the registry drops kwargs your factory doesn't declare).
       transforms (albumentations) are applied as transforms(image=img, mask=mask)
       with mask shaped [H, W, num_classes].
    3. __getitem__ must return:
       {"image": Tensor[C,H,W], "masks": Tensor[num_classes,H,W], "sample_id": str}
    See docs/dataset_classification_segmentation.md for a copy-paste skeleton.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.segmentation.registries.datasets import build_dataset
from src.segmentation.registries.lightning_modules import build_lightning_module
from src.segmentation.registries.losses import build_loss
from src.segmentation.registries.models import build_model
from src.segmentation.registries.optimizers import build_optimizer, build_scheduler
from src.segmentation.registries.samplers import build_sampler
from src.segmentation.transforms import build_transforms

from lightning.pytorch.callbacks import Callback


class EpochScorePrinter(Callback):
    """Print key metrics to stdout at the end of every validation epoch.

    Keeps a one-line per-epoch score record in the (nohup) log so progress is
    visible without TensorBoard. Metrics come from ``trainer.callback_metrics``;
    missing keys print ``n/a``.
    """

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics

        def g(key: str) -> str:
            v = m.get(key)
            return f"{float(v):.4f}" if v is not None else "n/a"

        print(
            f"[epoch {trainer.current_epoch:03d}] "
            f"train_loss={g('train_loss')} val_loss={g('val_loss')} "
            f"val_dice={g('val_dice')} val_num_cases={g('val_num_cases')}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for cross-validation",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID (default: use config or auto)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (limit samples)",
    )
    return parser.parse_args()


def build_dataloader(
    dataset,
    config: DictConfig,
    is_train: bool = True,
    sampler=None,
    generator=None,
) -> DataLoader:
    """Build a DataLoader from config.

    Args:
        dataset: Dataset instance
        config: Dataloader config section
        is_train: Whether this is training data
        sampler: Optional sampler
        generator: Optional torch.Generator seeding shuffle order

    Returns:
        DataLoader instance
    """
    from src.common.seed import seed_worker

    dl_config = config.train if is_train else config.valid

    return DataLoader(
        dataset,
        batch_size=dl_config.get("batch_size", 16),
        shuffle=dl_config.get("shuffle", is_train) if sampler is None else False,
        num_workers=dl_config.get("num_workers", 2),
        pin_memory=dl_config.get("pin_memory", True),
        drop_last=dl_config.get("drop_last", is_train),
        sampler=sampler,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def check_num_classes_consistency(config: DictConfig) -> None:
    """Fail fast if num_classes disagrees across config sections.

    num_classes is hand-written in several blocks (dataset / val_dataset /
    model / lightning_module). If they drift apart the run silently mis-trains
    (e.g. model emits N channels but the loss/metrics expect M). This validates
    that every section that *declares* num_classes agrees.

    Missing keys are ignored (a section that does not declare num_classes is
    skipped via .get); only sections that explicitly set it are compared.

    Args:
        config: Loaded experiment config.

    Raises:
        ValueError: If two or more sections declare conflicting num_classes.
    """
    sections = ["dataset", "val_dataset", "model", "lightning_module"]
    found: dict[str, int] = {}
    for name in sections:
        section = config.get(name)
        if section is None:
            continue
        value = section.get("num_classes")
        if value is None:
            continue
        found[name] = int(value)

    distinct = set(found.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{name}.num_classes={val}" for name, val in found.items())
        raise ValueError(
            "Inconsistent num_classes across config sections: "
            f"{detail}. All sections that declare num_classes must agree."
        )


def main():
    """Main training function."""
    args = parse_args()

    config = OmegaConf.load(args.config)

    if args.debug:
        config.debug = OmegaConf.create({"enabled": True, "max_samples": 50})

    # Fail fast on num_classes drift across dataset/model/lightning_module
    check_num_classes_consistency(config)

    # Fix all RNGs before building anything so model weights are reproducible
    from src.common.seed import set_seed

    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", False)
    set_seed(seed, deterministic=deterministic)
    print(f"Seed: {seed} (deterministic={deterministic})")

    # Generator seeding DataLoader shuffle order (workers seeded via seed_worker)
    import torch

    data_generator = torch.Generator()
    data_generator.manual_seed(seed)

    print("Task: segmentation")
    print(f"Config: {args.config}")
    print(f"Fold: {args.fold}")
    if args.gpu is not None:
        print(f"GPU: {args.gpu}")

    train_transform = build_transforms(
        preprocessing_config=config.preprocessing,
        augmentation_config=config.get("augmentation"),
        is_train=True,
    )
    val_transform = build_transforms(
        preprocessing_config=config.preprocessing,
        augmentation_config=None,
        is_train=False,
    )

    # `type` is read from the config block; transforms/fold/is_train are injected.
    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    train_dataset = build_dataset(transforms=train_transform, fold=args.fold, is_train=True, **dataset_config)

    val_dataset_config = OmegaConf.to_container(config.get("val_dataset", config.dataset), resolve=True)
    val_dataset = build_dataset(transforms=val_transform, fold=args.fold, is_train=False, **val_dataset_config)

    if config.get("debug", {}).get("enabled", False):
        max_samples = config.debug.get("max_samples", 50)
        from torch.utils.data import Subset

        train_dataset = Subset(train_dataset, list(range(min(max_samples, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(max_samples, len(val_dataset)))))
        print(f"Debug mode: limiting to {max_samples} samples")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    sampler = None
    if "sampler" in config and config.sampler is not None:
        sampler_config = OmegaConf.to_container(config.sampler, resolve=True)
        sampler = build_sampler(train_dataset, **sampler_config)
        print(f"Using sampler: {sampler_config['type']}")

    train_loader = build_dataloader(
        train_dataset, config.dataloader, is_train=True, sampler=sampler, generator=data_generator
    )
    val_loader = build_dataloader(val_dataset, config.dataloader, is_train=False, generator=data_generator)

    model_config = OmegaConf.to_container(config.model, resolve=True)
    model = build_model(**model_config)
    print(f"Model: {model_config['type']}")

    loss_config = OmegaConf.to_container(config.loss, resolve=True)
    loss = build_loss(**loss_config)
    print(f"Loss: {loss_config['type']}")

    opt_config = OmegaConf.to_container(config.optimizer, resolve=True)
    optimizer = build_optimizer(model.parameters(), **opt_config)
    print(f"Optimizer: {opt_config['type']}")

    scheduler = None
    if "scheduler" in config and config.scheduler is not None:
        sched_config = OmegaConf.to_container(config.scheduler, resolve=True)
        scheduler = build_scheduler(optimizer, **sched_config)
        print(f"Scheduler: {sched_config['type']}")

    module_config = OmegaConf.to_container(config.lightning_module, resolve=True)
    module = build_lightning_module(
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        **module_config,
    )
    print(f"Lightning Module: {module_config['type']}")

    trainer_config = config.trainer
    output_dir = (
        Path(trainer_config.output_root_path) / trainer_config.get("exp_name", "experiment") / f"fold{args.fold}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config for reproducibility
    import shutil

    config_save_path = output_dir / "config.yaml"
    shutil.copy(args.config, config_save_path)
    print(f"Config saved to: {config_save_path}")

    # Setup callbacks (configurable via the `trainer` section)
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    monitor = trainer_config.get("monitor", "val_loss")
    monitor_mode = trainer_config.get("monitor_mode", "min")
    save_top_k = trainer_config.get("save_top_k", 1)

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch:02d}-{" + monitor + ":.4f}",
            monitor=monitor,
            mode=monitor_mode,
            save_top_k=save_top_k,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EpochScorePrinter(),
    ]

    # Early stopping (optional; enabled by default)
    if trainer_config.get("early_stopping", True):
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=trainer_config.get("early_stopping_patience", 5),
                mode=monitor_mode,
            )
        )

    if args.gpu is not None:
        devices = [args.gpu]
        accelerator = "gpu"
    else:
        devices = trainer_config.get("devices", "auto")
        accelerator = trainer_config.get("accelerator", "auto")

    import lightning as L
    from lightning.pytorch.loggers import CSVLogger

    # Per-epoch metrics persisted to <output_dir>/metrics.csv
    csv_logger = CSVLogger(save_dir=str(output_dir), name="", version="")

    trainer = L.Trainer(
        max_epochs=trainer_config.num_epoch,
        accelerator=accelerator,
        devices=devices,
        precision=trainer_config.get("precision", "32-true"),
        default_root_dir=output_dir,
        callbacks=callbacks,
        logger=csv_logger,
        enable_progress_bar=True,
        log_every_n_steps=trainer_config.get("log_every_n_steps", 10),
        deterministic=deterministic,
    )

    print("\nStarting training...")
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_path = output_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")

    print(f"Best val_loss: {trainer.callback_metrics.get('val_loss', 'N/A')}")
    print(f"Best val_dice: {trainer.callback_metrics.get('val_dice', 'N/A')}")


if __name__ == "__main__":
    main()
