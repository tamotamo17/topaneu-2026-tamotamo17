"""Training-pipeline smoke tests on random data (CPU, a few steps).

Covers the 2.5D multiclass vessel pipeline (model_v2_auxbin + focal_tversky_ce +
segmentation_vol_mc) and the 3D Gaussian-heatmap detector (heatmap_3d).
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES = 4
IN_CHANNELS = 5
SIZE = 32


def _build_components(scheduler: bool = False):
    from src.segmentation.registries.lightning_modules import build_lightning_module
    from src.segmentation.registries.losses import build_loss
    from src.segmentation.registries.models import build_model
    from src.segmentation.registries.optimizers import build_optimizer, build_scheduler

    model = build_model(
        "model_v2_auxbin",
        arch="Unet",
        backbone="resnet18",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
    )
    loss = build_loss("focal_tversky_ce", dice_weight=0.3, aux_weight=0.3)
    optimizer = build_optimizer(model.parameters(), name="adamw", lr=1e-3, weight_decay=1e-4)
    sched = build_scheduler(optimizer, name="cosine", T_max=10) if scheduler else None
    module = build_lightning_module(
        "segmentation_vol_mc",
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=sched,
        num_classes=NUM_CLASSES,
    )
    return model, loss, module


def _make_batch(batch_size: int = 2, case_ids: list[str] | None = None) -> dict:
    """Random 2.5D batch with integer label masks [B, 1, H, W] and per-slice sample ids."""
    case_ids = case_ids or [f"case{i}_z{i:03d}" for i in range(batch_size)]
    return {
        "image": torch.randn(batch_size, IN_CHANNELS, SIZE, SIZE),
        "masks": torch.randint(0, NUM_CLASSES, (batch_size, 1, SIZE, SIZE)),
        "sample_id": case_ids,
    }


class _RandomSliceDataset(Dataset):
    """Tiny random dataset with the same sample contract as slices_2_5d_mc."""

    def __init__(self, n: int = 8):
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, IN_CHANNELS, SIZE, SIZE, generator=g)
        self.masks = torch.randint(0, NUM_CLASSES, (n, 1, SIZE, SIZE), generator=g)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return {"image": self.images[i], "masks": self.masks[i], "sample_id": f"case{i // 4}_z{i:03d}"}


def _silence_logging(module) -> dict:
    """Capture self.log calls so steps can run without a Trainer attached."""
    logged: dict = {}
    module.log = lambda k, v, **kw: logged.__setitem__(k, v)
    return logged


class TestTrainingPipeline:
    """Integration tests for the 2.5D multiclass training pipeline."""

    def test_forward_pass(self):
        model, _, _ = _build_components()
        output = model(_make_batch())
        assert output["masks"].shape == (2, NUM_CLASSES, SIZE, SIZE)
        assert output["aux_masks"].shape == (2, 1, SIZE, SIZE)

    def test_loss_and_backward(self):
        """The loss is a finite positive scalar and gradients reach every parameter."""
        model, loss_fn, _ = _build_components()
        batch = _make_batch()
        loss = loss_fn(model(batch), batch)
        assert loss.ndim == 0
        assert torch.isfinite(loss) and loss > 0
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, name

    def test_training_step(self):
        _, _, module = _build_components()
        logged = _silence_logging(module)
        loss = module.training_step(_make_batch(), 0)
        assert torch.isfinite(loss)
        assert "train_loss" in logged

    def test_validation_aggregates_per_case(self):
        """Validation accumulates Dice per case, grouping slices by stripping the _zNNN suffix."""
        _, _, module = _build_components()
        logged = _silence_logging(module)
        module.eval()
        module.on_validation_epoch_start()
        with torch.no_grad():
            module.validation_step(_make_batch(case_ids=["caseA_z000", "caseA_z001"]), 0)
            module.validation_step(_make_batch(case_ids=["caseB_z000", "caseA_z002"]), 1)
        assert set(module._val_acc) == {"caseA", "caseB"}
        module.on_validation_epoch_end()
        assert 0.0 <= float(logged["val_dice"]) <= 1.0
        assert logged["val_num_cases"] == 2.0

    def test_configure_optimizers_with_scheduler(self):
        _, _, module = _build_components(scheduler=True)
        cfg = module.configure_optimizers()
        assert isinstance(cfg["optimizer"], torch.optim.AdamW)
        assert isinstance(cfg["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_manual_training_loop_reduces_loss(self):
        """A few optimizer steps on a fixed batch reduce the loss (the pipeline can learn)."""
        torch.manual_seed(0)
        _, _, module = _build_components()
        _silence_logging(module)
        module.train()
        opt = module.configure_optimizers()
        batch = _make_batch(batch_size=4)

        losses = []
        for step in range(5):
            opt.zero_grad()
            loss = module.training_step(batch, step)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert all(torch.isfinite(torch.tensor(losses)))
        assert losses[-1] < losses[0]

    def test_lightning_trainer_fit(self, tmp_path):
        """One train + val batch through a real Lightning Trainer (fast_dev_run)."""
        import lightning.pytorch as pl

        _, _, module = _build_components(scheduler=True)
        loader = DataLoader(_RandomSliceDataset(8), batch_size=4, num_workers=0)
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            fast_dev_run=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=tmp_path,
        )
        trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
        assert trainer.state.finished
        assert "train_loss" in trainer.callback_metrics


class TestHeatmapTrainingLoop:
    """A few training steps of the 3D Gaussian-heatmap detector."""

    def test_heatmap_3d_loop(self):
        from src.segmentation.registries.lightning_modules import build_lightning_module
        from src.segmentation.registries.losses import build_loss
        from src.segmentation.registries.models import build_model
        from src.segmentation.registries.optimizers import build_optimizer

        torch.manual_seed(0)
        model = build_model("monai_segresnet3d", in_channels=2, num_classes=1, init_filters=8)
        optimizer = build_optimizer(model.parameters(), name="adamw", lr=1e-3)
        module = build_lightning_module(
            "heatmap_3d",
            model=model,
            loss=build_loss(type="heatmap_focal", gamma=2.0, fg_weight=50.0, normalize="mean"),
            optimizer=optimizer,
            roi_size=32,
            sw_batch_size=2,
        )
        _silence_logging(module)
        module.train()

        masks = torch.zeros(2, 1, 32, 32, 32)
        masks[0, 0, 16, 16, 16] = 1.0
        batch = {"image": torch.randn(2, 2, 32, 32, 32), "masks": masks}
        for step in range(3):
            optimizer.zero_grad()
            loss = module.training_step(batch, step)
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss)


class TestConfigHandling:
    """OmegaConf handling as used by the training scripts."""

    def test_config_merge(self):
        from omegaconf import OmegaConf

        base = OmegaConf.create(
            {
                "model": {"type": "model_v2_auxbin", "backbone": "efficientnet-b3", "num_classes": 37},
                "trainer": {"num_epoch": 55},
            }
        )
        override = OmegaConf.create({"model": {"num_classes": 20}, "trainer": {"num_epoch": 5}})
        merged = OmegaConf.merge(base, override)
        assert merged.model.num_classes == 20
        assert merged.model.backbone == "efficientnet-b3"
        assert merged.trainer.num_epoch == 5

    def test_build_from_config_blocks(self):
        """Config blocks (with `type`) can be passed straight to the builders."""
        from omegaconf import OmegaConf

        from src.segmentation.registries.losses import build_loss
        from src.segmentation.registries.models import build_model

        cfg = OmegaConf.create(
            {
                "model": {"type": "model_v2_auxbin", "arch": "Unet", "backbone": "resnet18",
                          "encoder_weights": None, "in_channels": 5, "num_classes": 4},
                "loss": {"type": "focal_tversky_ce", "class_weights": [0.2, 1.0, 1.0, 1.0]},
            }
        )
        model = build_model(**OmegaConf.to_container(cfg.model, resolve=True))
        loss_fn = build_loss(**OmegaConf.to_container(cfg.loss, resolve=True))
        batch = _make_batch()
        assert torch.isfinite(loss_fn(model(batch), batch))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
