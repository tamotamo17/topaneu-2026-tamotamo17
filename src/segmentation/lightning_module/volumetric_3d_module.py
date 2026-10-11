"""3D (patch-train / sliding-window-val) multiclass segmentation Lightning module.

Trains on 3D patches; validates/tests with MONAI ``sliding_window_inference`` over
the full volume, computing a per-case multiclass 3D Dice (mean over foreground
classes present in the GT), reported as ``val_dice`` — directly comparable to the
2.5D ``segmentation_vol_mc`` metric.
"""

from __future__ import annotations

from typing import Any

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from monai.inferers import sliding_window_inference

from src.segmentation.registries.lightning_modules import register_lightning_module


class Volumetric3DSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        num_classes: int = 37,
        roi_size: int = 96,
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        include_background: bool = False,
        dice_smooth: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.roi = (roi_size, roi_size, roi_size)
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.include_background = include_background
        self.dice_smooth = dice_smooth
        self._val_dices: list[float] = []
        self.save_hyperparameters(ignore=["model", "loss", "optimizer", "scheduler"])

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        loss = self.loss_fn(outputs, batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["image"].size(0))
        return loss

    # -- sliding-window validation ----------------------------------------------
    def _predict_full(self, image):
        def _pred(x):
            return self.model({"image": x})["masks"]
        return sliding_window_inference(image, self.roi, self.sw_batch_size, _pred,
                                        overlap=self.overlap, mode="gaussian")

    def _case_dice(self, pred, gt):
        start = 0 if self.include_background else 1
        dices = []
        for c in range(start, self.num_classes):
            g = gt == c
            gs = g.sum()
            if gs == 0:
                continue
            p = pred == c
            inter = (p & g).sum().float()
            dices.append((2 * inter + self.dice_smooth) / (p.sum().float() + gs.float() + self.dice_smooth))
        return torch.stack(dices).mean().item() if dices else None

    def on_validation_epoch_start(self):
        self._val_dices = []

    def validation_step(self, batch, batch_idx):
        image = batch["image"]           # [1, 1, X, Y, Z]
        gt = batch["masks"][:, 0]        # [1, X, Y, Z]
        logits = self._predict_full(image)
        pred = logits.argmax(dim=1)      # [1, X, Y, Z]
        d = self._case_dice(pred[0], gt[0])
        if d is not None:
            self._val_dices.append(d)
        # lightweight val loss on a center-cropped region is skipped; monitor val_dice.
        return None

    def on_validation_epoch_end(self):
        val = float(sum(self._val_dices) / len(self._val_dices)) if self._val_dices else 0.0
        self.log("val_dice", val, prog_bar=True, sync_dist=True)
        self.log("val_num_cases", float(len(self._val_dices)), sync_dist=True)

    def configure_optimizers(self) -> dict[str, Any] | Optimizer:
        if self.scheduler is not None:
            return {"optimizer": self.optimizer,
                    "lr_scheduler": {"scheduler": self.scheduler, "interval": "epoch", "frequency": 1}}
        return self.optimizer


@register_lightning_module("segmentation_3d", aliases=["seg_3d_vol", "volumetric_3d"])
def build_volumetric_3d_module(
    model: nn.Module,
    loss: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    num_classes: int = 37,
    roi_size: int = 96,
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    include_background: bool = False,
    dice_smooth: float = 1e-6,
    **kwargs,
) -> Volumetric3DSegmentationModule:
    return Volumetric3DSegmentationModule(
        model=model, loss=loss, optimizer=optimizer, scheduler=scheduler,
        num_classes=num_classes, roi_size=roi_size, sw_batch_size=sw_batch_size,
        overlap=overlap, include_background=include_background, dice_smooth=dice_smooth,
    )
