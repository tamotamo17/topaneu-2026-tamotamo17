"""Segmentation Lightning Module.

Provides a PyTorch Lightning module for segmentation tasks.
"""

from __future__ import annotations

from typing import Any

import lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.segmentation import MeanIoU

from src.segmentation.registries.lightning_modules import register_lightning_module


class SegmentationModule(pl.LightningModule):
    """Lightning module for segmentation tasks.

    Args:
        model: The segmentation model
        loss: Loss function
        optimizer: Optimizer instance
        scheduler: Optional learning rate scheduler
        num_classes: Number of segmentation classes
        threshold: Threshold for converting probabilities to binary masks
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        num_classes: int = 6,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.threshold = threshold

        # Metrics - mean IoU over classes. Masks are multi-channel [B, C, H, W]
        # binary maps, so use the one-hot input format (torchmetrics default).
        self.train_iou = MeanIoU(num_classes=num_classes, per_class=False)
        self.val_iou = MeanIoU(num_classes=num_classes, per_class=False)
        # Dedicated test metric so test_step does not pollute val_iou state
        self.test_iou = MeanIoU(num_classes=num_classes, per_class=False)

        self.save_hyperparameters(ignore=["model", "loss", "optimizer", "scheduler"])

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: Input batch dictionary with "image" key

        Returns:
            Model output dictionary with "masks" key
        """
        return self.model(batch)

    def _compute_dice_score(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0,
    ) -> torch.Tensor:
        """Compute Dice score per class.

        Args:
            pred: Predicted binary masks [B, C, H, W]
            target: Ground truth masks [B, C, H, W]
            smooth: Smoothing factor

        Returns:
            Dice scores per class [C]
        """
        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)

        intersection = (pred * target).sum(dim=(0, 2))
        union = pred.sum(dim=(0, 2)) + target.sum(dim=(0, 2))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Input batch with "image" and "masks" keys
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        outputs = self.forward(batch)
        loss = self.loss_fn(outputs, batch)

        pred_probs = torch.sigmoid(outputs["masks"])
        pred_binary = (pred_probs > self.threshold).long()
        target_binary = (batch["masks"] > self.threshold).long()

        dice_scores = self._compute_dice_score(pred_binary.float(), target_binary.float())
        mean_dice = dice_scores.mean()

        # Update IoU metric (one-hot/multi-hot [B, C, H, W] format)
        self.train_iou.update(pred_binary, target_binary)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_dice", mean_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)

        for c in range(self.num_classes):
            self.log(f"train_dice_class_{c}", dice_scores[c], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Input batch with "image" and "masks" keys
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        outputs = self.forward(batch)
        loss = self.loss_fn(outputs, batch)

        pred_probs = torch.sigmoid(outputs["masks"])
        pred_binary = (pred_probs > self.threshold).long()
        target_binary = (batch["masks"] > self.threshold).long()

        dice_scores = self._compute_dice_score(pred_binary.float(), target_binary.float())
        mean_dice = dice_scores.mean()

        # Update IoU metric (one-hot/multi-hot [B, C, H, W] format)
        self.val_iou.update(pred_binary, target_binary)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_dice", mean_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)

        for c in range(self.num_classes):
            self.log(f"val_dice_class_{c}", dice_scores[c], on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step.

        Mirrors validation_step but uses the dedicated test_iou metric so that
        test evaluation never contaminates validation metric state.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        outputs = self.forward(batch)
        loss = self.loss_fn(outputs, batch)

        pred_probs = torch.sigmoid(outputs["masks"])
        pred_binary = (pred_probs > self.threshold).long()
        target_binary = (batch["masks"] > self.threshold).long()

        dice_scores = self._compute_dice_score(pred_binary.float(), target_binary.float())
        mean_dice = dice_scores.mean()

        self.test_iou.update(pred_binary, target_binary)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_dice", mean_dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_iou", self.test_iou, on_step=False, on_epoch=True, prog_bar=True)

        for c in range(self.num_classes):
            self.log(f"test_dice_class_{c}", dice_scores[c], on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any] | Optimizer:
        """Configure optimizers and schedulers.

        Returns:
            Optimizer configuration
        """
        if self.scheduler is not None:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }
        return self.optimizer


@register_lightning_module("segmentation", aliases=["seg", "segment"])
def build_segmentation_module(
    model: nn.Module,
    loss: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    num_classes: int = 6,
    threshold: float = 0.5,
    **kwargs,
) -> SegmentationModule:
    """Build a segmentation lightning module.

    Args:
        model: Segmentation model
        loss: Loss function
        optimizer: Optimizer
        scheduler: Optional scheduler
        num_classes: Number of classes
        threshold: Threshold for binary masks

    Returns:
        SegmentationModule instance
    """
    return SegmentationModule(
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        num_classes=num_classes,
        threshold=threshold,
    )
