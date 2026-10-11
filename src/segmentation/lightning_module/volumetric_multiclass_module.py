"""Volumetric (3D, per-case) **multiclass** evaluation Lightning Module for 2.5D.

The multiclass counterpart of :class:`VolumetricSegmentationModule`. The model
produces mutually-exclusive class logits ``[B, C, H, W]`` (softmax), so the
per-case metric is a **multiclass** 3D Dice: for each case we accumulate the
intersection / predicted / ground-truth voxel counts **per class** across all of
that case's slices, compute one Dice per (case, class) at epoch end, and report
the mean over the **foreground** classes that are actually present in each case
(``val_dice``). Background (class 0) is excluded from the reported score.

Cases are identified from ``batch["sample_id"]`` (``"<case>_z<center>"``) by
stripping the ``_z<idx>`` suffix, so validation loaders must not shuffle or
drop_last (every slice of every case must be seen exactly once).

Used by Step2 (vessel, C=37) and Step3 (aneurysm-location, C=51).
"""

from __future__ import annotations

import re

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.segmentation import MeanIoU

from src.segmentation.lightning_module.module import SegmentationModule
from src.segmentation.registries.lightning_modules import register_lightning_module

_Z_SUFFIX = re.compile(r"_z\d+$")


class VolumetricMulticlassSegmentationModule(SegmentationModule):
    """2.5D multiclass segmentation module with per-case 3D Dice validation.

    Args:
        model, loss, optimizer, scheduler: as in :class:`SegmentationModule`.
        num_classes: Number of classes **including** background.
        dice_smooth: Smoothing added to per-(case, class) Dice.
        include_background: If False (default), class 0 is excluded from the
            reported mean Dice.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        num_classes: int = 37,
        dice_smooth: float = 1e-6,
        include_background: bool = False,
    ):
        super().__init__(
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            num_classes=num_classes,
            threshold=0.5,
        )
        self.dice_smooth = dice_smooth
        self.include_background = include_background
        # Replace the base (one-hot) IoU metrics with multiclass ones.
        self.train_iou = MeanIoU(num_classes=num_classes, per_class=False, input_format="index")
        self.val_iou = MeanIoU(num_classes=num_classes, per_class=False, input_format="index")
        self.test_iou = MeanIoU(num_classes=num_classes, per_class=False, input_format="index")
        # case_id -> tensor[3, num_classes] of [intersection, pred_sum, gt_sum]
        self._val_acc: dict[str, torch.Tensor] = {}
        self._test_acc: dict[str, torch.Tensor] = {}

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _case_id(sample_id: str) -> str:
        return _Z_SUFFIX.sub("", sample_id)

    @staticmethod
    def _target_labels(batch: dict) -> torch.Tensor:
        """Ground-truth integer labels [B, H, W] from a [B, 1, H, W] mask."""
        target = batch["masks"]
        if target.ndim == 4:
            target = target.squeeze(1)
        return target.long()

    def _accumulate(self, acc: dict[str, torch.Tensor], outputs: dict, batch: dict) -> None:
        pred = outputs["masks"].argmax(dim=1)  # [B, H, W]
        tgt = self._target_labels(batch)  # [B, H, W]
        c = self.num_classes
        # One-hot [B, C, H, W] then sum spatial -> per-sample per-class counts.
        pred_oh = torch.nn.functional.one_hot(pred.clamp(0, c - 1), c).permute(0, 3, 1, 2).float()
        tgt_oh = torch.nn.functional.one_hot(tgt.clamp(0, c - 1), c).permute(0, 3, 1, 2).float()
        dims = (2, 3)
        inter = (pred_oh * tgt_oh).sum(dim=dims)  # [B, C]
        psum = pred_oh.sum(dim=dims)
        gsum = tgt_oh.sum(dim=dims)

        for i, sid in enumerate(batch["sample_id"]):
            case = self._case_id(sid)
            stats = acc.get(case)
            if stats is None:
                stats = torch.zeros(3, c, device=self.device)
                acc[case] = stats
            stats[0] += inter[i]
            stats[1] += psum[i]
            stats[2] += gsum[i]

    def _reduce_across_ranks(self, acc: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not (dist.is_available() and dist.is_initialized()):
            return acc
        world = dist.get_world_size()
        local = {k: v.detach().cpu() for k, v in acc.items()}
        gathered: list = [None] * world
        dist.all_gather_object(gathered, local)
        merged: dict[str, torch.Tensor] = {}
        for part in gathered:
            for case, stats in part.items():
                merged[case] = merged.get(case, torch.zeros_like(stats)) + stats
        return merged

    def _epoch_dice(self, acc: dict[str, torch.Tensor], prefix: str) -> None:
        acc = self._reduce_across_ranks(acc)
        if not acc:
            return
        start = 0 if self.include_background else 1
        case_dices = []
        for stats in acc.values():
            inter, psum, gsum = stats[0], stats[1], stats[2]  # [C] each
            dice = (2.0 * inter + self.dice_smooth) / (psum + gsum + self.dice_smooth)  # [C]
            # Average over foreground classes **present in the ground truth**.
            # A case with no GT foreground (e.g. an aneurysm-free scan) carries no
            # segmentation signal, so it is skipped rather than scored 0 (that is a
            # detection/false-positive concern, not segmentation quality).
            gt_present = gsum[start:] > 0
            if not bool(gt_present.any()):
                continue
            case_dices.append(dice[start:][gt_present].mean())
        n_scored = len(case_dices)
        mean_dice = (
            torch.stack([d.float() for d in case_dices]).mean().to(self.device)
            if n_scored
            else torch.zeros((), device=self.device)
        )
        self.log(f"{prefix}_dice", mean_dice, prog_bar=True, rank_zero_only=True, sync_dist=False)
        self.log(f"{prefix}_num_cases", float(n_scored), rank_zero_only=True, sync_dist=False)

    # -- training ---------------------------------------------------------------

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = self.loss_fn(outputs, batch)
        pred = outputs["masks"].argmax(dim=1)
        self.train_iou.update(pred, self._target_labels(batch))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # -- validation -------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_acc = {}

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = self.loss_fn(outputs, batch)
        self._accumulate(self._val_acc, outputs, batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["masks"].size(0))
        return loss

    def on_validation_epoch_end(self) -> None:
        self._epoch_dice(self._val_acc, "val")

    # -- test -------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        self._test_acc = {}

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        loss = self.loss_fn(outputs, batch)
        self._accumulate(self._test_acc, outputs, batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["masks"].size(0))
        return loss

    def on_test_epoch_end(self) -> None:
        self._epoch_dice(self._test_acc, "test")


@register_lightning_module(
    "segmentation_vol_mc",
    aliases=["seg_vol_mc", "volumetric_multiclass_segmentation"],
)
def build_volumetric_multiclass_segmentation_module(
    model: nn.Module,
    loss: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    num_classes: int = 37,
    dice_smooth: float = 1e-6,
    include_background: bool = False,
    **kwargs,
) -> VolumetricMulticlassSegmentationModule:
    """Build a :class:`VolumetricMulticlassSegmentationModule`."""
    return VolumetricMulticlassSegmentationModule(
        model=model,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        num_classes=num_classes,
        dice_smooth=dice_smooth,
        include_background=include_background,
    )
