"""3D Gaussian-heatmap regression module (patch-train / sliding-window-val).

For center-peaked Gaussian sphere targets (continuous [0,1]) instead of a filled
binary ball. 1-channel sigmoid output trained with the penalty-reduced focal loss
built from the config's ``loss:`` section (``heatmap_focal``), which down-weights
the easy background and up-weights the near-peak neighbourhood.

Val = per-case Dice of (pred>thr) vs (gt>thr) as a localisation proxy. Note this
metric mixes detection (a missed aneurysm scores 0) with shape agreement over a
~3mm ball, so a few hit/miss flips move it by ~0.05 — use the coordinate FROC
(``scripts/froc_aneurysm_3d.py``) for decisions finer than that.

Optional auxiliary task (``cls_weight > 0``, needs a model that also returns
``cls_logits`` — e.g. ``monai_segresnet3d_auxcls``): a patch-level "contains an
aneurysm centre" binary head off the encoder bottleneck. It adds a whole-patch
gradient to a network whose only other signal covers 0.0087% of voxels, and its
sliding-window max is reported as ``val_cls_auc``, a case-level presence score
that can gate blob FPs downstream. Default ``cls_weight=0.0`` keeps every
existing config numerically unchanged.
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


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """ROC-AUC via the rank-sum identity; ``None`` if one class is absent.

    Kept dependency-free (no sklearn import in the training loop) and tie-safe:
    equal scores share their average rank, so a head that has collapsed to a
    constant scores 0.5 rather than 0 or 1.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, lab in zip(ranks, labels) if lab)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


class Heatmap3DModule(pl.LightningModule):
    def __init__(self, model, loss, optimizer, scheduler=None, roi_size=96, sw_batch_size=2,
                 overlap=0.5, val_thr=0.5, dice_smooth=1e-6,
                 cls_weight=0.0, cls_thr=0.5, cls_pos_weight=None, ds_weight=0.0):
        super().__init__()
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.roi = (roi_size,) * 3
        self.sw_batch_size, self.overlap = sw_batch_size, overlap
        self.val_thr = val_thr
        self.dice_smooth = dice_smooth
        # auxiliary patch-level "contains an aneurysm centre" head (0.0 => off)
        self.cls_weight = float(cls_weight)
        self.cls_thr = float(cls_thr)
        self.cls_pos_weight = cls_pos_weight
        # nnU-Net-style deep supervision (models that return ``ds_masks``)
        self.ds_weight = float(ds_weight)
        self._val = []
        self._cls_score, self._cls_label = [], []
        self.save_hyperparameters(ignore=["model", "loss", "optimizer", "scheduler"])

    def forward(self, batch):
        return self.model(batch)

    def _cls_loss(self, cls_logits, masks):
        """BCE on "this patch contains an aneurysm centre".

        The target is derived from the heatmap target with the same threshold the
        seg metric uses, so positive means the patch holds a Gaussian PEAK: a
        patch that merely clips the tail of a neighbouring aneurysm stays
        negative and the head is not asked to fire on partial evidence.
        """
        tgt = (masks[:, 0] > self.cls_thr).flatten(1).any(1).float()      # [B]
        pw = (torch.as_tensor(self.cls_pos_weight, dtype=cls_logits.dtype, device=cls_logits.device)
              if self.cls_pos_weight is not None else None)
        return F.binary_cross_entropy_with_logits(cls_logits[:, 0], tgt, pos_weight=pw)

    def _ds_loss(self, ds, target):
        """Weighted loss over the auxiliary deep-supervision heads.

        MONAI already upsamples every head to full resolution, so each one is
        scored against the same target. Weights halve per level (nnU-Net's
        rule) and are normalised, so ``ds_weight`` alone controls how much the
        auxiliary heads count relative to the main one.
        """
        n = ds.size(1)
        w = [0.5 ** i for i in range(n)]
        s = sum(w)
        return sum((wi / s) * self.loss_fn(ds[:, i], target) for i, wi in enumerate(w))

    def training_step(self, batch, _):
        out = self.forward(batch)
        bs = batch["image"].size(0)
        target = batch["masks"].float()
        seg = self.loss_fn(out["masks"], target)
        if self.ds_weight > 0 and "ds_masks" in out:
            ds = self._ds_loss(out["ds_masks"], target)
            seg = seg + self.ds_weight * ds
            self.log("train_ds", ds, on_step=True, on_epoch=True, batch_size=bs)
        if self.cls_weight <= 0 or "cls_logits" not in out:
            self.log("train_loss", seg, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
            return seg
        cls = self._cls_loss(out["cls_logits"], target)
        total = seg + self.cls_weight * cls
        self.log_dict({"train_loss": total, "train_seg": seg, "train_cls": cls},
                      on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        return total

    def on_validation_epoch_start(self):
        self._val = []
        self._cls_score, self._cls_label = [], []

    def validation_step(self, batch, _):
        image = batch["image"]
        gt = batch["masks"][:, 0].float()
        win_cls: list[torch.Tensor] = []

        def _predict(x):
            out = self.model({"image": x})
            if "cls_logits" in out:
                win_cls.append(out["cls_logits"].detach().float())
            return out["masks"]

        logits = sliding_window_inference(image, self.roi, self.sw_batch_size, _predict,
                                          overlap=self.overlap, mode="gaussian")
        p = (torch.sigmoid(logits[:, 0]) > self.val_thr).float()
        g = (gt > self.val_thr).float()
        for i in range(p.size(0)):
            if g[i].sum() > 0:
                inter = (p[i] * g[i]).sum()
                self._val.append(float((2 * inter + self.dice_smooth) / (p[i].sum() + g[i].sum() + self.dice_smooth)))
        # Case-level presence score = max aux logit over the sliding windows.
        # sliding_window_inference pools windows across the batch dim, so the
        # windows can only be attributed to a case when the batch holds one
        # (valid dataloader batch_size is 1); otherwise skip rather than mix.
        if win_cls and image.size(0) == 1:
            self._cls_score.append(float(torch.cat(win_cls)[:, 0].max()))
            self._cls_label.append(int(g[0].sum() > 0))
        return None

    def on_validation_epoch_end(self):
        self.log("val_dice", float(sum(self._val) / len(self._val)) if self._val else 0.0, prog_bar=True, sync_dist=True)
        if self._cls_label:
            auc = _roc_auc(self._cls_score, self._cls_label)
            if auc is not None:
                self.log("val_cls_auc", auc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> dict[str, Any] | Optimizer:
        if self.scheduler is not None:
            return {"optimizer": self.optimizer,
                    "lr_scheduler": {"scheduler": self.scheduler, "interval": "epoch", "frequency": 1}}
        return self.optimizer


@register_lightning_module("heatmap_3d", aliases=["gaussian_heatmap_3d"])
def build_heatmap_3d(model, loss, optimizer, scheduler=None, roi_size=96, sw_batch_size=2,
                     overlap=0.5, val_thr=0.5, dice_smooth=1e-6,
                     cls_weight=0.0, cls_thr=0.5, cls_pos_weight=None, ds_weight=0.0,
                     **kwargs) -> Heatmap3DModule:
    """Build the heatmap module.

    ``loss`` comes from the config's ``loss:`` section — use ``heatmap_focal``.
    The focal/foreground hyper-parameters live there, not here: they used to be
    module arguments while the built loss was silently discarded, so the config
    said ``ce_dice`` while training actually ran a hardcoded focal soft-BCE.

    ``cls_weight > 0`` enables the auxiliary patch-level presence head and
    requires a model returning ``cls_logits`` (``monai_segresnet3d_auxcls``).
    Asking for it with a seg-only model is a silent no-op in ``training_step``,
    so it is rejected here instead.
    """
    if "focal_gamma" in kwargs or "fg_weight" in kwargs:
        raise ValueError(
            "focal_gamma/fg_weight moved to the loss section: use "
            "loss: {type: heatmap_focal, gamma: ..., fg_weight: ...}")
    if float(cls_weight) > 0 and not hasattr(model, "cls_head"):
        raise ValueError(
            f"cls_weight={cls_weight} needs a model with an auxiliary classification head "
            f"(model: {{type: monai_segresnet3d_auxcls, ...}}), got {type(model).__name__}")
    return Heatmap3DModule(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler,
                           roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap,
                           val_thr=val_thr, dice_smooth=dice_smooth,
                           cls_weight=cls_weight, cls_thr=cls_thr, cls_pos_weight=cls_pos_weight,
                           ds_weight=ds_weight)
