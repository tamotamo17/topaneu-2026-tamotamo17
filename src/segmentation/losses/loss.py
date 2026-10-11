"""Segmentation Loss Functions.

Provides common loss functions for semantic segmentation tasks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.segmentation.registries.losses import register_loss


class DiceLoss(nn.Module):
    """Soft Dice Loss for segmentation.

    Args:
        smooth: Smoothing factor to avoid division by zero
        reduction: Reduction method ('mean', 'sum', 'none')

    Input/Output:
        outputs: dict with "masks" key -> Tensor[B, C, H, W] (logits)
        targets: dict with "masks" key -> Tensor[B, C, H, W] (binary)
    """

    def __init__(
        self,
        smooth: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            outputs: Model outputs with "masks" key (logits)
            targets: Ground truth with "masks" key (binary masks)

        Returns:
            Dice loss value
        """
        pred = outputs["masks"]
        target = targets["masks"]

        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)

        pred = pred.view(pred.size(0), pred.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)

        # Compute Dice score per class
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss = 1 - Dice score
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss.

    Args:
        bce_weight: Weight for BCE loss component
        dice_weight: Weight for Dice loss component
        smooth: Smoothing factor for Dice loss

    Input/Output:
        outputs: dict with "masks" key -> Tensor[B, C, H, W] (logits)
        targets: dict with "masks" key -> Tensor[B, C, H, W] (binary)
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss(smooth=smooth, reduction="mean")

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute combined BCE and Dice loss.

        Args:
            outputs: Model outputs with "masks" key (logits)
            targets: Ground truth with "masks" key (binary masks)

        Returns:
            Combined loss value
        """
        pred = outputs["masks"]
        target = targets["masks"]

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target)

        # Dice loss
        dice = self.dice_loss(outputs, targets)

        return self.bce_weight * bce + self.dice_weight * dice


class CEDiceLoss(nn.Module):
    """Combined multiclass Cross-Entropy + soft (softmax) Dice loss.

    For **mutually-exclusive** multiclass segmentation (each voxel has exactly
    one label), unlike the sigmoid-based :class:`BCEDiceLoss` used for binary /
    multi-label maps. Used by Step2 (vessels) and Step3 (aneurysm-location).

    Args:
        ce_weight: Weight for the cross-entropy component.
        dice_weight: Weight for the soft Dice component.
        smooth: Smoothing added to Dice numerator/denominator.
        include_background: If False, class 0 is excluded from the Dice term
            (the CE term always sees every class). Aneurysm/vessel foreground is
            sparse, so excluding the dominant background often helps.

    Input/Output:
        outputs: dict with "masks" -> Tensor[B, C, H, W] (logits)
        targets: dict with "masks" -> Tensor[B, 1, H, W] or [B, H, W] (int labels)
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        include_background: bool = True,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.include_background = include_background
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = outputs["masks"]  # [B, C, H, W]
        target = targets["masks"]
        if target.ndim == logits.ndim:  # [B, 1, H, W] -> [B, H, W]
            target = target.squeeze(1)
        target = target.long()

        num_classes = logits.size(1)
        ce = self.ce(logits, target.clamp(0, num_classes - 1))

        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes)
        onehot = onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        dims = (0, 2, 3)
        intersection = (probs * onehot).sum(dim=dims)
        union = probs.sum(dim=dims) + onehot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [C]

        dice_per_class = dice if self.include_background else dice[1:]
        dice_loss = 1.0 - dice_per_class.mean()

        return self.ce_weight * ce + self.dice_weight * dice_loss


@register_loss("ce_dice", aliases=["dice_ce", "cedice"])
def build_ce_dice_loss(
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    smooth: float = 1.0,
    include_background: bool = True,
    **kwargs,
) -> CEDiceLoss:
    """Build combined CrossEntropy + softmax Dice loss (multiclass)."""
    return CEDiceLoss(
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        smooth=smooth,
        include_background=include_background,
    )


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def _soft_skel(x: torch.Tensor, iters: int) -> torch.Tensor:
    """Differentiable 2D soft-skeleton (Shit et al., clDice) on [B, C, H, W]."""
    x1 = _soft_open(x)
    skel = F.relu(x - x1)
    for _ in range(iters):
        x = _soft_erode(x)
        x1 = _soft_open(x)
        delta = F.relu(x - x1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class CEDiceClDiceLoss(nn.Module):
    """CrossEntropy + softmax Dice + **soft clDice** for multiclass vessels.

    clDice (Shit et al., CVPR 2021) measures topology/connectivity via the soft
    skeleton, so it directly rewards keeping thin tubular structures connected —
    the failure mode of the plain Dice on small vessels (Pcom / Acom / AICA /
    AChA / OA), which collapse to background. Added on top of the CE + Dice base.

    Args:
        ce_weight / dice_weight / cldice_weight: term weights.
        smooth: Dice/clDice smoothing.
        include_background: exclude class 0 from Dice/clDice when False.
        iters: soft-skeleton iterations (more = fuller skeleton, slower).

    Input/Output:
        outputs: dict "masks" -> [B, C, H, W] logits
        targets: dict "masks" -> [B, 1, H, W] or [B, H, W] int labels
    """

    def __init__(
        self,
        ce_weight: float = 0.4,
        dice_weight: float = 0.4,
        cldice_weight: float = 0.2,
        smooth: float = 1.0,
        include_background: bool = False,
        iters: int = 5,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.smooth = smooth
        self.include_background = include_background
        self.iters = iters
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
        logits = outputs["masks"]
        target = targets["masks"]
        if target.ndim == logits.ndim:
            target = target.squeeze(1)
        target = target.long()
        num_classes = logits.size(1)

        ce = self.ce(logits, target.clamp(0, num_classes - 1))

        probs = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dim=dims)
        denom = probs.sum(dim=dims) + onehot.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)  # [C]

        # soft clDice per class
        sk_p = _soft_skel(probs, self.iters)
        sk_g = _soft_skel(onehot, self.iters)
        tprec = ((sk_p * onehot).sum(dim=dims) + self.smooth) / (sk_p.sum(dim=dims) + self.smooth)
        tsens = ((sk_g * probs).sum(dim=dims) + self.smooth) / (sk_g.sum(dim=dims) + self.smooth)
        cldice = (2.0 * tprec * tsens) / (tprec + tsens)  # [C]

        start = 0 if self.include_background else 1
        dice_loss = 1.0 - dice[start:].mean()
        cldice_loss = 1.0 - cldice[start:].mean()
        return self.ce_weight * ce + self.dice_weight * dice_loss + self.cldice_weight * cldice_loss


@register_loss("ce_dice_cldice", aliases=["cldice"])
def build_ce_dice_cldice_loss(
    ce_weight: float = 0.4,
    dice_weight: float = 0.4,
    cldice_weight: float = 0.2,
    smooth: float = 1.0,
    include_background: bool = False,
    iters: int = 5,
    **kwargs,
) -> CEDiceClDiceLoss:
    """Build CE + softmax Dice + soft clDice loss (see :class:`CEDiceClDiceLoss`)."""
    return CEDiceClDiceLoss(
        ce_weight=ce_weight, dice_weight=dice_weight, cldice_weight=cldice_weight,
        smooth=smooth, include_background=include_background, iters=iters,
    )


class FocalTverskyCELoss(nn.Module):
    """Focal-Tversky + class-weighted CrossEntropy for imbalanced multiclass seg.

    The Tversky index ``TP / (TP + alpha*FP + beta*FN)`` with ``beta > alpha``
    penalizes **false negatives** (missed voxels) far more than false positives,
    and the focal exponent ``gamma`` up-weights low-index (hard/collapsed)
    classes. Together with per-class ``class_weights`` (inverse frequency) this
    directly targets the failure where rare thin vessels (Pcom / Acom / AICA /
    AChA / OA) collapse to zero prediction — a recall problem that plain Dice /
    clDice / higher resolution did not fix.

    Args:
        alpha / beta: Tversky FP / FN weights (beta>alpha emphasizes recall).
        gamma: focal exponent on ``(1 - TverskyIndex)`` (>1 focuses hard classes).
        ft_weight / ce_weight: term weights.
        class_weights: optional length-``num_classes`` list (CE weight; also
            weights the per-class Tversky mean). None -> uniform.
        include_background: exclude class 0 from the Tversky term when False.
        smooth: Tversky smoothing.

    Input/Output:
        outputs: dict "masks" -> [B, C, H, W] logits
        targets: dict "masks" -> [B, 1, H, W] or [B, H, W] int labels
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.3333,
        ft_weight: float = 0.7,
        ce_weight: float = 0.3,
        dice_weight: float = 0.0,
        aux_weight: float = 0.0,
        skel_weight: float = 0.0,
        class_weights: list[float] | None = None,
        include_background: bool = False,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.ft_weight, self.ce_weight, self.dice_weight = ft_weight, ce_weight, dice_weight
        self.aux_weight = aux_weight
        self.skel_weight = skel_weight
        self.include_background = include_background
        self.smooth = smooth
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.register_buffer("class_weights", w)
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
        logits = outputs["masks"]
        target = targets["masks"]
        if target.ndim == logits.ndim:
            target = target.squeeze(1)
        target = target.long()
        num_classes = logits.size(1)

        ce = self.ce(logits, target.clamp(0, num_classes - 1))

        probs = torch.softmax(logits, dim=1)
        # ndim-agnostic (2D [B,C,H,W] or 3D [B,C,D,H,W]): channel dim -> position 1.
        onehot = torch.movedim(F.one_hot(target.clamp(0, num_classes - 1), num_classes), -1, 1).float()
        dims = (0,) + tuple(range(2, logits.ndim))  # sum over batch + all spatial
        tp = (probs * onehot).sum(dim=dims)
        fp = (probs * (1 - onehot)).sum(dim=dims)
        fn = ((1 - probs) * onehot).sum(dim=dims)
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)  # [C]
        ft = (1.0 - ti).clamp(min=0).pow(self.gamma)  # [C]

        start = 0 if self.include_background else 1
        ft_c = ft[start:]
        if self.class_weights is not None:
            w = self.class_weights[start:].to(ft_c.device)
            ftl = (w * ft_c).sum() / (w.sum() + 1e-8)
        else:
            ftl = ft_c.mean()

        total = self.ce_weight * ce + self.ft_weight * ftl
        if self.dice_weight > 0:
            # softmax Dice over foreground classes (overlap quality for the
            # large vessels that heavy FN-weighting can erode).
            dice = (2.0 * tp + self.smooth) / (probs.sum(dim=dims) + onehot.sum(dim=dims) + self.smooth)
            total = total + self.dice_weight * (1.0 - dice[start:].mean())

        # Auxiliary binary "vessel vs background" head (deep supervision): BCE +
        # soft Dice on outputs["aux_masks"] vs (target > 0). Reinforces vessel-
        # ness independently of class, helping thin vessels' foreground recall.
        aux = outputs.get("aux_masks")
        if self.aux_weight > 0 and aux is not None:
            bin_gt = (target > 0).float().unsqueeze(1)  # [B, 1, *spatial]
            bce = F.binary_cross_entropy_with_logits(aux, bin_gt)
            ap = torch.sigmoid(aux)
            adims = dims
            a_dice = (2.0 * (ap * bin_gt).sum(adims) + self.smooth) / (ap.sum(adims) + bin_gt.sum(adims) + self.smooth)
            total = total + self.aux_weight * (bce + (1.0 - a_dice.mean()))

        # Skeleton-Recall term (RSNA-2025 1st place): soft recall of the GT
        # per-class centerline skeleton by the predicted probabilities. Rewards
        # covering thin vessels' cores (connectivity) beyond volumetric overlap.
        skel = targets.get("skeleton") if self.skel_weight > 0 else None
        if skel is not None:
            sk = skel.squeeze(1) if skel.ndim == logits.ndim else skel
            sk_oh = torch.movedim(F.one_hot(sk.long().clamp(0, num_classes - 1), num_classes), -1, 1).float()
            sk_sum = sk_oh.sum(dim=dims)                       # [C] skeleton pixels per class
            recall = ((probs * sk_oh).sum(dim=dims) + self.smooth) / (sk_sum + self.smooth)  # [C]
            present = sk_sum[start:] > 0
            if bool(present.any()):
                total = total + self.skel_weight * (1.0 - recall[start:][present].mean())
        return total


@register_loss("focal_tversky_ce", aliases=["focal_tversky"])
def build_focal_tversky_ce_loss(
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 1.3333,
    ft_weight: float = 0.7,
    ce_weight: float = 0.3,
    dice_weight: float = 0.0,
    aux_weight: float = 0.0,
    skel_weight: float = 0.0,
    class_weights: list[float] | None = None,
    include_background: bool = False,
    smooth: float = 1.0,
    **kwargs,
) -> FocalTverskyCELoss:
    """Build Focal-Tversky + class-weighted CE (+ optional Dice/aux/skeleton)."""
    return FocalTverskyCELoss(
        alpha=alpha, beta=beta, gamma=gamma, ft_weight=ft_weight, ce_weight=ce_weight,
        dice_weight=dice_weight, aux_weight=aux_weight, skel_weight=skel_weight,
        class_weights=class_weights, include_background=include_background, smooth=smooth,
    )


@register_loss("dice", aliases=["diceloss"])
def build_dice_loss(
    smooth: float = 1.0,
    reduction: str = "mean",
    **kwargs,
) -> DiceLoss:
    """Build Dice loss.

    Args:
        smooth: Smoothing factor
        reduction: Reduction method

    Returns:
        DiceLoss instance
    """
    return DiceLoss(smooth=smooth, reduction=reduction)


@register_loss("bce", aliases=["bceloss"])
def build_bce_loss(**kwargs) -> nn.BCEWithLogitsLoss:
    """Build BCE loss.

    Returns:
        BCEWithLogitsLoss instance (wrapped for dict interface)
    """

    class BCELossWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()

        def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
            return self.bce(outputs["masks"], targets["masks"])

    return BCELossWrapper()


@register_loss("bce_dice", aliases=["dice_bce", "bcedice"])
def build_bce_dice_loss(
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    smooth: float = 1.0,
    **kwargs,
) -> BCEDiceLoss:
    """Build combined BCE and Dice loss.

    Args:
        bce_weight: Weight for BCE component
        dice_weight: Weight for Dice component
        smooth: Smoothing factor for Dice

    Returns:
        BCEDiceLoss instance
    """
    return BCEDiceLoss(
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        smooth=smooth,
    )


class HeatmapFocalLoss(nn.Module):
    """Foreground-weighted focal soft-BCE for Gaussian center-heatmap regression.

    The target is a continuous [0, 1] Gaussian peaked at each aneurysm centre, so
    the positive region is vanishingly small (measured on the 0.4mm crops:
    ``target > 0.5`` covers 0.0087% of a volume, ~180 voxels in a 128^3 patch).

    ``normalize`` controls the denominator, which is what makes or breaks
    stability at 3D scale:

    * ``"mean"`` (default) — divide by the total voxel count. Constant denominator.
    * ``"positive_mass"`` — divide by the batch's Gaussian mass (``target.sum()``),
      CenterNet-style. **This collapses training — do not use.** Measured on
      fold0 (70ep, identical otherwise): val_dice went 0.4940 -> 5e-10 and the
      coordinate FROC from 0.82 case-sensitivity to 0.03. train_loss never fell,
      oscillating in 10-60 for all 70 epochs.

      Why it fails: 23% of sampled patches contain no foreground at all
      (``fg_prob`` = 0.7), and those must fall back to mean reduction because the
      denominator is ~0. That fallback lands around 1e-4 while mass-normalized
      steps land around 10-60 — a ~1e5 discontinuity in gradient scale between two
      step types that alternate constantly. Adam's second-moment estimate never
      settles. Among the patches that do contain foreground the mass still spans
      4,100x, so the amplification is uneven there too.

      The lesson: with AdamW a *constant* rescaling of the loss is a no-op, so
      per-step normalization only changes the RELATIVE weighting between steps —
      and dividing by the per-step signal up-weights the steps carrying the least
      signal, which is backwards. Kept only so the experiment is reproducible.
    """

    def __init__(self, gamma: float = 2.0, fg_weight: float = 50.0,
                 normalize: str = "mean", eps: float = 1e-6):
        super().__init__()
        if normalize not in ("positive_mass", "mean"):
            raise ValueError(f"normalize must be 'positive_mass' or 'mean', got {normalize!r}")
        self.gamma, self.fg_weight = gamma, fg_weight
        self.normalize, self.eps = normalize, eps

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lg = logit[:, 0] if logit.dim() == target.dim() and logit.size(1) == 1 else logit
        t = target[:, 0] if target.dim() == lg.dim() + 1 else target
        bce = F.binary_cross_entropy_with_logits(lg, t, reduction="none")  # logits-stable
        p = torch.sigmoid(lg)
        focal = (t - p).abs().clamp(min=self.eps) ** self.gamma   # down-weight easy voxels
        w = 1.0 + self.fg_weight * t                              # up-weight near-peak
        per_voxel = w * focal * bce
        if self.normalize == "mean":
            return per_voxel.mean()
        denom = t.sum()
        if denom <= self.eps:          # background-only batch -> bounded fallback
            return per_voxel.mean()
        return per_voxel.sum() / denom


@register_loss("heatmap_focal", aliases=["gaussian_heatmap_focal"])
def build_heatmap_focal_loss(
    gamma: float = 2.0,
    fg_weight: float = 50.0,
    normalize: str = "mean",
    **kwargs,
) -> HeatmapFocalLoss:
    """Build the Gaussian-heatmap focal loss (Step3 aneurysm detector).

    Args:
        gamma: Focal exponent on |target - pred|
        fg_weight: Near-peak up-weighting (weight = 1 + fg_weight * target)
        normalize: "mean" (all voxels; the working setting) or "positive_mass"
            (per-step Gaussian mass — collapses training, see class docstring)

    Returns:
        HeatmapFocalLoss instance
    """
    return HeatmapFocalLoss(gamma=gamma, fg_weight=fg_weight, normalize=normalize)


class BCETopKLoss(nn.Module):
    """BCE over only the hardest ``k`` percent of voxels (online hard example mining).

    Ported from the RSNA-2025 intracranial-aneurysm 7th place solution
    (MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution,
    ``BCE_topK_loss_sep_channel``), which regresses soft blob targets with this
    loss alone — no focal term, no foreground weighting, no deep supervision.

    Motivation for a sparse 3D heatmap: with ``target > 0.5`` covering 0.0087% of
    a volume, a plain mean over all voxels is dominated by easy background, and
    the usual fix (focal + a large foreground weight) adds two hyper-parameters
    that interact. Selecting the top-k losses drops the easy background
    automatically and leaves a single knob.

    ``k`` is a PERCENTAGE of the spatial voxels, taken per (batch, channel) so one
    sample cannot monopolise the selection. The reference uses ``k=20``.
    """

    def __init__(self, k: float = 20.0):
        super().__init__()
        self.k = float(k)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.bce(logit, target.to(logit.dtype))       # [B, C, X, Y, Z]
        n_spatial = int(np.prod(loss.shape[-3:]))
        n = max(1, round(n_spatial * self.k / 100.0))
        loss = loss.reshape(*loss.shape[:2], -1)              # [B, C, N]
        loss = torch.topk(loss, k=min(n, loss.shape[-1]), dim=-1, sorted=False)[0]
        return loss.mean()


class PatchPresenceLoss(nn.Module):
    """Per-cell + whole-input presence loss for the Step3 grid detector.

    Ported from the RSNA-2025 3rd place solution's ``Aneurysm3DROILossV3``
    (tamotamo17/RSNA2025-3rd-place-solution): the cell target is built by
    **adaptive max pooling** the voxel mask down to the feature-map grid — "any
    positive voxel inside the cell makes the cell positive" — and the volume
    target is "any positive cell". No MIL aggregation is involved; both heads get
    explicit targets.

    Max pooling (not average) is the whole point: an aneurysm's near-peak region
    is a handful of voxels, so averaging into a 3.2mm cell would wash it out.

    Called with the model's output dict rather than a single tensor, because the
    two heads have different shapes; returns a dict so the module can log the
    terms separately.

    ``gamma > 0`` applies the focal down-weighting of easy cells. Positive cells
    are ~0.1% of a 16^3 grid, so the default keeps it on.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = -1.0, lambda_cell: float = 1.0,
                 lambda_volume: float = 0.2, volume_gamma: float = 0.0, cell_thr: float = 0.5,
                 pos_weight: float | None = None, eps: float = 1e-6):
        super().__init__()
        self.gamma, self.alpha = float(gamma), float(alpha)
        self.lambda_cell, self.lambda_volume = float(lambda_cell), float(lambda_volume)
        # The volume head is roughly balanced (fg_prob 0.7 of patches contain a
        # centre), so focal has nothing to fix there and only shrinks the term:
        # at logit 0 it scales ln2 by (1-0.5)^gamma = 0.25 for gamma=2, making the
        # effective lambda_volume 4x smaller than it reads. Default it off.
        self.volume_gamma = float(volume_gamma)
        self.cell_thr, self.pos_weight, self.eps = float(cell_thr), pos_weight, eps

    def cell_target(self, target: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """Voxel mask [B,1,D,H,W] -> binary cell target [B,1,d,h,w]."""
        t = target.float()
        if t.dim() == 4:                                   # [B,D,H,W] -> [B,1,D,H,W]
            t = t[:, None]
        pooled = F.adaptive_max_pool3d(t, grid)
        return (pooled > self.cell_thr).to(t.dtype)

    def _bce(self, logit: torch.Tensor, tgt: torch.Tensor, gamma: float,
             pos_weight: float | None) -> torch.Tensor:
        pw = (torch.as_tensor(pos_weight, dtype=logit.dtype, device=logit.device)
              if pos_weight is not None else None)
        loss = F.binary_cross_entropy_with_logits(logit, tgt, pos_weight=pw, reduction="none")
        if gamma > 0:
            p = torch.sigmoid(logit)
            pt = p * tgt + (1 - p) * (1 - tgt)
            loss = loss * (1 - pt).clamp(min=self.eps) ** gamma
        if self.alpha >= 0:
            loss = loss * (self.alpha * tgt + (1 - self.alpha) * (1 - tgt))
        return loss.mean()

    def forward(self, out: dict[str, torch.Tensor], target: torch.Tensor) -> dict[str, torch.Tensor]:
        cell_logits = out["cell_logits"]
        cell_tgt = self.cell_target(target, tuple(cell_logits.shape[2:]))
        cell = self._bce(cell_logits, cell_tgt, self.gamma, self.pos_weight)
        total = self.lambda_cell * cell
        vol = torch.zeros((), device=cell.device, dtype=cell.dtype)
        if self.lambda_volume > 0 and "volume_logits" in out:
            vol_tgt = cell_tgt.flatten(2).amax(-1)          # [B,C] "any positive cell"
            vol = self._bce(out["volume_logits"], vol_tgt, self.volume_gamma, None)
            total = total + self.lambda_volume * vol
        return {"loss": total, "cell": cell, "volume": vol}


@register_loss("patch_presence", aliases=["cell_presence", "patch_cls"])
def build_patch_presence_loss(
    gamma: float = 2.0,
    alpha: float = -1.0,
    lambda_cell: float = 1.0,
    lambda_volume: float = 0.2,
    volume_gamma: float = 0.0,
    cell_thr: float = 0.5,
    pos_weight: float | None = None,
    **kwargs,
) -> PatchPresenceLoss:
    """Build the per-cell presence loss (Step3 grid detector).

    Args:
        gamma: Focal exponent on the per-cell term; 0 disables focal (plain BCE)
        alpha: Focal positive-class weight in [0,1]; <0 disables it
        lambda_cell: Weight of the per-cell term
        lambda_volume: Weight of the whole-input presence term
        volume_gamma: Focal exponent on the volume term; 0 (default) because that
            head is balanced — see the class docstring
        cell_thr: Voxel-mask value above which a max-pooled cell counts positive.
            0.5 on a Gaussian target means "contains the aneurysm CENTRE", so a
            cell clipping only the Gaussian tail stays negative.
        pos_weight: Optional extra up-weight for positive cells (BCE pos_weight)

    Returns:
        PatchPresenceLoss instance
    """
    return PatchPresenceLoss(gamma=gamma, alpha=alpha, lambda_cell=lambda_cell,
                             lambda_volume=lambda_volume, volume_gamma=volume_gamma,
                             cell_thr=cell_thr, pos_weight=pos_weight)


@register_loss("bce_topk", aliases=["topk_bce"])
def build_bce_topk_loss(k: float = 20.0, **kwargs) -> BCETopKLoss:
    """Build the top-k BCE loss (Step3 aneurysm heatmap detector).

    Args:
        k: Percentage of spatial voxels kept per (batch, channel); the reference
           solution uses 20.

    Returns:
        BCETopKLoss instance
    """
    return BCETopKLoss(k=k)
