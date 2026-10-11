"""Loss Registry for Segmentation Task.

Supports custom segmentation losses and PyTorch built-in losses.

Usage:
    from src.segmentation.registries.losses import build_loss, list_losses, register_loss

    # Build custom or built-in loss
    loss = build_loss("dice")
    loss = build_loss("bce_dice")

    # Register a new loss
    @register_loss("my_loss")
    def build_my_loss(**kwargs):
        return MyLoss(**kwargs)
"""

from __future__ import annotations

import torch.nn as nn

from src.common.registry_base import BaseRegistry

_registry = BaseRegistry(
    name="loss",
    lazy_map={
        "dice": "src.segmentation.losses.loss",
        "diceloss": "src.segmentation.losses.loss",
        "bce": "src.segmentation.losses.loss",
        "bce_dice": "src.segmentation.losses.loss",
        "dice_bce": "src.segmentation.losses.loss",
        "bcedice": "src.segmentation.losses.loss",
        # Multiclass (softmax) losses for Step2/Step3 segmentation
        "ce_dice": "src.segmentation.losses.loss",
        "dice_ce": "src.segmentation.losses.loss",
        "cedice": "src.segmentation.losses.loss",
        # + soft clDice (connectivity/topology for thin tubular vessels)
        "ce_dice_cldice": "src.segmentation.losses.loss",
        "cldice": "src.segmentation.losses.loss",
        # Focal-Tversky (+ class-weighted CE) for rare/thin collapsed classes
        "focal_tversky_ce": "src.segmentation.losses.loss",
        "focal_tversky": "src.segmentation.losses.loss",
        # Step3 aneurysm detector: Gaussian center-heatmap regression (1ch sigmoid)
        "heatmap_focal": "src.segmentation.losses.loss",
        "gaussian_heatmap_focal": "src.segmentation.losses.loss",
        # top-k BCE (hard example mining) — RSNA2025 7th place solution's blob loss
        "bce_topk": "src.segmentation.losses.loss",
        "topk_bce": "src.segmentation.losses.loss",
        # Step3 grid detector: per-cell + volume presence — RSNA2025 3rd place solution
        "patch_presence": "src.segmentation.losses.loss",
        "cell_presence": "src.segmentation.losses.loss",
        "patch_cls": "src.segmentation.losses.loss",
    },
)


def _get_builtin_loss_class(name: str) -> type[nn.Module] | None:
    """Try to get a PyTorch built-in loss class by name."""
    # Exact match first
    if hasattr(nn, name):
        cls = getattr(nn, name)
        if isinstance(cls, type) and issubclass(cls, nn.Module):
            return cls

    # Try common variations
    variations = [
        name,
        name + "Loss",
        name.replace("_", ""),
        name.replace("_", "") + "Loss",
    ]
    for var in variations:
        if hasattr(nn, var):
            cls = getattr(nn, var)
            if isinstance(cls, type) and issubclass(cls, nn.Module):
                return cls
    return None


def build_loss(name: str | None = None, **kwargs) -> nn.Module:
    """Build a loss function by name.

    First checks custom registry, then falls back to PyTorch built-ins.
    The name can be given positionally or via a ``type`` key, so a whole config
    block can be passed directly: ``build_loss(**config["loss"])``.

    Args:
        name: Loss name (case-insensitive). If omitted, taken from ``type``.
        **kwargs: Arguments for the loss constructor

    Returns:
        Loss module instance
    """
    if name is None:
        name = kwargs.pop("type", None)
    if name is None:
        raise KeyError("build_loss requires a name (positional) or a 'type' key.")

    # Try custom registry first
    if _registry.is_registered(name):
        return _registry.build(name, **kwargs)

    # Fall back to PyTorch built-in
    builtin_cls = _get_builtin_loss_class(name)
    if builtin_cls is not None:
        return builtin_cls(**kwargs)

    # Not found
    known = list_losses(include_builtin=False)
    raise KeyError(f"Unknown loss '{name}'. Custom losses: {known}")


def list_losses(prefix: str | None = None, include_builtin: bool = False) -> list[str]:
    """List registered loss names.

    Args:
        prefix: Optional prefix filter
        include_builtin: If True, include PyTorch built-in losses

    Returns:
        Sorted list of loss names
    """
    names = _registry.list_registered(prefix)

    if include_builtin:
        builtin_losses = [
            "BCELoss",
            "BCEWithLogitsLoss",
            "CrossEntropyLoss",
            "MSELoss",
            "L1Loss",
            "SmoothL1Loss",
        ]
        names = sorted(set(names + [n.lower() for n in builtin_losses]))

    return names


register_loss = _registry.register
register_lazy_loss = _registry.register_lazy
