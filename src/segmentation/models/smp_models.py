"""Segmentation Models Pytorch (SMP) Wrappers.

Wrapper for segmentation_models_pytorch library, providing access to
various segmentation architectures (U-Net, FPN, DeepLabV3+, etc.) with
pre-trained encoders.

See: https://github.com/qubvel/segmentation_models.pytorch
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.segmentation.registries.models import register_model


def _init_smp_from_ckpt(smp_model: "nn.Module", ckpt_path: str) -> None:
    """Warm-start an SMP model from a previously trained segmentation checkpoint.

    Loads the encoder+decoder (and matching head) weights from a Lightning
    checkpoint whose ``model`` is an ``SegmentationModelV{1,2}`` wrapping an SMP
    net. Keys are remapped from the Lightning prefix (``model.model.<smp>``) to
    the bare SMP names and loaded with ``strict=False`` so that layers with a
    different shape — most importantly the final segmentation head when the
    class count differs (e.g. 37-class vessel -> 2-class aneurysm) — are left at
    their freshly-initialised values. This transfers everything the source model
    learned about the domain (same modality/resolution/input channels) except
    the task-specific head.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    prefix = "model.model."  # LightningModule.model = SegmentationModelVx; .model = smp
    src = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    tgt = smp_model.state_dict()
    kept, partial, dropped = {}, [], []
    for k, v in src.items():
        if k not in tgt:
            dropped.append(k)
        elif v.shape == tgt[k].shape:
            kept[k] = v
        elif (v.ndim == tgt[k].ndim >= 2
              and v.shape[0] == tgt[k].shape[0]
              and v.shape[2:] == tgt[k].shape[2:]):
            # Same out-channels/kernel but different in-channels (e.g. the input
            # conv: 5-ch vessel model -> 10-ch [image|aux] aneurysm model). Copy
            # the overlapping input channels onto the fresh weight, keep the rest
            # (image pathway transfers; the extra aux channels stay at init).
            w = tgt[k].clone()
            c = min(v.shape[1], tgt[k].shape[1])
            w[:, :c] = v[:, :c]
            kept[k] = w
            partial.append(f"{k} {tuple(v.shape)}->{tuple(tgt[k].shape)}")
        else:
            dropped.append(k)
    dropped = sorted(dropped)
    smp_model.load_state_dict(kept, strict=False)
    print(f"[init_from_ckpt] {ckpt_path}")
    print(f"[init_from_ckpt] loaded {len(kept)}/{len(src)} tensors "
          f"(partial in-ch copy: {len(partial)}); reinit(skipped): {len(dropped)}")
    for p in partial:
        print(f"[init_from_ckpt]   partial: {p}")
    if dropped:
        print(f"[init_from_ckpt]   skipped (reinit): {dropped[:6]}{' ...' if len(dropped) > 6 else ''}")

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise ImportError(
        "segmentation_models_pytorch not found. Install with: uv add segmentation-models-pytorch\n"
        "Or add to pyproject.toml dependencies."
    ) from e


class SegmentationModelV1(nn.Module):
    """Segmentation model v1 with SMP architecture and backbone.

    Provides a unified interface for SMP models with architecture and backbone
    as separate arguments.

    Args:
        arch: Architecture name ("Unet", "FPN", "DeepLabV3Plus", "PSPNet", etc.)
        backbone: Encoder/backbone name (e.g., "resnet50", "efficientnet-b3")
        encoder_weights: Pretrained weights ("imagenet", "ssl", "swsl", None)
        in_channels: Number of input channels
        num_classes: Number of output classes/channels
        **kwargs: Additional arguments passed to smp model

    Example YAML:
        ```yaml
        model:
          type: model_v1
          arch: Unet
          backbone: resnet50
          num_classes: 6
          encoder_weights: imagenet
        ```
    """

    def __init__(
        self,
        arch: str = "Unet",
        backbone: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        num_classes: int = 6,
        init_from_ckpt: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.arch = arch
        self.backbone = backbone
        self.num_classes = num_classes

        model_class = getattr(smp, arch)

        self.model = model_class(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )

        if init_from_ckpt:
            _init_smp_from_ckpt(self.model, init_from_ckpt)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: Dict with "image" key containing Tensor[B, C, H, W]

        Returns:
            Dict with "masks" key containing Tensor[B, num_classes, H, W]
        """
        x = batch["image"]
        masks = self.model(x)
        return {"masks": masks}


class SegmentationModelV2AuxBin(nn.Module):
    """SMP segmentation model with an auxiliary **binary vessel** head.

    Shares the encoder + decoder with the main multiclass head and adds a second
    1-channel head predicting "vessel vs background". Trained with an auxiliary
    binary loss, this gives the encoder an explicit vesselness signal that helps
    thin vessels whose exact class is uncertain (improves foreground recall).

    forward(batch) -> {"masks": [B, num_classes, H, W], "aux_masks": [B, 1, H, W]}
    """

    def __init__(
        self,
        arch: str = "Unet",
        backbone: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        num_classes: int = 6,
        **kwargs,
    ):
        super().__init__()
        model_class = getattr(smp, arch)
        self.model = model_class(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )
        # Aux head consumes the same decoder features as the main head.
        dec_ch = self.model.segmentation_head[0].in_channels
        self.aux_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["image"]
        features = self.model.encoder(x)
        decoder_output = self.model.decoder(features)
        masks = self.model.segmentation_head(decoder_output)
        aux_masks = self.aux_head(decoder_output)
        return {"masks": masks, "aux_masks": aux_masks}


@register_model("model_v2_auxbin", aliases=["smp_auxbin"])
def build_model_v2_auxbin(
    arch: str = "Unet",
    backbone: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 6,
    **kwargs,
) -> SegmentationModelV2AuxBin:
    """Build an SMP model with an auxiliary binary vessel head (see class)."""
    return SegmentationModelV2AuxBin(
        arch=arch, backbone=backbone, encoder_weights=encoder_weights,
        in_channels=in_channels, num_classes=num_classes, **kwargs,
    )


@register_model("model_v1", aliases=["segmentation_v1", "smp_v1"])
def build_model_v1(
    arch: str = "Unet",
    backbone: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 6,
    init_from_ckpt: str | None = None,
    **kwargs,
) -> SegmentationModelV1:
    """Build segmentation model v1 with SMP architecture and backbone.

    Args:
        arch: Architecture name ("Unet", "FPN", "DeepLabV3Plus", "PSPNet", etc.)
        backbone: Encoder/backbone name (e.g., "resnet50", "efficientnet-b3")
        encoder_weights: Pretrained weights ("imagenet", "ssl", "swsl", None)
        in_channels: Number of input channels
        num_classes: Number of output classes/channels
        **kwargs: Additional arguments passed to smp model

    Returns:
        SegmentationModelV1 instance

    Example YAML:
        ```yaml
        model:
          type: model_v1
          arch: Unet
          backbone: resnet50
          num_classes: 6
          encoder_weights: imagenet
        ```

    Available architectures:
        - Unet, UnetPlusPlus
        - FPN
        - DeepLabV3, DeepLabV3Plus
        - PSPNet
        - PAN
        - MAnet
        - Linknet

    Available backbones (partial list):
        - resnet18, resnet34, resnet50, resnet101
        - efficientnet-b0 to efficientnet-b7
        - resnext50_32x4d, resnext101_32x8d
        - timm-* (any timm model with `timm-` prefix)
    """
    return SegmentationModelV1(
        arch=arch,
        backbone=backbone,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=num_classes,
        init_from_ckpt=init_from_ckpt,
        **kwargs,
    )
