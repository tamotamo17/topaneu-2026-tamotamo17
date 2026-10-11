"""MONAI 3D segmentation models.

Thin wrappers exposing the project's ``forward(batch) -> {"masks": logits}``
contract over MONAI 3D networks, so they plug into the registry / Lightning
modules. Used for the 3D vessel segmentation backbone (Step2), the aneurysm
heatmap detector (Step3) and the fine aneurysm-shape segmenter (Step4).
"""

from __future__ import annotations

import glob

import torch
import torch.nn as nn

from src.segmentation.registries.models import register_model

try:
    from monai.networks.nets import DynUNet, SegResNet, UNet
except ImportError as e:  # pragma: no cover
    raise ImportError("MONAI not found. Install with: uv add monai") from e


def _as_norm(norm_name):
    """Normalise a YAML-sourced ``norm_name`` into what MONAI expects.

    OmegaConf hands us ListConfig/DictConfig for ``["group", {...}]``; MONAI
    wants a plain str or ``(str, dict)`` tuple.
    """
    if isinstance(norm_name, str):
        return norm_name
    parts = list(norm_name)
    if len(parts) == 1:
        return str(parts[0])
    return (str(parts[0]), {str(k): v for k, v in dict(parts[1]).items()})


def _init_net_from_ckpt(net: nn.Module, ckpt_path: str, prefix: str = "model.net.") -> None:
    """Warm-start a MONAI 3D net from a previously trained segmentation ckpt.

    Loads matching weights from a Lightning checkpoint whose ``model`` is a
    :class:`MonaiSeg3D` (keys ``model.net.<...>``). Layers whose shape differs
    are handled like the 2.5D warm-start: the input conv (same out-channels/
    kernel, different in-channels — e.g. 1-ch vessel -> 2-ch [image|aux] sphere)
    is partially copied over its overlapping input channels, and the final
    output conv (different class count, e.g. 37 vessel -> 2 sphere) is left at
    its fresh init. Everything the vessel model learned about 3D structure
    transfers; only the task head is new.

    ``ckpt_path`` may be a glob (e.g. ``.../fold0/best-*.ckpt``): checkpoint
    names embed the epoch and score, so the pattern lets a config refer to the
    best checkpoint of a run before that run exists. It must match exactly one file.
    """
    if glob.has_magic(str(ckpt_path)):
        hits = sorted(glob.glob(str(ckpt_path)))
        if len(hits) != 1:
            raise FileNotFoundError(f"init_from_ckpt pattern {ckpt_path!r} matched {len(hits)} files: {hits}")
        ckpt_path = hits[0]
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    src = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    tgt = net.state_dict()
    kept, partial, dropped = {}, [], []
    for k, v in src.items():
        if k not in tgt:
            dropped.append(k)
        elif v.shape == tgt[k].shape:
            kept[k] = v
        elif (v.ndim == tgt[k].ndim >= 2
              and v.shape[0] == tgt[k].shape[0]
              and v.shape[2:] == tgt[k].shape[2:]):
            w = tgt[k].clone()
            c = min(v.shape[1], tgt[k].shape[1])
            w[:, :c] = v[:, :c]
            kept[k] = w
            partial.append(f"{k} {tuple(v.shape)}->{tuple(tgt[k].shape)}")
        else:
            dropped.append(k)
    dropped = sorted(dropped)
    net.load_state_dict(kept, strict=False)
    print(f"[init_from_ckpt] {ckpt_path}")
    print(f"[init_from_ckpt] loaded {len(kept)}/{len(src)} tensors "
          f"(partial in-ch copy: {len(partial)}); reinit(skipped): {len(dropped)}")
    for p in partial:
        print(f"[init_from_ckpt]   partial: {p}")
    if dropped:
        print(f"[init_from_ckpt]   skipped (reinit): {dropped[:6]}{' ...' if len(dropped) > 6 else ''}")


class MonaiSeg3D(nn.Module):
    """Wrap a MONAI 3D network to the ``{"masks": [B,C,D,H,W]}`` contract."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"masks": self.net(batch["image"])}


class MonaiSeg3DAuxBin(nn.Module):
    """MONAI 3D net with an auxiliary binary-vessel output channel.

    The wrapped net produces ``num_classes + 1`` channels; the first
    ``num_classes`` are the multiclass logits (``masks``) and the last is a
    binary "vessel vs background" logit (``aux_masks``), the 3D analogue of the
    2.5D :class:`SegmentationModelV2AuxBin`. The aux channel shares the whole
    network (encoder+decoder+final conv), adding one extra filter, and is
    supervised by the ``aux_weight`` BCE+Dice term against ``mask > 0``.
    """

    def __init__(self, net: nn.Module, num_classes: int):
        super().__init__()
        self.net = net
        self.num_classes = num_classes

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.net(batch["image"])
        return {"masks": out[:, : self.num_classes], "aux_masks": out[:, self.num_classes :]}


class MonaiSegResNet3DAuxCls(nn.Module):
    """MONAI SegResNet (3D) with an auxiliary patch-level classification head.

    The head branches off the encoder bottleneck, so the segmentation path is
    byte-for-byte the one :class:`MonaiSeg3D` runs (``encode`` -> ``decode`` is
    exactly MONAI's ``forward``, modulo the ``down_x`` reversal it does inline)
    and a Step3 checkpoint trained without the head still loads.

    ``forward(batch) -> {"masks": [B,C,D,H,W], "cls_logits": [B,num_cls]}``

    Motivation for Step3: the Gaussian centre-heatmap head only ever sees local
    gradients (targets ``>0.5`` are 0.0087% of voxels), and scaling the training
    set 4.3x did not move val_dice — the detector is limited by supervision
    signal, not data. A "does this patch contain an aneurysm centre" head gives
    the shared encoder a whole-patch gradient, and its sliding-window max is a
    case-level presence score usable to gate blob FPs at inference time.

    ``pool`` decides how the bottleneck grid (16^3 for a 128^3 patch) collapses
    to a vector. An aneurysm occupies ~1.6 bottleneck cells out of 4096, so avg
    alone dilutes it while max alone routes the gradient to a single cell (which
    is close to what the heatmap head already learns); ``avgmax`` concatenates
    both and is the default.
    """

    def __init__(self, net: nn.Module, bottleneck_channels: int, num_cls: int = 1,
                 pool: str = "avgmax", dropout: float = 0.2):
        super().__init__()
        if pool not in ("avg", "max", "avgmax"):
            raise ValueError(f"pool must be one of avg/max/avgmax, got {pool!r}")
        self.net = net
        self.pool = pool
        feat = bottleneck_channels * (2 if pool == "avgmax" else 1)
        # LayerNorm, then a zero-initialised output layer. Both exist to make the
        # head's loss scale predictable: the bottleneck of a vessel-seg-warm-started
        # SegResNet has large activations, and without these the initial BCE
        # measured 59 instead of ln2, i.e. the aux term silently outweighed the
        # segmentation term by ~2.5x at the config's cls_weight. LayerNorm (not
        # BatchNorm) because the 3D patch batch size is 2. Zero-init makes the
        # first-step logits exactly 0 -> BCE starts at ln2 = 0.693, so cls_weight
        # can be reasoned about arithmetically, and a warm-started encoder gets no
        # random gradient shock from a freshly initialised head.
        self.cls_head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Dropout(dropout), nn.Linear(feat, bottleneck_channels), nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(bottleneck_channels, num_cls),
        )
        nn.init.zeros_(self.cls_head[-1].weight)
        nn.init.zeros_(self.cls_head[-1].bias)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        avg = nn.functional.adaptive_avg_pool3d(x, 1).flatten(1)
        if self.pool == "avg":
            return avg
        mx = nn.functional.adaptive_max_pool3d(x, 1).flatten(1)
        return mx if self.pool == "max" else torch.cat([avg, mx], dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x, down_x = self.net.encode(batch["image"])
        cls = self.cls_head(self._pool(x))
        down_x.reverse()                      # what SegResNet.forward does before decode
        return {"masks": self.net.decode(x, down_x), "cls_logits": cls}


@register_model("monai_segresnet3d_auxcls", aliases=["segresnet3d_auxcls"])
def build_monai_segresnet3d_auxcls(
    in_channels: int = 1,
    num_classes: int = 1,
    init_filters: int = 32,
    blocks_down: tuple[int, ...] = (1, 2, 2, 4),
    dropout: float = 0.0,
    cls_num: int = 1,
    cls_pool: str = "avgmax",
    cls_dropout: float = 0.2,
    init_from_ckpt: str | None = None,
    **kwargs,
) -> MonaiSegResNet3DAuxCls:
    """SegResNet (3D) + auxiliary patch-level classification head.

    Drop-in replacement for ``monai_segresnet3d`` in the Step3 heatmap config;
    the ``heatmap_3d`` module supervises ``cls_logits`` when its ``cls_weight``
    is > 0. ``init_from_ckpt`` warm-starts the SegResNet exactly as the plain
    builder does (the new head keeps its fresh init, since its weights are not
    in the source ckpt).
    """
    net = SegResNet(
        spatial_dims=3, in_channels=in_channels, out_channels=num_classes,
        init_filters=init_filters, blocks_down=tuple(blocks_down),
        dropout_prob=(dropout or None),
    )
    if init_from_ckpt:
        _init_net_from_ckpt(net, init_from_ckpt)
    # SegResNet doubles the filter count at every down layer after the first.
    bottleneck = init_filters * 2 ** (len(tuple(blocks_down)) - 1)
    return MonaiSegResNet3DAuxCls(net, bottleneck_channels=bottleneck, num_cls=cls_num,
                                  pool=cls_pool, dropout=cls_dropout)


@register_model("monai_unet3d_auxbin", aliases=["unet3d_auxbin"])
def build_monai_unet3d_auxbin(
    in_channels: int = 1,
    num_classes: int = 37,
    channels: tuple[int, ...] = (32, 64, 128, 256, 320),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.0,
    init_from_ckpt: str | None = None,
    **kwargs,
) -> MonaiSeg3DAuxBin:
    """MONAI 3D U-Net with an extra binary-vessel aux channel. Returns
    ``{"masks", "aux_masks"}``."""
    net = UNet(
        spatial_dims=3, in_channels=in_channels, out_channels=num_classes + 1,
        channels=channels, strides=strides, num_res_units=num_res_units, dropout=dropout,
    )
    if init_from_ckpt:
        _init_net_from_ckpt(net, init_from_ckpt)
    return MonaiSeg3DAuxBin(net, num_classes)


@register_model("monai_segresnet3d_auxbin", aliases=["segresnet3d_auxbin"])
def build_monai_segresnet3d_auxbin(
    in_channels: int = 1,
    num_classes: int = 37,
    init_filters: int = 32,
    dropout: float = 0.0,
    init_from_ckpt: str | None = None,
    **kwargs,
) -> MonaiSeg3DAuxBin:
    """MONAI SegResNet (3D) with a binary-vessel aux channel. ``{"masks","aux_masks"}``."""
    net = SegResNet(
        spatial_dims=3, in_channels=in_channels, out_channels=num_classes + 1,
        init_filters=init_filters, dropout_prob=(dropout or None),
    )
    if init_from_ckpt:
        _init_net_from_ckpt(net, init_from_ckpt)
    return MonaiSeg3DAuxBin(net, num_classes)


@register_model("monai_dynunet3d_auxbin", aliases=["dynunet3d_auxbin"])
def build_monai_dynunet3d_auxbin(
    in_channels: int = 1,
    num_classes: int = 37,
    kernel_size: tuple = (3, 3, 3, 3, 3),
    strides: tuple = (1, 2, 2, 2, 2),
    filters: tuple | None = None,
    dropout: float = 0.0,
    init_from_ckpt: str | None = None,
    **kwargs,
) -> MonaiSeg3DAuxBin:
    """MONAI DynUNet (nnU-Net-style 3D) with a binary-vessel aux channel.

    ``upsample_kernel_size`` mirrors ``strides[1:]``. Returns
    ``{"masks","aux_masks"}``.
    """
    net = DynUNet(
        spatial_dims=3, in_channels=in_channels, out_channels=num_classes + 1,
        kernel_size=list(kernel_size), strides=list(strides),
        upsample_kernel_size=list(strides[1:]),
        filters=list(filters) if filters is not None else None,
        dropout=(dropout or None), res_block=True,
    )
    if init_from_ckpt:
        _init_net_from_ckpt(net, init_from_ckpt)
    return MonaiSeg3DAuxBin(net, num_classes)


class MonaiSeg3DDeepSup(nn.Module):
    """DynUNet with nnU-Net-style deep supervision.

    With ``deep_supervision=True`` MONAI's DynUNet returns a stacked
    ``[B, 1+deep_supr_num, C, D, H, W]`` in TRAIN mode (every head already
    upsampled to full resolution) and the plain ``[B, C, D, H, W]`` in eval.
    This splits that apart so the head the pipeline consumes stays exactly the
    ``masks`` contract, and the auxiliary heads ride along under ``ds_masks``
    for the Lightning module to add a weighted loss term.

    Inference is untouched — in eval mode there are no extra heads at all, so
    ``sliding_window_inference`` and every downstream consumer see the same
    single-channel output as any other detector.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.net(batch["image"])
        if out.dim() == 6:                      # train: heads stacked on dim 1
            return {"masks": out[:, 0], "ds_masks": out[:, 1:]}
        return {"masks": out}


class MonaiDynUNet3DAuxCls(nn.Module):
    """MONAI DynUNet (3D) with the same auxiliary patch-level cls head as
    :class:`MonaiSegResNet3DAuxCls`, so the two can be ensembled as peers.

    DynUNet's forward is a RECURSION through ``skip_layers``, not a
    ``encode``/``decode`` pair, so the bottleneck cannot be tapped by splitting
    the forward the way the SegResNet variant does. A forward hook on
    ``net.bottleneck`` captures it instead, which keeps MONAI's own forward — and
    therefore the segmentation path and its deep-supervision handling — running
    completely untouched. A plain ``monai_dynunet3d`` checkpoint still loads:
    the network lives under the same ``net.`` prefix and only ``cls_head.*`` is new.

    ``forward(batch) -> {"masks": ..., "cls_logits": [B,num_cls]}`` (plus
    ``ds_masks`` when deep supervision stacks heads on dim 1, as in
    :class:`MonaiSeg3DDeepSup`).

    Why bother on DynUNet: on SegResNet the head's measured value was less its
    own accuracy than its DECORRELATION — ``mean(base+aux)`` gave CPM 0.709 vs
    0.671/0.668 for the members, the largest single gain in that line. DynUNet is
    the ensemble's architecture-diversity member (it is what took the 8-model
    ensemble to CPM 0.735) while also being its weakest member at 0.565, so a
    second, decorrelated DynUNet is the cheapest place to look for more.
    """

    def __init__(self, net: nn.Module, bottleneck_channels: int, num_cls: int = 1,
                 pool: str = "avgmax", dropout: float = 0.2):
        super().__init__()
        if pool not in ("avg", "max", "avgmax"):
            raise ValueError(f"pool must be one of avg/max/avgmax, got {pool!r}")
        self.net = net
        self.pool = pool
        feat = bottleneck_channels * (2 if pool == "avgmax" else 1)
        # Same LayerNorm + zero-init output as the SegResNet variant, and for the
        # same measured reason: a vessel-seg-warm-started bottleneck has large
        # activations, and without them the initial BCE came out at 59 instead of
        # ln2. Kept as a flat ``cls_head`` Sequential (not factored into a shared
        # submodule) so the parameter names match the SegResNet variant's and the
        # ``hasattr(model, "cls_head")`` guards keep working.
        self.cls_head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Dropout(dropout), nn.Linear(feat, bottleneck_channels), nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(bottleneck_channels, num_cls),
        )
        nn.init.zeros_(self.cls_head[-1].weight)
        nn.init.zeros_(self.cls_head[-1].bias)
        self._bottleneck_out: torch.Tensor | None = None
        net.bottleneck.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        self._bottleneck_out = output

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        avg = nn.functional.adaptive_avg_pool3d(x, 1).flatten(1)
        if self.pool == "avg":
            return avg
        mx = nn.functional.adaptive_max_pool3d(x, 1).flatten(1)
        return mx if self.pool == "max" else torch.cat([avg, mx], dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._bottleneck_out = None
        out = self.net(batch["image"])
        feat = self._bottleneck_out
        if feat is None:                      # hook did not fire => wrong module wrapped
            raise RuntimeError("DynUNet bottleneck hook did not fire; cannot compute cls_logits")
        self._bottleneck_out = None           # do not hold the activation past the step
        res = {"masks": out[:, 0], "ds_masks": out[:, 1:]} if out.dim() == 6 else {"masks": out}
        res["cls_logits"] = self.cls_head(self._pool(feat))
        return res


@register_model("monai_dynunet3d_auxcls", aliases=["dynunet3d_auxcls"])
def build_monai_dynunet3d_auxcls(
    in_channels: int = 1,
    num_classes: int = 1,
    kernel_size: tuple = (3, 3, 3, 3, 3),
    strides: tuple = (1, 2, 2, 2, 2),
    filters: tuple | None = None,
    dropout: float = 0.0,
    deep_supervision: bool = False,
    deep_supr_num: int = 1,
    norm_name=("INSTANCE", {"affine": True}),
    cls_num: int = 1,
    cls_pool: str = "avgmax",
    cls_dropout: float = 0.2,
    init_from_ckpt: str | None = None,
    **kwargs,
) -> MonaiDynUNet3DAuxCls:
    """DynUNet (3D) + auxiliary patch-level classification head.

    Drop-in replacement for ``monai_dynunet3d`` in the Step3 heatmap config; the
    ``heatmap_3d`` module supervises ``cls_logits`` when its ``cls_weight`` > 0.
    Every DynUNet argument behaves exactly as in :func:`build_monai_dynunet3d` —
    including the ``norm_name`` warning about the bare ``"instance"`` string —
    and ``init_from_ckpt`` warm-starts the same way (the new head simply keeps
    its fresh init).
    """
    net = DynUNet(
        spatial_dims=3, in_channels=in_channels, out_channels=num_classes,
        kernel_size=list(kernel_size), strides=list(strides),
        upsample_kernel_size=list(strides[1:]),
        filters=list(filters) if filters is not None else None,
        dropout=(dropout or None), res_block=True,
        deep_supervision=bool(deep_supervision), deep_supr_num=int(deep_supr_num),
        norm_name=_as_norm(norm_name),
    )
    if init_from_ckpt:
        _init_net_from_ckpt(net, init_from_ckpt)
    # DynUNet exposes its per-level widths; the bottleneck is the deepest one.
    return MonaiDynUNet3DAuxCls(net, bottleneck_channels=int(net.filters[-1]),
                                num_cls=cls_num, pool=cls_pool, dropout=cls_dropout)


@register_model("monai_dynunet3d", aliases=["dynunet3d"])
def build_monai_dynunet3d(
    in_channels: int = 1,
    num_classes: int = 37,
    kernel_size: tuple = (3, 3, 3, 3, 3),
    strides: tuple = (1, 2, 2, 2, 2),
    filters: tuple | None = None,
    dropout: float = 0.0,
    deep_supervision: bool = False,
    deep_supr_num: int = 1,
    norm_name=("INSTANCE", {"affine": True}),
    init_from_ckpt: str | None = None,
    **kwargs,
) -> MonaiSeg3D:
    """MONAI DynUNet (nnU-Net-style 3D). Returns ``{"masks": logits}``.

    The plain counterpart of :func:`build_monai_dynunet3d_auxbin`, for the Step3
    detector where the head is a single sigmoid channel and there is no auxiliary
    binary-vessel output to attach. ``init_from_ckpt`` warm-starts from the
    DynUNet vessel-seg model — architectures must match for the transfer to
    happen at all (a SegResNet ckpt shares no key names with DynUNet).

    ``strides`` sets the total downsampling: the shipped ``(1,2,2,2,2)`` gives
    stride 16, twice as coarse as the SegResNet detector's 8. Setting the last
    entry to 1 gives stride 8 at essentially no cost (measured batch4/patch128/
    bf16: 9.7 -> 9.9 GiB, 0.525 -> 0.531 s/step).

    ``norm_name`` defaults to MONAI's own ``("INSTANCE", {"affine": True})``.
    Do not shorten that to the bare string ``"instance"``: MONAI reads the plain
    string as affine=False, which silently drops 92 learnable norm parameters and
    breaks the vessel warm-start (measured: 146/148 tensors transferred -> 54/148,
    all 94 skipped keys being ``norm*.weight``/``norm*.bias``).

    It also swaps the normalisation. DynUNet defaults to ``"instance"``
    while SegResNet uses GroupNorm, and InstanceNorm normalises every channel
    INDEPENDENTLY over the whole volume — so on a target where 0.0087% of voxels
    are positive, a channel that fires only on a lesion and one that fires
    broadly both get rescaled to unit variance, destroying their relative
    magnitude. Pass ``["group", {"num_groups": 8}]`` to match SegResNet.

    ``deep_supervision`` adds nnU-Net's auxiliary heads; the model then returns
    ``ds_masks`` alongside ``masks`` during training and the Lightning module
    weights them in. Eval output is unchanged either way.
    """
    net = DynUNet(
        spatial_dims=3, in_channels=in_channels, out_channels=num_classes,
        kernel_size=list(kernel_size), strides=list(strides),
        upsample_kernel_size=list(strides[1:]),
        filters=list(filters) if filters is not None else None,
        dropout=(dropout or None), res_block=True,
        deep_supervision=bool(deep_supervision), deep_supr_num=int(deep_supr_num),
        norm_name=_as_norm(norm_name),
    )
    if init_from_ckpt:
        _init_net_from_ckpt(net, init_from_ckpt)
    return MonaiSeg3DDeepSup(net) if deep_supervision else MonaiSeg3D(net)


@register_model("monai_unet3d", aliases=["unet3d"])
def build_monai_unet3d(
    in_channels: int = 1,
    num_classes: int = 37,
    channels: tuple[int, ...] = (32, 64, 128, 256, 320),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.0,
    init_from_ckpt: str | None = None,
    **kwargs,
) -> MonaiSeg3D:
    """MONAI 3D U-Net (residual). Returns ``{"masks": logits}``.

    ``init_from_ckpt`` warm-starts from a prior 3D seg checkpoint (e.g. the 3D
    vessel-seg model) via :func:`_init_net_from_ckpt`.
    """
    net = UNet(
        spatial_dims=3, in_channels=in_channels, out_channels=num_classes,
        channels=channels, strides=strides, num_res_units=num_res_units, dropout=dropout,
    )
    if init_from_ckpt:
        _init_net_from_ckpt(net, init_from_ckpt)
    return MonaiSeg3D(net)


@register_model("monai_segresnet3d", aliases=["segresnet3d"])
def build_monai_segresnet3d(
    in_channels: int = 1,
    num_classes: int = 37,
    init_filters: int = 32,
    dropout: float = 0.0,
    init_from_ckpt: str | None = None,
    **kwargs,
) -> MonaiSeg3D:
    """MONAI SegResNet (3D). Returns ``{"masks": logits}``.

    ``init_from_ckpt`` warm-starts from a prior 3D seg checkpoint (e.g. the 3D
    SegResNet vessel-seg model) via :func:`_init_net_from_ckpt`.
    """
    net = SegResNet(
        spatial_dims=3, in_channels=in_channels, out_channels=num_classes,
        init_filters=init_filters, dropout_prob=(dropout or None),
    )
    if init_from_ckpt:
        _init_net_from_ckpt(net, init_from_ckpt)
    return MonaiSeg3D(net)
