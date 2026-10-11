"""2.5D sliding-slice inference: reconstruct a full 3D prediction volume.

Shared by Step1 (binary vessel) and Step2 (multiclass vessel) inference. For
each center slice along ``slice_axis`` a window of ``num_slices`` adjacent slices
is stacked as channels, percentile-normalized **exactly as in training**, padded
to a multiple of 32 (so the fully-convolutional SMP U-Net accepts it without
losing border voxels), run through the model, then cropped back to the original
slice size. Per-slice predictions are stacked into a 3D label / probability
volume in the same array-axis order as the input.

The optional ``aux`` volume (e.g. the vessel mask for Step3 conditioning) is
windowed and appended as extra channels, mirroring
:class:`~src.segmentation.dataloaders.slices_2_5d_multiclass.Slices25DMulticlassDataset`.
"""

from __future__ import annotations

import numpy as np
import torch


def _percentile_normalize(window: np.ndarray, lower: float, upper: float, eps: float = 1e-8) -> np.ndarray:
    """Match ``NormalizePercentile``: clip to [lo, hi] percentiles, scale to [0, 1]."""
    lo = np.percentile(window, lower)
    hi = np.percentile(window, upper)
    out = np.clip(window, lo, hi)
    return ((out - lo) / (hi - lo + eps)).astype(np.float32)


def _pad_to_multiple(arr: np.ndarray, multiple: int, value: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad the first two axes ([H, W, ...]) up to a multiple of ``multiple``.

    Returns the padded array and the (top, left) offsets to crop back.
    """
    h, w = arr.shape[0], arr.shape[1]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    top, left = ph // 2, pw // 2
    pad_width = [(top, ph - top), (left, pw - left)] + [(0, 0)] * (arr.ndim - 2)
    padded = np.pad(arr, pad_width, mode="constant", constant_values=value)
    return padded, (top, left)


@torch.no_grad()
def predict_volume(
    model: torch.nn.Module,
    image: np.ndarray,
    *,
    num_slices: int = 5,
    slice_axis: int = 2,
    num_classes: int = 1,
    threshold: float = 0.5,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    aux: np.ndarray | None = None,
    aux_mode: str = "label",
    aux_max: float = 36.0,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
    return_proba: bool = False,
    proba_class: int | None = None,
    return_class_proba: bool = False,
    pad_multiple: int = 32,
) -> np.ndarray:
    """Run 2.5D inference over ``image`` and return a 3D prediction volume.

    Args:
        model: Trained segmentation model returning ``{"masks": [B, C, H, W]}``.
        image: 3D image volume (array-axis order, e.g. [X, Y, Z]).
        num_slices / slice_axis: 2.5D window size and slicing axis (match config).
        num_classes: Model output channels. ``1`` -> binary (sigmoid); ``>1`` ->
            multiclass (argmax).
        threshold: Binary foreground threshold (ignored for multiclass).
        lower_percentile / upper_percentile: Percentile-normalization bounds
            (must match training ``preprocessing.normalization.percentile``).
        aux: Optional 3D conditioning volume (same shape as ``image``); windowed
            and appended as channels like the training dataset.
        aux_mode / aux_max: ``"label"`` -> ``aux / aux_max``; ``"binary"`` -> ``aux > 0``.
        batch_size: Number of slices per forward pass.
        return_proba: Binary only — return the sigmoid probability instead of the
            thresholded mask (``float32``); multiclass always returns labels.

    Returns:
        3D ``np.ndarray`` (same shape/axis-order as ``image``): ``uint8`` labels,
        or ``float32`` probabilities when ``return_proba`` and binary.
    """
    if num_slices % 2 == 0:
        raise ValueError(f"num_slices must be odd, got {num_slices}")
    if aux is not None and aux.shape != image.shape:
        raise ValueError(f"aux shape {aux.shape} != image shape {image.shape}")

    model.eval()
    device = torch.device(device)
    model.to(device)

    half = num_slices // 2
    n = image.shape[slice_axis]
    orig_hw = tuple(s for a, s in enumerate(image.shape) if a != slice_axis)

    def take(arr, idx):
        idx = min(max(idx, 0), arr.shape[slice_axis] - 1)
        return np.take(arr, idx, axis=slice_axis)

    out_dtype = np.float32 if ((return_proba and num_classes == 1) or proba_class is not None
                               or return_class_proba) else np.uint8
    pred_slices: list[np.ndarray] = [None] * n  # type: ignore[list-item]

    for start in range(0, n, batch_size):
        centers = list(range(start, min(start + batch_size, n)))
        batch, offsets_meta = [], []
        for c in centers:
            window = np.stack([take(image, c + o) for o in range(-half, half + 1)], axis=-1).astype(np.float32)
            window = _percentile_normalize(window, lower_percentile, upper_percentile)
            if aux is not None:
                aux_win = np.stack([take(aux, c + o) for o in range(-half, half + 1)], axis=-1).astype(np.float32)
                aux_win = (aux_win > 0).astype(np.float32) if aux_mode == "binary" else aux_win / aux_max
                window = np.concatenate([window, aux_win], axis=-1)
            padded, (top, left) = _pad_to_multiple(window, pad_multiple, float(window.min()))
            batch.append(padded.transpose(2, 0, 1))  # [C, Hp, Wp]
            offsets_meta.append((top, left))

        x = torch.from_numpy(np.stack(batch)).float().to(device)
        logits = model({"image": x})["masks"]  # [B, C, Hp, Wp]

        if num_classes == 1:
            proba = torch.sigmoid(logits)[:, 0]  # [B, Hp, Wp]
            result = proba if return_proba else (proba > threshold).to(torch.uint8)
        elif return_class_proba:
            # Full per-class softmax [B, C, Hp, Wp] for multiclass ENSEMBLING.
            result = torch.softmax(logits, dim=1)
        elif proba_class is not None:
            # Softmax probability of a specific class (e.g. aneurysm) for FROC /
            # threshold sweeps instead of a hard argmax label.
            result = torch.softmax(logits, dim=1)[:, proba_class]  # [B, Hp, Wp]
        else:
            result = logits.argmax(dim=1).to(torch.uint8)  # [B, Hp, Wp]
        result = result.cpu().numpy()

        for i, c in enumerate(centers):
            top, left = offsets_meta[i]
            h, w = orig_hw
            if return_class_proba:
                # [C, Hp, Wp] -> crop -> [H, W, C]
                pred_slices[c] = result[i][:, top:top + h, left:left + w].transpose(1, 2, 0).astype(out_dtype)
            else:
                pred_slices[c] = result[i][top:top + h, left:left + w].astype(out_dtype)

    volume = np.stack(pred_slices, axis=slice_axis)  # class_proba -> [...,C] with Z at slice_axis
    return volume.astype(out_dtype)
