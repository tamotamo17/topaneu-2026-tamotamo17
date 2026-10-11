"""Auxiliary presence-head scoring of detector blobs.

The Gaussian heatmap detector trained from ``step3_detector_fold*.yaml``
(``monai_segresnet3d_auxcls``) carries a second head that answers "does this
patch contain an aneurysm centre?" for the whole sliding-window patch. Scoring a
blob with a patch CENTRED on it mirrors training (positives are foreground-
centred patches), so the head sees an in-distribution crop, and the resulting
probability combines with the heatmap peak in :func:`paint_blobs` to drop false
positives.

Both the fold-val evaluator (``scripts/infer_label_eval.py``) and the submission
path (``scripts/pipeline_core.py`` -> ``docker/``) build the scorer here so the
two cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import torch


def make_aux_score_fn(model, image, vessel, *, roi, lo_p, hi_p, aux_max, device):
    """Return ``fn(centroid) -> aux presence probability`` for one volume.

    Args:
        model: detector with a ``cls_head`` (returns ``{"cls_logits": ...}``).
        image / vessel: the 0.4mm crop volume and its predicted vessel map, in
            the same array-axis order the detector was run on.
        roi: patch edge length (``lightning_module.roi_size``).
        lo_p / hi_p: percentile-normalization bounds (``dataset.*_percentile``).
        aux_max: vessel-label scale (``dataset.aux_max``).
        device: torch device the model lives on.

    The normalization is done once per volume; ``fn`` only cuts the patch, so
    scoring N blobs costs N forward passes and nothing else.
    """
    lo, hi = np.percentile(image, lo_p), np.percentile(image, hi_p)
    img_n = np.clip((image.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    ves_n = vessel.astype(np.float32) / aux_max

    def fn(center):
        half = roi // 2
        c = [int(round(x)) for x in center]
        out = np.zeros((2, roi, roi, roi), np.float32)
        dst, src = [], []
        for a in range(3):
            lo_, hi_ = c[a] - half, c[a] + half
            lo_c, hi_c = max(0, lo_), min(img_n.shape[a], hi_)
            dst.append(slice(lo_c, hi_c))
            src.append(slice(lo_c - lo_, lo_c - lo_ + (hi_c - lo_c)))
        out[0][tuple(src)] = img_n[tuple(dst)]
        out[1][tuple(src)] = ves_n[tuple(dst)]
        with torch.no_grad():
            x = torch.from_numpy(out)[None].to(device)
            return float(torch.sigmoid(model({"image": x})["cls_logits"])[0, 0])

    return fn
