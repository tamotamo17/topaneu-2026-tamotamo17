"""Detector blobs -> labelled 0-50 mask, with optional Step4 fine-shape refinement.

The Step3 detector regresses a fixed-radius SPHERE around each aneurysm, so its
blob voxels overestimate the true lesion by ~33x — good for localization (Task1)
but poor for Task2 DSC. Step4 (`configs/segmentation/step4_fineseg_fold*.yaml`) segments the
real aneurysm shape inside a 64^3 crop centred on the blob; :class:`FineSeg`
applies that model at inference so the painted mask carries the true shape
instead of the sphere.

Both `scripts/infer_label_eval.py` (fold-val official metric) and
`scripts/pipeline_core.py` (raw-nii / Docker submission) call :func:`paint_blobs`
so the two paths cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import label as cc_label, center_of_mass

from scripts.anatomical_label import label_blob, label_blob_candidates


def _crop_bounds(shape, center, half: int):
    """Slices selecting the ``(2*half)^3`` cube around ``center``.

    Returns ``(dst, src)``: ``dst`` indexes the full volume, ``src`` indexes the
    same region inside the (possibly zero-padded) crop, so a crop-space array can
    be written back with ``vol[dst] = crop[src]``.
    """
    c = [int(round(x)) for x in center]
    dst, src = [], []
    for a in range(3):
        lo, hi = c[a] - half, c[a] + half
        lo_c, hi_c = max(0, lo), min(shape[a], hi)
        dst.append(slice(lo_c, hi_c))
        src.append(slice(lo_c - lo, lo_c - lo + (hi_c - lo_c)))
    return tuple(dst), tuple(src)


def _crop(vol, center, half: int, cval: float) -> np.ndarray:
    """Cut the cube around ``center``, padding out-of-bounds with ``cval``."""
    dst, src = _crop_bounds(vol.shape, center, half)
    out = np.full((2 * half,) * 3, cval, dtype=vol.dtype)
    out[src] = vol[dst]
    return out


class FineSeg:
    """Step4 fine aneurysm-shape segmenter applied around a detector blob.

    Input construction mirrors
    :class:`~src.segmentation.dataloaders.fineseg_crops.FineSegCropDataset`:
    2 channels = crop-wise percentile-normalized image + predicted vessel/``aux_max``.
    """

    def __init__(self, config, ckpt, device="cuda:0", size=None, thr=0.5):
        from scripts.infer_segmentation_2_5d import load_model  # local: avoids an import cycle

        cfg = OmegaConf.load(config)
        ds = cfg.dataset
        if str(ds.get("aux_encoding", "scalar")) != "scalar":
            raise ValueError("FineSeg supports the 2-channel scalar aux encoding only "
                             f"(config says aux_encoding={ds.get('aux_encoding')})")
        self.model = load_model(cfg, ckpt).to(device).eval()
        self.device = device
        self.thr = float(thr)
        self.lo_p = float(ds.get("lower_percentile", 0.5))
        self.hi_p = float(ds.get("upper_percentile", 99.5))
        self.aux_max = float(ds.get("aux_max", 36.0))
        # crop size is a property of the crops the model was trained on
        self.size = int(size if size is not None else cfg.preprocessing.image_size.get("x", 64))

    @torch.no_grad()
    def proba(self, image, vessel, center) -> tuple[np.ndarray, tuple, tuple]:
        """Aneurysm probability crop around ``center``, plus its write-back slices.

        Split out of :meth:`refine` so :class:`FineSegEnsemble` can average the
        probabilities of several folds BEFORE thresholding — averaging binary
        masks instead would quantise the very signal the rejector grades on.
        """
        half = self.size // 2
        img = _crop(image.astype(np.float32), center, half, float(image.min()))
        ves = _crop(vessel.astype(np.float32), center, half, 0.0)
        lo, hi = np.percentile(img, self.lo_p), np.percentile(img, self.hi_p)
        img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        x = np.stack([img, ves / self.aux_max], 0)[None]
        logits = self.model({"image": torch.from_numpy(x).float().to(self.device)})["masks"]
        proba = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
        dst, src = _crop_bounds(image.shape, center, half)
        return proba, dst, src

    def refine(self, image, vessel, center) -> tuple[np.ndarray, tuple, tuple, float]:
        """Predict the aneurysm shape around ``center``.

        Returns ``(crop_mask, dst, src, peak)`` — a boolean crop-space mask, the
        slices needed to write it into the full volume, and the maximum aneurysm
        probability anywhere in the crop.

        ``peak`` exists because the binary mask is a near-useless rejection
        signal: measured on fold0, fine-seg returns a NON-empty mask for ~99% of
        detector blobs (68/69 and 72/73 in two runs), so "did it segment
        anything" filters nothing. The graded peak probability is what can
        separate a real lesion from a spurious blob.
        """
        proba, dst, src = self.proba(image, vessel, center)
        return proba >= self.thr, dst, src, float(proba.max())


class FineSegEnsemble:
    """Several :class:`FineSeg` folds fused by MEAN probability.

    Duck-types :class:`FineSeg` — ``paint_blobs`` only ever calls ``.refine()`` —
    so nothing downstream needs to know whether it holds one model or five.

    Unlike the Step3 ensemble this one has no hidden constraint: Step4 has no
    per-model auxiliary head, so folds can be stacked freely. The members must
    agree on crop size, since the crop geometry decides the write-back slices.
    """

    def __init__(self, specs, device="cuda:0", size=None, thr=0.5):
        self.members = [FineSeg(c, k, device=device, size=size, thr=thr) for c, k in specs]
        if not self.members:
            raise ValueError("FineSegEnsemble needs at least one (config, ckpt)")
        sizes = {m.size for m in self.members}
        if len(sizes) != 1:
            raise ValueError(f"Step4 ensemble members disagree on crop size: {sorted(sizes)}")
        self.size = self.members[0].size
        self.thr = float(thr)
        self.device = device

    @torch.no_grad()
    def refine(self, image, vessel, center) -> tuple[np.ndarray, tuple, tuple, float]:
        probas, dst, src = [], None, None
        for m in self.members:
            p, dst, src = m.proba(image, vessel, center)
            probas.append(p)
        proba = np.mean(probas, axis=0)
        return proba >= self.thr, dst, src, float(proba.max())


def build_fineseg(specs, *, device="cuda:0", thr=0.5, size=None):
    """One or many ``(config, ckpt)`` -> the matching Step4 refiner (or ``None``)."""
    specs = [s for s in (specs or []) if s and s[0] and s[1]]
    if not specs:
        return None
    if len(specs) == 1:
        return FineSeg(specs[0][0], specs[0][1], device=device, thr=thr, size=size)
    return FineSegEnsemble(specs, device=device, thr=thr, size=size)


def paint_blobs(proba, vessel, *, thr=0.5, min_vox=30, r_vox=18, nb_frac=0.05,
                image=None, fineseg: FineSeg | None = None,
                relabel_from_refined: bool = False,
                aux_score_fn=None, aux_weight: float = 0.75, aux_thr: float = 0.5,
                label_topk: int = 1,
                fineseg_reject_thr: float | None = None, fineseg_reject_weight: float = 0.75):
    """Threshold -> CC blobs -> anatomical label -> painted 0-50 mask + label list.

    With ``fineseg`` (and the crop ``image``) each blob's sphere voxels are
    replaced by the fine-shape prediction in a crop around the blob centroid. If
    the refinement comes back empty inside the blob's own crop the sphere voxels
    are kept, so refinement can shrink a detection but never delete one.

    ``relabel_from_refined`` re-runs the anatomical labeller on the centroid of
    the REFINED mask instead of keeping the label derived from the detector's
    sphere. Off by default because it changes the submission path's Task1 output.

    Why it can matter: the labeller reads the predicted vessel map in a
    ``r_vox=18`` (7.2mm) ball around the centroid, and the residual label errors
    are ICA-C7 subdivisions — a scale where a 1-2mm centroid shift flips the
    answer. The detector's sphere centroid carries that much error (measured
    0.79mm for the sigma-2.5 heatmap, 1.20mm for sigma-5.0), whereas the
    fine-seg mask is the actual lesion. Without this flag Task1 is provably
    invariant to Step4 (measured byte-identical with and without refinement),
    so the whole Step4 stage can only ever move Task2.

    A refined centroid that lands with no vessel evidence keeps the original
    label rather than dropping the detection — refinement should not delete
    findings, only relocate them.

    ``aux_score_fn(centroid) -> float`` enables two-stage rescoring with a
    detector that has an auxiliary patch-presence head: each candidate blob is
    kept only if ``peak^(1-aux_weight) * aux^aux_weight >= aux_thr``, where
    ``peak`` is the blob's maximum heatmap probability. Use it with a LOW ``thr``
    so the pool is generous and the combined score does the deciding.

    Measured on fold0 (sigma-2.5 heatmap + aux head, extract thr 0.20): CPM
    0.638 -> 0.668 at ``aux_weight`` 0.75, with the gain concentrated where it
    matters — sensitivity at 0.125 FP/case 0.518 -> 0.600, case sensitivity at
    0.5 FP/case 0.803 -> 0.836. The aux score alone discriminates TP from FP
    about as well as the peak (AUC 0.876 vs 0.872) but on different evidence
    (whole-patch context vs the local peak), so the two combine to 0.890.

    The confidence must be applied HERE, per blob, and not to the voxel field:
    it is constant across a patch, so adding it to the heatmap logits floods a
    whole 51.2mm patch over the threshold as one component whose centroid is
    nowhere near a lesion (measured: CPM 0.629 -> 0.091 at weight 0.5). Geometry
    and centroids stay purely the heatmap's; the confidence only decides which
    blobs survive.

    ``label_topk`` > 1 reports that many candidate locations per blob in the
    returned label LIST (Task1) while still painting only the top-1 into the mask
    (Task2 needs one label per voxel). The rule labeller is only ~52% accurate and
    its confusions are adjacent ICA-C7 subdivisions, so the truth is often the
    runner-up; Task1 scores a list, and its GT side is de-duplicated, so trading
    one extra false positive per blob for a recovered true positive can pay.

    ``fineseg_reject_thr`` turns Step4 into a false-positive REJECTOR: a blob is
    dropped when ``peak^(1-w) * fs_peak^w`` falls below it, where ``fs_peak`` is
    the fine-seg model's maximum aneurysm probability in the crop. This inverts
    the default policy (an empty refinement falls back to the sphere so nothing
    is ever deleted), and is meant for running the detector permissively and
    letting the shape model discard the spurious candidates.

    It only works with a fine-seg model trained for it. The shipped
    ``fs_ftonly`` is not: its loss says a missed voxel is 10x worse than an
    invented one (beta 0.7 > alpha 0.3, class_weights [0.3, 3.0], and with
    ``include_background: false`` on 2 classes the class weights cancel out of
    the Tversky term entirely, leaving background suppression to a CE term of
    effective weight 0.1*0.3 = 0.03). Measured, its fs_peak is 1.000 for both
    true and false blobs — completely saturated, AUC 0.764. Retrained with a
    neutral loss (alpha=beta=0.5, gamma 1.0, ce_weight 0.3, class_weights
    [1,1]) the same architecture reaches AUC 0.827 alone and 0.888 combined with
    the detector peak, versus 0.821 for the detector peak by itself.

    Returns ``(mask, labels, stats)``; ``stats`` counts how the blobs were painted.
    """
    out = np.zeros(proba.shape, np.uint8)
    labels: list[int] = []
    stats = {"blobs": 0, "refined": 0, "fallback": 0, "sphere_vox": 0, "refined_vox": 0,
             "relabel_changed": 0, "relabel_same": 0, "relabel_none": 0,
             "aux_kept": 0, "aux_rejected": 0, "extra_labels": 0, "fs_rejected": 0}

    lab, n = cc_label(proba >= thr)
    if n == 0:
        return out, labels, stats
    sizes = np.bincount(lab.ravel())

    for k in range(1, n + 1):
        if sizes[k] < min_vox:
            continue
        blob = lab == k
        centroid = center_of_mass(blob)
        if aux_score_fn is not None:
            combined = float(proba[blob].max()) ** (1 - aux_weight) * aux_score_fn(centroid) ** aux_weight
            if combined < aux_thr:
                stats["aux_rejected"] += 1
                continue
            stats["aux_kept"] += 1
        cands = label_blob_candidates(centroid, vessel, r_vox=r_vox, nb_frac=nb_frac,
                                      k=max(1, int(label_topk)))
        if not cands:              # no vessel evidence nearby -> not a reportable finding
            continue
        loc, extra = cands[0], cands[1:]
        painted = blob
        if fineseg is not None:
            if image is None:
                raise ValueError("paint_blobs(fineseg=...) also needs the crop `image`")
            crop_mask, dst, src, fs_peak = fineseg.refine(image, vessel, centroid)
            if fineseg_reject_thr is not None:
                w = float(fineseg_reject_weight)
                if float(proba[blob].max()) ** (1 - w) * fs_peak ** w < fineseg_reject_thr:
                    stats["fs_rejected"] += 1
                    continue
            stats["blobs"] += 1
            stats["sphere_vox"] += int(blob.sum())
            refined = np.zeros_like(blob)
            refined[dst] = crop_mask[src]
            if refined.any():
                painted, stats["refined"] = refined, stats["refined"] + 1
                if relabel_from_refined:
                    # centroid in CROP space, mapped back to volume coordinates:
                    # crop[src] = vol[dst], so vol_idx = crop_idx - src.start + dst.start.
                    # (Computing it here rather than on the full-volume `refined`
                    # keeps this O(crop) instead of O(volume) per blob.)
                    cc = center_of_mass(crop_mask)
                    rc = tuple(c - s.start + d.start for c, s, d in zip(cc, src, dst))
                    loc2 = label_blob(rc, vessel, r_vox=r_vox, nb_frac=nb_frac)
                    if loc2 is None:
                        stats["relabel_none"] += 1
                    elif loc2 == loc:
                        stats["relabel_same"] += 1
                    else:
                        stats["relabel_changed"] += 1
                        loc = loc2
            else:
                stats["fallback"] += 1
        else:
            stats["blobs"] += 1
            stats["sphere_vox"] += int(blob.sum())

        stats["refined_vox"] += int(painted.sum())
        out[painted] = loc                 # mask carries the top-1 only
        labels.append(int(loc))
        labels.extend(int(e) for e in extra)
        stats["extra_labels"] += len(extra)

    return out, labels, stats
