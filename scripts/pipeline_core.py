"""Reusable full-pipeline inference core (shared by scripts/infer_pipeline_raw.py
and the Docker submission containers).

`run_pipeline` takes a volume in (x,y,z) index order + its per-axis spacing +
modality, and returns the original-grid location mask (x,y,z, uint8 0-52) and the
sorted list of detected location ids. All stage models are passed as
(config, ckpt) paths so callers can point at baked-in weights.

The blob post-processing knobs (`aux_rescore_*`, `fineseg_reject_*`,
`label_topk`) mirror `scripts/infer_label_eval.py` one-for-one, so a setting
tuned on the fold-val official metric can be carried into the submission
container without re-deriving it.
"""

from __future__ import annotations

import time

import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import zoom

from src.detection.vessel_roi_preproc import make_projection, CTA_HU_WINDOW, CTA_SLAB_HALF
from scripts.infer_vessel_roi_detection import build_rfdetr, top_box
from scripts.make_highres_vessel_crop import resample_cube
from scripts.infer_segmentation_2_5d import load_model
from src.segmentation.inference.predict_volume import predict_volume
from src.segmentation.inference.aux_score import make_aux_score_fn
from src.segmentation.inference.blob_mask import build_fineseg, paint_blobs
from scripts.froc_aneurysm_3d import predict_proba_3d
from scripts.uncrop_to_original import uncrop


def _step1_center_mm(vol625, modality, det, thr, margin, center_sp=0.625):
    nX, nY, nZ = vol625.shape
    cor = make_projection(vol625, "cor", modality, CTA_HU_WINDOW, CTA_SLAB_HALF)
    sag = make_projection(vol625, "sag", modality, CTA_HU_WINDOW, CTA_SLAB_HALF)
    cb, _ = top_box(det.predict(np.repeat(cor[..., None], 3, 2), threshold=thr), cor.shape)
    sb, _ = top_box(det.predict(np.repeat(sag[..., None], 3, 2), threshold=thr), sag.shape)
    cx1, cy1, cx2, cy2 = cb
    sx1, sy1, sx2, sy2 = sb
    ranges = [(cy1, cy2, nX), (sy1, sy2, nY), (min(cx1, sx1), max(cx2, sx2), nZ)]  # X,Y,Z

    def clamp(lo, hi, n):
        lo = max(0, int(np.floor(lo)) - margin); hi = min(n, int(np.ceil(hi)) + margin)
        return (0, n) if hi <= lo else (lo, hi)

    center_mm = []
    for lo, hi, n in ranges:
        clo, chi = clamp(lo, hi, n)
        center_mm.append(((clo + chi) / 2) * center_sp)
    return center_mm


class _Timer:
    """Stage timer that reports through ``log`` (a no-op when ``log`` is None)."""

    def __init__(self, log):
        self.log = log
        self.t0 = time.time()

    def mark(self, stage, **extra):
        if self.log is None:
            self.t0 = time.time()
            return
        dt = time.time() - self.t0
        detail = "".join(f" {k}={v}" for k, v in extra.items())
        self.log(f"{stage}: {dt:.1f}s{detail}")
        self.t0 = time.time()


def run_pipeline(
    vol, orig_sp, modality, *,
    step1_config, step1_ckpt, step2_config, step2_ckpt,
    step3_config=None, step3_ckpt=None, step3_specs=None, step3_combine="mean",
    step4_config=None, step4_ckpt=None, step4_specs=None, step4_thr=0.5,
    relabel_from_refined=False,
    aux_rescore_weight=None, aux_rescore_thr=0.5,
    fineseg_reject_thr=None, fineseg_reject_weight=0.75, label_topk=1,
    device="cuda:0", half_mm=60.0, target_sp=0.4, size=300,
    det_thr=0.5, min_vox=30, r_vox=18, nb_frac=0.05, step1_thr=0.3, step1_margin=8,
    log=None,
):
    """``step3_specs`` = list of ``(config, ckpt)`` for a fold/arch ENSEMBLE; their
    class-1 probability maps are fused (``step3_combine`` = "mean" or "max") before
    blob extraction. Falls back to the single ``step3_config``/``step3_ckpt``.

    ``step4_config``/``step4_ckpt`` (optional) enable the fine aneurysm-shape seg:
    each blob's sphere voxels are replaced by the predicted lesion shape, which is
    what Task2's DSC measures. Task1's location list is unchanged either way —
    unless ``relabel_from_refined`` or ``fineseg_reject_thr`` is set.
    ``step4_specs`` = list of ``(config, ckpt)`` takes precedence and fuses the
    folds by MEAN probability. Step4 has no per-model auxiliary head, so unlike
    ``step3_specs`` it carries no restriction on the rest of the configuration.

    ``aux_rescore_weight`` turns on two-stage rescoring with the detector's
    auxiliary presence head (needs exactly one detector, built from
    ``monai_segresnet3d_auxcls``): a blob survives only when
    ``peak^(1-w) * aux^w >= aux_rescore_thr``. ``fineseg_reject_thr`` lets Step4
    veto a blob the same way. Both are documented on :func:`paint_blobs`.

    ``log`` (e.g. ``print``) receives one line per stage with its wall time; the
    submission containers use it so a Grand Challenge job log shows where the
    12-minute per-case budget went.

    Returns ``(mask, sorted location ids, stats)``.
    """
    if step3_specs is None:
        step3_specs = [(step3_config, step3_ckpt)]
    if aux_rescore_weight is not None and len(step3_specs) != 1:
        raise ValueError("aux_rescore_weight needs exactly one detector "
                         "(the presence head is per-model)")
    vol = np.asarray(vol, dtype=np.float32)
    orig_shape = vol.shape
    t = _Timer(log)

    # Step1: ROI center (mm from corner) via 0.625 resample + RF-DETR
    f = tuple(orig_sp[a] / 0.625 for a in range(3))
    vol625 = zoom(vol, f, order=1)
    s1 = OmegaConf.to_container(OmegaConf.load(step1_config).model, resolve=True)
    det = build_rfdetr(s1["type"].replace("rfdetr_", ""), s1.get("num_classes", 1),
                       s1.get("resolution", 384), str(step1_ckpt))
    center_mm = _step1_center_mm(vol625, modality, det, step1_thr, step1_margin)
    del det, vol625
    t.mark("step1 vessel-ROI detection", center_mm=[round(c, 1) for c in center_mm])

    # crop 0.4mm cube from the raw volume
    crop_img = resample_cube(vol, orig_sp, center_mm, half_mm, target_sp, size,
                             order=1, cval=float(vol.min()))
    t.mark("crop 0.4mm cube", shape=crop_img.shape)

    # Step2: 2.5D vessel seg (37-class)
    s2 = OmegaConf.load(step2_config)
    m2 = load_model(s2, step2_ckpt)
    ds2 = s2.dataset
    vessel = predict_volume(
        m2, crop_img, num_slices=int(ds2.get("num_slices", 5)), slice_axis=int(ds2.get("slice_axis", 2)),
        num_classes=int(s2.model.get("num_classes", 37)),
        lower_percentile=float(s2.preprocessing.normalization.percentile.get("lower_percentile", 0.5)),
        upper_percentile=float(s2.preprocessing.normalization.percentile.get("upper_percentile", 99.5)),
        device=device)
    del m2
    t.mark("step2 vessel seg", n_labels=int((np.unique(vessel) > 0).sum()))

    # Step3: 3D aneurysm detector(s) (image + predicted vessel) -> fused proba
    vessel_f = vessel.astype(np.float32)
    probas = []
    aux_fn = None
    for cfg3, ck3 in step3_specs:
        s3 = OmegaConf.load(cfg3)
        m3 = load_model(s3, ck3).to(device).eval()
        ds3, lm3 = s3.dataset, s3.lightning_module
        roi = int(lm3.get("roi_size", 96))
        lo_p, hi_p = float(ds3.get("lower_percentile", 0.5)), float(ds3.get("upper_percentile", 99.5))
        aux_max = float(ds3.get("aux_max", 36.0))
        # 1-class head = Gaussian-heatmap detector -> sigmoid; multiclass -> softmax.
        activation = "sigmoid" if int(s3.model.get("num_classes", 2)) == 1 else "softmax"
        probas.append(predict_proba_3d(
            m3, crop_img, vessel_f,
            lo_p=lo_p, hi_p=hi_p,
            aux_mode=str(ds3.get("aux_mode", "label")), aux_max=aux_max,
            aux_encoding=str(ds3.get("aux_encoding", "scalar")), n_vessel=int(ds3.get("n_vessel", 37)),
            roi=(roi,) * 3, sw_batch=int(lm3.get("sw_batch_size", 2)),
            overlap=float(lm3.get("overlap", 0.5)), device=device, proba_class=1, activation=activation))
        if aux_rescore_weight is not None:
            if not hasattr(m3, "cls_head"):
                raise ValueError(f"aux_rescore_weight needs a detector with an aux head "
                                 f"(monai_segresnet3d_auxcls); got {type(m3).__name__}")
            aux_fn = make_aux_score_fn(m3, crop_img, vessel_f, roi=roi, lo_p=lo_p,
                                       hi_p=hi_p, aux_max=aux_max, device=device)
        else:
            del m3
    proba = np.max(probas, axis=0) if step3_combine == "max" else np.mean(probas, axis=0)
    t.mark("step3 aneurysm heatmap", n_models=len(probas), peak=round(float(proba.max()), 3))

    # blobs -> anatomical label -> (Step4 fine shape) -> crop mask
    fineseg = build_fineseg(step4_specs if step4_specs is not None
                            else [(step4_config, step4_ckpt)],
                            device=device, thr=step4_thr)
    crop_mask, locs, stats = paint_blobs(proba, vessel, thr=det_thr, min_vox=min_vox,
                                         r_vox=r_vox, nb_frac=nb_frac,
                                         image=crop_img, fineseg=fineseg,
                                         relabel_from_refined=relabel_from_refined,
                                         aux_score_fn=aux_fn,
                                         aux_weight=(aux_rescore_weight or 0.75),
                                         aux_thr=aux_rescore_thr,
                                         label_topk=label_topk,
                                         fineseg_reject_thr=fineseg_reject_thr,
                                         fineseg_reject_weight=fineseg_reject_weight)
    t.mark("step4 fine-seg + labelling", blobs=stats["blobs"], locations=sorted(set(locs)))

    # uncrop to original grid
    meta = {"orig_shape": list(orig_shape), "orig_spacing": list(orig_sp),
            "target_spacing": target_sp, "center_mm": center_mm, "half_mm": half_mm}
    orig_mask = uncrop(crop_mask, meta)
    t.mark("uncrop to original grid", shape=orig_mask.shape)
    return orig_mask, sorted(set(locs)), stats
