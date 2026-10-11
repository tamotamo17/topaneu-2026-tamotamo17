#!/usr/bin/env python
"""Coordinate FROC for the **3D** aneurysm sphere detector.

Mirror of ``scripts/froc_aneurysm.py`` but the per-voxel aneurysm probability is
produced by MONAI ``sliding_window_inference`` over the full 3D volume (image +
optional vessel-aux channel), matching the ``patches_3d`` / ``segmentation_3d``
training pipeline. Predicted sphere-blob centroids are matched to the TRUE
aneurysm centroids (from the location masks) within ``--tol-mm``; reports
detection sensitivity, per-case sensitivity, FP/case and localization error —
directly comparable to the 2.5D sphere FROC.

Usage:
    PYTHONPATH=. uv run python scripts/froc_aneurysm_3d.py \
        --config configs/segmentation/sw_C_r5_cw5_b44.yaml \
        --ckpt   results/.../best-*.ckpt \
        --images-dir .../aneurysm_hr/crop60_sp0.4/images \
        --aux-dir    .../multiclass_vessel_hr/crop60_sp0.4/pred_lrflip/pred \
        --fold-yaml  .../aneurysm_hr/crop60_sp0.4/folds/fold0.yaml \
        --gt-dir     .../aneurysm_hr/crop60_sp0.4/masks --gpu 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import label, center_of_mass
from tqdm import tqdm

from monai.inferers import sliding_window_inference

from scripts.infer_segmentation_2_5d import load_model


@torch.no_grad()
def predict_proba_3d(model, image, aux, *, lo_p, hi_p, aux_mode, aux_max,
                     roi, sw_batch, overlap, device, proba_class=1,
                     aux_encoding="scalar", n_vessel=37, activation="softmax"):
    """Full-volume aneurysm probability via sliding window.

    ``activation``: "softmax" -> class-``proba_class`` of a multiclass head;
    "sigmoid" -> the single-channel Gaussian-heatmap head (1-class model).
    ``aux_encoding``: "scalar" (label/aux_max, 1ch) or "onehot" (n_vessel ch) — must
    match the detector's training input encoding.
    """
    img = image.astype(np.float32)
    lo, hi = np.percentile(img, lo_p), np.percentile(img, hi_p)
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    if aux is None:
        arr = img[None]
    elif aux_encoding == "onehot":
        oh = np.eye(n_vessel, dtype=np.float32)[np.clip(aux.astype(np.int64), 0, n_vessel - 1)]
        arr = np.concatenate([img[None], oh.transpose(3, 0, 1, 2)], 0)
    else:
        a = aux.astype(np.float32)
        a = (a > 0).astype(np.float32) if aux_mode == "binary" else a / aux_max
        arr = np.stack([img, a], 0)
    x = torch.from_numpy(arr[None]).float().to(device)  # [1,C,X,Y,Z]

    def _pred(t):
        return model({"image": t})["masks"]

    logits = sliding_window_inference(x, roi, sw_batch, _pred, overlap=overlap, mode="gaussian")
    if activation == "sigmoid":
        return torch.sigmoid(logits)[0, 0].cpu().numpy()
    return torch.softmax(logits, dim=1)[0, proba_class].cpu().numpy()


def centroids(binvol, minv):
    l, n = label(binvol)
    if n == 0:
        return []
    sizes = np.bincount(l.ravel())
    return [np.array(center_of_mass(l == k)) for k in range(1, n + 1) if sizes[k] >= minv]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--aux-dir", type=Path, default=None)
    ap.add_argument("--fold-yaml", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--min-vox", type=int, default=30)
    ap.add_argument("--spacing", type=float, default=0.4)
    ap.add_argument("--tol-mm", type=float, default=5.0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = f"cuda:{args.gpu}" if args.gpu is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.ckpt).to(device).eval()

    ds, lm = cfg.dataset, cfg.lightning_module
    pk = dict(
        lo_p=float(ds.get("lower_percentile", 0.5)), hi_p=float(ds.get("upper_percentile", 99.5)),
        aux_mode=str(ds.get("aux_mode", "label")), aux_max=float(ds.get("aux_max", 36.0)),
        aux_encoding=str(ds.get("aux_encoding", "scalar")), n_vessel=int(ds.get("n_vessel", 37)),
        roi=(int(lm.get("roi_size", 96)),) * 3, sw_batch=int(lm.get("sw_batch_size", 2)),
        overlap=float(lm.get("overlap", 0.5)), device=device, proba_class=1,
        activation="sigmoid" if int(cfg.model.get("num_classes", 2)) == 1 else "softmax",
    )

    val = [r["id"] for r in OmegaConf.load(args.fold_yaml).val]
    probs, gt_cens = {}, {}
    for c in tqdm(val, desc="proba3d"):
        img = np.load(args.images_dir / f"{c}.npy")
        aux = np.load(args.aux_dir / f"{c}.npy") if args.aux_dir else None
        probs[c] = predict_proba_3d(model, img, aux, **pk)
        gt_cens[c] = centroids(np.load(args.gt_dir / f"{c}.npy") > 0, 1)
    n_an = sum(len(v) for v in gt_cens.values())
    tolv = args.tol_mm / args.spacing

    print(f"\n=== 3D Coordinate FROC ({len(val)} cases, {n_an} aneurysms, "
          f"tol={args.tol_mm}mm, min_vox={args.min_vox}) ===")
    print(f"{'thr':>5}{'sens':>8}{'case_sens':>11}{'FP/case':>10}{'loc_err_mm':>12}")
    for t in args.thresholds:
        tp = fn = fp = 0; case_hit = case_tot = 0; errs = []
        for c in val:
            gc = gt_cens[c]; pc = centroids(probs[c] >= t, args.min_vox); used = set()
            for g in gc:
                bd, bi = 1e9, None
                for i, p in enumerate(pc):
                    if i in used:
                        continue
                    d = np.linalg.norm(g - p)
                    if d < bd:
                        bd, bi = d, i
                if bi is not None and bd <= tolv:
                    tp += 1; used.add(bi); errs.append(bd * args.spacing)
                else:
                    fn += 1
            fp += len(pc) - len(used)
            if gc:
                case_tot += 1; case_hit += len(used) > 0
        sens = tp / (tp + fn) if (tp + fn) else 0
        le = np.mean(errs) if errs else float("nan")
        print(f"{t:>5.2f}{sens:>8.3f}{case_hit/max(case_tot,1):>11.3f}{fp/len(val):>10.2f}{le:>12.2f}")


if __name__ == "__main__":
    main()
