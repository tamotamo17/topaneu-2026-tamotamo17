#!/usr/bin/env python
"""Detection-based Step1 inference: 2D vessel-ROI boxes -> 3D ROI (RSNA-3rd style).

Runs the trained RF-DETR vessel-region detector on each case's coronal and
sagittal MIP projections, then fuses the two 2D boxes into a 3D ROI:

    coronal  box (image.max(axis=1), [X,Z]) -> X range = [y1,y2], Z range = [x1,x2]
    sagittal box (image.max(axis=0), [Y,Z]) -> Y range = [y1,y2], Z range = [x1,x2]
    3D ROI: X from coronal, Y from sagittal, Z = union of both Z ranges (+margin)

Writes ``<out>/roi_meta/<case>.json`` in the same schema as
``scripts/crop_to_vessel_region.py`` (``orig_shape`` + ``bbox`` [[lo,hi]x3]), so
the detected ROI can drive cropping just like the segmentation bbox. Also stores
the raw 2D predicted boxes (``<out>/pred2d/<case>.json``) for visualization.

Usage:
    PYTHONPATH=. uv run python scripts/infer_vessel_roi_detection.py \
        --config configs/detection/step1_vessel_roi.yaml \
        --ckpt   results/.../fold0/checkpoint_best_total.pth \
        --images-dir data/vessel_lr/images \
        --out-dir    data/step1_vessel_roi/roi \
        --margin 8 --threshold 0.3 --gpu 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from src.detection.vessel_roi_preproc import make_projection, modality_of


def build_rfdetr(size: str, num_classes: int, resolution: int, ckpt: str):
    """Build an RF-DETR model natively with trained weights (bypasses the
    generic wrapper's checkpoint loader, which targets a different API)."""
    import rfdetr

    cls = {
        "nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium, "base": rfdetr.RFDETRBase,
        "large": rfdetr.RFDETRLarge,
    }[size]
    return cls(pretrain_weights=ckpt, num_classes=num_classes, resolution=resolution)


def to_uint8(proj: np.ndarray, lo_p=0.5, hi_p=99.5) -> np.ndarray:
    lo, hi = np.percentile(proj, lo_p), np.percentile(proj, hi_p)
    out = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    return (out * 255).astype(np.uint8)


def top_box(det, shape_hw):
    """Highest-confidence box as (x1,y1,x2,y2); full-image fallback if none."""
    if det is None or len(det) == 0:
        h, w = shape_hw
        return (0.0, 0.0, float(w), float(h)), False
    xyxy = np.asarray(det.xyxy)
    conf = np.asarray(det.confidence)
    i = int(conf.argmax())
    return tuple(float(v) for v in xyxy[i]), True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="detection training config (for model spec)")
    ap.add_argument("--ckpt", required=True, type=Path, help="checkpoint_best_total.pth")
    ap.add_argument("--images-dir", required=True, type=Path, help="3D image .npy volumes")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--margin", type=int, default=8, help="ROI margin in voxels")
    ap.add_argument("--threshold", type=float, default=0.3, help="detector confidence threshold")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--cta-window", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="CTA HU window (must match training)")
    ap.add_argument("--cta-slab", type=int, default=None,
                    help="CTA central-slab half-thickness (must match training)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from src.detection.vessel_roi_preproc import CTA_HU_WINDOW, CTA_SLAB_HALF
    cta_window = tuple(args.cta_window) if args.cta_window else CTA_HU_WINDOW
    cta_slab = args.cta_slab if args.cta_slab is not None else CTA_SLAB_HALF

    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = OmegaConf.load(args.config)
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    size = mcfg["type"].replace("rfdetr_", "")
    model = build_rfdetr(size, mcfg.get("num_classes", 1), mcfg.get("resolution", 384), str(args.ckpt))

    roi_dir = args.out_dir / "roi_meta"; roi_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.out_dir / "pred2d"; pred_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted(p.stem for p in args.images_dir.glob("*.npy"))
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases | model={mcfg['type']} thr={args.threshold} margin={args.margin}")

    n_missing = 0
    for case in tqdm(cases, desc="roi-infer"):
        image = np.load(args.images_dir / f"{case}.npy")
        nX, nY, nZ = image.shape
        modality = modality_of(case)
        cor = make_projection(image, "cor", modality, cta_window, cta_slab)  # [X, Z]
        sag = make_projection(image, "sag", modality, cta_window, cta_slab)  # [Y, Z]
        cor_rgb = np.repeat(cor[..., None], 3, axis=2)  # RF-DETR expects RGB
        sag_rgb = np.repeat(sag[..., None], 3, axis=2)

        cor_box, cor_ok = top_box(model.predict(cor_rgb, threshold=args.threshold), cor.shape)
        sag_box, sag_ok = top_box(model.predict(sag_rgb, threshold=args.threshold), sag.shape)
        if not (cor_ok and sag_ok):
            n_missing += 1

        # coronal: x=Z, y=X ; sagittal: x=Z, y=Y
        cx1, cy1, cx2, cy2 = cor_box
        sx1, sy1, sx2, sy2 = sag_box
        x0, x1 = cy1, cy2                     # X from coronal rows
        y0, y1 = sy1, sy2                     # Y from sagittal rows
        z0, z1 = min(cx1, sx1), max(cx2, sx2)  # Z union of both cols

        def clamp(lo, hi, n):
            lo = max(0, int(np.floor(lo)) - args.margin)
            hi = min(n, int(np.ceil(hi)) + args.margin)
            if hi <= lo:  # degenerate -> full axis
                return 0, n
            return lo, hi

        bbox = [list(clamp(x0, x1, nX)), list(clamp(y0, y1, nY)), list(clamp(z0, z1, nZ))]
        (roi_dir / f"{case}.json").write_text(json.dumps(
            {"case": case, "orig_shape": [nX, nY, nZ], "bbox": bbox, "margin": args.margin}, indent=2))
        (pred_dir / f"{case}.json").write_text(json.dumps(
            {"case": case, "cor_box": cor_box, "cor_ok": cor_ok,
             "sag_box": sag_box, "sag_ok": sag_ok,
             "cor_shape": list(cor.shape), "sag_shape": list(sag.shape)}, indent=2))

    print(f"Wrote ROIs to {roi_dir} ({n_missing} cases had a missing plane detection -> full-axis fallback)")


if __name__ == "__main__":
    main()
