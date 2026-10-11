#!/usr/bin/env python
"""High-resolution vessel-ROI crop from the ORIGINAL NIfTI (Step2 input).

For each case, take the detection ROI center (from the RF-DETR Step1 detector,
expressed in the coarse ``--center-spacing`` grid), and resample a fixed
``±--half-mm`` cube around it **directly from the original ``.nii.gz``** to
``--target-spacing`` (isotropic). Sampling from the native volume (MR in-plane
~0.30 mm, CT ~0.43 mm / Z 0.25 mm) recovers real detail that the 0.625 mm
preprocessing had discarded — the point of the higher-resolution Step2.

Crop + resample is done in one pass with an affine (diagonal scale + offset),
so out-of-bounds target voxels are filled by padding:
    - image : padded with the volume's own min intensity (linear interp),
    - mask  : padded with 0 = background (nearest interp, labels preserved).

Output (fixed cube, size = round(2*half_mm / target_spacing) per axis):
    <out>/images/<case>.npy   int16
    <out>/masks/<case>.npy    uint8   (multiclass vessel labels 0..36)
    <out>/crop_meta/<case>.json   center_mm / half_mm / target_spacing / size /
        orig_shape / orig_spacing   (enough to map predictions back)

Usage:
    PYTHONPATH=. uv run python scripts/make_highres_vessel_crop.py \
        --data-root data/raw/topaneu \
        --roi-meta-dir data/step1_vessel_roi/roi/roi_meta \
        --center-spacing 0.625 --half-mm 60 --target-spacing 0.4 \
        --mask-source vessel \
        --out-dir data/vessel_hr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
from tqdm import tqdm


def resample_cube(vol, orig_sp, center_mm, half_mm, target_sp, size, order, cval):
    """Sample a [size]^3 cube at ``target_sp`` centered on ``center_mm`` (mm from
    the index-0 corner) out of ``vol`` (spacing ``orig_sp``). OOB -> ``cval``."""
    start_mm = [center_mm[a] - half_mm for a in range(3)]
    # output idx -> input idx: in = out * (target_sp/orig_sp) + start_mm/orig_sp
    matrix = np.diag([target_sp / orig_sp[a] for a in range(3)])
    offset = [start_mm[a] / orig_sp[a] for a in range(3)]
    return affine_transform(
        vol.astype(np.float32), matrix=matrix, offset=offset,
        output_shape=(size, size, size), order=order, cval=float(cval), mode="constant",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path, help="topaneu_deployment (images/ + *_masks/)")
    ap.add_argument("--roi-meta-dir", required=True, type=Path, help="detection ROI meta (bbox in center-spacing grid)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--center-spacing", type=float, default=0.625, help="isotropic spacing of the ROI-center grid")
    ap.add_argument("--half-mm", type=float, default=60.0, help="crop half-size in mm (±)")
    ap.add_argument("--target-spacing", type=float, default=0.4, help="output isotropic spacing (mm)")
    ap.add_argument("--mask-source", choices=["vessel", "location"], default="vessel")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    mask_dir_name = {"vessel": "vessel_masks", "location": "location_masks"}[args.mask_source]
    images_dir = args.data_root / "images"
    masks_dir = args.data_root / mask_dir_name
    out_img = args.out_dir / "images"; out_msk = args.out_dir / "masks"; out_meta = args.out_dir / "crop_meta"
    for d in (out_img, out_msk, out_meta):
        d.mkdir(parents=True, exist_ok=True)

    size = int(round(2 * args.half_mm / args.target_spacing))
    cases = sorted(p.stem for p in args.roi_meta_dir.glob("*.json"))
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases | cube={size}^3 @ {args.target_spacing}mm | ±{args.half_mm}mm | mask={args.mask_source}")

    for case in tqdm(cases, desc="hr-crop"):
        if not args.overwrite and (out_img / f"{case}.npy").exists() and (out_msk / f"{case}.npy").exists():
            continue
        img_path = images_dir / f"{case}_0000.nii.gz"
        msk_path = masks_dir / f"{case}.nii.gz"
        if not img_path.exists() or not msk_path.exists():
            print(f"[skip] missing nii for {case}")
            continue
        bbox = json.loads((args.roi_meta_dir / f"{case}.json").read_text())["bbox"]
        center_vox = [(lo + hi) / 2.0 for lo, hi in bbox]              # in center-spacing grid
        center_mm = [c * args.center_spacing for c in center_vox]      # mm from index-0 corner

        img_nii = nib.load(str(img_path))
        orig_sp = tuple(float(z) for z in img_nii.header.get_zooms()[:3])
        image = np.asarray(img_nii.dataobj)
        mask = np.asarray(nib.load(str(msk_path)).dataobj)
        if image.shape != mask.shape:
            print(f"[skip] {case}: image {image.shape} != mask {mask.shape}")
            continue

        img_c = resample_cube(image, orig_sp, center_mm, args.half_mm, args.target_spacing, size,
                              order=1, cval=float(image.min()))
        msk_c = resample_cube(mask, orig_sp, center_mm, args.half_mm, args.target_spacing, size,
                              order=0, cval=0)
        np.save(out_img / f"{case}.npy", np.round(img_c).astype(np.int16))
        np.save(out_msk / f"{case}.npy", np.round(msk_c).astype(np.uint8))
        (out_meta / f"{case}.json").write_text(json.dumps({
            "case": case, "center_mm": center_mm, "half_mm": args.half_mm,
            "target_spacing": args.target_spacing, "size": size,
            "orig_shape": list(image.shape), "orig_spacing": list(orig_sp),
            "center_spacing": args.center_spacing,
        }, indent=2))

    print(f"Wrote high-res crops -> {args.out_dir}")


if __name__ == "__main__":
    main()
