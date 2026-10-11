#!/usr/bin/env python
"""Invert the Step2/3 crop: map a crop-space label mask (size^3 @ target_spacing)
back onto the ORIGINAL NIfTI voxel grid, using crop_meta written by
make_highres_vessel_crop.py.

Forward crop (make_highres_vessel_crop.resample_cube):
    in_idx = out_idx * (target_sp/orig_sp) + (center_mm - half_mm)/orig_sp
Inverse (this file) — resample the crop back onto the original grid:
    crop_idx = orig_idx * (orig_sp/target_sp) - (center_mm - half_mm)/target_sp
which is exactly scipy affine_transform(crop, matrix=diag(orig_sp/target_sp),
offset=-(center_mm-half_mm)/target_sp, output_shape=orig_shape, order=0).

Voxels outside the crop map to background 0. Nearest interpolation preserves
integer labels. Result shape == crop_meta["orig_shape"] == GT mask shape, as the
official evaluator requires (pred.shape == gt.shape).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform


def uncrop(crop_mask: np.ndarray, meta: dict) -> np.ndarray:
    orig_sp = meta["orig_spacing"]
    orig_shape = tuple(meta["orig_shape"])
    tsp = meta["target_spacing"]
    start_mm = [meta["center_mm"][a] - meta["half_mm"] for a in range(3)]
    matrix = np.diag([orig_sp[a] / tsp for a in range(3)])
    offset = [-start_mm[a] / tsp for a in range(3)]
    return affine_transform(
        crop_mask.astype(np.float32), matrix=matrix, offset=offset,
        output_shape=orig_shape, order=0, cval=0.0, mode="constant",
    ).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crop-mask", required=True, type=Path, help="crop-space label .npy")
    ap.add_argument("--meta", required=True, type=Path, help="crop_meta/<case>.json")
    ap.add_argument("--out", required=True, type=Path, help="output original-grid mask .npy")
    args = ap.parse_args()
    meta = json.load(open(args.meta))
    om = uncrop(np.load(args.crop_mask), meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, om)
    print(f"uncropped -> {args.out}  shape={om.shape}  nonzero_labels={sorted(np.unique(om[om>0]).tolist())}")


if __name__ == "__main__":
    main()
