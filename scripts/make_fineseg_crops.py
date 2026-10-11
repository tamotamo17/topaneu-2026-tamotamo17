#!/usr/bin/env python
"""CLI wrapper for Step4 fine aneurysm-seg crop generation.

Implementation: src/segmentation/dataloaders/fineseg_crops.py

Usage:
    PYTHONPATH=. uv run python scripts/make_fineseg_crops.py \
        --manifest <blob_manifest.json> \
        --images-dir .../aneurysm_hr/crop60_sp0.4/images \
        --vessel-dir .../multiclass_vessel_hr/crop60_sp0.4/pred_lrflip/pred \
        --gt-dir     .../aneurysm_hr/crop60_sp0.4/masks \
        --out        .../aneurysm_fineseg/fold0_sz64 --size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.segmentation.dataloaders.fineseg_crops import generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--vessel-dir", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--tol-mm", type=float, default=5.0)
    ap.add_argument("--spacing", type=float, default=0.4)
    args = ap.parse_args()

    s = generate(args.manifest, args.images_dir, args.vessel_dir, args.gt_dir, args.out,
                 size=args.size, tol_mm=args.tol_mm, spacing=args.spacing)
    print(f"wrote {s['n_crops']} crops (size {s['size']}) -> {args.out}")
    print(f"  centers: TP-blob={s['tp']} FP-blob={s['fp']} FN-gt={s['fn']}")
    print(f"  positive-target={s['pos']} empty={s['n_crops']-s['pos']}  | train={s['train']} val={s['val']}")


if __name__ == "__main__":
    main()
