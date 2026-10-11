#!/usr/bin/env python
"""Build a 2D vessel-region DETECTION dataset from 3D volumes (RSNA-3rd style).

Instead of segmenting the vessel tree, we localize the vessel region with a 2D
detector on orthogonal maximum-intensity projections (as in the RSNA 2025 3rd
place solution). For each case this writes two projection images and one
vessel-region bounding box per image:

- **coronal**  = ``image.max(axis=1)``  -> [X, Z]  (box spans the vessel X,Z extent)
- **sagittal** = ``image.max(axis=0)``  -> [Y, Z]  (box spans the vessel Y,Z extent)

Together the two boxes reconstruct a 3D ROI at inference
(``scripts/infer_vessel_roi_detection.py``): X from coronal, Y from sagittal,
Z from the union of both. Projections are percentile-normalized to 8-bit PNG.

Output:
    <out>/images/<case>_cor.png , <case>_sag.png
    <out>/annotations.csv         (filepath,label,x1,y1,x2,y2,split)

``split`` (train/valid) comes from the given fold YAML so it matches the rest of
the pipeline. Feed the CSV to ``scripts/convert_annotations_to_coco.py``.

Usage:
    PYTHONPATH=. uv run python scripts/make_vessel_roi_detection_data.py \
        --images-dir data/vessel_lr/images \
        --masks-dir  data/vessel_lr/masks \
        --fold-yaml  data/vessel_lr/folds/fold0.yaml \
        --out-dir    data/step1_vessel_roi/det
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from src.detection.vessel_roi_preproc import make_projection, modality_of

LABEL = "vessel_region"


def bbox_from_mask(mask2d: np.ndarray):
    """(x1, y1, x2, y2) of the foreground in a [rows, cols] mask, or None."""
    rows = np.any(mask2d, axis=1)
    cols = np.any(mask2d, axis=0)
    if not rows.any() or not cols.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    # x = column, y = row; +1 so the box includes the last voxel
    return float(c0), float(r0), float(c1 + 1), float(r1 + 1)


def save_png(arr2d: np.ndarray, path: Path) -> None:
    """Save a [rows, cols] array as an RGB PNG (rows=height, cols=width)."""
    img = Image.fromarray(arr2d).convert("RGB")
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--masks-dir", required=True, type=Path)
    ap.add_argument("--fold-yaml", required=True, type=Path, help="fold YAML providing the train/val split")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cta-window", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="CTA HU window (default: module constant)")
    ap.add_argument("--cta-slab", type=int, default=None,
                    help="CTA central-slab half-thickness (0=single slice; default: module constant)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from src.detection.vessel_roi_preproc import CTA_HU_WINDOW, CTA_SLAB_HALF
    cta_window = tuple(args.cta_window) if args.cta_window else CTA_HU_WINDOW
    cta_slab = args.cta_slab if args.cta_slab is not None else CTA_SLAB_HALF

    doc = OmegaConf.load(args.fold_yaml)
    split_of = {r["id"]: "train" for r in doc.train}
    split_of.update({r["id"]: "valid" for r in doc.val})

    img_out = args.out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    rows_csv = []

    cases = sorted(p.stem for p in args.images_dir.glob("*.npy") if p.stem in split_of)
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases | out={args.out_dir}")

    for case in tqdm(cases, desc="project"):
        image = np.load(args.images_dir / f"{case}.npy")
        mask = np.load(args.masks_dir / f"{case}.npy") > 0
        split = split_of[case]
        modality = modality_of(case)

        # Image: modality-aware (MRA=percentile MIP, CTA=HU-window central slice).
        # Box target: always the full 3D vessel extent (mask MIP over the axis).
        cor = make_projection(image, "cor", modality, cta_window, cta_slab)  # [X, Z]
        cor_bb = bbox_from_mask(mask.max(axis=1))
        sag = make_projection(image, "sag", modality, cta_window, cta_slab)  # [Y, Z]
        sag_bb = bbox_from_mask(mask.max(axis=0))

        for tag, arr, bb in [("cor", cor, cor_bb), ("sag", sag, sag_bb)]:
            if bb is None:
                print(f"[skip] {case}_{tag}: empty vessel mask projection")
                continue
            fname = f"{case}_{tag}.png"
            save_png(arr, img_out / fname)
            rows_csv.append({
                "filepath": f"images/{fname}", "label": LABEL,
                "x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3], "split": split,
            })

    csv_path = args.out_dir / "annotations.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filepath", "label", "x1", "y1", "x2", "y2", "split"])
        w.writeheader()
        w.writerows(rows_csv)

    n_tr = sum(r["split"] == "train" for r in rows_csv)
    n_va = sum(r["split"] == "valid" for r in rows_csv)
    print(f"\nWrote {len(rows_csv)} boxes ({n_tr} train / {n_va} valid) + {csv_path}")
    print(f"Next: convert_annotations_to_coco.py --csv {csv_path} --images-root {args.out_dir} --out <coco>")


if __name__ == "__main__":
    main()
