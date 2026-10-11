#!/usr/bin/env python
"""Rasterize aneurysm-center SPHERE targets from location masks (RSNA-1st style).

Instead of the tiny, hard-to-learn raw aneurysm voxels, the detection target is a
fixed-radius ball around each aneurysm's centroid. This gives a larger, rounder,
easier-to-learn positive region; the predicted blob's centroid is the detected
aneurysm coordinate. Radius is given in mm and converted with the isotropic
voxel spacing.

Input : <in-dir> of location-mask .npy volumes (labels >0 = aneurysm, any class)
Output: <out-dir>/<case>.npy  uint8 binary sphere mask (same shape)

Usage:
    PYTHONPATH=. uv run python scripts/make_sphere_targets.py \
        --masks-dir data/aneurysm_hr/masks \
        --out-dir   data/aneurysm_hr/sphere_g25 \
        --spacing 0.4 --radius-mm 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import label, center_of_mass
from tqdm import tqdm


def rasterize_spheres(shape, centers, r_vox: float) -> np.ndarray:
    """Filled binary ball (1 inside radius, 0 outside)."""
    out = np.zeros(shape, dtype=np.uint8)
    r = int(np.ceil(r_vox))
    r2 = r_vox * r_vox
    for cz, cy, cx in centers:
        z0, z1 = max(0, int(cz) - r), min(shape[0], int(cz) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(shape[1], int(cy) + r + 1)
        x0, x1 = max(0, int(cx) - r), min(shape[2], int(cx) + r + 1)
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        ball = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r2
        out[z0:z1, y0:y1, x0:x1][ball] = 1
    return out


def rasterize_gaussian(shape, centers, sigma_vox: float, trunc: float = 3.0) -> np.ndarray:
    """Center-peaked Gaussian heatmap: exp(-d^2/(2 sigma^2)), peak 1 at each center,
    max-combined across centers. Continuous [0,1] (regression target)."""
    out = np.zeros(shape, dtype=np.float32)
    r = int(np.ceil(trunc * sigma_vox))
    two_s2 = 2.0 * sigma_vox * sigma_vox
    for cz, cy, cx in centers:
        z0, z1 = max(0, int(cz) - r), min(shape[0], int(cz) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(shape[1], int(cy) + r + 1)
        x0, x1 = max(0, int(cx) - r), min(shape[2], int(cx) + r + 1)
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        g = np.exp(-d2 / two_s2).astype(np.float32)
        np.maximum(out[z0:z1, y0:y1, x0:x1], g, out=out[z0:z1, y0:y1, x0:x1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks-dir", required=True, type=Path, help="location-mask .npy volumes")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--spacing", type=float, default=0.4, help="isotropic voxel spacing (mm)")
    ap.add_argument("--radius-mm", type=float, default=4.0, help="ball radius (mm) [mode=ball]")
    ap.add_argument("--mode", choices=["ball", "gaussian"], default="ball",
                    help="ball = filled binary sphere; gaussian = center-peaked heatmap [0,1]")
    ap.add_argument("--sigma-mm", type=float, default=2.5, help="Gaussian sigma (mm) [mode=gaussian]")
    ap.add_argument("--min-comp-vox", type=int, default=1, help="ignore GT components smaller than this")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.masks_dir.glob("*.npy"))
    if args.mode == "ball":
        r_vox = args.radius_mm / args.spacing
        print(f"{len(files)} cases | mode=ball radius={args.radius_mm}mm = {r_vox:.1f}vox @ {args.spacing}mm")
    else:
        s_vox = args.sigma_mm / args.spacing
        print(f"{len(files)} cases | mode=gaussian sigma={args.sigma_mm}mm = {s_vox:.1f}vox @ {args.spacing}mm")

    n_sph = 0
    for f in tqdm(files, desc="spheres"):
        m = np.load(f) > 0
        centers = []
        if m.any():
            lab, n = label(m)
            sizes = np.bincount(lab.ravel())
            for k in range(1, n + 1):
                if sizes[k] >= args.min_comp_vox:
                    centers.append(center_of_mass(lab == k))
        if args.mode == "ball":
            sph = rasterize_spheres(m.shape, centers, args.radius_mm / args.spacing)
        else:
            sph = rasterize_gaussian(m.shape, centers, args.sigma_mm / args.spacing)
        n_sph += len(centers)
        np.save(args.out_dir / f.name, sph)

    print(f"Wrote {len(files)} target masks ({n_sph} aneurysm centers, mode={args.mode}) -> {args.out_dir}")


if __name__ == "__main__":
    main()
