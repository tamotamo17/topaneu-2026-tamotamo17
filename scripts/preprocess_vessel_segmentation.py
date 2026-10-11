#!/usr/bin/env python
"""Preprocess TopAneu 3D data into resampled ``.npy`` volumes for 2.5D training.

For each case the script:

1. loads the image (``images/<case>_0000.nii.gz``) and a label mask from one of
   ``vessel_masks`` / ``location_masks`` / ``type_masks`` (``--mask-source``);
2. converts the mask according to ``--label-mode``:
   - ``binary``     : any label ``> 0`` -> ``1`` (Step1 vessel region);
   - ``multiclass`` : labels kept as-is (Step2 vessels / Step3 aneurysm-location);
3. resamples both volumes to a target voxel spacing on **all three axes**
   (the Z spacing is the main knob to sweep when investigating preprocessing):
   - image : linear interpolation (``order=1``), intensity kept raw (``int16``);
   - mask  : nearest-neighbour (``order=0``) so labels are preserved (``uint8``);
4. optionally also resamples the multiclass vessel mask into ``aux/<case>.npy``
   (``--aux-mask vessel``) so it can be fed as an extra input channel for Step3;
5. saves one ``.npy`` per case for the image / mask / (aux), and appends a row to
   ``metadata.csv`` recording the original / resampled shape and spacing.

Volumes are stored in array-axis order (matching ``nibabel`` ``dataobj``); the
data is already reoriented to LPS+ upstream, so the axis order is consistent
across every case.

Usage (Step1 binary vessel — original behaviour, the defaults):
    PYTHONPATH=. uv run python scripts/preprocess_vessel_segmentation.py \
        --data-root data/raw/topaneu \
        --out-dir   data/vessel_lr \
        --target-spacing 0.625 0.625 0.625

Step2 (multiclass vessel, 36 classes):
    ... --mask-source vessel --label-mode multiclass \
        --out-dir .../multiclass_vessel/sp0.625

Step3 (aneurysm-location seg + vessel mask as aux channel):
    ... --mask-source location --label-mode multiclass --aux-mask vessel \
        --out-dir .../aneurysm_location/sp0.625

Quick test on a couple of cases:
    PYTHONPATH=. uv run python scripts/preprocess_vessel_segmentation.py \
        --data-root ... --out-dir ... --target-spacing 0.4 0.4 0.4 --limit 2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def _resample(arr: np.ndarray, src_spacing, dst_spacing, order: int) -> np.ndarray:
    """Resample ``arr`` from ``src_spacing`` to ``dst_spacing`` per axis.

    ``zoom_factor = src / dst`` (finer target spacing -> upsampling -> factor>1).
    """
    factors = tuple(float(s) / float(d) for s, d in zip(src_spacing, dst_spacing))
    return zoom(arr, factors, order=order)


def _load_mask(mask_path: Path, ref_spacing, case: str, ref_shape) -> np.ndarray:
    """Load a label mask, checking geometry against the reference image."""
    mask_nii = nib.load(str(mask_path))
    mask_spacing = tuple(float(z) for z in mask_nii.header.get_zooms()[:3])
    mask_labels = np.asarray(mask_nii.dataobj)
    if mask_labels.shape != ref_shape:
        raise ValueError(f"{case}: image shape {ref_shape} != mask shape {mask_labels.shape} ({mask_path.name})")
    if not np.allclose(ref_spacing, mask_spacing, atol=1e-3):
        raise ValueError(f"{case}: image spacing {ref_spacing} != mask spacing {mask_spacing} ({mask_path.name})")
    return mask_labels


def _process_case(
    case: str,
    image_path: Path,
    mask_path: Path,
    out_images: Path,
    out_masks: Path,
    target_spacing: tuple[float, float, float],
    label_mode: str,
    aux_mask_path: Path | None = None,
    out_aux: Path | None = None,
) -> dict:
    img_nii = nib.load(str(image_path))
    src_spacing = tuple(float(z) for z in img_nii.header.get_zooms()[:3])

    # Load lazily; image as float32 for interpolation, mask as raw labels.
    image = np.asarray(img_nii.dataobj).astype(np.float32)
    orig_shape = image.shape

    mask_labels = _load_mask(mask_path, src_spacing, case, orig_shape)

    # Image: linear interpolation, keep raw intensity range (cast back to int16).
    image_rs = _resample(image, src_spacing, target_spacing, order=1)
    image_rs = np.round(image_rs).astype(np.int16)

    # Mask: binarize (Step1) or keep labels (Step2/3), then nearest-neighbour.
    if label_mode == "binary":
        mask_src = (mask_labels > 0).astype(np.uint8)
    else:  # multiclass: preserve integer labels
        mask_src = np.asarray(mask_labels).astype(np.uint8)
    mask_rs = _resample(mask_src, src_spacing, target_spacing, order=0).astype(np.uint8)

    if image_rs.shape != mask_rs.shape:
        raise ValueError(f"{case}: resampled image {image_rs.shape} != mask {mask_rs.shape}")

    np.save(out_images / f"{case}.npy", image_rs)
    np.save(out_masks / f"{case}.npy", mask_rs)

    # Optional auxiliary vessel mask (multiclass labels) for Step3 conditioning.
    if aux_mask_path is not None and out_aux is not None:
        aux_labels = _load_mask(aux_mask_path, src_spacing, case, orig_shape)
        aux_rs = _resample(np.asarray(aux_labels).astype(np.uint8), src_spacing, target_spacing, order=0).astype(np.uint8)
        if aux_rs.shape != image_rs.shape:
            raise ValueError(f"{case}: resampled aux {aux_rs.shape} != image {image_rs.shape}")
        np.save(out_aux / f"{case}.npy", aux_rs)

    return {
        "case": case,
        "orig_shape": "x".join(map(str, orig_shape)),
        "orig_spacing": ",".join(f"{s:.4f}" for s in src_spacing),
        "new_shape": "x".join(map(str, image_rs.shape)),
        "new_spacing": ",".join(f"{s:.4f}" for s in target_spacing),
        "num_labels": int(mask_rs.max()),
        "fg_voxels": int((mask_rs > 0).sum()),
        "fg_ratio": float((mask_rs > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path, help="topaneu_deployment dir (has images/ and *_masks/)")
    parser.add_argument("--out-dir", required=True, type=Path, help="output dir (images/ and masks/ created inside)")
    parser.add_argument(
        "--target-spacing",
        required=True,
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="target voxel spacing in mm for the X Y Z array axes",
    )
    parser.add_argument(
        "--mask-source",
        choices=["vessel", "location", "type"],
        default="vessel",
        help="which *_masks folder to use as the target mask (default: vessel)",
    )
    parser.add_argument(
        "--label-mode",
        choices=["binary", "multiclass"],
        default="binary",
        help="binarize the mask (Step1) or keep integer labels (Step2/3) (default: binary)",
    )
    parser.add_argument(
        "--aux-mask",
        choices=["none", "vessel"],
        default="none",
        help="also export a resampled multiclass vessel mask to aux/ for Step3 conditioning",
    )
    parser.add_argument("--limit", type=int, default=None, help="process only the first N cases (for testing)")
    parser.add_argument("--overwrite", action="store_true", help="reprocess even if output .npy already exists")
    args = parser.parse_args()

    mask_dir_name = {"vessel": "vessel_masks", "location": "location_masks", "type": "type_masks"}[args.mask_source]
    images_dir = args.data_root / "images"
    masks_dir = args.data_root / mask_dir_name
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {masks_dir} to exist")

    aux_masks_dir = None
    out_aux = None
    if args.aux_mask == "vessel":
        aux_masks_dir = args.data_root / "vessel_masks"
        if not aux_masks_dir.is_dir():
            raise SystemExit(f"--aux-mask vessel requires {aux_masks_dir} to exist")
        out_aux = args.out_dir / "aux"
        out_aux.mkdir(parents=True, exist_ok=True)

    out_images = args.out_dir / "images"
    out_masks = args.out_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # Pair image (<case>_0000.nii.gz) with mask (<case>.nii.gz) by case id.
    image_files = sorted(images_dir.glob("*_0000.nii.gz"))
    cases = []
    for img_path in image_files:
        case = img_path.name.replace("_0000.nii.gz", "")
        mask_path = masks_dir / f"{case}.nii.gz"
        if not mask_path.exists():
            print(f"[skip] no {args.mask_source} mask for {case}")
            continue
        aux_path = (aux_masks_dir / f"{case}.nii.gz") if aux_masks_dir is not None else None
        if aux_path is not None and not aux_path.exists():
            print(f"[skip] no aux vessel mask for {case}")
            continue
        cases.append((case, img_path, mask_path, aux_path))

    if args.limit is not None:
        cases = cases[: args.limit]

    target_spacing = tuple(args.target_spacing)
    print(f"Found {len(cases)} cases | source={args.mask_source} | mode={args.label_mode} | aux={args.aux_mask}")
    print(f"Target spacing (X,Y,Z) = {target_spacing} mm | Output: {args.out_dir}")

    rows = []
    for case, img_path, mask_path, aux_path in tqdm(cases, desc="preprocess"):
        done = (out_images / f"{case}.npy").exists() and (out_masks / f"{case}.npy").exists()
        if out_aux is not None:
            done = done and (out_aux / f"{case}.npy").exists()
        if not args.overwrite and done:
            continue
        try:
            rows.append(
                _process_case(
                    case, img_path, mask_path, out_images, out_masks, target_spacing,
                    label_mode=args.label_mode, aux_mask_path=aux_path, out_aux=out_aux,
                )
            )
        except Exception as e:  # noqa: BLE001 - keep going, report at end
            print(f"[error] {case}: {e}")

    # Write / append metadata.
    meta_path = args.out_dir / "metadata.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        write_header = not meta_path.exists()
        with open(meta_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\nProcessed {len(rows)} cases -> {args.out_dir}")
        print(f"Metadata: {meta_path}")
    else:
        print("\nNo new cases processed.")


if __name__ == "__main__":
    main()
