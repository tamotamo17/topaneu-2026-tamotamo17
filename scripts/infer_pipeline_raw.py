#!/usr/bin/env python
"""Unified raw-NIfTI -> Task1/Task2 inference (submission pipeline, single fold).

Thin CLI over :func:`scripts.pipeline_core.run_pipeline` — the SAME function the
Docker submission containers call, so running this on a raw .nii.gz exercises the
exact code path that runs on Grand Challenge (only the file I/O differs: nibabel
here, SimpleITK there).

Chains every stage on one raw image, producing outputs in the ORIGINAL grid:
  1. load raw .nii.gz (nibabel) -> array + voxel spacing
  2. resample to 0.625 mm -> RF-DETR Step1 -> coronal/sagittal boxes -> 3D ROI
     -> ROI center (mm from corner)
  3. resample a 300^3 @0.4mm cube around the center directly from the raw volume
  4. Step2 2.5D vessel seg (37-class) on the cube
  5. Step3 3D aneurysm detector (image + predicted vessel) -> probability
  6. blobs -> anatomical location label (from predicted vessel) -> 0-52 crop mask
     (with --step4-*, each blob's sphere is replaced by the fine aneurysm shape)
  7. uncrop the crop mask back onto the original grid
  -> Task1 = sorted unique labels ; Task2 = original-grid uint8 mask

Run with a single GPU visible so every torch model + RF-DETR share cuda:0:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. uv run python scripts/infer_pipeline_raw.py \
        --image .../topaneu_deployment/images/<case>_0000.nii.gz --case <case> \
        --step1-config ... --step1-ckpt ... \
        --step2-config ... --step2-ckpt ... \
        --step3-config ... --step3-ckpt ... \
        --out-mask <case>_pred.npy --out-json <case>_pred.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from src.detection.vessel_roi_preproc import modality_of
from scripts.pipeline_core import run_pipeline

DEV = "cuda:0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--case", required=True, help="case id (for modality inference)")
    ap.add_argument("--step1-config", required=True, type=Path)
    ap.add_argument("--step1-ckpt", required=True, type=Path)
    ap.add_argument("--step2-config", required=True, type=Path)
    ap.add_argument("--step2-ckpt", required=True, type=Path)
    ap.add_argument("--step3-config", required=True, type=Path)
    ap.add_argument("--step3-ckpt", required=True, type=Path)
    ap.add_argument("--step4-config", type=Path, default=None,
                    help="fine aneurysm-shape seg (e.g. configs/segmentation/step4_fineseg_fold0.yaml); "
                         "without it the detector sphere is painted as-is")
    ap.add_argument("--step4-ckpt", type=Path, default=None)
    ap.add_argument("--step4-thr", type=float, default=0.5)
    ap.add_argument("--aux-rescore-weight", type=float, default=None,
                    help="enable aux-presence-head rescoring of blobs (needs a "
                         "monai_segresnet3d_auxcls detector)")
    ap.add_argument("--aux-rescore-thr", type=float, default=0.5)
    ap.add_argument("--fineseg-reject-thr", type=float, default=None,
                    help="let Step4 veto blobs below this combined score")
    ap.add_argument("--fineseg-reject-weight", type=float, default=0.75)
    ap.add_argument("--label-topk", type=int, default=1)
    ap.add_argument("--half-mm", type=float, default=60.0)
    ap.add_argument("--target-sp", type=float, default=0.4)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--det-thr", type=float, default=0.5)
    ap.add_argument("--min-vox", type=int, default=30)
    ap.add_argument("--r-vox", type=int, default=18)
    ap.add_argument("--nb-frac", type=float, default=0.05)
    ap.add_argument("--step1-thr", type=float, default=0.3)
    ap.add_argument("--step1-margin", type=int, default=8)
    ap.add_argument("--out-mask", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()
    if bool(args.step4_config) != bool(args.step4_ckpt):
        ap.error("--step4-config and --step4-ckpt must be given together")

    modality = modality_of(args.case)
    nii = nib.load(str(args.image))
    vol = np.asarray(nii.get_fdata(), dtype=np.float32)
    orig_sp = tuple(float(z) for z in nii.header.get_zooms()[:3])

    orig_mask, locs, st = run_pipeline(
        vol, orig_sp, modality,
        step1_config=args.step1_config, step1_ckpt=args.step1_ckpt,
        step2_config=args.step2_config, step2_ckpt=args.step2_ckpt,
        step3_config=args.step3_config, step3_ckpt=args.step3_ckpt,
        step4_config=args.step4_config, step4_ckpt=args.step4_ckpt,
        step4_thr=args.step4_thr,
        aux_rescore_weight=args.aux_rescore_weight, aux_rescore_thr=args.aux_rescore_thr,
        fineseg_reject_thr=args.fineseg_reject_thr,
        fineseg_reject_weight=args.fineseg_reject_weight,
        label_topk=args.label_topk,
        device=DEV, half_mm=args.half_mm, target_sp=args.target_sp, size=args.size,
        det_thr=args.det_thr, min_vox=args.min_vox, r_vox=args.r_vox, nb_frac=args.nb_frac,
        step1_thr=args.step1_thr, step1_margin=args.step1_margin,
        log=lambda m: print(f"  [{args.case}] {m}", flush=True))

    args.out_mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_mask, orig_mask)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump([int(x) for x in locs], open(args.out_json, "w"))
    print(f"[{args.case}] modality={modality} blobs={st['blobs']} locs={locs} "
          f"orig_shape={orig_mask.shape}")
    if args.step4_config:
        print(f"  step4: refined {st['refined']}/{st['blobs']} blobs "
              f"(sphere-fallback {st['fallback']}), voxels {st['sphere_vox']} -> {st['refined_vox']}")
    if args.fineseg_reject_thr is not None:
        print(f"  fineseg rejection (w={args.fineseg_reject_weight}, "
              f"thr={args.fineseg_reject_thr}): rejected {st['fs_rejected']}")
    if args.aux_rescore_weight is not None:
        print(f"  aux rescoring (w={args.aux_rescore_weight}, thr={args.aux_rescore_thr}): "
              f"kept {st['aux_kept']}, rejected {st['aux_rejected']}")


if __name__ == "__main__":
    main()
