#!/usr/bin/env python
"""End-to-end fold-val baseline on the OFFICIAL metric: detect -> anatomical
label -> (optional Step4 fine-shape seg) -> labelled mask -> eval_topaneu.

For each val case: run the 3D detector, extract blob connected components,
label each blob with a 50-location id from the predicted vessel map (approach A,
`anatomical_label.label_blob`), paint each blob's voxels with its location label
to build a 0-50 mask, then score against the GT location mask with the faithful
TopAneu evaluator.

With `--step4-config/--step4-ckpt` the painted voxels are the Step4 fine-shape
prediction around each blob instead of the (~33x oversized) detector sphere,
which is what Task2's DSC/VS/HD95 actually measure. Task1 is unaffected by
definition — refinement changes shapes, not the reported location set.

`--model CFG CKPT` may be repeated to score a detector ENSEMBLE, fused with
`--combine mean|max` exactly the way `pipeline_core.run_pipeline` fuses
`step3_specs`. Without this the submission pipeline could run an ensemble that
no evaluation script could score. `--config/--ckpt` remain as the single-model
shorthand.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from scripts.infer_segmentation_2_5d import load_model
from scripts.froc_aneurysm_3d import predict_proba_3d
from scripts.eval_topaneu import task1_sample, task2_sample, aggregate, average
from src.segmentation.inference.aux_score import make_aux_score_fn
from src.segmentation.inference.blob_mask import build_fineseg, paint_blobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="single-model shorthand for --model")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--model", nargs=2, action="append", metavar=("CONFIG", "CKPT"), default=[],
                    help="detector to include; repeat for an ensemble")
    ap.add_argument("--combine", choices=["mean", "max"], default="mean",
                    help="ensemble fusion of the per-model probability maps (ignored for one model)")
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--aux-dir", required=True, type=Path, help="predicted vessel seg (detector aux + labeller)")
    ap.add_argument("--gt-dir", required=True, type=Path, help="GT location masks (0-50)")
    ap.add_argument("--fold-yaml", required=True, type=Path)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5],
                    help="blob extraction threshold(s). Several values are swept in ONE inference "
                         "pass: the probability map does not depend on the threshold, so it is "
                         "computed once per case and reused. Matters for ensembles — an 8-model "
                         "pass costs ~50 min, so a 4-point sweep this way saves ~2.5 h.")
    ap.add_argument("--min-vox", type=int, default=30)
    ap.add_argument("--r-vox", type=int, default=18)
    ap.add_argument("--nb-frac", type=float, default=0.05)
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--step4-config", type=Path, default=None,
                    help="fine-shape seg config (e.g. configs/segmentation/step4_fineseg_fold0.yaml); "
                         "without it the detector sphere is painted as-is")
    ap.add_argument("--step4-ckpt", type=Path, default=None)
    ap.add_argument("--step4-model", nargs=2, action="append", metavar=("CONFIG", "CKPT"), default=[],
                    help="repeatable: score a Step4 ENSEMBLE, fused by mean probability exactly "
                         "the way pipeline_core fuses `step4_specs`. Without this the submission "
                         "path could ship a Step4 ensemble that no evaluation script can score. "
                         "--step4-config/--step4-ckpt remain as the single-model shorthand.")
    ap.add_argument("--step4-thr", type=float, default=0.5)
    ap.add_argument("--fineseg-reject-thr", type=float, default=None,
                    help="drop a blob when peak^(1-w)*fs_peak^w < this (needs --step4-*). "
                         "Inverts the default 'refinement never deletes a detection' policy. "
                         "Only meaningful with a fine-seg trained on a neutral loss.")
    ap.add_argument("--fineseg-reject-weight", type=float, default=0.75)
    ap.add_argument("--label-topk", type=int, default=1,
                    help="report this many candidate locations per blob in the Task1 list "
                         "(mask/Task2 always uses the top-1). The rule labeller is ~52%% "
                         "accurate and confuses adjacent ICA-C7 subdivisions, so the truth "
                         "is often the runner-up.")
    ap.add_argument("--aux-rescore-weight", type=float, default=None,
                    help="enable two-stage blob rescoring with the detector's aux presence head: "
                         "keep a blob iff peak^(1-w)*aux^w >= --aux-rescore-thr. Use a LOW --thr "
                         "(e.g. 0.20) so the candidate pool is generous. Single detector only.")
    ap.add_argument("--aux-rescore-thr", type=float, default=0.5)
    ap.add_argument("--label-no-position", action="store_true",
                    help="disable every positional rule in the labeller (reproduces the "
                         "neighbour-set-only baseline: matched-blob 0.644, Task1 MCC 0.2025)")
    ap.add_argument("--label-no-c6", action="store_true",
                    help="keep the M1 positional rule but drop the ICA C6-nonOA one, which "
                         "captures 12/14 C6 on train at the cost of 6/24 infraclinoid")
    ap.add_argument("--label-no-branch-dist", action="store_true",
                    help="keep the M1/C6 positional rules but judge ICA branches and "
                         "BA/VA junctions by neighbour-set membership again (reproduces "
                         "the Task1 MCC 0.2390 measurement)")
    ap.add_argument("--label-drop-weak", action="store_true",
                    help="drop the three rules that contribute EXACTLY ZERO on the external "
                         "folds (AChA branch, VA-BA junction, A2). Measured on fold1+2 val with "
                         "out-of-fold vessel maps: each +0.0000, while fold0 showed +0.0123/"
                         "+0.0123/+0.0000 — i.e. their apparent value was fold0 overfitting. "
                         "Cuts the distance-threshold count from 9 to 6.")
    ap.add_argument("--relabel-from-refined", action="store_true",
                    help="re-run the anatomical labeller on the Step4-refined mask's centroid "
                         "instead of the detector sphere's (needs --step4-*). Without it Task1 "
                         "is invariant to Step4 by construction.")
    args = ap.parse_args()
    if bool(args.step4_config) != bool(args.step4_ckpt):
        ap.error("--step4-config and --step4-ckpt must be given together")
    specs = [(Path(c), Path(k)) for c, k in args.model]
    if bool(args.config) != bool(args.ckpt):
        ap.error("--config and --ckpt must be given together")
    if args.config:
        specs.insert(0, (args.config, args.ckpt))
    if not specs:
        ap.error("give at least one detector via --model CONFIG CKPT (or --config/--ckpt)")

    # Labeller ablation. Set on the module rather than threaded through
    # paint_blobs -> label_blob, so the call chain stays as it ships.
    import scripts.anatomical_label as _AL
    if args.label_no_position:
        _AL.POSITION_ENABLED = False
    if args.label_no_c6:
        _AL.C6_NONOA_C1C5_MM = -1.0
    if args.label_no_branch_dist:
        _AL.BRANCH_BY_DISTANCE = False
    if args.label_drop_weak:
        _AL.ACHA_JUNC_MM = -1.0
        _AL.VA_BA_JUNC_MM = -1.0
        _AL.A2_ACOM_MM = 9999.0
    print("labeller: positional rules "
          + ("OFF (neighbour-set only)" if args.label_no_position
             else "ON, C6 rule off" if args.label_no_c6 else "ON")
          + (", branch tests by neighbour-set" if args.label_no_branch_dist else "")
          + (", weak rules (AChA/VA-BA/A2) dropped" if args.label_drop_weak else ""))

    dev = f"cuda:{args.gpu}"
    # Each detector is loaded once and kept, so every case runs through all of
    # them in a single pass — the same shape as pipeline_core's step3_specs loop.
    models = []
    for cfg_path, ckpt_path in specs:
        cfg = OmegaConf.load(cfg_path)
        ds, lm = cfg.dataset, cfg.lightning_module
        pk = dict(lo_p=float(ds.get("lower_percentile", 0.5)), hi_p=float(ds.get("upper_percentile", 99.5)),
                  aux_mode=str(ds.get("aux_mode", "label")), aux_max=float(ds.get("aux_max", 36.0)),
                  aux_encoding=str(ds.get("aux_encoding", "scalar")), n_vessel=int(ds.get("n_vessel", 37)),
                  roi=(int(lm.get("roi_size", 96)),) * 3, sw_batch=int(lm.get("sw_batch_size", 2)),
                  overlap=float(lm.get("overlap", 0.5)), device=dev, proba_class=1,
                  activation="sigmoid" if int(cfg.model.get("num_classes", 2)) == 1 else "softmax")
        models.append((load_model(cfg, ckpt_path).to(dev).eval(), pk))
        print(f"detector: {cfg_path.name} <- {ckpt_path.name}")
    if len(models) > 1:
        print(f"ensemble fusion: {args.combine} over {len(models)} detectors")
    fuse = np.max if args.combine == "max" else np.mean

    if args.aux_rescore_weight is not None and len(models) != 1:
        ap.error("--aux-rescore-weight needs exactly one detector (the aux head is per-model)")
    if args.aux_rescore_weight is not None and not hasattr(models[0][0], "cls_head"):
        ap.error(f"--aux-rescore-weight needs a detector with an aux head "
                 f"(monai_segresnet3d_auxcls); got {type(models[0][0]).__name__}")

    step4_specs = [(Path(c), Path(k)) for c, k in args.step4_model]
    if args.step4_config:
        step4_specs.insert(0, (args.step4_config, args.step4_ckpt))
    fineseg = build_fineseg(step4_specs, device=dev, thr=args.step4_thr)
    if fineseg is None:
        print("Step4 fine-shape seg: OFF (sphere blobs)")
    else:
        print(f"Step4 fine-shape seg: ON  " + ", ".join(Path(k).name for _, k in step4_specs)
              + (f"  (ensemble of {len(step4_specs)}, mean fusion)" if len(step4_specs) > 1 else ""))

    def make_aux_fn(img, vessel):
        """Aux presence probability on a detector-patch CENTRED on a blob.

        Mirrors training, where positives are patches centred on foreground —
        so the blob-centred crop is in-distribution for the head. Shared with
        the submission path via ``src.segmentation.inference.aux_score``.
        """
        model0, kw = models[0]
        return make_aux_score_fn(model0, img, vessel, roi=kw["roi"][0],
                                 lo_p=kw["lo_p"], hi_p=kw["hi_p"],
                                 aux_max=kw["aux_max"], device=kw["device"])

    val = [r["id"] for r in OmegaConf.load(args.fold_yaml).val]
    thrs = list(args.thr)
    s1 = {t: [] for t in thrs}
    s2 = {t: [] for t in thrs}
    totals = {t: {} for t in thrs}
    for cid in tqdm(val, desc="infer+label"):
        img = np.load(args.images_dir / f"{cid}.npy")
        vessel = np.load(args.aux_dir / f"{cid}.npy")
        probas = [predict_proba_3d(m, img, vessel, **kw) for m, kw in models]
        proba = probas[0] if len(probas) == 1 else fuse(probas, axis=0)
        gt = np.load(args.gt_dir / f"{cid}.npy").astype(np.int64)
        gt_locs = np.unique(gt[gt > 0]).tolist()
        aux_fn = make_aux_fn(img, vessel) if args.aux_rescore_weight is not None else None
        for t in thrs:
            pred_mask, pred_locs, st = paint_blobs(
                proba, vessel, thr=t, min_vox=args.min_vox, r_vox=args.r_vox,
                nb_frac=args.nb_frac, image=img, fineseg=fineseg,
                relabel_from_refined=args.relabel_from_refined,
                aux_score_fn=aux_fn,
                aux_weight=(args.aux_rescore_weight or 0.75), aux_thr=args.aux_rescore_thr,
                label_topk=args.label_topk,
                fineseg_reject_thr=args.fineseg_reject_thr,
                fineseg_reject_weight=args.fineseg_reject_weight)
            tot = totals[t]
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
            s1[t].append(task1_sample(pred_locs, gt_locs))
            s2[t].append(task2_sample(pred_mask.astype(np.int64), gt))
            if args.save_dir:
                d = args.save_dir if len(thrs) == 1 else args.save_dir / f"thr{t:g}"
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"{cid}.npy", pred_mask)

    for t in thrs:
        report(t, s1[t], s2[t], totals[t], args, len(val), len(thrs) > 1)


def report(thr, s1, s2, tot, args, n_val, show_thr):
    if show_thr:
        print(f"\n########## thr = {thr:g} ##########")
    print(f"\nblobs painted: {tot.get('blobs', 0)} "
          f"(refined {tot.get('refined', 0)}, sphere-fallback {tot.get('fallback', 0)}) | "
          f"voxels sphere {tot.get('sphere_vox', 0)} -> painted {tot.get('refined_vox', 0)}")
    if args.fineseg_reject_thr is not None:
        print(f"fineseg rejection (w={args.fineseg_reject_weight}, "
              f"thr={args.fineseg_reject_thr}): rejected {tot.get('fs_rejected', 0)}")
    if args.label_topk > 1:
        print(f"label top-{args.label_topk}: {tot.get('extra_labels', 0)} extra candidate "
              f"locations added to the Task1 list")
    if args.aux_rescore_weight is not None:
        print(f"aux rescoring (w={args.aux_rescore_weight}, thr={args.aux_rescore_thr}): "
              f"kept {tot.get('aux_kept', 0)}, rejected {tot.get('aux_rejected', 0)}")
    if args.relabel_from_refined:
        print(f"relabel from refined centroid: changed {tot.get('relabel_changed', 0)}, "
              f"unchanged {tot.get('relabel_same', 0)}, no-vessel(kept original) "
              f"{tot.get('relabel_none', 0)}")

    for task, samples, seg in [(1, s1, False), (2, s2, True)]:
        avg = average(aggregate(samples, seg), seg)
        print(f"\n=== OFFICIAL Task{task} ({n_val} val cases) ===")
        for k, v in avg.items():
            print(f"  {k:<10} {v:.4f}")


if __name__ == "__main__":
    main()
