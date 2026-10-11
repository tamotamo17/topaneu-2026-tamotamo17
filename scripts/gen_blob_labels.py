#!/usr/bin/env python
"""Stage-1 of the detect->classify pipeline: turn a detector's PREDICTED blobs
into labelled training samples for the 50-location classifier.

For every case we run the (class-agnostic) 3D aneurysm detector, extract
connected-component blob centroids from its probability map, and match each blob
to a ground-truth aneurysm (connected component of the location mask, whose value
is the location label 1..50) by centroid distance. A matched blob gets that
location label; an unmatched blob (false positive) gets label 0 (reject). This
keeps the classifier trained on the SAME predicted-blob distribution it will see
at inference (incl. FPs to reject).

Output: <out> manifest.json  = {case: {"split","blobs":[{c,size,label}],"gt":[{c,label}]}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import label as cc_label, center_of_mass
from tqdm import tqdm

from scripts.infer_segmentation_2_5d import load_model
from scripts.froc_aneurysm_3d import predict_proba_3d


def gt_components(mask: np.ndarray):
    """Return [(centroid_zyx, location_label)] for each aneurysm CC in a location mask."""
    out = []
    lab, n = cc_label(mask > 0)
    for k in range(1, n + 1):
        m = lab == k
        loc = int(np.bincount(mask[m].ravel()).argmax())  # majority location label in the CC
        out.append((np.array(center_of_mass(m)), loc))
    return out


def pred_blobs(proba: np.ndarray, thr: float, min_vox: int):
    out = []
    lab, n = cc_label(proba >= thr)
    if n == 0:
        return out
    sizes = np.bincount(lab.ravel())
    for k in range(1, n + 1):
        if sizes[k] >= min_vox:
            out.append((np.array(center_of_mass(lab == k)), int(sizes[k])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="detector config")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--aux-dir", required=True, type=Path, help="predicted vessel seg (Step2)")
    ap.add_argument("--gt-dir", required=True, type=Path, help="location masks (0-50)")
    ap.add_argument("--fold-yaml", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-vox", type=int, default=30)
    ap.add_argument("--tol-mm", type=float, default=5.0)
    ap.add_argument("--spacing", type=float, default=0.4)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    dev = f"cuda:{args.gpu}"
    model = load_model(cfg, args.ckpt).to(dev).eval()
    ds, lm = cfg.dataset, cfg.lightning_module
    pk = dict(lo_p=float(ds.get("lower_percentile", 0.5)), hi_p=float(ds.get("upper_percentile", 99.5)),
              aux_mode=str(ds.get("aux_mode", "label")), aux_max=float(ds.get("aux_max", 36.0)),
              aux_encoding=str(ds.get("aux_encoding", "scalar")), n_vessel=int(ds.get("n_vessel", 37)),
              roi=(int(lm.get("roi_size", 96)),) * 3, sw_batch=int(lm.get("sw_batch_size", 2)),
              overlap=float(lm.get("overlap", 0.5)), device=dev, proba_class=1,
              activation="sigmoid" if int(cfg.model.get("num_classes", 2)) == 1 else "softmax")

    doc = OmegaConf.load(args.fold_yaml)
    split = {r["id"]: "train" for r in doc.train}
    split.update({r["id"]: "val" for r in doc.val})
    tolv = args.tol_mm / args.spacing

    manifest = {}
    for cid in tqdm(sorted(split), desc="blobs"):
        img = np.load(args.images_dir / f"{cid}.npy")
        aux = np.load(args.aux_dir / f"{cid}.npy")
        proba = predict_proba_3d(model, img, aux, **pk)
        blobs = pred_blobs(proba, args.thr, args.min_vox)
        gts = gt_components(np.load(args.gt_dir / f"{cid}.npy"))
        used = set()
        blob_recs = []
        for c, sz in blobs:
            best_d, best_i = 1e9, None
            for i, (gc, _) in enumerate(gts):
                if i in used:
                    continue
                d = np.linalg.norm(c - gc)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None and best_d <= tolv:
                used.add(best_i); lbl = gts[best_i][1]
            else:
                lbl = 0
            blob_recs.append({"c": [float(x) for x in c], "size": sz, "label": lbl})
        manifest[cid] = {
            "split": split[cid],
            "blobs": blob_recs,
            "gt": [{"c": [float(x) for x in gc], "label": lbl} for gc, lbl in gts],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(args.out, "w"))

    # summary
    from collections import Counter
    tr = Counter(); va = Counter(); reject = Counter()
    for cid, m in manifest.items():
        for b in m["blobs"]:
            (tr if m["split"] == "train" else va)[b["label"]] += 1
            if b["label"] == 0:
                reject[m["split"]] += 1
    print(f"\nmanifest -> {args.out}")
    print(f"train blobs: {sum(tr.values())} (reject={reject['train']}), val blobs: {sum(va.values())} (reject={reject['val']})")
    print("train label distribution (label:count):")
    print({k: tr[k] for k in sorted(tr)})


if __name__ == "__main__":
    main()
