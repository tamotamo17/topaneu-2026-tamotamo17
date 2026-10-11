#!/usr/bin/env python
"""Faithful local port of the official TopAneu-26 evaluator.

Reproduces github.com/Bangulli/TopAneu-26 `eval/task1/evaluate.py` and the
ACTIVE volume-level `eval/task2/evaluate.py` (the instance-level function there
is commented out and NOT used). Lets us score our pipeline on the real metric
(per-location Precision / Recall / MCC, plus DSC / VolSim / HD95 for Task 2)
instead of the coordinate-FROC proxy.

Metric details replicated exactly:
  * N classes (1..N); background 0 ignored. Everything macro-averaged /N.
    N comes from ``location_mapping.json`` of the active release (50 in the
    original deployment, 52 after the 2026-08 MCA re-split), so switching the
    dataset switches the denominator. Override with $TOPANEU_NCLS if needed.
  * eps = 1e-6 in every denominator; MCC's eps is added OUTSIDE the sqrt.
  * Task1: per-class counts TP=min(pred,gt), FP=max(0,pred-gt), FN=max(0,gt-pred),
    TN = n_aneu-(tp+fn) with n_aneu = sum of GT label counts in that sample.
  * Task2 (volume level): per-class binary presence. is_tp = class in BOTH gt&pred
    AND their binary masks intersect (>0 voxels). tp/fp/fn in {0,1};
    TN = n_aneu-(tp+fn) with n_aneu = #distinct nonzero GT classes in the sample.
    DICE/HD95/VOLSIM computed iff class in pred OR gt (else 0); global per-class =
    sum / (TP+FN+FP+eps). FP/FN-only -> DSC=0, VS=0, HD95=1 (normalized).
  * TP/FP/FN/TN summed across samples BEFORE computing Prec/Rec/MCC (micro per class).

Usage (local, on .npy label volumes 0..50, pred.shape must == gt.shape):
    # Task 2 (labeled masks)
    PYTHONPATH=. uv run python scripts/eval_topaneu.py --task 2 \
        --pred-dir <dir of pred *.npy 0-50> --gt-dir <dir of gt *.npy 0-50> \
        --fold-yaml <.../fold0.yaml>
    # Task 1 (location lists derived from the same labeled masks, or --pred-json-dir)
    PYTHONPATH=. uv run python scripts/eval_topaneu.py --task 1 \
        --pred-dir <pred masks> --gt-dir <gt masks> --fold-yaml <fold0.yaml>
    # sanity: reproduce the published simulation points
    PYTHONPATH=. uv run python scripts/eval_topaneu.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

EPS = 1e-6


_DEFAULT_DEP = Path(__file__).resolve().parents[1] / "docker" / "mapping"   # bundled id mapping


def _default_ncls() -> int:
    """Number of location classes = max id in the active release's mapping.

    Hardcoding 50 silently mis-scores the 52-class release (its ids 47-52 were
    renumbered), so read it from the data and let $TOPANEU_NCLS force a value.
    """
    if os.environ.get("TOPANEU_NCLS"):
        return int(os.environ["TOPANEU_NCLS"])
    mapping = Path(os.environ.get("TOPANEU_DEPLOYMENT", _DEFAULT_DEP)) / "location_mapping.json"
    try:
        return max(v for v in json.load(open(mapping))["labels"].values() if v > 0)
    except Exception:
        return 50


NCLS = _default_ncls()


# ----------------------------- segmentation primitives -----------------------------
def dice_fn(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    return float((2 * np.bitwise_and(a, b).sum()) / (a.sum() + b.sum() + EPS))


def volsim_fn(a: np.ndarray, b: np.ndarray) -> float:
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    return float(1 - (abs(int(a.sum()) - int(b.sum())) / (a.sum() + b.sum() + EPS)))


def _surface(img: np.ndarray) -> np.ndarray:
    return img & ~binary_erosion(img)


def hd_fn(a: np.ndarray, b: np.ndarray, perc: int = 95, norm: bool = True) -> float:
    diag = float(np.linalg.norm(a.shape))
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return 1.0 if norm else diag
    ca = np.argwhere(_surface(a))
    cb = np.argwhere(_surface(b))
    d_ab, _ = cKDTree(cb).query(ca)
    d_ba, _ = cKDTree(ca).query(cb)
    hd = max(np.percentile(d_ab, perc), np.percentile(d_ba, perc))
    return float(hd / diag if norm else hd)


def check_tp_intersection(gt_c: np.ndarray, pr_c: np.ndarray) -> bool:
    return bool(np.bitwise_and(gt_c, pr_c).sum() > 0)


# ----------------------------- per-sample confusion -----------------------------
def task1_sample(pred_locs: list[int], gt_locs: list[int]) -> dict:
    def counts(locs):
        r = {i: 0 for i in range(1, NCLS + 1)}
        for l in locs:
            if 1 <= int(l) <= NCLS:
                r[int(l)] += 1
        return r

    p, g = counts(pred_locs), counts(gt_locs)
    n_aneu = sum(g.values())
    out = {}
    for c in range(1, NCLS + 1):
        tp = min(p[c], g[c]); fp = max(0, p[c] - g[c]); fn = max(0, g[c] - p[c])
        out[f"TP_{c}"], out[f"FP_{c}"], out[f"FN_{c}"], out[f"TN_{c}"] = tp, fp, fn, n_aneu - (tp + fn)
    return out


def task2_sample(pred: np.ndarray, gt: np.ndarray) -> dict:
    assert pred.shape == gt.shape, f"shape mismatch pred={pred.shape} gt={gt.shape}"
    gts = [v for v in np.unique(gt).tolist() if v != 0]
    n_aneu = len(gts)
    out = {}
    for c in range(1, NCLS + 1):
        gc = gt == c; pc = pred == c
        in_g, in_p = bool(gc.any()), bool(pc.any())
        is_tp = check_tp_intersection(gc, pc) if (in_g and in_p) else False
        tp = 1 if is_tp else 0
        fp = 1 if (in_p and not is_tp) else 0
        fn = 1 if (in_g and not is_tp) else 0
        out[f"TP_{c}"], out[f"FP_{c}"], out[f"FN_{c}"], out[f"TN_{c}"] = tp, fp, fn, n_aneu - (tp + fn)
        if in_p or in_g:
            out[f"DICE_{c}"] = dice_fn(gc, pc)
            out[f"HD95_{c}"] = hd_fn(gc, pc, 95, True)
            out[f"VOLSIM_{c}"] = volsim_fn(gc, pc)
        else:
            out[f"DICE_{c}"] = out[f"HD95_{c}"] = out[f"VOLSIM_{c}"] = 0.0
    return out


# ----------------------------- aggregation -----------------------------
def aggregate(samples: list[dict], seg: bool) -> dict:
    agg = {k: sum(s[k] for s in samples) for k in samples[0].keys()}
    for i in range(1, NCLS + 1):
        tp, fp, fn, tn = agg[f"TP_{i}"], agg[f"FP_{i}"], agg[f"FN_{i}"], agg[f"TN_{i}"]
        agg[f"PRECISION_{i}"] = tp / (tp + fp + EPS)
        agg[f"RECALL_{i}"] = tp / (tp + fn + EPS)
        num = tp * tn - fn * fp
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + EPS
        agg[f"MCC_{i}"] = num / den
        if seg:
            norm = tp + fn + fp + EPS
            agg[f"DICE_{i}"] /= norm
            agg[f"HD95_{i}"] /= norm
            agg[f"VOLSIM_{i}"] /= norm
    return agg


def average(agg: dict, seg: bool) -> dict:
    keys = ["PRECISION", "RECALL", "MCC"] + (["DICE", "HD95", "VOLSIM"] if seg else [])
    return {k: sum(agg[f"{k}_{i}"] for i in range(1, NCLS + 1)) / NCLS for k in keys}


# ----------------------------- drivers -----------------------------
def _ids(fold_yaml: Path | None, pred_dir: Path) -> list[str]:
    if fold_yaml:
        from omegaconf import OmegaConf
        return [r["id"] for r in OmegaConf.load(fold_yaml).val]
    return sorted(p.stem for p in pred_dir.glob("*.npy"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, choices=[1, 2])
    ap.add_argument("--pred-dir", type=Path, help="predicted label .npy volumes (0-50)")
    ap.add_argument("--gt-dir", type=Path, help="GT label .npy volumes (0-50)")
    ap.add_argument("--gt-json-dir", type=Path, default=None, help="Task1 GT location_jsons dir (else derive from gt masks)")
    ap.add_argument("--fold-yaml", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    ids = _ids(args.fold_yaml, args.pred_dir)
    seg = args.task == 2
    samples = []
    for cid in ids:
        pred = np.load(args.pred_dir / f"{cid}.npy")
        gt = np.load(args.gt_dir / f"{cid}.npy")
        if args.task == 2:
            samples.append(task2_sample(pred.astype(np.int64), gt.astype(np.int64)))
        else:
            pl = np.unique(pred[pred > 0]).tolist()
            if args.gt_json_dir:
                gl = json.load(open(args.gt_json_dir / f"{cid}.json"))["locations"]
            else:
                gl = np.unique(gt[gt > 0]).tolist()
            samples.append(task1_sample(pl, gl))
    agg = aggregate(samples, seg)
    avg = average(agg, seg)
    print(f"=== TopAneu Task{args.task} ({len(ids)} cases) ===")
    for k, v in avg.items():
        print(f"  {k:<10} {v:.4f}")


def _selftest() -> None:
    """Reproduce deterministic simulation points from evaluation.md."""
    rng = np.random.default_rng(0)
    # build 20 fake cases: gt has 1-2 random location spheres in a 64^3 vol
    shape = (64, 64, 64)
    cases = []
    for _ in range(20):
        gt = np.zeros(shape, np.int64)
        for _ in range(rng.integers(1, 3)):
            c = rng.integers(8, 56, 3); lbl = int(rng.integers(1, 51))
            gt[c[0]-3:c[0]+3, c[1]-3:c[1]+3, c[2]-3:c[2]+3] = lbl
        cases.append(gt)

    def run(preds, seg):
        samp = [task2_sample(p, g) if seg else
                task1_sample(np.unique(p[p > 0]).tolist(), np.unique(g[g > 0]).tolist())
                for p, g in zip(preds, cases)]
        return average(aggregate(samp, seg), seg)

    print("Task2 all_correct (expect P/R/MCC/DSC~1, HD95~0):", {k: round(v, 3) for k, v in run(cases, True).items()})
    zeros = [np.zeros(shape, np.int64) for _ in cases]
    print("Task2 all_zero    (expect all 0, HD95~1):        ", {k: round(v, 3) for k, v in run(zeros, True).items()})
    print("Task1 all_correct (expect P/R/MCC~1):            ", {k: round(v, 3) for k, v in run(cases, False).items()})
    print("Task1 all_zero    (expect all 0):                ", {k: round(v, 3) for k, v in run(zeros, False).items()})


if __name__ == "__main__":
    main()
