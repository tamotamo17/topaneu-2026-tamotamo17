#!/usr/bin/env python
"""Approach A: anatomical location labelling of detected blobs from the PREDICTED
vessel segmentation — no per-class training data needed.

The 50 location classes are vessel segments / junctions. For a detected blob we
read the predicted vessel labels (Step2, 36 classes) in a ball around its
centroid, take the dominant vessel + the set of neighbouring vessels, and apply
an anatomical rule table (vessel [+ neighbour] -> location base name) plus a
laterality (R/L from the dominant vessel's side). This maps any blob to one of
the 50 locations.

Provides `label_blob(centroid, vessel_vol, ...)` for inference and a `--validate`
mode that scores the rule against GT-matched train blobs in a blob manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Mappings come from the ACTIVE release: the 2026-08 release renumbered the MCA
# locations (47-52), so pointing this at the wrong release silently produces
# valid-looking but wrong label ids. Defaults to the copy bundled in
# docker/mapping/ (copied from the data release); override with
# $TOPANEU_DEPLOYMENT.
DEP = Path(os.environ.get("TOPANEU_DEPLOYMENT",
                          Path(__file__).resolve().parents[1] / "docker" / "mapping"))


def _load_maps():
    vmap = json.load(open(DEP / "vessel_mapping.json"))["labels"]      # name->id (36)
    lmap = json.load(open(DEP / "location_mapping.json"))["labels"]    # name->id (50)
    # vessel id -> (base, side)
    vid2 = {}
    for name, i in vmap.items():
        if i == 0:
            continue
        side = "R" if name.startswith("R-") else "L" if name.startswith("L-") else None
        base = re.sub(r"^[RL]-", "", name)
        vid2[i] = (base, side)
    # (loc base, side) -> id ; side None for central
    locbase = {}
    for name, i in lmap.items():
        if i == 0:
            continue
        side = "R" if name.startswith("R-") else "L" if name.startswith("L-") else None
        base = re.sub(r"^[RL]-", "", name)
        base = re.sub(r"^\d+\.\d+\s+", "", base)  # strip "3.4 " numeric code
        locbase[(base, side)] = i
    return vid2, locbase


VID2BASE, LOCBASE = _load_maps()


def _loc_id(base: str, side: str | None) -> int | None:
    """Resolve a location base name (+side) to its label id; try central then sided."""
    if (base, None) in LOCBASE:
        return LOCBASE[(base, None)]
    if (base, side) in LOCBASE:
        return LOCBASE[(base, side)]
    # some are only bilateral; fall back to either side present
    for s in ("R", "L"):
        if (base, s) in LOCBASE:
            return LOCBASE[(base, s)]
    return None


# --- positional thresholds (mm), fitted on the fold0 TRAIN split only ----------
# The neighbour set answers "which vessels are adjacent"; it cannot answer "where
# ALONG a vessel the aneurysm sits". Three official locations need exactly that,
# and without it the rule table could never emit them at all — measured: 8 classes
# unreachable by construction, holding 59/404 = 14.6% of all GT aneurysms.
# Each threshold is the F1-optimum of a 0.5mm scan on train.
#
# Which vessel wins the voxel count decides which branch we land in, and that is
# NOT the same vessel a clinician would name. Measured on train:
#   M1-M2 junction     -> primary is M2 in 51/58 (already handled by the M2 branch)
#   M1 early bifurcation -> primary is M1 in 11/12
#   ICA C6-nonOA       -> primary is ICA-C1-C5 in 16/23, ICA-C6-C7 in only 6
# So the positional test has to sit in the branch the data actually reaches. A first
# attempt put the C6 test under `primary == "ICA-C6-C7"`, where it fired for 0 of the
# 6 val cases because they all land in the C1-C5 branch — fitting a discriminator is
# useless without checking the rule PATH reaches it.
EARLY_BIF_M2_MM = 10.0    # "M2 is near" for a primary-M1 blob. Separates early
                          # bifurcation from M1 trunk (train dist(M2): 6.4 vs 12.1mm)
                          # and survives the neighbour-set 5% cutoff dropping M2 when
                          # a detected centroid sits a little off the GT one.
EARLY_BIF_ICA_MM = 13.0   # then early bifurcation vs ordinary M1-M2 junction, by how
                          # proximal the bifurcation is (train dist(ICA-C6-C7): 9.6 vs
                          # 17.9mm). Together: F1 0.947, 9/10 early-bif captured, 0/5
                          # M1-M2 junctions stolen.
C6_NONOA_C1C5_MM = 1.0    # ICA C6-nonOA when the count says ICA-C1-C5: the C6-C7 class
                          # is TOUCHING (train median 0.0mm) whereas a true infraclinoid
                          # aneurysm sits ~3.0mm off it. F1 0.750 — captures 12/14 but
                          # steals 6/24 infraclinoid, so this one has a real cost and is
                          # ablated separately in the Task1 measurement. Set to a
                          # negative value to disable.
BRANCH_BY_DISTANCE = True # Master switch: decide branches/junctions by distance.
                          # False restores the neighbour-set-only rule (to reproduce
                          # historical measurements).
# --- Thin branches are tested by DISTANCE, not neighbour-set membership --------
# The neighbour set drops vessels below 5% of the ball. Pcom/AChA/OA rarely clear
# that bar, so they never entered the set and C7-Pcom-junction was produced for only
# 5/26 train cases. Lowering nb_frac globally lost on Task1 before (it affects every
# vessel and scatters the labels of false-positive blobs, hurting the macro score);
# a per-vessel distance test targets only the vessels that need it.
# Each threshold is the train-F1 optimum (0.5 mm grid). Comments give
# positives captured / negatives wrongly absorbed.
PCOM_JUNC_MM = 9.5        # C7-Pcom-junction. Chosen by directly maximising the mean
                          # held-out accuracy of a 5-fold CV on train (0.6570 -> 0.6669).
                          # The pair-wise F1 optimum was 5.5 mm, but overall accuracy
                          # prefers a looser value: Pcom is thin and often far away, so a
                          # strict threshold loses more junctions than it saves.
ACHA_JUNC_MM = 4.5        # C7-AChA-junction: only 4/10 (60% of positives have no
                          # segmented AChA at all, so they are unreachable) / 2/79 (F1 0.500)
OA_JUNC_MM = 8.0          # C6-OA-junction. Also CV-selected (pair-wise F1 optimum 3.0 mm).
# On the BA/VA side the problem is inverted: the set test fires "too close" and
# wrongly produces junctions, so the thresholds are strict.
SCA_JUNC_MM = 1.0         # BA-SCA junction: 2/4 captured / 0/16 taken from BA tip/trunk
                          # (the set test fired on 38% of negatives, causing BA-tip errors)
VA_BA_JUNC_MM = 8.5       # VA-BA junction: 6/7 captured / 0/16 (F1 0.923) — the most reliable
PICA_JUNC_MM = 2.0        # VA-PICA junction. CV made this one stricter instead
                          # (pair-wise F1 optimum 4.0 mm); fewer VA-trunk cases are lost.
                          # (set test: same 3/5 captured but 6/16 absorbed -> distance wins)
A2_ACOM_MM = 5.0          # A1A2 primary with a distant Acom: A2 if the ICA is also far, else A1.
A2_ICA_MM = 4.0           # A2: 3/5 captured / loses 2/28 Acom complex. Small n (5).
A1_ICA_MM = 3.0           # A1 vs Acom complex: A1 runs from the ICA terminus to the
A1_ACOM_MM = 3.0          # Acom, so it hugs the ICA and stays off the Acom.
                          # Train: 6/8 A1 captured, 2/33 Acom stolen. WARNING: n=8 and
                          # fold0-val contains ZERO A1 aneurysms, so this one rule is
                          # unvalidated on held-out data — train evidence only.


def _nearest_branch(nb, d, dist):
    """Location name of the nearest ICA branch (Pcom/AChA/OA) within its threshold, else None.

    Without ``dist`` this falls back to the neighbour-set test unchanged.
    """
    if dist is None or not BRANCH_BY_DISTANCE:
        for v, locname in (("Pcom", "ICA C7-Pcom-junction"), ("AChA", "ICA C7-AChA-junction"),
                           ("OA", "ICA C6-OA-junction")):
            if v in nb:
                return locname
        return None
    inf = float("inf")
    cands = [(d.get("Pcom", inf), PCOM_JUNC_MM, "ICA C7-Pcom-junction"),
             (d.get("AChA", inf), ACHA_JUNC_MM, "ICA C7-AChA-junction"),
             (d.get("OA", inf), OA_JUNC_MM, "ICA C6-OA-junction")]
    ok = [(dd, locname) for dd, thr, locname in cands if dd <= thr]
    return min(ok)[1] if ok else None


def _rule(primary: str, nb: set[str], dist: dict[str, float] | None = None) -> str:
    """Anatomical rule: primary vessel base (+ neighbour bases) -> location base name.

    ``dist`` maps a vessel base name to the distance in mm from the blob centroid
    to that vessel's nearest predicted voxel (``inf`` when absent). It is optional
    and every positional refinement is gated on it, so calling without ``dist``
    reproduces the historical behaviour exactly — needed because the recorded
    baselines (matched-blob 0.644 on val, Task1 MCC 0.2025/0.2067) were measured
    with the neighbour-set-only rule.
    """
    d = dist or {}
    inf = float("inf")
    if primary == "ICA-C1-C5":
        # The count says the cavernous/petrous segment, but a C6 aneurysm often has
        # more C1-C5 voxels than C6-C7 voxels in its ball. Touching the C6-C7 class
        # is what distinguishes it from a genuine infraclinoid aneurysm.
        if (dist is not None and C6_NONOA_C1C5_MM >= 0 and "OA" not in nb
                and d.get("ICA-C6-C7", inf) <= C6_NONOA_C1C5_MM):
            return "ICA C6-nonOA"
        return "ICA infraclinoid C1-C5"
    if primary == "OA":
        return "ICA C6-OA-junction"
    if primary == "ICA-C6-C7":
        # Branches are tested by "within threshold", not set membership, and the
        # nearest one wins when several qualify (Pcom's threshold is loose, so a
        # fixed order would miss a closer OA).
        branch = _nearest_branch(nb, d, dist)
        if branch:
            return branch
        if "M1" in nb and "A1A2" in nb:
            return "ICA C7-terminus"
        # No C6/C7 split attempted here on purpose: among train blobs whose count
        # primary is ICA-C6-C7, the non-branch cases are 6 C6-nonOA vs 5 C7-nonBranch
        # — a coin flip with no separating distance found. The C6 rule lives in the
        # ICA-C1-C5 branch above, where 16 of the 23 train C6-nonOA cases actually land.
        return "ICA C7-nonBranch"
    if primary == "Pcom":
        return "ICA C7-Pcom-junction"
    if primary == "AChA":
        return "ICA C7-AChA-junction"
    if primary == "M1":
        # Test M2 by DISTANCE, not membership of the 5%-of-ball neighbour set: a
        # detected centroid a few mm off the annotation drops M2 below that cutoff
        # and the blob silently becomes "M1 trunk" (observed on val).
        m2_near = "M2" in nb or (dist is not None and d.get("M2", inf) <= EARLY_BIF_M2_MM)
        if m2_near:
            if dist is not None and d.get("ICA-C6-C7", inf) <= EARLY_BIF_ICA_MM:
                return "M1 early bifurcation"
            return "M1-M2 junction"
        return "M1 trunk"
    if primary in ("M2", "M3"):
        return "M1-M2 junction" if "M1" in nb else "Distal-M2M3"
    if primary == "A1A2":
        # The "no isolated A1/A2 cases" bias below was calibrated on the 98-case
        # dataset. The 417-case release contains 8 A1 aneurysms, so the blanket
        # mapping is now wrong for them: A1 runs from the ICA terminus to the Acom,
        # so it hugs the ICA and stays off the Acom.
        if dist is not None and d.get("Acom", inf) > A1_ACOM_MM:
            # An A1A2 blob far from the Acom is A1 or A2: A1 touches the ICA
            # terminus, A2 extends anteriorly.
            if d.get("ICA-C6-C7", inf) <= A1_ICA_MM:
                return "A1"
            if (BRANCH_BY_DISTANCE and d.get("Acom", inf) > A2_ACOM_MM
                    and d.get("ICA-C6-C7", inf) > A2_ICA_MM):
                return "A2"
        # ACA-region aneurysms in this dataset are overwhelmingly Acom-complex;
        # bias the remaining A1A2 -> Acom complex.
        return "Acom complex"
    if primary in ("A3",):
        return "A3"
    if primary in ("3rd-A2",):
        return "A2"
    if primary in ("3rd-A3",):
        return "A3"
    if primary == "Acom":
        return "Acom complex"
    if primary == "P1P2":
        return "P1P2"
    if primary == "P3P4":
        return "P3P4"
    if primary == "BA":
        if "AICA" in nb:
            return "BA-AICA junction"
        # Check VA-BA junction first: its distance test separates best (F1 0.923)
        # and was measured to steal no BA tip/trunk case, so the order costs nothing.
        if dist is not None and BRANCH_BY_DISTANCE and d.get("VA", inf) <= VA_BA_JUNC_MM:
            return "VA-BA junction"
        # The set test for SCA also fires near the BA tip (38% of negatives); use a strict distance.
        if (d.get("SCA", inf) <= SCA_JUNC_MM) if (dist is not None and BRANCH_BY_DISTANCE) else ("SCA" in nb):
            return "BA-SCA junction"
        if "P1P2" in nb or "Pcom" in nb:
            return "BA tip"
        if "VA" in nb:
            return "VA-BA junction"
        return "BA trunk"
    if primary == "VA":
        # PICA runs alongside the VA for a long stretch, so set membership alone does not mean a junction.
        if (d.get("PICA", inf) <= PICA_JUNC_MM) if (dist is not None and BRANCH_BY_DISTANCE) else ("PICA" in nb):
            return "VA-PICA junction"
        if "BA" in nb:
            return "VA-BA junction"
        return "VA trunk"
    if primary == "SCA":
        return "BA-SCA junction" if "BA" in nb else "SCA trunk"
    if primary == "AICA":
        return "BA-AICA junction" if "BA" in nb else "AICA trunk"
    if primary == "PICA":
        return "VA-PICA junction" if "VA" in nb else "PICA trunk"
    return "ICA C7-nonBranch"  # fallback


POSITION_ENABLED = True  # master switch for every positional refinement below.
                         # Exists so the recorded neighbour-set-only baselines
                         # (matched-blob 0.644, Task1 MCC 0.2025) stay reproducible
                         # end-to-end without threading a flag through paint_blobs.
SPACING_MM = 0.4         # crop60_sp0.4 grid; the positional thresholds are in mm
_POS_R_VOX = 45          # 18mm at 0.4mm spacing — must reach past the largest
                         # threshold above (16mm) or that vessel reads as absent.


def _dists(centroid, vessel_vol, r_vox=_POS_R_VOX):
    """Distance in mm from ``centroid`` to the nearest voxel of each vessel base.

    ``inf`` for a base with no voxel inside the search box, which is what the
    positional rules test against ("Acom is not there" is as informative as
    "Acom is 2mm away"). Deliberately a separate, wider box than the labelling
    ball: the labelling ball is sized for "which vessel is this blob ON", while
    these thresholds ask "how far along the tree is it".
    """
    z, y, x = [int(round(v)) for v in centroid]
    r = int(r_vox)
    sl = tuple(slice(max(0, c - r), c + r + 1) for c in (z, y, x))
    sub = vessel_vol[sl]
    idx = np.argwhere(sub > 0)
    if idx.size == 0:
        return {}
    off = np.array([z - sl[0].start, y - sl[1].start, x - sl[2].start])
    dd = np.linalg.norm(idx - off, axis=1) * SPACING_MM
    cls = sub[tuple(idx.T)]
    out = {}
    for vid in range(1, 37):
        base = VID2BASE[vid][0]
        m = cls == vid
        if m.any():
            v = float(dd[m].min())
            if v < out.get(base, float("inf")):
                out[base] = v
    return out


def label_blob_candidates(centroid, vessel_vol, r_vox=15, nb_frac=0.10, k=1,
                          use_position=True):
    """Ordered location-id candidates for a blob, most likely first (max ``k``).

    The rule keys off the SINGLE most frequent vessel class in the ball around
    the centroid. Candidate ``i`` simply keys off the i-th most frequent class
    instead, keeping the same neighbour set and laterality. That is the natural
    runner-up: the residual label errors are ICA-C7 subdivisions, where the ball
    genuinely contains two plausible parent vessels and the counts are close.

    Returns ``[]`` when no vessel is found nearby (the caller treats that as
    "not a reportable finding"). Duplicates are collapsed, so fewer than ``k``
    candidates may come back — several primaries can map to one location.
    """
    z, y, x = [int(round(v)) for v in centroid]
    r = int(r_vox)
    sub = vessel_vol[max(0, z - r):z + r + 1, max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    vals = sub[sub > 0]
    if vals.size == 0:
        # widen once
        r2 = r * 2
        sub = vessel_vol[max(0, z - r2):z + r2 + 1, max(0, y - r2):y + r2 + 1, max(0, x - r2):x + r2 + 1]
        vals = sub[sub > 0]
        if vals.size == 0:
            return []
    counts = np.bincount(vals, minlength=37)
    thr = max(1, int(nb_frac * vals.size))
    # computed once per blob, not per candidate — the distances do not depend on
    # which vessel we nominate as primary
    dist = _dists(centroid, vessel_vol) if (use_position and POSITION_ENABLED) else None
    # laterality is a property of the ball, not of which primary we pick
    order = [int(i) for i in np.argsort(counts)[::-1] if counts[i] > 0]
    out: list[int] = []
    for primary_id in order[:max(1, int(k))]:
        p_base, p_side = VID2BASE[primary_id]
        nb = {VID2BASE[i][0] for i in range(1, 37) if counts[i] >= thr and i != primary_id}
        side = p_side
        if side is None:
            sided = [VID2BASE[i][1] for i in range(1, 37) if counts[i] >= thr and VID2BASE[i][1]]
            side = max(set(sided), key=sided.count) if sided else "R"
        loc = _loc_id(_rule(p_base, nb, dist), side)
        if loc is not None and loc not in out:
            out.append(int(loc))
    return out


def label_blob(centroid, vessel_vol, r_vox=15, nb_frac=0.10, use_position=True):
    """Return a location id (1..50) for a blob at `centroid` given the predicted
    vessel volume, or None if no vessel found nearby.

    ``use_position=False`` restores the neighbour-set-only rule that every
    recorded baseline was measured with.
    """
    c = label_blob_candidates(centroid, vessel_vol, r_vox=r_vox, nb_frac=nb_frac, k=1,
                              use_position=use_position)
    return c[0] if c else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--vessel-dir", required=True, type=Path, help="predicted vessel .npy (0-36)")
    ap.add_argument("--r-vox", type=int, default=15)
    ap.add_argument("--nb-frac", type=float, default=0.10)
    ap.add_argument("--validate", action="store_true", help="score rule on GT-matched train blobs")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    vcache = {}
    n_ok = n_tot = 0
    per_class = {}   # gt_label -> [n_ok, n_tot]
    confus = {}
    for cid, m in manifest.items():
        if args.validate and m["split"] != "train":
            continue
        vv = vcache.setdefault(cid, np.load(args.vessel_dir / f"{cid}.npy"))
        for b in m["blobs"]:
            if b["label"] == 0:      # reject/FP blobs: no GT location to score against
                continue
            pred = label_blob(b["c"], vv, r_vox=args.r_vox, nb_frac=args.nb_frac)
            gt = b["label"]
            per_class.setdefault(gt, [0, 0])
            per_class[gt][1] += 1; n_tot += 1
            if pred == gt:
                per_class[gt][0] += 1; n_ok += 1
            else:
                confus[(gt, pred)] = confus.get((gt, pred), 0) + 1
    print(f"anatomical rule accuracy on matched train blobs: {n_ok}/{n_tot} = {n_ok/max(n_tot,1):.3f}\n")
    print(f"{'loc':>4} {'acc':>7} {'n':>4}")
    for g in sorted(per_class):
        ok, tot = per_class[g]
        print(f"{g:>4} {ok/tot:>7.2f} {tot:>4}")
    print("\ntop confusions (gt->pred: count):")
    for (g, p), c in sorted(confus.items(), key=lambda x: -x[1])[:12]:
        print(f"  {g}->{p}: {c}")


if __name__ == "__main__":
    main()
