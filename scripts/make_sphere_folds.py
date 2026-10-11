#!/usr/bin/env python
"""Derive sphere-target fold YAMLs from the base (location-mask) fold YAMLs.

Step3 trains on the SAME images and the SAME train/val split as the base folds,
but the target volume is the rasterized sphere mask written by
``scripts/make_sphere_targets.py`` instead of the raw location mask. Rather than
re-running the split (and risking a different assignment), this script copies
each ``fold{N}.yaml`` and rewrites only the ``mask`` path to point at the sphere
directory, so the case-to-fold mapping is guaranteed identical.

Usage:
    PYTHONPATH=. uv run python scripts/make_sphere_folds.py \
        --folds-dir  data/aneurysm_hr/folds \
        --sphere-dir data/aneurysm_hr/sphere_g25 \
        --out-dir    data/aneurysm_hr/folds_sphere_g25
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-dir", required=True, type=Path, help="base folds dir (fold{N}.yaml)")
    ap.add_argument("--sphere-dir", required=True, type=Path, help="dir with the sphere target .npy volumes")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    fold_files = sorted(args.folds_dir.glob("fold*.yaml"))
    if not fold_files:
        raise SystemExit(f"No fold*.yaml found in {args.folds_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sphere_dir = args.sphere_dir.resolve()

    for path in fold_files:
        doc = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        missing = []
        for split in ("train", "val"):
            for row in doc[split]:
                target = sphere_dir / f"{row['id']}.npy"
                if not target.exists():
                    missing.append(row["id"])
                row["mask"] = str(target)
        if missing:
            raise SystemExit(f"{path.name}: missing sphere targets for {len(missing)} cases, e.g. {missing[:3]}")

        doc["meta"] = {**doc.get("meta", {}), "note": f"{sphere_dir.name} targets", "sphere_dir": str(sphere_dir)}
        OmegaConf.save(OmegaConf.create(doc), args.out_dir / path.name)
        print(f"{path.name}: train={len(doc['train'])} val={len(doc['val'])} -> {args.out_dir / path.name}")


if __name__ == "__main__":
    main()
