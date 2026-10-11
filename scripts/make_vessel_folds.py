#!/usr/bin/env python
"""Generate and persist the cross-validation split for the vessel dataset.

The split is computed **once** here and written to per-fold YAML files so the
exact case IDs (and their image/mask paths) used for each fold are recorded and
reproducible, instead of being recomputed at runtime inside the dataset.

Output (one file per fold under ``--out-dir``)::

    folds/fold0.yaml
    folds/fold1.yaml
    ...

Each file has the structure::

    meta: {fold, num_folds, seed, stratify, data_root, num_train, num_val}
    train:
      - {id: <case>, image: <abs .npy>, mask: <abs .npy>}
      - ...
    val:
      - {id: <case>, image: <abs .npy>, mask: <abs .npy>}

Cases are paired by identical filename stems in ``<data-root>/images`` and
``<data-root>/masks``. By default the split is **stratified by modality**
(the ``center{N}_{ct|mr}`` token in the filename) so CT/MR ratios stay balanced
across folds; pass ``--stratify none`` for a plain shuffled round-robin.

With ``--assignment splits/folds.yaml`` the split is NOT recomputed: each case
takes the fold listed there (the submission's split). Every case in the data
must appear in the assignment and vice versa, so a mismatched data release
fails loudly instead of silently producing a different split.

Usage:
    PYTHONPATH=. uv run python scripts/make_vessel_folds.py \
        --data-root data/vessel_lr \
        --out-dir   data/vessel_lr/folds \
        --assignment splits/folds.yaml
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def modality_key(case: str) -> str:
    """Extract the ``center{N}_{ct|mr}`` stratification token from a case id."""
    m = re.search(r"(center\d+_(?:ct|mr))", case)
    return m.group(1) if m else "unknown"


def assign_folds(cases: list[str], num_folds: int, seed: int, stratify: str) -> dict[str, int]:
    """Return a deterministic mapping ``case -> fold`` in ``[0, num_folds)``."""
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}

    if stratify == "modality":
        groups = defaultdict(list)
        for c in cases:
            groups[modality_key(c)].append(c)
        offset = 0
        # Sort groups for determinism; round-robin within each, continuing the
        # offset across groups so the overall per-fold counts stay balanced.
        for _, members in sorted(groups.items()):
            members = sorted(members)
            order = rng.permutation(len(members))
            for rank, idx in enumerate(order):
                fold_of[members[idx]] = (offset + rank) % num_folds
            offset += len(members)
    elif stratify == "none":
        order = rng.permutation(len(cases))
        for rank, idx in enumerate(order):
            fold_of[cases[idx]] = rank % num_folds
    else:
        raise ValueError(f"Unknown stratify mode: {stratify}")

    return fold_of


def load_assignment(path: Path, cases: list[str]) -> dict[str, int]:
    """Read a fixed ``case -> fold`` mapping and check it covers exactly ``cases``."""
    doc = OmegaConf.to_container(OmegaConf.load(path))
    fold_of = {c: int(re.fullmatch(r"fold(\d+)", k).group(1)) for k, ids in doc.items() for c in ids}
    missing, extra = sorted(set(cases) - set(fold_of)), sorted(set(fold_of) - set(cases))
    if missing or extra:
        raise SystemExit(f"{path} does not match the data: {len(missing)} cases not in the assignment "
                         f"(e.g. {missing[:3]}), {len(extra)} assigned cases without data (e.g. {extra[:3]})")
    return fold_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path, help="dir with images/ and masks/ (*.npy)")
    parser.add_argument("--out-dir", required=True, type=Path, help="dir to write fold{N}.yaml files")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", choices=["modality", "none"], default="modality")
    parser.add_argument("--assignment", type=Path, default=None,
                        help="YAML {fold<N>: [val case ids]} to use instead of generating the split")
    args = parser.parse_args()

    images_dir = args.data_root / "images"
    masks_dir = args.data_root / "masks"
    cases = sorted(p.stem for p in images_dir.glob("*.npy") if (masks_dir / f"{p.stem}.npy").exists())
    if not cases:
        raise SystemExit(f"No paired .npy volumes found in {images_dir} / {masks_dir}")

    if args.assignment is not None:
        fold_of = load_assignment(args.assignment, cases)
        args.num_folds = max(fold_of.values()) + 1
    else:
        fold_of = assign_folds(cases, args.num_folds, args.seed, args.stratify)

    def entry(case: str) -> dict:
        return {
            "id": case,
            "image": str((images_dir / f"{case}.npy").resolve()),
            "mask": str((masks_dir / f"{case}.npy").resolve()),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(cases)} cases | {args.num_folds} folds | seed={args.seed} | stratify={args.stratify}\n")
    print("fold | val | CT MR | train")
    for f in range(args.num_folds):
        val_cases = [c for c in cases if fold_of[c] == f]
        train_cases = [c for c in cases if fold_of[c] != f]
        ct = sum("_ct_" in c for c in val_cases)
        mr = sum("_mr_" in c for c in val_cases)
        print(f"  {f}  | {len(val_cases):3d} | {ct:2d} {mr:2d} | {len(train_cases):3d}")

        doc = {
            "meta": {
                "fold": f,
                "num_folds": args.num_folds,
                "seed": args.seed,
                "stratify": args.stratify if args.assignment is None else f"fixed ({args.assignment})",
                "data_root": str(args.data_root.resolve()),
                "num_train": len(train_cases),
                "num_val": len(val_cases),
            },
            "train": [entry(c) for c in train_cases],
            "val": [entry(c) for c in val_cases],
        }
        out_path = args.out_dir / f"fold{f}.yaml"
        OmegaConf.save(OmegaConf.create(doc), out_path)

    print(f"\nWrote {args.num_folds} fold files to {args.out_dir}")


if __name__ == "__main__":
    main()
