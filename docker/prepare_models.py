#!/usr/bin/env python
"""Assemble the 8 submitted checkpoints (Step1, Step2, and Step3/Step4 as 3-fold
ensembles) into docker/model/ under the fixed names the container expects, then
pack docker/model.tar.gz.

The tarball is uploaded SEPARATELY to Grand Challenge as a Model and extracted at
runtime to /opt/ml/model. Task1 and Task2 run the same pipeline, so one tarball
serves both.

The Lightning checkpoints are STRIPPED to their `state_dict` (plain
tensors). That drops the optimizer/scheduler state — roughly two thirds of the
bytes — and, more importantly, removes the pickled Lightning objects, so the
container does not need pytorch-lightning installed just to `torch.load` a file
it only reads weights out of. The RF-DETR checkpoint is copied verbatim because
rfdetr's own loader parses its full structure.

Usage:
    PYTHONPATH=. uv run python docker/prepare_models.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

# (destination name, run directory under --results-dir, checkpoint glob, strip?)
# Keep in step with the config paths in docker/common/topaneu_infer.py: a ckpt
# loaded against the wrong config silently produces garbage. Each glob must match
# exactly one file (Lightning names the best checkpoint best-epoch=..-val_dice=...).
STAGES = [
    # Step1 vessel-ROI detection, RF-DETR nano (configs/detection/step1_vessel_roi.yaml)
    ("step1_rfdetr.pth", "step1_vessel_roi/step1_vessel_roi_rfdetr_nano/fold0",
     "checkpoint_best_total.pth", False),
    # Step2 2.5D 37-class vessel seg, fold0 (step2_vessel_2_5d_fold0.yaml)
    ("step2_vessel25d.ckpt", "step2_vessel_2_5d/step2_vessel_2_5d_e50/fold0", "best-*.ckpt", True),
    # Step3 heatmap detector + presence head, 3-fold ensemble fused by mean
    # (step3_detector_fold{0,1,2}.yaml). The ensemble disables aux rescoring: the
    # presence head is per-model, so pipeline_core refuses it with >1 detector.
    ("step3_gauss_auxcls.ckpt", "step3_detector/step3_detector_auxcls/fold0", "best-*.ckpt", True),
    ("step3_gauss_fold1.ckpt", "step3_detector/step3_detector_auxcls/fold1", "best-*.ckpt", True),
    ("step3_gauss_fold2.ckpt", "step3_detector/step3_detector_auxcls/fold2", "best-*.ckpt", True),
    # Step4 fine aneurysm-shape seg / FP rejector, 3-fold ensemble fused by mean
    # (step4_fineseg_fold{0,1,2}.yaml)
    ("step4_fineseg.ckpt", "step4_fineseg/step4_fineseg_reject/fold0", "best-*.ckpt", True),
    ("step4_fineseg_fold1.ckpt", "step4_fineseg/step4_fineseg_reject/fold1", "best-*.ckpt", True),
    ("step4_fineseg_fold2.ckpt", "step4_fineseg/step4_fineseg_reject/fold2", "best-*.ckpt", True),
]


def resolve(results_dir: Path, run: str, pattern: str) -> Path | str:
    """The one checkpoint matching ``pattern`` in the run dir, or an error string."""
    hits = sorted((results_dir / run).glob(pattern))
    if len(hits) != 1:
        return f"{results_dir / run / pattern}: {len(hits)} matches"
    return hits[0]


def strip_ckpt(src: Path, dst: Path) -> None:
    """Save only the model tensors, keyed the way `load_model` reads them."""
    obj = torch.load(src, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj)
    tensors = {k: v.detach().cpu() for k, v in sd.items() if torch.is_tensor(v)}
    dropped = len(sd) - len(tensors)
    torch.save({"state_dict": tensors}, dst)
    if dropped:
        print(f"    dropped {dropped} non-tensor entries")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("results"),
                    help="root of the training runs (scripts/run_pipeline.sh writes to results/)")
    ap.add_argument("--out", type=Path, default=HERE / "model")
    ap.add_argument("--no-tar", action="store_true", help="skip packing model.tar.gz")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sources = [(name, resolve(args.results_dir, run, pat), strip) for name, run, pat, strip in STAGES]
    missing = [str(src) for _, src, _ in sources if isinstance(src, str)]
    if missing:
        raise SystemExit("missing checkpoint(s):\n  " + "\n  ".join(missing))

    for name, src, strip in sources:
        dst = args.out / name
        print(f"{name} <- {src}")
        if strip:
            strip_ckpt(src, dst)
        else:
            shutil.copyfile(src, dst)
        print(f"    {src.stat().st_size / 2**20:.0f} MiB -> {dst.stat().st_size / 2**20:.0f} MiB")

    if args.no_tar:
        return
    tarball = HERE / "model.tar.gz"
    subprocess.run(["tar", "-czf", str(tarball), "-C", str(args.out), "."], check=True)
    print(f"\npacked -> {tarball} ({tarball.stat().st_size / 2**20:.0f} MiB)")
    print("upload it as a separate Model on the Grand Challenge Algorithm "
          "(the same tarball serves Task1 and Task2)")


if __name__ == "__main__":
    main()
