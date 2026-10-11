#!/usr/bin/env python
"""Run 2.5D segmentation inference and reconstruct 3D prediction volumes.

Shared by Step1 (binary vessel) and Step2 (multiclass vessel). Given the
``config.yaml`` + a checkpoint from a finished training run, it rebuilds the
model, loads the weights, and writes one predicted 3D ``.npy`` per input case
(``<out>/pred/<case>.npy``) using :func:`predict_volume` — matching the training
normalization and 2.5D windowing.

The predicted vessel volume feeds the next stage: Step1's binary prediction is
the bounding box for ``scripts/crop_to_vessel_region.py``; Step2's multiclass
prediction is the ``aux`` conditioning input for the Step3 aneurysm model.

Usage (Step1 binary vessel, predict on the preprocessed volumes):
    PYTHONPATH=. uv run python scripts/infer_segmentation_2_5d.py \
        --config results/.../fold0/config.yaml \
        --ckpt   results/.../fold0/best-*.ckpt \
        --images-dir  data/vessel_hr/images \
        --out-dir     data/vessel_hr/pred_fold0 \
        --gpu 0

Step2 multiclass vessel (same, pointing at the multiclass model/config):
    ... --config results/multiclass_vessel_segmentation/.../config.yaml --ckpt ...

Step3 aneurysm-location model needs the vessel mask as aux — pass ``--aux-dir``:
    ... --aux-dir data/vessel_hr/pred_fold0/pred
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.segmentation.inference.predict_volume import predict_volume
from src.segmentation.registries.models import build_model


def load_model(config, ckpt_path: Path) -> torch.nn.Module:
    """Rebuild the segmentation model and load weights from a Lightning ckpt."""
    model_cfg = OmegaConf.to_container(config.model, resolve=True)
    # We load our own trained weights; skip downloading encoder pretrained and
    # any warm-start init (the trained ckpt below supersedes it).
    model_cfg["encoder_weights"] = None
    model_cfg.pop("init_from_ckpt", None)
    model = build_model(**model_cfg)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    # LightningModule stores the model under `self.model`; strip that prefix so
    # keys line up with the standalone SegmentationModelV1 (whose own submodule
    # is also `model`, giving ckpt keys like "model.model.encoder...").
    stripped = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if unexpected:
        print(f"[warn] {len(unexpected)} unexpected keys (first: {unexpected[:3]})")
    if missing:
        print(f"[warn] {len(missing)} missing keys (first: {missing[:3]})")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="training config.yaml from the results dir")
    parser.add_argument("--ckpt", required=True, type=Path, help="checkpoint (.ckpt) to load")
    parser.add_argument("--images-dir", required=True, type=Path, help="dir of preprocessed image .npy volumes")
    parser.add_argument("--out-dir", required=True, type=Path, help="output dir (pred/ created inside)")
    parser.add_argument("--aux-dir", type=Path, default=None, help="optional aux volume dir (Step3 vessel conditioning)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id (default: cuda if available else cpu)")
    parser.add_argument("--batch-size", type=int, default=8, help="slices per forward pass")
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases (for testing)")
    parser.add_argument("--fold-yaml", type=Path, default=None,
                        help="restrict prediction to one split of this fold YAML (see --split). "
                             "Without it EVERY volume in --images-dir is predicted, which for an "
                             "out-of-fold evaluation silently includes cases the model trained on.")
    parser.add_argument("--split", choices=["train", "val"], default="val",
                        help="which split of --fold-yaml to predict (default val)")
    parser.add_argument("--save-proba", action="store_true", help="binary only: save sigmoid probability (float32)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = f"cuda:{args.gpu}" if args.gpu is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    ds = config.dataset
    num_slices = int(ds.get("num_slices", 5))
    slice_axis = int(ds.get("slice_axis", 2))
    num_classes = int(config.model.get("num_classes", 1))
    threshold = float(config.get("lightning_module", {}).get("threshold", 0.5))
    perc = config.preprocessing.get("normalization", {}).get("percentile", {})
    lower_p = float(perc.get("lower_percentile", 0.5))
    upper_p = float(perc.get("upper_percentile", 99.5))
    aux_mode = str(ds.get("aux_mode", "label"))
    aux_max = float(ds.get("aux_max", 36.0))

    model = load_model(config, args.ckpt)

    out_pred = args.out_dir / "pred"
    out_pred.mkdir(parents=True, exist_ok=True)

    cases = sorted(args.images_dir.glob("*.npy"))
    if args.fold_yaml:
        keep = {r["id"] for r in OmegaConf.load(args.fold_yaml)[args.split]}
        before = len(cases)
        cases = [c for c in cases if c.stem in keep]
        print(f"fold filter: {args.fold_yaml.name} [{args.split}] -> {len(cases)}/{before} cases")
    if args.limit is not None:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases | num_classes={num_classes} | K={num_slices} axis={slice_axis} | "
          f"aux={'on' if args.aux_dir else 'off'} | device={device}")

    for img_path in tqdm(cases, desc="infer"):
        case = img_path.stem
        out_path = out_pred / f"{case}.npy"
        if out_path.exists() and not args.overwrite:
            continue
        image = np.load(img_path)
        aux = None
        if args.aux_dir is not None:
            aux_path = args.aux_dir / f"{case}.npy"
            if not aux_path.exists():
                print(f"[skip] no aux for {case}")
                continue
            aux = np.load(aux_path)
        pred = predict_volume(
            model, image,
            num_slices=num_slices, slice_axis=slice_axis, num_classes=num_classes,
            threshold=threshold, lower_percentile=lower_p, upper_percentile=upper_p,
            aux=aux, aux_mode=aux_mode, aux_max=aux_max,
            batch_size=args.batch_size, device=device,
            return_proba=args.save_proba,
        )
        np.save(out_path, pred)

    print(f"Wrote predictions to {out_pred}")


if __name__ == "__main__":
    main()
