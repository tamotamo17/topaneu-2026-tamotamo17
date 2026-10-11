"""Shared inference core for the TopAneu-26 submission containers.

Both task images copy this file to ``/opt/app/topaneu_infer.py``; their
``inference.py`` is only the template's ``infer_ct`` / ``infer_mr`` shim on top,
so Task1 (location list) and Task2 (labelled mask) cannot diverge — they are two
projections of ONE pipeline run.

Layout inside the container
---------------------------
``/opt/app``      code (src/, scripts/, configs/, mapping/) baked into the image
``/opt/ml/model`` weights, mounted at runtime from the separately uploaded
                  ``model.tar.gz`` (fixed names, see ``docker/prepare_models.sh``)
``/input``        Grand Challenge inputs (read-only)
``/output``       Grand Challenge outputs

Debug behaviour
---------------
Every run prints an environment banner (GPU, versions, model files, image
geometry) and one line per pipeline stage with its wall time, so a single
sanity-check submission answers "does it run, where does the 12-minute per-case
budget go, and if it broke, where". On an exception the traceback is printed and
an EMPTY-but-valid output is returned, so a partial failure still produces a
well-formed submission instead of a dead job. Set ``TOPANEU_STRICT=1`` at build
time to make failures fatal instead.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK
import torch

# Container paths. The TOPANEU_APP / TOPANEU_MODEL_DIR overrides exist so this
# exact module can be exercised from a repository checkout without docker (see
# docker/README.md, "Running without docker"); on Grand Challenge they are unset and
# the GC paths apply.
APP = Path(os.environ.get("TOPANEU_APP", "/opt/app"))
MODEL = Path(os.environ.get("TOPANEU_MODEL_DIR", "/opt/ml/model"))
CFG = APP / "configs"

# Fail loudly instead of hanging if some dependency tries to reach the hub:
# Grand Challenge runs with `--network none`, so a silent retry loop would burn
# the whole per-case budget. Nothing here should need the network (RF-DETR loads
# our own weights and skips the DINOv2 download, and `load_model` sets
# encoder_weights=None), so a hit here is a real bug worth surfacing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# scripts.anatomical_label reads vessel/location_mapping.json from here at import.
# In the image that is /opt/app/mapping; from a repo checkout (TOPANEU_APP=<repo>)
# the same files live in docker/mapping.
_MAPPING = next((p for p in (APP / "mapping", APP / "docker/mapping") if p.is_dir()),
                APP / "mapping")
os.environ.setdefault("TOPANEU_DEPLOYMENT", str(_MAPPING))

STRICT = os.environ.get("TOPANEU_STRICT", "0") == "1"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- pipeline configuration -------------------------------------------------
# Step3 = Gaussian centre-heatmap detector with the auxiliary presence head
# (`monai_segresnet3d_auxcls`), one per fold, fused by `step3_combine` below.
# Aux rescoring needs exactly ONE detector (the presence head is per-model), so
# it is off for this ensemble.
_STEP3 = [
    (CFG / "segmentation/step3_detector_fold0.yaml", MODEL / "step3_gauss_auxcls.ckpt"),
    (CFG / "segmentation/step3_detector_fold1.yaml", MODEL / "step3_gauss_fold1.ckpt"),
    (CFG / "segmentation/step3_detector_fold2.yaml", MODEL / "step3_gauss_fold2.ckpt"),
]
# Step4 = fine aneurysm-shape seg / FP rejector, same three folds, fused by mean.
_STEP4 = [
    (CFG / "segmentation/step4_fineseg_fold0.yaml", MODEL / "step4_fineseg.ckpt"),
    (CFG / "segmentation/step4_fineseg_fold1.yaml", MODEL / "step4_fineseg_fold1.ckpt"),
    (CFG / "segmentation/step4_fineseg_fold2.yaml", MODEL / "step4_fineseg_fold2.ckpt"),
]
# The final submission: a 3-fold (fold0+1+2) ensemble. Its best single-fold
# predecessor (fold0 detector + aux rescoring + fine-seg rejection) measured, on
# fold0-val of the 2026-08 release (84 cases) with the legacy evaluator port:
#   Task1  P 0.2774 / R 0.2762 / MCC 0.2652
#   Task2  DICE 0.1349 / HD95 0.4114 / VOLSIM 0.1538
#
# The ensemble itself is UNMEASURED and cannot be measured on the released data:
# fold1 and fold2 were trained on all 84 fold0-val cases (the folds partition the
# release, so every case is in some fold's train split) — there is no held-out
# set left for a fold0+1+2 ensemble. Two deliberate departures from the
# single-fold setting:
#   - aux rescoring is OFF. The presence head is per-model, so pipeline_core
#     rejects it outright once step3_specs holds more than one detector. That
#     drops one of the two FP filters (it vetoed 5 of 71 blobs); the Step4
#     shape rejector below still runs.
#   - step3_combine is "mean", not "max". With one FP filter gone, max
#     (recall-biased, takes the most confident fold everywhere) is the wrong way
#     to lean; mean makes the folds have to agree.
_KW = dict(
    step1_config=CFG / "detection/step1_vessel_roi.yaml",
    step1_ckpt=MODEL / "step1_rfdetr.pth",
    step2_config=CFG / "segmentation/step2_vessel_2_5d_fold0.yaml",
    step2_ckpt=MODEL / "step2_vessel25d.ckpt",
    step3_specs=_STEP3,
    step3_combine="mean",
    det_thr=0.50,
    step4_specs=_STEP4,
    fineseg_reject_thr=0.80,
    fineseg_reject_weight=0.75,
    device=DEVICE,
)

_BANNER_DONE = False


def log(msg: str) -> None:
    print(f"[topaneu] {msg}", flush=True)


def _banner() -> None:
    """One-time environment dump. Cheap, and it is the whole point of a debug run."""
    global _BANNER_DONE
    if _BANNER_DONE:
        return
    _BANNER_DONE = True
    log(f"python {sys.version.split()[0]}  torch {torch.__version__}  numpy {np.__version__}")
    log(f"SimpleITK {SimpleITK.Version.VersionString()}  device={DEVICE}  strict={STRICT}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        log(f"gpu: {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} {total:.1f}GiB "
            f"cuda={torch.version.cuda}")
    else:
        log("gpu: NONE — running on CPU, this will not fit the 12 min/case budget")
    if MODEL.is_dir():
        for f in sorted(MODEL.rglob("*")):
            if f.is_file():
                log(f"model: {f.relative_to(MODEL)} {f.stat().st_size / 2**20:.0f} MiB")
    else:
        log(f"model: {MODEL} MISSING — upload model.tar.gz as a Model on the Algorithm")
    for key, path in sorted(_KW.items()):
        if isinstance(path, Path) and not path.exists():
            log(f"MISSING {key}: {path}")
    for stage, specs in (("step3", _STEP3), ("step4", _STEP4)):
        for cfg, ckpt in specs:
            for p in (cfg, ckpt):
                if not Path(p).exists():
                    log(f"MISSING {stage}: {p}")


def _empty_locations() -> list[int]:
    return []


def _empty_mask(image: "SimpleITK.Image") -> "SimpleITK.Image":
    """All-background mask matching the input grid — a valid, if useless, answer."""
    arr = np.zeros(SimpleITK.GetArrayFromImage(image).shape, np.uint8)
    out = SimpleITK.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def predict(image: "SimpleITK.Image", modality: str):
    """Run the full pipeline once. Returns ``(location ids, labelled mask)``.

    ``modality`` is "CTA" or "MRA", taken from the Grand Challenge input socket.
    The mask carries the input's origin/spacing/direction, which the evaluator
    requires (``pred.shape == gt.shape``); the stock template drops them.
    """
    _banner()
    t0 = time.time()
    arr = SimpleITK.GetArrayFromImage(image)                       # (z,y,x)
    vol = arr.transpose(2, 1, 0).astype(np.float32)                # -> (x,y,z)
    spacing = tuple(float(s) for s in image.GetSpacing())          # (x,y,z)
    log(f"input: modality={modality} shape(x,y,z)={vol.shape} spacing={spacing} "
        f"dtype={arr.dtype} range=[{float(vol.min()):.1f},{float(vol.max()):.1f}]")

    try:
        from scripts.pipeline_core import run_pipeline

        mask_xyz, locs, stats = run_pipeline(vol, spacing, modality, log=log, **_KW)
        log(f"blobs={stats['blobs']} refined={stats['refined']} "
            f"aux_kept={stats['aux_kept']} aux_rejected={stats['aux_rejected']} "
            f"fs_rejected={stats['fs_rejected']} locations={locs}")
    except Exception:                                              # noqa: BLE001
        log("PIPELINE FAILED — traceback follows")
        traceback.print_exc()
        if STRICT:
            raise
        log("falling back to an empty prediction so the job still produces valid output")
        log(f"total: {time.time() - t0:.1f}s (failed)")
        return _empty_locations(), _empty_mask(image)

    out = SimpleITK.GetImageFromArray(mask_xyz.transpose(2, 1, 0).astype(np.uint8))
    out.CopyInformation(image)
    log(f"total: {time.time() - t0:.1f}s (budget is 12 min/case)")
    return [int(v) for v in locs], out
