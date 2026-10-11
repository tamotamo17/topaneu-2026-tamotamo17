# TopAneu-26 submission containers (Task1 / Task2)

Two containers are submitted to Grand Challenge. Both run the **same single
pipeline pass**; Task1 outputs the list of location ids and Task2 the
location-labelled mask. They share `docker/common/topaneu_infer.py`, so their
findings always agree.

```
Step1 RF-DETR vessel-ROI detection (coronal/sagittal projections at 0.625 mm)
  → crop ROI centre ±60 mm at 0.4 mm / 300³
  → Step2 2.5D vessel segmentation (37 classes)
  → Step3 3D Gaussian-heatmap detector, 3-fold ensemble
  → Step4 fine-shape segmentation (3-fold ensemble) + false-positive rejection
  → anatomical location labelling
  → uncrop to the input grid
```

See the top-level [README](../README.md) for the method.

## Official interface ([topaneu-26.grand-challenge.org](https://topaneu-26.grand-challenge.org/participation/) / [Bangulli/TopAneu-26](https://github.com/Bangulli/TopAneu-26))

| Item | Value |
|---|---|
| Input sockets | `head-ct-angiography` → `/input/images/head-ct-angio/*.mha`<br>`head-mr-angiography` → `/input/images/head-mr-angio/*.mha` |
| Task1 output | `/output/detected-aneurysm-locations.json` = flat list of ints, e.g. `[34, 35]` |
| Task2 output | `/output/images/aneurysm-segmentation/output.mha` = 1 channel uint8, values **0-52**, same grid as the input |
| Runtime limits | **12 min** per case / 32 GB RAM / optional **NVIDIA T4 16 GB** / `--network none` |

`main.py` follows the official template (`templates/task[1/2]/main.py`). The only
addition is that **the written output is read back and validated** and the
result is logged (Task1: list of ints within 1-52; Task2: uint8, values 0-52,
shape and spacing equal to the input).

## Layout

```
docker/
├── common/topaneu_infer.py   # shared inference core (settings, environment banner, stage timing, fallback)
├── task1/{Dockerfile,main.py,inference.py,requirements.txt,do_build.sh,do_save.sh,do_test_run.sh}
├── task2/{same}
├── mapping/                  # vessel/location_mapping.json, baked into the image (see mapping/README.md)
├── prepare_models.py / .sh   # collect the 8 checkpoints into model/ -> model.tar.gz
├── make_test_input.py        # raw .nii.gz -> test/input/interf{0,1} (.mha + inputs.json, as on GC)
├── build.sh                  # build both images
├── model/                    # (generated) mounted at /opt/ml/model at runtime
└── test/                     # (generated) local test inputs/outputs
```

## Steps

```bash
# 1) collect the weights from results/ (8 checkpoints -> docker/model/ + model.tar.gz)
bash docker/prepare_models.sh

# 2) generate test inputs (one CTA and one MRA case from fold0-val)
PYTHONPATH=. uv run python docker/make_test_input.py --images-dir /path/to/release/images

# 3) build and run under Grand Challenge conditions (--network none, --gpus all, /opt/ml/model read-only)
bash docker/task1/do_test_run.sh
bash docker/task2/do_test_run.sh

# 4) export for upload (image .tar.gz and model.tar.gz)
bash docker/task1/do_save.sh
bash docker/task2/do_save.sh
```

`model.tar.gz` is uploaded **separately, as the Algorithm's Model**. Task1 and
Task2 use the same `model.tar.gz`.

## Shipped models and settings

`_STEP3` / `_STEP4` / `_KW` in `docker/common/topaneu_infer.py` are the single
source of truth.

| Step | Config | Checkpoint (name inside model.tar.gz) |
|---|---|---|
| 1 vessel ROI | `detection/step1_vessel_roi.yaml` | `step1_rfdetr.pth` |
| 2 vessel seg 2.5D | `segmentation/step2_vessel_2_5d_fold0.yaml` | `step2_vessel25d.ckpt` |
| 3 detector, fold0/1/2 | `segmentation/step3_detector_fold{0,1,2}.yaml` | `step3_gauss_auxcls.ckpt`, `step3_gauss_fold1.ckpt`, `step3_gauss_fold2.ckpt` |
| 4 fine-seg / FP rejector, fold0/1/2 | `segmentation/step4_fineseg_fold{0,1,2}.yaml` | `step4_fineseg.ckpt`, `step4_fineseg_fold1.ckpt`, `step4_fineseg_fold2.ckpt` |

Step3 maps fused by mean, detection threshold 0.50, Step4 maps fused by mean,
fine-seg rejection (weight 0.75, threshold 0.80), auxiliary-presence rescoring
off (the presence head is per-model, so it is only usable with one detector).

## Debugging (sanity-check phase)

Every run logs the following, so a single sanity-check submission answers "does
it run, where does the time go, and if it fails, where":

```
[topaneu] python 3.12.12  torch 2.8.0+cu128  numpy 1.26.4
[topaneu] gpu: ... sm_120 47.3GiB cuda=12.8
[topaneu] model: step1_rfdetr.pth 115 MiB          # contents of /opt/ml/model
[topaneu] MISSING step3: /opt/ml/model/...         # a missing checkpoint is named explicitly
[topaneu] input: modality=MRA shape(x,y,z)=... spacing=... range=[...]
[topaneu] step1 vessel-ROI detection: 3.6s center_mm=[70.0, 94.1, 50.9]
[topaneu] step2 vessel seg: 7.4s n_labels=34
[topaneu] step3 aneurysm heatmap: 4.3s n_models=3 peak=0.926
[topaneu] step4 fine-seg + labelling: 0.4s blobs=1 locations=[36]
[topaneu] total: 24.2s (budget is 12 min/case)
[topaneu] wrote .../output.mha: shape=(229,496,380) dtype=uint8 labels=[36] fg_voxels=271 (OK)
```

- **Fallback on failure**: if the pipeline raises, the traceback is printed, a
  valid but empty output (Task1 `[]` / Task2 all-zero mask) is written and the
  process exits 0 — a finished job tells you more during the sanity-check phase
  than a dead one. Build with `TOPANEU_STRICT=1` in the Dockerfile to make
  failures fatal. Under the official rules failed runs do not count against the
  submission limit, so for the final submission choose depending on whether a
  zero-score completed run is preferable.
- **Offline**: `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are baked in. Grand
  Challenge runs with `--network none`, so a dependency that tries to reach the
  hub fails immediately with a traceback instead of burning the 12 minutes on
  retries. RF-DETR only loads our own weights (the `Using a different number of
  positional encodings...` warning is harmless) and `load_model` sets
  `encoder_weights=None`, so a normal run never touches the network.

## Important notes

- **Always ship the label mappings.** `scripts/anatomical_label.py` reads
  `vessel_mapping.json` / `location_mapping.json` from `$TOPANEU_DEPLOYMENT` at
  import time. The data release is not in the image, so the two files in
  `docker/mapping/` are baked in and the Dockerfile sets
  `TOPANEU_DEPLOYMENT=/opt/app/mapping`. Details in [`mapping/README.md`](mapping/README.md).
- **Location ids are 0-52** (2026-08 release). The first release used 0-50 with
  a different assignment of 47-52; baking the old mapping produces ids that look
  valid but are all shifted.
- **Weights are not baked into the image.** `model.tar.gz` is uploaded as a
  separate Model and extracted to `/opt/ml/model` at runtime;
  `topaneu_infer.py` refers to the fixed file names listed above.
- **Checkpoints are stripped to their state_dict** (`prepare_models.py`). That
  more than halves their size and removes the pickled Lightning objects, so the
  container does not need pytorch-lightning. The RF-DETR `.pth` is copied as is
  because rfdetr's own loader parses its structure.
- **torch**: base image `pytorch:2.9.1-cuda12.6` (the same as the official
  template). Training used torch 2.8 / cu128; only tensors are stored, so the
  checkpoints load across versions. torch/numpy are deliberately not listed in
  `requirements.txt` (pip would replace them with a CPU or incompatible build).
- **Geometry**: the output `.mha` copies origin/spacing/direction from the input
  (`CopyInformation`). The evaluator requires `pred.shape == gt.shape`; the stock
  template drops the geometry.

## Verification

Built and run with `do_test_run.sh` (docker 26.1.3, A100) on one CTA and one MRA
case from fold0-val, Task1 and Task2, all with `--network none`:

| | CTA `center2_ct_107` | MRA `center1_mr_113` |
|---|---|---|
| GT / prediction | 26 / `[26]` | 36 / `[36]` |
| Task2 output | uint8, within 0-52, shape/spacing/origin/direction equal to the input | same |

> **Some bugs only show up inside the container.** Once, although
> `requirements.txt` asks for opencv-python-headless, rfdetr's dependencies also
> pulled in plain opencv-python, and `import rfdetr` failed for lack of
> `libGL`/`libxcb`. The host has those libraries, so the "without docker" run
> below passed, and with `TOPANEU_STRICT=0` the fallback wrote an empty
> prediction and exited 0 — on Grand Challenge this looks like a successful run
> with all metrics near zero. **Always run the real container before submitting.**

### Running without docker

The entry point (`main.py` → `inference.py` → `topaneu_infer.py` →
`pipeline_core`) can also be run from the repository root (for debugging only;
for the reason above it does not replace testing the container):

```bash
CUDA_VISIBLE_DEVICES=0 \
TOPANEU_APP=$PWD TOPANEU_MODEL_DIR=$PWD/docker/model \
TOPANEU_INPUT=$PWD/docker/test/input/interf1 TOPANEU_OUTPUT=/path/to/out \
PYTHONPATH=.:docker/common:docker/task2 uv run python docker/task2/main.py
```

`TOPANEU_APP` / `TOPANEU_MODEL_DIR` / `TOPANEU_INPUT` / `TOPANEU_OUTPUT` are
local overrides only; on Grand Challenge they are unset and `/opt/app`,
`/opt/ml/model`, `/input` and `/output` are used.
