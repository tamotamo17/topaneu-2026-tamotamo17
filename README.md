# Cerebral aneurysm detection and anatomical localisation — TopAneu-26

Source code of our submission to the [TopAneu-26 challenge](https://topaneu-26.grand-challenge.org/):
detection, anatomical location classification (Task1) and location-labelled
segmentation (Task2) of intracranial aneurysms in CTA and MRA.

The repository contains everything needed to retrain the submitted models from
the challenge training data and to build the two Grand Challenge containers.
The pipeline script reproduces the submitted training procedure — models,
cross-validation split, and which detector produced which training data
([docs/reproducibility.md](docs/reproducibility.md)).

## Method

```
raw CTA / MRA (.nii.gz or .mha, native spacing)
 │
 ├─ Step1  vessel-ROI detection ─ RF-DETR nano on coronal + sagittal MIPs of the
 │         0.625 mm volume (modality-aware projection) → two 2D boxes → 3D ROI centre
 ├─ crop   ROI centre ±60 mm resampled from the raw image at 0.4 mm → 300³ cube
 ├─ Step2  37-class vessel segmentation ─ 2.5D U-Net (EfficientNet-B3, 5 slices)
 ├─ Step3  aneurysm detection ─ 3D SegResNet on [image | vessel map] regressing a
 │         Gaussian centre heatmap (σ 2.5 mm) + patch-level presence head;
 │         3-fold ensemble (mean); blob centroids = candidate aneurysms
 ├─ Step4  fine aneurysm shape ─ 3D U-Net on 64³ crops around each blob; replaces
 │         the blob with the predicted shape and rejects blobs with weak evidence
 │         (3-fold ensemble, mean probability)
 ├─ label  anatomical location ─ rule-based: vessel classes around the blob
 │         (+ distances to thin branches such as Pcom/AChA/OA) → one of 52 locations
 └─ uncrop back to the input grid
         → Task1: list of location ids   Task2: uint8 mask, values 0-52
```

Design choices, in short:

- **Detect first, label second.** The location classes are too rare to learn
  directly (1–3 training examples for many of the 52 classes); a learned
  location classifier stayed below the anatomical rule labeller
  (`scripts/anatomical_label.py`), so the detector is location-agnostic and the
  location comes from the predicted vessel map.
- **Gaussian centre heatmap instead of a filled sphere** as the detection target
  (coordinate FROC 0.679 → 0.821 and Task1 MCC 0.054 → 0.094 on fold0-val in our
  ablation).
- **Every stage consumes the previous stage's predictions, not the ground truth**,
  so downstream models are trained on the error distribution they see at test time.
- **Step3 is warm-started** from a 3D vessel-segmentation SegResNet of the same fold.
- **Step4 doubles as a false-positive filter**: its training crops include the
  detector's false positives (all-background targets), and a blob is dropped
  when `det_peak^0.25 · fineseg_peak^0.75 < 0.80`.

## Results

Measured on the challenge training data with 5-fold cross-validation
(folds stratified by modality, seed 42). The submitted ensemble itself cannot be
evaluated offline: folds 1 and 2 were trained on fold0's validation cases and
vice versa, so no held-out case is left for a fold0+1+2 ensemble.

Per-stage validation (best checkpoint of each submitted model):

| Stage | Metric | fold0 | fold1 | fold2 |
|---|---|---|---|---|
| Step1 vessel ROI | mAP@50 | 0.955 | – | – |
| Step2 vessel seg (2.5D) | mean Dice (36 vessel classes) | 0.779 | – | – |
| Step3 detector | val Dice of the heatmap | 0.498 | 0.596 | 0.583 |
| Step4 fine shape | crop Dice | 0.554 | 0.432 | 0.597 |

End to end, out-of-fold over the validation cases of folds 0–2 (251 cases),
single-fold models, detection threshold 0.50 and fine-seg rejection 0.80, scored
with the official evaluator (`scripts/official_eval/`):

| Precision | Recall | F1 | MCC | Dice | VolSim | HD95 |
|---|---|---|---|---|---|---|
| 0.459 | 0.427 | 0.394 | 0.457 | 0.185 | 0.219 | 203.4 |

## Installation

Python 3.12 and a CUDA GPU. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # creates .venv from uv.lock (torch 2.8 / CUDA 12.8 wheels)
uv sync --extra dev     # + pytest / ruff
PYTHONPATH=. uv run pytest tests -q
```

The official evaluator in `scripts/official_eval/` additionally needs
[TopBrain_Eval_Metrics](https://github.com/CoWBenchmark/TopBrain_Eval_Metrics)
(it pins an older SimpleITK, so install it without its dependencies):

```bash
uv pip install --no-deps "git+https://github.com/CoWBenchmark/TopBrain_Eval_Metrics.git@03ba48e47790609317916bacfea8904b596cd888"
```

## Data

The challenge data are distributed by the organisers (see the challenge
website). We use the 2026-08-17 release (416 cases). Unpack it, unchanged, into
`data/raw/topaneu/` of this repository:

```
data/raw/topaneu/
├── images/<case>_0000.nii.gz                 # CTA or MRA volume
├── vessel_masks/<case>.nii.gz                # 37-class vessel labels (0 = background)
├── location_masks/<case>.nii.gz              # aneurysm voxels labelled with their location id (0-52)
├── vessel_mapping.json
└── location_mapping.json
```

This is the default location used by `scripts/run_pipeline.sh` and
`docker/make_test_input.py`. If the release lives elsewhere, either pass its
path (`RAW=/path/to/release`, `--images-dir /path/to/release/images`) or
symlink it there (`ln -s /path/to/release data/raw/topaneu`). The release is
only read; all derived data are written under `data/` (see
[Generated data](#generated-data)).

Only these files are read; other files in the release (`location_jsons/`,
`type_masks/`, ...) are not used. Case ids must keep the release naming
(`topaneu_center<k>_<ct|mr>_<nnn>`): the modality is taken from the `_ct_` /
`_mr_` part, both for the modality-stratified folds and for the
modality-specific projection in Step1.

The 5-fold split is fixed in `splits/folds.yaml` (validation cases per fold)
and is the split all submitted models were trained with. The fold files are
built from it, so the case IDs of the release must match it exactly; the script
stops with an error otherwise. The id mappings are also bundled in
`docker/mapping/` for the containers.

## Training (reproducing the submission)

```bash
RAW=/path/to/release GPU=0 bash scripts/run_pipeline.sh     # RAW defaults to data/raw/topaneu
# derived data (~160 GB) and checkpoints on another disk:
DATA_DIR=/big/disk/topaneu/data RESULTS_DIR=/big/disk/topaneu/results \
    RAW=/path/to/release GPU=0 bash scripts/run_pipeline.sh
```

The script runs every step in order — preprocessing, folds, the 11 training
runs (1 + 3 + 3 + 3 + 1 helper; 2.5D vessel models for folds 1/2 are also trained
because their predictions are the fold1/fold2 detector inputs), the per-fold
evaluation, and finally collects the 8 submitted checkpoints into
`docker/model/`. Derived data are written to `data/`, runs to `results/`.
Expect several days on a single GPU; each block can be commented out to resume.

Part of the submitted models were trained on the previous data release (417 cases,
before the organisers' label corrections); this repository trains everything on
the current release with the same split. The differences are listed in
[docs/reproducibility.md](docs/reproducibility.md#data-release).

Every model is defined by one YAML file in `configs/`; a single stage can be
trained on its own, e.g.

```bash
PYTHONPATH=. uv run python scripts/train_segmentation.py \
    --config configs/segmentation/step3_detector_fold0.yaml --fold 0 --gpu 0
```

| Config | Model | Shipped |
|---|---|---|
| `detection/step1_vessel_roi.yaml` | Step1 RF-DETR nano, fold0 | ✓ |
| `segmentation/step2_vessel_2_5d_fold0.yaml` | Step2 2.5D vessel seg, fold0 | ✓ |
| `segmentation/step2_vessel_2_5d_fold1_2.yaml` | Step2 2.5D vessel seg, folds 1/2 (detector inputs) | |
| `segmentation/step2_vessel_3d.yaml` | Step2 3D vessel seg, folds 0–2 (warm starts) | |
| `segmentation/step3_detector_fold{0,1,2}.yaml` | Step3 detector | ✓ ×3 |
| `segmentation/step3_blobgen_fold0.yaml` | Step3 helper (generates the fold0 Step4 crops) | |
| `segmentation/step4_fineseg_fold{0,1,2}.yaml` | Step4 fine shape / FP rejector | ✓ ×3 |

### Paths

- **Run everything from the repository root.** All paths in `configs/` and in
  `scripts/run_pipeline.sh` are relative to it (`data/...`, `results/...`).
- **Only the raw release is passed in** (`RAW`, default `data/raw/topaneu`).
  Everything else is generated by the pipeline at fixed locations.
- **Disk space.** A full run writes about **160 GB** of derived data and about
  6 GB of checkpoints (measured on the full training set), usually too much for
  the disk holding the code. Put them elsewhere with
  `DATA_DIR=/big/disk/topaneu/data RESULTS_DIR=/big/disk/topaneu/results bash scripts/run_pipeline.sh`:
  the script turns `data/` and `results/` into symlinks to those directories, so
  the paths in `configs/` stay valid (it refuses to replace an existing real
  directory). Making the symlinks by hand is equivalent. The raw release can stay
  where it is (`RAW=...`); it is only read. The only large files written inside
  the repository are the collected weights, `docker/model/` and
  `docker/model.tar.gz` (about 1 GB together).
- **Fold files store absolute paths.** `folds*/fold{N}.yaml` list the resolved
  path of every image and mask. If you move `data/` after they were created,
  re-run the fold steps of `run_pipeline.sh` (the split itself comes from
  `splits/folds.yaml` and does not change).

### Generated data

What each step writes, and which config reads it. To train a single stage, the
data it reads must already exist.

| Path | Produced by (run_pipeline.sh block) | Content | Read by |
|---|---|---|---|
| `data/vessel_lr/{images,masks}/` | STEP0 `preprocess_vessel_segmentation.py` | 0.625 mm volumes | Step1 data generation |
| `data/vessel_lr/folds/` | STEP0 `make_vessel_folds.py --assignment splits/folds.yaml` | fold files | Step1 data generation |
| `data/step1_vessel_roi/{det,coco}/` | STEP1 projection + COCO conversion | MIP images + boxes | `step1_vessel_roi.yaml` |
| `data/step1_vessel_roi/roi/roi_meta/` | STEP1 `infer_vessel_roi_detection.py` | predicted ROI per case | CROP |
| `data/vessel_hr/{images,masks,crop_meta}/` | CROP (vessel masks) | 300³ crops at 0.4 mm | Step2 |
| `data/vessel_hr/folds/` | CROP folds | fold files in crop space | `step2_*.yaml` |
| `data/vessel_hr/pred_fold{0,1,2}/pred/` | STEP2 `infer_segmentation_2_5d.py` | predicted vessel maps, all cases | `step3_*_fold{N}.yaml` (`aux_dir`), Step4 crops |
| `data/aneurysm_hr/{images,masks,crop_meta,folds}/` | CROP (location masks) + folds | 300³ crops with location labels | Step3/Step4 data generation |
| `data/aneurysm_hr/sphere_g25/`, `folds_sphere_g25/` | STEP3 `make_sphere_targets.py` / `make_sphere_folds.py` | Gaussian heatmap targets | `step3_*.yaml` |
| `data/aneurysm_fineseg/blob_manifest_fold{N}.json`, `fold{N}_sz64/` | STEP4 `gen_blob_labels.py` / `make_fineseg_crops.py` | 64³ crops around detections | `step4_fineseg_fold{N}.yaml` |
| `results/<stage>/<exp_name>/fold{N}/` | every training run | checkpoints (`best-*.ckpt`), saved `config.yaml`, metrics | warm start (`init_from_ckpt`), later steps, `docker/prepare_models.py` |

## Inference

The submission runs as two Grand Challenge containers (Task1 / Task2) that
share one inference core, `docker/common/topaneu_infer.py`. Building, local
testing (`--network none`, as on Grand Challenge) and export are described in
[docker/README.md](docker/README.md):

```bash
bash docker/prepare_models.sh                          # results/ → docker/model/ + model.tar.gz
PYTHONPATH=. uv run python docker/make_test_input.py   # Grand Challenge-style test inputs
bash docker/task2/do_test_run.sh                       # build + run like Grand Challenge
```

For a single raw volume without docker, `scripts/infer_pipeline_raw.py` runs the
same pipeline core (single-detector form) from the command line.

## Repository layout

```
configs/            one YAML per trained model (see the table above)
splits/folds.yaml   the 5-fold cross-validation split
scripts/
  run_pipeline.sh   end-to-end reproduction of the submission
  train_*.py        training entry points (detection / segmentation)
  pipeline_core.py  full inference pipeline shared with the containers
  anatomical_label.py  rule-based anatomical location labeller
  ...               preprocessing, cropping, target generation, evaluation
  official_eval/    official TopAneu-26 evaluation code (vendored)
src/
  common/           registry base class, seeding
  detection/        RF-DETR wrapper, COCO dataset, MIP projection
  segmentation/     models, losses, datasets, Lightning modules, inference helpers
docker/             Grand Challenge containers (Task1 / Task2)
docs/               reproducibility notes
tests/              unit and smoke tests
```

Models, losses, datasets and Lightning modules are looked up by the `type`
field of the YAML through small registries (`src/*/registries/`), so the
training scripts never change between experiments.

## License

[Apache License 2.0](LICENSE). The files in `scripts/official_eval/` are copied
from the challenge organisers' repository
([Bangulli/TopAneu-26](https://github.com/Bangulli/TopAneu-26), commit `660da7a`)
and remain under their original terms.

## Acknowledgements

Built on [PyTorch Lightning](https://lightning.ai/), [MONAI](https://monai.io/),
[segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
and [RF-DETR](https://github.com/roboflow/rf-detr). The MIP-based vessel-ROI
detection follows ideas from top solutions of the RSNA 2025 Intracranial
Aneurysm Detection challenge.
