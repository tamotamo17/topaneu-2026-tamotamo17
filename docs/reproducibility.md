# Reproducibility notes

This document records how the submitted models were produced and how the code in
this repository was checked against the submission.

## What was submitted

The final submission (Grand Challenge, Task1 and Task2; build tag `ensemble-v1`)
consists of one image per task and one model tarball shared by both:

| Artifact | SHA-256 |
|---|---|
| `topaneu-26-task1_ensemble-v1.tar.gz` (image) | `b1899abaeea577b1c26650c4ea20d10b6b83a997170dd8b4f3c51e69f7358c2e` |
| `topaneu-26-task2_ensemble-v1.tar.gz` (image) | `162786b8ac9ec6ee7818ede5baeec0b0c8c4ee66821b9b84ec369cd6c99b04e9` |
| `model_task1_ensemble-v1.tar.gz` (weights) | `c8316fa97ae545ba3dfb7393500543d02801a65faae8964a0fd9796506119b61` |
| `model_task2_ensemble-v1.tar.gz` (weights) | `35c377747c5b62d9f4067de9e47bd7c1f6556b04df54207f7c2346a5898bdf33` |

The two model tarballs hold identical files (they differ only in the gzip
timestamp); the two images differ only in their task-specific `main.py` /
`inference.py`. The model tarball ships 8 checkpoints:

| Container file | Model | Config |
|---|---|---|
| `step1_rfdetr.pth` | Step1 vessel-ROI detector, fold0 | `configs/detection/step1_vessel_roi.yaml` |
| `step2_vessel25d.ckpt` | Step2 2.5D vessel seg, fold0 | `configs/segmentation/step2_vessel_2_5d_fold0.yaml` |
| `step3_gauss_auxcls.ckpt` | Step3 detector, fold0 | `configs/segmentation/step3_detector_fold0.yaml` |
| `step3_gauss_fold1.ckpt` | Step3 detector, fold1 | `configs/segmentation/step3_detector_fold1.yaml` |
| `step3_gauss_fold2.ckpt` | Step3 detector, fold2 | `configs/segmentation/step3_detector_fold2.yaml` |
| `step4_fineseg.ckpt` | Step4 fine shape / FP rejector, fold0 | `configs/segmentation/step4_fineseg_fold0.yaml` |
| `step4_fineseg_fold1.ckpt` | Step4, fold1 | `configs/segmentation/step4_fineseg_fold1.yaml` |
| `step4_fineseg_fold2.ckpt` | Step4, fold2 | `configs/segmentation/step4_fineseg_fold2.yaml` |

Inference settings (`docker/common/topaneu_infer.py`): Step3 probability maps of
the three folds averaged, detection threshold 0.50, minimum blob size 30 voxels,
Step4 probabilities of the three folds averaged, fine-seg rejection with weight
0.75 and threshold 0.80, auxiliary-presence rescoring off.

## Training graph

Each arrow is a training or inference run in `scripts/run_pipeline.sh`.

```
release ──► Step1 (fold0) ──► ROI ──► crops (vessel masks + location masks)
                                          │
       ┌──────────────────────────────────┼───────────────────────────────┐
       │ fold0                            │ fold1                         │ fold2
 Step2 2.5D fold0 (shipped)        Step2 2.5D fold1               Step2 2.5D fold2
   └► vessel map pred_fold0          └► pred_fold1                  └► pred_fold2
 Step2 3D fold0 ─ warm start ┐     Step2 3D fold1 ─┐              Step2 3D fold2 ─┐
 Step3 detector fold0 ◄──────┘     Step3 detector fold1 ◄─┘       Step3 detector fold2 ◄─┘
 Step3 helper fold0 ─► crops       └► crops                       └► crops
 Step4 fold0                       Step4 fold1                    Step4 fold2
```

All runs use the same 5-fold split, `splits/folds.yaml`.

Points that are easy to get wrong when reproducing by hand:

- **Each fold's detector uses its own fold's vessel map** as auxiliary input
  during training (`data/vessel_hr/pred_fold{N}`, produced by that fold's 2.5D
  model on every case). At inference time all three detectors receive the
  vessel map of the shipped fold0 2.5D model.
- **The fold0 Step4 crops come from a different detector** than the shipped
  fold0 detector: the helper `step3_blobgen_fold0.yaml` (same detector without
  the presence head). The fold1/fold2 crops come from the shipped fold1/fold2
  detectors.
- The 2.5D vessel models for folds 1/2 ran 55 epochs with a cosine schedule of
  `T_max: 90`, i.e. the learning rate never reached zero. This is how they were
  trained and is kept as is.
- `step3_blobgen_fold0` was trained before the heatmap focal loss was moved from
  the Lightning module into the `loss:` section. The config uses the current
  form (`heatmap_focal`, `gamma 2.0`, `fg_weight 50`, `normalize: mean`), which
  computes exactly the same loss.

## Data release

The organisers revised the training data during the challenge. This repository
trains every model on the 2026-08-17 release (416 cases). Part of the submitted
models were trained on the preceding 2026-08 release (417 cases):

| Submitted model | Data it was trained on |
|---|---|
| Step1, Step2 2.5D fold0, Step2 3D folds 0–2 | 2026-08 release (417 cases) |
| Step3 detector fold0, Step3 helper fold0, Step4 fold0 | 2026-08 release (417 cases) |
| Step2 2.5D folds 1/2, Step3 detectors folds 1/2, Step4 folds 1/2 | 2026-08-17 release (416 cases) |

What differs between the two releases (compared file by file):

- images and vessel masks: identical for all 416 shared cases;
- location masks: the location ids of 17 aneurysms in 8 cases were corrected
  (mostly left/right swaps within a case; the aneurysm voxels are unchanged),
  2 missing aneurysms were added and one mask was corrected;
- `topaneu_center1_mr_150` was removed (it belonged to fold3, i.e. to the
  training set of folds 0–2);
- `vessel_mapping.json` / `location_mapping.json`: identical.

The split is unaffected: `splits/folds.yaml` is the original 417-case split with
the removed case left out. (Regenerating a stratified split from the 416 cases
would move 362 of them to a different fold, which is why the assignment is
shipped instead.) Training on the current release therefore differs from the
submission only by that one training case and by the corrected labels of the
models in the first two rows.

## How the repository was checked against the submission

1. **Configs.** Every config in `configs/` was generated from the `config.yaml`
   that the training script saved next to the corresponding checkpoint, and
   compared key by key after parsing. The only differences are data/output paths
   (made relative to the repository, one data directory instead of one per
   release), experiment names, and the warm-start checkpoint, which is now a
   glob (`.../fold{N}/best-*.ckpt`) because the file name embeds the score.
2. **Split.** `scripts/make_vessel_folds.py --assignment splits/folds.yaml`
   reproduces the fold files the fold1/fold2 models were trained with exactly
   (same cases, same order), and those equal the fold0 models' split minus the
   removed case.
3. **Training graph.** The inputs of each run were identified from the saved
   configs and by regenerating intermediate data from the candidate models:
   the vessel maps (`pred_fold{0,1,2}`) agree with a re-run of the respective
   2.5D model on 99.9999% of voxels, the blob manifests used to cut the Step4
   crops are reproduced by the detectors listed above to within one or two
   voxels (GPU non-determinism), and the Step4 crops match the vessel map of
   their own fold exactly.
4. **Weights.** The 8 files inside the submitted `model.tar.gz` are
   identical (`torch.equal` on every tensor; byte-identical for RF-DETR) to the
   checkpoints the configs above produced.
5. **Inference code.** The code inside the submitted image matches this
   repository apart from the renamed config files and their path fields, which
   inference never reads. Running the container entry point
   (`docker/task2/main.py`) from this repository with the submitted weights
   gives voxel-identical Task2 masks to the original code on the CTA and MRA
   test cases.

Retraining from scratch will not give bit-identical weights: training ran with
`deterministic: false` (cuDNN autotuning, non-deterministic scatter/atomics), so
expect small run-to-run differences in the scores.
