#!/usr/bin/env bash
# =============================================================================
# Reproduce the TopAneu-26 submission end to end:
#   raw data release -> every trained model -> the weights shipped in the container.
#
#   STEP0  resample to 0.625 mm + 5-fold split
#   STEP1  vessel-ROI detection (RF-DETR on 2D MIPs)            -> 1 model  (fold0)
#   CROP   ROI centre +/-60 mm, 0.4 mm isotropic, 300^3 cube
#   STEP2  37-class vessel segmentation, 2.5D and 3D            -> 3 + 3 models
#   STEP3  aneurysm detector (3D Gaussian heatmap + presence)   -> 3 models + 1 helper
#   STEP4  fine aneurysm shape / false-positive rejector        -> 3 models
#   EVAL   per-fold validation with the anatomical labeller
#   PACK   collect the 8 submitted checkpoints for the container
#
# The submission is an ensemble over folds 0, 1 and 2 for Step3 and Step4; Step1
# and Step2 ship their fold0 model only. See docs/reproducibility.md for how each
# config here maps to the checkpoint that was actually submitted.
#
# Data: the 2026-08-17 release of the training data (416 cases). The 5-fold
# split is read from splits/folds.yaml (the submission's split) instead of being
# regenerated. See docs/reproducibility.md for how this relates to the data the
# submitted models were trained on.
#
# Usage:
#   RAW=/path/to/release GPU=0 bash scripts/run_pipeline.sh
#   DATA_DIR=/big/disk/topaneu/data RESULTS_DIR=/big/disk/topaneu/results \
#       RAW=/path/to/release GPU=0 bash scripts/run_pipeline.sh
#
# Disk space: a full run writes about 160 GB of derived data (measured: 0.625 mm
# volumes 14 GB, vessel crops + predictions 65 GB, aneurysm crops + heatmap targets
# 74 GB, Step4 crops 1.5 GB) and about 6 GB of checkpoints. DATA_DIR / RESULTS_DIR
# put them on another disk: data/ and results/ in the repository become symlinks to
# those directories, so every path in configs/ keeps working unchanged.
#
# Expected release layout: images/<case>_0000.nii.gz, vessel_masks/, location_masks/
# (default location data/raw/topaneu). All derived data go to data/ and all runs to results/ (both
# relative to the repository root). Training everything takes several days on
# one GPU; to resume, comment out the blocks that have already finished.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

PY=${PY:-uv run python}
GPU=${GPU:-0}

# data/ and results/ -> DATA_DIR / RESULTS_DIR (see "Disk space" above).
link_dir() {
  local name=$1 target=$2
  [ -n "$target" ] || return 0
  if [ -L "$name" ]; then
    if [ "$(readlink -f "$name")" != "$(readlink -f "$target")" ]; then
      echo "$name is already a symlink to $(readlink -f "$name"), not $target" >&2; exit 1
    fi
  elif [ -e "$name" ]; then
    echo "$name already exists as a real directory; move it to $target or unset the variable" >&2; exit 1
  else
    mkdir -p "$target"
    ln -s "$(readlink -f "$target")" "$name"
  fi
  echo "$name -> $(readlink -f "$name")"
}
link_dir data "${DATA_DIR:-}"
link_dir results "${RESULTS_DIR:-}"
RAW=${RAW:-data/raw/topaneu}
SPLIT=splits/folds.yaml

# Vessel/location id mappings used by the labeller and the evaluator.
export TOPANEU_DEPLOYMENT=docker/mapping

LR=data/vessel_lr                     # 0.625 mm whole-head volumes + split
DET=data/step1_vessel_roi             # projections, COCO dataset, predicted ROIs
VHR=data/vessel_hr                    # 300^3 crops with vessel masks
HR=data/aneurysm_hr                   # 300^3 crops with location masks
FS=data/aneurysm_fineseg              # Step4 crops
CFG=configs/segmentation

# The single best checkpoint of a run (names embed epoch and score).
best_ckpt() {
  local hits=( "$1"/best-*.ckpt )
  if [ "${#hits[@]}" -ne 1 ] || [ ! -f "${hits[0]}" ]; then
    echo "expected exactly one best-*.ckpt in $1, got: ${hits[*]}" >&2
    exit 1
  fi
  echo "${hits[0]}"
}


# =============================================================================
# STEP0  raw .nii.gz -> 0.625 mm isotropic .npy, and the 5-fold split
# =============================================================================
echo "########## STEP0  resample + 5-fold split (fixed: $SPLIT) ##########"
$PY scripts/preprocess_vessel_segmentation.py \
    --data-root "$RAW" --out-dir "$LR" \
    --target-spacing 0.625 0.625 0.625 \
    --mask-source vessel --label-mode multiclass
$PY scripts/make_vessel_folds.py --data-root "$LR" --out-dir "$LR/folds" --assignment "$SPLIT"


# =============================================================================
# STEP1  vessel-ROI detection
# =============================================================================
echo "########## STEP1  coronal/sagittal MIPs + vessel bbox -> COCO -> RF-DETR nano ##########"
$PY scripts/make_vessel_roi_detection_data.py \
    --images-dir "$LR/images" --masks-dir "$LR/masks" \
    --fold-yaml "$LR/folds/fold0.yaml" --out-dir "$DET/det"
$PY scripts/convert_annotations_to_coco.py \
    --csv "$DET/det/annotations.csv" --images-root "$DET/det" --out "$DET/coco"
# RF-DETR requires a non-empty test split; as in the submitted run, it is a copy of valid.
mkdir -p "$DET/coco/test" && cp "$DET"/coco/valid/* "$DET/coco/test/"
$PY scripts/train_detection.py --config configs/detection/step1_vessel_roi.yaml --fold 0 --gpu "$GPU"

STEP1_CKPT=results/step1_vessel_roi/step1_vessel_roi_rfdetr_nano/fold0/checkpoint_best_total.pth
echo "########## STEP1  fuse the two 2D boxes into a 3D ROI for every case ##########"
$PY scripts/infer_vessel_roi_detection.py \
    --config configs/detection/step1_vessel_roi.yaml --ckpt "$STEP1_CKPT" \
    --images-dir "$LR/images" --out-dir "$DET/roi" --gpu "$GPU"


# =============================================================================
# CROP  ROI centre +/-60 mm resampled from the raw image at 0.4 mm (300^3).
#   60 mm covers every aneurysm in the training data except one (it needs 61.3 mm).
#   crop_meta/ records what scripts/uncrop_to_original.py needs to map back.
# =============================================================================
echo "########## CROP  vessel crops + location crops ##########"
$PY scripts/make_highres_vessel_crop.py --data-root "$RAW" --roi-meta-dir "$DET/roi/roi_meta" \
    --center-spacing 0.625 --half-mm 60 --target-spacing 0.4 --mask-source vessel --out-dir "$VHR"
$PY scripts/make_highres_vessel_crop.py --data-root "$RAW" --roi-meta-dir "$DET/roi/roi_meta" \
    --center-spacing 0.625 --half-mm 60 --target-spacing 0.4 --mask-source location --out-dir "$HR"

echo "########## CROP  fold files in crop space (same fixed split) ##########"
$PY scripts/make_vessel_folds.py --data-root "$VHR" --out-dir "$VHR/folds" --assignment "$SPLIT"
$PY scripts/make_vessel_folds.py --data-root "$HR"  --out-dir "$HR/folds"  --assignment "$SPLIT"


# =============================================================================
# STEP2  37-class vessel segmentation
#   2.5D: fold0 is the shipped model. Each fold's model predicts EVERY case;
#         those maps are the vessel input of the same fold's Step3 detector.
#   3D:   SegResNet per fold, used only as the warm start of that fold's detector.
# =============================================================================
echo "########## STEP2  2.5D vessel seg: fold0 (shipped) ##########"
$PY scripts/train_segmentation.py --config $CFG/step2_vessel_2_5d_fold0.yaml --fold 0 --gpu "$GPU"
$PY scripts/infer_segmentation_2_5d.py --config $CFG/step2_vessel_2_5d_fold0.yaml \
    --ckpt "$(best_ckpt results/step2_vessel_2_5d/step2_vessel_2_5d_e50/fold0)" \
    --images-dir "$VHR/images" --out-dir "$VHR/pred_fold0" --gpu "$GPU"

for F in 1 2; do
  echo "########## STEP2  2.5D vessel seg: fold$F (vessel input for the fold$F detector) ##########"
  $PY scripts/train_segmentation.py --config $CFG/step2_vessel_2_5d_fold1_2.yaml --fold $F --gpu "$GPU"
  $PY scripts/infer_segmentation_2_5d.py --config $CFG/step2_vessel_2_5d_fold1_2.yaml \
      --ckpt "$(best_ckpt results/step2_vessel_2_5d/step2_vessel_2_5d_e55/fold$F)" \
      --images-dir "$VHR/images" --out-dir "$VHR/pred_fold$F" --gpu "$GPU"
done

for F in 0 1 2; do
  echo "########## STEP2  3D vessel seg (SegResNet): fold$F (warm start for the fold$F detector) ##########"
  $PY scripts/train_segmentation.py --config $CFG/step2_vessel_3d.yaml --fold $F --gpu "$GPU"
done


# =============================================================================
# STEP3  aneurysm detector
#   Target: Gaussian heatmap (sigma 2.5 mm) at each aneurysm centroid; blob
#   centroids of the predicted heatmap are the detections.
# =============================================================================
echo "########## STEP3  Gaussian heatmap targets + their fold files ##########"
$PY scripts/make_sphere_targets.py --masks-dir "$HR/masks" --out-dir "$HR/sphere_g25" \
    --spacing 0.4 --mode gaussian --sigma-mm 2.5
$PY scripts/make_sphere_folds.py --folds-dir "$HR/folds" --sphere-dir "$HR/sphere_g25" \
    --out-dir "$HR/folds_sphere_g25"

for F in 0 1 2; do
  echo "########## STEP3  detector with presence head: fold$F (shipped) ##########"
  $PY scripts/train_segmentation.py --config $CFG/step3_detector_fold$F.yaml --fold $F --gpu "$GPU"
done

echo "########## STEP3  helper detector without presence head: fold0 (generates the fold0 Step4 crops) ##########"
$PY scripts/train_segmentation.py --config $CFG/step3_blobgen_fold0.yaml --fold 0 --gpu "$GPU"


# =============================================================================
# STEP4  fine aneurysm-shape segmentation / false-positive rejector
#   Crops (64^3) around every detected blob (true and false positives) and every
#   missed aneurysm. fold0 uses the helper detector's blobs, folds 1/2 use their
#   own shipped detector's blobs - exactly as the submitted models were trained.
# =============================================================================
for F in 0 1 2; do
  if [ "$F" = "0" ]; then
    DET_CFG=$CFG/step3_blobgen_fold0.yaml
    DET_CKPT=$(best_ckpt results/step3_detector/step3_blobgen_segresnet/fold0)
  else
    DET_CFG=$CFG/step3_detector_fold$F.yaml
    DET_CKPT=$(best_ckpt results/step3_detector/step3_detector_auxcls/fold$F)
  fi
  echo "########## STEP4  blob manifest + 64^3 crops: fold$F ##########"
  $PY scripts/gen_blob_labels.py --config "$DET_CFG" --ckpt "$DET_CKPT" \
      --images-dir "$HR/images" --aux-dir "$VHR/pred_fold$F/pred" --gt-dir "$HR/masks" \
      --fold-yaml "$HR/folds/fold$F.yaml" --out "$FS/blob_manifest_fold$F.json" --gpu "$GPU"
  $PY scripts/make_fineseg_crops.py --manifest "$FS/blob_manifest_fold$F.json" \
      --images-dir "$HR/images" --vessel-dir "$VHR/pred_fold$F/pred" --gt-dir "$HR/masks" \
      --out "$FS/fold${F}_sz64" --size 64

  echo "########## STEP4  train fine-seg / FP rejector: fold$F (shipped) ##########"
  $PY scripts/train_segmentation.py --config $CFG/step4_fineseg_fold$F.yaml --fold $F --gpu "$GPU"
done


# =============================================================================
# EVAL  each fold's (detector, fine-seg) pair on its own validation cases:
#   detect -> anatomical location label -> fine shape -> labelled mask -> metrics.
#   The submitted setting is thr 0.50 with fine-seg rejection (w 0.75, thr 0.80).
#   Metrics come from scripts/eval_topaneu.py, a port of the evaluator computed
#   in crop space. It predates the official evaluator's update (F1 added, HD95
#   no longer normalised), so its numbers are NOT comparable with the
#   leaderboard; the official code is vendored in scripts/official_eval/.
#   The 3-fold ensemble itself has no held-out data left to be evaluated on.
# =============================================================================
for F in 0 1 2; do
  echo "########## EVAL  fold$F validation ##########"
  $PY scripts/infer_label_eval.py \
      --config $CFG/step3_detector_fold$F.yaml \
      --ckpt "$(best_ckpt results/step3_detector/step3_detector_auxcls/fold$F)" \
      --step4-config $CFG/step4_fineseg_fold$F.yaml \
      --step4-ckpt "$(best_ckpt results/step4_fineseg/step4_fineseg_reject/fold$F)" \
      --fineseg-reject-thr 0.80 --fineseg-reject-weight 0.75 --thr 0.5 \
      --images-dir "$HR/images" --aux-dir "$VHR/pred_fold$F/pred" --gt-dir "$HR/masks" \
      --fold-yaml "$HR/folds/fold$F.yaml" --gpu "$GPU"
done


# =============================================================================
# PACK  collect the 8 submitted checkpoints (stripped to state_dict) into
#   docker/model/ + docker/model.tar.gz. Building and testing the images needs
#   docker; see docker/README.md.
# =============================================================================
echo "########## PACK  docker/model/ + docker/model.tar.gz ##########"
$PY docker/prepare_models.py --results-dir results

echo
echo "done."
