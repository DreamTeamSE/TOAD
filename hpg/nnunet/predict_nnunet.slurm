#!/bin/bash
#SBATCH --job-name=nnunet-predict
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs nnUNetv2_predict on the held-out test set, then
# src/baselines/nnunet/export_predictions.py to convert nnU-Net's native
# output into the exact docs/DATA_CONTRACT.md format (single-channel 8-bit
# PNG, class IDs 0-4, filename stem matching the input).
#
# CHECKPOINT selects which trained weights to predict with. Point this at the
# smoke-test checkpoint first. Re-run against the real run's checkpoint once
# that's ready, for the actual deliverable-quality predictions. Example
# usage and paths:
#   sbatch --export=CHECKPOINT=/path/to/checkpoint_final.pth hpg/nnunet/predict_nnunet.slurm
#   $nnUNet_results/Dataset001_Rat/nnUNetTrainer_1epoch__nnUNetPlans__2d/fold_0/checkpoint_final.pth     (smoke test)
#   $nnUNet_results/Dataset001_Rat/nnUNetTrainer_250epochs__nnUNetPlans__2d/fold_0/checkpoint_final.pth  (real run)

set -euo pipefail

DATASET_ID=1
DATASET_NAME=Rat
DATASET_DIR_NAME="Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}"
CONFIGURATION=2d
FOLD="${FOLD:-0}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a full path to a checkpoint_final.pth -- see header comment above}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/blue/dream_team/$USER/seko_storage}"  # must match prepare_data.slurm/train_nnunet.slurm's DATA_ROOT
mkdir -p "$REPO_ROOT/logs"

export nnUNet_raw="$DATA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$DATA_ROOT/nnUNet_results"

IMAGES_TS="$nnUNet_raw/$DATASET_DIR_NAME/imagesTs"
MANIFEST_CSV="$nnUNet_raw/$DATASET_DIR_NAME/conversion_manifest.csv"
PRED_DIR="$DATA_ROOT/predictions_raw/$DATASET_DIR_NAME"
EXPORT_DIR="$DATA_ROOT/predictions_contract/$DATASET_DIR_NAME"
mkdir -p "$PRED_DIR" "$EXPORT_DIR"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nnunet

cd "$REPO_ROOT"

echo "Dataset           : $DATASET_DIR_NAME"
echo "Checkpoint        : $CHECKPOINT"
echo "imagesTs          : $IMAGES_TS"
echo "Raw predictions   : $PRED_DIR"
echo "Contract-format   : $EXPORT_DIR"
echo

echo "=== Predicting ==="
# TODO: fill in the nnUNetv2_predict call. Reference shape (replace with the
# variables already defined above -- don't hardcode paths):
#   nnUNetv2_predict -i <imagesTs dir> -o <output dir> -d <dataset id> -c <configuration> -f <fold> -chk <checkpoint path>
#
# Run `nnUNetv2_predict -h` once (interactively, before scripting this) to
# see the full flag list and confirm you're not missing anything.

echo
echo "=== Exporting to DATA_CONTRACT.md format ==="
# TODO: once src/baselines/nnunet/export_predictions.py is implemented, call
# it here with:
#   --manifest-csv    $MANIFEST_CSV
#   --predictions-dir $PRED_DIR
#   --out-dir         $EXPORT_DIR

echo
echo "Done. Contract-format predictions under $EXPORT_DIR"
