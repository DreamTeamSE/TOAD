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
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs nnUNetv2_predict on the held-out test set, then
# src/baselines/nnunet/export_predictions.py to convert nnU-Net's native
# output into the exact docs/DATA_CONTRACT.md format (single-channel 8-bit
# PNG, class IDs 0-4, filename stem matching the input).
#
# TRAINER selects which trained weights to predict with -- nnUNetv2_predict
# builds its model folder path from -tr/-p/-c internally, so this must match
# whatever trainer the checkpoint was actually produced with (it does NOT
# take a checkpoint file path directly). Point this at the smoke-test
# trainer first, then switch to the real run's trainer once that checkpoint
# is ready, for the actual deliverable-quality predictions:
#   sbatch hpg/nnunet/predict_nnunet.slurm                                      # real run (default: nnUNetTrainer_250epochs)
#   sbatch --export=TRAINER=nnUNetTrainer_1epoch hpg/nnunet/predict_nnunet.slurm  # smoke-test checkpoint
#
# CHECKPOINT is just the checkpoint *filename* within that trainer's fold_N/
# directory (e.g. checkpoint_final.pth), not a full path.

set -euo pipefail

DATASET_ID=1
DATASET_NAME=Rat
DATASET_DIR_NAME="Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}"
CONFIGURATION=2d
FOLD="${FOLD:-0}"
TRAINER="${TRAINER:-nnUNetTrainer_250epochs}"
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"

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
echo "Trainer           : $TRAINER"
echo "Checkpoint        : $CHECKPOINT"
echo "imagesTs          : $IMAGES_TS"
echo "Raw predictions   : $PRED_DIR"
echo "Contract-format   : $EXPORT_DIR"
echo

echo "=== Predicting ==="
nnUNetv2_predict -i "$IMAGES_TS" -o "$PRED_DIR" -d "$DATASET_ID" -c "$CONFIGURATION" -f "$FOLD" -tr "$TRAINER" -chk "$CHECKPOINT"

echo
echo "=== Exporting to DATA_CONTRACT.md format ==="
python src/baselines/nnunet/export_predictions.py \
    --manifest-csv "$MANIFEST_CSV" \
    --predictions-dir "$PRED_DIR" \
    --out-dir "$EXPORT_DIR"

echo
echo "Done. Contract-format predictions under $EXPORT_DIR"
