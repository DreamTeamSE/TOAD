#!/bin/bash
#SBATCH --job-name=nnunet-train
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# nnUNetv2_train on a single 2D config/fold (full 5-fold CV is a later
# stretch goal). Team-confirmed run parameters: 250 epochs, hpg-turin
# partition, 1 L4 GPU, one day of walltime.
#
# Resource allocation below is sized from a real smoke test (job 46886051,
# nnUNetTrainer_1epoch, 2026-10-20):
#   --cpus-per-task=8 : measured avg CPU utilization was only ~20%, well
#                        within 8 cores for this dataset's small 2D patches.
#   --mem=32gb         : measured peak RSS was ~16.9GB; 32gb keeps ~2x headroom.
#   --time=24:00:00    : measured epoch time was 77.92s; 250 epochs ≈ 5.4h,
#                        well under the one-day budget given.
#
# No CUDA OOM: batch_size=14 @ patch_size (192,256) fit comfortably in the
# L4's 24GB VRAM.
#
# TRAINER/FOLD are overridable so this one script also covers a cheap smoke
# test if the pipeline ever needs re-validating after a change:
#   sbatch --time=00:30:00 --export=TRAINER=nnUNetTrainer_1epoch hpg/nnunet/train_nnunet.slurm   # smoke test
#   sbatch hpg/nnunet/train_nnunet.slurm                                                          # real run (250 epochs)
#
# Requires prepare_data.slurm to have already run successfully (reads
# $nnUNet_preprocessed/Dataset001_Rat/, does not touch $TOAD_DATA_ROOT).

set -euo pipefail

DATASET_ID=1
DATASET_NAME=Rat
DATASET_DIR_NAME="Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}"
CONFIGURATION=2d
FOLD="${FOLD:-0}"
TRAINER="${TRAINER:-nnUNetTrainer_250epochs}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/blue/dream_team/$USER/seko_storage}"  # must match prepare_data.slurm's DATA_ROOT -- that's where nnUNet_preprocessed was populated
mkdir -p "$REPO_ROOT/logs"

export nnUNet_raw="$DATA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$DATA_ROOT/nnUNet_results"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nnunet

cd "$REPO_ROOT"

echo "Dataset             : $DATASET_DIR_NAME"
echo "Configuration       : $CONFIGURATION"
echo "Fold                : $FOLD"
echo "Trainer             : $TRAINER"
echo "nnUNet_preprocessed : $nnUNet_preprocessed"
echo "nnUNet_results      : $nnUNet_results"
echo

echo "=== Training ==="
nnUNetv2_train "$DATASET_ID" "$CONFIGURATION" "$FOLD" -tr "$TRAINER"

echo
echo "Done. Checkpoint + logs under \$nnUNet_results/$DATASET_DIR_NAME/${TRAINER}__nnUNetPlans__${CONFIGURATION}/fold_${FOLD}/"
