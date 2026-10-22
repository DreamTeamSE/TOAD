#!/bin/bash
#SBATCH --job-name=nnunet-train
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Planned responsibilities: nnUNetv2_train XXX 2d <fold> — a single 2D
# config/fold for this first baseline (full 5-fold CV is a later stretch
# goal). No existing per-epoch timing on this cluster/data — first
# submission should use a short --time + reduced epochs as a smoke test,
# then size a real run's --time from the observed per-epoch cost.
#
# Resource allocation below is sized from a real smoke test (job 46886051,
# nnUNetTrainer_1epoch, --partition=hpg-turin --gpus=l4:1, 2026-10-20):
#   --cpus-per-task=8 : measured avg CPU utilization was only ~20%, well
#                        within 8 cores for this dataset's small 2D patches.
#   --mem=32gb         : measured peak RSS was ~16.9GB; 32gb keeps ~2x headroom.
#   --time=30:00:00    : measured epoch time was 77.92s; 1000 epochs ≈ 21.7h,
#                        plus ~1min startup and ~13s final validation. 30h
#                        gives buffer against cluster variance. Check the
#                        partition's actual max walltime
#                        (`sinfo -p hpg-turin -o "%l"`) before relying on this.
#   --gpus=a100:1      : placeholder — no A100s exist on this cluster. See
#                        the open GPU item in the project plan (pending
#                        confirmation from Ani) for why this isn't committed
#                        here yet; the smoke test used a submit-time
#                        override (--partition=hpg-turin --gpus=l4:1).
#
# No CUDA OOM: batch_size=14 @ patch_size (192,256) fit comfortably in the
# L4's 24GB VRAM.
#
# TRAINER/FOLD are overridable so this one script covers both the mandatory
# smoke test and the full run:
#   sbatch --time=00:30:00 --export=TRAINER=nnUNetTrainer_1epoch hpg/nnunet/train_nnunet.slurm   # smoke test
#   sbatch hpg/nnunet/train_nnunet.slurm                                                          # full run (default trainer, 1000 epochs)
#
# Requires prepare_data.slurm to have already run successfully (reads
# $nnUNet_preprocessed/Dataset001_Rat/, does not touch $TOAD_DATA_ROOT).

set -euo pipefail

DATASET_ID=1
DATASET_NAME=Rat
DATASET_DIR_NAME="Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}"
CONFIGURATION=2d
FOLD="${FOLD:-0}"
TRAINER="${TRAINER:-nnUNetTrainer}"

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
