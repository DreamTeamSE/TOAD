#!/bin/bash
#SBATCH --job-name=nnunet-train
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Planned responsibilities: nnUNetv2_train XXX 2d <fold> — a single 2D
# config/fold for this first baseline (full 5-fold CV is a later stretch
# goal). No existing per-epoch timing on this cluster/data — first
# submission should use a short --time + reduced epochs as a smoke test,
# then size a real run's --time from the observed per-epoch cost.
#
# Resource allocation above was copied from feature/sam-3's train_sam3.slurm
# as a starting point (matches the established HPG convention), NOT derived
# from nnU-Net's actual resource profile. Known mismatches to revisit once a
# real smoke test can be run:
#   --cpus-per-task=8 : sized for SAM-3's DataLoader(num_workers=2). nnU-Net
#                        leans much more heavily on CPU-side work — it runs
#                        real-time data augmentation (elastic deform, rotate,
#                        scale, gamma jitter) via batchgenerators in parallel
#                        background worker processes to keep the GPU fed, and
#                        typically wants more worker processes than this.
#                        Likely too low — bump up once smoke-tested.
#   --mem=64gb         : likely generous, not tight. nnU-Net's memory appetite
#                        scales with patch/volume size; nnU-Net is usually
#                        benchmarked on much larger 3D CT/MRI volumes, while
#                        our 2D histology patches are small (comparable to
#                        SEKO's 192x256 inputs). Safe to over-ask, but not a
#                        measured number.
#   --time=24:00:00    : likely generous, not tight, for the same reason (small
#                        2D images, a few hundred training images, single
#                        fold). Still a placeholder, not a measurement — do
#                        not trust it without a smoke test first.
#   --gpus=a100:1      : single GPU is correct (nnU-Net single-fold 2D config
#                        doesn't need multi-GPU), but whether A100 specifically
#                        is available/necessary depends on your group's actual
#                        HPG allocation — unconfirmed pending HPG account access.
#
# Resolution: submit once with a quick-epoch trainer variant and a short
# --time, read the real per-epoch wall-clock time + peak CPU/memory usage
# nnU-Net reports in logs/%x_%j.out, then size the real run's flags from
# measured numbers instead of this borrowed starting point.
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
