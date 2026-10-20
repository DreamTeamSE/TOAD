#!/bin/bash
#SBATCH --job-name=nnunet-prep
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-only job: converts the real rat dataset to nnU-Net format, then runs
# nnU-Net's own structural check. Run once before train_nnunet.slurm, and
# again whenever the source data or matching logic changes (use --overwrite
# below, already on by default here).
#
# Required: export TOAD_DATA_ROOT to the real rat_original data root before
# submitting. It is not hardcoded here, since the real path currently
# points into a specific teammate's personal directory, not a shared
# location. Ask the team for the current path if you don't have it. Either
# export it in your shell first, or pass it at submit time:
#   sbatch --export=TOAD_DATA_ROOT=/blue/dream_team/<path>/seko_storage/data hpg/nnunet/prepare_data.slurm

set -euo pipefail

: "${TOAD_DATA_ROOT:?TOAD_DATA_ROOT is not set -- see header comment above}"

DATASET_ID=1
DATASET_NAME=Rat
DATASET_DIR_NAME="Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/blue/dream_team/$USER/seko_storage}"  # nnU-Net's own working dir, separate from TOAD_DATA_ROOT (the raw input data)
mkdir -p "$REPO_ROOT/logs"

export TOAD_DATA_ROOT
export nnUNet_raw="$DATA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$DATA_ROOT/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nnunet

cd "$REPO_ROOT"

echo "TOAD_DATA_ROOT : $TOAD_DATA_ROOT"
echo "nnUNet_raw     : $nnUNet_raw"
echo

echo "=== Converting rat data to nnU-Net format ==="
python src/baselines/nnunet/convert_to_nnunet.py --dataset-id "$DATASET_ID" --dataset-name "$DATASET_NAME" --overwrite

echo
echo "=== Verifying dataset integrity ==="
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

echo
echo "Done. $DATASET_DIR_NAME ready under \$nnUNet_raw; preprocessed plans under \$nnUNet_preprocessed."
