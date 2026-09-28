#!/bin/bash
#SBATCH --job-name=toad-eval
#SBATCH --account=REPLACE_WITH_GROUP        # <-- EDIT ME
#SBATCH --qos=REPLACE_WITH_GROUP            # <-- EDIT ME
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=REPLACE_WITH_YOUR_UFL_EMAIL

# Evaluates a fine-tuned SAM-3 checkpoint on the 65-image test_seko.csv set.
# Requires prepare_data.slurm to have been run (test_seko.csv must exist)
# and a checkpoint from train_sam3.slurm.
#
# Usage:
#   sbatch --export=CKPT_PATH=/path/to/sam3_toad_best.pt hpg/eval_sam3.slurm
# or edit CKPT_PATH below directly.

set -euo pipefail

REPO_ROOT="/blue/REPLACE_WITH_GROUP/$USER/TOAD"           # <-- EDIT ME
DATA_ROOT="/blue/REPLACE_WITH_GROUP/$USER/seko_storage"   # <-- EDIT ME
CKPT_PATH="${CKPT_PATH:-REPLACE_WITH_PATH_TO_CHECKPOINT/sam3_toad_best.pt}"   # <-- EDIT ME or pass via --export
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="$DATA_ROOT/outputs/eval_${RUN_ID}"
mkdir -p "$REPO_ROOT/logs" "$OUTPUT_DIR"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seko-sam3

cd "$REPO_ROOT"
nvidia-smi

python src/eval_sam3.py \
    --ckpt     "$CKPT_PATH" \
    --csv      "$DATA_ROOT/data/rat_original/Testing_images/test_seko.csv" \
    --device   cuda \
    --out-json "$OUTPUT_DIR/results.json" \
    2>&1 | tee "$OUTPUT_DIR/eval_output.log"

echo
echo "Eval log + results.json written to: $OUTPUT_DIR"
