#!/bin/bash
#SBATCH --job-name=toad-train
#SBATCH --account=REPLACE_WITH_GROUP        # <-- EDIT ME
#SBATCH --qos=REPLACE_WITH_GROUP            # <-- EDIT ME
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1                       # adjust GPU type/count to what your group has access to
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                   # a few spare cores for the DataLoader's num_workers=2
#SBATCH --mem=64gb
#SBATCH --time=24:00:00                     # unverified estimate — see note below; run a 1-epoch smoke test first
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=REPLACE_WITH_YOUR_UFL_EMAIL

# Fine-tunes SAM-3 on TOAD histology segmentation. Requires prepare_data.slurm
# to have been run first (train_seko.csv must exist).
#
# NOTE on --time: this pipeline has never been run end-to-end (see repo
# history / conversation this was generated from) — there's no real
# per-epoch timing to base --time on. 24h is a placeholder. Recommended:
# first submit with --epochs 1 and a short --time (e.g. 00:30:00) as a
# smoke test, check logs/%x_%j.out for the per-epoch wall-clock time
# train_sam3.py prints, then size --time for a full run accordingly.

set -euo pipefail

REPO_ROOT="/blue/REPLACE_WITH_GROUP/$USER/TOAD"           # <-- EDIT ME
DATA_ROOT="/blue/REPLACE_WITH_GROUP/$USER/seko_storage"   # <-- EDIT ME
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="$DATA_ROOT/outputs/run_${RUN_ID}"
mkdir -p "$REPO_ROOT/logs" "$OUTPUT_DIR"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seko-sam3

cd "$REPO_ROOT"
nvidia-smi

python src/train_sam3.py \
    --csv        "$DATA_ROOT/data/rat_original/train_seko.csv" \
    --ckpt       "$REPO_ROOT/checkpoints/sam3.pt" \
    --output-dir "$OUTPUT_DIR" \
    --device     cuda \
    --epochs     20 \
    --lr         1e-4 \
    --val-frac   0.15 \
    --seed       42

echo
echo "Checkpoints + metrics.json written to: $OUTPUT_DIR"
