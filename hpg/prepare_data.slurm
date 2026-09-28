#!/bin/bash
#SBATCH --job-name=toad-prep
#SBATCH --account=REPLACE_WITH_GROUP        # <-- EDIT ME: your PI's HPG allocation group
#SBATCH --qos=REPLACE_WITH_GROUP            # <-- EDIT ME: usually same as --account (or "<group>-b" for burst QOS)
#SBATCH --partition=hpg-default             # CPU only — no GPU needed for this step
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=REPLACE_WITH_YOUR_UFL_EMAIL   # <-- uncomment + edit for notifications

# One-time (or re-run-when-data-changes) step: regenerate train.csv with
# HPG-absolute paths, then run SEKO over every train + test image to
# produce train_seko.csv / test_seko.csv — the box-prompt source
# train_sam3.py and eval_sam3.py both require.
#
# Run this once before train_sam3.slurm / eval_sam3.slurm. It's cheap
# (SEKO inference over ~350 images took ~20s locally on CPU) so hpg-default
# CPU partition is enough — no GPU request.

set -euo pipefail

REPO_ROOT="/blue/REPLACE_WITH_GROUP/$USER/TOAD"          # <-- EDIT ME
DATA_ROOT="/blue/REPLACE_WITH_GROUP/$USER/seko_storage"  # <-- EDIT ME: where seko_storage/ lives on HPG
mkdir -p "$REPO_ROOT/logs"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seko-sam3

cd "$REPO_ROOT"

echo "=== Regenerating train.csv with HPG paths ==="
python src/utils/build_train_csv.py \
    --data-root "$DATA_ROOT/data/rat_original" \
    --out       "$DATA_ROOT/data/rat_original/train.csv"

echo
echo "=== Caching SEKO predictions for training images ==="
python src/utils/cache_seko_masks.py \
    --csv      "$DATA_ROOT/data/rat_original/train.csv" \
    --model    "$DATA_ROOT/models/V2_best_model.keras" \
    --out-csv  "$DATA_ROOT/data/rat_original/train_seko.csv"

echo
echo "=== Caching SEKO predictions for test images ==="
python src/utils/cache_seko_masks.py \
    --test-dir "$DATA_ROOT/data/rat_original/Testing_images" \
    --model    "$DATA_ROOT/models/V2_best_model.keras" \
    --out-csv  "$DATA_ROOT/data/rat_original/Testing_images/test_seko.csv"

echo
echo "Done. train_seko.csv and test_seko.csv are ready."
