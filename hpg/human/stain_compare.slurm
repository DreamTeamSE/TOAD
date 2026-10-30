#!/bin/bash
#SBATCH --job-name=human-stain-compare
#SBATCH --account=dream_team
#SBATCH --qos=dream_team
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stain-normalization experiment on human data: preprocess the same human slides
# twice (--stain-norm none and reinhard), run the SEKO model on both tile sets, and
# compare. Tiles are picked before any color change, so both runs get the same tiles.
#
# Required:
#   TOAD_DATA_ROOT   data root used by configs/baseline.yml (see src/config.py)
# Optional:
#   HUMAN_INPUT_DIR  human slides (default: data_paths.human_input_dir under TOAD_DATA_ROOT)
#   MASK_DIR         human class-ID masks named like the slides; adds Dice/IoU vs ground truth
#   TARGET_MPP       rat training pixel size in um/px; resamples human slides to it.
#                    Get it with: python src/inspect_resolution.py <rat originals> --resize-to 256x192
#   SOURCE_MPP       human pixel size in um/px (default: auto, read from TIFF metadata)
#   SCALE            fixed resize factor instead of TARGET_MPP (e.g. 0.25)
#   MODEL            .keras checkpoint (default: data_paths.weights_in under TOAD_DATA_ROOT)
#   OUT_ROOT         output folder (default: outputs/stain_compare_<job id>)
#
#   sbatch --export=ALL,TOAD_DATA_ROOT=/blue/dream_team/$USER/seko_storage,TARGET_MPP=<um/px> \
#       hpg/human/stain_compare.slurm
#
# Full-resolution Leica TIFFs are loaded whole; raise --mem if a slide fails to load.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${TOAD_DATA_ROOT:?Set TOAD_DATA_ROOT to the data root (see src/config.py)}"
export TOAD_DATA_ROOT

if [[ -n "${TARGET_MPP:-}" && -n "${SCALE:-}" ]]; then
    echo "Set TARGET_MPP or SCALE, not both." >&2
    exit 2
fi

OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs/stain_compare_${SLURM_JOB_ID:-local}}"
mkdir -p "$REPO_ROOT/logs" "$OUT_ROOT"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate seko

cd "$REPO_ROOT"

INPUT_ARGS=()
[[ -n "${HUMAN_INPUT_DIR:-}" ]] && INPUT_ARGS=(--input-dir "$HUMAN_INPUT_DIR")

SCALE_ARGS=()
if [[ -n "${TARGET_MPP:-}" ]]; then
    SCALE_ARGS=(--source-mpp "${SOURCE_MPP:-auto}" --target-mpp "$TARGET_MPP")
elif [[ -n "${SCALE:-}" ]]; then
    SCALE_ARGS=(--scale "$SCALE")
else
    echo "NOTE: no TARGET_MPP or SCALE; human slides are tiled at native resolution."
fi

MASK_ARGS=()
[[ -n "${MASK_DIR:-}" ]] && MASK_ARGS=(--mask-dir "$MASK_DIR")

MODEL_ARGS=()
[[ -n "${MODEL:-}" ]] && MODEL_ARGS=(--model "$MODEL")

echo "Output            : $OUT_ROOT"
echo "Human input       : ${HUMAN_INPUT_DIR:-config default}"
echo "Rescale           : ${SCALE_ARGS[*]:-none}"
echo "Masks             : ${MASK_DIR:-none}"
echo

if [[ -n "${HUMAN_INPUT_DIR:-}" ]]; then
    echo "=== Human slide pixel sizes ==="
    python src/inspect_resolution.py "$HUMAN_INPUT_DIR" --csv "$OUT_ROOT/human_mpp.csv"
    echo
fi

for NORM in none reinhard; do
    echo "=== Preprocess: stain-norm $NORM ==="
    python src/preprocess_human_data.py "${INPUT_ARGS[@]}" --output-dir "$OUT_ROOT/tiles_$NORM" \
        --stain-norm "$NORM" "${SCALE_ARGS[@]}" "${MASK_ARGS[@]}"
    echo
    echo "=== Predict: stain-norm $NORM ==="
    # Tiles are already normalized, so predict.py must not normalize again.
    python src/predict.py "${MODEL_ARGS[@]}" --input-dir "$OUT_ROOT/tiles_$NORM" \
        --output-dir "$OUT_ROOT/pred_$NORM" --stain-norm none
    echo
done

GT_ARGS=()
[[ -n "${MASK_DIR:-}" ]] && GT_ARGS=(--gt-dir "$OUT_ROOT/tiles_none/masks")

echo "=== Compare none vs reinhard ==="
python src/compare_predictions.py \
    --a "$OUT_ROOT/pred_none/Original_Predictions" --name-a none \
    --b "$OUT_ROOT/pred_reinhard/Original_Predictions" --name-b reinhard \
    --species human "${GT_ARGS[@]}" --csv "$OUT_ROOT/stain_compare.csv" | tee "$OUT_ROOT/stain_compare.txt"

echo
echo "Done. Results in $OUT_ROOT (stain_compare.txt, stain_compare.csv)"
