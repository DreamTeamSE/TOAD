#!/bin/bash
# setup_env.sh — one-time environment setup for the TOAD nnU-Net baseline on HiPerGator.
#
# Run this once on a login node (or a short interactive job — `srun --pty bash`)
# before submitting any of the .slurm scripts in this directory.
#
# Usage:
#   cd hpg/nnunet/
#   bash setup_env.sh
#
# Unlike SAM-3's setup_env.sh, there's no external checkpoint to transfer and
# no HuggingFace login dependency — nnunetv2 is a normal PyPI package that
# pulls in a compatible torch automatically. Simpler bar to clear.
set -euo pipefail

ENV_NAME="nnunet"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/blue/REPLACE_WITH_GROUP/$USER/seko_storage}"   # <-- EDIT ME (or export before running)

echo "Repo root   : $REPO_ROOT"
echo "Data root   : $DATA_ROOT"
echo "Conda env   : $ENV_NAME"
echo

module load conda   # HPG environment-modules; adjust version if `module avail conda` shows one

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '$ENV_NAME' already exists — skipping creation. Delete it first"
    echo "(conda env remove -n $ENV_NAME) to rebuild from scratch."
else
    echo "Creating conda env from environment_nnunet.yml ..."
    conda env create -f "$REPO_ROOT/environment_nnunet.yml"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# nnU-Net's three required path variables. Exported here only for this
# script's own verification run below — they do NOT persist into separate
# sbatch jobs (each starts a fresh shell), so every hpg/nnunet/*.slurm script
# must export these same three lines itself before calling any nnUNetv2_*
# command.
export nnUNet_raw="$DATA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$DATA_ROOT/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo
echo "Verifying environment ..."
python - <<'PYEOF'
import torch
print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
try:
    import importlib.metadata
    import nnunetv2  # noqa: F401 — import check only, package exposes no __version__
    print(f"nnunetv2 {importlib.metadata.version('nnunetv2')} import OK")
except ImportError as e:
    print(f"nnunetv2 import FAILED: {e}")
PYEOF

echo
nnUNetv2_plan_and_preprocess --help > /dev/null && echo "nnUNetv2_plan_and_preprocess CLI OK"

echo
echo "Done. Activate with: conda activate $ENV_NAME"
echo "Remember to export nnUNet_raw/nnUNet_preprocessed/nnUNet_results in every .slurm script — see comment above."
