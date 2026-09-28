#!/bin/bash
# setup_env.sh — one-time environment setup for the TOAD SAM-3 pipeline on HiPerGator.
#
# Run this once on a login node (or a short interactive job — `srun --pty bash`)
# before submitting any of the .slurm scripts in this directory.
#
# Usage:
#   cd hpg/
#   bash setup_env.sh
#
# ── Fill in before running ──────────────────────────────────────────────────
#   SAM3_SRC_DIR : path to a local clone of the patched SAM-3 package.
#                  environment_sam3.yml notes SAM-3 "must be installed
#                  separately as a custom patched package — it is not
#                  available on PyPI." This script does NOT know where that
#                  clone lives; if it's on your laptop, rsync/scp it to HPG
#                  first (e.g. `rsync -av /path/to/sam3/ hpg:/blue/<group>/$USER/sam3/`)
#                  and point SAM3_SRC_DIR at the HPG copy below.
set -euo pipefail

SAM3_SRC_DIR="${SAM3_SRC_DIR:-/blue/REPLACE_WITH_GROUP/$USER/sam3}"   # <-- EDIT ME (or export before running)
ENV_NAME="seko-sam3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Repo root   : $REPO_ROOT"
echo "SAM3 source : $SAM3_SRC_DIR"
echo "Conda env   : $ENV_NAME"
echo

module load conda   # HPG environment-modules; adjust version if `module avail conda` shows one, e.g. `module load conda/24.9.2`

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '$ENV_NAME' already exists — skipping creation. Delete it first"
    echo "(conda env remove -n $ENV_NAME) to rebuild from scratch."
else
    echo "Creating conda env from environment_sam3.yml ..."
    conda env create -f "$REPO_ROOT/environment_sam3.yml"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# TensorFlow is already pinned in environment_sam3.yml (kept for SEKO
# inference alongside torch/SAM-3) — the macOS pip-tensorflow segfault we
# hit locally is specific to Apple's CoreML/ANE framework probing in a
# non-GUI session; it does not apply on a Linux CUDA node, so both
# frameworks are expected to coexist fine in this one env here.

if [ ! -d "$SAM3_SRC_DIR" ]; then
    echo
    echo "!! SAM3_SRC_DIR ($SAM3_SRC_DIR) does not exist."
    echo "!! Transfer your patched SAM-3 clone to HPG and re-run this script,"
    echo "!! or install it manually:  pip install -e /path/to/sam3"
    exit 1
fi

echo "Installing SAM-3 from local clone: $SAM3_SRC_DIR"
pip install -e "$SAM3_SRC_DIR"

echo
echo "Verifying environment ..."
python "$REPO_ROOT/verify_env.py" || true

python - <<'PYEOF'
import torch, tensorflow as tf
print(f"torch {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
print(f"tensorflow {tf.__version__}")
try:
    import sam3
    print("sam3 import OK")
except ImportError as e:
    print(f"sam3 import FAILED: {e}")
PYEOF

echo
echo "Done. Activate with: conda activate $ENV_NAME"
