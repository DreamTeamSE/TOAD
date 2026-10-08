#!/bin/bash
# setup_env.sh — one-time environment setup for the TOAD nnU-Net baseline on HiPerGator.
#
# Planned responsibilities:
#   - module load conda; create/activate a dedicated "nnunet" conda env
#     (kept separate from seko/seko-sam3, per environment_nnunet.yml).
#   - pip install nnunetv2 (pulls in a pinned torch).
#   - export nnUNet_raw / nnUNet_preprocessed / nnUNet_results, pointed at
#     /blue/<group>/$USER/... (never /home — too small a quota).
#   - Check: torch sees a GPU, `nnUNetv2_plan_and_preprocess --help` runs.
#
# Unlike SAM-3's setup_env.sh, there's no external checkpoint to transfer and
# no HuggingFace login dependency.

echo "setup_env.sh scaffold stub" >&2
exit 1
