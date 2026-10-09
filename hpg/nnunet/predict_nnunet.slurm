#!/bin/bash
#SBATCH --job-name=nnunet-predict
#SBATCH --account=REPLACE_WITH_GROUP
#SBATCH --qos=REPLACE_WITH_GROUP
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Planned responsibilities: nnUNetv2_predict on imagesTs, then
# src/baselines/nnunet/export_predictions.py to convert nnU-Net's native
# output into the exact docs/DATA_CONTRACT.md format (single-channel 8-bit
# PNG, class IDs 0-4, filename stem matching the input).

echo "predict_nnunet.slurm scaffold stub" >&2
exit 1
