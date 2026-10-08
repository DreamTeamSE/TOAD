#!/bin/bash
#SBATCH --job-name=nnunet-prep
#SBATCH --account=REPLACE_WITH_GROUP
#SBATCH --qos=REPLACE_WITH_GROUP
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Planned responsibilities: CPU-only job that runs
# src/baselines/nnunet/convert_to_nnunet.py against the real rat dataset on
# HPG storage (data access required, not just compute), producing
# $nnUNet_raw/DatasetXXX_Rat/, then nnUNetv2_plan_and_preprocess
# --verify_dataset_integrity.

echo "prepare_data.slurm scaffold stub" >&2
exit 1
