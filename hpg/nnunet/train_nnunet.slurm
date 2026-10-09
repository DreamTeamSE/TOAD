#!/bin/bash
#SBATCH --job-name=nnunet-train
#SBATCH --account=REPLACE_WITH_GROUP
#SBATCH --qos=REPLACE_WITH_GROUP
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Planned responsibilities: nnUNetv2_train XXX 2d <fold> — a single 2D
# config/fold for this first baseline (full 5-fold CV is a later stretch
# goal). No existing per-epoch timing on this cluster/data — first
# submission should use a short --time + reduced epochs as a smoke test,
# then size a real run's --time from the observed per-epoch cost.
#
# Resource allocation above was copied from feature/sam-3's train_sam3.slurm
# as a starting point (matches the established HPG convention), NOT derived
# from nnU-Net's actual resource profile. Known mismatches to revisit once a
# real smoke test can be run:
#   --cpus-per-task=8 : sized for SAM-3's DataLoader(num_workers=2). nnU-Net
#                        leans much more heavily on CPU-side work — it runs
#                        real-time data augmentation (elastic deform, rotate,
#                        scale, gamma jitter) via batchgenerators in parallel
#                        background worker processes to keep the GPU fed, and
#                        typically wants more worker processes than this.
#                        Likely too low — bump up once smoke-tested.
#   --mem=64gb         : likely generous, not tight. nnU-Net's memory appetite
#                        scales with patch/volume size; nnU-Net is usually
#                        benchmarked on much larger 3D CT/MRI volumes, while
#                        our 2D histology patches are small (comparable to
#                        SEKO's 192x256 inputs). Safe to over-ask, but not a
#                        measured number.
#   --time=24:00:00    : likely generous, not tight, for the same reason (small
#                        2D images, a few hundred training images, single
#                        fold). Still a placeholder, not a measurement — do
#                        not trust it without a smoke test first.
#   --gpus=a100:1      : single GPU is correct (nnU-Net single-fold 2D config
#                        doesn't need multi-GPU), but whether A100 specifically
#                        is available/necessary depends on your group's actual
#                        HPG allocation — unconfirmed pending HPG account access.
#
# Resolution: submit once with --epochs 1 and a short --time (e.g. 00:30:00),
# read the real per-epoch wall-clock time + peak CPU/memory usage nnU-Net
# reports in logs/%x_%j.out, then size the real run's flags from measured
# numbers instead of this borrowed starting point.

echo "train_nnunet.slurm scaffold stub" >&2
exit 1
