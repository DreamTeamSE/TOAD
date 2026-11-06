# nnU-Net Baseline (HiPerGator)

Setup, data conversion, training, and prediction-export steps for running
the **nnU-Net v2** rat baseline (`Dataset001_Rat`) on HiPerGator. Follow the
shared [data contract](../../docs/DATA_CONTRACT.md) for class IDs and
prediction format — all scripts in this directory produce/consume that
format.

Code only lives in this repo. Rat data and model results live on HPG group
storage (`$DATA_ROOT`, see below).

## 1. Repository setup

```bash
cd /blue/dream_team/<username>
git clone https://github.com/DreamTeamSE/TOAD.git
cd TOAD
git checkout feature/nnunet
```

## 2. Environment setup

One-time, from the repo root (or a short interactive job — `srun --pty bash`):

```bash
bash hpg/nnunet/setup_env.sh
```

This creates the `nnunet` conda env from `environment_nnunet.yml` and
verifies torch/CUDA/nnunetv2 import correctly.

Every session after that:

```bash
module load conda
conda activate nnunet
```

nnU-Net needs three path variables (`nnUNet_raw`, `nnUNet_preprocessed`,
`nnUNet_results`) in every shell/job that calls an `nnUNetv2_*` command —
they don't persist across separate `sbatch` jobs, so each `.slurm` script
below exports them itself from `$DATA_ROOT` (defaults to
`/blue/dream_team/$USER/seko_storage`, overridable via `--export=DATA_ROOT=...`).

## 3. Data conversion — `prepare_data.slurm`

Converts the real rat data (`$TOAD_DATA_ROOT`) into nnU-Net's
`Dataset001_Rat` layout via `src/baselines/nnunet/convert_to_nnunet.py`,
then runs nnU-Net's own `--verify_dataset_integrity` check. `TOAD_DATA_ROOT`
points at a teammate's personal copy of the raw data, not a shared path —
ask the team for the current location.

```bash
sbatch --export=TOAD_DATA_ROOT=/blue/dream_team/<path>/seko_storage/data hpg/nnunet/prepare_data.slurm
```

Writes `$nnUNet_raw/Dataset001_Rat/` (including `conversion_manifest.csv`,
the join key between nnU-Net case IDs and original filenames used later by
`export_predictions.py`) and `$nnUNet_preprocessed/Dataset001_Rat/`.
Re-run (with `--overwrite`, already the default here) whenever the source
data or matching logic changes.

## 4. Training — `train_nnunet.slurm`

Single 2D config/fold (full 5-fold CV is a later stretch goal). Requires
`prepare_data.slurm` to have already run.

```bash
sbatch hpg/nnunet/train_nnunet.slurm                                          # real run: 250 epochs
sbatch --time=00:30:00 --export=TRAINER=nnUNetTrainer_1epoch hpg/nnunet/train_nnunet.slurm   # smoke test
```

Checkpoint + logs land under
`$nnUNet_results/Dataset001_Rat/<TRAINER>__nnUNetPlans__2d/fold_0/`.

## 5. Prediction + export — `predict_nnunet.slurm`

Runs `nnUNetv2_predict` on the held-out test set (`imagesTs`), then
`src/baselines/nnunet/export_predictions.py` to convert nnU-Net's native
output into the exact data-contract format (single-channel 8-bit PNG, class
IDs 0-4, filename stem matching the original input image).

```bash
sbatch hpg/nnunet/predict_nnunet.slurm                                          # real run (default: nnUNetTrainer_250epochs)
sbatch --export=TRAINER=nnUNetTrainer_1epoch hpg/nnunet/predict_nnunet.slurm    # smoke-test checkpoint
```

`TRAINER` must match whatever trainer produced the checkpoint you want to
predict with (nnU-Net builds its model folder path from `-tr`/`-c`
internally — it doesn't take a checkpoint file path directly). `CHECKPOINT`
is just the checkpoint filename within that trainer's `fold_N/` directory
(default `checkpoint_final.pth`), not a full path.

Raw nnU-Net predictions land under
`$DATA_ROOT/predictions_raw/Dataset001_Rat/`; contract-format PNGs under
`$DATA_ROOT/predictions_contract/Dataset001_Rat/`.
