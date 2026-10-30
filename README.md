# TOAD (Dream Team Engineering)

TOAD is a deep learning pipeline for histology segmentation in knee osteoarthritis (OA), using rat (SEKO) data and adult human data.

This repo contains **code only**. No datasets, no model weights. Data lives on HiPerGator group storage.

## Before You Start

- Read [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md). It defines the class IDs and prediction format every model must follow.
- Tasks are tracked on the TOAD Fall 2026 GitHub Project.

## Repo Structure

```
TOAD/
├── configs/      # Config files
├── docs/         # Data contract and documentation
├── src/          # Source code
└── README.md
```

## Branching

- `dev` is the main branch. All work branches off it.
- Until you're added as a collaborator, fork the repo and work on a branch in your fork, e.g. `feature/nnunet-baseline`.
- Open a pull request into `dev` on DreamTeamSE/TOAD for review.
- `feature/sam-3` is frozen until Step 3. Do not branch off it.

## Environment

Requires Conda (Miniforge or Anaconda).

```bash
conda env create -f environment.yml
conda activate seko
```

Some components (e.g. nnU-Net) use their own environments. See the README in their directory.

## Running Inference (SEKO, Single Image)

Run a trained SEKO model on one image and save the predicted mask (class IDs), a grayscale visualization, and an overlay on the original image.

From the project root:

```bash
python src/forward_pass.py \
  --model "/path/to/model.keras" \
  --image "/path/to/input_image.tif" \
  --output "/path/to/output_mask.png"
```

## Human Data Preprocessing

```bash
# 1. Find the pixel size the rat model was trained at (original rat scans, squashed to 256x192)
python src/inspect_resolution.py /path/to/rat_original_scans --resize-to 256x192

# 2. Check the human scans' pixel size
python src/inspect_resolution.py /path/to/human_tiffs --csv outputs/human_mpp.csv

# 3. Tile human images at the rat scale (add --mask-dir if you have label masks).
#    Without --input-dir, data_paths.human_input_dir under $TOAD_DATA_ROOT is used.
python src/preprocess_human_data.py \
  --input-dir /path/to/human_tiffs --output-dir outputs/human_none \
  --source-mpp auto --target-mpp <rat um/px> --stain-norm none
```

`--stain-norm none|reinhard` (default `none`) is available on `preprocess_human_data.py`,
`forward_pass.py`, `predict.py`, and via `SEKO_STAIN_NORM` in `train.py`, so rat and human
data can be run under the same setting. `src/compare_predictions.py` compares two prediction
folders (e.g. none vs reinhard), with Dice when ground-truth masks are given.

On HiPerGator, `hpg/human/stain_compare.slurm` runs the whole none-vs-reinhard comparison
(preprocess both ways, predict, compare); its header lists the settings.

Tests: `pytest` from the project root.

## Project Status

| Area                     | Status          |
|--------------------------|-----------------|
| SEKO U-Net inference     | Working         |
| SEKO U-Net training      | Not maintained  |
| Human data preprocessing | Under review    |
| nnU-Net baseline         | In progress     |
| Shared evaluation        | In progress     |
| SAM 3 cascade            | On hold (Step 3)|