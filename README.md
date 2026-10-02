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

- `dev` is the main branch. Branch off it for all work.
- One feature branch per person, e.g. `feature/nnunet-baseline`.
- Merge back to `dev` through a pull request with review.
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

## Project Status

| Area                     | Status          |
|--------------------------|-----------------|
| SEKO U-Net inference     | Working         |
| SEKO U-Net training      | Not maintained  |
| Human data preprocessing | Under review    |
| nnU-Net baseline         | In progress     |
| Shared evaluation        | In progress     |
| SAM 3 cascade            | On hold (Step 3)|