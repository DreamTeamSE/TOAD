# TOAD Data Contract

All models, pipelines, and scripts must follow this contract so results are comparable. Changes to this file require a PR reviewed by the team lead.

## Class IDs

| ID | Class        |
|----|--------------|
| 0  | Background   |
| 1  | Bone         |
| 2  | Cartilage    |
| 3  | Growth plate |
| 4  | Marrow       |

Osteophyte (raw label 5) is merged into cartilage (2).

## Prediction Format

- One PNG mask per input image
- Same filename stem as the input image
- Single channel, pixel values = class IDs

## Branching

- Branch off `dev`, one feature branch per person
- Merge back to `dev` through a PR with review

## Human Data (proposed, pending research sign-off)

- Class IDs are unchanged. Models keep all 5 output channels, including growth plate (3).
- Adult human masks never contain growth plate (3): adult growth plates are fused.
  `labels.prepare_mask(mask, "human")` merges osteophyte and rejects any 3.
- On human data, a predicted 3 is an error. Growth plate is left out of per-class Dice
  and its predicted share is reported separately (`src/compare_predictions.py`).
- Human images are resampled to the rat training pixel size before tiling
  (`--source-mpp` / `--target-mpp` in `src/preprocess_human_data.py`).

## Open Questions

- Human label schema: adult human knees have no growth plate; osteophyte handling may differ since surgeons sometimes shave them. We are ignoring osteophytes for now. Research to confirm the final human class list.
- Rat training pixel size: rat images were stored at 256x192, so the target microns-per-pixel must come from the original rat scans (`src/inspect_resolution.py --resize-to 256x192`).
- Whether "marrow" (4) corresponds to trabecular bone for quantification (future project maybe).