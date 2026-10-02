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

## Open Questions

- Human label schema: adult human knees have no growth plate; osteophyte handling may differ since surgeons sometimes shave them. We are ignoring osteophytes for now.
- Whether "marrow" (4) corresponds to trabecular bone for quantification (future project maybe).