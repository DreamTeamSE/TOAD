"""
convert_to_nnunet.py — rat SEKO data -> nnU-Net dataset format.

Planned responsibilities:
  - Pair rat images/masks using the stain-aware matching logic from
    src/utils/build_train_csv.py (on feature/sam-3 — port the matching logic,
    don't branch off it, since that branch is frozen).
  - Apply the osteophyte -> cartilage label remap (raw label 5 -> 2) required
    by docs/DATA_CONTRACT.md, adapting src/utils/osteophyte_label_change.py.
  - Reuse feature/sam-3's existing Testing_images/ + osteophyte_relabelled_masks/
    held-out set as nnU-Net's imagesTs, so results are comparable to SAM-3 later.
  - Write $nnUNet_raw/DatasetXXX_Rat/{dataset.json, imagesTr/, labelsTr/, imagesTs/}.
  - Split each RGB input into three single-channel files (case_0000/0001/0002)
    and set "overwrite_image_reader_writer": "NaturalImage2DIO" in dataset.json
    — nnU-Net does not accept a single 3-channel PNG, even for plain RGB images.
  - Include a spot-check utility (reconstruct an RGB preview from the three
    split channels + a mask-value histogram) to verify correctness before
    trusting a full conversion run.
"""

raise NotImplementedError("convert_to_nnunet.py scaffold stub")
