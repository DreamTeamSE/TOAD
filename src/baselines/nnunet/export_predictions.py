"""
export_predictions.py — nnU-Net prediction output -> docs/DATA_CONTRACT.md format.

Planned responsibilities:
  - Take nnU-Net's native nnUNetv2_predict output (run on imagesTs) and convert
    it to the exact docs/DATA_CONTRACT.md spec: one PNG per input image, same
    filename stem as the input, single-channel, 8-bit, pixel values = class
    IDs 0-4.
  - Since dataset.json's labels already map 1:1 to the contract, this should
    mostly be a format/naming pass-through — explicitly verify:
      * bit depth is 8-bit (not nnU-Net's sometimes-16-bit output)
      * single-channel (not an RGB visualization)
      * filenames don't carry extra nnU-Net-added suffixes that would break
        stem-matching against the input images

BEFORE WRITING load_prediction() BELOW: actually open one real prediction
file yourself and look at it. On HPG, after predict_nnunet.slurm has run:
    conda activate nnunet
    ls $PRED_DIR                      # confirm filenames (printed by predict_nnunet.slurm)
    python3 -c "
    import cv2
    img = cv2.imread('<full path to one .png from $PRED_DIR>', cv2.IMREAD_UNCHANGED)
    print(img.shape, img.dtype, sorted(set(img.flatten().tolist())))
    "
Compare shape/dtype/values against DATA_CONTRACT.md (single-channel, 8-bit,
values 0-4) before deciding what load_prediction() needs to do.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, for labels
from labels import CLASS_NAMES, validate_mask  # noqa: E402


def load_manifest(manifest_csv: str) -> list[dict]:
    """
    TODO: Read conversion_manifest.csv (written by convert_to_nnunet.py
    during data conversion — look at that file's `manifest_rows`/
    `write_dataset_json` section for the exact columns it writes:
    case_id, split, image_path, mask_path, stain).

    Return the rows where split == "test" as a list of dicts (csv.DictReader
    already gives you dicts — just filter and return them).
    """
    raise NotImplementedError


def original_stem(image_path: str) -> str:
    """
    TODO: Given a manifest row's original image_path (e.g.
    ".../Testing_images/Inputs/H&E/Raw256x192/Aging_H&E_016.tif"), return the
    filename stem the exported PNG should use, so it matches the original
    input image's name per DATA_CONTRACT.md ("same filename stem as the
    input image"). Path(...).stem is probably all you need here — but check
    what a real image_path from the manifest actually looks like first.
    """
    raise NotImplementedError


def load_prediction(pred_path: str) -> np.ndarray:
    """
    TODO: Load one nnU-Net prediction file and return it as a 2D numpy array
    of class IDs (values should be 0-4 after this function runs).

    Inspect a real prediction file first (see the module docstring above for
    the exact steps) and write this to match what you see.
    """
    raise NotImplementedError


def export_one(pred_path: str, out_path: str) -> None:
    """
    TODO:
      1. mask = load_prediction(pred_path)
      2. validate_mask(mask)  # from src.labels — raises if any pixel value
         isn't in CLASS_NAMES (0-4). If this raises, the prediction isn't
         already in contract label space and you'll need to add a remap
         step above before this point — don't silently clip/ignore invalid
         values.
      3. Make sure it's saved as single-channel 8-bit (cv2.imwrite with a
         2D uint8 array does this correctly).
      4. cv2.imwrite(out_path, mask)
    """
    raise NotImplementedError


def export_all(manifest_csv: str, predictions_dir: str, out_dir: str) -> None:
    """
    TODO:
      1. rows = load_manifest(manifest_csv)
      2. os.makedirs(out_dir, exist_ok=True)
      3. For each row:
         - predicted file is likely named f"{row['case_id']}.png" in
           predictions_dir — confirm by listing predictions_dir yourself
           rather than assuming.
         - out_name = original_stem(row["image_path"]) + ".png"
         - export_one(pred_path, os.path.join(out_dir, out_name))
      4. Print a summary: how many exported, and flag (without skipping)
         any manifest test row with no matching prediction file. That
         shouldn't happen, and if it does it's worth investigating why
         before treating this as done.
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True,
                        help="conversion_manifest.csv from convert_to_nnunet.py")
    parser.add_argument("--predictions-dir", required=True,
                        help="Output directory from nnUNetv2_predict")
    parser.add_argument("--out-dir", required=True,
                        help="Where to write DATA_CONTRACT.md-format PNGs")
    args = parser.parse_args()

    export_all(args.manifest_csv, args.predictions_dir, args.out_dir)


if __name__ == "__main__":
    main()
