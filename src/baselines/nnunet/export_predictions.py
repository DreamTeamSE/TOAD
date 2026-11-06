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

An HPG prediction was inspected before implementing load_prediction():
shape (192, 256), dtype uint8, and pixel values [0, 1, 2, 3, 4].
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
from utils.build_train_csv import image_stem  # noqa: E402


def load_manifest(manifest_csv: str) -> list[dict]:
    """Read conversion_manifest.csv, return only the test-split rows."""
    with open(manifest_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["split"] == "test"]


def original_stem(image_path: str) -> str:
    """
    Filename stem the exported PNG should use, matching the original input
    image's name per DATA_CONTRACT.md. Reuses image_stem() so the
    _resized_gray suffix and stray-comma artifacts (e.g.
    ",Aging_H&E_029,_resized_gray.tif") are normalized the same way they
    already are for the train/test CSVs.
    """
    return image_stem(os.path.basename(image_path))


def load_prediction(pred_path: str) -> np.ndarray:
    """
    Load a single-channel 8-bit nnU-Net prediction containing contract class IDs.
    """
    mask = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read prediction file: {pred_path}")
    if mask.ndim != 2:
        raise ValueError(
            f"Expected a single-channel 2D prediction at {pred_path}; "
            f"got shape {mask.shape}"
        )
    if mask.dtype != np.uint8:
        raise ValueError(
            f"Expected an 8-bit prediction at {pred_path}; got dtype {mask.dtype}"
        )

    validate_mask(mask)
    return mask


def export_one(pred_path: str, out_path: str) -> None:
    """load_prediction() already validates; just read, then write 8-bit single-channel."""
    mask = load_prediction(pred_path)
    cv2.imwrite(out_path, mask)


def export_all(manifest_csv: str, predictions_dir: str, out_dir: str) -> None:
    """Export every test-split prediction to DATA_CONTRACT.md format, flagging any missing ones."""
    rows = load_manifest(manifest_csv)
    os.makedirs(out_dir, exist_ok=True)

    exported, missing = 0, []
    for row in rows:
        pred_path = os.path.join(predictions_dir, f"{row['case_id']}.png")
        if not os.path.isfile(pred_path):
            missing.append(row["case_id"])
            continue
        out_name = original_stem(row["image_path"]) + ".png"
        export_one(pred_path, os.path.join(out_dir, out_name))
        exported += 1

    print(f"Exported {exported}/{len(rows)} test predictions -> {out_dir}")
    if missing:
        print(f"[WARN] {len(missing)} test case(s) had no matching prediction file in {predictions_dir}:")
        for case_id in missing:
            print(f"  {case_id}")


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
