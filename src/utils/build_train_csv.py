"""
build_train_csv.py

Scans Labelled_images/ and Input_images/ under rat_original, matches
image-mask pairs by filename stem, and writes a CSV used by train_sam3.py.
Testing_images/ is matched the same way and written to a separate test CSV.
Folder names come from the rat_dataset section of configs/baseline.yml.

Matching strategy (applied in order, first hit wins):
  1. Exact stem match   — strip known suffixes, compare directly
  2. Suffix-token match — compare only the final alphanumeric token
     (handles TolBlu case/hyphen divergence: GALIDO_TOL_BLUE_06r ↔ Gal-IDO_Tol-Blue_06r)

Unmatched files are reported but not written to the CSV.

Output CSV columns:
  image_path, mask_path, stain, species

image_path and mask_path are relative to the data root (rat_original/), so
the same CSV works locally and on HPG.

Usage (defaults come from configs/baseline.yml + $TOAD_DATA_ROOT):
    python src/utils/build_train_csv.py
    python src/utils/build_train_csv.py --data-root /path/to/rat_original \\
        --out data/train.csv --test-out data/test.csv
"""

import argparse
import csv
import os
import re
import glob
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for config
from config import load_config

CONFIG_PATH = "configs/baseline.yml"


# ── Stem extraction ───────────────────────────────────────────────────────

def mask_stem(filename: str) -> str:
    """Strip _bone.tif / _bone_binary.tif suffix."""
    return re.sub(r'(_bone_binary|_bone)\.tif$', '', filename)


def image_stem(filename: str) -> str:
    """Strip ,_resized_gray.tif / _resized_gray.tif / plain .tif suffix."""
    s = re.sub(r',?_resized_gray\.tif$', '', filename)
    if s == filename:                       # no resized_gray suffix
        s = re.sub(r'\.tif$', '', filename) # fall back to stripping plain .tif
    return s


def suffix_token(stem: str) -> str:
    """
    Last alphanumeric token after splitting on [-_\\s]+.
    e.g. 'GALIDO_TOL_BLUE_06r' → '06r'
         'Gal-IDO_Tol-Blue_06r' → '06r'
         'MMT_71'               → '71'
    """
    tokens = re.split(r'[-_\s]+', stem.lower())
    return tokens[-1] if tokens else stem.lower()


# ── Per-stain pairing ─────────────────────────────────────────────────────

def pair_stain(labelled_dir: str, input_dir: str, stain: str) -> tuple[list, list]:
    """
    Match masks to images for one stain type.

    Returns
    -------
    pairs    : list of (image_path, mask_path, stain)
    unmatched: list of mask paths with no corresponding image
    """
    masks  = sorted(glob.glob(os.path.join(labelled_dir, "*.tif")))
    images = sorted(glob.glob(os.path.join(input_dir,    "*.tif")))

    # Build image lookup: exact stem → path
    img_by_exact  = {image_stem(os.path.basename(f)): f for f in images}
    # Fallback: suffix token → path (skip ambiguous tokens)
    img_by_suffix = defaultdict(list)
    for f in images:
        img_by_suffix[suffix_token(image_stem(os.path.basename(f)))].append(f)
    img_by_suffix = {k: v[0] for k, v in img_by_suffix.items() if len(v) == 1}

    pairs, unmatched = [], []

    for mpath in masks:
        mname  = os.path.basename(mpath)
        mstem  = mask_stem(mname)
        mtoken = suffix_token(mstem)

        if mstem in img_by_exact:
            pairs.append((img_by_exact[mstem], mpath, stain))
        elif mtoken in img_by_suffix:
            pairs.append((img_by_suffix[mtoken], mpath, stain))
        else:
            unmatched.append(mpath)

    return pairs, unmatched


# ── Main ──────────────────────────────────────────────────────────────────

def write_csv(pairs: list, unmatched: list, out_path: str, data_root: str, species: str):
    """Write pairs with image/mask paths relative to data_root, then report unmatched masks."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "stain", "species"])
        for image_path, mask_path, stain in pairs:
            writer.writerow([
                Path(os.path.relpath(image_path, data_root)).as_posix(),
                Path(os.path.relpath(mask_path, data_root)).as_posix(),
                stain,
                species,
            ])

    print(f"\nWrote {len(pairs)} pairs → {out_path}")

    if unmatched:
        print(f"\nUnmatched masks ({len(unmatched)}) — no corresponding image found:")
        for p in unmatched:
            print(f"  {p}")


def build(data_root: str, out_path: str, test_out_path: str, dataset: dict):
    all_pairs    = []
    all_unmatched = []

    for entry in dataset["stains"]:
        stain        = entry["stain"]
        labelled_dir = os.path.join(data_root, dataset["labelled_dir"], entry["labelled"])
        input_dir    = os.path.join(data_root, dataset["input_dir"],    entry["input"])

        if not os.path.isdir(labelled_dir):
            print(f"[WARN] Labelled dir not found, skipping {stain}: {labelled_dir}")
            continue
        if not os.path.isdir(input_dir):
            print(f"[WARN] Input dir not found, skipping {stain}: {input_dir}")
            continue

        pairs, unmatched = pair_stain(labelled_dir, input_dir, stain)
        all_pairs.extend(pairs)
        all_unmatched.extend(unmatched)

        print(f"  {stain:8s}  {len(pairs):3d} matched  {len(unmatched):3d} unmatched")

    write_csv(all_pairs, all_unmatched, out_path, data_root, dataset["species"])

    test = dataset["test"]
    test_masks  = os.path.join(data_root, test["masks"])
    test_inputs = os.path.join(data_root, test["inputs"])
    if not (os.path.isdir(test_masks) and os.path.isdir(test_inputs)):
        print(f"\n[WARN] Test dirs not found, skipping test CSV: {test_masks}, {test_inputs}")
        return

    print(f"\nTest set:")
    pairs, unmatched = pair_stain(test_masks, test_inputs, test["stain"])
    print(f"  {'test':8s}  {len(pairs):3d} matched  {len(unmatched):3d} unmatched")
    write_csv(pairs, unmatched, test_out_path, data_root, dataset["species"])


def main():
    parser = argparse.ArgumentParser(description="Build image-mask pair CSVs for TOAD training")
    parser.add_argument("--data-root", default=None,
                        help="Path to rat_original/ directory (default: data_paths.rat_root)")
    parser.add_argument("--out", default=None,
                        help="Train CSV path (default: data_paths.train_csv)")
    parser.add_argument("--test-out", default=None,
                        help="Test CSV path (default: data_paths.test_csv)")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    data_paths = cfg["data_paths"]
    data_root = args.data_root or str(data_paths["rat_root"])
    out_path = args.out or str(data_paths["train_csv"])
    test_out_path = args.test_out or str(data_paths["test_csv"])

    print(f"Scanning: {data_root}")
    build(data_root, out_path, test_out_path, cfg["rat_dataset"])


if __name__ == "__main__":
    main()
