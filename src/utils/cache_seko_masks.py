# -*- coding: utf-8 -*-
"""
cache_seko_masks.py

Runs the trained SEKO (Keras U-Net) model once over every image listed in
train.csv and caches its predicted per-pixel mask to disk. These cached
masks let SAM 3 training/eval derive box prompts from SEKO's predictions
instead of the ground-truth mask — needed so the same code path also works
on images that have no ground truth.

Preprocessing mirrors train.py / predict.py exactly: cv2.imread (BGR, no
RGB conversion by default) resized to (img_height, img_width), scaled to
[0, 1].

Usage
-----
  conda activate seko
  python src/utils/cache_seko_masks.py
  python src/utils/cache_seko_masks.py --csv path/to/train.csv --model path/to/model.keras
"""

import argparse
import os
import re
import time

import cv2
import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "baseline.yml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_seko_model(model_path):
    from keras.models import load_model
    return load_model(model_path, compile=False)


def image_stem(filename: str) -> str:
    """Mirrors build_train_csv.py's image_stem — strips resized_gray / .tif suffixes."""
    s = re.sub(r',?_resized_gray\.tif$', '', filename)
    if s == filename:
        s = re.sub(r'\.tif$', '', filename)
    return s


# ── Test-set pairing (mirrors eval_sam3.py's match_test_pairs) ─────────────

def mask_stem(filename: str) -> str:
    return re.sub(r'(_bone_binary|_bone)\.tif$', '', filename)


def suffix_token(stem: str) -> str:
    tokens = re.split(r'[-_\s]+', stem.lower())
    return tokens[-1] if tokens else stem.lower()


def match_test_pairs(test_dir: str):
    """Match input images to ground-truth masks under a Testing_images/ dir."""
    import glob
    from collections import defaultdict

    input_dir = os.path.join(test_dir, "Inputs")
    mask_dir = os.path.join(test_dir, "osteophyte_relabelled_masks")

    images = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

    img_by_exact = {image_stem(os.path.basename(f)): f for f in images}
    img_by_suffix = defaultdict(list)
    for f in images:
        img_by_suffix[suffix_token(image_stem(os.path.basename(f)))].append(f)
    img_by_suffix = {k: v[0] for k, v in img_by_suffix.items() if len(v) == 1}

    pairs, unmatched = [], []
    for mpath in masks:
        mstem = mask_stem(os.path.basename(mpath))
        mtoken = suffix_token(mstem)
        if mstem in img_by_exact:
            pairs.append((img_by_exact[mstem], mpath))
        elif mtoken in img_by_suffix:
            pairs.append((img_by_suffix[mtoken], mpath))
        else:
            unmatched.append(mpath)

    if unmatched:
        print(f"  [WARN] {len(unmatched)} masks had no matching input — skipped")
    return pairs


def load_image(img_path, img_height, img_width, rgb=False):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Cannot read image: {img_path}")
    if img_bgr.shape[:2] != (img_height, img_width):
        img_bgr = cv2.resize(img_bgr, (img_width, img_height))
    feed_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if rgb else img_bgr
    feed = (feed_src.astype(np.float32) / 255.0)[None, ...]  # (1, H, W, 3)
    return feed


def infer_seko(model, feed):
    pred = model.predict(feed, verbose=0)   # (1, H, W, K)
    return np.argmax(pred, axis=-1)[0].astype(np.uint8)


def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="Cache SEKO predicted masks for training or test images"
    )
    parser.add_argument("--csv", default=None,
                        help="CSV with image_path, mask_path, stain columns "
                             "(train mode; default: baseline.yml train_csv, "
                             "ignored if --test-dir is given)")
    parser.add_argument("--test-dir", default=None,
                        help="Testing_images/ directory (test mode; pairs "
                             "Inputs/ against osteophyte_relabelled_masks/ "
                             "the same way eval_sam3.py does)")
    parser.add_argument("--model", default=cfg["paths"]["weights_in"],
                        help="Path to SEKO .keras checkpoint")
    parser.add_argument("--out-dir", default=None,
                        help="Directory to write predicted masks "
                             "(default: <root>/SEKO_Predicted_Masks)")
    parser.add_argument("--out-csv", default=None,
                        help="Manifest CSV path "
                             "(default: <root>/train_seko.csv or test_seko.csv)")
    parser.add_argument("--height", type=int, default=cfg["data"]["img_height"])
    parser.add_argument("--width", type=int, default=cfg["data"]["img_width"])
    parser.add_argument("--rgb", action="store_true",
                        help="Convert BGR→RGB before feeding SEKO (default: off, "
                             "matches how train.py fed images)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute masks that already exist on disk")
    args = parser.parse_args()

    test_mode = args.test_dir is not None

    if test_mode:
        root = os.path.normpath(args.test_dir)
        pairs = match_test_pairs(root)
        rows = [{"image_path": img, "mask_path": mask, "stain": None} for img, mask in pairs]
        print(f"Matched {len(rows)} test pairs from {root}")
    else:
        csv_path = args.csv or cfg["paths"]["train_csv"]
        root = os.path.dirname(os.path.abspath(csv_path))
        df = pd.read_csv(csv_path)
        rows = df.to_dict("records")
        print(f"Loaded {len(rows)} rows from {csv_path}")

    out_dir = args.out_dir or os.path.join(root, "SEKO_Predicted_Masks")
    default_manifest = "test_seko.csv" if test_mode else "train_seko.csv"
    out_csv = args.out_csv or os.path.join(root, default_manifest)

    # Output masks are keyed by (stain, stem) — or just stem when there's no
    # stain subfolder, as in test mode. Two rows landing on the same key
    # would silently overwrite each other's cached mask, so fail loudly
    # instead of letting that happen quietly.
    seen_keys = {}
    for row in rows:
        stem = image_stem(os.path.basename(row["image_path"]))
        key = (row.get("stain"), stem)
        if key in seen_keys:
            raise RuntimeError(
                f"Duplicate output key {key} for both "
                f"'{seen_keys[key]}' and '{row['image_path']}' — they would "
                "overwrite each other's cached SEKO mask. Fix the collision "
                "(e.g. add a stain/subfolder distinction) before caching."
            )
        seen_keys[key] = row["image_path"]

    print(f"Loading SEKO model: {args.model}")
    model = load_seko_model(args.model)

    records, skipped = [], 0
    t_start = time.time()

    for i, row in enumerate(rows):
        img_path = row["image_path"]
        mask_path = row["mask_path"]
        stain = row.get("stain")

        stem = image_stem(os.path.basename(img_path))
        stain_dir = os.path.join(out_dir, stain) if stain else out_dir
        os.makedirs(stain_dir, exist_ok=True)
        seko_mask_path = os.path.join(stain_dir, stem + ".tif")

        try:
            if args.overwrite or not os.path.isfile(seko_mask_path):
                feed = load_image(img_path, args.height, args.width, args.rgb)
                pred_ids = infer_seko(model, feed)
                cv2.imwrite(seko_mask_path, pred_ids)
                labels = np.unique(pred_ids)
                status = "wrote"
            else:
                pred_ids = cv2.imread(seko_mask_path, cv2.IMREAD_UNCHANGED)
                labels = np.unique(pred_ids)
                status = "cached"

            records.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "seko_mask_path": seko_mask_path,
                "stain": stain,
            })
            print(f"  [{i + 1:3d}/{len(rows)}] {stem:40s} {status:6s} labels={labels}")

        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")
            skipped += 1

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False)

    elapsed = time.time() - t_start
    print(f"\nWrote {len(out_df)} predicted masks → {out_dir}")
    print(f"Manifest → {out_csv}")
    if skipped:
        print(f"Skipped {skipped} rows due to errors")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
