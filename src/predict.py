# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:51:11 2024

@author: justinjoseph

Batch inference with a trained SEKO model. Defaults come from configs/baseline.yml:
data_paths (weights_in, input_dir) resolve against $TOAD_DATA_ROOT (see src/config.py),
paths.output_dir against the repo. Any of them can be overridden on the command line,
in which case $TOAD_DATA_ROOT is only needed for the ones you leave out:

    python src/predict.py --input-dir outputs/human_none --output-dir outputs/pred_none
    python src/predict.py --input-dir outputs/human_reinhard --output-dir outputs/pred_reinhard

Pass --stain-norm reinhard here only for images that were NOT already normalized
during preprocessing.
"""

import argparse
import glob
import os

import cv2
import numpy as np
import yaml

from config import REPO_ROOT, get_data_root, resolve_path
from stain_norm import add_stain_norm_args, apply_stain_norm, check_stain_args, reference_from_args

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "baseline.yml"
)
IMAGE_PATTERNS = ("*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png")
LOSSLESS_EXTENSIONS = (".tif", ".tiff", ".png")


def read_raw_config(path=CONFIG_PATH):
    """The YAML as written, without resolving data paths (so $TOAD_DATA_ROOT is not required)."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def list_images(input_directory):
    paths = set()
    for pattern in IMAGE_PATTERNS:
        paths.update(glob.glob(os.path.join(input_directory, pattern)))
        paths.update(glob.glob(os.path.join(input_directory, pattern.upper())))
    return sorted(paths)


def predict_images(
    input_directory,
    output_directory,
    model,
    img_height=None,
    img_width=None,
    stain_norm="none",
    ref_means=None,
    ref_stds=None,
):
    if img_height is None or img_width is None:
        data_cfg = read_raw_config()["data"]
        img_height = data_cfg["img_height"] if img_height is None else img_height
        img_width = data_cfg["img_width"] if img_width is None else img_width

    original_pred_folder = os.path.join(output_directory, "Original_Predictions")
    visualized_pred_folder = os.path.join(output_directory, "Visualized_Predictions")

    os.makedirs(original_pred_folder, exist_ok=True)
    os.makedirs(visualized_pred_folder, exist_ok=True)

    for img_path in list_images(input_directory):
        img = cv2.imread(img_path, 1)
        if img is None:
            print(f"Skipping {img_path}: Could not read image.")
            continue
        if img.shape[2] != 3:
            print(f"Skipping {img_path}: Not an RGB image.")
            continue

        # Normalize on the full image, before resizing (same order as preprocessing).
        img = apply_stain_norm(img, stain_norm, ref_means=ref_means, ref_stds=ref_stds)

        if img.shape[:2] != (img_height, img_width):
            print(f"Resizing {img_path} to {img_width}x{img_height}")
            img = cv2.resize(img, (img_width, img_height))

        img = (img / 255.0).astype(np.float32)
        img_input = np.expand_dims(img, 0)
        prediction = model.predict(img_input, verbose=0)
        predicted_img = np.argmax(prediction, axis=-1)[0, :, :].astype(np.uint8)

        # Class IDs must be stored losslessly: a .jpg input gets a .png prediction.
        stem, ext = os.path.splitext(os.path.basename(img_path))
        if ext.lower() not in LOSSLESS_EXTENSIONS:
            ext = ".png"
        name = stem + ext
        cv2.imwrite(os.path.join(original_pred_folder, f"prediction_{name}"), predicted_img)

        pred_max = np.max(predicted_img)
        normalized_img = (
            (predicted_img / pred_max * 255) if pred_max > 0 else np.zeros_like(predicted_img, dtype=np.uint8)
        ).astype(np.uint8)
        cv2.imwrite(
            os.path.join(visualized_pred_folder, f"visualized_prediction_{name}"),
            normalized_img,
        )


def main():
    raw = read_raw_config()
    data_cfg = raw.get("data") or {}
    data_paths = raw.get("data_paths") or {}
    default_output = resolve_path((raw.get("paths") or {}).get("output_dir", "outputs"), REPO_ROOT)

    parser = argparse.ArgumentParser(description="Run a SEKO model over a folder of images")
    parser.add_argument("--model", default=None, help="Path to .keras model (default: data_paths.weights_in)")
    parser.add_argument("--input-dir", default=None, help="Image folder (default: data_paths.input_dir)")
    parser.add_argument("--output-dir", default=str(default_output))
    parser.add_argument("--height", type=int, default=data_cfg["img_height"])
    parser.add_argument("--width", type=int, default=data_cfg["img_width"])
    add_stain_norm_args(parser, default=data_cfg.get("stain_norm", "none"))
    args = parser.parse_args()
    check_stain_args(parser, args)
    ref_means, ref_stds = reference_from_args(args)

    if args.model is None or args.input_dir is None:
        data_root = get_data_root()  # explains how to set $TOAD_DATA_ROOT if it is missing
        if args.model is None:
            args.model = str(resolve_path(data_paths["weights_in"], data_root))
        if args.input_dir is None:
            args.input_dir = str(resolve_path(data_paths["input_dir"], data_root))

    from keras.models import load_model

    model = load_model(args.model)
    print(f"Stain normalization: {args.stain_norm}")
    predict_images(
        args.input_dir,
        args.output_dir,
        model,
        args.height,
        args.width,
        stain_norm=args.stain_norm,
        ref_means=ref_means,
        ref_stds=ref_stds,
    )


if __name__ == "__main__":
    main()
