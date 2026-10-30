"""
Yo! This script runs a forward pass of a trained SEKO model on a single input image and saves:
  1) The raw predicted mask (class IDs) as a .png
  2) A grayscale visualization of the mask (also .png)
  3) An overlay of the mask on the original image for easy visual inspection
"""

import argparse
import os
import time

import cv2
import numpy as np
from keras.models import load_model

# Reusable visualization helpers
from visualize_mask import save_grayscale, overlay_mask
from stain_norm import add_stain_norm_args, apply_stain_norm, check_stain_args, reference_from_args


def load_and_preprocess_image(
    image_path: str,
    height: int,
    width: int,
    rgb: bool,
    stain_norm: str = "none",
    ref_means=None,
    ref_stds=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stain normalization (if any) runs on the full image, before the resize.

    Returns:
      x: (1, H, W, 3) float32 in [0,1] for model input
      img_rgb: (H, W, 3) uint8 RGB for visualization/overlay
    """
    img_bgr = cv2.imread(image_path, 1)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Image must be 3-channel (got shape {img_bgr.shape})")

    img_bgr = apply_stain_norm(img_bgr, stain_norm, ref_means=ref_means, ref_stds=ref_stds)

    if img_bgr.shape[:2] != (height, width):
        img_bgr = cv2.resize(img_bgr, (width, height))

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # If model expects RGB, feed img_rgb. If it was trained on BGR, feed BGR.
    feed = img_rgb if rgb else img_bgr
    x = (feed.astype(np.float32) / 255.0)[None, ...]  # (1,H,W,3)

    return x, img_rgb


def forward_pass_single_image(
    model_path: str,
    image_path: str,
    output_path: str,
    height: int,
    width: int,
    rgb: bool,
    alpha: float,
    background_class: int,
    stain_norm: str = "none",
    ref_means=None,
    ref_stds=None,
):
    print(f"Loading model: {model_path}")
    model = load_model(model_path, compile=False)

    print(f"Preprocessing image: {image_path} (stain norm: {stain_norm})")
    x, img_rgb = load_and_preprocess_image(
        image_path, height, width, rgb=rgb, stain_norm=stain_norm, ref_means=ref_means, ref_stds=ref_stds
    )

    print(f"Input tensor shape: {x.shape}, dtype: {x.dtype}, min/max: {x.min():.4f}/{x.max():.4f}")

    print("Running forward pass...")
    t0 = time.time()
    pred = model.predict(x, verbose=0)  # (1, H, W, K)
    t1 = time.time()

    n_classes = int(pred.shape[-1])
    pred_ids = np.argmax(pred, axis=-1)[0].astype(np.uint8)  # (H, W)

    unique = np.unique(pred_ids)
    print(f"Forward pass time: {(t1 - t0) * 1000:.2f} ms")
    print(f"Pred unique labels: {unique} (max={int(pred_ids.max())}, n_classes={n_classes})")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1) Save raw mask (class IDs)
    cv2.imwrite(output_path, pred_ids)
    print(f"Saved raw mask: {output_path}")

    # 2) Save viewable grayscale visualization
    base, _ = os.path.splitext(output_path)
    vis_path = base + "_vis.png"
    save_grayscale(pred_ids, vis_path)
    print(f"Saved visualization: {vis_path}")

    # 3) Save overlay on the original image
    overlay = overlay_mask(
        image_rgb=img_rgb,
        mask_ids=pred_ids,
        n_classes=n_classes,
        alpha=alpha,
        background_class=background_class,
    )
    overlay_path = base + "_overlay.png"
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved overlay: {overlay_path}")

    if len(unique) == 1 and unique[0] == background_class:
        print(
            "NOTE: Prediction is all background. If unexpected, try:\n"
            "  - rerun with/without --rgb (BGR vs RGB mismatch)\n"
            "  - verify preprocessing matches training (size/normalization)\n"
            "  - confirm you loaded the correct checkpoint"
        )


def main():
    parser = argparse.ArgumentParser(description="Run SEKO forward pass on a single image with overlay outputs")
    parser.add_argument("--model", required=True, help="Path to trained model (.keras/.hdf5)")
    parser.add_argument("--image", required=True, help="Path to input image (quote if it has spaces)")
    parser.add_argument("--output", required=True, help="Path to save raw predicted mask (class IDs)")

    parser.add_argument("--height", type=int, default=192, help="Model input height")
    parser.add_argument("--width", type=int, default=256, help="Model input width")

    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Convert BGR->RGB before feeding model (use if model trained on RGB)",
    )

    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay blend strength (0..1)")
    parser.add_argument("--background-class", type=int, default=0, help="Which class id is background")
    add_stain_norm_args(parser)

    args = parser.parse_args()
    check_stain_args(parser, args)
    ref_means, ref_stds = reference_from_args(args)

    forward_pass_single_image(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        height=args.height,
        width=args.width,
        rgb=args.rgb,
        alpha=args.alpha,
        background_class=args.background_class,
        stain_norm=args.stain_norm,
        ref_means=ref_means,
        ref_stds=ref_stds,
    )


if __name__ == "__main__":
    main()
