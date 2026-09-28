# -*- coding: utf-8 -*-
"""
visualize_sam3_predictions.py

Runs a fine-tuned SAM-3 checkpoint on sample images from test_seko.csv and
saves a side-by-side comparison PNG per image:

    [ original | SEKO prediction (+ box prompts) | ground truth | SAM-3 prediction ]

using the same color palette for all three mask panels (src/visualize_mask.py)
so they're directly comparable.

Requires the seko-sam3 environment (needs torch + the sam3 package).

Usage
-----
  # N random images from test_seko.csv
  python src/utils/visualize_sam3_predictions.py --ckpt path/to/sam3_toad_best.pt --n 8

  # specific images
  python src/utils/visualize_sam3_predictions.py --ckpt path/to/best.pt \\
      --image-path /path/to/img1.tif /path/to/img2.tif

  # the worst/best-scoring images from a prior eval_sam3.py run
  python src/utils/visualize_sam3_predictions.py --ckpt path/to/best.pt \\
      --from-results results.json --n 5
"""

import argparse
import contextlib
import json
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(__file__) + "/..")   # allow `import eval_sam3`, `import train_sam3`

from sam3 import build_sam3_image_model
from train_sam3 import TISSUE_NAMES, build_find_input, build_find_target, build_prompt
from eval_sam3 import extract_classes, load_image, load_mask, _merge_osteophyte_into_cartilage
from visualize_mask import overlay_mask

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "baseline.yml")
N_CLASSES = 5   # background + bone, cartilage, gp, marrow


@torch.no_grad()
def predict_mask(model, image_tensor, boxes_cxcywh, binary_masks, label_ids, device, H, W):
    """
    Runs one forward pass and returns (pred_mask_full (H,W) uint8 in the
    TOAD label schema, per_class_dice dict) — like eval_sam3.evaluate_image,
    but also materializes the predicted mask for visualization instead of
    only the DICE score.
    """
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" else contextlib.nullcontext()
    )
    image_tensor = image_tensor.to(device)

    with autocast_ctx:
        backbone_out = model.backbone.forward_image(image_tensor)
        text_out     = model.backbone.forward_text(["visual"], device=device)
        backbone_out.update(text_out)

        prompt      = build_prompt(boxes_cxcywh, device)
        find_input  = build_find_input(device)
        find_target = build_find_target(boxes_cxcywh, binary_masks, device)

        out = model.forward_grounding(
            backbone_out=backbone_out, find_input=find_input,
            find_target=find_target, geometric_prompt=prompt,
        )

    pred_masks = out.get("pred_masks")
    pred_boxes = out.get("pred_boxes")
    pred_mask_full = np.zeros((H, W), dtype=np.uint8)
    if pred_masks is None or pred_boxes is None:
        return pred_mask_full, {}

    gt_boxes = boxes_cxcywh.to(device)
    pred_b   = pred_boxes[0].detach()
    cost     = torch.cdist(gt_boxes, pred_b, p=1).cpu().numpy()
    gt_idx, pred_idx = linear_sum_assignment(cost)
    if len(pred_idx) == 0:
        return pred_mask_full, {}

    matched_preds = pred_masks[0, pred_idx].sigmoid()           # (M, h, w)
    matched_preds = torch.nn.functional.interpolate(
        matched_preds.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(1)
    pred_bin = (matched_preds > 0.5).cpu().numpy()

    # Later classes overwrite earlier ones on overlap, same convention as
    # sam_adapter.py's SamAdapter.predict / predict.py.
    for i in range(len(gt_idx)):
        cls = label_ids[gt_idx[i]]
        pred_mask_full[pred_bin[i]] = cls

    return pred_mask_full, {}


def draw_boxes(img_rgb, boxes_cxcywh, color=(255, 255, 255)):
    out = img_rgb.copy()
    H, W = img_rgb.shape[:2]
    for cx, cy, bw, bh in boxes_cxcywh.tolist():
        x1 = int((cx - bw / 2) * W); x2 = int((cx + bw / 2) * W)
        y1 = int((cy - bh / 2) * H); y2 = int((cy + bh / 2) * H)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
    return out


def process_one(model, img_path, mask_path, seko_mask_path, img_h, img_w, device, out_dir):
    stem = os.path.splitext(os.path.basename(img_path))[0]

    image_rgb, image_tensor = load_image(img_path, img_h, img_w)
    gt_mask   = _merge_osteophyte_into_cartilage(load_mask(mask_path, img_h, img_w))
    seko_mask = load_mask(seko_mask_path, img_h, img_w)
    boxes, binary_masks, label_ids = extract_classes(seko_mask, gt_mask)

    if len(label_ids) == 0:
        print(f"  [skip] {stem}: no SEKO-detected classes")
        return None

    pred_mask, _ = predict_mask(
        model, image_tensor, boxes, binary_masks, label_ids, device, img_h, img_w
    )

    seko_panel = overlay_mask(image_rgb, seko_mask, N_CLASSES)
    seko_panel = draw_boxes(seko_panel, boxes)
    gt_panel   = overlay_mask(image_rgb, gt_mask, N_CLASSES)
    pred_panel = overlay_mask(image_rgb, pred_mask, N_CLASSES)

    labels_row = ["original", "SEKO pred + box prompt", "ground truth", "SAM-3 prediction"]
    stack = np.concatenate([image_rgb, seko_panel, gt_panel, pred_panel], axis=1)

    out_path = os.path.join(out_dir, f"{stem}_compare.png")
    cv2.imwrite(out_path, cv2.cvtColor(stack, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out_path}  ({' | '.join(labels_row)})")
    return out_path


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Visualize fine-tuned SAM-3 predictions vs SEKO/GT")
    parser.add_argument("--ckpt", required=True, help="Fine-tuned SAM-3 checkpoint (.pt)")
    parser.add_argument("--csv", default=cfg["paths"]["test_seko_csv"])
    parser.add_argument("--device", default=cfg["sam3"].get("device", "cuda"))
    parser.add_argument("--out-dir", default="viz")
    parser.add_argument("--n", type=int, default=6, help="Number of images (random, unless --from-results)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-path", nargs="*", default=None,
                        help="Specific image_path values from the CSV to visualize, instead of sampling")
    parser.add_argument("--from-results", default=None,
                        help="results.json from eval_sam3.py — pick the --n lowest- and "
                             "--n highest-DICE images instead of a random sample")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    by_path = {row["image_path"]: row for _, row in df.iterrows()}

    if args.image_path:
        rows = [by_path[p] for p in args.image_path if p in by_path]
    elif args.from_results:
        with open(args.from_results) as f:
            results = json.load(f)
        scored = sorted(
            [r for r in results["per_image"] if r["mean_dice"] is not None],
            key=lambda r: r["mean_dice"],
        )
        picks = scored[:args.n] + scored[-args.n:]
        rows = [by_path[r["image_path"]] for r in picks if r["image_path"] in by_path]
    else:
        random.seed(args.seed)
        rows = random.sample(list(df.to_dict("records")), min(args.n, len(df)))

    print(f"Visualizing {len(rows)} images")

    img_h = cfg["data"]["img_height"]
    img_w = cfg["data"]["img_width"]

    load_from_hf = not args.ckpt or not os.path.isfile(args.ckpt)
    print(f"Loading checkpoint: {'HuggingFace' if load_from_hf else args.ckpt}")
    model = build_sam3_image_model(
        checkpoint_path=args.ckpt if not load_from_hf else None,
        load_from_HF=load_from_hf, device=str(device), eval_mode=True,
    )
    model.eval()

    for row in rows:
        row = dict(row)
        process_one(
            model, row["image_path"], row["mask_path"], row["seko_mask_path"],
            img_h, img_w, device, args.out_dir,
        )

    print(f"\nDone. Panels saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
