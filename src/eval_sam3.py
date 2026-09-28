import argparse
import contextlib
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment

from sam3 import build_sam3_image_model
from train_sam3 import (
    TISSUE_LABELS,
    TISSUE_NAMES,
    _extract_prompted_instances,
    _merge_osteophyte_into_cartilage,
    _transform,
    build_find_input,
    build_find_target,
    build_prompt,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "baseline.yml")


# ── Image / mask loading ───────────────────────────────────────────────────

def load_image(path, img_h, img_w):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise IOError(f"Cannot read: {path}")
    if img_bgr.shape[:2] != (img_h, img_w):
        img_bgr = cv2.resize(img_bgr, (img_w, img_h))
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(image_rgb).permute(2, 0, 1)
    return image_rgb, _transform(t).unsqueeze(0)   # (1, 3, 1008, 1008)

def load_mask(path, img_h, img_w):
    mask = np.array(Image.open(path)).astype(np.uint8)
    if mask.shape[:2] != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return mask

def extract_classes(seko_mask, gt_mask):
    """
    Returns (boxes_cxcywh, binary_masks, label_ids) — boxes come from the
    SEKO-predicted mask (matching what train_sam3.py trains against and
    what's available at inference time with no GT), binary_masks (the DICE
    target) come from the GT mask.
    """
    boxes_list, masks_list, labels_list = _extract_prompted_instances(seko_mask, gt_mask)
    if not boxes_list:
        H, W = seko_mask.shape
        return torch.zeros((0, 4)), torch.zeros((0, H, W), dtype=torch.bool), []
    return (
        torch.tensor(boxes_list, dtype=torch.float32),
        torch.from_numpy(np.stack(masks_list)),
        labels_list,
    )


# ── Per-image inference + per-class DICE ──────────────────────────────────

@torch.no_grad()
def evaluate_image(model, image_tensor, boxes_cxcywh, binary_masks, label_ids, device):
    """
    Returns dict {label_id: dice_score} for each class present in the image.
    Uses Hungarian matching to pair predicted masks with GT classes.
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
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=find_target,
            geometric_prompt=prompt,
        )

    pred_masks = out.get("pred_masks")
    pred_boxes = out.get("pred_boxes")
    if pred_masks is None or pred_boxes is None:
        return {}

    gt_boxes = boxes_cxcywh.to(device)
    pred_b   = pred_boxes[0].detach()
    cost     = torch.cdist(gt_boxes, pred_b, p=1).cpu().numpy()
    gt_idx, pred_idx = linear_sum_assignment(cost)

    if len(pred_idx) == 0:
        return {}

    matched_preds = pred_masks[0, pred_idx]                      # (M, H_pred, W_pred)
    matched_gt    = binary_masks[gt_idx].to(device).float()      # (M, H_orig, W_orig)

    H_pred, W_pred = matched_preds.shape[-2], matched_preds.shape[-1]
    matched_gt = F.interpolate(
        matched_gt.unsqueeze(1), size=(H_pred, W_pred), mode="nearest"
    ).squeeze(1)

    pred_bin  = (matched_preds.sigmoid() > 0.5).float()
    gt_area   = matched_gt.sum(dim=(-2, -1))
    inter     = (pred_bin * matched_gt).sum(dim=(-2, -1))
    union     = pred_bin.sum(dim=(-2, -1)) + gt_area
    dice_vals = (2.0 * inter + 1e-6) / (union + 1e-6)

    # Boxes come from SEKO, targets from GT — the two can disagree on
    # whether a class is present at all (SEKO drew a box for it, GT has
    # none). Skip those: an empty-vs-empty match scores DICE=1.0 under the
    # epsilon smoothing above, which would silently inflate the reported
    # metric for a class that was never actually there to segment.
    return {
        label_ids[gt_idx[i]]: dice_vals[i].item()
        for i in range(len(gt_idx))
        if gt_area[i] > 0
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned SAM3 on TOAD test set")
    parser.add_argument("--ckpt",     default=cfg["sam3"].get("ckpt_path"),
                        help="Fine-tuned checkpoint (default: baseline.yml ckpt_path)")
    parser.add_argument("--csv",      default=cfg["paths"]["test_seko_csv"],
                        help="Path to test_seko.csv (image_path, mask_path, "
                             "seko_mask_path) — generated by "
                             "src/utils/cache_seko_masks.py --test-dir")
    parser.add_argument("--device",   default=cfg["sam3"].get("device", "cuda"))
    parser.add_argument("--out-json", default="results.json",
                        help="Where to write structured per-class + per-image "
                             "results (default: results.json in cwd) — see "
                             "src/utils/load_results.py")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    img_h = cfg["data"]["img_height"]
    img_w = cfg["data"]["img_width"]

    # ── Load test pairs ──────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    pairs = list(zip(df["image_path"], df["mask_path"], df["seko_mask_path"]))
    print(f"Test pairs found: {len(pairs)}")

    # ── Load model ───────────────────────────────────────────────────────
    ckpt = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", args.ckpt)) \
           if args.ckpt and not os.path.isabs(args.ckpt) else args.ckpt
    load_from_hf = not ckpt or not os.path.isfile(ckpt)
    print(f"Loading checkpoint: {'HuggingFace' if load_from_hf else ckpt}")

    model = build_sam3_image_model(
        checkpoint_path=ckpt if not load_from_hf else None,
        load_from_HF=load_from_hf,
        device=str(device),
        eval_mode=True,
    )
    model.eval()

    # ── Evaluate ─────────────────────────────────────────────────────────
    class_dice_lists = defaultdict(list)   # label_id → [dice, ...]
    per_image_results = []                 # for results.json / visualization picks
    skipped = 0

    for i, (img_path, mask_path, seko_mask_path) in enumerate(pairs):
        try:
            _, image_tensor = load_image(img_path, img_h, img_w)
            gt_mask   = _merge_osteophyte_into_cartilage(load_mask(mask_path, img_h, img_w))
            seko_mask = load_mask(seko_mask_path, img_h, img_w)
            boxes, binary_masks, label_ids = extract_classes(seko_mask, gt_mask)

            if len(label_ids) == 0:
                skipped += 1
                continue

            per_class = evaluate_image(
                model, image_tensor, boxes, binary_masks, label_ids, device
            )
            for cls_id, dice in per_class.items():
                class_dice_lists[cls_id].append(dice)

            per_image_results.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "seko_mask_path": seko_mask_path,
                "per_class_dice": {
                    TISSUE_NAMES[c]: round(d, 6) for c, d in per_class.items()
                },
                "mean_dice": round(np.mean(list(per_class.values())), 6) if per_class else None,
            })

            print(f"  [{i+1:3d}/{len(pairs)}] {os.path.basename(img_path):50s} "
                  + "  ".join(f"{TISSUE_NAMES[c]}={d:.3f}"
                               for c, d in sorted(per_class.items())),
                  flush=True)

        except Exception as e:
            print(f"  [ERROR] {os.path.basename(img_path)}: {e}")
            skipped += 1

    # ── Results table ─────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"{'Class':<20} {'N Images':>8}  {'Mean DICE':>9}  {'Std':>6}")
    print("-" * 52)

    all_dice = []
    for cls_id in TISSUE_LABELS:
        scores = class_dice_lists.get(cls_id, [])
        if scores:
            mean = np.mean(scores)
            std  = np.std(scores)
            all_dice.extend(scores)
            print(f"{TISSUE_NAMES[cls_id]:<20} {len(scores):>8}  {mean:>9.4f}  {std:>6.4f}")
        else:
            print(f"{TISSUE_NAMES[cls_id]:<20} {'—':>8}  {'—':>9}  {'—':>6}")

    print("-" * 52)
    if all_dice:
        print(f"{'Mean (all classes)':<20} {len(pairs)-skipped:>8}  {np.mean(all_dice):>9.4f}")
    print("=" * 52)
    if skipped:
        print(f"Skipped: {skipped} images (no valid tissue classes or load error)")

    # ── Structured results (for load_results.py / picking images to visualize) ──
    results = {
        "ckpt": ckpt,
        "csv": args.csv,
        "n_pairs": len(pairs),
        "n_skipped": skipped,
        "mean_dice": round(float(np.mean(all_dice)), 6) if all_dice else None,
        "per_class": {
            TISSUE_NAMES[cls_id]: {
                "n_images": len(class_dice_lists.get(cls_id, [])),
                "mean_dice": round(float(np.mean(class_dice_lists[cls_id])), 6)
                             if class_dice_lists.get(cls_id) else None,
                "std_dice": round(float(np.std(class_dice_lists[cls_id])), 6)
                            if class_dice_lists.get(cls_id) else None,
            }
            for cls_id in TISSUE_LABELS
        },
        "per_image": per_image_results,
    }
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to: {args.out_json}")


if __name__ == "__main__":
    main()
