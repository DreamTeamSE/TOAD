"""
evaluate_predictions.py — score nnU-Net's contract-format test predictions
against ground truth, for the issue #14 baseline showcase.

For each test-split row in conversion_manifest.csv:
  - ground truth = manifest's mask_path, with merge_osteophyte() applied
    (the manifest's mask_path is the *original* annotation, still in raw
    0-5 label space -- the contract-space remap only happens when
    convert_to_nnunet.py writes labelsTr/labelsTs, which predict_nnunet.slurm
    never populates for the test split).
  - prediction = predictions-dir/<original_stem(image_path)>.png, written by
    export_predictions.py.

Produces, under --out-dir:
  - metrics.csv          per-class Dice/IoU (mean +/- std across images),
                          precision/recall/F1 and pixel accuracy from the
                          pooled confusion matrix.
  - confusion_matrix.png row-normalized 5x5 heatmap.
  - overlays/*.png        a handful of original/ground-truth/prediction panels.
  - RESULTS.md            short written summary pulling in the above, for
                          the midpoint showcase.

No scikit-learn/seaborn dependency -- environment_nnunet.yml only grew a
matplotlib pin for this, confusion matrix + per-class stats are plain numpy.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, for labels
from labels import CLASS_COLORS, CLASS_NAMES, merge_osteophyte, validate_mask  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir, for export_predictions
from export_predictions import load_manifest, original_stem  # noqa: E402

NUM_CLASSES = len(CLASS_NAMES)


def load_ground_truth(mask_path: str) -> np.ndarray:
    """Original annotation -> contract label space (osteophyte merged into cartilage)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read ground-truth mask: {mask_path}")
    merged = merge_osteophyte(mask)
    validate_mask(merged)
    return merged


def pixel_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Rows = ground truth, columns = prediction."""
    idx = (y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)).ravel()
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def per_class_dice_iou(conf_mat: np.ndarray) -> dict:
    """Dice/IoU per class from one image's confusion matrix; NaN where the class is absent from both GT and prediction."""
    tp = np.diag(conf_mat)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = np.where(denom_dice > 0, (2 * tp) / denom_dice, np.nan)
        iou = np.where(denom_iou > 0, tp / denom_iou, np.nan)
    return {"dice": dice, "iou": iou}


def evaluate(manifest_csv: str, predictions_dir: str) -> dict:
    """
    Walk every test-split row, score prediction vs. ground truth.

    Returns a dict with: per-image per-class dice/iou arrays, the pooled
    confusion matrix (summed over all images), and the list of (case_id,
    image_path, mask_path, pred_path) tuples actually scored.
    """
    rows = load_manifest(manifest_csv)

    pooled_conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_image_dice, per_image_iou = [], []
    scored, missing = [], []

    for row in rows:
        pred_path = os.path.join(predictions_dir, f"{original_stem(row['image_path'])}.png")
        if not os.path.isfile(pred_path):
            missing.append(row["case_id"])
            continue

        gt = load_ground_truth(row["mask_path"])
        pred = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)
        if pred is None:
            raise ValueError(f"Could not read prediction: {pred_path}")
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {row['case_id']}: GT {gt.shape} vs prediction {pred.shape}")

        conf_mat = pixel_confusion_matrix(gt, pred)
        pooled_conf_mat += conf_mat
        stats = per_class_dice_iou(conf_mat)
        per_image_dice.append(stats["dice"])
        per_image_iou.append(stats["iou"])
        scored.append({**row, "pred_path": pred_path})

    if missing:
        print(f"[WARN] {len(missing)} test case(s) had no matching prediction file in {predictions_dir}:")
        for case_id in missing:
            print(f"  {case_id}")

    return {
        "per_image_dice": np.array(per_image_dice),   # (n_images, NUM_CLASSES)
        "per_image_iou": np.array(per_image_iou),
        "pooled_conf_mat": pooled_conf_mat,
        "scored": scored,
    }


def write_metrics_csv(results: dict, out_path: str) -> list[dict]:
    """Per-class Dice/IoU (mean +/- std across images) + precision/recall/F1/accuracy from the pooled confusion matrix."""
    conf_mat = results["pooled_conf_mat"]
    tp = np.diag(conf_mat)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), np.nan)

    rows = []
    for cid, name in CLASS_NAMES.items():
        dice_col = results["per_image_dice"][:, cid]
        iou_col = results["per_image_iou"][:, cid]
        rows.append({
            "class_id": cid,
            "class_name": name,
            "dice_mean": np.nanmean(dice_col),
            "dice_std": np.nanstd(dice_col),
            "iou_mean": np.nanmean(iou_col),
            "iou_std": np.nanstd(iou_col),
            "precision": precision[cid],
            "recall": recall[cid],
            "f1": f1[cid],
            "n_images_present": int(np.sum(~np.isnan(dice_col))),
            "pixel_accuracy": "",
        })

    pixel_accuracy = tp.sum() / conf_mat.sum()
    rows.append({
        "class_id": "overall",
        "class_name": "mean across classes",
        "dice_mean": np.nanmean(results["per_image_dice"]),
        "dice_std": np.nan,
        "iou_mean": np.nanmean(results["per_image_iou"]),
        "iou_std": np.nan,
        "precision": np.nanmean(precision),
        "recall": np.nanmean(recall),
        "f1": np.nanmean(f1),
        "n_images_present": "",
        "pixel_accuracy": pixel_accuracy,
    })

    import csv as csv_module
    with open(out_path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def plot_confusion_matrix(conf_mat: np.ndarray, out_path: str) -> None:
    """Row-normalized (% of each ground-truth class) confusion matrix heatmap."""
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    normalized = np.divide(conf_mat, row_sums, out=np.zeros_like(conf_mat, dtype=float), where=row_sums > 0) * 100
    names = list(CLASS_NAMES.values())

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Ground truth", fontweight="bold")
    ax.set_title("nnU-Net Rat Baseline -- Confusion Matrix (% of row)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{normalized[i, j]:.1f}", ha="center", va="center",
                     color="white" if normalized[i, j] > 50 else "black")
    fig.colorbar(im, ax=ax, label="% of ground-truth class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def colorize(mask: np.ndarray) -> np.ndarray:
    """Map class IDs to CLASS_COLORS for display."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        out[mask == cid] = color
    return out


def save_overlays(results: dict, out_dir: str, num_overlays: int, seed: int) -> list[str]:
    """Original image | ground truth | prediction, side by side, for a sample of scored test cases."""
    os.makedirs(out_dir, exist_ok=True)
    scored = results["scored"]
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(scored), size=min(num_overlays, len(scored)), replace=False)

    saved = []
    for idx in sample:
        row = scored[idx]
        image = cv2.cvtColor(cv2.imread(row["image_path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gt = colorize(load_ground_truth(row["mask_path"]))
        pred = colorize(cv2.imread(row["pred_path"], cv2.IMREAD_UNCHANGED))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, title in zip(axes, (image, gt, pred), ("Input", "Ground truth", "Prediction")):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(row["case_id"])
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{row['case_id']}_overlay.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)

    return saved


def write_results_md(metrics_rows: list[dict], num_scored: int, overlay_paths: list[str], out_path: str) -> None:
    lines = [
        "# nnU-Net Rat Baseline -- Results (Issue #14)",
        "",
        "Single 2D fold, 250 epochs, `hpg-turin`/L4, scored against the held-out "
        f"test split ({num_scored} cases).",
        "",
        "## Per-class scores",
        "",
        "| Class | Dice (mean +/- std) | IoU (mean +/- std) | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for row in metrics_rows:
        if row["class_id"] == "overall":
            continue
        lines.append(
            f"| {row['class_name']} "
            f"| {row['dice_mean']:.3f} +/- {row['dice_std']:.3f} "
            f"| {row['iou_mean']:.3f} +/- {row['iou_std']:.3f} "
            f"| {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )
    overall = next(r for r in metrics_rows if r["class_id"] == "overall")
    lines += [
        "",
        f"**Mean Dice (all classes): {overall['dice_mean']:.3f}** -- "
        f"**Mean IoU: {overall['iou_mean']:.3f}** -- "
        f"**Pixel accuracy: {overall['pixel_accuracy']:.3f}**",
        "",
        "![Confusion matrix](confusion_matrix.png)",
        "",
        "## Qualitative examples",
        "",
    ]
    for p in overlay_paths:
        lines.append(f"![{os.path.basename(p)}](overlays/{os.path.basename(p)})")
    lines += [
        "",
        "## Caveats",
        "",
        "- Single train/test split, single fold (fold 0) -- no cross-validation, "
        "so these numbers carry no confidence interval across data splits.",
        "- Class pixel counts are highly imbalanced (background/bone dominate; "
        "growth plate is a thin, small structure) -- a low score on a rare class "
        "moves the per-image Dice/IoU less than its clinical importance warrants.",
        "- Ground truth here is the original annotation with `merge_osteophyte()` "
        "applied, matching the contract space the prediction is already in.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "sbatch hpg/nnunet/predict_nnunet.slurm",
        "python src/baselines/nnunet/evaluate_predictions.py \\",
        "    --manifest-csv $MANIFEST_CSV --predictions-dir $EXPORT_DIR --out-dir <out>",
        "```",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True, help="conversion_manifest.csv from convert_to_nnunet.py")
    parser.add_argument("--predictions-dir", required=True, help="Contract-format predictions (export_predictions.py's --out-dir)")
    parser.add_argument("--out-dir", required=True, help="Where to write metrics.csv, confusion_matrix.png, overlays/, RESULTS.md")
    parser.add_argument("--num-overlays", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = evaluate(args.manifest_csv, args.predictions_dir)

    metrics_rows = write_metrics_csv(results, os.path.join(args.out_dir, "metrics.csv"))
    plot_confusion_matrix(results["pooled_conf_mat"], os.path.join(args.out_dir, "confusion_matrix.png"))
    overlay_paths = save_overlays(results, os.path.join(args.out_dir, "overlays"), args.num_overlays, args.seed)
    write_results_md(metrics_rows, len(results["scored"]), overlay_paths, os.path.join(args.out_dir, "RESULTS.md"))

    overall = next(r for r in metrics_rows if r["class_id"] == "overall")
    print(f"Scored {len(results['scored'])} test cases.")
    print(f"Mean Dice: {overall['dice_mean']:.3f}  Mean IoU: {overall['iou_mean']:.3f}  Pixel accuracy: {overall['pixel_accuracy']:.3f}")
    print(f"Wrote metrics.csv, confusion_matrix.png, overlays/, RESULTS.md -> {args.out_dir}")


if __name__ == "__main__":
    main()
