"""Compare two sets of predicted masks, e.g. --stain-norm none vs --stain-norm reinhard.

Files are paired by stem, ignoring a leading ``prediction_`` (predict.py's naming),
so ``prediction_X_tile_00000_00192.tif`` in A pairs with the same name in B and with
``X_tile_00000_00192.png`` in --gt-dir.

Always reports:
  - pixel agreement between A and B (how much the setting changes the output)
  - share of pixels assigned to each class in A and in B

With --gt-dir (masks written by preprocess_human_data.py --mask-dir) also reports
pooled Dice per class for A and B. For --species human, growth plate is excluded from
Dice (it is never labelled) and any pixel predicted as growth plate is counted as an
error and reported separately.

    python src/compare_predictions.py \\
        --a outputs/pred_none/Original_Predictions --name-a none \\
        --b outputs/pred_reinhard/Original_Predictions --name-b reinhard \\
        --gt-dir outputs/human_none/masks --species human --csv outputs/stain_compare.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from labels import CLASS_NAMES, GROWTH_PLATE, HUMAN_ABSENT_CLASSES, NUM_CLASSES

MASK_EXTENSIONS = {".png", ".tif", ".tiff"}
PREDICTION_PREFIX = "prediction_"


def pairing_key(path: Path) -> str:
    stem = path.stem
    return stem[len(PREDICTION_PREFIX):] if stem.startswith(PREDICTION_PREFIX) else stem


def index_dir(folder: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS:
            key = pairing_key(p)
            if key in index:
                raise ValueError(f"Duplicate key {key!r} in {folder}: {index[key].name}, {p.name}")
            index[key] = p
    return index


def read_ids(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not read {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int64)


def class_counts(mask: np.ndarray) -> np.ndarray:
    return np.bincount(mask.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]


def out_of_range(mask: np.ndarray) -> int:
    """Pixels with values outside the contract's class IDs (e.g. unmerged osteophyte, 255)."""
    return int(np.count_nonzero((mask < 0) | (mask >= NUM_CLASSES)))


class DiceAccumulator:
    """Pooled Dice: sum intersections and sizes over all images, then divide once."""

    def __init__(self) -> None:
        self.inter = np.zeros(NUM_CLASSES, dtype=np.int64)
        self.pred = np.zeros(NUM_CLASSES, dtype=np.int64)
        self.gt = np.zeros(NUM_CLASSES, dtype=np.int64)

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        for c in range(NUM_CLASSES):
            p, g = pred == c, gt == c
            self.inter[c] += int(np.logical_and(p, g).sum())
            self.pred[c] += int(p.sum())
            self.gt[c] += int(g.sum())

    def dice(self, c: int) -> float | None:
        denom = self.pred[c] + self.gt[c]
        return None if denom == 0 else 2.0 * self.inter[c] / denom


def _fmt(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two prediction folders (and optional ground truth).")
    parser.add_argument("--a", type=Path, required=True, help="Prediction folder A")
    parser.add_argument("--b", type=Path, required=True, help="Prediction folder B")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--gt-dir", type=Path, default=None, help="Ground-truth class-ID masks")
    parser.add_argument("--species", choices=("human", "rat"), default="human")
    parser.add_argument("--csv", type=Path, default=None, help="Per-image results CSV")
    args = parser.parse_args()
    if args.name_a == args.name_b:
        parser.error("--name-a and --name-b must differ")

    idx_a, idx_b = index_dir(args.a), index_dir(args.b)
    idx_gt = index_dir(args.gt_dir) if args.gt_dir else {}
    keys = sorted(set(idx_a) & set(idx_b))
    only_a, only_b = set(idx_a) - set(idx_b), set(idx_b) - set(idx_a)
    print(f"Paired images: {len(keys)}  (only in {args.name_a}: {len(only_a)}, only in {args.name_b}: {len(only_b)})")
    if not keys:
        raise SystemExit("Nothing to compare: no matching file names between the two folders.")
    if only_a or only_b:
        print("  Unpaired files are ignored. Different tile sets usually mean the runs used different tiling settings.")

    absent = HUMAN_ABSENT_CLASSES if args.species == "human" else frozenset()
    agree_px = total_px = 0
    counts_a = np.zeros(NUM_CLASSES, dtype=np.int64)
    counts_b = np.zeros(NUM_CLASSES, dtype=np.int64)
    dice_a, dice_b = DiceAccumulator(), DiceAccumulator()
    gt_pairs = 0
    rows = []

    for key in keys:
        pa, pb = read_ids(idx_a[key]), read_ids(idx_b[key])
        if pa.shape != pb.shape:
            print(f"[skip] {key}: shapes differ {pa.shape} vs {pb.shape}")
            continue
        for name, pred in ((args.name_a, pa), (args.name_b, pb)):
            bad = out_of_range(pred)
            if bad:
                print(f"[warn] {key}: {bad} pixels in {name} outside class IDs 0..{NUM_CLASSES - 1}")
        agree = int((pa == pb).sum())
        agree_px += agree
        total_px += pa.size
        ca, cb = class_counts(pa), class_counts(pb)
        counts_a += ca
        counts_b += cb
        row = {"key": key, "agreement": agree / pa.size}

        gt_path = idx_gt.get(key)
        if gt_path is not None:
            gt = read_ids(gt_path)
            bad = out_of_range(gt)
            if bad:
                print(f"[warn] {key}: {bad} ground-truth pixels outside class IDs 0..{NUM_CLASSES - 1} are ignored")
            if gt.shape != pa.shape:
                print(f"[skip gt] {key}: ground truth {gt.shape} vs prediction {pa.shape}")
            else:
                dice_a.add(pa, gt)
                dice_b.add(pb, gt)
                gt_pairs += 1
                row["accuracy_" + args.name_a] = float((pa == gt).mean())
                row["accuracy_" + args.name_b] = float((pb == gt).mean())
        for c in range(NUM_CLASSES):
            row[f"{args.name_a}_frac_{CLASS_NAMES[c]}"] = ca[c] / pa.size
            row[f"{args.name_b}_frac_{CLASS_NAMES[c]}"] = cb[c] / pb.size
        rows.append(row)

    if total_px == 0:
        raise SystemExit("No comparable images (all shapes differed).")

    print(f"\nPixel agreement {args.name_a} vs {args.name_b}: {agree_px / total_px:.4f}")
    print(f"\n{'class':<14}{args.name_a + ' share':>16}{args.name_b + ' share':>16}")
    for c in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[c]:<14}{counts_a[c] / counts_a.sum():>16.4f}{counts_b[c] / counts_b.sum():>16.4f}")
    if GROWTH_PLATE in absent:
        print(
            f"\nGrowth plate is absent in adult human tissue; predicted growth-plate share is "
            f"{counts_a[GROWTH_PLATE] / counts_a.sum():.4f} ({args.name_a}) vs "
            f"{counts_b[GROWTH_PLATE] / counts_b.sum():.4f} ({args.name_b}). Lower is better."
        )

    if args.gt_dir:
        print(f"\nGround truth paired for {gt_pairs} of {len(keys)} images.")
        if gt_pairs:
            print(f"\n{'class':<14}{'Dice ' + args.name_a:>16}{'Dice ' + args.name_b:>16}")
            scored = [c for c in range(NUM_CLASSES) if c not in absent]
            for c in scored:
                print(f"{CLASS_NAMES[c]:<14}{_fmt(dice_a.dice(c)):>16}{_fmt(dice_b.dice(c)):>16}")
            fg = [c for c in scored if c != 0]
            for name, acc in ((args.name_a, dice_a), (args.name_b, dice_b)):
                vals = [acc.dice(c) for c in fg if acc.dice(c) is not None]
                if vals:
                    print(f"Mean foreground Dice ({name}): {np.mean(vals):.3f} over {len(vals)} classes")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "key", k != "agreement", k))
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
