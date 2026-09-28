# -*- coding: utf-8 -*-
"""
load_results.py

Loads and plots results from a train_sam3.py run (metrics.json) and/or an
eval_sam3.py run (results.json), for pulling results off HPG onto a laptop
to inspect.

Usage
-----
  python src/utils/load_results.py --metrics /path/to/metrics.json
  python src/utils/load_results.py --results /path/to/results.json
  python src/utils/load_results.py --metrics metrics.json --results results.json --out-dir plots/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")   # no display needed — just save PNGs
import matplotlib.pyplot as plt


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_training_curves(metrics: dict, out_dir: str):
    epochs = metrics["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, metrics["train_loss"], label="train")
    axes[0].plot(epochs, metrics["val_loss"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, metrics["train_dice"], label="train")
    axes[1].plot(epochs, metrics["val_dice"], label="val")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("DICE")
    axes[1].set_title("DICE")
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Per-class val DICE over epochs, if present
    class_dice_per_epoch = metrics.get("val_class_dice", [])
    if class_dice_per_epoch:
        classes = sorted({k for d in class_dice_per_epoch for k in d})
        fig, ax = plt.subplots(figsize=(7, 4))
        for cls in classes:
            vals = [d.get(cls) for d in class_dice_per_epoch]
            ax.plot(epochs, vals, label=cls)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val DICE")
        ax.set_title("Per-class validation DICE")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(out_dir, "val_class_dice.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out_path}")

    best_epoch = max(range(len(epochs)), key=lambda i: metrics["val_dice"][i])
    print(f"\nBest val DICE: {metrics['val_dice'][best_epoch]:.4f} "
          f"at epoch {epochs[best_epoch]}")
    print(f"Final train/val loss: {metrics['train_loss'][-1]:.4f} / {metrics['val_loss'][-1]:.4f}")


def plot_eval_results(results: dict, out_dir: str):
    per_class = results["per_class"]
    names = list(per_class.keys())
    means = [per_class[n]["mean_dice"] or 0.0 for n in names]
    stds  = [per_class[n]["std_dice"] or 0.0 for n in names]
    ns    = [per_class[n]["n_images"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, means, yerr=stds, capsize=4)
    for i, n in enumerate(ns):
        ax.text(i, means[i] + stds[i] + 0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_ylabel("mean DICE")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Test-set DICE by class (ckpt: {os.path.basename(results['ckpt'] or '?')})")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "eval_per_class_dice.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    print(f"\nOverall mean DICE: {results['mean_dice']}")
    print(f"Images evaluated: {results['n_pairs'] - results['n_skipped']} "
          f"(skipped {results['n_skipped']})")
    print("\nPer-class:")
    for n in names:
        c = per_class[n]
        print(f"  {n:<16} n={c['n_images']:>3}  mean={c['mean_dice']}  std={c['std_dice']}")

    # Worst/best images by mean_dice — good starting points for
    # visualize_sam3_predictions.py
    scored = [r for r in results["per_image"] if r["mean_dice"] is not None]
    scored.sort(key=lambda r: r["mean_dice"])
    if scored:
        print("\nLowest-DICE images (worth visualizing to see failure modes):")
        for r in scored[:5]:
            print(f"  {r['mean_dice']:.4f}  {os.path.basename(r['image_path'])}")
        print("\nHighest-DICE images:")
        for r in scored[-5:]:
            print(f"  {r['mean_dice']:.4f}  {os.path.basename(r['image_path'])}")


def main():
    parser = argparse.ArgumentParser(description="Load and plot train_sam3.py / eval_sam3.py results")
    parser.add_argument("--metrics", default=None, help="Path to metrics.json from train_sam3.py")
    parser.add_argument("--results", default=None, help="Path to results.json from eval_sam3.py")
    parser.add_argument("--out-dir", default="plots", help="Directory to save PNG plots into")
    args = parser.parse_args()

    if not args.metrics and not args.results:
        raise SystemExit("Pass at least one of --metrics or --results")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.metrics:
        print(f"=== Training metrics: {args.metrics} ===")
        plot_training_curves(load_metrics(args.metrics), args.out_dir)

    if args.results:
        print(f"\n=== Eval results: {args.results} ===")
        plot_eval_results(load_results(args.results), args.out_dir)


if __name__ == "__main__":
    main()
