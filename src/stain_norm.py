"""Stain normalization shared by the human preprocessing and the rat (SEKO) scripts.

Every entry point exposes the same ``--stain-norm`` choice so the two sides can be
run under identical conditions:

  none      leave colors untouched (default)
  reinhard  Reinhard transfer in OpenCV LAB space toward a fixed reference

The default Reinhard reference was computed from 50 sampled rat training images
(human_preprocess_colab.ipynb). Pass ``--reinhard-ref-json`` to use another one.

Order: stain normalization runs before the final resize to the model input size.
In preprocess_human_data.py it runs after the physical rescale (--scale/--target-mpp)
and before tiling. That is deliberate: a full-resolution Leica stitch would need a
~13 GB float LAB buffer, and the Reinhard statistics barely change under an area
downscale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

STAIN_NORM_METHODS = ("none", "reinhard")
DEFAULT_STAIN_NORM = "none"

# OpenCV 8-bit LAB (L in 0..255, A/B offset by 128).
DEFAULT_REINHARD_REF_MEANS = (155.6966, 144.2858, 102.3828)  # L, A, B
DEFAULT_REINHARD_REF_STDS = (40.1614, 25.1199, 25.5855)

# Pixels at or above this gray level are treated as glass/background.
TISSUE_GRAY_THRESHOLD = 220


def reinhard_normalize(
    image_uint8: np.ndarray,
    ref_means: tuple[float, float, float] | None = None,
    ref_stds: tuple[float, float, float] | None = None,
    tissue_gray_threshold: int = TISSUE_GRAY_THRESHOLD,
) -> np.ndarray:
    """Shift the LAB mean/std of tissue pixels to the reference, per channel.

    Statistics are measured on tissue only (gray < threshold); the resulting
    linear map is applied to every pixel. Input and output are uint8 BGR.
    """
    if ref_means is None:
        ref_means = DEFAULT_REINHARD_REF_MEANS
    if ref_stds is None:
        ref_stds = DEFAULT_REINHARD_REF_STDS

    lab = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2LAB).astype(np.float32)

    gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    tissue_mask = gray < tissue_gray_threshold
    if not tissue_mask.any():
        tissue_mask = np.ones_like(gray, dtype=bool)

    for ch in range(3):
        values = lab[:, :, ch][tissue_mask]
        src_mean = float(values.mean())
        src_std = float(values.std())
        if src_std < 1e-6:
            continue
        lab[:, :, ch] = (lab[:, :, ch] - src_mean) * (ref_stds[ch] / src_std) + ref_means[ch]

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_reinhard_ref_json(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Read a reference from JSON with keys L_mean, L_std, A_mean, A_std, B_mean, B_std."""
    with Path(path).open(encoding="utf-8") as f:
        d = json.load(f)
    means = (float(d["L_mean"]), float(d["A_mean"]), float(d["B_mean"]))
    stds = (float(d["L_std"]), float(d["A_std"]), float(d["B_std"]))
    return means, stds


def apply_stain_norm(
    image_uint8: np.ndarray,
    method: str,
    ref_means: tuple[float, float, float] | None = None,
    ref_stds: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Apply the named stain normalization to a uint8 BGR image."""
    if method == "none":
        return image_uint8
    if method == "reinhard":
        return reinhard_normalize(image_uint8, ref_means=ref_means, ref_stds=ref_stds)
    raise ValueError(f"Unknown stain normalization {method!r}; expected one of {STAIN_NORM_METHODS}")


def add_stain_norm_args(parser: argparse.ArgumentParser, default: str = DEFAULT_STAIN_NORM) -> None:
    """Register --stain-norm and --reinhard-ref-json on a CLI parser."""
    parser.add_argument(
        "--stain-norm",
        choices=STAIN_NORM_METHODS,
        default=default,
        help=f"Stain normalization applied before resizing (default: {default})",
    )
    parser.add_argument(
        "--reinhard-ref-json",
        type=Path,
        default=None,
        help="JSON with L_mean,L_std,A_mean,A_std,B_mean,B_std for the Reinhard target "
        "(default: rat training reference)",
    )


def check_stain_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject a Reinhard reference passed without --stain-norm reinhard."""
    if getattr(args, "reinhard_ref_json", None) is not None and args.stain_norm != "reinhard":
        parser.error("--reinhard-ref-json requires --stain-norm reinhard")


def reference_from_args(args: argparse.Namespace):
    """Return (ref_means, ref_stds) from parsed args; (None, None) means the default reference."""
    if getattr(args, "reinhard_ref_json", None) is None:
        return None, None
    return load_reinhard_ref_json(args.reinhard_ref_json)
