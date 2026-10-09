"""
Preprocess human histology images (.jpg/.jpeg/.tif/.tiff) for the TOAD models.

Default input: ``Allen_Lab_SafO-FastGreen_images/TIFF_second_batch`` — collects images
from that folder and from each immediate subfolder (e.g. ``*_bone_TIFF/*.tif``),
excluding ``Processed_Images``.

Default output: ``TOAD/outputs/Processed_Images`` (created if missing).

Per image, in this order:
  1. load and convert to uint8 BGR (16-bit / grayscale / RGBA handled)
  2. optional rescale to a target pixel size (``--scale`` or ``--source-mpp/--target-mpp``)
  3. optional stain normalization (``--stain-norm none|reinhard``, default none)
  4. optional Gray World white balance / CLAHE (whole image, before tiling)
  5. tile export (default) or a single resize to ``--width`` x ``--height``

Optional label masks (``--mask-dir``) get the same rescale (nearest neighbour) and
the same tile coordinates, are checked against docs/DATA_CONTRACT.md, and are written
as PNG to ``<output-dir>/masks`` with the same stem as their image tile.

Very large TIFFs: ``OPENCV_IO_MAX_IMAGE_PIXELS`` is raised by default (see top of file);
set it in the environment before running if you need a different cap.

CLI summary: run ``python src/preprocess_human_data.py -h``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

# Large Leica stitches exceed OpenCV's default read cap (~2^30 pixels).
# Set before ``import cv2`` so ``imread`` allows huge TIFFs (override via env if needed).
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(2**40 - 1))

import cv2
import numpy as np

from labels import prepare_mask
from resolution import read_resolution
from stain_norm import (
    DEFAULT_STAIN_NORM,
    STAIN_NORM_METHODS,
    add_stain_norm_args,
    apply_stain_norm,
    check_stain_args,
    load_reinhard_ref_json,
)

# Subfolder under TIFF_second_batch where outputs go; excluded from input discovery
PROCESSED_SUBDIR_NAME = "Processed_Images"
MASK_SUBDIR_NAME = "masks"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".tif", ".tiff"}

MAPPING_FIELDS = [
    "output_filename",
    "original_path",
    "mask_filename",
    "y",
    "x",
    "tile_h",
    "tile_w",
    "scale_x",
    "scale_y",
    "scaled_h",
    "scaled_w",
    "stain_norm",
]


class ScaleError(ValueError):
    """The rescale factor for an image could not be determined."""


def gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """Apply Gray World white balance in BGR space."""
    img = img_bgr.astype(np.float32)
    channel_means = img.reshape(-1, 3).mean(axis=0)
    mean_gray = channel_means.mean()

    scales = np.where(channel_means > 1e-6, mean_gray / channel_means, 1.0).astype(np.float32)
    img *= scales

    return np.clip(img, 0, 255).astype(np.uint8)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert any bit-depth image to 3-channel uint8 BGR (Colab-compatible)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        img_f = image.astype(np.float32)
        mx = img_f.max()
        if mx > 0:
            img_f /= mx
        image = np.clip(img_f * 255, 0, 255).astype(np.uint8)
    return image


def load_image_bgr(path: Path) -> np.ndarray | None:
    """Load image with IMREAD_UNCHANGED, then normalize to uint8 BGR."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return normalize_to_uint8(raw)


def is_background_patch(patch_uint8: np.ndarray, threshold: float) -> bool:
    """True if fraction of near-white pixels >= threshold (Colab-compatible)."""
    gray = cv2.cvtColor(patch_uint8, cv2.COLOR_BGR2GRAY) if patch_uint8.ndim == 3 else patch_uint8
    return float(np.mean(gray > 220)) >= threshold


def tile_image(
    image_uint8: np.ndarray,
    tile_h: int,
    tile_w: int,
    overlap: int = 0,
    bg_threshold: float | None = 0.9,
    bg_reference: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int]]:
    """
    Tile into (tile_h x tile_w) patches. Returns (patches_uint8, (y,x) coords, grid_shape).
    If bg_threshold is None, do not skip background patches.

    ``bg_reference`` (same height/width as the image) is used for the background test
    instead of the image itself. Passing the pre-color-normalization image keeps the
    tile set identical across --stain-norm settings, so runs can be compared tile by tile.
    """
    if bg_reference is None:
        bg_reference = image_uint8
    elif bg_reference.shape[:2] != image_uint8.shape[:2]:
        raise ValueError("bg_reference must match the image's height and width")
    h, w = image_uint8.shape[:2]
    step_h = max(1, tile_h - overlap)
    step_w = max(1, tile_w - overlap)

    if h < tile_h or w < tile_w:
        return [], [], (0, 0)

    y_starts = list(range(0, h - tile_h + 1, step_h))
    x_starts = list(range(0, w - tile_w + 1, step_w))
    if y_starts[-1] + tile_h < h:
        y_starts.append(h - tile_h)
    if x_starts[-1] + tile_w < w:
        x_starts.append(w - tile_w)

    patches: list[np.ndarray] = []
    coords: list[tuple[int, int]] = []
    skipped = 0
    for y in y_starts:
        for x in x_starts:
            p = image_uint8[y : y + tile_h, x : x + tile_w]
            ref = bg_reference[y : y + tile_h, x : x + tile_w]
            if bg_threshold is not None and is_background_patch(ref, bg_threshold):
                skipped += 1
                continue
            patches.append(p.copy())
            coords.append((y, x))

    grid = (len(y_starts), len(x_starts))
    print(f"  Tiles: {len(patches)} kept, {skipped} bg-skipped (grid {grid[0]}x{grid[1]})")
    return patches, coords, grid


def normalize_color_stain_agnostic(
    img_bgr: np.ndarray,
    apply_white_balance: bool = False,
    apply_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
) -> np.ndarray:
    out = img_bgr

    if apply_white_balance:
        out = gray_world_white_balance(out)

    if apply_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_tile_grid), int(clahe_tile_grid)),
        )
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return out


def resize_image(img_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)


def rescale(image: np.ndarray, fx: float, fy: float, is_mask: bool = False) -> np.ndarray:
    """Rescale by (fx, fy). Masks use nearest neighbour so class IDs are never blended."""
    if fx == 1.0 and fy == 1.0:
        return image
    h, w = image.shape[:2]
    size = (max(1, round(w * fx)), max(1, round(h * fy)))
    if is_mask:
        interp = cv2.INTER_NEAREST
    elif fx <= 1.0 and fy <= 1.0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interp)


def resolve_scale(
    src_path: Path,
    scale: float,
    source_mpp: float | str | None,
    target_mpp: float | None,
) -> tuple[float, float]:
    """
    Return (fx, fy) for one image.

    ``--scale`` gives one factor for every image. ``--source-mpp/--target-mpp`` give
    fx = source/target per axis; ``source_mpp="auto"`` reads it from the file's metadata
    and refuses missing or default-looking values instead of guessing.
    """
    if target_mpp is None:
        return scale, scale
    if source_mpp == "auto":
        res = read_resolution(src_path)
        if res.mpp_x is None or res.mpp_y is None:
            raise ScaleError(f"no physical pixel size in metadata (source: {res.source})")
        if res.suspect:
            raise ScaleError(
                f"metadata pixel size {res.mpp_x:.4f} um/px looks like a software default "
                f"(source: {res.source}); pass --source-mpp with the scanner's value"
            )
        return res.mpp_x / target_mpp, res.mpp_y / target_mpp
    factor = float(source_mpp) / target_mpp
    return factor, factor


def iter_image_paths(input_dir: Path) -> list[Path]:
    """
    Collect images directly under ``input_dir`` and under each immediate subdirectory,
    except ``Processed_Images`` (so outputs are never treated as inputs).
    """
    found: set[Path] = set()
    if not input_dir.is_dir():
        return []

    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            found.add(p.resolve())

    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.lower() == PROCESSED_SUBDIR_NAME.lower():
            continue
        try:
            for p in sorted(sub.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    found.add(p.resolve())
        except OSError:
            continue

    return sorted(found)


def index_masks(mask_dir: Path) -> dict[str, Path]:
    """Map mask file stem -> path for every mask under mask_dir (recursive)."""
    index: dict[str, Path] = {}
    for p in sorted(mask_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in MASK_EXTENSIONS:
            continue
        if p.stem in index:
            raise ValueError(f"Two masks share the stem {p.stem!r}: {index[p.stem]} and {p}")
        index[p.stem] = p
    return index


def load_mask(path: Path) -> np.ndarray:
    """
    Load a class-ID mask as uint8.

    Accepts single-channel files, palette PNGs (palette indices are the class IDs),
    and RGB/RGBA files whose color channels are identical (alpha is ignored).
    """
    if path.suffix.lower() == ".png":
        from PIL import Image

        with Image.open(path) as im:
            # Palette ("P") images: np.asarray returns the indices, not the colors.
            raw = np.asarray(im)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"could not read mask {path.name}")
    if raw.ndim == 3:
        color = raw[:, :, :3]
        if not all(np.array_equal(color[:, :, 0], color[:, :, c]) for c in range(1, color.shape[2])):
            raise ValueError(f"mask {path.name} has differing color channels; expected class IDs")
        raw = color[:, :, 0]
    if raw.dtype == bool:
        raw = raw.astype(np.uint8)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"mask {path.name} has non-integer dtype {raw.dtype}")
    if raw.size and (int(raw.min()) < 0 or int(raw.max()) > 255):
        raise ValueError(f"mask {path.name} has values outside 0..255 (max {int(raw.max())})")
    return raw.astype(np.uint8)


def _normalize_ext(ext: str) -> str:
    ext = ext.strip()
    if not ext:
        return ".tif"
    return ext if ext.startswith(".") else f".{ext}"


def effective_output_extension(src_path: Path, output_ext: str) -> str:
    """
    Default behavior: use the source file's suffix (.tif, .jpeg, …).

    ``output_ext`` may be ``auto`` / ``same`` / empty for that behavior, or an
    explicit extension like ``.png`` to force a format.
    """
    policy = output_ext.strip().lower()
    if policy in ("", "auto", "same"):
        suf = src_path.suffix.lower()
        return _normalize_ext(suf) if suf else ".tif"
    return _normalize_ext(output_ext)


def processed_output_filename(src_path: Path, output_ext: str) -> str:
    ext = effective_output_extension(src_path, output_ext)
    return f"{src_path.stem}_processed{ext}"


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    apply_white_balance: bool,
    apply_clahe: bool,
    clahe_clip_limit: float,
    clahe_tile_grid: int,
    *,
    keep_original_names: bool = False,
    output_ext: str = "auto",
    write_mapping: bool = True,
    mapping_filename: str = "rename_mapping.csv",
    stain_norm: str = DEFAULT_STAIN_NORM,
    reinhard_ref_json: Path | None = None,
    export_tiles: bool = True,
    tile_overlap: int = 0,
    tile_skip_bg: bool = True,
    tile_bg_threshold: float = 0.9,
    scale: float = 1.0,
    source_mpp: float | str | None = None,
    target_mpp: float | None = None,
    mask_dir: Path | None = None,
    species: str = "human",
) -> dict[str, int]:
    """Run the pipeline over input_dir. Returns counts for logging and tests."""
    if stain_norm not in STAIN_NORM_METHODS:
        raise ValueError(f"stain_norm must be one of {STAIN_NORM_METHODS}, got {stain_norm!r}")
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_means: tuple[float, float, float] | None = None
    ref_stds: tuple[float, float, float] | None = None
    if reinhard_ref_json is not None:
        ref_means, ref_stds = load_reinhard_ref_json(reinhard_ref_json)

    image_paths = iter_image_paths(input_dir)
    if not image_paths:
        raise ValueError(f"No supported images (.jpg/.jpeg/.tif/.tiff) found in: {input_dir}")

    mask_index: dict[str, Path] = {}
    mask_out_dir = output_dir / MASK_SUBDIR_NAME
    if mask_dir is not None:
        mask_index = index_masks(mask_dir)
        mask_out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Found {len(image_paths)} image files under {input_dir} "
        f"(root + subfolders except '{PROCESSED_SUBDIR_NAME}')"
    )
    print(f"Writing to: {output_dir}")
    if export_tiles:
        print(f"Mode: export tiles {height}x{width} overlap={tile_overlap} skip_bg={tile_skip_bg}")
    else:
        print(f"Mode: single output per image (--no-export-tiles), resize to {width}x{height}")
    if target_mpp is not None:
        print(f"Rescale: source {source_mpp} um/px -> target {target_mpp} um/px")
    elif scale != 1.0:
        print(f"Rescale: fixed factor {scale}")
    print(f"Stain normalization: {stain_norm}")
    if stain_norm == "reinhard":
        src = reinhard_ref_json if reinhard_ref_json else "default (rat training, from Colab notebook)"
        print(f"  Reference stats: {src}")
    if apply_white_balance or apply_clahe:
        print("Extra color: white_balance=%s, clahe=%s" % (apply_white_balance, apply_clahe))
    if mask_dir is not None:
        print(f"Masks: {len(mask_index)} found under {mask_dir}; species={species}")
    if not keep_original_names and not export_tiles:
        pol = output_ext.strip().lower()
        if pol in ("", "auto", "same"):
            print("Output naming: <stem>_processed.<ext same as input>")
        else:
            print(f"Output naming: <stem>_processed{_normalize_ext(output_ext)}")

    counts = {"images_done": 0, "images_skipped": 0, "outputs_written": 0, "write_failures": 0}
    mapping_rows: list[dict] = []

    def skip(reason: str) -> None:
        counts["images_skipped"] += 1
        print(f"[skip] {reason}")

    for src_path in image_paths:
        img = load_image_bgr(src_path)
        if img is None:
            skip(f"Could not read image: {src_path.name}")
            continue

        mask = None
        if mask_dir is not None:
            mask_path = mask_index.get(src_path.stem)
            if mask_path is None:
                skip(f"No mask with stem {src_path.stem!r} in {mask_dir}")
                continue
            try:
                mask = prepare_mask(load_mask(mask_path), species)
            except ValueError as exc:
                skip(f"{src_path.name}: {exc}")
                continue
            if mask.shape[:2] != img.shape[:2]:
                skip(f"{src_path.name}: mask size {mask.shape[:2]} != image size {img.shape[:2]}")
                continue

        try:
            fx, fy = resolve_scale(src_path, scale, source_mpp, target_mpp)
        except ScaleError as exc:
            skip(f"{src_path.name}: {exc}")
            continue
        img = rescale(img, fx, fy)
        if mask is not None:
            mask = rescale(mask, fx, fy, is_mask=True)
        scaled_h, scaled_w = img.shape[:2]
        # Background tiles are chosen on the image before any color change (see tile_image).
        selection_img = img

        img = apply_stain_norm(img, stain_norm, ref_means=ref_means, ref_stds=ref_stds)

        base_row = {
            "original_path": str(src_path.resolve()),
            "scale_x": f"{fx:.6g}",
            "scale_y": f"{fy:.6g}",
            "scaled_h": scaled_h,
            "scaled_w": scaled_w,
            "stain_norm": stain_norm,
        }

        if export_tiles:
            # Color ops run once on the whole image so every tile gets the same transform.
            img = normalize_color_stain_agnostic(
                img,
                apply_white_balance=apply_white_balance,
                apply_clahe=apply_clahe,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_grid=clahe_tile_grid,
            )
            ext_resolved = effective_output_extension(src_path, output_ext)
            bg_thr = tile_bg_threshold if tile_skip_bg else None
            patches, coords, _ = tile_image(
                img, height, width, tile_overlap, bg_thr, bg_reference=selection_img
            )
            if not patches:
                skip(f"No tiles for {src_path.name} (smaller than tile after rescale, or all background)")
                continue
            for patch, (y, x) in zip(patches, coords):
                out_name = f"{src_path.stem}_tile_{y:05d}_{x:05d}{ext_resolved}"
                if not cv2.imwrite(str(output_dir / out_name), patch):
                    counts["write_failures"] += 1
                    print(f"[error] Could not write: {out_name}")
                    continue
                mask_name = ""
                if mask is not None:
                    mask_name = f"{Path(out_name).stem}.png"
                    tile_mask = mask[y : y + height, x : x + width]
                    if not cv2.imwrite(str(mask_out_dir / mask_name), tile_mask):
                        counts["write_failures"] += 1
                        print(f"[error] Could not write mask: {mask_name}")
                        continue
                counts["outputs_written"] += 1
                mapping_rows.append(
                    {**base_row, "output_filename": out_name, "mask_filename": mask_name,
                     "y": y, "x": x, "tile_h": height, "tile_w": width}
                )
            counts["images_done"] += 1
            continue

        img = resize_image(img, width=width, height=height)
        img = normalize_color_stain_agnostic(
            img,
            apply_white_balance=apply_white_balance,
            apply_clahe=apply_clahe,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid=clahe_tile_grid,
        )

        ext_resolved = effective_output_extension(src_path, output_ext)
        if keep_original_names:
            out_name = f"{src_path.stem}{ext_resolved}"
        else:
            out_name = processed_output_filename(src_path, output_ext)

        if not cv2.imwrite(str(output_dir / out_name), img):
            counts["write_failures"] += 1
            print(f"[error] Could not write image: {out_name}")
            continue

        mask_name = ""
        if mask is not None:
            mask_name = f"{Path(out_name).stem}.png"
            resized_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            if not cv2.imwrite(str(mask_out_dir / mask_name), resized_mask):
                counts["write_failures"] += 1
                print(f"[error] Could not write mask: {mask_name}")
                continue

        counts["outputs_written"] += 1
        counts["images_done"] += 1
        mapping_rows.append(
            {**base_row, "output_filename": out_name, "mask_filename": mask_name,
             "y": "", "x": "", "tile_h": "", "tile_w": ""}
        )

    if write_mapping and mapping_rows:
        map_path = output_dir / mapping_filename
        with map_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=MAPPING_FIELDS)
            writer.writeheader()
            writer.writerows(mapping_rows)
        print(f"Wrote mapping: {map_path}")

    print(
        "Done. images processed={images_done}, images skipped={images_skipped}, "
        "outputs written={outputs_written}, write failures={write_failures}".format(**counts)
    )
    return counts


def _source_mpp_arg(text: str) -> float | str:
    if text.strip().lower() == "auto":
        return "auto"
    try:
        return _positive_float(text)
    except argparse.ArgumentTypeError:
        raise argparse.ArgumentTypeError("--source-mpp must be a positive number or 'auto'") from None


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a positive, finite number")
    return value


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "Allen_Lab_SafO-FastGreen_images" / "TIFF_second_batch"
    default_output = repo_root / "outputs" / PROCESSED_SUBDIR_NAME

    parser = argparse.ArgumentParser(
        description="Preprocess human images: uint8 load, optional rescale, optional stain "
        "normalization, tile export (default) or whole-image resize."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=(
            "Root folder: images here and in each immediate subfolder are used "
            f"(excludes '{PROCESSED_SUBDIR_NAME}'). Default: {default_input}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Output folder (created if missing). Default: {default_output}",
    )
    parser.add_argument("--width", type=int, default=256, help="Output width (resize mode) or tile width")
    parser.add_argument("--height", type=int, default=192, help="Output height (resize mode) or tile height")

    stain = parser.add_argument_group("stain normalization")
    add_stain_norm_args(stain)
    stain.add_argument(
        "--no-reinhard",
        action="store_true",
        help="Deprecated: same as --stain-norm none (now the default)",
    )

    scaling = parser.add_argument_group(
        "rescaling (tile mode only; applied before stain normalization and tiling)"
    )
    scaling.add_argument(
        "--scale",
        type=_positive_float,
        default=1.0,
        help="Fixed resize factor for every image, e.g. 0.25 (default 1.0 = native resolution)",
    )
    scaling.add_argument(
        "--source-mpp",
        type=_source_mpp_arg,
        default=None,
        help="Microns per pixel of the input images, or 'auto' to read it from TIFF metadata",
    )
    scaling.add_argument(
        "--target-mpp",
        type=_positive_float,
        default=None,
        help="Microns per pixel to resample to (the rat training scale; see src/inspect_resolution.py)",
    )

    labels = parser.add_argument_group("label masks")
    labels.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Folder of class-ID masks (.png/.tif) named with the same stem as each image. "
        f"Written to <output-dir>/{MASK_SUBDIR_NAME}",
    )
    labels.add_argument(
        "--species",
        choices=("human", "rat"),
        default="human",
        help="Class rules for masks: human masks may not contain growth plate (3)",
    )

    parser.add_argument(
        "--no-export-tiles",
        action="store_true",
        help="Write one resized image per input instead of tiling (tiling is on by default)",
    )
    parser.add_argument("--tile-overlap", type=int, default=0, help="Pixel overlap between tiles")
    parser.add_argument(
        "--no-tile-skip-bg",
        action="store_true",
        help="Do not skip mostly-white (background) tiles",
    )
    parser.add_argument(
        "--tile-bg-threshold",
        type=float,
        default=0.9,
        help="Skip tile if this fraction of pixels are near-white (default 0.9)",
    )

    parser.add_argument(
        "--white-balance",
        action="store_true",
        help="Apply Gray World white balance (whole image, after stain normalization)",
    )
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Apply CLAHE on LAB luminance",
    )
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--clahe-tile-grid", type=int, default=8, help="CLAHE tile grid size")

    parser.add_argument(
        "--keep-original-names",
        action="store_true",
        help="Use {stem}<ext> without _processed; extension follows --output-ext / auto (resize mode only)",
    )
    parser.add_argument(
        "--output-ext",
        type=str,
        default="auto",
        help="Output extension: default 'auto' = same as each input (.tif, .jpeg, …); set e.g. .png to force",
    )
    parser.add_argument("--no-mapping", action="store_true", help="Do not write mapping CSV")
    parser.add_argument("--mapping-filename", type=str, default="rename_mapping.csv")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.no_reinhard:
        if args.stain_norm == "reinhard":
            parser.error("--no-reinhard conflicts with --stain-norm reinhard")
        print("Note: --no-reinhard is deprecated; stain normalization is off unless --stain-norm reinhard.")
        args.stain_norm = "none"

    if (args.source_mpp is None) != (args.target_mpp is None):
        parser.error("--source-mpp and --target-mpp must be given together")
    if args.target_mpp is not None and args.scale != 1.0:
        parser.error("use either --scale or --source-mpp/--target-mpp, not both")
    rescaling = args.target_mpp is not None or args.scale != 1.0
    if rescaling and args.no_export_tiles:
        parser.error("rescaling applies to tile export only; resize mode already squashes each image to WxH")
    check_stain_args(parser, args)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    if args.mask_dir is not None and not args.mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {args.mask_dir}")

    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        apply_white_balance=args.white_balance,
        apply_clahe=args.clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_grid=args.clahe_tile_grid,
        keep_original_names=args.keep_original_names,
        output_ext=args.output_ext,
        write_mapping=not args.no_mapping,
        mapping_filename=args.mapping_filename,
        stain_norm=args.stain_norm,
        reinhard_ref_json=args.reinhard_ref_json,
        export_tiles=not args.no_export_tiles,
        tile_overlap=args.tile_overlap,
        tile_skip_bg=not args.no_tile_skip_bg,
        tile_bg_threshold=args.tile_bg_threshold,
        scale=args.scale,
        source_mpp=args.source_mpp,
        target_mpp=args.target_mpp,
        mask_dir=args.mask_dir,
        species=args.species,
    )


if __name__ == "__main__":
    main()
