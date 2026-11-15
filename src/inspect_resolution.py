"""Report microns-per-pixel for every image in one or more folders.

Use this to pick ``--target-mpp`` for ``preprocess_human_data.py``.

Rat (SEKO) training images were stored at 256x192, so their own metadata does not
describe the pixel size the model learned on. Point this at the ORIGINAL rat scans
and pass ``--resize-to 256x192`` to get the effective pixel size after that resize:

    python src/inspect_resolution.py /path/to/rat_originals --resize-to 256x192
    python src/inspect_resolution.py /path/to/human_tiffs --csv outputs/human_mpp.csv

Then preprocess human data at the rat scale:

    python src/preprocess_human_data.py --source-mpp auto --target-mpp <rat median>
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from resolution import Resolution, read_resolution

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff"}


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected WIDTHxHEIGHT, got {text!r}") from None


def iter_images(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def effective_mpp(res: Resolution, resize_to: tuple[int, int] | None) -> tuple[float | None, float | None]:
    """Pixel size after resizing the full image to resize_to (width, height)."""
    if res.mpp_x is None or res.mpp_y is None:
        return None, None
    if resize_to is None or not res.width or not res.height:
        return res.mpp_x, res.mpp_y
    out_w, out_h = resize_to
    return res.mpp_x * res.width / out_w, res.mpp_y * res.height / out_h


def main() -> None:
    parser = argparse.ArgumentParser(description="Report microns-per-pixel from image metadata.")
    parser.add_argument("folders", nargs="+", type=Path, help="Folders of .tif/.tiff/.jpg images")
    parser.add_argument("--recursive", action="store_true", help="Include all subfolders")
    parser.add_argument(
        "--resize-to",
        type=_parse_size,
        default=None,
        metavar="WxH",
        help="Also report the pixel size after resizing whole images to WxH (e.g. 256x192)",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Write one row per image to this CSV")
    args = parser.parse_args()

    rows = []
    for folder in args.folders:
        if not folder.is_dir():
            parser.error(f"Not a folder: {folder}")
        paths = iter_images(folder, args.recursive)
        if not paths:
            print(f"{folder}: no images found")
            continue

        usable_x, usable_y, suspect = [], [], 0
        for path in paths:
            res = read_resolution(path)
            eff_x, eff_y = effective_mpp(res, args.resize_to)
            rows.append(
                {
                    "folder": str(folder),
                    "file": str(path),
                    "width": res.width,
                    "height": res.height,
                    "mpp_x": res.mpp_x,
                    "mpp_y": res.mpp_y,
                    "source": res.source,
                    "suspect": res.suspect,
                    "effective_mpp_x": eff_x,
                    "effective_mpp_y": eff_y,
                }
            )
            if res.usable:
                usable_x.append(eff_x)
                usable_y.append(eff_y)
            else:
                suspect += 1

        print(f"\n{folder}")
        print(f"  images: {len(paths)}  usable resolution: {len(usable_x)}  missing/suspect: {suspect}")
        if usable_x:
            label = f"after resize to {args.resize_to[0]}x{args.resize_to[1]}" if args.resize_to else "native"
            print(
                f"  median um/px ({label}): x={statistics.median(usable_x):.4f} "
                f"y={statistics.median(usable_y):.4f}  "
                f"range x=[{min(usable_x):.4f}, {max(usable_x):.4f}]"
            )
        if suspect:
            print("  Some files have no physical size or a default-looking DPI.")
            print("  Get the scanner's microns-per-pixel from the lab instead of trusting these.")

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
