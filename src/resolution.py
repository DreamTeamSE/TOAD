"""Read physical pixel size (microns per pixel) from image metadata.

Sources, in order of trust:
  1. OME-XML ``PhysicalSizeX/Y`` (OME-TIFF, Leica SCN)
  2. Aperio ``MPP = ...`` in the TIFF ImageDescription
  3. TIFF ``XResolution/YResolution`` + ``ResolutionUnit``
  4. JPEG/JFIF DPI (always marked suspect)

A value is marked ``suspect`` when it looks like a software default rather than a
calibrated scan (common screen/print DPIs, or JPEG DPI). Do not rescale from a
suspect value; pass the scanner's real microns-per-pixel instead.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

MICRONS_PER_INCH = 25400.0
MICRONS_PER_CM = 10000.0

# DPI values written by default by editors and capture software, not by scanners.
DEFAULT_LOOKING_DPI = {72.0, 96.0, 150.0, 200.0, 300.0, 600.0}

_OME_UNIT_TO_MICRONS = {
    "µm": 1.0,  # U+00B5 micro sign
    "μm": 1.0,  # U+03BC Greek mu
    "um": 1.0,
    "micron": 1.0,
    "nm": 1e-3,
    "mm": 1e3,
    "cm": 1e4,
}

_APERIO_MPP = re.compile(r"\bMPP\s*=\s*([0-9]*\.?[0-9]+)")


@dataclass(frozen=True)
class Resolution:
    path: Path
    width: int | None
    height: int | None
    mpp_x: float | None
    mpp_y: float | None
    source: str
    suspect: bool

    @property
    def usable(self) -> bool:
        return self.mpp_x is not None and self.mpp_y is not None and not self.suspect


def _looks_like_default_dpi(dpi: float | None) -> bool:
    # Tolerance matters: 300 dpi saved as px/cm comes back as 299.999.
    return dpi is not None and any(abs(dpi - d) < 0.5 for d in DEFAULT_LOOKING_DPI)


def _ome_resolution(xml_text: str) -> tuple[float, float] | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    for el in root.iter():
        if not el.tag.endswith("Pixels"):
            continue
        px, py = el.get("PhysicalSizeX"), el.get("PhysicalSizeY")
        if px is None or py is None:
            continue
        ux = _OME_UNIT_TO_MICRONS.get(el.get("PhysicalSizeXUnit", "µm"))
        uy = _OME_UNIT_TO_MICRONS.get(el.get("PhysicalSizeYUnit", "µm"))
        if ux is None or uy is None:
            continue
        return float(px) * ux, float(py) * uy
    return None


def _rational(value) -> float | None:
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        return float(num) / float(den) if den else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tiff_resolution(path: Path) -> Resolution:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        width, height = int(page.imagewidth), int(page.imagelength)

        if tif.is_ome and tif.ome_metadata:
            ome = _ome_resolution(tif.ome_metadata)
            if ome is not None:
                return Resolution(path, width, height, ome[0], ome[1], "ome", False)

        desc = page.description or ""
        match = _APERIO_MPP.search(desc)
        if match:
            mpp = float(match.group(1))
            return Resolution(path, width, height, mpp, mpp, "aperio", False)

        tags = page.tags
        xr = tags.get("XResolution")
        yr = tags.get("YResolution")
        unit_tag = tags.get("ResolutionUnit")
        if xr is None or yr is None:
            return Resolution(path, width, height, None, None, "none", True)

        x_per_unit = _rational(xr.value)
        y_per_unit = _rational(yr.value)
        unit = int(unit_tag.value) if unit_tag is not None else 2  # TIFF default is inch
        if unit == 2:
            unit_microns, dpi_x, dpi_y = MICRONS_PER_INCH, x_per_unit, y_per_unit
        elif unit == 3:
            unit_microns = MICRONS_PER_CM
            dpi_x = x_per_unit * 2.54 if x_per_unit else None
            dpi_y = y_per_unit * 2.54 if y_per_unit else None
        else:
            return Resolution(path, width, height, None, None, "tiff-unitless", True)

        if not x_per_unit or not y_per_unit:
            return Resolution(path, width, height, None, None, "tiff-zero", True)

        mpp_x = unit_microns / x_per_unit
        mpp_y = unit_microns / y_per_unit
        suspect = any(_looks_like_default_dpi(d) for d in (dpi_x, dpi_y))
        return Resolution(path, width, height, mpp_x, mpp_y, "tiff-tags", suspect)


def _jpeg_resolution(path: Path) -> Resolution:
    from PIL import Image

    with Image.open(path) as im:
        width, height = im.size
        dpi = im.info.get("dpi")
    if not dpi or not dpi[0] or not dpi[1]:
        return Resolution(path, width, height, None, None, "none", True)
    return Resolution(
        path,
        width,
        height,
        MICRONS_PER_INCH / float(dpi[0]),
        MICRONS_PER_INCH / float(dpi[1]),
        "jpeg-dpi",
        True,
    )


def read_resolution(path: Path) -> Resolution:
    """Best-effort physical resolution for one image file. Never reads pixel data."""
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in (".tif", ".tiff"):
            return _tiff_resolution(path)
        if suffix in (".jpg", ".jpeg"):
            return _jpeg_resolution(path)
    except Exception as exc:  # unreadable metadata should not stop a batch
        return Resolution(path, None, None, None, None, f"error: {exc}", True)
    return Resolution(path, None, None, None, None, "unsupported", True)
