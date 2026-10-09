import cv2
import numpy as np
import pytest
import tifffile

from inspect_resolution import effective_mpp
from resolution import read_resolution

DATA = np.zeros((40, 60, 3), dtype=np.uint8)


def test_tiff_tags_in_centimeters(tmp_path):
    path = tmp_path / "scan.tif"
    tifffile.imwrite(path, DATA, resolution=(20000, 20000), resolutionunit="CENTIMETER")
    res = read_resolution(path)
    assert res.source == "tiff-tags"
    assert res.mpp_x == pytest.approx(0.5)
    assert res.usable
    assert (res.width, res.height) == (60, 40)


def test_default_dpi_is_suspect(tmp_path):
    path = tmp_path / "export.tif"
    tifffile.imwrite(path, DATA, resolution=(72, 72), resolutionunit="INCH")
    res = read_resolution(path)
    assert res.mpp_x == pytest.approx(25400 / 72)
    assert res.suspect and not res.usable


def test_aperio_description(tmp_path):
    path = tmp_path / "slide.tif"
    tifffile.imwrite(path, DATA, description="Aperio Image Library v12 |AppMag = 40|MPP = 0.2520", metadata=None)
    res = read_resolution(path)
    assert res.source == "aperio"
    assert res.mpp_x == pytest.approx(0.252)


def test_ome_physical_size(tmp_path):
    path = tmp_path / "slide.ome.tif"
    tifffile.imwrite(
        path,
        DATA,
        ome=True,
        photometric="rgb",
        metadata={"axes": "YXS", "PhysicalSizeX": 0.3, "PhysicalSizeY": 0.4},
    )
    res = read_resolution(path)
    assert res.source == "ome"
    assert (res.mpp_x, res.mpp_y) == (pytest.approx(0.3), pytest.approx(0.4))


def test_jpeg_is_always_suspect(tmp_path):
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), DATA)
    assert not read_resolution(path).usable


def test_effective_mpp_after_fixed_resize(tmp_path):
    path = tmp_path / "rat.tif"
    # 1024x768 original at 1 um/px squashed to 256x192 -> 4 um/px per axis
    tifffile.imwrite(
        path, np.zeros((768, 1024, 3), np.uint8), resolution=(10000, 10000), resolutionunit="CENTIMETER"
    )
    res = read_resolution(path)
    assert effective_mpp(res, (256, 192)) == (pytest.approx(4.0), pytest.approx(4.0))


def test_default_dpi_stored_per_centimeter_is_suspect(tmp_path):
    # 300 dpi written as px/cm (118.11) reads back as 299.999 dpi
    path = tmp_path / "cm300.tif"
    tifffile.imwrite(path, DATA, resolution=(300 / 2.54, 300 / 2.54), resolutionunit="CENTIMETER")
    assert read_resolution(path).suspect
