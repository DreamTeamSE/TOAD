import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

import preprocess_human_data as pp
from labels import CARTILAGE, GROWTH_PLATE, RAW_OSTEOPHYTE

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "preprocess_human_data.py"
TILE_W, TILE_H = 64, 48
BLOCK = 64  # label block size at native resolution; even, so a 0.5x area resize never mixes blocks

# One distinct color per raw label (BGR); 255-white is glass.
COLORS = {0: (250, 250, 250), 1: (40, 160, 200), 2: (180, 60, 60), 4: (60, 60, 160), RAW_OSTEOPHYTE: (120, 40, 120)}


def make_sample(tmp_path, name="slide", mpp=0.5, labels_grid=None):
    """Write a TIFF + raw mask whose pixel colors encode the label at every pixel."""
    if labels_grid is None:
        labels_grid = np.array(
            [
                [0, 0, 1, 1, 2, 2],
                [0, 1, 1, 2, 2, RAW_OSTEOPHYTE],
                [4, 4, 1, 1, 2, 0],
                [4, 4, 4, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
    mask = np.kron(labels_grid, np.ones((BLOCK, BLOCK), dtype=np.uint8))
    img = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for label, color in COLORS.items():
        img[mask == label] = color
    img_dir, mask_dir = tmp_path / "images", tmp_path / "masks_raw"
    img_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    tifffile.imwrite(
        img_dir / f"{name}.tif", img[:, :, ::-1], photometric="rgb",
        resolution=(10000 / mpp, 10000 / mpp), resolutionunit="CENTIMETER",
    )
    cv2.imwrite(str(mask_dir / f"{name}.png"), mask)
    return img_dir, mask_dir, img, mask


def run(img_dir, out_dir, **kw):
    defaults = dict(
        width=TILE_W, height=TILE_H, apply_white_balance=False, apply_clahe=False,
        clahe_clip_limit=2.0, clahe_tile_grid=8,
    )
    defaults.update(kw)
    return pp.preprocess_dataset(img_dir, out_dir, **defaults)


def read_mapping(out_dir):
    with (out_dir / "rename_mapping.csv").open() as f:
        return list(csv.DictReader(f))


def test_tiles_and_masks_stay_aligned_after_rescale(tmp_path):
    img_dir, mask_dir, _, _ = make_sample(tmp_path, mpp=0.5)
    out = tmp_path / "out"
    counts = run(img_dir, out, mask_dir=mask_dir, source_mpp="auto", target_mpp=1.0)
    assert counts["images_done"] == 1 and counts["images_skipped"] == 0

    rows = read_mapping(out)
    assert rows and all(r["scale_x"] == "0.5" for r in rows)
    expected_color = {**{k: v for k, v in COLORS.items() if k != RAW_OSTEOPHYTE}}
    for row in rows:
        tile = cv2.imread(str(out / row["output_filename"]), cv2.IMREAD_UNCHANGED)
        tmask = cv2.imread(str(out / "masks" / row["mask_filename"]), cv2.IMREAD_UNCHANGED)
        assert tile.shape[:2] == tmask.shape == (TILE_H, TILE_W)
        assert Path(row["mask_filename"]).stem == Path(row["output_filename"]).stem
        assert RAW_OSTEOPHYTE not in np.unique(tmask)  # merged into cartilage
        for label in np.unique(tmask):
            pix = tile[tmask == label]
            if label == CARTILAGE:
                ok = np.all(pix == COLORS[2], axis=1) | np.all(pix == COLORS[RAW_OSTEOPHYTE], axis=1)
            else:
                ok = np.all(pix == expected_color[label], axis=1)
            assert ok.all(), f"label {label} misaligned in {row['output_filename']}"


def test_scaled_size_recorded_and_tile_names_match_coords(tmp_path):
    img_dir, _, img, _ = make_sample(tmp_path)
    out = tmp_path / "out"
    run(img_dir, out, scale=0.5)
    for row in read_mapping(out):
        assert int(row["scaled_h"]) == img.shape[0] // 2
        assert int(row["scaled_w"]) == img.shape[1] // 2
        assert row["output_filename"].endswith(f"_tile_{int(row['y']):05d}_{int(row['x']):05d}.tif")


def test_tile_set_identical_across_stain_norm(tmp_path):
    img_dir, _, _, _ = make_sample(tmp_path)
    run(img_dir, tmp_path / "none", stain_norm="none")
    run(img_dir, tmp_path / "rh", stain_norm="reinhard")
    none_rows, rh_rows = read_mapping(tmp_path / "none"), read_mapping(tmp_path / "rh")
    assert [r["output_filename"] for r in none_rows] == [r["output_filename"] for r in rh_rows]
    assert {r["stain_norm"] for r in rh_rows} == {"reinhard"}
    # and the setting actually changed the pixels
    a = cv2.imread(str(tmp_path / "none" / none_rows[0]["output_filename"]))
    b = cv2.imread(str(tmp_path / "rh" / rh_rows[0]["output_filename"]))
    assert not np.array_equal(a, b)


def test_background_tiles_skipped_and_have_no_masks(tmp_path):
    grid = np.zeros((4, 6), dtype=np.uint8)
    grid[0, 0] = 1  # one small piece of tissue in a corner
    img_dir, mask_dir, _, _ = make_sample(tmp_path, labels_grid=grid)
    out = tmp_path / "out"
    run(img_dir, out, mask_dir=mask_dir)
    rows = read_mapping(out)
    assert 0 < len(rows) < 30
    assert sorted(p.name for p in (out / "masks").iterdir()) == sorted(r["mask_filename"] for r in rows)


def test_white_balance_is_one_transform_for_the_whole_image(tmp_path):
    img_dir, _, img, _ = make_sample(tmp_path)
    out = tmp_path / "out"
    run(img_dir, out, apply_white_balance=True, tile_skip_bg=False)
    whole = pp.gray_world_white_balance(img)
    for row in read_mapping(out):
        y, x = int(row["y"]), int(row["x"])
        tile = cv2.imread(str(out / row["output_filename"]))
        assert np.array_equal(tile, whole[y : y + TILE_H, x : x + TILE_W])


def test_human_mask_with_growth_plate_is_skipped(tmp_path):
    grid = np.full((4, 6), 1, dtype=np.uint8)
    grid[1, 1] = GROWTH_PLATE
    img_dir, mask_dir, _, _ = make_sample(tmp_path, labels_grid=grid)
    counts = run(img_dir, tmp_path / "out", mask_dir=mask_dir, species="human")
    assert counts["images_skipped"] == 1 and counts["outputs_written"] == 0


def test_missing_mask_is_skipped(tmp_path):
    img_dir, mask_dir, _, _ = make_sample(tmp_path)
    (mask_dir / "slide.png").unlink()
    counts = run(img_dir, tmp_path / "out", mask_dir=mask_dir)
    assert counts["images_skipped"] == 1


def test_auto_mpp_refuses_default_dpi(tmp_path):
    path = tmp_path / "x.tif"
    tifffile.imwrite(path, np.zeros((8, 8, 3), np.uint8), resolution=(300, 300), resolutionunit="INCH")
    with pytest.raises(pp.ScaleError, match="software default"):
        pp.resolve_scale(path, 1.0, "auto", 2.0)
    assert pp.resolve_scale(path, 1.0, 0.25, 1.0) == (0.25, 0.25)


def test_mask_rescale_never_invents_classes():
    mask = np.kron(np.array([[0, 1], [2, 4]], np.uint8), np.ones((7, 7), np.uint8))
    out = pp.rescale(mask, 0.37, 0.61, is_mask=True)
    assert set(np.unique(out)) <= {0, 1, 2, 4}


def _cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_cli_end_to_end(tmp_path):
    img_dir, mask_dir, _, _ = make_sample(tmp_path)
    out = tmp_path / "cli_out"
    res = _cli(
        "--input-dir", str(img_dir), "--output-dir", str(out), "--mask-dir", str(mask_dir),
        "--width", str(TILE_W), "--height", str(TILE_H),
        "--source-mpp", "auto", "--target-mpp", "1.0", "--stain-norm", "reinhard",
    )
    assert res.returncode == 0, res.stderr
    assert "Stain normalization: reinhard" in res.stdout
    assert read_mapping(out)


@pytest.mark.parametrize(
    "args, message",
    [
        (["--no-export-tiles", "--scale", "0.5"], "tile export only"),
        (["--no-reinhard", "--stain-norm", "reinhard"], "conflicts"),
        (["--target-mpp", "1.0"], "together"),
        (["--scale", "0.5", "--source-mpp", "1", "--target-mpp", "2"], "not both"),
        (["--reinhard-ref-json", "ref.json"], "requires --stain-norm reinhard"),
    ],
)
def test_cli_rejects_conflicting_options(tmp_path, args, message):
    res = _cli("--input-dir", str(tmp_path), *args)
    assert res.returncode == 2
    assert message in res.stderr


def test_default_is_no_stain_norm():
    assert pp.parse_args([]).stain_norm == "none"


def test_palette_png_mask_reads_indices(tmp_path):
    from PIL import Image

    ids = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    im = Image.fromarray(ids, mode="P")
    im.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0] + [0] * 753)
    im.save(tmp_path / "m.png")
    assert pp.load_mask(tmp_path / "m.png").tolist() == ids.tolist()


def test_rgba_mask_ignores_alpha(tmp_path):
    ids = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    rgba = np.dstack([ids, ids, ids, np.full_like(ids, 255)])
    cv2.imwrite(str(tmp_path / "m.png"), rgba)
    assert pp.load_mask(tmp_path / "m.png").tolist() == ids.tolist()


def test_16bit_mask_out_of_range_rejected(tmp_path):
    cv2.imwrite(str(tmp_path / "m.tif"), np.full((4, 4), 258, dtype=np.uint16))
    with pytest.raises(ValueError, match="outside 0..255"):
        pp.load_mask(tmp_path / "m.tif")


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "abc"])
def test_cli_rejects_non_finite_scale(tmp_path, bad):
    res = _cli("--input-dir", str(tmp_path), "--scale", bad)
    assert res.returncode == 2
