import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _py(script, *args):
    return subprocess.run(
        [sys.executable, str(SRC / script), *args], capture_output=True, text=True, cwd=SRC.parent
    )


def test_compare_predictions_agreement_and_dice(tmp_path):
    a, b, gt = tmp_path / "a", tmp_path / "b", tmp_path / "gt"
    for d in (a, b, gt):
        d.mkdir()
    truth = np.zeros((4, 4), np.uint8)
    truth[:, 2:] = 1  # left half background, right half bone
    pred_a = truth.copy()  # perfect
    pred_b = truth.copy()
    pred_b[0, 2:] = 3  # 2 bone pixels predicted as growth plate
    cv2.imwrite(str(a / "prediction_s_tile_00000_00000.tif"), pred_a)
    cv2.imwrite(str(b / "prediction_s_tile_00000_00000.tif"), pred_b)
    cv2.imwrite(str(gt / "s_tile_00000_00000.png"), truth)

    csv_path = tmp_path / "cmp.csv"
    res = _py(
        "compare_predictions.py", "--a", str(a), "--b", str(b), "--name-a", "none", "--name-b", "reinhard",
        "--gt-dir", str(gt), "--species", "human", "--csv", str(csv_path),
    )
    assert res.returncode == 0, res.stderr
    assert "Pixel agreement none vs reinhard: 0.8750" in res.stdout  # 14 of 16
    # bone Dice for B: 2*6 / (6 + 8) = 0.857
    dice_table = res.stdout.split("Dice none")[1]
    bone_line = next(line for line in dice_table.splitlines() if line.startswith("bone"))
    assert bone_line.split()[1:] == ["1.000", "0.857"]
    assert not any(line.startswith("growth_plate") and "Dice" in line for line in res.stdout.splitlines())
    assert "predicted growth-plate share is 0.0000 (none) vs 0.1250 (reinhard)" in res.stdout
    assert csv_path.exists()


keras = pytest.importorskip("keras")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A 1x1-conv 5-class model: enough to exercise the real loading/preprocessing code."""
    inputs = keras.Input((192, 256, 3))
    outputs = keras.layers.Conv2D(5, 1, activation="softmax")(inputs)
    model = keras.Model(inputs, outputs)
    path = tmp_path_factory.mktemp("model") / "tiny.keras"
    model.save(path)
    return path


def _slide(path, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((384, 512, 3), 250, np.uint8)
    img[50:330, 60:450] = rng.integers(40, 200, size=(280, 390, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.mark.parametrize("stain", ["none", "reinhard"])
def test_forward_pass_runs_with_each_stain_setting(tmp_path, tiny_model, stain):
    _slide(tmp_path / "rat.tif")
    out = tmp_path / f"pred_{stain}.png"
    res = _py(
        "forward_pass.py", "--model", str(tiny_model), "--image", str(tmp_path / "rat.tif"),
        "--output", str(out), "--stain-norm", stain,
    )
    assert res.returncode == 0, res.stderr
    assert f"stain norm: {stain}" in res.stdout
    mask = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert mask.shape == (192, 256) and mask.max() < 5


def test_forward_pass_preprocessing_differs_by_stain_setting(tmp_path):
    sys.path.insert(0, str(SRC))
    from forward_pass import load_and_preprocess_image

    _slide(tmp_path / "rat.tif")
    x_none, _ = load_and_preprocess_image(str(tmp_path / "rat.tif"), 192, 256, rgb=False)
    x_rh, _ = load_and_preprocess_image(str(tmp_path / "rat.tif"), 192, 256, rgb=False, stain_norm="reinhard")
    assert x_none.shape == x_rh.shape == (1, 192, 256, 3)
    assert not np.allclose(x_none, x_rh)


def test_predict_folder_with_cli_overrides(tmp_path, tiny_model):
    in_dir = tmp_path / "tiles"
    in_dir.mkdir()
    _slide(in_dir / "a.tif", 1)
    _slide(in_dir / "b.jpg", 2)
    out = tmp_path / "pred"
    res = _py(
        "predict.py", "--model", str(tiny_model), "--input-dir", str(in_dir),
        "--output-dir", str(out), "--stain-norm", "reinhard",
    )
    assert res.returncode == 0, res.stderr
    written = sorted(p.name for p in (out / "Original_Predictions").iterdir())
    assert written == ["prediction_a.tif", "prediction_b.png"]
    mask = cv2.imread(str(out / "Original_Predictions" / "prediction_a.tif"), cv2.IMREAD_UNCHANGED)
    assert mask.dtype == np.uint8 and mask.max() < 5


def test_rat_side_rejects_ref_json_without_reinhard(tmp_path, tiny_model):
    res = _py(
        "forward_pass.py", "--model", str(tiny_model), "--image", "x.tif", "--output", "y.png",
        "--reinhard-ref-json", "ref.json",
    )
    assert res.returncode == 2 and "requires --stain-norm reinhard" in res.stderr


def test_compare_rejects_identical_names(tmp_path):
    res = _py("compare_predictions.py", "--a", str(tmp_path), "--b", str(tmp_path), "--name-a", "x", "--name-b", "x")
    assert res.returncode == 2 and "must differ" in res.stderr


def test_predict_without_data_root_explains_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("TOAD_DATA_ROOT", raising=False)
    res = _py("predict.py", "--output-dir", str(tmp_path))
    assert res.returncode != 0
    assert "TOAD_DATA_ROOT is not set" in res.stderr
