import json

import cv2
import numpy as np
import pytest

import labels
from stain_norm import (
    DEFAULT_REINHARD_REF_MEANS,
    DEFAULT_REINHARD_REF_STDS,
    apply_stain_norm,
    load_reinhard_ref_json,
)


def _tissue_image(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((120, 160, 3), 250, dtype=np.uint8)  # glass background
    img[20:100, 20:140] = rng.integers(60, 180, size=(80, 120, 3), dtype=np.uint8)
    return img


def test_none_returns_input_unchanged():
    img = _tissue_image()
    out = apply_stain_norm(img, "none")
    assert np.array_equal(out, img)


def test_reinhard_moves_tissue_stats_to_reference():
    img = _tissue_image()
    out = apply_stain_norm(img, "reinhard")
    tissue = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 220
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)[tissue]
    # Rounding, clipping and the LAB->BGR->LAB round trip keep this from being exact.
    assert np.allclose(lab.mean(axis=0), DEFAULT_REINHARD_REF_MEANS, atol=4.0)
    assert np.allclose(lab.std(axis=0), DEFAULT_REINHARD_REF_STDS, atol=4.0)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        apply_stain_norm(_tissue_image(), "macenko")


def test_reference_json_round_trip(tmp_path):
    ref = {"L_mean": 1, "L_std": 2, "A_mean": 3, "A_std": 4, "B_mean": 5, "B_std": 6}
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(ref))
    assert load_reinhard_ref_json(path) == ((1.0, 3.0, 5.0), (2.0, 4.0, 6.0))


def test_human_mask_rejects_growth_plate():
    mask = np.array([[0, 1], [2, labels.GROWTH_PLATE]], dtype=np.uint8)
    with pytest.raises(ValueError, match="growth_plate"):
        labels.prepare_mask(mask, "human")


def test_rat_mask_keeps_growth_plate_and_merges_osteophyte():
    mask = np.array([[0, 3], [labels.RAW_OSTEOPHYTE, 4]], dtype=np.uint8)
    out = labels.prepare_mask(mask, "rat")
    assert out.tolist() == [[0, 3], [labels.CARTILAGE, 4]]


def test_unknown_label_value_hint():
    with pytest.raises(ValueError, match="merge_osteophyte"):
        labels.validate_mask(np.array([[0, 9]], dtype=np.uint8))


def test_output_channel_count_unchanged():
    # Human data drops growth plate from labels, not from the model head.
    assert labels.NUM_CLASSES == 5
    assert labels.GROWTH_PLATE not in labels.HUMAN_CLASS_IDS
