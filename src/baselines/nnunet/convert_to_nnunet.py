"""
convert_to_nnunet.py — rat SEKO data -> nnU-Net dataset format.

Converts rat images/masks (paired across stains: H&E, Safranin-O, Toluidine
Blue) into nnU-Net v2's expected layout:

    $nnUNet_raw/DatasetXXX_NAME/
        dataset.json
        imagesTr/   case_0000.png case_0001.png case_0002.png  (R,G,B split)
        labelsTr/   case.png                                   (class IDs 0-4)
        imagesTs/ labelsTs/ (optional test split)

Two important steps:
  1. RGB split — nnU-Net needs each of R/G/B as its own file, with
     "overwrite_image_reader_writer": "NaturalImage2DIO" in dataset.json so
     it recombines them as one RGB image instead of 3 unrelated modalities.
  2. Osteophyte (raw label 5) -> cartilage (2) remap, per docs/DATA_CONTRACT.md.

LIMITATIONS:
  - Developed/tested only against synthetic fixtures (--selftest) — real rat
    data lives on HiPerGator, not yet reachable. This proves the conversion
    logic is correct; it doesn't prove real filenames/images behave the same.
  - The real Testing_images/ + osteophyte_relabelled_masks/ layout is assumed
    (reuses pair_stain's stem-matching), not confirmed.
  - Once HPG access lands: validate against real data, and always run
    `nnUNetv2_plan_and_preprocess -d XXX --verify_dataset_integrity`.
  - macOS local dev only: --selftest may need KMP_DUPLICATE_LIB_OK=TRUE set
    (libomp conflict, doesn't occur on HPG's Linux env — don't add it there).

Usage:
    python src/baselines/nnunet/convert_to_nnunet.py --selftest
    python src/baselines/nnunet/convert_to_nnunet.py \\
        --data-root /path/to/rat_original --dataset-id 1 --dataset-name Rat
    python src/baselines/nnunet/convert_to_nnunet.py --spotcheck /path/to/DatasetXXX_Rat
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np

# Stain subfolders (matches build_train_csv.py on feature/sam-3).
DEFAULT_STAIN_CONFIG = [
    ("H&E", "H&E/Raw256x192", "HE"),
    ("Safo", "SafO/Raw 256x192", "SafO"),
    ("TolBlu", "TolBlu/Raw 256x192", "TolBlu"),
]

# DATA_CONTRACT's 0-4 scheme. Osteophyte (raw label 5) is merged into
# cartilage, so there's no separate "osteophyte" key.
DATA_CONTRACT_LABELS = {
    "background": 0,
    "bone": 1,
    "cartilage": 2,
    "growth_plate": 3,
    "marrow": 4,
}

DEFAULT_FILE_ENDING = ".png"


# ───────────────────────────── matching ─────────────────────────────
# Ported from build_train_csv.py (feature/sam-3)

def mask_stem(filename: str) -> str:
    """Strip _bone.tif / _bone_binary.tif suffix."""
    return re.sub(r'(_bone_binary|_bone)\.tif$', '', filename)


def image_stem(filename: str) -> str:
    """Strip ,_resized_gray.tif / _resized_gray.tif / plain .tif suffix."""
    s = re.sub(r',?_resized_gray\.tif$', '', filename)
    if s == filename:  # no resized_gray suffix
        s = re.sub(r'\.tif$', '', filename)  # fall back to stripping plain .tif
    return s


def suffix_token(stem: str) -> str:
    """
    Last alphanumeric token after splitting on [-_\\s]+.
    e.g. 'GALIDO_TOL_BLUE_06r' -> '06r'
         'Gal-IDO_Tol-Blue_06r' -> '06r'
    """
    tokens = re.split(r'[-_\s]+', stem.lower())
    return tokens[-1] if tokens else stem.lower()


def pair_stain(labelled_dir: str, input_dir: str, stain: str, ext: str = ".tif"):
    """
    Match masks to images for one stain. Tries exact-stem match first, then
    falls back to an unambiguous suffix-token match (tokens shared by 2+
    images are excluded, not guessed). Returns (pairs, unmatched_mask_paths).
    """
    if not os.path.isdir(labelled_dir) or not os.path.isdir(input_dir):
        return [], []

    masks = sorted(
        os.path.join(labelled_dir, f)
        for f in os.listdir(labelled_dir)
        if f.lower().endswith(ext)
    )
    images = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(ext)
    )

    img_by_exact = {image_stem(os.path.basename(f)): f for f in images}

    img_by_suffix_raw = defaultdict(list)
    for f in images:
        img_by_suffix_raw[suffix_token(image_stem(os.path.basename(f)))].append(f)
    img_by_suffix = {k: v[0] for k, v in img_by_suffix_raw.items() if len(v) == 1}

    pairs, unmatched = [], []
    for mpath in masks:
        mname = os.path.basename(mpath)
        mstem = mask_stem(mname)
        mtoken = suffix_token(mstem)

        if mstem in img_by_exact:
            pairs.append((img_by_exact[mstem], mpath, stain))
        elif mtoken in img_by_suffix:
            pairs.append((img_by_suffix[mtoken], mpath, stain))
        else:
            unmatched.append(mpath)

    return pairs, unmatched


# ───────────────────────────── label remap ─────────────────────────────
# Vectorized rewrite of osteophyte_label_change.py's nested-loop version.

def remap_osteophyte_to_cartilage(mask: np.ndarray, raw_label: int = 5, target_label: int = 2) -> np.ndarray:
    unexpected = set(np.unique(mask).tolist()) - {0, 1, 2, 3, 4, 5}
    if unexpected:
        print(f"[WARN] mask contains unexpected label value(s): {sorted(unexpected)}", file=sys.stderr)
    out = mask.copy()
    out[mask == raw_label] = target_label
    return out


# ───────────────────────────── RGB split ─────────────────────────────

def split_rgb_channels(image_bgr: np.ndarray):
    """
    Returns (R, G, B) as (H, W) uint8 arrays. Replicates into R=G=B with a
    warning if given a 2D (grayscale) input — unconfirmed whether real rat
    inputs are true color or pre-converted (the `_resized_gray` filename
    suffix hints some may already be grayscale).
    """
    if image_bgr.ndim == 2:
        print("[WARN] input image is grayscale, not RGB -- replicating into R=G=B", file=sys.stderr)
        gray = image_bgr
        return gray.copy(), gray.copy(), gray.copy()

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return r, g, b


def sanitize_case_id(raw: str) -> str:
    """Replace characters nnU-Net case identifiers disallow with underscores."""
    return re.sub(r'[^A-Za-z0-9_]+', '_', raw).strip('_')


def make_case_id(stain: str, index: int, orig_stem: str) -> str:
    """Stain+index make cross-stain ID collisions impossible; orig_stem keeps it traceable."""
    return f"Rat_{stain}_{index:04d}_{sanitize_case_id(orig_stem)}"


# ───────────────────────────── per-case IO ─────────────────────────────

def write_image_case(case_id: str, image_path: str, images_out_dir: str, file_ending: str = DEFAULT_FILE_ENDING):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read image: {image_path}")
    r, g, b = split_rgb_channels(img)
    for idx, channel in enumerate((r, g, b)):
        out_path = os.path.join(images_out_dir, f"{case_id}_{idx:04d}{file_ending}")
        cv2.imwrite(out_path, channel)


def write_label_case(case_id: str, mask_path: str, labels_out_dir: str, file_ending: str = DEFAULT_FILE_ENDING) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Could not read mask: {mask_path}")
    remapped = remap_osteophyte_to_cartilage(mask)
    out_path = os.path.join(labels_out_dir, f"{case_id}{file_ending}")
    cv2.imwrite(out_path, remapped)
    return remapped


# ───────────────────────────── dataset.json ─────────────────────────────

def build_dataset_json(num_training: int, file_ending: str = DEFAULT_FILE_ENDING) -> dict:
    return {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": dict(DATA_CONTRACT_LABELS),
        "numTraining": num_training,
        "file_ending": file_ending,
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }


def write_dataset_json(dataset_dict: dict, dataset_dir: str):
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_dict, f, indent=2)


# ───────────────────────────── orchestration ─────────────────────────────

def convert(
    data_root: str,
    nnunet_raw: str,
    dataset_id: int = 1,
    dataset_name: str = "Rat",
    stain_config=DEFAULT_STAIN_CONFIG,
    test_images_dir: str | None = None,
    test_masks_dir: str | None = None,
    include_test_labels: bool = False,
    manifest_csv: str | None = None,
    overwrite: bool = False,
) -> dict:
    dataset_dir = os.path.join(nnunet_raw, f"Dataset{dataset_id:03d}_{dataset_name}")
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")
    labels_ts = os.path.join(dataset_dir, "labelsTs")

    if os.path.isdir(dataset_dir) and os.listdir(dataset_dir) and not overwrite:
        raise FileExistsError(
            f"{dataset_dir} already exists and is non-empty. "
            "Pass overwrite=True to re-run into it anyway."
        )

    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)

    all_pairs, all_unmatched = [], []
    for labelled_sub, input_sub, stain in stain_config:
        labelled_dir = os.path.join(data_root, "Labelled_images", labelled_sub)
        input_dir = os.path.join(data_root, "Input_images", input_sub)
        pairs, unmatched = pair_stain(labelled_dir, input_dir, stain)
        all_pairs.extend(pairs)
        all_unmatched.extend(unmatched)
        print(f"  {stain:8s}  {len(pairs):3d} matched  {len(unmatched):3d} unmatched")

    manifest_rows = []
    for i, (image_path, mask_path, stain) in enumerate(all_pairs):
        stem = mask_stem(os.path.basename(mask_path))
        case_id = make_case_id(stain, i, stem)
        write_image_case(case_id, image_path, images_tr)
        write_label_case(case_id, mask_path, labels_tr)
        manifest_rows.append((case_id, "train", image_path, mask_path, stain))

    test_pairs = []
    if test_images_dir and test_masks_dir:
        os.makedirs(images_ts, exist_ok=True)
        if include_test_labels:
            os.makedirs(labels_ts, exist_ok=True)
        test_pairs, test_unmatched = pair_stain(test_masks_dir, test_images_dir, "Test")
        all_unmatched.extend(test_unmatched)
        for i, (image_path, mask_path, stain) in enumerate(test_pairs):
            stem = mask_stem(os.path.basename(mask_path))
            case_id = make_case_id(stain, i, stem)
            write_image_case(case_id, image_path, images_ts)
            if include_test_labels:
                write_label_case(case_id, mask_path, labels_ts)
            manifest_rows.append((case_id, "test", image_path, mask_path, stain))
        print(f"  {'Test':8s}  {len(test_pairs):3d} matched  {len(test_unmatched):3d} unmatched")

    dataset_json = build_dataset_json(num_training=len(all_pairs))
    write_dataset_json(dataset_json, dataset_dir)

    if manifest_csv:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_csv)), exist_ok=True)
        with open(manifest_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["case_id", "split", "image_path", "mask_path", "stain"])
            writer.writerows(manifest_rows)

    if all_unmatched:
        print(f"\n[WARN] {len(all_unmatched)} unmatched mask(s) -- no corresponding image found:")
        for p in all_unmatched:
            print(f"  {p}")

    print(
        "\nLIMITATIONS: not validated against real rat data (HPG access pending) -- run "
        "`nnUNetv2_plan_and_preprocess -d <id> --verify_dataset_integrity` before trusting "
        "this, and re-run spot_check_dataset() once real data is reachable."
    )

    return {
        "dataset_dir": dataset_dir,
        "num_train": len(all_pairs),
        "num_test": len(test_pairs),
        "num_unmatched": len(all_unmatched),
    }


# ───────────────────────────── spot-check utilities ─────────────────────────────

def reconstruct_rgb_preview(images_dir: str, case_id: str, out_path: str | None = None, file_ending: str = DEFAULT_FILE_ENDING) -> np.ndarray:
    channels = []
    for idx in range(3):
        path = os.path.join(images_dir, f"{case_id}_{idx:04d}{file_ending}")
        ch = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if ch is None:
            raise IOError(f"Could not read channel file: {path}")
        channels.append(ch)
    rgb = np.stack(channels, axis=-1)  # (H, W, 3), R,G,B order

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return rgb


def mask_value_histogram(label_path: str) -> dict:
    mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Could not read mask: {label_path}")
    values, counts = np.unique(mask, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def spot_check_dataset(dataset_dir: str, n_samples: int = 3, preview_out_dir: str | None = None, seed: int = 0):
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    case_ids = sorted(
        os.path.splitext(f)[0] for f in os.listdir(labels_tr) if f.endswith(DEFAULT_FILE_ENDING)
    )

    rng = np.random.default_rng(seed)
    sample = rng.choice(case_ids, size=min(n_samples, len(case_ids)), replace=False)

    for case_id in sample:
        preview_path = None
        if preview_out_dir:
            preview_path = os.path.join(preview_out_dir, f"{case_id}_preview{DEFAULT_FILE_ENDING}")
        reconstruct_rgb_preview(images_tr, case_id, out_path=preview_path)
        hist = mask_value_histogram(os.path.join(labels_tr, f"{case_id}{DEFAULT_FILE_ENDING}"))
        print(f"  {case_id}: label histogram = {hist}" + (f" (preview -> {preview_path})" if preview_path else ""))


# ───────────────────────────── self-test ─────────────────────────────

def _make_synthetic_image(h: int, w: int, r: int, g: int, b: int) -> np.ndarray:
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:, :, 0] = b
    bgr[:, :, 1] = g
    bgr[:, :, 2] = r
    return bgr


def _make_synthetic_mask(h: int, w: int, include_osteophyte: bool) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[1:3, 1:3] = 1  # bone
    mask[3:5, 3:5] = 4  # marrow
    mask[5:7, 1:3] = 3  # growth plate
    mask[1:3, 5:7] = 2  # cartilage
    if include_osteophyte:
        mask[6:8, 6:8] = 5  # osteophyte -> must become cartilage (2)
    return mask


def build_selftest_fixtures(root: str) -> dict:
    data_root = os.path.join(root, "fixture_data")

    dirs = {
        "he_labelled": os.path.join(data_root, "Labelled_images", "H&E"),
        "he_input": os.path.join(data_root, "Input_images", "H&E", "Raw256x192"),
        "tb_labelled": os.path.join(data_root, "Labelled_images", "TolBlu"),
        "tb_input": os.path.join(data_root, "Input_images", "TolBlu", "Raw 256x192"),
        "test_images": os.path.join(data_root, "Testing_images"),
        "test_masks": os.path.join(data_root, "osteophyte_relabelled_masks"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    H = W = 8
    expected = {
        "exact_matches": set(),
        "fallback_matches": set(),
        "excluded_ambiguous_tokens": {"12"},
        "unmatched_masks": set(),
        "osteophyte_cases": set(),
    }

    # HE: exact-stem match, carries an osteophyte pixel
    cv2.imwrite(os.path.join(dirs["he_input"], "MMT_71_resized_gray.tif"), _make_synthetic_image(H, W, 10, 20, 30))
    cv2.imwrite(os.path.join(dirs["he_labelled"], "MMT_71_bone.tif"), _make_synthetic_mask(H, W, include_osteophyte=True))
    expected["exact_matches"].add("MMT_71")
    expected["osteophyte_cases"].add("MMT_71")

    # HE: unpaired image, no mask -- should just be ignored, not crash
    cv2.imwrite(os.path.join(dirs["he_input"], "MMT_72_resized_gray.tif"), _make_synthetic_image(H, W, 40, 50, 60))

    # HE: unmatched mask, no image
    cv2.imwrite(os.path.join(dirs["he_labelled"], "MMT_99_bone.tif"), _make_synthetic_mask(H, W, include_osteophyte=False))
    expected["unmatched_masks"].add("MMT_99")

    # TolBlu: suffix-token fallback, recreating the real case/hyphen divergence
    cv2.imwrite(os.path.join(dirs["tb_input"], "Gal-IDO_Tol-Blue_06r.tif"), _make_synthetic_image(H, W, 70, 80, 90))
    cv2.imwrite(os.path.join(dirs["tb_labelled"], "GALIDO_TOL_BLUE_06r_bone.tif"), _make_synthetic_mask(H, W, include_osteophyte=True))
    expected["fallback_matches"].add("06r")
    expected["osteophyte_cases"].add("GALIDO_TOL_BLUE_06r")

    # TolBlu: ambiguous token ("12" shared by 2 images) -- must be excluded from fallback
    cv2.imwrite(os.path.join(dirs["tb_input"], "Sample-A_12.tif"), _make_synthetic_image(H, W, 1, 2, 3))
    cv2.imwrite(os.path.join(dirs["tb_input"], "Other-B_12.tif"), _make_synthetic_image(H, W, 4, 5, 6))
    cv2.imwrite(os.path.join(dirs["tb_labelled"], "Mystery_12_bone.tif"), _make_synthetic_mask(H, W, include_osteophyte=False))
    expected["unmatched_masks"].add("Mystery_12")

    # Test split: one matched pair
    cv2.imwrite(os.path.join(dirs["test_images"], "TestRat_01.tif"), _make_synthetic_image(H, W, 100, 110, 120))
    cv2.imwrite(os.path.join(dirs["test_masks"], "TestRat_01_bone.tif"), _make_synthetic_mask(H, W, include_osteophyte=False))

    return {"data_root": data_root, "test_images_dir": dirs["test_images"], "test_masks_dir": dirs["test_masks"], "expected": expected}


def run_selftest(selftest_dir: str = "nnunet_selftest_output") -> bool:
    os.makedirs(selftest_dir, exist_ok=True)
    fixtures = build_selftest_fixtures(selftest_dir)
    expected = fixtures["expected"]

    stain_config = [
        ("H&E", "H&E/Raw256x192", "HE"),
        ("TolBlu", "TolBlu/Raw 256x192", "TolBlu"),
    ]

    nnunet_raw_fixture = os.path.join(selftest_dir, "nnUNet_raw_fixture")
    manifest_csv = os.path.join(selftest_dir, "conversion_manifest.csv")
    preview_dir = os.path.join(selftest_dir, "previews")

    checks = []  # (description, passed, detail)

    try:
        result = convert(
            data_root=fixtures["data_root"],
            nnunet_raw=nnunet_raw_fixture,
            dataset_id=999,
            dataset_name="SelfTest",
            stain_config=stain_config,
            test_images_dir=fixtures["test_images_dir"],
            test_masks_dir=fixtures["test_masks_dir"],
            include_test_labels=True,
            manifest_csv=manifest_csv,
            overwrite=True,
        )
    except Exception as e:
        print(f"RESULT: 0/1 checks passed -- convert() raised: {e}", file=sys.stderr)
        return False

    dataset_dir = result["dataset_dir"]
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")

    # Map mask stem -> manifest row, for looking up case_id/paths by stem below
    manifest_by_stem = {}
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            stem = mask_stem(os.path.basename(row["mask_path"]))
            manifest_by_stem[stem] = row

    # --- Matching counts ---
    expected_exact = expected["exact_matches"]
    expected_fallback_stems = {"GALIDO_TOL_BLUE_06r"}  # the mask stem for the fallback pair
    train_stems = {k for k, v in manifest_by_stem.items() if v["split"] == "train"}

    checks.append(("exact-match case present", expected_exact <= train_stems, f"expected {expected_exact} subset of {train_stems}"))
    checks.append(("fallback-match case present", expected_fallback_stems <= train_stems, f"expected {expected_fallback_stems} subset of {train_stems}"))
    checks.append(("ambiguous/unmatched cases excluded", not (expected["unmatched_masks"] & train_stems), f"{expected['unmatched_masks']} must not be in {train_stems}"))
    checks.append(("exactly 2 train cases", result["num_train"] == 2, f"got {result['num_train']}"))
    checks.append(("exactly 2 unmatched masks", result["num_unmatched"] == 2, f"got {result['num_unmatched']}"))

    # --- Remap correctness: every 5 became a 2, counts match exactly ---
    for stem in expected["osteophyte_cases"]:
        row = manifest_by_stem.get(stem)
        if row is None:
            checks.append((f"remap check for {stem}", False, "case not found in manifest"))
            continue
        orig_mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        orig_five_count = int(np.sum(orig_mask == 5))
        orig_two_count = int(np.sum(orig_mask == 2))
        case_id = row["case_id"]
        new_hist = mask_value_histogram(os.path.join(labels_tr, f"{case_id}.png"))
        new_five_count = new_hist.get(5, 0)
        new_two_count = new_hist.get(2, 0)
        checks.append((
            f"remap: no 5s remain ({stem})", new_five_count == 0, f"got {new_five_count}"
        ))
        checks.append((
            f"remap: 2-count increased by exactly the original 5-count ({stem})",
            new_two_count == orig_two_count + orig_five_count,
            f"expected {orig_two_count + orig_five_count}, got {new_two_count}",
        ))

    # --- RGB split/reconstruct: exact pixel equality ---
    for stem in train_stems:
        row = manifest_by_stem[stem]
        case_id = row["case_id"]
        original = cv2.imread(row["image_path"], cv2.IMREAD_UNCHANGED)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        reconstructed = reconstruct_rgb_preview(images_tr, case_id, out_path=os.path.join(preview_dir, f"{case_id}_preview.png"))
        checks.append((
            f"RGB reconstruct exact match ({stem})",
            np.array_equal(original_rgb, reconstructed),
            "pixel mismatch after split+reconstruct" if not np.array_equal(original_rgb, reconstructed) else "",
        ))

    # --- dataset.json correctness ---
    with open(os.path.join(dataset_dir, "dataset.json")) as f:
        dj = json.load(f)
    checks.append(("dataset.json: overwrite_image_reader_writer", dj.get("overwrite_image_reader_writer") == "NaturalImage2DIO", str(dj.get("overwrite_image_reader_writer"))))
    checks.append(("dataset.json: labels match DATA_CONTRACT scheme", dj.get("labels") == DATA_CONTRACT_LABELS, str(dj.get("labels"))))
    checks.append(("dataset.json: numTraining matches actual count", dj.get("numTraining") == result["num_train"], f"got {dj.get('numTraining')}"))

    # --- Test split written ---
    test_stems = {k for k, v in manifest_by_stem.items() if v["split"] == "test"}
    checks.append(("test split: one case written", len(test_stems) == 1, f"got {len(test_stems)}"))
    checks.append(("test split: imagesTs populated", os.path.isdir(os.path.join(dataset_dir, "imagesTs")) and len(os.listdir(os.path.join(dataset_dir, "imagesTs"))) > 0, ""))

    # --- Report ---
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print()
    for desc, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        suffix = f" ({detail})" if detail and not ok else ""
        print(f"  [{status}] {desc}{suffix}")

    print(f"\nRESULT: {passed}/{total} checks passed")
    print(f"Previews for manual eyeballing: {os.path.abspath(preview_dir)}")
    print(
        "\nLIMITATIONS: synthetic fixtures only -- doesn't validate real filename mess beyond "
        "the two recreated cases, real image dimensions/stain colors, or the real "
        "Testing_images/ layout (assumed, not confirmed). Real-data validation still required."
    )

    return passed == total


# ───────────────────────────── CLI ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert TOAD rat data to nnU-Net dataset format")
    parser.add_argument("--data-root", help="Path to rat_original/ directory")
    parser.add_argument("--nnunet-raw", help="Override for $nnUNet_raw (else read from environment)")
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--dataset-name", default="Rat")
    parser.add_argument("--test-images-dir", default=None)
    parser.add_argument("--test-masks-dir", default=None)
    parser.add_argument("--include-test-labels", action="store_true")
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--selftest", action="store_true", help="Run the full pipeline against synthetic fixtures, no other args required")
    parser.add_argument("--selftest-dir", default="nnunet_selftest_output")
    parser.add_argument("--spotcheck", default=None, help="Path to an existing converted DatasetXXX_NAME dir")
    parser.add_argument("--n-spotcheck", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.selftest:
        ok = run_selftest(args.selftest_dir)
        sys.exit(0 if ok else 1)

    if args.spotcheck:
        spot_check_dataset(args.spotcheck, n_samples=args.n_spotcheck, seed=args.seed)
        return

    nnunet_raw = args.nnunet_raw or os.environ.get("nnUNet_raw")
    if not args.data_root or not nnunet_raw:
        parser.error("--data-root and --nnunet-raw (or $nnUNet_raw) are required unless --selftest/--spotcheck")

    test_images_dir = args.test_images_dir or os.path.join(args.data_root, "Testing_images")
    test_masks_dir = args.test_masks_dir or os.path.join(args.data_root, "osteophyte_relabelled_masks")
    if not (os.path.isdir(test_images_dir) and os.path.isdir(test_masks_dir)):
        print(f"[WARN] test split dirs not found ({test_images_dir}, {test_masks_dir}) -- skipping imagesTs", file=sys.stderr)
        test_images_dir = test_masks_dir = None

    manifest_csv = args.manifest_csv or os.path.join(
        nnunet_raw, f"Dataset{args.dataset_id:03d}_{args.dataset_name}", "conversion_manifest.csv"
    )

    convert(
        data_root=args.data_root,
        nnunet_raw=nnunet_raw,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        test_images_dir=test_images_dir,
        test_masks_dir=test_masks_dir,
        include_test_labels=args.include_test_labels,
        manifest_csv=manifest_csv,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
