"""Class labels for TOAD.

Source of truth: docs/DATA_CONTRACT.md. If you change a label here,
update the data contract in the same PR.
"""

import numpy as np

# Class IDs
BACKGROUND = 0
BONE = 1
CARTILAGE = 2
GROWTH_PLATE = 3
MARROW = 4

CLASS_NAMES = {
    BACKGROUND: "background",
    BONE: "bone",
    CARTILAGE: "cartilage",
    GROWTH_PLATE: "growth_plate",
    MARROW: "marrow",
}

NUM_CLASSES = len(CLASS_NAMES)

# Raw annotation label for osteophyte. Merged into cartilage before use.
RAW_OSTEOPHYTE = 5

# RGB colors for visualizations, so every plot and overlay matches.
CLASS_COLORS = {
    BACKGROUND: (0, 0, 0),
    BONE: (241, 196, 15),
    CARTILAGE: (52, 152, 219),
    GROWTH_PLATE: (46, 204, 113),
    MARROW: (231, 76, 60),
}


def merge_osteophyte(mask: np.ndarray) -> np.ndarray:
    """Return a copy of a raw mask with osteophyte merged into cartilage."""
    merged = mask.copy()
    merged[merged == RAW_OSTEOPHYTE] = CARTILAGE
    return merged


def validate_mask(mask: np.ndarray) -> None:
    """Raise if a mask contains values outside the contract's class IDs."""
    invalid = set(np.unique(mask)) - set(CLASS_NAMES)
    if invalid:
        raise ValueError(
            f"Mask contains invalid class IDs {sorted(invalid)}. "
            f"Expected only {sorted(CLASS_NAMES)}. "
            "Did you forget merge_osteophyte()?"
        )