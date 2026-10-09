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


# Adult human knees have fused growth plates, so class 3 is never annotated in
# human masks. The model keeps the output channel (NUM_CLASSES stays 5) so rat and
# human models share one head; on human data a predicted 3 is simply an error.
HUMAN_ABSENT_CLASSES = frozenset({GROWTH_PLATE})
HUMAN_CLASS_IDS = frozenset(CLASS_NAMES) - HUMAN_ABSENT_CLASSES


def validate_mask(mask: np.ndarray, allowed=None) -> None:
    """Raise if a mask contains values outside ``allowed`` (default: all contract IDs)."""
    allowed = set(CLASS_NAMES) if allowed is None else set(allowed)
    present = {int(v) for v in np.unique(mask)}
    invalid = present - allowed
    if not invalid:
        return
    hints = []
    known = invalid & set(CLASS_NAMES)
    if known:
        names = ", ".join(f"{i} ({CLASS_NAMES[i]})" for i in sorted(known))
        hints.append(f"Class {names} is not allowed for this data.")
    if invalid - known:
        hints.append("Did you forget merge_osteophyte()?")
    raise ValueError(
        f"Mask contains invalid class IDs {sorted(invalid)}. "
        f"Expected only {sorted(allowed)}. {' '.join(hints)}"
    )


def prepare_mask(mask: np.ndarray, species: str) -> np.ndarray:
    """Merge osteophyte into cartilage and validate against the species' classes."""
    if species not in ("rat", "human"):
        raise ValueError(f"species must be 'rat' or 'human', got {species!r}")
    merged = merge_osteophyte(mask)
    validate_mask(merged, allowed=HUMAN_CLASS_IDS if species == "human" else None)
    return merged
