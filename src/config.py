"""Config loading for TOAD.

Config files are YAML with these special sections:

    paths:        # resolved against REPO_ROOT (e.g. outputs/)
      output_dir: outputs
    data_paths:   # resolved against $TOAD_DATA_ROOT (data lives on HPG, not in the repo)
      train_csv: rat_original/train.csv
    device: auto  # auto | cpu | gpu (cuda also accepted); PyTorch code: torch_device(cfg["device"])

Everything else is returned as-is. Set the data root once in your shell, e.g.
    export TOAD_DATA_ROOT=/blue/<group>/toad
so nobody's personal path ends up in a committed config.
"""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT_ENV = "TOAD_DATA_ROOT"
DEVICES = ("auto", "cpu", "gpu")


def get_data_root() -> Path:
    """Return the data root from $TOAD_DATA_ROOT, or raise if it is not set."""
    value = os.environ.get(DATA_ROOT_ENV)
    if not value:
        raise EnvironmentError(
            f"{DATA_ROOT_ENV} is not set. Point it at the data directory, e.g.\n"
            f"  export {DATA_ROOT_ENV}=/blue/<group>/toad"
        )
    return Path(value).expanduser().resolve()


def resolve_path(path, root: Path) -> Path:
    """Resolve a relative path against root. Absolute paths are kept as-is."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def gpu_available() -> bool:
    """True if TensorFlow or PyTorch can see a GPU. Imports are lazy."""
    try:
        import tensorflow as tf

        if tf.config.list_physical_devices("GPU"):
            return True
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    return False


def resolve_device(device: str = "auto") -> str:
    """Turn a device setting into "gpu" or "cpu". "auto" uses a GPU if one is available.

    "cuda" is accepted as an alias for "gpu".
    """
    device = str(device).lower()
    if device == "cuda":
        device = "gpu"
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES} (or 'cuda'), got {device!r}")
    if device == "auto":
        return "gpu" if gpu_available() else "cpu"
    return device


def torch_device(device: str) -> str:
    """Translate a resolved device ("gpu"/"cpu") into PyTorch's name ("cuda"/"cpu").

    Usage: model.to(torch_device(cfg["device"]))
    """
    return "cuda" if resolve_device(device) == "gpu" else "cpu"


def load_config(path) -> dict:
    """Load a YAML config and resolve its paths.

    A relative config path is looked up from REPO_ROOT, so
    load_config("configs/baseline.yml") works from any directory.
    """
    config_path = resolve_path(path, REPO_ROOT)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    cfg["paths"] = {
        key: resolve_path(value, REPO_ROOT)
        for key, value in (cfg.get("paths") or {}).items()
    }

    data_paths = cfg.get("data_paths") or {}
    if data_paths:
        data_root = get_data_root()
        data_paths = {key: resolve_path(value, data_root) for key, value in data_paths.items()}
    cfg["data_paths"] = data_paths

    cfg["device"] = resolve_device(cfg.get("device", "auto"))
    return cfg
