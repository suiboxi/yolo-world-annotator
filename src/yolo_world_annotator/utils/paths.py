"""Cross-platform writable paths used by source and frozen builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def user_data_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return (base / "yolo-world-annotator").resolve()


def default_weights_dir() -> Path:
    """Return a writable model directory, preserving source-checkout weights."""

    override = os.environ.get("YOLO_WORLD_WEIGHTS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return (Path(sys.executable).resolve().parent / "models" / "weights").resolve()
    source_root = Path(__file__).resolve().parents[3]
    source_weights = source_root / "models" / "weights"
    if (source_root / "pyproject.toml").is_file() and source_weights.is_dir():
        return source_weights.resolve()
    return (user_data_dir() / "weights").resolve()


__all__ = ["default_weights_dir", "user_data_dir"]
