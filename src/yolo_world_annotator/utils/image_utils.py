from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(folder: Path) -> list[Path]:
    """Return supported images in a stable, case-insensitive filename order."""
    if not folder.is_dir():
        return []
    return sorted(
        (
            item
            for item in folder.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ),
        key=lambda item: item.name.casefold(),
    )


def read_image(path: Path) -> np.ndarray:
    """Read an image from any Windows path, including non-ASCII filenames."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"无法读取图片文件：{path}") from exc
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"图片损坏或格式不受支持：{path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def bgr_to_qimage(image: np.ndarray) -> QImage:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("期望 H×W×3 的 BGR 图片")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    return QImage(
        rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
    ).copy()

