"""Safe crop helpers shared by semantic verifiers.

Coordinates are always expressed in original-image pixels.  The helpers are
intentionally tolerant of either a :class:`BoundingBox` or a four-value
sequence so they can be used by offline tools without importing the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from PIL import Image


@dataclass(frozen=True, slots=True)
class CropRegion:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


def _coordinates(box: object) -> tuple[float, float, float, float]:
    if all(hasattr(box, name) for name in ("x1", "y1", "x2", "y2")):
        return float(box.x1), float(box.y1), float(box.x2), float(box.y2)
    if isinstance(box, Sequence) and len(box) == 4:
        return tuple(float(value) for value in box)  # type: ignore[return-value]
    raise TypeError("bbox 必须是 BoundingBox 或 [x1, y1, x2, y2]")


def padded_region(
    image_size: tuple[int, int],
    box: object,
    padding_ratio: float = 0.10,
) -> CropRegion:
    """Return a bounded integer crop region with proportional context padding."""

    width, height = (max(1, int(value)) for value in image_size)
    x1, y1, x2, y2 = _coordinates(box)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    ratio = min(0.50, max(0.0, float(padding_ratio)))
    pad_x = max(1.0, right - left) * ratio
    pad_y = max(1.0, bottom - top) * ratio
    region = CropRegion(
        max(0, int(left - pad_x)),
        max(0, int(top - pad_y)),
        min(width, int(right + pad_x + 0.999999)),
        min(height, int(bottom + pad_y + 0.999999)),
    )
    # A malformed/degenerate detector box should still produce a valid one
    # pixel crop rather than an exception deep inside a batch.
    right_edge = max(region.left + 1, region.right)
    bottom_edge = max(region.top + 1, region.bottom)
    if region.left >= width:
        left_edge = max(0, width - 1)
        right_edge = width
    else:
        left_edge = region.left
        right_edge = min(width, right_edge)
    if region.top >= height:
        top_edge = max(0, height - 1)
        bottom_edge = height
    else:
        top_edge = region.top
        bottom_edge = min(height, bottom_edge)
    return CropRegion(left_edge, top_edge, right_edge, bottom_edge)


def crop_image(image: Image.Image, box: object, padding_ratio: float = 0.10) -> Image.Image:
    """Crop *image* around *box* with bounded padding and RGB conversion."""

    pil_image = image.convert("RGB")
    region = padded_region(pil_image.size, box, padding_ratio)
    return pil_image.crop(region.box)


# Compatibility aliases used by callers that prefer an explicit name.
crop_with_padding = crop_image
crop_box = crop_image
compute_crop_region = padded_region


__all__ = [
    "CropRegion",
    "padded_region",
    "compute_crop_region",
    "crop_image",
    "crop_with_padding",
    "crop_box",
]
