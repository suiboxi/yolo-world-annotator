"""A small, Ultralytics-compatible tiled inference runner.

The project intentionally does not require the optional SAHI package.  Every
tile is sent through the existing detector, mapped back to original-image
coordinates and merged once at the end.  This preserves the required order:
SAHI -> YOLO candidates -> merge -> crop-level semantic verification.
"""

from __future__ import annotations

import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from yolo_world_annotator.core.annotation import BoundingBox
from yolo_world_annotator.inference.merge_utils import merge_detections


@dataclass(slots=True)
class SAHIConfig:
    slice_width: int = 1024
    slice_height: int = 1024
    overlap_width_ratio: float = 0.20
    overlap_height_ratio: float = 0.20
    postprocess_type: str = "NMS"
    postprocess_match_threshold: float = 0.50
    postprocess_match_metric: str = "IOU"
    score_threshold: float = 0.0
    max_tiles: int = 0

    def __post_init__(self) -> None:
        self.slice_width = max(32, int(self.slice_width))
        self.slice_height = max(32, int(self.slice_height))
        self.overlap_width_ratio = min(0.95, max(0.0, float(self.overlap_width_ratio)))
        self.overlap_height_ratio = min(0.95, max(0.0, float(self.overlap_height_ratio)))
        self.postprocess_match_threshold = min(1.0, max(0.0, float(self.postprocess_match_threshold)))
        self.score_threshold = min(1.0, max(0.0, float(self.score_threshold)))
        self.max_tiles = max(0, int(self.max_tiles))

    @classmethod
    def from_mapping(cls, values: dict | None) -> "SAHIConfig":
        values = values or {}
        return cls(
            slice_width=values.get("slice_width", values.get("slice_w", 1024)),
            slice_height=values.get("slice_height", values.get("slice_h", 1024)),
            overlap_width_ratio=values.get("overlap_width_ratio", values.get("overlap_w", 0.20)),
            overlap_height_ratio=values.get("overlap_height_ratio", values.get("overlap_h", 0.20)),
            postprocess_type=values.get("postprocess_type", "NMS"),
            postprocess_match_threshold=values.get("postprocess_match_threshold", values.get("merge_threshold", 0.50)),
            postprocess_match_metric=values.get("postprocess_match_metric", values.get("merge_metric", "IOU")),
            score_threshold=values.get("score_threshold", 0.0),
            max_tiles=values.get("max_tiles", 0),
        )

    def to_dict(self) -> dict:
        return {
            "slice_width": self.slice_width,
            "slice_height": self.slice_height,
            "overlap_width_ratio": self.overlap_width_ratio,
            "overlap_height_ratio": self.overlap_height_ratio,
            "postprocess_type": self.postprocess_type,
            "postprocess_match_threshold": self.postprocess_match_threshold,
            "postprocess_match_metric": self.postprocess_match_metric,
            "score_threshold": self.score_threshold,
            "max_tiles": self.max_tiles,
        }

    @property
    def merge_threshold(self) -> float:
        return self.postprocess_match_threshold

    @property
    def merge_metric(self) -> str:
        return self.postprocess_match_metric


@dataclass(frozen=True, slots=True)
class Tile:
    index: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class SAHIResult(list):
    """List-compatible result with tile/merge timing metadata."""

    def __init__(self, boxes: Iterable[BoundingBox] = (), **metadata) -> None:
        super().__init__(boxes)
        self.tile_count = int(metadata.get("tile_count", 0))
        self.raw_box_count = int(metadata.get("raw_box_count", len(self)))
        self.merged_box_count = int(metadata.get("merged_box_count", len(self)))
        self.yolo_time = float(metadata.get("yolo_time", 0.0))
        self.merge_time = float(metadata.get("merge_time", 0.0))
        self.total_time = float(metadata.get("total_time", 0.0))
        self.fallback = bool(metadata.get("fallback", False))
        self.error = metadata.get("error")

    @property
    def boxes(self) -> list[BoundingBox]:
        return list(self)

    def to_dict(self) -> dict:
        return {
            "enabled": not self.fallback,
            "sahi_enabled": not self.fallback,
            "tile_count": self.tile_count,
            "raw_box_count": self.raw_box_count,
            "merged_box_count": self.merged_box_count,
            "yolo_time": self.yolo_time,
            "merge_time": self.merge_time,
            "total_time": self.total_time,
            "fallback": self.fallback,
            "error": self.error,
        }


def _starts(length: int, window: int, overlap: float) -> list[int]:
    length, window = max(1, int(length)), max(1, int(window))
    if length <= window:
        return [0]
    stride = max(1, int(round(window * (1.0 - overlap))))
    values = list(range(0, max(1, length - window + 1), stride))
    last = length - window
    if values[-1] != last:
        values.append(last)
    return values


def generate_tiles(image_width: int, image_height: int, config: SAHIConfig | dict | None = None) -> list[Tile]:
    """Generate deterministic, gap-free tiles including right/bottom edges."""

    if int(image_width) <= 0 or int(image_height) <= 0:
        return []
    cfg = config if isinstance(config, SAHIConfig) else SAHIConfig.from_mapping(config)
    x_values = _starts(image_width, min(cfg.slice_width, image_width), cfg.overlap_width_ratio)
    y_values = _starts(image_height, min(cfg.slice_height, image_height), cfg.overlap_height_ratio)
    tiles: list[Tile] = []
    for y in y_values:
        for x in x_values:
            tiles.append(
                Tile(
                    index=len(tiles),
                    x1=x,
                    y1=y,
                    x2=min(image_width, x + cfg.slice_width),
                    y2=min(image_height, y + cfg.slice_height),
                )
            )
    if cfg.max_tiles and len(tiles) > cfg.max_tiles:
        # Keep first tiles for reproducibility, but always retain the final
        # right/bottom tile so edge objects are not systematically discarded.
        retained = tiles[: cfg.max_tiles]
        final = tiles[-1]
        if final not in retained:
            retained[-1] = final
        tiles = [Tile(index=i, x1=t.x1, y1=t.y1, x2=t.x2, y2=t.y2) for i, t in enumerate(retained)]
    return tiles


def map_box_to_image(box: BoundingBox, tile: Tile, image_size: tuple[int, int]) -> BoundingBox:
    """Map one tile-local box to original coordinates and clamp it."""

    width, height = image_size
    mapped = deepcopy(box)
    mapped.x1 = float(box.x1) + tile.x1
    mapped.y1 = float(box.y1) + tile.y1
    mapped.x2 = float(box.x2) + tile.x1
    mapped.y2 = float(box.y2) + tile.y1
    return mapped.normalized(width, height)


def _as_pil(image: Path | str | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as loaded:
            return loaded.convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        # A generic ndarray has no reliable colour-space tag.  Preserve its
        # channel order; callers with OpenCV BGR can convert explicitly before
        # passing it to the runner.
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    raise TypeError("image 必须是路径、PIL.Image 或 numpy.ndarray")


class SAHIInferenceRunner:
    """Run tiled inference through any detector exposing ``predict``."""

    def __init__(self, detector) -> None:
        self.detector = detector

    def _predict_tile(self, tile_image: Image.Image, tile: Tile, *, confidence: float, iou: float, imgsz: int, temp_dir: Path) -> list[BoundingBox]:
        # A detector may support in-memory images; prefer that to disk.  The
        # current Ultralytics wrapper accepts paths, so the fallback is a
        # short-lived PNG in a private temporary directory.
        source = tile_image
        try:
            boxes = self.detector.predict(source, confidence=confidence, iou=iou, imgsz=imgsz)
            return list(boxes or [])
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise
            tile_path = temp_dir / f"tile_{tile.index:05d}.png"
            tile_image.save(tile_path)
            boxes = self.detector.predict(tile_path, confidence=confidence, iou=iou, imgsz=imgsz)
            return list(boxes or [])

    def run(
        self,
        image: Path | str | Image.Image | np.ndarray,
        config: SAHIConfig | dict | None = None,
        *,
        confidence: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> SAHIResult:
        cfg = config if isinstance(config, SAHIConfig) else SAHIConfig.from_mapping(config)
        pil_image = _as_pil(image)
        width, height = pil_image.size
        tiles = generate_tiles(width, height, cfg)
        raw: list[BoundingBox] = []
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="yolo_world_sahi_") as temp_name:
            temp_dir = Path(temp_name)
            for tile in tiles:
                tile_image = pil_image.crop((tile.x1, tile.y1, tile.x2, tile.y2))
                for box in self._predict_tile(tile_image, tile, confidence=confidence, iou=iou, imgsz=imgsz, temp_dir=temp_dir):
                    try:
                        mapped = map_box_to_image(box, tile, (width, height))
                    except ValueError:
                        continue
                    mapped.inference_mode = "SAHI"
                    mapped.sahi_enabled = True
                    mapped.sahi_tile_index = tile.index
                    mapped.sahi_tile_count = len(tiles)
                    raw.append(mapped)
        yolo_time = time.perf_counter() - started
        merge_started = time.perf_counter()
        merged = merge_detections(
            raw,
            postprocess_type=cfg.postprocess_type,
            match_threshold=cfg.postprocess_match_threshold,
            match_metric=cfg.postprocess_match_metric,
            score_threshold=cfg.score_threshold,
        )
        merge_time = time.perf_counter() - merge_started
        for box in merged:
            box.inference_mode = "SAHI"
            box.sahi_enabled = True
            box.sahi_tile_count = len(tiles)
        return SAHIResult(
            merged,
            tile_count=len(tiles),
            raw_box_count=len(raw),
            merged_box_count=len(merged),
            yolo_time=yolo_time,
            merge_time=merge_time,
            total_time=time.perf_counter() - started,
        )

    def run_boxes(self, *args, **kwargs) -> list[BoundingBox]:
        """Convenience API returning only final boxes."""

        return self.run(*args, **kwargs).boxes


def run_sahi(detector, image, **kwargs) -> SAHIResult:
    return SAHIInferenceRunner(detector).run(image, **kwargs)


SahiRunner = SAHIInferenceRunner
TiledInferenceRunner = SAHIInferenceRunner


__all__ = [
    "SAHIConfig",
    "Tile",
    "SAHIResult",
    "SAHIInferenceRunner",
    "generate_tiles",
    "map_box_to_image",
    "run_sahi",
    "SahiRunner",
    "TiledInferenceRunner",
]
