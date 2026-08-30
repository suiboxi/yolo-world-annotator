from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from yolo_world_annotator.core.annotation import BoundingBox


class BaseDetector(ABC):
    @abstractmethod
    def predict(
        self, image: Path, *, confidence: float, iou: float, imgsz: int
    ) -> list[BoundingBox]:
        raise NotImplementedError
