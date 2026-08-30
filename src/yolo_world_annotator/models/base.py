from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.annotation import BoundingBox


class BaseDetector(ABC):
    @abstractmethod
    def predict(
        self, image: Path, *, confidence: float, iou: float, imgsz: int
    ) -> list[BoundingBox]:
        raise NotImplementedError
