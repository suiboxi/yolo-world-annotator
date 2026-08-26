from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.annotation import BoundingBox


class BaseDetector(ABC):
    @abstractmethod
    def set_classes(self, classes: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, image: Path, *, confidence: float, iou: float, imgsz: int
    ) -> list[BoundingBox]:
        raise NotImplementedError

