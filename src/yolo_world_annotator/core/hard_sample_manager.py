"""Object-oriented facade for hard-sample persistence.

The original functional API remains available in :mod:`core.hard_samples`;
this facade gives benchmark/active-learning code a stable extension point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yolo_world_annotator.core.annotation import BoundingBox
from yolo_world_annotator.core.hard_samples import (
    append_hard_sample,
    load_hard_samples,
    record_auto_issues,
)


class HardSampleManager:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        return load_hard_samples(self.path)

    def append(self, *, image: Path, box: BoundingBox | None, error_type: str, **kwargs) -> dict[str, Any]:
        return append_hard_sample(self.path, image=image, box=box, error_type=error_type, **kwargs)

    def record_auto(self, image: Path, boxes: list[BoundingBox]) -> int:
        return record_auto_issues(self.path, image, boxes)

    record_auto_issues = record_auto


__all__ = ["HardSampleManager"]
