from __future__ import annotations

from copy import deepcopy

from yolo_world_annotator.core.annotation import BoundingBox


class HistoryManager:
    """Per-image bounded snapshot history; a drag gesture is committed once."""

    def __init__(self, capacity: int = 20) -> None:
        if capacity < 1:
            raise ValueError("撤销容量必须至少为 1")
        self.capacity = capacity
        self._past: list[list[BoundingBox]] = []
        self._future: list[list[BoundingBox]] = []

    def reset(self, boxes: list[BoundingBox]) -> None:
        self._past = [deepcopy(boxes)]
        self._future.clear()

    def push(self, boxes: list[BoundingBox]) -> None:
        snapshot = deepcopy(boxes)
        if self._past and self._past[-1] == snapshot:
            return
        self._past.append(snapshot)
        if len(self._past) > self.capacity + 1:
            self._past.pop(0)
        self._future.clear()

    def undo(self) -> list[BoundingBox] | None:
        if len(self._past) <= 1:
            return None
        self._future.append(self._past.pop())
        return deepcopy(self._past[-1])

    def redo(self) -> list[BoundingBox] | None:
        if not self._future:
            return None
        state = self._future.pop()
        self._past.append(deepcopy(state))
        return deepcopy(state)

    @property
    def undo_count(self) -> int:
        return max(0, len(self._past) - 1)

    @property
    def initialized(self) -> bool:
        return bool(self._past)
