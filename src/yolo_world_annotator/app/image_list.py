from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem

STATUS_PREFIX = {
    "UNLABELED": "○",
    "AUTO_LABELED": "●",
    "VERIFIED": "✓",
}


class ImageListWidget(QListWidget):
    image_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(210)
        self._all_images: list[Path] = []
        self._statuses: dict[str, str] = {}
        self._filter = None
        self.currentRowChanged.connect(self.image_selected)

    def set_images(self, images: list[Path], statuses: dict[str, str] | None = None) -> None:
        self._all_images = list(images)
        self._statuses = dict(statuses or {})
        self._rebuild()

    @property
    def visible_images(self) -> list[Path]:
        return [Path(self.item(index).data(Qt.ItemDataRole.UserRole)) for index in range(self.count())]

    def path_at(self, row: int) -> Path | None:
        item = self.item(row)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return Path(value) if value else None

    def row_for_path(self, path: Path) -> int:
        resolved = path.resolve()
        for row in range(self.count()):
            candidate = self.path_at(row)
            if candidate is not None and candidate.resolve() == resolved:
                return row
        return -1

    def set_filter(self, predicate) -> None:
        """Filter by a callable ``predicate(path, status) -> bool``."""

        self._filter = predicate
        self._rebuild()

    def _rebuild(self) -> None:
        statuses = self._statuses
        images = [
            path
            for path in self._all_images
            if self._filter is None or self._filter(path, statuses.get(path.name, "UNLABELED"))
        ]
        self.clear()
        for path in images:
            status = statuses.get(path.name, "UNLABELED")
            item = QListWidgetItem(f"{STATUS_PREFIX.get(status, '○')}  {path.name}")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.addItem(item)
        if images:
            self.setCurrentRow(0)

    def update_status(self, row: int, status: str) -> None:
        item = self.item(row)
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        self._statuses[path.name] = status
        item.setText(f"{STATUS_PREFIX.get(status, '○')}  {path.name}")
