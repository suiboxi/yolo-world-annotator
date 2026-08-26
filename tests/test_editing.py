from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtTest import QTest

from app.canvas import AnnotationCanvas
from core.annotation import BoundingBox
from core.history import HistoryManager


def _box(x1=10, y1=20, x2=110, y2=120) -> BoundingBox:
    return BoundingBox(0, "person", x1, y1, x2, y2, 0.8, "YOLO-World")


def test_canvas_clamps_and_commits_manual_edit(qapp) -> None:
    canvas = AnnotationCanvas()
    canvas._image_size = (200, 150)
    canvas.set_boxes([_box()])
    canvas.commit_item_rect(0, QRectF(-30, -40, 500, 300))
    edited = canvas.boxes[0]
    assert (edited.x1, edited.y1, edited.x2, edited.y2) == (0, 0, 200, 150)
    assert edited.source == "MANUAL"
    assert edited.confidence is None


def test_canvas_add_delete_and_class_change(qapp) -> None:
    canvas = AnnotationCanvas()
    canvas._image_size = (300, 200)
    canvas.set_boxes([_box()])
    canvas.set_box_class(0, 1, "bus")
    assert canvas.boxes[0].class_name == "bus"
    canvas.add_box(BoundingBox(0, "person", 150, 100, 250, 190))
    assert len(canvas.boxes) == 2
    canvas._box_items[-1].setSelected(True)
    assert canvas.delete_selected()
    assert len(canvas.boxes) == 1


def test_direct_drag_creates_box_and_popup_button_deletes_it(qapp) -> None:
    canvas = AnnotationCanvas()
    canvas.resize(700, 500)
    pixmap = QPixmap(400, 300)
    pixmap.fill(QColor("white"))
    canvas._pixmap_item = canvas._scene.addPixmap(pixmap)
    canvas._pixmap_item.setZValue(-1000)
    canvas._image_size = (400, 300)
    canvas._scene.setSceneRect(0, 0, 400, 300)
    canvas.show()
    canvas.fit_image()
    qapp.processEvents()

    requested: list[QRectF] = []
    canvas.new_box_requested.connect(requested.append)
    start = canvas.mapFromScene(QPointF(40, 50))
    end = canvas.mapFromScene(QPointF(180, 170))
    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(canvas.viewport(), end, 20)
    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)
    qapp.processEvents()

    assert len(requested) == 1
    rect = requested[0]
    assert abs(rect.left() - 40) < 1
    assert abs(rect.top() - 50) < 1
    assert abs(rect.right() - 180) < 1
    assert abs(rect.bottom() - 170) < 1

    canvas.add_box(BoundingBox(0, "person", rect.left(), rect.top(), rect.right(), rect.bottom()))
    center = canvas.mapFromScene(rect.center())
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=center)
    qapp.processEvents()

    button = canvas._delete_button_item
    assert canvas.selected_index == 0
    assert button is not None and button.isVisible()
    button_center = canvas.mapFromScene(button.mapToScene(button.rect().center()))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=button_center)
    qapp.processEvents()
    assert canvas.boxes == []

    canvas.close()
    qapp.processEvents()


def test_plain_click_does_not_create_tiny_box(qapp) -> None:
    canvas = AnnotationCanvas()
    canvas.resize(500, 400)
    pixmap = QPixmap(300, 200)
    pixmap.fill(QColor("white"))
    canvas._pixmap_item = canvas._scene.addPixmap(pixmap)
    canvas._pixmap_item.setZValue(-1000)
    canvas._image_size = (300, 200)
    canvas._scene.setSceneRect(0, 0, 300, 200)
    canvas.show()
    canvas.fit_image()
    qapp.processEvents()

    requested: list[QRectF] = []
    canvas.new_box_requested.connect(requested.append)
    point = canvas.mapFromScene(QPointF(100, 100))
    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=point)
    qapp.processEvents()
    assert requested == []

    canvas.close()
    qapp.processEvents()


def test_history_keeps_twenty_operations() -> None:
    history = HistoryManager(capacity=20)
    history.reset([_box()])
    for index in range(25):
        history.push([_box(x1=index + 11)])
    assert history.undo_count == 20
    last = None
    for _ in range(20):
        last = history.undo()
    assert last is not None
    assert history.undo() is None


def test_history_new_edit_clears_redo() -> None:
    history = HistoryManager()
    history.reset([_box()])
    history.push([_box(x1=30)])
    assert history.undo() is not None
    history.push([_box(x1=40)])
    assert history.redo() is None
