from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from core.annotation import BoundingBox
from core.verification import HUMAN_CONFIRMED
from utils.image_utils import bgr_to_qimage, read_image


PALETTE = ["#ff4f64", "#45d483", "#4ba3ff", "#d17bff", "#ffb84d", "#31d4d7"]


class DeleteBoxButtonItem(QGraphicsRectItem):
    """Fixed-screen-size scene button used to delete the selected box."""

    def __init__(self, canvas: "AnnotationCanvas") -> None:
        super().__init__(QRectF(0, 0, 82, 30))
        self.canvas = canvas
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(1000)
        self.setPen(QPen(QColor("#ff7a82"), 1))
        self.setBrush(QBrush(QColor("#d83b46")))
        self.setToolTip("删除当前选中的标注框，并立即保存 txt。")
        self.label = QGraphicsSimpleTextItem("删除此框", self)
        self.label.setFont(QFont("Microsoft YaHei UI", 10, QFont.Weight.DemiBold))
        self.label.setBrush(QBrush(QColor("white")))
        bounds = self.label.boundingRect()
        self.label.setPos((82 - bounds.width()) / 2, (30 - bounds.height()) / 2 - 1)

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        self.setBrush(QBrush(QColor("#f04b56")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self.setBrush(QBrush(QColor("#d83b46")))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.setBrush(QBrush(QColor("#ad2933")))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.setBrush(QBrush(QColor("#f04b56")))
            if self.rect().contains(event.pos()):
                self.canvas.delete_selected()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class EditableBoxItem(QGraphicsRectItem):
    """A scene-coordinate box supporting move and four-corner resize."""

    HANDLE_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")

    def __init__(
        self,
        canvas: "AnnotationCanvas",
        index: int,
        box: BoundingBox,
        review_threshold: float,
    ) -> None:
        super().__init__(QRectF(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1))
        self.canvas = canvas
        self.index = index
        self.box = deepcopy(box)
        self.review_threshold = review_threshold
        self._mode: str | None = None
        self._press_scene = QPointF()
        self._start_rect = QRectF()
        self.setFlags(
            QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsRectItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        self.label_item = QGraphicsSimpleTextItem(self)
        self.label_item.setFont(QFont("Microsoft YaHei UI", 11, QFont.Weight.DemiBold))
        self._update_style()

    def _scene_handle_size(self) -> float:
        scale = max(abs(self.canvas.transform().m11()), 1e-6)
        return min(30.0, max(3.0, 9.0 / scale))

    def _corner_points(self, rect: QRectF | None = None) -> dict[str, QPointF]:
        rect = rect or self.rect()
        return {
            "top_left": rect.topLeft(),
            "top_right": rect.topRight(),
            "bottom_left": rect.bottomLeft(),
            "bottom_right": rect.bottomRight(),
        }

    def _hit_handle(self, point: QPointF) -> str | None:
        radius = self._scene_handle_size() * 1.5
        for name, corner in self._corner_points().items():
            if abs(point.x() - corner.x()) <= radius and abs(point.y() - corner.y()) <= radius:
                return name
        return None

    def _update_style(self) -> None:
        color = QColor(PALETTE[self.box.class_id % len(PALETTE)])
        low_confidence = (
            self.box.confidence is not None and self.box.confidence < self.review_threshold
        )
        pen = QPen(QColor("#ffdf4d") if low_confidence else color, 3.0)
        if low_confidence:
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        label = self.box.class_name
        if self.box.confidence is not None:
            label += f"  {self.box.confidence:.2f}"
        self.label_item.setText(label)
        self.label_item.setBrush(QBrush(QColor("#ffdf4d") if low_confidence else color))
        self._position_label()

    def _position_label(self) -> None:
        rect = self.rect()
        label_y = rect.top() - 25 if rect.top() >= 25 else rect.top() + 2
        self.label_item.setPos(rect.left() + 2, label_y)

    def sync_box(self, box: BoundingBox, review_threshold: float) -> None:
        self.box = deepcopy(box)
        self.review_threshold = review_threshold
        self.setRect(QRectF(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1))
        self._update_style()
        if self.isSelected():
            self.canvas.position_delete_button(self)

    def hoverMoveEvent(self, event) -> None:  # noqa: N802
        handle = self._hit_handle(event.pos())
        cursors = {
            "top_left": Qt.CursorShape.SizeFDiagCursor,
            "bottom_right": Qt.CursorShape.SizeFDiagCursor,
            "top_right": Qt.CursorShape.SizeBDiagCursor,
            "bottom_left": Qt.CursorShape.SizeBDiagCursor,
        }
        self.setCursor(cursors.get(handle, Qt.CursorShape.SizeAllCursor))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSelected(True)
            self._mode = self._hit_handle(event.pos()) or "move"
            self._press_scene = event.scenePos()
            self._start_rect = QRectF(self.rect())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._mode:
            return super().mouseMoveEvent(event)
        delta = event.scenePos() - self._press_scene
        rect = QRectF(self._start_rect)
        if self._mode == "move":
            rect.translate(delta)
        elif self._mode == "top_left":
            rect.setTopLeft(rect.topLeft() + delta)
        elif self._mode == "top_right":
            rect.setTopRight(rect.topRight() + delta)
        elif self._mode == "bottom_left":
            rect.setBottomLeft(rect.bottomLeft() + delta)
        elif self._mode == "bottom_right":
            rect.setBottomRight(rect.bottomRight() + delta)
        rect = self.canvas.clamp_rect(rect, moving=self._mode == "move")
        self.setRect(rect)
        self._position_label()
        self.canvas.position_delete_button(self)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._mode and event.button() == Qt.MouseButton.LeftButton:
            changed = self.rect() != self._start_rect
            self._mode = None
            if changed:
                self.canvas.commit_item_rect(self.index, self.rect())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.canvas.change_class_requested.emit(self.index)
        event.accept()

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        if self.isSelected():
            size = self._scene_handle_size()
            painter.setPen(QPen(QColor("white"), max(1.0, size / 5)))
            painter.setBrush(QBrush(QColor("#1976d2")))
            for corner in self._corner_points().values():
                painter.drawRect(QRectF(corner.x() - size / 2, corner.y() - size / 2, size, size))


class AnnotationCanvas(QGraphicsView):
    load_failed = Signal(str)
    boxes_edited = Signal(object, int)
    new_box_requested = Signal(object)
    selection_changed = Signal(int)
    change_class_requested = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._scene.selectionChanged.connect(self._on_selection_changed)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_size = (0, 0)
        self._current_path: Path | None = None
        self._box_items: list[EditableBoxItem] = []
        self._boxes: list[BoundingBox] = []
        self._review_threshold = 0.5
        self._create_mode = False
        self._create_start: QPointF | None = None
        self._draft_item: QGraphicsRectItem | None = None
        self._delete_button_item: DeleteBoxButtonItem | None = None
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QBrush(QColor("#24262b")))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    @property
    def current_path(self) -> Path | None:
        return self._current_path

    @property
    def boxes(self) -> list[BoundingBox]:
        return deepcopy(self._boxes)

    @property
    def selected_index(self) -> int:
        for item in self._box_items:
            if item.isSelected():
                return item.index
        return -1

    def clear_image(self) -> None:
        self.cancel_create_mode()
        self._reset_delete_button()
        self._scene.clear()
        self._pixmap_item = None
        self._image_size = (0, 0)
        self._current_path = None
        self._box_items = []
        self._boxes = []

    def load_image(self, path: Path) -> bool:
        try:
            image = read_image(path)
        except Exception as exc:
            self.clear_image()
            self.load_failed.emit(str(exc))
            return False
        qimage = bgr_to_qimage(image)
        self.cancel_create_mode()
        self._reset_delete_button()
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(QPixmap.fromImage(qimage))
        self._pixmap_item.setZValue(-1000)
        height, width = image.shape[:2]
        self._image_size = (width, height)
        self._current_path = path
        self._box_items = []
        self._boxes = []
        self._scene.setSceneRect(0, 0, width, height)
        self.fit_image()
        return True

    def set_boxes(self, boxes: list[BoundingBox], review_threshold: float = 0.5) -> None:
        self._hide_delete_button()
        for item in self._box_items:
            self._scene.removeItem(item)
        self._box_items.clear()
        width, height = self._image_size
        clean: list[BoundingBox] = []
        for box in boxes:
            try:
                clean.append(box.normalized(width, height))
            except ValueError:
                continue
        self._boxes = deepcopy(clean)
        self._review_threshold = review_threshold
        for index, box in enumerate(clean):
            item = EditableBoxItem(self, index, box, review_threshold)
            self._scene.addItem(item)
            self._box_items.append(item)

    def clamp_rect(self, rect: QRectF, moving: bool = False) -> QRectF:
        width, height = self._image_size
        rect = rect.normalized()
        min_size = 2.0
        if moving:
            if rect.left() < 0:
                rect.moveLeft(0)
            if rect.top() < 0:
                rect.moveTop(0)
            if rect.right() > width:
                rect.moveRight(width)
            if rect.bottom() > height:
                rect.moveBottom(height)
        left = min(max(rect.left(), 0.0), max(0.0, width - min_size))
        top = min(max(rect.top(), 0.0), max(0.0, height - min_size))
        right = min(max(rect.right(), left + min_size), float(width))
        bottom = min(max(rect.bottom(), top + min_size), float(height))
        return QRectF(QPointF(left, top), QPointF(right, bottom))

    def commit_item_rect(self, index: int, rect: QRectF) -> None:
        if not 0 <= index < len(self._boxes):
            return
        old = self._boxes[index]
        clean = self.clamp_rect(rect)
        self._boxes[index] = BoundingBox(
            class_id=old.class_id,
            class_name=old.class_name,
            x1=clean.left(),
            y1=clean.top(),
            x2=clean.right(),
            y2=clean.bottom(),
            confidence=None,
            source="MANUAL",
            yolo_class_id=old.yolo_class_id,
            yolo_class_name=old.yolo_class_name,
            yolo_confidence=old.yolo_confidence,
            siglip_enabled=old.siglip_enabled,
            siglip_class_id=old.siglip_class_id,
            siglip_class_name=old.siglip_class_name,
            siglip_score=old.siglip_score,
            agreement=old.agreement,
            combined_confidence=old.combined_confidence,
            fusion_status=HUMAN_CONFIRMED,
            review_required=False,
            review_confirmed=True,
            human_modified=True,
            candidate_class_ids=old.candidate_class_ids,
            inference_mode=old.inference_mode,
            sahi_enabled=old.sahi_enabled,
            sahi_tile_count=old.sahi_tile_count,
            sahi_tile_index=old.sahi_tile_index,
            siglip_top2_class_id=old.siglip_top2_class_id,
            siglip_top2_class_name=old.siglip_top2_class_name,
            siglip_top2_score=old.siglip_top2_score,
            siglip_margin=old.siglip_margin,
            vlm_enabled=old.vlm_enabled,
            vlm_triggered=old.vlm_triggered,
            vlm_model=old.vlm_model,
            vlm_target_class=old.vlm_target_class,
            vlm_features=dict(old.vlm_features),
            vlm_final_result=old.vlm_final_result,
            vlm_confidence=old.vlm_confidence,
            vlm_parse_error=old.vlm_parse_error,
            decision_state="HUMAN_MODIFIED",
            decision_reason="人工修改拥有最高优先级",
        )
        self._box_items[index].sync_box(self._boxes[index], self._review_threshold)
        self.boxes_edited.emit(self.boxes, index)

    def delete_selected(self) -> bool:
        index = self.selected_index
        if index < 0:
            return False
        del self._boxes[index]
        self.set_boxes(self._boxes, self._review_threshold)
        self.boxes_edited.emit(self.boxes, -1)
        return True

    def _ensure_delete_button(self) -> None:
        if self._delete_button_item is not None:
            return
        button = DeleteBoxButtonItem(self)
        self._scene.addItem(button)
        button.hide()
        self._delete_button_item = button

    def _reset_delete_button(self) -> None:
        self._delete_button_item = None

    def _hide_delete_button(self) -> None:
        if self._delete_button_item is not None:
            self._delete_button_item.hide()

    def position_delete_button(self, item: EditableBoxItem | None = None) -> None:
        if item is None:
            index = self.selected_index
            item = self._box_items[index] if 0 <= index < len(self._box_items) else None
        if item is None or not item.isSelected() or item.scene() is None:
            self._hide_delete_button()
            return

        self._ensure_delete_button()
        if self._delete_button_item is None:
            return
        rect = item.sceneBoundingRect().normalized()
        width, height = self._image_size
        scale = max(abs(self.transform().m11()), 1e-6)
        button_width = 82.0 / scale
        button_height = 30.0 / scale
        gap = 7.0 / scale

        # Prefer the outside-right side. If the frame touches the image edge,
        # flip to the left, then above/below. This keeps all four resize handles clear.
        if rect.right() + gap + button_width <= width:
            x = rect.right() + gap
            y = min(max(rect.top(), 0.0), max(0.0, height - button_height))
        elif rect.left() - gap - button_width >= 0:
            x = rect.left() - gap - button_width
            y = min(max(rect.top(), 0.0), max(0.0, height - button_height))
        elif rect.top() - gap - button_height >= 0:
            x = min(max(rect.center().x() - button_width / 2, 0.0), max(0.0, width - button_width))
            y = rect.top() - gap - button_height
        else:
            x = min(max(rect.center().x() - button_width / 2, 0.0), max(0.0, width - button_width))
            y = min(rect.bottom() + gap, max(0.0, height - button_height))
        self._delete_button_item.setPos(x, y)
        self._delete_button_item.show()

    def _editable_box_at(self, viewport_point) -> EditableBoxItem | None:
        scene_point = self.mapToScene(viewport_point)
        for scene_item in self._scene.items(scene_point):
            current = scene_item
            while current is not None:
                if isinstance(current, EditableBoxItem):
                    return current
                current = current.parentItem()
        return None

    def _delete_button_at(self, viewport_point) -> bool:
        if self._delete_button_item is None or not self._delete_button_item.isVisible():
            return False
        scene_item = self.itemAt(viewport_point)
        while scene_item is not None:
            if scene_item is self._delete_button_item:
                return True
            scene_item = scene_item.parentItem()
        return False

    def set_box_class(self, index: int, class_id: int, class_name: str) -> None:
        if not 0 <= index < len(self._boxes):
            return
        old = self._boxes[index]
        self._boxes[index] = BoundingBox(
            class_id=class_id,
            class_name=class_name,
            x1=old.x1,
            y1=old.y1,
            x2=old.x2,
            y2=old.y2,
            confidence=None,
            source="MANUAL",
            yolo_class_id=old.yolo_class_id,
            yolo_class_name=old.yolo_class_name,
            yolo_confidence=old.yolo_confidence,
            siglip_enabled=old.siglip_enabled,
            siglip_class_id=old.siglip_class_id,
            siglip_class_name=old.siglip_class_name,
            siglip_score=old.siglip_score,
            agreement=old.agreement,
            combined_confidence=old.combined_confidence,
            fusion_status=HUMAN_CONFIRMED,
            review_required=False,
            review_confirmed=True,
            human_modified=True,
            candidate_class_ids=old.candidate_class_ids,
            inference_mode=old.inference_mode,
            sahi_enabled=old.sahi_enabled,
            sahi_tile_count=old.sahi_tile_count,
            sahi_tile_index=old.sahi_tile_index,
            siglip_top2_class_id=old.siglip_top2_class_id,
            siglip_top2_class_name=old.siglip_top2_class_name,
            siglip_top2_score=old.siglip_top2_score,
            siglip_margin=old.siglip_margin,
            vlm_enabled=old.vlm_enabled,
            vlm_triggered=old.vlm_triggered,
            vlm_model=old.vlm_model,
            vlm_target_class=old.vlm_target_class,
            vlm_features=dict(old.vlm_features),
            vlm_final_result=old.vlm_final_result,
            vlm_confidence=old.vlm_confidence,
            vlm_parse_error=old.vlm_parse_error,
            decision_state="HUMAN_MODIFIED",
            decision_reason="人工修改拥有最高优先级",
        )
        self._box_items[index].sync_box(self._boxes[index], self._review_threshold)
        self.boxes_edited.emit(self.boxes, index)

    def add_box(self, box: BoundingBox) -> None:
        width, height = self._image_size
        self._boxes.append(box.normalized(width, height))
        self.set_boxes(self._boxes, self._review_threshold)
        self._box_items[-1].setSelected(True)
        self.boxes_edited.emit(self.boxes, len(self._boxes) - 1)

    def start_create_mode(self) -> None:
        """Compatibility entry point; direct drawing no longer needs a mode."""
        if self._pixmap_item is None:
            return
        self._create_mode = True
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def cancel_create_mode(self) -> None:
        self._create_mode = False
        self._create_start = None
        if self._draft_item is not None and self._draft_item.scene() is not None:
            self._scene.removeItem(self._draft_item)
        self._draft_item = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.unsetCursor()

    def fit_image(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.position_delete_button()

    def viewport_to_image(self, point) -> QPointF:
        scene_point = self.mapToScene(point)
        width, height = self._image_size
        return QPointF(
            min(max(scene_point.x(), 0.0), float(width)),
            min(max(scene_point.y(), 0.0), float(height)),
        )

    def _on_selection_changed(self) -> None:
        self.position_delete_button()
        self.selection_changed.emit(self.selected_index)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.position().toPoint()
            if self._delete_button_at(point) or self._editable_box_at(point) is not None:
                super().mousePressEvent(event)
                return
            scene_point = self.mapToScene(point)
            image_rect = QRectF(0, 0, self._image_size[0], self._image_size[1])
            if self._pixmap_item is None or not image_rect.contains(scene_point):
                super().mousePressEvent(event)
                return
            self._scene.clearSelection()
            self._create_mode = True
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
            self._create_start = self.viewport_to_image(event.position().toPoint())
            self._draft_item = self._scene.addRect(
                QRectF(self._create_start, self._create_start),
                QPen(QColor("#ffffff"), 2, Qt.PenStyle.DashLine),
            )
            self._draft_item.setZValue(50)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._create_mode and self._create_start is not None and self._draft_item is not None:
            current = self.viewport_to_image(event.position().toPoint())
            self._draft_item.setRect(self.clamp_rect(QRectF(self._create_start, current)))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._create_mode and self._create_start is not None and event.button() == Qt.MouseButton.LeftButton:
            current = self.viewport_to_image(event.position().toPoint())
            raw_rect = QRectF(self._create_start, current).normalized()
            rect = self.clamp_rect(raw_rect)
            self.cancel_create_mode()
            if raw_rect.width() >= 2 and raw_rect.height() >= 2:
                self.new_box_requested.emit(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap_item is not None:
            self.fit_image()
            self.position_delete_button()

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._pixmap_item is None:
            return super().wheelEvent(event)
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        current_scale = self.transform().m11()
        target = current_scale * factor
        if 0.03 <= target <= 30:
            self.scale(factor, factor)
            self.position_delete_button()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Delete and self.delete_selected():
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_create_mode()
            event.accept()
            return
        super().keyPressEvent(event)


class SegmentationPreview(QGraphicsView):
    """Read-only preview of the image partition used before detection.

    The current pipeline performs tiled (SAHI) partitioning rather than
    producing pixel masks.  This view intentionally visualises those tiles so
    an operator can verify coverage, overlap and edge handling before trusting
    the YOLO/Qwen results.
    """

    load_failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._tile_items: list[tuple[QGraphicsRectItem, QGraphicsSimpleTextItem]] = []
        self._image_size = (0, 0)
        self._current_path: Path | None = None
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QBrush(QColor("#1d2024")))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    @property
    def current_path(self) -> Path | None:
        return self._current_path

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def clear_image(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._tile_items.clear()
        self._image_size = (0, 0)
        self._current_path = None

    def load_image(self, path: Path) -> bool:
        try:
            image = read_image(path)
        except Exception as exc:
            self.clear_image()
            self.load_failed.emit(str(exc))
            return False
        qimage = bgr_to_qimage(image)
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(QPixmap.fromImage(qimage))
        self._pixmap_item.setZValue(-1000)
        height, width = image.shape[:2]
        self._image_size = (width, height)
        self._current_path = path.resolve()
        self._tile_items.clear()
        self._scene.setSceneRect(0, 0, width, height)
        self.fit_image()
        return True

    def set_tiles(self, tiles: list[dict] | None) -> None:
        for rect_item, text_item in self._tile_items:
            self._scene.removeItem(rect_item)
            self._scene.removeItem(text_item)
        self._tile_items.clear()
        width, height = self._image_size
        if width <= 0 or height <= 0:
            return
        colors = ("#4ba3ff", "#45d483", "#ffb84d", "#d17bff", "#ff4f64")
        for fallback_index, tile in enumerate(tiles or []):
            if not isinstance(tile, dict):
                continue
            try:
                x1 = min(max(float(tile.get("x1", 0)), 0.0), float(width))
                y1 = min(max(float(tile.get("y1", 0)), 0.0), float(height))
                x2 = min(max(float(tile.get("x2", width)), 0.0), float(width))
                y2 = min(max(float(tile.get("y2", height)), 0.0), float(height))
                rect = QRectF(QPointF(min(x1, x2), min(y1, y2)), QPointF(max(x1, x2), max(y1, y2)))
                if rect.width() <= 0 or rect.height() <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            index = int(tile.get("index", fallback_index))
            color = QColor(colors[index % len(colors)])
            pen = QPen(color, 3.0)
            pen.setStyle(Qt.PenStyle.DashLine)
            color.setAlpha(35)
            rect_item = self._scene.addRect(rect, pen, QBrush(color))
            rect_item.setZValue(5)
            text_item = self._scene.addSimpleText(f"Tile {index + 1}")
            text_item.setBrush(QBrush(QColor("#ffffff")))
            text_item.setFont(QFont("Microsoft YaHei UI", 10, QFont.Weight.DemiBold))
            text_item.setPos(rect.left() + 4, rect.top() + 4)
            text_item.setZValue(6)
            self._tile_items.append((rect_item, text_item))

    def fit_image(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap_item is not None:
            self.fit_image()
