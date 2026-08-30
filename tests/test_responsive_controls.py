from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication, QWidget

from yolo_world_annotator.app.annotator_window import (
    AnnotatorWindow,
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)


def _send_wheel(widget, delta: int = 120) -> None:
    event = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(widget, event)


def test_parameter_widgets_ignore_mouse_wheel(qapp) -> None:
    parent = QWidget()
    integer = NoWheelSpinBox(parent)
    integer.setRange(0, 100)
    integer.setValue(50)
    decimal = NoWheelDoubleSpinBox(parent)
    decimal.setRange(0, 1)
    decimal.setValue(0.5)
    combo = NoWheelComboBox(parent)
    combo.addItems(["A", "B", "C"])
    combo.setCurrentIndex(1)

    _send_wheel(integer)
    _send_wheel(decimal, -120)
    _send_wheel(combo)

    assert integer.value() == 50
    assert decimal.value() == 0.5
    assert combo.currentIndex() == 1
    # On Windows' offscreen Qt platform, processing deferred deletion
    # immediately after synthetic wheel events can access an already-freed
    # native event. Synchronous close keeps the same behavior assertion stable.
    parent.close()


def test_settings_panel_has_no_horizontal_overflow(qapp) -> None:
    window = AnnotatorWindow()
    window.resize(1100, 700)
    window.show()
    qapp.processEvents()

    assert window.settings_scroll.width() >= 320
    assert window.settings_scroll.horizontalScrollBar().maximum() == 0
    assert window.model_path_edit.minimumWidth() == 0
    window.close()
    qapp.processEvents()
