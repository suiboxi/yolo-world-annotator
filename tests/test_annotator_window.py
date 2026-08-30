from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PySide6.QtCore import QRectF

from yolo_world_annotator.app.annotator_window import AnnotatorWindow
from yolo_world_annotator.core.annotation import BoundingBox


def _write_image(path: Path) -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    encoded.tofile(str(path))


def test_window_edits_are_saved_next_to_image(qapp, tmp_path: Path) -> None:
    image_path = tmp_path / "中文图片.jpg"
    _write_image(image_path)
    window = AnnotatorWindow()
    window.open_folder(tmp_path, select_path=image_path)
    window.classes_edit.setPlainText("person\ncar")
    window.prompts_edit.setPlainText("person\ncar")
    window.save_classes()

    window.canvas.add_box(BoundingBox(0, "person", 10, 10, 80, 70))
    label_path = image_path.with_suffix(".txt")
    assert label_path.is_file()
    assert label_path.read_text(encoding="utf-8").startswith("0 ")
    assert not (tmp_path / "labels").exists()

    window.canvas.commit_item_rect(0, QRectF(20, 15, 100, 60))
    moved = label_path.read_text(encoding="utf-8")
    assert moved != ""

    window.selected_class_combo.setCurrentIndex(1)
    window.canvas.set_box_class(0, 1, "car")
    assert label_path.read_text(encoding="utf-8").startswith("1 ")

    window.canvas._box_items[0].setSelected(True)
    assert window.canvas.delete_selected()
    assert label_path.read_text(encoding="utf-8") == ""
    window.close()
    qapp.processEvents()


def test_window_keeps_inference_enabled_when_auto_selects_cpu(
    qapp, monkeypatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    window = AnnotatorWindow()
    try:
        assert window.device_combo.currentData() == "auto"
        assert window.auto_current_button.isEnabled()
        assert window.auto_all_button.isEnabled()
        assert "CPU" in window.device_value.text()
    finally:
        window.close()
        qapp.processEvents()


def test_window_lists_every_detected_cuda_device(qapp, monkeypatch) -> None:
    monkeypatch.setenv("YOLO_WORLD_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    window = AnnotatorWindow()
    try:
        assert window.device_combo.findData("cuda:0") >= 0
        assert window.device_combo.findData("cuda:1") >= 0
        assert window.device_combo.findData("cuda:2") >= 0
    finally:
        window.close()
        qapp.processEvents()
