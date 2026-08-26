from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox

from app.canvas import AnnotationCanvas
from app.main_window import MainWindow
from core.dataset import DatasetProject
from core.annotation import BoundingBox
from utils.image_utils import discover_images, read_image


def _write_unicode_image(path: Path, width: int = 320, height: int = 200) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    ok, encoded = cv2.imencode(path.suffix, image)
    assert ok
    encoded.tofile(str(path))


def test_unicode_image_discovery_and_read(tmp_path: Path) -> None:
    image_path = tmp_path / "中文图片.jpg"
    _write_unicode_image(image_path)
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    assert discover_images(tmp_path) == [image_path]
    assert read_image(image_path).shape == (200, 320, 3)


def test_phase1_window_loads_folder(qapp, tmp_path: Path) -> None:
    image_path = tmp_path / "测试图像.png"
    _write_unicode_image(image_path, 640, 480)
    window = MainWindow()
    try:
        window.open_image_folder(tmp_path)
        assert window.image_list.count() == 1
        assert window.current_index == 0
        assert window.canvas.image_size == (640, 480)
    finally:
        window.close()


def test_canvas_clamps_viewport_coordinates(qapp) -> None:
    canvas = AnnotationCanvas()
    canvas._image_size = (100, 80)
    point = canvas.viewport_to_image(canvas.viewport().rect().topLeft())
    assert 0 <= point.x() <= 100
    assert 0 <= point.y() <= 80


def test_invalid_existing_label_is_write_protected(qapp, tmp_path: Path, monkeypatch) -> None:
    project = DatasetProject(tmp_path / "受保护项目")
    project.config["classes"] = ["person"]
    project.save_metadata()
    image_path = project.images_dir / "bad_label.jpg"
    _write_unicode_image(image_path)
    label_path = project.label_path(image_path)
    original = "this is not a valid YOLO label\n"
    label_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Cancel),
    )
    window = MainWindow()
    try:
        window.open_project(project.images_dir)
        assert image_path.resolve() in window.label_load_failed_paths
        assert window.save_current() is False
        assert label_path.read_text(encoding="utf-8") == original
    finally:
        window.close()


def test_stale_inference_result_cannot_write_into_new_project(qapp, tmp_path: Path) -> None:
    first = DatasetProject(tmp_path / "first")
    second = DatasetProject(tmp_path / "second")
    first_image = first.images_dir / "same_name.jpg"
    second_image = second.images_dir / "same_name.jpg"
    _write_unicode_image(first_image)
    _write_unicode_image(second_image)
    window = MainWindow()
    try:
        window.open_project(first.root)
        window._pending_single_root = first.root
        window._pending_single_path = first_image.resolve()
        window.open_project(second.root)
        window._on_prediction_ready(
            str(first_image), [BoundingBox(0, "person", 10, 10, 50, 60, 0.9, "YOLO-World")]
        )
        assert not second.label_path(second_image).exists()
    finally:
        window.close()
