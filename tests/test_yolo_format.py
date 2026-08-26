from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.annotation import AnnotationStatus, BoundingBox, ImageAnnotation
from core.dataset import DatasetProject
from core.yolo_format import parse_yolo, serialize_yolo, xyxy_to_yolo


def test_xyxy_yolo_round_trip() -> None:
    original = BoundingBox(2, "referee", 64, 48, 320, 300, 0.8, "YOLO-World")
    values = xyxy_to_yolo(original, 640, 480)
    text = f"2 {' '.join(str(value) for value in values)}\n"
    loaded = parse_yolo(text, 640, 480, ["player", "ball", "referee"])[0]
    assert (loaded.x1, loaded.y1, loaded.x2, loaded.y2) == pytest.approx(
        (64, 48, 320, 300), abs=1e-6
    )
    assert serialize_yolo([original], 640, 480).startswith("2 ")
    assert "0.800000" not in serialize_yolo([original], 640, 480)


@pytest.mark.parametrize(
    "text",
    ["9 0.5 0.5 0.2 0.2", "0 1.2 0.5 0.2 0.2", "0 0.5 0.5 -0.2 0.2", "bad"],
)
def test_invalid_yolo_is_rejected(text: str) -> None:
    with pytest.raises(ValueError):
        parse_yolo(text, 640, 480, ["person"])


def test_project_uses_in_place_labels_and_persists(tmp_path: Path) -> None:
    project = DatasetProject(tmp_path / "数据集")
    assert project.images_dir.is_dir()
    assert project.labels_dir.is_dir()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    image_path = project.images_dir / "中文样本.jpg"
    encoded.tofile(str(image_path))
    project.config["classes"] = ["person"]
    annotation = ImageAnnotation(
        AnnotationStatus.AUTO_LABELED,
        [BoundingBox(0, "person", 20, 10, 100, 70, 0.91, "YOLO-World")],
    )
    project.save_annotation(image_path, annotation, (200, 100))
    assert project.label_path(image_path) == image_path.with_suffix(".txt")
    assert project.label_path(image_path).parent == image_path.parent
    assert not (project.root / "images").exists()
    assert not (project.root / "labels").exists()
    assert project.label_path(image_path).read_text(encoding="utf-8").strip().split()[0] == "0"
    assert "0.91" not in project.label_path(image_path).read_text(encoding="utf-8")
    reopened = DatasetProject(project.root)
    loaded = reopened.get_annotation(image_path, (200, 100))
    assert loaded.status == AnnotationStatus.AUTO_LABELED
    assert loaded.objects[0].confidence == pytest.approx(0.91)
    assert (project.root / "classes.txt").read_text(encoding="utf-8").strip() == "person"
    assert (project.root / "project.json").is_file()
    assert (project.root / "annotations.json").is_file()


def test_newer_nonempty_txt_overrides_stale_empty_metadata(tmp_path: Path) -> None:
    project = DatasetProject(tmp_path / "stale")
    project.config["classes"] = ["raspberry"]
    image_path = project.root / "berry.jpg"
    image_path.touch()
    project.record_annotation(
        image_path, ImageAnnotation(AnnotationStatus.AUTO_LABELED, [])
    )
    project.save_metadata()
    label_path = image_path.with_suffix(".txt")
    label_path.write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")
    os.utime(project.annotations_path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(label_path, ns=(2_000_000_000, 2_000_000_000))

    reopened = DatasetProject(project.root)
    annotation = reopened.get_annotation(image_path, (200, 100))
    assert len(annotation.objects) == 1
    assert annotation.objects[0].class_name == "raspberry"
