from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from core.annotation import AnnotationStatus, BoundingBox, ImageAnnotation
from core.dataset import DatasetProject
from core.exporter import export_yolo_dataset
from core.statistics import collect_statistics


def _make_project(root: Path, count: int = 5) -> DatasetProject:
    project = DatasetProject(root)
    project.config["classes"] = ["player", "ball"]
    for index in range(count):
        image = np.full((100, 200, 3), index * 20, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        path = project.images_dir / f"sample_{index}.jpg"
        encoded.tofile(str(path))
        status = AnnotationStatus.VERIFIED if index < 2 else AnnotationStatus.AUTO_LABELED
        objects = [BoundingBox(index % 2, project.classes[index % 2], 20, 10, 100, 80)]
        project.save_annotation(path, ImageAnnotation(status, objects), (200, 100))
    return project


def test_statistics_counts_statuses_and_classes(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "项目")
    stats = collect_statistics(project)
    assert stats["images"] == 5
    assert stats["verified"] == 2
    assert stats["auto_labeled"] == 3
    assert stats["objects"] == 5
    assert stats["classes"] == {"player": 3, "ball": 2}


def test_export_creates_train_val_and_portable_yaml(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "项目")
    destination = tmp_path / "项目" / "dataset_export"
    result = export_yolo_dataset(project, destination, train_ratio=0.8, seed=7)
    assert result == {
        "destination": str(destination.resolve()),
        "total": 5,
        "train": 4,
        "val": 1,
    }
    assert len(list((destination / "images" / "train").glob("*.jpg"))) == 4
    assert len(list((destination / "labels" / "train").glob("*.txt"))) == 4
    assert len(list((destination / "images" / "val").glob("*.jpg"))) == 1
    data = yaml.safe_load((destination / "data.yaml").read_text(encoding="utf-8"))
    assert data["path"] == destination.resolve().as_posix()
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["names"] == {0: "player", 1: "ball"}


def test_one_image_export_keeps_train_and_val_loadable(tmp_path: Path) -> None:
    project = _make_project(tmp_path / "single", count=1)
    destination = project.root / "dataset_export"
    result = export_yolo_dataset(project, destination)
    assert result["total"] == 1
    assert result["train"] == 1
    assert result["val"] == 1
    assert len(list((destination / "images" / "train").glob("*.jpg"))) == 1
    assert len(list((destination / "images" / "val").glob("*.jpg"))) == 1
