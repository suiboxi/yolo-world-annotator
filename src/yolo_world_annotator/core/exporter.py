from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

from yolo_world_annotator.core.annotation import AnnotationStatus
from yolo_world_annotator.core.dataset import DatasetProject
from yolo_world_annotator.utils.config import atomic_write_text
from yolo_world_annotator.utils.image_utils import read_image


def export_yolo_dataset(
    project: DatasetProject,
    destination: Path,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("训练集比例必须位于 0~1 之间")
    destination = destination.resolve()
    if destination in (project.root, project.images_dir, project.labels_dir):
        raise ValueError("导出目录不能覆盖项目根目录或源 images 目录")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"导出目录非空：{destination}")

    candidates: list[Path] = []
    seen_stems: set[str] = set()
    for image_path in project.list_images():
        image = read_image(image_path)
        height, width = image.shape[:2]
        annotation = project.get_annotation(image_path, (width, height))
        if annotation.status == AnnotationStatus.UNLABELED and not project.label_path(image_path).exists():
            continue
        folded_stem = image_path.stem.casefold()
        if folded_stem in seen_stems:
            raise ValueError(f"存在同名不同扩展名图片，YOLO 标签会冲突：{image_path.stem}")
        seen_stems.add(folded_stem)
        candidates.append(image_path)
    if not candidates:
        raise ValueError("没有可导出的已标注图片")

    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        # Keep the YAML loadable for a one-image smoke-test project. Real datasets
        # should contain distinct validation images; the statistics page makes the
        # tiny project size visible before export.
        split_map = {"train": list(shuffled), "val": list(shuffled)}
    else:
        train_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * train_ratio)))
        split_map = {
            "train": shuffled[:train_count],
            "val": shuffled[train_count:],
        }

    for split, images in split_map.items():
        image_dir = destination / "images" / split
        label_dir = destination / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path in images:
            shutil.copy2(image_path, image_dir / image_path.name)
            label_path = project.label_path(image_path)
            target_label = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, target_label)
            else:
                atomic_write_text(target_label, "")

    data = {
        # Ultralytics resolves `path: .` against the process CWD in some versions.
        # An absolute POSIX-style Windows path keeps this YAML directly trainable
        # no matter which directory launches model.train().
        "path": destination.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(project.classes)},
    }
    atomic_write_text(
        destination / "data.yaml",
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
    )
    return {
        "destination": str(destination),
        "total": len(candidates),
        "train": len(split_map["train"]),
        "val": len(split_map["val"]),
    }
