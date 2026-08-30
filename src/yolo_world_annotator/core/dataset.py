from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from yolo_world_annotator.core.annotation import AnnotationStatus, ImageAnnotation
from yolo_world_annotator.core.class_profiles import ClassProfiles, ensure_class_profiles
from yolo_world_annotator.core.yolo_format import load_yolo, serialize_yolo
from yolo_world_annotator.utils.config import (
    DEFAULT_CONFIG,
    atomic_write_json,
    atomic_write_text,
    load_json,
)
from yolo_world_annotator.utils.image_utils import discover_images, read_image

LOGGER = logging.getLogger(__name__)


class DatasetProject:
    """Owns an in-place image folder and its sidecar YOLO labels.

    The selected folder is the project.  Images and ``.txt`` labels always
    live next to each other; no ``images`` or ``labels`` directory is created.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError(f"图片文件夹不存在：{self.root}")
        # Keep these aliases for callers that display the active folder, but
        # both deliberately point at the user-selected directory.
        self.images_dir = self.root
        self.labels_dir = self.root
        self.classes_path = self.root / "classes.txt"
        self.project_path = self.root / "project.json"
        self.annotations_path = self.root / "annotations.json"
        self.hard_samples_path = self.root / "hard_samples.json"
        self.class_profiles_path = self.root / "class_profiles.json"
        loaded_config = load_json(self.project_path, {})
        self.config = deepcopy(DEFAULT_CONFIG)
        if isinstance(loaded_config, dict):
            self.config.update(loaded_config)
        if self.classes_path.exists():
            classes = [line.strip() for line in self.classes_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if classes:
                self.config["classes"] = classes
        # Rich confidence metadata is optional; txt files are canonical.  A
        # very large legacy annotations.json can otherwise freeze startup or
        # exhaust RAM, so large projects rebuild visible boxes from txt.
        metadata_limit = 16 * 1024 * 1024
        if self.annotations_path.exists() and self.annotations_path.stat().st_size > metadata_limit:
            LOGGER.warning(
                "annotations.json 超过 16 MB，已跳过详细元数据并改从 txt 按需加载"
            )
            raw_annotations = {}
        else:
            raw_annotations = load_json(self.annotations_path, {})
        self.annotations: dict[str, ImageAnnotation] = {}
        self._unparsed_annotations: dict[str, Any] = {}
        if isinstance(raw_annotations, dict):
            for name, data in raw_annotations.items():
                try:
                    self.annotations[name] = ImageAnnotation.from_dict(data)
                except (KeyError, TypeError, ValueError) as exc:
                    LOGGER.error("跳过损坏的 annotations.json 条目 %s: %s", name, exc)
                    # Preserve malformed entries instead of silently deleting
                    # them the next time an otherwise unrelated annotation is
                    # saved.  A user can repair/remove them explicitly later.
                    self._unparsed_annotations[str(name)] = data
        self.class_profiles: ClassProfiles = ensure_class_profiles(
            self.class_profiles_path, self.classes
        )

    @property
    def classes(self) -> list[str]:
        return list(self.config.get("classes", []))

    def list_images(self) -> list[Path]:
        return discover_images(self.images_dir)

    def label_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(".txt")

    def _annotation_key(self, image_path: Path) -> str:
        try:
            return image_path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return image_path.name

    def status_map(self) -> dict[str, str]:
        return {name: annotation.status.value for name, annotation in self.annotations.items()}

    def record_annotation(self, image_path: Path, annotation: ImageAnnotation) -> None:
        """Update in-memory metadata after a worker has committed the txt."""

        self.annotations[self._annotation_key(image_path)] = annotation

    def discard_annotation(self, image_path: Path) -> None:
        """Drop stale rich metadata so the newly written txt is canonical."""

        self.annotations.pop(self._annotation_key(image_path), None)

    def get_annotation(self, image_path: Path, image_size: tuple[int, int]) -> ImageAnnotation:
        key = self._annotation_key(image_path)
        existing = self.annotations.get(key)
        label_path = self.label_path(image_path)
        if existing is not None:
            # A worker or external YOLO tool may have committed a newer txt
            # after annotations.json was written.  The txt is canonical; avoid
            # showing stale zero-box metadata over a valid non-empty label.
            metadata_mtime = (
                self.annotations_path.stat().st_mtime_ns
                if self.annotations_path.exists()
                else 0
            )
            if (
                label_path.exists()
                and label_path.stat().st_size > 0
                and label_path.stat().st_mtime_ns > metadata_mtime
            ):
                width, height = image_size
                refreshed = ImageAnnotation(
                    existing.status,
                    load_yolo(label_path, width, height, self.classes),
                    existing.verification,
                )
                self.annotations[key] = refreshed
                return refreshed
            return existing
        if label_path.exists():
            width, height = image_size
            boxes = load_yolo(label_path, width, height, self.classes)
            annotation = ImageAnnotation(AnnotationStatus.AUTO_LABELED, boxes)
        else:
            annotation = ImageAnnotation()
        self.annotations[key] = annotation
        return annotation

    def save_annotation(
        self,
        image_path: Path,
        annotation: ImageAnnotation,
        image_size: tuple[int, int],
        *,
        save_metadata: bool = True,
    ) -> None:
        width, height = image_size
        clean_boxes = [box.normalized(width, height) for box in annotation.objects]
        annotation.objects = clean_boxes
        # The editable canvas and YOLO txt must never disagree.  Confidence or
        # review metadata may change how a box is drawn, but every visible box
        # is a trainable label until the user explicitly deletes it.
        serialized = serialize_yolo(clean_boxes, width, height)
        written_lines = [line for line in serialized.splitlines() if line.strip()]
        if len(written_lines) != len(clean_boxes):
            raise RuntimeError("标注序列化行数与预览框数不一致，已拒绝保存。")
        atomic_write_text(self.label_path(image_path), serialized)
        self.annotations[self._annotation_key(image_path)] = annotation
        if save_metadata:
            self.save_metadata()

    def save_metadata(self) -> None:
        self.class_profiles = self.class_profiles.sync_classes(self.classes)
        from yolo_world_annotator.core.class_profiles import save_class_profiles

        save_class_profiles(self.class_profiles_path, self.class_profiles)
        atomic_write_text(self.classes_path, "\n".join(self.classes) + ("\n" if self.classes else ""))
        atomic_write_json(self.project_path, self.config)
        persisted = dict(self._unparsed_annotations)
        persisted.update(
            {name: annotation.to_dict() for name, annotation in sorted(self.annotations.items())}
        )
        atomic_write_json(self.annotations_path, persisted)

    def update_config(self, values: dict) -> None:
        self.config.update(values)
        self.save_metadata()

    def get_class_profile(self, class_id_or_name: int | str):
        """Return a profile while keeping class ids controlled by the GUI."""

        return self.class_profiles.get(class_id_or_name)

    def verify_images(self) -> list[str]:
        errors: list[str] = []
        for path in self.list_images():
            try:
                read_image(path)
            except ValueError as exc:
                errors.append(str(exc))
        return errors
