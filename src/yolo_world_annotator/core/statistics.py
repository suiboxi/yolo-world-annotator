from __future__ import annotations

from collections import Counter

from yolo_world_annotator.core.annotation import AnnotationStatus
from yolo_world_annotator.core.dataset import DatasetProject
from yolo_world_annotator.core.verification import AUTO_ACCEPT, REJECT, REVIEW, classify_yolo_only
from yolo_world_annotator.utils.image_utils import read_image


def collect_statistics(project: DatasetProject) -> dict:
    statuses = Counter({status.value: 0 for status in AnnotationStatus})
    class_counts = Counter({name: 0 for name in project.classes})
    object_total = 0
    errors: list[str] = []
    object_statuses = Counter({AUTO_ACCEPT: 0, REVIEW: 0, REJECT: 0})
    human_confirmed = 0
    siglip_objects = 0
    agreements = 0
    human_corrections = 0
    sahi_objects = 0
    vlm_triggered = 0
    vlm_uncertain = 0
    per_class: dict[str, dict[str, int]] = {
        name: {"objects": 0, "agreement": 0} for name in project.classes
    }
    images = project.list_images()
    for image_path in images:
        try:
            image = read_image(image_path)
            height, width = image.shape[:2]
            annotation = project.get_annotation(image_path, (width, height))
            statuses[annotation.status.value] += 1
            object_total += len(annotation.objects)
            for box in annotation.objects:
                class_counts[box.class_name] += 1
                per_class.setdefault(box.class_name, {"objects": 0, "agreement": 0})
                per_class[box.class_name]["objects"] += 1
                if box.siglip_enabled:
                    siglip_objects += 1
                    if box.agreement:
                        agreements += 1
                        per_class[box.class_name]["agreement"] += 1
                if box.human_modified:
                    human_corrections += 1
                if box.sahi_enabled:
                    sahi_objects += 1
                if box.vlm_triggered:
                    vlm_triggered += 1
                if box.vlm_final_result == "UNCERTAIN" or box.vlm_parse_error:
                    vlm_uncertain += 1
                status_key = box.fusion_status
                if status_key == "HUMAN_CONFIRMED":
                    human_confirmed += 1
                if status_key is None and box.confidence is not None:
                    status_key = classify_yolo_only(box)
                if status_key in object_statuses:
                    object_statuses[status_key] += 1
        except Exception as exc:
            statuses[AnnotationStatus.UNLABELED.value] += 1
            errors.append(f"{image_path.name}: {exc}")
    class_agreement = {
        name: {
            "objects": values["objects"],
            "agreement": values["agreement"],
            "agreement_rate": (
                values["agreement"] / values["objects"] if values["objects"] else 0.0
            ),
        }
        for name, values in per_class.items()
    }
    return {
        "images": len(images),
        "unlabeled": statuses[AnnotationStatus.UNLABELED.value],
        "auto_labeled": statuses[AnnotationStatus.AUTO_LABELED.value],
        "verified": statuses[AnnotationStatus.VERIFIED.value],
        "objects": object_total,
        "classes": dict(class_counts),
        "errors": errors,
        "auto_accept": object_statuses[AUTO_ACCEPT],
        "review": object_statuses[REVIEW],
        "reject": object_statuses[REJECT],
        "human_confirmed": human_confirmed,
        "siglip_objects": siglip_objects,
        "agreement": agreements,
        "agreement_rate": agreements / siglip_objects if siglip_objects else 0.0,
        "human_corrections": human_corrections,
        "sahi_objects": sahi_objects,
        "vlm_triggered": vlm_triggered,
        "vlm_uncertain": vlm_uncertain,
        "sahi_rate": sahi_objects / object_total if object_total else 0.0,
        "vlm_trigger_rate": vlm_triggered / object_total if object_total else 0.0,
        "class_agreement": class_agreement,
    }
