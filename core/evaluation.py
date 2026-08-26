"""Offline A/B evaluation for YOLO-only versus fused annotations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from core.annotation import BoundingBox
from core.dataset import DatasetProject
from utils.image_utils import read_image


def _iou(left: BoundingBox, right: dict[str, Any]) -> float:
    bbox = right.get("bbox", right)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = (float(value) for value in bbox)
    ix1, iy1 = max(left.x1, x1), max(left.y1, y1)
    ix2, iy2 = min(left.x2, x2), min(left.y2, y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_left = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    area_right = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def _class_id(item: dict[str, Any]) -> int:
    return int(item.get("class_id", -1))


def _normalise_ground_truth(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(raw, dict) and isinstance(raw.get("images"), dict):
        raw = raw["images"]
    if not isinstance(raw, dict):
        raise ValueError("ground-truth JSON 必须是 image -> objects 映射")
    result: dict[str, list[dict[str, Any]]] = {}
    for image, objects in raw.items():
        if isinstance(objects, dict):
            objects = objects.get("objects", [])
        if not isinstance(objects, list):
            continue
        result[str(image)] = [item for item in objects if isinstance(item, dict)]
    return result


def load_ground_truth(path: Path) -> dict[str, list[dict[str, Any]]]:
    return _normalise_ground_truth(json.loads(path.read_text(encoding="utf-8-sig")))


def _evaluate_variant(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[tuple[BoundingBox, int]]],
    *,
    iou_threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = 0
    class_tp: Counter[int] = Counter()
    class_fp: Counter[int] = Counter()
    class_fn: Counter[int] = Counter()
    class_correct = 0
    class_compared = 0
    all_images = set(ground_truth) | set(predictions)
    for image in all_images:
        truth_objects = ground_truth.get(image, [])
        candidates = list(predictions.get(image, predictions.get(Path(image).name, [])))
        matched: set[int] = set()
        for truth in truth_objects:
            best_index = -1
            best_iou = 0.0
            for index, (box, _) in enumerate(candidates):
                if index in matched:
                    continue
                score = _iou(box, truth)
                if score > best_iou:
                    best_iou, best_index = score, index
            truth_id = _class_id(truth)
            if best_index < 0 or best_iou < iou_threshold:
                fn += 1
                class_fn[truth_id] += 1
                continue
            matched.add(best_index)
            predicted_id = candidates[best_index][1]
            class_compared += 1
            if predicted_id == truth_id:
                tp += 1
                class_tp[truth_id] += 1
                class_correct += 1
            else:
                fp += 1
                fn += 1
                class_fp[predicted_id] += 1
                class_fn[truth_id] += 1
        for index, (_, predicted_id) in enumerate(candidates):
            if index not in matched:
                fp += 1
                class_fp[predicted_id] += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    classes = sorted(set(class_tp) | set(class_fp) | set(class_fn))
    per_class = {}
    for class_id in classes:
        ctp, cfp, cfn = class_tp[class_id], class_fp[class_id], class_fn[class_id]
        cprecision = ctp / (ctp + cfp) if ctp + cfp else 0.0
        crecall = ctp / (ctp + cfn) if ctp + cfn else 0.0
        per_class[str(class_id)] = {
            "precision": cprecision,
            "recall": crecall,
            "f1": 2 * cprecision * crecall / (cprecision + crecall)
            if cprecision + crecall
            else 0.0,
            "tp": ctp,
            "fp": cfp,
            "fn": cfn,
        }
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_accuracy": class_correct / class_compared if class_compared else 0.0,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "per_class": per_class,
    }


def evaluate_ab(
    project: DatasetProject,
    ground_truth: dict[str, list[dict[str, Any]]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare stored YOLO-only metadata with final fused class decisions."""

    if not 0.0 < float(iou_threshold) <= 1.0:
        raise ValueError("IoU threshold 必须在 (0, 1] 范围内")

    yolo: dict[str, list[tuple[BoundingBox, int]]] = defaultdict(list)
    fused: dict[str, list[tuple[BoundingBox, int]]] = defaultdict(list)
    agreement_total = 0
    agreement_count = 0
    annotations = dict(project.annotations)
    for image_path in project.list_images():
        if image_path.name not in annotations:
            try:
                image = read_image(image_path)
                height, width = image.shape[:2]
                annotations[image_path.name] = project.get_annotation(image_path, (width, height))
            except Exception:
                continue
    for image_name, annotation in annotations.items():
        for box in annotation.objects:
            yolo_id = box.yolo_class_id if box.yolo_class_id is not None else box.class_id
            yolo[image_name].append((box, int(yolo_id)))
            fused[image_name].append((box, int(box.class_id)))
            if box.siglip_enabled:
                agreement_total += 1
                agreement_count += int(bool(box.agreement))
    return {
        "iou_threshold": float(iou_threshold),
        "yolo_only": _evaluate_variant(ground_truth, yolo, iou_threshold=iou_threshold),
        "yolo_siglip2": _evaluate_variant(ground_truth, fused, iou_threshold=iou_threshold),
        "agreement": agreement_count / agreement_total if agreement_total else 0.0,
    }
