"""Class-aware post-processing for tiled detections.

The implementation is dependency-light and deterministic.  It supports NMS
and weighted box fusion (WBF) while preserving every detector metadata field.
No function mutates the input list or its boxes.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Sequence

from yolo_world_annotator.core.annotation import BoundingBox


def _xyxy(box: object) -> tuple[float, float, float, float]:
    if isinstance(box, BoundingBox):
        return float(box.x1), float(box.y1), float(box.x2), float(box.y2)
    if all(hasattr(box, name) for name in ("x1", "y1", "x2", "y2")):
        return tuple(float(getattr(box, name)) for name in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]
    if isinstance(box, Sequence) and len(box) == 4:
        return tuple(float(value) for value in box)  # type: ignore[return-value]
    raise TypeError("box 必须提供 x1/y1/x2/y2 或四值坐标")


def bbox_iou(left: object, right: object, *, metric: str = "IOU") -> float:
    """Return IoU (or intersection-over-min-area) for two boxes."""

    lx1, ly1, lx2, ly2 = _xyxy(left)
    rx1, ry1, rx2, ry2 = _xyxy(right)
    ix1, iy1 = max(min(lx1, lx2), min(rx1, rx2)), max(min(ly1, ly2), min(ry1, ry2))
    ix2, iy2 = min(max(lx1, lx2), max(rx1, rx2)), min(max(ly1, ly2), max(ry1, ry2))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, abs(lx2 - lx1)) * max(0.0, abs(ly2 - ly1))
    right_area = max(0.0, abs(rx2 - rx1)) * max(0.0, abs(ry2 - ry1))
    normalized_metric = str(metric or "IOU").upper().replace("-", "_")
    if normalized_metric in {"IOS", "INTERSECTION_OVER_SMALLER", "INTERSECTION_OVER_MIN"}:
        denominator = min(left_area, right_area)
    else:
        denominator = left_area + right_area - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _same_class(left: BoundingBox, right: BoundingBox) -> bool:
    left_id = left.yolo_class_id if left.yolo_class_id is not None else left.class_id
    right_id = right.yolo_class_id if right.yolo_class_id is not None else right.class_id
    return int(left_id) == int(right_id)


def nms(
    boxes: Iterable[BoundingBox],
    iou_threshold: float = 0.50,
    *,
    match_metric: str = "IOU",
) -> list[BoundingBox]:
    """Class-aware greedy non-maximum suppression."""

    candidates = [deepcopy(box) for box in boxes]
    threshold = min(1.0, max(0.0, float(iou_threshold)))
    candidates.sort(key=lambda item: float(item.confidence or item.yolo_confidence or 0.0), reverse=True)
    kept: list[BoundingBox] = []
    for candidate in candidates:
        if any(_same_class(candidate, previous) and bbox_iou(candidate, previous, metric=match_metric) > threshold for previous in kept):
            continue
        kept.append(candidate)
    return kept


def weighted_box_fusion(
    boxes: Iterable[BoundingBox],
    iou_threshold: float = 0.55,
    *,
    match_metric: str = "IOU",
    score_threshold: float = 0.0,
) -> list[BoundingBox]:
    """Merge overlapping same-class boxes using confidence-weighted geometry."""

    threshold = min(1.0, max(0.0, float(iou_threshold)))
    score_cutoff = min(1.0, max(0.0, float(score_threshold)))
    remaining = [
        deepcopy(box)
        for box in boxes
        if float(box.confidence if box.confidence is not None else box.yolo_confidence or 0.0) >= score_cutoff
    ]
    remaining.sort(key=lambda item: float(item.confidence or item.yolo_confidence or 0.0), reverse=True)
    fused: list[BoundingBox] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        survivors: list[BoundingBox] = []
        for candidate in remaining:
            if _same_class(seed, candidate) and bbox_iou(seed, candidate, metric=match_metric) >= threshold:
                cluster.append(candidate)
            else:
                survivors.append(candidate)
        remaining = survivors
        if len(cluster) == 1:
            fused.append(seed)
            continue
        weights = [max(1e-6, float(item.confidence if item.confidence is not None else item.yolo_confidence or 0.0)) for item in cluster]
        total = sum(weights)
        seed.x1 = sum(item.x1 * weight for item, weight in zip(cluster, weights)) / total
        seed.y1 = sum(item.y1 * weight for item, weight in zip(cluster, weights)) / total
        seed.x2 = sum(item.x2 * weight for item, weight in zip(cluster, weights)) / total
        seed.y2 = sum(item.y2 * weight for item, weight in zip(cluster, weights)) / total
        seed.confidence = max(
            float(item.confidence if item.confidence is not None else item.yolo_confidence or 0.0)
            for item in cluster
        )
        # Keep a compact provenance trail for audit/benchmarking.
        sources = [str(item.source) for item in cluster if item.source]
        seed.source = "+".join(dict.fromkeys(sources)) or seed.source
        fused.append(seed)
    return fused


def merge_detections(
    boxes: Iterable[BoundingBox],
    *,
    postprocess_type: str = "NMS",
    match_threshold: float = 0.50,
    match_metric: str = "IOU",
    score_threshold: float = 0.0,
) -> list[BoundingBox]:
    """Dispatch to NMS/WBF and validate unknown post-process values."""

    normalized = str(postprocess_type or "NMS").strip().upper().replace("-", "_")
    if normalized in {"NMS", "NON_MAX_SUPPRESSION"}:
        return nms(boxes, match_threshold, match_metric=match_metric)
    if normalized in {"WBF", "WEIGHTED_BOX_FUSION"}:
        return weighted_box_fusion(
            boxes,
            match_threshold,
            match_metric=match_metric,
            score_threshold=score_threshold,
        )
    if normalized in {"NONE", "NO_MERGE"}:
        return [deepcopy(box) for box in boxes]
    raise ValueError(f"不支持的切片后处理类型：{postprocess_type}")


# Common aliases used by scripts and older experiments.
compute_iou = bbox_iou
apply_nms = nms
iou = bbox_iou
merge_boxes = merge_detections


__all__ = [
    "bbox_iou",
    "compute_iou",
    "nms",
    "apply_nms",
    "iou",
    "merge_boxes",
    "weighted_box_fusion",
    "merge_detections",
]
