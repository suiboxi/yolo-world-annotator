"""Persistent difficult-sample records for later prompt tuning/fine-tuning."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.annotation import BoundingBox
from utils.config import atomic_write_json, load_json


ERROR_TYPES = {
    "CLASS_CONFLICT",
    "FALSE_POSITIVE",
    "FALSE_NEGATIVE",
    "LOW_CONFIDENCE",
    "HUMAN_CORRECTION",
    "VLM_UNCERTAIN",
    "VLM_REJECT",
    "BAD_BBOX",
    "SAHI_RELATED",
}


def _box_dict(box: BoundingBox | None) -> dict[str, Any] | None:
    return None if box is None else box.to_dict()


def load_hard_samples(path: Path) -> list[dict[str, Any]]:
    raw = load_json(path, [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def append_hard_sample(
    path: Path,
    *,
    image: Path,
    box: BoundingBox | None,
    error_type: str,
    original_box: BoundingBox | None = None,
    human_correction: BoundingBox | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Append one normalized, JSON-safe hard-sample record atomically."""

    error = str(error_type).upper()
    if error not in ERROR_TYPES:
        raise ValueError(f"未知 hard sample 类型：{error_type}")
    yolo_source = original_box
    if yolo_source is None and error != "FALSE_NEGATIVE":
        yolo_source = box
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": str(image),
        "bbox": _box_dict(box),
        "yolo_prediction": _box_dict(yolo_source),
        "siglip_prediction": (
            None
            if box is None or not box.siglip_enabled
            else {
                "class_id": box.siglip_class_id,
                "class_name": box.siglip_class_name,
                "score": box.siglip_score,
                "agreement": box.agreement,
            }
        ),
        "vlm_prediction": (
            None
            if box is None or not box.vlm_enabled
            else {
                "triggered": box.vlm_triggered,
                "model": box.vlm_model,
                "target_class": box.vlm_target_class,
                "features": dict(box.vlm_features or {}),
                "final_result": box.vlm_final_result,
                "self_reported_confidence": box.vlm_confidence,
                "parse_error": box.vlm_parse_error,
            }
        ),
        "pipeline": {
            "inference_mode": box.inference_mode if box is not None else None,
            "sahi_enabled": bool(box.sahi_enabled) if box is not None else False,
            "sahi_tile_count": int(box.sahi_tile_count or 0) if box is not None else 0,
            "decision_state": box.decision_state if box is not None else None,
            "decision_reason": box.decision_reason if box is not None else None,
        },
        "original_prediction": _box_dict(original_box),
        "human_correction": _box_dict(human_correction),
        "error_type": error,
    }
    if note:
        record["note"] = str(note)
    records = load_hard_samples(path)
    records.append(record)
    atomic_write_json(path, records)
    return record


def record_auto_issues(path: Path, image: Path, boxes: list[BoundingBox]) -> int:
    """Record conflict/low-confidence boxes and return the number written."""

    count = 0
    for box in boxes:
        original = deepcopy(box)
        if box.yolo_class_id is not None:
            original.class_id = int(box.yolo_class_id)
        if box.yolo_class_name is not None:
            original.class_name = str(box.yolo_class_name)
        if box.yolo_confidence is not None:
            original.confidence = float(box.yolo_confidence)
        original.source = "YOLO-World"
        if box.agreement is False:
            append_hard_sample(
                path,
                image=image,
                box=box,
                original_box=original,
                error_type="CLASS_CONFLICT",
            )
            count += 1
        elif box.fusion_status in {"REVIEW", "REJECT"}:
            error_type = "LOW_CONFIDENCE"
            if box.vlm_final_result == "UNCERTAIN" or box.vlm_parse_error:
                error_type = "VLM_UNCERTAIN"
            elif box.vlm_final_result == "NOT_MATCH":
                error_type = "VLM_REJECT"
            elif box.sahi_enabled:
                error_type = "SAHI_RELATED"
            append_hard_sample(
                path,
                image=image,
                box=box,
                original_box=original,
                error_type=error_type,
            )
            count += 1
    return count
