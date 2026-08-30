from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "model": "yolov8s-worldv2.pt",
    "classes": ["person", "bus", "car", "bicycle"],
    "confidence": 0.25,
    "iou": 0.45,
    "imgsz": 640,
    "review_threshold": 0.5,
    "train_ratio": 0.8,
    # Optional YOLO-World + SigLIP2 verification.  Disabled by default to
    # preserve the original YOLO-only workflow and its startup cost.
    "siglip_enabled": False,
    "siglip_model": "google/siglip2-base-patch16-224",
    "siglip_padding": 0.10,
    "yolo_weight": 0.65,
    "siglip_weight": 0.35,
    "auto_accept_threshold": 0.75,
    "siglip_batch_size": 4,
    "siglip_precision": "auto",
    "candidate_top_k": 0,
    "siglip_prompt_template": "a photo of a {}",
    "siglip_prompt_ensemble": False,
    "per_class_thresholds": {},
    # Optional tiled inference.  Disabled by default for backwards
    # compatibility; all values are persisted in project.json when changed.
    "inference_mode": "YOLO_ONLY",
    "sahi_enabled": False,
    "sahi_slice_width": 1024,
    "sahi_slice_height": 1024,
    "sahi_overlap_width_ratio": 0.20,
    "sahi_overlap_height_ratio": 0.20,
    "sahi_postprocess_type": "NMS",
    "sahi_postprocess_match_threshold": 0.50,
    "sahi_postprocess_match_metric": "IOU",
    "sahi_max_tiles": 0,
    # Qwen3-VL is lazy and opt-in because it is a large local model.
    "vlm_enabled": False,
    "vlm_model": "Qwen/Qwen3-VL-8B-Instruct",
    "vlm_lazy_load": True,
    "vlm_low_memory": True,
    "vlm_max_new_tokens": 128,
    "vlm_yolo_low_threshold": 0.45,
    "vlm_siglip_low_threshold": 0.55,
    "vlm_margin_threshold": 0.10,
    "vlm_force_special_classes": True,
    # When enabled, every YOLO image result is checked by Qwen for every
    # detected box instead of only checking policy-selected hard samples.
    "vlm_check_each_image": False,
}


def as_bool(value: Any, default: bool = False) -> bool:
    """Parse booleans from JSON/config text without treating ``"false"`` as true."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default
