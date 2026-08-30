"""Mode A/B/C/D benchmark utilities.

The benchmark is intentionally model-agnostic: it can consume stored
annotations or a caller-provided pipeline callback.  This makes it useful on
an RTX 4060 without forcing an expensive VLM run just to inspect statistics.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from yolo_world_annotator.core.annotation import BoundingBox
from yolo_world_annotator.core.dataset import DatasetProject
from yolo_world_annotator.core.evaluation import _evaluate_variant
from yolo_world_annotator.core.verification import AUTO_ACCEPT, REVIEW

MODE_A = "YOLO"
MODE_B = "YOLO+SAHI"
MODE_C = "YOLO+SAHI+SIGLIP"
MODE_D = "YOLO+SAHI+SIGLIP+VLM"
BENCHMARK_MODES = (MODE_A, MODE_B, MODE_C, MODE_D)


@dataclass(slots=True)
class BenchmarkEntry:
    mode: str
    metrics: dict[str, Any]
    performance: dict[str, float] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "metrics": dict(self.metrics),
            "performance": dict(self.performance),
            "rates": dict(self.rates),
        }


def _predictions_for_mode(project: DatasetProject, mode: str) -> dict[str, list[tuple[BoundingBox, int]]]:
    result: dict[str, list[tuple[BoundingBox, int]]] = defaultdict(list)
    for image_path in project.list_images():
        annotation = project.annotations.get(image_path.name)
        if annotation is None:
            continue
        for box in annotation.objects:
            yolo_id = int(box.yolo_class_id if box.yolo_class_id is not None else box.class_id)
            if mode == MODE_A:
                result[image_path.name].append((box, yolo_id))
            elif mode == MODE_B:
                if box.sahi_enabled or box.inference_mode.upper().startswith("SAHI"):
                    result[image_path.name].append((box, int(box.class_id)))
                else:
                    result[image_path.name].append((box, yolo_id))
            elif mode == MODE_C:
                result[image_path.name].append((box, int(box.class_id)))
            else:
                # D uses the final, VLM-aware decision but keeps rejected boxes
                # out of the trainable prediction set.
                if box.vlm_final_result == "NOT_MATCH" or box.fusion_status == "REJECT":
                    continue
                result[image_path.name].append((box, int(box.class_id)))
    return result


def _rates(project: DatasetProject, mode: str) -> dict[str, float]:
    boxes = [box for annotation in project.annotations.values() for box in annotation.objects]
    total = len(boxes) or 1
    accepted = sum(1 for box in boxes if box.fusion_status == AUTO_ACCEPT)
    review = sum(1 for box in boxes if box.fusion_status == REVIEW or box.review_required)
    corrections = sum(1 for box in boxes if box.human_modified)
    triggers = sum(1 for box in boxes if box.vlm_triggered)
    return {
        "auto_accept_rate": accepted / total,
        "review_rate": review / total,
        "human_correction_rate": corrections / total,
        "vlm_trigger_rate": triggers / total,
        "sahi_bring_back_rate": sum(1 for box in boxes if box.sahi_enabled) / total if mode != MODE_A else 0.0,
        "vlm_calls_per_image": triggers / max(1, len(project.list_images())),
        "average_tiles_per_image": sum(box.sahi_tile_count for box in boxes) / max(1, len(project.list_images())),
    }


def _stored_performance(project: DatasetProject) -> dict[str, float]:
    pipelines = [
        annotation.verification.get("pipeline", {})
        for annotation in project.annotations.values()
        if isinstance(annotation.verification.get("pipeline", {}), dict)
    ]
    if not pipelines:
        return {}
    count = max(1, len(pipelines))
    return {
        "average_yolo_time": sum(float(item.get("yolo_time", item.get("yolo_normal_time", 0.0)) or 0.0) for item in pipelines) / count,
        "average_sahi_time": sum(float(item.get("merge_time", 0.0) or 0.0) for item in pipelines) / count,
        "average_siglip_time": sum(float(item.get("siglip_time", 0.0) or 0.0) for item in pipelines) / count,
        "average_vlm_time": sum(float(item.get("vlm_time", 0.0) or 0.0) for item in pipelines) / count,
        "vlm_calls_per_image": sum(float(item.get("vlm_calls", 0.0) or 0.0) for item in pipelines) / count,
        "total_time": sum(float(item.get("total_time", 0.0) or 0.0) for item in pipelines),
        "average_tiles_per_image": sum(float(item.get("tile_count", 0) or 0.0) for item in pipelines) / count,
    }


def benchmark_project(
    project: DatasetProject,
    ground_truth: Mapping[str, list[dict[str, Any]]],
    *,
    iou_threshold: float = 0.5,
    modes: Sequence[str] = BENCHMARK_MODES,
    mode_predictions: Mapping[str, dict[str, list[tuple[BoundingBox, int]]]] | None = None,
    timings: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Compare requested modes and include quality, review and timing metrics."""

    entries: dict[str, dict[str, Any]] = {}
    for mode in modes:
        predictions = (mode_predictions or {}).get(mode) if mode_predictions else None
        predictions = predictions or _predictions_for_mode(project, mode)
        started = time.perf_counter()
        metrics = _evaluate_variant(ground_truth, predictions, iou_threshold=float(iou_threshold))
        elapsed = time.perf_counter() - started
        performance = {
            "average_yolo_time": 0.0,
            "average_sahi_time": 0.0,
            "average_siglip_time": 0.0,
            "average_vlm_time": 0.0,
            "total_time": elapsed,
            "gpu_vram_peak": 0.0,
        }
        performance.update(_stored_performance(project))
        if timings and mode in timings:
            performance.update({key: float(value) for key, value in timings[mode].items()})
        entries[mode] = BenchmarkEntry(mode, metrics, performance, _rates(project, mode)).to_dict()
    return {
        "iou_threshold": float(iou_threshold),
        "modes": entries,
        "mode_order": [mode for mode in modes if mode in entries],
    }


def run_benchmark(
    project: DatasetProject,
    ground_truth: Mapping[str, list[dict[str, Any]]],
    *,
    pipeline: Callable[[str, Path], Sequence[BoundingBox]] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Run a callback-backed benchmark, or benchmark stored metadata."""

    if pipeline is None:
        return benchmark_project(project, ground_truth, iou_threshold=iou_threshold)
    predictions: dict[str, dict[str, list[tuple[BoundingBox, int]]]] = {}
    timings: dict[str, dict[str, float]] = {}
    for mode in BENCHMARK_MODES:
        mode_values: dict[str, list[tuple[BoundingBox, int]]] = defaultdict(list)
        started = time.perf_counter()
        for path in project.list_images():
            for box in pipeline(mode, path) or []:
                mode_values[path.name].append((box, int(box.class_id)))
        elapsed = time.perf_counter() - started
        predictions[mode] = mode_values
        timings[mode] = {"total_time": elapsed, "average_yolo_time": elapsed / max(1, len(project.list_images()))}
    return benchmark_project(project, ground_truth, iou_threshold=iou_threshold, mode_predictions=predictions, timings=timings)


__all__ = [
    "MODE_A",
    "MODE_B",
    "MODE_C",
    "MODE_D",
    "BENCHMARK_MODES",
    "BenchmarkEntry",
    "benchmark_project",
    "run_benchmark",
]
