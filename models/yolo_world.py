from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

_appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
os.environ["YOLO_CONFIG_DIR"] = str(_appdata / "YOLO26")

from ultralytics import YOLOWorld
from ultralytics.utils.downloads import attempt_download_asset

from core.annotation import BoundingBox
from models.base import BaseDetector


LOGGER = logging.getLogger(__name__)


class YOLOWorldDetector(BaseDetector):
    """Persistent Ultralytics YOLO-World detector for official V1/V2 weights."""

    def __init__(self, model_path: Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "未检测到可用的 NVIDIA CUDA GPU。本程序不会回退到 CPU，"
                "请检查显卡驱动以及 CUDA 版 PyTorch。"
            )
        model_path = model_path.resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            attempt_download_asset(str(model_path))
        if not model_path.exists():
            raise FileNotFoundError(f"模型下载失败：{model_path}")
        self.model_path = model_path
        self.device: int = 0
        self.device_description = f"{torch.cuda.get_device_name(0)} / cuda:0"
        self.model = YOLOWorld(str(model_path))
        self.classes: list[str] = []
        self.last_image_size: tuple[int, int] | None = None

    def set_classes(self, classes: list[str]) -> None:
        clean = [item.strip() for item in classes if item.strip()]
        if not clean:
            raise ValueError("请至少输入一个检测类别")
        if clean == self.classes:
            return
        self.model.set_classes(clean)
        self.classes = clean

    @torch.inference_mode()
    def predict(
        self, image: Path, *, confidence: float, iou: float, imgsz: int
    ) -> list[BoundingBox]:
        if not self.classes:
            raise RuntimeError("尚未设置检测类别")
        try:
            results = self.model.predict(
                source=str(image),
                conf=float(confidence),
                iou=float(iou),
                imgsz=int(imgsz),
                device=self.device,
                quantize=16,
                verbose=False,
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "GPU 显存不足。请降低 Image Size，使用 yolov8s-worldv2，或关闭其他占用 GPU 的程序。"
            ) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "GPU 显存不足。请降低 Image Size，使用 yolov8s-worldv2，或关闭其他占用 GPU 的程序。"
                ) from exc
            raise
        boxes: list[BoundingBox] = []
        if not results or results[0].boxes is None:
            self.last_image_size = None
            return boxes
        original_height, original_width = results[0].orig_shape
        self.last_image_size = (int(original_width), int(original_height))
        result_boxes = results[0].boxes
        xyxy = result_boxes.xyxy.detach().cpu().tolist()
        confidences = result_boxes.conf.detach().cpu().tolist()
        class_ids = result_boxes.cls.detach().cpu().to(torch.int64).tolist()
        for coords, score, class_id in zip(xyxy, confidences, class_ids, strict=True):
            if not 0 <= class_id < len(self.classes):
                continue
            boxes.append(
                BoundingBox(
                    class_id=class_id,
                    class_name=self.classes[class_id],
                    x1=coords[0],
                    y1=coords[1],
                    x2=coords[2],
                    y2=coords[3],
                    confidence=score,
                    source="YOLO-World",
                )
            )
        return boxes
