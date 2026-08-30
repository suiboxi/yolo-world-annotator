from __future__ import annotations

import logging
from pathlib import Path

import torch
from ultralytics import YOLOWorld
from ultralytics.utils.downloads import attempt_download_asset

from yolo_world_annotator.core.annotation import BoundingBox
from yolo_world_annotator.models.base import BaseDetector
from yolo_world_annotator.utils.device import DeviceInfo, resolve_device

LOGGER = logging.getLogger(__name__)


class YOLOWorldDetector(BaseDetector):
    """Persistent Ultralytics YOLO-World detector with portable device selection."""

    def __init__(self, model_path: Path, *, device: str | None = "auto") -> None:
        self.device_info: DeviceInfo = resolve_device(device)
        model_path = model_path.expanduser().resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            attempt_download_asset(str(model_path))
        if not model_path.exists():
            raise FileNotFoundError(f"模型下载失败：{model_path}")
        self.model_path = model_path
        self.device = self.device_info.torch_device
        self.device_description = self.device_info.description
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
                quantize=16 if self.device_info.use_half else 32,
                verbose=False,
            )
        except torch.cuda.OutOfMemoryError as exc:
            self._raise_cuda_oom(exc)
        except RuntimeError as exc:
            if self.device_info.use_half and "out of memory" in str(exc).lower():
                self._raise_cuda_oom(exc)
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

    @staticmethod
    def _raise_cuda_oom(exc: BaseException) -> None:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU 显存不足。请降低推理图像尺寸、使用较小的 YOLO-World 权重，"
            "或关闭其他占用 GPU 的程序。"
        ) from exc


__all__ = ["YOLOWorldDetector"]
