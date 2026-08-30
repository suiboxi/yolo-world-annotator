from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _FakeYOLOWorld:
    instances: list["_FakeYOLOWorld"] = []

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.classes: list[str] = []
        self.predict_calls: list[dict] = []
        self.__class__.instances.append(self)

    def set_classes(self, classes: list[str]) -> None:
        self.classes = list(classes)

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return []


@pytest.fixture(autouse=True)
def _clear_fake_instances() -> None:
    _FakeYOLOWorld.instances.clear()


def _weight(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    path.write_bytes(b"test weight placeholder")
    return path


def test_cpu_detector_uses_float32_ultralytics_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from yolo_world_annotator.models import yolo_world

    monkeypatch.setattr(yolo_world, "YOLOWorld", _FakeYOLOWorld)
    detector = yolo_world.YOLOWorldDetector(_weight(tmp_path), device="cpu")
    detector.set_classes(["apple"])

    assert detector.predict(tmp_path / "image.jpg", confidence=0.25, iou=0.45, imgsz=640) == []
    assert detector.device == "cpu"
    assert detector.device_description == "CPU / float32"
    assert _FakeYOLOWorld.instances[0].predict_calls == [
        {
            "source": str(tmp_path / "image.jpg"),
            "conf": 0.25,
            "iou": 0.45,
            "imgsz": 640,
            "device": "cpu",
            "quantize": 32,
            "verbose": False,
        }
    ]


def test_cuda_detector_uses_requested_device_and_float16(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from yolo_world_annotator.models import yolo_world

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"Test GPU {index}")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=12 * 1024**3),
    )
    monkeypatch.setattr(yolo_world, "YOLOWorld", _FakeYOLOWorld)
    detector = yolo_world.YOLOWorldDetector(_weight(tmp_path), device="cuda:1")
    detector.set_classes(["apple"])

    detector.predict(tmp_path / "image.jpg", confidence=0.2, iou=0.4, imgsz=768)

    call = _FakeYOLOWorld.instances[0].predict_calls[0]
    assert call["device"] == "cuda:1"
    assert call["quantize"] == 16
    assert "half" not in call
    assert detector.device_description == "Test GPU 1 / cuda:1 / 12.0 GiB / float16"


def test_explicit_cuda_error_is_not_silently_changed_to_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from yolo_world_annotator.models import yolo_world
    from yolo_world_annotator.utils.device import DeviceSelectionError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(yolo_world, "YOLOWorld", _FakeYOLOWorld)

    with pytest.raises(DeviceSelectionError, match="CUDA.*不可用"):
        yolo_world.YOLOWorldDetector(_weight(tmp_path), device="cuda")
