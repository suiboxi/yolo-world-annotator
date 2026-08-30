from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch

from yolo_world_annotator.app.inference_worker import InferenceWorker
from yolo_world_annotator.app.settings_panel import SettingsPanel
from yolo_world_annotator.models.model_manager import ModelManager


class _FakeDetector:
    instances: list["_FakeDetector"] = []

    def __init__(self, model_path: Path, *, device: str | None = None) -> None:
        self.model_path = Path(model_path).resolve()
        self.device_request = device
        self.device_info = SimpleNamespace(
            requested=device or os.environ.get("YOLO_WORLD_DEVICE", "auto")
        )
        self.device_description = f"device={device}"
        self.classes: list[str] = []
        self.__class__.instances.append(self)

    def set_classes(self, classes: list[str]) -> None:
        self.classes = list(classes)


def test_advanced_settings_persist_device(qapp) -> None:
    panel = SettingsPanel()

    assert panel.config_values()["device"] == "auto"
    panel.load_config({"device": "cpu"})
    assert panel.config_values()["device"] == "cpu"


def test_advanced_settings_use_environment_and_list_all_gpus(qapp, monkeypatch) -> None:
    monkeypatch.setenv("YOLO_WORLD_DEVICE", "cuda:2")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    panel = SettingsPanel()

    assert panel.device_combo.findData("cuda:1") >= 0
    assert panel.device_combo.findData("cuda:2") >= 0
    assert panel.config_values()["device"] == "cuda:2"
    panel.load_config({})
    assert panel.config_values()["device"] == "cuda:2"


def test_inference_worker_reloads_detector_when_device_changes(monkeypatch, tmp_path: Path) -> None:
    from yolo_world_annotator.app import inference_worker

    _FakeDetector.instances.clear()
    monkeypatch.setattr(inference_worker, "YOLOWorldDetector", _FakeDetector)
    worker = InferenceWorker()
    payload = {
        "model_path": str(tmp_path / "model.pt"),
        "classes": ["apple"],
        "siglip_enabled": False,
        "device": "cpu",
    }

    worker.load_model(payload)
    worker.load_model(payload)
    worker.load_model({**payload, "device": "cuda:1"})

    assert [item.device_request for item in _FakeDetector.instances] == ["cpu", "cuda:1"]


def test_model_manager_cache_includes_device(monkeypatch, tmp_path: Path) -> None:
    from yolo_world_annotator.models import model_manager

    _FakeDetector.instances.clear()
    monkeypatch.setattr(model_manager, "YOLOWorldDetector", _FakeDetector)
    manager = ModelManager()
    model_path = tmp_path / "model.pt"

    manager.load_yolo(model_path, ["apple"], device="cpu")
    manager.load_yolo(model_path, ["apple"], device="cpu")
    manager.load_yolo(model_path, ["apple"], device="cuda:0")

    assert [item.device_request for item in _FakeDetector.instances] == ["cpu", "cuda:0"]


def test_default_device_cache_honors_environment(monkeypatch, tmp_path: Path) -> None:
    from yolo_world_annotator.app import inference_worker
    from yolo_world_annotator.models import model_manager

    monkeypatch.setenv("YOLO_WORLD_DEVICE", "cpu")
    monkeypatch.setattr(inference_worker, "YOLOWorldDetector", _FakeDetector)
    monkeypatch.setattr(model_manager, "YOLOWorldDetector", _FakeDetector)
    model_path = tmp_path / "model.pt"

    _FakeDetector.instances.clear()
    worker = InferenceWorker()
    payload = {
        "model_path": str(model_path),
        "classes": ["apple"],
        "siglip_enabled": False,
    }
    worker.load_model(payload)
    worker.load_model(payload)
    assert len(_FakeDetector.instances) == 1

    _FakeDetector.instances.clear()
    manager = ModelManager()
    manager.load_yolo(model_path, ["apple"])
    manager.load_yolo(model_path, ["apple"])
    assert len(_FakeDetector.instances) == 1
