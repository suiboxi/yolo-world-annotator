from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _cuda_available(monkeypatch: pytest.MonkeyPatch, *, count: int = 1) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"Test GPU {index}")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=(8 + index) * 1024**3),
    )


def _cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def test_auto_prefers_first_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from yolo_world_annotator.utils.device import resolve_device

    _cuda_available(monkeypatch, count=2)

    device = resolve_device("auto")

    assert device.requested == "auto"
    assert device.torch_device == "cuda:0"
    assert device.use_half is True
    assert device.description == "Test GPU 0 / cuda:0 / 8.0 GiB / float16"


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from yolo_world_annotator.utils.device import resolve_device

    _cuda_unavailable(monkeypatch)

    device = resolve_device("auto")

    assert device.torch_device == "cpu"
    assert device.use_half is False
    assert device.description == "CPU / float32"


def test_explicit_cpu_ignores_available_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from yolo_world_annotator.utils.device import resolve_device

    _cuda_available(monkeypatch)

    device = resolve_device(" CPU ")

    assert device.requested == "cpu"
    assert device.torch_device == "cpu"
    assert device.use_half is False


def test_explicit_cuda_index_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    from yolo_world_annotator.utils.device import resolve_device

    _cuda_available(monkeypatch, count=2)

    device = resolve_device("cuda:1")

    assert device.requested == "cuda:1"
    assert device.torch_device == "cuda:1"
    assert device.description == "Test GPU 1 / cuda:1 / 9.0 GiB / float16"


def test_cuda_alias_selects_cuda_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    from yolo_world_annotator.utils.device import resolve_device

    _cuda_available(monkeypatch)

    assert resolve_device("cuda").torch_device == "cuda:0"


def test_explicit_cuda_reports_unavailable_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yolo_world_annotator.utils.device import DeviceSelectionError, resolve_device

    _cuda_unavailable(monkeypatch)

    with pytest.raises(DeviceSelectionError, match="CUDA.*不可用"):
        resolve_device("cuda:0")


def test_cuda_index_must_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    from yolo_world_annotator.utils.device import DeviceSelectionError, resolve_device

    _cuda_available(monkeypatch, count=1)

    with pytest.raises(DeviceSelectionError, match="cuda:1.*1 个"):
        resolve_device("cuda:1")


@pytest.mark.parametrize("requested", ["mps", "gpu", "cuda:-1", "cuda:abc"])
def test_invalid_device_request_is_rejected(requested: str) -> None:
    from yolo_world_annotator.utils.device import DeviceSelectionError, resolve_device

    with pytest.raises(DeviceSelectionError, match="auto、cpu、cuda 或 cuda:N"):
        resolve_device(requested)


def test_environment_selects_default_device(monkeypatch: pytest.MonkeyPatch) -> None:
    from yolo_world_annotator.utils.device import resolve_device

    _cuda_unavailable(monkeypatch)
    monkeypatch.setenv("YOLO_WORLD_DEVICE", "cpu")

    assert resolve_device().requested == "cpu"
